from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

_IMPORT_EXCEPTION: Optional[Exception] = None

try:
    from .data_augmentation import VideoAugmentation
    from .monitoring import UncertaintyMonitor
    from .models.losses import VideoReIDCriterion
    from .models.reid_model import VideoReIDModel
except Exception as e1:
    try:
        import sys
        from pathlib import Path as _Path

        sys.path.insert(0, str(_Path(__file__).parent.parent))
        from src.data_augmentation import VideoAugmentation
        from src.monitoring import UncertaintyMonitor
        from src.models.losses import VideoReIDCriterion
        from src.models.reid_model import VideoReIDModel
    except Exception as e2:
        _IMPORT_EXCEPTION = e2
        VideoReIDModel = None  # type: ignore[assignment]
        VideoReIDCriterion = None  # type: ignore[assignment]
        VideoAugmentation = None  # type: ignore[assignment]
        UncertaintyMonitor = None  # type: ignore[assignment]
        if _IMPORT_EXCEPTION is None:
            _IMPORT_EXCEPTION = e1


def ensure_runtime_dependencies() -> None:
    if VideoReIDModel is not None and VideoReIDCriterion is not None:
        return
    err = _IMPORT_EXCEPTION
    lines = [
        "Failed to import model dependencies.",
        "Install in WSB:",
        "  conda run -n WSB pip install -U pip setuptools wheel",
        "  conda run -n WSB pip install einops tensorboard wandb",
        "  conda run -n WSB pip install causal-conv1d",
        "  conda run -n WSB pip install mamba-ssm --no-build-isolation",
        "If mamba-ssm fails, install VS C++ Build Tools (cl.exe).",
    ]
    if err is not None:
        lines.append(f"Original error: {repr(err)}")
    raise RuntimeError("\n".join(lines))


# ---------------------------------------------------------------------------
# Seek-based video loading (TODO-19)
# ---------------------------------------------------------------------------

def _load_video_seek(
    path: Path,
    frames_per_clip: int,
    resize: Tuple[int, int],
    random_sample: bool = False,
) -> torch.Tensor:
    """Seek-based 帧采样：先获取总帧数，再用 seek 读取目标帧，避免读取全部帧。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: 顺序读取全部帧
        cap.release()
        return _load_video_sequential(path, frames_per_clip, resize, random_sample)

    # 计算目标帧索引
    if random_sample:
        if total_frames >= frames_per_clip:
            target_idxs = sorted(random.sample(range(total_frames), frames_per_clip))
        else:
            target_idxs = sorted(random.choices(range(total_frames), k=frames_per_clip))
    else:
        target_idxs = np.linspace(0, total_frames - 1, num=frames_per_clip, dtype=int).tolist()

    sampled: List[np.ndarray] = []
    for idx in target_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            # Seek 失败，fallback 到顺序读取
            cap.release()
            return _load_video_sequential(path, frames_per_clip, resize, random_sample)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize[1], resize[0]))
        sampled.append(frame)
    cap.release()

    arr = np.stack(sampled, axis=0).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


def _load_video_sequential(
    path: Path,
    frames_per_clip: int,
    resize: Tuple[int, int],
    random_sample: bool = False,
) -> torch.Tensor:
    """顺序读取全部帧后采样（fallback）。"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize[1], resize[0]))
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Empty video file: {path}")

    if random_sample:
        if len(frames) >= frames_per_clip:
            idxs = sorted(random.sample(range(len(frames)), frames_per_clip))
        else:
            idxs = sorted(random.choices(range(len(frames)), k=frames_per_clip))
    else:
        idxs = np.linspace(0, len(frames) - 1, num=frames_per_clip, dtype=int).tolist()

    sampled = [frames[i] for i in idxs]
    arr = np.stack(sampled, axis=0).astype("float32") / 255.0
    return torch.from_numpy(arr).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class VideoCowClipsDataset(Dataset):
    """基于 clip1/clip2 的视频数据集（tracklet_halves 协议）。"""

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        root: Path,
        id_list: Optional[Sequence[str]] = None,
        use_clip: str = "both",
        frames_per_clip: int = 8,
        resize: Tuple[int, int] = (224, 224),
        augmentation: Optional["VideoAugmentation"] = None,
    ) -> None:
        super().__init__()
        if use_clip not in {"clip1", "clip2", "both"}:
            raise ValueError(f"Unsupported use_clip={use_clip}")
        self.root = Path(root)
        self.use_clip = use_clip
        self.frames_per_clip = frames_per_clip
        self.resize = resize
        self.augmentation = augmentation

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if id_list is None:
            self.id_list = sorted([p.name for p in self.root.iterdir() if p.is_dir()])
        else:
            self.id_list = sorted(list(id_list))
        if not self.id_list:
            raise RuntimeError(f"No IDs found under {self.root}")

        self.id2label = {id_name: idx for idx, id_name in enumerate(self.id_list)}
        self.samples: List[Tuple[Path, int]] = []
        for id_name in self.id_list:
            id_dir = self.root / id_name
            if not id_dir.is_dir():
                continue
            clip1_paths = sorted(id_dir.glob("*_clip1.mp4"))
            clip2_paths = sorted(id_dir.glob("*_clip2.mp4"))
            if self.use_clip in {"clip1", "both"}:
                self.samples.extend((p, self.id2label[id_name]) for p in clip1_paths)
            if self.use_clip in {"clip2", "both"}:
                self.samples.extend((p, self.id2label[id_name]) for p in clip2_paths)
        if not self.samples:
            raise RuntimeError(f"No clip files found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, path: Path) -> torch.Tensor:
        return _load_video_seek(path, self.frames_per_clip, self.resize, random_sample=False)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        video = self._load_video(video_path)
        if self.augmentation is not None:
            video = self.augmentation(video)
        video = (video - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return video, label


class StrictReIDDataset(Dataset):
    """支持原始视频加载的 Dataset 类，用于 strict_reid 协议 (TODO-2)。
    从 manifest 中解析的视频路径列表加载原始 .mp4 文件。
    训练模式：从完整原始视频中随机采样 frames_per_clip 帧。
    评测模式：从完整原始视频中均匀采样 frames_per_clip 帧（确定性）。
    """

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(
        self,
        entries: List[Dict[str, str]],
        data_root: Path,
        frames_per_clip: int = 8,
        resize: Tuple[int, int] = (224, 224),
        is_train: bool = True,
        augmentation: Optional["VideoAugmentation"] = None,
        id2label: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.frames_per_clip = frames_per_clip
        self.resize = resize
        self.is_train = is_train
        self.augmentation = augmentation
        self.data_root = Path(data_root)

        # Build id_list and id2label
        unique_ids = sorted(set(e["id"] for e in entries))
        self.id_list = unique_ids
        if id2label is not None:
            self.id2label = id2label
        else:
            self.id2label = {id_name: idx for idx, id_name in enumerate(unique_ids)}

        # Build samples: (path, label)
        self.samples: List[Tuple[Path, int]] = []
        for entry in entries:
            video_path = self.data_root / entry["path"]
            label = self.id2label[entry["id"]]
            self.samples.append((video_path, label))

        if not self.samples:
            raise RuntimeError("No video entries provided to StrictReIDDataset")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, path: Path) -> torch.Tensor:
        return _load_video_seek(path, self.frames_per_clip, self.resize, random_sample=self.is_train)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        video = self._load_video(video_path)
        if self.augmentation is not None:
            video = self.augmentation(video)
        video = (video - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return video, label


# ---------------------------------------------------------------------------
# Collate, sampler, class weights
# ---------------------------------------------------------------------------

def collate_fn(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos, dim=0), torch.tensor(labels, dtype=torch.long)


class PKBatchSampler:
    """P-K batch sampler，兼容 VideoCowClipsDataset 和 StrictReIDDataset。"""

    def __init__(self, dataset, p: int = 4, k: int = 2):
        self.k = k
        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(dataset.samples):
            self.label_to_indices.setdefault(label, []).append(idx)
        self.labels = sorted(self.label_to_indices.keys())
        self.p = min(p, len(self.labels))
        self.batch_size = max(1, self.p * self.k)
        self._num_batches = max(1, len(dataset) // self.batch_size)

    def __iter__(self):
        for _ in range(self._num_batches):
            batch: List[int] = []
            selected_labels = random.sample(self.labels, self.p)
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    selected = random.sample(indices, self.k)
                else:
                    selected = random.choices(indices, k=self.k)
                batch.extend(selected)
            yield batch

    def __len__(self):
        return self._num_batches


def compute_class_weights(dataset) -> torch.Tensor:
    """兼容 VideoCowClipsDataset 和 StrictReIDDataset。"""
    label_counts: Dict[int, int] = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1
    num_classes = max(label_counts.keys()) + 1
    total_samples = sum(label_counts.values())
    weights = torch.ones(num_classes)
    for label, count in label_counts.items():
        weights[label] = total_samples / (num_classes * count)
    return weights


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 初始化函数，确保每个 worker 有不同但可复现的随机种子。"""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lambda_kl_schedule(step: int, warmup_steps: int, target: float, ramp_steps: int) -> float:
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return target * progress


def get_lambda_mi_schedule(step: int, warmup_steps: int, target: float, ramp_steps: int) -> float:
    """MI loss 权重的 warmup + ramp schedule (TODO-9)。"""
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return target * progress


def get_lambda_pose_aux_schedule(step: int, warmup_steps: int, target: float, ramp_steps: int) -> float:
    """Pose aux loss 权重的 warmup + ramp schedule。"""
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return target * progress


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


# ---------------------------------------------------------------------------
# Fold management
# ---------------------------------------------------------------------------

def split_into_folds(ids: Sequence[str], num_folds: int) -> List[List[str]]:
    if num_folds <= 1:
        return [list(ids)]
    fold_size = len(ids) // num_folds
    remainder = len(ids) % num_folds
    folds: List[List[str]] = []
    start = 0
    for idx in range(num_folds):
        end = start + fold_size + (1 if idx < remainder else 0)
        folds.append(list(ids[start:end]))
        start = end
    return folds


def _build_balanced_outer_folds(id_to_count: Dict[str, int], num_folds: int) -> List[List[str]]:
    """按视频数量贪心分配 ID 到各折，使各折视频总数尽量均衡 (TODO-4)。"""
    if num_folds <= 1:
        return [sorted(id_to_count.keys())]
    if num_folds > len(id_to_count):
        raise RuntimeError(f"num_folds={num_folds} > num_ids={len(id_to_count)}")
    ordered_ids = sorted(id_to_count.items(), key=lambda item: (-item[1], item[0]))
    folds: List[List[str]] = [[] for _ in range(num_folds)]
    fold_sums = [0 for _ in range(num_folds)]
    for id_name, count in ordered_ids:
        target_idx = min(range(num_folds), key=lambda idx: (fold_sums[idx], len(folds[idx]), idx))
        folds[target_idx].append(id_name)
        fold_sums[target_idx] += count
    return [sorted(f) for f in folds]


@dataclass
class OnlineFoldSpec:
    fold_index: int
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


@dataclass
class FoldPathSpec:
    fold_name: str
    train_root: Path
    val_root: Path
    test_root: Path
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


@dataclass
class StrictFoldSpec:
    """strict_reid 协议的折规格 (TODO-1)。"""
    fold_index: int
    fold_name: str
    train_entries: List[Dict[str, str]]
    val_gallery_entries: List[Dict[str, str]]
    val_query_entries: List[Dict[str, str]]
    test_gallery_entries: List[Dict[str, str]]
    test_query_entries: List[Dict[str, str]]
    train_ids: List[str]
    val_ids: List[str]
    test_ids: List[str]


def build_online_folds(
    data_root: Path,
    num_folds: int,
    val_ratio: float,
    seed: int,
) -> List[OnlineFoldSpec]:
    """构建在线划分的折（TODO-4：改用按视频数平衡的贪心分配）。"""
    id_dirs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if len(id_dirs) < 3:
        raise RuntimeError(f"Need at least 3 IDs for train/val/test, found {len(id_dirs)}")
    if num_folds > len(id_dirs):
        raise RuntimeError(f"num_folds={num_folds} > num_ids={len(id_dirs)}")
    if not (0.0 < val_ratio < 0.8):
        raise RuntimeError(f"val_ratio must be in (0, 0.8), got {val_ratio}")

    # 统计每个 ID 的视频数量
    id_to_count: Dict[str, int] = {}
    for id_name in id_dirs:
        id_dir = data_root / id_name
        clip_count = len(list(id_dir.glob("*_clip1.mp4"))) + len(list(id_dir.glob("*_clip2.mp4")))
        id_to_count[id_name] = max(clip_count, 1)

    # 使用贪心算法按视频数平衡分配到各折
    test_folds = _build_balanced_outer_folds(id_to_count, num_folds)

    # 打印各折视频总数，确认均衡
    for i, fold_ids in enumerate(test_folds):
        total_vids = sum(id_to_count[x] for x in fold_ids)
        print(f"  [balanced fold] test fold {i + 1}: {len(fold_ids)} IDs, {total_vids} videos")

    rng = random.Random(seed)
    all_ids = list(id_to_count.keys())

    specs: List[OnlineFoldSpec] = []
    for fold_idx, test_ids in enumerate(test_folds):
        remain = [x for x in all_ids if x not in test_ids]
        rng_fold = random.Random(seed + 10000 + fold_idx)
        rng_fold.shuffle(remain)
        val_n = max(1, int(round(len(remain) * val_ratio)))
        val_n = min(val_n, len(remain) - 1)
        val_ids = remain[:val_n]
        train_ids = remain[val_n:]
        if not train_ids or not val_ids or not test_ids:
            raise RuntimeError(
                f"Invalid split at fold {fold_idx + 1}: "
                f"train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
            )
        specs.append(
            OnlineFoldSpec(
                fold_index=fold_idx + 1,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=list(test_ids),
            )
        )
    return specs


def discover_fold_paths(fold_root: Path) -> List[FoldPathSpec]:
    if not fold_root.exists():
        raise FileNotFoundError(f"fold_root does not exist: {fold_root}")
    fold_dirs = sorted([p for p in fold_root.iterdir() if p.is_dir()])
    if not fold_dirs:
        raise RuntimeError(f"No fold directories found in: {fold_root}")

    specs: List[FoldPathSpec] = []
    for fold_dir in fold_dirs:
        train_root = fold_dir / "train"
        val_root = fold_dir / "val"
        test_root = fold_dir / "test"
        for split_root in (train_root, val_root, test_root):
            if not split_root.is_dir():
                raise RuntimeError(f"Missing split directory: {split_root}")
        train_ids = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
        val_ids = sorted([p.name for p in val_root.iterdir() if p.is_dir()])
        test_ids = sorted([p.name for p in test_root.iterdir() if p.is_dir()])
        if not train_ids or not val_ids or not test_ids:
            raise RuntimeError(f"Fold {fold_dir.name} has empty train/val/test IDs.")
        set_train, set_val, set_test = set(train_ids), set(val_ids), set(test_ids)
        if set_train & set_val or set_train & set_test or set_val & set_test:
            raise RuntimeError(f"Fold {fold_dir.name} has overlapping IDs between splits.")
        specs.append(
            FoldPathSpec(
                fold_name=fold_dir.name,
                train_root=train_root,
                val_root=val_root,
                test_root=test_root,
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
            )
        )
    return specs


def parse_strict_manifest(manifest_path: Path) -> Tuple[List[StrictFoldSpec], Path]:
    """解析 strict_reid_splits.json (TODO-1)。"""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("protocol") != "strict_reid":
        raise RuntimeError(
            f"Expected protocol='strict_reid', got '{manifest.get('protocol')}'"
        )
    data_root = manifest_path.parent

    specs: List[StrictFoldSpec] = []
    for fold in manifest["folds"]:
        spec = StrictFoldSpec(
            fold_index=fold["fold_index"],
            fold_name=fold["fold_name"],
            train_entries=fold["train_videos"],
            val_gallery_entries=fold["val_gallery"],
            val_query_entries=fold["val_query"],
            test_gallery_entries=fold["test_gallery"],
            test_query_entries=fold["test_query"],
            train_ids=fold["train_ids"],
            val_ids=fold["val_ids"],
            test_ids=fold["test_ids"],
        )
        specs.append(spec)

    # 打印 manifest 概要
    print(f"[strict_reid] Loaded {len(specs)} folds from: {manifest_path}")
    print(f"  eval_ids: {manifest.get('eval_ids', [])}")
    print(f"  train_only_ids: {manifest.get('train_only_ids', [])}")
    print(f"  excluded_ids: {manifest.get('excluded_ids', [])}")

    return specs, data_root


# ---------------------------------------------------------------------------
# Eval loader builders
# ---------------------------------------------------------------------------

def summarize_clip_layout(root: Path, id_list: Optional[Sequence[str]] = None) -> Dict[str, int]:
    if id_list is None:
        ids = [p.name for p in root.iterdir() if p.is_dir()]
    else:
        ids = list(id_list)
    clip1 = 0
    clip2 = 0
    total_ids = 0
    for id_name in ids:
        id_dir = root / id_name
        if not id_dir.is_dir():
            continue
        total_ids += 1
        clip1 += len(list(id_dir.glob("*_clip1.mp4")))
        clip2 += len(list(id_dir.glob("*_clip2.mp4")))
    return {"ids": total_ids, "clip1": clip1, "clip2": clip2}


def build_eval_loaders(
    root: Path,
    id_list: Optional[Sequence[str]],
    frames_per_clip: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    """构建 tracklet_halves 协议的评测 DataLoader (TODO-3)。
    注意：此协议使用同一视频的前后半段(clip1/clip2)互检，
    存在同轨迹信息泄漏，仅用于辅助参考。主实验应使用 strict_reid 协议。
    """
    gallery_dataset = VideoCowClipsDataset(
        root=root,
        id_list=id_list,
        use_clip="clip1",
        frames_per_clip=frames_per_clip,
        resize=(224, 224),
        augmentation=None,
    )
    query_dataset = VideoCowClipsDataset(
        root=root,
        id_list=id_list,
        use_clip="clip2",
        frames_per_clip=frames_per_clip,
        resize=(224, 224),
        augmentation=None,
    )
    gallery_loader = DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    query_loader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    return gallery_loader, query_loader


def build_strict_eval_loaders(
    entries_gallery: List[Dict[str, str]],
    entries_query: List[Dict[str, str]],
    data_root: Path,
    frames_per_clip: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    """构建 strict_reid 协议的评测 DataLoader。
    gallery 与 query 共享统一的 id2label 映射，确保标签编码一致。
    """
    # ---- 路径不重叠断言 ----
    gallery_paths = set(e["path"] for e in entries_gallery)
    query_paths = set(e["path"] for e in entries_query)
    overlap = gallery_paths & query_paths
    assert len(overlap) == 0, (
        f"Gallery and query share {len(overlap)} video paths! "
        f"Strict_reid protocol violated. Overlapping: {overlap}"
    )

    # ---- ID 合法性校验 ----
    gallery_ids = set(e["id"] for e in entries_gallery)
    query_ids = set(e["id"] for e in entries_query)
    assert query_ids.issubset(gallery_ids), (
        f"Query IDs not subset of gallery IDs! "
        f"Missing in gallery: {query_ids - gallery_ids}"
    )
    assert gallery_ids == query_ids, (
        f"Gallery/query ID sets differ. "
        f"Gallery-only: {gallery_ids - query_ids}, Query-only: {query_ids - gallery_ids}"
    )

    # ---- 统一 id2label 映射 ----
    all_ids = sorted(gallery_ids | query_ids)
    shared_id2label = {id_name: idx for idx, id_name in enumerate(all_ids)}

    gallery_ds = StrictReIDDataset(
        entries=entries_gallery, data_root=data_root,
        frames_per_clip=frames_per_clip, is_train=False, augmentation=None,
        id2label=shared_id2label,
    )
    query_ds = StrictReIDDataset(
        entries=entries_query, data_root=data_root,
        frames_per_clip=frames_per_clip, is_train=False, augmentation=None,
        id2label=shared_id2label,
    )

    # ---- 打印评测集统计 ----
    print(f"    [strict eval] gallery: {len(entries_gallery)} samples, "
          f"{len(gallery_ids)} IDs | query: {len(entries_query)} samples, "
          f"{len(query_ids)} IDs | shared label map: {len(shared_id2label)} IDs")

    gallery_loader = DataLoader(
        gallery_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    query_loader = DataLoader(
        query_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
    )
    return gallery_loader, query_loader


# ---------------------------------------------------------------------------
# Training loop (TODO-12: single forward, TODO-14: batch-level KL, TODO-9: MI warmup)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer_model: torch.optim.Optimizer,
    optimizer_mine: Optional[torch.optim.Optimizer],
    device: torch.device,
    global_step: int,
    kl_schedule_args: Optional[Dict] = None,
    mi_schedule_args: Optional[Dict] = None,
    pose_aux_schedule_args: Optional[Dict] = None,
    monitor: Optional["UncertaintyMonitor"] = None,
) -> Tuple[Dict[str, float], int]:
    """
    训练一个 epoch。
    - TODO-12: 单次 model forward，MINE 用 detached 特征更新
    - TODO-14: 每个 batch 更新 lambda_kl
    - TODO-9: 每个 batch 更新 lambda_mi
    返回 (metrics_dict, updated_global_step)。
    """
    model.train()
    criterion.train()
    total_loss = 0.0
    total_samples = 0
    loss_meter: Dict[str, float] = {}
    monitor_stats_list = []
    sigma2_vals: List[float] = []
    weight_entropy_vals: List[float] = []
    weight_max_vals: List[float] = []

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        # TODO-14: 每个 step 更新 KL schedule
        if kl_schedule_args is not None:
            criterion.lambda_kl = get_lambda_kl_schedule(global_step, **kl_schedule_args)
        # TODO-9: 每个 step 更新 MI schedule
        if mi_schedule_args is not None:
            criterion.lambda_mi = get_lambda_mi_schedule(global_step, **mi_schedule_args)
        # 每个 step 更新 pose_aux schedule
        if pose_aux_schedule_args is not None:
            criterion.lambda_pose_aux = get_lambda_pose_aux_schedule(global_step, **pose_aux_schedule_args)

        # TODO-12: 单次 model forward（消除 MINE 的双重 forward）
        outputs = model(videos)

        # MINE 步：用 detached 特征更新 MINE 网络（不反传梯度到主模型）
        if optimizer_mine is not None and getattr(model, "mine", None) is not None:
            optimizer_mine.zero_grad(set_to_none=True)
            mi_mine = model.mine(
                outputs["vid_id"].detach(), outputs["vid_pose"].detach()
            )
            loss_mine = -mi_mine
            loss_mine.backward()
            torch.nn.utils.clip_grad_norm_(model.mine.parameters(), max_norm=5.0)
            optimizer_mine.step()
            optimizer_mine.zero_grad(set_to_none=True)
            set_requires_grad(model.mine, False)

            # 用更新后的 MINE 重新计算 MI（此次需要对主模型的梯度流）
            outputs["mi"] = model.mine(outputs["vid_id"], outputs["vid_pose"])

        # 主模型步
        loss, loss_dict = criterion(outputs, labels)

        # NaN/Inf 检查
        if not torch.isfinite(loss):
            print(f"[FATAL] Non-finite loss detected at step {global_step}: {loss.item()}")
            print(f"  loss_dict: {loss_dict}")
            print(f"  batch labels: {labels.tolist()}")
            raise RuntimeError(f"Training stopped: non-finite loss at step {global_step}")

        optimizer_model.zero_grad(set_to_none=True)
        loss.backward()
        main_params = [p for p in model.parameters() if p.requires_grad]
        criterion_params = [p for p in criterion.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(main_params + criterion_params, max_norm=5.0)
        optimizer_model.step()
        if optimizer_mine is not None and getattr(model, "mine", None) is not None:
            set_requires_grad(model.mine, True)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        for key, value in loss_dict.items():
            loss_meter[key] = loss_meter.get(key, 0.0) + float(value) * bs

        if monitor is not None:
            sigma2 = outputs["sigma2"]
            stats = monitor.update(sigma2)
            if stats is not None:
                monitor_stats_list.append(stats)

        # sigma2 和聚合权重统计
        with torch.no_grad():
            s2 = outputs["sigma2"]  # (B, T, 1)
            sigma2_vals.append(s2.mean().item())
            if s2.mean().item() < 1e-8 or s2.mean().item() > 1e4:
                print(f"  [WARN] sigma2 abnormal at step {global_step}: "
                      f"mean={s2.mean().item():.6f}, std={s2.std().item():.6f}")
            w = outputs.get("weights")  # (B, T, 1)
            if w is not None:
                w_squeezed = w.squeeze(-1)  # (B, T)
                # 权重熵：越高越均匀
                log_w = torch.log(w_squeezed.clamp(min=1e-8))
                entropy = -(w_squeezed * log_w).sum(dim=-1).mean().item()
                weight_entropy_vals.append(entropy)
                weight_max_vals.append(w_squeezed.max(dim=-1).values.mean().item())

        global_step += 1

    avg = {key: val / max(total_samples, 1) for key, val in loss_meter.items()}
    avg["total"] = total_loss / max(total_samples, 1)
    if monitor_stats_list:
        latest = monitor_stats_list[-1]
        avg["sigma2_mean"] = latest.get("mean", 0.0)
        avg["sigma2_std"] = latest.get("std", 0.0)
        avg["sigma2_frame_variance"] = latest.get("frame_variance_mean", 0.0)
        avg["sigma2_health_status"] = latest.get("health_status", "UNKNOWN")
    if sigma2_vals:
        avg["sigma2_mean"] = float(np.mean(sigma2_vals))
        avg["sigma2_std"] = float(np.std(sigma2_vals))
    if weight_entropy_vals:
        avg["weight_entropy"] = float(np.mean(weight_entropy_vals))
    if weight_max_vals:
        avg["weight_max"] = float(np.mean(weight_max_vals))
    return avg, global_step


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_video_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(videos)
        all_feats.append(outputs["vid_id"].detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


@torch.no_grad()
def compute_reid_metrics(
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_labels: torch.Tensor,
    topk: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    # vid_id 已在模型输出端 L2 归一化 (TODO-10)，此处直接计算余弦相似度
    q = torch.nn.functional.normalize(query_feats, dim=1)
    g = torch.nn.functional.normalize(gallery_feats, dim=1)
    sim = torch.matmul(q, g.t())

    num_q = q.size(0)
    if num_q == 0 or g.size(0) == 0:
        return {f"rank-{k}": 0.0 for k in topk} | {"mAP": 0.0}

    indices = torch.argsort(sim, dim=1, descending=True)
    max_rank = min(max(topk), g.size(0))
    cmc = torch.zeros(max_rank, dtype=torch.float32)
    all_ap: List[float] = []
    valid_queries = 0

    for i in range(num_q):
        q_label = query_labels[i].item()
        order = indices[i]
        matches = (gallery_labels[order] == q_label).float()
        num_rel = int(matches.sum().item())
        if num_rel == 0:
            continue
        valid_queries += 1

        match_pos = torch.nonzero(matches, as_tuple=False).view(-1)
        first_pos = int(match_pos[0].item())
        if first_pos < max_rank:
            cmc[first_pos:] += 1

        ranks = torch.arange(1, matches.numel() + 1, dtype=torch.float32)
        cum_hits = torch.cumsum(matches, dim=0)
        precision = cum_hits / ranks
        ap = (precision * matches).sum() / max(num_rel, 1)
        all_ap.append(float(ap.item()))

    if valid_queries == 0:
        return {f"rank-{k}": 0.0 for k in topk} | {"mAP": 0.0}

    cmc = cmc / valid_queries
    metrics: Dict[str, float] = {}
    for k in topk:
        idx = min(k, max_rank) - 1
        metrics[f"rank-{k}"] = float(cmc[idx].item())
    metrics["mAP"] = float(np.mean(all_ap)) if all_ap else 0.0
    return metrics


def evaluate_reid(
    model: nn.Module,
    gallery_loader: DataLoader,
    query_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    gallery_feats, gallery_labels = extract_video_features(model, gallery_loader, device)
    query_feats, query_labels = extract_video_features(model, query_loader, device)
    return compute_reid_metrics(query_feats, query_labels, gallery_feats, gallery_labels)


@torch.no_grad()
def export_frame_quality_analysis(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    output_path: Path,
    topk: int = 3,
) -> None:
    """导出帧级质量分析数据（sigma2、聚合权重、top-k 帧索引）供论文可视化。"""
    model.eval()
    records = []
    sample_idx = 0
    for videos, labels in loader:
        videos = videos.to(device)
        outputs = model(videos)
        sigma2 = outputs["sigma2"]      # (B, T, 1)
        weights = outputs["weights"]    # (B, T, 1)
        B = videos.size(0)
        for i in range(B):
            s2 = sigma2[i].squeeze(-1).cpu().tolist()      # list of T floats
            w = weights[i].squeeze(-1).cpu()                # (T,)
            w_list = w.tolist()
            T = len(w_list)
            k = min(topk, T)
            top_high = torch.topk(w, k).indices.tolist()
            top_low = torch.topk(w, k, largest=False).indices.tolist()
            records.append({
                "sample_idx": sample_idx,
                "split": split_name,
                "identity": int(labels[i].item()),
                "num_frames": T,
                "sigma2_per_frame": s2,
                "weight_per_frame": w_list,
                "top_high_weight_frames": top_high,
                "top_low_weight_frames": top_low,
            })
            sample_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"    [EXPORT] Frame quality analysis ({split_name}): "
          f"{len(records)} samples -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 5 video ReID training")
    # Data source (三选一)
    parser.add_argument("--data-root", type=str, default=None,
                        help="tracklet_halves online split 的数据目录")
    parser.add_argument("--fold-root", type=str, default=None,
                        help="tracklet_halves prepared folds 的目录")
    parser.add_argument("--strict-manifest", type=str, default=None,
                        help="strict_reid_splits.json 路径 (TODO-1)")

    parser.add_argument("--protocol", type=str, default="strict_reid",
                        choices=["strict_reid", "tracklet_halves"],
                        help="评测协议。strict_reid 为主实验；tracklet_halves 仅辅助参考")

    parser.add_argument("--run-all-folds", action="store_true", default=False)
    parser.add_argument("--fold-index", type=int, default=None)

    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mine-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--lambda-mi", type=float, default=0.1)
    parser.add_argument("--lambda-orth", type=float, default=0.01)
    parser.add_argument("--lambda-temp", type=float, default=0.1)
    parser.add_argument("--lambda-kl", type=float, default=0.01)
    parser.add_argument("--lambda-pose-aux", type=float, default=0.2,
                        help="辅助姿态任务 loss 权重 (TODO-8)")
    parser.add_argument("--kl-warmup-steps", type=int, default=50)
    parser.add_argument("--kl-ramp-steps", type=int, default=100)
    parser.add_argument("--mi-warmup-steps", type=int, default=50,
                        help="MI loss warmup 步数 (TODO-9)")
    parser.add_argument("--mi-ramp-steps", type=int, default=100,
                        help="MI loss ramp 步数 (TODO-9)")
    parser.add_argument("--pose-aux-warmup-steps", type=int, default=50,
                        help="Pose aux loss warmup 步数")
    parser.add_argument("--pose-aux-ramp-steps", type=int, default=100,
                        help="Pose aux loss ramp 步数")
    parser.add_argument("--lambda-pose-aux-final", type=float, default=None,
                        help="Pose aux loss 最终权重（默认等于 --lambda-pose-aux）")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="LR warmup epoch 数 (TODO-15)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (TODO-20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--select-by", type=str, default="val_mAP", choices=["val_mAP"])
    parser.add_argument("--log-dir", type=str, default="runs/video_reid_cowclips")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_cowclips")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="video_reid_cowclips")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    # Ablation switches
    parser.add_argument("--disable-quality-gate", action="store_true")
    parser.add_argument("--disable-bidirectional", action="store_true")
    parser.add_argument("--disable-pose-stream", action="store_true")
    parser.add_argument("--disable-pose-injection", action="store_true")
    parser.add_argument("--disable-uncertainty-weighting", action="store_true")
    parser.add_argument("--disable-mi-loss", action="store_true")
    parser.add_argument("--disable-orth-loss", action="store_true")
    parser.add_argument("--disable-temp-loss", action="store_true")
    parser.add_argument("--disable-kl-loss", action="store_true")
    parser.add_argument("--disable-pose-aux", action="store_true",
                        help="禁用辅助姿态任务 (TODO-8 消融)")
    parser.add_argument("--export-frame-quality-analysis", action="store_true",
                        help="在 best/final 阶段导出帧级质量分析 (sigma2, weights, top-k)")
    parser.add_argument("--export-topk", type=int, default=3,
                        help="导出每个视频的 top-k 高/低权重帧索引")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def select_fold_specs(
    all_specs,
    fold_index: Optional[int],
    run_all_folds: bool,
):
    if fold_index is None and run_all_folds:
        return list(all_specs)
    if fold_index is None and not run_all_folds:
        return [all_specs[0]]
    if fold_index < 1 or fold_index > len(all_specs):
        raise RuntimeError(f"fold-index out of range: {fold_index}, valid [1, {len(all_specs)}]")
    return [all_specs[fold_index - 1]]


def build_ablation_config(args: argparse.Namespace) -> Dict[str, bool]:
    return {
        "use_quality_gating": not args.disable_quality_gate,
        "bidirectional": not args.disable_bidirectional,
        "use_pose_stream": not args.disable_pose_stream,
        "use_pose_to_id": not args.disable_pose_injection and not args.disable_pose_stream,
        "use_uncertainty_weighting": not args.disable_uncertainty_weighting,
    }


def build_loss_config(args: argparse.Namespace, model_config: Dict[str, bool]) -> Dict[str, float]:
    use_pose_stream = model_config["use_pose_stream"]
    return {
        "lambda_mi": 0.0 if args.disable_mi_loss or not use_pose_stream else args.lambda_mi,
        "lambda_orth": 0.0 if args.disable_orth_loss or not use_pose_stream else args.lambda_orth,
        "lambda_temp": 0.0 if args.disable_temp_loss else args.lambda_temp,
        "lambda_kl": 0.0 if args.disable_kl_loss else args.lambda_kl,
        "lambda_pose_aux": 0.0 if args.disable_pose_aux or not use_pose_stream else args.lambda_pose_aux,
    }


def format_toggle_status(config: Dict[str, bool], loss_config: Dict[str, float]) -> str:
    parts = [
        f"quality_gate={'on' if config['use_quality_gating'] else 'off'}",
        f"bidirectional={'on' if config['bidirectional'] else 'off'}",
        f"pose_stream={'on' if config['use_pose_stream'] else 'off'}",
        f"pose_injection={'on' if config['use_pose_to_id'] else 'off'}",
        f"uncertainty_weighting={'on' if config['use_uncertainty_weighting'] else 'off'}",
        f"mi_loss={'on' if loss_config['lambda_mi'] > 0 else 'off'}",
        f"orth_loss={'on' if loss_config['lambda_orth'] > 0 else 'off'}",
        f"temp_loss={'on' if loss_config['lambda_temp'] > 0 else 'off'}",
        f"kl_loss={'on' if loss_config['lambda_kl'] > 0 else 'off'}",
        f"pose_aux={'on' if loss_config.get('lambda_pose_aux', 0) > 0 else 'off'}",
    ]
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()

    has_strict = args.strict_manifest is not None
    has_fold = args.fold_root is not None
    has_data = args.data_root is not None
    if not (has_strict or has_fold or has_data):
        raise RuntimeError(
            "Must provide one of: --strict-manifest, --fold-root, or --data-root."
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    set_seed(args.seed)
    pin_memory = bool(args.pin_memory and device.type == "cuda")

    log_dir = Path(args.log_dir)
    ckpt_dir = Path(args.ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir)) if SummaryWriter is not None else None
    if writer is None:
        print("[WARN] tensorboard is unavailable; logs will not be written to TensorBoard.")

    use_wandb = args.use_wandb
    if use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except ImportError:
            use_wandb = False
            print("[WARN] wandb not installed, disabling wandb logging.")

    # 确定协议和数据模式
    protocol = args.protocol
    use_strict_mode = has_strict or (protocol == "strict_reid")

    # ---- strict_reid 硬约束：禁止隐式回退 ----
    if protocol == "strict_reid" and not has_strict:
        raise ValueError(
            "protocol='strict_reid' requires --strict-manifest to be specified. "
            "Implicit fallback to tracklet_halves is not allowed."
        )
    if protocol == "strict_reid" and has_strict:
        manifest_path = Path(args.strict_manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"strict-manifest file does not exist: {manifest_path}"
            )

    if use_strict_mode and has_strict:
        # ---- strict_reid 主实验路径 (TODO-1) ----
        strict_specs, strict_data_root = parse_strict_manifest(Path(args.strict_manifest))
        selected_strict = select_fold_specs(strict_specs, args.fold_index, args.run_all_folds)
        protocol = "strict_reid"
        print(f"[Protocol] strict_reid, {len(selected_strict)} fold(s) selected")
    elif has_fold:
        # ---- tracklet_halves prepared folds ----
        all_fold_specs = discover_fold_paths(Path(args.fold_root))
        selected_specs = select_fold_specs(all_fold_specs, args.fold_index, args.run_all_folds)
        protocol = "tracklet_halves"
        print(f"[Protocol] tracklet_halves (prepared folds from {args.fold_root})")
        print("[WARN] tracklet_halves 协议使用同轨迹 clip1/clip2 互检，存在信息泄漏，仅用于辅助参考")
        selected_strict = None
    else:
        # ---- tracklet_halves online split ----
        data_root = Path(args.data_root)
        clip_stats = summarize_clip_layout(data_root)
        if clip_stats["clip1"] == 0 or clip_stats["clip2"] == 0:
            raise RuntimeError(
                f"No clip pairs found in {data_root} "
                f"(clip1={clip_stats['clip1']}, clip2={clip_stats['clip2']})."
            )
        online_specs = build_online_folds(data_root, args.num_folds, args.val_ratio, args.seed)
        all_fold_specs = [
            FoldPathSpec(
                fold_name=f"fold{spec.fold_index:02d}",
                train_root=data_root, val_root=data_root, test_root=data_root,
                train_ids=spec.train_ids, val_ids=spec.val_ids, test_ids=spec.test_ids,
            )
            for spec in online_specs
        ]
        selected_specs = select_fold_specs(all_fold_specs, args.fold_index, args.run_all_folds)
        protocol = "tracklet_halves"
        print(f"[Protocol] tracklet_halves (online split from {data_root})")
        print("[WARN] tracklet_halves 协议使用同轨迹 clip1/clip2 互检，存在信息泄漏，仅用于辅助参考")
        selected_strict = None

    model_config = build_ablation_config(args)
    loss_config = build_loss_config(args, model_config)
    all_fold_metrics: List[Dict[str, float]] = []

    # 构建迭代列表
    if selected_strict is not None:
        fold_iter = selected_strict
    else:
        fold_iter = selected_specs

    for fold_idx, fold_spec in enumerate(fold_iter, start=1):
        if isinstance(fold_spec, StrictFoldSpec):
            fold_name = fold_spec.fold_name
            train_ids = fold_spec.train_ids
            val_ids = fold_spec.val_ids
            test_ids = fold_spec.test_ids
        else:
            fold_name = fold_spec.fold_name
            train_ids = fold_spec.train_ids
            val_ids = fold_spec.val_ids
            test_ids = fold_spec.test_ids

        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(fold_iter)}: {fold_name}")
        print(f"Train IDs={len(train_ids)}, Val IDs={len(val_ids)}, Test IDs={len(test_ids)}")
        print(f"Protocol: {protocol}")
        if protocol == "strict_reid" and has_strict:
            print(f"Strict manifest: {args.strict_manifest}")
        print(f"Ablations: {format_toggle_status(model_config, loss_config)}")
        print(f"{'=' * 72}")

        # TODO-5: 打印 gallery/query 样本数和 ID 分布
        if isinstance(fold_spec, StrictFoldSpec):
            print(f"  Train entries: {len(fold_spec.train_entries)}")
            print(f"  Val gallery: {len(fold_spec.val_gallery_entries)}, "
                  f"Val query: {len(fold_spec.val_query_entries)}")
            print(f"  Test gallery: {len(fold_spec.test_gallery_entries)}, "
                  f"Test query: {len(fold_spec.test_query_entries)}")

            # 构建 strict 模式的数据加载器
            train_dataset = StrictReIDDataset(
                entries=fold_spec.train_entries,
                data_root=strict_data_root,
                frames_per_clip=args.frames_per_clip,
                is_train=True,
                augmentation=VideoAugmentation(
                    use_occlusion=True, use_blur=True, use_brightness=True,
                    occlusion_prob=0.5, blur_prob=0.5, brightness_prob=0.5,
                ),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=PKBatchSampler(train_dataset, p=4, k=2),
                num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
                worker_init_fn=_worker_init_fn,
            )

            # TODO-3, TODO-5: strict 模式的评测 DataLoader
            val_gallery_loader, val_query_loader = build_strict_eval_loaders(
                fold_spec.val_gallery_entries, fold_spec.val_query_entries,
                strict_data_root, args.frames_per_clip, args.batch_size,
                args.num_workers, pin_memory,
            )
            test_gallery_loader, test_query_loader = build_strict_eval_loaders(
                fold_spec.test_gallery_entries, fold_spec.test_query_entries,
                strict_data_root, args.frames_per_clip, args.batch_size,
                args.num_workers, pin_memory,
            )
        else:
            # tracklet_halves 模式
            train_stats = summarize_clip_layout(fold_spec.train_root, train_ids)
            val_stats = summarize_clip_layout(fold_spec.val_root, val_ids)
            test_stats = summarize_clip_layout(fold_spec.test_root, test_ids)
            print(f"Train clips: clip1={train_stats['clip1']}, clip2={train_stats['clip2']}")
            print(f"Val clips:   clip1={val_stats['clip1']}, clip2={val_stats['clip2']}")
            print(f"Test clips:  clip1={test_stats['clip1']}, clip2={test_stats['clip2']}")
            for split_name, stats in (("train", train_stats), ("val", val_stats), ("test", test_stats)):
                if stats["clip1"] == 0 or stats["clip2"] == 0:
                    raise RuntimeError(
                        f"Fold {fold_name} split={split_name} has missing clips: "
                        f"clip1={stats['clip1']}, clip2={stats['clip2']}"
                    )

            train_dataset = VideoCowClipsDataset(
                root=fold_spec.train_root, id_list=train_ids,
                use_clip="both", frames_per_clip=args.frames_per_clip,
                resize=(224, 224),
                augmentation=VideoAugmentation(
                    use_occlusion=True, use_blur=True, use_brightness=True,
                    occlusion_prob=0.5, blur_prob=0.5, brightness_prob=0.5,
                ),
            )
            train_loader = DataLoader(
                train_dataset,
                batch_sampler=PKBatchSampler(train_dataset, p=4, k=2),
                num_workers=args.num_workers, pin_memory=pin_memory, collate_fn=collate_fn,
                worker_init_fn=_worker_init_fn,
            )
            val_gallery_loader, val_query_loader = build_eval_loaders(
                fold_spec.val_root, val_ids, args.frames_per_clip,
                args.batch_size, args.num_workers, pin_memory,
            )
            test_gallery_loader, test_query_loader = build_eval_loaders(
                fold_spec.test_root, test_ids, args.frames_per_clip,
                args.batch_size, args.num_workers, pin_memory,
            )

        class_weights = compute_class_weights(train_dataset).to(device)
        model = VideoReIDModel(
            feat_dim=args.feat_dim,
            num_blocks=args.num_blocks,
            use_quality_gating=model_config["use_quality_gating"],
            bidirectional=model_config["bidirectional"],
            use_pose_stream=model_config["use_pose_stream"],
            use_pose_to_id=model_config["use_pose_to_id"],
            use_uncertainty_weighting=model_config["use_uncertainty_weighting"],
        ).to(device)
        criterion = VideoReIDCriterion(
            feat_dim=args.feat_dim,
            num_classes=len(train_dataset.id_list),
            lambda_mi=loss_config["lambda_mi"],
            lambda_orth=loss_config["lambda_orth"],
            lambda_temp=loss_config["lambda_temp"],
            lambda_kl=loss_config["lambda_kl"],
            lambda_pose_aux=loss_config.get("lambda_pose_aux", 0.0),
            margin=args.margin,
            use_batch_hard=True,
            class_weights=class_weights,
            use_pose_aux=not args.disable_pose_aux and model_config["use_pose_stream"],
        ).to(device)

        optimizer_model = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if "mine" not in n]
            + list(criterion.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        mine_params = [p for n, p in model.named_parameters() if "mine" in n]
        optimizer_mine = (
            torch.optim.AdamW(mine_params, lr=args.mine_lr, weight_decay=args.weight_decay)
            if mine_params and loss_config["lambda_mi"] > 0
            else None
        )
        if optimizer_mine is None and getattr(model, "mine", None) is not None:
            set_requires_grad(model.mine, False)

        # TODO-15: LR warmup + CosineAnnealing
        warmup_epochs = max(0, args.warmup_epochs)
        cosine_epochs = max(1, args.num_epochs - warmup_epochs)
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer_model,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer_model, T_max=cosine_epochs, eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer_model,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer_model, T_max=args.num_epochs, eta_min=1e-6)

        monitor = UncertaintyMonitor(window_size=100, threshold=0.01, check_interval=50)

        # Schedule args for batch-level updates
        kl_schedule_args = {
            "warmup_steps": args.kl_warmup_steps,
            "target": loss_config["lambda_kl"],
            "ramp_steps": args.kl_ramp_steps,
        } if loss_config["lambda_kl"] > 0 else None

        mi_schedule_args = {
            "warmup_steps": args.mi_warmup_steps,
            "target": loss_config["lambda_mi"],
            "ramp_steps": args.mi_ramp_steps,
        } if loss_config["lambda_mi"] > 0 else None

        pose_aux_target = (
            args.lambda_pose_aux_final
            if args.lambda_pose_aux_final is not None
            else loss_config.get("lambda_pose_aux", 0.0)
        )
        pose_aux_schedule_args = {
            "warmup_steps": args.pose_aux_warmup_steps,
            "target": pose_aux_target,
            "ramp_steps": args.pose_aux_ramp_steps,
        } if loss_config.get("lambda_pose_aux", 0.0) > 0 else None

        global_step = 0
        best_val_map = -1.0
        best_epoch = -1
        best_state = None
        best_ckpt_path = ckpt_dir / f"{fold_name}_best.pth"
        patience_counter = 0  # TODO-20: early stopping

        for epoch in range(args.num_epochs):
            train_metrics, global_step = train_one_epoch(
                model, criterion, train_loader, optimizer_model, optimizer_mine,
                device, global_step, kl_schedule_args, mi_schedule_args,
                pose_aux_schedule_args, monitor,
            )
            scheduler.step()

            val_metrics = evaluate_reid(model, val_gallery_loader, val_query_loader, device)
            current_lr = optimizer_model.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:03d}/{args.num_epochs:03d} "
                f"lr={current_lr:.2e} "
                f"loss={train_metrics.get('total', 0.0):.4f} "
                f"id={train_metrics.get('id', 0.0):.4f} "
                f"tri={train_metrics.get('triplet', 0.0):.4f} "
                f"pose_aux={train_metrics.get('pose_aux', 0.0):.4f} "
                f"(λ={criterion.lambda_pose_aux:.3f}) "
                f"val_mAP={val_metrics.get('mAP', 0.0):.4f} "
                f"val_rank1={val_metrics.get('rank-1', 0.0):.4f}"
            )

            prefix = f"{fold_name}/"
            if writer is not None:
                for key, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{prefix}train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{prefix}val/{key}", value, epoch)
                writer.add_scalar(f"{prefix}lr", current_lr, epoch)
                writer.add_scalar(f"{prefix}lambda_pose_aux", criterion.lambda_pose_aux, epoch)
                writer.add_scalar(f"{prefix}lambda_kl", criterion.lambda_kl, epoch)
                writer.add_scalar(f"{prefix}lambda_mi", criterion.lambda_mi, epoch)
                writer.flush()

            if use_wandb:
                import wandb
                wb = {
                    "fold": fold_name, "epoch": epoch + 1, "lr": current_lr,
                    "protocol": protocol,
                }
                wb.update(
                    {f"{prefix}train/{k}": v for k, v in train_metrics.items()
                     if isinstance(v, (int, float))}
                )
                wb.update(
                    {f"{prefix}val/{k}": v for k, v in val_metrics.items()
                     if isinstance(v, (int, float))}
                )
                wandb.log(wb)

            current_val_map = float(val_metrics.get("mAP", 0.0))
            if current_val_map > best_val_map:
                best_val_map = current_val_map
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "criterion": criterion.state_dict(),
                        "optimizer_model": optimizer_model.state_dict(),
                        "optimizer_mine": (
                            optimizer_mine.state_dict() if optimizer_mine is not None else None
                        ),
                        "epoch": best_epoch,
                        "best_val_mAP": best_val_map,
                        "fold_name": fold_name,
                        "train_ids": train_ids,
                        "val_ids": val_ids,
                        "test_ids": test_ids,
                        "model_config": model_config,
                        "loss_config": loss_config,
                        "args": vars(args),
                        "protocol": protocol,
                        "strict_manifest_path": args.strict_manifest if protocol == "strict_reid" else None,
                        "fold_index": fold_spec.fold_index if hasattr(fold_spec, "fold_index") else fold_idx,
                    },
                    best_ckpt_path,
                )
                print(f"    [CKPT] best val_mAP={best_val_map:.4f} "
                      f"at epoch {best_epoch} -> {best_ckpt_path}")
            else:
                patience_counter += 1

            # TODO-20: Early stopping
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"    [EARLY STOP] No improvement for {args.patience} epochs. "
                      f"Best val_mAP={best_val_map:.4f} at epoch {best_epoch}.")
                break

        if best_state is None:
            raise RuntimeError(f"No best checkpoint found for {fold_name}.")
        model.load_state_dict(best_state)
        test_metrics = evaluate_reid(model, test_gallery_loader, test_query_loader, device)

        # 帧级质量分析导出
        if args.export_frame_quality_analysis:
            export_dir = ckpt_dir / f"{fold_name}_analysis"
            export_frame_quality_analysis(
                model, test_gallery_loader, device, "test_gallery",
                export_dir / "test_gallery_frame_quality.json", topk=args.export_topk,
            )
            export_frame_quality_analysis(
                model, test_query_loader, device, "test_query",
                export_dir / "test_query_frame_quality.json", topk=args.export_topk,
            )

        fold_result: Dict[str, float] = {
            "best_val_mAP": best_val_map, "best_epoch": float(best_epoch),
        }
        fold_result.update({f"test_{k}": v for k, v in test_metrics.items()})
        all_fold_metrics.append(fold_result)
        print(
            f"  Fold result: best_val_mAP={best_val_map:.4f}, "
            f"test_mAP={test_metrics.get('mAP', 0.0):.4f}, "
            f"test_rank1={test_metrics.get('rank-1', 0.0):.4f}"
        )

    # Cross-fold summary
    print(f"\n{'=' * 72}")
    print(f"Cross-fold summary (protocol={protocol})")
    print(f"{'=' * 72}")
    if all_fold_metrics:
        keys = sorted(all_fold_metrics[0].keys())
        summary = {}
        for key in keys:
            values = [m[key] for m in all_fold_metrics]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            summary[key] = {"mean": mean_val, "std": std_val}
            print(f"{key}: {mean_val:.4f} \u00b1 {std_val:.4f}")
        summary_path = ckpt_dir / "cross_fold_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "protocol": protocol,
                    "strict_manifest_path": args.strict_manifest if protocol == "strict_reid" else None,
                    "num_folds": len(all_fold_metrics),
                    "metrics_per_fold": all_fold_metrics,
                    "summary": summary,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved summary -> {summary_path}")

    if writer is not None:
        writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
