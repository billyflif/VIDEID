from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
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


class VideoCowClipsDataset(Dataset):
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
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {path}")
        frames: List[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize[1], self.resize[0]))
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"Empty video file: {path}")
        idxs = np.linspace(0, len(frames) - 1, num=self.frames_per_clip, dtype=int)
        sampled = [frames[i] for i in idxs]
        arr = np.stack(sampled, axis=0).astype("float32") / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        video = self._load_video(video_path)
        if self.augmentation is not None:
            video = self.augmentation(video)
        video = (video - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return video, label


def collate_fn(batch):
    videos, labels = zip(*batch)
    return torch.stack(videos, dim=0), torch.tensor(labels, dtype=torch.long)


class PKBatchSampler:
    def __init__(self, dataset: VideoCowClipsDataset, p: int = 4, k: int = 2):
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


def compute_class_weights(dataset: VideoCowClipsDataset) -> torch.Tensor:
    label_counts: Dict[int, int] = {}
    for _, label in dataset.samples:
        label_counts[label] = label_counts.get(label, 0) + 1
    num_classes = max(label_counts.keys()) + 1
    total_samples = sum(label_counts.values())
    weights = torch.ones(num_classes)
    for label, count in label_counts.items():
        weights[label] = total_samples / (num_classes * count)
    return weights


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_lambda_kl_schedule(step: int, warmup_steps: int, target: float, ramp_steps: int) -> float:
    if step < warmup_steps:
        return 0.0
    progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return target * progress


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


def build_online_folds(
    data_root: Path,
    num_folds: int,
    val_ratio: float,
    seed: int,
) -> List[OnlineFoldSpec]:
    id_dirs = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if len(id_dirs) < 3:
        raise RuntimeError(f"Need at least 3 IDs for train/val/test, found {len(id_dirs)}")
    if num_folds > len(id_dirs):
        raise RuntimeError(f"num_folds={num_folds} > num_ids={len(id_dirs)}")
    if not (0.0 < val_ratio < 0.8):
        raise RuntimeError(f"val_ratio must be in (0, 0.8), got {val_ratio}")

    rng = random.Random(seed)
    shuffled = id_dirs[:]
    rng.shuffle(shuffled)
    test_folds = split_into_folds(shuffled, num_folds)

    specs: List[OnlineFoldSpec] = []
    for fold_idx, test_ids in enumerate(test_folds):
        remain = [x for x in shuffled if x not in test_ids]
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
        gallery_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    query_loader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return gallery_loader, query_loader


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = enabled


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer_model: torch.optim.Optimizer,
    optimizer_mine: Optional[torch.optim.Optimizer],
    device: torch.device,
    monitor: Optional["UncertaintyMonitor"] = None,
) -> Dict[str, float]:
    model.train()
    criterion.train()
    total_loss = 0.0
    total_samples = 0
    loss_meter: Dict[str, float] = {}
    monitor_stats_list = []

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        if optimizer_mine is not None and getattr(model, "mine", None) is not None:
            optimizer_mine.zero_grad(set_to_none=True)
            outputs_mine = model(videos)
            mi_mine = model.mine(outputs_mine["vid_id"].detach(), outputs_mine["vid_pose"].detach())
            loss_mine = -mi_mine
            loss_mine.backward()
            torch.nn.utils.clip_grad_norm_(model.mine.parameters(), max_norm=5.0)
            optimizer_mine.step()
            optimizer_mine.zero_grad(set_to_none=True)
            set_requires_grad(model.mine, False)
        outputs = model(videos)
        loss, loss_dict = criterion(outputs, labels)
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

    avg = {key: val / max(total_samples, 1) for key, val in loss_meter.items()}
    avg["total"] = total_loss / max(total_samples, 1)
    if monitor_stats_list:
        latest = monitor_stats_list[-1]
        avg["sigma2_mean"] = latest.get("mean", 0.0)
        avg["sigma2_std"] = latest.get("std", 0.0)
        avg["sigma2_frame_variance"] = latest.get("frame_variance_mean", 0.0)
        avg["sigma2_health_status"] = latest.get("health_status", "UNKNOWN")
    return avg


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 5 video ReID training")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--fold-root", type=str, default=None)
    parser.add_argument("--run-all-folds", action="store_true", default=True)
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
    parser.add_argument("--kl-warmup-steps", type=int, default=50)
    parser.add_argument("--kl-ramp-steps", type=int, default=100)
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
    parser.add_argument("--disable-quality-gate", action="store_true")
    parser.add_argument("--disable-bidirectional", action="store_true")
    parser.add_argument("--disable-pose-stream", action="store_true")
    parser.add_argument("--disable-pose-injection", action="store_true")
    parser.add_argument("--disable-uncertainty-weighting", action="store_true")
    parser.add_argument("--disable-mi-loss", action="store_true")
    parser.add_argument("--disable-orth-loss", action="store_true")
    parser.add_argument("--disable-temp-loss", action="store_true")
    parser.add_argument("--disable-kl-loss", action="store_true")
    return parser.parse_args()


def select_fold_specs(
    all_specs: Sequence[FoldPathSpec],
    fold_index: Optional[int],
    run_all_folds: bool,
) -> List[FoldPathSpec]:
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
    ]
    return ", ".join(parts)


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    if args.fold_root is None and args.data_root is None:
        raise RuntimeError("Either --fold-root or --data-root must be provided.")

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

    if args.fold_root is not None:
        all_fold_specs = discover_fold_paths(Path(args.fold_root))
        selected_specs = select_fold_specs(all_fold_specs, args.fold_index, args.run_all_folds)
        print(f"Using prepared folds from: {args.fold_root}")
    else:
        data_root = Path(args.data_root)
        clip_stats = summarize_clip_layout(data_root)
        if clip_stats["clip1"] == 0 or clip_stats["clip2"] == 0:
            raise RuntimeError(
                f"No clip pairs found in {data_root} (clip1={clip_stats['clip1']}, clip2={clip_stats['clip2']})."
            )
        online_specs = build_online_folds(data_root, args.num_folds, args.val_ratio, args.seed)
        all_fold_specs = [
            FoldPathSpec(
                fold_name=f"fold{spec.fold_index:02d}",
                train_root=data_root,
                val_root=data_root,
                test_root=data_root,
                train_ids=spec.train_ids,
                val_ids=spec.val_ids,
                test_ids=spec.test_ids,
            )
            for spec in online_specs
        ]
        selected_specs = select_fold_specs(all_fold_specs, args.fold_index, args.run_all_folds)
        print(f"Using online split mode from: {data_root}")

    model_config = build_ablation_config(args)
    loss_config = build_loss_config(args, model_config)
    all_fold_metrics: List[Dict[str, float]] = []

    for fold_idx, fold_spec in enumerate(selected_specs, start=1):
        print(f"\n{'=' * 72}")
        print(f"Fold {fold_idx}/{len(selected_specs)}: {fold_spec.fold_name}")
        print(
            f"Train IDs={len(fold_spec.train_ids)}, "
            f"Val IDs={len(fold_spec.val_ids)}, "
            f"Test IDs={len(fold_spec.test_ids)}"
        )
        print(f"Ablations: {format_toggle_status(model_config, loss_config)}")
        print(f"{'=' * 72}")

        train_stats = summarize_clip_layout(fold_spec.train_root, fold_spec.train_ids)
        val_stats = summarize_clip_layout(fold_spec.val_root, fold_spec.val_ids)
        test_stats = summarize_clip_layout(fold_spec.test_root, fold_spec.test_ids)
        print(f"Train clips: clip1={train_stats['clip1']}, clip2={train_stats['clip2']}")
        print(f"Val clips:   clip1={val_stats['clip1']}, clip2={val_stats['clip2']}")
        print(f"Test clips:  clip1={test_stats['clip1']}, clip2={test_stats['clip2']}")
        for split_name, stats in (("train", train_stats), ("val", val_stats), ("test", test_stats)):
            if stats["clip1"] == 0 or stats["clip2"] == 0:
                raise RuntimeError(
                    f"Fold {fold_spec.fold_name} split={split_name} has missing clips: "
                    f"clip1={stats['clip1']}, clip2={stats['clip2']}"
                )

        train_dataset = VideoCowClipsDataset(
            root=fold_spec.train_root,
            id_list=fold_spec.train_ids,
            use_clip="both",
            frames_per_clip=args.frames_per_clip,
            resize=(224, 224),
            augmentation=VideoAugmentation(
                use_occlusion=True,
                use_blur=True,
                use_brightness=True,
                occlusion_prob=0.5,
                blur_prob=0.5,
                brightness_prob=0.5,
            ),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=PKBatchSampler(train_dataset, p=4, k=2),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        val_gallery_loader, val_query_loader = build_eval_loaders(
            fold_spec.val_root,
            fold_spec.val_ids,
            args.frames_per_clip,
            args.batch_size,
            args.num_workers,
            pin_memory,
        )
        test_gallery_loader, test_query_loader = build_eval_loaders(
            fold_spec.test_root,
            fold_spec.test_ids,
            args.frames_per_clip,
            args.batch_size,
            args.num_workers,
            pin_memory,
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
            margin=args.margin,
            use_batch_hard=True,
            class_weights=class_weights,
        ).to(device)

        optimizer_model = torch.optim.AdamW(
            [p for n, p in model.named_parameters() if "mine" not in n] + list(criterion.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        mine_params = [p for n, p in model.named_parameters() if "mine" in n]
        optimizer_mine = (
            torch.optim.AdamW(
                mine_params,
                lr=args.mine_lr,
                weight_decay=args.weight_decay,
            )
            if mine_params and loss_config["lambda_mi"] > 0
            else None
        )
        if optimizer_mine is None and getattr(model, "mine", None) is not None:
            set_requires_grad(model.mine, False)
        scheduler = CosineAnnealingLR(optimizer_model, T_max=args.num_epochs, eta_min=1e-6)
        monitor = UncertaintyMonitor(window_size=100, threshold=0.01, check_interval=50)

        global_step = 0
        best_val_map = -1.0
        best_epoch = -1
        best_state = None
        best_ckpt_path = ckpt_dir / f"{fold_spec.fold_name}_best.pth"

        for epoch in range(args.num_epochs):
            criterion.lambda_kl = get_lambda_kl_schedule(
                global_step, args.kl_warmup_steps, loss_config["lambda_kl"], args.kl_ramp_steps
            )
            train_metrics = train_one_epoch(
                model, criterion, train_loader, optimizer_model, optimizer_mine, device, monitor
            )
            global_step += len(train_loader)
            scheduler.step()

            val_metrics = evaluate_reid(model, val_gallery_loader, val_query_loader, device)
            current_lr = optimizer_model.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:03d}/{args.num_epochs:03d} "
                f"lr={current_lr:.2e} "
                f"loss={train_metrics.get('total', 0.0):.4f} "
                f"val_mAP={val_metrics.get('mAP', 0.0):.4f} "
                f"val_rank1={val_metrics.get('rank-1', 0.0):.4f}"
            )

            prefix = f"{fold_spec.fold_name}/"
            if writer is not None:
                for key, value in train_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{prefix}train/{key}", value, epoch)
                for key, value in val_metrics.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f"{prefix}val/{key}", value, epoch)
                writer.add_scalar(f"{prefix}lr", current_lr, epoch)
                writer.flush()

            if use_wandb:
                import wandb

                wb = {"fold": fold_spec.fold_name, "epoch": epoch + 1, "lr": current_lr}
                wb.update(
                    {f"{prefix}train/{k}": v for k, v in train_metrics.items() if isinstance(v, (int, float))}
                )
                wb.update(
                    {f"{prefix}val/{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}
                )
                wandb.log(wb)

            current_val_map = float(val_metrics.get("mAP", 0.0))
            if current_val_map > best_val_map:
                best_val_map = current_val_map
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                torch.save(
                    {
                        "model": model.state_dict(),
                        "criterion": criterion.state_dict(),
                        "optimizer_model": optimizer_model.state_dict(),
                        "optimizer_mine": optimizer_mine.state_dict() if optimizer_mine is not None else None,
                        "epoch": best_epoch,
                        "best_val_mAP": best_val_map,
                        "fold_name": fold_spec.fold_name,
                        "train_ids": fold_spec.train_ids,
                        "val_ids": fold_spec.val_ids,
                        "test_ids": fold_spec.test_ids,
                        "model_config": model_config,
                        "loss_config": loss_config,
                        "args": vars(args),
                    },
                    best_ckpt_path,
                )
                print(f"    [CKPT] best val_mAP={best_val_map:.4f} at epoch {best_epoch} -> {best_ckpt_path}")

        if best_state is None:
            raise RuntimeError(f"No best checkpoint found for {fold_spec.fold_name}.")
        model.load_state_dict(best_state)
        test_metrics = evaluate_reid(model, test_gallery_loader, test_query_loader, device)
        fold_result: Dict[str, float] = {"best_val_mAP": best_val_map, "best_epoch": float(best_epoch)}
        fold_result.update({f"test_{k}": v for k, v in test_metrics.items()})
        all_fold_metrics.append(fold_result)
        print(
            f"  Fold result: best_val_mAP={best_val_map:.4f}, "
            f"test_mAP={test_metrics.get('mAP', 0.0):.4f}, "
            f"test_rank1={test_metrics.get('rank-1', 0.0):.4f}"
        )

    print(f"\n{'=' * 72}")
    print("Cross-fold summary")
    print(f"{'=' * 72}")
    if all_fold_metrics:
        keys = sorted(all_fold_metrics[0].keys())
        summary = {}
        for key in keys:
            values = [m[key] for m in all_fold_metrics]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            summary[key] = {"mean": mean_val, "std": std_val}
            print(f"{key}: {mean_val:.4f} 卤 {std_val:.4f}")
        summary_path = ckpt_dir / "cross_fold_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
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


