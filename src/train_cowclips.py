"""
基于 cow clips 的视频 ReID 训练脚本。

数据组织：
    dataset_root/
        id0001/
            0001_clip1.mp4
            0001_clip2.mp4
        id0002/
            0002_clip1.mp4
            0002_clip2.mp4

其中 *_clip1.mp4 和 *_clip2.mp4 来自 video_clip.py 对原始 0001.mp4 的前/后半段切分。

训练：
    - 使用 train_ids 中所有 ID 的 clip1 + clip2 作为训练样本。

验证（ReID 评估）：
    - 使用 test_ids 中 ID 的 clip1 作为 gallery。
    - 使用同一批 ID 的 clip2 作为 query。
    - 计算基于视频级特征 vid_id 的检索指标：Top-1 / Top-5 / Top-10 准确率和 mAP。

运行示例：
    python -m src.train_cowclips \
        --data-root ./dataset_root \
        --num-epochs 20 \
        --batch-size 8 \
        --device cuda
"""

from typing import Dict, List, Optional, Sequence, Tuple
import argparse
import random
from pathlib import Path

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

# 修复导入路径：支持作为模块和脚本两种运行方式
try:
    # 作为模块导入（python -m src.train_cowclips）
    from .models.reid_model import VideoReIDModel
    from .models.losses import VideoReIDCriterion
    from .data_augmentation import VideoAugmentation
    from .monitoring import UncertaintyMonitor
except ImportError:  # pragma: no cover - 兼容脚本直接运行
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).parent.parent))
    from src.models.reid_model import VideoReIDModel
    from src.models.losses import VideoReIDCriterion
    from src.data_augmentation import VideoAugmentation
    from src.monitoring import UncertaintyMonitor


class VideoCowClipsDataset(Dataset):
    """从 cow clips 数据集中读取视频片段。

    期待目录结构：
        root/idXXXX/*_clip1.mp4
        root/idXXXX/*_clip2.mp4

    Args:
        root: 数据根目录
        id_list: 本数据集中使用的 ID 目录名列表
        use_clip: "clip1" | "clip2" | "both"，决定使用哪些 clip
        frames_per_clip: 采样多少帧作为输入 T
        resize: (H, W)，统一 resize 到的分辨率
        augmentation: 视频增强（仅训练集使用），接收 (T, C, H, W) 的 torch.Tensor
    """

    def __init__(
        self,
        root: Path,
        id_list: Sequence[str],
        use_clip: str = "both",
        frames_per_clip: int = 8,
        resize: Tuple[int, int] = (224, 224),
        augmentation: Optional[VideoAugmentation] = None,
    ) -> None:
        super().__init__()
        assert use_clip in {"clip1", "clip2", "both"}
        self.root = Path(root)
        self.id_list = list(id_list)
        self.use_clip = use_clip
        self.frames_per_clip = frames_per_clip
        self.resize = resize
        self.augmentation = augmentation

        # 建立 id_name -> label 映射
        self.id_list.sort()
        self.id2label = {id_name: idx for idx, id_name in enumerate(self.id_list)}

        # 收集所有样本 (video_path, label)
        self.samples: List[Tuple[Path, int]] = []
        for id_name in self.id_list:
            id_dir = self.root / id_name
            if not id_dir.is_dir():
                continue
            clip1_paths = sorted(id_dir.glob("*_clip1.mp4"))
            clip2_paths = sorted(id_dir.glob("*_clip2.mp4"))

            if self.use_clip in {"clip1", "both"}:
                for p in clip1_paths:
                    self.samples.append((p, self.id2label[id_name]))
            if self.use_clip in {"clip2", "both"}:
                for p in clip2_paths:
                    self.samples.append((p, self.id2label[id_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"在 {root} 下未找到任何 *_clip[12].mp4 文件，请先运行 video_clip.py 进行切分。")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_video(self, path: Path) -> torch.Tensor:
        """读取视频并采样为 (T, 3, H, W) 张量，像素范围 [0, 1]。"""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {path}")

        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.resize[1], self.resize[0]))  # (W, H)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            raise RuntimeError(f"视频无帧: {path}")

        # 采样 frames_per_clip 帧
        T = self.frames_per_clip
        idxs = np.linspace(0, len(frames) - 1, num=T, dtype=int)
        sampled = [frames[i] for i in idxs]

        arr = np.stack(sampled, axis=0).astype("float32") / 255.0  # (T, H, W, 3)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (T, 3, H, W)
        return tensor

    # ImageNet归一化参数（匹配预训练ResNet50）
    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        video = self._load_video(video_path)
        if self.augmentation is not None:
            video = self.augmentation(video)
        # ImageNet标准归一化（预训练ResNet50需要）
        video = (video - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return video, label


def collate_fn(batch):
    videos, labels = zip(*batch)
    videos = torch.stack(videos, dim=0)  # (B, T, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return videos, labels


class PKBatchSampler:
    """PK采样器：每个batch采样P个ID，每个ID采样K个样本。

    确保每个batch中至少有P个不同的ID各K个正样本，
    使triplet loss在小batch场景下仍然有效。
    """

    def __init__(self, dataset: VideoCowClipsDataset, p: int = 4, k: int = 2):
        self.p = p
        self.k = k
        self.batch_size = p * k

        # 构建 label -> indices 映射
        self.label_to_indices: Dict[int, List[int]] = {}
        for idx, (_, label) in enumerate(dataset.samples):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())
        if len(self.labels) < p:
            self.p = len(self.labels)
            self.batch_size = self.p * k

        self._num_batches = max(1, len(dataset) // self.batch_size)

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = []
            selected_labels = random.sample(self.labels, min(self.p, len(self.labels)))
            for label in selected_labels:
                indices = self.label_to_indices[label]
                if len(indices) >= self.k:
                    selected = random.sample(indices, self.k)
                else:
                    # 样本不足时过采样
                    selected = random.choices(indices, k=self.k)
                batch.extend(selected)
            yield batch

    def __len__(self):
        return self._num_batches


def compute_class_weights(dataset: VideoCowClipsDataset) -> torch.Tensor:
    """根据训练集样本分布计算类权重（逆频率加权），处理类不均衡问题。"""
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


def build_id_splits(root: Path, train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """根据 ID 目录划分 train_ids 和 test_ids。"""
    id_dirs = [p.name for p in root.iterdir() if p.is_dir()]
    if not id_dirs:
        raise RuntimeError(f"在 {root} 下未找到任何 id 目录")

    id_dirs.sort()
    rng = random.Random(seed)
    rng.shuffle(id_dirs)

    n_ids = len(id_dirs)
    n_train = max(1, int(n_ids * train_ratio))
    n_train = min(n_train, n_ids - 1)  # 保证有至少1个test id

    train_ids = id_dirs[:n_train]
    test_ids = id_dirs[n_train:]
    return train_ids, test_ids


def build_kfold_splits(root: Path, num_folds: int, seed: int) -> List[Tuple[List[str], List[str]]]:
    """构建k-fold交叉验证的ID划分。

    Args:
        root: 数据根目录
        num_folds: 折数
        seed: 随机种子
    Returns:
        folds: 列表，每个元素为 (train_ids, test_ids) 元组
    """
    id_dirs = sorted([p.name for p in root.iterdir() if p.is_dir()])
    if not id_dirs:
        raise RuntimeError(f"在 {root} 下未找到任何ID目录")

    rng = random.Random(seed)
    rng.shuffle(id_dirs)

    folds = []
    fold_size = len(id_dirs) // num_folds
    remainder = len(id_dirs) % num_folds

    start = 0
    for i in range(num_folds):
        end = start + fold_size + (1 if i < remainder else 0)
        test_ids = id_dirs[start:end]
        train_ids = [d for d in id_dirs if d not in test_ids]
        folds.append((train_ids, test_ids))
        start = end

    return folds


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    optimizer_model: torch.optim.Optimizer,
    optimizer_mine: torch.optim.Optimizer,
    device: torch.device,
    monitor: Optional[UncertaintyMonitor] = None,
    global_step: int = 0,
) -> Dict[str, float]:
    """训练一个 epoch（标准 MINE 双优化器训练）。"""
    model.train()
    criterion.train()

    total_loss = 0.0
    total_samples = 0
    loss_meter: Dict[str, float] = {}
    monitor_stats_list = []

    for batch_idx, (videos, labels) in enumerate(loader):
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)

        # 1) 更新 MINE 网络（最大化 MI 估计）
        mi_est = outputs["mi"]
        loss_mine = -mi_est

        optimizer_mine.zero_grad()
        loss_mine.backward(retain_graph=True)
        if hasattr(model, "mine"):
            torch.nn.utils.clip_grad_norm_(model.mine.parameters(), max_norm=5.0)
        optimizer_mine.step()

        # 2) 更新主模型（最小化 MI + 其他损失）
        loss, loss_dict = criterion(outputs, labels)

        optimizer_model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer_model.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        for k, v in loss_dict.items():
            loss_meter[k] = loss_meter.get(k, 0.0) + float(v) * bs

        # 更新监控器
        if monitor is not None:
            sigma2 = outputs["sigma2"]  # (B, T, 1)
            stats = monitor.update(sigma2)
            if stats is not None:
                monitor_stats_list.append(stats)

    avg_loss = total_loss / max(total_samples, 1)
    avg_dict = {k: v / max(total_samples, 1) for k, v in loss_meter.items()}
    avg_dict["total"] = avg_loss

    if monitor_stats_list:
        latest_stats = monitor_stats_list[-1]
        avg_dict["sigma2_mean"] = latest_stats.get("mean", 0.0)
        avg_dict["sigma2_std"] = latest_stats.get("std", 0.0)
        avg_dict["sigma2_frame_variance"] = latest_stats.get("frame_variance_mean", 0.0)
        avg_dict["sigma2_health_status"] = latest_stats.get("health_status", "UNKNOWN")

    return avg_dict


@torch.no_grad()
def extract_video_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """提取视频级特征 vid_id 及其标签。

    Returns:
        feats: (N, D)
        labels: (N,)
    """
    model.eval()
    all_feats: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(videos)
        vid_id = outputs["vid_id"]  # (B, D)
        all_feats.append(vid_id.detach().cpu())
        all_labels.append(labels.detach().cpu())

    feats = torch.cat(all_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return feats, labels


@torch.no_grad()
def compute_reid_metrics(
    query_feats: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_feats: torch.Tensor,
    gallery_labels: torch.Tensor,
    topk: Sequence[int] = (1, 5, 10),
) -> Dict[str, float]:
    """基于余弦距离的简单 ReID 检索评估：CMC 与 mAP。

    假设：每个 query 在 gallery 中有且仅有一个同 ID 的样本。
    """
    # 归一化特征
    q = torch.nn.functional.normalize(query_feats, dim=1)
    g = torch.nn.functional.normalize(gallery_feats, dim=1)

    # 余弦相似度越大越近，这里使用负相似度当作距离
    sim = torch.matmul(q, g.t())  # (Nq, Ng)

    num_q = q.size(0)
    device = sim.device

    # 为了统一逻辑，取相似度从大到小排序
    indices = torch.argsort(sim, dim=1, descending=True)  # (Nq, Ng)

    max_rank = max(topk)
    cmc = torch.zeros(max_rank, dtype=torch.float32)
    all_ap: List[float] = []

    for i in range(num_q):
        q_label = query_labels[i].item()
        order = indices[i]  # gallery 排名索引
        matches = (gallery_labels[order] == q_label).float()  # (Ng,)

        # 找到第一个正确匹配的位置
        correct_positions = torch.nonzero(matches, as_tuple=False).view(-1)
        if len(correct_positions) == 0:
            all_ap.append(0.0)
            continue

        first_pos = correct_positions[0].item()

        # 更新 CMC
        if first_pos < max_rank:
            cmc[first_pos:] += 1

        # 由于每个 query 只有一个正样本，AP = 1 / (rank_position + 1)
        ap = 1.0 / float(first_pos + 1)
        all_ap.append(ap)

    cmc = cmc / max(1, num_q)
    mAP = float(np.mean(all_ap)) if all_ap else 0.0

    metrics: Dict[str, float] = {f"rank-{k}": float(cmc[k - 1]) for k in topk if k <= max_rank}
    metrics["mAP"] = mAP
    return metrics


def get_lambda_kl_schedule(
    step: int,
    warmup_steps: int = 5000,
    target: float = 0.01,
    ramp_steps: int = 5000,
) -> float:
    """KL 权重的 warmup + 线性爬坡策略。"""
    if step < warmup_steps:
        return 0.0
    ramp_progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return target * ramp_progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 cow clips 的视频 ReID 训练脚本")
    parser.add_argument("--data-root", type=str, required=True, help="数据根目录，例如 ./dataset_root")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--train-ratio", type=float, default=0.8, help="用于训练的 ID 比例（仅单折时使用）")
    parser.add_argument("--num-folds", type=int, default=5, help="交叉验证折数（1=单次划分，>1=k-fold交叉验证）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mine-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="runs/video_reid_cowclips")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_cowclips")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="video_reid_cowclips")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    set_seed(args.seed)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    use_wandb = args.use_wandb
    if use_wandb:
        try:
            import wandb

            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        except ImportError:
            use_wandb = False

    data_root = Path(args.data_root)

    # ========== 构建交叉验证划分 ==========
    if args.num_folds > 1:
        folds = build_kfold_splits(data_root, num_folds=args.num_folds, seed=args.seed)
        print(f"使用 {args.num_folds}-fold 交叉验证")
    else:
        train_ids, test_ids = build_id_splits(data_root, train_ratio=args.train_ratio, seed=args.seed)
        folds = [(train_ids, test_ids)]
        print(f"使用单次划分 (train_ratio={args.train_ratio})")

    all_fold_metrics: List[Dict[str, float]] = []

    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        num_classes = len(train_ids)  # 分类器仅覆盖训练ID

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{len(folds)} | Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")
        print(f"{'='*60}")

        # ========== 数据增强配置 ==========
        augmentation = VideoAugmentation(
            use_occlusion=True,
            use_blur=True,
            use_brightness=True,
            occlusion_prob=0.5,
            blur_prob=0.5,
            brightness_prob=0.5,
        )

        # ========== 构建数据集 ==========
        train_dataset = VideoCowClipsDataset(
            root=data_root,
            id_list=train_ids,
            use_clip="both",
            frames_per_clip=args.frames_per_clip,
            resize=(224, 224),
            augmentation=augmentation,
        )

        gallery_dataset = VideoCowClipsDataset(
            root=data_root,
            id_list=test_ids,
            use_clip="clip1",
            frames_per_clip=args.frames_per_clip,
            resize=(224, 224),
            augmentation=None,
        )

        query_dataset = VideoCowClipsDataset(
            root=data_root,
            id_list=test_ids,
            use_clip="clip2",
            frames_per_clip=args.frames_per_clip,
            resize=(224, 224),
            augmentation=None,
        )

        # ========== PK采样器：保证每个batch有P个ID各K个正样本 ==========
        pk_sampler = PKBatchSampler(train_dataset, p=4, k=2)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=pk_sampler,
            num_workers=0,
            collate_fn=collate_fn,
        )
        gallery_loader = DataLoader(
            gallery_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
        query_loader = DataLoader(
            query_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )

        # ========== 计算类权重（处理类不均衡） ==========
        class_weights = compute_class_weights(train_dataset).to(device)

        # ========== 模型与损失函数 ==========
        feat_dim = args.feat_dim
        model = VideoReIDModel(feat_dim=feat_dim, num_blocks=4).to(device)

        criterion = VideoReIDCriterion(
            feat_dim=feat_dim,
            num_classes=num_classes,
            lambda_mi=0.1,
            lambda_orth=0.01,
            lambda_temp=0.1,
            lambda_kl=0.01,
            use_batch_hard=True,
            class_weights=class_weights,
        ).to(device)

        # ========== 优化器 ==========
        model_params = [p for n, p in model.named_parameters() if "mine" not in n]
        optimizer_model = torch.optim.AdamW(
            model_params + list(criterion.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        mine_params = [p for n, p in model.named_parameters() if "mine" in n]
        optimizer_mine = torch.optim.AdamW(
            mine_params,
            lr=args.mine_lr,
            weight_decay=args.weight_decay,
        )

        # ========== 学习率调度器：余弦退火 ==========
        scheduler = CosineAnnealingLR(optimizer_model, T_max=args.num_epochs, eta_min=1e-6)

        monitor = UncertaintyMonitor(
            window_size=100,
            threshold=0.01,
            check_interval=50,
        )

        # ========== 训练循环 ==========
        global_step = 0
        best_map = 0.0

        for epoch in range(args.num_epochs):
            # KL warmup：50步后开始，150步达到全额（适配小数据量）
            current_lambda_kl = get_lambda_kl_schedule(
                global_step, warmup_steps=50, ramp_steps=100
            )
            criterion.lambda_kl = current_lambda_kl

            train_metrics = train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer_model,
                optimizer_mine,
                device,
                monitor=monitor,
                global_step=global_step,
            )
            global_step += len(train_loader)
            scheduler.step()  # 更新学习率

            # 提取测试集特征并计算 ReID 指标
            gallery_feats, gallery_labels = extract_video_features(model, gallery_loader, device)
            query_feats, query_labels = extract_video_features(model, query_loader, device)
            reid_metrics = compute_reid_metrics(query_feats, query_labels, gallery_feats, gallery_labels)

            current_lr = optimizer_model.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}/{args.num_epochs} (step {global_step}, lr={current_lr:.2e}):")
            print(f"    Loss: {train_metrics.get('total', 0):.4f}, "
                  + ", ".join(f"{k}={v:.4f}" for k, v in reid_metrics.items()))

            # TensorBoard 日志
            prefix = f"fold{fold_idx}/" if len(folds) > 1 else ""
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"{prefix}train/{k}", v, epoch)
            for k, v in reid_metrics.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(f"{prefix}reid/{k}", v, epoch)
            writer.add_scalar(f"{prefix}lr", current_lr, epoch)
            writer.flush()

            # 可选 wandb 日志
            if use_wandb:
                import wandb

                log_data = {f"{prefix}train/{k}": v for k, v in train_metrics.items()
                            if isinstance(v, (int, float))}
                log_data.update({f"{prefix}reid/{k}": v for k, v in reid_metrics.items()
                                 if isinstance(v, (int, float))})
                log_data["epoch"] = epoch + 1
                log_data["fold"] = fold_idx
                log_data["lambda_kl"] = current_lambda_kl
                log_data["lr"] = current_lr
                wandb.log(log_data)

            # 保存最佳模型（按 mAP）
            cur_map = reid_metrics.get("mAP", 0.0)
            if cur_map > best_map:
                best_map = cur_map
                ckpt_path = ckpt_dir / f"best_fold{fold_idx}.pth"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "criterion": criterion.state_dict(),
                        "optimizer_model": optimizer_model.state_dict(),
                        "optimizer_mine": optimizer_mine.state_dict(),
                        "epoch": epoch + 1,
                        "best_mAP": best_map,
                        "fold": fold_idx,
                        "train_ids": train_ids,
                        "test_ids": test_ids,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"    [CKPT] best mAP={best_map:.4f} -> {ckpt_path}")

        # 记录本折最终评估指标
        gallery_feats, gallery_labels = extract_video_features(model, gallery_loader, device)
        query_feats, query_labels = extract_video_features(model, query_loader, device)
        fold_metrics = compute_reid_metrics(query_feats, query_labels, gallery_feats, gallery_labels)
        fold_metrics["best_mAP"] = best_map
        all_fold_metrics.append(fold_metrics)
        print(f"\n  Fold {fold_idx+1} 最佳 mAP: {best_map:.4f}")

    # ========== 汇总所有折的指标 ==========
    print(f"\n{'='*60}")
    print("交叉验证汇总")
    print(f"{'='*60}")
    if all_fold_metrics:
        metric_keys = list(all_fold_metrics[0].keys())
        for key in metric_keys:
            values = [m[key] for m in all_fold_metrics]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {key}: {mean_val:.4f} ± {std_val:.4f}")

    writer.close()
    print("\n训练完成。")


if __name__ == "__main__":
    main()
