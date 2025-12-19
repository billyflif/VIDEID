"""
一个简洁的训练示例脚本，演示如何使用 VideoReIDModel 与 VideoReIDCriterion。

实际项目中，你只需要替换数据加载部分和训练配置即可。

新功能：
- 数据增强（occlusion, blur, brightness）
- σ_t²变化监控
- batch-hard triplet mining（可选）
"""

from typing import Dict, Optional
import argparse
import os
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# 修复导入路径：支持作为模块和脚本两种运行方式
try:
    # 作为模块导入（python -m src.train）
    from .models.reid_model import VideoReIDModel
    from .models.losses import VideoReIDCriterion
    from .data_augmentation import VideoAugmentation
    from .monitoring import UncertaintyMonitor
except ImportError:
    # 如果作为脚本直接运行（python src/train.py），使用绝对导入
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.models.reid_model import VideoReIDModel
    from src.models.losses import VideoReIDCriterion
    from src.data_augmentation import VideoAugmentation
    from src.monitoring import UncertaintyMonitor


class DummyVideoDataset(Dataset):
    """
    用于演示训练流程的占位数据集：
    - 随机生成 (B, T, C, H, W) 的视频片段
    - 随机生成 ID 标签
    - 支持数据增强
    """

    def __init__(
        self,
        num_samples: int = 64,
        num_classes: int = 10,
        T: int = 8,
        augmentation: Optional[VideoAugmentation] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.T = T
        self.augmentation = augmentation

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # 模拟 3x224x224 的视频帧序列（归一化到[0, 1]）
        video = torch.rand(self.T, 3, 224, 224)
        label = torch.randint(0, self.num_classes, (1,)).item()
        
        # 应用数据增强（如果提供）
        if self.augmentation is not None:
            video = self.augmentation(video)
        
        return video, label


def collate_fn(batch):
    videos, labels = zip(*batch)
    videos = torch.stack(videos, dim=0)  # (B, T, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return videos, labels


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    """
    训练一个epoch（标准MINE训练：双优化器）
    
    Args:
        model: 模型
        criterion: 损失函数
        loader: 数据加载器
        optimizer_model: 主模型优化器
        optimizer_mine: MINE网络优化器
        device: 设备
        monitor: 不确定性监控器（可选）
        global_step: 全局步数（用于超参数调度）
    Returns:
        avg_dict: 平均损失字典
    """
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
        
        # ========== 标准MINE训练：双优化器 ==========
        # 1. 更新MINE网络（最大化MI估计）
        mi_est = outputs["mi"]
        loss_mine = -mi_est  # 最大化MI = 最小化负MI
        
        optimizer_mine.zero_grad()
        loss_mine.backward(retain_graph=True)
        if hasattr(model, "mine"):
            torch.nn.utils.clip_grad_norm_(model.mine.parameters(), max_norm=5.0)
        optimizer_mine.step()
        
        # 2. 更新主模型（最小化MI估计和其他损失）
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
                # 如果有健康警告，打印
                if stats.get('health_warning', 0) > 0:
                    print(f"\n⚠️  Batch {global_step + batch_idx}: {stats.get('health_status', 'UNKNOWN')}")
                    print(f"   帧间方差均值: {stats.get('frame_variance_mean', 0):.6f}")
        
        global_step += 1

    avg_loss = total_loss / max(total_samples, 1)
    avg_dict = {k: v / max(total_samples, 1) for k, v in loss_meter.items()}
    avg_dict["total"] = avg_loss
    
    # 添加监控统计信息
    if monitor_stats_list:
        latest_stats = monitor_stats_list[-1]
        avg_dict["sigma2_mean"] = latest_stats.get('mean', 0.0)
        avg_dict["sigma2_std"] = latest_stats.get('std', 0.0)
        avg_dict["sigma2_frame_variance"] = latest_stats.get('frame_variance_mean', 0.0)
        avg_dict["sigma2_health_status"] = latest_stats.get('health_status', 'UNKNOWN')
    
    return avg_dict


@torch.no_grad()
def evaluate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()
    total_loss = 0.0
    total_samples = 0
    loss_meter: Dict[str, float] = {}
    correct = 0
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)
        outputs = model(videos)
        loss, loss_dict = criterion(outputs, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        for k, v in loss_dict.items():
            loss_meter[k] = loss_meter.get(k, 0.0) + float(v) * bs
        vid_id = outputs["vid_id"]
        logits = criterion.id_loss.classifier(vid_id)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    avg_loss = total_loss / max(total_samples, 1)
    avg_dict = {k: v / max(total_samples, 1) for k, v in loss_meter.items()}
    avg_dict["total"] = avg_loss
    avg_dict["acc"] = correct / max(total_samples, 1) if total_samples > 0 else 0.0
    return avg_dict


def get_lambda_kl_schedule(
    step: int,
    warmup_steps: int = 5000,
    target: float = 0.01,
    ramp_steps: int = 5000,
) -> float:
    """
    实现λ_kl的warmup + 线性爬坡策略
    文档要求：前warmup_steps步λ_kl=0，之后逐渐增加到target
    """
    if step < warmup_steps:
        return 0.0
    ramp_progress = min(1.0, (step - warmup_steps) / max(1, ramp_steps))
    return target * ramp_progress


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--feat-dim", type=int, default=512)
    parser.add_argument("--num-train", type=int, default=32)
    parser.add_argument("--num-val", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mine-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-dir", type=str, default="runs/video_reid")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="video_reid")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    return parser.parse_args()


def main():
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

    num_classes = args.num_classes
    feat_dim = args.feat_dim
    
    # ========== 数据增强配置 ==========
    augmentation = VideoAugmentation(
        use_occlusion=True,
        use_blur=True,
        use_brightness=True,
        occlusion_prob=0.5,
        blur_prob=0.5,
        brightness_prob=0.5,
    )
    
    # ========== 模型和损失函数 ==========
    model = VideoReIDModel(feat_dim=feat_dim, num_blocks=8).to(device)  # 文档建议 N=6~12
    
    # 使用batch-hard triplet mining（可选）
    use_batch_hard = True  # 设置为False使用简单采样
    
    criterion = VideoReIDCriterion(
        feat_dim=feat_dim,
        num_classes=num_classes,
        lambda_mi=0.1,
        lambda_orth=0.01,
        lambda_temp=0.1,
        lambda_kl=0.01,  # 初始值，实际会通过调度更新
        use_batch_hard=use_batch_hard,
    ).to(device)

    # ========== 标准MINE训练：双优化器 ==========
    # 主模型优化器：优化除MINE网络外的所有参数
    model_params = [p for n, p in model.named_parameters() if 'mine' not in n]
    optimizer_model = torch.optim.AdamW(
        model_params + list(criterion.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    # MINE网络优化器：只优化MINE网络参数
    mine_params = [p for n, p in model.named_parameters() if 'mine' in n]
    optimizer_mine = torch.optim.AdamW(
        mine_params,
        lr=args.mine_lr,  # MINE网络通常使用稍大的学习率
        weight_decay=args.weight_decay,
    )
    
    # ========== 不确定性监控器 ==========
    monitor = UncertaintyMonitor(
        window_size=100,
        threshold=0.01,
        check_interval=50,
    )

    # ========== 数据集和数据加载器 ==========
    full_dataset = DummyVideoDataset(
        num_samples=args.num_train + args.num_val,
        num_classes=num_classes,
        T=8,
        augmentation=augmentation,  # 应用数据增强
    )
    train_size = args.num_train
    val_size = args.num_val
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ========== 训练循环 ==========
    global_step = 0
    num_epochs = args.num_epochs
    best_val_acc = 0.0
    
    print("=" * 60)
    print("开始训练")
    print(f"使用batch-hard triplet mining: {use_batch_hard}")
    print(f"数据增强: occlusion={augmentation.occlusion is not None}, "
          f"blur={augmentation.blur is not None}, "
          f"brightness={augmentation.brightness is not None}")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 更新λ_kl（warmup策略）
        current_lambda_kl = get_lambda_kl_schedule(global_step, warmup_steps=5000)
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

        val_metrics = evaluate(
            model,
            criterion,
            val_loader,
            device,
        )
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} (step {global_step}):")
        print(f"  训练损失: {train_metrics.get('total', 0):.4f}")
        print(f"  验证损失: {val_metrics.get('total', 0):.4f}")
        print(f"  验证准确率: {val_metrics.get('acc', 0):.4f}")
        print(f"  ID损失: {train_metrics.get('id', 0):.4f}")
        print(f"  三元组损失: {train_metrics.get('triplet', 0):.4f}")
        print(f"  MI损失: {train_metrics.get('mi', 0):.4f}")
        print(f"  正交损失: {train_metrics.get('orth', 0):.4f}")
        print(f"  时序平滑: {train_metrics.get('temp', 0):.4f}")
        print(f"  KL损失: {train_metrics.get('kl', 0):.4f} (λ={current_lambda_kl:.4f})")
        
        for k, v in train_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_metrics.items():
            if isinstance(v, (int, float)):
                writer.add_scalar(f"val/{k}", v, epoch)
        writer.flush()

        if use_wandb:
            log_data = {}
            for k, v in train_metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"train/{k}"] = v
            for k, v in val_metrics.items():
                if isinstance(v, (int, float)):
                    log_data[f"val/{k}"] = v
            log_data["epoch"] = epoch + 1
            log_data["lambda_kl"] = current_lambda_kl
            wandb.log(log_data)

        val_acc = val_metrics.get("acc", 0.0)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = ckpt_dir / "best.pth"
            torch.save(
                {
                    "model": model.state_dict(),
                    "criterion": criterion.state_dict(),
                    "optimizer_model": optimizer_model.state_dict(),
                    "optimizer_mine": optimizer_mine.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                ckpt_path,
            )

    # 打印最终监控摘要
    print("\n" + "=" * 60)
    print("训练完成 - 监控摘要")
    print("=" * 60)
    summary = monitor.get_summary()
    print(f"总batch数: {summary['total_batches']}")
    print(f"警告数量: {summary['num_warnings']}")
    if 'overall_mean' in summary:
        print(f"σ²总体统计:")
        print(f"  均值: {summary['overall_mean']:.6f}")
        print(f"  标准差: {summary['overall_std']:.6f}")
        print(f"  最小值: {summary['overall_min']:.6f}")
        print(f"  最大值: {summary['overall_max']:.6f}")
    
    if summary['warnings']:
        print(f"\n最近警告:")
        for warning in summary['warnings'][-5:]:
            print(f"  - {warning}")


if __name__ == "__main__":
    main()


