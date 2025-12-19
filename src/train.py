"""
一个简洁的训练示例脚本，演示如何使用 VideoReIDModel 与 VideoReIDCriterion。

实际项目中，你只需要替换数据加载部分和训练配置即可。

新功能：
- 数据增强（occlusion, blur, brightness）
- σ_t²变化监控
- batch-hard triplet mining（可选）
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        optimizer_mine.step()
        
        # 2. 更新主模型（最小化MI估计和其他损失）
        loss, loss_dict = criterion(outputs, labels)
        
        optimizer_model.zero_grad()
        loss.backward()
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10
    feat_dim = 512
    
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
        lr=3e-4,
        weight_decay=1e-4,
    )
    
    # MINE网络优化器：只优化MINE网络参数
    mine_params = [p for n, p in model.named_parameters() if 'mine' in n]
    optimizer_mine = torch.optim.AdamW(
        mine_params,
        lr=1e-3,  # MINE网络通常使用稍大的学习率
        weight_decay=1e-4,
    )
    
    # ========== 不确定性监控器 ==========
    monitor = UncertaintyMonitor(
        window_size=100,
        threshold=0.01,
        check_interval=50,
    )

    # ========== 数据集和数据加载器 ==========
    dataset = DummyVideoDataset(
        num_samples=32,
        num_classes=num_classes,
        T=8,
        augmentation=augmentation,  # 应用数据增强
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ========== 训练循环 ==========
    global_step = 0
    num_epochs = 2
    
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
        
        loss_dict = train_one_epoch(
            model,
            criterion,
            loader,
            optimizer_model,
            optimizer_mine,
            device,
            monitor=monitor,
            global_step=global_step,
        )
        
        global_step += len(loader)
        
        print(f"\nEpoch {epoch + 1}/{num_epochs} (step {global_step}):")
        print(f"  损失: {loss_dict.get('total', 0):.4f}")
        print(f"  ID损失: {loss_dict.get('id', 0):.4f}")
        print(f"  三元组损失: {loss_dict.get('triplet', 0):.4f}")
        print(f"  MI损失: {loss_dict.get('mi', 0):.4f}")
        print(f"  正交损失: {loss_dict.get('orth', 0):.4f}")
        print(f"  时序平滑: {loss_dict.get('temp', 0):.4f}")
        print(f"  KL损失: {loss_dict.get('kl', 0):.4f} (λ={current_lambda_kl:.4f})")
        
        # 打印监控信息
        if 'sigma2_mean' in loss_dict:
            print(f"\n  σ²统计:")
            print(f"    均值: {loss_dict.get('sigma2_mean', 0):.6f}")
            print(f"    标准差: {loss_dict.get('sigma2_std', 0):.6f}")
            print(f"    帧间方差: {loss_dict.get('sigma2_frame_variance', 0):.6f}")
            print(f"    健康状态: {loss_dict.get('sigma2_health_status', 'UNKNOWN')}")
    
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


