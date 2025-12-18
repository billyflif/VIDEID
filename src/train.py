"""
一个简洁的训练示例脚本，演示如何使用 VideoReIDModel 与 VideoReIDCriterion。

实际项目中，你只需要替换数据加载部分和训练配置即可。
"""

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from . import VideoReIDModel, VideoReIDCriterion


class DummyVideoDataset(Dataset):
    """
    用于演示训练流程的占位数据集：
    - 随机生成 (B, T, C, H, W) 的视频片段
    - 随机生成 ID 标签
    """

    def __init__(self, num_samples: int = 64, num_classes: int = 10, T: int = 8):
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.T = T

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # 模拟 3x224x224 的视频帧序列
        video = torch.randn(self.T, 3, 224, 224)
        label = torch.randint(0, self.num_classes, (1,)).item()
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
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    criterion.train()

    total_loss = 0.0
    total_samples = 0
    loss_meter: Dict[str, float] = {}

    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        loss, loss_dict = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

        for k, v in loss_dict.items():
            loss_meter[k] = loss_meter.get(k, 0.0) + float(v) * bs

    avg_loss = total_loss / max(total_samples, 1)
    avg_dict = {k: v / max(total_samples, 1) for k, v in loss_meter.items()}
    avg_dict["total"] = avg_loss
    return avg_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = 10
    feat_dim = 512

    model = VideoReIDModel(feat_dim=feat_dim, num_blocks=4).to(device)
    criterion = VideoReIDCriterion(
        feat_dim=feat_dim,
        num_classes=num_classes,
        lambda_mi=0.1,
        lambda_orth=0.01,
        lambda_temp=0.1,
        lambda_kl=0.01,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=3e-4,
        weight_decay=1e-4,
    )

    dataset = DummyVideoDataset(num_samples=32, num_classes=num_classes, T=8)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    for epoch in range(2):
        loss_dict = train_one_epoch(model, criterion, loader, optimizer, device)
        print(f"Epoch {epoch}:")
        print({k: round(v, 4) for k, v in loss_dict.items()})


if __name__ == "__main__":
    main()


