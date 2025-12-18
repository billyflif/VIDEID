import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class BayesianVisualStem(nn.Module):
    """
    贝叶斯视觉前端 (BVS)
    - 使用 ResNet-50 提取帧特征
    - 输出：均值特征 mu (B, T, D)，不确定性 sigma2 (B, T, 1)
    """

    def __init__(self, feat_dim: int = 512, pretrained: bool = True):
        super().__init__()
        backbone = resnet50(pretrained=pretrained)
        # 去掉最后的池化与全连接层
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        in_channels = backbone.fc.in_features

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.mu_head = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

        # 输出标量不确定性（广义质量因子）
        self.var_head = nn.Sequential(
            nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, 1, kernel_size=1, bias=True),
        )

        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            mu: (B, T, D)
            sigma2: (B, T, 1)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feat = self.stem(x)  # (B*T, C', H', W')
        feat = self.global_pool(feat)  # (B*T, C', 1, 1)

        mu = self.mu_head(feat)  # (B*T, D, 1, 1)
        mu = mu.view(b, t, -1)

        log_var = self.var_head(feat)  # (B*T, 1, 1, 1)
        sigma2 = self.softplus(log_var).view(b, t, 1)

        return mu, sigma2


