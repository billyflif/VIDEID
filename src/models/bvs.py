import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class BayesianVisualStem(nn.Module):
    """
    贝叶斯视觉前端 (BVS)
    - 使用 ResNet-50 提取帧特征
    - 输出：均值特征 mu (B, T, D)，不确定性 sigma2 (B, T, 1) 或 (B, T, D)
    """

    def __init__(self, feat_dim: int = 512, pretrained: bool = True, uncertainty_dim: str = "scalar"):
        """
        Args:
            feat_dim: 特征维度
            pretrained: 是否使用预训练权重
            uncertainty_dim: 不确定性输出维度，"scalar"输出(B, T, 1)，"vector"输出(B, T, feat_dim)
        """
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
        
        # Mean Head: 使用Linear层（符合文档要求）
        self.mu_head = nn.Linear(in_channels, feat_dim)

        # Variance Head: 支持标量和向量输出（符合文档要求）
        self.uncertainty_dim = uncertainty_dim
        if uncertainty_dim == "scalar":
            # 标量不确定性：σ_t² ∈ R¹
            self.var_head = nn.Linear(in_channels, 1)
        elif uncertainty_dim == "vector":
            # 向量不确定性：σ_t² ∈ R^(d_model)，更细粒度的不确定性建模
            self.var_head = nn.Linear(in_channels, feat_dim)
        else:
            raise ValueError(f"Unsupported uncertainty_dim: {uncertainty_dim}, must be 'scalar' or 'vector'")

        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            mu: (B, T, D)
            sigma2: (B, T, 1) 或 (B, T, D)，取决于uncertainty_dim
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        feat = self.stem(x)  # (B*T, C', H', W')
        feat = self.global_pool(feat)  # (B*T, C', 1, 1)
        feat = feat.view(b * t, -1)  # (B*T, C') - 展平用于Linear层

        # Mean Head: Linear层（符合文档要求）
        mu = self.mu_head(feat)  # (B*T, D)
        mu = mu.view(b, t, -1)  # (B, T, D)

        # Variance Head: 支持标量和向量输出
        log_var = self.var_head(feat)  # (B*T, 1) 或 (B*T, D)
        sigma2 = self.softplus(log_var)
        
        if self.uncertainty_dim == "scalar":
            sigma2 = sigma2.view(b, t, 1)  # (B, T, 1)
        else:  # vector
            sigma2 = sigma2.view(b, t, -1)  # (B, T, D)

        return mu, sigma2


