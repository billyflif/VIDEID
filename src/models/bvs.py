import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

try:
    from torchvision.models import ResNet50_Weights
except Exception:
    ResNet50_Weights = None


class BayesianVisualStem(nn.Module):
    """
    贝叶斯视觉前端 (BVS)
    - 使用 ResNet-50 提取帧特征
    - 输出：均值特征 mu (B, T, D)，不确定性 sigma2 (B, T, 1) 或 (B, T, D)
    """

    def __init__(self, feat_dim: int = 512, pretrained: bool = True, uncertainty_dim: str = "scalar",
                 freeze_layers: int = 3, dropout: float = 0.5):
        """
        Args:
            feat_dim: 特征维度
            pretrained: 是否使用预训练权重
            uncertainty_dim: 不确定性输出维度，"scalar"输出(B, T, 1)，"vector"输出(B, T, feat_dim)
            freeze_layers: 冻结ResNet前N个layer（0=不冻结, 3=冻结conv1+bn1+layer1-3）
            dropout: Dropout概率，防止小数据集过拟合
        """
        super().__init__()
        if ResNet50_Weights is None:
            backbone = resnet50(pretrained=pretrained)
        else:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            backbone = resnet50(weights=weights)
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

        # 冻结预训练层以防止小数据过拟合
        # freeze_layers=3: 冻结 conv1, bn1, layer1, layer2, layer3，仅训练 layer4 + heads
        if freeze_layers > 0:
            frozen_parts = [backbone.conv1, backbone.bn1]
            layer_list = [backbone.layer1, backbone.layer2, backbone.layer3]
            for i in range(min(freeze_layers, len(layer_list))):
                frozen_parts.append(layer_list[i])
            for module in frozen_parts:
                for param in module.parameters():
                    param.requires_grad = False

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        
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
        mu = self.dropout(mu)    # Dropout防止过拟合
        mu = mu.view(b, t, -1)  # (B, T, D)

        # Variance Head: 支持标量和向量输出
        log_var = self.var_head(feat)  # (B*T, 1) 或 (B*T, D)
        sigma2 = self.softplus(log_var)
        
        if self.uncertainty_dim == "scalar":
            sigma2 = sigma2.view(b, t, 1)  # (B, T, 1)
        else:  # vector
            sigma2 = sigma2.view(b, t, -1)  # (B, T, D)

        return mu, sigma2


