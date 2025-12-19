from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    梯度反转层 (Gradient Reversal Layer, GRL)
    用于MINE的对抗训练：主模型尝试最小化MI，MINE网络尝试最大化MI估计
    """
    @staticmethod
    def forward(ctx, x, lambda_grl=1.0):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_grl, None


class GradientReversalLayer(nn.Module):
    """
    梯度反转层的Module包装
    """
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_grl)


class UncertaintyWeightedAggregator(nn.Module):
    """
    使用不确定性进行权重聚合：
        w_t = exp(-σ_t²) / Σ exp(-σ_t²)  （符合文档要求）
        F_video = Σ_t w_t · H_out[t]
    """

    def __init__(self):
        super().__init__()

    def forward(self, h: torch.Tensor, sigma2: torch.Tensor):
        """
        Args:
            h: (B, T, D)
            sigma2: (B, T, 1) 或 (B, T, D) - 支持标量和向量不确定性
        Returns:
            feat: (B, D)
            weights: (B, T, 1)
        """
        # 如果sigma2是向量，需要先聚合为标量用于权重计算
        if sigma2.dim() == 3 and sigma2.size(-1) > 1:
            # 向量不确定性：取均值或L2范数作为权重依据
            sigma2_scalar = sigma2.norm(dim=-1, keepdim=True)  # (B, T, 1)
        else:
            sigma2_scalar = sigma2  # (B, T, 1)
        
        # 按照文档公式：w_t = exp(-σ_t²) / Σ exp(-σ_t²)
        w = torch.exp(-sigma2_scalar)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        feat = (h * w).sum(dim=1)
        return feat, w


class MINEEstimator(nn.Module):
    """
    MINE (Mutual Information Neural Estimation) 实现
    用于估计两组特征间的互信息上界
    
    标准MINE实现：不包含GRL，需要在训练时使用双优化器
    - MINE网络（D_φ）最大化MI估计
    - 主模型最小化MI估计
    """

    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 512, ema_decay: float = 0.99, eps: float = 1e-8):
        super().__init__()
        
        # 三层MLP判别器网络（符合文档要求）
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.ema_decay = ema_decay
        self.eps = eps
        self.register_buffer("ma_et", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: (B, D1)
            y: (B, D2)
        Returns:
            mi_estimate: 标量，互信息估计
        """
        b = x.size(0)
        
        joint = torch.cat([x, y], dim=-1)

        # 打乱 y 构造边缘分布 P_ID ⊗ P_NID
        perm = torch.randperm(b, device=x.device)
        marg = torch.cat([x, y[perm]], dim=-1)

        t_joint = self.net(joint)
        t_marg = self.net(marg)

        # MINE损失: 使用exp(D_φ)的指数滑动平均稳定估计
        et = torch.exp(t_marg)
        if self.training:
            ma_et = self.ma_et * self.ema_decay + (1.0 - self.ema_decay) * et.mean()
            self.ma_et = ma_et.detach()
            norm_et = ma_et
        else:
            norm_et = self.ma_et
        mi = t_joint.mean() - torch.log(norm_et + self.eps)
        return mi


