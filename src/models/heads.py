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
        w_t = exp(-lambda * sigma2_t) / sum_k exp(-lambda * sigma2_k)
        F_video = sum_t w_t * H_out[t]
    """

    def __init__(self, lam: float = 1.0):
        super().__init__()
        self.lam = lam

    def forward(self, h: torch.Tensor, sigma2: torch.Tensor):
        """
        Args:
            h: (B, T, D)
            sigma2: (B, T, 1)
        Returns:
            feat: (B, D)
            weights: (B, T, 1)
        """
        w = torch.exp(-self.lam * sigma2)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
        feat = (h * w).sum(dim=1)
        return feat, w


class MINEEstimator(nn.Module):
    """
    MINE (Mutual Information Neural Estimation) 实现
    用于估计两组特征间的互信息上界
    
    使用梯度反转层(GRL)实现对抗训练：
    - 主模型尝试最小化MI（通过GRL反向梯度）
    - MINE网络尝试最大化MI估计（正常梯度）
    """

    def __init__(self, dim_x: int, dim_y: int, hidden_dim: int = 512, use_grl: bool = True, lambda_grl: float = 1.0):
        super().__init__()
        self.use_grl = use_grl
        self.grl = GradientReversalLayer(lambda_grl=lambda_grl) if use_grl else None
        
        self.net = nn.Sequential(
            nn.Linear(dim_x + dim_y, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x: (B, D1)
            y: (B, D2)
        Returns:
            mi_estimate: 标量，互信息估计
        """
        b = x.size(0)
        
        # 如果使用GRL，对输入应用梯度反转
        if self.use_grl and self.training:
            x = self.grl(x)
            y = self.grl(y)
        
        joint = torch.cat([x, y], dim=-1)

        # 打乱 y 构造边缘分布
        perm = torch.randperm(b, device=x.device)
        marg = torch.cat([x, y[perm]], dim=-1)

        t_joint = self.net(joint)
        t_marg = self.net(marg)

        # MINE损失: E[T_joint] - log(E[exp(T_marg)])
        mi = t_joint.mean() - torch.log(torch.exp(t_marg).mean() + 1e-8)
        return mi


