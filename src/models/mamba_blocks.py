from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from mamba_ssm import Mamba


def stop_gradient(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


class BiMambaLayer(nn.Module):
    """
    双向 Mamba：
    - 前向扫描 + 后向扫描
    - 可选：使用外部不确定性 u 调制步长（通过 gating 方式近似）
    """

    def __init__(
        self,
        d_model: int,
        use_quality_gating: bool = False,
    ):
        super().__init__()
        self.use_quality_gating = use_quality_gating

        self.fwd = Mamba(d_model=d_model)
        self.bwd = Mamba(d_model=d_model)

        self.proj = nn.Linear(2 * d_model, d_model)

        # 步长门控相关参数：
        # Δ_raw = Softplus(Linear(x))
        # Δ_id  = Δ_raw * exp(-α * σ_t^2)
        # 这里 α 为可学习缩放因子
        self.delta_linear = nn.Linear(d_model, d_model)
        self.softplus = nn.Softplus()
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float))

    def forward(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, T, D)
            u: (B, T, 1), 不确定性（可选）
        Returns:
            out: (B, T, D)
            fwd_out, bwd_out: (B, T, D)
        """
        b, t, d = x.shape

        if self.use_quality_gating and u is not None:
            # ---------- 质量门控步长（与文档形式对应） ----------
            # 计算 Δ_raw = Softplus(Linear(x))
            delta_raw = self.softplus(self.delta_linear(x))  # (B, T, D)，>0

            # 将标量不确定性 σ_t^2 扩展到特征维度，并计算
            # Δ_id = Δ_raw * exp(-α * σ_t^2)
            sigma2_expand = u.expand_as(delta_raw)
            delta_id = delta_raw * torch.exp(-self.alpha * sigma2_expand)

            # 为了作为 gate 使用，将步长映射到 (0, 1)：
            # gate = Δ_id / (1 + Δ_id)
            gate = delta_id / (1.0 + delta_id)

            # 低质量（σ^2 大）→ exp(-α σ^2) 小 → Δ_id 小 → gate 接近 0
            # 相当于减弱当前时刻的输入更新，近似实现 “h_t ≈ h_{t-1}” 的效果
            x_gated = x * gate
        else:
            x_gated = x

        # Mamba 默认输入形状为 (B, T, D)
        fwd_out = self.fwd(x_gated)

        # 反向扫描
        rev_x = torch.flip(x_gated, dims=[1])
        bwd_out = self.bwd(rev_x)
        bwd_out = torch.flip(bwd_out, dims=[1])

        cat = torch.cat([fwd_out, bwd_out], dim=-1)
        out = self.proj(cat)

        return out, fwd_out, bwd_out


class RDBMambaBlock(nn.Module):
    """
    Residual Decoupled Bi-Mamba Block
    - 身份流 (ID stream)
    - 非身份流 (Pose / Non-ID stream)
    - 双向扫描 + 不确定性门控（ID 流）
    - 残差连接 + 跨流信息注入
    """

    def __init__(self, d_model: int, quality_gated: bool = True, alpha_pose_to_id: float = 0.1):
        super().__init__()
        self.id_layer = BiMambaLayer(d_model, use_quality_gating=quality_gated)
        self.pose_layer = BiMambaLayer(d_model, use_quality_gating=False)

        self.id_norm = nn.LayerNorm(d_model)
        self.pose_norm = nn.LayerNorm(d_model)

        self.alpha = alpha_pose_to_id

    def forward(self, x_id: torch.Tensor, x_pose: torch.Tensor, sigma2: Optional[torch.Tensor] = None):
        """
        Args:
            x_id: (B, T, D)
            x_pose: (B, T, D)
            sigma2: (B, T, 1)
        Returns:
            out_id, out_pose: (B, T, D)
        """
        # 身份流：受不确定性影响
        id_res = x_id
        id_out, id_fwd, id_bwd = self.id_layer(x_id, sigma2)
        # 双向融合
        z = torch.sigmoid(self.id_norm(id_fwd + id_bwd))
        id_bi = z * id_fwd + (1.0 - z) * id_bwd
        id_out = self.id_norm(id_out + id_bi)
        id_out = id_out + id_res

        # 非身份流：正常 Mamba
        pose_res = x_pose
        pose_out, pose_fwd, pose_bwd = self.pose_layer(x_pose, None)
        pose_bi = torch.cat([pose_fwd, pose_bwd], dim=-1)
        pose_out = self.pose_norm(pose_out)
        pose_out = pose_out + pose_res

        # 姿态信息注入身份流 (no gradient)
        id_out = id_out + self.alpha * stop_gradient(pose_out)

        return id_out, pose_out


