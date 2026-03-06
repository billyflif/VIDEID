from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
    from mamba_ssm.modules.mamba_simple import Mamba

    HAS_SELECTIVE_SCAN = True
    MAMBA_BACKEND = "mamba_ssm_cuda"
except Exception as exc:
    from .mamba_reference import Mamba, selective_scan_fn

    mamba_inner_fn = None
    HAS_SELECTIVE_SCAN = True
    MAMBA_BACKEND = "torch_reference"
    warnings.warn(
        "mamba_ssm CUDA extensions are unavailable, falling back to the exact PyTorch "
        f"reference implementation. Training will be slower but the Mamba equations are unchanged. "
        f"Original import error: {exc!r}",
        RuntimeWarning,
    )


def stop_gradient(x: torch.Tensor) -> torch.Tensor:
    return x.detach()


class QualityGatedMamba(nn.Module):
    """Quality-gated Mamba layer with a custom uncertainty-modulated delta."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = max(16, self.d_model // 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_state * 2 + self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.delta_linear = nn.Linear(d_model, self.d_inner)
        self.softplus = nn.Softplus()
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.sigma_proj = nn.Linear(d_model, self.d_inner)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, sigma2: Optional[torch.Tensor] = None) -> torch.Tensor:
        if sigma2 is None or not HAS_SELECTIVE_SCAN:
            return self._forward_standard(x)
        return self._forward_with_custom_delta(x, sigma2)

    def _depthwise_conv(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seq_len]
        x = x.transpose(1, 2)
        return self.act(x)

    def _run_scan(
        self,
        x: torch.Tensor,
        delta: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
    ) -> torch.Tensor:
        A = -torch.exp(self.A_log.float()).to(x.device)
        D = self.D.float().to(x.device)
        y = selective_scan_fn(
            u=x.transpose(1, 2).contiguous(),
            delta=delta.transpose(1, 2).contiguous(),
            A=A,
            B=B_param.transpose(1, 2).contiguous(),
            C=C_param.transpose(1, 2).contiguous(),
            D=D,
        )
        return y.transpose(1, 2)

    def _forward_standard(self, x: torch.Tensor) -> torch.Tensor:
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self._depthwise_conv(x)

        x_dbl = self.x_proj(x)
        B_param, C_param, delta_raw = x_dbl.split(
            [self.d_state, self.d_state, self.dt_rank], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta_raw))
        y = self._run_scan(x, delta, B_param, C_param)
        y = y * self.act(z)
        return self.out_proj(y)

    def _forward_with_custom_delta(self, x: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        x_original = x

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = self._depthwise_conv(x)

        x_dbl = self.x_proj(x)
        B_param, C_param, _ = x_dbl.split([self.d_state, self.d_state, self.dt_rank], dim=-1)

        delta_raw = self.softplus(self.delta_linear(x_original))
        if sigma2.size(-1) == 1:
            sigma2_expand = sigma2.expand(batch_size, seq_len, self.d_inner)
        else:
            sigma2_expand = self.softplus(self.sigma_proj(sigma2))
        delta_custom = delta_raw * torch.exp(-self.alpha * sigma2_expand)

        y = self._run_scan(x, delta_custom, B_param, C_param)
        y = y * self.act(z)
        return self.out_proj(y)


class BiMambaLayer(nn.Module):
    """Bidirectional Mamba layer with optional quality gating."""

    def __init__(
        self,
        d_model: int,
        use_quality_gating: bool = False,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.use_quality_gating = use_quality_gating
        self.bidirectional = bidirectional

        if use_quality_gating:
            self.fwd = QualityGatedMamba(d_model=d_model)
            self.bwd = QualityGatedMamba(d_model=d_model) if bidirectional else None
        else:
            self.fwd = Mamba(d_model=d_model)
            self.bwd = Mamba(d_model=d_model) if bidirectional else None

    def forward(self, x: torch.Tensor, u: Optional[torch.Tensor] = None):
        if self.use_quality_gating and u is not None:
            fwd_out = self.fwd(x, u)
        else:
            fwd_out = self.fwd(x)

        if not self.bidirectional or self.bwd is None:
            return fwd_out, None

        rev_x = torch.flip(x, dims=[1])
        if self.use_quality_gating and u is not None:
            rev_u = torch.flip(u, dims=[1])
            bwd_out = self.bwd(rev_x, rev_u)
        else:
            bwd_out = self.bwd(rev_x)
        bwd_out = torch.flip(bwd_out, dims=[1])
        return fwd_out, bwd_out


class RDBMambaBlock(nn.Module):
    """Residual decoupled bidirectional Mamba block."""

    def __init__(
        self,
        d_model: int,
        quality_gated: bool = True,
        bidirectional: bool = True,
        use_pose_stream: bool = True,
        use_pose_to_id: bool = True,
        alpha_pose_to_id: float = 0.1,
    ) -> None:
        super().__init__()
        self.use_pose_stream = use_pose_stream
        self.use_pose_to_id = use_pose_stream and use_pose_to_id
        self.id_layer = BiMambaLayer(
            d_model,
            use_quality_gating=quality_gated,
            bidirectional=bidirectional,
        )
        self.pose_layer = (
            BiMambaLayer(d_model, use_quality_gating=False, bidirectional=bidirectional)
            if use_pose_stream
            else None
        )

        self.id_norm = nn.LayerNorm(d_model)
        self.pose_norm = nn.LayerNorm(d_model) if use_pose_stream else None
        self.pose_fusion = nn.Linear(2 * d_model, d_model) if bidirectional and use_pose_stream else None
        self.fusion_gate = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            if bidirectional
            else None
        )

        if self.use_pose_to_id:
            self.gamma = nn.Parameter(torch.tensor(alpha_pose_to_id, dtype=torch.float32))
        else:
            self.register_parameter("gamma", None)

    def forward(
        self,
        x_id: torch.Tensor,
        x_pose: Optional[torch.Tensor],
        sigma2: Optional[torch.Tensor] = None,
    ):
        id_res = x_id
        id_fwd, id_bwd = self.id_layer(x_id, sigma2)

        if id_bwd is None or self.fusion_gate is None:
            id_bi = id_fwd
        else:
            z = self.fusion_gate(id_fwd + id_bwd)
            id_bi = z * id_fwd + (1.0 - z) * id_bwd
        id_out = self.id_norm(id_bi) + id_res

        pose_out = None
        if self.use_pose_stream and self.pose_layer is not None and x_pose is not None:
            pose_res = x_pose
            pose_fwd, pose_bwd = self.pose_layer(x_pose, None)

            if pose_bwd is None or self.pose_fusion is None:
                pose_bi = pose_fwd
            else:
                pose_cat = torch.cat([pose_fwd, pose_bwd], dim=-1)
                pose_bi = self.pose_fusion(pose_cat)
            pose_out = self.pose_norm(pose_bi) + pose_res  # type: ignore[arg-type]

            if self.gamma is not None:
                id_out = id_out + self.gamma * stop_gradient(pose_out)

        return id_out, pose_out
