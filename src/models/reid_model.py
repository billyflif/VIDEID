from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bvs import BayesianVisualStem
from .heads import MINEEstimator, UncertaintyWeightedAggregator
from .mamba_blocks import RDBMambaBlock


SUPPORTED_MODEL_ARCHS = (
    "dual_mamba",
    "single_mamba",
    "avgpool",
    "gru",
    "lstm",
    "transformer",
)


class IdentityTemporalEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class BidirectionalRecurrentEncoder(nn.Module):
    def __init__(
        self,
        rnn_type: str,
        feat_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if rnn_type not in {"gru", "lstm"}:
            raise ValueError(f"Unsupported rnn_type={rnn_type}")

        rnn_cls = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * 2
        self.norm = nn.LayerNorm(out_dim)
        self.proj = nn.Identity() if out_dim == feat_dim else nn.Linear(out_dim, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = self.norm(out)
        return self.proj(out)


class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, feat_dim))
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=int(feat_dim * ffn_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feat_dim)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def _get_pos_embed(self, seq_len: int) -> torch.Tensor:
        if seq_len <= self.max_seq_len:
            return self.pos_embed[:, :seq_len, :]
        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=seq_len, mode="linear", align_corners=False)
        return pos.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = self._get_pos_embed(x.size(1)).to(dtype=x.dtype, device=x.device)
        x = self.dropout(x + pos)
        x = self.encoder(x)
        return self.norm(x)


class VideoReIDModel(nn.Module):
    def __init__(
        self,
        feat_dim: int = 512,
        num_blocks: int = 4,
        mine_hidden_dim: int = 512,
        block_dropout: float = 0.3,
        use_quality_gating: bool = True,
        bidirectional: bool = True,
        use_pose_stream: bool = True,
        use_pose_to_id: bool = True,
        use_uncertainty_weighting: bool = True,
        model_arch: str = "dual_mamba",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 2,
        temporal_dropout: float = 0.1,
        transformer_num_layers: int = 2,
        transformer_num_heads: int = 8,
        transformer_ffn_ratio: float = 4.0,
        transformer_max_seq_len: int = 32,
    ):
        super().__init__()
        if model_arch not in SUPPORTED_MODEL_ARCHS:
            raise ValueError(
                f"Unsupported model_arch={model_arch}, "
                f"must be one of {SUPPORTED_MODEL_ARCHS}"
            )

        self.model_arch = model_arch
        self.is_mamba_family = model_arch in {"dual_mamba", "single_mamba"}
        self.use_pose_stream = bool(use_pose_stream and model_arch == "dual_mamba")
        self.use_quality_gating = bool(use_quality_gating and self.is_mamba_family)
        self.use_uncertainty_weighting = bool(
            use_uncertainty_weighting and self.is_mamba_family
        )
        self.bvs = BayesianVisualStem(feat_dim=feat_dim, pretrained=True)

        self.blocks = nn.ModuleList()
        self.block_dropout = nn.Dropout(p=block_dropout)
        self.temporal_encoder: Optional[nn.Module] = None

        if self.is_mamba_family:
            use_pose_to_id = bool(use_pose_to_id and self.use_pose_stream)
            self.blocks = nn.ModuleList(
                [
                    RDBMambaBlock(
                        d_model=feat_dim,
                        quality_gated=self.use_quality_gating,
                        bidirectional=bidirectional,
                        use_pose_stream=self.use_pose_stream,
                        use_pose_to_id=use_pose_to_id,
                    )
                    for _ in range(num_blocks)
                ]
            )
        elif model_arch == "avgpool":
            self.temporal_encoder = IdentityTemporalEncoder()
        elif model_arch in {"gru", "lstm"}:
            self.temporal_encoder = BidirectionalRecurrentEncoder(
                rnn_type=model_arch,
                feat_dim=feat_dim,
                hidden_dim=rnn_hidden_dim,
                num_layers=rnn_num_layers,
                dropout=temporal_dropout,
            )
        elif model_arch == "transformer":
            self.temporal_encoder = TemporalTransformerEncoder(
                feat_dim=feat_dim,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                ffn_ratio=transformer_ffn_ratio,
                dropout=temporal_dropout,
                max_seq_len=transformer_max_seq_len,
            )

        self.agg = UncertaintyWeightedAggregator(
            use_uncertainty=self.use_uncertainty_weighting
        )
        self.mine = (
            MINEEstimator(dim_x=feat_dim, dim_y=feat_dim, hidden_dim=mine_hidden_dim)
            if self.use_pose_stream
            else None
        )

    def _forward_mamba(self, mu: torch.Tensor, sigma2: torch.Tensor):
        h_id = mu
        h_pose = mu.detach().clone() if self.use_pose_stream else None

        for blk in self.blocks:
            h_id, h_pose = blk(h_id, h_pose, sigma2)
            h_id = self.block_dropout(h_id)
            if h_pose is not None:
                h_pose = self.block_dropout(h_pose)

        vid_id, weights = self.agg(h_id, sigma2)
        if h_pose is not None:
            vid_pose, _ = self.agg(h_pose, sigma2)
        else:
            vid_pose = vid_id.detach()
        return h_id, h_pose, vid_id, vid_pose, weights

    def _forward_baseline(self, mu: torch.Tensor, sigma2: torch.Tensor):
        if self.temporal_encoder is None:
            raise RuntimeError(f"temporal_encoder is not initialized for {self.model_arch}")

        h_id = self.temporal_encoder(mu)
        vid_id, weights = self.agg(h_id, sigma2)
        h_pose = h_id.detach()
        vid_pose = vid_id.detach()
        return h_id, h_pose, vid_id, vid_pose, weights

    def forward(self, x: torch.Tensor):
        mu, sigma2 = self.bvs(x)

        if self.is_mamba_family:
            h_id, h_pose, vid_id, vid_pose, weights = self._forward_mamba(mu, sigma2)
        else:
            h_id, h_pose, vid_id, vid_pose, weights = self._forward_baseline(mu, sigma2)

        vid_id = F.normalize(vid_id, p=2, dim=-1)
        vid_pose = F.normalize(vid_pose, p=2, dim=-1)
        mi_est = self.mine(vid_id, vid_pose) if self.mine is not None else vid_id.new_zeros(())

        return {
            "mu": mu,
            "sigma2": sigma2,
            "h_id": h_id,
            "h_pose": h_pose if h_pose is not None else h_id.detach(),
            "vid_id": vid_id,
            "vid_pose": vid_pose,
            "weights": weights,
            "mi": mi_est,
        }

