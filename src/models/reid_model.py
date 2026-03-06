from typing import Dict

import torch
import torch.nn as nn

from .bvs import BayesianVisualStem
from .mamba_blocks import RDBMambaBlock
from .heads import UncertaintyWeightedAggregator, MINEEstimator


class VideoReIDModel(nn.Module):
    """
    整体视频 ReID 模型：
    - BVS 提取帧级特征与不确定性
    - 多层 RDB-Mamba 进行时序建模与身份/非身份解耦
    - 聚合得到视频级身份特征
    - MINE 用于互信息最小化
    """

    def __init__(
        self,
        feat_dim: int = 512,
        num_blocks: int = 4,  # 小数据场景减少层数防止过拟合（原默认8）
        mine_hidden_dim: int = 512,
        block_dropout: float = 0.3,
        use_quality_gating: bool = True,
        bidirectional: bool = True,
        use_pose_stream: bool = True,
        use_pose_to_id: bool = True,
        use_uncertainty_weighting: bool = True,
    ):
        super().__init__()
        self.use_pose_stream = use_pose_stream
        self.bvs = BayesianVisualStem(feat_dim=feat_dim, pretrained=True)

        self.blocks = nn.ModuleList(
            [
                RDBMambaBlock(
                    d_model=feat_dim,
                    quality_gated=use_quality_gating,
                    bidirectional=bidirectional,
                    use_pose_stream=use_pose_stream,
                    use_pose_to_id=use_pose_to_id,
                )
                for _ in range(num_blocks)
            ]
        )
        # Mamba blocks之间的Dropout，防止过拟合
        self.block_dropout = nn.Dropout(p=block_dropout)

        self.agg = UncertaintyWeightedAggregator(use_uncertainty=use_uncertainty_weighting)
        self.mine = (
            MINEEstimator(dim_x=feat_dim, dim_y=feat_dim, hidden_dim=mine_hidden_dim)
            if use_pose_stream
            else None
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            outputs: dict，包含中间结果与特征
        """
        mu, sigma2 = self.bvs(x)  # (B, T, D), (B, T, 1)

        # 初始时刻，身份流和非身份流共享同一特征
        h_id = mu
        h_pose = mu.detach().clone() if self.use_pose_stream else None

        for blk in self.blocks:
            h_id, h_pose = blk(h_id, h_pose, sigma2)
            h_id = self.block_dropout(h_id)
            if h_pose is not None:
                h_pose = self.block_dropout(h_pose)

        # 视频级聚合
        vid_id, weights = self.agg(h_id, sigma2)
        if h_pose is not None:
            vid_pose, _ = self.agg(h_pose, sigma2)
        else:
            vid_pose = vid_id.detach()

        # MINE 估计互信息
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


