from typing import Dict, Tuple

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
        num_blocks: int = 8,  # 文档建议 N=6~12，默认8
        mine_hidden_dim: int = 512,
    ):
        super().__init__()
        self.bvs = BayesianVisualStem(feat_dim=feat_dim, pretrained=True)

        self.blocks = nn.ModuleList(
            [RDBMambaBlock(d_model=feat_dim) for _ in range(num_blocks)]
        )

        self.agg = UncertaintyWeightedAggregator()
        self.mine = MINEEstimator(dim_x=feat_dim, dim_y=feat_dim, hidden_dim=mine_hidden_dim)

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
        h_pose = mu.detach().clone()

        for blk in self.blocks:
            h_id, h_pose = blk(h_id, h_pose, sigma2)

        # 视频级聚合
        vid_id, weights = self.agg(h_id, sigma2)
        vid_pose, _ = self.agg(h_pose, sigma2)

        # MINE 估计互信息
        mi_est = self.mine(vid_id, vid_pose)

        return {
            "mu": mu,
            "sigma2": sigma2,
            "h_id": h_id,
            "h_pose": h_pose,
            "vid_id": vid_id,
            "vid_pose": vid_pose,
            "weights": weights,
            "mi": mi_est,
        }


