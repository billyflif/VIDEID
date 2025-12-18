from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDLoss(nn.Module):
    """
    身份损失：分类交叉熵 + 简单三元组约束。
    """

    def __init__(self, feat_dim: int, num_classes: int, margin: float = 0.3):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            feats: (B, D)
            labels: (B,)
        Returns:
            id_loss: 交叉熵损失
            triplet_loss: 三元组损失
            logits: (B, num_classes)
        """
        logits = self.classifier(feats)
        id_loss = self.ce(logits, labels)

        # 简单构造三元组（仅在一个 batch 内随机采样）
        # 实际项目中可替换为更稳定的 batch-hard triplet
        with torch.no_grad():
            anchors, positives, negatives = [], [], []
            for y in labels.unique():
                idx = (labels == y).nonzero(as_tuple=False).view(-1)
                if len(idx) < 2:
                    continue
                # 随机取一对正样本
                a, p = idx[0], idx[1]
                # 随机取一个负样本
                neg_idx = (labels != y).nonzero(as_tuple=False).view(-1)
                if len(neg_idx) == 0:
                    continue
                n = neg_idx[0]
                anchors.append(a)
                positives.append(p)
                negatives.append(n)

        triplet_loss = feats.new_tensor(0.0)
        if anchors:
            anchors = torch.stack(anchors)
            positives = torch.stack(positives)
            negatives = torch.stack(negatives)
            triplet_loss = self.triplet(
                feats[anchors], feats[positives], feats[negatives]
            )

        return id_loss, triplet_loss, logits


def orthogonal_loss(id_feat: torch.Tensor, pose_feat: torch.Tensor) -> torch.Tensor:
    """
    正交约束：最小化两组特征的内积。
    Args:
        id_feat: (B, D)
        pose_feat: (B, D)
    """
    id_n = F.normalize(id_feat, dim=-1)
    pose_n = F.normalize(pose_feat, dim=-1)
    prod = (id_n * pose_n).sum(dim=-1)
    return (prod**2).mean()


def temporal_smoothness_loss(feat: torch.Tensor) -> torch.Tensor:
    """
    时序平滑：鼓励相邻帧的特征变化平滑（典型用于身份流）。
    Args:
        feat: (B, T, D)
    """
    if feat.size(1) <= 1:
        return feat.new_tensor(0.0)
    diff = feat[:, 1:, :] - feat[:, :-1, :]
    return (diff**2).mean()


def kl_gaussian_regularizer(
    sigma2: torch.Tensor, prior_var: float = 1.0
) -> torch.Tensor:
    """
    对不确定性施加先验约束：KL(N(0, sigma2) || N(0, prior_var))
    这里采用简化形式，假设均值为 0。
    Args:
        sigma2: (B, T, 1)
    """
    sigma2 = sigma2.clamp(min=1e-6)
    prior_var = float(prior_var)
    kl = 0.5 * (sigma2 / prior_var - 1.0 - torch.log(sigma2 / prior_var))
    return kl.mean()


class VideoReIDCriterion(nn.Module):
    """
    总体损失组合：
    - 身份损失（分类 + 三元组）
    - 互信息最小化（MINE 输出）
    - 正交约束
    - 身份流时序平滑
    - 不确定性 KL 正则

    通过可配置权重进行加权求和，方便实验调参。
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        lambda_mi: float = 0.1,
        lambda_orth: float = 0.01,
        lambda_temp: float = 0.1,
        lambda_kl: float = 0.01,
        margin: float = 0.3,
    ):
        super().__init__()
        self.id_loss = IDLoss(feat_dim=feat_dim, num_classes=num_classes, margin=margin)
        self.lambda_mi = lambda_mi
        self.lambda_orth = lambda_orth
        self.lambda_temp = lambda_temp
        self.lambda_kl = lambda_kl

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            outputs: 来自 VideoReIDModel 的输出字典
            labels: (B,)
        Returns:
            total_loss: 加权后的总损失
            loss_dict: 各项损失详细分量
        """
        vid_id = outputs["vid_id"]  # (B, D)
        vid_pose = outputs["vid_pose"]  # (B, D)
        h_id = outputs["h_id"]  # (B, T, D)
        sigma2 = outputs["sigma2"]  # (B, T, 1)
        mi_est = outputs["mi"]  # 标量

        id_loss, triplet_loss, logits = self.id_loss(vid_id, labels)
        mi_loss = mi_est
        orth_loss = orthogonal_loss(vid_id, vid_pose)
        temp_loss = temporal_smoothness_loss(h_id)
        kl_loss = kl_gaussian_regularizer(sigma2)

        total_loss = (
            id_loss
            + triplet_loss
            + self.lambda_mi * mi_loss
            + self.lambda_orth * orth_loss
            + self.lambda_temp * temp_loss
            + self.lambda_kl * kl_loss
        )

        loss_dict = {
            "total": total_loss.detach(),
            "id": id_loss.detach(),
            "triplet": triplet_loss.detach(),
            "mi": mi_loss.detach(),
            "orth": orth_loss.detach(),
            "temp": temp_loss.detach(),
            "kl": kl_loss.detach(),
        }

        return total_loss, loss_dict


