from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_hard_triplet_loss(
    feats: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
    squared: bool = False,
) -> torch.Tensor:
    """
    Batch-hard triplet loss实现
    
    对于每个anchor，选择：
    - hardest positive: 同ID中距离最远的样本
    - hardest negative: 不同ID中距离最近的样本
    
    Args:
        feats: (B, D) 特征向量
        labels: (B,) 标签
        margin: triplet margin
        squared: 是否使用平方距离
    Returns:
        loss: 标量损失值
    """
    # 计算所有样本对之间的成对距离
    pairwise_dist = torch.cdist(feats, feats, p=2)  # (B, B)
    if squared:
        pairwise_dist = pairwise_dist ** 2
    
    # 创建mask：相同ID为True，不同ID为False
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
    
    # 对于每个anchor，找到hardest positive和hardest negative
    losses = []
    
    for i in range(len(feats)):
        # Hardest positive: 同ID中距离最远的
        positive_mask = labels_equal[i].clone()
        positive_mask[i] = False  # 排除自己
        if positive_mask.any():
            hardest_positive_dist = pairwise_dist[i][positive_mask].max()
        else:
            # 如果没有其他正样本，跳过这个anchor
            continue
        
        # Hardest negative: 不同ID中距离最近的
        negative_mask = ~labels_equal[i]
        if negative_mask.any():
            hardest_negative_dist = pairwise_dist[i][negative_mask].min()
        else:
            # 如果没有负样本，跳过这个anchor
            continue
        
        # 计算triplet loss
        loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + margin, min=0.0)
        losses.append(loss)
    
    if len(losses) == 0:
        return feats.new_tensor(0.0)
    
    return torch.stack(losses).mean()


class IDLoss(nn.Module):
    """
    身份损失：分类交叉熵 + 三元组约束（支持简单采样和batch-hard两种模式）
    """

    def __init__(
        self,
        feat_dim: int,
        num_classes: int,
        margin: float = 0.3,
        use_batch_hard: bool = False,
    ):
        """
        Args:
            feat_dim: 特征维度
            num_classes: 类别数
            margin: triplet loss的margin
            use_batch_hard: 是否使用batch-hard triplet mining
        """
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.ce = nn.CrossEntropyLoss()
        self.margin = margin
        self.use_batch_hard = use_batch_hard
        
        # 如果使用batch-hard，不需要TripletMarginLoss
        if not use_batch_hard:
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

        # 根据模式选择三元组损失计算方式
        if self.use_batch_hard:
            # Batch-hard triplet mining
            triplet_loss = batch_hard_triplet_loss(feats, labels, margin=self.margin)
        else:
            # 简单构造三元组（仅在一个 batch 内随机采样）
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
    正交约束：最小化两组特征的矩阵乘法Frobenius范数
    L_orth = ||H_ID^T · H_NID||²_F
    
    如果输入是(B, D)，则计算 (D, B) × (B, D) = (D, D) 的Frobenius范数
    如果输入是(B, T, D)，则对时间维度进行矩阵乘法
    
    Args:
        id_feat: (B, D) 或 (B, T, D)
        pose_feat: (B, D) 或 (B, T, D)
    """
    if id_feat.dim() == 2:
        # (B, D) -> (D, B) × (B, D) = (D, D)
        # H_ID^T: (D, B), H_NID: (B, D)
        prod = torch.matmul(id_feat.t(), pose_feat)  # (D, D)
        return torch.norm(prod, p='fro') ** 2
    elif id_feat.dim() == 3:
        # (B, T, D) -> 对每个batch计算 (D, T) × (T, D) = (D, D)
        # H_ID^T: (B, D, T), H_NID: (B, T, D)
        prod = torch.bmm(id_feat.transpose(1, 2), pose_feat)  # (B, D, D)
        return torch.norm(prod, p='fro', dim=(1, 2)).mean() ** 2
    else:
        raise ValueError(f"Unsupported feature dimension: {id_feat.dim()}")


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
        use_batch_hard: bool = False,
    ):
        super().__init__()
        self.id_loss = IDLoss(
            feat_dim=feat_dim,
            num_classes=num_classes,
            margin=margin,
            use_batch_hard=use_batch_hard,
        )
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


