import random
from typing import Dict, Optional, Tuple

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
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            feat_dim: 特征维度
            num_classes: 类别数
            margin: triplet loss的margin
            use_batch_hard: 是否使用batch-hard triplet mining
            class_weights: 类别权重张量 (num_classes,)，用于处理类不均衡
        """
        super().__init__()
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
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
                    # 真正随机取一对正样本
                    idx_list = idx.tolist()
                    sampled = random.sample(idx_list, 2)
                    a, p = sampled[0], sampled[1]
                    # 真正随机取一个负样本
                    neg_idx = (labels != y).nonzero(as_tuple=False).view(-1)
                    if len(neg_idx) == 0:
                        continue
                    n = random.choice(neg_idx.tolist())
                    anchors.append(torch.tensor(a, device=labels.device))
                    positives.append(torch.tensor(p, device=labels.device))
                    negatives.append(torch.tensor(n, device=labels.device))

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
    中心化归一化交叉协方差矩阵的非对角线惩罚 (Centered Normalized Cross-Covariance Penalty)。
    比原始 Frobenius 范数版本更鲁棒：
    - 中心化消除均值偏移的影响
    - L2 归一化防止模型通过缩小特征范数来降低 loss
    - 按特征维度归一化使 loss 与维度无关

    Args:
        id_feat: (B, D) 或 (B, T, D) 身份流特征
        pose_feat: (B, D) 或 (B, T, D) 非身份流特征
    Returns:
        loss: 标量
    """
    if id_feat.dim() == 3:
        id_feat = id_feat.reshape(-1, id_feat.size(-1))    # (B*T, D)
        pose_feat = pose_feat.reshape(-1, pose_feat.size(-1))
    # 中心化
    id_feat = id_feat - id_feat.mean(dim=0, keepdim=True)
    pose_feat = pose_feat - pose_feat.mean(dim=0, keepdim=True)
    # L2 归一化（沿样本维度）
    id_feat = F.normalize(id_feat, dim=0)
    pose_feat = F.normalize(pose_feat, dim=0)
    # 交叉协方差矩阵
    cross_cov = id_feat.T @ pose_feat  # (D, D)
    return cross_cov.pow(2).sum() / id_feat.size(1)


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


class TemporalOrderPredictionLoss(nn.Module):
    """
    时序顺序预测辅助任务：给非身份流提供正向监督信号。
    随机打乱帧序列后，让一个小分类头预测是否被打乱（二分类）。
    确保 pose stream 学习到有意义的时序/动态特征，避免塌缩为无语义残差。
    """

    def __init__(self, feat_dim: int, shuffle_prob: float = 0.5):
        super().__init__()
        self.shuffle_prob = shuffle_prob
        # 简单的分类头：对时序特征做 mean pooling 后二分类
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim // 2, 2),
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, h_pose: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_pose: (B, T, D) 非身份流帧级输出
        Returns:
            loss: 标量
        """
        B, T, D = h_pose.shape
        if T <= 1:
            return h_pose.new_tensor(0.0)

        labels = []
        features = []
        for i in range(B):
            if random.random() < self.shuffle_prob:
                # 打乱帧顺序
                perm = torch.randperm(T, device=h_pose.device)
                features.append(h_pose[i, perm, :].mean(dim=0))
                labels.append(1)  # 1 = shuffled
            else:
                features.append(h_pose[i].mean(dim=0))
                labels.append(0)  # 0 = original order

        feat = torch.stack(features, dim=0)  # (B, D)
        label = torch.tensor(labels, device=h_pose.device, dtype=torch.long)
        logits = self.classifier(feat)
        return self.ce(logits, label)


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
        lambda_pose_aux: float = 0.2,
        margin: float = 0.3,
        use_batch_hard: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        use_pose_aux: bool = True,
    ):
        super().__init__()
        self.id_loss = IDLoss(
            feat_dim=feat_dim,
            num_classes=num_classes,
            margin=margin,
            use_batch_hard=use_batch_hard,
            class_weights=class_weights,
        )
        self.lambda_mi = lambda_mi
        self.lambda_orth = lambda_orth
        self.lambda_temp = lambda_temp
        self.lambda_kl = lambda_kl
        self.lambda_pose_aux = lambda_pose_aux
        self.use_pose_aux = use_pose_aux
        self.pose_aux_loss = TemporalOrderPredictionLoss(feat_dim) if use_pose_aux else None

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
        h_pose = outputs["h_pose"]  # (B, T, D)
        sigma2 = outputs["sigma2"]  # (B, T, 1)
        mi_est = outputs["mi"]  # 标量

        id_loss, triplet_loss, logits = self.id_loss(vid_id, labels)
        mi_loss = mi_est
        # 使用帧级特征计算正交损失（更充分利用样本，不受 batch size 限制）
        orth_loss = orthogonal_loss(h_id, h_pose)
        temp_loss = temporal_smoothness_loss(h_id)
        kl_loss = kl_gaussian_regularizer(sigma2)

        # 非身份流辅助监督：时序顺序预测
        if self.pose_aux_loss is not None and h_pose is not None:
            pose_aux = self.pose_aux_loss(h_pose)
        else:
            pose_aux = vid_id.new_tensor(0.0)

        total_loss = (
            id_loss
            + triplet_loss
            + self.lambda_mi * mi_loss
            + self.lambda_orth * orth_loss
            + self.lambda_temp * temp_loss
            + self.lambda_kl * kl_loss
            + self.lambda_pose_aux * pose_aux
        )

        loss_dict = {
            "total": total_loss.detach(),
            "id": id_loss.detach(),
            "triplet": triplet_loss.detach(),
            "mi": mi_loss.detach(),
            "orth": orth_loss.detach(),
            "temp": temp_loss.detach(),
            "kl": kl_loss.detach(),
            "pose_aux": pose_aux.detach(),
        }

        return total_loss, loss_dict


