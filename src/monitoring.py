"""
训练监控模块：监控σ_t²变化
按照文档要求：观察训练日志中的σ_t²变化，如果所有帧的σ²预测都一致，说明BVS模块失效
"""

import torch
from typing import Dict, List, Optional
import numpy as np


class UncertaintyMonitor:
    """
    不确定性监控器：监控BVS模块输出的σ_t²变化
    """
    
    def __init__(
        self,
        window_size: int = 100,
        threshold: float = 0.01,
        check_interval: int = 50,
    ):
        """
        Args:
            window_size: 用于计算统计量的窗口大小（batch数）
            threshold: 方差阈值，如果σ_t²的方差小于此值，认为BVS失效
            check_interval: 检查间隔（每N个batch检查一次）
        """
        self.window_size = window_size
        self.threshold = threshold
        self.check_interval = check_interval
        
        # 存储历史记录
        self.sigma2_history: List[torch.Tensor] = []
        self.batch_count = 0
        self.warnings: List[str] = []
    
    def update(self, sigma2: torch.Tensor) -> Optional[Dict[str, float]]:
        """
        更新监控状态
        
        Args:
            sigma2: (B, T, 1) 或 (B, T, D) 不确定性张量（支持标量和向量）
        Returns:
            stats: 统计信息字典，如果达到检查间隔则返回，否则返回None
        """
        self.batch_count += 1
        
        # 如果是不确定性向量，先聚合为标量用于监控
        if sigma2.dim() == 3 and sigma2.size(-1) > 1:
            # 向量不确定性：使用L2范数作为标量表示
            sigma2_scalar = sigma2.norm(dim=-1, keepdim=True)  # (B, T, 1)
        else:
            sigma2_scalar = sigma2  # (B, T, 1)
        
        # 存储当前batch的σ_t²
        # 计算每个batch的统计量
        batch_stats = {
            'mean': sigma2_scalar.mean().item(),
            'std': sigma2_scalar.std().item(),
            'min': sigma2_scalar.min().item(),
            'max': sigma2_scalar.max().item(),
            'variance': sigma2_scalar.var().item(),
        }
        
        # 计算帧间方差（检查是否所有帧的σ²都一致）
        # 对每个样本，计算T个帧的σ²的方差
        frame_variance = sigma2_scalar.var(dim=1, keepdim=False)  # (B, 1)
        batch_stats['frame_variance_mean'] = frame_variance.mean().item()
        batch_stats['frame_variance_min'] = frame_variance.min().item()
        
        # 存储标量版本用于历史记录
        self.sigma2_history.append(sigma2_scalar.detach().cpu())
        
        # 保持窗口大小
        if len(self.sigma2_history) > self.window_size:
            self.sigma2_history.pop(0)
        
        # 每check_interval个batch检查一次
        if self.batch_count % self.check_interval == 0:
            return self._check_health(batch_stats)
        
        return None
    
    def _check_health(self, current_stats: Dict[str, float]) -> Dict[str, float]:
        """
        检查BVS模块的健康状态
        
        Returns:
            stats: 包含当前统计量和健康检查结果的字典
        """
        stats = current_stats.copy()
        
        # 检查1: 帧间方差是否过小（所有帧的σ²都一致）
        if stats['frame_variance_mean'] < self.threshold:
            warning = (
                f"Batch {self.batch_count}: BVS可能失效！"
                f"帧间方差均值={stats['frame_variance_mean']:.6f} < {self.threshold}"
            )
            self.warnings.append(warning)
            stats['health_warning'] = 1.0
            stats['health_status'] = 'WARNING'
        else:
            stats['health_warning'] = 0.0
            stats['health_status'] = 'OK'
        
        # 检查2: σ²是否全为0或全为相同值
        if len(self.sigma2_history) >= 10:
            recent_sigma2 = torch.cat(self.sigma2_history[-10:], dim=0)  # (10*B, T, 1)
            recent_std = recent_sigma2.std().item()
            if recent_std < self.threshold:
                warning = (
                    f"Batch {self.batch_count}: σ²变化过小！"
                    f"最近10个batch的σ²标准差={recent_std:.6f} < {self.threshold}"
                )
                if warning not in self.warnings[-5:]:  # 避免重复警告
                    self.warnings.append(warning)
                stats['health_warning'] = 1.0
                stats['health_status'] = 'WARNING'
        
        # 计算窗口内的统计量
        if len(self.sigma2_history) >= self.window_size:
            window_sigma2 = torch.cat(self.sigma2_history, dim=0)  # (window_size*B, T, 1)
            stats['window_mean'] = window_sigma2.mean().item()
            stats['window_std'] = window_sigma2.std().item()
            stats['window_min'] = window_sigma2.min().item()
            stats['window_max'] = window_sigma2.max().item()
        
        return stats
    
    def get_summary(self) -> Dict[str, any]:
        """
        获取监控摘要
        
        Returns:
            summary: 包含所有警告和统计信息的字典
        """
        summary = {
            'total_batches': self.batch_count,
            'warnings': self.warnings[-10:],  # 最近10个警告
            'num_warnings': len(self.warnings),
        }
        
        if len(self.sigma2_history) > 0:
            all_sigma2 = torch.cat(self.sigma2_history, dim=0)
            summary['overall_mean'] = all_sigma2.mean().item()
            summary['overall_std'] = all_sigma2.std().item()
            summary['overall_min'] = all_sigma2.min().item()
            summary['overall_max'] = all_sigma2.max().item()
        
        return summary
    
    def reset(self):
        """重置监控器状态"""
        self.sigma2_history.clear()
        self.batch_count = 0
        self.warnings.clear()

