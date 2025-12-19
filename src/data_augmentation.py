"""
数据增强模块：模拟各种degradation factors
按照文档要求实现：
- random occlusion（随机遮挡）
- artificial motion blur（人工运动模糊）
- brightness perturbation（亮度扰动）
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
import random
import math


class RandomOcclusion:
    """
    随机遮挡增强
    在视频帧上随机放置矩形遮挡区域
    """
    
    def __init__(
        self,
        prob: float = 0.5,
        min_occlusion_ratio: float = 0.1,
        max_occlusion_ratio: float = 0.3,
        num_patches: int = 1,
    ):
        """
        Args:
            prob: 应用遮挡的概率
            min_occlusion_ratio: 最小遮挡比例（相对于图像尺寸）
            max_occlusion_ratio: 最大遮挡比例
            num_patches: 遮挡块的数量
        """
        self.prob = prob
        self.min_occlusion_ratio = min_occlusion_ratio
        self.max_occlusion_ratio = max_occlusion_ratio
        self.num_patches = num_patches
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (T, C, H, W) 或 (B, T, C, H, W)
        Returns:
            augmented_video: 相同形状
        """
        if random.random() > self.prob:
            return video
        
        # 处理不同的输入维度
        is_batch = video.dim() == 5
        if is_batch:
            B, T, C, H, W = video.shape
            video = video.view(B * T, C, H, W)
        else:
            T, C, H, W = video.shape
            video = video.view(T, C, H, W)
        
        augmented = video.clone()
        
        for _ in range(self.num_patches):
            # 随机选择遮挡区域大小
            occlusion_h = int(H * random.uniform(self.min_occlusion_ratio, self.max_occlusion_ratio))
            occlusion_w = int(W * random.uniform(self.min_occlusion_ratio, self.max_occlusion_ratio))
            
            # 随机选择位置
            top = random.randint(0, max(1, H - occlusion_h))
            left = random.randint(0, max(1, W - occlusion_w))
            
            # 对每一帧应用遮挡（可以随机选择部分帧）
            num_frames_to_occlude = random.randint(1, video.size(0))
            frames_to_occlude = random.sample(range(video.size(0)), num_frames_to_occlude)
            
            for frame_idx in frames_to_occlude:
                # 使用随机值或黑色遮挡
                if random.random() < 0.5:
                    # 黑色遮挡
                    augmented[frame_idx, :, top:top+occlusion_h, left:left+occlusion_w] = 0.0
                else:
                    # 随机噪声遮挡
                    noise = torch.randn_like(augmented[frame_idx, :, top:top+occlusion_h, left:left+occlusion_w])
                    augmented[frame_idx, :, top:top+occlusion_h, left:left+occlusion_w] = noise
        
        if is_batch:
            augmented = augmented.view(B, T, C, H, W)
        
        return augmented


class ArtificialMotionBlur:
    """
    人工运动模糊
    使用卷积核模拟运动模糊效果
    """
    
    def __init__(
        self,
        prob: float = 0.5,
        kernel_size: int = 15,
        angle_range: Tuple[float, float] = (0, 360),
        length_range: Tuple[float, float] = (5.0, 15.0),
    ):
        """
        Args:
            prob: 应用模糊的概率
            kernel_size: 模糊核大小（必须是奇数）
            angle_range: 运动方向角度范围（度）
            length_range: 模糊长度范围（像素）
        """
        self.prob = prob
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.angle_range = angle_range
        self.length_range = length_range
    
    def _create_motion_blur_kernel(
        self,
        length: float,
        angle: float,
        device: torch.device,
    ) -> torch.Tensor:
        """
        创建运动模糊核
        
        Args:
            length: 模糊长度
            angle: 运动角度（度）
        Returns:
            kernel: (1, 1, kernel_size, kernel_size)
        """
        kernel = torch.zeros((self.kernel_size, self.kernel_size), device=device)
        
        # 将角度转换为弧度
        angle_rad = math.radians(angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # 计算运动方向
        center = self.kernel_size // 2
        
        # 在运动方向上绘制线条
        for i in range(int(length)):
            x = int(center + i * cos_angle)
            y = int(center + i * sin_angle)
            
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1.0
        
        # 归一化
        kernel = kernel / (kernel.sum() + 1e-8)
        
        return kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (T, C, H, W) 或 (B, T, C, H, W)
        Returns:
            augmented_video: 相同形状
        """
        if random.random() > self.prob:
            return video
        
        # 处理不同的输入维度
        is_batch = video.dim() == 5
        if is_batch:
            B, T, C, H, W = video.shape
            video = video.view(B * T, C, H, W)
        else:
            T, C, H, W = video.shape
            video = video.view(T, C, H, W)
        
        device = video.device
        augmented = video.clone()
        
        # 随机选择要模糊的帧
        num_frames_to_blur = random.randint(1, video.size(0))
        frames_to_blur = random.sample(range(video.size(0)), num_frames_to_blur)
        
        for frame_idx in frames_to_blur:
            # 随机生成模糊参数
            length = random.uniform(*self.length_range)
            angle = random.uniform(*self.angle_range)
            
            # 创建模糊核
            kernel = self._create_motion_blur_kernel(length, angle, device)
            
            # 对每个通道应用模糊
            for c in range(C):
                frame_channel = augmented[frame_idx:frame_idx+1, c:c+1, :, :]
                blurred = F.conv2d(
                    frame_channel,
                    kernel,
                    padding=self.kernel_size // 2,
                )
                augmented[frame_idx:frame_idx+1, c:c+1, :, :] = blurred
        
        if is_batch:
            augmented = augmented.view(B, T, C, H, W)
        
        return augmented


class BrightnessPerturbation:
    """
    亮度扰动
    随机调整视频帧的亮度
    """
    
    def __init__(
        self,
        prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.5, 1.5),
        per_frame: bool = True,
    ):
        """
        Args:
            prob: 应用扰动的概率
            brightness_range: 亮度调整范围（乘法因子）
            per_frame: 是否对每一帧独立应用扰动
        """
        self.prob = prob
        self.brightness_range = brightness_range
        self.per_frame = per_frame
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: (T, C, H, W) 或 (B, T, C, H, W)
        Returns:
            augmented_video: 相同形状
        """
        if random.random() > self.prob:
            return video
        
        augmented = video.clone()
        
        if self.per_frame:
            # 对每一帧独立应用扰动
            if augmented.dim() == 5:  # (B, T, C, H, W)
                B, T, C, H, W = augmented.shape
                for b in range(B):
                    for t in range(T):
                        factor = random.uniform(*self.brightness_range)
                        augmented[b, t] = augmented[b, t] * factor
            else:  # (T, C, H, W)
                T, C, H, W = augmented.shape
                for t in range(T):
                    factor = random.uniform(*self.brightness_range)
                    augmented[t] = augmented[t] * factor
        else:
            # 对整个视频应用相同的扰动
            factor = random.uniform(*self.brightness_range)
            augmented = augmented * factor
        
        # 裁剪到[0, 1]范围（假设输入是归一化的）
        augmented = torch.clamp(augmented, 0.0, 1.0)
        
        return augmented


class VideoAugmentation:
    """
    组合所有数据增强策略
    """
    
    def __init__(
        self,
        use_occlusion: bool = True,
        use_blur: bool = True,
        use_brightness: bool = True,
        occlusion_prob: float = 0.5,
        blur_prob: float = 0.5,
        brightness_prob: float = 0.5,
    ):
        """
        Args:
            use_occlusion: 是否使用遮挡增强
            use_blur: 是否使用模糊增强
            use_brightness: 是否使用亮度增强
            occlusion_prob: 遮挡增强的应用概率
            blur_prob: 模糊增强的应用概率
            brightness_prob: 亮度增强的应用概率
        """
        self.occlusion = RandomOcclusion(prob=occlusion_prob) if use_occlusion else None
        self.blur = ArtificialMotionBlur(prob=blur_prob) if use_blur else None
        self.brightness = BrightnessPerturbation(prob=brightness_prob) if use_brightness else None
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """
        按顺序应用所有启用的增强策略
        
        Args:
            video: (T, C, H, W) 或 (B, T, C, H, W)
        Returns:
            augmented_video: 相同形状
        """
        augmented = video.clone()
        
        if self.occlusion is not None:
            augmented = self.occlusion(augmented)
        
        if self.blur is not None:
            augmented = self.blur(augmented)
        
        if self.brightness is not None:
            augmented = self.brightness(augmented)
        
        return augmented

