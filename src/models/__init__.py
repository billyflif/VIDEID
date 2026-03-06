"""Model package exports with optional imports."""

__all__ = []

try:
    from .bvs import BayesianVisualStem
    from .mamba_blocks import RDBMambaBlock
    from .heads import MINEEstimator, UncertaintyWeightedAggregator
    from .reid_model import VideoReIDModel
    from .losses import (
        IDLoss,
        VideoReIDCriterion,
        kl_gaussian_regularizer,
        orthogonal_loss,
        temporal_smoothness_loss,
    )

    __all__ = [
        "BayesianVisualStem",
        "RDBMambaBlock",
        "UncertaintyWeightedAggregator",
        "MINEEstimator",
        "IDLoss",
        "VideoReIDCriterion",
        "orthogonal_loss",
        "temporal_smoothness_loss",
        "kl_gaussian_regularizer",
        "VideoReIDModel",
    ]
except Exception:
    pass

