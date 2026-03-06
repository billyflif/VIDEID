"""Top-level package exports with optional imports.

This file avoids hard failures when optional runtime dependencies
(for example mamba-ssm) are not installed yet.
"""

__all__ = []

try:
    from .models.reid_model import VideoReIDModel
    from .models.losses import VideoReIDCriterion

    __all__ += ["VideoReIDModel", "VideoReIDCriterion"]
except Exception:
    pass

try:
    from .data_augmentation import (
        ArtificialMotionBlur,
        BrightnessPerturbation,
        RandomOcclusion,
        VideoAugmentation,
    )
    from .monitoring import UncertaintyMonitor

    __all__ += [
        "VideoAugmentation",
        "RandomOcclusion",
        "ArtificialMotionBlur",
        "BrightnessPerturbation",
        "UncertaintyMonitor",
    ]
except Exception:
    pass

