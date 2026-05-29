"""Person ReID descriptors (2 sources): torchreid, FastReID."""
from .base import ReIDExtractor, crop_boxes, l2_normalize
from .factory import build_reid, AVAILABLE

__all__ = ["ReIDExtractor", "crop_boxes", "l2_normalize",
           "build_reid", "AVAILABLE"]
