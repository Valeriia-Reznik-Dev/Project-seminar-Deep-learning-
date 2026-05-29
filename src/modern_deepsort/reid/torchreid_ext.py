"""ReID via torchreid (source #1: KaiyangZhou/deep-person-reid).

Exposes the whole model zoo; we use two distinct families/models:
    - osnet_x1_0    (light, real-time)
    - resnet50      / resnet50_ibn_a (heavier, more accurate)
torchreid's FeatureExtractor handles its own preprocessing from RGB crops.
"""
from __future__ import annotations

import numpy as np

from .base import ReIDExtractor

# feature dimensions for the common backbones
_DIMS = {"osnet_x1_0": 512, "osnet_x0_25": 512, "resnet50": 2048,
         "resnet50_ibn_a": 2048, "mlfn": 1024}


class TorchReID(ReIDExtractor):
    name = "torchreid"

    def __init__(self, model_name: str = "osnet_x1_0", model_path: str = "",
                 device: str = "cuda", input_size=(256, 128)):
        super().__init__(device, input_size)
        from torchreid.utils import FeatureExtractor  # lazy
        self.extractor = FeatureExtractor(
            model_name=model_name, model_path=model_path or "",
            image_size=list(input_size), device=device)
        self._feature_dim = _DIMS.get(model_name, 512)
        self.model_name = model_name

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        # FeatureExtractor accepts a list of HxWx3 RGB numpy arrays
        feats = self.extractor(crops)
        return feats.cpu().numpy()
