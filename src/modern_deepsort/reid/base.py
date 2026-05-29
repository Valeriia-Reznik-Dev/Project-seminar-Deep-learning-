"""Common ReID interface.

A ReID extractor turns a frame + person boxes (tlwh) into one L2-normalized
appearance descriptor per box. Cropping/resizing is shared here; subclasses
implement only `_embed(crops)` for a list of RGB uint8 HxWx3 crops.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np


def crop_boxes(frame_bgr: np.ndarray, boxes_tlwh: np.ndarray,
               size=(256, 128)) -> list[np.ndarray]:
    """Crop person boxes and resize to (H, W); returns RGB uint8 crops.

    Boxes are clipped to the image; degenerate boxes yield a zero crop so the
    output length always matches the number of input boxes (index alignment
    with detections must be preserved).
    """
    H, W = frame_bgr.shape[:2]
    out = []
    for x, y, w, h in np.asarray(boxes_tlwh, float).reshape(-1, 4):
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        if x2 <= x1 or y2 <= y1:
            out.append(np.zeros((size[0], size[1], 3), np.uint8))
            continue
        patch = frame_bgr[y1:y2, x1:x2]
        patch = cv2.resize(patch, (size[1], size[0]))
        out.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    return out


def _crop_boxes_masked(frame_bgr, boxes_tlwh, masks, size=(256, 128)):
    """Like crop_boxes but zeroes the background (mask==0) within each box."""
    H, W = frame_bgr.shape[:2]
    out = []
    for (x, y, w, h), m in zip(np.asarray(boxes_tlwh, float).reshape(-1, 4), masks):
        x1, y1 = max(0, int(round(x))), max(0, int(round(y)))
        x2, y2 = min(W, int(round(x + w))), min(H, int(round(y + h)))
        if x2 <= x1 or y2 <= y1:
            out.append(np.zeros((size[0], size[1], 3), np.uint8)); continue
        patch = frame_bgr[y1:y2, x1:x2].copy()
        if m is not None:
            mp = m[y1:y2, x1:x2].astype(bool)
            patch[~mp] = 0
        patch = cv2.resize(patch, (size[1], size[0]))
        out.append(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
    return out


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


class ReIDExtractor(ABC):
    name = "base"

    def __init__(self, device: str = "cuda", input_size=(256, 128)):
        self.device = device
        self.input_size = input_size

    @abstractmethod
    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """RGB crops -> (N, D) raw features (not yet normalized)."""
        raise NotImplementedError

    def extract(self, frame_bgr: np.ndarray, boxes_tlwh: np.ndarray,
                masks=None) -> np.ndarray:
        """Embed person boxes. If ``masks`` (a list of full-frame bool arrays
        aligned with the boxes) is given, the background is zeroed inside each
        box before cropping (segmentation-cleaned ReID, Stage 6)."""
        boxes = np.asarray(boxes_tlwh, float).reshape(-1, 4)
        if len(boxes) == 0:
            return np.zeros((0, self.feature_dim), np.float32)
        if masks is not None:
            frame_bgr = frame_bgr  # keep original; apply per-box during crop
            crops = _crop_boxes_masked(frame_bgr, boxes, masks, self.input_size)
        else:
            crops = crop_boxes(frame_bgr, boxes, self.input_size)
        feats = self._embed(crops)
        return l2_normalize(feats)

    @property
    def feature_dim(self) -> int:
        return getattr(self, "_feature_dim", 512)


if __name__ == "__main__":  # crop alignment / normalization self-test
    img = np.zeros((100, 200, 3), np.uint8)
    boxes = np.array([[10, 10, 30, 60], [-5, -5, 5, 5], [180, 90, 50, 50]])
    crops = crop_boxes(img, boxes, (256, 128))
    assert len(crops) == 3 and all(c.shape == (256, 128, 3) for c in crops)
    feats = l2_normalize(np.random.rand(3, 512))
    assert np.allclose(np.linalg.norm(feats, axis=1), 1.0)
    print("reid.base self-test OK")
