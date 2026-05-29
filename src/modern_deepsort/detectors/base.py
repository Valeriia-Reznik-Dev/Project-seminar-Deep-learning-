"""Common detector interface + box helpers.

Every detector adapter returns a list of `Detection(tlwh, confidence)` with
boxes already filtered to the *person* class and to a minimum confidence.
The DeepSORT core consumes tlwh, so adapters convert from their native xyxy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    tlwh: np.ndarray          # (4,) [x, y, w, h]
    confidence: float

    @property
    def tlbr(self) -> np.ndarray:
        x, y, w, h = self.tlwh
        return np.array([x, y, x + w, y + h], dtype=float)


def xyxy_to_tlwh(boxes: np.ndarray) -> np.ndarray:
    """(N,4) [x1,y1,x2,y2] -> (N,4) [x,y,w,h]."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    out = boxes.copy()
    out[:, 2] = boxes[:, 2] - boxes[:, 0]
    out[:, 3] = boxes[:, 3] - boxes[:, 1]
    return out


class Detector(ABC):
    """Abstract person detector. `name` is used by the factory/config."""

    name: str = "base"

    def __init__(self, min_confidence: float = 0.3, device: str = "cuda"):
        self.min_confidence = min_confidence
        self.device = device

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """Detect persons in a BGR image -> list of Detection (tlwh)."""
        raise NotImplementedError

    def __call__(self, frame_bgr: np.ndarray) -> list[Detection]:
        return self.detect(frame_bgr)


if __name__ == "__main__":  # box-conversion self-test
    xyxy = np.array([[10, 20, 40, 80]])
    tlwh = xyxy_to_tlwh(xyxy)
    assert tlwh.tolist() == [[10, 20, 30, 60]], tlwh
    d = Detection(tlwh[0], 0.9)
    assert d.tlbr.tolist() == [10, 20, 40, 80], d.tlbr
    print("detectors.base self-test OK")
