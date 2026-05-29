"""Detector factory — choose the detector before execution (Rules requirement).

    build_detector({"type": "yolo", "weights": "yolov8n.pt", "min_confidence": 0.3})

Adapters are imported lazily so that installing only one detector's
dependencies is enough to use it (avoids torch/mmcv/nanodet version clashes).
"""
from __future__ import annotations

AVAILABLE = ("yolo", "nanodet", "mmdet", "detectron2")


def build_detector(cfg: dict):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "yolo":
        from .yolo import YoloDetector
        return YoloDetector(**cfg)
    if kind == "nanodet":
        from .nanodet import NanoDetDetector
        return NanoDetDetector(**cfg)
    if kind == "mmdet":
        from .mmdet import MMDetDetector
        return MMDetDetector(**cfg)
    if kind == "detectron2":  # segmentation model, switchable like a detector
        from ..segmentation.detectron2_seg import Detectron2Seg
        return Detectron2Seg(**cfg)
    raise ValueError(f"unknown detector type {kind!r}; available: {AVAILABLE}")
