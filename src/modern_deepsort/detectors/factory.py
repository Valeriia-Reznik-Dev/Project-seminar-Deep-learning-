"""Detector factory — choose the detector before execution (Rules requirement).

    build_detector({"type": "yolo", "weights": "yolov8n.pt", "min_confidence": 0.3})

Adapters are imported lazily so that installing only one detector's
dependencies is enough to use it (avoids torch/mmcv/nanodet version clashes).
"""
from __future__ import annotations

AVAILABLE = ("yolo", "nanodet", "mmdet")


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
    raise ValueError(f"unknown detector type {kind!r}; available: {AVAILABLE}")
