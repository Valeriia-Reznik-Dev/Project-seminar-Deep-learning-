"""YOLO person detector (source #1: Ultralytics).

Supports the YOLOv8 family (n/s/m/...). COCO class 0 == person.
Weights are auto-downloaded by ultralytics on first use, or pass a local path.
"""
from __future__ import annotations

import numpy as np

from .base import Detector, Detection, xyxy_to_tlwh

PERSON_CLASS = 0


class YoloDetector(Detector):
    name = "yolo"

    def __init__(self, weights: str = "yolov8n.pt", min_confidence: float = 0.3,
                 device: str = "cuda", imgsz: int = 640):
        super().__init__(min_confidence, device)
        from ultralytics import YOLO  # lazy import
        self.model = YOLO(weights)
        self.imgsz = imgsz

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame_bgr, imgsz=self.imgsz, conf=self.min_confidence,
            classes=[PERSON_CLASS], device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        tlwh = xyxy_to_tlwh(xyxy)
        return [Detection(tlwh[i], float(conf[i])) for i in range(len(conf))]
