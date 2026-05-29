"""MMDetection person detector (source #3: open-mmlab/mmdetection).

Gives access to high-accuracy detectors (RTMDet, Faster R-CNN, ...) -> the
quality option. COCO class 0 == person. Pass a config + checkpoint.
"""
from __future__ import annotations

import numpy as np

from .base import Detector, Detection, xyxy_to_tlwh

PERSON_CLASS = 0


class MMDetDetector(Detector):
    name = "mmdet"

    def __init__(self, config: str, weights: str, min_confidence: float = 0.3,
                 device: str = "cuda"):
        super().__init__(min_confidence, device)
        from mmdet.apis import init_detector, inference_detector  # lazy
        self._infer = inference_detector
        self.model = init_detector(config, weights, device=device)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        result = self._infer(self.model, frame_bgr)
        # mmdet 3.x: result.pred_instances has bboxes/scores/labels
        inst = result.pred_instances
        bboxes = inst.bboxes.cpu().numpy()
        scores = inst.scores.cpu().numpy()
        labels = inst.labels.cpu().numpy()
        keep = (labels == PERSON_CLASS) & (scores >= self.min_confidence)
        bboxes, scores = bboxes[keep], scores[keep]
        tlwh = xyxy_to_tlwh(bboxes)
        return [Detection(tlwh[i], float(scores[i])) for i in range(len(scores))]
