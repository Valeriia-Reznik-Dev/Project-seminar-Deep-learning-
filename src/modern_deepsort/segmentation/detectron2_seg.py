"""Person instance segmentation via Detectron2 Mask R-CNN.

Implemented as a `Detector` so it is interchangeable with the box detectors in
the same pipeline/config (Rules: "switch between detection and segmentation
models before execution"). Each Detection carries a full-frame boolean mask so
the ReID stage can zero the background for cleaner appearance descriptors.
COCO person class == 0.
"""
from __future__ import annotations

import numpy as np

from ..detectors.base import Detector, Detection, xyxy_to_tlwh

PERSON_CLASS = 0


class Detectron2Seg(Detector):
    name = "detectron2"

    def __init__(self,
                 config_file: str =
                 "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                 weights: str | None = None, min_confidence: float = 0.5,
                 device: str = "cuda"):
        super().__init__(min_confidence, device)
        from detectron2 import model_zoo                     # lazy
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence
        cfg.MODEL.WEIGHTS = weights or model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        inst = self.predictor(frame_bgr)["instances"].to("cpu")
        if len(inst) == 0:
            return []
        classes = inst.pred_classes.numpy()
        scores = inst.scores.numpy()
        boxes = inst.pred_boxes.tensor.numpy()           # xyxy
        masks = inst.pred_masks.numpy() if inst.has("pred_masks") else None
        out = []
        for i in range(len(inst)):
            if classes[i] != PERSON_CLASS or scores[i] < self.min_confidence:
                continue
            tlwh = xyxy_to_tlwh(boxes[i:i + 1])[0]
            m = masks[i] if masks is not None else None
            out.append(Detection(tlwh, float(scores[i]), mask=m))
        return out
