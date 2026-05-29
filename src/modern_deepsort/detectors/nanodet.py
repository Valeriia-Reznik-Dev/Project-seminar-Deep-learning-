"""NanoDet person detector (source #2: RangiLyu/nanodet).

NanoDet-Plus is a very light anchor-free detector -> the high-FPS option.
Requires the nanodet package + a config/checkpoint pair. COCO class 0 == person.
"""
from __future__ import annotations

import numpy as np

from .base import Detector, Detection, xyxy_to_tlwh

PERSON_CLASS = 0


class NanoDetDetector(Detector):
    name = "nanodet"

    def __init__(self, config: str, weights: str, min_confidence: float = 0.35,
                 device: str = "cuda"):
        super().__init__(min_confidence, device)
        # lazy imports (nanodet is an optional heavy dependency)
        import torch
        from nanodet.util import cfg, load_config, Logger
        from nanodet.model.arch import build_model
        from nanodet.util import load_model_weight
        from nanodet.data.transform import Pipeline

        load_config(cfg, config)
        self.cfg = cfg
        model = build_model(cfg.model)
        ckpt = torch.load(weights, map_location="cpu")
        load_model_weight(model, ckpt, Logger(-1, use_tensorboard=False))
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)
        self._torch = torch

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        torch = self._torch
        img_info = {"id": 0, "height": frame_bgr.shape[0],
                    "width": frame_bgr.shape[1]}
        meta = {"img_info": img_info, "raw_img": frame_bgr, "img": frame_bgr}
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)) \
            .unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            results = self.model.inference(meta)
        # results: {image_id: {class_id: [ [x1,y1,x2,y2,score], ... ]}}
        dets = results[0].get(PERSON_CLASS, [])
        out = []
        for x1, y1, x2, y2, score in dets:
            if score < self.min_confidence:
                continue
            tlwh = xyxy_to_tlwh([[x1, y1, x2, y2]])[0]
            out.append(Detection(tlwh, float(score)))
        return out
