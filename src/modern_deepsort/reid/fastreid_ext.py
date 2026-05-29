"""ReID via FastReID (source #2: JDAI-CV/fast-reid).

A different repository/family (e.g. BoT, SBS, AGW, or a transformer backbone)
-> satisfies "more than two model sources" together with torchreid. Build from
a FastReID config + checkpoint.
"""
from __future__ import annotations

import numpy as np

from .base import ReIDExtractor


class FastReID(ReIDExtractor):
    name = "fastreid"

    def __init__(self, config_file: str, weights: str, device: str = "cuda",
                 input_size=(256, 128)):
        super().__init__(device, input_size)
        import torch
        from fastreid.config import get_cfg          # lazy
        from fastreid.modeling.meta_arch import build_model
        from fastreid.utils.checkpoint import Checkpointer

        cfg = get_cfg()
        cfg.merge_from_file(config_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = device
        cfg.freeze()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.model.eval()
        Checkpointer(self.model).load(weights)
        self._feature_dim = cfg.MODEL.BACKBONE.FEAT_DIM
        self._torch = torch
        # FastReID expects (H, W) from config
        self.input_size = tuple(cfg.INPUT.SIZE_TEST)

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self._torch
        H, W = self.input_size
        batch = []
        for c in crops:
            import cv2
            img = cv2.resize(c, (W, H))                # c is RGB
            batch.append(torch.as_tensor(img.transpose(2, 0, 1), dtype=torch.float32))
        x = torch.stack(batch).to(self.device)          # FastReID normalizes inside
        with torch.no_grad():
            feats = self.model({"images": x})
        return feats.cpu().numpy()
