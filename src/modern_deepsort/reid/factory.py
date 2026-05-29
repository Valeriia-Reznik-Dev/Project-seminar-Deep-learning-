"""ReID factory — choose the descriptor before execution (Rules requirement).

    build_reid({"type": "torchreid", "model_name": "osnet_x1_0"})
    build_reid({"type": "fastreid", "config_file": ..., "weights": ...})

>= 3 models across >= 2 sources (torchreid + fastreid). Lazy imports.
"""
from __future__ import annotations

AVAILABLE = ("torchreid", "fastreid")


def build_reid(cfg: dict):
    cfg = dict(cfg)
    kind = cfg.pop("type")
    if kind == "torchreid":
        from .torchreid_ext import TorchReID
        return TorchReID(**cfg)
    if kind == "fastreid":
        from .fastreid_ext import FastReID
        return FastReID(**cfg)
    raise ValueError(f"unknown reid type {kind!r}; available: {AVAILABLE}")
