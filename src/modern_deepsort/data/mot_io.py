"""Read MOTChallenge gt.txt / det.txt files.

MOT format (comma separated):
    frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
For MOT16 gt there are extra meaningful columns:
    ..., conf(=considered flag 0/1), class, visibility
We filter MOT16 gt to pedestrians (class == 1) and the "considered" flag == 1,
matching the standard evaluation. MOT15 gt has all-ones confidence and no class.
"""
from __future__ import annotations

import numpy as np


def _read_raw(path):
    rows = np.loadtxt(path, delimiter=",", ndmin=2) if _nonempty(path) else \
        np.zeros((0, 10))
    return rows


def _nonempty(path):
    try:
        with open(path) as fh:
            return any(line.strip() for line in fh)
    except OSError:
        return False


def load_detections(path, min_confidence=None):
    """Return {frame: ndarray[K,4]} of tlwh boxes (and conf if needed)."""
    rows = _read_raw(path)
    by_frame = {}
    for r in rows:
        frame = int(r[0])
        conf = r[6] if len(r) > 6 else 1.0
        if min_confidence is not None and conf < min_confidence:
            continue
        by_frame.setdefault(frame, []).append(r[2:6])
    return {f: np.asarray(b, float).reshape(-1, 4) for f, b in by_frame.items()}


def load_gt(path, is_mot16=False, min_visibility=0.0):
    """Return ({frame: ndarray[K,4] tlwh}, {frame: ndarray[K] ids}).

    For MOT16 gt: keep only pedestrian class (==1) with considered flag (==1)
    and visibility >= min_visibility.
    """
    rows = _read_raw(path)
    boxes, ids = {}, {}
    for r in rows:
        if is_mot16 and len(r) >= 9:
            considered, cls = r[6], int(r[7])
            visibility = r[8] if len(r) > 8 else 1.0
            if considered < 1 or cls != 1 or visibility < min_visibility:
                continue
        frame = int(r[0])
        boxes.setdefault(frame, []).append(r[2:6])
        ids.setdefault(frame, []).append(int(r[1]))
    boxes = {f: np.asarray(b, float).reshape(-1, 4) for f, b in boxes.items()}
    ids = {f: np.asarray(i, int) for f, i in ids.items()}
    return boxes, ids
