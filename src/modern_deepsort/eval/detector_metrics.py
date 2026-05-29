"""Detector quality vs ground-truth bboxes: Precision / Recall / F1.

Used in Stage 3 to pick detectors. Matching is greedy by IoU at a fixed
threshold (default 0.5), per frame, one GT <-> one detection. Follows the
spec: "compare Precision + Recall or F1 score between ground-truth bboxes and
bboxes from your detector".

bbox format everywhere here: (x, y, w, h) in pixels (MOTChallenge tlwh).
"""
from __future__ import annotations

import numpy as np


def iou_matrix(gt: np.ndarray, det: np.ndarray) -> np.ndarray:
    """IoU between every GT (M,4) and detection (N,4), tlwh -> (M,N)."""
    if len(gt) == 0 or len(det) == 0:
        return np.zeros((len(gt), len(det)), dtype=float)
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    dx1, dy1 = det[:, 0], det[:, 1]
    dx2, dy2 = det[:, 0] + det[:, 2], det[:, 1] + det[:, 3]

    ix1 = np.maximum(gx1[:, None], dx1[None, :])
    iy1 = np.maximum(gy1[:, None], dy1[None, :])
    ix2 = np.minimum(gx2[:, None], dx2[None, :])
    iy2 = np.minimum(gy2[:, None], dy2[None, :])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    area_g = (gt[:, 2] * gt[:, 3])[:, None]
    area_d = (det[:, 2] * det[:, 3])[None, :]
    union = area_g + area_d - inter
    return np.where(union > 0, inter / union, 0.0)


def match_frame(gt: np.ndarray, det: np.ndarray, iou_thr: float = 0.5):
    """Greedy one-to-one IoU matching for a single frame.

    Returns (tp, fp, fn).
    """
    M, N = len(gt), len(det)
    if M == 0:
        return 0, N, 0
    if N == 0:
        return 0, 0, M
    iou = iou_matrix(gt, det)
    pairs = [(iou[i, j], i, j) for i in range(M) for j in range(N)
             if iou[i, j] >= iou_thr]
    pairs.sort(reverse=True)
    used_g, used_d = set(), set()
    tp = 0
    for _, i, j in pairs:
        if i in used_g or j in used_d:
            continue
        used_g.add(i); used_d.add(j); tp += 1
    fp = N - len(used_d)
    fn = M - len(used_g)
    return tp, fp, fn


def prf1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    return precision, recall, f1


def evaluate_detector(gt_by_frame: dict, det_by_frame: dict, iou_thr: float = 0.5):
    """Aggregate P/R/F1 over a sequence.

    gt_by_frame / det_by_frame: {frame_id: np.ndarray[K,4] tlwh}.
    """
    TP = FP = FN = 0
    for frame in sorted(set(gt_by_frame) | set(det_by_frame)):
        gt = np.asarray(gt_by_frame.get(frame, np.zeros((0, 4))), float).reshape(-1, 4)
        det = np.asarray(det_by_frame.get(frame, np.zeros((0, 4))), float).reshape(-1, 4)
        tp, fp, fn = match_frame(gt, det, iou_thr)
        TP += tp; FP += fp; FN += fn
    p, r, f = prf1(TP, FP, FN)
    return {"precision": p, "recall": r, "f1": f, "tp": TP, "fp": FP, "fn": FN}


if __name__ == "__main__":  # tiny self-test
    gt = {1: [[0, 0, 10, 10], [50, 50, 10, 10]]}
    det = {1: [[1, 1, 10, 10], [200, 200, 5, 5]]}  # 1 good match, 1 FP, 1 FN
    res = evaluate_detector(gt, det, 0.5)
    assert res["tp"] == 1 and res["fp"] == 1 and res["fn"] == 1, res
    assert abs(res["f1"] - 0.5) < 1e-9, res
    print("detector_metrics self-test OK:", res)
