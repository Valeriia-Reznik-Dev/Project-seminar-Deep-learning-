"""Evaluate a detector vs ground-truth bboxes (Precision / Recall / F1).

Runs the chosen detector over every frame of each sequence and compares to GT
with IoU-greedy matching (Stage 3 detector selection).

    python scripts/eval_detectors.py --mot_root data/mot \
        --detector yolo --weights yolov8n.pt --conf 0.3
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modern_deepsort.detectors import build_detector            # noqa: E402
from modern_deepsort.data.mot_io import load_gt                 # noqa: E402
from modern_deepsort.eval.detector_metrics import evaluate_detector  # noqa: E402

MOT16 = {"MOT16-09", "MOT16-11"}


def run_sequence(detector, seq_dir):
    imgs = sorted(glob.glob(os.path.join(seq_dir, "img1", "*.jpg")))
    det_by_frame = {}
    for frame_id, path in enumerate(imgs, start=1):
        frame = cv2.imread(path)
        dets = detector(frame)
        det_by_frame[frame_id] = (np.array([d.tlwh for d in dets], float)
                                  if dets else np.zeros((0, 4)))
    return det_by_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mot_root", default="data/mot")
    p.add_argument("--detector", required=True, choices=["yolo", "nanodet", "mmdet"])
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--config", default=None, help="config for nanodet/mmdet")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    cfg = {"type": args.detector, "min_confidence": args.conf, "device": args.device}
    if args.detector == "yolo":
        cfg["weights"] = args.weights
    else:
        cfg.update({"config": args.config, "weights": args.weights})
    detector = build_detector(cfg)

    summary = {}
    seqs = sorted(d for d in os.listdir(args.mot_root)
                  if os.path.isdir(os.path.join(args.mot_root, d)))
    for seq in seqs:
        seq_dir = os.path.join(args.mot_root, seq)
        gt_boxes, _ = load_gt(os.path.join(seq_dir, "gt", "gt.txt"),
                              is_mot16=(seq in MOT16))
        det_by_frame = run_sequence(detector, seq_dir)
        res = evaluate_detector(gt_boxes, det_by_frame, args.iou)
        summary[seq] = res
        print(f"{seq:<16} P={res['precision']:.3f} R={res['recall']:.3f} "
              f"F1={res['f1']:.3f}")

    mean_f1 = float(np.mean([s["f1"] for s in summary.values()]))
    print(f"{'MEAN F1':<16} {mean_f1:.3f}")
    out = args.out_json or f"results/detector_{args.detector}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"per_sequence": summary, "mean_f1": mean_f1}, open(out, "w"), indent=2)
    print(f"written {out}")


if __name__ == "__main__":
    main()
