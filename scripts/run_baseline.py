"""Reproduce the ORIGINAL (unmodified) DeepSORT on the six test sequences.

Pipeline (all original components):
  det/det.txt  ->  mars-small128 appearance features (tools.generate_detections)
               ->  deep_sort_app.run (original Kalman + matching cascade)
               ->  results/baseline/<seq>.txt   (MOTChallenge format)

This is the Stage-1 reference point. HOTA scoring of the output is done in
Stage 2 (TrackEval). Nothing in the original core is modified here.
"""
import argparse
import os
import sys

# repo root on path so we can import the upstream modules as-is
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import deep_sort_app  # noqa: E402  (upstream single-sequence runner)
from tools import generate_detections as gdet  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Original DeepSORT baseline runner")
    p.add_argument("--mot_root", required=True,
                   help="Dir with staged sequences (data/mot/<SEQ>/...)")
    p.add_argument("--reid_model", default="weights/mars-small128.pb",
                   help="Path to the original mars-small128.pb descriptor")
    p.add_argument("--output_dir", default="results/baseline")
    p.add_argument("--det_cache", default="results/baseline/_detections",
                   help="Where per-sequence <seq>.npy feature files are written")
    # original DeepSORT defaults (0.3 confidence reproduces the paper)
    p.add_argument("--min_confidence", type=float, default=0.3)
    p.add_argument("--min_detection_height", type=int, default=0)
    p.add_argument("--nms_max_overlap", type=float, default=1.0)
    p.add_argument("--max_cosine_distance", type=float, default=0.2)
    p.add_argument("--nn_budget", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.det_cache, exist_ok=True)

    if not os.path.isfile(args.reid_model):
        raise SystemExit(
            f"ReID model not found: {args.reid_model}\n"
            f"Run scripts/download_baseline_reid.sh first.")

    # 1) original appearance descriptor over each sequence's det/det.txt
    print("== generating mars-small128 features for all sequences ==")
    encoder = gdet.create_box_encoder(args.reid_model, batch_size=32)
    gdet.generate_detections(encoder, args.mot_root, args.det_cache)

    # 2) original DeepSORT per sequence
    sequences = sorted(d for d in os.listdir(args.mot_root)
                       if os.path.isdir(os.path.join(args.mot_root, d)))
    for seq in sequences:
        seq_dir = os.path.join(args.mot_root, seq)
        det_npy = os.path.join(args.det_cache, f"{seq}.npy")
        out_txt = os.path.join(args.output_dir, f"{seq}.txt")
        if not os.path.isfile(det_npy):
            print(f"[skip] {seq}: no detections npy"); continue
        print(f"== tracking {seq} ==")
        deep_sort_app.run(
            seq_dir, det_npy, out_txt,
            args.min_confidence, args.nms_max_overlap,
            args.min_detection_height, args.max_cosine_distance,
            args.nn_budget, display=False)

    print(f"== baseline results written to {args.output_dir} ==")


if __name__ == "__main__":
    main()
