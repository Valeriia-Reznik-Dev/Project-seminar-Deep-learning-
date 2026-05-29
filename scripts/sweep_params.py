"""Parameter sweep: show the influence of tracker parameters on HOTA & FPS.

Fixes a detector + ReID (from a base YAML) and sweeps a small grid of tracker
parameters, logging averaged HOTA and mean FPS per configuration. The produced
CSV feeds the "parameter evolution" tables/plots in the report.

    python scripts/sweep_params.py --config configs/realtime.yaml \
        --param max_cosine_distance --values 0.1,0.2,0.3,0.4
"""
import argparse
import csv
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modern_deepsort.detectors import build_detector            # noqa: E402
from modern_deepsort.reid import build_reid                     # noqa: E402
from modern_deepsort.tracking import ModernDeepSort             # noqa: E402
from modern_deepsort.eval.trackeval_wrap import score_hota      # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--param", required=True,
                   help="tracker param to sweep (e.g. max_cosine_distance)")
    p.add_argument("--values", required=True, help="comma-separated values")
    p.add_argument("--mot_root", default="data/mot")
    p.add_argument("--out_csv", default="results/sweep.csv")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    values = [float(v) for v in args.values.split(",")]

    # build detector/ReID once (reused across the sweep)
    detector = build_detector(cfg["detector"])
    reid = build_reid(cfg["reid"])
    base_tp = dict(cfg.get("tracker", {}))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []
    seqs = sorted(d for d in os.listdir(args.mot_root)
                  if os.path.isdir(os.path.join(args.mot_root, d)))

    for val in values:
        tp = dict(base_tp); tp[args.param] = val
        pipe = ModernDeepSort(
            detector, reid,
            max_cosine_distance=tp.get("max_cosine_distance", 0.2),
            nn_budget=tp.get("nn_budget", 100),
            nms_max_overlap=tp.get("nms_max_overlap", 1.0),
            min_confidence=tp.get("min_confidence", 0.3),
            min_detection_height=tp.get("min_detection_height", 0),
            use_segmentation_crops=tp.get("use_segmentation_crops", False))

        run_dir = os.path.join("results", f"_sweep_{args.param}_{val}")
        os.makedirs(run_dir, exist_ok=True)
        fps = []
        for seq in seqs:
            info = pipe.run_sequence(os.path.join(args.mot_root, seq),
                                     os.path.join(run_dir, f"{seq}.txt"))
            fps.append(info["fps"])
        res = score_hota(args.mot_root, run_dir,
                         os.path.join("results", "_trackeval_sweep"),
                         f"sweep_{val}")
        mean_fps = sum(fps) / max(len(fps), 1)
        rows.append({args.param: val, "avg_HOTA": res["average_HOTA"],
                     "mean_FPS": mean_fps})
        print(f"{args.param}={val}: HOTA={res['average_HOTA']*100:.2f} "
              f"FPS={mean_fps:.1f}")

    with open(args.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[args.param, "avg_HOTA", "mean_FPS"])
        w.writeheader(); w.writerows(rows)
    print(f"written {args.out_csv}")


if __name__ == "__main__":
    main()
