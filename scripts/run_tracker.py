"""Run the modern DeepSORT pipeline over all sequences from a YAML config.

    python scripts/run_tracker.py --config configs/realtime.yaml --score

Config selects detector + ReID + tracker params (Rules: chosen before run).
Writes results/<run_name>/<seq>.txt, a timing.json (FPS per sequence), and -
with --score - HOTA via TrackEval.
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modern_deepsort.detectors import build_detector            # noqa: E402
from modern_deepsort.reid import build_reid                     # noqa: E402
from modern_deepsort.tracking import ModernDeepSort             # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--mot_root", default="data/mot")
    p.add_argument("--out_root", default="results")
    p.add_argument("--score", action="store_true", help="run HOTA after tracking")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))
    run_name = cfg.get("name", os.path.splitext(os.path.basename(args.config))[0])
    out_dir = os.path.join(args.out_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    detector = build_detector(cfg["detector"])
    reid = build_reid(cfg["reid"])
    identity_db = None
    if cfg.get("identity"):
        from modern_deepsort.identity import build_identity_db
        identity_db = build_identity_db(cfg["identity"])
    tp = cfg.get("tracker", {})
    pipe = ModernDeepSort(detector, reid, identity_db=identity_db,
                          max_cosine_distance=tp.get("max_cosine_distance", 0.2),
                          nn_budget=tp.get("nn_budget", 100),
                          nms_max_overlap=tp.get("nms_max_overlap", 1.0),
                          min_confidence=tp.get("min_confidence", 0.3),
                          min_detection_height=tp.get("min_detection_height", 0),
                          use_segmentation_crops=tp.get("use_segmentation_crops", False))

    timing = {}
    for seq in sorted(d for d in os.listdir(args.mot_root)
                      if os.path.isdir(os.path.join(args.mot_root, d))):
        seq_dir = os.path.join(args.mot_root, seq)
        out_file = os.path.join(out_dir, f"{seq}.txt")
        info = pipe.run_sequence(seq_dir, out_file)
        timing[seq] = info
        print(f"{seq:<16} {info['fps']:5.1f} FPS (infer {info['infer_fps']:5.1f})")

    mean_fps = sum(t["fps"] for t in timing.values()) / max(len(timing), 1)
    timing["_mean_fps"] = mean_fps
    json.dump(timing, open(os.path.join(out_dir, "timing.json"), "w"), indent=2)
    print(f"{'MEAN':<16} {mean_fps:5.1f} FPS  (>=5 required for real-time)")

    if args.score:
        from modern_deepsort.eval.trackeval_wrap import score_hota
        res = score_hota(args.mot_root, out_dir,
                         os.path.join(args.out_root, "_trackeval"), run_name)
        print("\n== HOTA ==")
        for seq, h in sorted(res["per_sequence"].items()):
            print(f"  {seq:<16} {h*100:6.2f}")
        print(f"  {'AVERAGE':<16} {res['average_HOTA']*100:6.2f}")
        json.dump(res, open(os.path.join(out_dir, "hota.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
