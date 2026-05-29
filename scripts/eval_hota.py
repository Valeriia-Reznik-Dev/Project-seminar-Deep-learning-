"""Score a results directory with HOTA (averaged across the six videos).

Usage:
    python scripts/eval_hota.py \
        --mot_root data/mot \
        --results_dir results/baseline \
        --tracker baseline
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modern_deepsort.eval.trackeval_wrap import score_hota  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mot_root", default="data/mot")
    p.add_argument("--results_dir", required=True)
    p.add_argument("--tracker", default="modern_deepsort")
    p.add_argument("--work_dir", default="results/_trackeval")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    res = score_hota(args.mot_root, args.results_dir, args.work_dir, args.tracker)
    print("\n== HOTA per sequence ==")
    for seq, h in sorted(res["per_sequence"].items()):
        print(f"  {seq:<16} {h*100:6.2f}")
    print(f"  {'AVERAGE':<16} {res['average_HOTA']*100:6.2f}")

    out = args.out_json or os.path.join(args.results_dir, "hota.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
