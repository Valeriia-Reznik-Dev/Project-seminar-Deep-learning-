"""Render a tracking-overlay video from MOTChallenge results.

Draws one consistently-coloured box (+ id) per track over the sequence frames
and writes an .mp4. Used to produce the required overlays for the original
(unmodified) implementation and for the best one.

    python scripts/render_overlay.py \
        --seq_dir data/mot/MOT16-09 \
        --results results/baseline/MOT16-09.txt \
        --out overlays/original/MOT16-09.mp4
"""
import argparse
import glob
import os

import cv2
import numpy as np


def color_for(idx: int):
    rng = np.random.default_rng(int(idx) * 9973 + 1)
    return tuple(int(c) for c in rng.integers(64, 256, size=3))


def load_results(path):
    by_frame = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            v = line.split(",")
            frame, tid = int(float(v[0])), int(float(v[1]))
            x, y, w, h = (float(v[2]), float(v[3]), float(v[4]), float(v[5]))
            by_frame.setdefault(frame, []).append((tid, x, y, w, h))
    return by_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq_dir", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=None)
    args = p.parse_args()

    images = sorted(glob.glob(os.path.join(args.seq_dir, "img1", "*.jpg")))
    if not images:
        raise SystemExit(f"no frames under {args.seq_dir}/img1")
    by_frame = load_results(args.results)

    h0, w0 = cv2.imread(images[0]).shape[:2]
    fps = args.fps or 25.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w0, h0))
    for frame_idx, path in enumerate(images, start=1):
        img = cv2.imread(path)
        for tid, x, y, w, h in by_frame.get(frame_idx, []):
            c = color_for(tid)
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(img, p1, p2, c, 2)
            cv2.putText(img, str(tid), (int(x), max(0, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        writer.write(img)
    writer.release()
    print(f"wrote {args.out} ({len(images)} frames @ {fps:.0f} fps)")


if __name__ == "__main__":
    main()
