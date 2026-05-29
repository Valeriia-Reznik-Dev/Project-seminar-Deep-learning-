"""Standalone ReID evaluation on ground-truth body crops (Stage 4).

Extracts GT person crops (with their track IDs) from the sequences, embeds them
with the chosen ReID model, and reports clustering metrics
(Silhouette / Calinski-Harabasz, and Fowlkes-Mallows against an agglomerative
clustering). This selects the best ReID model independently of the tracker.

    python scripts/eval_reid.py --mot_root data/mot \
        --reid torchreid --model_name osnet_x1_0
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

from modern_deepsort.reid import build_reid                       # noqa: E402
from modern_deepsort.data.mot_io import load_gt                   # noqa: E402
from modern_deepsort.eval.reid_metrics import evaluate_reid       # noqa: E402

MOT16 = {"MOT16-09", "MOT16-11"}


def gather_crops(reid, seq_dir, is_mot16, max_per_id=30):
    gt_boxes, gt_ids = load_gt(os.path.join(seq_dir, "gt", "gt.txt"),
                               is_mot16=is_mot16)
    imgs = {int(os.path.splitext(os.path.basename(p))[0]): p
            for p in glob.glob(os.path.join(seq_dir, "img1", "*.jpg"))}
    feats, labels = [], []
    counts = {}
    for frame in sorted(gt_boxes):
        if frame not in imgs:
            continue
        keep = []
        for box, tid in zip(gt_boxes[frame], gt_ids[frame]):
            if counts.get(tid, 0) >= max_per_id:
                continue
            counts[tid] = counts.get(tid, 0) + 1
            keep.append((box, tid))
        if not keep:
            continue
        frame_img = cv2.imread(imgs[frame])
        boxes = np.array([b for b, _ in keep], float)
        emb = reid.extract(frame_img, boxes)
        feats.append(emb)
        labels.extend(t for _, t in keep)
    if not feats:
        return np.zeros((0, reid.feature_dim)), np.array([])
    return np.vstack(feats), np.asarray(labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mot_root", default="data/mot")
    p.add_argument("--reid", required=True, choices=["torchreid", "fastreid"])
    p.add_argument("--model_name", default="osnet_x1_0")
    p.add_argument("--model_path", default="")
    p.add_argument("--config_file", default="")
    p.add_argument("--weights", default="")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_per_id", type=int, default=30)
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    cfg = {"type": args.reid, "device": args.device}
    if args.reid == "torchreid":
        cfg.update({"model_name": args.model_name, "model_path": args.model_path})
        tag = args.model_name
    else:
        cfg.update({"config_file": args.config_file, "weights": args.weights})
        tag = "fastreid"
    reid = build_reid(cfg)

    from sklearn.cluster import AgglomerativeClustering
    summary = {}
    for seq in sorted(d for d in os.listdir(args.mot_root)
                      if os.path.isdir(os.path.join(args.mot_root, d))):
        emb, y = gather_crops(reid, os.path.join(args.mot_root, seq),
                              seq in MOT16, args.max_per_id)
        if len(y) == 0:
            continue
        n_clusters = len(np.unique(y))
        pred = AgglomerativeClustering(n_clusters=n_clusters,
                                       metric="cosine", linkage="average"
                                       ).fit_predict(emb)
        res = evaluate_reid(emb, y, labels_pred=pred, metric="cosine")
        summary[seq] = res
        print(f"{seq:<16} sil={res['silhouette']:.3f} "
              f"CH={res['calinski_harabasz']:.1f} FMI={res['fowlkes_mallows']:.3f}")

    out = args.out_json or f"results/reid_{tag}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print(f"written {out}")


if __name__ == "__main__":
    main()
