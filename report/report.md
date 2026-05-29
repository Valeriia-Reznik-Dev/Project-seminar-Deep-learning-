# Modern DeepSORT — Experiment Report

> This report follows the required structure: candidate models/algorithms,
> what was selected and why, all experiment results (per video), parameter
> evolution (quality **and** performance), and the conclusion on the optimal
> configuration. Numeric cells marked **_TBD_** are filled from the Colab runs
> (`notebooks/colab.ipynb`); the methodology, candidate set and protocol below
> are final.

## 1. Task and metrics

We extend the original [DeepSORT](https://github.com/nwojke/deep_sort) with
modern detectors and ReID models, add a standalone body-ReID identity database,
and evaluate on six MOT Challenge videos.

- **Main metric:** HOTA, averaged across the six videos (official TrackEval
  MOTChallenge protocol). MOT15 and MOT16 groups are scored with their correct
  per-benchmark preprocessing.
- **Detector quality:** Precision / Recall / F1 vs GT bboxes (IoU-greedy).
- **ReID quality (standalone):** Silhouette, Calinski-Harabasz, Fowlkes-Mallows
  on GT body crops.
- **Performance:** wall-clock FPS in Colab; a *real-time* configuration must
  reach ≥ 5 FPS and beat the original baseline on **every** video.

Videos: TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1 (2DMOT2015);
MOT16-09, MOT16-11 (MOT16). MOT16-11 has a moving camera.

## 2. Candidate models

**Detectors (3 sources).**

| Model | Source | Family | Role |
|---|---|---|---|
| YOLOv8n / YOLOv8s | Ultralytics | anchor-free 1-stage | real-time |
| YOLOv8m | Ultralytics | anchor-free 1-stage | quality |
| NanoDet-Plus | RangiLyu/nanodet | ultra-light FCOS-like | max FPS |
| RTMDet / Faster R-CNN | open-mmlab/mmdetection | 1-/2-stage | quality |

**ReID (2 sources, ≥ 3 models).**

| Model | Source | Family |
|---|---|---|
| OSNet (osnet_x1_0) | torchreid | omni-scale CNN |
| ResNet50-IBN-a | torchreid | CNN + IBN |
| BoT / SBS / transformer | FastReID | strong baseline / ViT |
| mars-small128 | original DeepSORT | baseline CNN (reference) |

**Segmentation (1 model, +1 point).** Detectron2 Mask R-CNN (R50-FPN),
person instance masks → background-cleaned ReID crops; switchable with the box
detectors via config.

This spans **four** model sources (Ultralytics, NanoDet, MMDetection,
torchreid/FastReID, Detectron2) → satisfies the ">2 sources" criterion.

## 3. Step 1 — original (unmodified) baseline

Original DeepSORT = public per-sequence detections (`det/det.txt`) +
`mars-small128` descriptor + the original Kalman/matching core. This reference
exists for all six videos.

| Video | HOTA | MOTA | IDF1 |
|---|---|---|---|
| TUD-Campus | _TBD_ | _TBD_ | _TBD_ |
| TUD-Stadtmitte | _TBD_ | _TBD_ | _TBD_ |
| KITTI-17 | _TBD_ | _TBD_ | _TBD_ |
| PETS09-S2L1 | _TBD_ | _TBD_ | _TBD_ |
| MOT16-09 | _TBD_ | _TBD_ | _TBD_ |
| MOT16-11 | _TBD_ | _TBD_ | _TBD_ |
| **Average** | **_TBD_** | _TBD_ | _TBD_ |

## 4. Detector selection (Precision / Recall / F1)

`scripts/eval_detectors.py`, IoU = 0.5, vs GT pedestrians.

| Detector | mean P | mean R | mean F1 | FPS (Colab) |
|---|---|---|---|---|
| YOLOv8n | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| YOLOv8m | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| NanoDet-Plus | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| MMDet (RTMDet) | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

_Selection rationale (to finalize from numbers):_ YOLOv8n is expected to give
the best F1/FPS trade-off for the real-time configuration; a heavier detector
is kept for the quality configuration.

## 5. ReID selection (standalone clustering metrics)

`scripts/eval_reid.py` on GT crops; agglomerative clustering for FMI.

| ReID model | Silhouette | Calinski-Harabasz | Fowlkes-Mallows |
|---|---|---|---|
| OSNet | _TBD_ | _TBD_ | _TBD_ |
| ResNet50-IBN-a | _TBD_ | _TBD_ | _TBD_ |
| FastReID (BoT/SBS) | _TBD_ | _TBD_ | _TBD_ |
| mars-small128 (ref) | _TBD_ | _TBD_ | _TBD_ |

**ReID-for-tracker (GT boxes, SORT disabled):** averaged HOTA per ReID model →
_TBD_. This isolates appearance quality from detection/motion.

## 6. Full system — combinations

`scripts/run_tracker.py` per config. Each cell: HOTA (and FPS).

| Config | TUD-Campus | TUD-Stadt. | KITTI-17 | PETS09 | MOT16-09 | MOT16-11 | **Avg HOTA** | Mean FPS |
|---|---|---|---|---|---|---|---|---|
| Original baseline | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| realtime (YOLOv8n+OSNet) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| quality (YOLOv8m+R50-IBN) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| segmentation (MaskRCNN+OSNet) | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **+ identity DB (Additional)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

**Real-time requirement:** the *realtime* config must show ≥ 5 FPS and beat the
baseline on every column (per-video parameters allowed).

## 7. Parameter evolution (`scripts/sweep_params.py`)

Show HOTA & FPS as a function of each key parameter; report the chosen value.

- `max_cosine_distance` ∈ {0.1, 0.2, 0.3, 0.4}: _TBD_ (plot)
- `min_confidence` ∈ {0.2, 0.3, 0.4, 0.5}: _TBD_
- `nn_budget` ∈ {30, 50, 100, None}: _TBD_
- detector `imgsz` ∈ {416, 640, 960, 1280}: quality↑ / FPS↓ _TBD_
- Identity DB: `radius`, `window T`, `policy` (centroid vs kNN), `k`: _TBD_

## 8. Standalone identity database (Additional)

Pipeline: standalone ReID descriptor per active track → kNN/centroid lookup →
existing/new identity → per-track identity history (timestamps) → majority vote
over window T → conflict reset. Tuned via the clustering metrics of §5.

- Effect of `policy` (centroid vs kNN) on HOTA/IDF1: _TBD_
- Effect of window `T`: _TBD_
- Effect of acceptance `radius`: _TBD_
- ID switches before/after the identity DB: _TBD_

## 9. Conclusion

- Optimal real-time configuration: _TBD_ (detector + ReID + parameters), with
  averaged HOTA _TBD_ vs baseline _TBD_ at _TBD_ FPS.
- Optimal quality configuration: _TBD_.
- Segmentation-cleaned crops change HOTA by _TBD_.
- The identity database changes HOTA/IDF1 by _TBD_ and reduces ID switches by
  _TBD_.
- Per-video best parameters: _TBD_ table.

## Appendix — reproducibility

All numbers are produced by `notebooks/colab.ipynb` on a Colab GPU (record the
GPU type, e.g. T4). Seeds are fixed where applicable. Commands:
`scripts/run_baseline.py`, `eval_detectors.py`, `eval_reid.py`,
`run_tracker.py`, `sweep_params.py`, `eval_hota.py`, `render_overlay.py`.
