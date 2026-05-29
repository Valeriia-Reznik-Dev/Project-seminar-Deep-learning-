# Modern DeepSORT

Course project (Deep Learning seminar): extend the original
[DeepSORT](https://github.com/nwojke/deep_sort) with modern, more efficient
**person detection** and **person re-identification (ReID)** models, add a
**standalone body-ReID identity database**, and evaluate on
[MOT Challenge](https://motchallenge.net/) videos using the official
**HOTA** protocol.

This repository is built **on top of the original `nwojke/deep_sort`** and
**preserves its full upstream commit history**. The original DeepSORT core
(Kalman filter, matching cascade, Hungarian assignment) is kept; only the
*ends* of the pipeline (detector, ReID feature extractor) and the identity
layer are added/replaced through adapters.

> See [PLAN.md](PLAN.md) for the full architecture, model choices, evaluation
> protocol, staged roadmap and pitfalls. The original DeepSORT README is kept
> as [ORIGINAL_README.md](ORIGINAL_README.md).

## Test videos (MOT Challenge)

| Sequence | Dataset | Note |
|---|---|---|
| TUD-Campus | 2DMOT2015 | static cam, short |
| TUD-Stadtmitte | 2DMOT2015 | static cam |
| KITTI-17 | 2DMOT2015 | car-mounted, slow |
| PETS09-S2L1 | 2DMOT2015 | surveillance |
| MOT16-09 | MOT16 | static, crowded |
| MOT16-11 | MOT16 | **moving camera** |

## Repository layout

```
deep_sort/            # original DeepSORT core (upstream, kept intact)
application_util/     # original visualization / preprocessing (upstream)
deep_sort_app.py      # original single-sequence runner (upstream)
tools/                # original feature extractor (mars-small128, TF) (upstream)

scripts/              # data + model download, baseline runner   (added)
src/modern_deepsort/  # new modular code (detectors/reid/identity/eval) (added)
configs/              # per-run model & parameter selection (added)
report/               # experiment report (added)
notebooks/            # Colab notebook (added)
```

## Stage 1 — reproduce the original (unmodified) baseline

The baseline is the **original** DeepSORT: the public per-sequence detections
that ship with each MOT sequence (`det/det.txt`) + the original
`mars-small128` appearance descriptor. This gives a fair reference point that
exists for **all six** videos (including the MOT15 ones, for which the
DeepSORT authors never released precomputed features).

```bash
# 1. environment (CPU is enough for the baseline feature extractor via TF2)
pip install -r requirements/baseline.txt

# 2. data: MOT15 (TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1) + MOT16 (09, 11)
bash scripts/download_mot.sh

# 3. original appearance model (mars-small128.pb)
bash scripts/download_baseline_reid.sh

# 4. run the original DeepSORT on all six sequences -> results/baseline/<seq>.txt
python scripts/run_baseline.py \
    --mot_root data/mot \
    --reid_model weights/mars-small128.pb \
    --output_dir results/baseline
```

HOTA scoring of `results/baseline/` is wired up in **Stage 2** (TrackEval).

## Run everything (Colab)

Open [notebooks/colab.ipynb](notebooks/colab.ipynb) on a GPU runtime and run
top to bottom: it installs deps, downloads data + models, reproduces the
baseline, selects detector/ReID, runs the real-time / quality / segmentation /
identity-DB configurations, sweeps parameters, scores HOTA, and renders
overlays. Fill the result tables in [report/report.md](report/report.md) with
the printed numbers.

## Configurations

| Config | Detector | ReID | Notes |
|---|---|---|---|
| [realtime](configs/realtime.yaml) | YOLOv8n | OSNet | ≥ 5 FPS target |
| [quality](configs/quality.yaml) | YOLOv8m | ResNet50-IBN | higher HOTA |
| [segmentation](configs/segmentation.yaml) | Mask R-CNN | OSNet (masked crops) | +1 segmentation |
| [additional](configs/additional.yaml) | YOLOv8n | OSNet + identity DB (ResNet50-IBN) | Additional task |

```bash
python scripts/run_tracker.py --config configs/realtime.yaml --score
python scripts/sweep_params.py --config configs/realtime.yaml \
    --param max_cosine_distance --values 0.1,0.2,0.3,0.4
python scripts/render_overlay.py --seq_dir data/mot/MOT16-09 \
    --results results/realtime_yolov8n_osnet/MOT16-09.txt \
    --out overlays/best/MOT16-09.mp4
```

## Status

See the staged roadmap in [PLAN.md](PLAN.md). Heavy runs (model inference,
HOTA over all sequences) run in the Colab notebook with a GPU; the code here is
structured so the same entry points work locally and in Colab. The pipeline,
metrics, integration with the original core, segmentation mask path, identity
database and overlay renderer are all unit/smoke-tested locally with dummy
models.
