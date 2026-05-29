# Modern DeepSORT — Project Plan

> Course project: extend the original DeepSORT with modern detection & ReID models,
> add a standalone body-ReID identity database, evaluate on MOT Challenge videos.
> Target: **10/10** (Main full update + Additional task + diverse sources + segmentation + report).

---

## 0. Scoring strategy

Formal grading implies a single path to 10:

| Deliverable | Points |
|---|---|
| Full DeepSORT update (detector **+** ReID) | 5–6 |
| Additional task (standalone ReID DB, clustering, identity & conflict resolution) | +2 |
| > 2 model sources (different repos / hubs) | +1 |
| Working segmentation model | +1 |
| Well-written report | +1 |

Hard constraints (not optional):
- At least **one real-time (≥ 5 FPS in Colab) model combination** that **beats the original
  (unmodified) DeepSORT on EVERY test video** (parameters may differ per video).
- Preserve the **original commit history** of `nwojke/deep_sort`.
- Provide **overlays** for the original implementation and for the best one.
- Provide a **fully-operational Colab** with execution instructions.
- **No barefaced copy-paste** from modern DeepSORT forks.

Recommended literature actually relevant here: *Jiao et al., "A Survey of Deep Learning-based
Object Detection"* — basis for the detector survey section of the report. The other two
recommended sources (snoring detection / NLP) are course-generic and not used directly.

---

## 1. System architecture

Core idea: **keep the original `nwojke/deep_sort` core** (Kalman filter, matching cascade,
Hungarian assignment, `Track`/`Tracker`) and replace only the *ends* via adapters. This both
protects against the copy-paste penalty and preserves the upstream commit history.

```
                              Frame
                                |
        +-----------------------v------------------------+
        |  DetectorFactory  (chosen before run)          |
        |   - YOLOv8        (ultralytics)                |
        |   - NanoDet-Plus  (RangiLyu/nanodet)           |
        |   - RTMDet/FRCNN  (open-mmlab/mmdetection)     |
        |   - [seg] Mask R-CNN (detectron2) / SMP        |
        |  -> normalized bbox (tlwh) + conf, class=person|
        +-----------------------+------------------------+
                                | person crops
        +-----------------------v------------------------+
        |  ReIDFactory  (chosen before run)              |
        |   - OSNet           (torchreid)                |
        |   - ResNet50-IBN    (torchreid / fast-reid)    |
        |   - TransReID/other (fast-reid -> 2nd source)  |
        |  -> L2-normalized descriptors                  |
        +---------------+--------------------+-----------+
                        | tracker feat       | standalone reid feat
        +---------------v-----------+   +----v-----------------------+
        | DeepSORT core (original)  |   | Identity DB (Additional)   |
        | Kalman + matching cascade |   | kNN / centroid clusters    |
        | -> track_id per detection |   | -> identity per descriptor |
        +---------------+-----------+   +----+-----------------------+
                        | track_id          | identity + timestamp
                        +---------+---------+
                                  |
        +-------------------------v----------------------+
        | Identity resolution (window T, majority vote)  |
        | Conflict resolution (reset conflicting tracks) |
        +-------------------------+----------------------+
                                  |
              +-------------------v-------------------+
              | Overlay render + MOTChallenge txt out |
              +---------------------------------------+
```

### Modularity (key principle)
- `detectors/base.py` — abstract `Detector.detect(frame) -> list[Detection(tlwh, conf)]`;
  each model is a separate adapter behind one interface.
- `reid/base.py` — `ReIDExtractor.extract(frame, boxes) -> np.ndarray[N, D]` (L2-norm).
- `deep_sort/` — **almost untouched original**; only the input format changes
  (live detections instead of precomputed `det.txt` + `.npy`).
- `tracking/pipeline.py` — glue: detect -> reid -> tracker.update -> identity resolution.
- `eval/` — TrackEval wrapper (HOTA) + sklearn metrics for detector and ReID.
- `configs/*.yaml` — choose models & parameters before run (per-video allowed).

---

## 2. Model selection (targets the +1 for diverse sources)

To secure "> 2 sources", pick models from **different repos and architecture families**:

**Detectors (≥ 3, need 3 sources):**
- YOLOv8n/s — `ultralytics` (anchor-free, fast, real-time candidate)
- NanoDet-Plus — `RangiLyu/nanodet` (ultra-light, FPS mode)
- RTMDet or Faster R-CNN — `open-mmlab/mmdetection` (accurate, quality mode)

**ReID (≥ 3, ≥ 2 sources):**
- OSNet — `torchreid` (KaiyangZhou) — light, real-time
- ResNet50-IBN — `torchreid` or `fast-reid` — more accurate
- TransReID / a `fast-reid` model — different source & family (transformer)

**Segmentation (+1, additional):**
- Mask R-CNN (`detectron2`) **or** `segmentation_models.pytorch` (Unet/DeepLab).
  Use the mask to produce cleaner crops for ReID (remove background -> cleaner descriptors).

Source count: ultralytics + nanodet + mmdetection + torchreid + (fast-reid) + detectron2
= comfortably > 2 -> +1.

---

## 3. Additional task — standalone ReID + Identity DB (the 9–10 part)

A standalone module on top of the tracker, following the spec pipeline:

1. **Identity init** when a track appears -> create a cluster record in the DB.
2. Per frame: for each detection -> descriptor (standalone ReID model, may differ from the
   tracker's ReID).
3. **Identity lookup**: kNN over the DB -> known identity, or "new" (distance > radius).
4. Append `(identity, frame_timestamp)` to the track's identity history.
5. **Identity resolution**: per active track, take identities over window `[t-T, t]`
   -> majority vote -> final identity for the track.
6. **Conflict resolution**: if two active tracks claim the same identity -> reset
   (or resolve by distance to centroid).

Tunable parameters (show evolution in the report): `k` neighbors, search radius, cluster
representation (per-descriptor vs centroid/center), window `T`, conflict policy, new-identity
threshold.

Standalone ReID metrics (independent of tracker, on GT crops with track_id): Fowlkes–Mallows,
Silhouette, Calinski–Harabasz (sklearn). Use them to pick the best ReID model and clustering
policy **before** integrating into the tracker.

---

## 4. Evaluation protocol

| Measured | How | Tool |
|---|---|---|
| Detector quality | Precision/Recall/F1, GT bbox vs detections (IoU match) | sklearn |
| ReID inside tracker | "disable" the SORT part, run on GT bboxes, measure HOTA | TrackEval |
| Standalone ReID | clustering metrics on GT crops | sklearn |
| Whole system | **HOTA averaged across videos** (main metric) | TrackEval (MOTChallenge protocol) |
| Performance | FPS in Colab (≥ 5 for real-time combo) | custom timer |

**Test videos:** TUD-Campus, TUD-Stadtmitte, KITTI-17, PETS09-S2L1 (MOT15),
MOT16-09, MOT16-11 (MOT16).

---

## 5. Stages / roadmap (tied to commit history)

- **Stage 0 — Repository.** Clone `nwojke/deep_sort` *preserving original history*
  (clone -> new remote, NO squash). First own commit goes on top of their last one.
- **Stage 1 — Baseline.** Run original DeepSORT (their precomputed features + detections)
  on all 6 videos -> HOTA. This is "step 1", the reference. Save the original overlay.
- **Stage 2 — Eval harness.** Wire up TrackEval, prepare seqmaps / folder structure, verify
  the original HOTA reproduces. Nothing proceeds without this.
- **Stage 3 — Detectors.** 3 detector adapters + F1 vs GT comparison. Pick candidates.
- **Stage 4 — ReID.** 3 ReID adapters + standalone evaluation on GT crops (clustering metrics).
- **Stage 5 — Integration.** Replace in core, run HOTA across (detector x ReID) combos, find
  the ≥ 5 FPS combo that beats the original on every video.
- **Stage 6 — Segmentation.** Add seg model, mask crops for ReID, measure HOTA gain.
- **Stage 7 — Additional.** Identity DB + clustering + resolution + conflict. Tune parameters.
- **Stage 8 — Tuning (video-wise).** Parameter evolution, quality/speed tables per video.
- **Stage 9 — Report + Colab + overlays.** Original & best overlays, report with all
  experiments and the conclusion on the optimal model/algorithm/parameter set.

---

## 6. Pitfalls

**Code compatibility**
- The original `deep_sort` is old (numpy/TF1-era) and expects *precomputed* detections
  (`det.txt`) and features (`.npy`). Need a "live detections -> their format" adapter without
  touching the core.
- Coordinate format: the core uses `tlwh`, detectors emit `xyxy` — a frequent bug source.
  One converter, covered by a test.

**MOT data formats (treacherous)**
- MOT15 (TUD, KITTI-17, PETS) and MOT16 have **different GT formats**: MOT16 has a visibility
  flag, confidence, and an **object class** + ignore zones + distractor classes. Failing to
  filter by `class=pedestrian` and to honor ignore regions yields wrong/low HOTA.
- TrackEval is picky about folder structure and seqmaps — half the time goes here. Reproduce
  the baseline first, then everything else.

**Performance vs accuracy (explicitly in the spec)**
- Mask R-CNN / Faster R-CNN can easily drop below 5 FPS in Colab -> unfit for the real-time
  combo. Build the real-time combo on YOLOv8n/NanoDet + OSNet. Heavy models -> "quality" runs.
- Colab FPS depends on the GPU (T4 vs other) — fix the GPU type in the report and time honestly
  (exclude model load / IO).

**MOT16-11 — moving camera** -> Kalman degrades. Optional camera-motion compensation (ECC),
but written ourselves, not copy-pasted from existing trackers.

**ReID details:** ImageNet normalization (mean/std), crop resize (256x128), batch the crops
(else slow), L2-normalize descriptors (else cosine/euclidean are wrong).

**Identity DB:** a growing gallery slows kNN and leaks memory -> cap the size, use centroids
instead of all descriptors. Conflict resolution can oscillate -> tune window `T` and threshold.

**Colab dependencies:** `detectron2` + `torchreid` + `ultralytics` + `mmdetection` conflict on
torch/CUDA versions. Pin versions; install heavy packages optionally (per mode flag), not all
at once.

**Anti-copy-paste:** take the core from the original (allowed and required), but write all new
parts (adapters, DB, resolution) ourselves. Do not pull `deep_sort_realtime` /
`deep_sort_pytorch` verbatim.

---

## 7. Planned repository structure

```
.
├── deep_sort/              # upstream nwojke (history preserved)
├── detectors/{base,yolo,nanodet,mmdet}.py
├── reid/{base,torchreid_ext,fastreid_ext}.py
├── segmentation/{base,detectron2_seg,smp_seg}.py
├── identity/{database,clustering,resolution}.py   # Additional
├── tracking/pipeline.py
├── eval/{trackeval_wrap,detector_metrics,reid_metrics}.py
├── configs/*.yaml          # model/parameter selection
├── scripts/{download_data,run_baseline,run_eval}.py
├── notebooks/colab.ipynb   # working Colab
├── report/report.md
└── overlays/{original,best}/
```

---

## 8. Deliverables checklist

- [ ] Repo with preserved upstream commit history + real development history
- [ ] ≥ 3 selectable detectors from ≥ 3 sources
- [ ] ≥ 3 selectable ReID models from ≥ 2 sources
- [ ] ≥ 1 segmentation model, switchable with detection
- [ ] Standalone body ReID + identity DB (Additional)
- [ ] Real-time (≥ 5 FPS) combo beating original on every video
- [ ] HOTA per video + averaged; detector F1; standalone ReID clustering metrics
- [ ] Parameter-evolution tables (quality & performance)
- [ ] Overlays: original + best
- [ ] Fully-operational Colab with instructions
- [ ] Detailed report (English)
