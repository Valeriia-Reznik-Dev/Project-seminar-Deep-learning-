"""Modern DeepSORT pipeline.

Keeps the ORIGINAL DeepSORT core (Kalman filter + matching cascade via
``deep_sort.tracker.Tracker``) and only swaps the ends:
    detection : a modern Detector  (YOLO / NanoDet / MMDet)
    appearance: a modern ReIDExtractor (torchreid / FastReID)

Mirrors the original deep_sort_app.run logic (min-confidence filter, NMS,
predict/update, confirmed-track output) but runs detection + appearance live
per frame instead of reading precomputed .npy files. Output is MOTChallenge
format, identical to the original.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np

# original DeepSORT core lives at the repository root
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2  # noqa: E402
from deep_sort import nn_matching                    # noqa: E402
from deep_sort.detection import Detection as DSDetection  # noqa: E402
from deep_sort.tracker import Tracker                # noqa: E402
from application_util import preprocessing           # noqa: E402


class ModernDeepSort:
    def __init__(self, detector, reid, max_cosine_distance=0.2, nn_budget=100,
                 nms_max_overlap=1.0, min_confidence=0.3, min_detection_height=0,
                 use_segmentation_crops=False,
                 identity_db=None, time_window=30):
        self.detector = detector
        self.reid = reid
        self.nms_max_overlap = nms_max_overlap
        self.min_confidence = min_confidence
        self.min_detection_height = min_detection_height
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget
        # Stage 6: zero the background of ReID crops using segmentation masks
        self.use_segmentation_crops = use_segmentation_crops
        # Additional task (Stage 7): optional standalone identity resolver
        self.identity_db = identity_db
        self.time_window = time_window

    def _new_tracker(self):
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", self.max_cosine_distance, self.nn_budget)
        return Tracker(metric)

    def run_sequence(self, seq_dir, output_file, gt_boxes_by_frame=None):
        """Track a sequence; write MOTChallenge txt. Returns timing info.

        If ``gt_boxes_by_frame`` is given, detection is replaced by those boxes
        (used to evaluate ReID in isolation: "disable the SORT component" /
        ground-truth-box mode from the spec).
        """
        tracker = self._new_tracker()
        if self.identity_db is not None:
            self.identity_db.reset()
        results = []
        images = sorted(glob.glob(os.path.join(seq_dir, "img1", "*.jpg")))

        t_infer, t_total0 = 0.0, time.time()
        for frame_idx, path in enumerate(images, start=1):
            frame = cv2.imread(path)

            # 1) detections (live detector, or GT boxes in ReID-only mode)
            masks = None
            if gt_boxes_by_frame is not None:
                boxes = np.asarray(gt_boxes_by_frame.get(frame_idx,
                                   np.zeros((0, 4))), float).reshape(-1, 4)
                scores = np.ones(len(boxes), float)
            else:
                t0 = time.time()
                dets = self.detector(frame)
                t_infer += time.time() - t0
                boxes = np.array([d.tlwh for d in dets], float) if dets \
                    else np.zeros((0, 4))
                scores = np.array([d.confidence for d in dets], float) if dets \
                    else np.zeros(0)
                if self.use_segmentation_crops and dets:
                    masks = [d.mask for d in dets]

            # height + confidence filter (keep masks aligned)
            if len(boxes):
                keep = (boxes[:, 3] >= self.min_detection_height) & \
                       (scores >= self.min_confidence)
                boxes, scores = boxes[keep], scores[keep]
                if masks is not None:
                    masks = [m for m, k in zip(masks, keep) if k]

            # 2) appearance descriptors (optionally background-masked)
            if len(boxes):
                use_masks = (masks is not None and any(m is not None for m in masks))
                t0 = time.time()
                features = self.reid.extract(frame, boxes,
                                             masks=masks if use_masks else None)
                t_infer += time.time() - t0
            else:
                features = np.zeros((0, getattr(self.reid, "feature_dim", 512)))

            detections = [DSDetection(boxes[i], float(scores[i]), features[i])
                          for i in range(len(boxes))]

            # NMS (as in the original)
            if detections:
                bb = np.array([d.tlwh for d in detections])
                sc = np.array([d.confidence for d in detections])
                idx = preprocessing.non_max_suppression(bb, self.nms_max_overlap, sc)
                detections = [detections[i] for i in idx]

            # 3) tracking step (original core)
            tracker.predict()
            tracker.update(detections)

            # 4) optional standalone identity resolution (Additional task)
            if self.identity_db is not None:
                self.identity_db.step(frame_idx, frame, tracker, detections)

            # 5) collect confirmed tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                x, y, w, h = track.to_tlwh()
                tid = track.track_id
                if self.identity_db is not None:
                    tid = self.identity_db.resolved_id(track.track_id, tid)
                results.append([frame_idx, tid, x, y, w, h])

        if self.identity_db is not None:
            self.identity_db.resolve_conflicts()

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w") as fh:
            for r in results:
                fh.write("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1\n"
                         % (r[0], r[1], r[2], r[3], r[4], r[5]))

        wall = time.time() - t_total0
        n = max(len(images), 1)
        return {"frames": len(images), "wall_s": wall,
                "fps": len(images) / wall if wall > 0 else 0.0,
                "infer_fps": len(images) / t_infer if t_infer > 0 else 0.0}
