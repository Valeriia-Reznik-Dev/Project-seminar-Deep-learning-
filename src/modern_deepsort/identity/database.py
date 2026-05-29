"""Standalone body-ReID identity database (Additional task).

Pipeline implemented (per the task spec):
  1. A *standalone* ReID model (may differ from the tracker's) describes each
     active track's current crop.
  2. The descriptor is matched against the identity database with a nearest-
     neighbour search (centroid- or kNN-based). The result is an existing
     identity or - if nothing is close enough - a newly created one.
  3. The (identity, frame_timestamp) pair is appended to the track's identity
     history.
  4. After all detections on the frame are processed, each active track's
     identity is *resolved* over a time window T by simple majority vote.
  5. Identity conflicts among active tracks (same identity claimed by >1 track)
     are resolved by resetting the conflicting tracks (simplest policy).

Search policy is configurable: nearest centroid (default) or kNN over stored
descriptors, with an acceptance radius; cluster size is capped.

Output ids: resolved identities are offset by IDENTITY_OFFSET so they never
collide with raw track ids used as a fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

IDENTITY_OFFSET = 100_000


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance from vector a (D,) to rows of b (N,D). Inputs assumed
    L2-normalized; returns (N,)."""
    return 1.0 - b @ a


class IdentityDatabase:
    def __init__(self, reid=None, radius=0.3, k=5, window=30,
                 max_cluster=50, policy="centroid"):
        """
        reid       : standalone ReIDExtractor (or None to disable)
        radius     : max cosine distance to accept an existing identity
        k          : neighbours for the kNN policy
        window     : T, in frames, for majority-vote identity resolution
        max_cluster: cap on descriptors stored per identity
        policy     : 'centroid' (nearest centroid) or 'knn' (k nearest desc.)
        """
        self.reid = reid
        self.radius = radius
        self.k = k
        self.window = window
        self.max_cluster = max_cluster
        self.policy = policy
        self.reset()

    # ---------------------------------------------------------------- lifecycle
    def reset(self):
        self.identities = {}                       # id -> dict(centroid, descs)
        self._next_id = 0
        self.history = defaultdict(list)           # track_id -> [(frame, ident)]
        self.resolved = {}                         # track_id -> output id
        self.current_frame = 0

    # ----------------------------------------------------------------- matching
    def _new_identity(self, desc):
        ident = self._next_id
        self._next_id += 1
        self.identities[ident] = {"centroid": desc.copy(),
                                  "descs": [desc.copy()]}
        return ident

    def _best_identity(self, desc):
        """Return (identity_id, distance) of the closest identity, or (None,inf)."""
        if not self.identities:
            return None, np.inf
        ids = list(self.identities.keys())
        if self.policy == "knn":
            best_id, best_d = None, np.inf
            for ident in ids:
                descs = np.asarray(self.identities[ident]["descs"])
                d = np.sort(_cosine_dist(desc, descs))[:self.k].mean()
                if d < best_d:
                    best_id, best_d = ident, d
            return best_id, best_d
        # centroid policy
        cents = np.asarray([self.identities[i]["centroid"] for i in ids])
        dists = _cosine_dist(desc, cents)
        j = int(np.argmin(dists))
        return ids[j], float(dists[j])

    def _assign(self, desc):
        ident, dist = self._best_identity(desc)
        if ident is None or dist > self.radius:
            return self._new_identity(desc)
        return ident

    def _update_cluster(self, ident, desc):
        c = self.identities[ident]
        c["descs"].append(desc.copy())
        if len(c["descs"]) > self.max_cluster:
            c["descs"].pop(0)
        m = np.mean(c["descs"], axis=0)
        c["centroid"] = m / max(np.linalg.norm(m), 1e-12)

    # -------------------------------------------------------------------- step
    def _descriptor(self, frame, track):
        if self.reid is None:
            return None
        emb = self.reid.extract(frame, np.asarray([track.to_tlwh()], float))
        return emb[0] if len(emb) else None

    def step(self, frame_idx, frame, tracker, detections):
        """Assign identities to active tracks on this frame, then resolve."""
        self.current_frame = frame_idx
        if self.reid is None:
            return
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            desc = self._descriptor(frame, track)
            if desc is None:
                continue
            ident = self._assign(desc)
            self.history[track.track_id].append((frame_idx, ident))
            self._update_cluster(ident, desc)
        self._resolve()

    # --------------------------------------------------------------- resolution
    def _vote(self, track_id):
        lo = self.current_frame - self.window
        votes = [i for (f, i) in self.history[track_id] if f >= lo]
        if not votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    def _resolve(self):
        """Majority vote per active track + reset conflicting tracks."""
        active = [tid for tid, h in self.history.items()
                  if h and h[-1][0] == self.current_frame]
        provisional = {tid: self._vote(tid) for tid in active}

        # detect identities claimed by more than one active track
        owners = defaultdict(list)
        for tid, ident in provisional.items():
            if ident is not None:
                owners[ident].append(tid)
        conflicted = {ident for ident, t in owners.items() if len(t) > 1}

        self.resolved = {}
        for tid, ident in provisional.items():
            if ident is None or ident in conflicted:
                continue  # fall back to raw track id (conflict reset)
            self.resolved[tid] = IDENTITY_OFFSET + ident

    def resolve_conflicts(self):
        """Public hook (final call); resolution already runs every step."""
        self._resolve()

    def resolved_id(self, track_id, fallback):
        return self.resolved.get(track_id, fallback)


def build_identity_db(cfg: dict):
    """Build an IdentityDatabase from config; its ReID is built via build_reid."""
    cfg = dict(cfg)
    reid_cfg = cfg.pop("reid", None)
    reid = None
    if reid_cfg:
        from ..reid import build_reid
        reid = build_reid(reid_cfg)
    return IdentityDatabase(reid=reid, **cfg)
