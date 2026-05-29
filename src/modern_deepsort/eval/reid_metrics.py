"""Standalone ReID quality via clustering metrics (Stage 4 / Stage 7).

The spec recommends evaluating the ReID model independently from the tracker
on preliminary saved ground-truth body crops with their track IDs, using
clustering metrics available in sklearn:
  - Fowlkes-Mallows  (supervised, needs true labels)  -> higher is better
  - Silhouette       (unsupervised, geometry of embedding) -> higher is better
  - Calinski-Harabasz(unsupervised) -> higher is better

Here `labels_true` are the GT track IDs and `embeddings` are the L2-normalized
ReID descriptors. `labels_pred` (e.g. from KMeans / Agglomerative on the
descriptors) is optional and only needed for Fowlkes-Mallows.
"""
from __future__ import annotations

import numpy as np
from sklearn import metrics


def evaluate_reid(embeddings: np.ndarray,
                  labels_true: np.ndarray,
                  labels_pred: np.ndarray | None = None,
                  metric: str = "cosine") -> dict:
    """Compute clustering metrics for a set of ReID embeddings.

    embeddings : (N, D) float
    labels_true: (N,) int  -- GT identity/track id per crop
    labels_pred: (N,) int  -- optional predicted cluster id (for FMI)
    """
    embeddings = np.asarray(embeddings, dtype=float)
    labels_true = np.asarray(labels_true)
    out = {}

    # silhouette / CH use labels_true as the grouping to measure separability
    # of the embedding space w.r.t. the real identities.
    n_labels = len(np.unique(labels_true))
    if 2 <= n_labels < len(labels_true):
        out["silhouette"] = float(
            metrics.silhouette_score(embeddings, labels_true, metric=metric))
        # CH is defined on euclidean geometry
        out["calinski_harabasz"] = float(
            metrics.calinski_harabasz_score(embeddings, labels_true))
    else:
        out["silhouette"] = float("nan")
        out["calinski_harabasz"] = float("nan")

    if labels_pred is not None:
        out["fowlkes_mallows"] = float(
            metrics.fowlkes_mallows_score(labels_true, np.asarray(labels_pred)))
    return out


if __name__ == "__main__":  # tiny self-test: two identities in distinct directions
    rng = np.random.default_rng(0)
    a = rng.normal([1, 0, 0], 0.02, size=(20, 3))
    b = rng.normal([0, 1, 0], 0.02, size=(20, 3))
    emb = np.vstack([a, b])
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9  # L2-normalize
    y = np.array([0] * 20 + [1] * 20)
    res = evaluate_reid(emb, y, labels_pred=y, metric="cosine")
    assert res["silhouette"] > 0.8, res
    assert res["fowlkes_mallows"] == 1.0, res
    print("reid_metrics self-test OK:", res)
