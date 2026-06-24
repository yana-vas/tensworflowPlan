"""Reconstruction quality metrics: IoU, Chamfer-L1, Normal Consistency, F-Score.

Standard single-view reconstruction metrics (ONet / 3D-RETR). All functions take
numpy arrays so they are easy to unit-test without a GPU.
"""

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree


def volumetric_iou(
    pred_prob: np.ndarray,
    gt_occ: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Intersection-over-union of occupancy over a shared set of points.

    Args:
        pred_prob: ``(N,)`` predicted occupancy probabilities in [0, 1].
        gt_occ: ``(N,)`` ground-truth occupancy in {0, 1}.
        threshold: Probability above which a point is predicted occupied.

    Returns:
        IoU in [0, 1] (1.0 if both sets are empty).
    """
    pred = pred_prob > threshold
    gt = gt_occ > 0.5
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def _nn_distances(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """For each point in ``a`` return distance to and index of nearest in ``b``."""
    tree = cKDTree(b)
    dist, idx = tree.query(a, k=1)
    return dist, idx


def chamfer_l1(pred_points: np.ndarray, gt_points: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance between two point sets.

    Args:
        pred_points: ``(P, 3)`` points sampled on the predicted surface.
        gt_points: ``(Q, 3)`` points sampled on the ground-truth surface.

    Returns:
        Mean of both directional nearest distances (lower is better);
        ``inf`` if either set is empty.
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")
    d_pred_to_gt, _ = _nn_distances(pred_points, gt_points)
    d_gt_to_pred, _ = _nn_distances(gt_points, pred_points)
    return float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))


def normal_consistency(
    pred_points: np.ndarray,
    pred_normals: np.ndarray,
    gt_points: np.ndarray,
    gt_normals: np.ndarray,
) -> float:
    """Symmetric absolute-cosine agreement of normals at nearest neighbors.

    Args:
        pred_points/pred_normals: ``(P, 3)`` predicted surface points and normals.
        gt_points/gt_normals: ``(Q, 3)`` ground-truth surface points and normals.

    Returns:
        Mean absolute cosine similarity in [0, 1] (higher is better);
        0.0 if either set is empty.
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0

    def _unit(n: np.ndarray) -> np.ndarray:
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-10)

    pn, gn = _unit(pred_normals), _unit(gt_normals)
    _, idx_p2g = _nn_distances(pred_points, gt_points)
    _, idx_g2p = _nn_distances(gt_points, pred_points)
    cos_p2g = np.abs(np.sum(pn * gn[idx_p2g], axis=1))
    cos_g2p = np.abs(np.sum(gn * pn[idx_g2p], axis=1))
    return float(0.5 * (cos_p2g.mean() + cos_g2p.mean()))


def f_score(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    tau: float = 0.02,
) -> float:
    """F-Score: harmonic mean of precision and recall at distance threshold ``tau``.

    Args:
        pred_points: ``(P, 3)`` predicted surface points.
        gt_points: ``(Q, 3)`` ground-truth surface points.
        tau: Distance threshold counting a point as "matched".

    Returns:
        F-score in [0, 1] (higher is better); 0.0 if either set is empty.
    """
    if len(pred_points) == 0 or len(gt_points) == 0:
        return 0.0
    d_pred_to_gt, _ = _nn_distances(pred_points, gt_points)
    d_gt_to_pred, _ = _nn_distances(gt_points, pred_points)
    precision = float((d_pred_to_gt < tau).mean())
    recall = float((d_gt_to_pred < tau).mean())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    rng = np.random.RandomState(0)
    # Identical point sets => perfect scores / zero distance.
    pts = rng.rand(500, 3).astype(np.float32)
    normals = rng.randn(500, 3).astype(np.float32)
    assert chamfer_l1(pts, pts) == 0.0
    assert abs(normal_consistency(pts, normals, pts, normals) - 1.0) < 1e-5
    assert f_score(pts, pts, tau=0.01) == 1.0
    # IoU sanity
    pred = np.array([0.9, 0.9, 0.1, 0.1])
    gt = np.array([1.0, 0.0, 0.0, 1.0])
    iou = volumetric_iou(pred, gt)  # pred occ {0,1}; gt occ {0,3} -> inter 1, union 3
    print("metrics.py self-test: chamfer 0.0, NC 1.0, F 1.0, IoU", round(iou, 3))
    assert abs(iou - (1 / 3)) < 1e-6