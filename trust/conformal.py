"""Conformal prediction sets (CLAUDE.md section 10.4), with a separate threshold per camera (Mondrian).

1. For each calibration (val-B) photo: score = 1 - p(true disease).
2. Sort the n scores; q_hat = the k-th smallest, k = ceil((n + 1)(1 - alpha)).
3. New photo: set = every disease with 1 - p(d) <= q_hat.
Guarantee: if the new photo comes from the same distribution as the calibration photos, its true disease is in the set
at least 1 - alpha of the time. A camera shift breaks that, so each camera group gets its own q_hat; a group with
fewer than MIN_GROUP calibration photos (or an unknown camera) uses the most conservative (largest) q_hat.
Worked example (n = 9, alpha = 0.1): k = 9, scores 0.05 ... 0.70 -> q_hat = 0.70; p = AION 0.52, vein occlusion 0.40,
glaucoma 0.05, DR 0.03 -> scores 0.48, 0.60, 0.95, 0.97 -> set {AION, vein occlusion}.
"""
from __future__ import annotations

import math

import numpy as np

MIN_GROUP = 50  # decision log: camera groups under 50 calibration photos use the most conservative q_hat


def conformal_k(n: int, alpha: float) -> int:
    return math.ceil((n + 1) * (1 - alpha))


def q_hat(scores: np.ndarray, alpha: float) -> float:
    """k-th smallest calibration score; inf when k > n (too few photos to promise 1 - alpha)."""
    k = conformal_k(len(scores), alpha)
    return float(np.sort(scores)[k - 1]) if k <= len(scores) else float("inf")


def calib_scores(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1 - p[np.arange(len(y)), y]


def prediction_sets(p: np.ndarray, qh) -> np.ndarray:
    """(N, D) boolean: disease d is in photo i's set. qh: one value or one per photo."""
    qh = np.broadcast_to(np.asarray(qh, dtype=float), (len(p),))
    return (1 - p) <= qh[:, None]


class Mondrian:
    """One q_hat per camera group, the largest one for small or unknown groups."""

    def __init__(self, alpha: float = 0.1, min_group: int = MIN_GROUP):
        self.alpha, self.min_group, self.q, self.n = alpha, min_group, {}, {}

    def fit(self, p: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "Mondrian":
        s = calib_scores(p, y)
        self.q_all = q_hat(s, self.alpha)
        for g in np.unique(groups):
            m = groups == g
            self.n[g] = int(m.sum())
            if m.sum() >= self.min_group:
                self.q[g] = q_hat(s[m], self.alpha)
        self.q_conservative = max([self.q_all, *self.q.values()])
        return self

    def threshold(self, groups: np.ndarray) -> np.ndarray:
        return np.array([self.q.get(g, self.q_conservative) for g in groups])

    def sets(self, p: np.ndarray, groups: np.ndarray) -> np.ndarray:
        return prediction_sets(p, self.threshold(groups))
