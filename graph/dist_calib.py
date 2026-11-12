"""Distribution calibration for rare diseases (Yang et al., "Free Lunch for Few-shot Learning", ICLR 2021,
arXiv 2101.06395): borrow the spread of well-sampled diseases to create extra feature points for a rare one.

With 14 RP photos we cannot estimate how RP photos vary. For each real rare photo x (frozen feature, after the
Tukey transform x^lambda):
  1. find the k common diseases whose mean feature is nearest to x (Euclidean),
  2. calibrated mean   mu'    = (w * x + sum of their means) / (w + k)      (paper: w = 1; w = None keeps mu' = x)
     calibrated spread Sigma' = average of their covariances + alpha * I
  3. draw n points from N(mu', Sigma').
A classifier is then trained on real photos of every disease + the rare support photos + the drawn points.

Two departures from the paper's code, both stated:
  - alpha is added to the diagonal (shrinkage), not to every entry, and is tuned on the practice task.
  - no covariance matrix is ever formed (2048 x 2048 from ~100 photos is rank-deficient): a draw from the empirical
    covariance of class i is R_i^T g with R_i = (X_i - mean_i) / sqrt(n_i - 1), g ~ N(0, I_{n_i}). It has exactly
    covariance R_i^T R_i = Sigma_i, so the average over k classes is sum_i sqrt(1/k) R_i^T g_i.
"""
from __future__ import annotations

import numpy as np


def tukey(X: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """Tukey's ladder of powers (paper section 3.2): makes skewed non-negative features more Gaussian."""
    X = np.clip(np.asarray(X, np.float64), 0, None)
    return np.log(X + 1e-6) if lam == 0 else np.power(X, lam)


class BaseStats:
    """Mean and covariance factor of each well-sampled (base) disease, in the transformed space."""

    def __init__(self, X_by_class: dict):
        self.names = list(X_by_class)
        self.means = np.stack([np.asarray(X_by_class[c], np.float64).mean(0) for c in self.names])
        self.R = [(np.asarray(X_by_class[c], np.float64) - m) / np.sqrt(len(X_by_class[c]) - 1)
                  for c, m in zip(self.names, self.means)]

    def nearest(self, x: np.ndarray, k: int) -> np.ndarray:
        return np.argsort(np.linalg.norm(self.means - x, axis=1), kind="stable")[:k]

    def calibrated_mean(self, x: np.ndarray, k: int, w: float | None = 1.0) -> np.ndarray:
        if w is None:
            return np.asarray(x, np.float64)
        top = self.nearest(x, k)
        return (w * x + self.means[top].sum(0)) / (w + k)

    def sample(self, x: np.ndarray, n: int, k: int = 2, alpha: float = 0.0, w: float | None = 1.0,
               rng: np.random.Generator | None = None) -> np.ndarray:
        """n points from N(calibrated mean, average covariance of the k nearest base diseases + alpha I)."""
        rng = rng or np.random.default_rng(0)
        top = self.nearest(x, k)
        out = np.tile(self.calibrated_mean(x, k, w), (n, 1))
        for i in top:
            out += np.sqrt(1.0 / k) * (rng.standard_normal((n, self.R[i].shape[0])) @ self.R[i])
        if alpha > 0:
            out += np.sqrt(alpha) * rng.standard_normal(out.shape)
        return out


def augment(support: np.ndarray, stats: BaseStats, n_total: int, k: int = 2, alpha: float = 0.0,
            w: float | None = 1.0, seed: int = 0) -> np.ndarray:
    """About n_total synthetic points for one rare disease, split evenly over its support photos (as the paper)."""
    rng = np.random.default_rng(seed)
    per = max(1, int(round(n_total / len(support))))
    return np.concatenate([stats.sample(x, per, k, alpha, w, rng) for x in support])
