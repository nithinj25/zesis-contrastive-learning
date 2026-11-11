"""Calibrated confidence (CLAUDE.md section 10.2): temperature scaling and expected calibration error.

    p = softmax(z / T)       T chosen to minimise log-loss on val-A; the ranking of classes never changes
    ECE = sum_bins (n_b / N) * | accuracy_b - confidence_b |
A bin with 100 of 1,000 predictions that says 0.9 and is right 70% of the time adds 0.1 x 0.2 = 0.02.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    a = z / T
    a = a - a.max(1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(1, keepdims=True)


def nll(z: np.ndarray, y: np.ndarray, T: float) -> float:
    p = softmax(z, T)
    return float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean())


def fit_temperature(z: np.ndarray, y: np.ndarray, bounds=(0.05, 20.0)) -> float:
    """T minimising the mean negative log-likelihood of the true class (searched on log T)."""
    r = minimize_scalar(lambda t: nll(z, y, float(np.exp(t))), bounds=np.log(bounds), method="bounded")
    return float(np.exp(r.x))


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width confidence bins (top-class confidence)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)
