"""OOD check, "have I seen anything like this?" (CLAUDE.md section 10.3).

FAISS inner-product index over L2-normalised training features (inner product = cosine). Score of a new photo =
1 - cosine to its k-th nearest training photo (k = 10): large = unfamiliar. Threshold = the 95th percentile of val-A
scores, so 5% of in-distribution photos are flagged. The same index returns the nearest confirmed cases for the
explanation panel. Per camera (decision log, 9 Nov: 91% of neighbours share the photo's camera): a camera group with
>= MIN_GROUP val-A photos gets its own threshold; others and unseen cameras use the pooled one.
"""
from __future__ import annotations

import faiss
import numpy as np

K = 10
PCT = 95.0
MIN_GROUP = 50


def unit(X: np.ndarray) -> np.ndarray:
    X = np.ascontiguousarray(X, dtype=np.float32)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


class KnnOOD:
    def __init__(self, k: int = K, pct: float = PCT, min_group: int = MIN_GROUP):
        self.k, self.pct, self.min_group = k, pct, min_group

    def fit(self, X_index: np.ndarray, ids=None) -> "KnnOOD":
        Xn = unit(X_index)
        self.index = faiss.IndexFlatIP(Xn.shape[1])
        self.index.add(Xn)
        self.ids = np.asarray(ids) if ids is not None else np.arange(len(Xn))
        return self

    def neighbours(self, X: np.ndarray, n: int):
        sims, idx = self.index.search(unit(X), n)
        return sims, idx

    def score(self, X: np.ndarray) -> np.ndarray:
        sims, _ = self.neighbours(X, self.k)
        return 1 - sims[:, -1]

    def calibrate(self, X_val: np.ndarray, groups=None) -> "KnnOOD":
        s = self.score(X_val)
        self.thr_all = float(np.percentile(s, self.pct))
        self.thr = {}
        if groups is not None:
            groups = np.asarray(groups)
            for g in np.unique(groups):
                if (groups == g).sum() >= self.min_group:
                    self.thr[g] = float(np.percentile(s[groups == g], self.pct))
        return self

    def threshold(self, groups) -> np.ndarray:
        return np.array([self.thr.get(g, self.thr_all) for g in groups])

    def flag(self, X: np.ndarray, groups=None) -> np.ndarray:
        th = self.threshold(groups) if groups is not None else self.thr_all
        return self.score(X) > th
