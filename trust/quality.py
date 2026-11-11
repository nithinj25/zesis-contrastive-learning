"""Quality gate (CLAUDE.md section 10.1): "is this photo readable, or should it be retaken?"

Multinomial logistic regression on the frozen global feature, trained on EyeQ grades of EyePACS photos
(good / usable / reject). Split by patient (EyePACS file names give patient and eye), so both eyes of one person are on
one side. A photo is sent back for a retake when p(reject) > THRESHOLD; THRESHOLD is set on the training patients
(out-of-fold) to catch 90% of rejects there, and reported on the held-out patients.
Why not the photo-statistics tree of phase 1: its sharpness threshold depends on camera resolution and compression, so
it called 33-61% of RFMiD, RFMiD 2.0 and Paraguay photos "reject" (decision log, 6 Nov).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

GRADES = ["good", "usable", "reject"]
TARGET_RECALL = 0.90


def patient_split(patients: np.ndarray, test_frac: float = 0.2, seed: int = 0) -> np.ndarray:
    """Boolean mask of test rows; every patient's rows fall on one side."""
    rng = np.random.default_rng(seed)
    up = np.unique(patients)
    test_p = set(rng.choice(up, int(round(test_frac * len(up))), replace=False))
    return np.array([p in test_p for p in patients])


class QualityGate:
    def __init__(self, C: float = 0.01):
        self.C = C

    def fit(self, X: np.ndarray, grade: np.ndarray):
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(self.sc.transform(X), grade)
        return self

    def p_reject(self, X: np.ndarray) -> np.ndarray:
        # by name: sklearn sorts classes alphabetically (good, reject, usable), not in GRADES order
        return self.lr.predict_proba(self.sc.transform(X))[:, list(self.lr.classes_).index("reject")]

    def set_threshold(self, p_rej: np.ndarray, is_reject: np.ndarray, recall: float = TARGET_RECALL):
        """Largest threshold that still catches `recall` of the rejects."""
        self.threshold = float(np.quantile(p_rej[is_reject], 1 - recall))
        return self

    def retake(self, X: np.ndarray) -> np.ndarray:
        return self.p_reject(X) > self.threshold


def oof_p_reject(X, grade, groups, C: float, n_splits: int = 5) -> np.ndarray:
    out = np.zeros(len(X))
    for tr, te in GroupKFold(n_splits).split(X, groups=groups):
        out[te] = QualityGate(C).fit(X[tr], grade[tr]).p_reject(X[te])
    return out
