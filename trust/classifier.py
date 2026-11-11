"""The calibrated disease classifier the trust layer qualifies (CLAUDE.md sections 10.2, 10.4).

Scope (10.4: conformal needs one true class): the 6 common diseases with >= 40 head-training photos (normal, DR,
myopia, BRVO, ARMD, CSR) plus a 7th class, OTHER = "abnormal, but none of the 6": photos with findings and no
disease label (43% of val-A). Without it the classifier forced those photos into the 6 (on val-A, 58 of 162 were
called normal: missed abnormalities), and conformal sets were calibrated on a cleaner mix than real photos.
Photos with several diseases are left out (no single true class). Rare diseases (AION, CRVO, RP) are not classes;
the disease bank's margin rule flags them (graph/bank.py, eval/rare_bank.py) and trust/decide.py refers them.
Two candidates, chosen on val-A only (log-loss after temperature, then accuracy):
  logistic : multinomial logistic regression on the frozen global feature (StandardScaler; C by grouped 5-fold CV
             on head training)
  prototype: cosine to each class's mean feature, divided by a temperature
Temperature T is fitted on val-A. Val-B is never touched here (it is the conformal calibration set).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from trust.calibrate import fit_temperature, nll, softmax

OTHER = "other_abnormal"
CLASSES = ["normal", "DR", "myopia", "BRVO", "ARMD", "CSR", OTHER]
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)


def single_dx(meta: pd.DataFrame, rows: pd.Index, classes=CLASSES) -> pd.DataFrame:
    """Labelled photos with one true class: exactly one disease among `classes`, or (OTHER) findings and no disease.
    Adds dz (class name) and y (class index)."""
    m = meta.loc[rows]
    m = m[m.label_ok]
    one = m.diseases.map(len) == 1
    other = (m.diseases.map(len) == 0) & (m.findings.map(len) > 0)
    dz = pd.Series(np.where(one, m.diseases.map(lambda v: v[0] if len(v) else ""), np.where(other, OTHER, "")), index=m.index)
    m = m.assign(dz=dz)
    m = m[m.dz.isin(classes)]
    return m.assign(y=m.dz.map({c: i for i, c in enumerate(classes)}))


class Logistic:
    def __init__(self, C: float):
        self.C = C

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(C=self.C, max_iter=3000).fit(self.sc.transform(X), y)
        return self

    def logits(self, X):
        return self.lr.decision_function(self.sc.transform(X))


class Prototype:
    def fit(self, X, y):
        P = np.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])
        self.P = P / np.linalg.norm(P, axis=1, keepdims=True)
        return self

    def logits(self, X):
        return (X / np.linalg.norm(X, axis=1, keepdims=True)) @ self.P.T


def choose_C(X, y, groups) -> float:
    best, best_c = np.inf, C_GRID[0]
    for C in C_GRID:
        losses = []
        for tr, te in GroupKFold(5).split(X, groups=groups):
            z = Logistic(C).fit(X[tr], y[tr]).logits(X[te])
            losses.append(nll(z, y[te], fit_temperature(z, y[te])))
        if np.mean(losses) < best:
            best, best_c = float(np.mean(losses)), C
    return best_c


class Calibrated:
    """A fitted model + temperature: proba(X) = softmax(logits / T)."""

    def __init__(self, model, T: float, classes=CLASSES):
        self.model, self.T, self.classes = model, T, list(classes)

    def proba(self, X):
        return softmax(self.model.logits(X), self.T)
