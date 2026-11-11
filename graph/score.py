"""Evidence for and against each disease (CLAUDE.md section 9.5).

    present = log(p / p')        absent = log((1 - p) / (1 - p'))
    E_f(d)  = q_f * present + (1 - q_f) * absent           q_f = calibrated detector probability
    score(d) = log prior(d) + sum_f r_f * E_f(d)           posterior = softmax over diseases
A missing detector output (NaN: finding switched off or localisation failed) is skipped, E = 0 (section 19.8).
The list of E terms is the explanation: positive = evidence for, negative = evidence against.
"""
from __future__ import annotations

import numpy as np

CLAMP = (0.01, 0.99)


def evidence(q: np.ndarray, p: np.ndarray, p_not: np.ndarray) -> np.ndarray:
    """E for every (photo, disease, finding). q: (N, F); p, p_not: (D, F). Returns (N, D, F); NaN q -> 0."""
    p, p_not = np.clip(p, *CLAMP), np.clip(p_not, *CLAMP)
    present = np.log(p / p_not)
    absent = np.log((1 - p) / (1 - p_not))
    qq = np.nan_to_num(q, nan=0.0)[:, None, :]
    E = qq * present[None] + (1 - qq) * absent[None]
    return np.where(np.isnan(q)[:, None, :], 0.0, E)


def score(q: np.ndarray, p: np.ndarray, p_not: np.ndarray, prior: np.ndarray, r: np.ndarray):
    """-> (scores (N, D), posterior (N, D), weighted evidence terms (N, D, F))."""
    terms = evidence(q, p, p_not) * r[None, None, :]
    s = np.log(prior)[None, :] + terms.sum(-1)
    z = np.exp(s - s.max(1, keepdims=True))
    return s, z / z.sum(1, keepdims=True), terms


def explain(terms_row: np.ndarray, findings: list[str], k: int = 3) -> tuple[list, list]:
    """Top-k evidence for and against one disease: lists of (finding, E)."""
    order = np.argsort(terms_row)
    against = [(findings[i], float(terms_row[i])) for i in order[:k] if terms_row[i] < 0]
    for_ = [(findings[i], float(terms_row[i])) for i in order[::-1][:k] if terms_row[i] > 0]
    return for_, against
