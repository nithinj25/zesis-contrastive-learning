"""Edge probabilities p(f | d), p(f | not d) and disease priors for the Bayesian scorer (CLAUDE.md 9.3, 9.4, 19.8).

Two kinds of disease:
  zero-shot (in graph/edges.csv, no training photos: AION, CRVO): a cited row gives mu = the share of eyes with the
      SIGN. The scorer's evidence is the DETECTOR's output, so mu is put on the detector's scale (measurement model):
          p(f | d) = mu * sens_f + (1 - mu) * fpr_f
      sens_f = mean out-of-fold detector output on head-training photos labelled f, fpr_f = the same on photos not
      labelled f (head-training only, nothing held out). Using mu directly (the first version) failed: detectors
      calibrated to RFMiD's sparse labels give modest outputs even on clear signs, so every "typical" finding read as
      strong evidence against, and all 108 AION / CRVO photos ranked those diseases near last.
      A finding with no row gets the background rate (no information either way, E ~ 0), not a guess.
  data diseases (enough head-training photos): soft counts. RFMiD labels the disease, not its signs (of 57 CRVO
      photos only 3 also carry a haemorrhage label), so label co-occurrence would describe labelling habits, not
      eyes. Instead the counts are the calibrated detector outputs on the disease's photos, predicted out-of-fold:
          p(f | d) = (sum_{i in d} q_if + s * mu0_f) / (n_d + s)        mu0_f = mean q_f over head-training, s = 10
      This is section 9.4's Beta update with a neutral prior (no medical claim), and it matches what the detectors
      will output at test time.
  p(f | not d) = (sum_{i not in d} q_if + 1) / (n_not_d + 2) over head-training (section 9.4's (n + 1) / (N + 2)).
Priors: (n_d + 1) / (N + D), zero-shot diseases get pi_min, all floored at pi_min and renormalised (19.8).
Clamp every probability to [0.01, 0.99] (9.3).
"""
from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

S_PRIOR = 10.0
PI_MIN = 0.005
MIN_DISEASE_PHOTOS = 5  # data diseases with fewer head-training photos (CRAO 3, BRAO 2) are left out


def on_detector_scale(mu: float, sens: float, fpr: float) -> float:
    """Expected detector output for a disease whose eyes show the sign with probability mu."""
    return mu * sens + (1 - mu) * fpr


def beta_smooth(n_fd: float, n_d: float, mu: float, s: float = S_PRIOR) -> float:
    """Section 9.4: (n_fd + s mu) / (n_d + s). Example: (7 + 9) / (9 + 10) = 0.842."""
    return (n_fd + s * mu) / (n_d + s)


def priors(counts: dict, zero_shot: list, n_total: int, pi_min: float = PI_MIN) -> dict:
    D = len(counts) + len(zero_shot)
    raw = {d: (n + 1) / (n_total + D) for d, n in counts.items()}
    raw.update({d: pi_min for d in zero_shot})
    floored = {d: max(v, pi_min) for d, v in raw.items()}
    z = sum(floored.values())
    return {d: v / z for d, v in floored.items()}


def build(name: str, edges_path: str = "graph/edges.csv", statuses=("draft", "reviewed"), pi_min: float = PI_MIN) -> dict:
    art = joblib.load(os.path.join("artifacts", name, "findings.joblib"))
    findings = art["detectors"] + art["prototypes"]
    r = np.array([art["report"][f]["reliability"] for f in findings])
    q = pd.read_parquet(os.path.join("features", name, "train_findings_oof.parquet")).set_index("image_id")[findings]
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id").loc[q.index]
    Q = q.values.astype(np.float64)
    mu0 = Q.mean(0)
    lab = np.array([[f in list(x) for f in findings] for x in meta.findings])
    sens = np.array([Q[lab[:, j], j].mean() for j in range(len(findings))])
    fpr = np.array([Q[~lab[:, j], j].mean() for j in range(len(findings))])
    edges = pd.read_csv(edges_path)
    edges = edges[edges.status.isin(statuses)]
    zero_shot = sorted(edges.disease.unique())
    counts = pd.Series([d for x in meta.diseases for d in x]).value_counts()
    data_d = [d for d, n in counts.items() if n >= MIN_DISEASE_PHOTOS and d not in zero_shot]
    diseases = data_d + zero_shot
    P, Pn = np.zeros((len(diseases), len(findings))), np.zeros((len(diseases), len(findings)))
    for i, d in enumerate(diseases):
        has = meta.diseases.map(lambda x: d in list(x)).values
        if d in zero_shot:
            P[i] = mu0
            for _, e in edges[edges.disease == d].iterrows():
                if e.finding in findings:
                    j = findings.index(e.finding)
                    P[i, j] = on_detector_scale(e.mu, sens[j], fpr[j])
        else:
            P[i] = [beta_smooth(Q[has, j].sum(), has.sum(), mu0[j]) for j in range(len(findings))]
        Pn[i] = (Q[~has].sum(0) + 1) / ((~has).sum() + 2)
    pri = priors({d: int(counts[d]) for d in data_d}, zero_shot, len(meta), pi_min)
    return {"diseases": diseases, "zero_shot": zero_shot, "findings": findings, "r": r,
            "p": np.clip(P, 0.01, 0.99), "p_not": np.clip(Pn, 0.01, 0.99),
            "prior": np.array([pri[d] for d in diseases]), "edges": edges, "sens": sens, "fpr": fpr}
