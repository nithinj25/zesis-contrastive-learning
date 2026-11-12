"""Does the camera cost accuracy? Leave-one-camera-out on RFMiD's 3 cameras (CLAUDE.md sections 11, 19.7).

    python -m eval.leave_camera_out trace_v1_384 trace_v1 trace_v1_e100 moco_plain_e15 moco_plain imagenet_r50

For each held-out camera c (test photos of c only):
  out-of-domain: classifier fitted on RFMiD train photos of the OTHER two cameras
  in-domain    : classifier fitted on a random set of RFMiD train photos from ALL cameras, the same size as the
                 out-of-domain set (3 draws). Holding out Topcon 3D OCT-2000 leaves 308 training photos against
                 1,735, so without size matching the gap would mostly measure less data, not the camera.
  gap = in - out (positive = the unseen camera costs accuracy). 95% interval: 1,000 resamples of c's test photos,
  each scoring the mean-over-draws in-domain AUROC minus the out-of-domain AUROC on the same resample.
Classifiers: linear probe (StandardScaler + balanced logistic regression, C = 0.01, the value val-A picked every
time) and kNN (k = 20, t = 0.07). Tasks: macro-AUROC over labels with >= 5 positives in c's test photos and >= 3
in every training set, and the binary Disease_Risk AUROC.

Camera centring (kNN only): subtract each camera's mean feature, then re-normalise. The means come from that
camera's RFMiD photos with labels unused; for the held-out camera this uses its unlabelled test photos, which is
an ablation pending mentor question 5. A constant shift of all of one camera's photos cannot change a linear
probe's AUROC on that camera (every score moves together), so centring is only tested where distances matter.
Also reports the share of each photo's 20 nearest neighbours from its own camera, before and after centring.
Writes outputs/eval/leave_camera_out.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eval.compare import macro_auc
from eval.probes import knn_scores, load_features, probe

C_FIXED = 0.01
DRAWS = 3
MIN_POS_TEST, MIN_POS_TRAIN = 5, 3


def centre_by_camera(X: np.ndarray, cams: np.ndarray) -> np.ndarray:
    """Subtract each camera's mean feature, then L2-normalise again."""
    Z = X.copy()
    for c in np.unique(cams):
        Z[cams == c] -= X[cams == c].mean(0)
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def same_camera_share(X: np.ndarray, cams: np.ndarray, groups: np.ndarray, k: int = 20) -> tuple[float, float]:
    """Share of each photo's k nearest neighbours (cosine; itself and same-eye copies excluded) from its own
    camera, and the chance level (each photo's camera's share of the other photos)."""
    S = X @ X.T
    S[groups[:, None] == groups[None, :]] = -9.0
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    share = (cams[nn] == cams[:, None]).mean()
    freq = pd.Series(cams).value_counts(normalize=True)
    return float(share), float(freq[cams].mean())


def gap_bootstrap(Y: np.ndarray, S_in: list[np.ndarray], S_out: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Mean-over-draws in-domain AUROC minus out-of-domain AUROC, on the same resampled test photos."""
    rng = np.random.default_rng(seed)

    def gap(i=None):
        return np.mean([macro_auc(Y, s, i) for s in S_in]) - macro_auc(Y, S_out, i)

    diffs = np.array([gap(rng.integers(0, len(Y), len(Y))) for _ in range(n)])
    diffs = diffs[~np.isnan(diffs)]
    return {"in": float(np.mean([macro_auc(Y, s) for s in S_in])), "out": macro_auc(Y, S_out), "gap": float(gap()),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}


def fit_scores(kind: str, Xtr, Ytr, Xte) -> np.ndarray:
    if kind == "knn":
        return knn_scores(Xtr, Ytr, Xte)
    return np.column_stack([probe(C_FIXED).fit(Xtr, Ytr[:, j]).predict_proba(Xte)[:, 1] for j in range(Ytr.shape[1])])


def loco(X: np.ndarray, r: pd.DataFrame, kinds=("linear", "knn")) -> dict:
    """r: labelled RFMiD rows (bucket train / test) with columns source, labels, disease_risk, row."""
    labels = sorted({lab for x in r.labels for lab in x})
    Y = np.array([[lab in list(x) for lab in labels] for x in r.labels], float)
    risk = r.disease_risk.fillna(0).values.astype(float)[:, None]
    tr_all = np.where(r.bucket.values == "train")[0]
    out = {}
    for c in sorted(r.source.unique()):
        te = np.where((r.bucket.values == "test") & (r.source.values == c))[0]
        tr_out = tr_all[r.source.values[tr_all] != c]
        rng = np.random.default_rng(0)
        tr_in = [rng.choice(tr_all, len(tr_out), replace=False) for _ in range(DRAWS)]
        keep = [j for j in range(len(labels)) if Y[te, j].sum() >= MIN_POS_TEST
                and min(Y[s, j].sum() for s in [tr_out, *tr_in]) >= MIN_POS_TRAIN]
        res = {"n_test": len(te), "n_train": len(tr_out), "labels": [labels[j] for j in keep]}
        for kind in kinds:
            for task, T in (("labels", Y[:, keep]), ("risk", risk)):
                if T.shape[1] == 0:
                    continue
                s_out = fit_scores(kind, X[tr_out], T[tr_out], X[te])
                s_in = [fit_scores(kind, X[s], T[s], X[te]) for s in tr_in]
                res[f"{kind}_{task}"] = gap_bootstrap(T[te], s_in, s_out)
        out[c] = res
    return out


def run(meta: pd.DataFrame, name: str) -> dict:
    f = load_features(name)
    X = f.attrs["X"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    r = meta[(meta.dataset == "rfmid") & meta.label_ok & meta.bucket.isin(["train", "test"])]
    r = r.merge(f[["image_id", "row"]], on="image_id").reset_index(drop=True)
    Xr = X[r.row.values]
    cams, groups = r.source.values, r.dup_group.fillna(r.image_id).astype(str).values
    Xc = centre_by_camera(Xr, cams)
    res = {"plain": loco(Xr, r), "centred": loco(Xc, r, kinds=("knn",))}
    res["same_camera_share"] = {"plain": same_camera_share(Xr, cams, groups), "centred": same_camera_share(Xc, cams, groups)}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default="outputs/eval/leave_camera_out.json", help="result file (ablation runs write elsewhere)")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet")
    result = {}
    for n in args.names:
        result[n] = run(meta, n)
        print(f"{n} done", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)

    def fmt(g):
        return f"{g['in']:.3f} vs {g['out']:.3f} gap {g['gap']:+.3f} [{g['ci95'][0]:+.3f}, {g['ci95'][1]:+.3f}]"

    cams = list(result[args.names[0]]["plain"].keys())
    for c in cams:
        p0 = result[args.names[0]]["plain"][c]
        print(f"\nHeld-out camera {c}: {p0['n_test']} test photos, {p0['n_train']} training photos, "
              f"{len(p0['labels'])} labels {p0['labels']}")
        for key, part in (("linear_labels", "plain"), ("linear_risk", "plain"), ("knn_labels", "plain"),
                          ("knn_labels", "centred"), ("knn_risk", "plain"), ("knn_risk", "centred")):
            print(f"  {key:14s} {part:8s}")
            for n in args.names:
                g = result[n][part][c].get(key)
                if g:
                    print(f"     {n:16s} in {fmt(g)}")
    print("\nSame-camera share of 20 nearest neighbours (chance in brackets): plain -> centred")
    for n in args.names:
        s = result[n]["same_camera_share"]
        print(f"  {n:16s} {s['plain'][0]:.3f} ({s['plain'][1]:.3f}) -> {s['centred'][0]:.3f}")
