"""Few-shot disease recognition on frozen features (CLAUDE.md sections 11 and 19.7; "Situation A" in section 4).

    python -m eval.fewshot trace_v1_384 moco_plain_e15 moco_plain imagenet_r50

Dataset: JSIEC-1000 (39 classes, one label per photo, many rare; held out, never in the pool). For each K in
{1, 5, 10} and each of 5 seeds: pick K support photos per class (distinct same-eye groups), score every other
photo of that class whose eye is not in the support set. Classes with fewer than K + 1 eye groups are skipped at
that K. Two classifiers on the same draw:
  prototype: cosine to each class's mean support feature (what TRACE's case index does for a rare disease)
  logistic : multinomial logistic regression on the support photos (C = 1). Features are standardised with a
             StandardScaler fitted once on unlabelled pretraining-pool photos: fitted on K support photos per class,
             it blows noise dimensions up to the size of the signal (the toy test caught this at K = 1)
Metric: top-1 and top-5 accuracy averaged over classes (balanced), mean and 95% t-interval over seeds; plus
per-class top-1 / top-5 for the rare targets. Nothing is fitted on the query photos. Writes outputs/eval/fewshot.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

KS = (1, 5, 10)
SEEDS = 5
RARE = ("Retinitis pigmentosa", "CRVO", "RAO", "Bietti crystalline dystrophy", "VKH disease", "Optic atrophy",
        "Disc swelling and elevation", "Fundus neoplasm")


def draw(y: np.ndarray, groups: np.ndarray, K: int, rng: np.random.Generator):
    """Support / query indices. Support: K photos per class from K different eye groups. Query: the class's
    other photos whose group is not in the support. Returns (support, query, classes kept)."""
    sup, qry, kept = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        gs = np.unique(groups[idx])
        if len(gs) < K + 1:
            continue
        chosen = set(rng.choice(gs, K, replace=False))
        s = [rng.choice(idx[groups[idx] == g]) for g in chosen]
        q = [i for i in idx if groups[i] not in chosen]
        sup += s
        qry += q
        kept.append(c)
    return np.array(sup), np.array(qry), np.array(kept)


def rank_of_truth(scores: np.ndarray, truth_col: np.ndarray) -> np.ndarray:
    """1-based rank of the true class in each row (1 = top). Ties count against the truth."""
    t = scores[np.arange(len(scores)), truth_col]
    return (scores >= t[:, None]).sum(1)


def balanced_topk(rank: np.ndarray, y: np.ndarray, k: int) -> tuple[float, dict]:
    per = {c: float((rank[y == c] <= k).mean()) for c in np.unique(y)}
    return float(np.mean(list(per.values()))), per


def fewshot(X: np.ndarray, y: np.ndarray, groups: np.ndarray, K: int, seeds: int = SEEDS,
            scale_ref: np.ndarray | None = None) -> dict:
    """X: (N, D) L2-normalised features; y: class names; groups: same-eye group ids.
    scale_ref: unlabelled features to fit the logistic model's standardiser on (None = no standardising)."""
    scaler = StandardScaler().fit(scale_ref) if scale_ref is not None else None
    Z = scaler.transform(X) if scaler is not None else X
    res = {"prototype": [], "logistic": []}
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        sup, qry, kept = draw(y, groups, K, rng)
        col = {c: i for i, c in enumerate(kept)}
        tq = np.array([col[c] for c in y[qry]])
        P = np.stack([X[sup][y[sup] == c].mean(0) for c in kept])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Z[sup], y[sup])
        S_lr = lr.predict_proba(Z[qry])[:, [list(lr.classes_).index(c) for c in kept]]
        for name, S in (("prototype", X[qry] @ P.T), ("logistic", S_lr)):
            r = rank_of_truth(S, tq)
            t1, per1 = balanced_topk(r, y[qry], 1)
            t5, per5 = balanced_topk(r, y[qry], 5)
            res[name].append({"top1": t1, "top5": t5, "per_class_top1": per1, "per_class_top5": per5,
                              "n_classes": len(kept), "n_query": len(qry)})
    out = {}
    for name, runs in res.items():
        summ = {"n_classes": runs[0]["n_classes"], "n_query_mean": float(np.mean([r["n_query"] for r in runs]))}
        for m in ("top1", "top5"):
            v = np.array([r[m] for r in runs])
            h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
            summ[m] = float(v.mean())
            summ[f"{m}_ci95"] = [float(v.mean() - h), float(v.mean() + h)]
        classes = runs[0]["per_class_top1"].keys()
        summ["per_class"] = {c: {"top1": float(np.mean([r["per_class_top1"][c] for r in runs])),
                                 "top5": float(np.mean([r["per_class_top5"][c] for r in runs]))} for c in classes}
        out[name] = summ
    return out


def jsiec(meta: pd.DataFrame, name: str):
    from eval.probes import load_features

    f = load_features(name)
    X = f.attrs["X"]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    j = meta[(meta.dataset == "jsiec") & meta.label_ok].merge(f[["image_id", "row"]], on="image_id")
    y = np.array([lab[0] for lab in j.labels])
    groups = j.dup_group.fillna(j.image_id).astype(str).values
    pool = meta[meta.bucket == "pool"].merge(f[["image_id", "row"]], on="image_id").sample(5000, random_state=0)
    return X[j.row.values], y, groups, X[pool.row.values]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default="outputs/eval/fewshot.json", help="result file (ablation runs write elsewhere)")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet")
    result = {}
    for n in args.names:
        X, y, g, ref = jsiec(meta, n)
        result[n] = {f"K{K}": fewshot(X, y, g, K, scale_ref=ref) for K in KS}
    print(f"JSIEC few-shot, {len(y)} photos, 39 classes; balanced top-1 / top-5, mean [95% CI] over {SEEDS} seeds")
    for K in KS:
        for clf in ("prototype", "logistic"):
            print(f"\nK = {K}, {clf} ({result[args.names[0]][f'K{K}'][clf]['n_classes']} classes; chance top-1 "
                  f"{1 / result[args.names[0]][f'K{K}'][clf]['n_classes']:.3f})")
            for n in args.names:
                r = result[n][f"K{K}"][clf]
                rare = "  ".join(f"{c.split()[0]} {r['per_class'][c]['top1']:.2f}/{r['per_class'][c]['top5']:.2f}"
                                 for c in RARE if c in r["per_class"])
                print(f"  {n:16s} top-1 {r['top1']:.3f} [{r['top1_ci95'][0]:.3f}, {r['top1_ci95'][1]:.3f}]  "
                      f"top-5 {r['top5']:.3f}   | {rare}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
