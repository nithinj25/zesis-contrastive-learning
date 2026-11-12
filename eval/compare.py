"""Is encoder A really better than encoder B? And does an encoder still see the camera? (CLAUDE.md sections 11, 19.7)

    python -m eval.compare trace_v1 moco_plain_e15 moco_plain imagenet_r50 trace_v1_e100

1. Paired bootstrap (section 11, "1,000-sample bootstrap CIs for every ours-vs-baseline comparison"): the FIRST
   encoder is compared with each of the others on the same test photos. Each resample draws test photos with
   replacement and scores both encoders on that same draw, so the photos' difficulty cancels out. Reported: the
   difference, its 95% interval and the share of resamples where the first encoder is not better (a one-sided p).
   Tasks: RFMiD linear macro-AUROC, IDRiD any-DR AUROC, Paraguay any-DR AUROC (probe fitted on IDRiD).
2. Clean camera probe: RFMiD's three cameras (same hospital, same kind of patients; section 5.5 camera from the
   photo size). Run on all RFMiD photos and on normal eyes only (no label at all), so the disease mix cannot give the
   camera away. Balanced accuracy, 5-fold CV grouped by same-eye group; chance 1/3. The 7-source probe in
   eval/probes.py mixes camera with country and disease mix.
Writes outputs/eval/compare.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

from eval.probes import MIN_TEST_POS, choose_C, grade, load_features, probe


def macro_auc(Y: np.ndarray, S: np.ndarray, i: np.ndarray | None = None) -> float:
    i = np.arange(len(Y)) if i is None else i
    cols = [roc_auc_score(Y[i, j], S[i, j]) for j in range(Y.shape[1]) if 0 < Y[i, j].sum() < len(i)]
    return float(np.mean(cols)) if cols else float("nan")  # a resample with one class only: no AUROC


def paired_bootstrap(Y: np.ndarray, Sa: np.ndarray, Sb: np.ndarray, n: int = 1000, seed: int = 0) -> dict:
    """Y, Sa, Sb: (N, L) labels and two encoders' scores on the SAME photos (L = 1 for a single task)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(Y), len(Y))
        diffs.append(macro_auc(Y, Sa, i) - macro_auc(Y, Sb, i))
    diffs = np.array(diffs)
    diffs = diffs[~np.isnan(diffs)]
    return {"a": macro_auc(Y, Sa), "b": macro_auc(Y, Sb), "diff": macro_auc(Y, Sa) - macro_auc(Y, Sb),
            "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            "p_not_better": float((diffs <= 0).mean())}


def task_scores(meta: pd.DataFrame, name: str) -> dict:
    """Test-set labels and probe scores per task, same protocol as eval/probes.py. Row order follows meta, so
    every encoder's scores line up photo by photo."""
    feats = load_features(name)
    X = feats.attrs["X"]
    m = meta.merge(feats, on="image_id")
    out = {}
    tr = m[(m.bucket == "train") & (m.dataset == "rfmid") & m.label_ok]
    te = m[(m.bucket == "test") & (m.dataset == "rfmid") & m.label_ok]
    labels = sorted({lab for x in m[m.dataset == "rfmid"].labels for lab in x})
    Ytr = np.array([[lab in list(x) for lab in labels] for x in tr.labels], float)
    Yte = np.array([[lab in list(x) for lab in labels] for x in te.labels], float)
    keep = [j for j in range(len(labels)) if Yte[:, j].sum() >= MIN_TEST_POS and Ytr[:, j].sum() >= 2]
    Xtr, Xte = X[tr.row.values], X[te.row.values]
    C, _ = choose_C(Xtr, Ytr[:, keep], tr.dup_group.values)
    S = np.column_stack([probe(C).fit(Xtr, Ytr[:, j]).predict_proba(Xte)[:, 1] for j in keep])
    out["rfmid_linear"] = (Yte[:, keep], S, te.image_id.values)

    idr = m[(m.dataset == "idrid") & m.label_ok].copy()
    idr["g"] = [grade(list(x), "idrid_dr") for x in idr.labels]
    itr, ite = idr[idr.idrid_split == "train"], idr[idr.idrid_split == "test"]
    ytr, yte = (itr.g >= 1).astype(int).values, (ite.g >= 1).astype(int).values
    C, _ = choose_C(X[itr.row.values], ytr[:, None], itr.dup_group.values)
    p = probe(C).fit(X[itr.row.values], ytr).predict_proba(X[ite.row.values])[:, 1]
    out["idrid_any_dr"] = (yte[:, None].astype(float), p[:, None], ite.image_id.values)

    par = m[(m.dataset == "paraguay") & m.label_ok & ~m.label_conflict].copy()
    ypar = np.array([int(grade(list(x), "paraguay_dr") >= 2) for x in par.labels])
    yall = (idr.g >= 1).astype(int).values
    C, _ = choose_C(X[idr.row.values], yall[:, None], idr.dup_group.values)
    p = probe(C).fit(X[idr.row.values], yall).predict_proba(X[par.row.values])[:, 1]
    out["paraguay_any_dr"] = (ypar[:, None].astype(float), p[:, None], par.image_id.values)
    out["_camera"] = camera_rfmid(m, X)
    return out


def camera_rfmid(m: pd.DataFrame, X: np.ndarray, seed: int = 0) -> dict:
    """Guess which of RFMiD's 3 cameras took the photo, from frozen features."""
    r = m[m.dataset == "rfmid"]
    res = {}
    for name, rows in [("all", r), ("normal_only", r[r.label_ok & (r.labels.map(len) == 0)])]:
        y = rows.source.values
        groups = rows.dup_group.fillna(rows.image_id).values
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        pred = cross_val_predict(probe(1.0), X[rows.row.values], y, cv=cv, groups=groups)
        res[name] = {"bal_acc": float(balanced_accuracy_score(y, pred)), "chance": 1 / len(set(y)), "n": len(rows),
                     "per_camera": rows.source.value_counts().to_dict()}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", help="first = the encoder compared against every other")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default="outputs/eval/compare.json", help="result file (ablation runs write elsewhere)")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet")
    scores = {n: task_scores(meta, n) for n in args.names}
    a, result = args.names[0], {"camera_rfmid": {n: scores[n]["_camera"] for n in args.names}, "paired": {}}
    print("Clean camera probe, RFMiD's 3 cameras (balanced accuracy; chance 0.333):")
    for n in args.names:
        c = scores[n]["_camera"]
        print(f"  {n:16s} all photos {c['all']['bal_acc']:.3f} (n {c['all']['n']})   "
              f"normal eyes only {c['normal_only']['bal_acc']:.3f} (n {c['normal_only']['n']})")
    print(f"\nPaired bootstrap, {a} minus each other encoder ({args.n} resamples of the same test photos):")
    for b in args.names[1:]:
        for task in ("rfmid_linear", "idrid_any_dr", "paraguay_any_dr"):
            Ya, Sa, ida = scores[a][task]
            Yb, Sb, idb = scores[b][task]
            assert (ida == idb).all(), "test photos must line up"
            r = paired_bootstrap(Ya, Sa, Sb, args.n)
            result["paired"][f"{a}_vs_{b}_{task}"] = r
            print(f"  vs {b:16s} {task:16s} {r['a']:.3f} vs {r['b']:.3f}: diff {r['diff']:+.3f} "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}], p(not better) {r['p_not_better']:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
