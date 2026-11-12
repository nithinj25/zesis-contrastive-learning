"""Frozen-feature evaluation (CLAUDE.md section 19.7): linear probes, kNN, camera probe.

Run from the repo root:  python -m eval.probes --name imagenet_r50
Reads features/<name>/ and data/meta.parquet. Writes outputs/eval/<name>.json and appends a row to
outputs/eval/baselines.csv. Nothing here trains the encoder; every probe sees frozen, L2-normalised features.
Regularisation C is chosen by 5-fold cross-validation on the training rows, grouped by same-eye group
(dup_group), never on test.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0, 10.0)
MIN_TEST_POS = 5
KNN_K, KNN_T = 20, 0.07


def load_features(name: str, buckets=("train", "val_a", "val_b", "test", "ldo_test", "pool")) -> pd.DataFrame:
    """One row per photo: image_id and a 'feat' row index into the returned matrix (attached as .attrs['X'])."""
    frames, mats = [], []
    for b in buckets:
        path = os.path.join("features", name, f"{b}_global.npy")
        if not os.path.exists(path):
            continue
        idx = pd.read_parquet(os.path.join("features", name, f"{b}_index.parquet"))
        frames.append(idx)
        mats.append(np.load(path).astype(np.float32))
    ids = pd.concat(frames, ignore_index=True)
    ids["row"] = np.arange(len(ids))
    ids.attrs["X"] = np.concatenate(mats)
    return ids


def probe(C: float, multiclass: bool = False):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def choose_C(X, Y, groups, seed=0) -> tuple[float, dict]:
    """Y: (n, L) binary. Picks the C with the best mean AUROC over labels, by grouped 5-fold CV."""
    folds = list(GroupKFold(n_splits=5).split(X, groups=groups))
    scores = {}
    for C in C_GRID:
        aucs = []
        for j in range(Y.shape[1]):
            y = Y[:, j]
            pred = np.zeros(len(y))
            ok = True
            for tr, te in folds:
                if y[tr].min() == y[tr].max():
                    ok = False
                    break
                pred[te] = probe(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            if ok and 0 < y.sum() < len(y):
                aucs.append(roc_auc_score(y, pred))
        scores[C] = float(np.mean(aucs)) if aucs else float("nan")
    return max(scores, key=lambda c: scores[c]), scores


def boot_ci(y_true: np.ndarray, y_score: np.ndarray, n: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap interval of macro-AUROC over the columns (test photos resampled)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        cols = [roc_auc_score(y_true[i, j], y_score[i, j]) for j in range(y_true.shape[1]) if 0 < y_true[i, j].sum() < len(i)]
        vals.append(np.mean(cols))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def knn_scores(Xtr, Ytr, Xte, k=KNN_K, t=KNN_T) -> np.ndarray:
    """Similarity-weighted kNN vote per label: features are unit length, so X @ X.T is cosine."""
    sim = Xte @ Xtr.T
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    w = np.exp(np.take_along_axis(sim, idx, axis=1) / t)
    return np.einsum("nk,nkl->nl", w, Ytr[idx]) / w.sum(1, keepdims=True)


def rfmid_tasks(meta, feats, X) -> dict:
    m = meta.merge(feats, on="image_id")
    tr = m[(m.bucket == "train") & (m.dataset == "rfmid") & m.label_ok]
    te = m[(m.bucket == "test") & (m.dataset == "rfmid") & m.label_ok]
    labels = sorted({lab for x in m[m.dataset == "rfmid"].labels for lab in x})
    Ytr = np.array([[lab in list(x) for lab in labels] for x in tr.labels], float)
    Yte = np.array([[lab in list(x) for lab in labels] for x in te.labels], float)
    keep = [j for j, lab in enumerate(labels) if Yte[:, j].sum() >= MIN_TEST_POS and Ytr[:, j].sum() >= 2]
    left_out = {labels[j]: (int(Ytr[:, j].sum()), int(Yte[:, j].sum())) for j in range(len(labels)) if j not in keep and Yte[:, j].sum() >= MIN_TEST_POS}
    Xtr, Xte = X[tr.row.values], X[te.row.values]
    Ytr_k, Yte_k = Ytr[:, keep], Yte[:, keep]

    C, cv = choose_C(Xtr, Ytr_k, tr.dup_group.values)
    S = np.column_stack([probe(C).fit(Xtr, Ytr_k[:, j]).predict_proba(Xte)[:, 1] for j in range(len(keep))])
    per = {labels[j]: float(roc_auc_score(Yte_k[:, i], S[:, i])) for i, j in enumerate(keep)}
    risk_tr, risk_te = tr.disease_risk.astype(float).values, te.disease_risk.astype(float).values
    risk = probe(C).fit(Xtr, risk_tr).predict_proba(Xte)[:, 1]
    K = knn_scores(Xtr, Ytr_k, Xte)
    knn_per = [roc_auc_score(Yte_k[:, i], K[:, i]) for i in range(len(keep))]
    return {
        "rfmid_n_train": len(tr), "rfmid_n_test": len(te), "rfmid_C": C, "rfmid_C_cv": cv,
        "rfmid_labels_scored": [labels[j] for j in keep],
        "rfmid_labels_left_out": left_out,
        "rfmid_linear_macro_auroc": float(np.mean(list(per.values()))),
        "rfmid_linear_macro_auroc_ci95": boot_ci(Yte_k, S),
        "rfmid_linear_per_label": per,
        "rfmid_disease_risk_auroc": float(roc_auc_score(risk_te, risk)),
        "rfmid_knn_macro_auroc": float(np.mean(knn_per)),
    }


def grade(labels, prefix) -> int:
    return int([x for x in labels if x.startswith(prefix)][0][len(prefix):])


def dr_tasks(meta, feats, X) -> dict:
    m = meta.merge(feats, on="image_id")
    idr = m[(m.dataset == "idrid") & m.label_ok].copy()
    idr["g"] = [grade(list(x), "idrid_dr") for x in idr.labels]
    tr, te = idr[idr.idrid_split == "train"], idr[idr.idrid_split == "test"]
    out = {"idrid_n_train": len(tr), "idrid_n_test": len(te)}
    for name, cut in [("any_dr", 1), ("referable_dr", 2)]:
        ytr, yte = (tr.g >= cut).astype(int).values, (te.g >= cut).astype(int).values
        C, _ = choose_C(X[tr.row.values], ytr[:, None], tr.dup_group.values)
        p = probe(C).fit(X[tr.row.values], ytr).predict_proba(X[te.row.values])[:, 1]
        out[f"idrid_{name}_auroc"] = float(roc_auc_score(yte, p))
        out[f"idrid_{name}_C"] = C
    five = probe(1.0).fit(X[tr.row.values], tr.g.values).predict(X[te.row.values])
    out["idrid_5grade_balanced_acc"] = float(balanced_accuracy_score(te.g.values, five))

    # A new site: probe fitted on all IDRiD, tested on Paraguay (folder 1 = no DR, 2-7 = DR grades).
    par = m[(m.dataset == "paraguay") & m.label_ok & ~m.label_conflict].copy()
    par["any_dr"] = [int(grade(list(x), "paraguay_dr") >= 2) for x in par.labels]
    yall = (idr.g >= 1).astype(int).values
    C, _ = choose_C(X[idr.row.values], yall[:, None], idr.dup_group.values)
    p = probe(C).fit(X[idr.row.values], yall).predict_proba(X[par.row.values])[:, 1]
    out["paraguay_any_dr_auroc_from_idrid"] = float(roc_auc_score(par.any_dr.values, p))
    out["paraguay_n"] = len(par)
    return out


def camera_tasks(meta, feats, X, per_group: int = 400, seed: int = 0) -> dict:
    """Balanced accuracy of guessing the camera (training-side sources) and the dataset (all) from features."""
    m = meta.merge(feats, on="image_id")
    out = {}
    for name, rows, col in [("source", m[m.bucket.isin(["train", "pool", "val_a", "val_b"])], "source"),
                            ("dataset", m, "dataset")]:
        s = rows.groupby(col, group_keys=False).apply(lambda g: g.sample(min(len(g), per_group), random_state=seed))
        y = s[col].values
        pred = cross_val_predict(probe(1.0), X[s.row.values], y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
        out[f"camera_probe_{name}_bal_acc"] = float(balanced_accuracy_score(y, pred))
        out[f"camera_probe_{name}_chance"] = 1 / len(set(y))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--out_dir", default="outputs/eval", help="per-encoder JSON + summary CSV go here")
    args = ap.parse_args()
    t0 = time.time()
    meta = pd.read_parquet("data/meta.parquet")
    feats = load_features(args.name)
    X = feats.attrs["X"]
    res = {"encoder": args.name}
    for part in (rfmid_tasks, dr_tasks, camera_tasks):
        res.update(part(meta, feats, X))
        print(f"{part.__name__} done ({time.time() - t0:.0f} s)")
    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(args.out_dir, f"{args.name}.json"), "w"), indent=1)
    keys = ["rfmid_linear_macro_auroc", "rfmid_knn_macro_auroc", "rfmid_disease_risk_auroc", "idrid_any_dr_auroc",
            "idrid_referable_dr_auroc", "idrid_5grade_balanced_acc", "paraguay_any_dr_auroc_from_idrid",
            "camera_probe_source_bal_acc", "camera_probe_dataset_bal_acc"]
    row = pd.DataFrame([{"encoder": args.name, **{k: round(res[k], 4) for k in keys}}])
    path = os.path.join(args.out_dir, "baselines.csv")
    row.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    ci = res["rfmid_linear_macro_auroc_ci95"]
    print(f"\n{args.name}: RFMiD linear macro-AUROC {res['rfmid_linear_macro_auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] over "
          f"{len(res['rfmid_labels_scored'])} labels; kNN {res['rfmid_knn_macro_auroc']:.3f}; Disease_Risk {res['rfmid_disease_risk_auroc']:.3f}")
    print(f"IDRiD any-DR {res['idrid_any_dr_auroc']:.3f}, referable {res['idrid_referable_dr_auroc']:.3f}, 5-grade bal. acc "
          f"{res['idrid_5grade_balanced_acc']:.3f}; Paraguay any-DR from an IDRiD probe {res['paraguay_any_dr_auroc_from_idrid']:.3f}")
    print(f"camera probe: source {res['camera_probe_source_bal_acc']:.3f} (chance {res['camera_probe_source_chance']:.3f}), "
          f"dataset {res['camera_probe_dataset_bal_acc']:.3f} (chance {res['camera_probe_dataset_chance']:.3f})")
    print("left out (leave-disease-out or < 2 train positives):", res["rfmid_labels_left_out"])
