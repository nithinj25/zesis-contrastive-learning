"""Finding detectors on frozen features (CLAUDE.md sections 8.2, 8.3).

    python -m findings.probes --name trace_v2

Input per photo, routed by anatomy: a finding the ontology places at the disc (edema, pallor, cupping, tilted disc,
shunt) reads global + disc crop (4096); every other finding reads the global feature (2048). Why not the spec's
6144 for everything (8.1): out-of-fold, all three views for every finding gave no gain (0.9169 vs 0.9169), but the
matching crop helped the finding that lives there (logistic, OOF: disc pallor 0.739 -> 0.786, disc edema 0.961 ->
0.973). The macula crop was mixed (macular scar up, drusen down), so it is not used. The routing is fixed by the
ontology's location, not picked per finding by score. Labels: the harmonised findings of the head-training photos (RFMiD + RFMiD 2.0 train;
leave-disease-out photos are not in it).
By training positives (8.3): >= 20 -> trained detector; 5-19 -> prototype (cosine to the mean feature of the
positives); < 5 -> dropped.
Two trained detectors are compared: balanced logistic regression per finding, and the spec's MLP (in -> 512 -> F,
dropout 0.3, BCE with pos_weight = negatives / positives). Each is scored by out-of-fold (OOF) AUROC: 5 folds
grouped by same-eye group, every training photo predicted by a model that never saw it. The model is chosen by mean OOF AUROC
over the detector findings; val-A AUROC and the global-only input are printed next to it.
Why OOF and not val-A alone (deviation from 8.3): val-A holds 2-13 positives for most findings (haemorrhage 5,
cotton wool spots 4), too few for an AUROC or a calibration fit to mean anything.
Calibration: Platt scaling per finding (a logistic fit on the raw score), fitted on OOF + val-A scores together.
Reliability r_f from the pooled OOF + val-A AUROC: >= 0.75 -> 1.0, 0.60-0.75 -> 0.5, < 0.60 -> 0 (off; the 0.60
floor is a judgment call). Val-B and the test buckets are only predicted, never fitted on.
Writes features/<name>/<bucket>_findings.parquet (calibrated probabilities), train_findings_oof.parquet
(head-training photos predicted out-of-fold, for the graph's edge counts), artifacts/<name>/findings.joblib and
outputs/eval/findings_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

BUCKETS = ["train", "val_a", "val_b", "test", "ldo_test"]
MIN_DETECTOR, MIN_PROTOTYPE = 20, 5
C_GRID = (0.01, 0.1)


def pos_weight(n_pos: int, n: int) -> float:
    """negatives / positives (8.2): 40 positives in 1,920 photos -> 47."""
    return (n - n_pos) / n_pos


def reliability(auroc: float) -> float:
    return 1.0 if auroc >= 0.75 else 0.5 if auroc >= 0.60 else 0.0


def platt_fit(score: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """p = sigmoid(a * score + b), fitted by logistic regression on the raw score."""
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(score[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(score: np.ndarray, ab: tuple[float, float]) -> np.ndarray:
    return 1 / (1 + np.exp(-(ab[0] * score + ab[1])))


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


# ---- the two trained detectors: each returns raw scores (logits) for the rows of Xte, shape (n, F) ----
def fit_predict_lr(Xtr, Ytr, Xte, C: float) -> np.ndarray:
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    return np.column_stack([LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(Ztr, Ytr[:, j])
                            .decision_function(Zte) for j in range(Ytr.shape[1])])


def fit_predict_mlp(Xtr, Ytr, Xte, epochs: int = 60, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sc = StandardScaler().fit(Xtr)
    xt = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=dev)
    yt = torch.tensor(Ytr, dtype=torch.float32, device=dev)
    xe = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=dev)
    net = nn.Sequential(nn.Dropout(0.3), nn.Linear(xt.shape[1], 512), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(512, yt.shape[1])).to(dev)
    pw = torch.tensor([pos_weight(int(c), len(Ytr)) for c in Ytr.sum(0)], device=dev)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        net.train()
        for i in torch.randperm(len(xt), device=dev).split(128):
            opt.zero_grad()
            loss_fn(net(xt[i]), yt[i]).backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        return net(xe).cpu().numpy()


def prototype_scores(Xtr, y, Xte) -> np.ndarray:
    """Cosine of each test row to the mean feature of the training positives (features are block-normalised)."""
    p = Xtr[y > 0].mean(0)
    return (Xte @ p) / (np.linalg.norm(Xte, axis=1) * np.linalg.norm(p) + 1e-9)


def oof(fn, X, Y, groups, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold scores: every row predicted by a model fitted on the other folds."""
    out = np.zeros(Y.shape, dtype=np.float64)
    for tr, te in GroupKFold(n_splits).split(X, groups=groups):
        out[te] = fn(X[tr], Y[tr], X[te])
    return out


def load_split(name: str, bucket: str):
    d = os.path.join("features", name)
    idx = pd.read_parquet(os.path.join(d, f"{bucket}_index.parquet"))
    g, dc, mc = (np.load(os.path.join(d, f"{bucket}_{k}.npy")).astype(np.float32) for k in ("global", "disc", "macula"))
    return idx, {"global": g, "global+disc": np.concatenate([g, dc], 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    data = {b: load_split(args.name, b) for b in BUCKETS}
    tr_idx, va_idx = data["train"][0], data["val_a"][0]
    tr_meta, va_meta = meta.loc[tr_idx.image_id], meta.loc[va_idx.image_id]
    keep_tr = (tr_meta.in_heads & tr_meta.label_ok).values
    keep_va = va_meta.label_ok.values
    counts = pd.Series([f for x in tr_meta[keep_tr].findings for f in x]).value_counts()
    detectors = [f for f in counts.index if counts[f] >= MIN_DETECTOR]
    protos = [f for f in counts.index if MIN_PROTOTYPE <= counts[f] < MIN_DETECTOR]
    lab = lambda m, names: np.array([[f in list(x) for f in names] for x in m.findings], dtype=np.float32)
    Ytr, Yva = lab(tr_meta[keep_tr], detectors), lab(va_meta[keep_va], detectors)
    Ptr, Pva = lab(tr_meta[keep_tr], protos), lab(va_meta[keep_va], protos)
    groups = tr_meta[keep_tr].dup_group.fillna(-1).astype(str).values
    groups = np.where(groups == "-1.0", tr_meta[keep_tr].index.values, groups)
    print(f"{keep_tr.sum()} head-training photos, {keep_va.sum()} val-A; {len(detectors)} detectors, {len(protos)} prototypes; "
          f"loc_ok train {tr_idx.loc_ok[keep_tr].mean():.1%}, val-A {va_idx.loc_ok[keep_va].mean():.1%}")

    # ---- routing: a finding located at the disc reads global + disc crop; every other finding reads global ----
    onto = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))["findings"]
    at_disc = lambda f: onto[f]["location"] == "disc"
    route = {"global": lambda f: "global", "routed": lambda f: "global+disc" if at_disc(f) else "global"}
    models = {f"logistic C={C}": (lambda a, b, c, C=C: fit_predict_lr(a, b, c, C)) for C in C_GRID}
    models["mlp"] = fit_predict_mlp
    Xin = lambda b, inp, keep=None: data[b][1][inp] if keep is None else data[b][1][inp][keep]
    table, cache = [], {}
    for mname, fn in models.items():
        per_input = {inp: (oof(fn, Xin("train", inp, keep_tr), Ytr, groups), fn(Xin("train", inp, keep_tr), Ytr, Xin("val_a", inp, keep_va)))
                     for inp in ("global", "global+disc")}
        for rname, r in route.items():
            S_oof = np.column_stack([per_input[r(f)][0][:, j] for j, f in enumerate(detectors)])
            S_va = np.column_stack([per_input[r(f)][1][:, j] for j, f in enumerate(detectors)])
            a_oof = [auc(Ytr[:, j], S_oof[:, j]) for j in range(len(detectors))]
            a_va = [auc(Yva[:, j], S_va[:, j]) for j in range(len(detectors))]
            cache[(rname, mname)] = (S_oof, S_va, a_oof, a_va)
            table.append({"input": rname, "model": mname, "oof": float(np.mean(a_oof)), "val_a": float(np.nanmean(a_va))})
            print(f"  {rname:8s} {mname:16s} OOF macro-AUROC {table[-1]['oof']:.4f}   val-A {table[-1]['val_a']:.4f}", flush=True)
    # model by OOF; routing is fixed in advance by anatomy (not picked by score) and reported against global
    mname = max((r for r in table if r["input"] == "routed"), key=lambda r: r["oof"])["model"]
    best = next(r for r in table if r["input"] == "routed" and r["model"] == mname)
    inp = "routed"
    S_oof, S_va, a_oof, a_va = cache[(inp, mname)]
    print(f"chosen: routed + {mname}")

    # ---- final fit on all head-training photos; Platt on OOF + val-A; reliability ----
    report, cal, raw, oof_q = {}, {}, {b: {} for b in BUCKETS}, {}
    S_all = {i: {b: models[mname](Xin("train", i, keep_tr), Ytr, Xin(b, i)) for b in BUCKETS} for i in ("global", "global+disc")}
    for j, f in enumerate(detectors):
        y_cal, s_cal = np.concatenate([Ytr[:, j], Yva[:, j]]), np.concatenate([S_oof[:, j], S_va[:, j]])
        cal[f] = platt_fit(s_cal, y_cal)
        pooled = auc(y_cal, s_cal)
        report[f] = {"kind": "detector", "input": route[inp](f), "train_pos": int(Ytr[:, j].sum()), "val_a_pos": int(Yva[:, j].sum()),
                     "oof_auroc": a_oof[j], "val_a_auroc": a_va[j], "pooled_auroc": pooled, "reliability": reliability(pooled)}
        for b in BUCKETS:
            raw[b][f] = platt_apply(S_all[route[inp](f)][b][:, j], cal[f])
        oof_q[f] = platt_apply(S_oof[:, j], cal[f])
    proto_vecs = {}
    for j, f in enumerate(protos):
        y, i = Ptr[:, j], route[inp](f)
        Xtr = Xin("train", i, keep_tr)
        s_oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(Xtr, groups=groups):
            s_oof[te] = prototype_scores(Xtr[tr], y[tr], Xtr[te]) if y[tr].sum() else 0.0
        s_va = prototype_scores(Xtr, y, Xin("val_a", i, keep_va))
        y_cal, s_cal = np.concatenate([y, Pva[:, j]]), np.concatenate([s_oof, s_va])
        cal[f] = platt_fit(s_cal, y_cal)
        pooled = auc(y_cal, s_cal)
        report[f] = {"kind": "prototype", "input": i, "train_pos": int(y.sum()), "val_a_pos": int(Pva[:, j].sum()),
                     "oof_auroc": auc(y, s_oof), "val_a_auroc": auc(Pva[:, j], s_va), "pooled_auroc": pooled,
                     "reliability": reliability(pooled)}
        proto_vecs[f] = Xtr[y > 0].mean(0)
        for b in BUCKETS:
            raw[b][f] = platt_apply(prototype_scores(Xtr, y, Xin(b, i)), cal[f])
        oof_q[f] = platt_apply(s_oof, cal[f])

    for b in BUCKETS:
        out = pd.DataFrame(raw[b]).astype(np.float32)
        out.insert(0, "image_id", data[b][0].image_id.values)
        out.to_parquet(os.path.join("features", args.name, f"{b}_findings.parquet"), index=False)
    # head-training photos, each predicted by a model that never saw it: what the graph's edge counts use (graph/build.py)
    oof_df = pd.DataFrame(oof_q).astype(np.float32)
    oof_df.insert(0, "image_id", tr_idx.image_id.values[keep_tr])
    oof_df.to_parquet(os.path.join("features", args.name, "train_findings_oof.parquet"), index=False)
    os.makedirs(os.path.join("artifacts", args.name), exist_ok=True)
    joblib.dump({"input": inp, "model": mname, "detectors": detectors, "prototypes": protos, "platt": cal,
                 "prototype_vectors": proto_vecs, "report": report},
                os.path.join("artifacts", args.name, "findings.joblib"))
    os.makedirs("outputs/eval", exist_ok=True)
    json.dump({"chosen": best, "table": table, "findings": report, "dropped": [f for f in counts.index if counts[f] < MIN_PROTOTYPE]},
              open(f"outputs/eval/findings_{args.name}.json", "w"), indent=1)
    print(f"\n{'finding':20s} kind       train+  valA+   OOF    val-A  pooled  r_f")
    for f, r in sorted(report.items(), key=lambda t: -t[1]["pooled_auroc"]):
        print(f"{f:20s} {r['kind']:10s} {r['train_pos']:5d} {r['val_a_pos']:5d}  {r['oof_auroc']:.3f}  {r['val_a_auroc']:.3f}  "
              f"{r['pooled_auroc']:.3f}  {r['reliability']}")
