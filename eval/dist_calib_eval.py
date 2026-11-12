"""Does distribution calibration (graph/dist_calib.py) beat the prototype bank on rare diseases?

    python -m eval.dist_calib_eval --name trace_v2            # practice task only: picks the configuration
    python -m eval.dist_calib_eval --name trace_v2 --final    # + ONE run of the picked configuration on real rare photos

Practice task (no rare photo is touched): the 6 common diseases with >= 40 single-diagnosis head-training photos
(normal, DR, myopia, ARMD, BRVO, CSR). Each of myopia, ARMD, BRVO, CSR in turn is made "rare": it keeps K = 5 or 14
photos (14 = RP's count) from distinct eyes; its other photos (other eyes) are the queries. 5 seeds.
  margin(d)   = score of d minus the best other disease's score (cosine for prototypes, logit for logistic)
  threshold   = 95th percentile of margin(d) over val-A photos of the other diseases (5% false alarm, the rule the
                deployed bank uses)
  called      = share of query photos of d above the threshold   <- the number that picks the configuration
  top1        = share with margin > 0
Methods: proto (the current bank: cosine to the mean), lr (logistic on real photos, balanced class weights), and
dc_* (logistic on real photos + synthetic rare points). Logistic models read the Tukey-transformed feature (sqrt).

Final run (--final, once): the picked configuration against prototypes on the real rare diseases, same splits for both:
  RFMiD / RFMiD 2.0 AION, CRVO, RP photos, 5 folds grouped by eye (support = other folds); JSIEC CRVO, RP (new
  hospital, never in training) with all rare photos as support. Rare call = top rare disease beats the best common
  disease by the val-A 5%-false-alarm margin (eval/rare_bank.py's rule). Writes outputs/eval/dist_calib_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from eval.probes import load_features
from graph.bank import unit
from graph.dist_calib import BaseStats, augment, tukey

BASE = ["normal", "DR", "myopia", "ARMD", "BRVO", "CSR"]
PSEUDO_RARE = ["myopia", "ARMD", "BRVO", "CSR"]
KS = (5, 14)
SEEDS = range(5)
N_SYN = 500
C = 0.01

# name: (k nearest base diseases, mean weight w (None = keep the photo's own position), alpha in units of the mean
# per-dimension base variance)
DC_CONFIGS = {
    "dc_paper": (2, 1.0, 0.0),      # the paper's recipe (k = 2, w = 1)
    "dc_k1": (1, 1.0, 0.0),
    "dc_spread_k2": (2, None, 0.0),  # borrow only the spread, not the position
    "dc_spread_k1": (1, None, 0.0),
    "dc_spread_k2_a1": (2, None, 1.0),
}


def groups_of(meta: pd.DataFrame, ids) -> np.ndarray:
    g = meta.loc[list(ids)].dup_group.fillna(-1).astype(str).values
    return np.where(g == "-1.0", np.array(list(ids)), g)


def single_dx_rows(meta: pd.DataFrame, rows, classes) -> pd.Series:
    m = meta.loc[rows]
    m = m[m.label_ok & (m.diseases.map(len) == 1)]
    dz = m.diseases.map(lambda v: v[0])
    return dz[dz.isin(classes)]


def fit_lr(X, y, classes, balanced=False):
    """Logistic scorer whose columns follow `classes` (sklearn sorts classes_ alphabetically)."""
    sc = StandardScaler().fit(X)
    lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced" if balanced else None).fit(sc.transform(X), y)
    order = [list(lr.classes_).index(c) for c in classes]
    return lambda Z: lr.decision_function(sc.transform(Z))[:, order]


def proto_scorer(X_by_class: dict, classes):
    P = np.stack([unit(X_by_class[c].mean(0)) for c in classes])
    return lambda Z: unit(Z) @ P.T


def margin(s: np.ndarray, j: int) -> np.ndarray:
    other = np.delete(s, j, axis=1)
    return s[:, j] - other.max(1)


def practice_one(d, K, seed, U, T, lab, grp, va_U, va_T, va_lab, mean_var):
    """One pseudo-rare task -> {method: (called, top1, val-A top-1 false alarm)}."""
    rng = np.random.default_rng(1000 * seed + K)
    idx_d = np.where(lab == d)[0]
    eyes = rng.permutation(np.unique(grp[idx_d]))[:K]
    sup = np.array([rng.choice(idx_d[grp[idx_d] == e]) for e in eyes])
    qry = idx_d[~np.isin(grp[idx_d], eyes)]
    others = np.where(lab != d)[0]
    j = BASE.index(d)
    fa = va_lab != d
    res = {}

    def score(name, f, Q, VA):
        mq, mv = margin(f(Q), j), margin(f(VA), j)
        thr = np.percentile(mv, 95)
        res[name] = (float((mq > thr).mean()), float((mq > 0).mean()), float((mv > 0).mean()))

    byc = {c: U[others][lab[others] == c] for c in BASE if c != d}
    byc[d] = U[sup]
    score("proto", proto_scorer(byc, BASE), U[qry], va_U[fa])
    Xtr = np.concatenate([T[others], T[sup]])
    ytr = np.concatenate([lab[others], lab[sup]])
    score("lr", fit_lr(Xtr, ytr, BASE, balanced=True), T[qry], va_T[fa])
    stats = BaseStats({c: T[others][lab[others] == c] for c in BASE if c != d})
    for name, (k, w, a) in DC_CONFIGS.items():
        syn = augment(T[sup], stats, N_SYN, k=k, alpha=a * mean_var, w=w, seed=seed)
        f = fit_lr(np.concatenate([Xtr, syn]), np.concatenate([ytr, np.full(len(syn), d)]), BASE)
        score(name, f, T[qry], va_T[fa])
    return d, K, seed, res


def practice(meta, U_all, T_all, row) -> pd.DataFrame:
    head = single_dx_rows(meta, meta.index[meta.in_heads], BASE)
    head = head[head.index.isin(row.index)]
    va = single_dx_rows(meta, meta.index[meta.bucket == "val_a"], BASE)
    va = va[va.index.isin(row.index)]
    r, rv = row.loc[head.index].values, row.loc[va.index].values
    U, T, lab, grp = U_all[r], T_all[r], head.values, groups_of(meta, head.index)
    mean_var = float(np.mean([T[lab == c].var(0).mean() for c in BASE]))
    jobs = [delayed(practice_one)(d, K, s, U, T, lab, grp, U_all[rv], T_all[rv], va.values, mean_var)
            for d in PSEUDO_RARE for K in KS for s in SEEDS]
    out = Parallel(n_jobs=6)(jobs)
    recs = [{"disease": d, "K": K, "seed": s, "method": m, "called": v[0], "top1": v[1], "val_a_fa_top1": v[2]}
            for d, K, s, res in out for m, v in res.items()]
    return pd.DataFrame(recs)


def final(meta, U_all, T_all, row, rare, pick) -> dict:
    k, w, a = DC_CONFIGS[pick]
    common = BASE + ["hypertensive_retinopathy"]
    head = single_dx_rows(meta, meta.index[meta.in_heads], common)
    head = head[head.index.isin(row.index)]
    va_ids = [i for i in meta.index[(meta.bucket == "val_a") & meta.label_ok] if i in row.index]
    classes = common + list(rare)
    cU = {c: U_all[row.loc[head.index[head == c]].values] for c in common}
    cT = {c: T_all[row.loc[head.index[head == c]].values] for c in common}
    stats = BaseStats({c: cT[c] for c in BASE})
    mean_var = float(np.mean([cT[c].var(0).mean() for c in BASE]))
    rare_ids = {d: [i for i in meta.index[(meta.ldo_disease == d) & meta.label_ok & (meta.bucket != "excluded")
                                          & meta.dataset.isin(["rfmid", "rfmid2"])] if i in row.index] for d in rare}
    is_rare = np.isin(classes, rare)

    def models(sup_ids: dict):
        byU = dict(cU, **{d: U_all[row.loc[sup_ids[d]].values] for d in rare})
        Xs = [cT[c] for c in common] + [T_all[row.loc[sup_ids[d]].values] for d in rare]
        ys = [np.full(len(cT[c]), c) for c in common] + [np.full(len(sup_ids[d]), d) for d in rare]
        for d in rare:
            syn = augment(T_all[row.loc[sup_ids[d]].values], stats, N_SYN, k=k, alpha=a * mean_var, w=w)
            Xs.append(syn)
            ys.append(np.full(len(syn), d))
        n_real = sum(len(cT[c]) for c in common) + sum(len(sup_ids[d]) for d in rare)  # real rows first, synthetic after
        Xr, yr = np.concatenate(Xs[:len(common) + len(rare)]), np.concatenate(ys[:len(common) + len(rare)])
        assert len(Xr) == n_real
        return {"proto": (proto_scorer(byU, classes), U_all),
                "lr": (fit_lr(Xr, yr, classes, balanced=True), T_all),
                pick: (fit_lr(np.concatenate(Xs), np.concatenate(ys), classes), T_all)}

    def judge(f, F, ids, d, tau):
        s = f(F[row.loc[ids].values])
        j = classes.index(d)
        rank = (s > s[:, [j]]).sum(1) + 1
        rm = s[:, is_rare].max(1) - s[:, ~is_rare].max(1)
        top_rare = np.array(classes)[np.where(is_rare)[0][s[:, is_rare].argmax(1)]]
        return rank, (top_rare == d) & (rm > tau)

    def taus(ms):
        out = {}
        for m, (f, F) in ms.items():
            s = f(F[row.loc[va_ids].values])
            out[m] = float(np.percentile(s[:, is_rare].max(1) - s[:, ~is_rare].max(1), 95))
        return out

    res = {"pick": pick, "rfmid_5fold": {}, "jsiec": {}}
    acc = {m: {d: ([], []) for d in rare} for m in ("proto", "lr", pick)}
    folds = {d: list(GroupKFold(5).split(rare_ids[d], groups=groups_of(meta, rare_ids[d]))) for d in rare}
    for fi in range(5):
        sup = {d: [rare_ids[d][i] for i in folds[d][fi][0]] for d in rare}
        ms = models(sup)
        tz = taus(ms)
        for d in rare:
            q = [rare_ids[d][i] for i in folds[d][fi][1]]
            for m, (f, F) in ms.items():
                rk, cl = judge(f, F, q, d, tz[m])
                acc[m][d][0].extend(rk)
                acc[m][d][1].extend(cl)
    for m in acc:
        res["rfmid_5fold"][m] = {d: {"n": len(acc[m][d][0]), "top1": float(np.mean(np.array(acc[m][d][0]) == 1)),
                                     "top3": float(np.mean(np.array(acc[m][d][0]) <= 3)), "called": float(np.mean(acc[m][d][1]))}
                                 for d in rare}
    ms = models(rare_ids)
    tz = taus(ms)
    res["tau"] = tz
    js = meta[(meta.dataset == "jsiec") & meta.label_ok]
    for m, (f, F) in ms.items():
        res["jsiec"][m] = {}
        for d in rare:
            ids = [i for i in js.index[js.diseases.map(lambda v, d=d: d in list(v))] if i in row.index]
            if ids:
                rk, cl = judge(f, F, ids, d, tz[m])
                res["jsiec"][m][d] = {"n": len(ids), "top1": float((rk == 1).mean()), "top3": float((rk <= 3).mean()),
                                      "called": float(cl.mean())}
        s = f(F[row.loc[va_ids].values])
        res.setdefault("val_a_false_alarm_top1", {})[m] = float(is_rare[s.argmax(1)].mean())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    rare = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))["leave_disease_out"]
    f = load_features(args.name)
    row = f.set_index("image_id").row
    U = unit(f.attrs["X"].astype(np.float64))
    T = tukey(U, 0.5)
    df = practice(meta, U, T, row)
    summ = df.groupby(["method", "K"])[["called", "top1", "val_a_fa_top1"]].mean().unstack("K")
    print("Practice task (myopia / ARMD / BRVO / CSR made rare; mean over 4 diseases x 5 seeds)")
    print(summ.round(3).to_string())
    print("\ncalled at K = 5 / 14, per disease")
    print(df.groupby(["method", "disease", "K"]).called.mean().unstack(["disease", "K"]).round(2).to_string())
    mean_called = df.groupby("method").called.mean()
    pick = mean_called.drop(["proto", "lr"]).idxmax()
    out = {"practice": df.to_dict("records"), "practice_mean_called": mean_called.to_dict(), "pick": pick}
    print(f"\npicked on the practice task: {pick} (mean called {mean_called[pick]:.3f} vs proto {mean_called['proto']:.3f},"
          f" lr {mean_called['lr']:.3f})")
    if args.final:
        out["final"] = final(meta, U, T, row, rare, pick)
        fr = out["final"]
        for part in ("rfmid_5fold", "jsiec"):
            print(f"\n{part}: top1 / top3 / called")
            for m, per in fr[part].items():
                print(f"  {m:16s} " + "   ".join(f"{d} {v['top1']:.2f}/{v['top3']:.2f}/{v['called']:.2f} (n {v['n']})"
                                                for d, v in per.items()))
        print(f"\nval-A top-1 false alarm (top-1 is a rare disease): {fr['val_a_false_alarm_top1']}; margins {fr['tau']}")
    os.makedirs("outputs/eval", exist_ok=True)
    json.dump(out, open(f"outputs/eval/dist_calib_{args.name}.json", "w"), indent=1, default=float)
