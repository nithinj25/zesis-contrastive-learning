"""Phase 5 evaluation: calibration, conformal coverage per camera, risk-coverage and OOD (CLAUDE.md 10, 14, 19.7).

    python -m eval.trust_eval --name trace_v2

Fitting (no test photo touches any of it):
  classifier  : multinomial logistic on head-training single-diagnosis photos, 6 classes (trust/classifier.py), C by
                grouped CV on head training; temperature T on val-A.
  conformal   : Mondrian (per camera) on val-B, alpha = 0.1 (90% target coverage); groups < 50 photos and unseen
                cameras use the most conservative q_hat.
  OOD         : 10th-neighbour cosine distance to the pool + head-training photos; thresholds at the 95th percentile
                of val-A, per camera where >= 50 photos.
Tests:
  in scope    : RFMiD + RFMiD 2.0 test photos with one diagnosis among the 6 classes (coverage and set size per camera).
  new sites   : the DR photos of Paraguay (ZEISS VISUCAM 500, unseen camera), IDRiD and JSIEC, and JSIEC BRVO: does the
                set contain the true disease? Their "no DR" photos are not used: "no DR signs" is not "normal".
  risk-cover. : in-scope test photos sorted by calibrated top probability; error at 100% ... 10% coverage; AURC.
  OOD         : in-distribution = all RFMiD + RFMiD 2.0 labelled test photos. Out = (a) 13 JSIEC classes with no
                counterpart in our vocabulary (judgment call, listed in UNKNOWN), (b) non-fundus images when
                features/<name>/ood_cifar_global.npy exists. AUROC and FPR at 95% TPR. Also: share of Paraguay photos
                flagged (an unseen camera, mostly familiar diseases: a high share means the camera itself looks unfamiliar).
Writes outputs/eval/trust_<name>.json and artifacts/<name>/trust.joblib.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from eval.probes import load_features
from trust.calibrate import ece, fit_temperature, nll
from trust.classifier import CLASSES, Calibrated, Logistic, choose_C, single_dx
from trust.conformal import Mondrian
from trust.ood import KnnOOD

ALPHA = 0.1
UNKNOWN = ["Rhegmatogenous RD", "Silicon oil in eye", "Fundus neoplasm", "Bietti crystalline dystrophy", "VKH disease",
           "Chorioretinal atrophy-coloboma", "Myelinated nerve fiber", "Congenital disc abnormality", "Dragged Disc",
           "Peripheral retinal degeneration and break", "Vitreous particles", "Yellow-white spots-flecks", "Fibrosis"]


def risk_coverage(conf: np.ndarray, correct: np.ndarray) -> dict:
    order = np.argsort(-conf)
    c = correct[order]
    curve = {}
    for cov in np.arange(1.0, 0.05, -0.1):
        n = max(1, int(round(cov * len(c))))
        curve[f"{cov:.1f}"] = float(1 - c[:n].mean())
    risks = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return {"error_at_coverage": curve, "aurc": float(risks.mean())}


def ood_metrics(s_in: np.ndarray, s_out: np.ndarray) -> dict:
    y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    thr = np.percentile(s_in, 95)  # keeps 95% of in-distribution photos
    return {"auroc": float(roc_auc_score(y, np.r_[s_in, s_out])), "fpr_at_95tpr": float((s_out <= thr).mean()),
            "n_in": len(s_in), "n_out": len(s_out)}


def coverage(sets: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(len(y)), "coverage": float(sets[np.arange(len(y)), y].mean()), "mean_set_size": float(sets.sum(1).mean()),
            "share_singleton": float((sets.sum(1) == 1).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    f = load_features(args.name)
    X = f.attrs["X"] / np.linalg.norm(f.attrs["X"], axis=1, keepdims=True)
    row = f.set_index("image_id").row
    feats = lambda ids: X[row.loc[list(ids)].values]
    has = lambda ids: [i for i in ids if i in row.index]

    tr = single_dx(meta, has(meta.index[meta.in_heads]))
    va = single_dx(meta, has(meta.index[meta.bucket == "val_a"]))
    vb = single_dx(meta, has(meta.index[meta.bucket == "val_b"]))
    g = tr.dup_group.fillna(-1).astype(str).values
    g = np.where(g == "-1.0", tr.index.values, g)
    C = choose_C(feats(tr.index), tr.y.values, g)
    model = Logistic(C).fit(feats(tr.index), tr.y.values)
    z_va = model.logits(feats(va.index))
    T = fit_temperature(z_va, va.y.values)
    clf = Calibrated(model, T)
    p_va = clf.proba(feats(va.index))
    res = {"classes": CLASSES, "C": C, "T": T, "n_train": len(tr), "n_val_a": len(va), "n_val_b": len(vb),
           "val_a": {"nll_T1": nll(z_va, va.y.values, 1.0), "nll_T": nll(z_va, va.y.values, T),
                     "accuracy": float((p_va.argmax(1) == va.y.values).mean()),
                     "ece": ece(p_va.max(1), p_va.argmax(1) == va.y.values)}}

    mond = Mondrian(alpha=ALPHA).fit(clf.proba(feats(vb.index)), vb.y.values, vb.source.values)
    res["conformal"] = {"alpha": ALPHA, "q_hat": {k: float(v) for k, v in mond.q.items()}, "q_all": mond.q_all,
                        "q_conservative": mond.q_conservative, "val_b_n_by_camera": mond.n}

    te = single_dx(meta, has(meta.index[(meta.bucket == "test") & meta.dataset.isin(["rfmid", "rfmid2"])]))
    p_te = clf.proba(feats(te.index))
    sets = mond.sets(p_te, te.source.values)
    res["test_in_scope"] = {"all": coverage(sets, te.y.values), "accuracy": float((p_te.argmax(1) == te.y.values).mean()),
                            "ece": ece(p_te.max(1), p_te.argmax(1) == te.y.values),
                            "by_camera": {c: coverage(sets[(te.source == c).values], te.y.values[(te.source == c).values])
                                          for c in te.source.unique()}}
    res["risk_coverage"] = risk_coverage(p_te.max(1), p_te.argmax(1) == te.y.values)

    sites = {}
    for name, sel in {"paraguay_DR": (meta.dataset == "paraguay"), "idrid_DR": (meta.dataset == "idrid"),
                      "jsiec_DR": (meta.dataset == "jsiec"), "jsiec_BRVO": (meta.dataset == "jsiec")}.items():
        dz = name.split("_")[1]
        ids = has(meta.index[sel & meta.label_ok & meta.diseases.map(lambda v, dz=dz: list(v) == [dz])])
        p = clf.proba(feats(ids))
        s = mond.sets(p, meta.loc[ids].source.values)
        sites[name] = coverage(s, np.full(len(ids), CLASSES.index(dz)))
        sites[name]["top1"] = float((p.argmax(1) == CLASSES.index(dz)).mean())
    res["new_sites"] = sites

    index_ids = has(meta.index[meta.bucket.isin(["pool", "train"])])
    ood = KnnOOD().fit(feats(index_ids), index_ids)
    va_all = has(meta.index[(meta.bucket == "val_a") & meta.label_ok])
    ood.calibrate(feats(va_all), meta.loc[va_all].source.values)
    s_in = ood.score(feats(has(meta.index[(meta.bucket == "test") & meta.dataset.isin(["rfmid", "rfmid2"]) & meta.label_ok])))
    js = meta[(meta.dataset == "jsiec") & meta.label_ok]
    unk = has(js.index[js.labels.map(lambda v: v[0] in UNKNOWN)])
    res["ood"] = {"thresholds": {"pooled": ood.thr_all, **ood.thr},
                  "jsiec_unknown_diseases": ood_metrics(s_in, ood.score(feats(unk)))}
    cif = os.path.join("features", args.name, "ood_cifar_global.npy")
    if os.path.exists(cif):
        res["ood"]["non_fundus_cifar"] = ood_metrics(s_in, ood.score(np.load(cif).astype(np.float32)))
    par = has(meta.index[(meta.dataset == "paraguay") & meta.label_ok])
    res["ood"]["paraguay_flagged"] = float(ood.flag(feats(par), meta.loc[par].source.values).mean())
    te_all = has(meta.index[(meta.bucket == "test") & meta.dataset.isin(["rfmid", "rfmid2"]) & meta.label_ok])
    res["ood"]["rfmid_test_flagged"] = float(ood.flag(feats(te_all), meta.loc[te_all].source.values).mean())

    os.makedirs(os.path.join("artifacts", args.name), exist_ok=True)
    joblib.dump({"classifier": clf, "mondrian": mond, "ood_thresholds": {"pooled": ood.thr_all, **ood.thr},
                 "ood_index_ids": index_ids, "alpha": ALPHA}, os.path.join("artifacts", args.name, "trust.joblib"))
    os.makedirs("outputs/eval", exist_ok=True)
    json.dump(res, open(f"outputs/eval/trust_{args.name}.json", "w"), indent=1, default=float)

    print(f"classifier: logistic C {C}, T {T:.3f}; val-A accuracy {res['val_a']['accuracy']:.3f}, log-loss "
          f"{res['val_a']['nll_T1']:.3f} -> {res['val_a']['nll_T']:.3f}, ECE {res['val_a']['ece']:.3f}")
    print(f"conformal (alpha {ALPHA}): q_hat {res['conformal']['q_hat']}, pooled {mond.q_all:.3f}, conservative "
          f"{mond.q_conservative:.3f}; val-B photos per camera {mond.n}")
    t = res["test_in_scope"]
    print(f"\ntest in scope ({t['all']['n']} photos): accuracy {t['accuracy']:.3f}, ECE {t['ece']:.3f}, coverage "
          f"{t['all']['coverage']:.3f}, mean set size {t['all']['mean_set_size']:.2f}, single-disease sets {t['all']['share_singleton']:.2f}")
    for c, v in t["by_camera"].items():
        print(f"  {c:20s} n {v['n']:4d}  coverage {v['coverage']:.3f}  set size {v['mean_set_size']:.2f}  singletons {v['share_singleton']:.2f}")
    print("new sites (true disease in the set):")
    for k, v in sites.items():
        print(f"  {k:12s} n {v['n']:4d}  coverage {v['coverage']:.3f}  set size {v['mean_set_size']:.2f}  top-1 {v['top1']:.3f}")
    rc = res["risk_coverage"]
    print("risk-coverage (error at coverage): " + ", ".join(f"{k} {v:.3f}" for k, v in rc["error_at_coverage"].items())
          + f"; AURC {rc['aurc']:.4f}")
    o = res["ood"]
    print(f"\nOOD: unknown JSIEC diseases AUROC {o['jsiec_unknown_diseases']['auroc']:.3f}, FPR@95TPR "
          f"{o['jsiec_unknown_diseases']['fpr_at_95tpr']:.3f} (n {o['jsiec_unknown_diseases']['n_out']})"
          + (f"; non-fundus AUROC {o['non_fundus_cifar']['auroc']:.3f}, FPR@95 {o['non_fundus_cifar']['fpr_at_95tpr']:.3f}"
             if "non_fundus_cifar" in o else "")
          + f"\n     flagged: RFMiD test {o['rfmid_test_flagged']:.3f} (target ~0.05), Paraguay (unseen camera) {o['paraguay_flagged']:.3f}")
