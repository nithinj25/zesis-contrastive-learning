"""Quality gate evaluation (CLAUDE.md 10.1; trust/quality.py).

    python -m eval.quality_eval --name trace_v2

Data: EyeQ-graded EyePACS photos with features. Train / test split by patient (80 / 20). C by grouped out-of-fold
reject AUROC on the training patients; threshold set out-of-fold on them for 90% reject recall.
Reported on the held-out patients: reject AUROC, reject recall, and how often good / usable photos are sent back.
Shortcut check: good and usable pool photos were in the encoder's pretraining pool and rejects never were, so a gate
could learn "seen in pretraining". Two checks: (1) the 322 EyeQ-graded photos that never entered the pool (dropped as
copies of test photos), all three grades; (2) the share of RFMiD, RFMiD 2.0, IDRiD and Paraguay test photos (never in
the pool, mostly readable) sent back for a retake.
Version 2 (--readable-extra, the default): the EyeQ-only gate sent back 34-67% of doctor-graded photos from other
cameras (it learned what a bad EyePACS photo looks like). A photo an eye doctor diagnosed from is readable, so the
RFMiD + RFMiD 2.0 head-training photos join the training patients as "usable" (test photos stay out). False retakes
are then measured on doctor-graded test photos (RFMiD, RFMiD 2.0, IDRiD, JSIEC test; Paraguay = an untouched camera).
Trade-off: rejects come from EyePACS only, so the gate may be lenient with bad photos from other cameras.
Writes outputs/eval/quality_<name>.json and artifacts/<name>/quality.joblib.
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
from trust.quality import QualityGate, oof_p_reject, patient_split

C_GRID = (0.001, 0.01, 0.1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--eyeq-only", action="store_true", help="version 1: EyeQ grades only")
    ap.add_argument("--onboard", type=int, default=0,
                    help="new-site check: add N Paraguay photos (whole eyes) as readable; score the rest of Paraguay")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    f = load_features(args.name, buckets=("train", "val_a", "val_b", "test", "ldo_test", "pool", "spare", "excluded"))
    X = f.attrs["X"] / np.linalg.norm(f.attrs["X"], axis=1, keepdims=True)
    row = f.set_index("image_id").row
    eq = meta[meta.eyeq_quality.notna() & meta.index.isin(row.index)]
    clean = eq[eq.bucket == "excluded"]            # never in the pool: the shortcut check
    eq = eq[eq.bucket != "excluded"]
    te_mask = patient_split(eq.patient.values, 0.2, seed=0)
    tr, te = eq[~te_mask], eq[te_mask]
    if not args.eyeq_only:
        extra = meta[meta.in_heads & meta.index.isin(row.index)].copy()
        extra["eyeq_quality"] = "usable"
        extra["patient"] = extra.dup_group.fillna(-1).astype(str).where(extra.dup_group.notna(), extra.index.to_series())
        tr = pd.concat([tr, extra])
    onboard_ids = []
    if args.onboard:
        par = meta[(meta.dataset == "paraguay") & meta.index.isin(row.index)].copy()
        grp = par.dup_group.fillna(-1).astype(str).where(par.dup_group.notna(), par.index.to_series())
        n = 0
        for g in np.random.default_rng(0).permutation(grp.unique()):
            if n >= args.onboard:
                break
            onboard_ids += list(grp.index[grp == g])
            n = len(onboard_ids)
        add = par.loc[onboard_ids].assign(eyeq_quality="usable", patient=grp.loc[onboard_ids])
        tr = pd.concat([tr, add])
    Xtr, Xte = X[row.loc[tr.index].values], X[row.loc[te.index].values]
    gtr = tr.eyeq_quality.values
    best = max(C_GRID, key=lambda C: roc_auc_score(gtr == "reject", oof_p_reject(Xtr, gtr, tr.patient.values, C)))
    p_oof = oof_p_reject(Xtr, gtr, tr.patient.values, best)
    gate = QualityGate(best).fit(Xtr, gtr).set_threshold(p_oof, gtr == "reject")

    def report(d: pd.DataFrame) -> dict:
        p = gate.p_reject(X[row.loc[d.index].values])
        g = d.eyeq_quality.values
        out = {"n": len(d), "counts": pd.Series(g).value_counts().to_dict()}
        if 0 < (g == "reject").sum() < len(g):
            out["reject_auroc"] = float(roc_auc_score(g == "reject", p))
        for grade in ("good", "usable", "reject"):
            if (g == grade).any():
                out[f"retake_share_{grade}"] = float((p[g == grade] > gate.threshold).mean())
        return out

    res = {"version": "eyeq_only" if args.eyeq_only else "eyeq_plus_doctor_graded", "C": best, "threshold": gate.threshold, "train_patients": int(tr.patient.nunique()),
           "test_patients": int(te.patient.nunique()), "held_out": report(te), "never_pooled": report(clean)}
    other = {}
    for ds in ("rfmid", "rfmid2", "idrid", "paraguay", "jsiec"):
        ids = [i for i in meta.index[(meta.dataset == ds) & (meta.bucket == "test")] if i in row.index and i not in set(onboard_ids)]
        other[ds] = {"n": len(ids), "retake_share": float(gate.retake(X[row.loc[ids].values]).mean())}
    res["other_datasets_retake_share"] = other
    os.makedirs(os.path.join("artifacts", args.name), exist_ok=True)
    if not args.onboard and not args.eyeq_only:  # the deployed gate is version 2 without any Paraguay photo
        joblib.dump(gate, os.path.join("artifacts", args.name, "quality.joblib"))
    tag = "_eyeq_only" if args.eyeq_only else f"_onboard{args.onboard}" if args.onboard else ""
    json.dump(res, open(f"outputs/eval/quality_{args.name}{tag}.json", "w"), indent=1)
    print(f"quality gate: C {best}, retake threshold p(reject) > {gate.threshold:.3f} (90% reject recall on training patients)")
    for k in ("held_out", "never_pooled"):
        r = res[k]
        print(f"  {k:12s} n {r['n']:5d} {r['counts']}: reject AUROC {r.get('reject_auroc', float('nan')):.3f}; sent back: "
              + ", ".join(f"{g} {r[f'retake_share_{g}']:.3f}" for g in ("good", "usable", "reject") if f"retake_share_{g}" in r))
    print("  sent back on other test sets: " + ", ".join(f"{d} {v['retake_share']:.3f} (n {v['n']})" for d, v in other.items()))
