"""New-site onboarding (CLAUDE.md 10.4): can 100 labelled photos from an unseen camera restore conformal coverage?

    python -m eval.site_onboarding --name trace_v2

Site: Paraguay (ZEISS VISUCAM 500, never in the pool or any fit). Labels: DR grade folders; for this DR-screening
site, "no DR signs" (folder 1) is taken as normal (an assumption, stated). The classifier and its temperature are the
ones fitted in eval/trust_eval.py (artifacts/<name>/trust.joblib), unchanged; only the conformal threshold changes.
20 random splits by same-eye group: N_CAL photos calibrate Paraguay's own q_hat, the rest are scored. Compared with
the threshold an unseen camera gets by default (the most conservative one from val-B).
Writes outputs/eval/site_onboarding_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from eval.probes import load_features
from trust.classifier import CLASSES
from trust.conformal import calib_scores, prediction_sets, q_hat

N_CAL, SPLITS = 100, 20

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    art = joblib.load(os.path.join("artifacts", args.name, "trust.joblib"))
    clf, mond, alpha = art["classifier"], art["mondrian"], art["alpha"]
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    f = load_features(args.name)
    X = f.attrs["X"] / np.linalg.norm(f.attrs["X"], axis=1, keepdims=True)
    row = f.set_index("image_id").row
    par = meta[(meta.dataset == "paraguay") & meta.label_ok & ~meta.label_conflict]
    par = par[[i in row.index for i in par.index]]
    y = np.where(par.diseases.map(lambda v: list(v) == ["DR"]), CLASSES.index("DR"), CLASSES.index("normal"))
    p = clf.proba(X[row.loc[par.index].values])
    groups = par.dup_group.fillna(-1).astype(str).values
    groups = np.where(groups == "-1.0", par.index.values, groups)
    ug = np.unique(groups)
    default = prediction_sets(p, mond.q_conservative)
    runs = []
    for s in range(SPLITS):
        rng = np.random.default_rng(s)
        order = rng.permutation(ug)
        cal_g, n = [], 0
        for g in order:  # whole eyes into calibration until it holds N_CAL photos
            if n >= N_CAL:
                break
            cal_g.append(g)
            n += int((groups == g).sum())
        cal = np.isin(groups, cal_g)
        qh = q_hat(calib_scores(p[cal], y[cal]), alpha)
        sets = prediction_sets(p[~cal], qh)
        yt = y[~cal]
        runs.append({"n_cal": int(cal.sum()), "q_hat": qh,
                     "coverage": float(sets[np.arange(len(yt)), yt].mean()),
                     "coverage_DR": float(sets[np.arange(len(yt)), yt][yt == CLASSES.index("DR")].mean()),
                     "set_size": float(sets.sum(1).mean()),
                     "default_coverage": float(default[~cal][np.arange(len(yt)), yt].mean()),
                     "default_set_size": float(default[~cal].sum(1).mean())})
    r = pd.DataFrame(runs)
    out = {"n": len(par), "n_dr": int((y == CLASSES.index("DR")).sum()), "alpha": alpha, "q_conservative": mond.q_conservative,
           "summary": {c: [float(r[c].mean()), float(r[c].min()), float(r[c].max())] for c in r.columns}, "runs": runs}
    os.makedirs("outputs/eval", exist_ok=True)
    json.dump(out, open(f"outputs/eval/site_onboarding_{args.name}.json", "w"), indent=1)
    sm = out["summary"]
    print(f"Paraguay {out['n']} photos ({out['n_dr']} DR); {SPLITS} splits, ~{N_CAL} calibration photos each; target coverage {1 - alpha:.0%}")
    print(f"  default (unseen-camera threshold {mond.q_conservative:.3f}): coverage {sm['default_coverage'][0]:.3f}, "
          f"set size {sm['default_set_size'][0]:.2f}")
    print(f"  onboarded (Paraguay's own q_hat {sm['q_hat'][0]:.3f} [{sm['q_hat'][1]:.3f}, {sm['q_hat'][2]:.3f}]): coverage "
          f"{sm['coverage'][0]:.3f} [{sm['coverage'][1]:.3f}, {sm['coverage'][2]:.3f}], DR photos {sm['coverage_DR'][0]:.3f}, "
          f"set size {sm['set_size'][0]:.2f}")
