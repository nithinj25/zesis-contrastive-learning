"""The whole trust pipeline on held-out photos (CLAUDE.md 10.5, 10.6, 14 phase 5).

    python -m eval.decision_eval --name trace_v2

Per photo: quality gate -> OOD -> rare-disease bank -> conformal set (+ disagreement flag) -> one action
(trust/decide.py). Everything is fitted without test photos:
  quality gate, classifier, temperature, conformal, OOD: artifacts/<name>/{quality,trust}.joblib (eval/quality_eval.py,
      eval/trust_eval.py).
  rare bank: common diseases from head training, AION / CRVO / RP from their photos OUTSIDE the test bucket only, so
      no test photo is in the bank; margin tau on val-A for 5% false alarm (eval/rare_bank.py's rule).
  disagreement flag (10.6): gap = |classifier p(d) - graph posterior p(d)| for the diagnosed disease (graph/score.py on
      calibrated findings). Threshold 0.5; kept only if, on val-A, diagnoses with a large gap are less accurate than the
      rest (decided and printed here, never on test).
Outcome groups: RFMiD + RFMiD 2.0 test (all labelled photos); rare-disease test photos (RFMiD / RFMiD 2.0 test bucket +
JSIEC CRVO, RP); 13 JSIEC classes unknown to TRACE; Paraguay (unseen camera). "Unsafe" = a single DIAGNOSE that is
wrong. A diagnosis of normal is right only for a photo labelled normal; for Paraguay (a DR-screening site) the
"no DR signs" folder counts as normal, the same stated assumption as eval/site_onboarding.py.
Writes outputs/eval/decision_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from eval.leave_disease_out import findings_q
from eval.probes import load_features
from eval.rare_bank import rare_margin
from eval.trust_eval import UNKNOWN
from graph.bank import Bank, unit
from graph.build import build
from graph.score import score
from trust.classifier import CLASSES
from trust.decide import DIAGNOSE, REFER, SHORTLIST, decide
from trust.ood import KnnOOD

GAP = 0.5


def correct_dx(d: str, diseases) -> bool:
    return d in list(diseases)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    rare = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))["leave_disease_out"]
    art = joblib.load(os.path.join("artifacts", args.name, "trust.joblib"))
    gate = joblib.load(os.path.join("artifacts", args.name, "quality.joblib"))
    clf, mond = art["classifier"], art["mondrian"]
    f = load_features(args.name)
    X = unit(f.attrs["X"])
    row = f.set_index("image_id").row
    feats = lambda ids: X[row.loc[list(ids)].values]
    has = lambda ids: [i for i in ids if i in row.index]

    ood = KnnOOD().fit(feats(art["ood_index_ids"]))
    ood.thr_all, ood.thr = art["ood_thresholds"]["pooled"], {k: v for k, v in art["ood_thresholds"].items() if k != "pooled"}

    bank = Bank(args.name)
    head = meta[meta.in_heads & meta.label_ok]
    for d in ["normal", "DR", "myopia", "ARMD", "BRVO", "CSR", "hypertensive_retinopathy"]:
        bank.add(d, feats(has(head.index[head.diseases.map(lambda v, d=d: d in list(v))])))
    for d in rare:
        ids = has(meta.index[(meta.ldo_disease == d) & meta.label_ok & (meta.bucket == "ldo_test")])
        bank.add(d, feats(ids), ids)
    va_ids = has(meta.index[(meta.bucket == "val_a") & meta.label_ok])
    tau = float(np.percentile(rare_margin(bank, feats(va_ids), rare), 95))

    G = build(args.name)
    gcol = {j: G["diseases"].index(c) for j, c in enumerate(CLASSES) if c in G["diseases"]}  # other_abnormal has no node

    def truth(image_id) -> list:
        r = meta.loc[image_id]
        t = list(r.diseases)
        if not t and r.dataset == "paraguay":
            return ["normal"]
        if not t and len(r.findings) > 0:  # abnormal, none of the 6: the classifier's 7th class
            return ["other_abnormal"]
        return t

    def run(ids) -> pd.DataFrame:
        ids = has(ids)
        Xg, src = feats(ids), meta.loc[ids].source.values
        p = clf.proba(Xg)
        sets = mond.sets(p, src)
        q = findings_q(args.name, ids, G["findings"]) if all(meta.loc[ids].bucket.isin(["train", "val_a", "val_b", "test", "ldo_test"])) else None
        if q is not None:
            _, post, _ = score(q, G["p"], G["p_not"], G["prior"], G["r"])
            top = p.argmax(1)
            gap = np.array([abs(p[i, t] - post[i, gcol[t]]) if t in gcol else 0.0 for i, t in enumerate(top)])
        else:
            gap = np.zeros(len(ids))
        s = bank.scores(Xg)
        is_rare = np.isin(bank.names, rare)
        top_rare = np.array(bank.names)[np.where(is_rare)[0][s[:, is_rare].argmax(1)]]
        rare_call = np.where(rare_margin(bank, Xg, rare) > tau, top_rare, None)
        retake, unfam = gate.retake(Xg), ood.flag(Xg, src)
        out = []
        for i, image_id in enumerate(ids):
            cs = [CLASSES[j] for j in np.where(sets[i])[0]]
            dec = decide(bool(retake[i]), bool(unfam[i]), rare_call[i], cs, disagree=bool(gap[i] > GAP) if use_flag else False)
            out.append({"image_id": image_id, "action": dec.action, "diseases": dec.diseases, "gap": float(gap[i]),
                        "top": CLASSES[int(p[i].argmax())], "truth": truth(image_id)})
        return pd.DataFrame(out)

    # ---- disagreement flag: keep it only if large-gap diagnoses are less accurate on val-A ----
    use_flag = False
    dv = run(va_ids)
    dx = dv[dv.action == DIAGNOSE]
    ok = np.array([correct_dx(d[0], t) for d, t in zip(dx.diseases, dx.truth)])
    big = dx.gap.values > GAP
    flag_check = {"val_a_diagnoses": int(len(dx)), "large_gap": int(big.sum()),
                  "accuracy_large_gap": float(ok[big].mean()) if big.any() else None,
                  "accuracy_small_gap": float(ok[~big].mean()) if (~big).any() else None}
    use_flag = bool(big.any() and ok[big].mean() < ok[~big].mean())
    flag_check["kept"] = use_flag

    groups = {
        "rfmid_test": meta.index[(meta.bucket == "test") & meta.dataset.isin(["rfmid", "rfmid2"]) & meta.label_ok & (meta.ldo_disease == "")],
        "rare_test": meta.index[(meta.ldo_disease.isin(rare)) & meta.label_ok & (meta.bucket == "test")],
        "jsiec_unknown": meta.index[(meta.dataset == "jsiec") & meta.label_ok & meta.labels.map(lambda v: len(v) > 0 and v[0] in UNKNOWN)],
        "paraguay": meta.index[(meta.dataset == "paraguay") & meta.label_ok & ~meta.label_conflict],
    }
    res = {"tau": tau, "disagreement": flag_check, "groups": {}}
    for gname, ids in groups.items():
        d = run(ids)
        share = d.action.value_counts(normalize=True).to_dict()
        dxs = d[d.action == DIAGNOSE]
        wrong = [not correct_dx(x[0], t) for x, t in zip(dxs.diseases, dxs.truth)]
        sl = d[d.action == SHORTLIST]
        r = {"n": len(d), "actions": {k: float(v) for k, v in share.items()},
             "diagnose_accuracy": float(1 - np.mean(wrong)) if len(dxs) else None,
             "unsafe_share": float(np.sum(wrong) / len(d)),
             "shortlist_contains_truth": float(np.mean([any(x in t for x in s) for s, t in zip(sl.diseases, sl.truth)])) if len(sl) else None}
        if gname == "rare_test":
            ref = d[d.action == REFER]
            r["referred_naming_true_rare"] = float(np.mean([any(x in t for x in s) for s, t in zip(ref.diseases, ref.truth)])) if len(ref) else None
            r["by_dataset"] = meta.loc[d.image_id].dataset.value_counts().to_dict()
        res["groups"][gname] = r
    json.dump(res, open(f"outputs/eval/decision_{args.name}.json", "w"), indent=1, default=float)

    print(f"rare-call margin tau {tau:.4f} (val-A, 5% false alarm). Disagreement flag on val-A: {flag_check}")
    for g, r in res["groups"].items():
        acts = ", ".join(f"{a} {r['actions'].get(a, 0):.2f}" for a in ("diagnose", "shortlist", "refer", "retake"))
        extra = f", shortlist holds truth {r['shortlist_contains_truth']:.2f}" if r["shortlist_contains_truth"] is not None else ""
        if g == "rare_test" and r.get("referred_naming_true_rare") is not None:
            extra += f", referrals naming the true rare disease {r['referred_naming_true_rare']:.2f}"
        dxa = f"{r['diagnose_accuracy']:.3f}" if r["diagnose_accuracy"] is not None else "-"
        print(f"  {g:14s} n {r['n']:5d}: {acts} | diagnoses right {dxa}, UNSAFE (confident wrong) {r['unsafe_share']:.3f}{extra}")
