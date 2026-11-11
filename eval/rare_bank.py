"""How well does the disease bank (graph/bank.py) recognise rare diseases? (CLAUDE.md sections 4, 11)

    python -m eval.rare_bank trace_v2 moco_plain_e15 imagenet_r50

Bank: 10 diseases (7 common from head training, AION / CRVO / RP from their RFMiD + RFMiD 2.0 photos). Each
encoder gets its own bank, built the same way.
  Test 1, new dataset: JSIEC photos (other hospital and cameras, never in the bank) whose disease is in the bank.
                       Rare: CRVO 22, RP 22. Common ones are reported too.
  Test 2, leave one eye out: every RFMiD / RFMiD 2.0 photo of a rare disease, ranked by a bank whose rare
                       prototype is rebuilt without that eye's photos.
  False alarm: share of val-A photos (no AION, CRVO or RP among them) whose top-1 is a rare disease.
Rare-call margin: cosine ranks ignore how rare a disease is (17% of val-A photos came out rare). So a rare disease
is only CALLED when its cosine beats the best common disease by a margin tau, set on val-A so at most 5% of its
photos would raise a rare call (the same rule as the OOD threshold, 10.3; needs no rare photos). Reported: the
share of rare photos still called correctly (sensitivity) at that tau, next to the plain top-1.
Metric: top-1 and top-3 among the 10 diseases (chance 0.10 / 0.30). Writes outputs/eval/rare_bank.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import yaml

from eval.probes import load_features
from graph.bank import Bank, disease_rows, unit


def rank_of(bank: Bank, X: np.ndarray, disease: str) -> np.ndarray:
    s = bank.scores(X)
    c = bank.names.index(disease)
    return (s > s[:, [c]]).sum(1) + 1


def rare_margin(bank: Bank, X: np.ndarray, rare: list[str]) -> np.ndarray:
    """Best rare cosine minus best common cosine, per photo."""
    s = bank.scores(X)
    is_rare = np.isin(bank.names, rare)
    return s[:, is_rare].max(1) - s[:, ~is_rare].max(1)


def called(bank: Bank, X: np.ndarray, disease: str, rare: list[str], tau: float) -> np.ndarray:
    """True where `disease` is the top rare disease AND clears the margin."""
    s = bank.scores(X)
    is_rare = np.isin(bank.names, rare)
    top_rare = np.array(bank.names)[np.where(is_rare)[0][s[:, is_rare].argmax(1)]]
    return (top_rare == disease) & (rare_margin(bank, X, rare) > tau)


def run(meta: pd.DataFrame, name: str, rare: list[str]) -> dict:
    f = load_features(name)
    row = f.set_index("image_id").row
    X = unit(f.attrs["X"])
    feats = lambda ids: X[row.loc[ids].values]
    rows = {d: [i for i in ids if i in row.index] for d, ids in disease_rows(meta, rare).items()}
    bank = Bank(name)
    for d, ids in rows.items():
        bank.add(d, feats(ids), ids)
    va = [i for i in meta.index[(meta.bucket == "val_a") & meta.label_ok] if i in row.index]
    tau = float(np.percentile(rare_margin(bank, feats(va), rare), 95))
    out = {"jsiec": {}, "leave_one_eye_out": {}, "tau": tau}
    js = meta[(meta.dataset == "jsiec") & meta.label_ok]
    for d in bank.names:
        ids = [i for i in js.index[js.diseases.map(lambda x, d=d: d in list(x))] if i in row.index]
        if ids:
            r = rank_of(bank, feats(ids), d)
            out["jsiec"][d] = {"n": len(ids), "top1": float((r == 1).mean()), "top3": float((r <= 3).mean())}
            if d in rare:
                out["jsiec"][d]["called"] = float(called(bank, feats(ids), d, rare, tau).mean())
    for d in rare:
        ids = rows[d]
        groups = meta.loc[ids].dup_group.fillna(-1).astype(str).values
        groups = np.where(groups == "-1.0", np.array(ids), groups)
        ranks, calls = [], []
        for i, g in enumerate(groups):
            keep = [j for j, gg in zip(ids, groups) if gg != g]
            b = Bank(name)
            for dd, ii in rows.items():
                b.add(dd, feats(keep) if dd == d else feats(ii))
            ranks.append(int(rank_of(b, feats([ids[i]]), d)[0]))
            calls.append(bool(called(b, feats([ids[i]]), d, rare, tau)[0]))
        r = np.array(ranks)
        out["leave_one_eye_out"][d] = {"n": len(ids), "top1": float((r == 1).mean()), "top3": float((r <= 3).mean()),
                                       "called": float(np.mean(calls))}
    top = np.array(bank.names)[bank.scores(feats(va)).argmax(1)]
    out["val_a_false_alarm"] = float(np.isin(top, rare).mean())
    out["val_a_top1_calls"] = pd.Series(top).value_counts().to_dict()
    out["val_a_false_alarm_with_margin"] = float((rare_margin(bank, feats(va), rare) > tau).mean())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    rare = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))["leave_disease_out"]
    res = {n: run(meta, n, rare) for n in args.names}
    os.makedirs("outputs/eval", exist_ok=True)
    json.dump(res, open("outputs/eval/rare_bank.json", "w"), indent=1)
    print("top-1 / top-3 among 10 diseases (chance 0.10 / 0.30)")
    for part in ("jsiec", "leave_one_eye_out"):
        print(f"\n{'JSIEC (new dataset, never in the bank)' if part == 'jsiec' else 'RFMiD rare photos, leave one eye out'}")
        diseases = list(res[args.names[0]][part].keys())
        for d in diseases:
            n = res[args.names[0]][part][d]["n"]
            cells = "   ".join(f"{nm} {res[nm][part][d]['top1']:.2f}/{res[nm][part][d]['top3']:.2f}"
                               + (f" called {res[nm][part][d]['called']:.2f}" if "called" in res[nm][part][d] else "")
                               for nm in args.names)
            print(f"  {d:26s} n {n:3d}   {cells}")
    print("\nval-A false alarm (top-1 is AION, CRVO or RP; val-A has none): "
          + "   ".join(f"{nm} {res[nm]['val_a_false_alarm']:.3f} -> {res[nm]['val_a_false_alarm_with_margin']:.3f} "
                       f"with margin {res[nm]['tau']:+.4f}" for nm in args.names))
