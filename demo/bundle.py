"""Build the demo bundle: everything the one-photo pipeline needs that otherwise lives in data/meta.parquet and
features/, so the demo runs from artifacts/ alone (CLAUDE.md 19.10).

    python -m demo.bundle --name trace_v2

Writes artifacts/<name>/demo_bundle.joblib with feature vectors and labels only, no photos:
  ood        : the trust layer's OOD index (pool photos, float16) and its thresholds (from trust.joblib)
  bank       : the rare-disease bank and its margin tau, built exactly as eval/decision_eval.py builds them
  library    : the retrieval case library (eval/retrieval.py: head training + rare photos) with ids, diagnoses and
               processed_path, so thumbnails can be shown when processed/ exists locally
  graph      : graph.build(<name>) (disease x finding tables for the evidence for / against)
  cameras    : camera ids that have stored colour statistics (step 7)
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml

from eval.probes import load_features
from eval.rare_bank import rare_margin
from eval.retrieval import setup
from graph.bank import Bank, unit
from graph.build import build

COMMON = ["normal", "DR", "myopia", "ARMD", "BRVO", "CSR", "hypertensive_retinopathy"]


def main(name: str):
    meta = pd.read_parquet("data/meta.parquet")
    mi = meta.set_index("image_id")
    rare = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))["leave_disease_out"]
    art = joblib.load(os.path.join("artifacts", name, "trust.joblib"))
    f = load_features(name)
    X = unit(f.attrs["X"])
    row = f.set_index("image_id").row
    feats = lambda ids: X[row.loc[list(ids)].values]
    has = lambda ids: [i for i in ids if i in row.index]

    bank = Bank(name)  # as eval/decision_eval.py
    head = mi[mi.in_heads & mi.label_ok]
    for d in COMMON:
        bank.add(d, feats(has(head.index[head.diseases.map(lambda v, d=d: d in list(v))])))
    for d in rare:
        ids = has(mi.index[(mi.ldo_disease == d) & mi.label_ok & (mi.bucket == "ldo_test")])
        bank.add(d, feats(ids), ids)
    va_ids = has(mi.index[(mi.bucket == "val_a") & mi.label_ok])
    tau = float(np.percentile(rare_margin(bank, feats(va_ids), rare), 95))

    lib = setup(meta)[0]
    out = {
        "encoder": name,
        "ood": {"ids": np.array(art["ood_index_ids"]), "X": feats(art["ood_index_ids"]).astype(np.float16),
                "thresholds": art["ood_thresholds"]},
        "bank": {"names": list(bank.names), "protos": np.stack(bank.protos), "n": list(bank.n), "rare": rare, "tau": tau},
        "library": {"ids": lib.image_id.values, "X": feats(lib.image_id).astype(np.float16), "dx": lib.dx.values,
                    "diseases": [list(x) for x in lib.diseases], "processed_path": lib.processed_path.values,
                    "dataset": lib.dataset.values},
        "graph": build(name),
        "cameras": sorted(json.load(open("data/trace_preprocess.json"))["cameras"]),
    }
    path = os.path.join("artifacts", name, "demo_bundle.joblib")
    joblib.dump(out, path, compress=3)
    print(f"{path}: {os.path.getsize(path) / 1e6:.0f} MB; OOD index {len(out['ood']['ids'])}, library {len(lib)}, "
          f"bank {bank.names}, tau {tau:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="trace_v2")
    main(ap.parse_args().name)
