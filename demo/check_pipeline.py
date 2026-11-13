"""Does the one-photo demo pipeline reproduce the evaluation? (run once after building the bundle)

    python -m demo.check_pipeline --name trace_v2

On the evaluation's "rfmid_test" group (RFMiD + RFMiD 2.0 test photos), photo by photo from processed/: feature
agreement with the cached features, finding agreement with the saved finding probabilities, and the share of each
action against outputs/eval/decision_<name>.json. The demo encodes in fp32, the cache in fp16, so a few photos close
to a threshold may change action.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from data.process import load_processed
from demo.pipeline import TracePipeline
from eval.probes import load_features

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="trace_v2")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    ids = meta.index[(meta.bucket == "test") & meta.dataset.isin(["rfmid", "rfmid2"]) & meta.label_ok & (meta.ldo_disease == "")]
    f = load_features(args.name, ("test",))
    X = f.attrs["X"][f.set_index("image_id").row.loc[ids].values]
    saved_q = pd.read_parquet(os.path.join("features", args.name, "test_findings.parquet")).set_index("image_id").loc[ids]
    pipe = TracePipeline(args.name, device=args.device)
    acts, cos, qdiff = [], [], []
    for n, i in enumerate(ids):
        r = pipe.analyse(processed=load_processed(os.path.join("processed", meta.loc[i].processed_path)), camera=meta.loc[i].source)
        acts.append(r["action"])
        cos.append(float(r["feature"] @ X[n] / np.linalg.norm(X[n])))
        qdiff.append(np.nanmax(np.abs(np.array([r["findings"][c] for c in saved_q.columns]) - saved_q.loc[i].values)))
    ours = pd.Series(acts).value_counts(normalize=True).round(3).to_dict()
    ref = {k: round(v, 3) for k, v in json.load(open(f"outputs/eval/decision_{args.name}.json"))["groups"]["rfmid_test"]["actions"].items()}
    print(f"{len(ids)} photos; feature cosine to cache: min {min(cos):.5f}, mean {np.mean(cos):.6f}")
    print(f"finding probabilities vs saved: max |diff| {max(qdiff):.4f}, median {np.median(qdiff):.5f}")
    print(f"actions, demo pipeline: {ours}")
    print(f"actions, evaluation   : {ref}")
