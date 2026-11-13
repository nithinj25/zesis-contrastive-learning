"""Save the finding detectors so a NEW photo can be scored (the demo; CLAUDE.md 19.10).

    python -m findings.export --name trace_v2

findings/probes.py fitted its final detectors and saved only their outputs (features/<name>/<bucket>_findings.parquet)
and calibration (artifacts/<name>/findings.joblib). This refits the same final models (same head-training photos,
same detector list, same MLP, seed 0) on both inputs ("global", "global+disc") and saves scaler + network weights to
artifacts/<name>/finding_models.pt. Check: the refit, Platt-calibrated, must reproduce the saved test-bucket finding
probabilities (printed: max absolute difference per input; GPU arithmetic can differ in the last digits).
"""
from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import torch

from findings.probes import BUCKETS, fit_mlp, load_split, platt_apply, predict_mlp


def finding_scores(models: dict, fj: dict, g: np.ndarray, d: np.ndarray | None) -> dict:
    """Calibrated p(finding) for photos with global features g (N, 2048) and disc-crop features d (N, 2048 or None
    when the disc was not found). Disc-routed findings without a disc crop are NaN (skipped, section 19.8)."""
    rep = fj["report"]
    inputs = {"global": g, "global+disc": None if d is None else np.concatenate([g, d], 1)}
    out = {}
    for inp, m in models.items():
        X = inputs[inp]
        S = predict_mlp(m["scaler"], m["net"], X) if X is not None else None
        for j, f in enumerate(fj["detectors"]):
            if rep[f]["input"] == inp:
                out[f] = platt_apply(S[:, j], fj["platt"][f]) if S is not None else np.full(len(g), np.nan)
    for f in fj["prototypes"]:
        X = inputs[rep[f]["input"]]
        if X is None:
            out[f] = np.full(len(g), np.nan)
            continue
        p = fj["prototype_vectors"][f]
        out[f] = platt_apply((X @ p) / (np.linalg.norm(X, axis=1) * np.linalg.norm(p) + 1e-9), fj["platt"][f])
    return out


def load_models(name: str, device: str = "cpu") -> dict:
    blob = torch.load(os.path.join("artifacts", name, "finding_models.pt"), map_location=device, weights_only=False)
    out = {}
    for inp, m in blob.items():
        net = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(m["in"], 512), torch.nn.ReLU(),
                                  torch.nn.Dropout(0.3), torch.nn.Linear(512, m["out"]))
        net.load_state_dict(m["state"])
        out[inp] = {"scaler": m["scaler"], "net": net.to(device).eval()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    fj = joblib.load(os.path.join("artifacts", args.name, "findings.joblib"))
    data = {b: load_split(args.name, b) for b in BUCKETS}
    tr_meta = meta.loc[data["train"][0].image_id]
    keep_tr = (tr_meta.in_heads & tr_meta.label_ok).values
    Ytr = np.array([[f in list(x) for f in fj["detectors"]] for x in tr_meta[keep_tr].findings], dtype=np.float32)

    blob, models = {}, {}
    for inp in ("global", "global+disc"):
        sc, net = fit_mlp(data["train"][1][inp][keep_tr], Ytr)
        net = net.cpu()
        blob[inp] = {"scaler": sc, "state": net.state_dict(), "in": net[1].in_features, "out": net[4].out_features}
        models[inp] = {"scaler": sc, "net": net}
    torch.save(blob, os.path.join("artifacts", args.name, "finding_models.pt"))

    test = data["test"]
    q = finding_scores(models, fj, test[1]["global"], np.asarray(test[1]["global+disc"])[:, 2048:])
    saved = pd.read_parquet(os.path.join("features", args.name, "test_findings.parquet")).set_index("image_id")
    saved = saved.loc[test[0].image_id]
    diffs = {f: float(np.nanmax(np.abs(q[f] - saved[f].values))) for f in q}
    print("max |refit - saved| per finding on the test bucket:")
    for f, v in sorted(diffs.items(), key=lambda t: -t[1]):
        print(f"  {f:20s} {v:.5f}")
    print(f"saved artifacts/{args.name}/finding_models.pt")
