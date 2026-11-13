"""Approach 2's stress test on our encoders: how gracefully does each degrade as photo quality drops?

Run from the repo root:  python -m eval.stress trace_v2 imagenet_r50 moco_plain_e15 supervised_r50
Photos: RFMiD test (639). Each is corrupted with one of Approach 2's 11 artifacts at level 1-5
(augment/stress_artifacts.py; identical draws for every encoder) on the processed photo, then finished exactly as
at evaluation (step 7 for a known camera, step 8, 384 px). Probes (one logistic per label, C from the encoder's
probe run, else 0.01) are fitted once on CLEAN cached RFMiD-train features, so any drop is the encoder's.
Per (encoder, artifact, level): macro-AUROC over the 18 scored RFMiD labels, Disease_Risk AUROC, stability (mean
cosine between a photo's clean and corrupted feature) and flip rate (Disease_Risk call at 0.5 that changes).
mCE (section 19.6): CE_artifact = sum over levels of (1 - macro-AUROC) / the same sum for ImageNet; mCE = mean.
Weights are read from the main repo's checkpoints (read only). Writes outputs/stress/stress.csv and summary.json.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

MAIN = r"C:\Users\Nithin J\fast_projects\zesis"
WEIGHTS = {"imagenet_r50": None, "trace_v2": "checkpoints/trace_v2/encoder.pt",
           "moco_plain_e15": "checkpoints/moco_plain_e15/encoder.pt", "supervised_r50": "checkpoints/supervised_r50/encoder.pt"}
_S: dict = {}


def _init(paths, sources, keys):
    import cv2

    cv2.setNumThreads(1)
    _S.update(paths=paths, sources=sources, keys=keys, stats=json.load(open("data/trace_preprocess.json")))


def _make(job):
    from augment.stress_artifacts import corrupt
    from data.process import load_processed
    from encoder.export import finish_view

    idx, name, level = job
    out = []
    for i in idx:
        rgb, mask = load_processed(os.path.join("processed", _S["paths"][i]))
        img = corrupt(rgb, mask, name, level, _S["keys"][i])
        out.append((finish_view(img, mask, _S["sources"][i], _S["stats"], 384) * 255).round().byte().numpy())
    return np.stack(out)


def main(encoders: list[str]):
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from augment.stress_artifacts import ARTIFACTS
    from encoder.export import load_trace_encoder
    from eval.probes import load_features

    meta = pd.read_parquet("data/meta.parquet")
    pick = lambda b: meta[(meta.bucket == b) & (meta.dataset == "rfmid") & meta.label_ok & meta.fov_ok].reset_index(drop=True)
    tr, te = pick("train"), pick("test")
    scored = json.load(open(os.path.join(MAIN, "outputs/eval/trace_v1_384.json")))["rfmid_labels_scored"]
    Y = lambda d: np.array([[l in set(x) for l in scored] for x in d.labels], float)
    Ytr, Yte = Y(tr), Y(te)
    rtr, rte = tr.disease_risk.astype(float).values, te.disease_risk.astype(float).values

    dev = "cuda"
    nets, probes, cached = {}, {}, {}
    for e in encoders:
        w = WEIGHTS[e]
        nets[e] = load_trace_encoder(None if w is None else os.path.join(MAIN, w), dev).half()
        f = load_features(e, ("train", "test"))
        row = f.set_index("image_id").row
        Xtr, cached[e] = f.attrs["X"][row.loc[tr.image_id].values], f.attrs["X"][row.loc[te.image_id].values]
        pj = os.path.join(MAIN, f"outputs/eval/{'trace_v1_384' if e == 'trace_v2' else e}.json")
        C = json.load(open(pj)).get("rfmid_C", 0.01) if os.path.exists(pj) else 0.01
        fit = lambda y: make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000)).fit(Xtr, y)
        probes[e] = ([fit(Ytr[:, j]) for j in range(len(scored))], fit(rtr), C)
        print(f"{e}: probes fitted (C = {C})", flush=True)

    settings = [("clean", 0)] + [(n, l) for n in ARTIFACTS for l in range(1, 6)]
    chunks = [list(range(i, min(i + 32, len(te)))) for i in range(0, len(te), 32)]
    pool = mp.get_context("spawn").Pool(6, _init, (te.processed_path.tolist(), te.source.tolist(), te.image_id.tolist()))
    clean, rows = {}, []
    t0 = time.time()
    for name, level in settings:
        feats = {e: [] for e in encoders}
        for batch in pool.imap(_make, [(c, name, level) for c in chunks]):
            x = torch.from_numpy(batch).to(dev).half().div_(255)
            with torch.no_grad():
                for e in encoders:
                    feats[e].append(nets[e](x).float().cpu().numpy())
        for e in encoders:
            F = np.concatenate(feats[e])
            if level == 0:
                clean[e] = F
                agree = float((F * cached[e] / np.linalg.norm(cached[e], axis=1, keepdims=True)).sum(1).mean())
                print(f"{e}: clean features vs cached, mean cosine {agree:.4f}", flush=True)
            heads, risk_head, _ = probes[e]
            S = np.column_stack([h.predict_proba(F)[:, 1] for h in heads])
            rp, rc = risk_head.predict_proba(F)[:, 1], risk_head.predict_proba(clean[e])[:, 1]
            rows.append({"encoder": e, "artifact": name, "level": level,
                         "macro_auroc": float(np.mean([roc_auc_score(Yte[:, j], S[:, j]) for j in range(len(scored))])),
                         "risk_auroc": float(roc_auc_score(rte, rp)),
                         "stability": float((F * clean[e]).sum(1).mean()),
                         "flip_rate": float(((rp > 0.5) != (rc > 0.5)).mean())})
        print(f"{name} level {level}: " + "  ".join(f"{r['encoder']} {r['macro_auroc']:.3f}" for r in rows[-len(encoders):])
              + f"   ({time.time() - t0:.0f} s)", flush=True)
    pool.close()

    df = pd.DataFrame(rows)
    os.makedirs("outputs/stress", exist_ok=True)
    df.to_csv("outputs/stress/stress.csv", index=False)
    cor = df[df.level > 0]
    summary = {}
    ref = "imagenet_r50" if "imagenet_r50" in encoders else None
    for e in encoders:
        d = cor[cor.encoder == e]
        s = {"clean_macro_auroc": float(df[(df.encoder == e) & (df.level == 0)].macro_auroc.iloc[0]),
             "corrupted_mean_macro_auroc": float(d.macro_auroc.mean()),
             "level5_mean_macro_auroc": float(d[d.level == 5].macro_auroc.mean()),
             "mean_stability": float(d.stability.mean()), "mean_flip_rate": float(d.flip_rate.mean()),
             "worst_artifact": d.groupby("artifact").macro_auroc.mean().idxmin()}
        if ref:
            r = cor[cor.encoder == ref]
            ce = {a: float((1 - d[d.artifact == a].macro_auroc).sum() / (1 - r[r.artifact == a].macro_auroc).sum()) for a in d.artifact.unique()}
            s["mCE_vs_imagenet"] = float(np.mean(list(ce.values())))
            s["CE_per_artifact"] = ce
        summary[e] = s
    json.dump(summary, open("outputs/stress/summary.json", "w"), indent=1)
    for e, s in summary.items():
        print(f"{e:16s} clean {s['clean_macro_auroc']:.3f}  corrupted mean {s['corrupted_mean_macro_auroc']:.3f}  "
              f"level-5 {s['level5_mean_macro_auroc']:.3f}  mCE {s.get('mCE_vs_imagenet', float('nan')):.3f}  "
              f"stability {s['mean_stability']:.3f}  flips {s['mean_flip_rate']:.3f}  worst {s['worst_artifact']}")


if __name__ == "__main__":
    main(sys.argv[1:])
