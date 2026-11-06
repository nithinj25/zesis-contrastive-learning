"""Does preprocessing remove camera signal but keep biology? (CLAUDE.md section 6, "How we know it worked")

Run from the repo root after data.splits:  python scripts/camera_probe.py [--per-dataset 400]
1. Camera probe: a logistic regression guesses the dataset from simple photo statistics (colour mean/std,
   black share, sharpness; image size is not a feature). Balanced accuracy should DROP after preprocessing.
   Three versions: raw photo, after steps 3-6, and after step 7 too (training datasets only, as step 7 needs
   a known camera).
2. Biology probe: the same statistics predicting disc pallor (ODP) and media haze (MH) on RFMiD train + val.
   AUROC should stay ABOUT THE SAME; a big drop means a step is too strong.
The "before" number is the Phase 1 exit check (section 14).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.getcwd())
from data.preprocess import camera_color  # noqa: E402
from data.process import load_processed, probe_features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--per-dataset", type=int, default=400)
ap.add_argument("--out", default="outputs/camera_probe.json")
args = ap.parse_args()

meta = pd.read_parquet("data/meta.parquet")
proc = pd.read_parquet("data/process.parquet")
d = meta[meta.fov_ok & (meta.bucket != "excluded")][["image_id", "dataset", "source", "split", "labels"]].merge(proc, on="image_id")
raw_cols = sorted(c for c in proc.columns if c.startswith("raw_"))
proc_cols = [c.replace("raw_", "proc_") for c in raw_cols]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def camera_probe(table, cols, label="dataset"):
    X, y = table[cols].to_numpy(), table[label].to_numpy()
    pred = cross_val_predict(model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return float(balanced_accuracy_score(y, pred))


rng = np.random.default_rng(0)
sample = d.groupby("dataset", group_keys=False).apply(lambda g: g.sample(min(len(g), args.per_dataset), random_state=0))
n_cls = sample.dataset.nunique()
results = {"datasets": sample.dataset.value_counts().to_dict(), "chance": 1 / n_cls}
results["camera_probe_raw"] = camera_probe(sample, raw_cols)
results["camera_probe_after_steps_3_6"] = camera_probe(sample, proc_cols)
print(f"camera probe over {n_cls} datasets ({len(sample)} photos; chance {1 / n_cls:.3f}): "
      f"raw {results['camera_probe_raw']:.3f} -> after steps 3-6 {results['camera_probe_after_steps_3_6']:.3f}")

# Step 7 needs a known camera: training datasets only, each photo shifted with its own camera's stats.
stats = json.load(open("data/trace_preprocess.json"))
train_side = sample[sample.source.isin(stats["cameras"])].copy()
step7 = []
for pid, src, path in zip(train_side.image_id, train_side.source, train_side.processed_path):
    rgb, mask = load_processed(os.path.join("processed", path))
    c = stats["cameras"][src]
    shifted = camera_color(rgb, mask, c["mean"], c["std"], stats["pool"]["mean"], stats["pool"]["std"], stats["params"]["colour_alpha"])
    step7.append(probe_features(shifted, "s7_"))
s7 = pd.DataFrame(step7, index=train_side.index)
train_side = pd.concat([train_side, s7], axis=1)
s7_cols = [c.replace("raw_", "s7_") for c in raw_cols]
k = train_side.dataset.nunique()
for name, cols in [("raw", raw_cols), ("after_steps_3_6", proc_cols), ("after_step_7", s7_cols)]:
    results[f"train_datasets_{name}"] = camera_probe(train_side, cols)
print(f"training datasets only ({k}; chance {1 / k:.3f}): raw {results['train_datasets_raw']:.3f} -> steps 3-6 "
      f"{results['train_datasets_after_steps_3_6']:.3f} -> step 7 {results['train_datasets_after_step_7']:.3f}")

# Biology probe on RFMiD train + val.
bio = d[(d.dataset == "rfmid") & d.split.isin(["train", "val"])]
for lab in ["ODP", "MH"]:
    y = np.array([lab in list(x) for x in bio.labels])
    for name, cols in [("raw", raw_cols), ("after_steps_3_6", proc_cols)]:
        p = cross_val_predict(model(), bio[cols].to_numpy(), y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                              method="predict_proba")[:, 1]
        results[f"bio_{lab}_{name}"] = float(roc_auc_score(y, p))
    print(f"biology probe {lab} ({y.sum()} positives of {len(y)}): AUROC raw {results[f'bio_{lab}_raw']:.3f} -> "
          f"after steps 3-6 {results[f'bio_{lab}_after_steps_3_6']:.3f}")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(results, open(args.out, "w"), indent=1)
print("saved", args.out)
