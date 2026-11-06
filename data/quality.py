"""Step 5: good / usable / reject for every photo (CLAUDE.md section 6 step 5, section 19.9).

Run from the repo root, after data.process:  python -m data.quality
Thresholds are not hand-picked: a depth-3 decision tree (a handful of readable thresholds) is fitted to EyeQ's
human grades on EyePACS photos and checked on held-out patients. Where EyeQ graded a photo, its human grade is
kept (q_source = "eyeq").

Grades are applied to EyePACS ONLY (q_source "tree" there; "ungraded" elsewhere). Section 19.9 said to apply
them everywhere, but on 2026-11-06 the tree called 33% of RFMiD, 38% of RFMiD 2.0 and 61% of Paraguay
"reject", and the contact sheet showed many of those were readable, some with clear lesions or media haze,
which is itself an RFMiD disease label. Its main threshold (Laplacian sharpness) depends on each camera's
resolution and compression. q_pred is still stored for every photo, for analysis only.
Writes data/quality.parquet and outputs/quality_tree.txt.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

GRADED_DATASETS = ("eyepacs",)  # where the tree is in-domain
FEATURES = ["q_sharpness", "q_detail", "q_L_mean", "q_L_std", "q_dark_frac", "q_bright_frac", "q_centre_rim"]
GRADES = ["good", "usable", "reject"]


def make_tree(seed: int = 0) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, class_weight="balanced", random_state=seed)


def evaluate(graded: pd.DataFrame, seed: int = 0) -> dict:
    """Fit on 80% of patients, score on the other 20%. graded needs FEATURES, eyeq_quality, patient."""
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(split.split(graded, groups=graded.patient))
    tree = make_tree(seed).fit(graded.iloc[tr][FEATURES], graded.iloc[tr].eyeq_quality)
    pred = tree.predict(graded.iloc[te][FEATURES])
    true = graded.iloc[te].eyeq_quality
    return {
        "n_test": len(te),
        "accuracy": float((pred == true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "confusion": pd.DataFrame(confusion_matrix(true, pred, labels=GRADES), index=GRADES, columns=GRADES),
        # A good photo called reject loses data; a reject called good lets a bad photo into the pool.
        "reject_recall": float((pred[true.values == "reject"] == "reject").mean()),
    }


def grade_all(processed: pd.DataFrame, graded: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, DecisionTreeClassifier]:
    """processed: one row per photo with FEATURES, fov_ok, dataset and eyeq_quality (None if not graded)."""
    tree = make_tree(seed).fit(graded[FEATURES], graded.eyeq_quality)
    out = processed[["image_id"]].copy()
    ok = processed.fov_ok.values
    out["q_pred"] = None
    out.loc[ok, "q_pred"] = tree.predict(processed.loc[ok, FEATURES])
    human = processed.eyeq_quality.notna().values
    in_domain = processed.dataset.isin(GRADED_DATASETS).values
    out["q_grade"] = np.where(human, processed.eyeq_quality, np.where(in_domain, out.q_pred, "ungraded"))
    out["q_source"] = np.where(human, "eyeq", np.where(in_domain, "tree", "ungraded"))
    out.loc[~ok, ["q_grade", "q_source"]] = ["reject", "fov_fail"]
    return out, tree


if __name__ == "__main__":
    meta = pd.read_parquet("data/raw_meta.parquet")[["image_id", "dataset", "patient", "eyeq_quality"]]
    proc = pd.read_parquet("data/process.parquet").merge(meta, on="image_id")
    graded = proc[proc.fov_ok & proc.eyeq_quality.notna()]
    print(f"EyeQ-graded photos with features: {len(graded)} ({graded.eyeq_quality.value_counts().to_dict()})")
    for seed in range(3):
        r = evaluate(graded, seed)
        print(f"held-out patients, seed {seed}: n {r['n_test']}, accuracy {r['accuracy']:.3f}, "
              f"balanced {r['balanced_accuracy']:.3f}, reject recall {r['reject_recall']:.3f}")
    print("confusion (rows = EyeQ, columns = tree), seed 2:\n" + r["confusion"].to_string())
    out, tree = grade_all(proc, graded)
    out.to_parquet("data/quality.parquet")
    with open("outputs/quality_tree.txt", "w") as f:
        f.write(export_text(tree, feature_names=FEATURES, decimals=3))
    print("\nthe fitted thresholds:\n" + export_text(tree, feature_names=FEATURES, decimals=3))
    print("grades per dataset (used):\n" + out.merge(meta, on="image_id").groupby(["dataset", "q_grade"]).size().unstack(fill_value=0).to_string())
    print("tree predictions per dataset (analysis only):\n"
          + out.merge(meta, on="image_id").groupby(["dataset", "q_pred"]).size().unstack(fill_value=0).to_string())
