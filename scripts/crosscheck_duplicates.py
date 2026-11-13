"""Cross-check our deduplication against Approach 3's duplicate list (trace-lsnm-, data/manifests/duplicate_pairs.csv).

Run from the repo root:  python -m scripts.crosscheck_duplicates <path to duplicate_pairs.csv>
For each of their pairs: our photo ids, our match score (data/dup_pairs.parquet; NaN = the pair was never shortlisted),
whether we put both photos in one same-eye group (dup_group), and where each side ended up (bucket). A pair is a
LEAK if one side is on the training side (pretraining pool, head training, val-A / val-B) and the other is a test
photo, and neither was dropped. Writes outputs/crosscheck/duplicates.csv and prints a summary.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd

TRAIN_SIDE = {"train", "val_a", "val_b", "pool"}
TEST_SIDE = {"test", "ldo_test"}


def to_our_id(rec: str) -> str | None:
    """Their 'dataset:split:name' -> our image_id; None if we have no such dataset."""
    ds, split, name = rec.split(":")
    num = re.sub(r"\D", "", name)
    if ds == "rfmid":
        return f"rfmid_{ {'training': 'train', 'validation': 'val', 'testing': 'test'}[split]}_{int(num)}"
    if ds == "idrid_grading":
        return f"idrid_{ {'training': 'train', 'testing': 'test'}[split]}_{int(num):03d}"
    if ds == "idrid_segmentation":
        return f"idrid_seg_{int(num):02d}"
    return None


def side(row) -> str:
    if row.bucket == "excluded":
        return "dropped"
    if row.in_pool or row.bucket in TRAIN_SIDE:
        return "train-side"
    return "test-side" if row.bucket in TEST_SIDE else row.bucket


def main(path: str):
    theirs = pd.read_csv(path)
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    pairs = pd.read_parquet("data/dup_pairs.parquet")
    score = {}
    for a, b, s in zip(pairs.image_a, pairs.image_b, pairs.score):
        score[(a, b)] = score[(b, a)] = max(s, score.get((a, b), -1))
    rows = []
    for _, r in theirs.iterrows():
        a, b = to_our_id(r.left_record_id), to_our_id(r.right_record_id)
        found = a in meta.index and b in meta.index
        ra, rb = (meta.loc[a], meta.loc[b]) if found else (None, None)
        sa, sb = (side(ra), side(rb)) if found else ("?", "?")
        same_group = bool(found and pd.notna(ra.dup_group) and ra.dup_group == rb.dup_group)
        rows.append({"theirs_a": r.left_record_id, "theirs_b": r.right_record_id, "relation": r.relation,
                     "their_pixel_corr": r.pixel_correlation, "ours_a": a, "ours_b": b, "found": found,
                     "our_score": score.get((a, b), np.nan), "same_eye_group": same_group,
                     "bucket_a": ra.bucket if found else None, "bucket_b": rb.bucket if found else None,
                     "side_a": sa, "side_b": sb,
                     "leak": {sa, sb} == {"train-side", "test-side"}})
    out = pd.DataFrame(rows)
    os.makedirs("outputs/crosscheck", exist_ok=True)
    out.to_csv("outputs/crosscheck/duplicates.csv", index=False)

    print(f"{len(out)} pairs, {int(out.found.sum())} mapped to our photos")
    for rel, g in out.groupby("relation"):
        print(f"\n{rel}: {len(g)} pairs")
        print(f"  our score >= 0.8: {int((g.our_score >= 0.8).sum())}, 0.6-0.8: {int(g.our_score.between(0.6, 0.8, inclusive='left').sum())}, "
              f"< 0.6: {int((g.our_score < 0.6).sum())}, never shortlisted: {int(g.our_score.isna().sum())}")
        print(f"  same-eye group in ours: {int(g.same_eye_group.sum())}")
        print("  where the two sides are:", (g.side_a + " / " + g.side_b).value_counts().to_dict())
        print(f"  LEAKS (train-side photo paired with a kept test photo): {int(g.leak.sum())}")
    if out.leak.any():
        print("\nLeaking pairs:\n" + out[out.leak][["ours_a", "ours_b", "relation", "our_score", "bucket_a", "bucket_b"]].to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
