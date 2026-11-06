"""Audit the dedupe output: leaks left, drops per dataset, rare labels, and a contact sheet.

Run from the repo root after data.dedupe:
    python scripts/dedupe_audit.py [sheet.jpg] [--previous old_dedupe.parquet old_pairs.parquet]
--previous checks that an earlier run's drops are still drops (a regression check after code changes).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from data.dedupe import MATCH, SAME_EYE, SPLIT_RANK  # noqa: E402
from data.preprocess import find_fov, standardise_geometry  # noqa: E402

RARE = ("AION", "CRVO", "RP", "CRAO", "Retinitis pigmentosa", "RAO")
WATCH = list(RARE) + ["ODE", "ODP", "MCA", "PRH", "CWS", "TV"]

ap = argparse.ArgumentParser()
ap.add_argument("sheet", nargs="?", default="outputs/dedupe_cross.jpg")
ap.add_argument("--previous", nargs=2, metavar=("DEDUPE", "PAIRS"))
args = ap.parse_args()

meta = pd.read_parquet("data/raw_meta.parquet").set_index("image_id")
res = pd.read_parquet("data/dedupe.parquet").set_index("image_id")
pairs = pd.read_parquet("data/dup_pairs.parquet")
ds, sp = meta.dataset, meta.split

# 1. After drops, does any kept image still match a kept image of another split rank?
kept = set(res.index[~res.dup_drop])
p = pairs[(pairs.score >= MATCH) & pairs.image_a.isin(kept) & pairs.image_b.isin(kept)]
across = p[[SPLIT_RANK[sp[a]] != SPLIT_RANK[sp[b]] for a, b in zip(p.image_a, p.image_b)]]
print("kept pairs still matching across split ranks:", len(across))
print("drops:", int(res.dup_drop.sum()), "of", len(res))
print("kept per split:", res[~res.dup_drop].groupby(["dataset", "split"]).size().to_dict())

# 2. Regression against an earlier run.
if args.previous:
    old = pd.read_parquet(args.previous[0]).set_index("image_id")
    lost = [i for i in old.index[old.dup_drop] if i in res.index and not res.dup_drop[i]]
    print(f"\nprevious drops: {int(old.dup_drop.sum())}, no longer dropped: {len(lost)} {lost[:10]}")
    old_pairs = pd.read_parquet(args.previous[1])
    now = pairs[pairs.score >= SAME_EYE]
    now_keys = set(zip(now.image_a, now.image_b)) | set(zip(now.image_b, now.image_a))
    strong_old = old_pairs[old_pairs.score >= SAME_EYE]
    missing = [(a, b) for a, b in zip(strong_old.image_a, strong_old.image_b) if (a, b) not in now_keys]
    pool_pool = [m for m in missing if sp[m[0]] == "pool" and sp[m[1]] == "pool"]
    print(f"previous >=0.8 pairs: {len(strong_old)}, missing now: {len(missing)}, "
          f"of which pool-pool (skipped on purpose): {len(pool_pool)}; others: {[m for m in missing if m not in pool_pool][:10]}")

# 3. Score bands and pairs between datasets.
bands = pd.cut(pairs.score, [-1, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], right=False)
print("\npairs checked:", len(pairs), "| score bands:", {str(k): v for k, v in bands.value_counts().sort_index().items()})
pairs["ds_pair"] = ["-".join(sorted([ds[a], ds[b]])) for a, b in zip(pairs.image_a, pairs.image_b)]
x = pairs[pairs.score >= MATCH]
print("\nmatches by dataset pair:")
print(pd.DataFrame({">=0.6": x.groupby("ds_pair").size(), ">=0.8": x[x.score >= SAME_EYE].groupby("ds_pair").size()})
      .fillna(0).astype(int).to_string())
print("\ndrops by dataset, split and reason:")
print(res[res.dup_drop].groupby(["dataset", "split", "dup_reason"]).size().to_string())

# 4. Rare and important labels: before and after.
rows = {}
for lab in WATCH:
    has = meta.index[[lab in list(v) for v in meta.labels]]
    for d in sorted(set(ds[has])):
        h = has[ds[has] == d]
        drop = res.loc[h, "dup_drop"]
        rows[f"{lab} [{d}]"] = dict(all=len(h), dropped=int(drop.sum()), kept=int((~drop).sum()),
                                    unique_eyes=res.loc[h][~drop].dup_group.nunique(),
                                    kept_test=int(((sp[h] == "test") & ~drop).sum()))
print("\nlabels before/after:\n" + pd.DataFrame(rows).T.to_string())
rare_drops = [(i, [lab for lab in meta.labels[i] if lab in RARE], res.dup_reason[i])
              for i in res.index[res.dup_drop] if any(lab in RARE for lab in meta.labels[i])]
print("\nrare-label drops:", rare_drops or "none")


# 5. Contact sheet: random pairs between different datasets, strong and borderline.
def thumb(image_id):
    img = cv2.imread(os.path.join("raw", meta.path[image_id]))[..., ::-1].copy()
    std, _ = standardise_geometry(img, find_fov(img))
    return cv2.resize(std, (170, 170), interpolation=cv2.INTER_AREA)


rng = np.random.default_rng(1)
two = pairs.ds_pair.str.split("-")
xb = pairs[(pairs.score >= MATCH) & (two.str[0] != two.str[1])]
cells_rows = []
for lo, hi, n in [(0.8, 1.01, 8), (0.6, 0.8, 12)]:
    band = xb[(xb.score >= lo) & (xb.score < hi)]
    if not len(band):
        continue
    pick = band.iloc[rng.choice(len(band), min(n, len(band)), replace=False)]
    print(f"\nsheet {lo}-{min(hi, 1.0)}:", list(zip(pick.ds_pair, pick.score.round(2))))
    cells = [np.hstack([thumb(r.image_a), thumb(r.image_b), np.full((170, 8, 3), 255, np.uint8)]) for r in pick.itertuples()]
    for k in range(0, len(cells), 4):
        row = cells[k:k + 4]
        while len(row) < 4:
            row.append(np.zeros_like(cells[0]))
        cells_rows.append(np.hstack(row))
if cells_rows:
    os.makedirs(os.path.dirname(args.sheet) or ".", exist_ok=True)
    cv2.imwrite(args.sheet, cv2.cvtColor(np.vstack(cells_rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("sheet written:", args.sheet)
