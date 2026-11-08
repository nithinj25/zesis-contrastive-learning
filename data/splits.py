"""Write data/meta.parquet: one row per photo, with the bucket every later step uses (section 5.3).

Run from the repo root after ingest, dedupe, process and quality:  python -m data.splits
Also writes data/trace_preprocess.json (per-camera colour stats for step 7, section 19.1).

Buckets:
  excluded   dropped by dedupe, or the retina was not found (exclude_reason says which)
  test       a held-out split (RFMiD / RFMiD 2.0 test, IDRiD, JSIEC, Paraguay, ...)
  ldo_test   a train or val photo of a leave-disease-out target (section 9.6): all of a target's photos are its test set
  val_a      tuning (section 10)        val_b   conformal threshold only
  train      RFMiD / RFMiD 2.0 train: head training, and the pool unless graded reject
  pool       a pool-only photo (APTOS, EyePACS) chosen for pretraining
  spare      a pool-only photo left out to keep the pool at its target size, or graded reject
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import yaml

from data.ontology import Ontology, load
from data.preprocess import pooled_stats

CONFIG = "configs/data.yaml"
VAL_SPLIT = "data/val_split.csv"  # committed: the val-A / val-B membership, frozen


def split_val(rows: pd.DataFrame, fraction_b: float, seed: int) -> pd.Series:
    """Assign val photos to val_a / val_b, keeping each dup_group whole and balancing camera and labels.

    Greedy multi-label stratification: groups holding the rarest keys are placed first, each into the half
    that is furthest below its target for that key. Keys: the source, and every finding and disease.
    """
    rng = np.random.default_rng(seed)
    keys = {i: [f"src:{s}"] + [f"lab:{x}" for x in list(f) + list(d)]
            for i, s, f, d in zip(rows.index, rows.source, rows.findings, rows.diseases)}
    groups: dict[int, list] = {}
    for i, g in zip(rows.index, rows.dup_group):
        groups.setdefault(g, []).append(i)
    freq = Counter(k for ks in keys.values() for k in ks)
    order = list(groups)
    rng.shuffle(order)
    order.sort(key=lambda g: min(freq[k] for i in groups[g] for k in keys[i]))  # stable: random within ties
    want = {"val_b": fraction_b, "val_a": 1 - fraction_b}
    count = {h: Counter() for h in want}
    size = {h: 0 for h in want}
    out = {}
    for g in order:
        members = groups[g]
        gk = Counter(k for i in members for k in keys[i])
        rarest = min(gk, key=lambda k: freq[k])

        def deficit(h):  # how far below its share this half is, for the rarest key, then overall
            return (count[h][rarest] - want[h] * freq[rarest], size[h] - want[h] * len(rows))

        half = min(want, key=deficit)
        for i in members:
            out[i] = half
        count[half].update(gk)
        size[half] += len(members)
    return pd.Series(out).reindex(rows.index)


def choose_pool(meta: pd.DataFrame, cfg: dict) -> pd.Series:
    """True for pool-only photos that enter the pool: every eligible non-EyePACS photo, then EyePACS in priority order."""
    rng = np.random.default_rng(cfg["seed"])
    eligible = (meta.bucket == "pool_candidate") & meta.q_grade.isin(cfg["pool_grades"])
    chosen = eligible & (meta.dataset != "eyepacs")
    already = int(((meta.bucket == "train") & meta.q_grade.isin(cfg["pool_grades"])).sum() + chosen.sum())
    room = max(cfg["pool_target"] - already, 0)
    picked = []
    step = meta.q_source.astype(str) + ":" + meta.q_grade.astype(str)
    for key in cfg["eyepacs_priority"]:
        ids = meta.index[eligible & (meta.dataset == "eyepacs") & (step == key)].to_numpy()
        rng.shuffle(ids)
        picked.extend(ids[: room - len(picked)])
        if len(picked) >= room:
            break
    chosen.loc[picked] = True
    return chosen


def mine_sources(meta: pd.DataFrame, merge: dict | None) -> pd.Series:
    """The source ID the mining loss uses (section 7.4). Normally the same as `source` (which also picks step 7's
    colour stats), except groups the pseudo-domain check (scripts/pseudo_domains.py, section 5.5) could not tell
    apart from a known camera: those share that camera's mining ID, so their pairs never count as cross-source.
    merge: {"dataset:size_group": source}."""
    out = meta.source.astype(str).copy()
    for key, src in (merge or {}).items():
        ds, size = key.split(":")
        out[(meta.dataset == ds) & (meta.size_group == size)] = src
    return out


def build(raw: pd.DataFrame, dedupe: pd.DataFrame, proc: pd.DataFrame, quality: pd.DataFrame,
          onto: Ontology, cfg: dict) -> pd.DataFrame:
    meta = raw.merge(dedupe[["image_id", "dup_group", "dup_drop", "dup_reason"]], on="image_id", how="left")
    cols = ["image_id", "fov_ok", "fail_reason", "processed_path", "fov_fit", "fov_ratio"]
    meta = meta.merge(proc[[c for c in cols if c in proc.columns]], on="image_id", how="left")
    meta = meta.merge(quality[["image_id", "q_grade", "q_source"]], on="image_id", how="left")
    meta["dup_drop"] = meta.dup_drop.fillna(False).astype(bool)
    meta["fov_ok"] = meta.fov_ok.fillna(False).astype(bool)
    mapped = [onto.map(d, labs, ok, risk) for d, labs, ok, risk in zip(meta.dataset, meta.labels, meta.label_ok, meta.disease_risk)]
    meta["findings"] = [f for f, _ in mapped]
    meta["diseases"] = [d for _, d in mapped]
    # Same eye, same dataset, different labels (real: 45 Paraguay same-eye pairs differ by DR grade). The kept
    # copy's label is not more trustworthy than the dropped one's, so evaluation can exclude these eyes.
    key = meta.dataset + ":" + meta.dup_group.astype(str)
    distinct = pd.Series([tuple(sorted(x)) for x in meta.labels], index=meta.index).groupby(key).nunique()
    meta["label_conflict"] = key.map(distinct).fillna(1).gt(1).values
    targets = set(onto.leave_out)
    meta["ldo_disease"] = [",".join(sorted(set(d) & targets)) for d in meta.diseases]
    meta = meta.set_index("image_id", drop=False)

    bucket = pd.Series("", index=meta.index)
    reason = pd.Series("", index=meta.index)
    reason[meta.dup_drop] = "dedupe:" + meta.dup_reason[meta.dup_drop]
    reason[(reason == "") & ~meta.fov_ok] = "fov_fail"
    # A photo of the same eye as a leave-disease-out photo, outside the test set, would leak that eye.
    ldo_groups = set(meta.dup_group[(meta.ldo_disease != "") & (reason == "")])
    same_eye = meta.dup_group.isin(ldo_groups) & (meta.ldo_disease == "") & (meta.split != "test")
    reason[(reason == "") & same_eye] = "same_eye_as_ldo"
    bucket[reason != ""] = "excluded"
    free = bucket == ""
    bucket[free & (meta.split == "test")] = "test"
    free = bucket == ""
    bucket[free & (meta.ldo_disease != "")] = "ldo_test"
    free = bucket == ""
    bucket[free & (meta.split == "train")] = "train"
    bucket[free & (meta.split == "pool")] = "pool_candidate"
    val = free & (meta.split == "val")
    meta["bucket"] = bucket
    meta.loc[val, "bucket"] = split_val(meta[val], cfg["val_b_fraction"], cfg["seed"])
    frozen = cfg.get("frozen_val")
    if frozen is not None and len(frozen):
        # Once anything has been tuned on val-A, its membership must not move (a later label fix would
        # otherwise reshuffle the halves). Photos in the frozen file keep their half; new ones are assigned fresh.
        keep = val & meta.image_id.isin(frozen.index)
        meta.loc[keep, "bucket"] = frozen.reindex(meta.image_id[keep]).values

    in_pool = choose_pool(meta, cfg)
    meta.loc[meta.bucket == "pool_candidate", "bucket"] = np.where(in_pool[meta.bucket == "pool_candidate"], "pool", "spare")
    meta["in_pool"] = (meta.bucket == "pool") | ((meta.bucket == "train") & meta.q_grade.isin(cfg["pool_grades"]))
    meta["in_heads"] = (meta.bucket == "train") & meta.label_ok
    meta["exclude_reason"] = reason
    meta["mine_source"] = mine_sources(meta, cfg.get("mine_merge"))
    return meta.reset_index(drop=True)


def preprocess_stats(meta: pd.DataFrame, proc: pd.DataFrame, cfg: dict) -> dict:
    """Pool and per-camera LAB stats (after step 6) for step 7. Only training sources: no target statistics
    are taken from held-out data (section 19.1, mentor question 5)."""
    p = proc.set_index("image_id")
    cols_m, cols_s = [f"lab_{c}_mean" for c in "Lab"], [f"lab_{c}_std" for c in "Lab"]

    def stats(ids):
        q = p.loc[ids]
        m, s = pooled_stats(q[cols_m].to_numpy(), q[cols_s].to_numpy(), q.lab_n.to_numpy())
        return {"mean": [round(float(v), 3) for v in m], "std": [round(float(v), 3) for v in s], "n_photos": len(ids)}

    pool_ids = meta.image_id[meta.in_pool]
    train_side = meta[meta.bucket.isin(["train", "val_a", "val_b", "pool", "spare"]) & meta.fov_ok]
    return {
        "pool": stats(pool_ids),
        "cameras": {src: stats(g.image_id) for src, g in train_side.groupby("source")},
        "params": {"colour_alpha": cfg["colour_alpha"], "clahe_clip": cfg["clahe_clip"],
                   "illumination_strength": cfg["illumination_strength"],
                   "illumination_sigma_fraction": cfg["illumination_sigma_fraction"], "size": 512,
                   "lab": "OpenCV 8-bit LAB (L 0-255, a and b offset by 128)"},
    }


def summary(meta: pd.DataFrame) -> str:
    lines = ["photos per bucket:\n" + meta.groupby(["bucket", "dataset"]).size().unstack(fill_value=0).to_string()]
    lines.append(f"\npool: {int(meta.in_pool.sum())} photos from " + str(meta[meta.in_pool].dataset.value_counts().to_dict()))
    lines.append(f"head training: {int(meta.in_heads.sum())}")
    lines.append("excluded: " + str(meta[meta.bucket == "excluded"].exclude_reason.str.split(":").str[0].value_counts().to_dict()))
    heads = meta[meta.in_heads]
    counts = Counter(f for fs in heads.findings for f in fs)
    rule = lambda n: "detector" if n >= 20 else ("prototype" if n >= 5 else "dropped")  # noqa: E731
    lines.append("\nfinding positives in head training (section 8.3 rule):")
    lines += [f"  {f:18s} {n:4d}  {rule(n)}" for f, n in counts.most_common()]
    ldo = meta[meta.ldo_disease != ""]
    lines.append("\nleave-disease-out photos by bucket:\n" + ldo.groupby(["ldo_disease", "bucket"]).size().unstack(fill_value=0).to_string())
    return "\n".join(lines)


if __name__ == "__main__":
    cfg = yaml.safe_load(open(CONFIG))
    if os.path.exists(VAL_SPLIT):
        cfg["frozen_val"] = pd.read_csv(VAL_SPLIT).set_index("image_id").bucket
        print(f"using the frozen val split in {VAL_SPLIT} ({len(cfg['frozen_val'])} photos)")
    raw = pd.read_parquet("data/raw_meta.parquet")
    meta = build(raw, pd.read_parquet("data/dedupe.parquet"), pd.read_parquet("data/process.parquet"),
                 pd.read_parquet("data/quality.parquet"), Ontology(load()), cfg)
    meta.to_parquet("data/meta.parquet")
    if not os.path.exists(VAL_SPLIT):
        meta[meta.bucket.isin(["val_a", "val_b"])][["image_id", "bucket"]].to_csv(VAL_SPLIT, index=False)
        print(f"val split frozen to {VAL_SPLIT}; delete it only before any tuning has used val-A")
    stats = preprocess_stats(meta, pd.read_parquet("data/process.parquet"), cfg)
    with open("data/trace_preprocess.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(summary(meta))
    print("\ncamera colour stats written for:", list(stats["cameras"]))
