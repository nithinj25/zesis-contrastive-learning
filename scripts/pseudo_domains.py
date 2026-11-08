"""Pseudo-domain check (CLAUDE.md section 5.5): can the source IDs that mining relies on be trusted?

    python scripts/pseudo_domains.py            # writes outputs/pseudo_domains.json

Mining only attracts pairs from DIFFERENT sources. If one physical camera carries two source IDs, its photos form
"cross-source" pairs that share every camera cue, and attraction would pull them together for the camera, not the
retina: the dangerous error. Merging two real cameras into one ID only loses some attraction.

Part 1 (section 5.5 validation): k-means (k = 3..8) on size-free acquisition statistics of RFMiD, where the true
camera is known from the photo size. Completeness of a camera = share of its photos in its dominant cluster; it must
be >= 0.9 for clustering to be safe on mixed sets (EyePACS).
Part 2 (clash test): APTOS has photo sizes equal to known cameras (4288x2848 Kowa VX-10a, 2144x1424 Topcon 3D
OCT-2000, 2048x1536 Topcon TRC-NW300, 1504x1000 RFMiD 2.0's group). For each, a classifier tries to tell the APTOS
subset from that camera (5-fold CV AUROC, same-eye copies kept in one fold). Two feature sets: acquisition
statistics, and frozen plain-MoCo encoder features (the camera probe on them is 0.97). Read against references:
same camera / same dataset, same camera model / other dataset (IDRiD vs RFMiD Kowa), and different cameras.
AUROC near the "same camera" references -> treat as one camera (merge); near 1 -> distinct capture.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOUR = ([f"raw_{c}_{s}" for c in "RGB" for s in ("mean", "std")] + [f"lab_{c}_{s}" for c in "Lab" for s in ("mean", "std")])
QUALITY = ["raw_sharpness", "q_sharpness", "q_detail", "q_dark_frac", "q_bright_frac", "q_centre_rim"]
SHAPE = ["raw_black_frac", "fov_fit", "fov_ratio", "fov_radius_frac", "mask_frac"]  # framing of the retina in the frame
STATS = COLOUR + QUALITY + SHAPE
# the same photo statistics measured on the PROCESSED photo (after steps 3-6: crop, geometry, illumination), i.e. the
# camera cues that survive preprocessing and reach the encoder; black share left out (it is the mask's shape)
PROC = [f"proc_{c}_{s}" for c in "RGBLab" for s in ("mean", "std")] + ["proc_sharpness"]
CLASH = {"4288x2848": "kowa_vx10a", "2144x1424": "topcon_3d_oct2000", "2048x1536": "topcon_trc_nw300",
         "1504x1000": "rfmid2_1504x1000"}


def completeness(cam: np.ndarray, cluster: np.ndarray) -> dict:
    """Per camera: share of its photos in its most common cluster."""
    out = {}
    for c in np.unique(cam):
        _, n = np.unique(cluster[cam == c], return_counts=True)
        out[str(c)] = float(n.max() / n.sum())
    return out


def two_sample_auroc(Xa: np.ndarray, Xb: np.ndarray, ga: np.ndarray, gb: np.ndarray, C: float = 1.0,
                     seed: int = 0) -> float:
    """CV AUROC of a classifier telling group a from group b. ~0.5: indistinguishable; ~1: easily told apart.
    ga / gb: group ids (same-eye copies share one) so a copy never sits in train and test at once."""
    X = np.vstack([Xa, Xb])
    y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    g = np.r_[ga, gb]
    n_splits = int(min(5, len(Xa), len(Xb)))
    score = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, g):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        score[te] = clf.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, score))


def load_encoder_features(name: str) -> pd.DataFrame:
    parts = []
    for split in ("pool", "train", "val_a", "val_b", "test"):
        ix = pd.read_parquet(f"features/{name}/{split}_index.parquet")
        x = np.load(f"features/{name}/{split}_global.npy").astype(np.float32)
        parts.append(pd.DataFrame(x, index=ix.image_id))
    f = pd.concat(parts)
    return f[~f.index.duplicated()]


if __name__ == "__main__":
    meta = pd.read_parquet("data/meta.parquet")
    proc = pd.read_parquet("data/process.parquet")
    extra = [c for c in STATS + PROC if c not in meta.columns]  # fov_fit / fov_ratio are already in meta
    d = meta[meta.fov_ok & (meta.bucket != "excluded")].merge(proc[["image_id"] + extra], on="image_id")
    d["group"] = d.dup_group.fillna(d.image_id).astype(str)
    res: dict = {}

    # ---- Part 1: does size-free clustering keep each RFMiD camera together?
    r = d[d.dataset == "rfmid"]
    print(f"Part 1: RFMiD, {len(r)} photos, cameras {r.source.value_counts().to_dict()}")
    res["part1"] = {}
    for fname, cols in [("colour+quality+shape", STATS), ("colour+quality", COLOUR + QUALITY)]:
        X = StandardScaler().fit_transform(r[cols].to_numpy())
        for k in range(3, 9):
            lab = KMeans(k, n_init=10, random_state=0).fit_predict(X)
            comp = completeness(r.source.to_numpy(), lab)
            res["part1"][f"{fname} k={k}"] = comp
            ok = "OK" if min(comp.values()) >= 0.9 else "FAIL"
            print(f"  {fname:22s} k={k}: " + ", ".join(f"{c} {v:.2f}" for c, v in comp.items()) + f"  [{ok}]")

    # ---- Part 2: APTOS photos at a known camera's size vs that camera
    enc = load_encoder_features("moco_plain")
    d = d[d.image_id.isin(enc.index)]

    KINDS = {"raw": STATS, "raw_noshape": COLOUR + QUALITY, "processed": PROC, "encoder": None}

    def feats(rows, kind):
        return enc.loc[rows.image_id].to_numpy() if kind == "encoder" else rows[KINDS[kind]].to_numpy()

    def sel(**kw):
        m = np.ones(len(d), bool)
        for k, v in kw.items():
            m &= d[k].isin(v if isinstance(v, list) else [v]).to_numpy()
        return d[m]

    tests = {}
    kowa = sel(source="kowa_vx10a", dataset="rfmid")
    tests["REF same camera+dataset: RFMiD Kowa train vs test"] = (sel(source="kowa_vx10a", dataset="rfmid", split="train"),
                                                                   sel(source="kowa_vx10a", dataset="rfmid", split="test"))
    tests["REF same camera+dataset: RFMiD OCT-2000 train vs test"] = (sel(source="topcon_3d_oct2000", dataset="rfmid", split="train"),
                                                                       sel(source="topcon_3d_oct2000", dataset="rfmid", split="test"))
    tests["REF same camera model, other dataset: IDRiD vs RFMiD Kowa"] = (sel(dataset="idrid"), kowa)
    tests["REF different cameras: RFMiD Kowa vs RFMiD TRC-NW300"] = (kowa, sel(source="topcon_trc_nw300", dataset="rfmid"))
    tests["REF different cameras: RFMiD Kowa vs RFMiD OCT-2000"] = (kowa, sel(source="topcon_3d_oct2000", dataset="rfmid"))
    tests["REF different datasets: APTOS 1050x1050 vs RFMiD TRC-NW300"] = (sel(dataset="aptos", size_group="1050x1050"),
                                                                           sel(source="topcon_trc_nw300"))
    for size, cam in CLASH.items():
        tests[f"CLASH APTOS {size} vs {cam}"] = (sel(dataset="aptos", size_group=size), sel(source=cam, dataset=["rfmid", "rfmid2"]))
    tests["SPLIT? RFMiD 2.0 512x512 vs TRC-NW300"] = (sel(source="rfmid2_512x512"), sel(source="topcon_trc_nw300"))
    tests["SPLIT? RFMiD 2.0 1504x1000 vs TRC-NW300"] = (sel(source="rfmid2_1504x1000"), sel(source="topcon_trc_nw300"))
    tests["SPLIT? RFMiD 2.0 512x512 vs 1504x1000"] = (sel(source="rfmid2_512x512"), sel(source="rfmid2_1504x1000"))
    tests["SPLIT? RFMiD vs RFMiD 2.0 within TRC-NW300"] = (sel(source="topcon_trc_nw300", dataset="rfmid"),
                                                          sel(source="topcon_trc_nw300", dataset="rfmid2"))

    print("\nPart 2: two-sample AUROC (0.5 = indistinguishable, 1 = easily told apart)")
    print(f"  {'test':62s} {'n_a':>5s} {'n_b':>5s} " + " ".join(f"{k:>11s}" for k in KINDS))
    res["part2"] = {}
    for name, (a, b) in tests.items():
        a = a[~a.group.isin(b.group)]  # a copy of the same photo on both sides would fake "indistinguishable"
        if len(a) < 5 or len(b) < 5:
            print(f"  {name:62s} too few photos ({len(a)}, {len(b)})")
            continue
        out = {"n_a": len(a), "n_b": len(b)}
        for kind in KINDS:
            C = 0.1 if kind == "encoder" else 1.0
            out[kind] = two_sample_auroc(feats(a, kind), feats(b, kind), a.group.to_numpy(), b.group.to_numpy(), C)
        res["part2"][name] = out
        print(f"  {name:62s} {len(a):5d} {len(b):5d} " + " ".join(f"{out[k]:11.3f}" for k in KINDS))

    os.makedirs("outputs", exist_ok=True)
    json.dump(res, open("outputs/pseudo_domains.json", "w"), indent=1)
    print("\nwrote outputs/pseudo_domains.json")
