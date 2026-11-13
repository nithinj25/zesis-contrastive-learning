"""UMAP of the frozen features (a required deliverable): does the map group eyes by disease and mix the cameras?

Run from the repo root:  python -m scripts.umap_plots [--encoders imagenet_r50 trace_v2]
Photos: every labelled RFMiD and RFMiD 2.0 photo (train, val, test and the leave-disease-out photos; 3,835).
One UMAP per encoder (cosine, 30 neighbours, seed 0), shown three ways: common diseases, rare diseases, and the three
camera models of RFMiD (RFMiD 2.0 TRC-NW300 photos share that camera id). A 2-D map can mislead, so each
panel carries a number measured in the full 2048-d space: the share of each photo's 20 nearest neighbours with
the same disease (single-diagnosis photos) or the same camera (photos of those three cameras), against chance.
Writes outputs/figures/umap_<enc1>_vs_<enc2>.png and outputs/figures/umap_stats.json.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.probes import load_features

BUCKETS = ("train", "val_a", "val_b", "test", "ldo_test")
COMMON = ["DR", "myopia", "ARMD"]
RARE = ["CRVO", "AION", "RP"]
CAMERAS = {"topcon_3d_oct2000": "Topcon 3D OCT-2000", "kowa_vx10a": "Kowa VX-10α", "topcon_trc_nw300": "Topcon TRC-NW300"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1-3: the only three that pass all-pairs (scatter)
CONTEXT, INK, MUTED = "#d9d8d3", "#0b0b0b", "#52514e"
NAMES = {"imagenet_r50": "ImageNet ResNet-50 (before)", "trace_v2": "TRACE (ours)", "moco_plain_e15": "Plain MoCo v2"}
LABEL = {"DR": "Diabetic retinopathy", "myopia": "Myopia", "ARMD": "ARMD", "CRVO": "CRVO", "AION": "AION",
         "RP": "Retinitis pigmentosa"}


def same_share(X: np.ndarray, group: np.ndarray, k: int = 20, block: np.ndarray | None = None) -> float:
    """Mean share of each row's k nearest neighbours (cosine; X unit rows) with the same group.
    block: rows with equal block ids are never neighbours (same-eye copies)."""
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    if block is not None:
        S[block[:, None] == block[None, :]] = -np.inf
    idx = np.argpartition(-S, k, axis=1)[:, :k]
    return float((group[idx] == group[:, None]).mean())


def chance(group: np.ndarray) -> float:
    """Expected share if neighbours were random: sum of squared group shares."""
    p = pd.Series(group).value_counts(normalize=True).values
    return float((p ** 2).sum())


def main():
    import umap

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs=2, default=["imagenet_r50", "trace_v2"])
    args = ap.parse_args()

    meta = pd.read_parquet("data/meta.parquet")
    rows = meta[meta.dataset.isin(["rfmid", "rfmid2"]) & meta.label_ok & meta.fov_ok & meta.bucket.isin(BUCKETS)]
    rows = rows[["image_id", "source", "diseases", "dup_group"]].reset_index(drop=True)
    rows["disease"] = [x[0] if len(x) == 1 else ("none" if len(x) == 0 else "multiple") for x in rows.diseases]
    block = pd.factorize(rows.dup_group.fillna(rows.image_id))[0]

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 12.5), dpi=150)
    stats = {}
    for r, enc in enumerate(args.encoders):
        f = load_features(enc, BUCKETS)
        m = rows.merge(f, on="image_id", how="inner")
        assert len(m) == len(rows), f"{enc}: {len(rows) - len(m)} photos without features"
        X = f.attrs["X"][m.row.values]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        xy = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine", random_state=0).fit_transform(X)

        single = m.disease.isin(["normal", *COMMON, *RARE, "BRVO", "CSR"]).values
        cam = m.source.isin(list(CAMERAS)).values
        s = {"disease_same": same_share(X[single], m.disease.values[single], block=block[single]),
             "disease_chance": chance(m.disease.values[single]),
             "camera_same": same_share(X[cam], m.source.values[cam], block=block[cam]),
             "camera_chance": chance(m.source.values[cam]),
             "n_photos": len(m), "n_single_diagnosis": int(single.sum()), "n_rfmid_cameras": int(cam.sum())}
        stats[enc] = s

        panels = [
            ("Common diseases", m.disease.values, COMMON, f"same disease among 20 nearest: {s['disease_same']:.0%} (chance {s['disease_chance']:.0%})"),
            ("Rare diseases", m.disease.values, RARE, "these eyes were left out of encoder training"),
            ("Camera (3 camera models)", m.source.values, list(CAMERAS), f"same camera among 20 nearest: {s['camera_same']:.0%} (chance {s['camera_chance']:.0%})"),
        ]
        for c, (title, groups, keys, note) in enumerate(panels):
            ax = axes[r, c]
            ax.scatter(xy[:, 0], xy[:, 1], s=4, c=CONTEXT, linewidths=0, rasterized=True)
            for key, col in zip(keys, SERIES):
                sel = groups == key
                big = c == 1  # rare groups are small: bigger markers with a surface ring so they stay visible
                ax.scatter(xy[sel, 0], xy[sel, 1], s=34 if big else 7, c=col, linewidths=0.8 if big else 0,
                           edgecolors="white", rasterized=True,
                           label=f"{CAMERAS.get(key, LABEL.get(key, key))} ({sel.sum()})")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_title(f"{NAMES.get(enc, enc)}: {title}", fontsize=12, color=INK, loc="left")
            # legend and note sit below the map, never on the points
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.01), ncol=2, fontsize=9, frameon=False,
                      markerscale=1.6 if c != 1 else 1.0, handletextpad=0.3, columnspacing=1.0, labelcolor=INK)
            ax.text(0, -0.17, f"grey = all other photos · {note}", transform=ax.transAxes, fontsize=9.5,
                    color=MUTED, va="top")
    fig.suptitle("UMAP of frozen 2048-d features, 3,835 labelled RFMiD + RFMiD 2.0 photos (same map per row, coloured three ways)",
                 fontsize=13, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=4.5)
    os.makedirs("outputs/figures", exist_ok=True)
    out = f"outputs/figures/umap_{args.encoders[0]}_vs_{args.encoders[1]}.png"
    fig.savefig(out, facecolor="#fcfcfb")
    json.dump(stats, open("outputs/figures/umap_stats.json", "w"), indent=1)
    print(out)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
