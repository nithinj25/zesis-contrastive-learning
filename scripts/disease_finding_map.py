"""Disease x finding map: what the finding detectors see in each disease's photos (slides, model card).

    python -m scripts.disease_finding_map --name trace_v2

Cell = mean calibrated detector output over the disease's photos. Common diseases: head-training photos, each
predicted out-of-fold (a model that never saw it). Rare diseases (AION, CRVO, RP): their held-out photos, which no
detector trained on. Descriptive only, nothing is fitted here. Columns are grouped by where the finding sits.
Writes outputs/figures/disease_finding_map.png.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import LinearSegmentedColormap

from eval.leave_disease_out import findings_q

# sequential blue ramp, steps 100 -> 700 (dataviz reference palette); the lightest step means "near zero"
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, MUTED, SURFACE = "#1f1f1e", "#6b6a66", "#ffffff"
LOCATION_ORDER = ["disc", "anywhere", "posterior_pole", "vessels", "macula", "background", "periphery", "whole_image"]
NICE = {"hypertensive_retinopathy": "Hypertensive ret.", "normal": "Normal", "myopia": "Myopia"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    onto = yaml.safe_load(open("data/ontology.yaml", encoding="utf-8"))
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    oof = pd.read_parquet(os.path.join("features", args.name, "train_findings_oof.parquet")).set_index("image_id")
    loc = {f: v["location"] for f, v in onto["findings"].items()}
    F = sorted(oof.columns, key=lambda f: (LOCATION_ORDER.index(loc[f]) if loc[f] in LOCATION_ORDER else 99, f))
    head = meta.loc[oof.index]
    rows, labels = [], []
    counts = pd.Series([d for x in head.diseases for d in x]).value_counts()
    for d in [d for d, n in counts.items() if n >= 5]:
        has = head.diseases.map(lambda x, d=d: d in list(x)).values
        rows.append(oof[F].values[has].mean(0))
        labels.append(f"{NICE.get(d, d)}  (n {has.sum()})")
    for d in onto["leave_disease_out"]:
        ids = meta.index[(meta.ldo_disease == d) & meta.label_ok & (meta.bucket != "excluded")]
        rows.append(np.nanmean(findings_q(args.name, ids, F), 0))
        labels.append(f"{d}  (n {len(ids)}, held out)")
    M = np.array(rows)

    cmap = LinearSegmentedColormap.from_list("blue", BLUE)
    fig, ax = plt.subplots(figsize=(15, 6.2), facecolor=SURFACE)
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=0.5, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        if v >= 0.05:  # selective labels: only cells with a real signal
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v >= 0.3 else INK)
    for x in range(1, len(F)):
        if loc[F[x]] != loc[F[x - 1]]:
            ax.axvline(x - 0.5, color=SURFACE, lw=3)
    n_common = len(M) - len(onto["leave_disease_out"])
    ax.axhline(n_common - 0.5, color=SURFACE, lw=6)
    ax.set_xticks(range(len(F)), [f.replace("_", " ") for f in F], rotation=40, ha="right", fontsize=9, color=INK)
    ax.set_yticks(range(len(labels)), labels, fontsize=9, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("mean detector output (0.5 or more = darkest)", color=MUTED, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_title("What the finding detectors see in each disease", loc="left", fontsize=13, color=INK, pad=24)
    ax.text(0, -0.9, "Rows above the gap: training photos, each scored by a detector that never saw it. Below: held-out "
            "rare diseases, never used for training. Columns grouped by location (disc first).",
            transform=ax.get_yaxis_transform(), fontsize=9, color=MUTED, va="bottom")
    os.makedirs("outputs/figures", exist_ok=True)
    out = "outputs/figures/disease_finding_map.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print("saved", out)
