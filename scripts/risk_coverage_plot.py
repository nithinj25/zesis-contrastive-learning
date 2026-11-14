"""Risk-coverage curves: how fast the error falls when the system answers only its most confident photos.

    python -m scripts.risk_coverage_plot            # -> outputs/figures/risk_coverage.png + outputs/eval/risk_coverage.json

Same protocol as eval/trust_eval.py (CLAUDE.md 10.2, 19.7): 7-class disease classifier (logistic on the frozen
global feature, C by grouped CV on head training, temperature on val-A), RFMiD + RFMiD 2.0 test photos with one
diagnosis, sorted by calibrated top probability. TRACE uses its saved classifier (artifacts/trace_v2/trust.joblib),
so its curve is the one in the model card; the baselines get the same recipe refitted in memory (nothing written to
artifacts/). Classifier only: no quality gate, OOD check or conformal set.
"""
from __future__ import annotations

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eval.probes import load_features
from trust.calibrate import fit_temperature
from trust.classifier import Calibrated, Logistic, choose_C, single_dx

ENCODERS = [("imagenet_r50", "ImageNet ResNet-50"), ("moco_plain_e15", "Plain MoCo v2"),
            ("supervised_r50", "Supervised ResNet-50"), ("trace_v2", "TRACE")]
COLOURS = {"trace_v2": "#2a78d6", "imagenet_r50": "#eb6834", "moco_plain_e15": "#1baf7a", "supervised_r50": "#eda100"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def curve(conf: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Coverage (n/N) and error among the n most confident photos, for every n; AURC = mean error over n."""
    c = correct[np.argsort(-conf, kind="stable")].astype(float)
    n = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / n
    return n / len(c), risk, float(risk.mean())


def scores(name: str, meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    f = load_features(name)
    X = f.attrs["X"] / np.linalg.norm(f.attrs["X"], axis=1, keepdims=True)
    row = f.set_index("image_id").row
    feats = lambda ids: X[row.loc[list(ids)].values]
    has = lambda ids: [i for i in ids if i in row.index]
    te = single_dx(meta, has(meta.index[(meta.bucket == "test") & meta.dataset.isin(["rfmid", "rfmid2"])]))
    if name == "trace_v2":
        clf = joblib.load(os.path.join("artifacts", name, "trust.joblib"))["classifier"]
    else:  # eval/trust_eval.py's recipe
        tr = single_dx(meta, has(meta.index[meta.in_heads]))
        va = single_dx(meta, has(meta.index[meta.bucket == "val_a"]))
        g = tr.dup_group.fillna(-1).astype(str).values
        g = np.where(g == "-1.0", tr.index.values, g)
        model = Logistic(choose_C(feats(tr.index), tr.y.values, g)).fit(feats(tr.index), tr.y.values)
        clf = Calibrated(model, fit_temperature(model.logits(feats(va.index)), va.y.values))
    p = clf.proba(feats(te.index))
    return p.max(1), p.argmax(1) == te.y.values


def main():
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    res, curves = {}, {}
    for name, label in ENCODERS:
        conf, ok = scores(name, meta)
        cov, risk, aurc = curve(conf, ok)
        curves[name] = (cov, risk)
        at = lambda c: float(risk[max(1, int(round(c * len(risk)))) - 1])
        res[name] = {"label": label, "n": int(len(ok)), "aurc": aurc,
                     "error_at_coverage": {f"{c:.1f}": at(c) for c in np.arange(1.0, 0.05, -0.1)}}
        print(f"{label:22s} n {len(ok)}  error at 100% {at(1.0):.3f}  80% {at(0.8):.3f}  50% {at(0.5):.3f}  AURC {aurc:.3f}")

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    start = 0.1  # below ~10% coverage only a few dozen photos remain, so the curve is noise
    for name, label in ENCODERS:
        cov, risk = curves[name]
        m = cov >= start
        lw, z = (2.4, 4) if name == "trace_v2" else (1.6, 3)
        ax.plot(cov[m] * 100, risk[m] * 100, color=COLOURS[name], lw=lw, zorder=z, label=label,
                solid_capstyle="round")
    # end labels (error when answering all), nudged apart so none overlap
    ends = sorted((curves[n][1][-1] * 100, n, l) for n, l in ENCODERS)
    ys = [e[0] for e in ends]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + 1.4)
    for (y0, name, label), y in zip(ends, ys):
        ax.annotate(f"{label}  {y0:.1f}%", (100, y0), xytext=(103, y), textcoords="data", va="center", fontsize=8.5,
                    color=INK, fontweight="bold" if name == "trace_v2" else "normal", annotation_clip=False)
    t = curves["trace_v2"]
    for c, (tx, ty) in ((0.8, (86, 4.0)), (0.5, (62, 2.2))):  # notes in the empty lower-right corner
        i = int(round(c * len(t[1]))) - 1
        x, y = t[0][i] * 100, t[1][i] * 100
        ax.plot(x, y, "o", ms=6, color=COLOURS["trace_v2"], mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"TRACE answering {c:.0%}: {y:.1f}% error", (x, y), xytext=(tx, ty), textcoords="data", ha="center",
                    fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, shrinkB=4))
    ax.set_xlim(start * 100, 100)
    ax.set_ylim(0, None)
    ax.set_xlabel("Photos answered (most confident first), %")
    ax.set_ylabel("Error among answered photos, %")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title("Answering only confident photos cuts the error", loc="left", fontsize=12, color=INK, pad=22)
    ax.text(0, 1.03, f"RFMiD + RFMiD 2.0 test, {res['trace_v2']['n']} photos, 7 classes; calibrated classifier on "
            "frozen features. Labels at the right edge: error when answering all.", transform=ax.transAxes,
            fontsize=8, color=INK2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=4, frameon=False, fontsize=8.5,
              labelcolor=INK)
    fig.tight_layout()
    os.makedirs("outputs/figures", exist_ok=True)
    fig.savefig("outputs/figures/risk_coverage.png", facecolor=SURFACE, bbox_inches="tight")
    json.dump(res, open("outputs/eval/risk_coverage.json", "w"), indent=1)
    print("wrote outputs/figures/risk_coverage.png, outputs/eval/risk_coverage.json")


if __name__ == "__main__":
    main()
