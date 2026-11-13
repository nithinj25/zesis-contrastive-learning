"""Score the 8 ablation-ladder encoders on the held-out tests and build the ablation table (CLAUDE.md 11, 19.15).

    python -m scripts.ablation_eval              # run what is missing, then build the table
    python -m scripts.ablation_eval --table-only # rebuild the table from results already on disk

Runs the existing evaluation scripts with their output redirected to outputs/ablation/ (the headline results in
outputs/eval/ are never touched): eval.probes per encoder, then eval.leave_camera_out, eval.fewshot and two paired
bootstraps (eval.compare: full TRACE vs every rung, plain MoCo vs every rung). A step is skipped when its output file
exists. Every rung is read at its final epoch (30), no checkpoint picking. Label the table "short budget".

Seed noise: plain MoCo and full TRACE were trained twice (seeds 0 and 1). The table's last row is the larger of the
two seed gaps per column; a rung-to-rung difference smaller than that is not read as an effect.
Writes outputs/ablation/table.csv and outputs/ablation/table.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pandas as pd

# ladder order: each rung adds one component to the one above it
RUNGS = [
    ("abl_r0_plain", "Plain MoCo v2"),
    ("abl_r1_views", "+ retina-safe views"),
    ("abl_r2_artifacts", "+ artifact simulator"),
    ("abl_r3_multicrop", "+ multi-crop"),
    ("abl_r3_lesion", "r3 with Approach 2 lesion-guided locals + self-masking"),
    ("abl_r4_beta", "+ same-camera negatives x2 (beta)"),
    ("abl_r5_full", "+ cross-camera mining (full TRACE)"),
]
SEED_PAIRS = [("abl_r0_plain", "abl_r0_plain_s1"), ("abl_r5_full", "abl_r5_full_s1")]
NAMES = [n for n, _ in RUNGS] + [b for _, b in SEED_PAIRS]
OUT = "outputs/ablation"
CAMS = {"topcon_3d_oct2000": "OCT2000", "kowa_vx10a": "Kowa"}


def sh(args: list[str], log: str) -> None:
    print(" ".join(args), flush=True)
    with open(os.path.join(OUT, log), "a", encoding="utf-8") as f:
        code = subprocess.run([sys.executable, "-m"] + args, stdout=f, stderr=subprocess.STDOUT).returncode
    if code:
        raise SystemExit(f"{args[0]} failed (exit {code}); see {OUT}/{log}")


def run_all() -> None:
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    for n in NAMES:
        if not os.path.exists(os.path.join(OUT, "eval", f"{n}.json")):
            sh(["eval.probes", "--name", n, "--out_dir", os.path.join(OUT, "eval")], "eval_probes.log")
    steps = [
        ("leave_camera_out.json", ["eval.leave_camera_out", *NAMES]),
        ("fewshot.json", ["eval.fewshot", *NAMES]),
        ("compare_full.json", ["eval.compare", "abl_r5_full", *[n for n in NAMES if n != "abl_r5_full"]]),
        ("compare_plain.json", ["eval.compare", "abl_r0_plain", *[n for n in NAMES if n != "abl_r0_plain"]]),
    ]
    for out, args in steps:
        path = os.path.join(OUT, out)
        if not os.path.exists(path):
            sh(args + ["--out", path], "eval_" + out.replace(".json", ".log"))


def val_final(n: str) -> tuple[float, float]:
    v = pd.read_csv(f"outputs/train/{n}/val.csv").iloc[-1]
    assert int(v.epoch) == 30, f"{n}: last val-A check is epoch {v.epoch}, not 30"
    return float(v.val_linear_macro_auroc), float(v.val_knn_macro_auroc)


def row(n: str, loco: dict, few: dict, cam: dict) -> dict:
    p = json.load(open(os.path.join(OUT, "eval", f"{n}.json")))
    va_lin, va_knn = val_final(n)
    r = {"encoder": n, "valA_linear": va_lin, "valA_knn": va_knn,
         "rfmid_linear": p["rfmid_linear_macro_auroc"], "rfmid_knn": p["rfmid_knn_macro_auroc"],
         "idrid_any_dr": p["idrid_any_dr_auroc"], "paraguay_any_dr": p["paraguay_any_dr_auroc_from_idrid"]}
    for c, short in CAMS.items():
        g = loco[n]["plain"][c]["linear_labels"]
        r[f"unseen_{short}"], r[f"gap_{short}"] = g["out"], g["gap"]
    r["same_cam_nn"] = loco[n]["same_camera_share"]["plain"][0]
    r["jsiec_k5_proto"] = few[n]["K5"]["prototype"]["top1"]
    r["jsiec_k5_logistic"] = few[n]["K5"]["logistic"]["top1"]
    r["camera_probe_rfmid"] = cam[n]["all"]["bal_acc"]
    return r


def table() -> pd.DataFrame:
    loco = json.load(open(os.path.join(OUT, "leave_camera_out.json")))
    few = json.load(open(os.path.join(OUT, "fewshot.json")))
    cam = json.load(open(os.path.join(OUT, "compare_full.json")))["camera_rfmid"]
    rows = {n: row(n, loco, few, cam) for n in NAMES}
    df = pd.DataFrame([{**rows[n], "rung": label} for n, label in RUNGS])
    for a, b in SEED_PAIRS:
        df = pd.concat([df, pd.DataFrame([{**rows[b], "rung": f"(seed 1 of {a})"}])])
    cols = [c for c in df.columns if c not in ("encoder", "rung")]
    noise = {c: max(abs(rows[a][c] - rows[b][c]) for a, b in SEED_PAIRS) for c in cols}
    df = pd.concat([df, pd.DataFrame([{"encoder": "", "rung": "seed noise (max |seed 0 - seed 1|)", **noise}])])
    return df[["rung", "encoder"] + cols].reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-only", action="store_true")
    args = ap.parse_args()
    if not args.table_only:
        run_all()
    df = table()
    df.to_csv(os.path.join(OUT, "table.csv"), index=False)
    md = df.drop(columns="encoder").to_markdown(index=False, floatfmt=".3f")
    head = ("Short-budget ablation ladder (10k photos, 30 epochs at 224 px, K 4,096), each rung read at epoch 30.\n"
            "unseen_* = linear macro-AUROC on an RFMiD camera never seen by the probe; gap_* = in-domain minus that.\n"
            "same_cam_nn = share of 20 nearest neighbours from the same camera (lower = less camera-driven).\n\n")
    open(os.path.join(OUT, "table.md"), "w", encoding="utf-8").write(head + md + "\n")
    print(head + md)
