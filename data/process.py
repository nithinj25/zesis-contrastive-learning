"""One offline pass over every photo (CLAUDE.md section 6, steps 3, 4 and 6).

Run from the repo root, after data.ingest:  python -m data.process [--limit N] [--workers 6]
For each photo it:
  1. finds the retina (step 3) and maps it to a 512 px circle (step 4),
  2. measures quality on that image, before any correction (step 5 is fitted later on these numbers),
  3. fixes the lighting (step 6) and saves processed/<dataset>/<image_id>.png, RGBA: alpha = retina mask,
  4. measures the colour (LAB mean/std inside the mask), for the per-camera colour step 7,
  5. measures simple statistics of the raw and the processed photo, for the camera probe (section 6).
Writes data/process.parquet, one row per photo. A photo that fails step 3 gets fov_ok = False and no file.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from data.preprocess import correct_illumination, find_fov, standardise_geometry

OUT_ROOT = "processed"
SIZE = 512
_yy, _xx = np.ogrid[:SIZE, :SIZE]
_R = np.hypot(_xx - (SIZE - 1) / 2, _yy - (SIZE - 1) / 2) / (SIZE / 2)  # 0 at the centre, 1 at the rim


def quality_features(std: np.ndarray, mask: np.ndarray) -> dict:
    """Numbers a quality grade can be fitted on (section 19.9). Computed inside the retina only.

    sharpness: variance of the Laplacian of green (blur lowers it).
    detail: RMS of fine band-pass detail in the central retina (vessel edges; blur and haze lower it).
    L_mean, L_std: exposure and contrast. dark_frac, bright_frac: share of nearly black / blown-out pixels.
    centre_rim: centre minus rim brightness (vignetting, uneven light).
    """
    inside = cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0  # keep the mask edge out of the Laplacian
    g = std[..., 1].astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    band = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 6.0)
    centre = inside & (_R < 0.5)
    L = cv2.cvtColor(std, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    Li = L[inside]
    rim, mid = inside & (_R > 0.8), inside & (_R < 0.3)
    return {
        "q_sharpness": float(lap[inside].var()),
        "q_detail": float(np.sqrt((band[centre] ** 2).mean())) if centre.any() else 0.0,
        "q_L_mean": float(Li.mean()),
        "q_L_std": float(Li.std()),
        "q_dark_frac": float((Li < 20).mean()),
        "q_bright_frac": float((Li > 240).mean()),
        "q_centre_rim": float(L[mid].mean() - L[rim].mean()) if rim.any() and mid.any() else 0.0,
    }


def colour_stats(img: np.ndarray, mask: np.ndarray) -> dict:
    """LAB mean and std inside the mask, plus the pixel count, so camera stats can be pooled later."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float64)[mask > 0]
    out = {"lab_n": int(len(lab))}
    for c, name in enumerate("Lab"):
        out[f"lab_{name}_mean"], out[f"lab_{name}_std"] = float(lab[:, c].mean()), float(lab[:, c].std())
    return out


def probe_features(img: np.ndarray, prefix: str) -> dict:
    """Simple whole-photo statistics for the camera probe. Same recipe before and after processing.

    The photo is resized to 256 x 256 first, so image size itself is not a feature.
    """
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    rgb = small.reshape(-1, 3).astype(np.float32)
    g = small[..., 1].astype(np.float32)
    out = {f"{prefix}black_frac": float((rgb.max(1) < 15).mean()),
           f"{prefix}sharpness": float(cv2.Laplacian(g, cv2.CV_32F).var())}
    for c, name in enumerate("RGB"):
        out[f"{prefix}{name}_mean"], out[f"{prefix}{name}_std"] = float(rgb[:, c].mean()), float(rgb[:, c].std())
    for c, name in enumerate("Lab"):
        out[f"{prefix}{name}_mean"], out[f"{prefix}{name}_std"] = float(lab[:, c].mean()), float(lab[:, c].std())
    return out


def process_one(job: tuple[str, str, str, str]) -> dict:
    image_id, src_path, out_path, _ = job
    cv2.setNumThreads(1)
    row = {"image_id": image_id, "fov_ok": False}
    bgr = cv2.imread(src_path)
    if bgr is None:
        row["fail_reason"] = "unreadable"
        return row
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    row.update(probe_features(rgb, "raw_"))
    fov = find_fov(rgb)
    if fov is None:
        row["fail_reason"] = "fov_fail"
        return row
    std, mask = standardise_geometry(rgb, fov, size=SIZE)
    lit = correct_illumination(std, mask)
    row.update(fov_ok=True, fail_reason="", fov_fit=fov.fit, fov_ratio=fov.ry / fov.rx,
               fov_radius_frac=fov.rx / rgb.shape[1], mask_frac=float(mask.mean()))
    row.update(quality_features(std, mask))
    row.update(colour_stats(lit, mask))
    row.update(probe_features(lit, "proc_"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgba = np.dstack([cv2.cvtColor(lit, cv2.COLOR_RGB2BGR), mask * 255])
    cv2.imwrite(out_path, rgba, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    row["processed_path"] = os.path.relpath(out_path, OUT_ROOT).replace(os.sep, "/")
    return row


def load_processed(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read a processed PNG back as (RGB uint8, mask 0/1)."""
    bgra = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return cv2.cvtColor(bgra[..., :3], cv2.COLOR_BGR2RGB), (bgra[..., 3] > 127).astype(np.uint8)


def run(meta: pd.DataFrame, raw_root: str = "raw", out_root: str = OUT_ROOT, workers: int = 6) -> pd.DataFrame:
    jobs = [(i, os.path.join(raw_root, p), os.path.join(out_root, d, f"{i}.png"), d)
            for i, p, d in zip(meta.image_id, meta.path, meta.dataset)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(process_one, jobs, chunksize=16))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N photos of each dataset (a trial run)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_parquet("data/raw_meta.parquet")
    if args.limit:
        meta = meta.groupby("dataset", group_keys=False).head(args.limit)
    out = run(meta, workers=args.workers)
    out.to_parquet("data/process.parquet" if not args.limit else "outputs/process_trial.parquet")
    print(f"photos {len(out)}, retina found {out.fov_ok.sum()}, failed: {out[~out.fov_ok].fail_reason.value_counts().to_dict()}")
