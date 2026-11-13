"""Precompute Approach 2's lesion-guided crop-sampling map (augment/lesions.py) for every pool photo, once.

Run from the repo root:  python -m scripts.lesion_maps [--pool_ids data/ablation_pool.csv] [--workers 6]
Writes data/lesion_maps.npy ((N, 64, 64) float16, each map sums to 1) and data/lesion_maps_paths.npy (processed_path
per row). The map is fixed per photo (computed on the processed 512 px photo, before any view edits), so the loader
reads it instead of recomputing it every time a photo is loaded.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd


def one(path: str) -> np.ndarray:
    import cv2

    cv2.setNumThreads(1)
    from augment.lesions import blob_response, crop_sampling_map
    from data.process import load_processed

    rgb, mask = load_processed(os.path.join("processed", path))
    return crop_sampling_map(blob_response(rgb, mask), mask).astype(np.float16)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool_ids", default="data/ablation_pool.csv")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet")
    rows = meta[meta.in_pool]
    if args.pool_ids:
        rows = rows[rows.image_id.isin(set(pd.read_csv(args.pool_ids).image_id))]
    paths = rows.processed_path.tolist()
    t = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        maps = np.stack(pool.map(one, paths, chunksize=32))
    np.save("data/lesion_maps.npy", maps)
    np.save("data/lesion_maps_paths.npy", np.array(paths))
    print(f"{len(paths)} maps in {time.time() - t:.0f} s; row sums {maps.astype(np.float32).sum((1, 2)).min():.3f}"
          f"-{maps.astype(np.float32).sum((1, 2)).max():.3f}")
