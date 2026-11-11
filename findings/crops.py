"""Disc and macula crop features for the finding detectors (CLAUDE.md sections 8.1 and 19.2).

    python -m findings.crops --name trace_v2 --weights checkpoints/trace_v2/encoder.pt

For every labelled photo (buckets train, val_a, val_b, test, ldo_test; the pool has no labels, so it is skipped):
  1. find the disc and the fovea on the processed 512 px photo (findings/localise.py);
  2. apply the evaluation steps to the WHOLE photo at 512 px (step 7 for a known camera, step 8 CLAHE), exactly
     as for the global feature, so a crop shows the same pixels the global view does;
  3. cut the disc crop (2 DD) and the macula crop (3 DD), resize each to 384 px, encode with the frozen encoder.
Writes features/<name>/<bucket>_disc.npy and <bucket>_macula.npy (float16, (N, 2048), L2-normalised; a row of zeros
where localisation failed) and adds loc_ok, disc_x, disc_y, fovea_x, fovea_y to <bucket>_index.parquet.
The rows follow the existing <bucket>_index.parquet, so they line up with <bucket>_global.npy.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from data.process import load_processed
from encoder.export import finish_view, load_trace_encoder
from findings.localise import DD, crop, localise

BUCKETS = ["train", "val_a", "val_b", "test", "ldo_test"]
OUT = 384


def crops_for_photo(rgb: np.ndarray, mask: np.ndarray, source: str | None, stats: dict):
    """-> (disc crop, macula crop) as (3, 384, 384) float tensors in [0, 1], landmarks. Zeros if not found."""
    lm = localise(rgb, mask)
    if not lm.ok:
        z = torch.zeros(3, OUT, OUT)
        return z, z.clone(), lm
    view = finish_view(rgb, mask, source, stats, size=rgb.shape[0])  # steps 7-8 on the whole photo, no resize
    img = (view.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).float() / 255.0
    return (to_t(crop(img, (lm.disc_x, lm.disc_y), 2 * DD, OUT)),
            to_t(crop(img, (lm.fovea_x, lm.fovea_y), 3 * DD, OUT)), lm)


class CropPhotos(Dataset):
    def __init__(self, rows: pd.DataFrame, stats: dict, root: str = "processed"):
        self.paths = [os.path.join(root, p) for p in rows.processed_path]
        self.sources, self.stats = rows.source.tolist(), stats

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb, mask = load_processed(self.paths[i])
        d, m, lm = crops_for_photo(rgb, mask, self.sources[i], self.stats)
        return d, m, torch.tensor([lm.disc_x, lm.disc_y, lm.fovea_x, lm.fovea_y, float(lm.ok)])


@torch.no_grad()
def extract(encoder, rows: pd.DataFrame, stats: dict, device: str, batch: int = 32, workers: int = 4):
    loader = DataLoader(CropPhotos(rows, stats), batch_size=batch, num_workers=workers)
    disc, mac, info = [], [], []
    for d, m, lm in loader:
        ok = lm[:, 4] > 0
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            fd = encoder(d.to(device)).float().cpu() * ok[:, None]  # failed rows become zeros
            fm = encoder(m.to(device)).float().cpu() * ok[:, None]
        disc.append(fd.numpy().astype(np.float16))
        mac.append(fm.numpy().astype(np.float16))
        info.append(lm.numpy())
    return np.concatenate(disc), np.concatenate(mac), np.concatenate(info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--buckets", nargs="*", default=BUCKETS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    stats = json.load(open("data/trace_preprocess.json"))
    enc = load_trace_encoder(args.weights, device)
    out_dir = os.path.join("features", args.name)
    for bucket in args.buckets:
        idx_path = os.path.join(out_dir, f"{bucket}_index.parquet")
        index = pd.read_parquet(idx_path)
        rows = meta.loc[index.image_id].reset_index()
        t = time.time()
        disc, mac, info = extract(enc, rows, stats, device, workers=args.workers)
        np.save(os.path.join(out_dir, f"{bucket}_disc.npy"), disc)
        np.save(os.path.join(out_dir, f"{bucket}_macula.npy"), mac)
        index = index[["image_id"]].assign(loc_ok=info[:, 4] > 0, disc_x=info[:, 0], disc_y=info[:, 1],
                                           fovea_x=info[:, 2], fovea_y=info[:, 3])
        index.to_parquet(idx_path)
        print(f"{bucket}: {len(rows)} photos, loc_ok {index.loc_ok.mean():.1%}, {time.time() - t:.0f} s", flush=True)
