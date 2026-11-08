"""Feature cache (CLAUDE.md section 19.2): run a frozen encoder over every photo once.

Run from the repo root:  python -m encoder.features --name imagenet_r50 [--weights encoder.pt] [--size 384]
Writes features/<name>/<bucket>_global.npy (float16, (N, 2048), L2-normalised) and <bucket>_index.parquet
(image_id in row order) for each bucket in data/meta.parquet except "excluded" and "spare".
Every head, graph, trust and eval script reads features only through this name, so swapping encoders means
changing one config value. Disc and macula crop features (6,144-d) come in Phase 4.
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

BUCKETS = ["train", "val_a", "val_b", "test", "ldo_test", "pool"]


class ProcessedPhotos(Dataset):
    def __init__(self, rows: pd.DataFrame, stats: dict, size: int, root: str = "processed"):
        self.paths = [os.path.join(root, p) for p in rows.processed_path]
        self.sources = rows.source.tolist()
        self.stats, self.size = stats, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rgb, mask = load_processed(self.paths[i])
        return finish_view(rgb, mask, self.sources[i], self.stats, self.size)


@torch.no_grad()
def extract(encoder, rows: pd.DataFrame, stats: dict, size: int, device: str, batch: int = 64, workers: int = 4) -> np.ndarray:
    loader = DataLoader(ProcessedPhotos(rows, stats, size), batch_size=batch, num_workers=workers,
                        pin_memory=True, persistent_workers=False)
    out = []
    for x in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(encoder(x.to(device, non_blocking=True)).cpu().numpy().astype(np.float16))
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="encoder name, e.g. imagenet_r50, moco_plain, trace_v1")
    ap.add_argument("--weights", default=None, help="backbone state_dict; omit for the ImageNet baseline")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--buckets", nargs="*", default=BUCKETS)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = pd.read_parquet("data/meta.parquet")
    stats = json.load(open("data/trace_preprocess.json"))
    enc = load_trace_encoder(args.weights, device)
    out_dir = os.path.join("features", args.name)
    os.makedirs(out_dir, exist_ok=True)
    for bucket in args.buckets:
        rows = meta[(meta.bucket == bucket) & meta.fov_ok].reset_index(drop=True)
        t = time.time()
        feats = extract(enc, rows, stats, args.size, device, workers=args.workers)
        np.save(os.path.join(out_dir, f"{bucket}_global.npy"), feats)
        rows[["image_id"]].to_parquet(os.path.join(out_dir, f"{bucket}_index.parquet"))
        print(f"{bucket}: {len(rows)} photos, {time.time() - t:.0f} s, {len(rows) / (time.time() - t):.0f} photos/s")
    json.dump({"name": args.name, "weights": args.weights, "size": args.size}, open(os.path.join(out_dir, "info.json"), "w"))
