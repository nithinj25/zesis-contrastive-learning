"""Embed a folder of fundus photos with the frozen TRACE encoder: one command for hidden test data.

    python -m encoder.embed --images path/to/photos --out features.npy
    python -m encoder.embed --images photos/ --out f.npy --camera kowa_vx10a     # known camera: colour correction

Writes two files:
  features.npy          float32 (N, 2048), L2-normalised, one row per photo (cosine = dot product)
  features_index.csv    row, path (relative to --images), ok, error. Rows with ok = False (unreadable photo, or
                        no retina found) are all zeros, so the row order always matches the file list.

Every photo gets the evaluation preprocessing of CLAUDE.md section 19.1 (retina ellipse, geometry, illumination,
CLAHE, 384 px). Unknown camera (the default) skips the per-camera colour step; --camera applies a stored camera's
statistics from data/trace_preprocess.json. Folders are searched recursively for .jpg/.jpeg/.png/.tif/.tiff/.bmp.
Preprocessing runs in worker processes that never import torch (each torch import costs ~1.7 GB on Windows).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
_W: dict = {}


def list_images(root: str) -> list[str]:
    """All photo files under root, as sorted paths relative to root (forward slashes)."""
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.relpath(os.path.join(d, f), root).replace(os.sep, "/") for f in files if f.lower().endswith(EXTS)]
    return sorted(out)


def _init(root: str, stats: dict, camera: str | None):
    import cv2

    cv2.setNumThreads(1)
    _W.update(root=root, stats=stats, camera=camera)


def load_one(rel: str) -> tuple[np.ndarray | None, str]:
    """One photo -> (uint8 (384, 384, 3), "") or (None, reason). Runs in a worker (no torch)."""
    import cv2

    from data.preprocess import prepare_photo

    try:
        bgr = cv2.imdecode(np.fromfile(os.path.join(_W["root"], rel), np.uint8), cv2.IMREAD_COLOR)  # unicode-safe
    except OSError as e:
        return None, f"unreadable ({e.__class__.__name__})"
    if bgr is None:
        return None, "unreadable"
    try:
        return prepare_photo(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), _W["camera"], _W["stats"]), ""
    except ValueError as e:
        return None, str(e)


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Embed a folder of fundus photos with the frozen TRACE encoder.")
    ap.add_argument("--images", required=True, help="folder of photos (searched recursively)")
    ap.add_argument("--out", required=True, help="output .npy; the index is written next to it as <name>_index.csv")
    ap.add_argument("--weights", default="checkpoints/trace_v2/encoder.pt")
    ap.add_argument("--stats", default="data/trace_preprocess.json")
    ap.add_argument("--camera", default=None, help="camera id from --stats for every photo; default: unknown camera")
    ap.add_argument("--device", default="auto", help="auto, cpu or cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)),
                    help="preprocessing processes; 0 = in this process")
    return ap.parse_args(argv)


def main(argv=None) -> dict:
    args = parse(argv)
    with open(args.stats) as f:
        stats = json.load(f)
    if args.camera is not None and args.camera not in stats.get("cameras", {}):
        sys.exit(f"unknown --camera {args.camera!r}; known: {', '.join(sorted(stats.get('cameras', {})))}")
    if not os.path.isfile(args.weights):
        sys.exit(f"weights not found: {args.weights} (see README section 7 for the download)")
    files = list_images(args.images)
    if not files:
        sys.exit(f"no photos ({', '.join(EXTS)}) under {args.images}")

    import torch

    from encoder.export import load_trace_encoder, to_tensor

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    enc = load_trace_encoder(args.weights, device=device)
    feats = np.zeros((len(files), 2048), np.float32)
    errors = [""] * len(files)
    print(f"{len(files)} photos, device {device}, camera {args.camera or 'unknown (no colour step)'}", flush=True)

    pool = None
    if args.workers > 0:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.images, stats, args.camera))
        results = pool.imap(load_one, files, chunksize=4)
    else:
        _init(args.images, stats, args.camera)
        results = map(load_one, files)

    t0, rows, xs = time.time(), [], []

    def flush():
        if xs:
            with torch.inference_mode():
                feats[rows] = enc(torch.stack(xs).to(device)).cpu().numpy()
            rows.clear(), xs.clear()

    try:
        for i, (img, err) in enumerate(results):
            if img is None:
                errors[i] = err
            else:
                rows.append(i), xs.append(to_tensor(img))
                if len(xs) == args.batch:
                    flush()
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(files)}  {(i + 1) / (time.time() - t0):.1f} photos/s", flush=True)
        flush()
    finally:
        if pool is not None:
            pool.close(), pool.join()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, feats)
    index = os.path.splitext(args.out)[0] + "_index.csv"
    with open(index, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "ok", "error"])
        w.writerows((i, p, not e, e) for i, (p, e) in enumerate(zip(files, errors)))
    n_bad = sum(bool(e) for e in errors)
    print(f"wrote {args.out} {feats.shape} and {index}; {len(files) - n_bad} ok, {n_bad} failed "
          f"({time.time() - t0:.0f} s)")
    return {"features": args.out, "index": index, "n": len(files), "failed": n_bad}


if __name__ == "__main__":
    main()
