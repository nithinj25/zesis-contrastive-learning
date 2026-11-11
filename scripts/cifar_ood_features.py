"""Non-fundus images for the OOD test (CLAUDE.md 19.7): 500 CIFAR-10 test images through TRACE's own front door.

    python -m scripts.cifar_ood_features --name trace_v2 --weights checkpoints/trace_v2/encoder.pt

Source: CIFAR-10 binary test batch (raw/ood_cifar10/, MD5 of cifar-10-binary.tar.gz checked against the published
c32a1d4ab5d03f1284b67883e8d87530; parsed as raw bytes, no pickle). Each 32 x 32 image is upscaled to 512 x 512.
  1. preprocess() as for any photo: share whose retina is not found (rejected before the encoder).
  2. features for ALL 500 with the whole image treated as retina (resize to 384, CLAHE, no colour step), so the OOD
     score is measured even for images that would never reach it. Saved as features/<name>/ood_cifar_global.npy.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from encoder.export import finish_view, load_trace_encoder, preprocess

N = 500

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", required=True)
    args = ap.parse_args()
    raw = np.fromfile("raw/ood_cifar10/cifar-10-batches-bin/test_batch.bin", dtype=np.uint8).reshape(-1, 3073)[:N]
    imgs = raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    stats = json.load(open("data/trace_preprocess.json"))
    enc = load_trace_encoder(args.weights, "cuda" if torch.cuda.is_available() else "cpu")
    dev = next(enc.parameters()).device
    fov_fail, views = 0, []
    for im in imgs:
        big = cv2.resize(np.ascontiguousarray(im), (512, 512), interpolation=cv2.INTER_CUBIC)
        try:
            preprocess(big, source_id=None)
        except ValueError:
            fov_fail += 1
        views.append(finish_view(big, np.ones((512, 512), np.uint8), None, stats))
    with torch.no_grad():
        F = torch.cat([enc(torch.stack(views[i:i + 50]).to(dev)).cpu() for i in range(0, N, 50)]).numpy()
    np.save(os.path.join("features", args.name, "ood_cifar_global.npy"), F.astype(np.float16))
    json.dump({"n": N, "fov_fail": fov_fail, "fov_fail_share": fov_fail / N},
              open(os.path.join("features", args.name, "ood_cifar_info.json"), "w"))
    print(f"{N} CIFAR-10 images: retina not found for {fov_fail} ({fov_fail / N:.1%}); features saved for all {N}")
