"""Contact sheet of training views, to review by eye before training (CLAUDE.md section 19.5).

    python scripts/view_sheet.py [--policy retina] [--artifacts] [--rows 8] [--views 7] [--size 224] [--scale 0.6 1.0]
    python scripts/view_sheet.py --each-artifact      # one column per artifact at its strongest setting

Each row: one pool photo (one per camera first). Column 0 is the evaluation image (steps 7 + 8, no augmentation);
the rest are random training views. Writes outputs/sheets/views_<name>.png and prints the time per view.
--each-artifact: column 0 is a clean view; each further column is the SAME view with one artifact at the top of
its range (section 19.5). If a lesion visible in column 0 disappears, the setting is too strong.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from augment.artifacts import DEFAULT_P, Artifacts  # noqa: E402
from augment.views import plain_view_np, prepare, retina_view_np  # noqa: E402
from data.preprocess import camera_color, clahe_l  # noqa: E402
from data.process import load_processed  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="retina", choices=["retina", "plain"])
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--views", type=int, default=7)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--scale", type=float, nargs=2, default=[0.6, 1.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--artifacts", action="store_true", help="retina views with the artifact simulator")
    ap.add_argument("--each-artifact", action="store_true")
    args = ap.parse_args()
    name = "each_artifact" if args.each_artifact else args.policy + ("_artifacts" if args.artifacts else "")

    meta = pd.read_parquet("data/meta.parquet")
    stats = json.load(open("data/trace_preprocess.json"))
    pool = meta[meta.in_pool]
    pick = pool.groupby("source", group_keys=False).apply(lambda g: g.sample(1, random_state=args.seed))
    if len(pick) < args.rows:  # prefer photos with findings so lesions can be checked
        extra = pool[(pool.findings.str.len() > 0) & ~pool.image_id.isin(pick.image_id)]
        pick = pd.concat([pick, extra.sample(args.rows - len(pick), random_state=args.seed)])
    pick = pick.head(args.rows)

    rng = np.random.default_rng(args.seed)
    p = stats["params"]
    tiles, times = [], []
    for r in pick.itertuples():
        rgb, mask = load_processed(os.path.join("processed", r.processed_path))
        cam = stats["cameras"].get(r.source)
        ref = camera_color(rgb, mask, cam["mean"], cam["std"], stats["pool"]["mean"], stats["pool"]["std"],
                           p["colour_alpha"]) if cam else rgb
        ref = cv2.resize(clahe_l(ref, mask, clip=p["clahe_clip"]), (args.size, args.size), interpolation=cv2.INTER_AREA)
        row = [ref]
        if args.each_artifact:
            seed = int(rng.integers(1 << 30))
            calm = dict(colour_p=0, jitter_p=0, blur_p=0, clahe_clip=p["clahe_clip"], camera=cam, pool=stats["pool"])
            row = [retina_view_np(rgb, mask, np.random.default_rng(seed), args.size, **calm).transpose(1, 2, 0)]
            for k in DEFAULT_P:
                art = Artifacts(stats, p={k: 1.0}, strong=True)
                v = retina_view_np(rgb, mask, np.random.default_rng(seed), args.size, artifacts=art, **calm)
                row.append(v.transpose(1, 2, 0))
        art = Artifacts(stats) if args.artifacts else None
        for _ in range(0 if args.each_artifact else args.views):
            t = time.perf_counter()
            if args.policy == "retina":
                v = retina_view_np(rgb, mask, rng, args.size, scale=tuple(args.scale), camera=cam, pool=stats["pool"],
                                   colour_alpha=p["colour_alpha"], clahe_clip=p["clahe_clip"], artifacts=art)
            else:
                v = plain_view_np(prepare(rgb, mask), rng, args.size)
            times.append(time.perf_counter() - t)
            row.append(v.transpose(1, 2, 0))
        label = f"{r.source} {','.join(r.findings) if len(r.findings) else ''}"[:40]
        strip = np.ascontiguousarray(np.concatenate(row, axis=1))
        cv2.putText(strip, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(strip)
    sheet = np.concatenate(tiles, axis=0)
    os.makedirs("outputs/sheets", exist_ok=True)
    if args.each_artifact:  # column titles
        for j, k in enumerate(["clean"] + list(DEFAULT_P)):
            cv2.putText(sheet, k, (j * args.size + 4, args.size - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    out = f"outputs/sheets/views_{name}.png"
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    if times:
        print(f"{out}: {len(pick)} photos x {args.views} views; {1000 * np.mean(times):.1f} ms per view "
              f"(median {1000 * np.median(times):.1f} ms) on one core")
    else:
        print(f"{out}: {len(pick)} photos, one column per artifact at full strength")
