"""IDRiD's optic-disc and fovea centres, moved from raw pixels into our processed 512 px space (CLAUDE.md 8.1).

    python -m findings.landmarks        # writes data/idrid_landmarks.parquet

Used only to CHECK the disc / macula localiser: develop on idrid_split == "train", report on "test". Step 4
(data/preprocess.py: standardise_geometry) crops the window [cx - rx, cx + rx] x [cy - ry, cy + ry] around the
retina ellipse and resizes it to size x size, so a raw point (x, y) lands at
    x' = (x - x0) * size / (x1 - x0),   y' = (y - y0) * size / (y1 - y0)
with the same integer rounding of x0, x1, y0, y1 as step 4.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from data.preprocess import TARGET_DIAMETER, find_fov


def map_point(x: float, y: float, cx: float, cy: float, rx: float, ry: float, size: int = TARGET_DIAMETER):
    x0, x1 = int(round(cx - rx)), int(round(cx + rx))
    y0, y1 = int(round(cy - ry)), int(round(cy + ry))
    return (x - x0) * size / (x1 - x0), (y - y0) * size / (y1 - y0)


def _one(job):
    image_id, path, pts = job
    cv2.setNumThreads(1)
    rgb = cv2.cvtColor(cv2.imread(os.path.join("raw", path)), cv2.COLOR_BGR2RGB)
    fov = find_fov(rgb)
    if fov is None:
        return None
    od = map_point(pts[0], pts[1], fov.cx, fov.cy, fov.rx, fov.ry)
    fv = map_point(pts[2], pts[3], fov.cx, fov.cy, fov.rx, fov.ry)
    return {"image_id": image_id, "od_x": od[0], "od_y": od[1], "fovea_x": fv[0], "fovea_y": fv[1]}


if __name__ == "__main__":
    meta = pd.read_parquet("data/meta.parquet")
    i = meta[(meta.dataset == "idrid") & meta.od_x.notna() & meta.processed_path.notna()]
    jobs = [(r.image_id, r.path, (r.od_x, r.od_y, r.fovea_x, r.fovea_y)) for r in i.itertuples()]
    with ProcessPoolExecutor(6) as ex:
        rows = [r for r in ex.map(_one, jobs, chunksize=8) if r is not None]
    out = pd.DataFrame(rows).merge(i[["image_id", "idrid_split", "processed_path"]], on="image_id")
    out.to_parquet("data/idrid_landmarks.parquet", index=False)
    d = np.hypot(out.od_x - out.fovea_x, out.od_y - out.fovea_y)
    tr = out.idrid_split == "train"
    print(f"{len(out)} of {len(i)} photos mapped; disc-fovea distance, train split: median {d[tr].median():.1f} px "
          f"(IQR {d[tr].quantile(.25):.1f}-{d[tr].quantile(.75):.1f}) at a {TARGET_DIAMETER} px retina")
