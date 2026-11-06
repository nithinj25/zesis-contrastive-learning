"""Trace one photo through the data pipeline, printing the real numbers at each step.

Run from the repo root:  python scripts/pipeline_walkthrough.py [image_id]
Writes outputs/pipeline_walkthrough.jpg (not committed: it shows dataset images).
Used to write docs/DATA_PIPELINE.md.
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())
from data.dedupe import coarse_signature, fine_signature, match_score  # noqa: E402
from data.preprocess import correct_illumination, find_fov, standardise_geometry  # noqa: E402

meta = pd.read_parquet("data/raw_meta.parquet").set_index("image_id")
image_id = sys.argv[1] if len(sys.argv) > 1 else meta.index[(meta.dataset == "rfmid2") & (meta.size_group == "512x512")][0]
row = meta.loc[image_id]
rgb = cv2.cvtColor(cv2.imread(os.path.join("raw", row.path)), cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
print(f"image {image_id}: {w}x{h}, dataset {row.dataset}, split {row.split}, source {row.source}, labels {list(row.labels)}")

# Step 3 internals, repeated here to print them.
red = cv2.GaussianBlur(rgb[..., 0].astype(np.float32), (0, 0), max(1.0, min(h, w) / 512))
p = max(8, min(h, w) // 32)
corners = np.median([np.median(a) for a in (red[:p, :p], red[:p, -p:], red[-p:, :p], red[-p:, -p:])])
centre = np.median(red[h // 4: 3 * h // 4, w // 4: 3 * w // 4])
threshold = max(0.06 * red.max(), corners + max(8.0, 0.10 * (centre - corners)))
print(f"step 3: red max {red.max():.1f}, corners {corners:.1f}, centre {centre:.1f}, "
      f"threshold = max(0.06*{red.max():.1f}, {corners:.1f} + max(8, 0.10*({centre:.1f}-{corners:.1f}))) = {threshold:.1f}")
blob = cv2.morphologyEx((red > threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
fov = find_fov(rgb)
print(f"        ellipse centre ({fov.cx:.1f}, {fov.cy:.1f}), rx {fov.rx:.1f}, ry {fov.ry:.1f}, ry/rx {fov.ry / fov.rx:.3f}, fit {fov.fit:.2f}, "
      f"mask covers {fov.mask.mean():.1%} of the photo")

std, mask = standardise_geometry(rgb, fov)
print(f"step 4: crop {2 * fov.rx:.0f}x{2 * fov.ry:.0f} px -> 512x512; horizontal scale {512 / (2 * fov.rx):.3f}, "
      f"vertical {512 / (2 * fov.ry):.3f}; mask covers {mask.mean():.1%} (a full circle is {np.pi / 4:.1%})")

lit = correct_illumination(std, mask)
L0 = cv2.cvtColor(std, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
L1 = cv2.cvtColor(lit, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
yy, xx = np.ogrid[:512, :512]
r = np.hypot(xx - 255.5, yy - 255.5) / 256
inner, rim = (r < 0.3) & (mask > 0), (r > 0.8) & (mask > 0)
print(f"step 6: L centre {L0[inner].mean():.1f} vs rim {L0[rim].mean():.1f} (gap {L0[inner].mean() - L0[rim].mean():.1f}) -> "
      f"centre {L1[inner].mean():.1f} vs rim {L1[rim].mean():.1f} (gap {L1[inner].mean() - L1[rim].mean():.1f})")

# Dedupe signatures: the photo against a darker, slightly blurred copy of itself and against another eye.
copy = cv2.GaussianBlur((std * 0.75).astype(np.uint8), (0, 0), 0.8)
other_id = meta.index[(meta.dataset == row.dataset) & (meta.index != image_id)][5]
o = cv2.cvtColor(cv2.imread(os.path.join("raw", meta.path[other_id])), cv2.COLOR_BGR2RGB)
ostd, _ = standardise_geometry(o, find_fov(o))
c_self = float(coarse_signature(std) @ coarse_signature(copy))
c_other = float(coarse_signature(std) @ coarse_signature(ostd))
print(f"dedupe: coarse vector length {coarse_signature(std).size}; coarse similarity to own copy {c_self:.2f}, to {other_id} {c_other:.2f}")
print(f"        fine score to own copy {match_score(fine_signature(std), fine_signature(copy)):.2f}, "
      f"to {other_id} {match_score(fine_signature(std), fine_signature(ostd)):.2f}")

# Figure: raw + outline/ellipse, threshold blob, standardised, illumination-corrected, detail band.
n = 256
over = rgb.copy()
cnt, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(over, cnt, -1, (255, 255, 0), max(1, w // 300))
cv2.ellipse(over, (int(fov.cx), int(fov.cy)), (int(fov.rx), int(fov.ry)), 0, 0, 360, (0, 255, 0), max(1, w // 300))
g = std[..., 1].astype(np.float32)
band = cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 6.0)
band = np.clip(128 + band * 6, 0, 255).astype(np.uint8)
panels = [cv2.resize(over, (n, n)), cv2.resize(cv2.cvtColor(blob * 255, cv2.COLOR_GRAY2RGB), (n, n)),
          cv2.resize(std, (n, n)), cv2.resize(lit, (n, n)), cv2.resize(cv2.cvtColor(band, cv2.COLOR_GRAY2RGB), (n, n))]
for k, t in enumerate(["1 raw + outline + ellipse", "2 rough outline", "3 geometry 512", "4 lighting fixed", "5 dedupe detail"]):
    cv2.putText(panels[k], t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/pipeline_walkthrough.jpg", cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("figure: outputs/pipeline_walkthrough.jpg")
