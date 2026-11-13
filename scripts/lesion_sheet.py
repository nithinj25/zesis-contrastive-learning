"""Contact sheet: where do lesion-guided local crops land on real photos? (visual check before training)

Run from the repo root:  python -m scripts.lesion_sheet
8 random pool photos (seed 0): 4 with DR grade >= 2 (EyePACS / APTOS labels, monitoring only) and 4 without DR.
Top: photo + 12 guided crop boxes. Bottom: its sampling map. Writes outputs/sheets/lesion_crops.png.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from augment.lesions import sample_guided_crop
from data.process import load_processed

meta = pd.read_parquet("data/meta.parquet")
maps = np.load("data/lesion_maps.npy", mmap_mode="r")
paths = [str(p) for p in np.load("data/lesion_maps_paths.npy")]
row = {p: i for i, p in enumerate(paths)}
pool = meta[meta.processed_path.isin(set(paths))]
grade = pool.labels.apply(lambda x: max([int(l[-1]) for l in x if l[:-1] in ("eyepacs_dr", "aptos_dr")] or [-1]))
rng = np.random.default_rng(0)
pick = list(rng.choice(pool[grade >= 2].processed_path.values, 4, replace=False)) + \
       list(rng.choice(pool[grade == 0].processed_path.values, 4, replace=False))

fig, axes = plt.subplots(2, 8, figsize=(24, 6.6), dpi=110)
for c, p in enumerate(pick):
    rgb, mask = load_processed(os.path.join("processed", p))
    pm = np.asarray(maps[row[p]], np.float32)
    axes[0, c].imshow(rgb)
    for _ in range(12):
        crop = sample_guided_crop(rng, pm, mask)
        if crop:
            x, y, s = crop
            axes[0, c].add_patch(Rectangle((x - s / 2, y - s / 2), s, s, fill=False, ec="#ffffff", lw=1.2))
    axes[1, c].imshow(pm, cmap="Blues")
    axes[0, c].set_title(("DR grade >= 2 · " if c < 4 else "no DR · ") + p.split("/")[-1][:22], fontsize=8)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("Lesion-guided local crops (12 draws per photo, white boxes) and the sampling map (darker = likelier)", x=0.01, ha="left")
fig.tight_layout()
os.makedirs("outputs/sheets", exist_ok=True)
fig.savefig("outputs/sheets/lesion_crops.png")
print("outputs/sheets/lesion_crops.png")
