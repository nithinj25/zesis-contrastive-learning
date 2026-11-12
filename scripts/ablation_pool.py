"""The ~10k-photo pool for the short ablation ladder (CLAUDE.md section 19.15).

    python -m scripts.ablation_pool        # writes data/ablation_pool.csv (image_id), deterministic (seed 0)

Every non-EyePACS pool photo (RFMiD train, RFMiD 2.0 train, APTOS: the small sources where camera mixing and mining
matter), then EyePACS photos drawn at random to reach 10,000, so the mix still resembles the full 22.5k pool. ODIR is
not ingested, so it is not in the ladder. Queue K = 4,096 (about 0.4 x pool, section 7.6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

N = 10_000

if __name__ == "__main__":
    meta = pd.read_parquet("data/meta.parquet")
    pool = meta[meta.in_pool]
    small = pool[pool.dataset != "eyepacs"]
    eye = pool[pool.dataset == "eyepacs"]
    pick = eye.iloc[np.sort(np.random.default_rng(0).choice(len(eye), N - len(small), replace=False))]
    out = pd.concat([small, pick])
    out[["image_id"]].to_csv("data/ablation_pool.csv", index=False)
    print(f"{len(out)} photos: " + ", ".join(f"{d} {n}" for d, n in out.dataset.value_counts().items())
          + f"; {out.mine_source.nunique()} mining sources")
