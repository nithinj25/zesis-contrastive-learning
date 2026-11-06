"""Find the same eye across images and datasets (CLAUDE.md section 5.4, section 6 step 2).

Run from the repo root, after data.ingest:  python -m data.dedupe
Reads data/raw_meta.parquet. Writes data/dup_pairs.parquet and data/dedupe.parquet.

pHash does not work on fundus photos: the black border and shared layout dominate it, so it
matched thousands of unrelated eyes. So we compare fine retinal detail (vessels) instead.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from data.preprocess import find_fov, standardise_geometry

COARSE_PX = 64
FINE_PX = 256
SHIFT_PX = 8  # small misalignment allowed between two photos
TOP_K = 50  # neighbours per image, in each dataset, to check at high resolution
TOP_K_BIG = 150  # in a dataset bigger than BIG_GROUP, where look-alikes crowd the list
BIG_GROUP = 5000
COARSE_MIN = 0.7  # below this, a pair is not checked

MATCH = 0.6  # at or above: count as a match when protecting a held-out split. Cautious on purpose.
SAME_EYE = 0.8  # at or above: these were the same eye on the contact sheets

SPLIT_RANK = {"test": 2, "val": 1, "train": 0, "pool": 0}  # pool = pretraining only, like train

_yy, _xx = np.ogrid[:COARSE_PX, :COARSE_PX]
_centre = (COARSE_PX - 1) / 2
_KEEP = ((_xx - _centre) ** 2 + (_yy - _centre) ** 2) <= (0.82 * COARSE_PX / 2) ** 2


def _detail(gray: np.ndarray, fine_sigma: float, wide_sigma: float) -> np.ndarray:
    """Band-pass filter: keeps vessels and lesions, removes overall brightness and colour."""
    gray = gray.astype(np.float32)
    return cv2.GaussianBlur(gray, (0, 0), fine_sigma) - cv2.GaussianBlur(gray, (0, 0), wide_sigma)


def coarse_signature(std_rgb: np.ndarray) -> np.ndarray:
    """Short unit vector for the fast all-pairs search. Inner product = correlation."""
    small = cv2.resize(std_rgb[..., 1], (COARSE_PX, COARSE_PX), interpolation=cv2.INTER_AREA)
    v = _detail(small, 0.8, 4.0)[_KEEP]
    v = v - v.mean()
    return v / (np.linalg.norm(v) + 1e-6)


def fine_signature(std_rgb: np.ndarray) -> np.ndarray:
    """High-resolution detail of the central retina, always inside the circle."""
    small = cv2.resize(std_rgb[..., 1], (FINE_PX, FINE_PX), interpolation=cv2.INTER_AREA)
    a, b = FINE_PX // 4 - SHIFT_PX, 3 * FINE_PX // 4 + SHIFT_PX
    return _detail(small, 1.0, 6.0)[a:b, a:b].astype(np.float16)


MIN_DETAIL = 0.3  # std of the fine signature; real photos measured >= 0.77, a blank white photo 0.0


def match_score(a: np.ndarray, b: np.ndarray) -> float:
    """Best correlation over small shifts, tried in both directions. 1 = identical detail.

    A photo with no detail (blank, fully over-exposed) has no vessel pattern to compare. Correlation is
    undefined there and came out as 1.0: one white EyePACS photo "matched" 23 different Paraguay eyes.
    So it scores 0.
    """
    a, b = a.astype(np.float32), b.astype(np.float32)
    m = SHIFT_PX
    if a[m:-m, m:-m].std() < MIN_DETAIL or b[m:-m, m:-m].std() < MIN_DETAIL:
        return 0.0
    return float(
        max(
            cv2.matchTemplate(b, a[m:-m, m:-m], cv2.TM_CCOEFF_NORMED).max(),
            cv2.matchTemplate(a, b[m:-m, m:-m], cv2.TM_CCOEFF_NORMED).max(),
        )
    )


def candidate_pairs(
    coarse: np.ndarray,
    top_k: int = TOP_K,
    minimum: float = COARSE_MIN,
    groups: np.ndarray | None = None,
    queries: np.ndarray | None = None,
    top_k_big: int = TOP_K_BIG,
    big_group: int = BIG_GROUP,
) -> np.ndarray:
    """Closest neighbours as (i, j, similarity) rows with i < j, using the GPU if present.

    groups (the dataset of each image): the neighbours are taken inside every group separately, so a
    big dataset of look-alikes cannot push a real copy in a small one out of the list (adding APTOS
    lost one real RFMiD pair that way). A group bigger than big_group gets top_k_big neighbours.
    queries (True/False per image): only these images search. Leaving out pool images skips
    pool-pool pairs, which can never cause a drop; a pool copy of a test image is still found,
    because the test image searches the pool.
    """
    import torch  # here, not at the top: worker processes must not each load the GPU libraries

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.tensor(coarse, dtype=torch.float32, device=device)
    groups = np.zeros(len(x), int) if groups is None else np.asarray(groups)
    query_rows = np.arange(len(x)) if queries is None else np.flatnonzero(queries)
    columns = [torch.tensor(np.flatnonzero(groups == g), device=device) for g in np.unique(groups)]
    found: dict[tuple[int, int], float] = {}
    for start in range(0, len(query_rows), 512):
        rows_here = torch.tensor(query_rows[start : start + 512], device=device)
        sim = x[rows_here] @ x.T
        sim[torch.arange(len(rows_here), device=device), rows_here] = -1  # never match an image with itself
        for cols in columns:
            k = min(top_k_big if len(cols) > big_group else top_k, len(cols))
            values, picked = sim[:, cols].topk(k, dim=1)
            values, indices = values.cpu().numpy(), cols[picked].cpu().numpy()
            for row, i in enumerate(rows_here.cpu().numpy()):
                for value, j in zip(values[row], indices[row]):
                    if value >= minimum and int(j) != int(i):
                        found[(min(int(i), int(j)), max(int(i), int(j)))] = float(value)
    return np.array([(i, j, v) for (i, j), v in found.items()], dtype=np.float64).reshape(-1, 3)


def same_eye_groups(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    """Union-find: if A matches B and B matches C, all three share one group number."""
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[root(a)] = root(b)
    roots = [root(i) for i in range(n)]
    numbers = {r: g for g, r in enumerate(dict.fromkeys(roots))}
    return np.array([numbers[r] for r in roots])


def decide_drops(
    splits: list[str], pairs: list[tuple[int, int, float]], datasets: list[str] | None = None
) -> dict[int, str]:
    """Which images to remove, so a held-out split never shares an eye with anything used earlier.

    Different splits: the lower-ranked image goes (test beats val beats train/pool).
    Same split, same dataset: inside test or val, a same-eye copy goes, or it would be counted twice.
    Same split, different datasets (IDRiD grading vs IDRiD lesion masks): both stay, because they are
    scored in different tests; they share a dup_group, so a pooled count can take each eye once.
    Train-train copies stay; they do not leak.
    """
    datasets = datasets or [""] * len(splits)
    drops: dict[int, str] = {}
    for i, j, score in pairs:
        if score < MATCH:
            continue
        ri, rj = SPLIT_RANK[splits[i]], SPLIT_RANK[splits[j]]
        if ri != rj:
            low, high = (i, j) if ri < rj else (j, i)
            drops.setdefault(low, f"matches_{splits[high]}")
        elif ri > 0 and score >= SAME_EYE and datasets[i] == datasets[j]:
            drops.setdefault(max(i, j), f"copy_within_{splits[i]}")
    return drops


def _signatures(path: str):
    cv2.setNumThreads(1)
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    fov = find_fov(rgb)
    if fov is None:
        return None
    std, _ = standardise_geometry(rgb, fov)
    return coarse_signature(std), fine_signature(std)


def _score_chunk(job) -> list[float]:
    cv2.setNumThreads(1)
    return [match_score(a, b) for a, b in job]


def run(meta: pd.DataFrame, raw_root: str = "raw", workers: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [os.path.join(raw_root, p) for p in meta.path]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_signatures, paths, chunksize=16))
        usable = [i for i, r in enumerate(results) if r is not None]
        coarse = np.stack([results[i][0] for i in usable])
        fine = [results[i][1] for i in usable]
        split = meta.split.values[usable]
        cand = candidate_pairs(coarse, groups=meta.dataset.values[usable], queries=split != "pool")
        # A pair of two training-only images (pool or train) where one is pool can never cause a drop
        # or matter for grouping, so it is not scored. Train-train stays: same-eye groups keep CV folds clean.
        a, b = cand[:, 0].astype(int), cand[:, 1].astype(int)
        rank = np.array([SPLIT_RANK[s] for s in split])
        keep = (np.maximum(rank[a], rank[b]) > 0) | ((split[a] != "pool") & (split[b] != "pool"))
        cand, a, b = cand[keep], a[keep], b[keep]
        jobs = [(fine[x], fine[y]) for x, y in zip(a, b)]
        chunks = [jobs[i : i + 256] for i in range(0, len(jobs), 256)]
        scores = [s for chunk in ex.map(_score_chunk, chunks) for s in chunk]

    index = np.array(usable)
    pairs = pd.DataFrame(
        {"i": index[a], "j": index[b], "coarse": cand[:, 2], "score": scores}
    )
    pairs["image_a"] = meta.image_id.values[pairs.i]
    pairs["image_b"] = meta.image_id.values[pairs.j]

    strong = pairs[pairs.score >= SAME_EYE]
    groups = same_eye_groups(len(meta), list(zip(strong.i, strong.j)))
    drops = decide_drops(meta.split.tolist(), list(zip(pairs.i, pairs.j, pairs.score)), meta.dataset.tolist())
    out = meta[["image_id", "dataset", "split", "source"]].copy()
    out["dup_group"] = groups
    out["dup_drop"] = [i in drops for i in range(len(meta))]
    out["dup_reason"] = [drops.get(i, "") for i in range(len(meta))]
    out["signature_ok"] = [r is not None for r in results]
    return pairs, out


if __name__ == "__main__":
    pair_table, result = run(pd.read_parquet("data/raw_meta.parquet"))
    pair_table.to_parquet("data/dup_pairs.parquet")
    result.to_parquet("data/dedupe.parquet")
    print(f"images {len(result)}, no signature {(~result.signature_ok).sum()}, pairs checked {len(pair_table)}")
    print(f"pairs with score >= {MATCH}: {(pair_table.score >= MATCH).sum()}, >= {SAME_EYE}: {(pair_table.score >= SAME_EYE).sum()}")
    sizes = result.groupby("dup_group").size()
    print(f"same-eye groups with 2+ images: {(sizes > 1).sum()}, largest {sizes.max()}")
    print("\ndrops by split and reason:")
    print(result[result.dup_drop].groupby(["dataset", "split", "dup_reason"]).size().to_string())
