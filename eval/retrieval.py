"""Case retrieval, "show me the 5 most similar confirmed cases" (a required deliverable; CLAUDE.md sections 10.3, 10.5).

Run from the repo root:  python -m eval.retrieval [--encoders imagenet_r50 moco_plain_e15 trace_v2] [--figure trace_v2]
Case library = labelled RFMiD + RFMiD 2.0 head-training photos + the rare-disease photos (AION, CRVO, RP), i.e. the
photos the deployed rare-disease bank holds. Search = cosine on frozen, L2-normalised global features (FAISS flat
inner product, the trust layer's index). A query's own eye (same dup_group) is never returned.

Metric: precision@5 = share of the 5 returned cases whose diagnoses include the query's diagnosis (single-diagnosis
queries). Three query sets: RFMiD + RFMiD 2.0 test photos against head training only (common diseases); each rare
photo against the library minus its own eye (leave one eye out); JSIEC photos (a dataset never in the library).
Chance = the class's share of the library. Writes outputs/eval/retrieval.json and, for --figure, a picture of 4
random queries (seed 0, not hand-picked) with their 5 cases: outputs/figures/retrieval_<encoder>.png.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eval.probes import load_features
from trust.ood import KnnOOD

COMMON = ["normal", "DR", "myopia", "ARMD", "BRVO", "CSR"]
RARE = ["AION", "CRVO", "RP"]
JSIEC = ["DR", "BRVO", "CRVO", "RP"]
BUCKETS = ("train", "val_a", "val_b", "test", "ldo_test", "excluded")


def diagnosis(diseases) -> str:
    """One label per photo for display and queries: the single diagnosis, 'several' or 'findings only'."""
    return diseases[0] if len(diseases) == 1 else ("several" if len(diseases) else "findings only")


def precision_at_k(query_dx: np.ndarray, lib_diseases: list, nbr: np.ndarray) -> np.ndarray:
    """query_dx: (n,) diagnosis per query; nbr: (n, k) library rows. Share of neighbours whose disease list holds it."""
    return np.array([np.mean([q in lib_diseases[j] for j in row]) for q, row in zip(query_dx, nbr)])


def search(index: KnnOOD, Xq: np.ndarray, q_block: np.ndarray, lib_block: np.ndarray, k: int = 5) -> tuple:
    """k nearest library rows per query, skipping rows from the query's own eye (block id)."""
    sims, idx = index.neighbours(Xq, k + 20)
    out_i, out_s = np.zeros((len(Xq), k), int), np.zeros((len(Xq), k))
    for n in range(len(Xq)):
        keep = [(i, s) for i, s in zip(idx[n], sims[n]) if lib_block[i] != q_block[n]][:k]
        assert len(keep) == k, "more than 20 same-eye copies in the library"
        out_i[n], out_s[n] = [i for i, _ in keep], [s for _, s in keep]
    return out_i, out_s


def setup(meta: pd.DataFrame):
    lab = meta[meta.dataset.isin(["rfmid", "rfmid2"]) & meta.label_ok & meta.fov_ok].copy()
    lab["dx"] = [diagnosis(list(x)) for x in lab.diseases]
    lab["is_rare"] = [any(d in RARE for d in x) for x in lab.diseases]
    lib = lab[(lab.bucket == "train") | (lab.is_rare & (lab.bucket != "excluded"))].reset_index(drop=True)
    test = lab[(lab.bucket == "test") & lab.dx.isin(COMMON)].reset_index(drop=True)
    rare_q = lab[lab.dx.isin(RARE) & (lab.bucket != "excluded")].reset_index(drop=True)
    js = meta[(meta.dataset == "jsiec") & meta.fov_ok & (meta.bucket == "test")].copy()
    js["dx"] = [diagnosis(list(x)) for x in js.diseases]
    js = js[js.dx.isin(JSIEC)].reset_index(drop=True)
    return lib, test, rare_q, js


def block_ids(*frames):
    """Same-eye group id per row across frames (dup_group, else the photo itself)."""
    keys = pd.concat([f.dup_group.fillna(f.image_id) for f in frames], ignore_index=True)
    codes = pd.factorize(keys)[0]
    out, s = [], 0
    for f in frames:
        out.append(codes[s:s + len(f)])
        s += len(f)
    return out


def evaluate(enc: str, meta: pd.DataFrame) -> tuple[dict, dict]:
    lib, test, rare_q, js = setup(meta)
    f = load_features(enc, BUCKETS)
    row = f.set_index("image_id").row
    X = f.attrs["X"]
    get = lambda d: X[row.loc[d.image_id].values]
    b_lib, b_test, b_rare, b_js = block_ids(lib, test, rare_q, js)
    lib_dis = [list(x) for x in lib.diseases]
    common_lib = ~lib.is_rare.values  # common queries search head training only (no rare photos, no test photos)
    full = KnnOOD().fit(get(lib))
    part = KnnOOD().fit(get(lib)[common_lib])
    sub_dis = [d for d, c in zip(lib_dis, common_lib) if c]

    res, nbrs = {"encoder": enc, "library": len(lib)}, {}
    i, s = search(part, get(test), b_test, b_lib[common_lib])
    p = precision_at_k(test.dx.values, sub_dis, i)
    res["rfmid_test"] = {d: float(p[test.dx.values == d].mean()) for d in COMMON}
    res["rfmid_test_macro"] = float(np.mean(list(res["rfmid_test"].values())))
    nbrs["rfmid_test"] = (test, np.flatnonzero(common_lib)[i], s)

    i, s = search(full, get(rare_q), b_rare, b_lib)
    p = precision_at_k(rare_q.dx.values, lib_dis, i)
    res["rare_leave_one_eye_out"] = {d: float(p[rare_q.dx.values == d].mean()) for d in RARE}
    i, s = search(full, get(js), b_js, b_lib)
    p = precision_at_k(js.dx.values, lib_dis, i)
    res["jsiec"] = {d: float(p[js.dx.values == d].mean()) for d in JSIEC}
    nbrs["jsiec"] = (js, i, s)
    res["chance"] = {d: float(np.mean([d in x for x in lib_dis])) for d in sorted(set(COMMON + RARE))}
    res["n_queries"] = {"rfmid_test": test.dx.value_counts().to_dict(), "rare": rare_q.dx.value_counts().to_dict(),
                        "jsiec": js.dx.value_counts().to_dict()}
    return res, nbrs | {"lib": lib}


def figure(enc: str, nbrs: dict, out: str, seed: int = 0):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from data.process import load_processed

    rng = np.random.default_rng(seed)
    lib = nbrs["lib"]
    picks = []
    for key, dx in [("rfmid_test", "DR"), ("rfmid_test", "myopia"), ("jsiec", "RP"), ("jsiec", "CRVO")]:
        q, idx, sims = nbrs[key]
        n = rng.choice(np.flatnonzero(q.dx.values == dx))
        picks.append((q.iloc[n], idx[n], sims[n], "RFMiD test" if key == "rfmid_test" else "JSIEC (never in library)"))

    def thumb(path):
        rgb, mask = load_processed(os.path.join("processed", path))
        return np.where(mask[..., None] > 0, rgb, 252).astype(np.uint8)

    fig, axes = plt.subplots(len(picks), 6, figsize=(15, 2.95 * len(picks)), dpi=130)
    for r, (q, idx, sims, src) in enumerate(picks):
        ax = axes[r, 0]
        ax.imshow(thumb(q.processed_path))
        ax.set_title(f"QUERY · {src}\n{q.dx}", fontsize=9.5, color="#0b0b0b", loc="left")
        for c, (j, s) in enumerate(zip(idx, sims), start=1):
            case = lib.iloc[j]
            hit = q.dx in list(case.diseases)
            axes[r, c].imshow(thumb(case.processed_path))
            axes[r, c].set_title(f"{'✓' if hit else '✗'} {case.dx}\ncosine {s:.3f}", fontsize=9.5, loc="left",
                                 color="#0b0b0b" if hit else "#52514e")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
    fig.suptitle(f"{enc}: 5 most similar confirmed cases for 4 random queries (seed {seed}; ✓ = same diagnosis, "
                 "query's own eye excluded)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, facecolor="#fcfcfb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["imagenet_r50", "moco_plain_e15", "trace_v2"])
    ap.add_argument("--figure", default="trace_v2")
    ap.add_argument("--out", default="outputs/eval/retrieval.json")
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet")
    results = []
    for enc in args.encoders:
        res, nbrs = evaluate(enc, meta)
        results.append(res)
        print(f"{enc}: RFMiD macro {res['rfmid_test_macro']:.3f}  rare {res['rare_leave_one_eye_out']}  "
              f"JSIEC {res['jsiec']}")
        if enc == args.figure:
            os.makedirs("outputs/figures", exist_ok=True)
            figure(enc, nbrs, f"outputs/figures/retrieval_{enc}.png")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
