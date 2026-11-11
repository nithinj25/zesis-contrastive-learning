"""Leave-disease-out: recognise AION and CRVO with ZERO training photos (CLAUDE.md sections 9.6, 11, 14 phase 4).

    python -m eval.leave_disease_out --name trace_v2

Queries: every photo of a held-out disease (none was in the pool, head training, edge counts or calibration):
AION 30 (RFMiD + RFMiD 2.0), CRVO 79 (RFMiD + RFMiD 2.0 + JSIEC). Candidates: the graph's diseases (data diseases
with >= 5 head-training photos + the zero-shot ones). Metric: rank of the true held-out disease; top-1 and top-3
(chance = 1/D and 3/D). Two methods:
  graph       : Bayesian scorer on calibrated findings (graph/score.py), 0 photos of the disease.
  image K-shot: cosine to class means of global features (data diseases from head training; the held-out disease
                from K of its OTHER photos, other eyes only), 5 seeds. What the case index can do with K photos.
False alarm (val-A, which holds no AION or CRVO photo): share of photos whose top-1 is a zero-shot disease.
pi_min (19.8) is kept at 0.005 unless its val-A false alarm exceeds 5%; the sweep is printed as a sensitivity
check, and the choice never looks at the held-out photos.
Writes outputs/eval/leave_disease_out_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from eval.probes import load_features
from graph.build import build
from graph.score import explain, score

PI_GRID = (0.001, 0.005, 0.02)
SEEDS, KS = 5, (1, 5)


def findings_q(name: str, ids, cols) -> np.ndarray:
    frames = [pd.read_parquet(os.path.join("features", name, f"{b}_findings.parquet"))
              for b in ("train", "val_a", "val_b", "test", "ldo_test")]
    q = pd.concat(frames).drop_duplicates("image_id").set_index("image_id")
    return q.loc[ids, cols].values.astype(np.float64)


def ranks_of(scores: np.ndarray, col: int) -> np.ndarray:
    t = scores[:, col]
    return (scores > t[:, None]).sum(1) + 1


def topk(r: np.ndarray, k: int) -> float:
    return float((r <= k).mean())


def image_kshot(X, ids, meta, diseases, target, Xq, q_groups, K, seed):
    """Rank of `target` for each query by cosine to class means; target mean from K other-eye query photos."""
    rng = np.random.default_rng(seed)
    head = meta.loc[ids]
    protos = []
    for d in diseases:
        if d == target:
            protos.append(None)
            continue
        has = head.diseases.map(lambda x: d in list(x)).values
        protos.append(X[has].mean(0))
    col = diseases.index(target)
    out = []
    for i in range(len(Xq)):
        pool = np.where(q_groups != q_groups[i])[0]
        sup = rng.choice(pool, K, replace=False)
        P = np.stack([Xq[sup].mean(0) if p is None else p for p in protos])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        s = Xq[i] @ P.T
        out.append(int((s > s[col]).sum() + 1))
    return np.array(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    G = build(args.name)
    D, F = G["diseases"], G["findings"]
    print(f"graph: {len(D)} diseases {D}; zero-shot {G['zero_shot']}; {len(F)} findings; "
          f"{len(G['edges'])} cited edges (status {sorted(G['edges'].status.unique())})")

    va_ids = meta.index[(meta.bucket == "val_a") & meta.label_ok]
    q_va = findings_q(args.name, va_ids, F)
    sweep = {}
    for pm in PI_GRID:
        Gp = build(args.name, pi_min=pm)
        _, post, _ = score(q_va, Gp["p"], Gp["p_not"], Gp["prior"], Gp["r"])
        top = np.array(D)[post.argmax(1)]
        sweep[pm] = {"val_a_false_alarm": float(np.isin(top, G["zero_shot"]).mean())}
    pi_min = 0.005 if sweep[0.005]["val_a_false_alarm"] <= 0.05 else min(pm for pm in PI_GRID if sweep[pm]["val_a_false_alarm"] <= 0.05)
    G = build(args.name, pi_min=pi_min)

    f = load_features(args.name)
    X = f.attrs["X"] / np.linalg.norm(f.attrs["X"], axis=1, keepdims=True)
    row = f.set_index("image_id").row
    head_ids = meta.index[meta.in_heads & meta.label_ok]
    result = {"diseases": D, "pi_min": pi_min, "sweep": sweep, "targets": {}}
    for target in G["zero_shot"]:
        sel = meta[(meta.ldo_disease == target) & meta.label_ok & (meta.bucket != "excluded")]
        ids = [i for i in sel.index if i in row.index]
        q = findings_q(args.name, ids, F)
        s, post, terms = score(q, G["p"], G["p_not"], G["prior"], G["r"])
        col = D.index(target)
        r_graph = ranks_of(s, col)
        res = {"n": len(ids), "by_dataset": sel.loc[ids].dataset.value_counts().to_dict(),
               "graph": {"top1": topk(r_graph, 1), "top3": topk(r_graph, 3), "median_rank": float(np.median(r_graph)),
                         "median_posterior": float(np.median(post[:, col]))}}
        for pm in PI_GRID:
            Gp = build(args.name, pi_min=pm)
            sp, _, _ = score(q, Gp["p"], Gp["p_not"], Gp["prior"], Gp["r"])
            sweep[pm][f"{target}_top1"], sweep[pm][f"{target}_top3"] = topk(ranks_of(sp, col), 1), topk(ranks_of(sp, col), 3)
        Xq = X[row.loc[ids].values]
        groups = sel.loc[ids].dup_group.fillna(-1).astype(str).values
        groups = np.where(groups == "-1.0", np.array(ids), groups)
        for K in KS:
            rr = np.concatenate([image_kshot(X[row.loc[head_ids].values], head_ids, meta, D, target, Xq, groups, K, sd)
                                 for sd in range(SEEDS)])
            res[f"image_{K}shot"] = {"top1": topk(rr, 1), "top3": topk(rr, 3), "median_rank": float(np.median(rr))}
        i0 = int(np.argmax(post[:, col]))
        fr, ag = explain(terms[i0, col], F)
        res["example"] = {"image_id": ids[i0], "posterior": float(post[i0, col]), "for": fr, "against": ag,
                          "top3": [D[j] for j in np.argsort(-s[i0])[:3]]}
        res["predicted_top1"] = pd.Series(np.array(D)[s.argmax(1)]).value_counts().to_dict()
        result["targets"][target] = res

    os.makedirs("outputs/eval", exist_ok=True)
    json.dump(result, open(f"outputs/eval/leave_disease_out_{args.name}.json", "w"), indent=1, default=float)
    print(f"pi_min {pi_min}; val-A false alarm (top-1 is a zero-shot disease, val-A has none): "
          f"{sweep[pi_min]['val_a_false_alarm']:.1%}; chance top-1 {1 / len(D):.3f}, top-3 {3 / len(D):.3f}")
    for t, r in result["targets"].items():
        g = r["graph"]
        print(f"\n{t}: {r['n']} photos {r['by_dataset']}")
        print(f"  graph, 0 photos : top-1 {g['top1']:.3f}  top-3 {g['top3']:.3f}  median rank {g['median_rank']:.0f}  "
              f"median posterior {g['median_posterior']:.3f}")
        for K in KS:
            m = r[f"image_{K}shot"]
            print(f"  image, {K} photo{'s' if K > 1 else ' '}: top-1 {m['top1']:.3f}  top-3 {m['top3']:.3f}  median rank {m['median_rank']:.0f}")
        print(f"  graph top-1 calls: {r['predicted_top1']}")
        e = r["example"]
        print(f"  best example {e['image_id']}: p({t}) {e['posterior']:.2f}, top-3 {e['top3']}; for {e['for']}; against {e['against']}")
    print("\npi_min sensitivity (not used to choose):")
    for pm, v in sweep.items():
        print(f"  {pm}: " + ", ".join(f"{k} {x:.3f}" for k, x in v.items()))
