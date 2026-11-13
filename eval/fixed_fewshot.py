"""Approach 3's fixed few-shot exam (trace-lsnm-, data/protocols/trace_fewshot_v1, frozen 11 Nov 2026) on our features.

Run from the repo root:  python -m eval.fixed_fewshot <protocol dir> trace_v2 imagenet_r50 moco_plain_e15 [...]
Ported from their scripts/run_dinov2_fewshot_benchmark.py and trace_protocols/evaluation.py, unchanged in substance:
  * train records = their 1,917 RFMiD training records (base_adapter_pool.csv); queries = their 639 fixed RFMiD
    validation records (fixed_query_records.csv); labels = RFMiD's own labels, in RFMiD's class order;
  * heads: prototype margin, kNN class margin (m in 1, 3, 5 <= K), balanced liblinear logistic (C 0.01, 0.1, 1), all
    on L2-normalised features; support protocols "abundant" (K positives + every training negative) and "strict"
    (K positives + the episode's K negatives);
  * the head per (protocol, K) is locked on base-label episodes (replicates 0-9) by macro AUPRC lift over query
    prevalence (ties: smaller parameter), then the 1,650 novel-label episodes (11 labels x K 1/3/5 x 50) are scored;
    macro AUPRC = mean over labels of the mean over episodes.
Our features come from features/<encoder>/ (buckets train, val_a, val_b, ldo_test, excluded cover every RFMiD
train / val photo). Writes outputs/fixed_fewshot/<encoder>.json (+ episodes CSV).

Caveat: their queries are RFMiD val. TRACE's checkpoint was chosen on val-A, which holds about half of those photos
(by a 17-label linear probe, not these episodes), so TRACE has a mild selection advantage over DINOv2 / RETFound /
FLAIR here; ImageNet and plain MoCo have none. supervised_r50 was trained on RFMiD train labels, novel labels included.
"""
from __future__ import annotations

import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

KNN = (1, 3, 5)
CS = (0.01, 0.1, 1.0)
TUNING_REPLICATES = 10
BUCKETS = ("train", "val_a", "val_b", "ldo_test", "excluded", "test")


def l2(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def prototype_scores(S, pos, neg, Q):
    p, n = l2(S[pos].mean(0, keepdims=True))[0], l2(S[neg].mean(0, keepdims=True))[0]
    return Q @ p - Q @ n


def knn_scores(S, pos, neg, Q, m):
    ps, ns = Q @ S[pos].T, Q @ S[neg].T
    top = lambda a: np.partition(a, a.shape[1] - m, axis=1)[:, -m:].mean(1)
    return top(ps) - top(ns)


def logistic_scores(S, pos, neg, Q, C, seed):
    sel = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos), np.uint8), np.zeros(len(neg), np.uint8)]
    m = LogisticRegression(C=C, class_weight="balanced", solver="liblinear", dual=len(sel) < S.shape[1],
                           max_iter=500, random_state=int(seed))
    return m.fit(S[sel], y).decision_function(Q).astype(np.float32)


def score(head, param, S, pos, neg, Q, seed):
    if head == "prototype":
        return prototype_scores(S, pos, neg, Q)
    if head == "knn":
        return knn_scores(S, pos, neg, Q, int(param))
    return logistic_scores(S, pos, neg, Q, float(param), seed)


def our_id(rec: str) -> str:
    ds, split, name = rec.split(":")
    assert ds == "rfmid", rec
    return f"rfmid_{ {'training': 'train', 'validation': 'val', 'testing': 'test'}[split]}_{int(re.sub(r'\\D', '', name))}"


def load(protocol_dir: str, encoder: str):
    from eval.probes import load_features

    train_ids = pd.read_csv(os.path.join(protocol_dir, "base_adapter_pool.csv")).record_id.tolist()
    query_ids = pd.read_csv(os.path.join(protocol_dir, "fixed_query_records.csv")).record_id.tolist()
    classes = [r.label for r in pd.read_csv(os.path.join(protocol_dir, "label_eligibility.csv"))
               .sort_values("class_index").itertuples()]
    meta = pd.read_parquet("data/meta.parquet").set_index("image_id")
    f = load_features(encoder, BUCKETS)
    row = f.set_index("image_id").row
    X = f.attrs["X"]

    def block(ids):
        ours = [our_id(r) for r in ids]
        missing = [o for o in ours if o not in row.index]
        assert not missing, f"{encoder}: {len(missing)} records without features, e.g. {missing[:3]}"
        Y = np.array([[c in set(meta.loc[o].labels) for c in classes] for o in ours], np.uint8)
        return l2(X[row.loc[ours].values]), Y

    S, Ys = block(train_ids)
    Q, Yq = block(query_ids)
    return S, Ys, Q, Yq, {r: i for i, r in enumerate(train_ids)}, classes


def run_episodes(ep: pd.DataFrame, S, Ys, Q, Yq, index, heads_for, jobs=6):
    neg_pool = {c: np.flatnonzero(Ys[:, c] == 0) for c in range(Ys.shape[1])}

    def one(e):
        pos = np.array([index[r] for r in json.loads(e.support_positive_record_ids_json)])
        strict = np.array([index[r] for r in json.loads(e.strict_support_negative_record_ids_json)])
        y = Yq[:, e.class_index]
        prev = float(y.mean())
        out = []
        for proto, neg in (("abundant_negatives", neg_pool[e.class_index]), ("strict_k_negatives", strict)):
            for head, param in heads_for(proto, e.k):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    ap = float(average_precision_score(y, score(head, param, S, pos, neg, Q, e.seed)))
                out.append({"episode_id": e.episode_id, "label": e.label, "k": int(e.k), "replicate": int(e.replicate),
                            "support_protocol": proto, "head": head, "parameter": "none" if param is None else str(param),
                            "query_prevalence": prev, "auprc": ap, "lift": ap - prev})
        return out

    rows = Parallel(n_jobs=jobs)(delayed(one)(e) for e in ep.itertuples(index=False))
    return pd.DataFrame([r for rs in rows for r in rs])


def macro(rows: pd.DataFrame) -> pd.DataFrame:
    per = rows.groupby(["head", "parameter", "support_protocol", "k", "label"], as_index=False)[["query_prevalence", "auprc", "lift"]].mean()
    return per.groupby(["head", "parameter", "support_protocol", "k"], as_index=False)[["query_prevalence", "auprc", "lift"]].mean()


def lock_heads(tuning_macro: pd.DataFrame) -> dict:
    locked = {}
    for (proto, k), g in tuning_macro.groupby(["support_protocol", "k"]):
        best = [hg.sort_values(["lift", "parameter"], ascending=[False, True]).iloc[0] for _, hg in g.groupby("head")]
        sel = max(best, key=lambda r: float(r["lift"]))
        p = None if sel["parameter"] == "none" else (int(float(sel["parameter"])) if sel["head"] == "knn" else float(sel["parameter"]))
        locked[(proto, int(k))] = (sel["head"], p)
    return locked


def evaluate(protocol_dir: str, encoder: str, shuffle_seed: int | None = None) -> dict:
    S, Ys, Q, Yq, index, classes = load(protocol_dir, encoder)
    if shuffle_seed is not None:  # their control: shuffled query labels must give AUPRC ~ prevalence
        Yq = Yq[np.random.default_rng(shuffle_seed).permutation(len(Yq))]
    base = pd.read_csv(os.path.join(protocol_dir, "base_tuning_episode_plan.csv"))
    base = base[base.replicate < TUNING_REPLICATES]
    novel = pd.read_csv(os.path.join(protocol_dir, "episode_plan.csv"))

    def all_heads(proto, k):
        return [("prototype", None)] + [("knn", m) for m in KNN if m <= k] + [("logistic", c) for c in CS]

    tuning = macro(run_episodes(base, S, Ys, Q, Yq, index, all_heads))
    locked = lock_heads(tuning)
    rows = run_episodes(novel, S, Ys, Q, Yq, index, lambda proto, k: [locked[(proto, int(k))]])
    per_label = rows.groupby(["support_protocol", "k", "label"]).auprc.mean().reset_index()
    res = {"encoder": encoder, "shuffled": shuffle_seed is not None,
           "locked_heads": {f"{p}_k{k}": list(v) for (p, k), v in locked.items()}, "macro_auprc": {}, "macro_lift": {},
           "per_label": {}}
    for (proto, k), g in rows.groupby(["support_protocol", "k"]):
        m = macro(g)
        res["macro_auprc"][f"{proto}_k{k}"] = float(m.auprc.iloc[0])
        res["macro_lift"][f"{proto}_k{k}"] = float(m.lift.iloc[0])
        res["per_label"][f"{proto}_k{k}"] = per_label[(per_label.support_protocol == proto) & (per_label.k == k)].set_index("label").auprc.round(4).to_dict()
    return res, rows


def main():
    protocol_dir, encoders = sys.argv[1], sys.argv[2:]
    os.makedirs("outputs/fixed_fewshot", exist_ok=True)
    for enc in encoders:
        shuffle = enc.endswith(":shuffled")
        name = enc.split(":")[0]
        res, rows = evaluate(protocol_dir, name, shuffle_seed=0 if shuffle else None)
        tag = name + ("_shuffled" if shuffle else "")
        json.dump(res, open(f"outputs/fixed_fewshot/{tag}.json", "w"), indent=1)
        rows.to_csv(f"outputs/fixed_fewshot/{tag}_episodes.csv", index=False)
        a = res["macro_auprc"]
        print(f"{tag:22s} abundant K1/3/5 {a['abundant_negatives_k1']:.4f} {a['abundant_negatives_k3']:.4f} "
              f"{a['abundant_negatives_k5']:.4f}   strict {a['strict_k_negatives_k1']:.4f} {a['strict_k_negatives_k3']:.4f} "
              f"{a['strict_k_negatives_k5']:.4f}   heads {res['locked_heads']}", flush=True)


if __name__ == "__main__":
    main()
