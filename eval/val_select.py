"""Choose the pretraining epoch on val-A, never on test (CLAUDE.md sections 5.3 and 7.7).

During pretraining, every few epochs the current backbone encodes the head-training photos and the val-A photos.
A linear probe (same protocol as eval/probes.py: StandardScaler + balanced LogisticRegression, C by grouped 5-fold
CV on train) is fitted on head-training and scored on val-A, plus a kNN score. The labels are the harmonised
findings + diseases (data/ontology.yaml), so RFMiD and RFMiD 2.0 form one task. A label is scored only with at
least MIN_VAL_POS val-A positives and 2 training positives.

The selection metric is val-A linear macro-AUROC. The test buckets are never read here.

The photos are preprocessed once (section 19.1 evaluation policy, via finish_view) into a uint8 array on disk,
read through a memory map, so a check costs GPU encoding plus small probes (about a minute) and little RAM.

Score saved encoders after the fact:
    python -m eval.val_select checkpoints/moco_plain_e15/encoder.pt checkpoints/moco_plain/epoch_090.pt ...
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.process import load_processed
from encoder.export import IMAGENET_MEAN, IMAGENET_STD, build_backbone, finish_view
from eval.probes import choose_C, knn_scores, probe

MIN_VAL_POS = 5
CACHE_DIR = os.path.join("outputs", "cache")
METRIC = "val_linear_macro_auroc"


def eval_rows(meta: pd.DataFrame) -> pd.DataFrame:
    rows = meta[meta.bucket.isin(["train", "val_a"]) & meta.fov_ok & meta.label_ok].copy()
    rows["targets"] = [sorted(set(f) | set(d)) for f, d in zip(rows.findings, rows.diseases)]
    return rows.sort_values(["bucket", "image_id"]).reset_index(drop=True)


def label_matrix(targets, vocab) -> np.ndarray:
    return np.array([[v in t for v in vocab] for t in targets], dtype=float)


def build_cache(rows: pd.DataFrame, stats: dict, size: int, path: str, root: str = "processed") -> np.memmap:
    """Preprocess every row once and store it as (N, 3, size, size) uint8 on disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.uint8, shape=(len(rows), 3, size, size))
    t = time.time()
    for i, (p, s) in enumerate(zip(rows.processed_path, rows.source)):
        rgb, mask = load_processed(os.path.join(root, p))
        arr[i] = (finish_view(rgb, mask, s, stats, size) * 255).round().byte().numpy()
    arr.flush()
    del arr
    os.replace(path + ".tmp", path)
    rows[["image_id"]].to_parquet(path.replace(".npy", "_index.parquet"))
    print(f"val-select cache: {len(rows)} photos at {size} px in {time.time() - t:.0f} s -> {path}")
    return np.load(path, mmap_mode="r")


def score_features(Xtr, Ytr, Xva, Yva, groups, vocab) -> dict:
    keep = [j for j in range(len(vocab)) if Yva[:, j].sum() >= MIN_VAL_POS and Ytr[:, j].sum() >= 2]
    Ytr_k, Yva_k = Ytr[:, keep], Yva[:, keep]
    C, _ = choose_C(Xtr, Ytr_k, groups)
    S = np.column_stack([probe(C).fit(Xtr, Ytr_k[:, i]).predict_proba(Xva)[:, 1] for i in range(len(keep))])
    K = knn_scores(Xtr, Ytr_k, Xva)
    lin = {vocab[j]: float(roc_auc_score(Yva_k[:, i], S[:, i])) for i, j in enumerate(keep)}
    knn = [roc_auc_score(Yva_k[:, i], K[:, i]) for i in range(len(keep))]
    return {METRIC: float(np.mean(list(lin.values()))), "val_knn_macro_auroc": float(np.mean(knn)),
            "val_C": C, "val_n_labels": len(keep), "val_per_label": lin}


class ValSelector:
    def __init__(self, meta: pd.DataFrame, stats: dict, size: int = 384, cache_dir: str = CACHE_DIR):
        rows = eval_rows(meta)
        path = os.path.join(cache_dir, f"valselect_{size}.npy")
        idx_path = path.replace(".npy", "_index.parquet")
        fresh = os.path.exists(path) and os.path.exists(idx_path) and \
            pd.read_parquet(idx_path).image_id.tolist() == rows.image_id.tolist()
        self.x = np.load(path, mmap_mode="r") if fresh else build_cache(rows, stats, size, path)
        self.vocab = sorted({t for ts in rows.targets for t in ts})
        self.tr = (rows.bucket == "train").values
        Y = label_matrix(rows.targets, self.vocab)
        self.Ytr, self.Yva = Y[self.tr], Y[~self.tr]
        self.groups = rows.dup_group.values[self.tr]
        self.mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    @torch.no_grad()
    def features(self, backbone: torch.nn.Module, device: str = "cuda", batch: int = 32) -> np.ndarray:
        """L2-normalised 2048-d backbone features in eval mode (BatchNorm uses running stats and is not updated).
        The caller's train/eval mode is restored afterwards."""
        was_training = backbone.training
        backbone.eval()
        mean, std = self.mean.to(device), self.std.to(device)
        out = []
        try:
            for i in range(0, len(self.x), batch):
                x = torch.from_numpy(np.array(self.x[i:i + batch])).to(device).float().div_(255)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                    f = backbone((x - mean) / std)
                out.append(F.normalize(f.float(), dim=1).cpu().numpy())
        finally:
            backbone.train(was_training)
        return np.concatenate(out)

    def score(self, backbone: torch.nn.Module, device: str = "cuda") -> dict:
        t = time.time()
        X = self.features(backbone, device)
        res = score_features(X[self.tr], self.Ytr, X[~self.tr], self.Yva, self.groups, self.vocab)
        res["val_seconds"] = round(time.time() - t, 1)
        return res


def load_backbone(path: str) -> torch.nn.Module:
    """Accepts a backbone state_dict (encoder.pt) or a full training checkpoint (last.pt / epoch_XXX.pt)."""
    net = build_backbone("imagenet")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if "model" in sd:
        pre = "q_enc.backbone."
        sd = {k[len(pre):]: v for k, v in sd["model"].items() if k.startswith(pre)}
    net.load_state_dict(sd)
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", nargs="*", help="encoder.pt or checkpoint files; 'imagenet' for the ImageNet start")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()
    meta = pd.read_parquet("data/meta.parquet")
    sel = ValSelector(meta, json.load(open("data/trace_preprocess.json")), args.size)
    print(f"val-A: {int((~sel.tr).sum())} photos, head-training: {int(sel.tr.sum())}")
    for w in args.weights:
        net = (build_backbone("imagenet") if w == "imagenet" else load_backbone(w)).cuda()
        r = sel.score(net)
        print(f"{w}: val-A linear {r[METRIC]:.4f}, kNN {r['val_knn_macro_auroc']:.4f} "
              f"({r['val_n_labels']} labels, C {r['val_C']}, {r['val_seconds']} s)")
