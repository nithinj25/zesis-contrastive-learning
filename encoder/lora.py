"""PEFT adapter (barrier 04; CLAUDE.md section 7.8): LoRA on the frozen TRACE encoder's layer4 convolutions.

Run from the repo root:  python -m encoder.lora [--fractions 0.01 0.1 1.0]
The TRACE backbone stays frozen (weights and BatchNorm statistics). Each layer4 convolution gets a low-rank branch,
out = conv(x) + (alpha / r) * B(A(x)): A has the conv's kernel and stride with r output channels, B is a 1x1 conv
back to the full width and starts at zero, so before training the adapted model equals the frozen one exactly.
Trained with BCE (pos_weight capped at 50, as encoder/supervised.py) on 1 / 10 / 100% of RFMiD's training photos,
drawn by multi-label stratified sampling (section 19.7); 3 draws for 1% and 10%, 1 for 100%. The step is chosen on
val-A. Against it, on exactly the same photos: the frozen-feature linear probe (StandardScaler + logistic, C = 0.01,
the value cross-validation picked in every earlier run). Both are scored on RFMiD test, on the labels with >= 2
positives in that draw and >= 5 test positives. Writes outputs/eval/lora_trace_v2.json.

Deviation from section 7.8: written here (~30 lines) instead of the `peft` library, whose dependencies (transformers,
accelerate) would be installed into an environment where a numpy upgrade already broke pandas once (13 Nov).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class LoRAConv2d(nn.Module):
    """A frozen Conv2d plus a trainable rank-r branch with the same kernel, stride and padding."""

    def __init__(self, base: nn.Conv2d, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.A = nn.Conv2d(base.in_channels, r, base.kernel_size, base.stride, base.padding, base.dilation, bias=False)
        self.B = nn.Conv2d(r, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + self.scale * self.B(self.A(x))


def add_lora(backbone: nn.Module, r: int = 8, alpha: float = 16.0, layer: str = "layer4") -> list[nn.Parameter]:
    """Freeze the backbone and wrap every Conv2d inside `layer` (including the downsample). Returns the new params."""
    backbone.requires_grad_(False)
    block = getattr(backbone, layer)
    targets = [(name, m) for name, m in block.named_modules() if isinstance(m, nn.Conv2d)]
    for name, conv in targets:
        parent = block
        *path, last = name.split(".")
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, last, LoRAConv2d(conv, r, alpha))
    return [p for n, p in backbone.named_parameters() if p.requires_grad]


def sample_fraction(Y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Row indices of a multi-label stratified sample (all rows for frac = 1)."""
    if frac >= 1.0:
        return np.arange(len(Y))
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

    split = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=frac, test_size=1 - frac, random_state=seed)
    idx, _ = next(split.split(np.zeros(len(Y)), Y))
    return np.sort(idx)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from augment.loader import ViewLoader
    from data.process import load_processed
    from encoder.export import IMAGENET_MEAN, IMAGENET_STD, build_backbone, finish_view
    from encoder.supervised import label_matrix, lr_at, macro_auroc, pos_weight
    from eval.probes import load_features

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="trace_v2")
    ap.add_argument("--fractions", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500, help="optimiser steps per run (fewer for tiny draws)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.8)
    dev = "cuda"
    meta = pd.read_parquet("data/meta.parquet")
    stats = json.load(open("data/trace_preprocess.json"))
    pick = lambda b: meta[(meta.bucket == b) & (meta.dataset == "rfmid") & meta.label_ok & meta.fov_ok].reset_index(drop=True)
    tr, va, te = pick("train"), pick("val_a"), pick("test")
    scored = json.load(open("outputs/eval/trace_v1_384.json"))["rfmid_labels_scored"]
    Ytr, Yva, Yte = (label_matrix(d.labels, scored) for d in (tr, va, te))

    f = load_features(args.encoder, ("train", "test"))
    row = f.set_index("image_id").row
    Ftr, Fte = f.attrs["X"][row.loc[tr.image_id].values], f.attrs["X"][row.loc[te.image_id].values]

    def eval_set(rows):
        return torch.stack([(finish_view(*load_processed(os.path.join("processed", p)), s, stats, 384) * 255).round().to(torch.uint8)
                            for p, s in zip(rows.processed_path, rows.source)])

    Xva, Xte = eval_set(va), eval_set(te)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)

    def run_lora(sub: np.ndarray, cols: list[int], seed: int) -> np.ndarray:
        torch.manual_seed(seed)
        bb = build_backbone("random")
        bb.load_state_dict(torch.load(f"checkpoints/{args.encoder}/encoder.pt", weights_only=True))
        lora = add_lora(bb, args.rank)
        head = nn.Linear(2048, len(cols))
        bb, head = bb.to(dev).eval(), head.to(dev)  # eval(): BatchNorm keeps the frozen encoder's statistics
        opt = torch.optim.AdamW([{"params": lora, "lr": 5e-4}, {"params": head.parameters(), "lr": 1e-3}], weight_decay=0.01)
        base = [g["lr"] for g in opt.param_groups]
        Y = torch.from_numpy(Ytr[sub][:, cols])
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight(Ytr[sub][:, cols])).to(dev))
        batch = min(args.batch, len(sub) // 2)
        loader = ViewLoader(tr.processed_path.values[sub].tolist(), batch, args.workers, seed=seed, size=384,
                            policy="retina", sources=tr.source.values[sub].tolist(), stats=stats)
        total = min(args.steps, max(300, 40 * len(loader)))
        every, best, best_state, step, ep = max(1, total // 10), -1.0, None, 0, 0
        scaler = torch.amp.GradScaler()

        @torch.no_grad()
        def predict(X):
            head.eval()
            out = []
            for i in range(0, len(X), 32):
                x = (X[i:i + 32].to(dev).float() / 255 - mean) / std
                with torch.autocast("cuda", dtype=torch.float16):
                    out.append(torch.sigmoid(head(bb(x)).float()).cpu().numpy())
            head.train()
            return np.concatenate(out)

        while step < total:
            for v1, v2, _, idx in loader.epoch(ep):
                for g, b in zip(opt.param_groups, base):
                    g["lr"] = b * lr_at(step, total, max(1, total // 20))
                x = (torch.from_numpy(np.concatenate([v1, v2])).to(dev).float() / 255 - mean) / std
                y = torch.cat([Y[idx], Y[idx]]).to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = head(bb(x))
                loss = lossf(logits.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                step += 1
                if step % every == 0 or step == total:
                    auc = macro_auroc(Yva[:, cols], predict(Xva), min_pos=3)
                    if auc > best:
                        best = auc
                        best_state = ({k: v.clone() for k, v in bb.state_dict().items() if ".A." in k or ".B." in k},
                                      {k: v.clone() for k, v in head.state_dict().items()})
                if step >= total:
                    break
            ep += 1
        loader.close()
        bb.load_state_dict(best_state[0], strict=False)
        head.load_state_dict(best_state[1])
        return predict(Xte), best, total

    results = []
    n_lora = sum(p.numel() for p in add_lora(build_backbone("random"), args.rank))
    for frac in args.fractions:
        for seed in ([0] if frac >= 1.0 else [0, 1, 2]):
            t = time.time()
            sub = sample_fraction(Ytr, frac, seed)
            cols = [j for j in range(len(scored)) if Ytr[sub, j].sum() >= 2 and Yte[:, j].sum() >= 5]
            lin = np.column_stack([make_pipeline(StandardScaler(), LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000))
                                   .fit(Ftr[sub], Ytr[sub, j]).predict_proba(Fte)[:, 1] for j in cols])
            S, val_best, steps = run_lora(sub, cols, seed)
            r = {"fraction": frac, "seed": seed, "n_photos": int(len(sub)), "labels": [scored[j] for j in cols],
                 "steps": steps, "lora_valA": val_best,
                 "linear_probe_test": macro_auroc(Yte[:, cols], lin), "lora_test": macro_auroc(Yte[:, cols], S),
                 "minutes": round((time.time() - t) / 60, 1)}
            results.append(r)
            print(f"{frac:.0%} seed {seed}: {len(sub)} photos, {len(cols)} labels  linear {r['linear_probe_test']:.4f}  "
                  f"LoRA {r['lora_test']:.4f}  ({r['minutes']} min)", flush=True)
            json.dump({"encoder": args.encoder, "rank": args.rank, "lora_params": n_lora, "runs": results},
                      open(f"outputs/eval/lora_{args.encoder}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
