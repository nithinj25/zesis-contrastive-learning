"""Supervised ResNet-50 baseline (CLAUDE.md section 11): the whole network fine-tuned on RFMiD's labels.

Run from the repo root:  python -m encoder.supervised [--epochs 25]
Same start (ImageNet ResNet-50), same retina-safe views at 384 px and same photos as TRACE's head training, but
trained WITH labels. Writes checkpoints/supervised_r50/{encoder.pt (backbone only, loads like any TRACE encoder),
head.pt, log.csv} and outputs/eval/supervised_r50_e2e.json (its own head on RFMiD test). Its frozen backbone then
goes through the same transfer tests as every other encoder:  python -m encoder.features --name supervised_r50 ...

Choices: labels with >= 5 training positives + Disease_Risk; BCE with pos_weight = negatives / positives (section
8.2), capped at 50 (a judgment call: uncapped, a 5-positive label gets weight ~350); AdamW, backbone LR 1e-4, head
1e-3, cosine with a 1-epoch warm-up; the epoch is chosen on val-A, never on test. CRVO / AION / RP photos are not in
the training bucket (leave-disease-out), so the baseline never sees them either.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MIN_TRAIN_POS = 5
POS_WEIGHT_CAP = 50.0


def label_matrix(labels_col, names: list[str]) -> np.ndarray:
    """labels_col: iterable of label lists -> (n, len(names)) float32 0/1."""
    return np.array([[n in set(x) for n in names] for x in labels_col], np.float32).reshape(-1, len(names))


def pos_weight(Y: np.ndarray, cap: float = POS_WEIGHT_CAP) -> np.ndarray:
    """negatives / positives per column (section 8.2: 40 positives in 1,920 -> 47), capped."""
    pos = Y.sum(0)
    return np.minimum((len(Y) - pos) / np.maximum(pos, 1), cap).astype(np.float32)


def macro_auroc(Y: np.ndarray, S: np.ndarray, min_pos: int = 1) -> float:
    cols = [roc_auc_score(Y[:, j], S[:, j]) for j in range(Y.shape[1]) if min_pos <= Y[:, j].sum() < len(Y)]
    return float(np.mean(cols)) if cols else float("nan")


def lr_at(step: int, total: int, warm: int) -> float:
    """Multiplier: linear warm-up, then cosine to 0."""
    if step < warm:
        return (step + 1) / warm
    return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))


def main():
    import torch
    import torch.nn as nn

    from augment.loader import ViewLoader
    from data.process import load_processed
    from encoder.export import IMAGENET_MEAN, IMAGENET_STD, build_backbone, finish_view

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="supervised_r50")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8, help="photos per step; each gives 2 views, so 16 images")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(0.8)  # Windows pages GPU memory instead of failing (section 7.6)
    dev = "cuda"
    out_dir = os.path.join("checkpoints", args.name)
    os.makedirs(out_dir, exist_ok=True)

    meta = pd.read_parquet("data/meta.parquet")
    stats = json.load(open("data/trace_preprocess.json"))
    pick = lambda b: meta[(meta.bucket == b) & (meta.dataset == "rfmid") & meta.label_ok & meta.fov_ok].reset_index(drop=True)
    tr, va, te = pick("train"), pick("val_a"), pick("test")
    counts = pd.Series([lab for x in tr.labels for lab in x]).value_counts()
    names = sorted(counts[counts >= MIN_TRAIN_POS].index.tolist())
    Ytr = np.column_stack([label_matrix(tr.labels, names), tr.disease_risk.astype(np.float32).values])
    heads = names + ["Disease_Risk"]
    print(f"train {len(tr)}, val-A {len(va)}, test {len(te)}; {len(heads)} outputs: {heads}")

    def eval_set(rows):  # evaluation images exactly as the frozen-feature tests see them (finish_view)
        xs = []
        for p, s in zip(rows.processed_path, rows.source):
            rgb, mask = load_processed(os.path.join("processed", p))
            xs.append((finish_view(rgb, mask, s, stats, args.size) * 255).round().to(torch.uint8))
        return torch.stack(xs)

    t0 = time.time()
    Xva, Xte = eval_set(va), eval_set(te)
    Yva = np.column_stack([label_matrix(va.labels, names), va.disease_risk.astype(np.float32).values])
    print(f"eval images ready in {time.time() - t0:.0f} s")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = build_backbone("imagenet")
            self.head = nn.Linear(2048, len(heads))
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, x):  # x: float in [0, 1]
            return self.head(self.backbone((x - self.mean) / self.std))

    net = Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": 1e-4},
                             {"params": net.head.parameters(), "lr": 1e-3}], weight_decay=0.01)
    base_lr = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler()
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pos_weight(Ytr)).to(dev))
    loader = ViewLoader(tr.processed_path.tolist(), args.batch, args.workers, seed=args.seed, size=args.size,
                        policy="retina", sources=tr.source.tolist(), stats=stats)
    total, warm = args.epochs * len(loader), len(loader)

    @torch.no_grad()
    def predict(X):
        net.eval()
        out = []
        for i in range(0, len(X), 32):
            x = X[i:i + 32].to(dev).float().div_(255).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(torch.sigmoid(net(x).float()).cpu().numpy())
        net.train()
        return np.concatenate(out)

    log = open(os.path.join(out_dir, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["epoch", "loss", "valA_macro_auroc", "valA_risk_auroc", "seconds"])
    best, step = -1.0, 0
    Yt = torch.from_numpy(Ytr)
    for ep in range(args.epochs):
        t, losses = time.time(), []
        for v1, v2, _, idx in loader.epoch(ep):
            for g, b in zip(opt.param_groups, base_lr):
                g["lr"] = b * lr_at(step, total, warm)
            x = torch.from_numpy(np.concatenate([v1, v2])).to(dev, non_blocking=True).float().div_(255)
            x = x.contiguous(memory_format=torch.channels_last)
            y = torch.cat([Yt[idx], Yt[idx]]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(x)
            loss = lossf(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            step += 1
        S = predict(Xva)
        auc = macro_auroc(Yva[:, :-1], S[:, :-1], min_pos=3)
        risk = roc_auc_score(Yva[:, -1], S[:, -1])
        w.writerow([ep + 1, f"{np.mean(losses):.4f}", f"{auc:.4f}", f"{risk:.4f}", f"{time.time() - t:.0f}"])
        log.flush()
        print(f"epoch {ep + 1}: loss {np.mean(losses):.4f}  val-A macro-AUROC {auc:.4f}  risk {risk:.4f}  {time.time() - t:.0f} s")
        if auc > best:
            best = auc
            torch.save(net.backbone.state_dict(), os.path.join(out_dir, "encoder.pt"))
            torch.save({"head": net.head.state_dict(), "heads": heads, "epoch": ep + 1}, os.path.join(out_dir, "head.pt"))
    loader.close()

    # Its own head on RFMiD test, at the val-A-selected epoch; scored on the same labels as the frozen probes.
    net.backbone.load_state_dict(torch.load(os.path.join(out_dir, "encoder.pt"), weights_only=True))
    h = torch.load(os.path.join(out_dir, "head.pt"), weights_only=False)
    net.head.load_state_dict(h["head"])
    S = predict(Xte)
    ref = json.load(open("outputs/eval/trace_v1_384.json"))["rfmid_labels_scored"]
    scored = [n for n in ref if n in names]
    Yte = label_matrix(te.labels, scored)
    St = S[:, [heads.index(n) for n in scored]]
    from eval.probes import boot_ci

    res = {"name": args.name, "selected_epoch": h["epoch"], "valA_macro_auroc": best,
           "labels_scored": scored, "labels_not_trained": [n for n in ref if n not in names],
           "rfmid_e2e_macro_auroc": macro_auroc(Yte, St), "rfmid_e2e_macro_auroc_ci95": boot_ci(Yte, St),
           "rfmid_e2e_per_label": {n: float(roc_auc_score(Yte[:, i], St[:, i])) for i, n in enumerate(scored)},
           "rfmid_e2e_disease_risk_auroc": float(roc_auc_score(te.disease_risk.astype(float), S[:, -1]))}
    os.makedirs("outputs/eval", exist_ok=True)
    json.dump(res, open(f"outputs/eval/{args.name}_e2e.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rfmid_e2e_per_label"}, indent=1))


if __name__ == "__main__":
    main()
