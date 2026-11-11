"""Contrastive pretraining (CLAUDE.md sections 7.5, 7.6, 19.3, 19.4).

    python -m encoder.train --config-name pretrain_plain                  # Phase 2 baseline
    python -m encoder.train --config-name pretrain_plain train.max_steps=200   # timing run
    python -m encoder.train ... ckpt.resume=checkpoints/moco_plain/last.pt     # continue

One step (section 19.4): the key encoder (no gradients, shuffle-BN) encodes both global views -> k1, k2; the query
encoder encodes them -> q1, q2, and the n_local local views (one batch of n B small views) -> l_j; loss = mean
over the pairs (q1, k2), (q2, k1) [mined after warm-up] and every (l_j, k1), (l_j, k2) [never mined]: 10 pairs at
n_local = 4, the plain two-view loss at 0; backward; SGD step; momentum update; enqueue view 2's keys only, so the
oldest queue key is K / B steps old.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Heavy modules (torch, the model, the loss) are imported inside main(). Views are made by augment/loader.py,
# whose workers never import torch (on Windows each torch worker committed ~1.7 GB).


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def record_val(val_path: str, epoch: int, step: int, res: dict, metric: str) -> bool:
    """Append one val-A row; return True if this epoch is the best so far (epoch 0 = the start, never 'best')."""
    new = not os.path.exists(val_path)
    with open(val_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "step", metric, "val_knn_macro_auroc", "val_C", "val_n_labels", "val_seconds"])
        w.writerow([epoch, step, f"{res[metric]:.4f}", f"{res['val_knn_macro_auroc']:.4f}", res["val_C"],
                    res["val_n_labels"], res["val_seconds"]])
    rows = list(csv.DictReader(open(val_path)))
    trained = [r for r in rows if int(r["epoch"]) > 0]
    if epoch == 0 or not trained:
        return False
    best = max(trained, key=lambda r: float(r[metric]))  # ties: the earliest epoch wins
    return int(best["epoch"]) == epoch


MINING_COLS = ["attracted_per_anchor", "anchors_with_attraction", "candidates_per_anchor", "blocked_by_topk",
               "blocked_by_mutual", "purity", "purity_pairs", "chance", "lift", "dr_purity", "dr_pairs", "dr_chance", "dr_lift",
               "same_eye_share"]


def write_mining(path: str, epoch: int, step: int, s: dict, source_names: list[str]):
    """One row per epoch in mining.csv; the source-pair table goes to mining_sources.jsonl (one line per epoch)."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["epoch", "step"] + MINING_COLS)
        w.writerow([epoch, step] + [f"{s[c]:.4f}" if isinstance(s[c], float) else s[c] for c in MINING_COLS])
    with open(path.replace(".csv", "_sources.jsonl"), "a") as f:
        f.write(json.dumps({"epoch": epoch, "sources": source_names, "observed": s["source_obs"],
                            "ratio_to_expected": s["source_ratio"],
                            **{k: s[k] for k in ("source_purity", "source_purity_pairs", "source_dr_purity",
                                                 "source_dr_pairs")}}) + "\n")
    print(f"mining, epoch {epoch}: {s['attracted_per_anchor']:.3f} attracted per anchor ({100 * s['anchors_with_attraction']:.1f}% "
          f"of anchors); candidates blocked by top-k {100 * s['blocked_by_topk']:.0f}%, by mutual {100 * s['blocked_by_mutual']:.0f}%; "
          f"purity {s['purity']:.3f} vs chance {s['chance']:.3f} (lift {s['lift']:.2f}, {s['purity_pairs']} pairs); "
          f"DR {s['dr_purity']:.3f} vs {s['dr_chance']:.3f} (lift {s['dr_lift']:.2f}, {s['dr_pairs']} pairs); "
          f"same eye {100 * s['same_eye_share']:.1f}%")


def save_ckpt(path: str, model, opt, scaler, epoch: int, step: int, cfg, epoch_done: bool = True):
    import torch

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "global_step": step, "epoch_done": epoch_done, "config": OmegaConf.to_container(cfg, resolve=True),
                "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, path)


@hydra.main(config_path="../configs", config_name="pretrain", version_base=None)
def main(cfg: DictConfig) -> None:
    import pandas as pd
    import torch

    from augment.loader import ViewLoader
    from encoder.loss import multicrop_loss
    from encoder.mining_monitor import MiningMonitor, pool_labels
    from encoder.moco import MoCo

    root = hydra.utils.get_original_cwd()
    os.chdir(root)
    random.seed(cfg.seed), np.random.seed(cfg.seed), torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    torch.cuda.set_per_process_memory_fraction(cfg.train.mem_fraction)  # Windows pages instead of OOM otherwise

    meta = pd.read_parquet(cfg.data.meta)
    rows = meta[meta.in_pool].reset_index(drop=True)
    # the loss's source IDs (section 5.5: mine_source merges groups that may be one camera); step 7 keeps `source`
    mine_src = rows.mine_source if "mine_source" in rows.columns else rows.source
    source_ids = {s: i for i, s in enumerate(sorted(mine_src.unique()))}
    src_all = torch.tensor([source_ids[x] for x in mine_src])
    stats = json.load(open(cfg.data.get("stats", "data/trace_preprocess.json")))
    view_kw = None
    if cfg.views.policy == "retina":
        art = cfg.views.get("artifacts", False)
        view_kw = {"scale": tuple(cfg.views.global_scale),
                   "artifacts": OmegaConf.to_container(art) if OmegaConf.is_config(art) else bool(art)}
    n_local = int(cfg.views.get("n_local", 0)) if cfg.views.policy == "retina" else 0
    ds = ViewLoader(rows.processed_path.tolist(), cfg.optim.batch, cfg.train.workers, cfg.seed, cfg.views.global_px,
                    cfg.views.policy, cfg.data.img_root, sources=rows.source.tolist(), stats=stats, view_kw=view_kw,
                    n_local=n_local, local_size=cfg.views.local_px, local_scale=tuple(cfg.views.local_scale))
    B = cfg.optim.batch
    steps_per_epoch = len(ds)
    total = steps_per_epoch * cfg.optim.epochs
    warmup = steps_per_epoch * cfg.optim.warmup_epochs
    base_lr = cfg.optim.lr_base * B / 256

    model = MoCo(cfg.model.init, cfg.model.proj_dims[-1], cfg.moco.K, cfg.moco.m, cfg.mining.get("knn", 10),
                 cfg.train.shuffle_chunks).to(device)
    init_from = cfg.model.get("init_from")
    if init_from and not cfg.ckpt.resume:  # e.g. the 384 px phase (section 19.4) from the val-selected 224 px encoder
        model.load_backbone(torch.load(init_from, map_location="cpu"))
        print(f"backbones initialised from {init_from}; new projection head and an empty queue")
    opt = torch.optim.SGD(model.q_enc.parameters(), lr=base_lr, momentum=cfg.optim.momentum, weight_decay=cfg.optim.wd)
    scaler = torch.amp.GradScaler(enabled=cfg.amp)
    out_dir = os.path.join(cfg.ckpt.dir, cfg.encoder_name)
    os.makedirs(out_dir, exist_ok=True)
    # Mining monitor (section 7.4): labels measure what mining does, they never reach the loss.
    monitor, mine_csv = None, os.path.join("outputs", "train", cfg.encoder_name, "mining.csv")
    if cfg.mining.enabled:
        sets, dr = pool_labels(rows)
        eyes = pd.factorize(rows.dup_group.fillna(rows.image_id).astype(str))[0]
        monitor = MiningMonitor(sets, dr, src_all.numpy(), eyes, len(source_ids), device)
        print(f"mining monitor: chance purity {monitor.chance:.3f} (findings/diseases), {monitor.dr_chance:.3f} (DR)")
    epoch0, step = 0, 0
    if cfg.ckpt.resume:
        ck = torch.load(cfg.ckpt.resume, map_location="cpu", weights_only=False)  # RNG states must stay on the CPU
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), scaler.load_state_dict(ck["scaler"])
        step = ck["global_step"]
        epoch0 = ck["epoch"] + 1 if ck.get("epoch_done", True) else ck["epoch"]
        random.setstate(ck["rng"]["python"]), np.random.set_state(ck["rng"]["numpy"]), torch.set_rng_state(ck["rng"]["torch"])
        if ck["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resumed from {cfg.ckpt.resume}: epoch {epoch0}, step {step} (batch {step - epoch0 * len(ds)} of the epoch)")

    log_path = os.path.join("outputs", "train", cfg.encoder_name, "log.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if step == 0:
        log.writerow(["step", "epoch", "loss", "lr", "img_per_s", "gpu_gb", "q_dim_std", "attracted_per_anchor",
                      "loss_global", "loss_local"])
    collapse = math.log(cfg.moco.K + 1)
    try:
        import psutil

        print(f"RAM available at start: {psutil.virtual_memory().available / 1e9:.1f} GB")
    except ImportError:
        pass
    print(f"views: {cfg.views.policy}, 2 x {cfg.views.global_px} px globals + {n_local} x {cfg.views.local_px} px locals "
          f"({2 + 2 * n_local} pairs per photo)")
    print(f"pool {len(rows)} photos, {len(source_ids)} sources, {steps_per_epoch} steps/epoch, total {total}, "
          f"lr {base_lr:.5f}, collapse level log(K+1) = {collapse:.2f}")

    # Model selection on val-A (section 7.7): never on test.
    selector, val_path = None, os.path.join("outputs", "train", cfg.encoder_name, "val.csv")
    if cfg.get("eval_every", 0):
        import shutil

        from eval.val_select import METRIC, ValSelector

        selector = ValSelector(meta, stats, cfg.eval_size)

        def run_val(epoch_count: int):
            res = selector.score(model.q_enc.backbone, device)
            is_best = record_val(val_path, epoch_count, step, res, METRIC)
            if epoch_count > 0:
                path = os.path.join(out_dir, f"encoder_e{epoch_count:03d}.pt")
                torch.save(model.q_enc.backbone.state_dict(), path)
                if is_best:
                    shutil.copyfile(path, os.path.join(out_dir, "encoder_best.pt"))
                    json.dump({"epoch": epoch_count, METRIC: res[METRIC]}, open(os.path.join(out_dir, "best.json"), "w"))
            print(f"val-A after {epoch_count} epochs: linear {res[METRIC]:.4f}, kNN {res['val_knn_macro_auroc']:.4f} "
                  f"({res['val_n_labels']} labels, {res['val_seconds']} s){' <- best so far' if is_best else ''}")

        if step == 0:
            run_val(0)  # the starting point (ImageNet), for reference

    max_steps = cfg.train.max_steps or total
    model.train()
    for epoch in range(epoch0, cfg.optim.epochs):
        mine = cfg.mining.enabled and epoch >= cfg.mining.warmup_epochs
        t0, seen = time.time(), 0
        for v1, v2, loc, idx in ds.epoch(epoch, start=max(0, step - epoch * steps_per_epoch)):
            x1 = torch.from_numpy(v1).to(device, non_blocking=True).float().div_(255)
            x2 = torch.from_numpy(v2).to(device, non_blocking=True).float().div_(255)
            xl = torch.from_numpy(loc.reshape(-1, *loc.shape[2:])).to(device, non_blocking=True).float().div_(255) if n_local else None
            src = src_all[idx].to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warmup, base_lr)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                k1, k2 = model.keys(x1), model.keys(x2)
                q1, q2 = model.q_enc(x1), model.q_enc(x2)
                ql = model.q_enc(xl).view(n_local, B, -1) if n_local else None  # locals: own forward pass (own size)
            kw = dict(tau=cfg.moco.tau, beta=cfg.mining.beta, alpha=cfg.mining.get("alpha", 0.5),
                      top_frac=cfg.mining.get("top_frac", 0.01), knn=cfg.mining.get("knn", 10))
            loss, st = multicrop_loss(q1, q2, k1, k2, ql, src, model.queue, model.queue_src, model.queue_kth, mine=mine, **kw)
            idx_t = torch.from_numpy(idx).long().to(device)
            if mine and monitor is not None:  # before enqueue: the masks refer to the queue as the loss saw it
                for j, a in enumerate(st["attract"]):
                    monitor.update(a, idx_t, src, model.queue_idx, model.queue_src, st["counts"] if j == 0 else None)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.momentum_update()
            model.enqueue(k2.float(), src, idx_t)
            step += 1
            seen += B
            every = cfg.ckpt.get("every_steps", 0)
            if every and step % every == 0 and step % steps_per_epoch:  # mid-epoch save: a crash loses <= `every` steps
                save_ckpt(os.path.join(out_dir, "last.pt"), model, opt, scaler, epoch, step, cfg, epoch_done=False)
            if step % cfg.train.log_every == 0 or step == max_steps:
                torch.cuda.synchronize()
                rate = seen / (time.time() - t0)
                qstd = float(q1.detach().float().std(0).mean())
                att = st["attracted_per_anchor"]
                gb = torch.cuda.max_memory_allocated() / 1e9
                log.writerow([step, epoch, f"{loss.item():.4f}", f"{opt.param_groups[0]['lr']:.6f}", f"{rate:.1f}", f"{gb:.2f}",
                              f"{qstd:.4f}", f"{att:.3f}", f"{st['loss_global']:.4f}", f"{st.get('loss_local', float('nan')):.4f}"])
                log_f.flush()
                print(f"step {step} ep {epoch} loss {loss.item():.3f} (collapse {collapse:.2f}) lr {opt.param_groups[0]['lr']:.5f} "
                      f"{rate:.0f} img/s gpu {gb:.2f} GB q-dim-std {qstd:.4f} (healthy ~{1 / math.sqrt(q1.shape[1]):.4f}) attracted {att:.2f}")
            if step >= max_steps:
                break
        done = step >= (epoch + 1) * steps_per_epoch
        save_ckpt(os.path.join(out_dir, "last.pt"), model, opt, scaler, epoch, step, cfg, epoch_done=done)
        if (epoch + 1) % 10 == 0:
            save_ckpt(os.path.join(out_dir, f"epoch_{epoch + 1:03d}.pt"), model, opt, scaler, epoch, step, cfg)
            old = sorted(f for f in os.listdir(out_dir) if f.startswith("epoch_"))[:-2]
            for f in old:
                os.remove(os.path.join(out_dir, f))
        torch.save(model.q_enc.backbone.state_dict(), os.path.join(out_dir, "encoder.pt"))
        print(f"epoch {epoch} done: {seen} photos in {time.time() - t0:.0f} s; saved {out_dir}/last.pt and encoder.pt")
        if mine and monitor is not None:
            write_mining(mine_csv, epoch, step, monitor.summary(), sorted(source_ids, key=source_ids.get))
            monitor.reset()
        if selector is not None and done and ((epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.optim.epochs):
            run_val(epoch + 1)
        if step >= max_steps:
            break
    log_f.close()
    ds.close()


if __name__ == "__main__":
    main()
