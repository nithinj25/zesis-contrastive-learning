"""A torch-free batch loader for pretraining views.

Why not torch's DataLoader: on Windows every DataLoader worker imports PyTorch, and each one committed ~1.7 GB
(about 500 MB resident). Six of them plus training left ~400 MB of RAM free on a 16 GB laptop, so everything
paged and slowed down. Making views needs only numpy and OpenCV, so these workers never import torch; the
main process turns their uint8 batches into tensors.

Each batch's randomness comes from (seed, epoch, batch number), so the same run gives the same views, and a
resumed run continues with exactly the views it would have seen.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from collections import deque

import numpy as np

_STATE: dict = {}


def _init(paths: list[str], size: int, policy: str, sources: list | None = None, stats: dict | None = None,
          view_kw: dict | None = None, local: tuple = (0, 96, (0.15, 0.35)), guided: dict | None = None):
    import cv2

    cv2.setNumThreads(1)
    view_kw = dict(view_kw or {})
    art = view_kw.pop("artifacts", None)  # True, or a dict of per-artifact probabilities
    if art:
        from augment.artifacts import Artifacts

        view_kw["artifacts"] = Artifacts(stats, None if art is True else dict(art))
    _STATE.update(paths=paths, size=size, policy=policy, sources=sources, stats=stats or {}, view_kw=view_kw,
                  local=local, guided=None)
    if guided:  # lesion-guided locals: per-photo crop-sampling maps, memory-mapped so workers share one copy
        maps = np.load(guided["maps"], mmap_mode="r")
        keys = [str(x) for x in np.load(guided["maps"].replace(".npy", "_paths.npy"))]
        row = {k: i for i, k in enumerate(keys)}
        _STATE["guided"] = dict(maps=maps, row=[row[os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p)]
                                                for p in paths], side=tuple(guided["side"]),
                                min_retina=guided.get("min_retina", 0.85))


def make_batch(job: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """job = (indices, seed, epoch, batch_number). Returns the two global view stacks (B, 3, H, W), the local views
    (n_local, B, 3, h, h) and the indices, all uint8. Each photo's locals are drawn after its two globals from the
    same generator, so n_local = 0 gives exactly the two-view batches of earlier runs."""
    from augment.views import plain_view_np, prepare, retina_view_np
    from data.process import load_processed

    idx, seed, epoch, b = job
    rng = np.random.default_rng((seed, epoch, b))
    size, policy, stats = _STATE["size"], _STATE["policy"], _STATE["stats"]
    n_local, lsize, lscale = _STATE["local"]
    v1, v2, loc = [], [], []
    for i in idx:
        rgb, mask = load_processed(_STATE["paths"][i])
        if policy == "plain":
            img = prepare(rgb, mask)
            v1.append(plain_view_np(img, rng, size))
            v2.append(plain_view_np(img, rng, size))
        elif policy == "retina":
            src = _STATE["sources"][i] if _STATE["sources"] is not None else None
            kw = dict(camera=stats.get("cameras", {}).get(src), pool=stats.get("pool"), **_STATE["view_kw"])
            if "params" in stats:
                kw.update(colour_alpha=stats["params"]["colour_alpha"], clahe_clip=stats["params"]["clahe_clip"])
            v1.append(retina_view_np(rgb, mask, rng, size, **kw))
            v2.append(retina_view_np(rgb, mask, rng, size, **kw))
            kw_l = {**kw, "scale": tuple(lscale)}  # same recipe, smaller crop (section 7.3 multi-crop)
            gd = _STATE["guided"]
            if gd is None:
                loc.append([retina_view_np(rgb, mask, rng, lsize, **kw_l) for _ in range(n_local)])
            else:  # Approach 2's lesion-guided locals; a failed draw falls back to the uniform local crop
                from augment.lesions import sample_guided_crop

                pmap = np.asarray(gd["maps"][gd["row"][i]], np.float32)
                loc.append([retina_view_np(rgb, mask, rng, lsize, crop=sample_guided_crop(
                    rng, pmap, mask, gd["side"], gd["min_retina"]), **kw_l) for _ in range(n_local)])
        else:
            raise ValueError(f"unknown view policy {policy!r}")
    if policy != "retina" or n_local == 0:
        loc_arr = np.zeros((0, len(idx), 3, lsize, lsize), np.uint8)
    else:
        loc_arr = np.ascontiguousarray(np.stack([np.stack(v) for v in loc]).transpose(1, 0, 2, 3, 4))
    return np.stack(v1), np.stack(v2), loc_arr, np.asarray(idx)


class ViewLoader:
    """for v1, v2, loc, idx in loader.epoch(e): ...   (numpy uint8 arrays, drop_last batches, shuffled per epoch)"""

    def __init__(self, paths: list[str], batch: int, workers: int, seed: int = 0, size: int = 224,
                 policy: str = "plain", root: str = "processed", prefetch: int | None = None,
                 sources: list | None = None, stats: dict | None = None, view_kw: dict | None = None,
                 n_local: int = 0, local_size: int = 96, local_scale=(0.15, 0.35), guided: dict | None = None):
        """sources: camera id per photo and stats: data/trace_preprocess.json (both for policy "retina", step 7).
        view_kw: extra retina_view_np settings, e.g. scale. n_local: multi-crop local views per photo (retina only)."""
        self.paths = [os.path.join(root, p) for p in paths]
        self.batch, self.seed = batch, seed
        self.prefetch = prefetch or 2 * workers
        self.pool = mp.get_context("spawn").Pool(workers, initializer=_init,
                                                 initargs=(self.paths, size, policy, sources, stats, view_kw,
                                                           (n_local, local_size, tuple(local_scale)), guided))

    def __len__(self):
        return len(self.paths) // self.batch

    def epoch(self, epoch: int, start: int = 0):
        """start: skip the first `start` batches (resuming mid-epoch gives exactly the remaining batches)."""
        order = np.random.default_rng((self.seed, epoch)).permutation(len(self.paths))
        jobs = [(order[b * self.batch:(b + 1) * self.batch].tolist(), self.seed, epoch, b) for b in range(len(self))]
        pending: deque = deque()
        it = iter(jobs[start:])
        for job in it:  # bounded prefetch: never more than `prefetch` batches waiting in memory
            pending.append(self.pool.apply_async(make_batch, (job,)))
            if len(pending) >= self.prefetch:
                break
        while pending:
            result = pending.popleft().get()
            nxt = next(it, None)
            if nxt is not None:
                pending.append(self.pool.apply_async(make_batch, (nxt,)))
            yield result

    def close(self):
        self.pool.close()
        self.pool.join()
