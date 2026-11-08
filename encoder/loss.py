"""InfoNCE and the TRACE loss: cross-source false-negative mining + same-source beta weighting (section 7.4).

For anchor q from source d_i, a queue key j is "attracted" (an extra positive) only if ALL hold:
  1. different source, 2. similarity in the top `top_frac` of cross-source similarities in this batch,
  3. mutual: j is among the anchor's top-k cross-source keys AND the anchor is at least as close as j's own
     k-th cross-source neighbour (stored when j entered the queue), 4. after warm-up (the caller sets mine).
Negatives from the same source get weight beta.

    Z = sum_{p in P} exp(s_p / tau) + sum_{j not in P} w_j exp(s_j / tau)     w_j = beta if same source else 1
    L = -(1/|P|) sum_{p in P} lambda_p log(exp(s_p / tau) / Z)                lambda = 1 for k+, alpha for attracted
With no attracted keys and beta = 1 this is exactly MoCo's InfoNCE. Computed in fp32 (-1e9 overflows fp16).
"""
from __future__ import annotations

import torch


def contrastive_loss(pos: torch.Tensor, s: torch.Tensor, same: torch.Tensor, valid: torch.Tensor,
                     attract: torch.Tensor, tau: float = 0.2, beta: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    """Per-anchor loss. pos: (B,) similarity to the true partner. s: (B, K) similarities to queue keys.
    same, valid, attract: (B, K) bool. Returns (B,)."""
    pos, s = pos.float(), s.float()
    w = torch.where(same, torch.full_like(s, beta), torch.ones_like(s))
    log_w = torch.log(w).masked_fill(attract, 0.0)  # attracted keys count as positives: weight 1
    logits = torch.cat([pos[:, None] / tau, s / tau + log_w], dim=1)  # + log w == multiply exp() by w
    pad = torch.cat([torch.zeros_like(pos[:, None], dtype=torch.bool), ~valid], dim=1)
    logits = logits.masked_fill(pad, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    lam = torch.cat([torch.ones_like(pos[:, None]), attract.float() * alpha], dim=1)
    return -(lam * logp).sum(1) / (1 + attract.sum(1))


@torch.no_grad()
def mine_attract(s: torch.Tensor, src: torch.Tensor, queue_src: torch.Tensor, queue_kth: torch.Tensor,
                 knn: int = 10, top_frac: float = 0.01) -> tuple[torch.Tensor, dict]:
    """The four-condition attraction rule (warm-up is the caller's). Returns (attract mask, counts per condition)."""
    valid = (queue_src >= 0)[None, :].expand_as(s)
    cross = (src[:, None] != queue_src[None, :]) & valid
    if not cross.any():
        z = torch.zeros_like(s, dtype=torch.bool)
        return z, {"cross": 0, "high": 0, "top": 0, "mutual": 0, "attracted": 0}
    vals = s[cross]
    if vals.numel() > 4_000_000:  # torch.quantile has an input-size limit
        vals = vals[torch.randint(0, vals.numel(), (4_000_000,), device=vals.device)]
    thr = torch.quantile(vals.float(), 1 - top_frac)
    k = min(knn, s.shape[1])
    top = torch.zeros_like(cross).scatter_(1, s.masked_fill(~cross, -2.0).topk(k, dim=1).indices, True) & cross
    mutual = s >= queue_kth[None, :]
    high = cross & (s > thr)
    attract = high & top & mutual
    # counts: cross-source keys; "high" = also in the top fraction (condition 2); of those, how many also pass
    # condition 3a (top-k) and 3b (mutual): tells which condition blocks attraction most
    return attract, {"cross": int(cross.sum()), "high": int(high.sum()), "top": int((high & top).sum()),
                     "mutual": int((high & mutual).sum()), "attracted": int(attract.sum())}


def trace_loss(q: torch.Tensor, k_pos: torch.Tensor, src: torch.Tensor, queue: torch.Tensor, queue_src: torch.Tensor,
               queue_kth: torch.Tensor, tau: float = 0.2, beta: float = 2.0, alpha: float = 0.5, mine: bool = True,
               top_frac: float = 0.01, knn: int = 10) -> tuple[torch.Tensor, dict]:
    """q, k_pos: (B, D) unit vectors (anchor and its true partner key). queue: (K, D). Returns (mean loss, stats)."""
    q, k_pos = q.float(), k_pos.float()
    pos = (q * k_pos).sum(1)
    s = q @ queue.float().T
    valid = (queue_src >= 0)[None, :].expand_as(s)
    same = (src[:, None] == queue_src[None, :]) & valid
    if mine:
        attract, stats = mine_attract(s.detach(), src, queue_src, queue_kth, knn, top_frac)
    else:
        attract, stats = torch.zeros_like(s, dtype=torch.bool), {"attracted": 0}
    loss = contrastive_loss(pos, s, same, valid, attract, tau, beta, alpha)
    stats["attracted_per_anchor"] = stats["attracted"] / len(q)
    stats["attract"] = attract  # (B, K) bool, for the mining monitor (never used for training)
    return loss.mean(), stats


def multicrop_loss(q1: torch.Tensor, q2: torch.Tensor, k1: torch.Tensor, k2: torch.Tensor, q_loc: torch.Tensor | None,
                   src: torch.Tensor, queue: torch.Tensor, queue_src: torch.Tensor, queue_kth: torch.Tensor,
                   mine: bool = True, **kw) -> tuple[torch.Tensor, dict]:
    """Section 19.4: the mean over all (query, key) pairs. Globals: (q1, k2) and (q2, k1), mined after warm-up.
    Each local view (q_loc: (n, B, D)) is paired with BOTH global keys, mine=False (beta still applies); locals
    are never compared with each other. n = 4 gives 2 + 4 x 2 = 10 pairs; n = 0 gives plain two-view MoCo.

    A local's queue similarities are the same for k1 and k2, so they are computed once for both pairs."""
    l12, s1 = trace_loss(q1, k2, src, queue, queue_src, queue_kth, mine=mine, **kw)
    l21, s2 = trace_loss(q2, k1, src, queue, queue_src, queue_kth, mine=mine, **kw)
    stats = {"attracted_per_anchor": (s1["attracted_per_anchor"] + s2["attracted_per_anchor"]) / 2,
             "loss_global": float((l12 + l21).detach()) / 2, "attract": (s1["attract"], s2["attract"]),
             "counts": {k: s1[k] + s2[k] for k in ("cross", "high", "top", "mutual", "attracted") if k in s1}}
    n = 0 if q_loc is None else q_loc.shape[0]
    if n == 0:
        return (l12 + l21) / 2, stats
    tau, beta = kw.get("tau", 0.2), kw.get("beta", 2.0)
    ql = q_loc.reshape(-1, q_loc.shape[-1]).float()  # (n B, D): local j of photo b is row j B + b
    srcl = src.repeat(n)
    s = ql @ queue.float().T
    valid = (queue_src >= 0)[None, :].expand_as(s)
    same = (srcl[:, None] == queue_src[None, :]) & valid
    none = torch.zeros_like(s, dtype=torch.bool)
    loc = 0.0
    for k in (k1, k2):
        pos = (ql * k.float().repeat(n, 1)).sum(1)
        loc = loc + contrastive_loss(pos, s, same, valid, none, tau, beta).mean()
    stats["loss_local"] = float(loc.detach()) / 2
    # sum over pairs / number of pairs; each local term above is already a mean over its n pairs
    return (l12 + l21 + n * loc) / (2 + 2 * n), stats
