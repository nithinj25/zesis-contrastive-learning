"""MoCo v2 with a queue that remembers each key's source and k-th cross-source neighbour (section 7.5).

The query encoder is ResNet-50 + projection head 2048 -> 2048 -> 256 (the head feeds the loss only and is
discarded after pretraining). The key encoder is a momentum copy: theta_k <- m theta_k + (1 - m) theta_q.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.export import IMAGENET_MEAN, IMAGENET_STD, build_backbone


class QueryNet(nn.Module):
    """Normalise -> ResNet-50 -> projection head -> unit vector."""

    def __init__(self, init: str = "imagenet", dims=(2048, 2048, 256)):
        super().__init__()
        self.backbone = build_backbone(init)
        self.head = nn.Sequential(nn.Linear(dims[0], dims[1]), nn.ReLU(inplace=True), nn.Linear(dims[1], dims[2]))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.head(self.backbone((x - self.mean) / self.std)), dim=1)


class MoCo(nn.Module):
    def __init__(self, init: str = "imagenet", dim: int = 256, K: int = 8192, m: float = 0.999, knn: int = 10,
                 shuffle_chunks: int = 2):
        super().__init__()
        self.q_enc = QueryNet(init, (2048, 2048, dim))
        self.k_enc = copy.deepcopy(self.q_enc).requires_grad_(False)
        self.register_buffer("queue", F.normalize(torch.randn(K, dim), dim=1))
        self.register_buffer("queue_src", torch.full((K,), -1, dtype=torch.long))  # -1 = empty slot
        self.register_buffer("queue_kth", torch.ones(K))  # each key's k-th cross-source neighbour similarity
        self.register_buffer("queue_idx", torch.full((K,), -1, dtype=torch.long))  # pool row of each key (monitoring)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.K, self.m, self.knn, self.shuffle_chunks = K, m, knn, shuffle_chunks

    @torch.no_grad()
    def momentum_update(self):
        for pq, pk in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            pk.mul_(self.m).add_(pq.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def keys(self, x: torch.Tensor) -> torch.Tensor:
        """Key encoder with shuffle-BN on one GPU: shuffle the batch, run it in sub-batches (each with its own
        BatchNorm statistics), then unshuffle. Twins then never share a batch-statistics fingerprint."""
        perm = torch.randperm(len(x), device=x.device)
        chunks = [self.k_enc(c) for c in x[perm].chunk(self.shuffle_chunks)]
        out = torch.empty_like(torch.cat(chunks))
        out[perm] = torch.cat(chunks)
        return out

    @torch.no_grad()
    def enqueue(self, k: torch.Tensor, src: torch.Tensor, idx: torch.Tensor | None = None):
        """Add B keys; store each one's k-th cross-source neighbour similarity among the current queue.
        idx: the keys' pool rows, kept only so the mining monitor can look up labels of attracted pairs."""
        k = k.float()
        cross = (src[:, None] != self.queue_src[None, :]) & (self.queue_src[None, :] >= 0)
        kk = min(self.knn, self.K)
        kth = (k @ self.queue.T).masked_fill(~cross, -2.0).topk(kk, dim=1).values[:, -1]
        kth = kth.masked_fill(kth < -1, 2.0)  # too few cross-source keys yet: it cannot be attracted
        slot = (torch.arange(k.shape[0], device=k.device) + int(self.ptr)) % self.K
        self.queue[slot], self.queue_src[slot], self.queue_kth[slot] = k, src, kth
        if idx is not None:
            self.queue_idx[slot] = idx.long()  # loader indices are int32 on Windows
        self.ptr[0] = (int(self.ptr) + k.shape[0]) % self.K
