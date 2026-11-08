"""Mining monitor (CLAUDE.md section 7.4, "Log every epoch"). Labels are used here ONLY to measure what mining
does; they never reach the loss.

Per epoch:
- attraction: attracted keys per anchor, share of anchors with at least one;
- which condition blocks: of the cross-source keys in the top fraction ("candidates"), the share failing the
  anchor's top-k (3a) and the share failing the mutual check (3b);
- purity and lift, two label types:
  * findings/diseases (labelled RFMiD / RFMiD 2.0 train photos): an attracted pair is "pure" if the two photos share
    a non-normal finding or disease. Normal-normal pairs are left out (section 7.4);
  * DR (every pool photo: EyePACS / APTOS grade >= 1, or RFMiD "DR"): pure = both have DR; pairs with no DR on
    either side are left out, the same idea;
  lift = purity / chance, chance = the same rate over random cross-source pool pairs (computed once);
- same eye: attracted pairs that are copies of one eye (same dup_group): true positives, but not "learned";
- sources: attracted pairs per (anchor source, key source) / the count expected from the queue's make-up. A ratio
  far above 1 for one pair would mean those sources attract for a shared camera cue (section 5.5 limit).
"""
from __future__ import annotations

import numpy as np
import torch

COND_KEYS = ("cross", "high", "top", "mutual", "attracted")


def pool_labels(rows) -> tuple[list, list]:
    """From the pool rows of meta.parquet: per photo a label set (None = unknown) and DR status (None = unknown)."""
    sets, dr = [], []
    for ds, ok, f, d, labs in zip(rows.dataset, rows.label_ok, rows.findings, rows.diseases, rows.labels):
        labs = list(labs) if labs is not None else []
        grade = [int(x[-1]) for x in labs if str(x).startswith(("eyepacs_dr", "aptos_dr"))]
        if grade:
            sets.append(None)  # graded for DR only: other findings unknown
            dr.append(grade[0] >= 1)
            continue
        s = set(f if f is not None else []) | set(d if d is not None else [])
        known = bool(ok) and len(s) > 0
        sets.append(s if known else None)
        dr.append(("DR" in s) if known else None)
    return sets, dr


def pair_rate(known: np.ndarray, share: np.ndarray, both_neg: np.ndarray) -> tuple[float, int]:
    """Share of pairs that are 'pure', among known pairs that are not negative on both sides."""
    m = known & ~both_neg
    return (float(share[m].mean()) if m.any() else float("nan")), int(m.sum())


class MiningMonitor:
    def __init__(self, label_sets: list, dr: list, src: np.ndarray, groups: np.ndarray, n_src: int, device="cpu",
                 chance_pairs: int = 2_000_000, seed: int = 0):
        n = len(label_sets)
        vocab = sorted({x for s in label_sets if s for x in s} - {"normal"})
        L = np.zeros((n, max(1, len(vocab))), bool)
        col = {v: i for i, v in enumerate(vocab)}
        for i, s in enumerate(label_sets):
            for x in s or ():
                if x in col:
                    L[i, col[x]] = True
        known = np.array([s is not None for s in label_sets])
        normal = np.array([s is not None and "normal" in s for s in label_sets])
        dr_known = np.array([x is not None for x in dr])
        dr_pos = np.array([bool(x) for x in dr])
        self.n_src = n_src
        self.vocab = vocab
        dev = torch.device(device)
        self.L, self.known, self.normal = (torch.from_numpy(x).to(dev) for x in (L, known, normal))
        self.dr_known, self.dr_pos = torch.from_numpy(dr_known).to(dev), torch.from_numpy(dr_pos).to(dev)
        self.groups = torch.from_numpy(np.asarray(groups, np.int64)).to(dev)

        # chance: random cross-source pool pairs, same exclusions
        rng = np.random.default_rng(seed)
        a, b = rng.integers(0, n, chance_pairs), rng.integers(0, n, chance_pairs)
        x = src[a] != src[b]
        a, b = a[x], b[x]
        share = (L[a] & L[b]).any(1)
        self.chance, _ = pair_rate(known[a] & known[b], share, normal[a] & normal[b])
        self.dr_chance, _ = pair_rate(dr_known[a] & dr_known[b], dr_pos[a] & dr_pos[b], ~dr_pos[a] & ~dr_pos[b])
        self.reset()

    def reset(self):
        S = self.n_src
        self.c = dict.fromkeys(COND_KEYS, 0)
        self.anchors = self.with_attr = 0
        self.lab_n = self.lab_pure = self.dr_n = self.dr_pure = self.same_eye = 0
        self.obs = torch.zeros(S, S, dtype=torch.float64)
        # purity per (anchor source, key source): biology vs a shared camera cue (high ratio + low purity = camera)
        self.pair = {k: torch.zeros(S * S, dtype=torch.float64) for k in ("lab_n", "lab_pure", "dr_n", "dr_pure")}
        self.exp = torch.zeros(S, S, dtype=torch.float64)

    @torch.no_grad()
    def update(self, attract: torch.Tensor, anchor_idx: torch.Tensor, anchor_src: torch.Tensor,
               queue_idx: torch.Tensor, queue_src: torch.Tensor, counts: dict | None = None):
        """attract: (B, K) bool from one (query, key) pair of the loss. anchor_idx / queue_idx: pool rows."""
        S = self.n_src
        self.anchors += attract.shape[0]
        self.with_attr += int(attract.any(1).sum())
        for k in COND_KEYS:
            self.c[k] += (counts or {}).get(k, 0)
        r, c = attract.nonzero(as_tuple=True)
        if len(r) == 0:
            return
        a, b = anchor_idx[r], queue_idx[c]
        ok = b >= 0  # keys enqueued without a pool row cannot be looked up
        a, b, sa, sb = a[ok], b[ok], anchor_src[r][ok], queue_src[c][ok]
        cell = (sa * S + sb).cpu()

        def add(key, mask):
            self.pair[key] += torch.bincount(cell, weights=mask.double().cpu(), minlength=S * S)

        m = self.known[a] & self.known[b] & ~(self.normal[a] & self.normal[b])
        pure = (self.L[a] & self.L[b]).any(1) & m
        self.lab_n += int(m.sum())
        self.lab_pure += int(pure.sum())
        add("lab_n", m), add("lab_pure", pure)
        m = self.dr_known[a] & self.dr_known[b] & (self.dr_pos[a] | self.dr_pos[b])
        pure = self.dr_pos[a] & self.dr_pos[b] & m
        self.dr_n += int(m.sum())
        self.dr_pure += int(pure.sum())
        add("dr_n", m), add("dr_pure", pure)
        self.same_eye += int((self.groups[a] == self.groups[b]).sum())
        self.obs += torch.bincount(sa * S + sb, minlength=S * S).view(S, S).double().cpu()
        # expected: each anchor's attracted count spread over the sources of its cross-source queue keys
        comp = torch.bincount(queue_src[queue_src >= 0], minlength=S).double().cpu()
        D = comp[None, :].repeat(S, 1)
        D.fill_diagonal_(0)
        D = D / D.sum(1, keepdim=True).clamp_min(1e-12)
        n_from = torch.bincount(sa, minlength=S).double().cpu()
        self.exp += n_from[:, None] * D

    def summary(self) -> dict:
        c, n = self.c, max(1, self.anchors)
        high = max(1, c["high"])
        pur = self.lab_pure / self.lab_n if self.lab_n else float("nan")
        drp = self.dr_pure / self.dr_n if self.dr_n else float("nan")
        attracted = int(self.obs.sum())
        ratio = (self.obs / self.exp.clamp_min(1e-12)).where(self.exp > 0, torch.full_like(self.obs, float("nan")))
        return {"attracted_per_anchor": c["attracted"] / n, "anchors_with_attraction": self.with_attr / n,
                "candidates_per_anchor": c["high"] / n, "blocked_by_topk": 1 - c["top"] / high,
                "blocked_by_mutual": 1 - c["mutual"] / high,
                "purity": pur, "purity_pairs": self.lab_n, "chance": self.chance, "lift": pur / self.chance,
                "dr_purity": drp, "dr_pairs": self.dr_n, "dr_chance": self.dr_chance, "dr_lift": drp / self.dr_chance,
                "same_eye_share": self.same_eye / attracted if attracted else float("nan"),
                "source_obs": self.obs.tolist(), "source_ratio": ratio.tolist(),
                "source_purity": self._rate("lab_pure", "lab_n"), "source_purity_pairs": self._shape("lab_n"),
                "source_dr_purity": self._rate("dr_pure", "dr_n"), "source_dr_pairs": self._shape("dr_n")}

    def _shape(self, key):
        return self.pair[key].view(self.n_src, self.n_src).tolist()

    def _rate(self, num, den):
        n, d = self.pair[num], self.pair[den]
        return (n / d.clamp_min(1)).where(d > 0, torch.full_like(d, float("nan"))).view(self.n_src, self.n_src).tolist()
