"""MoCo + TRACE loss against the hand-checked numbers in CLAUDE.md section 16 and section 7.4."""
import math

import torch

from encoder.loss import contrastive_loss, mine_attract, multicrop_loss, trace_loss
from encoder.moco import MoCo


def infonce(sims, tau=0.2):
    """sims[0] is the positive. Plain InfoNCE through contrastive_loss with no attraction and beta = 1."""
    s = torch.tensor([sims[1:]], dtype=torch.float32, requires_grad=True)
    pos = torch.tensor([sims[0]], dtype=torch.float32, requires_grad=True)
    z = torch.zeros_like(s, dtype=torch.bool)
    loss = contrastive_loss(pos, s, z, ~z, z, tau=tau, beta=1.0)
    return loss, pos, s


def test_infonce_easy_negatives():
    loss, _, _ = infonce([0.8, 0.3, 0.1])
    total = sum(math.exp(x / 0.2) for x in [0.8, 0.3, 0.1])
    assert abs(total - 60.73) < 0.01 and abs(math.exp(4) / total - 0.899) < 0.001
    assert abs(loss.item() - 0.106) < 0.001


def test_infonce_look_alike():
    loss, _, _ = infonce([0.8, 0.75, 0.1])
    total = sum(math.exp(x / 0.2) for x in [0.8, 0.75, 0.1])
    assert abs(total - 98.77) < 0.01 and abs(loss.item() - 0.593) < 0.001


def test_infonce_lesson_dry_run_gradients():
    loss, pos, s = infonce([0.9397, 0.7660, 0.0, -0.8660])
    loss.backward()
    assert abs(loss.item() - 0.357) < 0.001
    grads = [pos.grad.item()] + s.grad[0].tolist()  # dL/ds_k = (p_k - y_k) / tau
    for g, want in zip(grads, [-1.50, 1.47, 0.03, 0.00]):
        assert abs(g - want) < 0.01


def test_section_7_4_worked_example():
    # pos 0.90; DR eye camera B 0.82 (attracted); healthy eye camera A 0.80 (same source); glaucoma camera C 0.15.
    pos = torch.tensor([0.90])
    s = torch.tensor([[0.82, 0.80, 0.15]])
    valid = torch.ones_like(s, dtype=torch.bool)
    none = torch.zeros_like(s, dtype=torch.bool)
    same = torch.tensor([[False, True, False]])
    attract = torch.tensor([[True, False, False]])
    plain = contrastive_loss(pos, s, none, valid, none, tau=0.2, beta=1.0)
    trace = contrastive_loss(pos, s, same, valid, attract, tau=0.2, beta=2.0, alpha=0.5)
    e = [math.exp(x / 0.2) for x in [0.90, 0.82, 0.80, 0.15]]
    assert abs(sum(e) - 207.07) < 0.01 and abs(e[0] + e[1] + 2 * e[2] + e[3] - 261.67) < 0.01
    assert abs(plain.item() - 0.833) < 0.001 and abs(trace.item() - 0.900) < 0.001
    assert abs(e[2] / sum(e) - 0.264) < 0.001 and abs(2 * e[2] / 261.67 - 0.417) < 0.001


def test_trace_loss_reduces_to_infonce():
    torch.manual_seed(0)
    q = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    k = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    queue = torch.nn.functional.normalize(torch.randn(16, 8), dim=1)
    src, qsrc = torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1] * 8)
    a, _ = trace_loss(q, k, src, queue, qsrc, torch.ones(16), beta=1.0, mine=False)
    logits = torch.cat([(q * k).sum(1, keepdim=True), q @ queue.T], 1) / 0.2
    b = torch.nn.functional.cross_entropy(logits, torch.zeros(4, dtype=torch.long))
    assert torch.allclose(a, b, atol=1e-6)


def test_mining_needs_every_condition():
    # One anchor (source 0); keys: [same source & close, cross & close & mutual, cross & close but not mutual, far]
    s = torch.tensor([[0.95, 0.90, 0.92, 0.10] + [0.0] * 96])
    qsrc = torch.tensor([0, 1, 1, 1] + [1] * 96)
    kth = torch.tensor([0.5, 0.85, 0.99, 0.5] + [0.5] * 96)  # key 2's own 10th neighbour is closer than the anchor
    attract, counts = mine_attract(s, torch.tensor([0]), qsrc, kth, knn=10, top_frac=0.05)
    assert attract[0].nonzero().flatten().tolist() == [1]
    assert counts["attracted"] == 1


def test_momentum_update_numbers():
    mo = MoCo(init="random", K=8, m=0.999)
    with torch.no_grad():
        for p in mo.k_enc.parameters():
            p.fill_(1.0)
        for p in mo.q_enc.parameters():
            p.fill_(2.0)
    p = next(mo.k_enc.parameters())
    for want in (1.001, 1.002, 1.003):
        mo.momentum_update()
        assert abs(p.flatten()[0].item() - want) < 1e-4
    for _ in range(997):
        mo.momentum_update()
    assert abs(p.flatten()[0].item() - 1.632) < 1e-3


def test_queue_is_circular_and_stores_sources():
    mo = MoCo(init="random", K=6, knn=2)
    for step in range(4):
        k = torch.nn.functional.normalize(torch.randn(2, 256), dim=1)
        mo.enqueue(k, torch.tensor([step, step]))
    assert int(mo.ptr) == 2 and mo.queue_src.tolist() == [3, 3, 1, 1, 2, 2]


def test_collapse_loss_level():
    # A collapsed model scores every candidate equally: loss = log(#candidates).
    assert abs(math.log(16385) - 9.70) < 0.005
    loss, _, _ = infonce([0.5] * 8)
    assert abs(loss.item() - math.log(8)) < 1e-5


def _mc_setup(n_local, seed=0, B=4, D=8, K=32):
    g = torch.Generator().manual_seed(seed)
    unit = lambda *s: torch.nn.functional.normalize(torch.randn(*s, generator=g), dim=-1)
    q1, q2, k1, k2 = (unit(B, D) for _ in range(4))
    ql = unit(n_local, B, D) if n_local else None
    queue = unit(K, D)
    return q1, q2, k1, k2, ql, torch.tensor([0, 1, 2, 0][:B]), queue, torch.tensor([0, 1, 2, 3] * (K // 4))


def test_multicrop_equals_the_ten_pair_loop():
    q1, q2, k1, k2, ql, src, queue, qsrc = _mc_setup(4)
    for t in (q1, q2, ql):
        t.requires_grad_(True)
    kw = dict(tau=0.2, beta=2.0)
    fast, _ = multicrop_loss(q1, q2, k1, k2, ql, src, queue, qsrc, torch.ones(32), mine=False, **kw)
    pairs = [(q1, k2), (q2, k1)] + [(ql[j], k) for j in range(4) for k in (k1, k2)]
    assert len(pairs) == 10  # 2 global-global + 4 locals x 2 global keys (section 7.3)
    slow = sum(trace_loss(q, k, src, queue, qsrc, torch.ones(32), mine=False, **kw)[0] for q, k in pairs) / 10
    assert torch.allclose(fast, slow, atol=1e-6)
    g_fast = torch.autograd.grad(fast, (q1, q2, ql))
    g_slow = torch.autograd.grad(slow, (q1, q2, ql))
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(g_fast, g_slow))


def test_multicrop_without_locals_is_two_view_moco():
    q1, q2, k1, k2, _, src, queue, qsrc = _mc_setup(0)
    a, _ = multicrop_loss(q1, q2, k1, k2, None, src, queue, qsrc, torch.ones(32), mine=False, beta=1.0)
    l1, _ = trace_loss(q1, k2, src, queue, qsrc, torch.ones(32), mine=False, beta=1.0)
    l2, _ = trace_loss(q2, k1, src, queue, qsrc, torch.ones(32), mine=False, beta=1.0)
    assert torch.allclose(a, (l1 + l2) / 2)
    empty = torch.zeros(0, 4, 8)
    b, _ = multicrop_loss(q1, q2, k1, k2, empty, src, queue, qsrc, torch.ones(32), mine=False, beta=1.0)
    assert torch.allclose(a, b)


def test_locals_never_mined_but_beta_applies():
    # mining on, and queue_kth = -1 makes the mutual condition always true, so global anchors do attract keys;
    # the locals must still be scored exactly as unmined pairs
    q1, q2, k1, k2, ql, src, queue, qsrc = _mc_setup(2, K=400)
    kth = -torch.ones(400)
    kw = dict(beta=2.0, top_frac=0.2, knn=10)
    fast, st = multicrop_loss(q1, q2, k1, k2, ql, src, queue, qsrc, kth, mine=True, **kw)
    assert st["attracted_per_anchor"] > 0  # the globals were mined
    glob = [trace_loss(q, k, src, queue, qsrc, kth, mine=True, **kw)[0] for q, k in [(q1, k2), (q2, k1)]]
    loc = [trace_loss(ql[j], k, src, queue, qsrc, kth, mine=False, **kw)[0] for j in range(2) for k in (k1, k2)]
    assert torch.allclose(fast, (sum(glob) + sum(loc)) / 6, atol=1e-6)
    lo1, _ = multicrop_loss(q1, q2, k1, k2, ql, src, queue, qsrc, kth, mine=False, beta=1.0)
    lo2, _ = multicrop_loss(q1, q2, k1, k2, ql, src, queue, qsrc, kth, mine=False, beta=2.0)
    assert lo2 > lo1  # same-source negatives weigh more on every pair, locals included
