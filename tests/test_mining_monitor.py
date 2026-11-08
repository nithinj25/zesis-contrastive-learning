"""Mining monitor (encoder/mining_monitor.py) on a hand-worked six-photo example."""
import numpy as np
import pandas as pd
import pytest
import torch

from encoder.mining_monitor import MiningMonitor, pool_labels

# photo: source, label set, DR status, eye group
#   0: src 0, {DR}        DR+  eye A
#   1: src 1, {DR}        DR+  eye A (a copy of photo 0's eye)
#   2: src 1, {normal}    DR-
#   3: src 0, {normal}    DR-
#   4: src 2, unknown     DR+  (EyePACS grade 2)
#   5: src 2, unknown     DR-  (EyePACS grade 0)
SETS = [{"DR"}, {"DR"}, {"normal"}, {"normal"}, None, None]
DR = [True, True, False, False, True, False]
SRC = np.array([0, 1, 1, 0, 2, 2])
GROUPS = np.array([10, 10, 12, 13, 14, 15])


def monitor():
    return MiningMonitor(SETS, DR, SRC, GROUPS, n_src=3, chance_pairs=2_000_000)


def test_chance_matches_hand_count():
    # labels: cross-source pairs of known photos {0, 3 | 1, 2}: (0,1) pure, (0,2) no, (3,1) no, (3,2) normal-normal
    # left out -> 1/3. DR: 12 cross-source pairs, 3 with neither DR left out, of the other 9 three are DR+/DR+ -> 1/3
    m = monitor()
    assert m.chance == pytest.approx(1 / 3, abs=0.003) and m.dr_chance == pytest.approx(1 / 3, abs=0.003)


def test_one_step_dry_run():
    m = monitor()
    queue_idx = torch.tensor([1, 2, 4, 5, 3])  # queue holds photos 1, 2, 4, 5, 3
    queue_src = torch.tensor(SRC)[queue_idx]
    attract = torch.tensor([[1, 1, 1, 0, 0],   # anchor photo 0 attracts photos 1, 2, 4
                            [0, 1, 0, 1, 0]],  # anchor photo 3 attracts photos 2, 5
                           dtype=torch.bool)
    m.update(attract, torch.tensor([0, 3]), torch.tensor([0, 0]), queue_idx, queue_src,
             {"cross": 8, "high": 10, "top": 6, "mutual": 8, "attracted": 5})
    s = m.summary()
    assert s["attracted_per_anchor"] == 2.5 and s["anchors_with_attraction"] == 1.0
    assert s["blocked_by_topk"] == pytest.approx(0.4) and s["blocked_by_mutual"] == pytest.approx(0.2)
    # labels: (0,1) pure, (0,2) impure; (0,4), (3,5) unknown; (3,2) normal-normal left out -> 1 of 2
    assert s["purity"] == 0.5 and s["purity_pairs"] == 2 and s["lift"] == pytest.approx(1.5, abs=0.02)
    # DR: (0,1) +/+, (0,2) +/-, (0,4) +/+ -> 2 of 3; (3,2), (3,5) -/- left out
    assert s["dr_purity"] == pytest.approx(2 / 3) and s["dr_pairs"] == 3
    assert s["same_eye_share"] == pytest.approx(1 / 5)  # (0,1) is one eye
    # sources: anchor source 0 attracted keys of source 1 three times, source 2 twice. The queue holds sources
    # [1, 1, 2, 2, 0]: cross-source keys for source 0 are half source 1, half source 2 -> expected 2.5 each
    assert s["source_obs"][0][1] == 3 and s["source_obs"][0][2] == 2
    assert s["source_ratio"][0][1] == pytest.approx(1.2) and s["source_ratio"][0][2] == pytest.approx(0.8)
    # per source pair: 0 -> 1 holds (0,1) and (0,2): labels 1 of 2, DR 1 of 2; 0 -> 2 holds (0,4): DR 1 of 1
    assert s["source_purity"][0][1] == 0.5 and s["source_purity_pairs"][0][1] == 2
    assert s["source_dr_purity"][0][1] == 0.5 and s["source_dr_purity"][0][2] == 1.0 and s["source_dr_pairs"][0][2] == 1
    assert np.isnan(s["source_purity"][0][2])  # (0,4): photo 4's labels unknown
    m.reset()
    assert m.summary()["attracted_per_anchor"] == 0


def test_pool_labels_from_meta_rows():
    rows = pd.DataFrame({"dataset": ["rfmid", "rfmid", "eyepacs", "aptos", "rfmid"],
                         "label_ok": [True, True, True, True, True],
                         "findings": [["haemorrhage"], [], [], [], []],
                         "diseases": [["DR"], ["normal"], [], [], []],
                         "labels": [["DR", "HR"], [], ["eyepacs_dr0"], ["aptos_dr3"], ["XYZ"]]})
    sets, dr = pool_labels(rows)
    assert sets == [{"haemorrhage", "DR"}, {"normal"}, None, None, None]  # last: only an unmapped label -> unknown
    assert dr == [True, False, False, True, None]


def test_queue_remembers_pool_rows_and_loss_returns_the_mask():
    from encoder.loss import trace_loss
    from encoder.moco import MoCo

    mo = MoCo("random", dim=8, K=6, knn=2)
    k = torch.nn.functional.normalize(torch.randn(4, 8), dim=1)
    mo.enqueue(k, torch.tensor([0, 1, 0, 1]), torch.tensor([7, 8, 9, 10]))
    mo.enqueue(k[:3], torch.tensor([1, 1, 0]), torch.tensor([11, 12, 13], dtype=torch.int32))  # loader dtype on Windows
    # K = 6: slots 0-3 get rows 7-10, then 11, 12 fill slots 4, 5 and 13 wraps round into slot 0
    assert mo.queue_idx.tolist() == [13, 8, 9, 10, 11, 12]
    _, st = trace_loss(k, k, torch.tensor([0, 1, 0, 1]), mo.queue, mo.queue_src, mo.queue_kth, mine=True, top_frac=0.5)
    assert st["attract"].shape == (4, 6) and st["attract"].dtype == torch.bool
