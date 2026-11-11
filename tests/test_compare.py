"""Paired bootstrap (eval/compare.py) on hand-checkable cases."""
import numpy as np
import pytest

from eval.compare import macro_auc, paired_bootstrap

Y = np.array([0, 0, 1, 1, 0, 1, 0, 1], float)[:, None]


def test_macro_auc_hand_example():
    # positives score 0.6, 0.7, 0.9, 0.8; negatives 0.1, 0.2, 0.65, 0.3: of 16 pairs one is wrong (0.6 < 0.65)
    s = np.array([0.1, 0.2, 0.6, 0.7, 0.65, 0.9, 0.3, 0.8])[:, None]
    assert macro_auc(Y, s) == pytest.approx(15 / 16)


def test_identical_encoders_differ_by_zero():
    rng = np.random.default_rng(0)
    Yb, s = (rng.random((100, 1)) < 0.5).astype(float), rng.random((100, 1))
    r = paired_bootstrap(Yb, s, s, n=200)
    assert r["diff"] == 0 and r["ci95"] == [0.0, 0.0] and r["p_not_better"] == 1.0


def test_perfect_beats_random_on_every_draw():
    rng = np.random.default_rng(1)
    Yb = (rng.random((200, 3)) < 0.4).astype(float)
    perfect, noise = Yb + 0.01 * rng.random(Yb.shape), rng.random(Yb.shape)
    r = paired_bootstrap(Yb, perfect, noise, n=200)
    assert r["a"] == 1.0 and r["ci95"][0] > 0.3 and r["p_not_better"] == 0.0
