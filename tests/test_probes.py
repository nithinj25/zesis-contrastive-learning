"""Unit tests for the frozen-feature evaluation helpers."""
import numpy as np

from eval.probes import boot_ci, choose_C, knn_scores


def test_knn_vote_by_hand():
    # Two training points: label 1 at angle 0, label 0 at 90 degrees. Query at 30 degrees, k = 2, t = 0.5:
    # weights exp(cos30/0.5) = 5.65, exp(cos60/0.5) = 2.72 -> score 5.65 / 8.37 = 0.675
    Xtr = np.array([[1.0, 0.0], [0.0, 1.0]])
    Ytr = np.array([[1.0], [0.0]])
    q = np.array([[np.cos(np.pi / 6), np.sin(np.pi / 6)]])
    s = knn_scores(Xtr, Ytr, q, k=1, t=0.5)  # k=1 keeps the nearest only
    assert s[0, 0] == 1.0
    s2 = knn_scores(np.vstack([Xtr, [[-1.0, 0.0]]]), np.vstack([Ytr, [[0.0]]]), q, k=2, t=0.5)
    w1, w2 = np.exp(np.cos(np.pi / 6) / 0.5), np.exp(np.cos(np.pi / 3) / 0.5)
    assert abs(s2[0, 0] - w1 / (w1 + w2)) < 1e-9 and abs(s2[0, 0] - 0.675) < 0.001


def test_choose_C_and_bootstrap_on_separable_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    Y = (X[:, :2] > 0).astype(float)
    C, scores = choose_C(X, Y, groups=np.arange(200) // 2)
    assert C in scores and scores[C] > 0.9
    lo, hi = boot_ci(Y, X[:, :2] + 0.1 * rng.normal(size=(200, 2)), n=200)
    assert 0.85 < lo <= hi <= 1.0
