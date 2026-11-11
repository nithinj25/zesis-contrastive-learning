"""Finding detectors (findings/probes.py): section 16 numbers and toy checks."""
import numpy as np
import pytest

from findings.probes import (fit_predict_lr, fit_predict_mlp, oof, platt_apply, platt_fit, pos_weight,
                             prototype_scores, reliability)


def test_pos_weight_section_16():
    assert pos_weight(40, 1920) == 47  # 1,880 negatives / 40 positives


def test_reliability_bands():
    assert [reliability(a) for a in (0.9, 0.75, 0.74, 0.60, 0.59)] == [1.0, 1.0, 0.5, 0.5, 0.0]


def test_platt_makes_scores_mean_what_they_say():
    # raw scores are 3x too confident: true probability = sigmoid(score / 3)
    rng = np.random.default_rng(0)
    s = rng.normal(0, 4, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-s / 3))).astype(float)
    a, b = platt_fit(s, y)
    assert a == pytest.approx(1 / 3, abs=0.02) and b == pytest.approx(0, abs=0.05)
    p = platt_apply(s, (a, b))
    hot = (p > 0.85) & (p < 0.95)
    assert y[hot].mean() == pytest.approx(0.9, abs=0.03)  # "0.9" is right about 90% of the time


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, 2)) < 0.3).astype(np.float32)
    X = rng.standard_normal((n, 12)).astype(np.float32)
    X[:, 0] += 3 * Y[:, 0]
    X[:, 1] += 3 * Y[:, 1]
    return X, Y


def test_both_detectors_learn_a_separable_toy_out_of_fold():
    from sklearn.metrics import roc_auc_score

    X, Y = _toy()
    groups = np.arange(len(X)).astype(str)
    for fn in (lambda a, b, c: fit_predict_lr(a, b, c, 0.1), lambda a, b, c: fit_predict_mlp(a, b, c, epochs=200)):
        S = oof(fn, X, Y, groups)
        assert S.shape == Y.shape and min(roc_auc_score(Y[:, j], S[:, j]) for j in range(2)) > 0.9


def test_prototype_scores_point_at_the_positives():
    X = np.array([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    y = np.array([1, 1, 0, 0])
    s = prototype_scores(X, y, X)
    assert s[:2].min() > s[2:].max() and s[0] == pytest.approx(0.9986, abs=1e-3)  # cos((1,0), (0.95,0.05))
