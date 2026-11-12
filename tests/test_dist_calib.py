"""Distribution calibration (graph/dist_calib.py) on hand-checkable toy numbers."""
import numpy as np

from graph.dist_calib import BaseStats, augment, tukey


def toy_stats(rng):
    # three base diseases in 2-d: means (0,0), (4,0), (10,10), with different known spreads
    A = rng.normal([0, 0], [1.0, 0.2], (4000, 2))
    B = rng.normal([4, 0], [0.2, 1.0], (4000, 2))
    C = rng.normal([10, 10], [3.0, 3.0], (4000, 2))
    return BaseStats({"A": A, "B": B, "C": C}), (A, B, C)


def test_tukey():
    assert np.allclose(tukey(np.array([0.25, 0.0, 1.0]), 0.5), [0.5, 0.0, 1.0])


def test_calibrated_mean_dry_run():
    # x = (1, 2): distances to the means 2.24, 3.61, 12.04 -> nearest two are A and B
    # mu' = ((1,2) + (0,0) + (4,0)) / 3 = (5/3, 2/3)
    s = BaseStats({"A": np.array([[-1.0, 0], [1, 0]]), "B": np.array([[3.0, 0], [5, 0]]),
                   "C": np.array([[10.0, 9], [10, 11]])})
    x = np.array([1.0, 2.0])
    assert list(s.nearest(x, 2)) == [0, 1]
    assert np.allclose(s.calibrated_mean(x, 2), [5 / 3, 2 / 3])
    assert np.allclose(s.calibrated_mean(x, 2, w=None), x)  # spread-only variant keeps the photo's own position


def test_sample_covariance_is_average_of_nearest_plus_alpha():
    rng = np.random.default_rng(0)
    s, (A, B, _) = toy_stats(rng)
    x = np.array([1.0, 2.0])
    pts = s.sample(x, 100_000, k=2, alpha=0.5, rng=np.random.default_rng(1))
    want = (np.cov(A.T) + np.cov(B.T)) / 2 + 0.5 * np.eye(2)  # [[1.02+.5, 0], [0, 0.52+.5]] approx
    assert np.allclose(np.cov(pts.T), want, atol=0.03)
    assert np.allclose(pts.mean(0), s.calibrated_mean(x, 2), atol=0.02)


def test_augment_splits_evenly_over_support():
    s, _ = toy_stats(np.random.default_rng(0))
    out = augment(np.array([[1.0, 2.0], [9.0, 9.0]]), s, n_total=100, k=1)
    assert out.shape == (100, 2)
    # first photo borrows A's spread around (0.5, 1); second borrows C's around (9.5, 9.5)
    assert np.allclose(out[:50].mean(0), [0.5, 1.0], atol=0.4) and np.allclose(out[50:].mean(0), [9.5, 9.5], atol=1.2)
