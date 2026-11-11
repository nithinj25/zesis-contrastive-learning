"""Leave-one-camera-out helpers (eval/leave_camera_out.py) on toy data."""
import numpy as np

from eval.leave_camera_out import centre_by_camera, gap_bootstrap, same_camera_share


def _two_cameras(seed=0, n=40, offset=8.0):
    rng = np.random.default_rng(seed)
    cams = np.repeat(["a", "b"], n)
    X = rng.standard_normal((2 * n, 8))
    X[cams == "b", 0] += offset  # camera b sits far away along one direction
    return X / np.linalg.norm(X, axis=1, keepdims=True), cams


def test_centring_removes_each_cameras_mean_and_keeps_unit_length():
    X, cams = _two_cameras()
    Z = centre_by_camera(X, cams)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0)
    # before re-normalising, each camera's mean is zero, so after it the means are small
    assert all(np.abs(Z[cams == c].mean(0)).max() < 0.1 for c in "ab")


def test_offset_cameras_are_each_others_strangers_until_centred():
    X, cams = _two_cameras()
    groups = np.arange(len(X)).astype(str)
    share, chance = same_camera_share(X, cams, groups, k=10)
    assert chance == 0.5 and share > 0.85
    share_c, _ = same_camera_share(centre_by_camera(X, cams), cams, groups, k=10)
    assert share_c < 0.7


def test_same_eye_copies_are_not_neighbours():
    X = np.array([[1.0, 0], [1.0, 0], [0, 1.0], [0.6, 0.8]])
    cams = np.array(["a", "a", "b", "b"])
    groups = np.array(["e1", "e1", "e2", "e3"])  # rows 0 and 1 are the same eye
    share, _ = same_camera_share(X, cams, groups, k=1)
    # rows 0, 1 -> row 3 (other camera; their true nearest, each other, is a copy and excluded);
    # row 2 -> row 3 (0.8, same camera); row 3 -> row 2 (0.8, same camera). Share 2/4. Counting copies would give 0.75.
    assert share == 0.5


def test_identical_classifiers_have_zero_gap():
    rng = np.random.default_rng(0)
    Y = (rng.random((60, 2)) < 0.4).astype(float)
    S = rng.random((60, 2))
    g = gap_bootstrap(Y, [S, S], S, n=100)
    assert g["gap"] == 0 and g["ci95"] == [0.0, 0.0] and g["in"] == g["out"]
