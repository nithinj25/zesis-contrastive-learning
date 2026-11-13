import numpy as np

from scripts.umap_plots import chance, same_share


def _toy():
    X = np.eye(4)[[0, 0, 1, 1]] + 0.01 * np.arange(4)[:, None]
    return X / np.linalg.norm(X, axis=1, keepdims=True), np.array(["a", "a", "b", "b"])


def test_same_share_finds_own_group():
    X, g = _toy()
    assert same_share(X, g, k=1) == 1.0


def test_same_share_excludes_same_eye_copies():
    X, g = _toy()  # each photo's only same-group neighbour is its own copy, so masking it leaves only the other group
    assert same_share(X, g, k=1, block=np.array([0, 0, 1, 1])) == 0.0


def test_chance_is_sum_of_squared_shares():
    assert chance(np.array(["a", "a", "b", "b"])) == 0.5
    assert abs(chance(np.array(["a", "a", "a", "b"])) - 0.625) < 1e-12
