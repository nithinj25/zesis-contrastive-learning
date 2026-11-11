"""Few-shot evaluation (eval/fewshot.py) on toy data with known answers."""
import numpy as np

from eval.fewshot import balanced_topk, draw, fewshot, rank_of_truth


def test_rank_of_truth_hand_example():
    # row 0: truth col 2 has the top score -> rank 1; row 1: truth col 0 is third -> rank 3; ties count against
    S = np.array([[0.1, 0.2, 0.9], [0.1, 0.5, 0.4], [0.5, 0.5, 0.0]])
    assert rank_of_truth(S, np.array([2, 0, 0])).tolist() == [1, 3, 2]


def test_balanced_topk_weights_classes_equally():
    # class a: 3 of 4 right, class b: 0 of 1 right -> balanced 0.375 (plain accuracy would be 0.6)
    rank = np.array([1, 1, 1, 2, 3])
    y = np.array(["a", "a", "a", "a", "b"])
    acc, per = balanced_topk(rank, y, 1)
    assert per == {"a": 0.75, "b": 0.0} and acc == 0.375


def test_support_and_query_never_share_an_eye():
    rng = np.random.default_rng(0)
    y = np.repeat(["a", "b", "c"], 6)
    groups = np.array([f"g{i // 2}" for i in range(18)])  # every eye photographed twice
    sup, qry, kept = draw(y, groups, 2, rng)
    assert len(sup) == 6 and set(groups[sup]).isdisjoint(groups[qry])
    assert all((y[sup] == c).sum() == 2 for c in kept)


def test_class_with_too_few_eyes_is_skipped():
    y = np.array(["a"] * 5 + ["b"] * 2)
    groups = np.arange(7).astype(str)
    _, qry, kept = draw(y, groups, 2, np.random.default_rng(0))  # b has 2 eyes: no query left at K = 2
    assert kept.tolist() == ["a"] and set(y[qry]) == {"a"}


def test_separated_clusters_are_perfect():
    rng = np.random.default_rng(1)
    centres = np.eye(6)[:3] * 5
    X = np.concatenate([c + 0.1 * rng.standard_normal((10, 6)) for c in centres])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.repeat(["a", "b", "c"], 10)
    r = fewshot(X, y, np.arange(30).astype(str), K=1, seeds=3)
    assert r["prototype"]["top1"] == 1.0 and r["logistic"]["top1"] == 1.0 and r["prototype"]["n_classes"] == 3
