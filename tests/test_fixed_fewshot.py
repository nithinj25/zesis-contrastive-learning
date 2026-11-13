import numpy as np
import pandas as pd

from eval.fixed_fewshot import knn_scores, l2, lock_heads, logistic_scores, our_id, prototype_scores


def _toy(seed=0):
    rng = np.random.default_rng(seed)
    S = l2(np.r_[rng.normal(1, 0.3, (10, 4)), rng.normal(-1, 0.3, (10, 4))])
    Q = l2(np.r_[rng.normal(1, 0.3, (5, 4)), rng.normal(-1, 0.3, (5, 4))])
    return S, Q, np.arange(10), np.arange(10, 20)


def test_heads_rank_positives_first():
    S, Q, pos, neg = _toy()
    for s in (prototype_scores(S, pos, neg, Q), knn_scores(S, pos, neg, Q, 3), logistic_scores(S, pos, neg, Q, 1.0, 0)):
        assert s[:5].min() > s[5:].max()


def test_prototype_margin_is_cosine_difference():
    S = l2(np.array([[1, 0], [0, 1]], float))
    s = prototype_scores(S, np.array([0]), np.array([1]), l2(np.array([[1, 0]], float)))
    assert np.isclose(s[0], 1.0)


def test_record_ids_map_to_ours():
    assert our_id("rfmid:training:95") == "rfmid_train_95"
    assert our_id("rfmid:validation:1") == "rfmid_val_1"


def test_lock_prefers_best_lift_then_smaller_parameter():
    m = pd.DataFrame([
        {"head": "logistic", "parameter": "0.1", "support_protocol": "a", "k": 3, "lift": 0.2},
        {"head": "logistic", "parameter": "0.01", "support_protocol": "a", "k": 3, "lift": 0.2},
        {"head": "prototype", "parameter": "none", "support_protocol": "a", "k": 3, "lift": 0.1},
    ])
    assert lock_heads(m)[("a", 3)] == ("logistic", 0.01)
