"""Unit tests for quality grading, on synthetic feature tables with a known rule."""
import numpy as np
import pandas as pd

from data.quality import FEATURES, evaluate, grade_all


def table(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({f: rng.uniform(0, 1, n) for f in FEATURES})
    # The hidden rule: blurry -> reject; dark but sharp -> usable; otherwise good.
    d["eyeq_quality"] = np.where(d.q_sharpness < 0.3, "reject", np.where(d.q_L_mean < 0.4, "usable", "good"))
    d["patient"] = np.arange(n) // 2  # two eyes per patient
    d["image_id"] = [f"p{i}" for i in range(n)]
    d["fov_ok"] = True
    d["dataset"] = "eyepacs"
    return d


def test_tree_recovers_a_threshold_rule_on_held_out_patients():
    r = evaluate(table())
    assert r["accuracy"] > 0.95 and r["reject_recall"] > 0.95


def test_human_grade_wins_and_fov_failures_are_rejects():
    d = table(600)
    graded = d.copy()
    d.loc[0, "eyeq_quality"] = "usable"  # a human grade that the rule would not give
    d.loc[0, "q_sharpness"] = 0.9
    d.loc[0, "q_L_mean"] = 0.9
    d.loc[1:, "eyeq_quality"] = None
    d.loc[2, "fov_ok"] = False
    out, _ = grade_all(d, graded)
    out = out.set_index("image_id")
    assert out.q_grade["p0"] == "usable" and out.q_source["p0"] == "eyeq"
    assert out.q_grade["p2"] == "reject" and out.q_source["p2"] == "fov_fail"
    assert (out.q_source.drop(["p0", "p2"]) == "tree").all()


def test_other_datasets_stay_ungraded():
    d = table(400)
    graded = d.copy()
    d["eyeq_quality"] = None
    d.loc[:9, "dataset"] = "rfmid"
    d.loc[:9, "q_sharpness"] = 0.0  # the tree would call these reject
    out, _ = grade_all(d, graded)
    out = out.set_index("image_id")
    assert (out.q_grade[[f"p{i}" for i in range(10)]] == "ungraded").all()
    assert (out.q_pred[[f"p{i}" for i in range(10)]] == "reject").all()  # kept for analysis only
