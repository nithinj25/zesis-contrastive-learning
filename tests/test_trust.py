"""Trust layer maths (trust/calibrate.py, trust/conformal.py) against the section 16 numbers."""
import numpy as np
import pytest

from trust.calibrate import ece, fit_temperature, nll, softmax
from trust.conformal import Mondrian, calib_scores, conformal_k, prediction_sets, q_hat


def test_ece_bin_section_16():
    # 100 of 1,000 predictions say 0.9 and are right 70%; the other 900 are perfectly calibrated at 0.5
    conf = np.r_[np.full(100, 0.9), np.full(900, 0.5)]
    correct = np.r_[np.r_[np.ones(70), np.zeros(30)], np.r_[np.ones(450), np.zeros(450)]]
    assert ece(conf, correct) == pytest.approx(0.02)


def test_temperature_recovers_known_overconfidence_and_keeps_ranking():
    rng = np.random.default_rng(0)
    z_true = rng.normal(0, 1.5, (20000, 4))
    y = np.array([rng.choice(4, p=p) for p in softmax(z_true)])
    z = 3 * z_true                          # a model three times too confident
    T = fit_temperature(z, y)
    assert T == pytest.approx(3.0, rel=0.05)
    assert nll(z, y, T) < nll(z, y, 1.0)
    assert (softmax(z, T).argmax(1) == z.argmax(1)).all()


def test_conformal_worked_example_section_10_4():
    assert conformal_k(9, 0.1) == 9
    scores = np.array([0.05, 0.10, 0.12, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70])
    qh = q_hat(scores, 0.1)
    assert qh == pytest.approx(0.70)
    p = np.array([[0.52, 0.40, 0.05, 0.03]])  # AION, vein occlusion, glaucoma, DR
    assert prediction_sets(p, qh)[0].tolist() == [True, True, False, False]


def test_too_few_calibration_photos_cannot_promise_coverage():
    assert q_hat(np.array([0.1, 0.2]), 0.1) == float("inf")  # k = 3 > n = 2: every disease goes in


def test_coverage_holds_on_exchangeable_data():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (6000, 5))
    p = softmax(z)
    y = np.array([rng.choice(5, p=q) for q in p])
    cal, test = slice(0, 3000), slice(3000, None)
    qh = q_hat(calib_scores(p[cal], y[cal]), 0.1)
    cover = prediction_sets(p[test], qh)[np.arange(3000), y[test]].mean()
    assert 0.88 <= cover <= 0.93


def test_mondrian_small_or_unknown_groups_use_the_most_conservative_threshold():
    rng = np.random.default_rng(2)
    p = softmax(rng.normal(0, 2, (400, 3)))
    y = rng.integers(0, 3, 400)
    groups = np.array(["a"] * 300 + ["b"] * 80 + ["c"] * 20)
    m = Mondrian(alpha=0.1, min_group=50).fit(p, y, groups)
    assert set(m.q) == {"a", "b"} and m.n["c"] == 20
    th = m.threshold(np.array(["a", "c", "zeiss_new"]))
    assert th[0] == m.q["a"] and th[1] == th[2] == m.q_conservative >= max(m.q.values())


def test_knn_ood_flags_a_far_cluster_and_five_percent_of_familiar_photos():
    from trust.ood import KnnOOD

    rng = np.random.default_rng(3)
    centre = np.zeros(16); centre[0] = 1
    near = lambda n: centre + 0.15 * rng.standard_normal((n, 16))
    far = np.zeros(16); far[1] = 1
    ood = KnnOOD(k=10).fit(near(2000)).calibrate(near(400))
    fam = ood.flag(near(2000))
    assert 0.03 <= fam.mean() <= 0.08                     # ~5% of familiar photos flagged by design
    assert ood.flag(far + 0.15 * rng.standard_normal((200, 16))).mean() > 0.99
    sims, idx = ood.neighbours(near(1), 5)
    assert sims.shape == idx.shape == (1, 5) and (np.diff(sims[0]) <= 1e-6).all()   # nearest first


def test_knn_ood_per_group_thresholds_fall_back_for_small_and_unseen_groups():
    from trust.ood import KnnOOD

    rng = np.random.default_rng(4)
    X = rng.standard_normal((500, 8))
    ood = KnnOOD(k=5, min_group=50).fit(X)
    val = rng.standard_normal((120, 8))
    groups = np.array(["a"] * 100 + ["b"] * 20)
    ood.calibrate(val, groups)
    assert set(ood.thr) == {"a"}
    th = ood.threshold(["a", "b", "paraguay"])
    assert th[0] == ood.thr["a"] and th[1] == th[2] == ood.thr_all


def test_patient_split_keeps_both_eyes_together():
    from trust.quality import patient_split

    patients = np.repeat(np.arange(50), 2)            # two eyes each
    te = patient_split(patients, 0.2, seed=0)
    assert te.sum() == 20 and set(patients[te]).isdisjoint(set(patients[~te]))


def test_quality_threshold_catches_the_target_share_of_rejects():
    from trust.quality import GRADES, QualityGate

    rng = np.random.default_rng(5)
    grade = rng.choice(GRADES, 900, p=[0.6, 0.2, 0.2])
    X = rng.standard_normal((900, 6))
    X[grade == "reject", 0] += 3                      # rejects differ along one direction
    gate = QualityGate(C=1.0).fit(X, grade)
    p = gate.p_reject(X)
    gate.set_threshold(p, grade == "reject", recall=0.9)
    assert abs(gate.retake(X)[grade == "reject"].mean() - 0.9) < 0.02
    assert gate.retake(X)[grade == "good"].mean() < 0.1


def test_decision_rule_order_and_outcomes():
    from trust.decide import DIAGNOSE, REFER, RETAKE, SHORTLIST, decide

    assert decide(True, True, "CRVO", ["DR"]).action == RETAKE          # quality first, stops everything else
    assert decide(False, True, "CRVO", ["DR"]).action == REFER           # unfamiliar beats a rare call
    d = decide(False, False, "CRVO", ["DR"])
    assert d.action == REFER and d.diseases == ["CRVO"]                   # rare suspicion is referred, not diagnosed
    assert decide(False, False, None, ["DR"]).action == DIAGNOSE
    assert decide(False, False, None, ["DR"], disagree=True).action == SHORTLIST
    assert decide(False, False, None, ["DR", "BRVO"]).action == SHORTLIST
    assert decide(False, False, None, ["DR", "BRVO", "ARMD"]).action == SHORTLIST
    assert decide(False, False, None, ["a", "b", "c", "d"]).action == REFER
    assert decide(False, False, None, []).action == REFER


def test_other_abnormal_alone_is_referred_never_diagnosed():
    from trust.decide import REFER, SHORTLIST, decide

    assert decide(False, False, None, ["other_abnormal"]).action == REFER
    assert decide(False, False, None, ["DR", "other_abnormal"]).action == SHORTLIST


def test_single_dx_adds_other_abnormal_for_findings_only_photos():
    import pandas as pd

    from trust.classifier import CLASSES, OTHER, single_dx

    meta = pd.DataFrame({"label_ok": [True] * 5,
                         "diseases": [["DR"], ["normal"], [], ["DR", "BRVO"], []],
                         "findings": [[], [], ["disc_cupping"], [], []]}, index=list("abcde"))
    s = single_dx(meta, meta.index)
    assert s.dz.to_dict() == {"a": "DR", "b": "normal", "c": OTHER}       # several diseases / nothing -> left out
    assert s.y["c"] == CLASSES.index(OTHER)
