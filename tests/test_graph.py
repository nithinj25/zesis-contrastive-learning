"""Knowledge-graph scorer (graph/score.py, graph/build.py) against the section 16 numbers."""
import numpy as np
import pytest

from graph.build import beta_smooth, priors
from graph.score import evidence, explain, score


def test_beta_smoothing_section_16():
    assert beta_smooth(7, 9, 0.9) == pytest.approx(0.842, abs=1e-3)  # RP -> pigment
    assert beta_smooth(5, 6, 0.8) == pytest.approx(0.8125)            # AION -> disc edema
    assert beta_smooth(360, 400, 0.95) == pytest.approx(0.901, abs=1e-3)


def test_evidence_term_section_16():
    p, pn = np.array([[0.9]]), np.array([[0.1]])
    E = evidence(np.array([[0.9], [0.5], [0.1]]), p, pn)[:, 0, 0]
    assert E == pytest.approx([1.758, 0.0, -1.758], abs=1e-3)


def test_missing_detector_output_is_skipped():
    E = evidence(np.array([[np.nan, 0.9]]), np.array([[0.9, 0.9]]), np.array([[0.1, 0.1]]))
    assert E[0, 0, 0] == 0.0 and E[0, 0, 1] > 1.7


def test_worked_example_B_section_9_5():
    # diseases RP, DR, Normal; findings pigment, thin vessels, haemorrhage; shared background p'
    p = np.array([[0.90, 0.85, 0.02], [0.02, 0.10, 0.85], [0.01, 0.02, 0.01]])
    pn = np.tile([0.05, 0.10, 0.20], (3, 1))
    prior, r = np.array([0.01, 0.30, 0.69]), np.ones(3)
    s, post, _ = score(np.array([[0.9, 0.8, 0.1]]), p, pn, prior, r)
    assert s[0] == pytest.approx([-0.92, -3.39, -3.19], abs=0.01)
    assert post[0] == pytest.approx([0.842, 0.072, 0.087], abs=0.001)
    _, post_unsure, _ = score(np.array([[0.5, 0.8, 0.1]]), p, pn, prior, r)  # pigment unsure
    assert post_unsure[0] == pytest.approx([0.283, 0.275, 0.442], abs=0.002)


def test_worked_example_A_section_9_5():
    q = np.array([[0.91, 0.78, 0.03]])
    p = np.array([[0.875, 0.80, 0.05], [0.03, 0.03, 0.90]])     # AION, DR
    pn = np.array([[0.03, 0.04, 0.15], [0.04, 0.05, 0.05]])
    s, _, terms = score(q, p, pn, np.array([0.002, 0.3]), np.ones(3))
    assert terms[0, 0] == pytest.approx([2.885, 1.992, 0.075], abs=2e-3)
    assert s[0] == pytest.approx([-1.263, -3.956], abs=3e-3)    # AION beats DR despite a 150x smaller prior
    for_, against = explain(terms[0, 1], ["disc_edema", "disc_pallor", "microaneurysm"])
    assert against[0][0] == "microaneurysm" and for_ == []


def test_priors_floor_and_renormalise():
    pr = priors({"normal": 498, "DR": 359}, ["AION"], n_total=1000, pi_min=0.005)
    assert sum(pr.values()) == pytest.approx(1.0) and pr["AION"] >= 0.005 * 0.99
    assert pr["normal"] > pr["DR"] > pr["AION"]


def test_measurement_model_puts_literature_on_the_detector_scale():
    from graph.build import on_detector_scale

    # a detector that outputs 0.6 on average on true signs and 0.05 elsewhere; 80% of eyes show the sign
    assert on_detector_scale(0.8, 0.6, 0.05) == pytest.approx(0.8 * 0.6 + 0.2 * 0.05)  # 0.49, not 0.80
    assert on_detector_scale(1.0, 0.6, 0.05) == 0.6 and on_detector_scale(0.0, 0.6, 0.05) == 0.05
