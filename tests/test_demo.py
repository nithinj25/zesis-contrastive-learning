"""Demo display helpers (the pipeline itself is checked against the evaluation by demo/check_pipeline.py)."""
import numpy as np
import pytest

pytest.importorskip("gradio")
pytest.importorskip("joblib")


@pytest.fixture(scope="module")
def app():
    import os

    if not os.path.exists("artifacts/trace_v2/demo_bundle.joblib"):
        pytest.skip("demo artifacts not built")
    import demo.app as a

    return a


def _result(**over):
    r = {"action": "diagnose", "reason": "one disease fits at 90% confidence", "diseases": ["DR"], "retina_found": True,
         "quality": {"p_reject": 0.01, "retake": False}, "ood": {"score": 0.05, "threshold": 0.09, "unfamiliar": False},
         "rare": {"margin": -0.1, "tau": 0.013, "called": None}, "conformal_set": ["DR"], "camera": None,
         "evidence": {"DR": {"for": [("haemorrhage", 0.8), ("disc_edema", 0.02)], "against": [("drusen", -0.3)]}}}
    r.update(over)
    return r


def test_card_names_the_action_and_disease(app):
    html = app.card(_result())
    assert "DIAGNOSE" in html and "Diabetic retinopathy" in html and "not a medical device" in html


def test_card_for_a_photo_without_retina(app):
    html = app.card({"action": "retake", "reason": "retina not found: please retake the photo", "diseases": [], "retina_found": False})
    assert "RETAKE" in html


def test_weak_evidence_is_hidden(app):
    md = app.evidence_md(_result())
    assert "haemorrhage (+0.80)" in md and "drusen (-0.30)" in md and "disc edema" not in md


def test_overlay_draws_boxes(app):
    from findings.localise import Landmarks

    lit, mask = np.full((512, 512, 3), 100, np.uint8), np.ones((512, 512), np.uint8)
    out = app.overlay(lit, mask, Landmarks(300, 250, 150, 260, True))
    assert out.shape == lit.shape and (out != 100).any()
