"""scripts/risk_coverage_plot.py: the curve behind docs/figures/risk_coverage.png."""
import numpy as np

from scripts.risk_coverage_plot import curve


def test_curve_by_hand():
    # confidences 0.9, 0.8, 0.7, 0.6 with correct = T, T, F, T -> errors among the top n: 0, 0, 1/3, 1/4
    cov, risk, aurc = curve(np.array([0.7, 0.9, 0.6, 0.8]), np.array([False, True, True, True]))
    assert np.allclose(cov, [0.25, 0.5, 0.75, 1.0])
    assert np.allclose(risk, [0, 0, 1 / 3, 1 / 4])
    assert np.isclose(aurc, (1 / 3 + 1 / 4) / 4)


def test_perfect_ranking_beats_reversed():
    conf = np.linspace(1, 0, 100)
    good = curve(conf, np.arange(100) < 80)[2]   # all 20 errors are the least confident
    bad = curve(conf, np.arange(100) >= 20)[2]   # all 20 errors are the most confident
    assert good < bad and np.isclose(curve(conf, np.arange(100) < 80)[1][-1], 0.2)
