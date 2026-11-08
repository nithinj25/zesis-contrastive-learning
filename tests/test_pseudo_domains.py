"""Unit tests for the pseudo-domain check (scripts/pseudo_domains.py, CLAUDE.md section 5.5)."""
import importlib.util
import os

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "pseudo_domains", os.path.join(os.path.dirname(__file__), "..", "scripts", "pseudo_domains.py"))
pdm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdm)


def test_completeness_dry_run():
    # camera a: 3 photos in clusters 0, 0, 1 -> dominant share 2/3; camera b: both in cluster 2 -> 1.0
    out = pdm.completeness(np.array(["a", "a", "a", "b", "b"]), np.array([0, 0, 1, 2, 2]))
    assert abs(out["a"] - 2 / 3) < 1e-12 and out["b"] == 1.0


def test_two_sample_auroc_same_vs_different():
    rng = np.random.default_rng(0)
    n, d = 200, 6
    a, b = rng.normal(0, 1, (n, d)), rng.normal(0, 1, (n, d))
    ga, gb = np.arange(n), np.arange(n, 2 * n)
    same = pdm.two_sample_auroc(a, b, ga, gb)
    shifted = pdm.two_sample_auroc(a, b + 1.5, ga, gb)
    assert abs(same - 0.5) < 0.1  # one distribution: a classifier cannot tell the groups apart
    assert shifted > 0.95  # mean shifted by 1.5 SD in every feature: easily told apart


def test_two_sample_auroc_keeps_copies_in_one_fold():
    # every photo in b is an exact copy of one in a, in the same group: no fold trains on one copy and tests the
    # other, so the classifier cannot "recognise" copies; with identical rows the AUROC must sit at chance
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, (100, 4))
    g = np.arange(100)
    assert abs(pdm.two_sample_auroc(a, a.copy(), g, g) - 0.5) < 0.1
