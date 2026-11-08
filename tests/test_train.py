"""Unit tests for the training schedule."""
import math

from encoder.train import lr_at


def test_warmup_then_cosine():
    base, total, warm = 0.00375, 1000, 100
    assert abs(lr_at(0, total, warm, base) - base / 100) < 1e-12  # step 1 of 100
    assert abs(lr_at(99, total, warm, base) - base) < 1e-12  # end of warm-up
    assert abs(lr_at(100, total, warm, base) - base) < 1e-12  # cosine starts at the top
    assert abs(lr_at(550, total, warm, base) - base / 2) < 1e-12  # halfway down: cos(pi/2) = 0
    assert lr_at(1000, total, warm, base) < 1e-12  # ends at zero
    mid = lr_at(325, total, warm, base)
    assert abs(mid - 0.5 * base * (1 + math.cos(math.pi * 0.25))) < 1e-12
