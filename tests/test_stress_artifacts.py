import cv2
import numpy as np

from augment.stress_artifacts import ARTIFACTS, corrupt


def _photo():
    rgb = np.zeros((128, 128, 3), np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    cv2.circle(mask, (64, 64), 60, 1, -1)
    rgb[mask > 0] = (200, 110, 50)
    return rgb, mask


def test_eleven_artifacts():
    assert len(ARTIFACTS) == 11


def test_level_zero_is_identity_and_levels_are_reproducible():
    rgb, mask = _photo()
    assert np.array_equal(corrupt(rgb, mask, "haze", 0, "x"), rgb)
    for name in ARTIFACTS:
        a, b = corrupt(rgb, mask, name, 3, "x"), corrupt(rgb, mask, name, 3, "x")
        assert np.array_equal(a, b), name
        assert a.shape == rgb.shape and a.dtype == np.uint8
        assert (a[mask == 0] == 0).all(), name  # never touches the background


def test_stronger_level_changes_more():
    rgb, mask = _photo()
    for name in ("haze", "defocus_blur", "color_cast", "vignetting"):
        d = [np.abs(corrupt(rgb, mask, name, l, "x").astype(int) - rgb).mean() for l in (1, 5)]
        assert d[1] > d[0], name


def test_rgb_bgr_round_trip_keeps_their_colours():
    # their haze veil is BGR (150, 170, 190): in RGB the veil must push towards (190, 170, 150)
    rgb, mask = _photo()
    out = corrupt(np.zeros_like(rgb), mask, "haze", 5, "x")
    r, g, b = out[mask > 0].mean(0)
    assert r > g > b
