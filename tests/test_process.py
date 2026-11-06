"""Unit tests for the offline processing pass, on synthetic retinas."""
import os

import cv2
import numpy as np

from data.process import colour_stats, load_processed, probe_features, process_one, quality_features
from data.preprocess import find_fov, standardise_geometry


def eye(seed=1, size=512, shape=None):
    rng = np.random.default_rng(seed)
    h, w = shape or (size, size)
    img = np.zeros((h, w, 3), np.uint8)
    r = min(h, w) // 2 - 4
    cv2.circle(img, (w // 2, h // 2), r, (170, 90, 40), -1)
    for _ in range(40):
        pts = np.cumsum(rng.normal(0, 18 * r / 252, (12, 2)), axis=0) + [w // 2, h // 2] + rng.integers(-r // 2, r // 2, 2)
        cv2.polylines(img, [pts.astype(np.int32)], False, (110, 40, 20), max(1, int(2 * r / 252)), cv2.LINE_AA)
    cv2.circle(img, (w // 2 + r // 2, h // 2), r // 6, (235, 190, 130), -1)
    return img


def std_of(img):
    return standardise_geometry(img, find_fov(img))


def test_blur_lowers_sharpness_and_detail():
    std, mask = std_of(eye())
    blurred = cv2.GaussianBlur(std, (0, 0), 3) * mask[..., None]
    a, b = quality_features(std, mask), quality_features(blurred, mask)
    assert b["q_sharpness"] < 0.5 * a["q_sharpness"] and b["q_detail"] < 0.5 * a["q_detail"]


def test_dark_photo_has_lower_exposure_and_more_dark_pixels():
    std, mask = std_of(eye())
    dark = (std * 0.15).astype(np.uint8)
    a, b = quality_features(std, mask), quality_features(dark, mask)
    assert b["q_L_mean"] < a["q_L_mean"] - 20 and b["q_dark_frac"] > a["q_dark_frac"]


def test_vignetting_shows_as_centre_rim_gap():
    std, mask = std_of(eye())
    yy, xx = np.mgrid[:512, :512]
    r2 = ((xx - 255.5) ** 2 + (yy - 255.5) ** 2) / 256 ** 2
    vignetted = (std * (1 - 0.6 * r2)[..., None]).clip(0, 255).astype(np.uint8)
    assert quality_features(vignetted, mask)["q_centre_rim"] > quality_features(std, mask)["q_centre_rim"] + 5


def test_colour_stats_of_a_flat_colour():
    img = np.full((64, 64, 3), (200, 100, 50), np.uint8)
    mask = np.ones((64, 64), np.uint8)
    s = colour_stats(img, mask)
    L = cv2.cvtColor(img[:1, :1], cv2.COLOR_RGB2LAB)[0, 0]
    assert s["lab_n"] == 64 * 64 and abs(s["lab_L_mean"] - L[0]) < 1e-6 and s["lab_L_std"] == 0


def test_probe_features_do_not_depend_on_image_size():
    # Image size identifies a camera trivially; the probe must not see it.
    small, big = eye(shape=(512, 512)), cv2.resize(eye(shape=(512, 512)), (2048, 2048), interpolation=cv2.INTER_CUBIC)
    a, b = probe_features(small, "x_"), probe_features(big, "x_")
    for k in a:
        if k != "x_sharpness":  # sharpness after downscaling differs a little by design
            assert abs(a[k] - b[k]) < 2.0, k


def test_process_one_writes_a_png_that_reads_back(tmp_path):
    src = tmp_path / "a.png"
    cv2.imwrite(str(src), cv2.cvtColor(eye(shape=(600, 800)), cv2.COLOR_RGB2BGR))
    out = tmp_path / "processed" / "x" / "a.png"
    row = process_one(("x_a", str(src), str(out), "x"))
    assert row["fov_ok"] and os.path.exists(out)
    rgb, mask = load_processed(str(out))
    assert rgb.shape == (512, 512, 3) and set(np.unique(mask)) <= {0, 1}
    assert abs(mask.mean() - np.pi / 4) < 0.02  # a full circle
    assert (rgb[mask == 0] == 0).all()  # nothing outside the retina


def test_process_one_reports_a_failure_instead_of_crashing(tmp_path):
    src = tmp_path / "black.png"
    cv2.imwrite(str(src), np.zeros((300, 400, 3), np.uint8))
    row = process_one(("x_b", str(src), str(tmp_path / "o.png"), "x"))
    assert not row["fov_ok"] and row["fail_reason"] == "fov_fail"
