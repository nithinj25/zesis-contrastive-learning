"""Unit tests for the camera-artifact simulator (augment/artifacts.py), with hand-worked numbers."""
import cv2
import numpy as np
import pytest

from augment.artifacts import (DEFAULT_P, Artifacts, colour_profile, glint, haze, jpeg, lashes, uneven, vignette)

S = 225
GEOM = (112.0, 112.0, 100.0)  # retina centre (112, 112), radius 100 px


def flat(v=128):
    img = np.full((S, S, 3), v, np.uint8)
    yy, xx = np.ogrid[:S, :S]
    mask = (((xx - 112) ** 2 + (yy - 112) ** 2) <= 100**2).astype(np.uint8)
    return img * mask[..., None], mask


def test_vignette_dry_run():
    # s = 0.3: centre keeps 100%, half radius 1 - 0.3 * 0.25 = 92.5%, rim 70%
    img = np.full((S, S, 3), 200, np.uint8)
    out = vignette(img, GEOM, 0.3)
    assert out[112, 112, 0] == pytest.approx(200)
    assert out[112, 162, 0] == pytest.approx(200 * 0.925)
    assert out[112, 212, 0] == pytest.approx(200 * 0.70)


def test_uneven_light_ramp():
    img = np.full((S, S, 3), 100, np.uint8)
    out = uneven(img, GEOM, 0.1, 0.0)  # brighter to the right
    assert out[112, 12, 0] == pytest.approx(90) and out[112, 212, 0] == pytest.approx(110)
    assert out[112, 112, 0] == pytest.approx(100) and out[12, 112, 0] == pytest.approx(100)  # unchanged across the ramp


def test_haze_dry_run():
    # flat 0.5, h = 0.4: (1 - 0.4) 0.5 + 0.4 (0.9 * 0.5 + 0.1) = 0.52 -> 132.6
    img = np.full((S, S, 3), 127.5, np.float32)
    assert haze(img, 0.4, 10)[112, 112, 0] == pytest.approx(0.52 * 255, abs=0.01)
    edge = np.zeros((S, S, 3), np.float32)
    edge[:, 112:] = 255
    before = edge[:, 140].mean() - edge[:, 84].mean()
    out = haze(edge, 0.4, 10)
    assert out[:, 140].mean() - out[:, 84].mean() < before  # contrast drops, like a cataract


def test_glint_peak_and_locality():
    img = np.full((S, S, 3), 100, np.uint8)
    out = glint(img, [(60.0, 60.0)], [10.0], 0.3, [1.0], [0.0])
    assert out[60, 60, 0] == pytest.approx(100 + 0.3 * 255)  # + intensity at the centre
    assert out[60, 70, 0] == pytest.approx(100 + 0.3 * 255 * np.exp(-0.5 * 4), rel=1e-4)  # one radius = 2 sigma away
    assert out[160, 160, 0] == pytest.approx(100, abs=1e-3)  # far away: untouched


def test_lashes_only_darken_the_rim():
    img = np.full((S, S, 3), 200, np.uint8)
    out = lashes(img, GEOM, [(0.0, 0.0, 0.9, 240, 300, 0.06)], 0.3, 2.0)  # an arc across the top
    assert out.max() <= 200 + 1e-3
    assert out[112 - 90, 112, 0] < 200 * 0.8  # dark on the arc (top rim)
    assert out[112, 112, 0] == pytest.approx(200) and out[112 + 90, 112, 0] == pytest.approx(200)  # centre and bottom clear


def test_colour_profile_reaches_camera_mean():
    pool = {"mean": [120.0, 135.0, 150.0], "std": [20.0, 5.0, 8.0]}
    cam = {"mean": [90.0, 140.0, 160.0], "std": [20.0, 5.0, 8.0]}
    lab = np.zeros((S, S, 3), np.uint8)
    lab[:] = (120, 135, 150)  # a photo sitting exactly at the pool's mean
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    mask = np.ones((S, S), np.uint8)
    full = cv2.cvtColor(colour_profile(img, mask, pool, cam, 1.0), cv2.COLOR_RGB2LAB)[112, 112].astype(float)
    half = cv2.cvtColor(colour_profile(img, mask, pool, cam, 0.5), cv2.COLOR_RGB2LAB)[112, 112].astype(float)
    start = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[112, 112].astype(float)
    assert np.abs(full - np.array(cam["mean"])).max() <= 2  # alpha 1: lands on the camera's mean
    assert np.abs(half - (start + np.array(cam["mean"])) / 2).max() <= 2  # alpha 0.5: half way


def test_jpeg_changes_little_at_quality_40():
    rng = np.random.default_rng(0)
    img = cv2.GaussianBlur(rng.integers(0, 256, (S, S, 3)).astype(np.uint8), (0, 0), 3)
    out = jpeg(img, 40)
    mse = ((out.astype(float) - img) ** 2).mean()
    assert 0 < mse and 10 * np.log10(255**2 / mse) > 28  # visible loss, but PSNR > 28 dB


def test_never_dust_and_identity_when_off():
    assert "dust" not in DEFAULT_P
    with pytest.raises(AssertionError):
        Artifacts(p={"dust": 0.1})
    img, mask = flat()
    out = Artifacts(p={})(img, mask, GEOM, np.random.default_rng(0))
    assert np.array_equal(out, img)


def test_hook_rates_mask_and_reproducibility():
    stats = {"pool": {"mean": [120.0, 135.0, 150.0], "std": [20.0, 5.0, 8.0]},
             "cameras": {"a": {"mean": [100.0, 140.0, 155.0], "std": [18.0, 6.0, 9.0]}}}
    art = Artifacts(stats)
    img, mask = flat()
    rng = np.random.default_rng(1)
    counts = {k: 0 for k in DEFAULT_P}
    n = 1500
    for _ in range(n):
        out = art(img, mask, GEOM, rng)
        for k in art.last:
            counts[k] += 1
        assert out[mask == 0].max() == 0 and out.dtype == np.uint8 and out.shape == img.shape
    for k, p in DEFAULT_P.items():
        assert abs(counts[k] / n - p) < 0.04, (k, counts[k] / n, p)  # each artifact fires at its own rate
    a = Artifacts(stats)(img, mask, GEOM, np.random.default_rng(7))
    b = Artifacts(stats)(img, mask, GEOM, np.random.default_rng(7))
    assert np.array_equal(a, b)


def test_artifacts_in_retina_view_and_loader_stays_torch_free():
    import subprocess
    import sys

    from augment.views import retina_view_np

    img, mask = flat()
    big = cv2.resize(img, (512, 512)), cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
    art = Artifacts(p={"vignette": 1.0})
    v = retina_view_np(*big, np.random.default_rng(0), 64, artifacts=art, colour_p=0, jitter_p=0, blur_p=0, clahe_clip=0)
    assert art.last == ["vignette"] and v.shape == (3, 64, 64)
    code = "import sys, augment.artifacts; print('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "False"


def test_step7_blend_is_exact():
    # the colour-profile artifact is told the view's colour after step 7: mean and std are (1 - a) cam + a pool
    from data.preprocess import shift_channel

    rng = np.random.default_rng(0)
    x = rng.normal(80, 12, 100000)
    a, cam_m, cam_s, pool_m, pool_s = 0.7, 80.0, 12.0, 120.0, 20.0
    y = shift_channel(x, cam_m, cam_s, pool_m, pool_s, a)
    assert y.mean() == pytest.approx((1 - a) * cam_m + a * pool_m, abs=0.2)
    assert y.std() == pytest.approx((1 - a) * cam_s + a * pool_s, abs=0.2)


def test_glints_stay_on_the_rim():
    art = Artifacts(p={"glint": 1.0}, strong=True)
    img = np.full((S, S, 3), 60, np.uint8)
    mask = np.ones((S, S), np.uint8)
    rng = np.random.default_rng(3)
    r, _, _ = __import__("augment.artifacts", fromlist=["_grid"])._grid(img.shape, GEOM)
    for _ in range(50):
        out = art(img, mask, GEOM, rng)
        bright = out[..., 0] > 60 + 0.5 * 0.5 * 255  # above half the glint's peak
        assert bright.any() and r[bright].min() > 0.55  # nothing bright near the disc / macula region
