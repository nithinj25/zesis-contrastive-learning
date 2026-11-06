"""Unit tests for section 6 steps 3, 4 and 6, on synthetic retinas with known answers."""
import cv2
import numpy as np
import pytest

from data.preprocess import correct_illumination, find_fov, standardise_geometry


def retina(h=1000, w=1400, cx=700, cy=500, r=450, colour=(170, 90, 40)):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (cx, cy), r, colour, -1)
    return img


def disc_mask(size=512):
    yy, xx = np.ogrid[:size, :size]
    return (((xx - 255.5) ** 2 + (yy - 255.5) ** 2) <= 256**2).astype(np.uint8)


def lightness(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]


def test_find_fov_recovers_known_circle():
    fov = find_fov(retina())
    assert fov.cx == pytest.approx(700, abs=2)
    assert fov.cy == pytest.approx(500, abs=2)
    assert fov.radius == pytest.approx(450, abs=3)


def test_find_fov_flat_topped_photo():
    img = retina()
    img[:150] = 0  # the top of the circle is cut off, as on some cameras
    img[850:] = 0
    fov = find_fov(img)
    assert fov.cx == pytest.approx(700, abs=2)
    assert fov.radius == pytest.approx(450, abs=3)


def test_find_fov_ignores_dim_non_black_corners():
    # Real RFMiD 2.0 photos: square frame, retina nearly filling it, corners dim but not black.
    img = np.full((512, 512, 3), (35, 40, 70), np.uint8)
    cv2.circle(img, (256, 256), 250, (170, 90, 40), -1)
    fov = find_fov(img)
    assert fov.radius == pytest.approx(250, abs=4)  # not 362, the circle around the whole square
    assert fov.mask[3, 3] == 0
    _, mask = standardise_geometry(img, fov)
    cols = np.where(mask[256] > 0)[0]
    assert cols.max() - cols.min() + 1 == pytest.approx(512, abs=4)


@pytest.mark.parametrize(
    "img",
    [np.zeros((500, 500, 3), np.uint8), retina(h=1000, w=1000, cx=500, cy=500, r=100)],
    ids=["all_black", "tiny_circle"],
)
def test_find_fov_failures_return_none(img):
    assert find_fov(img) is None


def test_find_fov_never_cuts_dark_retina():
    # Regression: a brightness-only mask blacked out dark areas reaching the rim (real macula on RFMiD 2.0).
    img = np.full((512, 512, 3), (35, 40, 70), np.uint8)
    cv2.circle(img, (256, 256), 250, (170, 90, 40), -1)
    cv2.ellipse(img, (256, 256), (250, 250), 0, -40, 40, (30, 35, 60), -1)  # dark sector out to the rim
    fov = find_fov(img)
    assert fov.radius == pytest.approx(250, abs=4)
    assert fov.cx == pytest.approx(256, abs=4)
    assert fov.mask[256, 450] == 1  # inside the dark sector


def test_find_fov_keeps_dim_rim_when_not_flat_cut():
    img = retina()
    yy, xx = np.ogrid[:1000, :1400]
    r = np.sqrt((xx - 700.0) ** 2 + (yy - 500.0) ** 2) / 450
    img = (img * np.clip(1.25 - r**4, 0.25, 1)[..., None]).astype(np.uint8)  # fades towards the rim
    fov = find_fov(img)
    assert fov.radius == pytest.approx(450, abs=8)
    assert fov.mask[int(500 - 0.97 * 450), 700] == 1


def test_stretched_photo_is_made_round_again():
    # Real RFMiD 2.0: 61% of training photos are 4:3 originals squeezed into 512x512, so discs are tall ovals.
    img = retina()
    cv2.circle(img, (950, 500), 60, (250, 230, 200), -1)  # round disc in the original
    img[:80] = 0
    img[920:] = 0
    stretched = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    fov = find_fov(stretched)
    assert fov.ry / fov.rx == pytest.approx(1400 / 1000, abs=0.05)
    out, _ = standardise_geometry(stretched, fov)
    rows, cols = np.where(lightness(out) > 200)
    assert (rows.max() - rows.min()) / (cols.max() - cols.min()) == pytest.approx(1.0, abs=0.08)


def test_round_photo_stays_round():
    fov = find_fov(retina())
    assert fov.rx == fov.ry


def test_geometry_gives_same_size_for_different_cameras():
    for img in (retina(), retina(h=1424, w=2144, cx=1072, cy=712, r=700), retina(h=600, w=800, cx=400, cy=300, r=280)):
        out, mask = standardise_geometry(img, find_fov(img))
        assert out.shape == (512, 512, 3)
        cols = np.where(mask.any(0))[0]
        assert cols.max() - cols.min() + 1 == pytest.approx(512, abs=4)


def test_geometry_flat_top_is_not_called_retina():
    img = retina()
    img[:150] = 0
    out, mask = standardise_geometry(img, find_fov(img))
    assert mask[5, 256] == 0  # padded or cut area
    assert mask[256, 256] == 1
    assert out[~mask.astype(bool)].max() == 0


def test_geometry_handles_circle_past_image_edge():
    img = retina(h=800, w=1400, cx=700, cy=400, r=500)  # circle taller than the image
    out, mask = standardise_geometry(img, find_fov(img))
    assert out.shape == (512, 512, 3)
    assert mask[256, 256] == 1 and mask[2, 256] == 0


def test_geometry_does_not_flip():
    img = retina()
    cv2.circle(img, (1000, 500), 40, (250, 250, 250), -1)  # marker on the right
    out, _ = standardise_geometry(img, find_fov(img))
    # 300 px right of centre at scale 512/900 lands at x = 256 + 171 = 427; its mirror is x = 85
    assert lightness(out)[256, 427] > lightness(out)[256, 85] + 20


def test_illumination_formula_dry_run():
    # L = 100, local brightness 80, retina mean 100 -> L_flat = 125 -> 0.4*100 + 0.6*125 = 115
    L, bg, mean, strength = 100.0, 80.0, 100.0, 0.6
    L_flat = L / bg * mean
    assert L_flat == pytest.approx(125)
    assert (1 - strength) * L + strength * L_flat == pytest.approx(115)


def vignetted():
    mask = disc_mask()
    yy, xx = np.mgrid[:512, :512].astype(np.float32)
    r2 = ((xx - 255.5) ** 2 + (yy - 255.5) ** 2) / 256**2
    base = np.full((512, 512, 3), (170, 90, 40), np.float32)
    img = (base * (1 - 0.4 * r2)[..., None]).clip(0, 255).astype(np.uint8)
    return img * mask[..., None], mask, r2


def test_illumination_flattens_vignetting():
    img, mask, r2 = vignetted()
    out = correct_illumination(img, mask)
    centre, rim = (r2 < 0.05) & (mask > 0), (r2 > 0.7) & (r2 < 0.9) & (mask > 0)
    before = lightness(img)[centre].mean() - lightness(img)[rim].mean()
    after = lightness(out)[centre].mean() - lightness(out)[rim].mean()
    # Sigma 64 deliberately removes less vignetting than 32, to protect the disc (section 20).
    assert abs(after) < 0.65 * abs(before)


def test_illumination_keeps_disc_and_lesion_contrast():
    mask = disc_mask()
    img = np.full((512, 512, 3), (170, 90, 40), np.uint8)
    cv2.circle(img, (380, 256), 40, (235, 190, 130), -1, cv2.LINE_AA)  # optic disc, 80 px wide
    cv2.circle(img, (150, 180), 3, (110, 30, 20), -1, cv2.LINE_AA)  # small dark lesion
    img = img * mask[..., None]

    def contrast(im, cx, cy, r):
        Y, X = np.ogrid[:512, :512]
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        inside = d2 <= (0.7 * r) ** 2
        ring = (d2 >= (1.5 * r) ** 2) & (d2 <= (2.2 * r) ** 2) & (mask > 0)
        return lightness(im)[inside].mean() - lightness(im)[ring].mean()

    out = correct_illumination(img, mask)
    assert contrast(out, 380, 256, 40) >= 0.85 * contrast(img, 380, 256, 40)
    assert abs(contrast(out, 150, 180, 3)) >= 0.95 * abs(contrast(img, 150, 180, 3))


def test_illumination_leaves_colour_channels_alone():
    img, mask, _ = vignetted()
    out = correct_illumination(img, mask)
    a = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1:]
    b = cv2.cvtColor(out, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 1:]
    assert np.abs(a - b)[mask > 0].mean() < 2.5


def test_illumination_strength_zero_is_identity_and_border_stays_black():
    img, mask, _ = vignetted()
    out = correct_illumination(img, mask, strength=0.0)
    assert np.abs(out.astype(int) - img.astype(int))[mask > 0].mean() < 2
    assert correct_illumination(img, mask)[mask == 0].max() == 0


def test_camera_shift_matches_the_documented_dry_run():
    # docs/DATA_PIPELINE.md step 7: camera mean 60, std 20; pool 70, 25; alpha 0.7.
    from data.preprocess import shift_channel

    assert abs(shift_channel(80.0, 60, 20, 70, 25) - 90.5) < 1e-9
    assert abs(shift_channel(60.0, 60, 20, 70, 25) - 67.0) < 1e-9
    # the pale disc stays paler than its neighbour: the gap grows by 0.3 + 0.7 * 25 / 20 = 1.175
    assert abs((shift_channel(80.0, 60, 20, 70, 25) - shift_channel(60.0, 60, 20, 70, 25)) - 20 * 1.175) < 1e-9


def test_camera_color_with_pool_equal_to_camera_changes_nothing():
    from data.preprocess import camera_color

    rng = np.random.default_rng(0)
    img = rng.integers(40, 200, (64, 64, 3)).astype(np.uint8)
    mask = np.ones((64, 64), np.uint8)
    out = camera_color(img, mask, [100, 130, 140], [20, 5, 5], [100, 130, 140], [20, 5, 5])
    round_trip = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2LAB), cv2.COLOR_LAB2RGB)
    assert np.abs(out.astype(int) - round_trip.astype(int)).max() <= 1  # only 8-bit LAB rounding remains


def test_clahe_raises_local_contrast_and_keeps_outside_black():
    from data.preprocess import clahe_l

    img = np.zeros((128, 128, 3), np.uint8)
    img[:, :] = (120, 60, 30)
    img[60:68, 60:68] = (110, 55, 28)  # a faint dark spot
    mask = np.zeros((128, 128), np.uint8)
    mask[8:120, 8:120] = 1
    img = img * mask[..., None]
    out = clahe_l(img, mask)
    contrast = lambda a: abs(int(a[64, 64].astype(int).sum()) - int(a[40, 40].astype(int).sum()))  # noqa: E731
    assert contrast(out) >= contrast(img)
    assert (out[mask == 0] == 0).all()


def test_pooled_stats_equal_stats_of_all_pixels():
    from data.preprocess import pooled_stats

    rng = np.random.default_rng(0)
    a, b = rng.normal(50, 10, (1000, 3)), rng.normal(80, 5, (3000, 3))
    m, s = pooled_stats(np.stack([a.mean(0), b.mean(0)]), np.stack([a.std(0), b.std(0)]), np.array([1000, 3000]))
    allpx = np.vstack([a, b])
    assert np.allclose(m, allpx.mean(0)) and np.allclose(s, allpx.std(0))
