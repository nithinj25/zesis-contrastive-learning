"""Disc / fovea localiser (findings/localise.py) and landmark mapping (findings/landmarks.py) on known answers."""
import cv2
import numpy as np
import pytest

from findings.landmarks import map_point
from findings.localise import DD, crop, disc_and_macula_crops, find_disc, localise


def synthetic_eye(disc=(110, 250), fovea=(305, 262), glare_rim=False):
    """512 px retina: flat orange background, a bright disc blob one DD wide, a darker fovea spot.
    Optional bright glare crescent on the rim (what fooled the plain-brightness finder)."""
    yy, xx = np.mgrid[:512, :512]
    mask = (((xx - 255.5) ** 2 + (yy - 255.5) ** 2) <= 256 ** 2).astype(np.uint8)
    img = np.zeros((512, 512, 3), np.float32) + (170, 90, 40)
    blob = np.exp(-((xx - disc[0]) ** 2 + (yy - disc[1]) ** 2) / (2 * (DD / 3) ** 2))
    img += blob[..., None] * (80, 110, 90)
    spot = np.exp(-((xx - fovea[0]) ** 2 + (yy - fovea[1]) ** 2) / (2 * (DD / 3) ** 2))
    img -= spot[..., None] * (50, 35, 15)
    if glare_rim:
        rim = (np.hypot(xx - 255.5, yy - 255.5) > 236) & (yy > 380)
        img[rim] += (80, 110, 90)
    img = np.clip(img, 0, 255).astype(np.uint8) * mask[..., None]
    return img, mask


def test_finds_disc_and_fovea_on_a_synthetic_eye():
    img, mask = synthetic_eye()
    lm = localise(img, mask)
    assert lm.ok
    assert np.hypot(lm.disc_x - 110, lm.disc_y - 250) < 0.1 * DD
    assert np.hypot(lm.fovea_x - 305, lm.fovea_y - 262) < 0.1 * DD


def test_rim_glare_does_not_win_over_the_disc():
    img, mask = synthetic_eye(glare_rim=True)
    x, y = find_disc(img, mask)
    assert np.hypot(x - 110, y - 250) < 0.1 * DD


def test_fovea_on_the_photo_centre_side_for_a_right_hand_disc():
    img, mask = synthetic_eye(disc=(400, 250), fovea=(205, 262))
    lm = localise(img, mask)
    assert np.hypot(lm.fovea_x - 205, lm.fovea_y - 262) < 0.1 * DD


def test_crop_is_centred_and_padded_past_the_edge():
    img = np.zeros((512, 512, 3), np.uint8)
    img[240:272, 240:272] = 255  # a white square in the middle
    c = crop(img, (256, 256), 64, out=64)
    assert c.shape == (64, 64, 3) and c[16:48, 16:48].min() == 255 and c[:12].max() == 0
    edge = crop(img, (0, 0), 64, out=64)  # three quarters of this window lie outside the photo
    assert edge.shape == (64, 64, 3) and edge.max() == 0


def test_crop_sizes_follow_the_disc_diameter():
    img, mask = synthetic_eye()
    d, m = disc_and_macula_crops(img, localise(img, mask), out=96)
    assert d.shape == m.shape == (96, 96, 3)
    assert d[48, 48].mean() > m[48, 48].mean()  # disc crop centred on bright, macula crop on dark


def test_map_point_matches_step4_window():
    # ellipse centre (1000, 700), radii 500 x 400 -> window x 500..1500, y 300..1100 -> 512 x 512
    x, y = map_point(1000, 700, 1000, 700, 500, 400)
    assert (x, y) == pytest.approx((256, 256))
    x, y = map_point(500, 1100, 1000, 700, 500, 400)
    assert (x, y) == pytest.approx((0, 512))


def test_crops_for_photo_shapes_and_failure_rows():
    import json

    from findings.crops import crops_for_photo

    stats = json.load(open("data/trace_preprocess.json"))
    img, mask = synthetic_eye()
    d, m, lm = crops_for_photo(img, mask, None, stats)
    assert lm.ok and d.shape == m.shape == (3, 384, 384) and 0 <= float(d.min()) and float(d.max()) <= 1
    assert d[:, 192, 192].mean() > m[:, 192, 192].mean()  # disc crop centred on bright, macula crop on dark
    d0, m0, lm0 = crops_for_photo(img, np.zeros_like(mask), None, stats)  # no retina: nothing to find
    assert not lm0.ok and float(d0.abs().sum()) == 0 and float(m0.abs().sum()) == 0
