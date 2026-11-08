"""Unit tests for the training views."""
import cv2
import numpy as np

from augment.views import color_jitter, plain_view_np, prepare, random_resized_crop


def photo():
    rng = np.random.default_rng(0)
    img = rng.integers(30, 220, (512, 512, 3)).astype(np.uint8)
    yy, xx = np.ogrid[:512, :512]
    mask = (((xx - 255.5) ** 2 + (yy - 255.5) ** 2) <= 256**2).astype(np.uint8)
    return img * mask[..., None], mask


def test_view_shape_range_and_randomness():
    rgb, mask = photo()
    rng = np.random.default_rng(0)
    img = prepare(rgb, mask)
    a, b = plain_view_np(img, rng), plain_view_np(img, rng)
    assert a.shape == (3, 224, 224) and a.dtype == np.uint8
    assert not np.array_equal(a, b)  # two views of one photo differ


def test_same_seed_same_view():
    rgb, mask = photo()
    img = prepare(rgb, mask)
    a = plain_view_np(img, np.random.default_rng(5))
    b = plain_view_np(img, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_crop_scale_respected():
    img = np.zeros((512, 512, 3), np.uint8)
    img[:256] = 255  # top half white
    rng = np.random.default_rng(0)
    whites = [random_resized_crop(img, rng, 64, scale=(1.0, 1.0), ratio=(1.0, 1.0)).mean() for _ in range(3)]
    assert all(abs(w - 127.5) < 3 for w in whites)  # full-image crops keep the half/half split


def test_jitter_keeps_range_and_changes_colour():
    rgb, _ = photo()
    out = color_jitter(rgb, np.random.default_rng(1))
    assert out.dtype == np.uint8 and out.shape == rgb.shape and not np.array_equal(out, rgb)


def test_views_module_does_not_import_torch():
    # Loader workers import it; torch there cost ~1.7 GB of commit per worker on Windows.
    import subprocess
    import sys

    code = "import sys, augment.views, augment.loader, data.process; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_loader_batches_are_reproducible(tmp_path):
    import cv2

    from augment.loader import ViewLoader

    rgb, mask = photo()
    for i in range(6):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.dstack([rgb[..., ::-1], mask * 255]))
    paths = [f"{i}.png" for i in range(6)]
    a = ViewLoader(paths, batch=2, workers=2, seed=3, size=32, root=str(tmp_path))
    first = list(a.epoch(0))
    again = list(a.epoch(0))
    other = list(a.epoch(1))
    resumed = list(a.epoch(0, start=1))
    a.close()
    assert len(first) == 3 and first[0][0].shape == (2, 3, 32, 32) and first[0][0].dtype == np.uint8
    assert first[0][2].shape == (0, 2, 3, 96, 96)  # plain policy: no local views
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(first, again))  # same epoch, same views
    assert sorted(np.concatenate([b[3] for b in first]).tolist()) == list(range(6))  # each photo once
    assert not all(np.array_equal(x[3], y[3]) for x, y in zip(first, other))  # a new order next epoch
    assert len(resumed) == 2 and all(np.array_equal(x[0], y[0]) for x, y in zip(first[1:], resumed))  # mid-epoch resume


# ---------------------------------------------------------------- retina-safe views
from augment.views import crop_rotate, retina_geometry, retina_view_np, sample_crop  # noqa: E402

OFF = dict(colour_p=0.0, jitter_p=0.0, blur_p=0.0)  # switch the random colour edits off


def test_crop_area_and_centre_rules():
    rng = np.random.default_rng(0)
    cx, cy, R = retina_geometry(np.zeros((512, 512, 3)))
    assert (cx, cy, R) == (255.5, 255.5, 256.0)
    for _ in range(500):
        x, y, side = sample_crop(rng, cx, cy, R, (0.6, 1.0))
        assert 0.6 - 1e-9 <= side**2 / (np.pi * R * R) <= 1.0 + 1e-9  # 60-100% of the retina's area
        assert np.hypot(x - cx, y - cy) <= R - side / 2 + 1e-9  # centre kept near the middle
    # dry run from the explanation: 60% -> side 351.5, centre may move 80.3 px; 100% -> 453.7 and 29.1 px
    side60 = np.sqrt(0.6 * np.pi * 256**2)
    assert abs(side60 - 351.5) < 0.1 and abs((256 - side60 / 2) - 80.3) < 0.1


def test_global_views_stay_mostly_retina():
    rgb, mask = photo()
    rng = np.random.default_rng(1)
    fracs = []
    for _ in range(100):
        x, y, side = sample_crop(rng, 255.5, 255.5, 256, (0.6, 1.0))
        _, m, _ = crop_rotate(rgb, mask, x, y, side, rng.uniform(-15, 15), 224, False)
        fracs.append(m.mean())
    assert min(fracs) > 0.75, min(fracs)


def test_retina_centre_is_mapped_through_rotation_and_flip():
    img = np.zeros((512, 512, 3), np.uint8)
    yy, xx = np.ogrid[:512, :512]
    mask = (((xx - 255.5) ** 2 + (yy - 255.5) ** 2) <= 256**2).astype(np.uint8)
    img[(xx - 255.5) ** 2 + (yy - 255.5) ** 2 <= 4] = 255  # a bright dot at the retina centre
    img[300:310, 200:210] = 90  # an off-centre patch, so a wrong rotation or flip would show
    seen = {}

    def hook(v, m, geom, rng, colour_from=None):
        seen["geom"] = geom
        return v

    for seed in range(20):
        rng = np.random.default_rng(seed)
        view = retina_view_np(img, mask, rng, 224, artifacts=hook, clahe_clip=0, **OFF)
        g = view[1].astype(float)
        yx = np.unravel_index(np.argmax(g), g.shape)
        assert np.hypot(yx[1] - seen["geom"][0], yx[0] - seen["geom"][1]) < 2.0, (seed, yx, seen["geom"])
        assert seen["geom"][2] > 112  # the retina radius in view pixels is bigger than half the view


def test_retina_views_never_grey_and_black_outside():
    rgb, mask = photo()
    rng = np.random.default_rng(2)
    for _ in range(30):
        v = retina_view_np(rgb, mask, rng, 96)
        assert v.shape == (3, 96, 96) and v.dtype == np.uint8
        assert np.abs(v[0].astype(int) - v[1].astype(int)).max() > 0  # colour survives: no greyscale
    edge = np.zeros((512, 512, 3), np.uint8)
    edge[mask > 0] = (200, 80, 40)
    v = retina_view_np(edge, mask, np.random.default_rng(3), 224, scale=(1.0, 1.0))
    retina = v.max(0) > 0
    assert 0.6 < retina.mean() < 1.0  # a full-size crop shows black corners, and they stay black (0)


def test_camera_colour_step_moves_towards_pool():
    img = np.full((512, 512, 3), (180, 90, 60), np.uint8)
    yy, xx = np.ogrid[:512, :512]
    mask = (((xx - 255.5) ** 2 + (yy - 255.5) ** 2) <= 256**2).astype(np.uint8)
    img = img * mask[..., None]
    lab = cv2.cvtColor(img[256:257, 256:257], cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    cam = {"mean": lab.tolist(), "std": [10.0, 5.0, 5.0]}
    pool = {"mean": [lab[0] - 30, lab[1], lab[2]], "std": [10.0, 5.0, 5.0]}  # the pool is darker
    no = retina_view_np(img, mask, np.random.default_rng(4), 64, camera=cam, pool=pool, clahe_clip=0, **OFF)
    yes = retina_view_np(img, mask, np.random.default_rng(4), 64, camera=cam, pool=pool, clahe_clip=0,
                         **{**OFF, "colour_p": 1.0})
    L = lambda v: cv2.cvtColor(np.ascontiguousarray(v.transpose(1, 2, 0)), cv2.COLOR_RGB2LAB)[32, 32, 0]
    assert abs((float(L(no)) - float(L(yes))) - 0.7 * 30) < 3  # alpha 0.7 of the 30-level gap


def test_retina_view_reproducible():
    rgb, mask = photo()
    a = retina_view_np(rgb, mask, np.random.default_rng(9), 224)
    b = retina_view_np(rgb, mask, np.random.default_rng(9), 224)
    c = retina_view_np(rgb, mask, np.random.default_rng(10), 224)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_masked_jitter_keeps_outside_black():
    rgb, mask = photo()
    out = color_jitter(rgb, np.random.default_rng(1), mask=mask)
    assert out[mask == 0].max() == 0


def test_local_views_are_small_and_mostly_retina():
    # local scale 0.15-0.35 of the retina's area: side sqrt(0.15 pi 256^2) = 175.7 to sqrt(0.35 pi 256^2) = 268.4 px
    rgb, mask = photo()
    rng = np.random.default_rng(0)
    cx, cy, R = retina_geometry(rgb)
    sides, share = [], []
    for _ in range(500):
        x, y, side = sample_crop(rng, cx, cy, R, (0.15, 0.35))
        sides.append(side)
        share.append(crop_rotate(rgb, mask, x, y, side, rng.uniform(-15, 15), 96, False)[1].mean())
    assert 175.7 - 0.1 <= min(sides) and max(sides) <= 268.4 + 0.1
    assert min(share) > 0.9  # measured minimum 0.95 over 2,000 draws


def test_loader_multicrop_batches(tmp_path):
    import cv2

    from augment.loader import ViewLoader

    rgb, mask = photo()
    for i in range(4):
        cv2.imwrite(str(tmp_path / f"{i}.png"), np.dstack([rgb[..., ::-1], mask * 255]))
    paths = [f"{i}.png" for i in range(4)]
    kw = dict(batch=2, workers=1, seed=3, size=64, policy="retina", root=str(tmp_path))
    multi = ViewLoader(paths, n_local=4, local_size=32, **kw)
    two = ViewLoader(paths, **kw)
    m, t = list(multi.epoch(0)), list(two.epoch(0))
    again = list(multi.epoch(0))
    multi.close(), two.close()
    v1, v2, loc, idx = m[0]
    assert v1.shape == (2, 3, 64, 64) and loc.shape == (4, 2, 3, 32, 32) and loc.dtype == np.uint8
    assert all(np.array_equal(a[2], b[2]) for a, b in zip(m, again))  # reproducible locals
    assert t[0][2].shape[0] == 0
    # photo 0's globals come first from the generator, so they are identical with or without locals
    assert np.array_equal(m[0][0][0], t[0][0][0]) and np.array_equal(m[0][1][0], t[0][1][0])
    assert not np.array_equal(loc[0, 0], loc[1, 0])  # four different local crops
