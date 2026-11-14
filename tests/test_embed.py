"""encoder/embed.py: the one-command folder embedder (on real photos it matches the cached evaluation features,
cosine 1.00000; that check needs raw data and is recorded in CLAUDE.md, not repeated here)."""
import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from encoder.embed import list_images, main
from encoder.export import build_backbone, preprocess

STATS = {"params": {"illumination_strength": 0.6, "clahe_clip": 1.5, "colour_alpha": 0.7},
         "pool": {"mean": [120.0, 135.0, 150.0], "std": [40.0, 8.0, 10.0]},
         "cameras": {"cam_a": {"mean": [100.0, 140.0, 155.0], "std": [35.0, 9.0, 11.0]}}}


def fundus(seed=0):
    """A synthetic fundus-like photo: an orange disc of retina with a bright optic disc, black border."""
    rng = np.random.default_rng(seed)
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(img, (320, 240), 200, (170, 80, 40), -1)
    cv2.circle(img, (400, 230), 25, (240, 200, 150), -1)
    return np.clip(img.astype(int) + rng.integers(-10, 10, img.shape) * (img > 0), 0, 255).astype(np.uint8)


def write(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1], cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    buf.tofile(path)  # unicode-safe


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    d = tmp_path_factory.mktemp("embed")
    root = d / "photos"
    write(str(root / "b.png"), fundus(1))
    write(str(root / "sub" / "fotos_ñ" / "a.JPG"), fundus(2))
    write(str(root / "blank.png"), np.zeros((300, 400, 3), np.uint8))
    (root / "notes.txt").write_text("not a photo")
    torch.manual_seed(0)
    torch.save(build_backbone("random").state_dict(), d / "enc.pt")
    (d / "stats.json").write_text(json.dumps(STATS))
    return d


def test_list_images_is_recursive_sorted_and_filters_extensions(setup):
    assert list_images(str(setup / "photos")) == ["b.png", "blank.png", "sub/fotos_ñ/a.JPG"]


def test_folder_to_features_and_index(setup):
    d = setup
    out = main(["--images", str(d / "photos"), "--out", str(d / "out" / "f.npy"), "--weights", str(d / "enc.pt"),
                "--stats", str(d / "stats.json"), "--device", "cpu", "--workers", "0", "--batch", "1"])
    F = np.load(out["features"])
    idx = pd.read_csv(out["index"])
    assert F.shape == (3, 2048) and F.dtype == np.float32 and out["failed"] == 1
    assert list(idx.path) == ["b.png", "blank.png", "sub/fotos_ñ/a.JPG"] and list(idx.ok) == [True, False, True]
    assert "fov_fail" in idx.error[1] and not F[1].any()
    assert np.allclose(np.linalg.norm(F[[0, 2]], axis=1), 1, atol=1e-5)


def test_same_features_as_the_python_interface(setup):
    """The tool must give exactly what load_trace_encoder + preprocess give, with and without a camera."""
    from encoder.export import load_trace_encoder

    d = setup
    enc = load_trace_encoder(str(d / "enc.pt"))
    for cam in (None, "cam_a"):
        args = ["--images", str(d / "photos"), "--out", str(d / f"o_{cam}" / "f.npy"), "--weights", str(d / "enc.pt"),
                "--stats", str(d / "stats.json"), "--device", "cpu", "--workers", "0"]
        F = np.load(main(args + (["--camera", cam] if cam else []))["features"])
        with torch.inference_mode():
            ref = enc(preprocess(fundus(1), cam, str(d / "stats.json"))[None]).numpy()[0]
        assert np.allclose(F[0], ref, atol=1e-5)


def test_unknown_camera_is_refused(setup):
    d = setup
    with pytest.raises(SystemExit, match="unknown --camera"):
        main(["--images", str(d / "photos"), "--out", str(d / "x.npy"), "--weights", str(d / "enc.pt"),
              "--stats", str(d / "stats.json"), "--camera", "nikon"])
