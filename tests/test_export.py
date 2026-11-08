"""Unit tests for the encoder interface (section 19.1)."""
import json

import cv2
import numpy as np
import torch

from encoder.export import TraceEncoder, build_backbone, finish_view, load_trace_encoder, preprocess

STATS = {"pool": {"mean": [100.0, 135.0, 150.0], "std": [30.0, 6.0, 8.0]},
         "cameras": {"cam_a": {"mean": [80.0, 140.0, 160.0], "std": [20.0, 5.0, 7.0]}},
         "params": {"colour_alpha": 0.7, "clahe_clip": 1.5, "illumination_strength": 0.6}}


def photo(h=600, w=800):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w // 2, h // 2), min(h, w) // 2 - 5, (170, 90, 40), -1)
    cv2.circle(img, (w // 2 + 100, h // 2), 40, (235, 190, 130), -1)
    return img


def test_features_are_unit_length_2048_and_size_agnostic():
    enc = TraceEncoder(build_backbone("random")).eval()
    with torch.no_grad():
        for size in (224, 384):
            f = enc(torch.rand(2, 3, size, size))
            assert f.shape == (2, 2048) and torch.allclose(f.norm(dim=1), torch.ones(2), atol=1e-5)


def test_saved_backbone_loads_back_identically(tmp_path):
    torch.manual_seed(0)
    a = TraceEncoder(build_backbone("random")).eval()
    torch.save(a.backbone.state_dict(), tmp_path / "encoder.pt")
    b = load_trace_encoder(str(tmp_path / "encoder.pt"))
    x = torch.rand(1, 3, 128, 128)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)


def test_step7_only_for_a_known_camera():
    rgb = np.full((512, 512, 3), (150, 80, 40), np.uint8)
    mask = np.ones((512, 512), np.uint8)
    known = finish_view(rgb, mask, "cam_a", STATS, size=64)
    unknown = finish_view(rgb, mask, "hidden_site", STATS, size=64)
    assert known.shape == (3, 64, 64) and 0 <= known.min() and known.max() <= 1
    assert not torch.allclose(known, unknown)  # the known camera was colour-shifted


def test_preprocess_raw_photo_to_tensor(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(STATS))
    x = preprocess(photo(), source_id=None, stats_path=str(path))
    assert x.shape == (3, 384, 384) and x.dtype == torch.float32 and 0 <= x.min() and x.max() <= 1
    assert x[:, 0, 0].sum() == 0  # the corner is outside the retina


def test_preprocess_rejects_a_photo_without_retina(tmp_path):
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(STATS))
    try:
        preprocess(np.zeros((300, 400, 3), np.uint8), stats_path=str(path))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
