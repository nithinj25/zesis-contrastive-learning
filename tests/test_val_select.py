"""Unit tests for val-A model selection (eval/val_select.py) and its bookkeeping in encoder/train.py."""
import json
import os

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from encoder.export import build_backbone
from encoder.train import record_val
from eval.val_select import METRIC, ValSelector, build_cache, label_matrix, load_backbone, score_features


def test_label_matrix():
    Y = label_matrix([["a"], [], ["a", "c"]], ["a", "b", "c"])
    assert Y.tolist() == [[1, 0, 0], [0, 0, 0], [1, 0, 1]]


def test_score_features_signal_and_exclusion():
    rng = np.random.default_rng(0)
    vocab = ["good", "noise", "rare"]
    ntr, nva = 300, 120
    Ytr = (rng.random((ntr, 3)) < 0.3).astype(float)
    Yva = (rng.random((nva, 3)) < 0.3).astype(float)
    Yva[:, 2] = 0
    Yva[:3, 2] = 1  # only 3 val-A positives: below MIN_VAL_POS, so "rare" is not scored

    def feats(Y):
        X = rng.normal(size=(len(Y), 16))
        X[:, 0] += 4 * Y[:, 0]  # "good" is visible in dimension 0; "noise" is visible nowhere
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    res = score_features(feats(Ytr), Ytr, feats(Yva), Yva, np.arange(ntr), vocab)
    assert res["val_n_labels"] == 2 and "rare" not in res["val_per_label"]
    assert res["val_per_label"]["good"] > 0.95
    assert abs(res["val_per_label"]["noise"] - 0.5) < 0.15
    assert res[METRIC] == pytest.approx(np.mean(list(res["val_per_label"].values())))


def test_features_keep_bn_and_mode():
    net = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    net.train()
    sel = object.__new__(ValSelector)  # skip the photo cache; give it a tiny array instead
    sel.x = np.random.default_rng(0).integers(0, 256, (10, 3, 8, 8), dtype=np.uint8)
    sel.mean, sel.std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    before = net[1].running_mean.clone()
    f = sel.features(net, device="cpu", batch=4)
    assert f.shape == (10, 4)
    assert np.allclose(np.linalg.norm(f, axis=1), 1, atol=1e-5)
    assert torch.equal(net[1].running_mean, before)  # eval mode: BatchNorm statistics untouched
    assert net.training  # mode restored for training


def test_record_val_best(tmp_path):
    p = str(tmp_path / "val.csv")
    r = lambda v: {METRIC: v, "val_knn_macro_auroc": 0.5, "val_C": 0.1, "val_n_labels": 10, "val_seconds": 1.0}
    assert record_val(p, 0, 0, r(0.99), METRIC) is False  # the start never counts as best
    assert record_val(p, 5, 100, r(0.80), METRIC) is True
    assert record_val(p, 10, 200, r(0.85), METRIC) is True
    assert record_val(p, 15, 300, r(0.82), METRIC) is False
    assert record_val(p, 20, 400, r(0.85), METRIC) is False  # a tie keeps the earlier epoch
    assert len(pd.read_csv(p)) == 5


def test_load_backbone_from_checkpoint(tmp_path):
    net = build_backbone("random")
    with torch.no_grad():
        net.conv1.weight.fill_(0.123)
    ck = {"model": {f"q_enc.backbone.{k}": v for k, v in net.state_dict().items()} | {"q_enc.head.0.weight": torch.zeros(1)}}
    torch.save(ck, tmp_path / "last.pt")
    torch.save(net.state_dict(), tmp_path / "encoder.pt")
    for name in ("last.pt", "encoder.pt"):
        got = load_backbone(str(tmp_path / name))
        assert torch.allclose(got.conv1.weight, torch.full_like(got.conv1.weight, 0.123))


@pytest.mark.skipif(not os.path.exists("data/trace_preprocess.json"), reason="needs the stored preprocessing stats")
def test_cache_round_trip(tmp_path):
    rgba = np.zeros((64, 64, 4), np.uint8)
    cv2.circle(rgba, (32, 32), 30, (40, 80, 160, 255), -1)  # an orange-ish disc, alpha = mask (BGRA on disk)
    os.makedirs(tmp_path / "proc")
    cv2.imwrite(str(tmp_path / "proc" / "a.png"), rgba)
    rows = pd.DataFrame({"image_id": ["a"], "processed_path": ["a.png"], "source": [None]})
    stats = json.load(open("data/trace_preprocess.json"))
    arr = build_cache(rows, stats, 32, str(tmp_path / "cache" / "c.npy"), root=str(tmp_path / "proc"))
    assert arr.shape == (1, 3, 32, 32) and arr.dtype == np.uint8
    assert arr[0, :, 16, 16].max() > 0 and arr[0, :, 0, 0].max() == 0  # retina kept, outside the mask black
    assert pd.read_parquet(str(tmp_path / "cache" / "c_index.parquet")).image_id.tolist() == ["a"]
