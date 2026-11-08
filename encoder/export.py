"""The encoder interface the organisers run on hidden data (CLAUDE.md section 19.1).

    enc = load_trace_encoder("encoder.pt")          # or build_backbone("imagenet") for the baseline
    x = preprocess(photo_rgb_uint8, source_id=None)  # (3, 384, 384) in [0, 1]
    feats = enc(x[None])                             # (1, 2048), L2-normalised

The backbone is torchvision's ResNet-50 with fc = Identity. ImageNet mean/std normalisation happens INSIDE
the encoder, so callers pass plain [0, 1] RGB. Only the backbone state_dict is saved (encoder.pt).
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from data.preprocess import camera_color, clahe_l, correct_illumination, find_fov, standardise_geometry

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EVAL_SIZE = 384


def build_backbone(init: str = "imagenet") -> nn.Module:
    """ResNet-50 with the classifier removed: (B, 3, H, W) -> (B, 2048). init: 'imagenet' or 'random'."""
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if init == "imagenet" else None
    net = torchvision.models.resnet50(weights=weights)
    net.fc = nn.Identity()
    return net


class TraceEncoder(nn.Module):
    """Frozen TRACE backbone. Input: float (B, 3, H, W), RGB in [0, 1], already TRACE-preprocessed.
    Output: (B, 2048) L2-normalised features. Size-agnostic (global average pool); default 384 x 384."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone((x - self.mean) / self.std).float(), dim=1)


def load_trace_encoder(weights_path: str | None = None, device: str = "cpu", init: str = "imagenet") -> TraceEncoder:
    """weights_path: a backbone state_dict (encoder.pt). None gives the ImageNet (or random) baseline."""
    backbone = build_backbone(init if weights_path is None else "random")
    if weights_path is not None:
        backbone.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return TraceEncoder(backbone).eval().to(device)


def finish_view(rgb: np.ndarray, mask: np.ndarray, source_id: str | None, stats: dict, size: int = EVAL_SIZE) -> torch.Tensor:
    """Steps 7 (only for a known camera) and 8 on a processed 512 px photo, then resize: (3, size, size) in [0, 1].

    Evaluation policy (section 19.1): a camera with stored statistics gets step 7; an unknown camera skips it
    and relies on learned invariance. Per-image colour normalisation is never used (it erases pallor).
    """
    cams = stats.get("cameras", {})
    if source_id in cams:
        c, p = cams[source_id], stats["pool"]
        rgb = camera_color(rgb, mask, c["mean"], c["std"], p["mean"], p["std"], stats["params"]["colour_alpha"])
    rgb = clahe_l(rgb, mask, clip=stats["params"]["clahe_clip"])
    if rgb.shape[0] != size:
        import cv2

        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0


def preprocess(img_rgb_uint8: np.ndarray, source_id: str | None = None,
               stats_path: str = "data/trace_preprocess.json") -> torch.Tensor:
    """Raw photo -> (3, 384, 384) in [0, 1]: steps 3, 4, 6, then 7 (known camera only) and 8.

    Raises ValueError if the retina cannot be found (the quality gate should ask for a retake).
    """
    with open(stats_path) as f:
        stats = json.load(f)
    fov = find_fov(img_rgb_uint8)
    if fov is None:
        raise ValueError("retina not found (fov_fail)")
    std, mask = standardise_geometry(img_rgb_uint8, fov)
    lit = correct_illumination(std, mask, strength=stats["params"]["illumination_strength"])
    return finish_view(lit, mask, source_id, stats)
