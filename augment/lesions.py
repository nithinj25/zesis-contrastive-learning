"""Label-free lesion-candidate map for aiming local crops (from Approach 2, trace-ai-retinal-diagnostics,
trace_ssl/data/lesions.py, commit 33faec5; ported to TRACE's RGB processed photos).

Not a detector: a cheap "where might something small and round be?" map. On the green channel, the scale-normalised
determinant of the Hessian is large where intensity curves in both directions (round blobs: microaneurysms,
haemorrhages, exudates, drusen) and ~0 or negative along vessels, which curve in one direction only. It is divided by
the local contrast, so dull and vivid photos compare; the optic disc (the brightest region) and a rim band are
zeroed. crop_sampling_map mixes it with a uniform floor (default 20%), so healthy tissue is still visited.
"""
from __future__ import annotations

import cv2
import numpy as np

SIGMAS = (1.5, 3.0)  # blob scales at 512 px: microaneurysm ~3 px, small haemorrhage / exudate ~6-12 px


def masked_blur(x: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur averaging only pixels inside the mask (normalised convolution)."""
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(x * m, (0, 0), sigma) / np.maximum(cv2.GaussianBlur(m, (0, 0), sigma), 1e-6)


def _disc_mask(green: np.ndarray, mask: np.ndarray) -> np.ndarray:
    size = green.shape[0]
    smooth = masked_blur(green, mask, size * 0.03)
    smooth[mask == 0] = 0
    y, x = np.unravel_index(np.argmax(smooth), smooth.shape)
    disc = np.zeros(mask.shape, np.uint8)
    cv2.circle(disc, (int(x), int(y)), int(0.11 * size), 1, cv2.FILLED)
    return disc


def blob_response(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """(H, W) float32 blob strength, 0 outside the usable retina."""
    green = rgb[..., 1].astype(np.float32)
    mask = mask.astype(np.uint8)
    size = green.shape[0]
    rim = int(0.03 * size) | 1
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim, rim)))
    resp = np.zeros_like(green)
    for s in SIGMAS:
        g = masked_blur(green, mask, s)
        gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        det = (gxx * gyy - gxy ** 2) * s ** 4
        resp = np.maximum(resp, np.sqrt(np.clip(det, 0, None)))
    local_std = np.sqrt(masked_blur((green - masked_blur(green, inner, size * 0.05)) ** 2, inner, size * 0.05))
    resp /= np.maximum(local_std, 2.0)
    resp[(inner == 0) | (_disc_mask(green, mask) == 1)] = 0
    return resp


def crop_sampling_map(resp: np.ndarray, mask: np.ndarray, floor: float = 0.2, grid: int = 64) -> np.ndarray:
    """(grid, grid) probabilities for local-crop centres: (1 - floor) x sharpened blob response + floor x uniform."""
    r = cv2.resize(resp, (grid, grid), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA) > 0.5
    r = r ** 2 * m
    p = (1 - floor) * r / max(r.sum(), 1e-6) + floor * m / max(m.sum(), 1)
    return (p / p.sum()).astype(np.float32)


def sample_guided_crop(rng: np.random.Generator, pmap: np.ndarray, mask: np.ndarray, side_px=(96, 160),
                       min_retina: float = 0.85, tries: int = 20) -> tuple[float, float, float] | None:
    """(centre x, centre y, side) of a square local crop: centre drawn from pmap (a cell, then uniform inside it),
    side ~ U(side_px) photo pixels, kept only if >= min_retina of it is retina. None if no try succeeds."""
    h, w = mask.shape
    g = pmap.shape[0]
    flat = pmap.ravel().astype(np.float64)
    flat /= flat.sum()
    for _ in range(tries):
        cy, cx = divmod(int(rng.choice(flat.size, p=flat)), g)
        x, y = (cx + rng.random()) * w / g, (cy + rng.random()) * h / g
        side = float(rng.uniform(*side_px))
        x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
        if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
            continue
        if mask[y0:y0 + int(side), x0:x0 + int(side)].mean() >= min_retina:
            return x, y, side
    return None
