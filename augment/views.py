"""Training views for contrastive pretraining.

policy "plain": the standard MoCo v2 recipe (the Phase 2 baseline): random resized crop 20-100%, colour jitter
0.4/0.4/0.4/0.1 with p 0.8, greyscale p 0.2, Gaussian blur sigma 0.1-2 with p 0.5, horizontal flip.
Section 7.3 explains why TRACE replaces it (greyscale hides red lesions; small crops can cut the only lesion;
strong jitter erases colour evidence).

policy "retina" (section 7.3, TRACE): square crop covering 60-100% of the retina's area, centred so it stays mostly
retina; rotation +-15 deg and flip 50% in the same resampling; step 7 (per-camera colour) on half the views;
camera artifacts (augment/artifacts.py, optional hook); jitter 0.2/0.2/0.1/0.02 with p 0.8; never greyscale;
blur sigma 0.1-1.0 with p 0.3; step 8 (CLAHE) last, with its tile count scaled to the crop so a tile covers the
same patch of retina as at evaluation. All of it runs at view size, which is what keeps the loader fast.

Views are made from the processed 512 px photo (steps 3-6 done) and always end with step 8 (CLAHE), which is
part of preprocessing, not augmentation. Output: uint8 array (3, size, size).
"""
from __future__ import annotations

import cv2
import numpy as np

from data.preprocess import camera_color, clahe_l


def random_resized_crop(img: np.ndarray, rng: np.random.Generator, size: int, scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)) -> np.ndarray:
    h, w = img.shape[:2]
    for _ in range(10):
        area = h * w * rng.uniform(*scale)
        r = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw, ch = int(round(np.sqrt(area * r))), int(round(np.sqrt(area / r)))
        if 0 < cw <= w and 0 < ch <= h:
            x0, y0 = rng.integers(0, w - cw + 1), rng.integers(0, h - ch + 1)
            return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (size, size), interpolation=cv2.INTER_AREA if cw > size else cv2.INTER_LINEAR)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def color_jitter(img: np.ndarray, rng: np.random.Generator, b=0.4, c=0.4, s=0.4, hue=0.1,
                 mask: np.ndarray | None = None) -> np.ndarray:
    """torchvision-style jitter in random order (brightness, contrast, saturation, hue) on uint8 RGB.
    With a mask, the contrast mean is taken inside the retina and the outside stays black."""
    x = img.astype(np.float32)
    grey_w = np.array([0.299, 0.587, 0.114], np.float32)
    for op in rng.permutation(4):
        if op == 0:
            x = x * np.float32(rng.uniform(1 - b, 1 + b))
        elif op == 1:
            grey = x @ grey_w
            m = np.float32(grey[mask > 0].mean() if mask is not None and mask.any() else grey.mean())
            x = (x - m) * np.float32(rng.uniform(1 - c, 1 + c)) + m
        elif op == 2:
            g = (x @ grey_w)[..., None]
            x = (x - g) * np.float32(rng.uniform(1 - s, 1 + s)) + g
        else:
            hsv = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)  # 8-bit hue: 0-179
            hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + int(round(rng.uniform(-hue, hue) * 180))) % 180).astype(np.uint8)
            x = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float32)
        np.clip(x, 0, 255, out=x)
    out = x.astype(np.uint8)
    return out * mask[..., None] if mask is not None else out


def plain_view_np(img: np.ndarray, rng: np.random.Generator, size: int = 224) -> np.ndarray:
    """One MoCo v2 view as uint8 (3, size, size). img: the processed photo AFTER step 8 (see prepare).
    No torch here: loader workers import this module (augment/loader.py explains why)."""
    img = random_resized_crop(img, rng, size)
    if rng.random() < 0.8:
        img = color_jitter(img, rng)
    if rng.random() < 0.2:
        img = np.repeat(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=2)
    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.1, 2.0))
    if rng.random() < 0.5:
        img = img[:, ::-1]
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def prepare(rgb: np.ndarray, mask: np.ndarray, clahe_clip: float = 1.5) -> np.ndarray:
    """Step 8 on the whole processed photo, once; every plain view is then cut from the result."""
    return clahe_l(rgb, mask, clip=clahe_clip)


# ---------------------------------------------------------------- retina-safe views (TRACE, section 7.3)

def retina_geometry(img: np.ndarray) -> tuple[float, float, float]:
    """Centre and radius of the retina circle. Step 4 maps the retina to a circle whose diameter is the photo's
    width, centred, so the geometry comes from the size. (The mask is smaller on flat-cut photos, so measuring
    its area would underestimate the radius.)"""
    h, w = img.shape[:2]
    return (w - 1) / 2, (h - 1) / 2, w / 2


def sample_crop(rng: np.random.Generator, cx: float, cy: float, R: float, scale) -> tuple[float, float, float]:
    """Square crop with area = s * (retina area), s ~ U(scale). Its centre is uniform in a disc of radius
    R - side/2 around the retina centre, so the crop stays mostly retina. Returns (centre x, centre y, side)."""
    side = float(np.sqrt(rng.uniform(*scale) * np.pi * R * R))
    room = max(0.0, R - side / 2)
    r, phi = room * np.sqrt(rng.random()), rng.uniform(0, 2 * np.pi)
    return cx + r * np.cos(phi), cy + r * np.sin(phi), side


def crop_rotate(rgb: np.ndarray, mask: np.ndarray, x: float, y: float, side: float, angle: float, out: int,
                flip: bool, point=None) -> tuple[np.ndarray, np.ndarray, tuple[float, float] | None]:
    """Cut the square (x, y, side), rotated by angle degrees, resized to out x out. The photo is first shrunk
    with area averaging (anti-aliasing, so fine vessels do not alias), then one affine warp does the rotation and
    the cut. RGB and mask travel together as one RGBA image. point: a photo (x, y) to map into view pixels."""
    rgba = np.dstack([rgb, mask.astype(np.uint8) * 255])
    k = out / side
    kx = ky = 1.0
    if k < 1:
        rgba = cv2.resize(rgba, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        kx, ky = rgba.shape[1] / rgb.shape[1], rgba.shape[0] / rgb.shape[0]
        x, y, scale = x * kx, y * ky, 1.0
    else:
        scale = k
    M = cv2.getRotationMatrix2D((x, y), angle, scale)
    M[0, 2] += out / 2 - x
    M[1, 2] += out / 2 - y
    v = cv2.warpAffine(rgba, M, (out, out), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mapped = None
    if point is not None:
        px, py = M @ np.array([point[0] * kx, point[1] * ky, 1.0])
        mapped = (out - 1 - px if flip else px, py)
    if flip:
        v = v[:, ::-1]
    m = (v[..., 3] > 127).astype(np.uint8)
    return np.ascontiguousarray(v[..., :3]) * m[..., None], m, mapped


RETINA_DEFAULTS = dict(rot=15.0, flip_p=0.5, colour_p=0.5, jitter_p=0.8, jitter=(0.2, 0.2, 0.1, 0.02),
                       blur_p=0.3, blur_sigma=(0.1, 1.0), clahe_clip=1.5, clahe_tiles=8)


def retina_view_np(rgb: np.ndarray, mask: np.ndarray, rng: np.random.Generator, size: int = 224, scale=(0.6, 1.0),
                   camera: dict | None = None, pool: dict | None = None, colour_alpha: float = 0.7,
                   artifacts=None, **kw) -> np.ndarray:
    """One retina-safe view as uint8 (3, size, size), from the processed photo BEFORE step 8.

    camera / pool: LAB {mean, std} of this photo's camera and of the pool (data/trace_preprocess.json); step 7 is
    applied with probability colour_p when they are given. artifacts(img, mask, geom, rng) -> img is the optional
    camera-artifact hook, geom = (centre x, centre y, radius) of the retina in view pixels.
    Order (section 6): crop -> step 7 -> artifacts -> jitter -> blur -> step 8. Blur and jitter stand for capture
    differences, so they come before step 8, which is processing."""
    p = {**RETINA_DEFAULTS, **kw}
    cx, cy, R = retina_geometry(rgb)
    x, y, side = sample_crop(rng, cx, cy, R, scale)
    angle = rng.uniform(-p["rot"], p["rot"])
    flip = rng.random() < p["flip_p"]
    img, m, (vx, vy) = crop_rotate(rgb, mask, x, y, side, angle, size, flip, point=(cx, cy))
    geom = (vx, vy, R * size / side)  # retina centre and radius in view pixels
    current = camera  # the view's colour distribution now (LAB mean/std), handed to the colour-profile artifact
    if camera is not None and pool is not None and rng.random() < p["colour_p"]:
        img = camera_color(img, m, camera["mean"], camera["std"], pool["mean"], pool["std"], colour_alpha)
        # step 7 is linear, so the result's mean and std are exact blends: (1 - a) camera + a pool
        current = {k: [(1 - colour_alpha) * c + colour_alpha * q for c, q in zip(camera[k], pool[k])] for k in ("mean", "std")}
    if artifacts is not None:
        img = artifacts(img, m, geom, rng, colour_from=current)
    if rng.random() < p["jitter_p"]:
        img = color_jitter(img, rng, *p["jitter"], mask=m)
    if rng.random() < p["blur_p"]:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*p["blur_sigma"])) * m[..., None]
    if p["clahe_clip"] > 0:  # 0 switches step 8 off (tests; the non-CLAHE image of section 6 step 8)
        tiles = max(2, int(round(p["clahe_tiles"] * side / (2 * R))))  # a tile covers 1/8 of the retina's diameter
        img = clahe_l(img, m, clip=p["clahe_clip"], tiles=tiles)
    return np.ascontiguousarray(img.transpose(2, 0, 1))
