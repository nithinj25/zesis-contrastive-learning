"""Approach 2's stress-test corruptions (trace-ai-retinal-diagnostics, trace_ssl/data/artifacts.py, commit 33faec5),
ported unchanged in effect. Their functions take BGR images; `corrupt` converts our RGB photos to BGR and back, so
colours (haze veil, glare tint, colour cast) are exactly theirs. 11 corruptions x 5 severities (0.2 ... 1.0); the
random draws come from crc32(image key | name | level), so every encoder sees the identical corrupted set.
"""
from __future__ import annotations

import zlib

import cv2
import numpy as np


def _radius_grid(mask, center_shift=(0.0, 0.0)):
    h, w = mask.shape
    r = w / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2 + center_shift[0] * r, h / 2 + center_shift[1] * r
    return np.hypot(xx - cx, yy - cy) / r


def _smooth_field(shape, rng, cells=4):
    small = rng.random((cells, cells)).astype(np.float32)
    field = cv2.resize(small, shape[::-1], interpolation=cv2.INTER_CUBIC)
    field -= field.min()
    return field / max(field.max(), 1e-6)


def _finish(out, mask):
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[mask == 0] = 0
    return out


def vignetting(img, mask, s, rng):
    shift = rng.uniform(-0.15, 0.15, 2) * s
    rho = _radius_grid(mask, shift)
    falloff = np.clip(1.0 - 0.85 * s * np.clip(rho, 0, 1.3) ** 2.5, 0.05, 1.0)
    return _finish(img * falloff[..., None], mask)


def defocus_blur(img, mask, s, rng):
    r = max(1, int(round(1 + 7 * s)))
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, cv2.FILLED)
    return _finish(cv2.filter2D(img, -1, k / k.sum()), mask)


def motion_blur(img, mask, s, rng):
    length = max(3, int(round(3 + 15 * s)))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    return _finish(cv2.filter2D(img, -1, k / max(k.sum(), 1e-6)), mask)


def haze(img, mask, s, rng):
    alpha = (0.25 + 0.75 * _smooth_field(mask.shape, rng)) * 0.65 * s
    veil = np.array([150, 170, 190], np.float32)  # BGR
    return _finish(img.astype(np.float32) * (1 - alpha[..., None]) + veil * alpha[..., None], mask)


def reflections(img, mask, s, rng):
    h, w = mask.shape
    glare = np.zeros((h, w), np.float32)
    start = rng.uniform(0, 360)
    cv2.ellipse(glare, (w // 2, h // 2), (int(0.9 * w / 2),) * 2, 0, start, start + rng.uniform(40, 120),
                1.0, thickness=max(2, w // 40))
    for _ in range(rng.integers(1, 4)):
        c = rng.uniform(0.2, 0.8, 2) * (w, h)
        cv2.circle(glare, (int(c[0]), int(c[1])), int(w * rng.uniform(0.01, 0.04)), 1.0, cv2.FILLED)
    glare = cv2.GaussianBlur(glare, (0, 0), w / 60)
    glare /= max(glare.max(), 1e-6)
    tint = np.array([200, 235, 255], np.float32)  # BGR
    return _finish(img + glare[..., None] * tint * s, mask)


def uneven_illumination(img, mask, s, rng):
    gain = 1.0 + (_smooth_field(mask.shape, rng) - 0.5) * 1.2 * s
    return _finish(img * gain[..., None], mask)


def exposure(img, mask, s, rng):
    sign = rng.choice([-1, 1])
    gain = 1 + sign * 0.6 * s
    gamma = 1 - sign * 0.4 * s
    return _finish(255.0 * (np.clip(img * gain, 0, 255) / 255.0) ** gamma, mask)


def color_cast(img, mask, s, rng):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    theta = rng.uniform(0, 2 * np.pi)
    lab[..., 1] += 18 * s * np.cos(theta)
    lab[..., 2] += 18 * s * np.sin(theta)
    return _finish(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR), mask)


def sensor_noise(img, mask, s, rng):
    x = img.astype(np.float32)
    noise = rng.normal(0, 1, x.shape).astype(np.float32) * 25 * s * np.sqrt(np.clip(x, 16, 255) / 128)
    return _finish(x + noise, mask)


def jpeg(img, mask, s, rng):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(round(95 - 85 * s))])
    return _finish(cv2.imdecode(buf, cv2.IMREAD_COLOR), mask)


def low_resolution(img, mask, s, rng):
    h, w = mask.shape
    f = 1 + 5 * s
    small = cv2.resize(img, (max(8, int(w / f)), max(8, int(h / f))), interpolation=cv2.INTER_AREA)
    return _finish(cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR), mask)


ARTIFACTS = {f.__name__: f for f in (vignetting, defocus_blur, motion_blur, haze, reflections, uneven_illumination,
                                      exposure, color_cast, sensor_noise, jpeg, low_resolution)}
SEVERITIES = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}


def stress_rng(key: str, name: str, level: int) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{key}|{name}|{level}".encode()))


def corrupt(rgb: np.ndarray, mask: np.ndarray, name: str, level: int, key: str) -> np.ndarray:
    """Our RGB photo -> the same photo with Approach 2's corruption `name` at `level` (1-5), RGB out."""
    if level == 0:
        return rgb.copy()
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    out = ARTIFACTS[name](bgr, mask, SEVERITIES[level], stress_rng(key, name, level))
    return np.ascontiguousarray(out[..., ::-1])
