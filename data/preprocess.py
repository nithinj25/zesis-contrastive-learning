"""TRACE preprocessing (CLAUDE.md section 6). This file covers steps 3, 4 and 6 so far.

Images are RGB uint8 arrays. Masks are uint8 arrays holding 0 or 1.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TARGET_DIAMETER = 512
MIN_RADIUS_FRACTION = 0.25  # section 19.9: a smaller circle counts as fov_fail
MIN_ASPECT, MAX_ASPECT = 0.7, 1.5  # ry / rx; 4:3 squeezed into a square gives 1.33


@dataclass
class Fov:
    mask: np.ndarray  # retina area: the fitted ellipse, minus any flat-cut top or bottom
    cx: float
    cy: float
    rx: float
    ry: float  # rx != ry means the photo was stretched (e.g. 4:3 squeezed into a 512x512 square)
    fit: float  # share of outline points on the ellipse; low means treat with suspicion

    @property
    def radius(self) -> float:
        return self.rx


def _solve_ellipse(pts: np.ndarray) -> np.ndarray:
    """Axis-aligned ellipse A x^2 + B y^2 + C x + D y = 1. Returns (..., [cx, cy, rx, ry]); NaN if invalid."""
    x, y = pts[..., 0], pts[..., 1]
    M = np.stack([x**2, y**2, x, y], axis=-1)
    if pts.ndim == 2:
        sol = np.linalg.lstsq(M, np.ones(len(x)), rcond=None)[0]  # least squares over all points
    else:
        sol = np.linalg.solve(M, np.ones(M.shape[:2])[..., None])[..., 0]  # exact, per 4-point sample
    A, B, C, D = sol[..., 0], sol[..., 1], sol[..., 2], sol[..., 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        cx, cy = -C / (2 * A), -D / (2 * B)
        k = 1 + A * cx**2 + B * cy**2
        rx, ry = np.sqrt(k / A), np.sqrt(k / B)
    bad = (A <= 0) | (B <= 0) | (k <= 0)
    return np.where(np.asarray(bad)[..., None], np.nan, np.stack([cx, cy, rx, ry], axis=-1))


def _fit_ellipse(points: np.ndarray, h: int, w: int) -> tuple[float, float, float, float, float] | None:
    """RANSAC, so dark bites in the outline don't pull the ellipse inwards."""
    rng = np.random.default_rng(0)  # fixed, so preprocessing is repeatable
    s = max(h, w) / 2
    if len(points) > 2000:
        points = points[rng.choice(len(points), 2000, replace=False)]
    p = (points - [w / 2, h / 2]) / s  # normalised, for numerical stability
    tol = (0.01 * max(h, w) + 1) / s

    def residual(e: np.ndarray) -> np.ndarray:
        d = np.hypot((p[:, 0, None] - e[..., 0]) / e[..., 2], (p[:, 1, None] - e[..., 1]) / e[..., 3])
        return np.abs(d - 1) * (e[..., 2] + e[..., 3]) / 2

    samples = p[rng.integers(0, len(p), (400, 4))]
    x, y = samples[..., 0], samples[..., 1]
    usable = np.abs(np.linalg.det(np.stack([x**2, y**2, x, y], axis=-1))) > 1e-12
    if not usable.any():
        return None
    e = _solve_ellipse(samples[usable])
    ratio = e[:, 3] / e[:, 2]
    keep = np.isfinite(e).all(1) & (ratio >= MIN_ASPECT) & (ratio <= MAX_ASPECT)
    keep &= (e[:, 2] * s >= MIN_RADIUS_FRACTION * w) & (e[:, 2] * s <= max(h, w))
    if not keep.any():
        return None
    e = e[keep]
    best = e[(residual(e) < tol).sum(0).argmax()]
    for _ in range(2):  # refine on the inliers
        inliers = p[residual(best[None])[:, 0] < tol]
        if len(inliers) < 8:
            return None
        refined = _solve_ellipse(inliers)
        if not np.isfinite(refined).all():
            return None
        best = refined
    fit = float((residual(best[None])[:, 0] < tol).mean())
    cx, cy, rx, ry = best[0] * s + w / 2, best[1] * s + h / 2, best[2] * s, best[3] * s
    if abs(ry / rx - 1) < 0.02:  # no real stretch: treat it as a circle
        rx = ry = (rx + ry) / 2
    return float(cx), float(cy), float(rx), float(ry), fit


def find_fov(img: np.ndarray) -> Fov | None:
    """Step 3: find the retina. Returns None on failure (tag it reject / fov_fail).

    Brightness cannot define the retina: diseased or dim retina can be as dark as the
    background. So brightness only gives a rough outline, an ellipse is fitted to its edge,
    and the mask comes from that ellipse, so nothing inside the retina is ever cut.
    """
    h, w = img.shape[:2]
    red = cv2.GaussianBlur(img[..., 0].astype(np.float32), (0, 0), max(1.0, min(h, w) / 512))
    if red.max() <= 0:
        return None
    p = max(8, min(h, w) // 32)
    corners = np.median([np.median(a) for a in (red[:p, :p], red[:p, -p:], red[-p:, :p], red[-p:, -p:])])
    centre = np.median(red[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4])
    threshold = max(0.06 * red.max(), corners + max(8.0, 0.10 * (centre - corners)))
    blob = (red > threshold).astype(np.uint8)
    blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outline = max(contours, key=cv2.contourArea)[:, 0, :].astype(np.float64)
    # Points on the photo's frame say nothing about the retina's edge.
    frame = 3
    inner = (outline[:, 0] >= frame) & (outline[:, 0] < w - frame) & (outline[:, 1] >= frame) & (outline[:, 1] < h - frame)
    if inner.sum() < 30:
        return None
    fitted = _fit_ellipse(outline[inner], h, w)
    if fitted is None:
        return None
    cx, cy, rx, ry, fit = fitted

    yy, xx = np.ogrid[:h, :w]
    mask = (((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1).astype(np.uint8)
    # Flat-cut photos: a sudden jump from almost no retina to a wide row means a real cut.
    # A gradual change is just a dim rim, which stays in the mask.
    filled = np.zeros((h, w), np.uint8)
    cv2.drawContours(filled, [outline.astype(np.int32)[:, None, :]], -1, 1, thickness=cv2.FILLED)
    widths = filled.sum(1).astype(np.float64)
    chord = 2 * rx * np.sqrt(np.maximum(1 - ((np.arange(h) - cy) / ry) ** 2, 0))
    wide = np.where((chord > 0) & (widths >= 0.7 * chord))[0]
    if len(wide):
        top, bottom = int(wide[0]), int(wide[-1])
        if top - (cy - ry) > 0.02 * ry and top >= 4 and widths[top - 4] < 0.2 * chord[top]:
            mask[:top] = 0
        if (cy + ry) - bottom > 0.02 * ry and bottom < h - 4 and widths[bottom + 4] < 0.2 * chord[bottom]:
            mask[bottom + 1 :] = 0
    return Fov(mask, cx, cy, rx, ry, fit)


def standardise_geometry(
    img: np.ndarray, fov: Fov, size: int = TARGET_DIAMETER
) -> tuple[np.ndarray, np.ndarray]:
    """Step 4: crop around the ellipse and resize so the retina is a circle of diameter = size.

    Mapping the ellipse to a circle also undoes any stretch in the photo. It can run past the
    image edge (flat-topped photos), so missing areas are padded with black. No laterality
    flip (section 2.3 item 5).
    """
    x0, x1 = int(round(fov.cx - fov.rx)), int(round(fov.cx + fov.rx))
    y0, y1 = int(round(fov.cy - fov.ry)), int(round(fov.cy + fov.ry))
    h, w = img.shape[:2]
    pad = (max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w))

    def window(a: np.ndarray) -> np.ndarray:
        a = cv2.copyMakeBorder(a, *pad, cv2.BORDER_CONSTANT, value=0)
        top, left = y0 + pad[0], x0 + pad[2]
        return a[top : top + (y1 - y0), left : left + (x1 - x0)]

    interp = cv2.INTER_AREA if (x1 - x0) > size else cv2.INTER_CUBIC
    out = cv2.resize(window(img), (size, size), interpolation=interp)
    blob = cv2.resize(window(fov.mask), (size, size), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.ogrid[:size, :size]
    centre = (size - 1) / 2
    circle = (xx - centre) ** 2 + (yy - centre) ** 2 <= (size / 2) ** 2
    mask = (circle & (blob > 0)).astype(np.uint8)
    return out * mask[..., None], mask


def correct_illumination(img: np.ndarray, mask: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Step 6: flatten slow brightness changes in LAB L only, ignoring the black border."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, m = lab[..., 0], mask.astype(np.float32)
    # 64 px at 512: wide enough to leave the optic disc's contrast alone (section 20).
    sigma = img.shape[0] / 8
    # Dividing by the blurred mask stops the black border darkening the estimate at the edges.
    bg = cv2.GaussianBlur(L * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)
    L_flat = L / (bg + 1e-6) * L[m > 0].mean()
    lab[..., 0] = (1 - strength) * L + strength * L_flat
    out = cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out * mask[..., None]


def shift_channel(x, cam_mean: float, cam_std: float, pool_mean: float, pool_std: float, alpha: float = 0.7):
    """Step 7 on one channel: move a camera's distribution part of the way to the pool's."""
    mapped = (x - cam_mean) / cam_std * pool_std + pool_mean
    return (1 - alpha) * x + alpha * mapped


def camera_color(img: np.ndarray, mask: np.ndarray, cam_mean, cam_std, pool_mean, pool_std, alpha: float = 0.7) -> np.ndarray:
    """Step 7: per-CAMERA colour shift in LAB (stats over all of that camera's photos, never per image).

    A per-image shift would pull a pale eye back to average and erase pallor; a per-camera shift keeps
    each eye's difference from its camera's average. Training applies it to a random half of views;
    evaluation applies it only when the camera is known (section 19.1).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        lab[..., c] = shift_channel(lab[..., c], cam_mean[c], cam_std[c], pool_mean[c], pool_std[c], alpha)
    out = cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out * mask[..., None]


def clahe_l(img: np.ndarray, mask: np.ndarray, clip: float = 1.5, tiles: int = 8) -> np.ndarray:
    """Step 8: mild local contrast on LAB L only, so small lesions show while colour is untouched."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * mask[..., None]


def pooled_stats(means: np.ndarray, stds: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-photo (mean, std, pixel count) into the mean and std of all pixels together.

    means, stds: (n_photos, 3). Uses E[x^2] = std^2 + mean^2, weighted by pixel count.
    """
    w = counts[:, None].astype(np.float64)
    mean = (w * means).sum(0) / w.sum()
    second = (w * (stds**2 + means**2)).sum(0) / w.sum()
    return mean, np.sqrt(np.maximum(second - mean**2, 0))


def eval_view(rgb: np.ndarray, mask: np.ndarray, source_id: str | None, stats: dict, size: int = 384) -> np.ndarray:
    """Evaluation-time steps 7 (known camera only) and 8 on a processed 512 px photo, then resize: uint8 (size, size, 3).

    CLAUDE.md section 19.1 policy: a camera with stored statistics (data/trace_preprocess.json) gets step 7; an
    unknown camera skips it. No torch here, so worker processes can run it cheaply (encoder/export.py wraps it).
    """
    cams = stats.get("cameras", {})
    if source_id in cams:
        c, p = cams[source_id], stats["pool"]
        rgb = camera_color(rgb, mask, c["mean"], c["std"], p["mean"], p["std"], stats["params"]["colour_alpha"])
    rgb = clahe_l(rgb, mask, clip=stats["params"]["clahe_clip"])
    if rgb.shape[0] != size:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def prepare_photo(img_rgb: np.ndarray, source_id: str | None, stats: dict, size: int = 384) -> np.ndarray:
    """Raw RGB photo -> uint8 (size, size, 3): steps 3, 4, 6, then eval_view. ValueError if no retina is found."""
    fov = find_fov(img_rgb)
    if fov is None:
        raise ValueError("retina not found (fov_fail)")
    std, mask = standardise_geometry(img_rgb, fov)
    lit = correct_illumination(std, mask, strength=stats["params"]["illumination_strength"])
    return eval_view(lit, mask, source_id, stats, size)
