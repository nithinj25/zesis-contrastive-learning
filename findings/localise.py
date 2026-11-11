"""Find the optic disc and the fovea on a processed 512 px photo, and cut the disc and macula crops (CLAUDE.md 8.1).

Disc: the most disc-like blob, i.e. brighter than its surroundings at disc scale:
    blob = blur(brightness, DD/4) - blur(brightness, DD),   brightness = (R + G) / 2
Blurs are mask-normalised (divided by the blurred mask) so the black border does not darken the rim. The centre must
lie half a disc (39 px) inside the retina edge: without that margin, bright rim glare (common in RFMiD 2.0's
1504x1000 photos) won. Chosen on IDRiD *training* photos (disc within 0.5 DD): raw brightness + vessel density 90.3%,
blob + 0.5 x vessel density 97.3%, blob alone 98.1%; the margin made no difference there. On a by-eye sheet of
other cameras, the vessel cue pulled the guess onto exudate clusters, so it is not used.
Fovea: the darkest compact spot, i.e. darker than its surroundings,
    spot = blur(L, DD/3) - blur(L, DD)       (LAB lightness)
1.8 to 3.2 disc diameters from the disc, at least half a disc inside the retina edge, within 35 degrees of
horizontal on the photo-centre side of the disc. Chosen on IDRiD *training* photos (fovea within 0.5 DD): darkest
blurred spot 42% (guesses sat on the dark flat-cut edge), with the edge margin 71% (guesses sat in the darker
periphery), darker-than-surroundings 90%. Ring width (1.9-3.0 vs 1.8-3.2) and direction rule (45 degrees towards
the photo centre vs 35 degrees from horizontal) moved it by about 1 point; the wide ring is kept for cameras with
another field of view.
Disc diameter DD = 78 px at a 512 px retina: IDRiD training photos put the fovea a median 196 px from the disc,
and the disc-fovea distance is about 2.5 DD. Cameras with a narrower field show a larger disc; DD stays fixed.
Crops: disc = a 2 DD square around the disc; macula = a 3 DD square around the fovea (the macula is ~5.5 mm
across, about 3 disc diameters). On IDRiD test the disc crop contains the true disc centre in 99.0% of photos and
the macula crop the true fovea in 93.2%. loc_ok = False only if no candidate exists. A confidence score (how far
the winning spot stands out from the other candidates) was tried and dropped: on IDRiD train it was the same for
right and wrong guesses (fovea: median 2.35 vs 2.16 standard deviations).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

DD = 78.0
SIZE = 512


@dataclass
class Landmarks:
    disc_x: float
    disc_y: float
    fovea_x: float
    fovea_y: float
    ok: bool
    reason: str = ""


def _blur_in_mask(a: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    m = mask.astype(np.float32)
    return cv2.GaussianBlur(a * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)


def _z(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    v = a[mask > 0]
    return (a - v.mean()) / (v.std() + 1e-6)


def find_disc(rgb: np.ndarray, mask: np.ndarray, dd: float = DD) -> tuple[float, float] | None:
    """Centre of the most disc-like blob: brighter than its surroundings at disc scale (difference of blurs)."""
    bright = (rgb[..., 0].astype(np.float32) + rgb[..., 1].astype(np.float32)) / 2
    blob = _blur_in_mask(bright, mask, dd / 4) - _blur_in_mask(bright, mask, dd)
    allowed = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dd) | 1,) * 2))
    if allowed.sum() == 0:
        return None
    score = np.where(allowed > 0, blob, -np.inf)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    return float(x), float(y)


def find_fovea(rgb: np.ndarray, mask: np.ndarray, disc: tuple[float, float], dd: float = DD,
               r_min: float = 1.8, r_max: float = 3.2, max_angle: float = 35.0) -> tuple[float, float] | None:
    """Centre of the darkest compact spot in the search ring on the photo-centre side of the disc."""
    h, w = mask.shape
    # half a disc inside the retina edge: the flat-cut top and bottom of a photo look dark next to the black border
    inner = cv2.erode(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(dd) | 1,) * 2))
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = xx - disc[0], yy - disc[1]
    dist = np.hypot(dx, dy)
    side = 1.0 if disc[0] <= (w - 1) / 2 else -1.0  # the fovea lies on the photo-centre side of the disc
    near_horizontal = np.abs(np.arctan2(dy, dx * side)) <= np.deg2rad(max_angle)
    ring = (dist >= r_min * dd) & (dist <= r_max * dd) & (inner > 0) & near_horizontal
    if not ring.any():
        return None
    L = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32)
    spot = _blur_in_mask(L, mask, dd / 3) - _blur_in_mask(L, mask, dd)  # darker than its surroundings
    score = np.where(ring, spot, np.inf)
    y, x = np.unravel_index(np.argmin(score), score.shape)
    return float(x), float(y)


def localise(rgb: np.ndarray, mask: np.ndarray, dd: float = DD) -> Landmarks:
    disc = find_disc(rgb, mask, dd)
    if disc is None:
        return Landmarks(np.nan, np.nan, np.nan, np.nan, False, "no_disc")
    fov = find_fovea(rgb, mask, disc, dd)
    if fov is None:
        return Landmarks(*disc, np.nan, np.nan, False, "no_fovea")
    return Landmarks(*disc, *fov, True)


def crop(rgb: np.ndarray, centre: tuple[float, float], side: float, out: int = 384) -> np.ndarray:
    """Square crop of `side` px around `centre`, padded with black past the image edge, resized to out x out."""
    half = side / 2
    x0, y0 = int(round(centre[0] - half)), int(round(centre[1] - half))
    s = int(round(side))
    h, w = rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
    a = cv2.copyMakeBorder(rgb, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    patch = a[y0 + pad : y0 + pad + s, x0 + pad : x0 + pad + s]
    return cv2.resize(patch, (out, out), interpolation=cv2.INTER_CUBIC)


def disc_and_macula_crops(rgb: np.ndarray, lm: Landmarks, dd: float = DD, out: int = 384):
    return crop(rgb, (lm.disc_x, lm.disc_y), 2 * dd, out), crop(rgb, (lm.fovea_x, lm.fovea_y), 3 * dd, out)
