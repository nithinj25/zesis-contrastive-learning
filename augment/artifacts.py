"""Camera-artifact simulator (CLAUDE.md sections 7.3 and 19.5).

Why: contrastive learning teaches the encoder to ignore whatever differs between the two views of one eye. If both
views always share their camera's look (tint, vignetting, haze, compression), the camera stays a free shortcut, and
the camera probe on plain MoCo features is 0.97 (chance 0.14). Each artifact below makes a view look as if another
camera, or a worse capture, took it, so only the retina itself is left to match on.

Rule: an artifact must look like the camera or the capture, never like disease. So there are NO dust spots: they look
like microaneurysms, and the encoder would learn that small dark dots do not matter.

Every artifact is applied independently per view, with its own probability. Lengths in section 19.5 are given for a
512 px retina diameter and are rescaled to the retina's diameter in the view (geom from augment/views.py).
Order follows the light path: haze, lash shadow and glint (eye and lens), vignetting and uneven light (optics),
camera colour (sensor), JPEG (storage). Everything stays inside the retina mask. No torch: loader workers import this.
"""
from __future__ import annotations

import cv2
import numpy as np

from data.preprocess import shift_channel

DEFAULT_P = dict(haze=0.2, lashes=0.1, glint=0.15, vignette=0.3, uneven=0.3, colour=0.5, jpeg=0.3)  # never "dust"
REF_DIAMETER = 512.0


def _grid(shape, geom):
    """Distance from the retina centre as a fraction of the radius, plus the x / y offsets in radii."""
    h, w = shape[:2]
    cx, cy, R = geom
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    dx, dy = (xx - cx) / R, (yy - cy) / R
    return np.sqrt(dx * dx + dy * dy), dx, dy


def vignette(img: np.ndarray, geom, s: float) -> np.ndarray:
    """Darker towards the rim: multiply by (1 - s r^2), r = distance from the centre / radius."""
    r, _, _ = _grid(img.shape, geom)
    f = 1 - s * np.minimum(r, 1.0) ** 2
    return img.astype(np.float32) * f[..., None]


def uneven(img: np.ndarray, geom, g: float, angle: float) -> np.ndarray:
    """A smooth brightness ramp in direction angle: factor 1 - g on one side of the retina, 1 + g on the other."""
    _, dx, dy = _grid(img.shape, geom)
    t = np.clip(dx * np.cos(angle) + dy * np.sin(angle), -1, 1)
    return img.astype(np.float32) * (1 + g * t)[..., None]


def haze(img: np.ndarray, h: float, sigma: float) -> np.ndarray:
    """Cataract-like veil, with img in [0, 1]: out = (1 - h) img + h (0.9 blur(img) + 0.1)."""
    x = img.astype(np.float32) / 255
    veil = 0.9 * cv2.GaussianBlur(x, (0, 0), sigma) + 0.1
    return ((1 - h) * x + h * veil) * 255


def glint(img: np.ndarray, centres, radii, intensity: float, aspect, angles) -> np.ndarray:
    """Soft bright elliptical spots (flash reflections): + intensity * 255 at each spot's centre, Gaussian fall-off,
    sigma = radius / 2."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    add = np.zeros((h, w), np.float32)
    for (x0, y0), r, a, th in zip(centres, radii, aspect, angles):
        u = (xx - x0) * np.cos(th) + (yy - y0) * np.sin(th)
        v = -(xx - x0) * np.sin(th) + (yy - y0) * np.cos(th)
        s = r / 2
        add = np.maximum(add, np.exp(-0.5 * ((u / s) ** 2 + (v / (s * a)) ** 2)))
    return img.astype(np.float32) + intensity * 255 * add[..., None]


def lashes(img: np.ndarray, geom, arcs, opacity: float, blur: float) -> np.ndarray:
    """Blurred dark curved bands near the top or bottom rim. arcs: (centre dx, centre dy, radius, start, end,
    thickness), lengths in retina radii, angles in degrees (OpenCV: 90 = bottom, 270 = top)."""
    h, w = img.shape[:2]
    cx, cy, R = geom
    band = np.zeros((h, w), np.float32)
    for dx, dy, rad, a0, a1, th in arcs:
        cv2.ellipse(band, (int(round(cx + dx * R)), int(round(cy + dy * R))), (int(rad * R), int(rad * R)), 0, a0, a1,
                    1.0, max(1, int(round(th * R))), cv2.LINE_AA)
    band = cv2.GaussianBlur(band, (0, 0), max(0.5, blur))
    return img.astype(np.float32) * (1 - opacity * np.clip(band, 0, 1))[..., None]


def colour_profile(img: np.ndarray, mask: np.ndarray, src: dict, camera: dict, alpha: float) -> np.ndarray:
    """Step 7 run towards a camera: move the view's LAB colour from its current distribution src (LAB mean/std)
    towards one camera's. With src = the pool's statistics this is exactly step 7 inverted (section 19.5)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        lab[..., c] = shift_channel(lab[..., c], src["mean"][c], src["std"][c], camera["mean"][c], camera["std"][c], alpha)
    out = cv2.cvtColor(np.clip(np.rint(lab), 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out * mask[..., None]


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


class Artifacts:
    """The hook retina_view_np calls: Artifacts(stats)(img, mask, geom, rng) -> img (uint8 RGB).

    stats: data/trace_preprocess.json (pool + per-camera LAB stats, for the colour profile). p: probabilities per
    artifact (DEFAULT_P; a missing key means never). strong=True uses the top of every range (contact sheets).
    self.last lists what was applied to the latest view (for monitoring and tests)."""

    def __init__(self, stats: dict | None = None, p: dict | None = None, strong: bool = False):
        self.p = dict(DEFAULT_P if p is None else p)
        assert "dust" not in self.p, "dust spots look like microaneurysms: never simulate them"
        self.pool = (stats or {}).get("pool")
        self.cams = list((stats or {}).get("cameras", {}).values())
        self.strong = strong
        self.last: list[str] = []

    def _u(self, rng, lo, hi):
        return hi if self.strong else rng.uniform(lo, hi)

    def __call__(self, img: np.ndarray, mask: np.ndarray, geom, rng: np.random.Generator,
                 colour_from: dict | None = None) -> np.ndarray:
        """colour_from: the view's current LAB mean/std (retina_view_np passes it: its camera's, or the step 7 blend);
        unknown -> the pool's."""
        self.last = []
        fire = {k: rng.random() < self.p.get(k, 0.0) for k in DEFAULT_P}  # one draw each, fixed order: reproducible
        k = 2 * geom[2] / REF_DIAMETER  # view pixels per pixel of a 512 px retina
        x = img
        if fire["haze"]:
            x = _u8(haze(x, self._u(rng, 0.1, 0.4), self._u(rng, 8, 20) * k))
            self.last.append("haze")
        if fire["lashes"]:
            arcs = []
            for _ in range(int(rng.integers(1, 4))):
                top = rng.random() < 0.5
                mid, span = (270.0 if top else 90.0) + rng.uniform(-25, 25), rng.uniform(50, 110)
                arcs.append((rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), rng.uniform(0.78, 0.95), mid - span / 2,
                             mid + span / 2, rng.uniform(0.03, 0.08)))
            x = _u8(lashes(x, geom, arcs, self._u(rng, 0.1, 0.3), 0.02 * 2 * geom[2]))
            self.last.append("lashes")
        if fire["glint"]:
            n = int(rng.integers(1, 4))
            # outer rim only (70-95% of the radius): a white blob of this size at the posterior pole would look like a
            # cotton wool spot or exudate, the same trap as dust and microaneurysms
            rr = rng.uniform(0.70, 0.95, n) * geom[2]
            phi = rng.uniform(0, 2 * np.pi, n)
            centres = list(zip(geom[0] + rr * np.cos(phi), geom[1] + rr * np.sin(phi)))
            radii = [self._u(rng, 0.02, 0.06) * 2 * geom[2] for _ in range(n)]
            x = _u8(glint(x, centres, radii, self._u(rng, 0.2, 0.5), rng.uniform(0.5, 1.0, n), rng.uniform(0, np.pi, n)))
            self.last.append("glint")
        if fire["vignette"]:
            x = _u8(vignette(x, geom, self._u(rng, 0.1, 0.4)))
            self.last.append("vignette")
        if fire["uneven"]:
            x = _u8(uneven(x, geom, self._u(rng, 0.05, 0.15), rng.uniform(0, 2 * np.pi)))
            self.last.append("uneven")
        if fire["colour"] and self.cams and self.pool is not None:
            cam = self.cams[int(rng.integers(len(self.cams)))]
            x = colour_profile(x, mask, colour_from or self.pool, cam, self._u(rng, 0.3, 1.0))
            self.last.append("colour")
        if fire["jpeg"]:
            x = jpeg(x, int(round(40 if self.strong else rng.uniform(40, 95))))
            self.last.append("jpeg")
        return x * mask[..., None]
