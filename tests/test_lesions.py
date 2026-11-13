import cv2
import numpy as np
import torch
import torch.nn.functional as F

from augment.lesions import blob_response, crop_sampling_map, sample_guided_crop
from encoder.loss import trace_loss


def _retina(size=512):
    """Flat orange retina disc, one dark round blob (a 'haemorrhage') and one dark straight line (a 'vessel')."""
    rgb = np.zeros((size, size, 3), np.uint8)
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 2, 1, -1)
    rgb[mask > 0] = (200, 110, 50)
    cv2.circle(rgb, (180, 300), 5, (120, 40, 20), -1)                 # blob
    cv2.line(rgb, (300, 150), (420, 330), (120, 40, 20), 3)          # vessel
    rgb = cv2.GaussianBlur(rgb, (0, 0), 1.0) * mask[..., None]
    return rgb.astype(np.uint8), mask


def test_round_blob_beats_vessel():
    rgb, mask = _retina()
    r = blob_response(rgb, mask)
    blob = r[290:311, 170:191].max()
    vessel = r[200:280, 330:390].max()
    assert blob > 3 * vessel


def test_sampling_map_is_a_distribution_aimed_at_the_blob():
    rgb, mask = _retina()
    p = crop_sampling_map(blob_response(rgb, mask), mask)
    assert abs(p.sum() - 1) < 1e-5 and p.min() >= 0
    cell = p[300 * 64 // 512, 180 * 64 // 512]
    assert cell > 50 * np.median(p[p > 0])            # the blob's cell is far likelier than a typical cell
    assert (p > 0).sum() > 0.5 * (cv2.resize(mask.astype(np.float32), (64, 64)) > 0.5).sum()  # floor: all retina reachable


def test_guided_crops_stay_on_retina_and_near_blobs():
    rgb, mask = _retina()
    p = crop_sampling_map(blob_response(rgb, mask), mask)
    rng = np.random.default_rng(0)
    crops = [sample_guided_crop(rng, p, mask) for _ in range(200)]
    assert all(c is not None for c in crops)
    for x, y, side in crops:
        assert 96 <= side <= 160
        x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
        assert mask[y0:y0 + int(side), x0:x0 + int(side)].mean() >= 0.85
    near = np.mean([abs(x - 180) < 40 and abs(y - 300) < 40 for x, y, _ in crops])
    assert near > 0.5  # 80% of the mass sits on the blob


def test_mask_self_equals_dropping_that_key():
    torch.manual_seed(0)
    q, k = F.normalize(torch.randn(2, 8), dim=1), F.normalize(torch.randn(2, 8), dim=1)
    queue = F.normalize(torch.randn(6, 8), dim=1)
    qsrc, src, kth = torch.tensor([0, 1, 0, 1, 0, 1]), torch.tensor([0, 1]), torch.ones(6)
    excl = torch.zeros(2, 6, dtype=torch.bool)
    excl[0, 3] = True  # anchor 0's own old key sits at slot 3
    masked, _ = trace_loss(q, k, src, queue, qsrc, kth, beta=2.0, mine=False, exclude=excl)
    keep = [0, 1, 2, 4, 5]
    a, _ = trace_loss(q[:1], k[:1], src[:1], queue[keep], qsrc[keep], kth[keep], beta=2.0, mine=False)
    b, _ = trace_loss(q[1:], k[1:], src[1:], queue, qsrc, kth, beta=2.0, mine=False)
    assert torch.allclose(masked, (a + b) / 2, atol=1e-6)


def test_no_exclude_is_unchanged():
    torch.manual_seed(1)
    q, k = F.normalize(torch.randn(3, 8), dim=1), F.normalize(torch.randn(3, 8), dim=1)
    queue = F.normalize(torch.randn(5, 8), dim=1)
    args = (q, k, torch.tensor([0, 1, 0]), queue, torch.tensor([0, 1, 1, 0, -1]), torch.ones(5))
    a, _ = trace_loss(*args, mine=False)
    b, _ = trace_loss(*args, mine=False, exclude=torch.zeros(3, 5, dtype=torch.bool))
    assert torch.equal(a, b)
