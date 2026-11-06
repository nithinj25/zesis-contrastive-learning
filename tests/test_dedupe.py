"""Unit tests for dedupe, on synthetic retinas with random vessel patterns."""
import cv2
import numpy as np

from data.dedupe import (
    MATCH,
    SAME_EYE,
    candidate_pairs,
    coarse_signature,
    decide_drops,
    fine_signature,
    match_score,
    same_eye_groups,
)


def eye(seed, size=512):
    """Same disc, macula and border every time; only the vessel pattern differs by seed."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 2, (170, 90, 40), -1)
    for _ in range(40):
        pts = np.cumsum(rng.normal(0, 18, (12, 2)), axis=0) + rng.integers(100, size - 100, 2)
        cv2.polylines(img, [pts.astype(np.int32)], False, (110, 40, 20), int(rng.integers(1, 4)), cv2.LINE_AA)
    cv2.circle(img, (380, 256), 40, (235, 190, 130), -1)
    cv2.circle(img, (180, 256), 25, (120, 60, 30), -1)
    return img


def test_same_photo_at_different_brightness_and_blur_matches():
    a = eye(1)
    b = cv2.GaussianBlur((a * 0.7).astype(np.uint8), (0, 0), 0.7)
    assert match_score(fine_signature(a), fine_signature(b)) >= SAME_EYE


def test_small_shift_still_matches():
    a = eye(1)
    b = cv2.warpAffine(a, np.float32([[1, 0, 6], [0, 1, -5]]), (512, 512))
    assert match_score(fine_signature(a), fine_signature(b)) >= SAME_EYE


def test_different_eyes_with_same_layout_do_not_match():
    # What fooled pHash: same border, disc and macula position, different vessels.
    assert match_score(fine_signature(eye(1)), fine_signature(eye(2))) < MATCH


def test_blank_photo_matches_nothing():
    # Real case: a fully white EyePACS photo scored 1.0 against 23 different Paraguay eyes.
    blank = np.zeros((512, 512, 3), np.uint8)
    cv2.circle(blank, (256, 256), 256, (255, 255, 255), -1)
    assert match_score(fine_signature(blank), fine_signature(eye(1))) == 0.0
    assert match_score(fine_signature(blank), fine_signature(blank)) == 0.0


def test_coarse_search_finds_the_real_pair():
    sigs = np.stack([coarse_signature(eye(s)) for s in range(6)] + [coarse_signature((eye(3) * 0.8).astype(np.uint8))])
    pairs = candidate_pairs(sigs, top_k=3, minimum=0.7)
    # A shortlist: look-alikes may get through for the high-resolution step to reject,
    # but the real pair must be there, and ranked first.
    best = pairs[pairs[:, 2].argmax()]
    assert (int(best[0]), int(best[1])) == (3, 6)


def test_a_big_dataset_cannot_crowd_out_a_real_pair():
    # The real pair: eye 3 in dataset A, a blurred copy in dataset B. Dataset C holds 5 vectors
    # halfway between them, closer to each of the two than they are to each other.
    real_a = coarse_signature(eye(3))
    real_b = coarse_signature(cv2.GaussianBlur(eye(3), (0, 0), 1.5))
    middle = (real_a + real_b) / np.linalg.norm(real_a + real_b)
    assert real_a @ real_b >= 0.7 and middle @ real_a > real_a @ real_b
    sigs = np.stack([real_a, real_b] + [middle] * 5)
    groups = np.array(["A", "B", "C", "C", "C", "C", "C"])
    found = {(int(i), int(j)) for i, j, _ in candidate_pairs(sigs, top_k=3, minimum=0.7, groups=groups)}
    assert (0, 1) in found
    # Without groups, the copies fill both images' shortlists and the real pair is missed.
    found_flat = {(int(i), int(j)) for i, j, _ in candidate_pairs(sigs, top_k=3, minimum=0.7)}
    assert (0, 1) not in found_flat


def test_only_queries_search_but_pool_copies_are_still_found():
    # Images 0-2: pool (do not search). Image 0 and 1 are a pool-pool copy: skipped on purpose.
    # Image 3: a test image whose copy is pool image 2: must be found.
    a, b = eye(1), eye(5)
    sigs = np.stack([coarse_signature(x) for x in [a, (a * 0.8).astype(np.uint8), b, (b * 0.8).astype(np.uint8)]])
    queries = np.array([False, False, False, True])
    found = {(int(i), int(j)) for i, j, _ in candidate_pairs(sigs, top_k=3, minimum=0.7, queries=queries)}
    assert (2, 3) in found and (0, 1) not in found


def test_big_group_gets_a_longer_list():
    # One query, 8 near-identical look-alikes in a big group: with top_k 2 but top_k_big 8, all are kept.
    sigs = np.stack([coarse_signature(eye(1))] * 9)
    groups = np.array(["q"] + ["big"] * 8)
    queries = np.array([True] + [False] * 8)
    found = candidate_pairs(sigs, top_k=2, minimum=0.7, groups=groups, queries=queries, top_k_big=8, big_group=5)
    assert len(found) == 8


def test_groups_follow_chains_and_keep_others_apart():
    g = same_eye_groups(5, [(0, 1), (1, 2)])
    assert g[0] == g[1] == g[2]
    assert len({g[0], g[3], g[4]}) == 3


def test_held_out_split_always_wins():
    splits = ["train", "test", "val", "test"]
    drops = decide_drops(splits, [(0, 1, 0.9), (2, 3, 0.65)])
    assert drops == {0: "matches_test", 2: "matches_test"}


def test_below_match_is_kept():
    assert decide_drops(["train", "test"], [(0, 1, MATCH - 0.01)]) == {}


def test_train_copies_stay_but_test_copies_go():
    drops = decide_drops(["train", "train", "test", "test"], [(0, 1, 0.99), (2, 3, 0.99)])
    assert drops == {3: "copy_within_test"}


def test_uncertain_same_split_pair_is_not_removed():
    # 0.6 to 0.8 is a mix of real and false matches: worth protecting a held-out split, not worth losing test images.
    assert decide_drops(["test", "test"], [(0, 1, 0.7)]) == {}


def test_pool_image_matching_a_held_out_set_is_dropped():
    # APTOS (pool) vs JSIEC (test): the pool copy goes. Pool vs train stays: both only train.
    drops = decide_drops(["pool", "test", "pool", "train"], [(0, 1, 0.7), (2, 3, 0.95)])
    assert drops == {0: "matches_test"}


def test_copy_shared_by_two_test_sets_is_kept():
    # IDRiD lesion-mask photo vs the same photo in IDRiD grading: both are needed, in different tests.
    drops = decide_drops(["test", "test", "test"], [(0, 1, 0.99), (1, 2, 0.99)], ["idrid_seg", "idrid", "idrid"])
    assert drops == {2: "copy_within_test"}
