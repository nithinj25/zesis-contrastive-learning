"""Unit tests for split building, on a small synthetic table that exercises every rule."""
import numpy as np
import pandas as pd

from data.ontology import Ontology, load
from data.splits import build, preprocess_stats

CFG = {"pool_target": 8, "pool_grades": ["good", "usable", "ungraded"],
       "eyepacs_priority": ["eyeq:good", "eyeq:usable", "tree:good", "tree:usable"],
       "val_b_fraction": 0.5, "seed": 0, "colour_alpha": 0.7, "clahe_clip": 1.5,
       "illumination_strength": 0.6, "illumination_sigma_fraction": 0.125}


def make(rows):
    """rows: (image_id, dataset, split, source, labels, grade, dup_group, dup_drop, fov_ok)."""
    raw = pd.DataFrame([dict(image_id=r[0], dataset=r[1], split=r[2], source=r[3], labels=r[4], label_ok=True,
                             disease_risk=1 if r[4] else 0, path="x") for r in rows])
    dedupe = pd.DataFrame([dict(image_id=r[0], dup_group=r[6], dup_drop=r[7], dup_reason="matches_test" if r[7] else "") for r in rows])
    proc = pd.DataFrame([dict(image_id=r[0], fov_ok=r[8], fail_reason="" if r[8] else "fov_fail", processed_path="p",
                              fov_fit=1.0, fov_ratio=1.0, lab_n=100, lab_L_mean=100.0, lab_a_mean=140.0, lab_b_mean=150.0,
                              lab_L_std=10.0, lab_a_std=3.0, lab_b_std=4.0) for r in rows])
    quality = pd.DataFrame([dict(image_id=r[0], q_grade=r[5], q_source="tree") for r in rows])
    return raw, dedupe, proc, quality


def base_rows():
    rows = [
        ("t1", "rfmid", "test", "kowa", ["DR"], "good", 0, False, True),
        ("d1", "rfmid", "train", "kowa", ["DR"], "good", 1, True, True),       # dedupe drop
        ("f1", "rfmid", "train", "kowa", ["DR"], "reject", 2, False, False),   # retina not found
        ("a1", "rfmid", "train", "kowa", ["AION", "ODE"], "good", 3, False, True),  # leave-disease-out target
        ("a2", "rfmid", "train", "kowa", [], "good", 3, False, True),          # same eye as a1, no label
        ("r1", "rfmid", "train", "kowa", ["ODE"], "good", 4, False, True),
        ("r2", "rfmid", "train", "kowa", ["DR"], "reject", 5, False, True),    # reject: heads yes, pool no
        ("ap1", "aptos", "pool", "aptos", [], "good", 6, False, True),
        ("ap2", "aptos", "pool", "aptos", [], "reject", 7, False, True),
    ]
    rows += [(f"e{i}", "eyepacs", "pool", "eyepacs", [], "good" if i < 3 else "usable", 10 + i, False, True) for i in range(10)]
    # val: two cameras, 12 photos each, one same-eye pair
    for s in ("kowa", "topcon"):
        for i in range(12):
            g = 100 if (s == "kowa" and i < 2) else 200 + i + (50 if s == "topcon" else 0)
            rows.append((f"v{s}{i}", "rfmid", "val", s, ["DR"] if i % 3 == 0 else [], "good", g, False, True))
    return rows


def test_buckets_follow_every_rule():
    meta = build(*make(base_rows()), Ontology(load()), CFG).set_index("image_id")
    assert meta.bucket["t1"] == "test"
    assert meta.bucket["d1"] == "excluded" and meta.exclude_reason["d1"].startswith("dedupe")
    assert meta.bucket["f1"] == "excluded" and meta.exclude_reason["f1"] == "fov_fail"
    assert meta.bucket["a1"] == "ldo_test" and meta.ldo_disease["a1"] == "AION"
    assert meta.bucket["a2"] == "excluded" and meta.exclude_reason["a2"] == "same_eye_as_ldo"
    assert meta.in_heads["r2"] and not meta.in_pool["r2"]
    assert not meta.in_pool["ap2"] and meta.bucket["ap2"] == "spare"
    assert not meta.in_pool[["a1", "t1", "d1"]].any() and not meta.in_heads[["a1", "t1"]].any()


def test_same_eye_with_different_labels_is_flagged():
    rows = base_rows() + [("p1", "paraguay", "test", "paraguay", ["paraguay_dr3"], "good", 500, False, True),
                          ("p2", "paraguay", "test", "paraguay", ["paraguay_dr4"], "good", 500, True, True),
                          ("p3", "paraguay", "test", "paraguay", ["paraguay_dr1"], "good", 501, False, True)]
    meta = build(*make(rows), Ontology(load()), CFG).set_index("image_id")
    assert meta.label_conflict["p1"] and meta.label_conflict["p2"] and not meta.label_conflict["p3"]
    assert meta.label_conflict["vkowa0"]  # the base table's same-eye val pair: DR vs no labels, a real conflict


def test_pool_hits_its_target_with_eyepacs_best_grade_first():
    meta = build(*make(base_rows()), Ontology(load()), CFG).set_index("image_id")
    assert meta.in_pool.sum() == CFG["pool_target"]
    eye = meta[meta.dataset == "eyepacs"]
    # pool already holds r1 and ap1 (2 photos), so EyePACS fills 6: all 3 good ones, then 3 usable
    assert eye[eye.q_grade == "good"].in_pool.all() and eye.in_pool.sum() == 6


def test_val_halves_keep_groups_whole_and_balance_cameras():
    meta = build(*make(base_rows()), Ontology(load()), CFG).set_index("image_id")
    val = meta[meta.bucket.isin(["val_a", "val_b"])]
    assert val.bucket["vkowa0"] == val.bucket["vkowa1"]  # same eye stays together
    per = val.groupby(["source", "bucket"]).size().unstack()
    assert (abs(per.val_a - per.val_b) <= 2).all()
    dr = val[[("DR" in d) for d in val.diseases]].bucket.value_counts()
    assert abs(dr.get("val_a", 0) - dr.get("val_b", 0)) <= 2


def test_frozen_val_split_is_kept():
    base = build(*make(base_rows()), Ontology(load()), CFG).set_index("image_id")
    val = base[base.bucket.isin(["val_a", "val_b"])].bucket
    flipped = val.map({"val_a": "val_b", "val_b": "val_a"})
    again = build(*make(base_rows()), Ontology(load()), {**CFG, "frozen_val": flipped}).set_index("image_id")
    assert (again.bucket[val.index] == flipped).all()


def test_camera_stats_never_use_held_out_photos():
    rows = base_rows() + [("j1", "jsiec", "test", "jsiec_x", ["CRVO"], "good", 999, False, True)]
    raw, dedupe, proc, quality = make(rows)
    meta = build(raw, dedupe, proc, quality, Ontology(load()), CFG)
    stats = preprocess_stats(meta, proc, CFG)
    assert "jsiec_x" not in stats["cameras"] and set(stats["cameras"]) == {"kowa", "topcon", "aptos", "eyepacs"}
    assert np.allclose(stats["pool"]["mean"], [100, 140, 150]) and stats["pool"]["n_photos"] == CFG["pool_target"]
