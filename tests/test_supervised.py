import numpy as np

from encoder.supervised import label_matrix, lr_at, macro_auroc, pos_weight


def test_pos_weight_matches_section_8_2():
    Y = np.zeros((1920, 1), np.float32)
    Y[:40] = 1
    assert pos_weight(Y)[0] == 47.0  # 1,880 / 40


def test_pos_weight_is_capped():
    Y = np.zeros((1735, 1), np.float32)
    Y[:5] = 1
    assert pos_weight(Y)[0] == 50.0  # uncapped it would be 346


def test_label_matrix():
    Y = label_matrix([["DR", "MH"], [], ["ODC"]], ["DR", "MH", "ODC"])
    assert Y.tolist() == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]


def test_macro_auroc_skips_columns_without_both_classes():
    Y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]], np.float32)
    S = np.array([[0.9, 0.1], [0.2, 0.5], [0.8, 0.3], [0.1, 0.2]])
    assert macro_auroc(Y, S) == 1.0  # column 2 has no positives and is skipped


def test_lr_schedule():
    assert lr_at(0, 100, 10) == 0.1
    assert lr_at(9, 100, 10) == 1.0
    assert abs(lr_at(10, 100, 10) - 1.0) < 1e-9
    assert lr_at(100, 100, 10) == 0.0
