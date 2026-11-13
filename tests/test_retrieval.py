import numpy as np

from eval.retrieval import diagnosis, precision_at_k, search
from trust.ood import KnnOOD


def test_diagnosis_labels():
    assert diagnosis(["DR"]) == "DR"
    assert diagnosis(["DR", "ARMD"]) == "several"
    assert diagnosis([]) == "findings only"


def test_precision_counts_multi_label_cases():
    lib = [["DR"], ["DR", "ARMD"], ["normal"], ["ARMD"]]
    p = precision_at_k(np.array(["DR", "ARMD"]), lib, np.array([[0, 1, 2, 3], [0, 1, 2, 3]]))
    assert p.tolist() == [0.5, 0.5]


def test_search_skips_the_query_eye():
    lib = np.eye(4)[[0, 0, 1, 2]].astype(np.float32)  # rows 0 and 1 are the same eye (block 7)
    index = KnnOOD().fit(lib)
    i, s = search(index, lib[:1], q_block=np.array([7]), lib_block=np.array([7, 7, 8, 9]), k=2)
    assert 0 not in i[0] and 1 not in i[0]
