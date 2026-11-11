"""Disease bank (graph/bank.py): prototypes, ranking, add / remove, save / load."""
import numpy as np
import pytest

from graph.bank import Bank


def _bank():
    b = Bank("toy")
    b.add("normal", np.array([[1.0, 0, 0], [0.9, 0.1, 0]]), ["n1", "n2"])
    b.add("DR", np.array([[0, 1.0, 0]]), ["d1"])
    return b


def test_prototype_is_unit_mean_and_ranking_is_by_cosine():
    b = _bank()
    assert np.linalg.norm(b.protos[0]) == pytest.approx(1.0)
    top = b.rank(np.array([0.2, 1.0, 0]))
    assert [d for d, _ in top] == ["DR", "normal"] and top[0][1] > top[1][1]


def test_adding_a_disease_is_one_vector_and_changes_the_answer():
    b = _bank()
    x = np.array([0, 0.1, 1.0])
    assert b.rank(x, 1)[0][0] == "DR"          # nothing better exists yet
    b.add("RP", np.array([[0, 0, 1.0], [0, 0.2, 1.0]]), ["r1", "r2"])
    assert len(b.protos) == 3 and b.rank(x, 1)[0][0] == "RP"
    b.add("RP", np.array([[0, 1.0, 0]]), ["r3"])  # re-adding replaces, never duplicates
    assert b.names.count("RP") == 1 and b.n[b.names.index("RP")] == 1


def test_save_load_round_trip(tmp_path):
    b = _bank()
    p = str(tmp_path / "bank.npz")
    b.save(p)
    c = Bank.load(p)
    assert c.names == b.names and c.n == b.n and c.members == b.members and c.encoder == "toy"
    assert np.allclose(np.stack(c.protos), np.stack(b.protos))
