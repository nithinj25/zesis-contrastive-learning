"""Unit tests for the label vocabulary and its mapping."""
from data.ontology import Ontology, load


def test_vocabulary_file_loads_and_every_entry_has_labels_and_status():
    data = load()
    for section in ("findings", "diseases"):
        for name, entry in data[section].items():
            assert entry.get("labels"), name
            assert entry.get("status") in {"sourced", "draft", "uncertain", "reviewed"}, name


def test_unsourced_and_unplaced_labels_are_never_mapped():
    data = load()
    o = Ontology(data)
    held = [(d, lab) for d, labs in data["needs_source"].items() for lab in labs]
    held += [(d, lab) for d, labs in data["sourced_unmapped"].items() for lab in labs]
    for key in held:
        assert key not in o.findings and key not in o.diseases, key


def test_mca_means_different_lesions_in_the_two_rfmid_datasets():
    # RFMiD paper: MCA = macroaneurysm. RFMiD 2.0 paper: MCA = microaneurysm.
    o = Ontology(load())
    assert o.map("rfmid", ["MCA"], True) == ([], [])
    assert o.map("rfmid2", ["MCA"], True) == (["microaneurysm"], [])


def test_rfmid2_csc_is_not_central_serous():
    o = Ontology(load())
    assert o.map("rfmid2", ["CSC"], True) == ([], [])
    assert o.map("rfmid2", ["CSR"], True) == ([], ["CSR"])
    assert o.map("rfmid2", ["HTN"], True) == ([], ["hypertensive_retinopathy"])


def test_mapping_examples():
    o = Ontology(load())
    assert o.map("rfmid2", ["HR", "MCA"], True) == (["haemorrhage", "microaneurysm"], [])
    assert o.map("rfmid", ["ODE", "ODP", "AION"], True) == (["disc_edema", "disc_pallor"], ["AION"])
    assert o.map("jsiec", ["Retinitis pigmentosa"], True) == ([], ["RP"])
    assert o.map("jsiec", ["RAO"], True) == ([], [])  # not CRAO without a source
    assert o.map("paraguay", ["paraguay_dr1"], True) == ([], [])  # 'No DR signs'
    assert o.map("paraguay", ["paraguay_dr4"], True) == ([], ["DR"])


def test_rfmid_normal_rule_and_unlabelled_photos():
    o = Ontology(load())
    assert o.map("rfmid", [], True, disease_risk=0) == ([], ["normal"])
    assert o.map("rfmid", [], True, disease_risk=1) == ([], [])  # risk without a named label: not normal
    assert o.map("rfmid2", ["DR"], False) == ([], [])  # label row missing: nothing is guessed


def test_leave_disease_out_targets_are_diseases():
    data = load()
    assert set(data["leave_disease_out"]) <= set(data["diseases"])
