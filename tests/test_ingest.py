"""Unit tests for ingest, on a tiny fake dataset with the same quirks as the real files."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from data.ingest import (
    SplitSpec,
    ingest,
    ingest_aptos,
    ingest_eyepacs,
    ingest_idrid,
    ingest_jsiec,
    ingest_split,
    jsiec_class,
    jsiec_code,
    read_labels,
    source_id,
)


def make_split(root, name, ids, header, rows, size=(2144, 1424), bom=False):
    folder = root / name
    folder.mkdir(parents=True)
    for i in ids:
        Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(folder / f"{i}.png")
    text = header + "\n" + "\n".join(rows) + "\n"
    (folder / "labels.csv").write_bytes(("﻿" if bom else "").encode() + text.encode("utf-8"))
    return SplitSpec("rfmid", name, name, f"{name}/labels.csv")


def test_read_labels_cleans_bom_spaces_and_empty_columns(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes("﻿ID,DR,ON ,\n1,1,0,\n".encode("utf-8"))
    d = read_labels(str(path))
    assert list(d.columns) == ["ID", "DR", "ON"]


def test_read_labels_survives_non_utf8_bytes(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"ID,DR,ON\xa0\n1,1,0\n")  # as in the real RFMiD 2.0 headers
    assert list(read_labels(str(path)).columns) == ["ID", "DR", "ON"]


def test_labels_and_risk_are_read(tmp_path):
    spec = make_split(tmp_path, "train", [1, 2], "ID,Disease_Risk,DR,AION", ["1,1,1,0", "2,0,0,0"], bom=True)
    t, orphans = ingest_split(str(tmp_path), spec)
    one, two = t[t.image_id == "rfmid_train_1"].iloc[0], t[t.image_id == "rfmid_train_2"].iloc[0]
    assert one.labels == ["DR"] and one.disease_risk == 1 and one.label_ok
    assert two.labels == [] and two.disease_risk == 0 and two.label_ok  # labelled, with nothing found
    assert orphans == []


def test_mismatched_ids_are_flagged_not_guessed(tmp_path):
    # Real case: RFMiD 2.0 has image 251 but a label row for 151.
    spec = make_split(tmp_path, "test", [1, 251], "ID,DR", ["1,1", "151,1"])
    t, orphans = ingest_split(str(tmp_path), spec)
    bad = t[t.image_id == "rfmid_test_251"].iloc[0]
    assert not bad.label_ok and bad.labels == []
    assert orphans == [151]


def test_ids_restarting_per_split_stay_unique(tmp_path):
    a = make_split(tmp_path, "train", [1], "ID,DR", ["1,1"])
    b = make_split(tmp_path, "val", [1], "ID,DR", ["1,0"])
    meta, _ = ingest(str(tmp_path), [a, b])
    assert meta.image_id.is_unique and len(meta) == 2


@pytest.mark.parametrize(
    "dataset,size,expected",
    [
        ("rfmid", (2144, 1424), "topcon_3d_oct2000"),
        ("rfmid", (4288, 2848), "kowa_vx10a"),
        ("rfmid", (2048, 1536), "topcon_trc_nw300"),
        ("rfmid", (999, 999), "rfmid_999x999"),
        ("rfmid2", (2048, 1536), "topcon_trc_nw300"),  # shares duplicate photos with RFMiD's NW300
        ("rfmid2", (1504, 1000), "rfmid2_1504x1000"),  # camera not proven
        ("rfmid2", (512, 512), "rfmid2_512x512"),  # camera not proven
        ("aptos", (4288, 2848), "aptos"),  # one source (section 5.5), even at an RFMiD camera's size
        ("jsiec", (3046, 2572), "jsiec_3046x2572"),
    ],
)
def test_source_id_never_guesses_a_camera(dataset, size, expected):
    assert source_id(dataset, *size) == expected


def save(path, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((size[1], size[0], 3), np.uint8)).save(path)


@pytest.mark.parametrize(
    "folder,name,code",
    [("15.0.Retinitis pigmentosa", "Retinitis pigmentosa", "15_0"), ("3.RAO", "RAO", "3"), ("29.1.Blur fundus with suspected PDR", "Blur fundus with suspected PDR", "29_1")],
)
def test_jsiec_folder_names(folder, name, code):
    assert jsiec_class(folder) == name and jsiec_code(folder) == code


def test_jsiec_same_file_name_in_two_classes_stays_unique(tmp_path):
    save(tmp_path / "jsiec/1000images/2.1.CRVO/a.JPG")
    save(tmp_path / "jsiec/1000images/3.RAO/a.jpg")
    save(tmp_path / "jsiec/1000images/3.RAO/b.TIF")
    (tmp_path / "jsiec/1000images/3.RAO/notes.txt").write_text("x")
    t = ingest_jsiec(str(tmp_path))
    assert sorted(t.image_id) == ["jsiec_2_1_a", "jsiec_3_a", "jsiec_3_b"]
    assert set(t.split) == {"test"} and t.label_ok.all()
    assert t.set_index("image_id").labels["jsiec_2_1_a"] == ["CRVO"]


def test_aptos_goes_to_the_pool_with_its_grade_kept(tmp_path):
    (tmp_path / "aptos").mkdir()
    for csv, rows in [("train_1.csv", "x1,2\n"), ("valid.csv", "x2,0\n"), ("test.csv", "")]:
        (tmp_path / "aptos" / csv).write_text("id_code,diagnosis\n" + rows + "\n\n")  # real files end in blank lines
    save(tmp_path / "aptos/train_images/train_images/x1.png")
    save(tmp_path / "aptos/val_images/val_images/x2.png")
    save(tmp_path / "aptos/test_images/test_images/x3.png")  # no grade row
    t = ingest_aptos(str(tmp_path)).set_index("image_id")
    assert set(t.split) == {"pool"} and set(t.source) == {"aptos"}
    assert t.labels["aptos_x1"] == ["aptos_dr2"] and t.labels["aptos_x2"] == ["aptos_dr0"]
    assert not t.label_ok["aptos_x3"] and t.aptos_split["aptos_x3"] == "test"


def test_missing_dataset_folder_is_skipped(tmp_path):
    spec = make_split(tmp_path, "train", [1], "ID,DR", ["1,1"])
    meta, _ = ingest(str(tmp_path), [spec, SplitSpec("rfmid", "val", "nowhere", "nowhere.csv")])
    assert len(meta) == 1  # no JSIEC or APTOS folder either, and no crash


def test_eyepacs_keeps_patient_and_eye(tmp_path):
    save(tmp_path / "eyepacs/resized_train_cropped/resized_train_cropped/10003_left.jpeg")
    save(tmp_path / "eyepacs/resized_train_cropped/resized_train_cropped/10003_right.jpeg")
    (tmp_path / "eyepacs/trainLabels_cropped.csv").write_text(",Unnamed: 0,image,level\n0,0,10003_left,2\n")
    (tmp_path / "eyeq").mkdir()
    (tmp_path / "eyeq/Label_EyeQ_train.csv").write_text(",image,quality,DR_grade\n0,10003_left.jpeg,2,2\n")
    t = ingest_eyepacs(str(tmp_path)).set_index("image_id")
    assert t.eyeq_quality["eyepacs_10003_left"] == "reject" and t.eyeq_quality["eyepacs_10003_right"] is None
    assert set(t.patient) == {"eyepacs_10003"} and set(t.split) == {"pool"}
    assert t.labels["eyepacs_10003_left"] == ["eyepacs_dr2"] and not t.label_ok["eyepacs_10003_right"]
    assert t.laterality["eyepacs_10003_right"] == "right"


def test_idrid_names_restart_per_set_and_masks_become_labels(tmp_path):
    r = tmp_path / "idrid"
    g = r / "B. Disease Grading"
    loc = r / "C. Localization/2. Groundtruths"
    for split, folder, word in [("train", "a. Training Set", "Training"), ("test", "b. Testing Set", "Testing")]:
        save(g / "1. Original Images" / folder / "IDRiD_001.jpg")
        (g / "2. Groundtruths").mkdir(parents=True, exist_ok=True)
        (g / "2. Groundtruths" / f"{folder[0]}. IDRiD_Disease Grading_{word} Labels.csv").write_text(
            "Image name,Retinopathy grade,Risk of macular edema ,,\nIDRiD_001,3,2,,\n")
        for sub in ["1. Optic Disc Center Location", "2. Fovea Center Location"]:
            (loc / sub).mkdir(parents=True, exist_ok=True)
            (loc / sub / f"IDRiD_{word}_Markups.csv").write_text("Image No,X- Coordinate,Y - Coordinate,,\nIDRiD_001,10,20,,\n,,,,\n")
    seg = r / "A. Segmentation"
    save(seg / "1. Original Images/a. Training Set/IDRiD_01.jpg")
    save(seg / "2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/IDRiD_01_MA.tif")
    save(seg / "2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/IDRiD_01_OD.tif")
    t = ingest_idrid(str(tmp_path)).set_index("image_id")
    assert sorted(t.index) == ["idrid_seg_01", "idrid_test_001", "idrid_train_001"]
    assert set(t.split) == {"test"}
    assert t.labels["idrid_test_001"] == ["idrid_dr3", "idrid_dme2"]
    assert (t.od_x["idrid_train_001"], t.fovea_y["idrid_train_001"]) == (10, 20)
    assert t.labels["idrid_seg_01"] == ["idrid_MA", "idrid_OD"]  # no HE, EX, SE mask files: absent
    assert set(t.source) == {"idrid"}


def test_paraguay_labels_come_from_folders_and_disagreements_are_flagged(tmp_path):
    from data.ingest import ingest_paraguay

    save(tmp_path / "paraguay/1. No DR signs/1.jpg")
    save(tmp_path / "paraguay/3. Moderate NPDR/2.jpg")
    save(tmp_path / "paraguay/3. Moderate NPDR/3.jpg")
    sheet = pd.DataFrame({"Image": [1, 2, 3], "Format": "jpg", "Status": ["No DR signs", "Moderate NPDR", "Severe NPDR"]})
    sheet.to_excel(tmp_path / "paraguay/Annotations of the classifications.xlsx", index=False)
    t = ingest_paraguay(str(tmp_path)).set_index("image_id")
    assert t.labels["paraguay_1"] == ["paraguay_dr1"] and t.labels["paraguay_2"] == ["paraguay_dr3"]
    assert not t.label_ok["paraguay_3"] and list(t.labels["paraguay_3"]) == []  # folder and sheet disagree
    assert set(t.split) == {"test"} and set(t.source) == {"paraguay"}
