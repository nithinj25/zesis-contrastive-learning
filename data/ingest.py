"""Read each dataset's files into one table (CLAUDE.md section 6 step 1, section 5.5).

Run from the repo root:  python -m data.ingest
Writes data/raw_meta.parquet, one row per image. So far: RFMiD, RFMiD 2.0, JSIEC, APTOS, IDRiD, EyePACS.
A dataset whose folder is missing is skipped with a message.

Splits: train / val / test as the dataset defines them; "pool" = pretraining only, never
used for head training or calibration (APTOS). JSIEC is held out, so all of it is "test".
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import pandas as pd
from PIL import Image

# RFMiD camera by image size (section 5.5).
RFMID_CAMERAS = {
    (2144, 1424): "topcon_3d_oct2000",
    (4288, 2848): "kowa_vx10a",
    (2048, 1536): "topcon_trc_nw300",
}
NOT_DISEASE_LABELS = {"ID", "Disease_Risk"}


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    split: str
    image_dir: str  # relative to the raw root
    label_csv: str  # relative to the raw root


SPECS = [
    SplitSpec("rfmid", "train", "rfmid/Training_Set/Training_Set/Training", "rfmid/Training_Set/Training_Set/RFMiD_Training_Labels.csv"),
    SplitSpec("rfmid", "val", "rfmid/Evaluation_Set/Evaluation_Set/Validation", "rfmid/Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"),
    SplitSpec("rfmid", "test", "rfmid/Test_Set/Test_Set/Test", "rfmid/Test_Set/Test_Set/RFMiD_Testing_Labels.csv"),
    SplitSpec("rfmid2", "train", "rfmid2/Training_set", "rfmid2/Training_set/RFMiD_2_Training_labels.csv"),
    SplitSpec("rfmid2", "val", "rfmid2/Validation_set", "rfmid2/Validation_set/RFMiD_2_Validation_labels.csv"),
    SplitSpec("rfmid2", "test", "rfmid2/Test_set", "rfmid2/Test_set/RFMiD_2_Testing_labels.csv"),
]


def read_labels(path: str) -> pd.DataFrame:
    """Read a label CSV, cleaning a leading byte-order mark, stray spaces and empty columns."""
    d = pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")
    d.columns = [c.replace("�", " ").replace("\xa0", " ").strip() for c in d.columns]
    d = d[[c for c in d.columns if not c.startswith("Unnamed")]]
    d = d.dropna(subset=["ID"]).copy()
    d["ID"] = d["ID"].astype(int)
    return d


def source_id(dataset: str, width: int, height: int) -> str:
    """Camera where the size proves it, otherwise a size group, never a guessed camera."""
    if dataset == "rfmid" and (width, height) in RFMID_CAMERAS:
        return RFMID_CAMERAS[(width, height)]
    # Shown by the data: RFMiD 2.0's 2048x1536 group contains 19 of the same eyes as RFMiD's TRC-NW300.
    if dataset == "rfmid2" and (width, height) == (2048, 1536):
        return "topcon_trc_nw300"
    # Section 5.5: APTOS is one source. Its sizes overlap RFMiD cameras; the pseudo-domain step decides that.
    # One source per dataset until the pseudo-domain step (section 5.5): size groups would only reflect resizing.
    if dataset in ("aptos", "eyepacs", "idrid", "idrid_seg", "paraguay"):
        return "idrid" if dataset == "idrid_seg" else dataset
    return f"{dataset}_{width}x{height}"


def image_row(raw_root: str, path: str, dataset: str, split: str, image_id: str) -> dict:
    """The columns every dataset shares. Labels start empty and label_ok False."""
    with Image.open(path) as im:
        width, height = im.size
    return {
        "image_id": image_id,
        "path": os.path.relpath(path, raw_root).replace(os.sep, "/"),
        "dataset": dataset,
        "split": split,
        "width": width,
        "height": height,
        "size_group": f"{width}x{height}",
        "source": source_id(dataset, width, height),
        "label_ok": False,
        "labels": [],
        "disease_risk": None,
        "patient": None,  # only where the dataset gives it (EyePACS)
    }


def ingest_split(raw_root: str, spec: SplitSpec) -> tuple[pd.DataFrame, list[int]]:
    """Returns one row per image, plus label IDs that have no image.

    An image whose ID has no label row gets label_ok = False and no labels. Matching such
    images to orphan labels by guesswork could attach the wrong disease to an eye.
    """
    labels = read_labels(os.path.join(raw_root, spec.label_csv))
    disease_cols = [c for c in labels.columns if c not in NOT_DISEASE_LABELS]
    by_id = labels.set_index("ID")
    files = sorted(
        f for ext in ("png", "jpg") for f in glob.glob(os.path.join(raw_root, spec.image_dir, f"*.{ext}"))
    )
    rows, seen = [], set()
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        # IDs restart in every split, so the ID alone is not unique.
        row = image_row(raw_root, path, spec.dataset, spec.split, f"{spec.dataset}_{spec.split}_{stem}")
        if stem.isdigit() and int(stem) in by_id.index:
            seen.add(int(stem))
            entry = by_id.loc[int(stem)]
            row["label_ok"] = True
            row["labels"] = [c for c in disease_cols if entry[c] == 1]
            if "Disease_Risk" in by_id.columns:
                row["disease_risk"] = int(entry["Disease_Risk"])
        rows.append(row)
    return pd.DataFrame(rows), sorted(set(by_id.index) - seen)


JSIEC_DIR = "jsiec/1000images"
APTOS_SPLITS = {"train": "train_1.csv", "val": "valid.csv", "test": "test.csv"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


JSIEC_CODE = re.compile(r"^(\d+(?:\.\d+)?)\.")


def jsiec_class(folder: str) -> str:
    """'15.0.Retinitis pigmentosa' -> 'Retinitis pigmentosa'; '3.RAO' -> 'RAO'."""
    return JSIEC_CODE.sub("", folder).strip()


def jsiec_code(folder: str) -> str:
    """'15.0.Retinitis pigmentosa' -> '15_0'; '3.RAO' -> '3'."""
    return JSIEC_CODE.match(folder).group(1).replace(".", "_")


def ingest_jsiec(raw_root: str) -> pd.DataFrame:
    """JSIEC-1000: one folder per class, one class per image. Held out, so split = test."""
    rows = []
    for folder in sorted(os.listdir(os.path.join(raw_root, JSIEC_DIR))):
        for name in sorted(os.listdir(os.path.join(raw_root, JSIEC_DIR, folder))):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(name)[0]
            # File names are not guaranteed unique across class folders, so the folder's number is kept.
            path = os.path.join(raw_root, JSIEC_DIR, folder, name)
            row = image_row(raw_root, path, "jsiec", "test", f"jsiec_{jsiec_code(folder)}_{stem}")
            row["label_ok"], row["labels"] = True, [jsiec_class(folder)]
            rows.append(row)
    return pd.DataFrame(rows)


def ingest_aptos(raw_root: str) -> pd.DataFrame:
    """APTOS 2019: pretraining pool only (section 5.3), so its own train/val/test all become "pool".

    The DR grade is kept as a label (aptos_dr0 to aptos_dr4) for monitoring; it never trains anything.
    """
    rows = []
    for split, csv in APTOS_SPLITS.items():
        grades = pd.read_csv(os.path.join(raw_root, "aptos", csv)).dropna().set_index("id_code").diagnosis
        for path in sorted(glob.glob(os.path.join(raw_root, "aptos", f"{split}_images", f"{split}_images", "*.png"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            row = image_row(raw_root, path, "aptos", "pool", f"aptos_{stem}")
            row["aptos_split"] = split
            if stem in grades.index:
                row["label_ok"], row["labels"] = True, [f"aptos_dr{int(grades[stem])}"]
            rows.append(row)
    return pd.DataFrame(rows)


IDRID = "idrid"
IDRID_GRADES = "B. Disease Grading"
IDRID_SEG = "A. Segmentation"
IDRID_LOC = "C. Localization/2. Groundtruths"
IDRID_SETS = {"train": "a. Training Set", "test": "b. Testing Set"}
IDRID_LESIONS = {"1. Microaneurysms": "MA", "2. Haemorrhages": "HE", "3. Hard Exudates": "EX", "4. Soft Exudates": "SE", "5. Optic Disc": "OD"}


def _clean_csv(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = [c.strip() for c in d.columns]
    return d.loc[:, ~d.columns.str.startswith("Unnamed")].dropna(how="all")


def ingest_idrid(raw_root: str) -> pd.DataFrame:
    """IDRiD, all held out (split = test). Its own train/test is kept in idrid_split.

    "idrid": the 516 grading photos, labels idrid_dr0-4 and idrid_dme0-2, plus disc and fovea centres
    from part C. Part C's photos are byte-identical copies of part B's, so they are not read twice.
    "idrid_seg": the 81 photos with lesion masks; labels name the masks present (idrid_MA, ...).
    Both are one camera, so one source ("idrid").
    """
    root = os.path.join(raw_root, IDRID)
    rows = []
    for split, folder in IDRID_SETS.items():
        csv = glob.glob(os.path.join(root, IDRID_GRADES, "2. Groundtruths", f"{folder[0]}. *Labels.csv"))[0]
        grades = _clean_csv(csv).set_index("Image name")
        centres = {}
        for kind, sub in [("od", "1. Optic Disc Center Location"), ("fovea", "2. Fovea Center Location")]:
            word = "Training" if split == "train" else "Testing"
            d = _clean_csv(glob.glob(os.path.join(root, IDRID_LOC, sub, f"*{word}*.csv"))[0]).dropna(subset=["Image No"])
            centres[kind] = d.set_index("Image No").iloc[:, :2]
        for path in sorted(glob.glob(os.path.join(root, IDRID_GRADES, "1. Original Images", folder, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            # Names restart in each set (IDRiD_001 is in both), so the set is part of the ID.
            row = image_row(raw_root, path, "idrid", "test", f"idrid_{split}_{stem[6:]}")
            row["idrid_split"] = split
            if stem in grades.index:
                g = grades.loc[stem]
                row["label_ok"] = True
                row["labels"] = [f"idrid_dr{int(g.iloc[0])}", f"idrid_dme{int(g.iloc[1])}"]
            for kind, table in centres.items():
                xy = table.loc[stem].tolist() if stem in table.index else [None, None]
                row[f"{kind}_x"], row[f"{kind}_y"] = xy
            rows.append(row)
        mask_root = os.path.join(root, IDRID_SEG, "2. All Segmentation Groundtruths", folder)
        for path in sorted(glob.glob(os.path.join(root, IDRID_SEG, "1. Original Images", folder, "*.jpg"))):
            stem = os.path.splitext(os.path.basename(path))[0]
            row = image_row(raw_root, path, "idrid_seg", "test", f"idrid_seg_{stem[6:]}")
            row["idrid_split"] = split
            row["label_ok"] = True  # a missing mask file means that lesion type is absent
            row["labels"] = [f"idrid_{code}" for sub, code in IDRID_LESIONS.items()
                             if glob.glob(os.path.join(mask_root, sub, f"{stem}_*.tif"))]
            rows.append(row)
    return pd.DataFrame(rows)


# The "cropped" version: made from the original photos, with ~50% more fine detail than "resized"
# at the same retina size (59 of 60 sampled). Never ingest both: they are the same eyes.
EYEPACS_DIR = "eyepacs/resized_train_cropped/resized_train_cropped"
EYEPACS_LABELS = "eyepacs/trainLabels_cropped.csv"


def ingest_eyepacs(raw_root: str) -> pd.DataFrame:
    """EyePACS (Kaggle 2015 train): pretraining pool only. DR grade kept as eyepacs_dr0-4, monitoring only.

    File names are <patient>_<left|right>, so this is the one dataset with real patient IDs.
    """
    grades = _clean_csv(os.path.join(raw_root, EYEPACS_LABELS)).set_index("image").level
    quality = eyeq_quality(raw_root)
    rows = []
    for name in sorted(os.listdir(os.path.join(raw_root, EYEPACS_DIR))):
        if not name.lower().endswith(IMAGE_EXTS):
            continue
        stem = os.path.splitext(name)[0]
        row = image_row(raw_root, os.path.join(raw_root, EYEPACS_DIR, name), "eyepacs", "pool", f"eyepacs_{stem}")
        row["patient"] = "eyepacs_" + stem.split("_")[0]
        row["laterality"] = stem.split("_")[-1]
        if stem in grades.index:
            row["label_ok"], row["labels"] = True, [f"eyepacs_dr{int(grades[stem])}"]
        row["eyeq_quality"] = quality.get(stem)
        rows.append(row)
    return pd.DataFrame(rows)


# EyeQ quality grades for EyePACS photos. 0/1/2 = good/usable/reject: checked by eye on a contact sheet.
# Only the train file matches our photos; the test file grades EyePACS test photos, which we do not have.
EYEQ_FILE = "eyeq/Label_EyeQ_train.csv"
EYEQ_NAMES = {0: "good", 1: "usable", 2: "reject"}


def eyeq_quality(raw_root: str) -> dict[str, str]:
    """EyePACS stem -> 'good' / 'usable' / 'reject'. Empty if EyeQ is not downloaded."""
    path = os.path.join(raw_root, EYEQ_FILE)
    if not os.path.exists(path):
        print(f"no EyeQ grades: {EYEQ_FILE} not found")
        return {}
    d = _clean_csv(path)
    return {os.path.splitext(i)[0]: EYEQ_NAMES[int(q)] for i, q in zip(d.image, d.quality)}


PARAGUAY = "paraguay"
PARAGUAY_SHEET = "paraguay/Annotations of the classifications.xlsx"


def ingest_paraguay(raw_root: str) -> pd.DataFrame:
    """Paraguay DR: one folder per grade ('3. Moderate NPDR'), images numbered 1-1437. Held out (test).

    Labels paraguay_dr1 to paraguay_dr7 from the folder number. If the spreadsheet's Status disagrees
    with the folder, label_ok is False: we do not pick one.
    """
    sheet = {}
    if os.path.exists(os.path.join(raw_root, PARAGUAY_SHEET)):
        d = pd.read_excel(os.path.join(raw_root, PARAGUAY_SHEET))
        sheet = {str(int(i)): str(s).strip().lower() for i, s in zip(d["Image"], d["Status"])}
    rows = []
    for folder in sorted(os.listdir(os.path.join(raw_root, PARAGUAY))):
        full = os.path.join(raw_root, PARAGUAY, folder)
        if not os.path.isdir(full):
            continue
        number, name = folder.split(".", 1)
        for f in sorted(os.listdir(full)):
            if not f.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(f)[0]
            row = image_row(raw_root, os.path.join(full, f), "paraguay", "test", f"paraguay_{stem}")
            agrees = stem not in sheet or sheet[stem] == name.strip().lower()
            row["label_ok"], row["labels"] = agrees, [f"paraguay_dr{int(number)}"] if agrees else []
            rows.append(row)
    return pd.DataFrame(rows)


def ingest(raw_root: str, specs: list[SplitSpec] = SPECS) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    tables, orphans = [], {}
    for spec in specs:
        if not os.path.isdir(os.path.join(raw_root, spec.image_dir)):
            print(f"skipped {spec.dataset}/{spec.split}: {spec.image_dir} not found")
            continue
        table, missing = ingest_split(raw_root, spec)
        tables.append(table)
        if missing:
            orphans[f"{spec.dataset}/{spec.split}"] = missing
    readers = [
        ("jsiec", JSIEC_DIR, ingest_jsiec),
        ("aptos", "aptos/train_images", ingest_aptos),
        ("idrid", f"{IDRID}/{IDRID_GRADES}", ingest_idrid),
        ("eyepacs", EYEPACS_DIR, ingest_eyepacs),
        ("paraguay", f"{PARAGUAY}/1. No DR signs", ingest_paraguay),
    ]
    for name, folder, reader in readers:
        if os.path.isdir(os.path.join(raw_root, folder)):
            tables.append(reader(raw_root))
        else:
            print(f"skipped {name}: {folder} not found")
    return pd.concat(tables, ignore_index=True), orphans


if __name__ == "__main__":
    meta, orphan_labels = ingest("raw")
    meta.to_parquet("data/raw_meta.parquet")
    print(meta.groupby(["dataset", "split"]).size().to_string())
    print("\nsources:\n" + meta.groupby(["dataset", "source"]).size().to_string())
    unlabelled = meta[~meta.label_ok]
    print(f"\nimages with no label row: {len(unlabelled)}: {unlabelled.image_id.tolist()}")
    print(f"label rows with no image: {orphan_labels}")
