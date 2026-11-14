# TRACE data card

What data TRACE uses, where it came from, how it was cleaned and split, and what is known to be wrong with it.
Built by `make data` (`data.ingest` → `data.dedupe` → `data.process` → `data.quality` → `data.splits`); details in
[docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md). Numbers are from the run of 2026-11-06. Images are not
redistributed; each dataset must be downloaded from its source under its own terms.

## 1. Datasets

| Dataset | Photos | Role | Source | Licence |
|---|---|---|---|---|
| RFMiD | 3,200 (train 1,920 / val 640 / test 640) | Head labels, val, held-out test | IEEE DataPort | to verify |
| RFMiD 2.0 | 860 (train 509 / val 177 / test 174) | Extra rare labels, same rules as RFMiD | Zenodo [10.5281/zenodo.7505822](https://doi.org/10.5281/zenodo.7505822) (Panchal, Naik, Kokare) | CC BY 4.0 (Zenodo record) |
| APTOS 2019 | 3,662 | Pretraining pool, labels unused | Kaggle | Kaggle competition terms, to verify |
| EyePACS (Kaggle 2015 train, "cropped" version) | 35,108 | Pretraining pool, labels unused | Kaggle | Kaggle competition terms, to verify |
| EyeQ grades (`Label_EyeQ_train.csv`) | 12,543 grades of EyePACS photos | Quality grading | GitHub HzFu/EyeQ | to verify |
| IDRiD (grading, parts B and C) | 516 | Held-out test (DR transfer) | IEEE DataPort | CC BY 4.0 (licence file in the download; Porwal, Pachade, Kokare) |
| IDRiD (segmentation, part A) | 81 | Held-out test (lesion masks) | IEEE DataPort | CC BY 4.0 (same) |
| JSIEC-1000 | 1,000 in 39 classes | Held-out test (rare diseases, OOD) | Zenodo [10.5281/zenodo.3477553](https://doi.org/10.5281/zenodo.3477553) | "other-open" on Zenodo, terms to verify |
| Paraguay DR | 1,437 in 7 grades | Held-out test (unseen site, ZEISS camera) | Zenodo [10.5281/zenodo.4891308](https://doi.org/10.5281/zenodo.4891308) | CC BY 4.0 (Zenodo record) |
| REFUGE, ODIR-5K | not yet downloaded | REFUGE: held-out glaucoma test. ODIR: optional pool | | |

Not used: the EyeQ test file (it grades EyePACS *test* photos, which are not downloaded), EyePACS "resized"
(same photos as "cropped", less detail), IDRiD part C photos (byte-identical to part B; only its disc and fovea
coordinates are used).

## 2. Cleaning

| Dataset | Photos | Dropped by dedupe | Retina not found | Kept |
|---|---|---|---|---|
| RFMiD | 3,200 | 168 | 0 | 3,032 |
| RFMiD 2.0 | 860 | 51 | 0 | 809 |
| APTOS | 3,662 | 275 | 0 | 3,387 |
| EyePACS | 35,108 | 1,327 | 2 | 33,779 |
| IDRiD (grading) | 516 | 8 | 0 | 508 |
| IDRiD (masks) | 81 | 0 | 0 | 81 |
| JSIEC | 1,000 | 7 | 0 | 993 |
| Paraguay | 1,437 | 203 | 0 | 1,234 |
| **Total** | **45,864** | **2,039** | **2** | **43,823** |

**Deduplication.** Photos are compared by their fine vessel pattern (perceptual hashes matched thousands of
different eyes). A photo is dropped if it matches (score ≥ 0.6) a photo in a higher split (test > val >
train/pool), or if it is a same-eye copy (≥ 0.8) inside one dataset's test or val split. After dropping, no kept
photo matches a kept photo of another split. Main findings:
- **84 RFMiD photos are IDRiD photos** (60 exact copies; RFMiD train 60, val 9, test 15, matching 102 IDRiD photos in 103 pairs). Train and val copies are dropped; test copies are kept in both test sets, linked by `dup_group`. Full list: [docs/idrid_in_rfmid.csv](docs/idrid_in_rfmid.csv).
- **Paraguay contains the same eye photographed more than once**: 251 same-eye pairs, 203 copies dropped.
- Cautious by design: most 0.6–0.8 matches are different eyes that look alike, so some photos were dropped
  needlessly (mainly pool photos, which is cheap).

**Retina finding failures:** 2 EyePACS photos (`eyepacs_36416_right`, `eyepacs_41176_left`).

## 3. Splits (`data/meta.parquet`, column `bucket`)

| Bucket | Photos | Contents |
|---|---|---|
| pool (pretraining) | 22,500 | EyePACS 16,918 · APTOS 3,387 · RFMiD train 1,735 · RFMiD 2.0 train 460 |
| head training | 2,193 | RFMiD + RFMiD 2.0 train with a label row |
| val-A / val-B | 377 / 378 | RFMiD + RFMiD 2.0 val, halves balanced by camera and label, same-eye groups kept whole |
| test | 3,626 | RFMiD test 639 · RFMiD 2.0 test 171 · IDRiD 508 + 81 · JSIEC 993 · Paraguay 1,234 |
| ldo_test | 81 | Train/val photos of the leave-disease-out targets (below) |
| spare | 16,861 | EyePACS photos not needed to reach the pool size, and EyePACS rejects |
| excluded | 2,041 | Dedupe drops and retina failures |

**Leave-disease-out (AION, CRVO, RP; provisional):** every photo of these diseases is out of the pool and head
training and forms its test set. AION 30, CRVO 78, RP 36 photos (one CRVO photo was dropped as the same eye as an
IDRiD test photo). Labels were used only to choose what to leave out. Pool-only datasets (APTOS, EyePACS) have no
such labels, so unlabelled cases of these diseases may still be in the pool.

**Pool size and EyePACS selection.** The pool is capped at 22,500 so the MoCo queue (8,192 ≈ 0.4 × pool) and the
compute budget hold. EyePACS fills what the other datasets leave, in this order: EyeQ good (8,136 used), EyeQ
usable (1,829), then tree-graded good (6,953). No reject enters the pool.

## 4. Quality grades

A depth-3 decision tree on simple measurements (sharpness, fine detail, exposure, contrast, dark and bright
pixel share, centre–rim brightness) was fitted to EyeQ's human grades. On held-out EyePACS patients it is right
75–78% of the time and catches 54–71% of rejects (3 seeds). **It is used on EyePACS only.** On other cameras it
called 33% of RFMiD, 38% of RFMiD 2.0 and 61% of Paraguay "reject", and many of those photos are readable,
some with clear lesions or media haze (itself an RFMiD label). Every other dataset is `ungraded` and unfiltered.

## 5. Camera / source IDs (`source`)

| Source | How it is known |
|---|---|
| `topcon_3d_oct2000`, `kowa_vx10a`, `topcon_trc_nw300` | RFMiD image size (2144×1424, 4288×2848, 2048×1536). RFMiD 2.0's 2048×1536 group shares photos with RFMiD's TRC-NW300 |
| `rfmid2_512x512`, `rfmid2_1504x1000` | Size groups; camera not proven |
| `aptos`, `eyepacs`, `idrid`, `paraguay` | One source per dataset |
| `jsiec_<w>x<h>` | 17 size groups; camera not proven |

Per-camera colour statistics for preprocessing step 7 (`data/trace_preprocess.json`) come from training-side
photos only; nothing is computed from held-out data.

## 6. Labels

Dataset labels map to 20 findings and 14 diseases through `data/ontology.yaml`. Each label's meaning is quoted
from the dataset's own paper (RFMiD: Pachade et al., *Data* 2021, 6, 14; RFMiD 2.0: Panchal et al., *Data* 2023,
8, 29); the mapping onto TRACE findings is the project spec's and has not yet been reviewed by a clinician.
What the papers settled:
- **HR** means haemorrhagic retinopathy in both datasets (1 photo in RFMiD, 84 in RFMiD 2.0), so it counts as
  haemorrhage evidence.
- **MCA differs:** RFMiD's MCA is *macroaneurysm* (dilated retinal arteriole, 1 photo), RFMiD 2.0's is
  *microaneurysm*. Only RFMiD 2.0's counts as microaneurysm.
- **RFMiD 2.0's CSC is cysticercosis**, not central serous chorioretinopathy; its HTN column is hypertensive retinopathy.
- 30 further labels have a sourced meaning but no TRACE role yet (`sourced_unmapped`); MNF (RFMiD) and CNV
  (RFMiD 2.0) have no definition in either paper. None of these are used.
- **Camera check:** the RFMiD paper's Table 3 gives each class's count per camera. Our size-based camera IDs match
  it exactly for 43 of 46 classes (the rest differ by 2–8 photos, and one paper row does not add up).

Finding positives in head training: 14 findings have ≥ 20 (trained detectors), 6 have 5–19 (prototypes).
The val-A / val-B membership is frozen in `data/val_split.csv`, so later label fixes cannot reshuffle it.

## 7. Known issues

- **Label conflicts:** same-eye copies within a dataset with different labels are flagged `label_conflict`
  (kept photos: Paraguay 26, RFMiD 7, IDRiD 6). One eye is labelled CRVO in RFMiD and proliferative DR in IDRiD.
- **Not proven patient-level:** only EyePACS has patient IDs. Elsewhere, two photos of one eye can score below the
  dedupe threshold, and false matches can chain different eyes into one same-eye group (largest: 9 Paraguay photos
  with grades 4, 5 and 7).
- **Same-size cameras:** some APTOS photos have the exact sizes of RFMiD's three cameras (28, 52, 351 photos), and
  JSIEC has photos at Paraguay's 2124×2056 size. The pseudo-domain check (CLAUDE.md §5.5) must settle the APTOS case
  before the mining loss uses `source`.
- **8 JSIEC photos (2100×1592)** have a bright line along the bottom edge inside the retina mask.
- **Stretched photos:** 61% of RFMiD 2.0 training photos and 133 APTOS photos are stretched; preprocessing undoes it.

## 8. Does preprocessing remove camera signal? (`scripts/camera_probe.py`)

A logistic regression guesses the dataset from simple photo statistics (image size excluded):

| | Raw | After steps 3–6 | After step 7 |
|---|---|---|---|
| 8 datasets (chance 0.125) | 0.687 | 0.614 | (needs a known camera) |
| 4 training datasets (chance 0.25) | 0.805 | 0.717 | 0.493 |
| Disc pallor AUROC (RFMiD) | 0.615 | 0.634 | |
| Media haze AUROC (RFMiD) | 0.753 | 0.794 | |

Camera signal falls; the two colour- and haze-sensitive findings do not.
