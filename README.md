# TRACE: Trustworthy Retinal Contrastive Encoder

> **TRACE learns from less, ignores the camera, and knows when it doesn't know.**

TRACE is a foundation model for **retinal fundus photos**, the colour photos of the back of the eye that
eye doctors use to find diabetic retinopathy, glaucoma, vein blockages and many rarer diseases. It is our entry
for the **ZEISS India Campus Hackathon "Beyond Code 2026", Theme 2: Contrastive Learning for Rare Eye Diseases**.

**Status (8 Nov 2026):** the data pipeline and the baselines are done. The TRACE encoder is built and its
100-epoch training run is underway. The reasoning, trust layer and demo come next. See [Where we are](#5-where-we-are).

---

## Contents
1. [The problem in plain words](#1-the-problem-in-plain-words)
2. [The key idea](#2-the-key-idea)
3. [How TRACE works](#3-how-trace-works)
4. [What's new in our training](#4-whats-new-in-our-training)
5. [Where we are](#5-where-we-are)
6. [Results so far](#6-results-so-far)
7. [Data and the no-leakage rules](#7-data-and-the-no-leakage-rules)
8. [Repo layout](#8-repo-layout)
9. [Running it yourself](#9-running-it-yourself)
10. [Further reading](#10-further-reading)

---

## 1. The problem in plain words

AI models for eye photos have three well-known weaknesses:

1. **They need lots of labelled photos.** A doctor has to label every training image, and rare diseases have
   only a handful of examples.
2. **They learn the camera, not the eye.** A model trained on one hospital's camera often fails on another,
   because it partly learned "this tint and lighting means disease X". This is called *domain shift*.
3. **They are always confident.** A normal model gives an answer even for a blurry photo, a disease it has never
   seen, or a picture that isn't an eye at all.

ZEISS asked for one reusable encoder that learns from **unlabelled** photos, transfers to new hospitals and
cameras with **few labels**, and is **calibrated, explainable and able to say "I don't know"**.

---

## 2. The key idea

> **Rare diseases are rare, but their findings are not.**

Take AION, a rare optic-nerve stroke with almost no training photos. A doctor recognises it from two
*findings*: a **swollen optic disc** and a **pale optic disc**. Both findings also appear in many commoner
diseases, so there are plenty of photos to learn them from.

So TRACE learns the **findings** from plenty of data, then **composes** rare diseases from them using
medical knowledge. That leads to three situations:

| Situation | What TRACE does |
|---|---|
| **A**: a few photos of the disease exist | Compare the new photo with them (nearest neighbours / prototypes) |
| **B**: no photos at all | Reason from findings with a medical knowledge graph ("disc edema + pallor → AION") |
| **C**: something it has never seen | Notice that the photo is unfamiliar and **refer it to a specialist** |

---

## 3. How TRACE works

### One photo in, one decision out

```
Fundus photo (any camera)
   │
   ▼
Quality gate ──bad photo──► "Please retake the photo"
   │ ok
   ▼
Preprocess (find the retina · even out lighting · tone down camera colour · gentle contrast boost)
   │
   ▼
Encoder (ResNet-50, frozen) ──► 2048-number "fingerprint" of the eye
   │
   ├──► ~20 finding detectors ──► p(disc edema), p(haemorrhage), ... ──► Knowledge graph ──► disease scores + reasons
   │
   └──► Similar-case search (FAISS) ──► "these 5 past patients looked like this" + "is this photo unusual?"
                                │
                                ▼
                Trust layer: unusual? → calibrate → conformal prediction set
                                │
                                ▼
                 Diagnose  |  Short-list  |  Refer to a specialist
         (always shown with the findings, the evidence for and against, and the similar cases)
```

### The four layers

| Layer | Job | Learns from | Model |
|---|---|---|---|
| 1. Perception | Turn a photo into a fingerprint (embedding) | ~22,500 **unlabelled** photos | ResNet-50 trained with MoCo v2 + our cross-source mining |
| 2. Findings | Detect ~20 visible signs (disc swelling, haemorrhages, drusen …) | RFMiD / RFMiD 2.0 finding labels | Small MLP on the frozen fingerprint |
| 3. Reasoning | Turn findings into disease scores with reasons | Literature (cited per edge) + label counts | Bayesian evidence scorer over a knowledge graph |
| 4. Trust | Decide whether to answer, short-list or refer | Held-out calibration photos, per camera | Temperature scaling, kNN out-of-distribution check, conformal prediction |

The encoder is trained **once, without labels, and then frozen**. Everything after it is small and quick to
train, and swapping encoders means changing a single config value.

### Is the encoder pretrained or built from scratch?

Both, in two stages:
1. It **starts** from the standard ImageNet ResNet-50 (torchvision), which already knows general vision: edges,
   textures, blobs.
2. It is then **trained further on fundus photos, with no labels**, using contrastive learning. This turns it into
   a retina encoder.

Training from scratch on ~22k photos would need far more data and compute than we have.

---

## 4. What's new in our training

### Contrastive learning in one paragraph
Take one photo and make two slightly edited copies (crop, rotate, a small colour change). The model is
rewarded when the two copies of the same photo get **similar** fingerprints, and copies of **other** photos
(the "negatives") get **different** ones. No labels are needed. The choice of edits decides what the model
learns to ignore. We use **MoCo v2**, which keeps a queue of 8,192 past fingerprints as negatives, so it works
with the small batches a laptop GPU can fit.

### Problem 1: standard edits erase disease
Normal contrastive recipes use strong colour changes, grayscale and tiny crops. In an eye photo these
**destroy the evidence**: a pale disc is a colour, red haemorrhages vanish in grayscale, and a small crop can cut
out the only lesion. Our rule is that **an edit is allowed only if a doctor would give both copies the same
diagnosis.**

| Edit | Standard recipe | TRACE |
|---|---|---|
| Crop | 8–100% of the image | 60–100%, kept inside the retina |
| Colour jitter | strong | gentle (brightness/contrast 0.2, hue 0.02) |
| Grayscale | 20% of the time | **never** |
| Blur | strong | light, 30% of the time (microaneurysms are a few pixels wide) |
| Camera artifacts | none | vignetting, camera colour profiles, haze, glints, eyelash shadows, JPEG, so the model learns to ignore the device |
| Multi-crop | none | 2 large + 4 small views per photo, linking lesion-scale detail to the whole eye |

We deliberately **never** add fake dust spots, because they look just like microaneurysms.

### Problem 2: false negatives (our main contribution)
Two *different* patients with the *same* disease get pushed apart as "negatives". Look-alikes get pushed
hardest, so this hurts rare diseases most.

The usual fix is "stop pushing photos that look very similar". But early in training, photos from the **same
camera** look similar just because they share tint and lighting, so that fix would **strengthen** camera bias.

**TRACE's rule:** a negative is turned into an extra positive (pulled closer) only if it:
1. comes from a **different camera/source**, so the similarity can't be a camera cue;
2. is in the **top 1%** of cross-source similarities;
3. is a **mutual nearest neighbour** (each is in the other's top 10);
4. appears **after a 20-epoch warm-up**.

Separately, negatives from the **same** camera are pushed **twice as hard** (β = 2). This stops the model from
telling photos apart by camera cues.

| Key | Plain MoCo | TRACE |
|---|---|---|
| Diabetic eye, *other* camera, looks alike | pushed away (wrong) | **pulled closer** |
| Healthy eye, *same* camera, looks alike | pushed (26% of the push) | **pushed harder** (42%) |

With no attraction and β = 1, the loss is exactly the standard MoCo loss, which gives clean ablations.

We **monitor** the rule every epoch. Labels are used only to measure it, never to train. In a short check run,
attracted pairs shared a finding or disease **4.9–5.4× more often than chance**, with no same-eye copies.

---

## 5. Where we are

| Phase | What | Status |
|---|---|---|
| 0. Setup | Repo, configs, tests | ✅ Done |
| 1. Data | Ingest 8 datasets, dedupe, preprocess, label mapping, leak-free splits | ✅ Done (45,864 photos) |
| 2. Baselines | ImageNet ResNet-50, plain MoCo v2 (15 and 100 epochs), evaluation suite | ✅ Done |
| 3. TRACE encoder | Retina-safe views, artifact simulator, multi-crop, cross-source mining, mining monitor | 🟡 Built and tested; 100-epoch run underway (epoch 12 of 100) |
| 4. Findings + graph | Disc/macula crops, finding detectors, knowledge graph, evidence scorer | ⏳ Next |
| 5. Trust | Quality gate, calibration, OOD, per-camera conformal sets, decision rule | ⏳ Planned |
| 6. Package | Gradio demo, UMAPs, model card | ⏳ Planned |

**144 unit tests pass.** Every formula is checked against a hand-worked example.

---

## 6. Results so far

All numbers come from real runs. "Frozen" means the encoder is never fine-tuned; a small linear classifier is
trained on top of its fingerprints.

### Baselines (held-out test sets)

| Encoder | RFMiD linear macro-AUROC | RFMiD kNN | IDRiD any-DR | Paraguay any-DR (unseen site, ZEISS camera) | Camera probe ↓ |
|---|---|---|---|---|---|
| ImageNet ResNet-50 | 0.885 | 0.759 | 0.865 | 0.863 | 0.971 |
| Plain MoCo v2, 15 epochs | **0.918** | **0.851** | 0.859 | **0.903** | 0.961 |
| Plain MoCo v2, 100 epochs | 0.892 | 0.819 | **0.887** | 0.819 | 0.969 |

What the table shows:
- Self-supervised training on fundus photos **beats ImageNet**, and the gain shows up mostly in the first epochs.
- **Longer plain training isn't better.** The training epoch is chosen on a validation split (val-A), never on test.
- **The camera probe tells us how strongly the fingerprint reveals which camera took the photo.** It stays near
  0.97 (chance is 0.14) for every baseline. Plain contrastive learning does **not** remove camera bias. That gap
  is what TRACE targets, and it is the comparison we will report.

**TRACE encoder (in progress), val-A linear macro-AUROC:**

| Epoch | Score |
|---|---|
| 0 | 0.883 |
| 5 | 0.916 |
| 10 | 0.917 |

Plain MoCo's best on the same split was 0.914. Mining only starts at epoch 20, so the real test comes later.

### Preprocessing removes camera signal but keeps the biology
A simple classifier guessing the source dataset from photo statistics (4 training datasets, chance 0.25) scores:

| After | Score |
|---|---|
| Raw photos | 0.805 |
| Our preprocessing | 0.717 |
| Camera colour step | **0.493** |

Meanwhile, disc pallor and media haze detection did **not** drop (AUROC 0.615 → 0.634 and 0.753 → 0.794).

### Things we found in the data
- **IDRiD photos are inside RFMiD.** 84 RFMiD photos match IDRiD test photos, and 76 are exact copies. Pretraining
  on RFMiD and testing on IDRiD would leak. We drop the copies from our training side.
- **Public datasets copy each other.** We dropped 2,039 duplicate or same-eye photos so that no test eye hides in
  training.
- **61% of RFMiD 2.0 photos are squashed** (4:3 photos stored as squares). Our retina-finding step fits an ellipse
  and un-stretches them.
- **The same label code means different things in different datasets.** "MCA" is *macro*aneurysm in RFMiD and
  *micro*aneurysm in RFMiD 2.0. All labels are mapped through one checked vocabulary in
  [data/ontology.yaml](data/ontology.yaml).

---

## 7. Data and the no-leakage rules

| Dataset | Photos | Used for |
|---|---|---|
| RFMiD (provided by ZEISS) | 3,200 | Pretraining (train split, labels hidden), finding labels, validation, held-out test |
| RFMiD 2.0 | 860 | Same as RFMiD; extra rare diseases |
| EyePACS (Kaggle 2015) | 35,108 | Pretraining pool (16,918 quality-filtered), labels ignored |
| EyeQ | 12,543 grades | Photo-quality labels for EyePACS |
| APTOS 2019 | 3,662 | Pretraining pool, labels ignored |
| IDRiD | 516 + 81 | **Held-out test**: DR grading, lesion masks |
| JSIEC-1000 | 1,000 | **Held-out test**: 39 classes, rare diseases, unfamiliar photos |
| Paraguay DR | 1,437 | **Held-out test**: unseen site on a ZEISS VISUCAM 500 |
| REFUGE, ODIR-5K | — | Planned |

**Photos are not included in this repo.** Each dataset must be downloaded from its own source under its own licence;
see [DATA_CARD.md](DATA_CARD.md).

**Leakage rules we never break:**
- Every photo is compared with every other for duplicates and same-eye copies (matched on fine retinal detail,
  not just pHash). A copy of a test photo never enters training.
- Splits keep the same eye together.
- Validation is split in two: **val-A** tunes settings and **val-B** sets the conformal threshold. Neither trains
  any weights.
- **Leave-disease-out:** when we test a disease "zero-shot" (AION, CRVO, RP), its photos are removed from
  *everything*, including the unlabelled pretraining pool.
- Test sets are touched only at the end.

---

## 8. Repo layout

```
zesis/
├── CLAUDE.md            # full design spec + decision log (the "source of truth")
├── DATA_CARD.md         # every dataset, licence, counts, cleaning, known problems
├── docs/DATA_PIPELINE.md# how a photo goes from download to training view
├── Makefile             # make data / test / audit / probe
├── configs/
│   ├── data.yaml        # pool size, split rules, source merges
│   ├── pretrain.yaml    # full TRACE run (trace_v1)
│   └── pretrain_plain.yaml  # plain MoCo v2 baseline
├── data/                # Phase 1: ingest, dedupe, preprocess, quality, ontology, splits
├── augment/             # retina-safe views, artifact simulator, torch-free multi-process loader
├── encoder/             # MoCo, TRACE loss, mining monitor, training loop, export, feature cache
├── eval/                # frozen-feature probes, kNN, camera probe, val-A epoch selection
├── scripts/             # audits and checks: camera probe, dedupe audit, view contact sheets, pseudo-domains
└── tests/               # 144 unit tests (pytest)
```

Not in git (too big, or not ours to share): `raw/` (downloads), `processed/` (512 px photos, ~14 GB),
`features/`, `checkpoints/`, `outputs/`, and all `*.parquet`, `*.npy` and `*.pt` files.

---

## 9. Running it yourself

### Hardware
Developed on a laptop with an **RTX 4060 (8 GB)**, 16 GB RAM and Windows 11. Kaggle or Colab works for the bigger runs.

| Run | Time |
|---|---|
| Data pipeline | ~1 h on 6 CPU workers |
| Plain MoCo, 100 epochs | ~5 h |
| Full TRACE, 100 epochs | ~7.5 h |

### Setup
```bash
git clone https://github.com/nithinj25/zesis-contrastive-learning.git
cd zesis-contrastive-learning
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/Mac:  source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
pytest -q                     # should print: 144 passed
```

### Put the datasets in `raw/`
```
raw/
├── rfmid/      Training_Set/…, Evaluation_Set/…, Test_Set/…   (as unzipped from IEEE DataPort)
├── rfmid2/     Training_set/, Validation_set/, Test_set/
├── eyepacs/    resized_train_cropped/resized_train_cropped/*.jpeg + trainLabels_cropped.csv
├── eyeq/       Label_EyeQ_train.csv
├── aptos/      train_images/train_images/, … + train_1.csv, valid.csv, test.csv
├── idrid/      A. Segmentation/, B. Disease Grading/, C. Localization/
├── jsiec/      1000images/<class folders>/
└── paraguay/   <grade folders>/ + "Annotations of the classifications.xlsx"
```
A missing dataset is skipped with a message. The exact paths are at the top of [data/ingest.py](data/ingest.py).

### Pipeline
```bash
make data          # ingest → dedupe → preprocess → quality → splits  → data/meta.parquet + processed/
                   # (no make on Windows? run the five "python -m data.…" lines in the Makefile)
make probe         # camera signal before/after preprocessing
python scripts/view_sheet.py --artifacts       # look at training views before training

# Pretrain
python -m encoder.train --config-name pretrain_plain     # plain MoCo v2 baseline
python -m encoder.train                                  # full TRACE (configs/pretrain.yaml)
python -m encoder.train ckpt.resume=checkpoints/trace_v1/last.pt   # resume after a stop or crash

# Evaluate a frozen encoder
python -m encoder.features --name imagenet_r50                                   # ImageNet baseline
python -m encoder.features --name trace_v1 --weights checkpoints/trace_v1/encoder.pt
python -m eval.probes --name trace_v1        # → outputs/eval/trace_v1.json + a row in baselines.csv
```
Any config value can be overridden on the command line (Hydra), e.g. `train.workers=8 optim.epochs=30`.

**Windows tips:**
- Keep DataLoader workers modest (8 on 16 GB RAM). Our loader workers never import torch, to save memory.
- Training saves a resumable checkpoint every 300 steps. A laptop GPU at its thermal limit can crash, and
  resuming continues from that exact step.

---

## 10. Further reading

| Read | What it covers |
|---|---|
| [CLAUDE.md](CLAUDE.md) | The complete design: every decision, formula, worked example and the dated decision log (§20) |
| [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md) | The data pipeline step by step |
| [DATA_CARD.md](DATA_CARD.md) | Datasets, licences, cleaning numbers, known issues |

**Built on:** MoCo v2 (He et al. 2019; Chen et al. 2020), SimCLR (Chen et al. 2020), alignment and uniformity
(Wang & Isola 2020), conformal prediction, and the HPO frequency annotations for the knowledge graph.

---

*Research prototype for a hackathon. **Not a medical device**, and not for diagnosing real patients.*
