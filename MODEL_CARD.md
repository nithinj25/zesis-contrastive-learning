# TRACE model card

**TRACE (Trustworthy Retinal Contrastive Encoder)** is a frozen ResNet-50 encoder for colour fundus photos, trained
without disease labels, plus small heads and a trust layer on top of it. Team div4maxxers, BMSIT Bengaluru, for the
ZEISS India Campus Hackathon "Beyond Code 2026", Theme 2. Every number below comes from a script in this repo; the
full log, with dates and the reasoning behind each choice, is section 20 of [CLAUDE.md](CLAUDE.md).

> **Version of 13 Nov 2026.** Code: MIT licence. Weights: research use only (section 9).

---

## 1. Intended use

- **For:** research on label-efficient retinal representation learning; a screening-support prototype that
  suggests findings, a short-list of diseases and similar past cases, and says when it is unsure.
- **Not for:** diagnosis or any clinical decision. TRACE is not a medical device, has not been validated
  prospectively and has not been reviewed by clinicians. Every output must be read by an eye-care professional.
- **Users:** researchers and engineers; in a demo, an eye-care professional reading the decision card.

## 2. What it is

| Part | What | Trained on |
|---|---|---|
| Encoder | torchvision ResNet-50, ImageNet start, 2048-d L2-normalised output at 384 px; frozen after pretraining | 22,500 unlabelled photos (EyePACS 16,918, APTOS 3,387, RFMiD train 1,735, RFMiD 2.0 train 460) |
| Pretraining | MoCo v2 (queue 8,192, momentum 0.999, τ 0.2) with retina-safe views, a camera-artifact simulator, multi-crop and same-camera negatives ×2 (β = 2); the val-A-chosen checkpoint after 20 epochs at 224 px, then 10 more epochs at 384 px (also chosen on val-A) | Labels never used |
| Finding detectors | 14 small MLPs + 6 prototypes over 20 findings (disc edema, haemorrhage, drusen, …), calibrated | RFMiD + RFMiD 2.0 head-training labels (2,193 photos) |
| Disease classifier | Logistic regression, 6 diseases + "other abnormal", temperature-scaled | Same |
| Rare-disease bank | One prototype per rare disease (AION, CRVO, RP), called only above a margin set for 5% false alarms on val-A | The rare photos only |
| Trust layer | Quality gate, kNN out-of-distribution check (FAISS), per-camera conformal prediction sets, decision rule | Val-A (thresholds, temperature) and val-B (conformal), never the training photos |

The cross-camera false-negative mining described in our proposal was built and tested, but **is not in the final
encoder**: the checkpoint chosen on val-A came before mining started, and the ablation (section 5) shows mining trades
few-shot accuracy for camera mixing. It is reported as a tested idea, not as part of the model.

**Interface** ([encoder/export.py](encoder/export.py)): `load_trace_encoder("encoder.pt")`, `preprocess(photo_rgb)`
→ `(3, 384, 384)` in [0, 1] → `(1, 2048)` features. A known camera gets its stored colour correction; an unknown camera
skips it (no per-image colour normalisation, which would erase pallor).

## 3. Data and leakage control

Details in [DATA_CARD.md](DATA_CARD.md). Nine public datasets, 45,864 photos; deduplicated by vessel-pattern matching
(2,039 dropped). **84 RFMiD photos are IDRiD photos** (76 exact copies); the train and val copies were removed, so the
organisers' "pretrain on RFMiD, test on IDRiD" flow would leak unless they do the same. AION, CRVO and RP photos were
removed from pretraining and from every head, so tests on them are on diseases the system never trained on.
Held out until evaluation: RFMiD test, RFMiD 2.0 test, IDRiD, JSIEC-1000, Paraguay (ZEISS VISUCAM 500).

## 4. Results

All with the encoder frozen. 95% intervals from 1,000 bootstrap resamples of the test photos, or over 5 seeds.

### 4.1 Transfer (linear probe on frozen features, macro-AUROC)

| Test | ImageNet ResNet-50 | Plain MoCo v2 (15 ep) | Supervised ResNet-50 | **TRACE** |
|---|---|---|---|---|
| RFMiD test, 18 labels | 0.885 | 0.917 | 0.918 | **0.917** [0.897, 0.935] |
| IDRiD, any DR | 0.865 | 0.859 | 0.864 | **0.871** |
| Paraguay (new hospital, ZEISS camera), any DR, probe trained on IDRiD | 0.863 | 0.902 | 0.933 | 0.886 |
| Unseen camera, Kowa VX-10α held out | 0.867 | 0.903 | not valid* | **0.921** |
| Unseen camera, Topcon 3D OCT-2000 held out | 0.796 | 0.809 | not valid* | **0.858** |

The supervised baseline is the whole ResNet-50 fine-tuned on RFMiD's labels. With its own head it scores **0.941** on
RFMiD test, above TRACE. \*Its backbone saw labelled photos from every RFMiD camera, so a held-out-camera test needs a
retrain per camera; not run.

### 4.2 Few labels, new diseases

| Test | ImageNet | Plain MoCo | Supervised | **TRACE** |
|---|---|---|---|---|
| JSIEC-1000, 39 diseases, 1 photo each (prototype top-1) | 0.297 | 0.384 | 0.357 | **0.403** |
| … 5 photos each | 0.536 | 0.611 | 0.571 | **0.628** |
| … 10 photos each | 0.610 | 0.679 | 0.621 | **0.716** |

Chance is 0.026. Support and query photos never come from the same eye; 5 seeds.

**A teammate's fixed few-shot exam** (11 rarer RFMiD labels, K = 1 / 3 / 5 × 50 fixed episodes, macro AUPRC):
TRACE 0.144 / 0.232 / 0.264; plain MoCo 0.117 / 0.198 / 0.226; ImageNet 0.095 / 0.141 / 0.169. Their published
DINOv2-S run on the same episodes: 0.123 / 0.199 / 0.227 (their preprocessing at 448 px, so not a paired comparison).
TRACE's checkpoint was chosen on val-A, which overlaps these queries, so it has a slight selection advantage.

**Rare diseases never trained on** (share caught at a 5% false-alarm rate on val-A; logistic model with distribution
calibration): AION 0.47 (30 photos), CRVO 0.79 (56), RP 0.79 (14); on JSIEC, a dataset never seen, CRVO 0.95 and RP
0.95 (22 each). Small samples: read these as indications, not precise rates.

**Zero-shot from a knowledge graph failed:** scoring AION and CRVO from literature-based finding profiles, with no
photos, got 0 of 30 and 0 of 78 right (the detectors barely respond to CRVO's signs). Rare diseases therefore need a
few photos; the graph is kept to explain decisions (evidence for and against).

### 4.3 Parameter-efficient adaptation (LoRA)

Rank-8 LoRA on the last ResNet block (262k weights, 1.1% of the network); encoder otherwise frozen. RFMiD test,
linear probe → LoRA on the same photos: 1% of labels (17 photos, 3 draws) 0.807 → 0.817; 10% (173 photos, 3 draws)
0.879 → 0.884; 100% (1 run) 0.917 → 0.932. Full fine-tuning of the supervised baseline: 0.941.

### 4.4 Trust

| Check | Result |
|---|---|
| Calibration (RFMiD + RFMiD 2.0 test, 732 photos, 7 classes) | Accuracy 0.796, ECE 0.029 |
| Conformal sets, target 90% | Coverage 0.906, mean set size 1.39; per camera 0.895 (OCT-2000), 0.943 (Kowa), 0.918 (TRC-NW300), 0.923 (RFMiD 2.0 512 px), 0.722 (RFMiD 2.0 1504 px, 18 photos) |
| New camera (Paraguay) | Coverage drops to 0.575 with existing thresholds; re-calibrated on ~100 local photos: 0.895 [0.836, 0.946], with larger sets (3.6) |
| Out-of-distribution | Non-fundus images (CIFAR-10): AUROC 1.000. Fundus diseases unknown to TRACE (13 JSIEC classes): AUROC 0.823, but catching 95% of them would also flag 66% of familiar photos. RFMiD test flagged 5.8% (target 5%); Paraguay 21.5% |
| Quality gate (EyeQ reject photos) | Catches 88.8%; asks for a retake on 3.6% of RFMiD, 11.1% of RFMiD 2.0, 20% of JSIEC and 55.5% of Paraguay photos (27.8% after onboarding with 100 local photos) |
| Full decision rule, RFMiD + RFMiD 2.0 test (787) | Diagnose 34% · short-list 31% · refer 30% · retake 5%. Diagnoses right 90.3%; **confidently wrong 3.3%**; short-lists hold the true disease 84% |
| Rare-disease photos (63) | Refer 79%, **confidently wrong 0%** |
| Diseases TRACE does not know (JSIEC, 219) | Confidently wrong 7.8% |
| Paraguay (1,208) | Confidently wrong 2.0%, but retake 55% (the quality gate at an unseen camera) |

### 4.5 Robustness to poor photos

A teammate's stress test: 11 corruptions (blur, haze, glare, exposure, colour cast, noise, JPEG, low resolution, …)
× 5 severities on RFMiD test, probes trained on clean photos only.

| | ImageNet | Plain MoCo | Supervised | **TRACE** |
|---|---|---|---|---|
| Mean macro-AUROC, 55 corrupted settings | 0.845 | 0.889 | 0.885 | **0.897** |
| At the worst severity | 0.802 | 0.851 | 0.844 | **0.872** |
| Mean corruption error vs ImageNet (lower is better) | 1.00 | 0.72 | 0.74 | **0.67** |
| Normal / abnormal calls that flip | 11.8% | 8.3% | 8.9% | **5.8%** |

Several corruptions resemble TRACE's own training artifacts; the top two severities go beyond its training ranges.

### 4.6 Camera bias

A linear probe still tells RFMiD's three cameras apart from TRACE features (balanced accuracy 0.995; ImageNet and
every other encoder ≈ 0.99, chance 0.33), even on normal eyes. Camera *detectability* is therefore not removed. What
improved is how much the camera shapes the feature map: share of a photo's 20 nearest neighbours from its own camera
0.91 for TRACE vs 0.96 for ImageNet (chance 0.64), and the drop in AUROC on an unseen camera (section 4.1).

### 4.7 Explaining the embedding

[UMAP](outputs/figures/umap_imagenet_r50_vs_trace_v2.png): share of each photo's 20 nearest neighbours with the same
disease 0.74 for TRACE vs 0.65 for ImageNet (chance 0.28). [Retrieval](outputs/figures/retrieval_trace_v2.png)
(5 most similar confirmed cases, query's own eye excluded): right diagnosis in 50% of returned cases on RFMiD (ImageNet
41%, supervised 62%); JSIEC CRVO 77%, but RP only 12% and AION 9%, because the library holds few photos of them.

## 5. What each training idea did (ablation, short budget)

10,000 photos, 30 epochs, read at the final epoch; differences smaller than the seed-to-seed spread are not read.
Retina-safe views carry the unseen-camera gain (OCT-2000 held out 0.819 → 0.848; Paraguay 0.910 → 0.935). Same-camera
negatives ×2 lower the same-camera neighbour share (0.929 → 0.907) and change nothing else. Cross-camera mining mixes
cameras most (0.907 → 0.868) but costs few-shot accuracy (JSIEC 5-shot 0.600 → 0.520) and IDRiD (0.880 → 0.845).
Multi-crop lowers JSIEC 5-shot (0.635 → 0.594); a teammate's lesion-guided local crops, tested in its place, lowered
it further (0.550) and made neighbours more camera-driven (0.929 → 0.958), so they were not adopted. Full table:
`outputs/ablation/table.md`.

## 6. Known failure modes

- **Diagnoses that supervised learning sees well:** a ResNet-50 fine-tuned on RFMiD labels beats TRACE on RFMiD's own
  diseases (0.941 vs 0.917) and on DR at Paraguay (0.933 vs 0.886). TRACE's advantage is on diseases without many labels.
- **New cameras:** thresholds, conformal coverage and the quality gate all drift at a new site (Paraguay: coverage
  0.575, 55% retake requests). Rankings transfer; cut-offs need ~100 local labelled photos.
- **Camera identity stays in the features**, so retrieval and the familiarity check partly match by camera.
- **Rare eyes can look familiar:** most unknown fundus diseases pass the OOD check; referral cannot rest on it alone.
- **Small rare-disease samples** (14–56 photos): wide uncertainty on every rare-disease number.
- **Multi-label data, single-label guarantee:** conformal sets cover the primary diagnosis on single-diagnosis photos only.
- **Labels can disagree across datasets** (one eye is CRVO in RFMiD and DR grade 4 in IDRiD).
- Not tested: REFUGE / glaucoma transfer, ODIR, prospective or clinician-reviewed evaluation, children, populations
  other than those in these datasets, widefield or OCT images.

## 7. Compute

One laptop GPU (NVIDIA RTX 4060 Laptop, 8 GB), mixed precision. The final checkpoint needs ≈ 4 GPU-hours (20 epochs at 224 px ≈ 1.6 h,
10 at 384 px ≈ 2.2 h); the runs that produced it continued to 100 and 20 epochs (≈ 12 h). All runs, including
baselines and the 8-run ablation, ≈ 30 GPU-hours (estimate from the logs). Heads, calibration and evaluation run on
CPU in minutes. Inference on a laptop CPU: about 0.3 s to preprocess and 0.15 s to encode one photo (one measurement).

## 8. Reproduce

See [README.md](README.md). Datasets must be downloaded by the user under their own terms.

## 9. Licence and citation

- **Weights: research use only.** They were trained on datasets whose terms are research-only or still to be verified
  (EyePACS and APTOS Kaggle terms; RFMiD; see [DATA_CARD.md](DATA_CARD.md)).
- **Code:** MIT licence ([LICENSE](LICENSE)).
- Cite the datasets as listed in the data card.
