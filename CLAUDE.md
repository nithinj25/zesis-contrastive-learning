# CLAUDE.md — TRACE (Trustworthy Retinal Contrastive Encoder)

> Project context file for Claude Code. It holds every decision made while designing TRACE, so a new
> session can start building without the original chat history. Read it fully before writing code.
> Last updated: 2026-11-05.

---

## 0. How to work with me (read first)

I'm Nithin J, a 3rd-year CS (AI/ML) student at BMSIT Bengaluru. I'm building this for the
ZEISS India Campus Hackathon "Beyond Code 2026", Theme 2.

**Working style I want:**
1. **Discussion first, then code.** Before writing a module, explain in simple words what it does,
   why, and how it fits into TRACE. Then write it.
2. **Simple language, small worked examples.** When a formula is involved, show a tiny dry run with
   real numbers, and verify the formula against that trace (run it).
3. **Reasoning over code dumps.** Explain the reasoning chain. Avoid writing 500 lines at once.
   Build in small, testable pieces.
4. **Follow the build order (section 14).** Don't start a phase until the previous one meets its
   "done when" line.
5. **Locked decisions (section 2) don't change silently.** If you think one is wrong, say so and
   explain why before changing anything.
6. **Every module gets a unit test**, using the sanity-check numbers in section 16 where they apply.
7. **No data leakage, ever.** See section 5.4. If unsure whether something leaks, stop and ask.
8. Put labels and numbers marked "illustrative" in this file only into examples, never into results.
9. Section 19 holds the build-level specs (interfaces, configs, parameters). Follow it unless I say otherwise.

---

## 1. The hackathon (facts from the official brief and kickoff deck)

### 1.1 Theme 2: Contrastive Learning for Rare Eye Diseases
- **Mentor:** Dr Raghu Prasad, Head of Advanced Development & AI, Centre for Application Research, India.
- **General expectation:** build a robust, scalable foundation model using contrastive learning that
  identifies hidden pathological patterns in retinal fundus images with much less labelled data and
  without explicit supervision.
- **Core problem named by ZEISS:** domain shift. A model trained on one hospital's machines and
  population fails on another.
- **Focus areas:** Retina-Preserving Architecture · Modality Agnosticism (beating domain shift) ·
  Low-Label Transfer · Trustworthy Explainability (resisting false negatives).
- **Provided data:** RFMiD (IEEE DataPort). The brief says the open repositories hold "over 5,000 images".
- The brief says the focus areas are guides, and new ideas are welcome. I've also been told we are
  **not restricted to RFMiD images**.

### 1.2 Grand challenge (kickoff deck)
**"Trustworthy Contrastive Foundation Models for Retinal Imaging. Can a model learn retinal biology,
not dataset shortcuts, without seeing disease labels?"**

**Mission:** build a compute-efficient retinal foundation encoder using contrastive self-supervision on
heterogeneous, mostly unlabelled fundus images. The encoder must preserve subtle pathology, resist false
negatives and camera bias, transfer with few labels, detect unfamiliar inputs, and explain its
embedding structure.

**Five barriers to solve:**

| # | Barrier | ZEISS wording | TRACE answer |
|---|---|---|---|
| 01 | Safe pairs | Retina-preserving views; don't erase tiny lesions | Retina-safe augmentations + multi-crop (§7.3) |
| 02 | False negatives | Multi-positive, prototype or debiased contrast | Cross-source false-negative mining (§7.4) |
| 03 | Domain shift | Suppress camera and dataset signatures; keep biology | Per-camera preprocessing + artifact simulator + same-source β weighting (§6, §7) |
| 04 | Low-label transfer | Prove linear-probe, few-shot and **PEFT** capability | Frozen encoder + probes + prototypes; optional LoRA adapter (§7.8) |
| 05 | Trust | Calibration, OOD detection, interpretable retrieval | Quality gate, temperature scaling, FAISS OOD + retrieval, per-camera conformal (§10) |

**Organiser data flow:**
1. PRETRAIN: RFMiD images, labels hidden.
2. ADAPT: a small organiser-defined labelled subset.
3. TEST: held-out RFMiD + IDRiD, with no tuning on the target.
4. STRESS: quality and domain shifts, hidden evaluation.

**Submission:** encoder, code, model card, UMAP, retrieval, ablations.

**Win condition:** one reusable encoder that beats strong supervised and ImageNet baselines on hidden
transfer while staying efficient, calibrated and auditable.

**Scoring:**

| Component | Weight |
|---|---|
| Hidden transfer | 25% |
| Few-shot / linear probe | 20% |
| Contrastive innovation | 20% |
| Trust & OOD | 20% |
| Efficiency, interpretability & reproducibility | 15% |

Hidden transfer + linear probes = 45%, and **both freeze the encoder**. That's why the encoder is
trained once, without labels, and then frozen (§2).

### 1.3 Timeline (tentative, from the kickoff deck)
- Idea submission: 14–24 Oct 2026 (done: description, tech stack, pitch video, slides).
- Hackathon sprint: 1–8 Nov 2026 (tentative). **Final submission: Sun 15 Nov 2026** (per Nithin; confirm with the organisers).
- Finalist shortlist: 16–20 Nov 2026.
- Grand finale: 1 Dec 2026, ZEISS GCC Bangalore office.

### 1.4 Questions to ask the mentor (still open)
1. Are external pretrained weights (ImageNet, RETFound) allowed?
2. Is there a compute budget or a required image resolution?
3. Which datasets are in the hidden evaluation, so we can keep them out of the pretraining pool?
4. Does "modality agnosticism" mean cameras only, or also widefield / OCT?
5. May we compute colour statistics from **unlabelled** target images (e.g. the hidden test set), or does
   "no target tuning" forbid that? (Affects §19.1.)
6. Do the organisers run our `preprocess()` on hidden data, and do they pass a camera ID? (Affects §19.1.)
7. Does the deck allow a public pretraining pool beyond RFMiD? §1.2 records "PRETRAIN: RFMiD images", so check the deck itself. (Affects §2.3 item 1.)

---

## 2. Problem statement and locked decisions

### 2.1 Problem statement (final)
> **How can we build a contrastive foundation model that learns hidden pathology from unlabelled fundus
> images, stays reliable across devices, diagnoses rare diseases from few or zero examples, and explains
> and qualifies every decision?**

**Tagline:** "TRACE learns from less, ignores the camera, and knows when it doesn't know."

**Key insight:** rare diseases are rare, but their **findings are not**. Disc swelling, pallor,
haemorrhages and pigment appear across many diseases, so we learn them from plenty of data and
**compose** rare diseases from them. Example: AION has almost no images, but its two findings
(disc edema + disc pallor) are learned from many other diseases.

### 2.2 Locked decisions
| Decision | Value | Why |
|---|---|---|
| Backbone | ResNet-50, ImageNet initialisation (torchvision) | Fits one GPU, works on ~40k images, suits small local lesions |
| SSL method | MoCo v2 (queue + momentum encoder) | Many negatives with small batches; explicit negatives we can re-weight |
| Our novelty | Cross-source false-negative mining + same-source β weighting | Fixes false negatives without strengthening camera bias |
| Encoder after pretraining | **Frozen**. Everything downstream trains small heads on its features | ZEISS scores frozen-encoder transfer (45%); stable distances for OOD and conformal |
| Downstream feature | 2048-d backbone output, L2-normalised. Projection head (2048→2048→256) is **loss only** and discarded | Backbone features transfer better (SimCLR found >10%) |
| Reasoning | Findings → knowledge graph (Bayesian evidence scorer first, GNN later) | Zero-shot + explainable |
| Trust | Quality gate → OOD → calibrated scores → per-camera conformal → decision rule | Knows when it doesn't know |
| Compute | Local RTX 4060 Laptop (8 GB) first; Kaggle/Colab for the 384 px phase and overflow; mixed precision | Changed by Nithin 2026-11-05. See §20 |

### 2.3 Open decisions (discuss before building that part)
Items 1 and 3–6 were decided on 2026-11-05; see §20 for the reasoning.
1. **Pool breadth:** decided. Headline encoder uses a ~20–25k public pool; RFMiD-only is reported as a second row. Pending mentor question 7.
2. **The 3–5 rare target diseases:** still open. Provisional: AION, CRVO, RP. CRAO is out on the current data (5 images, none in a test split). A JSIEC class only if its findings are in the vocabulary. Real counts are in §20.
3. **PEFT adapter** (§7.8): decided. Minimal LoRA, no SupCon unless time is left.
4. **GNN alignment** (§9.7): decided. Skip unless everything else is done by about 13 Nov.
5. **Flip vs laterality standardisation:** decided. No standardisation; keep the 50% flip in pretraining.
6. **Disagreement flag** (§10.6): decided. Include, threshold tuned on val-A.

---

## 3. Architecture overview

### 3.1 Run time: one photo in, one decision out
```
Fundus photo (any camera)
   │
   ▼
Quality gate ──reject──► "Retake the photo"
   │ ok
   ▼
Preprocess (crop · illumination · camera colour · mild CLAHE)
   │
   ▼
Encoder (ResNet-50, frozen) ──► 2048-d feature
   │
   ├──► Finding probes (~20, calibrated) ──► p(finding) ──► Knowledge graph ──► graph scores (+ reasons)
   │
   └──► Case index (FAISS + prototypes) ──► image scores, kNN distance, 5 similar cases
                                │
                                ▼
                   Trust layer: OOD check → fusion → calibration → conformal set
                                │
                                ▼
               Diagnose  |  Short-list  |  Refer  (+ findings, evidence for/against, similar cases)
```

### 3.2 Build time: five stages, only the first is expensive
| Stage | What | Learns from | Cost |
|---|---|---|---|
| A | Pretrain encoder (MoCo + mining) | Unlabelled pool | Hours of GPU |
| B | Finding probes | RFMiD / RFMiD 2.0 finding labels (encoder frozen) | Minutes |
| C | Graph + alignment | Literature + label counts; small GNN | Minutes |
| D | Calibrate | Validation splits, per camera | Minutes |
| E | Evaluate | 6 held-out sets, never trained on | Minutes–hours |

Stages B–E can start on **frozen ImageNet ResNet-50 features** while stage A trains, then swap in the
TRACE encoder's features. Only the feature file changes.

### 3.3 Layer contract
| Layer | Input | Output | Learns from | Model |
|---|---|---|---|---|
| 1 Perception | Preprocessed photo, 384×384 | 2048-d feature | ~40k unlabelled images | ResNet-50, MoCo v2 + cross-source mining |
| 2 Findings | Features of whole image, disc crop, macula crop (3 × 2048 = 6144) | ~20 calibrated probabilities | RFMiD / RFMiD 2.0 finding labels | MLP 6144→512→20 |
| 3 Reasoning | Finding probabilities | Score per disease + evidence for/against | Literature + counts | Bayesian scorer (+ GraphSAGE alignment) |
| 4 Trust | Disease scores + distance to known cases | Diagnose / short-list / refer + similar cases | Calibration splits, per camera | Temperature scaling, kNN OOD, conformal |

---

## 4. Concepts quick reference (what I've learned; keep explanations at this level)

- **Embedding / pin on a map:** the encoder turns a photo into a vector. Similar eyes → close vectors.
- **Cosine similarity:** for unit vectors, a dot product. distance² = 2 − 2·cos.
- **Softmax + cross-entropy:** p_i = e^{z_i}/Σe^{z_j}, loss = −log p_correct.
- **Temperature τ:** divide scores by τ. Small τ = a sharp focus on the hardest look-alikes.
- **Contrastive learning = the find-the-twin game:** two edited copies of one photo are pulled
  together; copies of other photos (negatives) are pushed apart. Edits decide what the model ignores.
- **InfoNCE:** L = −log( e^{s₊/τ} / Σ_k e^{s_k/τ} ). Gradient on each score: ∂L/∂s_k = (p_k − y_k)/τ.
  A negative's push is proportional to how much it fooled the model.
- **Collapse:** with only "pull twins together", the model gives every image the same vector.
  Negatives prevent it. A collapsed model's loss = log(#candidates). Partial (dimensional) collapse =
  most dimensions have ~0 spread; healthy per-dimension std ≈ 1/√d.
- **SimCLR:** negatives from the batch only, so it needs huge batches.
- **MoCo:** queue of past keys + slow momentum key encoder θk ← mθk + (1−m)θq, m = 0.999.
- **False negatives:** two different patients with the same disease get pushed apart. Look-alikes get
  the hardest push, so this harms rare diseases most.
- **Projection head:** absorbs the "forgetting" the loss forces (colour, rotation). We keep the backbone.
- **Rare disease logic:** Situation A (few photos → prototypes / nearest pin), B (zero photos → graph),
  C (unknown → refer).

---

## 5. Data

### 5.1 Datasets (all public; verify counts after download, since versions differ)
| Dataset | Images | Camera(s) | Labels | Our use | Where |
|---|---|---|---|---|---|
| RFMiD (provided) | 3,200 (1,920 train / 640 val / 640 test) | Topcon 3D OCT-2000, Kowa VX-10α, Topcon TRC-NW300 | 45 conditions + disease-risk flag | Train: pool + labels. Val: calibration. Test: held out | IEEE DataPort (free account); Kaggle mirrors |
| RFMiD 2.0 | ~860 | Topcon TRC-NW300, ZEISS FF450plus | 49 classes + normal (incl. AION, CRAO, RP, MCA, NV) | Extra rare labels; same split rule | Zenodo record 7505822 (CC BY 4.0) |
| EyePACS (Kaggle DR 2015) | 88,702 | Many | DR grade 0–4 | Bulk of pool: ~25k quality-filtered subset; labels ignored | Kaggle |
| EyeQ | 28,792 (EyePACS images) | — | Quality: Good / Usable / Reject | Trains the quality gate; filters the pool | GitHub |
| APTOS 2019 | 3,662 | Mixed, India | DR grade | Pool (labels ignored) | Kaggle |
| ODIR-5K | ~7,000 | Canon, Zeiss, Kowa | 8 patient-level classes | Pool (labels ignored); keep both eyes in one split | ODIR / Kaggle |
| Messidor-2 (optional) | 1,748 | Topcon TRC NW6 | DR / DME | Pool, if access arrives | ADCIS request form |
| IDRiD | 516 + 81 | Kowa VX-10α | DR grade; pixel masks MA, HE, EX, SE, optic disc | **Held out**: DR transfer, lesion test, explanation check | IEEE DataPort |
| REFUGE | 1,200 | ZEISS Visucam 500 + Canon CR-2 | Glaucoma; disc/cup masks | **Held out**: glaucoma transfer, cupping, 2 cameras | Registration |
| JSIEC-1000 | 1,000 | ZEISS FF450 Plus + Topcon | 39 classes, many rare | **Held out**: rare / unseen-disease + OOD | Zenodo record 3477553 |
| Paraguay DR | 1,437 | ZEISS VISUCAM 500 | 7 DR classes | **Held out**: unseen site on a ZEISS camera (wording in §20) | Zenodo record 4891308 |
| RETFound weights | — | — | — | Baseline only (ViT-L ~300M params, gated download), if allowed | Hugging Face |

**Note on the organiser flow:** they pretrain on RFMiD with labels hidden and test on held-out RFMiD + IDRiD.
Never put IDRiD or RFMiD test into the pool.

### 5.2 The five jobs data does
1. **Pretraining pool:** unlabelled photos from many cameras, labels discarded.
2. **Label sets:** finding/disease labels for probes and graph alignment.
3. **Calibration sets:** labelled images the heads never trained on; only thresholds and temperature.
4. **Held-out tests:** never touched until evaluation, including an unseen site on a ZEISS camera.
5. **Quality / OOD:** quality-graded photos + non-fundus + rare-class images.

### 5.3 Split buckets
| Bucket | Contents | Rule |
|---|---|---|
| Pretraining pool | RFMiD train, RFMiD 2.0 train, EyePACS subset, APTOS, ODIR (training patients), Messidor-2 | Labels discarded; only train splits of RFMiD / RFMiD 2.0 |
| Head training | RFMiD train + RFMiD 2.0 train labels | Same images as the pool, now with labels |
| Val-A (tuning, 50%) | Half of RFMiD val + RFMiD 2.0 val | β, encoder early stopping, π_min, temperature, reliability weights, OOD and disagreement thresholds |
| Val-B (calibration, 50%) | The other half | Conformal q̂ only. Stratify by camera and rare class |
| Held-out test | RFMiD test, RFMiD 2.0 test, IDRiD, REFUGE, JSIEC, Paraguay | Never seen until the end; dedupe against the pool first |

### 5.4 Leakage traps (hard rules)
- **Duplicates across datasets:** public fundus sets reuse images. pHash everything; drop any pool image
  matching a test image; report how many were removed.
- **Both eyes of one patient:** keep them in the same split (ODIR especially). Splits are patient-level. Check RFMiD's files for patient IDs: if there are none, group near-duplicates (pHash + embedding), keep each group together in val-A/val-B and CV folds, and state that limit in the data card.
- **ZEISS hidden test:** remove whatever they reserve (ask the mentor).
- **Leave-disease-out:** for a disease tested zero-shot, remove its images from probe training, edge
  counts and alignment. Its graph edges come from literature only.
- **Calibration data never fits weights.**

### 5.5 Source (camera) IDs, needed by the mining loss
| Dataset | How |
|---|---|
| RFMiD | From image size: 2144×1424 = Topcon 3D OCT-2000; 4288×2848 = Kowa VX-10α; ~2048×1536 = Topcon TRC-NW300 |
| Single-camera sets | One ID per dataset (APTOS treated as one source) |
| Mixed sets (EyePACS, ODIR, RFMiD 2.0) | Pseudo-domains: k-means (k≈4–8) on acquisition statistics (size, FOV mask shape, mean LAB colour, sharpness, JPEG quantisation tables). Validate on RFMiD, where the true camera is known: with image-size features removed, each camera's **completeness** (share of its images in its dominant cluster) should be ≥ 0.9, because splitting one camera creates false cross-source pairs. Fallback: one source per dataset, merging datasets that share a known camera model |

### 5.6 One label vocabulary (= the graph's finding nodes)
| Finding | Location | Label sources |
|---|---|---|
| Optic disc edema | Disc | RFMiD ODE · RFMiD 2.0 ODE |
| Optic disc pallor | Disc | ODP |
| Optic disc cupping | Disc | ODC · REFUGE cup/disc masks (test only) |
| Tilted disc · optociliary shunt | Disc | TD · ST |
| Haemorrhages | Anywhere | HR, PRH; IDRiD HE masks (test only) |
| Microaneurysms | Posterior pole | MCA; IDRiD MA masks (test only) |
| Hard exudates / exudation | Macula | EDN · EX; IDRiD EX masks (test only) |
| Cotton wool spots | Posterior pole | CWS; IDRiD SE masks (test only) |
| Drusen | Macula | DN |
| Macular edema · macular scar · macular hole | Macula | ME/CME · MS · MHL |
| Tortuous vessels · neovascularisation | Vessels | TV · NV |
| RPE changes · tessellation · choroidal folds | Background | RPEC · TSLN · CF |
| Laser scars · retinal traction | Periphery | LS · RT |
| Media haze | Whole image | MH |

Candidate extra findings for rare targets (only if labels exist): pigment clumps (bone-spicule),
vessel attenuation, chorioretinal atrophy, crystals, yellow flecks, bull's-eye maculopathy.

Diseases (graph targets): DR, ARMD, BRVO, CRVO, CRAO, AION, CSR, RP, myopia, hypertensive retinopathy,
optic neuritis, glaucoma, plus the other RFMiD / RFMiD 2.0 disease labels. Mark uncertain mappings in
`data/ontology.yaml` (e.g. RFMiD "HR" = haemorrhagic retinopathy, used as haemorrhage evidence).

**A finding enters the vocabulary only if it is (1) visible in a fundus photo, (2) labelled often enough
to train a detector, and (3) useful for telling diseases apart.**

### 5.7 Practical
- EyePACS, APTOS, ODIR and RFMiD mirrors exist on Kaggle: attach them as notebook inputs.
- Store processed images at 512 px (~40k images = a few GB; fits Kaggle's 20 GB output).
- Record each licence and citation in `DATA_CARD.md`. Most sets are research-use only.

---

## 6. Preprocessing: "camera signal down, biology intact"

Run once over every image up to step 6. Output: a folder of 512 px images (stored after step 6) + one `meta.parquet`. Steps 7–8 run on the fly in the dataloader, per view, in this order: crop → step 7 (p = 0.5) → artifacts → step 8. See §19.1 for evaluation.

| # | Step | What | Why |
|---|---|---|---|
| 1 | Ingest | Decode RGB, keep originals, read size/EXIF, assign dataset ID + source ID | Source IDs feed mining |
| 2 | Deduplicate | pHash all images; check close pairs by embedding similarity | No test image hides in the pool |
| 3 | Find the retina | Low threshold for a rough outline, then a RANSAC axis-aligned ellipse fit to its edge. The mask comes from the ellipse, never from brightness | Works for round, flat-topped and stretched photos; dark retina is never cut |
| 4 | Standardise geometry | Crop around the ellipse and map it to a circle, which undoes any stretch; resize so retina diameter = 512 px; clean circular mask. No laterality flip (§2.3 item 5); laterality is kept as metadata where available | Lesions similar in pixel size across cameras |
| 5 | Quality tags | Laplacian variance, exposure, centre-vs-edge brightness, contrast → good / usable / reject | Rejects are kept for the quality gate and OOD tests |
| 6 | Fix illumination | Flatten LAB L channel only, mask-aware, strength 0.6 | Removes vignetting without touching colour |
| 7 | Camera colour | Shift each camera's mean/std LAB towards the pool's, alpha 0.7. **Per camera, not per image** | A pale disc stays paler than its neighbours |
| 8 | Mild CLAHE | L channel only, clip limit ~1.5 | Small lesions show; haze stays visible. Colour-sensitive findings can also use the non-CLAHE image |
| 9 | Harmonise labels | Map every label into the vocabulary (§5.6) | One language for probes and graph |
| 10 | Write splits | One Parquet row per image: path, dataset, source, quality, labels, patient, split | Every later step reads only this table |

**Reference code (steps 3, 6, 7):**
```python
import cv2, numpy as np

# find_fov: see data/preprocess.py. The original threshold + enclosing circle failed on real photos (§20).

def correct_illumination(img, mask, strength=0.6):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, m = lab[..., 0], mask.astype(np.float32)
    sigma = img.shape[0] / 8                         # 64 px at 512; S/16 dimmed the disc (§20)
    # dividing by the blurred mask stops the black border darkening the estimate at the edges
    bg = cv2.GaussianBlur(L * m, (0, 0), sigma) / (cv2.GaussianBlur(m, (0, 0), sigma) + 1e-6)
    L_flat = L / (bg + 1e-6) * L[m > 0].mean()
    lab[..., 0] = (1 - strength) * L + strength * L_flat
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out * mask[..., None]

def camera_color(img, mask, cam_mean, cam_std, pool_mean, pool_std, alpha=0.7):
    # cam_* are computed once per camera over all its images (inside the mask), not per image
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    for c in range(3):
        mapped = (lab[..., c] - cam_mean[c]) / cam_std[c] * pool_std[c] + pool_mean[c]
        lab[..., c] = (1 - alpha) * lab[..., c] + alpha * mapped
    out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return out * mask[..., None]
```

**How we know it worked:** before and after preprocessing, train a tiny classifier on simple image
statistics to guess the source dataset. Its accuracy should **drop** after preprocessing. A probe for
disc pallor and media haze should score **about the same**. If it drops, a step is too strong: lower
that step's strength.

---

## 7. Layer 1: the encoder

### 7.1 What it produces
ResNet-50: 384×384 image → 2048-d backbone feature. During pretraining, a projection head
2048 → 2048 → 256 feeds the loss. After pretraining, drop the head and use the 2048-d feature,
L2-normalised, for everything downstream. (Slides say "256-d embedding" as shorthand.)

### 7.2 Why these choices
| Option | Params | Verdict |
|---|---|---|
| ResNet-50, ImageNet start | 25M | **Main model.** Fits one GPU with mixed precision; convolutions suit small lesions and small data |
| ViT-S/16, ImageNet start | 22M | Optional ablation; ViTs usually need more data |
| RETFound (ViT-L) | ~300M | Too big to pretrain; baseline only, frozen, if allowed |
| From scratch | — | Not used; ~40k images is too few |

| Method | Why / why not |
|---|---|
| SimCLR | Needs batches of 1,000+; one GPU fits ~64 |
| BYOL / DINO | No negatives, but our novelty re-weights explicit negatives |
| MAE | Weaker features for similarity search, prototypes and OOD |
| **MoCo v2** | Queue of 16k negatives with small batches; re-weightable; well understood |

### 7.3 Retina-safe views
**Golden rule: an edit is safe only if a doctor would give both copies the same diagnosis.**

| Edit | Standard SimCLR/MoCo | TRACE | Why |
|---|---|---|---|
| Random crop | 8–100% | 60–100%, inside the retina circle | A small crop can cut out the only lesion |
| Colour jitter | Strong (0.4) | Brightness/contrast 0.2, saturation 0.1, hue 0.02 | Colour is evidence (pallor, exudates, haemorrhage) |
| Grayscale | 20% | **Never** | Red lesions vanish in grey |
| Rotation / flip | Rare | ±15°, flip 50% | Anatomy stays valid |
| Blur | Strong | σ 0.1–1.0, 30% of the time | Microaneurysms are a few pixels wide |
| Camera artifacts | None | Vignetting, sampled camera colour profiles, flash glints, eyelash shadows, haze, JPEG 40–95 | Forces the model to ignore the device |
| Multi-crop | None | 2 global views at 384 px + 4 local views at 160 px (15–35% of the retina) | Local views match lesion-scale detail to the whole eye |

**Never use:** grayscale, big hue shifts, crops outside the retina circle, cutout/random erasing
(can erase the lesion), strong blur, **simulated dust spots** (they look like microaneurysms, so the
model would learn "small dark dots don't matter").

**Multi-crop pairing:** 6 views per image per step. Local views are compared **only against global
views** (as keys), never against each other. Positive pairs per image per step: 2 (global↔global) +
4 × 2 (local→global) = 10. Cost ≈ 1.35× the 2-global-view setup (397,312 vs 294,912 pixels).
Views are regenerated every load; nothing is saved to disk.

### 7.4 Cross-source false-negative mining (main contribution)
**The conflict:** the simple label-free fix for false negatives is "stop pushing away negatives that look
very similar". But early on, two photos from the **same camera** look similar because they share tint
and lighting. Naive masking would protect those pairs and **strengthen** camera bias.

**Our rule.** For anchor q from source dᵢ, a queue key j becomes an extra positive ("attracted") only if
**all** of these hold:
1. **Different source:** dⱼ ≠ dᵢ. Cross-camera similarity can't come from camera cues.
2. **Very similar:** sᵢⱼ is in the top 1% of cross-source similarities in this batch (adaptive threshold).
3. **Mutual neighbours:** j is in the anchor's top 10 **cross-source** keys, **and** the anchor is at least as
   close as j's own 10th cross-source neighbour (stored when j entered the queue). Same-source keys are
   masked in both directions, otherwise they crowd the list and attraction silently stops firing.
4. **After warm-up:** only from epoch 20.

Separately, **negatives from the same source get weight β = 2**, so separating same-camera images by
camera cues stops working.

```
P  = { true partner k+ } ∪ { attracted keys }
Z  = Σ_{p∈P} exp(s_p/τ)  +  Σ_{j∉P} w_j · exp(s_j/τ)        w_j = β if same source, else 1
L  = −(1/|P|) · Σ_{p∈P} λ_p · log( exp(s_p/τ) / Z )          λ = 1 for k+, α = 0.5 for attracted keys
```

**Sanity check:** with no attracted keys and β = 1, this is exactly MoCo's InfoNCE. That gives free
ablations: β only, attraction only, both.

**Worked example** (τ = 0.2, β = 2, α = 0.5; anchor = DR eye from camera A):

| Key | s | exp(s/τ) | Plain MoCo | TRACE |
|---|---|---|---|---|
| True partner (same eye) | 0.90 | 90.02 | Pulled | Pulled |
| DR eye, camera B | 0.82 | 60.34 | Pushed, share 29.1% | **Pulled** (attracted, weight 0.5) |
| Healthy eye, camera A | 0.80 | 54.60 | Pushed, share 26.4% | Pushed harder, share 41.7% (β = 2) |
| Glaucoma eye, camera C | 0.15 | 2.12 | Pushed, share 1.0% | Pushed, share 0.8% |
| Z | | | 207.07 | 261.67 |
| Loss | | | 0.833 | 0.900 |

The loss rises; that's expected, because it's a harder objective. The direction of each push or pull is what matters.

**Collapse safety:** attraction is pulling, and pulling drives collapse. The brakes (top 1%, mutual
top-10, 20-epoch warm-up, α = 0.5) exist to prevent it. **Log every epoch:** attracted keys per anchor
and their **label purity** (on RFMiD train, how often an attracted pair shares a disease label;
monitoring only, never training), plus which condition blocks attraction most. Report **lift** = purity ÷ chance,
where chance is the same rate for random cross-source pairs. Exclude normal–normal pairs from **both**, so the
ratio is like for like. If lift < 2 (a judgment call) or purity falls over epochs, tighten the threshold or extend the warm-up.

### 7.5 Reference code sketch: queue + loss (verify and unit-test before use)
```python
import copy, torch, torch.nn as nn, torch.nn.functional as F

class MoCo(nn.Module):
    def __init__(self, encoder, dim=256, K=8192, m=0.999, knn=10):
        super().__init__()
        self.q_enc = encoder                                   # ResNet-50 + MLP head -> 256
        self.k_enc = copy.deepcopy(encoder).requires_grad_(False)
        self.register_buffer("queue", F.normalize(torch.randn(K, dim), dim=1))
        self.register_buffer("queue_src", torch.full((K,), -1, dtype=torch.long))
        self.register_buffer("queue_kth", torch.ones(K))       # each key's 10th-neighbour similarity
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.K, self.m, self.knn = K, m, knn

    @torch.no_grad()
    def momentum_update(self):
        for pq, pk in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            pk.mul_(self.m).add_(pq.detach(), alpha=1 - self.m)

    @torch.no_grad()
    def enqueue(self, k, src):
        cross = (src[:, None] != self.queue_src[None, :]) & (self.queue_src[None, :] >= 0)
        kth = (k @ self.queue.T).masked_fill(~cross, -2.0).topk(self.knn, dim=1).values[:, -1]
        kth = kth.masked_fill(kth < -1, 2.0)                   # too few cross-source keys: can't be attracted
        idx = (torch.arange(k.shape[0], device=k.device) + int(self.ptr)) % self.K
        self.queue[idx], self.queue_src[idx], self.queue_kth[idx] = k, src, kth
        self.ptr[0] = (int(self.ptr) + k.shape[0]) % self.K

def trace_loss(q, k_pos, src, mo, tau=0.2, beta=2.0, alpha=0.5, mine=True, top_frac=0.01):
    q, k_pos = q.float(), k_pos.float()                             # compute the loss in fp32
    pos = (q * k_pos).sum(1, keepdim=True) / tau                    # (B, 1)
    s = q @ mo.queue.T.float()                                      # (B, K) cosine similarities
    valid = mo.queue_src[None, :] >= 0
    same = (src[:, None] == mo.queue_src[None, :]) & valid
    attract = torch.zeros_like(s, dtype=torch.bool)
    if mine:
        with torch.no_grad():
            cross = ~same & valid
            thr = torch.quantile(s[cross], 1 - top_frac)            # adaptive top-1% threshold
            top = torch.zeros_like(attract).scatter_(1, s.masked_fill(~cross, -2.0).topk(mo.knn, dim=1).indices, True)
            mutual = s >= mo.queue_kth[None, :]                     # anchor is in the key's cross-source top-k
            attract = cross & (s > thr) & top & mutual
    w = torch.where(same, torch.tensor(beta, device=s.device), torch.tensor(1.0, device=s.device))
    log_w = torch.log(w).masked_fill(attract, 0.0)                  # attracted keys count as positives
    logits = torch.cat([pos, s / tau + log_w], dim=1)               # + log w  ==  multiply exp() by w
    pad = torch.cat([torch.zeros_like(pos, dtype=torch.bool), ~valid.expand_as(s)], 1)
    logits = logits.masked_fill(pad, -1e9)
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    lam = torch.cat([torch.ones_like(pos), attract.float() * alpha], dim=1)
    loss = -(lam * logp).sum(1) / (1 + attract.sum(1))
    return loss.mean(), attract.sum(1).float().mean()

# one step (§19.4): key encoder on both globals -> k1, k2 (no grad); query encoder on all 6 views.
# loss = mean of 10 pairs: (q1,k2), (q2,k1) with mine=epoch >= warm-up, plus each local vs k1 and k2 with mine=False
# backward -> optimiser step -> model.momentum_update() -> model.enqueue(k2, src)   # view 2 keys only
```
Implementation notes:
- Compute the loss in **fp32**. Under autocast, −1e9 overflows fp16.
- `torch.quantile` has an input-size limit (~16M elements). B×K = 64×16,384 ≈ 1M is fine; subsample if larger.
- **BatchNorm leak:** twins in one batch share BN statistics, so the model can match twins by
  "batch fingerprint". MoCo uses shuffle-BN across GPUs. On one GPU: run the key encoder on sub-batches
  with shuffled order, or simulate shuffling. Test with a baseline that this doesn't hurt.
- Compare our custom loop against a `lightly` MoCo v2 baseline to catch bugs.

### 7.6 Training recipe
| Setting | Value | Note |
|---|---|---|
| Start weights | torchvision ResNet-50 ImageNet | Saves many epochs |
| Batch / queue | 32 / 8,192 on the 4060 with the ~20–25k pool | Keep K ≈ 0.4 × pool (below). K/B = 256 steps |
| Optimiser | SGD, momentum 0.9, weight decay 1e-4 | LR 0.03 × batch/256 (0.00375 at batch 32), cosine decay |
| Temperature / momentum | τ = 0.2 / m = 0.999 | MoCo v2 defaults |
| Mining | warm-up 20 epochs, top 1%, k = 10, β = 2, α = 0.5 | Tune β ∈ {1.5, 2, 3} on validation |
| Resolution | 224 px for experiments, then ~20 epochs at 384 px | Cheap iteration, then lesion detail |
| Epochs | ~100 at 224 px | Stop early if the validation probe plateaus |
| Precision | Mixed (torch.autocast + GradScaler) | ~2× faster on T4 |

**Queue size rule:** an image's own older key can still be in the queue when the image comes round again,
where it acts as a negative (a self-inflicted false negative). Being smaller than the dataset isn't enough.
Simulated share of anchors affected, by pool N and queue K:

| N | K | Own key in queue |
|---|---|---|
| 40,000 | 16,384 | 8.5% |
| 22,500 | 16,384 | 26.4% |
| 22,500 | 8,192 | 6.6% |
| 10,000 | 4,096 | 8.4% |
| 1,920 | 1,024 | 14.1% |
| 1,920 | 768 | 8.1% |

So keep **K ≈ 0.4 × pool**: 8,192 for ~20–25k, 4,096 for the 10k ablation ladder, 768 for RFMiD-only.

**Why m = 0.999:** the oldest key is K/B steps old (8,192/32 = 256). If K/B rises above 256, scale m to hold drift at about 23%, e.g. 0.9995 at 512. Key-encoder drift over 256 steps:
m = 0.999 → 23% (OK); 0.99 → 92%; 0.9 → 100% (fails). Half-gap time at m = 0.999 ≈ 693 steps.

**Compute (measured 2026-11-05 on the RTX 4060 Laptop):** full multi-crop MoCo step, mixed precision, random
tensors, so **no dataloader cost and no sustained-heat test**. These are best-case speeds. Redo with real data in phase 2.

| Batch | Views | Peak memory | Speed |
|---|---|---|---|
| 32 | 224/96 px | 4.41 GB | 108 img/s |
| 48 | 224/96 px | 6.33 GB | 102 img/s (205 steps) |
| 64 | 224/96 px | Over the 6.4 GB cap | — |
| 12 | 384/160 px | 4.72 GB | 36 img/s |
| 16 | 384/160 px | 6.25 GB | 35 img/s |
| 24 | 384/160 px | Over the 6.4 GB cap | — |

**Measured on real data (2026-11-06, plain MoCo, 2 views, 224 px, batch 32): ~125 img/s steady, 3.4 GB peak; one epoch
on the 22.5k pool ≈ 3 min, 100 epochs ≈ 5 h.** The view loader, not the GPU, sets this speed (§20).
Speed is flat across batch sizes, so use batch 32 and keep the headroom. **On Windows, going over budget can page GPU
memory into system RAM and crawl instead of raising an out-of-memory error**, so set
`torch.cuda.set_per_process_memory_fraction`. Best-case times: 22.5k pool, 100 epochs ≈ 6 h; 384 px phase, 20 epochs
≈ 3.5 h; one 10k ablation run, 30 epochs ≈ 48 min. If it doesn't fit, shrink the EyePACS subset first.

### 7.7 Checks while it trains
- Every 5 epochs: **kNN accuracy + linear probe** on RFMiD val (frozen features). This is the main signal.
- **Camera probe:** logistic regression predicting source ID from features. Should fall towards chance.
- **Mining stats:** attracted keys per anchor + label purity.
- **Collapse watch:** loss not stuck at log(#candidates) (log 8,193 = 9.01; log 769 = 6.65);
  per-dimension std of normalised 256-d outputs ≈ 1/√256 = 0.0625, not ~0.
- **UMAP:** coloured by disease (should cluster) and by camera (should mix).

### 7.8 PEFT adapter (§2.3 item 3)
Decided: minimal version. LoRA (r = 8) on layer4's convolutions via the `peft` library (verify Conv2d support on
the installed version), trained with BCE on the 1 / 10 / 100% label fractions and compared against the linear
probe. It covers barrier 04 (PEFT). SupCon only if time is left; say so in the model card. The frozen-encoder
results stay the headline; the adapter is reported separately.

---

## 8. Layer 2: finding detectors

### 8.1 Inputs
Three features from the same frozen encoder:
- **Global:** the whole image.
- **Disc crop:** a square of ~2 disc diameters around the optic disc. The disc = the brightest large blob
  after a heavy blur of the green and red channels.
- **Macula crop:** centred ~2.5 disc diameters from the disc towards the centre, at the darkest region.

Concatenate: 3 × 2048 = **6,144**. This is a heuristic. Check it on 50 images by eye and flag failures
with a quality tag.

### 8.2 Model and loss
MLP 6,144 → 512 → 20, dropout 0.3, sigmoid per output, BCE (multi-label).
```
pos_weight_f = negatives_f / positives_f
example: disc edema has 40 positives in 1,920 images → 1,880 / 40 = 47
```
Without it, the model learns "never say disc edema" and is still 98% accurate.

### 8.3 Calibration and reliability
- Temperature/Platt-scale each finding on val-A ("0.9" must mean right ~90%).
- **Reliability weight r_f:** AUROC on val-A; below ~0.75 → r_f = 0.5; very poor → off.
- **By training positives (decided 2026-11-05; a judgment call):** at least 20 → trained detector; 5–19 → **prototype** (mean feature of the
  positives, cosine); under 5 → dropped from the vocabulary.

Example output: `disc_edema 0.91 · disc_pallor 0.78 · haemorrhage 0.12 · exudates 0.05 · microaneurysm 0.03`

### 8.4 Where finding labels come from
| Source | Gives | Catch |
|---|---|---|
| RFMiD / RFMiD 2.0 | Sign-type labels (ODE, ODP, ODC, TV, CWS, DN, …) | Only some findings |
| IDRiD masks | MA, HE, EX, SE pixel masks | **Held-out test only** in our split plan |
| Own labelling | A few hundred images, doctor-checked | Time, but high value |
| Derived from disease labels | "DR ⇒ probably haemorrhage" | Noisy; avoid or flag |

**Count positives per finding before finalising the vocabulary.** That's the main data risk.

---

## 9. Layer 3: knowledge graph

### 9.1 What it is
Basically a reviewable table, loaded with NetworkX. No graph database is needed.

| Element | Example | Count |
|---|---|---|
| Disease nodes | AION, CRVO, DR, glaucoma, RP… | ~45 |
| Finding nodes | disc edema, pallor, haemorrhage… | ~20 |
| Location nodes | disc, macula, vessels, periphery, whole image | 5 |
| Edge disease → finding | with p(finding \| disease) | ~300 |
| Edge finding → location | disc edema → disc | ~20 |
| Edge disease → family | CRVO → vascular occlusion | ~45 |

### 9.2 File format (`graph/edges.csv`, in git, every edge cited)
```
disease,finding,literature,source
AION,disc_edema,typical,"EyeWiki: AION"
AION,disc_pallor,typical,"EyeWiki: AION"
DR,microaneurysm,required,"ICDR scale"
DR,disc_edema,rare,"textbook"
RP,pigment_clumps,very_frequent,"HPO: Orphanet annotation"
```

### 9.3 Literature → prior μ
| Textbook word | μ | HPO frequency band (HP id, range) | μ |
|---|---|---|---|
| required | 0.95 | Obligate (HP:0040280, 100%) | 0.95 |
| typical | 0.80 | Very frequent (HP:0040281, 80–99%) | 0.90 |
| sometimes | 0.40 | Frequent (HP:0040282, 30–79%) | 0.55 |
| rare | 0.10 | Occasional (HP:0040283, 5–29%) | 0.17 |
| excludes | 0.02 | Very rare (HP:0040284, 1–4%) / Excluded (HP:0040285, 0%) | 0.025 / 0.02 |

- HPO's `phenotype.hpoa` gives frequency in column 8: an HP band id, n/m counts (preferred when present),
  or a percentage. Use n/m directly when available.
- **Clamp every probability to [0.01, 0.99]**, because log(0) = −∞ makes a disease impossible forever.

### 9.4 Update with data (Beta prior, literature worth s = 10 images)
```
a = s·μ,  b = s·(1−μ)
p(f | d) = (n_fd + a) / (n_d + a + b)          n_d = images with disease d; n_fd = of those, with finding f
p(f | ¬d) = (n + 1) / (N + 2)                  from all other images (plentiful)
```
| Edge | μ | Counts (illustrative) | Calculation | p |
|---|---|---|---|---|
| AION → disc edema | typical 0.8 | 5 of 6 | (5 + 8) / (6 + 10) | 0.8125 |
| DR → microaneurysm | required 0.95 | 360 of 400 | (360 + 9.5) / (400 + 10) | 0.901 |
| RP → pigment | very frequent 0.9 | 7 of 9 | (7 + 9) / (9 + 10) | 0.842 |

A rare disease mostly follows the literature, and a common one mostly follows the data.

### 9.5 Scoring: evidence for and against
```
present  = log( p(f|d) / p(f|¬d) )
absent   = log( (1−p(f|d)) / (1−p(f|¬d)) )
E_f(d)   = q_f · present + (1 − q_f) · absent          q_f = calibrated detector probability
score(d) = log prior(d) + Σ_f r_f · E_f(d)
```
- Positive E = evidence for; negative = against. The list of E terms **is** the explanation.
- A missing expected finding counts against the disease. An unsure detector (q = 0.5 with p, p′
  symmetric) gives ≈ 0.
- Example: p = 0.9, p′ = 0.1 → q = 0.9 gives +1.76; q = 0.5 gives 0; q = 0.1 gives −1.76.

**Worked example A** (illustrative, r_f = 1): detected disc edema 0.91, pallor 0.78, microaneurysm 0.03.

| Disease | Finding | p(f\|d) | p(f\|¬d) | present | absent | E |
|---|---|---|---|---|---|---|
| AION (prior 0.002 → −6.215) | disc edema 0.91 | 0.875 | 0.03 | 3.373 | −2.049 | +2.885 |
| | disc pallor 0.78 | 0.80 | 0.04 | 2.996 | −1.569 | +1.992 |
| | microaneurysm 0.03 | 0.05 | 0.15 | −1.099 | 0.111 | +0.075 |
| | **total** | | | | | **−1.263** |
| DR (prior 0.3 → −1.204) | disc edema 0.91 | 0.03 | 0.04 | −0.288 | 0.010 | −0.261 |
| | disc pallor 0.78 | 0.03 | 0.05 | −0.511 | 0.021 | −0.394 |
| | microaneurysm 0.03 | 0.90 | 0.05 | 2.890 | −2.251 | −2.097 |
| | **total** | | | | | **−3.956** |

AION beats DR despite a prior 150× smaller: "disc edema and pallor support AION; missing microaneurysms
argue against DR."

**Worked example B** (shared background p′; priors RP 0.01, DR 0.30, Normal 0.69; q: pigment 0.9,
thin vessels 0.8, haemorrhage 0.1):

| Finding | p(f\|RP) | p(f\|DR) | p(f\|Normal) | p′ |
|---|---|---|---|---|
| Pigment | 0.90 | 0.02 | 0.01 | 0.05 |
| Thin vessels | 0.85 | 0.10 | 0.02 | 0.10 |
| Haemorrhage | 0.02 | 0.85 | 0.01 | 0.20 |

Scores: RP −0.92, DR −3.39, Normal −3.19 → softmax **RP 84.2%**, DR 7.2%, Normal 8.7%.
With pigment unsure (q = 0.5): RP 28.3%, DR 27.5%, Normal 44.2%. One detector swings the answer, which
is why the trust layer exists.

**Honest limitation:** this is naive Bayes. It over-counts correlated findings (e.g. haemorrhage + cotton
wool spots). Mitigations: merge strongly correlated findings into one evidence term; calibrate final
scores (§10.2).

### 9.6 Leave-disease-out rule
When testing zero-shot on disease X: remove all X images from probe training, edge counts and alignment.
X's edges come from literature only. Otherwise the test isn't zero-shot.
**Also remove X's images from the pretraining pool** (decided 2026-11-05), so the encoder never sees them either. All of X's images then
form its test set, which raises n: AION 30, CRVO 57, RP 14. Labels are used only to choose what to leave out; say so in the model card.

### 9.7 Graph–image alignment (skip unless everything else is done by about 13 Nov)
- 2-layer **GraphSAGE** (PyTorch Geometric) over the graph. Disease embeddings come from finding/location
  neighbours with edge weight p(f|d). Output: 256-d per disease.
- Image side: a small projection of the frozen 2048-d feature to 256-d.
- **Sigmoid pairwise (SigLIP-style) loss**: each (image, disease) pair is yes/no. The data is multi-label.
- Zero-shot: a disease with no images still gets an embedding from its findings.
- Compare against the Bayesian scorer, and keep whichever is better.

### 9.8 Fusion (image prototype vs graph)
Convert both to calibrated log-probabilities first, then:
```
score(d) = w_n · image(d) + (1 − w_n) · graph(d),     w_n = n / (n + 10),  n = labelled images of d
n: 0 → 0.00 · 1 → 0.09 · 5 → 0.33 · 10 → 0.50 · 50 → 0.83 · 100 → 0.91
```
image(d) = cosine of the image to the mean feature (prototype) of d's labelled images.
Example (photo 0.25, graph 0.85): rare n = 1 → 0.80 (graph wins); common n = 100 → 0.30 (photos win).

---

## 10. Layer 4: trust (temperature and thresholds fitted on val-A; conformal q̂ on val-B only)

### 10.1 Quality gate
Logistic regression on frozen features, trained on EyeQ Good / Usable / Reject. Reject → ask for a
retake **before** any diagnosis. Tested with Fundus-C (§11).

### 10.2 Calibrated confidence
```
p = softmax(z / T)            T chosen to minimise log-loss on val-A (ranking unchanged)
ECE = Σ_bins (n_b / N) · | accuracy_b − confidence_b |
example: a bin with 100 of 1,000 predictions says 0.9, is right 70% → adds 0.1 × 0.2 = 0.02
```

### 10.3 OOD: "have I seen anything like this?"
FAISS index of all training features (inner product on normalised vectors = cosine). For a new image,
take the distance to its **10th nearest neighbour**. Threshold = **95th percentile** of those distances on
val-A images (5% of normal images flagged). Above the threshold → unfamiliar. The same index returns
the **5 nearest confirmed cases** for the explanation panel.

Note: rare-disease eyes often look unusual, so a referral must carry the graph's short-list
("unusual eye; findings suggest RP").

### 10.4 Conformal prediction sets
1. For each val-B image: score = 1 − p(true disease).
2. Sort n scores; q̂ = the k-th smallest with **k = ⌈(n+1)(1−α)⌉**.
3. New image: set = every disease with 1 − p(d) ≤ q̂.

Example: n = 9, α = 0.1 → k = 9. Sorted: 0.05, 0.10, 0.12, 0.20, 0.25, 0.30, 0.40, 0.55, 0.70 → q̂ = 0.70.
New image: AION 0.52, vein occlusion 0.40, glaucoma 0.05, DR 0.03 → scores 0.48, 0.60, 0.95, 0.97 →
set = {AION, vein occlusion} → short-list.

**Mondrian (per camera):** separate q̂ per camera group, because the guarantee breaks under camera shift.
At a new site, collect ~100 labelled images to calibrate; until then, use the most conservative existing
q̂. RFMiD is multi-label: apply conformal to the primary diagnosis on single-diagnosis images and state
this scope in the model card.

### 10.5 Decision rule (checks run in this order; earlier ones stop later ones)
| Condition | Output |
|---|---|
| Quality gate says reject | Ask for retake |
| OOD distance above threshold | Refer to specialist (unfamiliar), with graph short-list attached |
| Conformal set has exactly 1 disease | Diagnose with calibrated confidence |
| Set has 2–3 diseases | Short-list for clinician review |
| Set empty or > 3 diseases | Refer to specialist |

Every output shows: findings, evidence for/against, the 5 most similar confirmed cases.

### 10.6 Disagreement flag (proposed addition)
If |image score − graph score| > 0.5 for the top disease, **never output a single diagnosis**. At most a
short-list marked "evidence disagrees". Optionally also require ≥ 2 key findings with high reliability
before a graph-driven diagnosis. Tune the 0.5 on val-A: in large-gap cases, count who was right
more often (image or graph).

### 10.7 Example decision card (demo target)
```
Possible retinitis pigmentosa   (short-list: RP, choroideremia)
  ✔ Pigment clumps (reliable detector)   ✔ Thin vessels   ✘ No haemorrhages
  3 most similar past cases: [img] [img] [img]
  ⚠ Evidence disagrees — refer for specialist review
```

---

## 11. Evaluation: how each claim is proved

| Claim | Test | Data | Metric |
|---|---|---|---|
| Encoder transfers | Linear probe + kNN on frozen features | RFMiD test, IDRiD, REFUGE | Macro-AUROC, balanced accuracy |
| Few labels suffice | 1 / 5 / 10-shot; 1 / 10 / 100% labels | RFMiD test | Mean ± 95% CI over 5 seeds |
| Rare diseases zero-shot | Leave-disease-out (e.g. AION, CRVO, RP) | RFMiD / RFMiD 2.0 test, JSIEC | Top-1 / top-5 vs image-only |
| Works on new cameras | Leave-one-camera-out (RFMiD camera groups via resolution) | Paraguay (ZEISS VISUCAM 500), REFUGE halves | In- vs out-of-domain gap. Claim wording in §20 |
| Camera is ignored | Predict source from features | All | Accuracy vs chance |
| Lesions are kept | Inpaint lesions vs random areas vs add artifacts | IDRiD masks | Lesion-vs-nuisance ratio |
| Robust to bad photos | **Fundus-C** (our suite): 7 corruptions × 5 severities | RFMiD test | Mean corruption error |
| Knows when it doesn't know | OOD + conformal + risk–coverage | JSIEC rare, non-fundus, Fundus-C 4–5 | AUROC, FPR@95, coverage, ECE, AURC |
| Explanations faithful | Remove a cited lesion: does the score drop? | IDRiD, REFUGE masks | Score drop; mask overlap |

**Baselines:** frozen ImageNet ResNet-50; plain MoCo v2 on the same pool; supervised ResNet-50 fine-tuned
on RFMiD with class weights; frozen RETFound + prototypes (if allowed); FLAIR zero-shot (if weights usable);
TRACE encoder without the graph.

**Ablations** (each on top of plain MoCo): retina-safe views → artifact simulator → multi-crop → β
re-weighting → cross-source attraction. Also **naive similarity masking**: we expect it to increase
camera leakage, which shows the conflict our method solves.

**Statistics:** 5 seeds for few-shot sampling; 1,000-sample bootstrap CIs for every ours-vs-baseline comparison.

**Required deliverables:** encoder weights, code, `MODEL_CARD.md`, UMAP plots, retrieval demo, ablation table.

---

## 12. Differentiation (what other teams likely won't have)
Most teams will submit SimCLR/MoCo/RETFound + fine-tune + one AUC + Grad-CAM. We win on:
1. **Proof, not claims:** leave-one-camera-out, leave-disease-out, ablations, baselines.
2. **Measured trust:** risk–coverage curve, conformal coverage per camera, ECE.
3. **Organiser stress test:** robustness is core (artifact simulator, per-camera colour, quality gate).
4. **A doctor-usable output:** the decision card (§10.7), not just a heatmap.
5. **ZEISS fit:** new-camera onboarding with per-camera calibration; retake prompt at capture time for
   screening camps; ResNet-50 is small enough to export (ONNX) and run on a clinic PC.
6. **Honest failure analysis slide:** where TRACE fails.

**If time is short, do first:** (1) the two key experiments, (2) the decision-card demo on a hidden
rare disease, (3) the risk–coverage curve.

**Pitch line:** "Other models give an answer. TRACE gives an answer only when it's sure, shows why, and
proved it on a site and a disease it never saw."

**Closest outside competitors:** vision-language models (FLAIR; RET-CLIP; RetiZero, ~342k fundus images
with text, >400 diseases, reports 84.3% top-5 zero-shot on 15 diseases and 75.6% on 52). Weakness:
hidden reasoning, and they need huge photo–text data to build.

---

## 13. Tech stack and repo

| Area | Tool |
|---|---|
| Language | Python 3.11 |
| Deep learning | PyTorch 2.x, torchvision, timm |
| Contrastive baselines | lightly (MoCo v2 / SimCLR) |
| Images / augmentation | OpenCV, Albumentations, Pillow, custom artifact simulator |
| Dedup | imagehash (pHash) |
| Data | pandas + Parquet (single `meta.parquet`) |
| Graph | NetworkX; PyTorch Geometric (GraphSAGE) |
| Search / OOD | FAISS (CPU) |
| Heads / calibration / metrics | scikit-learn |
| Conformal | Own module (~50 lines) |
| Analysis | umap-learn, Matplotlib |
| Config / tracking | Hydra, Weights & Biases |
| Demo | Gradio |
| Compute | Local RTX 4060 Laptop first; Kaggle / Colab for the 384 px phase and overflow; mixed precision |
| Code | Git + GitHub; Makefile targets |

The trained model and demo run on a laptop CPU at a few hundred ms per image.

```
trace/
├── configs/               # Hydra: data.yaml, pretrain.yaml, probes.yaml, graph.yaml, trust.yaml
├── data/
│   ├── ingest.py          # read every dataset -> raw metadata, source IDs
│   ├── dedupe.py          # pHash + embedding duplicate check
│   ├── preprocess.py      # FOV, geometry, illumination, camera colour, CLAHE, quality tags
│   ├── ontology.yaml      # every dataset label -> finding / disease
│   └── splits.py          # writes meta.parquet
├── augment/
│   ├── retina_views.py    # retina-safe crops, multi-crop
│   └── artifacts.py       # vignetting, colour profiles, glints, lashes, haze, JPEG
├── encoder/
│   ├── moco.py            # queue, momentum encoder
│   ├── loss.py            # InfoNCE + cross-source mining
│   └── train.py
├── findings/
│   ├── localise.py        # disc + macula crops
│   └── probes.py          # MLP, pos_weight, calibration, reliability
├── graph/
│   ├── edges.csv          # reviewed medical knowledge, cited per edge
│   ├── build.py           # Beta-smoothed edge probabilities
│   ├── score.py           # evidence for / against
│   └── align.py           # GraphSAGE + sigmoid alignment (stretch)
├── trust/
│   ├── calibrate.py  ood.py  conformal.py  decide.py
├── eval/
│   ├── probes.py  fewshot.py  leave_disease_out.py  leave_camera_out.py
│   ├── fundus_c.py  lesion_test.py  faithfulness.py  risk_coverage.py  report.py
├── demo/app.py            # Gradio: upload → findings → reasons → cases → decision
├── tests/                 # unit tests (section 16)
├── MODEL_CARD.md  DATA_CARD.md  README.md
└── Makefile               # make data / pretrain / heads / eval / demo
```

---

## 14. Build order (each phase has a "done when")

| Phase | Work | Done when |
|---|---|---|
| 0 Setup | Repo, Kaggle notebook template, Hydra, W&B, datasets attached | A dummy run logs to W&B from a config |
| 1 Data | Ingest, dedupe, preprocess, ontology, splits; **count labels per finding and per rare disease** | `meta.parquet` covers every image; contact sheets look right; "before" camera-probe number recorded |
| 2 Baselines | Frozen ImageNet features + probes; short plain-MoCo run; time 200 steps | Baseline table filled; compute plan redone with real timings |
| 3 TRACE encoder | Retina views, artifacts, multi-crop, mining; ablations one at a time | Beats plain MoCo on few-shot AUROC or the camera probe, with CIs |
| 4 Findings + graph | Localiser, probes, calibration, edges.csv, Bayesian scorer, then (stretch) GNN | Leave-disease-out top-5 reported vs image-only |
| 5 Trust | Quality gate, temperature, FAISS OOD, per-camera conformal, decision rule, disagreement flag | Coverage + set size per camera; OOD AUROC; risk–coverage curve |
| 6 Package | Gradio demo, UMAPs, model card, data card, README, one-command reproduce | A teammate reproduces results from a fresh clone |

Phases 4–5 can start on baseline ImageNet features while phase 3 trains.

**Owners (team roles):** Data (§5–6, artifact simulator, Fundus-C, data card) · Encoder (§7) ·
Reasoning (§8–9; get a clinician or optometry student to review `edges.csv`) · Trust + eval + demo
(§10–11, Gradio app, model card).

---

## 15. Risks and mitigations
| Risk | Mitigation |
|---|---|
| Compute runs out | Time steps on day one; 224 px; shrink EyePACS subset; start phases 4–5 on baseline features |
| Rare classes have only a handful of images | Pool RFMiD 2.0 + JSIEC; report per-class counts; prototypes for tiny classes |
| Rare **findings** lack labels | Count first; own labelling; drop findings that can't be trained |
| Graph edges wrong | Cite per edge; clinician review; edge-noise test (±20%) |
| Mining pulls wrong pairs | Warm-up, mutual neighbours, purity logging, α < 1 |
| Collapse / dimensional collapse | Monitor loss vs log(#candidates), per-dim std, kNN probe |
| Own keys in the queue | K ≈ 0.4 × pool: 8,192 at ~22k, 768 for RFMiD-only (§7.6) |
| BatchNorm leak | Shuffle-BN or equivalent; verify |
| Conformal is single-label, data multi-label | Primary diagnosis only; state scope |
| Duplicates leak into tests | Dedupe; report count removed |
| Naive-Bayes overconfidence | Merge correlated findings; calibrate |
| Novelty claims | Literature check before calling anything "new" |

---

## 16. Unit-test targets (verified numbers)
| Test | Input | Expected |
|---|---|---|
| InfoNCE, easy negatives | τ 0.2; sims 0.8 (pos), 0.3, 0.1 | Σexp = 60.73; p = 0.899; loss = 0.106 |
| InfoNCE, look-alike | sims 0.8, 0.75, 0.1 | Σexp = 98.77; p = 0.553; loss = 0.593 |
| InfoNCE, lesson dry run | cos 0.9397 (pos), 0.7660, 0.0, −0.8660 | loss = 0.357; gradients (−1.50, +1.47, +0.03, 0.00) |
| `trace_loss` reduces to InfoNCE | no attraction, β = 1 | equals plain InfoNCE |
| `trace_loss` worked example | §7.4 table (pos 0.90; attracted 0.82; same-source 0.80; other 0.15) | Z: 207.07 → 261.67; loss 0.833 → 0.900; same-source share 26.4% → 41.7% |
| Momentum update | θk = 1, θq = 2, m = 0.999 | 1.001, 1.002, 1.003; after 1,000 steps 1.632 |
| Beta smoothing | (7, 9, μ 0.9, s 10) | 0.842 |
| Evidence term | p 0.9, p′ 0.1, q 0.9 / 0.5 / 0.1 | +1.76 / 0 / −1.76 |
| KG example B | §9.5 | RP 0.842, DR 0.072, Normal 0.087 |
| Fusion weight | n = 0, 1, 5, 10, 50, 100 | 0, 0.09, 0.33, 0.50, 0.83, 0.91 |
| Conformal k | n = 9, α = 0.1 | k = 9; q̂ = 0.70; set {AION, vein occlusion} |
| ECE bin | 100/1000, conf 0.9, acc 0.7 | 0.02 |
| pos_weight | 40 pos in 1,920 | 47 |
| Collapse loss | 16,385 candidates | log = 9.70 |

---

## 17. Existing materials (already made)
- Team brief PDF (6 pages), pitch deck (11 slides: cover, problem, insight, architecture, encoder,
  reasoning, trust, impact, data, plan, close), 158 s pitch video (team names removed).
- TRACE Build Guide (Claude doc) and the "TRACE Explained From Zero" explainer doc.
- Idea submission done: 150–200-word description + tech stack.

## 18. Sources
- ZEISS Beyond Code 2026 Themes & Brief; Campus Hackathon kick-off deck (16 Oct 2026).
- RFMiD (IEEE DataPort); RFMiD 2.0 (Zenodo 7505822); JSIEC-1000 (Zenodo 3477553); Paraguay DR (Zenodo 4891308);
  EyeQ (GitHub); IDRiD; REFUGE; ODIR-5K; APTOS 2019; EyePACS (Kaggle).
- HPO frequency annotations: https://obophenotype.github.io/human-phenotype-ontology/annotations/frequency
- RetiZero: https://arxiv.org/abs/2406.09317 · FLAIR: https://arxiv.org/abs/2308.07898 · RET-CLIP: https://arxiv.org/abs/2405.14137
- MoCo (He et al. 2019), MoCo v2 (Chen et al. 2020), SimCLR (Chen et al. 2020), alignment & uniformity
  (Wang & Isola 2020), false-negative cancellation, domain-aware contrastive learning, UIOS open-set retinal anomaly identification.
- Verify every dataset count, licence and record ID after download.

---

## 19. Implementation specs (build-level details; follow these unless I say otherwise)

### 19.1 Encoder interface (fix this early; the organisers run the encoder on hidden data)
```python
# trace/encoder/export.py
class TraceEncoder(nn.Module):
    """Frozen TRACE backbone. Input: float tensor (B, 3, H, W), RGB in [0, 1], already
    TRACE-preprocessed (§6: steps 3–6 and 8; step 7 only for a known camera). ImageNet mean/std normalisation is applied INSIDE.
    Output: (B, 2048) L2-normalised features. Size-agnostic (global average pool); default 384×384."""

def load_trace_encoder(weights_path: str, device="cpu") -> TraceEncoder: ...
def preprocess(img_rgb_uint8: np.ndarray, source_id: str | None = None,
               stats_path="trace_preprocess.json") -> torch.Tensor: ...   # returns (3, 384, 384) in [0, 1]
```
- Backbone = torchvision `resnet50` with `fc = nn.Identity()`. Save the backbone `state_dict` only (`encoder.pt`).
- `trace_preprocess.json` stores the pool LAB mean/std, per-camera LAB mean/std, and all step parameters.
- **Fixed evaluation policy:** known camera → apply step 7; unknown → skip it. Training applies step 7 per view with p = 0.5, so skipping it isn't a new distribution shift.
- **IDRiD:** the headline number uses the unknown-camera policy; "RFMiD Kowa stats applied" is a second row.
- **Unknown camera (e.g. the hidden test):** skip step 7 (camera colour) and rely on learned invariance.
  Per-image colour normalisation is not allowed (it erases pallor). Dataset-level stats from unlabelled
  target images are an ablation only, pending a mentor answer ("no target tuning" may forbid it).
- Optional: ONNX export of the backbone for the CPU demo.

### 19.2 Feature cache (how stages B–E swap encoders)
- `features/<encoder_name>/<split>_global.npy`: float16, (N, 2048), L2-normalised.
- `features/<encoder_name>/<split>_disc.npy`, `<split>_macula.npy`: same shape; rows for failed
  localisation are zeros, with `loc_ok = False` in the index.
- `features/<encoder_name>/<split>_index.parquet`: `image_id` in row order, `loc_ok`.
- `encoder_name ∈ {imagenet_r50, moco_plain, trace_v1, …}`. Every head, graph, trust and eval script reads
  features only through `encoder_name` in the config. Swapping encoders = changing one config value.

### 19.3 Kaggle checkpoint / resume (sessions time out, at about 12 h; check the current limit)
- Save every epoch to `/kaggle/working/ckpt/last.pt`, plus `epoch_XXX.pt` every 10 epochs. Keep the last 2.
- The checkpoint contains: `q_enc`, `k_enc`, **`queue`, `queue_src`, `queue_kth`, `ptr`**, optimiser,
  GradScaler, LR scheduler, `epoch`, `global_step`, RNG states (python, numpy, torch, cuda), resolved config.
- Across sessions: upload `last.pt` as a W&B artifact (or a new Kaggle dataset version) at the end of each
  session; the next session attaches it and calls `--resume`. Resuming must reproduce the LR schedule exactly.
- Before a long run, test resume: train 50 steps, save, reload, and check the loss continues smoothly.

### 19.4 Multi-crop loss and the 384 px phase
- Key encoder sees **both globals** (no gradients). Enqueue only view 2's B keys per step, so the oldest key is K/B steps old.
- Query encoder sees both globals + 4 locals. Total loss = **mean over the 10 (query, key) pairs**. The
  key for a global query is the other global's key; local queries use both global keys. Mining
  (`mine=True` after warm-up) applies to both global–global pairs; local pairs use `mine=False`, β still applies.
- Optional 5-epoch linear LR warm-up.
- Phase 1: ~100 epochs, globals 224 px, locals 96 px. Phase 2 (Kaggle/Colab preferred for a larger BatchNorm batch; locally at batch 12; skipped if time is short): ~20 epochs, globals 384 px, locals 160 px,
  LR restarted at 0.1× base with cosine decay; mining stays on; keep the queue (it refreshes within K/B steps).
- Seeds: set python/numpy/torch seeds and log them; `cudnn.benchmark = True` is acceptable (speed over bit-exactness).

### 19.5 Artifact simulator (`augment/artifacts.py`); each artifact is applied independently per view
| Artifact | Prob | Parameters (at 512 px retina diameter) |
|---|---|---|
| Camera colour profile | 0.5 | Inverse of §6 step 7 towards a randomly sampled stored camera's LAB stats, alpha 0.3–1.0 |
| Vignetting | 0.3 | Multiply by (1 − s·r²), r = normalised radius, s 0.1–0.4 |
| Uneven illumination | 0.3 | Smooth linear gradient in a random direction, ±5–15% brightness |
| Haze (cataract-like) | 0.2 | out = (1−h)·img + h·(0.9·blur(img, σ 8–20) + 0.1), h 0.1–0.4 |
| Flash glint | 0.15 | 1–3 soft bright ellipses, radius 2–6% of diameter, +0.2–0.5 intensity |
| Eyelash shadow | 0.1 | 1–3 blurred dark curved bands at the top/bottom rim, opacity 0.1–0.3 |
| JPEG | 0.3 | Quality 40–95 |
| **Dust spots** | **never** | They mimic microaneurysms |

Everything stays inside the FOV mask. Check visually: save a contact sheet of 64 augmented views per
setting and review it before training. If a lesion becomes invisible, the setting is too strong.

### 19.6 Fundus-C robustness suite (`eval/fundus_c.py`): 7 corruptions × 5 severities, fixed seed
| Corruption | Severity 1 → 5 |
|---|---|
| Defocus blur (disk radius px) | 1, 2, 3, 4, 6 |
| Motion blur (kernel length px) | 3, 5, 9, 13, 17 |
| Gaussian noise (σ, image in [0, 1]) | 0.02, 0.04, 0.06, 0.08, 0.10 |
| Under-exposure (gamma) | 1.3, 1.6, 2.0, 2.5, 3.0 |
| Over-exposure + glare (brightness ×) | 1.1, 1.2, 1.3, 1.4, 1.5 |
| Haze (h) | 0.1, 0.2, 0.3, 0.4, 0.5 |
| JPEG (quality) | 60, 40, 25, 15, 10 |

Applied to RFMiD test **after** preprocessing. Severities 4–5 exceed the training ranges on purpose;
state the overlap with training artifacts honestly. Metric: error = 1 − macro-AUROC of the frozen
linear probe; CE_c = Σ_s E(model) / Σ_s E(ImageNet-R50 baseline); **mCE = mean over corruptions**.
Severity 4–5 images also serve as OOD / quality-gate test inputs.

### 19.7 Evaluation protocol details
- **Features:** frozen, L2-normalised global 2048-d for transfer tests; 6144-d for findings.
- **Linear probe:** `StandardScaler` (fit on train) + `LogisticRegression` (L2, one-vs-rest for multi-label),
  C ∈ {0.01, 0.1, 1, 10} chosen by patient-level 5-fold CV on train (not val), `class_weight="balanced"`, `max_iter=2000`.
- **kNN:** k = 20, cosine, similarity-weighted votes (weight = exp(s / 0.07)).
- **RFMiD multi-label:** macro-AUROC over labels with ≥ 5 test positives (list the excluded labels),
  plus binary AUROC on the `Disease_Risk` column.
- **Few-shot:** K ∈ {1, 5, 10} per class from RFMiD train, 5 seeds; skip classes with < K images at that
  K. Report prototypes (mean feature, cosine) and logistic regression; mean ± 95% CI over seeds.
- **Label fractions 1 / 10 / 100%:** multi-label stratified sampling (iterative stratification, e.g. `iterstrat`).
- **Leave-one-camera-out:** fit probes on two RFMiD camera groups, test on the third; also fit on all
  RFMiD and test on Paraguay and REFUGE. Report gap = in-domain − out-of-domain macro-AUROC.
- **Camera probe:** logistic regression predicting source ID, 5-fold CV, balanced accuracy vs chance (1 / #sources).
- **OOD:** in-distribution = RFMiD test; OOD = JSIEC classes outside the training vocabulary, a few hundred
  non-fundus natural images (e.g. STL-10 via torchvision), Fundus-C severity 4–5. AUROC + FPR@95%TPR.
- **Risk–coverage:** sort by calibrated max probability; error at coverage 100% → 10% in 10% steps; AURC.
- **Bootstrap:** 1,000 resamples of test images for every ours-vs-baseline difference.

### 19.8 Knowledge-graph details not covered above
- **Disease priors:** prior(d) = (n_d + 1) / (N + D) from training prevalence, then floored at
  **π_min = 0.005** and renormalised. Zero-image (zero-shot) diseases get π_min. Pick π_min on val-A;
  never let a rare target's prior sit so low that strong evidence can't win (check with worked example B).
- **Normal node:** a disease node whose lesion-finding edges are 0.01 (clamped), updated with data counts.
  Without it, healthy eyes get forced into a disease.
- **Missing detector output** (finding switched off, or localisation failed for disc/macula findings):
  the term is skipped (E = 0), not treated as absent.
- **Drafting `edges.csv`:** Claude Code may draft edges from sources it actually opens (HPO annotations,
  EyeWiki, review papers). Every row needs a `source` and a `status` column (`draft` / `reviewed`).
  **Never invent a medical fact without a source.** Only `reviewed` rows are used in reported results.

### 19.9 Preprocessing thresholds
- Quality tags (§6 step 5): don't hard-code. Fit thresholds on Laplacian variance, exposure and contrast to
  best match EyeQ Good / Usable / Reject on EyeQ images. **Changed 2026-11-06: applied to EyePACS only** (§20).
- FOV failure (no blob, no usable ellipse, or radius < 25% of the image width): tag `reject` with reason `fov_fail`.
- Keep `fit` (share of outline points on the ellipse) and `ry/rx` in `meta.parquet`. Review low-fit photos on a contact sheet; flat-cut photos score low normally (median 0.67 on the 1504×1000 group).

### 19.10 Demo (`demo/app.py`, Gradio, CPU)
Input: one fundus image + optional camera choice. Output panels:
1. Quality status (and a retake message if rejected).
2. Preprocessed image with disc and macula boxes.
3. Top findings with calibrated probabilities.
4. Disease ranking: top 3 evidence-for and evidence-against terms each.
5. Conformal set + decision (diagnose / short-list / refer), plus OOD and disagreement flags.
6. The 5 most similar confirmed cases (thumbnails + labels) from the FAISS index.

Loads `encoder.pt`, `trace_preprocess.json`, probe weights, `edges.csv`, calibration values and the FAISS index from `artifacts/`.

### 19.11 Config sketch (`configs/pretrain.yaml`)
```yaml
encoder_name: trace_v1
data: {meta: data/meta.parquet, pool_splits: [pool], img_root: processed/}
model: {arch: resnet50, init: imagenet, proj_dims: [2048, 2048, 256]}
moco: {K: 8192, m: 0.999, tau: 0.2}           # K ≈ 0.4 × pool: 768 if RFMiD-only, 4096 for the ladder
mining: {enabled: true, warmup_epochs: 20, top_frac: 0.01, knn: 10, beta: 2.0, alpha: 0.5}
views: {global_px: 224, local_px: 96, n_local: 4, global_scale: [0.6, 1.0], local_scale: [0.15, 0.35]}
optim: {name: sgd, lr_base: 0.03, momentum: 0.9, wd: 1.0e-4, batch: 32, epochs: 100, warmup_epochs: 5}
amp: true
ckpt: {dir: checkpoints/, every: 1, keep: 2, resume: null}   # /kaggle/working/ckpt on Kaggle
eval_every: 5
seed: 0
```

### 19.12 Environment (`requirements.txt`; pin exact versions with `pip freeze` after the first working install)
```
torch>=2.2
torchvision>=0.17
timm>=0.9
lightly>=1.5
albumentations>=1.4      # check the API if 2.x is installed
opencv-python-headless>=4.9
pillow>=10
imagehash>=4.3
pandas>=2.1
pyarrow>=15
numpy>=1.26
scikit-learn>=1.4
iterative-stratification>=0.1.7
faiss-cpu>=1.7.4
networkx>=3.2
torch-geometric>=2.5     # only for the GNN stretch goal
peft>=0.10               # LoRA adapter (§7.8); verify Conv2d support
umap-learn>=0.5
matplotlib>=3.8
hydra-core>=1.3
wandb>=0.16
gradio>=4.0
pytest>=8
```

### 19.13 Makefile targets
`make data` (ingest → dedupe → preprocess → splits) · `make features ENC=<name>` · `make pretrain` ·
`make heads` (probes + calibration) · `make graph` · `make trust` · `make eval` · `make demo` · `make test`.

### 19.14 Cards (reproducibility is 15% of the score)
- **MODEL_CARD.md:** intended use (screening support, not diagnosis); training data; architecture; metrics per
  dataset and camera; calibration and conformal scope (primary diagnosis only); known failure modes;
  compute used; licence.
- **DATA_CARD.md:** every dataset with licence, citation, counts before and after dedupe, camera/source IDs,
  split rules, and duplicates removed.

### 19.15 Short ablation ladder (§20)
All 7 ablation runs: ~10k images from at least 3 sources (RFMiD train, RFMiD 2.0 train, APTOS, an ODIR subset), 30 epochs at
224 px, queue 4,096, mining warm-up 6 epochs. Run plain MoCo and full TRACE with 2 seeds each to measure noise; don't
over-read differences smaller than the seed spread. Label the table "short budget". Full-length headline: plain MoCo vs
full TRACE, same pool, epochs and seed.

### 19.16 Ask me instead of guessing
Dataset paths and download status · real label counts · mentor answers (§1.4) · the rare target diseases ·
any medical fact without a source · any change to a locked decision (§2.2).

---

## 20. Decision log (2026-11-05)
Reviewed answers to the open design questions. The sections above already reflect these; this records why.

| Topic | Decision | Reason |
|---|---|---|
| Compute | Local RTX 4060 first, then Kaggle/Colab | Nithin's call. Replaces the locked Kaggle-first choice |
| Multi-crop keys | Both globals through the key encoder; enqueue view 2 only | The 10 pairs need a key for each global; one queued key per image keeps K/B staleness correct |
| Validation | Val-A tunes, val-B sets conformal q̂, C uses CV on train | Temperature and q̂ must not see the same data. Coverage SD ≈ 3% at n = 100, 4% at n = 50. Camera groups under 50 use the most conservative q̂ |
| Pool | ~20–25k headline; RFMiD-only second row | RFMiD-only (1,920 images) is likely weak, and the second row protects us if external data is ruled out |
| Ablations | Short ladder + 2 full runs | 7 full runs cost too much; the main claim still needs one matched full comparison |
| Mining | Cross-source top-10 in both directions; lift instead of a fixed 70% | Stops same-camera keys silently blocking attraction; purity means little without chance |
| Pseudo-domains | Completeness ≥ 0.9, no size features | Splitting a camera is the dangerous error; merging only loses attraction. 0.9 is a judgment call |
| Camera colour | Step 7 per view with p = 0.5 | Per view (not per image) so twin pairs teach the invariance |
| Laterality | No standardisation; keep 50% flip | Most datasets lack laterality, and the flip already covers it |
| Illumination blur | sigma = image size / 8 (64 px at 512), strength 0.6 | Synthetic 80 px disc: sigma 32 kept 68% of disc contrast, 64 keeps 91%; vignette gap 37 becomes 16.8 vs 21.0. Confirm on real photos |
| ZEISS claim | Reworded, below | The pool holds other ZEISS images |
| Rare targets | At least 10 test images, 20+ preferred; below 10 = case study | 95% Wilson interval at 50%: 0.24–0.76 (n = 10), 0.30–0.70 (n = 20). If counts fall short, a held-out disease's train/val images can raise n, but the encoder saw them unlabelled: report those separately |

**Real-data findings, RFMiD 2.0 (official training folder only, 509 photos, 2026-11-05):**
- **Stretched photos:** 313 of 509 (61%) are 512×512 with ry/rx ≈ 1.31, consistent with 4:3 originals squeezed into a square, so discs are tall ovals. Step 3–4 now undoes this. Check every other dataset for the same thing.
- **Retina finding:** the original threshold + enclosing circle shrank many of these photos by about 30%. A brightness mask instead cut out dark retina. Hence the ellipse fit, which had 0 failures on this set.
- **Label counts (train / val / test):** AION 2/1/1, CRAO 1/0/0, RP 2/1/1, CRVO 7/3/2, ODE 11/5/4, ODP 10/4/4. These are too few on their own, so RFMiD and JSIEC counts decide the targets.
- **Patient IDs:** none in the RFMiD 2.0 label files, so near-duplicate grouping is needed (§5.4). No per-image camera column either.

**Real-data findings, RFMiD + RFMiD 2.0 (4,060 images, 2026-11-05):**
- **Target counts, RFMiD + RFMiD 2.0 (test splits / all splits):** AION 5 / 30, CRVO 11 / 57, RP 3 / 14, CRAO 0 / 5. Only CRVO reaches the bar of 10 test images on the test splits alone.
- **Finding positives in train splits (RFMiD + RFMiD 2.0):** MH 343, ODC 303, TSLN 206, DN 142, ODP 75, ODE 69, LS 59, HR 57, EDN 51, RT 38, MS 33, CME 27, RPEC 24, CWS 22, TV 15, MHL 15, TD 12, ST 7, PRH 7, MCA 6, CF 6, ME 1. No column named NV or EX exists in either dataset, so §5.1 and §5.6 need correcting. HR is 57 in RFMiD 2.0 train but 0 in RFMiD train, so check that it means the same thing in both before merging.
- **Duplicates (`data/dedupe.py`):** pHash is useless here (6,863 pairs at Hamming ≤ 4, because the black border and layout dominate), so we match fine retinal detail. 31,204 pairs checked; 47 at score ≥ 0.8, 506 at ≥ 0.6. At least 19 pairs link RFMiD test to RFMiD 2.0 (13 train, 3 val, 3 test), all normal eyes. **136 of 4,060 images dropped (3.3%)**, and no kept image matches a kept image in another split. **No AION, CRVO, RP or CRAO image was dropped**, and none of those images share a same-eye group, so the counts below are unique eyes.
- **What the scores mean:** ≥ 0.8 is usually the same eye, but not always (1 of 8 sampled was a clear false match). In 0.6–0.8, most sampled pairs do not look like the same eye, and 421 of 459 are from the same camera. They probably match on fixed camera marks (sensor dust, a lens ring), though that comes from looking at the sheets and is not proven. Dropping them is deliberately cautious, because it only protects held-out splits. Same-eye photos that score lower can still slip through, so splits are not proven patient-level.
- **Possible camera fingerprint:** if fixed marks are confirmed, they let a model identify the camera. Check with the camera probe (§7.7) before relying on the artifact simulator to remove this.
- **Windows memory:** do not import torch at module level in code that worker processes load. 14 workers loading the GPU libraries exhausted the page file (WinError 1455). This laptop has 15.7 GB RAM, so keep DataLoader workers modest and measure.
- **Label disagreements on the same eye:** 3 real ones among the 36 pairs, all "no labels" vs one label: RFMiD train 1754 vs val 601 (ODP), train 1877 vs test 541 (DN), train 1886 vs val 550 (MH). RFMiD's empty labels and RFMiD 2.0's WNL both mean normal.
- **ID mismatches, RFMiD 2.0:** images train 222, 79 and test 251, 291, 404, 489 have no label row, and label rows train 89, 216 and test 151, 191, 405, 490 have no image. They look like typos, but they are not matched by guesswork: `label_ok = False`, excluded from labelled sets.
- **Sources:** RFMiD 2.0's 2048×1536 group merges into `topcon_trc_nw300` because they share photos. 1504×1000 and 512×512 stay as size groups.
- **Retina finding, RFMiD train:** 0 failures in 1,920, median fit 0.98, ry/rx 1.00 for all three cameras, so RFMiD is not stretched.

**Real-data findings, JSIEC + APTOS (2026-11-06; 8,722 images in total):**
- **Ingest:** JSIEC 1,000 images in 39 class folders, one class each, all `test` (held out); IDs keep the folder number because file names
  are not unique across folders. APTOS 3,662 images, all `pool` (its own train/val/test kept as `aptos_split`); DR grade kept as `aptos_dr0`–`4`,
  monitoring only. APTOS `test.csv` has 366 rows; a line count says 868 because of blank lines. Every image now has a `size_group` column.
- **Retina finding:** 0 failures on all 4,662. APTOS 819×614 (133 photos) is squashed, ry/rx = 0.949; the ellipse step undoes it.
  JSIEC 2100×1592 (8 photos, test only) has a bright white line along the bottom edge, inside the mask. Not fixed; note it in the data card.
- **APTOS sizes match RFMiD cameras:** 2144×1424 (28), 4288×2848 (52), 2048×1536 (351). A shared size does not prove a shared camera, but if
  it is the same camera, `aptos` vs `kowa_vx10a` would be a false cross-source pair, the dangerous error (§5.5). APTOS is one source
  (`aptos`) for now; the pseudo-domain step must check this before mining uses it.
- **Dedupe fix:** the shortlist took each image's top 30 over all images, so 3,662 APTOS look-alikes pushed one real RFMiD ≥ 0.8 pair out
  (47 → 46). It now takes the top 30 inside each dataset; the 47 pairs are back. EyePACS would have made this much worse.
- **Dedupe result:** 102,763 pairs checked; 320 of 8,722 dropped (APTOS 176 of 3,662, 4.8%; JSIEC 7 same-class copies, 4 of them exact).
  No kept image matches a kept image of another split rank. No AION, CRVO, RP, CRAO, JSIEC CRVO, RP or RAO image dropped.
  The APTOS–RFMiD matches (744 at 0.6–0.8, none ≥ 0.8) looked like different eyes on the sheet, normal eyes with the same layout; JSIEC–RFMiD
  matches were blurry photos. Cautious drops, cheap because APTOS is pool only. Expect ~5% of EyePACS to go the same way.
- **Rare-target test counts with JSIEC:** CRVO 11 + 22 = 33; RP 3 + 22 = 25 (RP now passes the 10-image bar). JSIEC "RAO" (16) means
  retinal artery occlusion, which may include branch occlusions, so it is not counted as CRAO without checking.

**Real-data findings, IDRiD + EyePACS + EyeQ (2026-11-06; 44,427 images in `raw_meta.parquet`):**
- **IDRiD:** part C's 516 photos are byte-identical to part B's, so they are read once (`idrid`, labels `idrid_dr0`–`4`, `idrid_dme0`–`2`,
  plus `od_x/od_y/fovea_x/fovea_y` from C, none missing). Part A's 81 lesion-mask photos are `idrid_seg` (labels = masks present: MA 81,
  EX 81, OD 81, HE 80, SE 40); only 3 are byte-identical to B photos. All IDRiD is `test`, one source `idrid`. All photos are 4288×2848,
  the same size as RFMiD's Kowa VX-10α group.
- **Dedupe rule for two test sets:** a same-eye copy inside one dataset's test split is dropped; a copy shared by two different test sets
  (IDRiD grading vs IDRiD lesion masks, JSIEC vs RFMiD test) is kept in both, linked by `dup_group`, because they are scored in different
  tests. Pooled counts (leave-disease-out across datasets) must count each `dup_group` once.
- **EyePACS:** Kaggle 2015 train only, in two versions of the same photos. "Cropped" (35,108) keeps ~50% more fine detail than "resized"
  (35,126) at the same retina size (59 of 60 sampled), so it was made from the originals: we use cropped only, never both. 17,561 patients;
  file names give patient and eye (`patient`, `laterality` columns). All `pool`, one source `eyepacs`; DR grade kept as `eyepacs_dr0`–`4`,
  monitoring only.
- **EyeQ:** only `Label_EyeQ_train.csv` matches our photos (12,543 rows, all EyePACS train; DR grades agree with Kaggle's 100%). The test
  file (16,249) grades EyePACS *test* photos, which we do not have. 0/1/2 = good/usable/reject, checked by eye. Stored as `eyeq_quality`
  on 12,529 EyePACS rows: good 8,347, usable 1,876, reject 2,306; 22,579 rows ungraded. So the quality gate trains and is tested on a
  patient-level split of these 12.5k, and §19.9 thresholds fitted on them grade the rest.
- **Pool size (to decide at `splits.py`):** pool-eligible images now exceed the planned ~20–25k (EyePACS alone is 35k). The EyePACS subset
  size is set so the pool lands in that range, which keeps K = 8,192 and the 6 h budget (§7.6).
- **Dedupe at 44k (`scripts/dedupe_audit.py`):** only non-pool images search (pool-pool pairs can never cause a drop); 50 neighbours per
  dataset, 150 in a dataset over 5,000 images. 197,520 pairs, 12.5 min with 6 workers. **1,470 of 44,427 dropped** (EyePACS 1,050 = 3.0%,
  APTOS 193, RFMiD train 134, RFMiD val 30); no kept image matches a kept image of another split rank. 2 EyePACS photos failed retina
  finding (`eyepacs_36416_right`, `eyepacs_41176_left`). Regression vs the 8,722-image run: no ≥ 0.8 pair lost except pool-pool ones
  (skipped on purpose); 3 earlier drops are RFMiD 2.0 test copies of RFMiD test photos, now kept by the two-test-sets rule; 3 APTOS photos
  matched only at 0.60–0.68 (mostly look-alikes) and are missed now that pool photos do not search.
- **IDRiD photos are inside RFMiD.** 84 RFMiD photos match IDRiD at ≥ 0.8, 60 of them exact copies (score > 0.99), almost all from RFMiD's
  Kowa VX-10α group (IDRiD's camera, which also confirms §5.5's size-to-camera mapping): RFMiD train 60, val 9, test 15
  (corrected 14 Nov: the earlier "76 exact; train 72, val 9, test 22" counted pairs, 103 in all, not photos). The organisers'
  flow (pretrain on RFMiD, test on IDRiD) would leak these; ours drops the train and val copies. **Tell the mentor.** RFMiD test ↔ IDRiD
  copies are two test sets, so both are kept, linked by `dup_group`.
- **Same eye, different diagnosis:** `rfmid_train_1282` (RFMiD: CRVO) is the same eye as `idrid_test_057` (IDRiD: DR grade 4, DME 2),
  confirmed by eye. Dropped from train. Labels can disagree across datasets; the knowledge graph and evaluation must allow for it.
- **Kept after dedupe:** RFMiD train 1,786 / val 610 / test 639; RFMiD 2.0 train 474 / val 168 / test 171; JSIEC 993; IDRiD 508 + 81;
  APTOS 3,469; EyePACS 34,058. Rare targets: only the one CRVO above was dropped (RFMiD CRVO 45 → 44).

**Phase 1 completion (2026-11-06; `data/meta.parquet`, 45,864 rows; [DATA_CARD.md](DATA_CARD.md)):**
- **Paraguay** (1,437, `test`, source `paraguay`): 7 grade folders, all agreeing with the spreadsheet. Mostly 2124×2056; JSIEC has 11 photos
  at that size too. It holds the same eye photographed more than once: 251 same-eye pairs, 203 copies dropped; in 45 pairs the copies
  have different DR grades.
- **Blank-photo bug fixed:** a fully white EyePACS photo has no detail, and its correlation came out as 1.0 against 23 different Paraguay
  eyes, chaining them into one same-eye group. `match_score` now returns 0 when a signature's std < 0.3 (real photos ≥ 0.77). Re-run:
  only the 26 pairs involving that photo changed. Some ≥ 0.8 matches are still different eyes whose bright discs line up (seen on
  EyePACS–Paraguay pairs at 0.80–0.82); they cause cautious drops, not leaks.
- **Processing pass** (`data/process.py`, 29.5 min, 6 workers): steps 3, 4, 6 for every photo, saved as 512 px RGBA PNG (alpha = mask) in
  `processed/` (14 GB). 2 retina failures. LAB conversions now round instead of truncating (the spec's reference code truncated,
  darkening every pixel by about half a level).
- **Quality grades: EyePACS only (deviation from §19.9).** The depth-3 tree fitted to EyeQ gets 75–78% accuracy and 54–71% reject recall
  on held-out EyePACS patients, but called 33% of RFMiD, 38% of RFMiD 2.0 and 61% of Paraguay "reject"; on the contact sheet many were
  readable, some with clear lesions or media haze (an RFMiD label). Its main threshold, Laplacian sharpness, depends on camera resolution
  and compression. So it only orders EyePACS into the pool; other datasets are `ungraded` and unfiltered. The §10.1 quality gate on
  encoder features is the real gate.
- **Splits** (`data/splits.py`, `configs/data.yaml`): pool 22,500 (EyePACS 16,918 = EyeQ good 8,136 + EyeQ usable 1,829 + tree good 6,953;
  APTOS 3,387; RFMiD train 1,735; RFMiD 2.0 train 460). Head training 2,193. Val-A 377 / val-B 378. Test 3,626. Leave-disease-out
  photos: AION 30, CRVO 78, RP 36. Findings in head training: 14 with ≥ 20 positives (detectors), 6 with 5–19 (prototypes), 0 dropped.
  `label_conflict` flags same-eye copies within a dataset with different labels (kept: Paraguay 26, RFMiD 7, IDRiD 6).
- **Ontology** (`data/ontology.yaml`): 20 findings, 14 diseases, meanings now quoted from both RFMiD papers (PDFs in `raw/papers/`,
  not committed). HR = haemorrhagic retinopathy in both. **MCA = macroaneurysm in RFMiD but microaneurysm in RFMiD 2.0**, so only
  RFMiD 2.0's MCA maps to microaneurysm. RFMiD 2.0's CSC = cysticercosis; HTN = hypertensive retinopathy. MNF and CNV are undefined
  in both papers and unused. The RFMiD paper's per-camera class counts (Table 3) match our size-based camera IDs exactly for 43 of
  46 classes. Val-A / val-B frozen in `data/val_split.csv` (a mapping change had swapped 204 photos between the halves).
- **Step 7 / 8 code:** `camera_color`, `clahe_l`, `pooled_stats` in `data/preprocess.py`; per-camera stats from training-side photos only in
  `data/trace_preprocess.json` (7 sources).
- **Phase 1 exit check, camera probe** (`scripts/camera_probe.py`): dataset guessed from simple photo statistics, 8 datasets (chance 0.125):
  raw 0.687 → after steps 3–6 0.614. 4 training datasets (chance 0.25): 0.805 → 0.717 → 0.493 after step 7. Biology probes on RFMiD:
  disc pallor AUROC 0.615 → 0.634, media haze 0.753 → 0.794. Camera signal falls, the findings do not.
- **Still open from Phase 1:** REFUGE and ODIR ingest when downloaded (ODIR: one source, per the §5.5 fallback below). The pseudo-domain
  check was settled in Phase 3 (below).

**Phase 2 (2026-11-06):**
- **Encoder interface** (`encoder/export.py`, §19.1): `TraceEncoder`, `load_trace_encoder`, `preprocess`; ImageNet normalisation inside;
  step 7 only for a camera with stored statistics. Feature cache `encoder/features.py` (§19.2): 26,962 photos in 7 min.
- **ImageNet ResNet-50 baseline** (`eval/probes.py`, `outputs/eval/baselines.csv`): RFMiD linear macro-AUROC 0.885 [0.863, 0.905] over 18
  labels (CRVO left out: leave-disease-out), kNN 0.759, Disease_Risk 0.963; IDRiD any-DR 0.865, referable 0.903, 5-grade balanced accuracy
  0.394; Paraguay any-DR from an IDRiD-trained probe 0.863. **Camera probe on its features: 0.971 balanced accuracy over 7 training
  sources (chance 0.143)**: ImageNet features identify the camera almost perfectly, the problem TRACE targets.
- **MoCo core** (`encoder/moco.py`, `encoder/loss.py`): reproduces every §16 number and the §7.4 worked example (0.833 → 0.900). Shuffle-BN
  on one GPU = key encoder on shuffled sub-batches.
- **Windows RAM:** torch DataLoader workers each committed ~1.7 GB (~500–690 MB resident); 6 of them left ~400 MB free and training
  crawled. `augment/loader.py` replaces them with workers that never import torch (~100 MB resident each); views travel as uint8.
  Views are fixed by (seed, epoch, batch), so runs are reproducible and resume mid-epoch exactly: stop at step 100 + resume to 200 matched a
  straight run within 0.003 loss (GPU arithmetic order only).
- **View cost** (one core): PNG decode 14 ms, CLAHE 16 ms (now once per photo, not per view), jitter cut from 14 ms to a few ms.
- **Plain MoCo v2, short budget** (15 epochs, full 22.5k pool, standard MoCo v2 views, β = 1, no mining; 47 min, final loss 5.27, per-dim
  std 0.0625, no collapse): RFMiD linear macro-AUROC **0.917** [0.898, 0.935] vs ImageNet 0.885 [0.863, 0.905]; kNN **0.851** vs 0.759;
  Disease_Risk 0.970 vs 0.963; Paraguay any-DR from an IDRiD probe **0.902** vs 0.863; IDRiD any-DR 0.859 vs 0.865, referable 0.906 vs 0.903,
  5-grade balanced accuracy 0.348 vs 0.394 (103 test photos; noisy). The intervals overlap, so a paired bootstrap on the same test photos is
  still needed before calling the linear-probe gain significant. **Camera probe 0.960 vs 0.971: plain MoCo does not remove camera signal.**
  That is the gap TRACE's views, colour step and β weighting must close (barrier 03), and it is the comparison to report.
- **Plain MoCo v2, full 100 epochs** (2026-11-07; 4 h 55 min, 133 img/s, 3.45 GB, final loss 4.75, per-dim std 0.0626, no collapse):
  RFMiD linear **0.892** [0.864, 0.917] (15 ep: 0.917); kNN 0.819 (0.851); Disease_Risk 0.965 (0.970); IDRiD any-DR **0.887** (0.859),
  referable 0.920 (0.906), 5-grade 0.410 (0.348); Paraguay any-DR from IDRiD **0.819** (0.902); camera probe 0.969 (0.960).
  **Longer is not better on RFMiD or Paraguay**; it gains on IDRiD (DR, the pool's majority task: EyePACS is 75% of it). The epoch must
  be chosen on **val-A** (in-loop kNN/probe, §7.7), never by comparing these test numbers. Camera signal is untouched at 100 epochs.
- **Val-A selection** (`eval/val_select.py`, in `encoder/train.py` every `eval_every` epochs; 377 val-A photos, 17 finding/disease labels
  with ≥ 5 positives; probe trained on head-training; ~60 s a check): ImageNet 0.883 (kNN 0.787); 15-epoch run at epoch 10 0.914,
  epoch 15 0.914 (kNN 0.833); 100-epoch run at epoch 90 0.909, epoch 100 0.908 (kNN 0.822). Val-A agrees with test: no gain from 100
  epochs. The 0.006 gap is small for 377 photos. The 15-epoch run had its own (shorter) cosine schedule, so it is not epoch 15 of the long run.
  **C = 0.01, the smallest value in the §19.7 grid, won every time**, so the best C may lie below the grid.
  A 1-epoch smoke run (whole cosine schedule in one epoch) already reached val-A 0.9085 (kNN 0.836), the same as 100 epochs: with
  plain MoCo almost all of the gain over ImageNet comes in the first epoch. Why is not tested (one guess: BatchNorm adapting to fundus photos).

**Phase 3 (2026-11-07):**
- **Retina-safe views** (`augment/views.py`, policy `retina`): square crop of 60–100% of the retina's area (side 351–454 px at R = 256),
  centre within R − side/2 of the retina centre (views ≥ 75% retina); ±15° rotation + flip in one anti-aliased resample; step 7 with
  p = 0.5; artifact hook; jitter 0.2/0.2/0.1/0.02 with p = 0.8 (contrast mean taken inside the mask); blur σ 0.1–1.0 with p = 0.3; step 8
  last, CLAHE tiles = 8 × side / 512 so a tile covers the same retina patch as at evaluation. Retina geometry comes from the photo size
  (step 4 makes the retina circle span the width; mask area underestimates it on flat-cut photos). 7.6 ms a view on one core; training
  with 2 retina views runs at ~130 img/s, the same as plain. Contact sheet `scripts/view_sheet.py`: lesions survive in every view.
- **Artifact simulator** (`augment/artifacts.py`, `views.artifacts: true`; §19.5 probabilities and ranges, lengths rescaled to the
  retina's diameter in the view). Two choices §19.5 leaves open, made after the per-artifact contact sheet:
  (1) **glints only on the outer rim (70–95% of the radius)**: a glint is 2–6% of the diameter (~20–60 px at 512), the size of a cotton
  wool spot at the posterior pole, the same trap as dust (a judgment call, no source); (2) **the colour profile starts from the view's
  actual colour**: its own camera's LAB stats, or after step 7 the exact blend (1 − α)·camera + α·pool (step 7 is linear). Starting from
  the pool's stats when step 7 had not run made dark photos nearly black. Views with artifacts cost ~10 ms; training stays at
  ~125 img/s. On the random-mix sheet about 1 view in 56 (dark Kowa photo, colour + vignette together) hides most lesion detail: watch it.
- **Multi-crop** (`views.n_local: 4`; `encoder/loss.py: multicrop_loss`, equal to the explicit 10-pair loop in value and gradients):
  locals use the same retina recipe at 96 px and 15–35% of the retina's area (side 176–268 px at 512; ≥ 95% of every local is retina,
  2,000 draws). Each photo's locals are drawn after its globals, so `n_local: 0` reproduces the two-view runs exactly. The query encoder runs
  the 4B locals as one separate batch (own BatchNorm statistics, own size). 600-step real-data run: GPU alone 103 img/s (130 without locals);
  loader-bound at 66 img/s with 6 workers, **84 img/s steady with 10** (12 ran out of RAM); 4.54 GB peak. Loss peaks near 9.0 at step
  ~256 (= K/B, the queue has just filled with real keys) and then falls; local loss above global (8.34 vs 7.94 at step 600); no collapse.
- **Pseudo-domain check** (§5.5; `scripts/pseudo_domains.py`, `outputs/pseudo_domains.json`):
  (1) **k-means fails, so the fallback holds: one source per dataset** (EyePACS, APTOS, later ODIR). On RFMiD without image size, k = 3–8,
  Topcon 3D OCT-2000 never keeps more than 0.59 of its photos in one cluster (bar 0.9), with or without mask-shape features.
  (2) **APTOS photos at a known camera's size**: two-sample AUROC (5-fold, same-eye copies in one fold) on photo statistics of the
  *processed* photo, i.e. cues that survive preprocessing. References: same camera, same dataset 0.60–0.78; same camera model, other
  dataset (IDRiD vs RFMiD Kowa) 0.70; same camera, other dataset (TRC-NW300, RFMiD vs RFMiD 2.0) 0.94 [0.912, 0.961]; different cameras
  ≈ 1.00. Clashes: 2144×1424 vs OCT-2000 1.000, 2048×1536 vs TRC-NW300 0.992, 1504×1000 vs RFMiD 2.0's group 0.992: distinct, kept
  separate. **4288×2848 vs Kowa 0.967 [0.953, 0.982] overlaps the same-camera reference, so its 52 photos share Kowa's mining ID**
  (the safe error; a judgment call). Mining reads the new `mine_source` column (`configs/data.yaml: mine_merge`); step 7 keeps `source`,
  because those photos' colour differs from RFMiD's Kowa photos. Encoder features can't settle this (IDRiD vs Kowa already 0.97: they pick up
  population and processing too), and the dedupe score is not a camera fingerprint (APTOS 1050×1050, a size no RFMiD camera has, matches
  TRC-NW300 and Kowa as often as same-size groups do). RFMiD 2.0's 512×512 and 1504×1000 groups are distinct from TRC-NW300 and from each
  other (≥ 0.996). Limit: "separable" does not prove no shared cue, so the mining step must log attracted pairs by (anchor source, key source).
- **Mining monitor** (`encoder/mining_monitor.py`; `outputs/train/<name>/mining.csv` + `mining_sources.jsonl`, per epoch once mining is on).
  Labels only measure, never train. Purity two ways: findings/diseases on labelled RFMiD / RFMiD 2.0 pool photos (normal–normal left out;
  chance 0.059), and DR on every pool photo (EyePACS / APTOS grade ≥ 1 or RFMiD "DR"; no-DR pairs left out; chance 0.182). Also the
  condition that blocks candidates, same-eye share, and attracted pairs per (anchor source, key source) vs the queue's make-up, with purity
  per source pair (high ratio + low purity would mean a camera cue). The queue now stores each key's pool row (`queue_idx`).
- **Mining check run** (`trace_check`: full TRACE config, mining from epoch 2 instead of 20, 7 epochs, 83 img/s with the monitor): attraction
  fires, 2.4–2.8 keys per anchor, 57–67% of anchors; of ~33 top-1% candidates per anchor, 80% fail the top-10 condition and ~82% the mutual
  one. **Purity 0.29–0.32 vs chance 0.059: lift 4.9–5.4** (bar 2), flat over epochs 2–6; DR lift 1.43 → 1.57; same eye 0%; per-dim std
  0.062, no collapse. The global loss drops when mining starts (6.95 → 5.08) because attracted keys become positives, not collapse.
  Sources: pairs among the small non-EyePACS sources are attracted 2–6× more than the queue's make-up predicts (spread over many pairs, while
  all of them attract EyePACS keys at 0.2–0.9×); the highest ratios involve RFMiD 2.0's two small size groups but are < 0.5% of attracted
  pairs. Per-pair purity was added after this run to tell biology from a camera cue. Mining started on a 2-epoch encoder here; the real
  run starts it at epoch 20.
- **GPU crashes on the full run (trace_v1, 8 Nov):** `CUDNN_STATUS_EXECUTION_FAILED` in backward at epoch 4 after 23 min, then again
  ~2 min after resuming. GPU at 86–87 °C (its target temperature, 3 °C from the slowdown limit), ~210 s of software thermal slowdown
  logged; no driver-reset event in the Windows log. Likely heat (multi-crop + 10 CPU workers on a laptop with shared cooling), not proven.
  Mitigation: `ckpt.every_steps: 300` (mid-epoch `last.pt`, resume is exact), 8 workers, and a restart loop (90 s cool-down, up to 30
  tries; `outputs/trace_v1_restarts.txt`). After a crash `log.csv` repeats the steps since the last save (keep the last row per step), and
  that epoch's mining row covers only the steps after the resume.
- **Later crashes were RAM, not heat (9 Nov):** 5 more crashes, about hourly, were `MemoryError` in a loader worker. Windows commit was
  63–64 GB of a 65 GB limit: training 16.3 GB (main process ~10 GB, 8 workers ~0.8 GB each), VS Code 11.8 GB, Chrome 9.3 GB,
  Claude/ChatGPT apps 5.3 GB. No crash after the other apps were closed (06:07). Close other apps before a long run. The validation checks
  at epochs 35, 45, 55 and 65 were lost because crashes hit them.
- **trace_v1 full run (8–9 Nov; 100 epochs, mining from epoch 20; ~80 img/s; per-dim std 0.0624, no collapse):**
  val-A linear/kNN: ep 0 0.883/0.787, 10 0.917/0.815, **20 0.919/0.842 (best)**, 30 0.919/0.834, 40 0.914/0.843, 60 0.914/0.833,
  80 0.917/0.831, 100 0.914/0.831. **The val-selected checkpoint (epoch 20) was measured before mining started**: mining never
  raised val-A. Mining: 3.0–3.3 attracted keys per anchor, same eye 0%; label lift 5.6 → 4.7 and DR lift 1.78 → 1.61, falling
  slowly but above the bar of 2 (§7.4 says tighten or extend warm-up if purity falls).
  **Test** (`outputs/eval/baselines.csv`): trace_v1 (= epoch 20) RFMiD linear **0.922** [0.906, 0.936], kNN 0.829, Disease_Risk 0.973,
  IDRiD any-DR 0.871, referable 0.902, 5-grade 0.376, Paraguay 0.860, **camera probe 0.970**. Epoch 100: 0.905 [0.876, 0.927], kNN 0.807,
  0.968, 0.862, 0.906, 0.334, 0.827, camera 0.978. Matched against plain MoCo at 100 epochs: linear 0.905 vs 0.892, kNN 0.807 vs 0.819,
  IDRiD 0.862 vs 0.887, Paraguay 0.827 vs 0.819, camera 0.978 vs 0.969. **The camera probe did not fall for any encoder** (ImageNet 0.971):
  the 7 sources also differ in population and disease mix, so this probe may be measuring more than the camera. Next: a cleaner
  RFMiD-only 3-camera probe, leave-one-camera-out (§19.7), the paired bootstrap, and the ablation ladder to find out what mining does.
- **Paired bootstrap** (`eval/compare.py`, 1,000 resamples of the same test photos; `outputs/eval/compare.json`), trace_v1 minus:
  ImageNet RFMiD +0.037 [+0.023, +0.052]; plain MoCo 100 ep RFMiD +0.031 [+0.006, +0.058], Paraguay +0.041 [+0.016, +0.065], IDRiD
  −0.016 [−0.056, +0.017]; plain MoCo 15 ep RFMiD +0.005 [−0.011, +0.023] (not significant), Paraguay **−0.043 [−0.064, −0.023]**.
  So TRACE v1 beats ImageNet and the matched-length plain run, but not the short plain run.
- **The camera is visible in coarse colour and light layout (9 Nov):** a clean probe on RFMiD's 3 cameras (same hospital; 5-fold
  grouped CV) gives 0.990–0.998 for every encoder, ImageNet included, and 0.997–0.999 on normal eyes only (no disease to give it
  away). Pixel diagnostics: a 16×16 thumbnail of the evaluation image identifies the camera at 0.995, still 0.990 with every photo
  cut to the same outline (Kowa photos are flat-cut, h/w 0.83; both Topcons are full circles), and fine-detail energy alone at 0.959.
  Camera *detectability* is therefore not a reachable target, and an encoder probe at ~1.0 does not show harm. **The camera claim
  must be measured as a performance gap** (leave-one-camera-out, §19.7; new-site transfer), not as probe accuracy.
- **But the camera does shape the feature map (9 Nov; scratchpad `camera_neighbours.py`):** on RFMiD (3,032 photos), the share of each
  photo's 20 nearest neighbours (cosine, same-eye copies excluded) from its own camera, with chance 0.653, is: ImageNet 0.969, plain MoCo
  15 ep 0.940, plain 100 ep 0.958, trace_v1 (ep 20) 0.921, **trace_v1 ep 100 0.868**. On normal eyes only it's 0.876–0.968, so this isn't
  disease mix. When a diseased photo's neighbour comes from another camera, it shares a label as often as a same-camera neighbour does
  (trace_v1 0.488 vs 0.491). TRACE's ep 20 → 100 drop is the first sign that the 80 mining epochs did change something, even though val-A
  didn't move. **This risks the distance-based parts:** kNN OOD (§10.3) and "5 similar cases" retrieval are camera-driven, and a new
  camera's photos will look unfamiliar. Measure the leave-one-camera-out gap and per-camera OOD before trusting them.
- **384 px phase (`configs/pretrain_384.yaml`, `trace_v1_384`, started 9 Nov):** starts from the val-selected epoch-20 backbone
  (`model.init_from`, new head, empty queue); batch 12, m 0.99963 (same ~23% key drift at K/B = 683), LR 0.03 × 12/256 with a
  1-epoch warm-up, 20 epochs, ~30 img/s (≈ 4.4 h), 4.86 GB. **Mining off** (deviation from §19.4): epoch 20 was chosen before mining
  started, so this continues the recipe that produced it (β = 2, no attraction) and measures resolution alone. The epoch-0 val-A
  check reproduces epoch 20 (0.9194).
- **384 px result (9 Nov; 4.3 h, 1 crash = CUDA out of memory at 13:31, probably another app taking GPU memory; resumed exactly):**
  val-A linear/kNN by epoch: 0 0.919/0.842, 5 0.925/0.841, **10 0.928/0.846 (best → `trace_v1_384`)**, 15 0.924/0.851,
  20 0.925/0.850. Test: RFMiD linear 0.917 [0.897, 0.935], kNN 0.838, Disease_Risk 0.977, IDRiD any-DR 0.871, referable 0.909,
  5-grade 0.352, Paraguay 0.886, RFMiD 3-camera probe 0.995, same-camera neighbours 0.914. Paired bootstrap, trace_v1_384 minus:
  trace_v1 RFMiD −0.006 [−0.019, +0.006], **Paraguay +0.027 [+0.015, +0.039]**; ImageNet RFMiD +0.032 [+0.011, +0.049], Paraguay
  +0.023 [+0.001, +0.046]; plain 100 ep Paraguay +0.067 [+0.045, +0.090]; plain 15 ep RFMiD −0.001, Paraguay −0.016 [−0.035, +0.001].
  So 384 px helps transfer to a new site and leaves RFMiD unchanged. It still doesn't beat the short plain run, and the gap to it
  on Paraguay is now not significant (it was −0.043).
- **One-photo demo, not a result (9 Nov):** an internet RP photo (Hamel 2006, Orphanet J Rare Dis, Wikimedia, CC BY 2.0) is not in our
  data (best vessel match 0.23 over all 45,862 photos, mirrored too). Its cosine to JSIEC's 39 class means (gallery only, nothing
  fitted) ranks RP **1st** for trace_v1_384 (0.918 vs BRVO 0.892), 2nd for plain MoCo 15 ep (CRVO first), and 30th for ImageNet. kNN
  votes favour pathological myopia for all three (TRACE: myopia 29%, BRVO 19%, RP 17%). All three call it abnormal (p = 1.00). RFMiD
  has no pigment label, so the top finding probes are ARMD, MYA and BRVO. The 10th-neighbour OOD check calls it *familiar* (0.914 vs
  0.906), so a rare eye can pass the OOD gate; §10.3's referral can't be the only safety net.
- **JSIEC few-shot** (`eval/fewshot.py`, `outputs/eval/fewshot.json`; 993 photos, 39 classes, support and query never the same eye,
  5 seeds; balanced top-1, chance 0.026): prototype K = 1 / 5 / 10: trace_v1_384 **0.403 / 0.628 / 0.716**, plain 15 ep 0.384 / 0.611 / 0.679,
  plain 100 ep 0.346 / 0.559 / 0.628, ImageNet 0.297 / 0.536 / 0.610. Logistic K = 1 / 5 / 10: 0.420 / 0.677 / 0.782 vs 0.415 / 0.669 / 0.764,
  0.357 / 0.610 / 0.716, 0.337 / 0.620 / 0.713. TRACE is best in all 6 settings: +0.06 to +0.11 over ImageNet, +0.004 to +0.037 over plain
  15 ep (seed intervals overlap with the latter). The logistic model's standardiser is fitted on 5,000 unlabelled pool photos: fitted on the
  support photos, it inflated noise dimensions at K = 1 (the toy test caught it). RP top-1 at K = 5 / 10: TRACE 0.75 / 0.93, ImageNet
  0.72 / 0.80; at K = 1 ImageNet is better on RP (0.57 vs 0.39).
- **Disc / macula localiser** (`findings/localise.py`, §8.1; IDRiD landmarks mapped into processed space by `findings/landmarks.py`,
  `data/idrid_landmarks.parquet`). Developed on IDRiD *train* landmarks only (IDRiD train already fits probes in our DR protocol), reported
  once on IDRiD test. Disc = brighter than its surroundings at disc scale, 39 px inside the edge; fovea = darker than its surroundings,
  1.8–3.2 DD from the disc, within 35° of horizontal on the photo-centre side; DD = 78 px (median disc–fovea distance 196 px / 2.5).
  Each step's train numbers are in the module docstring. **IDRiD test (103): disc within 0.5 DD 99.0%, fovea 84.5% (within 1 DD 89.3%)**,
  median errors 0.12 / 0.14 DD. By eye on 24 photos from 8 other sources: disc 23/24, macula ~18/24 (misses slide towards the darker rim,
  e.g. EyePACS). Next: a weak-spot `loc_ok` flag so bad macula crops are skipped (§19.8), not used.
- **Leave-one-camera-out** (`eval/leave_camera_out.py`, `outputs/eval/leave_camera_out.json`; RFMiD, in-domain training size-matched,
  3 draws; linear C = 0.01). Linear macro-AUROC, in-domain → unseen camera (gap):
  Kowa held out (128 test photos, 6 labels): trace_v1_384 0.940 → **0.921** (+0.019 [+0.006, +0.033]); plain 15 ep 0.917 → 0.903 (+0.014);
  plain 100 ep 0.899 → 0.881 (+0.018); ImageNet 0.883 → 0.867 (+0.016).
  Topcon OCT-2000 held out (438 test photos, 11 labels, only 308 training photos): trace_v1_384 0.879 → **0.858** (+0.021 [+0.001, +0.040]);
  trace_v1_e100 +0.041; plain 15 ep 0.860 → 0.809 (**+0.051** [+0.031, +0.071]); plain 100 ep +0.049; ImageNet 0.815 → 0.796 (+0.019).
  TRC-NW300 (73 photos, 2 labels): intervals too wide to read. Disease_Risk gaps are small and mixed. **So an unseen camera costs
  ~0.02 AUROC; TRACE 384 has the best unseen-camera score on both readable cameras, and on OCT-2000 its gap is under half plain
  MoCo's.** Not yet a head-to-head paired test of the gaps. kNN gaps are larger (Kowa +0.04 to +0.06 for the SSL encoders).
  **Camera centring: rejected.** It lowers the same-camera neighbour share (trace_v1_384 0.908 → 0.814, chance 0.642) but doesn't
  improve unseen-camera kNN: Kowa 0.880 → 0.887, OCT-2000 0.833 → 0.798. Camera means also carry each camera's disease mix.
- **Two cheap probe ideas tested on val-A and rejected (10 Nov; trace_v1_384, 377 photos):** (1) Head2Toe-style multi-layer
  features (average-pooled layer outputs, each block L2-normalised, concatenated): layer4 only 0.9278 (kNN 0.846); layer3+4 0.9293
  (0.853); layer2+3+4 0.9275; all four 0.9277. The best gain, +0.0015, is far inside the noise of 377 photos. (2) Wider C grid
  (0.0003–0.1): CV on train picks 0.003, val-A 0.9263 vs 0.9278 at 0.01. So the §19.7 grid and the 2048-d feature stay.
- **Short-schedule retrain started (10 Nov; `configs/pretrain_short.yaml`, `trace_s20`):** trace_v1's first-20-epoch recipe (β = 2, no
  attraction) with the cosine schedule finishing at epoch 20: the fair match for the 15-epoch plain run, and the reference row for two
  literature ideas to test as ablations (Fourier low-frequency amplitude mixing across cameras; cross-source nearest-neighbour positives).
- **Short-schedule result (10 Nov; 1 h 20 min, no crash): no gain.** val-A linear/kNN by epoch: 5 0.916/0.808, 10 0.914/0.814,
  15 0.913/0.832, 20 0.913/0.838. Below trace_v1's epoch 20 (0.919/0.842, long schedule) and level with plain 15 ep (0.914/0.833).
  So the "learning rate never decayed" guess was wrong. Every SSL run now sits at val-A 0.91–0.93, with run-to-run differences
  about the size of the noise on 377 photos: the linear probe on RFMiD is saturated for this backbone and pool.
- **Encoder frozen (10 Nov):** `trace_v1_384` is final. Its weights and features are read-only; Phase 4 works on a byte-identical copy
  named `trace_v2` (`checkpoints/trace_v2/encoder.pt`, `features/trace_v2/`). Nothing in Phase 4 changes encoder weights.
- **Crop features** (`findings/crops.py`): disc (2 DD) and macula (3 DD) crops cut from the whole photo after steps 7–8 at 512 px,
  resized to 384 px; 6,657 labelled photos, no hard localisation failure. On IDRiD test the disc crop contains the true disc centre
  in 99.0% of photos and the macula crop the fovea in 93.2%. A confidence score for crops was tried and dropped (same for right and
  wrong guesses), so `loc_ok` marks hard failures only.
- **Finding detectors** (`findings/probes.py`, `outputs/eval/findings_trace_v2.json`, `artifacts/trace_v2/findings.joblib`; calibrated
  probabilities in `features/trace_v2/<bucket>_findings.parquet`). 2,193 head-training photos, 14 detectors + 6 prototypes.
  **Three deviations from §8, with reasons:** (1) scored by out-of-fold AUROC on head-training (5 folds, grouped by eye) as well as
  val-A, and Platt + reliability use OOF + val-A pooled, because val-A has 2–13 positives for most findings (haemorrhage 5, CWS 4).
  (2) **Input routed by anatomy, not 6,144 for all:** all three views for every finding gave no gain (OOF 0.9169 vs 0.9169), but the
  matching crop helps the finding that lives there (logistic OOF: disc pallor 0.739 → 0.786, disc edema 0.961 → 0.973; macular scar
  0.866 → 0.907 on the macula crop alone, drusen 0.862 → 0.850). Rule fixed by the ontology's location: disc findings read global +
  disc crop, all others global; macula crop unused (mixed). (3) Reliability floor 0.60 for "off" (a judgment call).
  Result (routed + MLP, OOF macro 0.920 vs global 0.917; logistic 0.918): disc_edema 0.976, haemorrhage 0.961, retinal_traction
  0.957, tessellation 0.957, media_haze 0.952, laser_scars 0.947, hard_exudates 0.937, cotton_wool_spots 0.932, macular_edema 0.921,
  disc_cupping 0.918, tilted_disc 0.886 (prototype), rpe_changes 0.862, tortuous_vessels 0.857 (prototype), drusen 0.854,
  macular_scar 0.850, microaneurysm 0.834 (prototype, 5 positives), disc_pallor 0.812, choroidal_folds 0.783; r_f = 0.5 for
  macular_hole 0.696 and optociliary_shunt 0.654. **Go / no-go: every finding AION and CRVO need is ≥ 0.81, so both go through the graph.**
- **Sources read for edges (10 Nov):** EyeWiki NAION (reviewed 10 Aug 2026) and CRVO (edited 9 Nov 2026). The NAION page carries a notice
  against copying its content into AI programs: store no EyeWiki text; cite the primary studies it draws on, pending Nithin's decision.
  AION is time-dependent (acute = disc edema, later = pallor), so the graph must not require both at once.
- **Knowledge graph, zero-shot: a negative result (10 Nov; `graph/edges.csv`, `graph/build.py`, `graph/score.py`,
  `eval/leave_disease_out.py`, `outputs/eval/leave_disease_out_trace_v2*.json`).** 12 cited `draft` edges (AION 6, CRVO 6; primary
  papers with DOIs where PubMed abstracts supported the fact, EyeWiki in our own words otherwise). 9 candidate diseases (7 data
  diseases with ≥ 5 head-training photos, edges from out-of-fold detector outputs; AION, CRVO from literature), π_min 0.005 (val-A
  false alarm 0%). **RFMiD labels the disease, not its signs** (1 of 30 AION photos also labelled disc_edema; 3 of 57 CRVO photos
  labelled haemorrhage), so data edges use detector outputs, and the planned oracle test is impossible.
  Version 1 (literature μ used directly) and version 2 (μ put on the detector's scale: μ·sens + (1 − μ)·fpr, from head-training only)
  both give **top-1 0 of 30 AION and 0 of 78 CRVO** (top-3: AION 0.30 → 0.33, CRVO 0.00); almost all are called normal. Stopped there:
  further changes would be tuned on the held-out photos. Image route on the same photos (cosine to class means, K other-eye photos,
  5 seeds): **AION K = 1 top-1 0.087 / top-3 0.200, K = 5 0.407 / 0.740; CRVO K = 1 0.274 / 0.546, K = 5 0.662 / 0.897.**
  Why (mean detector output, diagnostic only): AION photos 0.454 disc_edema vs normal 0.020, so the finding is seen, but one finding
  at q ≈ 0.45 cannot beat a ~40× prior gap to normal. CRVO photos 0.054 haemorrhage (BRVO 0.064, normal 0.003), 0.063 disc edema,
  0.011 tortuous vessels: the detectors, trained on RFMiD's HR / PRH labels, barely respond to CRVO's signs.
  **Decision: rare diseases go through the image prototype bank (few photos); the graph stays as the explanation layer (evidence
  for / against from the 18 good detectors), and the zero-shot failure goes on the honest-failure slide.** Phase 4's "done when"
  (leave-disease-out reported vs image-only) is met.
- **Disease bank (step 7; `graph/bank.py`, `artifacts/trace_v2/bank.npz`, `eval/rare_bank.py`, `outputs/eval/rare_bank.json`):** one
  prototype (unit mean of global features) per disease; adding a disease = adding one vector. 10 diseases: 7 common from head
  training, AION 30 / CRVO 56 / RP 14 from their RFMiD + RFMiD 2.0 photos; JSIEC never enters it. Top-1 / top-3 among 10 (chance
  0.10 / 0.30), trace_v2 vs plain 15 ep vs ImageNet. **JSIEC (new dataset, never in the bank):** CRVO (22) **1.00** / 1.00 vs 0.91 /
  1.00 vs 0.91 / 0.95; RP (22) 0.86 / 1.00 vs 0.59 / 0.82 vs **1.00** / 1.00; DR (103) **0.56** vs 0.46 vs 0.17; BRVO (44) **0.50** /
  0.95 vs 0.48 / 0.91 vs 0.11 / 0.55. **Leave one eye out (RFMiD):** AION 0.50 / 0.73 vs 0.47 / 0.73 vs 0.43 / 0.73; CRVO 0.73 / 0.91 vs
  0.71 / 0.91 vs 0.64 / 0.73; RP 0.57 / 0.86 vs 0.43 / 0.71 vs 0.71 / 0.86. RP is the one place ImageNet matches or beats us (its
  pigment is a colour cue). **False alarms:** plain cosine ranking calls 17% of val-A photos (none rare) a rare disease. A rare disease
  is now only *called* when it beats the best common disease by a margin set on val-A for a 5% false alarm (τ = 0.016 for trace_v2).
  Share still called at that margin, trace_v2: JSIEC CRVO 1.00, RP 0.45; RFMiD AION 0.37, CRVO 0.54, RP 0.43. Fewer calls, but
  each is trustworthy; the rest go to the short-list / refer path (Phase 5).
- **Disease × finding map** (`scripts/disease_finding_map.py`, `outputs/figures/disease_finding_map.png`): mean detector output per
  disease (common: head-training out-of-fold; rare: held-out photos). AION's row lights up on disc edema (0.45); CRVO's row is nearly
  blank (haemorrhage 0.05), which is the zero-shot failure in one picture; myopia shows tessellation 0.31 and disc cupping 0.23.
**Phase 5, trust layer (11 Nov; `trust/`, `eval/trust_eval.py`, `eval/site_onboarding.py`, `eval/quality_eval.py`, `eval/decision_eval.py`;
artifacts in `artifacts/trace_v2/{trust,quality}.joblib`; results in `outputs/eval/{trust,site_onboarding,quality,decision}_trace_v2*.json`):**
- **Maths verified first** (`trust/calibrate.py`, `trust/conformal.py`, `tests/test_trust.py`): §16 ECE bin 0.02 and the §10.4 conformal example
  (k = 9, q̂ = 0.70, set {AION, vein occlusion}) reproduce exactly; temperature recovers a known 3× overconfidence; coverage holds on
  exchangeable toy data.
- **Classifier** (`trust/classifier.py`): logistic on the frozen global feature beat prototypes on val-A (6 classes: accuracy 0.882 vs 0.706,
  log-loss 0.301 vs 0.970, ECE 0.032 vs 0.247). **A 7th class `other_abnormal` was added** ("abnormal, none of the 6": findings, no disease
  label): 43% of val-A photos (162 / 377) are of that kind, and the 6-class model called 58 of them normal (missed abnormalities) and the rest
  DR / myopia / …; on test the full pipeline then gave 30.9% confident wrong diagnoses. With 7 classes (val-A accuracy 0.727, T 1.103,
  ECE 0.060), RFMiD + RFMiD 2.0 test on the real mix (732 photos): accuracy 0.796, ECE 0.029, **conformal coverage 0.906** (target 0.90),
  mean set 1.39, 65% single answers; per camera OCT-2000 0.895, Kowa 0.943, TRC-NW300 0.918, RFMiD 2.0 512 0.923, 1504 0.722 (18 photos).
  Risk–coverage: error 0.204 at 100% coverage, 0.137 at 80%, 0.085 at 50%; AURC 0.087 (6-class, clean photos only: 0.116 → 0.032 at
  80%, AURC 0.020). Conformal: only OCT-2000 has ≥ 50 val-B photos (q̂ 0.827); others and unseen cameras use the conservative 0.833.
- **New camera breaks conformal, onboarding restores it** (Paraguay, ZEISS VISUCAM 500; "no DR signs" taken as normal, stated): default
  threshold coverage 0.575 (6-class version 0.673); with Paraguay's own q̂ from ~100 local photos (20 eye-grouped splits) **0.895 [0.836,
  0.946]**, but sets grow to 3.6 diseases (6-class: 2.4): the classifier is unsure there, so the promise is kept by widening, i.e. more
  short-lists and referrals, not confident errors. DR coverage at other sites (7-class, default threshold): IDRiD 0.732, JSIEC 0.748,
  Paraguay 0.225.
- **OOD** (`trust/ood.py`; FAISS cosine, 10th neighbour over pool + head training; 95th pct of val-A, per camera for groups ≥ 50): RFMiD test
  flagged 0.058 (target 0.05); **Paraguay flagged 0.215** (the camera itself looks unfamiliar); 13 JSIEC classes unknown to TRACE (judgment-call
  list in `eval/trust_eval.py: UNKNOWN`, 219 photos): AUROC 0.823, but FPR@95%TPR 0.662: most unknown fundus diseases still look familiar.
  **Non-fundus** (`scripts/cifar_ood_features.py`; 500 CIFAR-10 test images, binary batch parsed as bytes, archive MD5 checked against the
  published c32a1d4…7530, in `raw/ood_cifar10/`): the retina finder rejects 30.2% before the encoder; scoring all 500 anyway (whole image
  treated as retina): **AUROC 1.000, FPR@95%TPR 0.000**. A non-fundus image never passes as a familiar eye.
- **Quality gate** (`trust/quality.py`, `eval/quality_eval.py`; logistic on frozen features, EyeQ grades, split by patient, threshold for 90%
  reject recall on training patients). EyeQ only: held-out EyePACS reject AUROC 0.991, rejects caught 0.878, good sent back 0.002; shortcut
  check on 320 graded photos that never entered the pool (rejects were never in the pool, goods were): AUROC 0.989, caught 0.919, good 0.000,
  so no "seen in pretraining" shortcut. **But it sent back 34–67% of doctor-graded photos from other cameras** (RFMiD 0.344, RFMiD 2.0 0.608,
  IDRiD 0.100, Paraguay 0.671, JSIEC 0.419). Version 2 (deployed): RFMiD + RFMiD 2.0 head-training photos added as readable ("a doctor
  diagnosed from it"): rejects caught 0.888, false retakes RFMiD 0.036, RFMiD 2.0 0.111, IDRiD 0.004, JSIEC 0.200, **Paraguay 0.555**;
  onboarding the gate with 100 Paraguay photos: Paraguay 0.278, rejects still 0.888. Trade-off: rejects come from EyePACS only.
  Bug caught by the unit test: sklearn orders classes alphabetically (good, reject, usable), so the reject column was read by position wrongly.
- **Decision rule** (`trust/decide.py`): retake → unfamiliar → rare-bank call (margin τ 0.013 on val-A, bank built without any test photo)
  → {other_abnormal} alone = refer → one disease = diagnose → 2–3 = short-list → else refer. **Disagreement flag (§10.6) dropped:** on
  val-A, large-gap diagnoses were as accurate as the rest (0.931 vs 0.919). Full pipeline (`eval/decision_eval.py`):
  RFMiD + RFMiD 2.0 test (787): diagnose 0.34 / short-list 0.31 / refer 0.30 / retake 0.05; diagnoses right 0.903; **confident wrong 0.033**
  (6-class: 0.309); short-lists hold the truth 0.84. Rare test photos (63, RFMiD test bucket + JSIEC CRVO / RP): refer 0.79, **confident
  wrong 0.000**, referrals naming the true rare disease 0.50. JSIEC unknown diseases (219): confident wrong 0.078 (6-class 0.361).
  Paraguay (1,208): confident wrong 0.020, diagnoses right 0.925, but retake 0.55 (the quality gate at an unseen camera).
- **Distribution calibration for rare diseases (11 Nov; `graph/dist_calib.py`, `eval/dist_calib_eval.py`,
  `outputs/eval/dist_calib_trace_v2.json`; Yang et al., ICLR 2021, arXiv 2101.06395).** Each rare photo borrows the spread of its
  nearest common disease(s) to make ~500 synthetic feature points; a logistic model is trained on real + synthetic. Two departures:
  alpha on the diagonal only; covariance never formed (exact draws via R^T g). **Chosen on a practice task, not on rare photos:**
  myopia / ARMD / BRVO / CSR each made "rare" (K = 5 or 14 photos, 5 seeds), share caught at 5% val-A false alarm: prototype
  0.61 / 0.67, logistic (real only, balanced) 0.68 / 0.77, logistic + DC 0.71 / 0.80. Paired over the 20 tasks: DC − logistic
  +0.03 [+0.01, +0.05], DC − prototype +0.10 / +0.13. All 5 DC variants within ~0.01 of each other; the pick (`dc_spread_k1`:
  keep the photo's own position, borrow the nearest disease's spread) is by a hair. Myopia is the one class where prototypes win.
  **Final run, once** (RFMiD 5 folds by eye; JSIEC with all rare photos as support), called at 5% false alarm, prototype → logistic
  → DC: AION 0.37 → 0.43 → 0.47 (n 30), CRVO 0.57 → 0.77 → 0.79 (56), RP 0.36 → 0.71 → 0.79 (14); JSIEC CRVO 1.00 → 1.00 → 0.95 (22),
  **JSIEC RP 0.50 → 0.86 → 0.95** (22). Top-1 among 10: AION 0.47 → 0.63, CRVO 0.75 → 0.89, RP 0.64 → 0.86, JSIEC RP 0.86 → 1.00.
  As in the practice task, most of the gain is from logistic over cosine; DC adds a little on top. Small n (RP 14, JSIEC 22 each).
  Not yet wired into `eval/decision_eval.py` (still the cosine bank).
- **Ablation ladder started (11 Nov, 23:23; `scripts/ablation_pool.py`, `configs/ablation.yaml`, `scripts/run_queue.py`).** Pool
  `data/ablation_pool.csv`: all non-EyePACS pool photos (APTOS 3,387, RFMiD 1,735, RFMiD 2.0 460) + 4,418 random EyePACS = 10,000,
  7 mining sources; K 4,096, 30 epochs, mining warm-up 6, val-A every 10 epochs. 8 runs, plain and full first: r0 plain, r5 full,
  r1 + retina views, r2 + artifacts, r3 + multi-crop, r4 + β = 2, then seed-1 repeats of r0 and r5 (noise). Every rung is read at
  its final epoch (no checkpoint picking); features cached after training. Naive similarity masking (§11) not run: it needs
  same-source neighbour lists the queue doesn't store. Estimate ~7 h training + ~1 h features. Log: `outputs/ablation/queue.log`.
- **Ablation ladder result (12 Nov; `scripts/ablation_eval.py`, `outputs/ablation/table.md`; short budget, final epoch,
  seed noise = max |seed 0 − seed 1| over the plain and full repeats).** Read only steps larger than that noise.
  **Retina-safe views carry the camera gain:** unseen camera OCT-2000 0.819 → 0.848 (gap 0.043 → 0.017; noise 0.011),
  Kowa 0.883 → 0.932 (noise 0.006), same-camera neighbours 0.941 → 0.920, Paraguay 0.910 → 0.935 (paired +0.025
  [+0.012, +0.038]); RFMiD kNN falls 0.848 → 0.824. Artifacts: RFMiD linear 0.919 → 0.924, but Paraguay 0.935 → 0.848
  (unexplained; one seed, and the next rung is back at 0.923). Multi-crop: JSIEC 5-shot prototype 0.635 → 0.594.
  β = 2: same-camera neighbours 0.929 → 0.907, nothing else moves. **Mining is a trade-off:** same-camera neighbours
  0.907 → 0.868 (both seeds), the largest camera-mixing step, but JSIEC 5-shot prototype 0.600 → 0.520, logistic
  0.674 → 0.630, IDRiD any-DR 0.880 → 0.845 (paired −0.035 [−0.069, −0.005]), unseen Kowa 0.928 → 0.906. RFMiD linear
  is flat throughout (0.915–0.924), the 3-camera probe ~0.99 for all. So mining mixes cameras in neighbour space at the
  cost of few-shot and retrieval accuracy; the headline `trace_v1_384` was selected at epoch 20, before mining
  started, which fits this. Not yet tested: Medha's lesion-guided crops and camera-balanced sampling as rungs.
- **Final vs copy (13 Nov):** `trace_v2` is final, tagged `final-trace-v2` (local tag, not pushed); its weights, features
  and trust files are read-only on disk. Teammates' ideas go only into a git worktree, `../zesis_combined` on branch
  `combined`, with its own features / outputs / checkpoints copies; `raw/` and `processed/` are shared junctions.
- **Supervised ResNet-50 baseline (13 Nov; `encoder/supervised.py`, `checkpoints/supervised_r50/`):** whole network
  fine-tuned from ImageNet on RFMiD train (1,735 photos; 25 outputs = 24 labels with ≥ 5 positives + Disease_Risk),
  BCE pos_weight capped at 50, retina views at 384 px, AdamW, epoch chosen on val-A (epoch 16). Own head on RFMiD test
  **0.941** [0.929, 0.953] (TRACE frozen + linear 0.917). Frozen features through the same tests, supervised vs
  trace_v1_384 (paired): RFMiD linear 0.918 vs 0.917 (−0.001 [−0.024, +0.026]); IDRiD any-DR 0.864 vs 0.871;
  **Paraguay 0.933 vs 0.886 (TRACE −0.047 [−0.066, −0.029])**; retrieval precision@5 RFMiD 0.62 vs 0.50; **JSIEC
  39-class prototype top-1 K = 1 / 5 / 10: 0.357 / 0.571 / 0.621 vs TRACE 0.403 / 0.628 / 0.716** (logistic 0.375 /
  0.642 / 0.750 vs 0.420 / 0.677 / 0.782). So supervised wins on its labelled diseases (incl. DR at a new site);
  TRACE wins on 39 new diseases from few photos. **Leave-one-camera-out is not valid for this baseline** (its backbone
  saw labelled photos from every RFMiD camera; a fair run needs one retrain per held-out camera), so it isn't reported.
- **UMAP (13 Nov; `scripts/umap_plots.py`, `outputs/figures/umap_imagenet_r50_vs_trace_v2.png`):** 3,835 labelled
  RFMiD + RFMiD 2.0 photos; share of 20 nearest neighbours (2048-d, same eye excluded) with the same disease: ImageNet
  0.65, TRACE 0.74 (chance 0.28); same camera 0.97 vs 0.92 (chance 0.58). TRC-NW300 still forms its own island.
- **Retrieval (13 Nov; `eval/retrieval.py`, `outputs/eval/retrieval.json`, `outputs/figures/retrieval_trace_v2.png`,
  4 random queries, seed 0):** library = head training + rare photos; precision@5, own eye excluded. RFMiD test macro:
  ImageNet 0.41, plain 15 ep 0.50, TRACE 0.50, supervised 0.62. JSIEC CRVO 0.24 / 0.75 / 0.77 / 0.84; JSIEC RP
  0.21 / 0.02 / 0.12 / 0.08; RFMiD AION (leave one eye out) TRACE 0.09 (chance 0.01). With 14 RP and 26 AION photos
  in the library, nearest neighbours rarely are all that class: similar cases explain, the bank decides.
- **LoRA adapter (13 Nov; `encoder/lora.py`, `outputs/eval/lora_trace_v2.json`):** own ~30-line LoRA (rank 8, alpha 16)
  on all 10 layer4 convs of frozen trace_v2, 262k weights = 1.1%; deviation from §7.8's `peft` (its dependencies risk
  the environment; see below). Same photos, RFMiD test, linear probe → LoRA: 1% (17 photos, 3 draws, 4 labels)
  0.807 → 0.817; 10% (173, 3 draws, 14 labels) 0.879 → 0.884; 100% (1 run, 18 labels) 0.917 → **0.932** (supervised
  full fine-tune 0.941). LoRA wins 6 of 7 runs; the 100% gain is one run without an interval.
- **Approach 2's lesion-guided local crops, tested and rejected (13 Nov; copy only: `../zesis_combined`, branch
  `combined`; `augment/lesions.py`, `scripts/lesion_maps.py`, rung `abl_r3_lesion`, `outputs/ablation/lesion/`):**
  r3 multi-crop with Approach 2's locals (centres from a determinant-of-Hessian blob map, 80% guided / 20% uniform,
  side 96–160 px at 512) + same-image queue masking; everything else as `abl_r3_multicrop`, whose re-run reproduced
  the ladder numbers exactly. lesion vs multicrop (seed noise): JSIEC 5-shot prototype 0.550 vs 0.594 (0.004),
  logistic 0.640 vs 0.664; unseen Kowa 0.894 vs 0.925 (0.006); same-camera neighbours 0.958 vs 0.929 (0.003);
  RFMiD kNN 0.802 vs 0.818; RFMiD linear, IDRiD, Paraguay paired differences not significant. No local crops
  (r2) gives JSIEC 5-shot 0.635. So it worsens the multi-crop few-shot drop and makes features more camera-driven;
  not adopted. On the contact sheet the map hits exudates in DR eyes but vessel crossings in healthy eyes. The three
  changes were tested together, so no single part is blamed.
- **Teammates' evaluation protocols on the final encoder (13 Nov; code on branch `combined`; encoders unchanged):**
  (1) **Duplicate cross-check** (`scripts/crosscheck_duplicates.py`, Approach 3's 143 pairs): all 60 confirmed + 2
  candidate IDRiD ↔ RFMiD copies are ≥ 0.8 in our dedupe and in one same-eye group; 0 leaks (no train-side photo
  paired with a kept test photo); their 81 "possibly related" IDRiD mask ↔ grading pairs are test-only on both sides.
  (2) **Approach 3's fixed few-shot exam** (`eval/fixed_fewshot.py`, their trace_fewshot_v1 episodes ported: 11
  novel RFMiD labels, K 1/3/5 × 50, 639 RFMiD val queries, head locked on base labels, macro AUPRC). Abundant
  negatives: TRACE 0.144 / 0.232 / 0.264, plain MoCo 15 ep 0.117 / 0.198 / 0.226, ImageNet 0.095 / 0.141 / 0.169;
  their reported DINOv2-S 0.123 / 0.199 / 0.227, DINOv2-L K=3 0.204, their best SSL K=3 0.238 (their pipeline, 448 px,
  not paired). Paired over episodes, TRACE − plain +0.027 / +0.034 / +0.038 (better on 8–9 of 11 labels), − ImageNet
  +0.050 / +0.091 / +0.095 (10–11 of 11). Strict K+K: TRACE 0.085 / 0.147 / 0.175. Label-shuffle control lift
  0.005–0.009 (pass). Caveats: their queries are RFMiD val and TRACE's checkpoint was chosen on val-A (about half of
  them); supervised_r50 (0.358 / 0.412 / 0.432) was trained on 9 of the 11 labels, so it is not comparable.
  (3) **Approach 2's stress test** (`eval/stress.py`, their 11 artifacts × 5 levels ported; RFMiD test, probes on
  clean train features; clean re-run matches cached features, cosine 1.0000). Mean macro-AUROC over 55 corrupted
  settings / level 5 / mCE vs ImageNet / Disease_Risk flip rate: TRACE 0.897 / 0.872 / **0.667** / 5.8%; plain 0.889 /
  0.851 / 0.719 / 8.3%; supervised 0.885 / 0.844 / 0.742 / 8.9%; ImageNet 0.845 / 0.802 / 1.000 / 11.8%. TRACE best or
  tied on 10 of 11 artifacts (JPEG level 5: 0.843 vs ≤ 0.745); plain MoCo better on colour cast. Several artifacts
  overlap TRACE's training simulator; levels 4–5 exceed its training ranges; no interval on mCE.
- **Environment:** `.venv` uses system site-packages (numpy 1.26.4, pandas, sklearn live there). A plain `pip install
  umap-learn` put numpy 2.2.6 into the venv and broke pandas (killed a training run); removed again. Install new
  packages with `--no-deps`. `iterstrat` needs `test_size` passed explicitly with sklearn 1.8.
- **Packaging (13–14 Nov):** demo app (`demo/app.py`, `demo/pipeline.py`; Gradio 5.20 + pydantic ≥ 2.11 needs the
  `get_type` patch and `show_api=False`), `Makefile` rewritten, MIT licence for code (weights research use only).
  **Weights on Hugging Face: `higgsboson25/trace-retinal-encoder`** (encoder.pt + artifacts/trace_v2, 197 MB, public);
  an anonymous download was byte-identical to the local files. Windows path limit: download into a short folder.
  `master` now includes the `combined` branch (merged via GitHub PR #1 on 13 Nov): teammates' eval scripts plus the
  rejected lesion-crop code, switched off by default (`views.local_guided`, `moco.mask_self` false).
- **Embed a folder (14 Nov; `encoder/embed.py`, `make embed`):** `python -m encoder.embed --images <dir> --out f.npy
  [--camera id]` → (N, 2048) float32 + `<name>_index.csv`; no-retina photos are zero rows with `ok = False`. Workers
  never import torch (the pixel steps moved to torch-free `data.preprocess.eval_view` / `prepare_photo`; `export.py`
  outputs byte-identical on 64 checks). On 40 real test photos (Kowa with `--camera`, Paraguay without) cosine to the
  cached features 1.00000; CPU vs GPU 0.999999; ~6 photos/s on CPU with 4 workers at 4288 × 2848.
- **`requirements.txt` pinned (14 Nov)** to the versions behind the results; unused `timm`, `lightly`, `albumentations`,
  `peft`, `torch-geometric`, `imagehash`, `networkx`, `wandb` dropped (deviation from §19.12; LoRA is our own code).
  `umap-learn` pinned to 0.5.7, because 0.6 declares numpy ≥ 2 (it ran on 1.26 here, but a fresh install would
  refuse). A pip dry run resolves all 85 packages with numpy 1.26.4. A fresh install gets newer fastapi / starlette than
  the demo was tested with (gradio's own pins forbid the tested ones); not run end to end (fresh-clone check skipped).
- **Risk–coverage figure (14 Nov; `scripts/risk_coverage_plot.py`, `docs/figures/risk_coverage.png`):** classifier
  alone, 732 RFMiD + RFMiD 2.0 test photos, 7 classes. Error answering all / 80% / 50%, AURC: TRACE 0.204 / 0.137 /
  0.085, 0.086; supervised 0.195 / 0.140 / 0.090, 0.092; plain MoCo 15 ep 0.225 / 0.162 / 0.071, 0.094; ImageNet
  0.246 / 0.183 / 0.087, 0.110. TRACE best AURC but close to supervised and plain; no interval computed.
- **Rare-disease wording fixed (14 Nov):** README and model card now say the decision rule and demo use the prototype
  bank (called at 5% false alarm: AION 0.37, CRVO 0.57, RP 0.36, JSIEC CRVO 1.00, RP 0.50); the logistic +
  distribution-calibration rates are a separately evaluated model, not yet wired in.
- **Compute plan with real timings:** plain MoCo 100 epochs ≈ 5 h. TRACE (multi-crop, artifacts) **measured 84 img/s → 100 epochs ≈ 7.4 h**
  (was estimated at 6.3 h);
  384 px phase (20 epochs at ~35 img/s) ≈ 3.6 h; ablation ladder (10k photos, 30 epochs) ≈ 50 min a run. About 21 GPU-hours for the
  headline runs and the ladder: fits locally if runs go overnight, with Kaggle for parallel ladder runs.
- **Phase 2 status:** done when the baseline table is filled and the compute plan is redone. Both hold for ImageNet and plain MoCo. Still
  missing from the §11 baseline list: supervised ResNet-50 fine-tuned on RFMiD, and RETFound (only if external weights are allowed).

**ZEISS claim wording:** "Paraguay is an unseen site and dataset captured on a ZEISS VISUCAM 500, a model not identified in
our pool. The pool does contain other ZEISS models (FF450plus via RFMiD 2.0) and ODIR images from unspecified Zeiss devices."
REFUGE's Visucam 500 half takes the same wording. Make no camera-novelty claim for JSIEC.

**Unverified, check before relying on it:** the claim that the kickoff deck allows a public pool with EyePACS as an optional
scale-up (§1.2 records "PRETRAIN: RFMiD images"; mentor question 7); RFMiD patient IDs (§5.4); per-camera val counts.

**Queue and batch (differs from the answer received):** that answer kept K = 16,384 with m = 0.9995 at batch 32, but it was given
before the pool shrank. At 22.5k images, 26% of anchors would have their own key in the queue, so we use K = 8,192, which also
keeps m = 0.999 and K/B = 256. Batch 32 instead of 48 because speed is the same and it leaves room for other apps (§7.6).
