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
- **IDRiD photos are inside RFMiD.** 84 RFMiD photos match IDRiD at ≥ 0.8, 76 of them exact copies (score > 0.99), almost all from RFMiD's
  Kowa VX-10α group (IDRiD's camera, which also confirms §5.5's size-to-camera mapping): RFMiD train 72, val 9, test 22. The organisers'
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
- **Still open from Phase 1:** pseudo-domain check (§5.5; must settle the APTOS sizes that equal RFMiD cameras before mining), REFUGE and
  ODIR ingest when downloaded.

**ZEISS claim wording:** "Paraguay is an unseen site and dataset captured on a ZEISS VISUCAM 500, a model not identified in
our pool. The pool does contain other ZEISS models (FF450plus via RFMiD 2.0) and ODIR images from unspecified Zeiss devices."
REFUGE's Visucam 500 half takes the same wording. Make no camera-novelty claim for JSIEC.

**Unverified, check before relying on it:** the claim that the kickoff deck allows a public pool with EyePACS as an optional
scale-up (§1.2 records "PRETRAIN: RFMiD images"; mentor question 7); RFMiD patient IDs (§5.4); per-camera val counts.

**Queue and batch (differs from the answer received):** that answer kept K = 16,384 with m = 0.9995 at batch 32, but it was given
before the pool shrank. At 22.5k images, 26% of anchors would have their own key in the queue, so we use K = 8,192, which also
keeps m = 0.999 and K/B = 256. Batch 32 instead of 48 because speed is the same and it leaves room for other apps (§7.6).
