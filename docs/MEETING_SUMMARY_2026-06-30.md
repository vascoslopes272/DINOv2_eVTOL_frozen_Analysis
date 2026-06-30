# Preliminary frozen-DINOv2 analysis — results to show

Order of work this week. Each section = what was done, what came out, how to read it.

---

## 0. Setup

Frozen `facebook/dinov2-large` (no fine-tuning) as a fixed feature extractor over
patent figures. Testbed: shrouded-vs-open rotor (binary, balanced-ish, ground-truth
labels available) — **a pipeline validation, not the thesis question.** Real target
(configuration types: tilt-rotor / tilt-wing / lift+cruise / multirotor / ducted) is
scaffolded and ready, waiting on labelling (see §6).

Extracted at 3 depths × 2 poolings = 6 representations per image:
- layers **18, 22, 24** (of 24 total)
- pooling **CLS token** vs **mean of patch tokens**

**Dataset (after this week's relabelling cleanup):** 72 patent figures, 1 per
patent, **32 shrouded / 40 open**.

---

## 1. Stage 0 — Sanity / QC gate

**Question:** are the embeddings even usable? (broken values, dead vectors, collapsed representation?)

| layer | pooling | n_figures | nan | inf | all_zero | cos_mean | cos_p50 |
|---|---|---|---|---|---|---|---|
| 18 | cls | 72 | 0 | 0 | 0 | 0.959 | 0.963 |
| 18 | mean_patch | 72 | 0 | 0 | 0 | 0.855 | 0.863 |
| 22 | cls | 72 | 0 | 0 | 0 | 0.881 | 0.889 |
| 22 | mean_patch | 72 | 0 | 0 | 0 | 0.863 | 0.868 |
| **24** | **cls** | 72 | 0 | 0 | 0 | **0.454** | **0.454** |
| 24 | mean_patch | 72 | 0 | 0 | 0 | 0.824 | 0.830 |

**Result: PASS** — zero NaN / Inf / dead vectors across all 6 representations.

**One useful early observation:** `cos_mean` (average similarity between any two
images) drops from ~0.96 at layer 18 to ~0.45 at layer 24 — i.e. **early layers see
all patent drawings as nearly identical; later layers spread images out**, giving
more room to separate classes. First hint that later layers are more discriminative.

---

## 2. Stage 1 — Structure (per representation)

**Question:** how much real information is in each representation, and is there any
shrouded/open structure at all?

| layer | pooling | intrinsic dim | Hopkins | sep. ratio | p (sep.) | silhouette | probe AUC | p (probe) |
|---|---|---|---|---|---|---|---|---|
| 18 | cls | 13.2 | 0.64 | 1.023 | 0.079 | 0.023 | 0.689 | 0.0066 |
| 18 | mean_patch | 15.2 | 0.65 | 1.066 | 0.001 | 0.049 | 0.708 | 0.150 |
| 22 | cls | 21.0 | 0.67 | 1.087 | <0.001 | 0.069 | 0.787 | 0.0100 |
| **22** | **mean_patch** | 18.5 | 0.59 | 1.076 | <0.001 | 0.060 | **0.840** | 0.0033 |
| 24 | cls | 26.2 | 0.65 | 1.114 | <0.001 | 0.095 | 0.836 | 0.0033 |
| **24** | **mean_patch** | 18.0 | 0.58 | **1.122** | <0.001 | 0.097 | 0.839 | 0.0066 |

**Read it as:**
- **Intrinsic dim** (~13–26 of 1024): each representation really lives on a much
  smaller effective surface than its raw size — heavy compression is safe.
- **Hopkins** (~0.6–0.67, vs 0.5 = random): mild but real tendency to cluster.
- **Separation ratio** >1 everywhere, strongest at layer 24 (1.122) — different-class
  images sit further apart than same-class images.
- **Probe AUC** (held-out, cross-validated): best at **layer 22 mean_patch (0.840)**
  and layer 24 (0.836–0.839). 0.5 = chance, 1.0 = perfect.

**Best representations: layer 22 mean_patch and layer 24 (both poolings).**

---

## 3. Deep dive — is the separation real, or luck? (permutation test)

**Method:** compute the real between-class vs within-class distance ratio, then
shuffle the 32/40 labels 10,000 times to see what pure chance produces. The
p-value = how often chance beats the real result.

![permutation test](file:///mnt/storage_11tb/Drive_files_to_syncronize/4%20-%20Intelligence%20Models%20%26%20Post%20Process%20Outputs/Preliminary_analysis/outputs/analysis/permutation_separation.png)

(if the image doesn't render: open directly at
`outputs/analysis/permutation_separation.png` under the configured `output_dir`)

| | layer22_cls | layer24_mean_patch |
|---|---|---|
| observed ratio | 1.087 | 1.122 |
| chance distribution centre | 1.000 | 1.000 |
| z-score | 6.8 | 10.6 |
| p-value | 0.0001 | 0.0001 |

**Read it as:** the red line (real result) sits far outside the entire blue
histogram (10,000 chance outcomes) — chance essentially never reaches this. **The
separation is statistically real, not a fluke.**

**The honest caveat to say out loud:** significance and effect size are different
things. With this many pairwise comparisons, even a modest effect becomes hugely
significant. So: *highly confident it's real* (p≈0.0001), but *moderate in size*
(ratio ~1.1, not 2x).

---

## 4. Confound check — is it really design, or drawing style / date?

**Question:** could the model be reading *who filed the patent* or *when*, instead
of the actual rotor design?

**(a) Is the confound skewed across shrouded/open?**

| confound | test | shrouded | open | p-value | verdict |
|---|---|---|---|---|---|
| filing_year | t-test | 2017.1 | 2017.4 | 0.846 | balanced |
| is_bell (applicant) | chi² | 0.42 | 0.45 | 1.000 | balanced |

**(b) Can the embedding decode the confound?** (best matrices)

| layer/pooling | design AUC | bell AUC | year R² |
|---|---|---|---|
| 22 mean_patch | 0.840 | 0.859 | −2.51 |
| 24 mean_patch | 0.839 | 0.729 | −1.84 |

**Read it as:** the model *can* decode applicant identity quite well (AUC up to
0.86) — so it clearly picks up on drawing style — **but applicant is statistically
balanced across shrouded/open (p=1.0)**, so that style signal runs parallel to the
result, it doesn't explain it. Filing year is both balanced **and** undecodable
(negative R² = worse than guessing the mean) — no temporal leakage.

**One-line takeaway: the shrouded/open signal is genuine, not an artifact of who
drew the patent or when it was filed.**

---

## 5. Effect size with confidence intervals (how big, and how sure)

Bootstrap (2,000 resamples) on the two best representations:

| | sep. ratio [95% CI] | AUC [95% CI] |
|---|---|---|
| layer22_mean_patch | 1.076 [1.037, 1.149] | 0.816 [0.707, 0.904] |
| layer24_mean_patch | 1.122 [1.071, 1.210] | 0.825 [0.716, 0.918] |

**Read it as:** both intervals sit clearly above the chance level (ratio>1.0,
AUC>0.5) → real. Centred around 0.82 AUC → moderate effect. Still fairly wide →
expected with only 72 images; will tighten once the dataset scales.

---

## 6. What's already built for the real thesis question

Shrouded/open was the validation testbed. The same statistical machinery has
already been extended to the actual target — **configuration type** (tilt-rotor,
tilt-wing, lift+cruise, multirotor, ducted) — and is sitting ready in
`notebooks/12_configuration_types.ipynb`: multinomial probe, confusion matrix
(which configurations get mistaken for which), class-similarity matrix. It will run
automatically once the `Configuration` column is added to the labelled Excel.

---

## One-paragraph summary to read out loud

> "I built and validated a statistical pipeline for evaluating frozen DINOv2
> embeddings, using shrouded-vs-open rotor as a controlled testbed. The pipeline
> passed every sanity check, found a statistically real separation between classes
> (permutation p≈0.0001), of moderate size (AUC≈0.82-0.84, with bootstrap
> confidence intervals clearly above chance), and ruled out the two most likely
> confounds — applicant drawing style and filing year. The same machinery is
> already built and ready for my real question, configuration-type classification,
> and will run as soon as labelling is complete."
