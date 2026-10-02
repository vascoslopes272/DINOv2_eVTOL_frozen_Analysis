# eVTOL Visual Evaluation — Embedding Evaluation

> **Current pipeline (2026-09-30):** this pillar has three notebooks, run in order, each a thin driver of `src/`:
>
> | notebook | what it reproduces |
> |---|---|
> | [`30_embedding_evaluation.ipynb`](notebooks/30_embedding_evaluation.ipynb) | the figure sets and embeddings of `eVTOL-Embedding-Extraction` (notebooks 20–22), label-free; writes [`docs/embedding_evaluation/embedding_evaluation_report.md`](../docs/embedding_evaluation/embedding_evaluation_report.md) |
> | [`31_evtolnews_evaluation.ipynb`](notebooks/31_evtolnews_evaluation.ipynb) | the evtol.news photo set: class metrics, parent check, patent ↔ photo matching, the advisor report |
> | [`32_setup_selection_and_analysis.ipynb`](notebooks/32_setup_selection_and_analysis.ipynb) | the DINOv2 analysis document: registers check, split-half setup selection, preprocessing variants and controls, view erasure, the review fixes of 2026-09-30, the registers twin reports (optional), the patent 3-D pages, the attention figure and the analysis brief (md, pdf, docx); every step is skipped when its outputs are newer than its inputs (`src/setup_analysis.py`) |
>
> The reasons behind the figure selection are in
> [`SELECTION_DECISIONS.md`](../docs/embedding_evaluation/SELECTION_DECISIONS.md).
> The report below is the **earlier** supervisor-report split (Batches 01 + 05,
> 264 images), kept for reference. Its notebook is archived under `archive/notebooks/`.

**Pillar:** embedding extraction, visual feature analysis and visual-learning metrics.
**Companion pillar:** [Labeling Evaluation](../labeling_evaluation/README.md) (taxonomy, review data, spatial design characteristics, convergence studies).

**Prepared for:** Phase Supervisor
**Pipeline stage:** [`21_structure_clustering.ipynb`](archive/notebooks/21_structure_clustering.ipynb) (Batches 01 + 05, 1639-patent dataset)
**Model:** `facebook/dinov2-large` (frozen), layers {18, 22, 24} × pooling {cls, mean_patch}

*Split from `docs/Supervisor_Report_Taxonomy_Structure_Analysis_summary.md` (the summary edition of the original combined supervisor report, with Section 5 restored). This file carries the embedding-side sections; the batch, duplicate, taxonomy and design-characteristic sections live in the companion file. All figure and output paths below are relative to this repository.*

---

## 1. Reference Matrix Selection (Layer × Pooling)

> **What this stage checks:** which DINOv2 layer/pooling configuration the rest of the analysis standardizes on.

*Purpose: compare all 6 candidate DINOv2 layer × pooling configurations and decide which one the rest of the analysis standardizes on.*

The input set is the **264 embedded main images** that survive the review pipeline (see [Labeling Evaluation §3](../labeling_evaluation/README.md#3-pipeline--data-discrepancy-audit) for how 563 reviewed patents narrow down to 264 embedded images). `G1`/`M1`/`M2`/`M3` are taxonomy schema section codes (topology / fuselage-gear-boom / wing-empennage / propulsion), used downstream as labels against clusters.

**Note:** only 3 of 5 columns are populated for all 6 candidates; silhouette/NMI were only run for layer 22 mean_patch (the bolded row), so this table does not show a head-to-head winner — on Hopkins alone, layer 24 cls (0.724) scores higher. The bolded row is what was carried forward, decided by the criteria below the table, not by this table.

**Table 1.** Layer × pooling baseline comparison (all 264 embedded images).

| Layer | Pooling | Effective dim (participation ratio) | Dims for 90% variance | Hopkins (clusterability) | Best clustering silhouette[^1] | Cluster→taxonomy alignment (best NMI)[^1] |
|---|---|---:|---:|---:|---:|---:|
| 18 | cls | 15.7 | 52 | 0.683 | — | — |
| 18 | mean_patch | 12.2 | 51 | 0.718 | — | — |
| **22** | **mean_patch** | **12.1** | **57** | **0.687** | **0.197 (k-means, k=2)** | **0.260 (M3_emp_chord)** |
| 22 | cls | 20.2 | 61 | 0.706 | — | — |
| 24 | cls | 26.0 | 61 | **0.724** | — | — |
| 24 | mean_patch | 12.1 | 57 | 0.664 | — | — |

[^1]: Only computed for the reference matrix (layer 22, mean_patch); not run for the other 5. Set as `taxonomy.reference_matrix` in `config.yaml`.

**Selection declaration:** **Layer 22, mean-patch pooling** is used for all clustering/alignment work (Sections 4–6). Not derived from Table 1 — on Hopkins alone it is the second-worst of six. Justified instead by the cosine-similarity QC histograms (Table 2): layer 22 mean_patch is the most balanced of the six (mean 0.940, std 0.029), avoiding two failure modes:

- **Too collapsed** — layer 18, mean ≈ 0.94–0.98, little spread.
- **Too noisy** — layer 24 cls, mean 0.434, std 0.133, consistent with an unstable final-block `[CLS]` token.

Section 5 then confirms it in practice: its best-aligning attribute (`M3_emp_chord`, NMI 0.260) beats every confound tested.

**Scope — best available, not proven optimal.** The spread argument cuts both ways: a more spread-out matrix (layer 24) could carry more signal if that spread is structure rather than noise, and that cannot be settled from histograms alone. The six matrices were never compared on taxonomy alignment directly — Section 6 does that, and in fact finds layer 22 **cls** ahead of layer 22 mean_patch on that criterion. **Treat this selection as provisional**, pending that follow-up comparison.

*Set as `taxonomy.reference_matrix` in `config.yaml`.*

**Status:** RECONCILED. **Core finding:** layer 22 mean_patch is the working reference matrix, chosen on cosine-spread balance, not proven optimality.

---

## 2. Embedding Sanity Check (QC)

> **What this stage checks:** whether the embeddings are technically sound before drawing conclusions from them — no corrupted, duplicate, or dead vectors, no NaN/Inf, across all 6 matrices.

### 2.1 Embedding QC Matrix (Stage 1)

No corrupted/duplicate/dead embeddings found (N = 264 figures):

**Table 2.** Embedding QC matrix — cosine-similarity statistics and integrity checks, per layer/pooling.

| Layer | Pooling | Cosine mean | Cosine std | Cosine p05 | Cosine p95 | Exact duplicates | Near-dead dims | NaN/Inf |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 18 | cls | 0.9758 | 0.0118 | 0.9509 | 0.9896 | 0 | 0 | 0 |
| 18 | mean_patch | 0.9421 | 0.0348 | 0.8908 | 0.9783 | 0 | 0 | 0 |
| 22 | cls | 0.9029 | 0.0333 | 0.8369 | 0.9478 | 0 | 0 | 0 |
| 22 | mean_patch | 0.9398 | 0.0294 | 0.8905 | 0.9761 | 0 | 0 | 0 |
| 24 | cls | 0.4337 | 0.1332 | 0.2263 | 0.6666 | 0 | 0 | 0 |
| 24 | mean_patch | 0.8964 | 0.0417 | 0.8201 | 0.9541 | 0 | 0 | 0 |

Layer 24 cls is the outlier (mean cosine similarity drops to 0.43) — expected for the final block's `[CLS]` token, unstable late in ViT backbones.

**Figure 1.** Cosine similarity histograms per matrix (individual panels for legibility).

| L18 cls | L18 mean_patch | L22 cls |
|:---:|:---:|:---:|
| ![L18 cls](../docs/figs/cosine_L18_cls.png) | ![L18 mean_patch](../docs/figs/cosine_L18_mean_patch.png) | ![L22 cls](../docs/figs/cosine_L22_cls.png) |
| **L22 mean_patch** | **L24 cls** | **L24 mean_patch** |
| ![L22 mean_patch](../docs/figs/cosine_L22_mean_patch.png) | ![L24 cls](../docs/figs/cosine_L24_cls.png) | ![L24 mean_patch](../docs/figs/cosine_L24_mean_patch.png) |

**How to read:** each panel is pairwise cosine similarity over 500 random pairs (1.0 = identical, 0 = unrelated). Pure QC, not a design-signal check — a histogram near 1.0 (layer 18) means little room for real design differences to show; a wider spread (layer 24 cls) means more room for structure, but also more noise-sensitivity. Whether the spread is real signal is for Sections 3–6 to test.

### 2.2 Sanity Check Validation

**Status:** PASSED. **Core finding:** all 6 matrices technically sound — zero duplicates/dead dims/NaN-Inf, distributions behave as expected per layer depth. Whether the near-1.0 concentration leaves enough *usable* separation is deferred to Sections 3–6.

---

## 3. Global Embedding Structure vs. Random Noise

> **What this stage checks:** whether the embeddings encode genuine geometric structure or are statistically indistinguishable from random noise — via PCA concentration and pairwise-distance vs. a random baseline.

### 3.1 PCA Structure Across Layers 18–24, cls vs. mean_patch

**Table 3.** PCA structure across layers 18–24, cls vs. mean_patch.

| Layer | Pooling | PC1 variance ratio vs. random | Effective dim | Dims for 90% var |
|---|---|---:|---:|---:|
| 18 | cls | 20.43× | 15.7 | 52 |
| 18 | mean_patch | 25.42× | 12.2 | 51 |
| 22 | cls | 22.01× | 20.2 | 61 |
| 22 | mean_patch | 26.57× | 12.1 | 57 |
| 24 | cls | 18.27× | 26.0 | 61 |
| 24 | mean_patch | **28.49×** | 12.1 | 57 |

`mean_patch` pooling shows a stronger dominant direction than `cls`, which spreads variance across more dimensions — mean-patch embeddings are more concentrated, cls more diffuse.

### 3.2 Pairwise Cosine Distance: Real Embeddings vs. Random Baseline

Random baseline = Gaussian noise of identical shape. All 6 matrices comfortably clear the pipeline's > 3× pass threshold.

**Figure 2.** PCA spectrum, real vs. random (per layer/pooling).

| L18 cls | L18 mean_patch | L22 cls |
|:---:|:---:|:---:|
| ![L18 cls](../docs/figs/pcarand_L18_cls.png) | ![L18 mean_patch](../docs/figs/pcarand_L18_mean_patch.png) | ![L22 cls](../docs/figs/pcarand_L22_cls.png) |
| **L22 mean_patch** | **L24 cls** | **L24 mean_patch** |
| ![L22 mean_patch](../docs/figs/pcarand_L22_mean_patch.png) | ![L24 cls](../docs/figs/pcarand_L24_cls.png) | ![L24 mean_patch](../docs/figs/pcarand_L24_mean_patch.png) |

**Figure 3.** Pairwise cosine-distance distribution, real vs. random (per layer/pooling).

| L18 cls | L18 mean_patch | L22 cls |
|:---:|:---:|:---:|
| ![L18 cls](../docs/figs/basecomp_L18_cls.png) | ![L18 mean_patch](../docs/figs/basecomp_L18_mean_patch.png) | ![L22 cls](../docs/figs/basecomp_L22_cls.png) |
| **L22 mean_patch** | **L24 cls** | **L24 mean_patch** |
| ![L22 mean_patch](../docs/figs/basecomp_L22_mean_patch.png) | ![L24 cls](../docs/figs/basecomp_L24_cls.png) | ![L24 mean_patch](../docs/figs/basecomp_L24_mean_patch.png) |

### 3.3 Structural Integrity Conclusion

All 6 matrices pass real-vs-random (PC1 18–28× baseline) and show non-trivial local structure (Hopkins ≈ 0.66–0.72 — a smooth continuum, not sharply separated islands). **Embeddings encode genuine geometric structure, not noise.** What that continuum is *made of* (design signal vs. drawing style) is answered downstream — Section 4.1 finds a branch grouped by rendering style, and Section 5 quantifies drawing perspective as a genuine confound (ARI 0.288).

**Status:** PASSED. **Core finding:** all 6 matrices clear the ≥ 3× threshold by a wide margin, proving real geometric structure. Hopkins scores predict Section 4 will find coarse, gradient-like groupings rather than crisp clusters — which is what it finds.

---

## 4. Unsupervised Image-Level Clustering

> **What this stage checks:** whether unsupervised clustering recovers meaningful aircraft-design groupings, or just superficial drafting/applicant style.

*Using layer 22, mean_patch pooling — the pipeline's configured reference matrix (Section 1).*

### 4.1 Hierarchical Dendrogram

*Purpose: visualize how the 264 architectures group hierarchically, before any cluster count is chosen.*

Ward-linkage clustering over the 264-figure, layer-22-mean_patch matrix (PCA-57):

![Figure 4. Dendrogram, labeled 1–14 at the cut used for the branch folders below](../docs/figs/dendrogram_labeled.png)

**Figure 4.** Dendrogram, labeled 1–14 at the cut used for the branch folders below.

**Branch-level image folders:** cut at Ward distance 0.60 → exactly **14 branches** (a strict 0.47 cut gives 21 instead). Each labeled branch has a folder of its actual main images for auditing:

```text
docs/dendrogram_clusters/cluster_01/ … cluster_14/   (1–43 images each; cluster_sizes.json)
```

**Visual spot-check:** genuine consistency is confirmed, but the common thread differs by branch — some branches share a design configuration, others share only drawing perspective or rendering style. The branch-by-branch design reading is in [Labeling Evaluation §5](../labeling_evaluation/README.md#5-design-family-reading-of-the-dendrogram-branches); the style effect is the same one quantified as the drawing-perspective confound in Section 5 below (ARI 0.288).

### 4.2 Dimensionality Reduction Comparison

*Purpose: find and validate the cluster count/algorithm that Sections 5–6 test against the taxonomy.*

**Conclusions up front:**

1. A **coarse, stable 2-way split** — every algorithm agrees k=2 is best-supported (agglomerative silhouette 0.254, k-means 0.197, bootstrap ARI 0.724).
2. **Finer structure exists but is fragile** — HDBSCAN finds 5 sub-groups at the cost of 43% "noise," and k=3–10 silhouettes all drop below 0.14.
3. Taxonomy attributes appear as **gradients, not islands** in UMAP, matching Section 3's "smooth continuum" verdict.

**Figure 5.** UMAP colored by unsupervised cluster (k-means k=2, HDBSCAN).

| UMAP — k-means k=2 | UMAP — HDBSCAN |
|:---:|:---:|
| ![UMAP kmeans k=2](../docs/figs/umap_cluster_kmeans_k2.png) | ![UMAP HDBSCAN](../docs/figs/umap_cluster_hdbscan.png) |

HDBSCAN's largest cluster (129 images) occupies the same region as one k-means cluster; its "noise" points (114) fill the diffuse space between cores.

**Figure 6.** UMAP colored by taxonomy attribute (`G1_topType`, `M2_wCount`, `M2_wingConf`, `M1_boomsPresent`, `M1_fusShape`, `M1_gearArch`).

| G1_topType | M2_wCount |
|:---:|:---:|
| ![G1_topType](../docs/figs/umap_attr_G1_topType.png) | ![M2_wCount](../docs/figs/umap_attr_M2_wCount.png) |
| **M2_wingConf** | **M1_boomsPresent** |
| ![M2_wingConf](../docs/figs/umap_attr_M2_wingConf.png) | ![M1_boomsPresent](../docs/figs/umap_attr_M1_boomsPresent.png) |
| **M1_fusShape** | **M1_gearArch** |
| ![M1_fusShape](../docs/figs/umap_attr_M1_fusShape.png) | ![M1_gearArch](../docs/figs/umap_attr_M1_gearArch.png) |

No single attribute cleanly reproduces the cluster boundary — hence the quantitative test in Section 5.

**Clustering sweep:** agglomerative k=2 (silhouette 0.254) best overall; k-means k=2 close behind (0.197); HDBSCAN settles on 5 clusters at 43.2% noise. Bootstrap stability (ARI) = 0.724 for k-means k=2.

---

## 5. Cluster ↔ Taxonomy Alignment (Architectures)

*Purpose: the key confound test — do clusters track real design attributes more than nuisance factors like applicant identity or drawing perspective?*

**Metrics.**

- **NMI** (0–1) measures how much an attribute reduces uncertainty about cluster membership, but is *not* chance-corrected — many small categories can inflate it artificially.
- **ARI** measures the same agreement but *is* chance-corrected (0 = random, can go negative).
- High NMI + comparable ARI = real signal; high NMI + near-zero ARI = artifact.
- **"Categories / N"** = number of label values / number of images actually carrying that label (many fields only apply to a subset of aircraft). More categories over fewer labelled images drives NMI's inflation problem.

Best-aligning attributes to the 2-cluster (k-means) partition, out of 79 tested. Attribute definitions are in [Labeling Evaluation §4](../labeling_evaluation/README.md#4-taxonomy-schema-and-the-attributes-under-test).

**Table 4.** Best-aligning taxonomy attributes vs. confounds, 2-cluster (k-means) partition.

| Attribute | NMI | ARI (chance-corrected) | Categories / N labelled |
|---|---:|---:|---:|
| **M3_emp_chord** | **0.260** | 0.218 | 2 / 39 |
| M3_emp_zone | 0.235 | 0.123 | 5 / 41 |
| M3_boom_t2_count | 0.145 | −0.013 | 9 / 94 |
| M2_wing1_posV | 0.139 | 0.210 | 4 / 211 |
| M1_boom1_circSym | 0.135 | 0.109 | 2 / 111 |
| *assignee (confound)* | 0.207 | **0.036** | **109 / 264** |
| *drawing perspective (confound)* | 0.201 | 0.288 | 6 / 264 |
| *assignee_country (confound)* | 0.169 | 0.182 | 19 / 261 |

Cross-referencing NMI and ARI:

- **Assignee (mirage):** 109 companies / 264 patents inflates NMI to 0.207; ARI is 0.036 — real agreement is practically zero.
- **M3_boom_t2_count (false positive):** NMI 0.145 but ARI −0.013 — aligns worse than chance.
- **M3_emp_chord (real signal):** ARI (0.218) tracks NMI (0.260) closely — legitimate.
- **Drawing perspective (genuine confound):** ARI (0.288) is *higher* than its NMI (0.201) — a real, strong relationship, and the one most worth controlling for (see Section 7).

**Status:** PASSED, with a caveat. On NMI (the pipeline's actual gate), `M3_emp_chord` (0.260) beats every confound. Chance-corrected: `M3_emp_chord` is confirmed real (ARI 0.218), `assignee` dissolves (ARI 0.036), but drawing perspective is a genuine confound (ARI 0.288, above `M3_emp_chord`'s own). **Core finding:** real design signal exists, but not yet *clean* — perspective must be explicitly controlled for before cluster membership is read as pure design signal.

---

## 6. Intra- vs. Inter-Class Distances (Notebook Stage 5 — Layer Comparison)

*Purpose: per-attribute test of whether same-class patents sit closer together in embedding space, and which layer/pooling separates classes best.*

**Separation ratio** = mean between-class cosine distance ÷ mean within-class distance (1.0 = attribute invisible to the embedding; 1.54 = different-class pairs 54% farther apart). Cohen's d is the standardized effect size; the p-value checks it is not a shuffling accident. Across 264 figures × 79 attributes (492 rows): **174 separate significantly at p < 0.05** (excluding `cluster`).

**`cluster`** = the unsupervised k-means assignment fed back through this same test, as a reference ceiling — since clusters were fit to minimize exactly this distance, they should (and do) separate more strongly than any real attribute.

Top individual separations (excluding `cluster`):

**Table 5.** Top individual attribute × layer separations (intra- vs. inter-class distance).

| Attribute | Layer | Pooling | Separation ratio | Cohen's d | p-value |
|---|---|---|---:|---:|---:|
| **M1_boom1_circSym** | 22 | mean_patch | 1.543 | 1.248 | 0.001 |
| M1_boom1_circSym | 22 | cls | 1.412 | 1.149 | 0.001 |
| M1_boom1_circSym | 24 | mean_patch | 1.423 | 1.031 | 0.001 |
| M1_boom1_circSym | 18 | mean_patch | 1.415 | 0.911 | 0.004 |
| M3_core_layout_zone | 22 | cls | 1.146 | 0.735 | 0.005 |
| M3_wing1_chord | 22 | cls | 1.263 | 0.726 | 0.009 |

For reference, `cluster` itself separates most strongly (layer 22 mean_patch: ratio 1.560, d = 1.03, p = 0.001) — expected.

Averaging Cohen's d per layer × pooling, **layer 22 cls** has the highest effect size (mean d = 0.363), ahead of layer 22 mean_patch (0.315) and the rest (0.26–0.30).

![Figure 7. Separation heatmap](../docs/figs/separation_heatmap.png)

**Figure 7.** Separation heatmap.

**What this means for the reference matrix:** this is the one test comparing all 6 matrices on a taxonomy-relevant criterion, and layer 22 **cls** edges out layer 22 mean_patch (0.363 vs. 0.315) — the layer-22 *depth* is validated, but the *pooling* choice is genuinely open. The margin is modest and this metric does not cover clustering quality or confound resistance, so rerunning the clustering and alignment stages on layer 22 cls (and layer 24 mean_patch as a secondary candidate) is listed as future work rather than done inline.

---

## 7. Phase Progress & Next Steps (Embedding Side)

> **What this section is for:** not a new test — it rolls up the pass/fail results from Sections 1–6 into an overall phase status and lists the concrete embedding-side next steps still open. The data/taxonomy-side roll-up is in [Labeling Evaluation §6](../labeling_evaluation/README.md#6-phase-progress--next-steps-labeling-side).

**Embeddings show real, non-random geometric structure** (PC1 18–28× random baseline, Hopkins 0.66–0.72) — sufficient to proceed to deeper unsupervised production pipelines. Zero embedding QC failures (no corrupt/duplicate/dead vectors across 6 layer × pooling matrices, now covering all 264 embedded architectures rather than one per patent). Embedding every architecture (not just each patent's main one) improved the downstream result: the clearest 2-way partition (k-means, silhouette 0.197, stable ARI 0.724) now aligns with a genuine design attribute (`M3_emp_chord`, NMI 0.260, confirmed real by its ARI of 0.218) more strongly than with any confound on the pipeline's NMI gate. The chance-corrected view adds one caveat that must carry into the next phase: drawing perspective's ARI (0.288) exceeds even `M3_emp_chord`'s, making it the confound that must be explicitly controlled before cluster membership is read as pure design signal. Layer 22 mean_patch remains the configured reference matrix, with Section 6 flagging layer 22 cls as the candidate to compare against in the next iteration.

**Immediate next steps:**

1. **Fill baseline comparison gaps.** Complete the layer matrix table: populate the "Best clustering silhouette" and "Cluster → taxonomy alignment (best NMI)" columns for the other 5 candidate layer/pooling configurations in Section 1. Currently they are only computed for the reference matrix (layer 22, mean_patch), leaving the baseline comparison incomplete.
2. **Address clustering weaknesses.**
    - Optimize HDBSCAN hyperparameters: fine-tune the density-based clustering parameters to reduce the high noise fraction (43.2%), which currently discards nearly half the dataset.
    - Implement confound control: explicitly apply residualization to the embeddings before clustering. While the design attribute `M3_emp_chord` (NMI 0.260) beats the confounds on NMI, applicant identity (NMI 0.207, ARI 0.036) and especially drawing perspective (NMI 0.201, ARI 0.288) are strong enough to distort the unsupervised groupings.
3. **Re-evaluate the pooling choice.** Rerun the clustering and alignment stages on layer 22 cls (and layer 24 mean_patch as a secondary candidate), since Section 6 found layer 22 cls ahead on per-attribute separation.

**Status:** IN PROGRESS.

---

*All tables/figures sourced from [`notebooks/21_structure_clustering.ipynb`](archive/notebooks/21_structure_clustering.ipynb) (formerly `DINOv2_eVTOL_frozen_Analysis/notebooks/11_taxonomy_structure_separation.ipynb`). Stage outputs live outside the repo on the sync drive under `4 - Intelligence Models & Post Process Outputs/Preliminary_analysis/outputs/`: embeddings and Stage-0 QC in `analysis_1639_518/`, Stage 1–5 tables and figures in `analysis_taxonomy/`. Figures referenced above are vendored copies in [`docs/figs/`](../docs/figs/); see [ANALYSIS_GUIDE.md](../docs/ANALYSIS_GUIDE.md) for metric definitions.*
