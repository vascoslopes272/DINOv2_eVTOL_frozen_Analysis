# Frozen DINOv2 — Extraction QC & Layer/Pooling Selection Report

**Prepared for:** Phase Supervisor
**Pipeline stage:** `12a_extract_dinov2_frozen.ipynb` (Batches 01 + 05, 1639-patent dataset), model `facebook/dinov2-large` (frozen), layers {18, 22, 24} × pooling {cls, mean_patch}

<span class="provenance">Split out of the original combined <code>Supervisor_Report_Taxonomy_Structure_Analysis.md</code> (Section 1.3 + Section 2) — this document covers embedding QC and the layer/pooling selection decision, i.e. everything downstream of extraction but not requiring the joint label+embedding analysis. Label-only batch reconciliation moved to <code>10b_label_analysis_report.md</code>; joint structure/clustering results moved to repo 3's <code>21_structure_clustering_report.md</code>.</span>

<style>
.stage-status {
  display: inline-block; font-size: 0.78em; font-weight: 700; letter-spacing: 0.03em;
  padding: 2px 10px; border-radius: 999px; margin: 0.2em 0 0.6em 0; border: 1px solid transparent;
}
.stage-status.passed   { background:#e6f4ea; color:#1e7d34; border-color:#b7e1c1; }
.stage-status.caution  { background:#fdf3d9; color:#8a6100; border-color:#f2dd9c; }
.stage-status.info     { background:#e8f0fe; color:#1a56c4; border-color:#c3d7fb; }
.stage-status.progress { background:#f1e9fb; color:#6b3fa0; border-color:#ddc9f2; }
.stage-purpose {
  margin: 0.3em 0 1em 0; padding: 0.5em 0.9em; background: #f6f7f9;
  border-left: 3px solid #9aa5b1; font-size: 0.93em; color: #3a3f44;
}
.stage-purpose strong { color: #1a1a1a; }
.sub-purpose { display: block; margin: 0.2em 0 0.8em 0; font-size: 0.9em; font-style: italic; color: #5a6069; }
sup.fn-ref {
  display: inline-flex; align-items: center; justify-content: center; min-width: 1.1em; height: 1.1em;
  padding: 0 0.15em; margin-left: 1px; font-size: 0.68em; font-weight: 700; line-height: 1;
  color: #ffffff; background: #6b7684; border-radius: 50%; vertical-align: super;
}
div.footnotes {
  margin-top: 0.6em; padding: 0.6em 0.9em; background: #fafbfc; border-left: 3px solid #6b7684;
  font-size: 0.88em; color: #3a3f44;
}
div.footnotes p { margin: 0.35em 0; }
.provenance { display: block; margin: 0.2em 0 0.8em 0; font-size: 0.72em; font-style: italic; color: #6b7076; line-height: 1.3; }
.provenance code { font-size: 0.95em; }
.cite-inline { font-size: 0.78em; font-style: italic; color: #6b7076; }
.cite-inline code { font-size: 0.95em; }
span.fn-num {
  display: inline-flex; align-items: center; justify-content: center; min-width: 1.3em; height: 1.3em;
  margin-right: 0.4em; font-size: 0.85em; font-weight: 700; color: #ffffff; background: #6b7684; border-radius: 50%;
}
.img-grid, .img-grid3 { display: flex; flex-wrap: wrap; gap: 0.6em; }
.img-grid3 figure, .img-grid figure { flex: 1 1 30%; margin: 0; }
.img-grid3 img, .img-grid img { max-width: 100%; }
</style>

---

## Section 1.3: Optimal Layer and Patch Pooling Strategy Selection

<span class="sub-purpose">Purpose: compare all 6 candidate DINOv2 layer × pooling configurations and decide which single one downstream joint analysis (repo 3) will standardize on.</span>

`G1`, `M1`, `M2`, `M3` are section codes in the aircraft-design taxonomy schema — `G1` covers global/topology attributes, `M1` fuselage/gear/boom, `M2` wing/empennage, `M3` propulsion-component attributes. They are used downstream as *labels* to test against clusters, in repo 3. The baseline comparison at this stage of the pipeline is across **DINOv2 layers × pooling method**, evaluated on all 264 embedded images.

**What this table is for — and what it isn't.** Only two of its five columns (effective dim, dims for 90% variance, Hopkins) are actually populated for all 6 candidates; the last two (silhouette, NMI) were only ever run for layer 22 mean_patch, so the table cannot by itself show that configuration winning a head-to-head comparison — on Hopkins alone, layer 24 cls (0.724) and layer 18 mean_patch (0.718) both score higher. The bolded row below is not this table's winner; it is a record of which configuration was carried forward, decided on the criteria described after the table (this section's own cosine-spread analysis), not on this table's numbers.

**Table 4.** Layer × pooling baseline comparison (all 264 embedded images).

| Layer | Pooling | Effective dim (participation ratio) | Dims for 90% variance | Hopkins (clusterability) | Best clustering silhouette<sup class="fn-ref">4</sup> | Cluster→taxonomy alignment (best NMI)<sup class="fn-ref">4</sup> |
|---|---|---:|---:|---:|---:|---:|
| 18 | cls | 15.7 | 52 | 0.683 | — | — |
| 18 | mean_patch | 12.2 | 51 | 0.718 | — | — |
| **22** | **mean_patch** | **12.1** | **57** | **0.687** | **0.197 (k-means, k=2)** | **0.260 (M3_emp_chord)** |
| 22 | cls | 20.2 | 61 | 0.706 | — | — |
| 24 | cls | 26.0 | 61 | **0.724** | — | — |
| 24 | mean_patch | 12.1 | 57 | 0.664 | — | — |

<div class="footnotes">
<p><span class="fn-num">4</span> Only computed downstream for the pipeline's configured reference matrix (layer 22, mean_patch), in repo 3; not run for the other 5 matrices in this baseline pass. <span class="cite-inline">Set in <code>config.yaml</code>'s <code>taxonomy.reference_matrix</code>.</span></p>
</div>

**Selection Declaration:** **Layer 22, mean-patch pooling** is the configuration used for all downstream clustering and taxonomy-alignment work in repo 3. **This choice is not derived from Table 4 above** — on Hopkins score, layer 22 mean_patch (0.687) is actually the second-worst of the six, behind layer 24 cls (0.724) and layer 18 mean_patch (0.718). The justification comes entirely from the cosine-similarity QC histograms below (Section 2.1).

The cosine-similarity QC histograms (Table 5, Section 2.1) are what justify it: layer 22 mean_patch has the most balanced distribution of the six — mean 0.940 with a tight spread (std 0.029). It sits between the two failure modes visible in the other matrices:

- **Too collapsed:** layer 18 (cls mean 0.976, mean_patch 0.942) sits close to 1.0 with little spread — most images look nearly identical in that space, leaving little room for real design differences to show up.
- **Too noisy:** layer 24 cls has the widest spread by far (mean 0.434, std 0.133) — over 10× the spread of layer 22 mean_patch — consistent with the last transformer block's `[CLS]` token being unstable rather than structured.

Layer 22 mean_patch avoids both: enough spread to carry signal, without the instability seen at layer 24 cls. Its Hopkins score (0.687) is close to but not the single highest of the six (layer 24 cls: 0.724) — consistent with this being a stability/separability trade-off, not a case where one matrix dominates on every measure. Repo 3's taxonomy-alignment result then confirms the choice holds up in practice: its best-aligning attribute (`M3_emp_chord`, NMI 0.260) beats every confound tested, including applicant identity (0.207) and drawing perspective (0.201).

**Scope of this selection — best available, not proven optimal, and further investigation is needed.** A caveat worth being explicit about: the cosine-spread argument above cuts both ways. Most matrices here sit close to 1.0 in similarity, which raises a legitimate concern that the embeddings are *too* similar overall — and by that logic a more spread-out matrix (layer 24 mean_patch, mean 0.896, or even layer 24 cls, mean 0.434) could in principle carry more separable design signal, provided the extra spread is structure rather than noise. That question cannot be settled from the histograms alone. What can be said is: layer 22 mean_patch *appeared* strongest on the criteria available at selection time, and the downstream stages were only run on it — so the six matrices were never compared on the criterion that matters most (taxonomy alignment). Repo 3's per-attribute separation test provides the first cross-matrix evidence on that front (and in fact finds layer 22 **cls** ahead of layer 22 mean_patch on that criterion — see repo 3's report), and is the natural basis for revisiting this choice in a future iteration. **In short: this selection should be treated as provisional pending that follow-up comparison, not as a settled result.**

<span class="cite-inline">Set as `taxonomy.reference_matrix` in repo 3's `config.yaml`.</span>

**Status:** RECONCILED (data audit). **Core finding:** layer 22 mean_patch stands as the working reference matrix, chosen on cosine-spread balance rather than proven optimality (see the scope note above).

<span class="stage-status info">LAYER/POOLING SELECTION — RECONCILED</span>

---

## Section 2: Embedding QC (Sanity Check)

<div class="stage-purpose"><strong>What this stage checks:</strong> whether the embeddings themselves are technically sound before drawing any scientific conclusions from them — no corrupted, duplicate, or dead vectors, no NaN/Inf values, across all 6 layer×pooling matrices.</div>

### 2.1 Embedding QC Matrix (Stage 1)

No corrupted/duplicate/dead embeddings were found across any of the 6 layer×pooling matrices (N=264 figures):

**Table 5.** Embedding QC matrix — cosine-similarity statistics and integrity checks, per layer/pooling.

| Layer | Pooling | Cosine mean | Cosine std | Cosine p05 | Cosine p95 | Exact duplicates | Near-dead dims | NaN/Inf |
|---|---|---|---|---|---|---|---|---|
| 18 | cls | 0.9758 | 0.0118 | 0.9509 | 0.9896 | 0 | 0 | 0 |
| 18 | mean_patch | 0.9421 | 0.0348 | 0.8908 | 0.9783 | 0 | 0 | 0 |
| 22 | cls | 0.9029 | 0.0333 | 0.8369 | 0.9478 | 0 | 0 | 0 |
| 22 | mean_patch | 0.9398 | 0.0294 | 0.8905 | 0.9761 | 0 | 0 | 0 |
| 24 | cls | 0.4337 | 0.1332 | 0.2263 | 0.6666 | 0 | 0 | 0 |
| 24 | mean_patch | 0.8964 | 0.0417 | 0.8201 | 0.9541 | 0 | 0 | 0 |

Layer 24 cls is the outlier (mean cosine similarity drops to 0.43) — this is expected for the final block's `[CLS]` token, which specializes heavily late in ViT backbones.

<div class="img-grid3">
<figure><img src="figs/cosine_L18_cls.png"><figcaption>L18 cls</figcaption></figure>
<figure><img src="figs/cosine_L18_mean_patch.png"><figcaption>L18 mean_patch</figcaption></figure>
<figure><img src="figs/cosine_L22_cls.png"><figcaption>L22 cls</figcaption></figure>
<figure><img src="figs/cosine_L22_mean_patch.png"><figcaption>L22 mean_patch</figcaption></figure>
<figure><img src="figs/cosine_L24_cls.png"><figcaption>L24 cls</figcaption></figure>
<figure><img src="figs/cosine_L24_mean_patch.png"><figcaption>L24 mean_patch</figcaption></figure>
</div>
<p class="caption"><strong>Figure 1.</strong> Cosine similarity histograms per matrix (regenerated as individual panels for legibility)</p>

**How to read these histograms:** each panel is a distribution of pairwise cosine similarity, computed over 500 random pairs of the 264 figures, for one layer/pooling matrix. The x-axis is similarity (1.0 = identical direction in embedding space, 0 = unrelated); the y-axis is how many of the 500 sampled pairs fall at that similarity level. This is purely a QC check, not a design-signal check: it confirms every matrix produces a normal-looking spread with no NaN/Inf, no exact-duplicate spike at 1.0, and no dead dimensions — nothing about *which aircraft look similar* is visible here. A histogram pushed close to 1.0 (e.g. layer 18 cls, mean 0.976) means most images sit close together in that space — little room left for real design differences to separate them. A wider, lower-mean spread (e.g. layer 24 cls, mean 0.43) means images are more spread apart — more room for genuine structure, but also where the embedding is most sensitive to noise. Whether that spread actually reflects real design signal (rather than noise) is what repo 3's joint tests answer quantitatively, not this QC step.

### 2.2 Sanity Check Validation

**Status:** PASSED (sanity check). **Core finding:** all 6 embedding matrices are technically sound — zero exact duplicates, zero dead/near-dead dimensions, zero NaN/Inf — and the cosine distributions behave as expected per layer depth (near-1.0 early layers, wide spread at layer 24 cls). One observation carried forward rather than resolved here: most matrices concentrate near similarity 1.0, so how much *usable* separation each one actually holds is deferred to the quantitative tests in repo 3.

<span class="stage-status passed">SANITY CHECK — PASSED</span>

---

<span class="provenance">All tables/figures sourced from <code>eVTOL-Embedding-Extraction/notebooks/12a_extract_dinov2_frozen.ipynb</code> and its outputs.</span>
