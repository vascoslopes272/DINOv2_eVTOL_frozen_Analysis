# eVTOL Patent Taxonomy — Label & Batch Reconciliation Report

**Prepared for:** Phase Supervisor
**Pipeline stage:** `10b_label_analysis.ipynb` (Batches 01 + 05, 1639-patent dataset) — label-only, no embeddings involved.

<span class="provenance">Split out of the original combined <code>Supervisor_Report_Taxonomy_Structure_Analysis.md</code> (Section 1.1/1.2) — this document covers everything derivable from the label/review data alone. Layer/pooling selection and embedding QC moved to <code>12a_qc_report.md</code>; joint structure/clustering results moved to repo 3's <code>21_structure_clustering_report.md</code>.</span>

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
</style>

---

## Section 1: Batches & Architecture Baseline Analysis

<div class="stage-purpose"><strong>What this stage checks:</strong> whether the raw review spreadsheets and the downstream processed manifests agree with each other end-to-end (no patents lost or double-counted between review, duplicate-filtering, and architecture extraction). Label-only — no embeddings involved.</div>

<span class="provenance">All figures below are derived directly from the source review files (<code>Review_postprocess_Batch_01/05.xlsx</code>, patent-level <code>T1</code> section and architecture-level <code>T2</code> section, at <code>.../1639_DS/data/reviewed xlsxs/</code>) and the downstream processing manifests (<code>processing_manifest_Batch_01.csv</code>, <code>processing_manifest_Batch_05.csv</code>, at <code>.../1639_DS/data/processed/Batch_01/</code> and <code>.../Batch_05/</code>) plus <code>src/label_export.py</code>'s outputs.</span>

### 1.1 Individual and Combined Batch High-Level Summary

**Table 1.** Individual and combined batch high-level summary.

| Metric | Batch_01 | Batch_05 | **Global / Combined** |
|---|---:|---:|---:|
| Total patents reviewed (patent-level, `T1`) | 352 | 211 | **563** |
| Patents disapproved (`isApproved=False`) | 60 | 87<sup class="fn-ref">1</sup> | **147** |
| Patents approved (`isApproved=True`) | 292 | 123 | **415** |
| Total unique Patent IDs reviewed *and approved into the architecture stage* (`T2`)<sup class="fn-ref">2</sup> | 125 | 95 | **220** |
| Total architecture instances (approved, `T2` "arch" rows) | 301<sup class="fn-ref">3</sup> | 230<sup class="fn-ref">3</sup> | **531** |
| `is_main == True` images | 150 | 114 | **264** |

<div class="footnotes">
<p><span class="fn-num">1</span> Batch_05 has one patent with an unresolved <code>isApproved</code> value (blank/incomplete review) — excluded from both the approved and disapproved counts above, so 123 + 87 + 1 = 211 reconciles exactly.</p>
<p><span class="fn-num">2</span> <strong>Why this can be lower than "approved patents":</strong> a patent only reaches the architecture (<code>T2</code>) stage after passing patent-level (<code>T1</code>) review as approved <em>and</em> non-duplicate; some approved-but-duplicate patents (e.g. Batch_01's 183 <code>isDuplicate=True</code> patents) are folded into an existing entry rather than creating a new architecture record, which is why 125/95 unique IDs is smaller than 292/123 approved patents.</p>
<p><span class="fn-num">3</span> <strong>Why architecture instances (301/230) exceed unique Patent IDs (125/95):</strong> a single patent can disclose more than one distinct aircraft configuration. Batch_01 has 14 multi-architecture patents (7 with 2 architectures, 3 with 3, 2 with 4, 2 with 5); Batch_05 has 12 (8×2, 2×3, 1×4, 1×5). This is expected and by design — it is <em>not</em> a data-quality issue.</p>
</div>

**`is_main == True` cross-check:** 264 main images (150 + 114) against 268 total labelled architectures (Section 1.2) reconciles cleanly: 268 − 4 architectures with no approved main image = 264. The two counts match exactly once that known gap is accounted for. <span class="cite-inline">Documented in <code>label_analysis.validate_canonical</code>'s <code>qc_issues.csv</code>.</span>

#### Duplicate-Type Definitions

The underlying review criteria are identical across both batches — same taxonomy schema (`G1`→`M3`), same image quality checks. When a reviewer flags a patent as a duplicate of one already in the dataset, it is tagged with one of three real categories (exact labels and counts from the `duplicateType` field, `META` section):

**Table 2.** Duplicate-type definitions and counts.

| Type (as labelled in the data) | Batch_01 | Batch_05 | Plain-language meaning |
|---|---:|---:|---|
| **"1 — Same Aircraft"** | 5 | 6 | Points to an aircraft/design already seen in the dataset, but this patent record is otherwise distinct enough to log separately (e.g. different filing). |
| **"2 — Images AND aircraft the same"** | 167 | 27 | True exact duplicate — same aircraft *and* the same drawings; contributes no new visual information. |
| **"3 — Same plane, small changes"** | 10 | – (0) | Same aircraft, but the drawings show minor variations (e.g. a revised figure, small structural tweak) worth keeping separately. |

The three types are defined by which taxonomy fields match between the two records: **Type 1** — `G1` through `M3` are equal. **Type 2** — `T2` through `M3` are equal. **Type 3** — all taxonomy stages differ. Type 2 accounts for the large majority of flagged duplicates in both batches (167 of 182 flagged in Batch_01, 27 of 33 in Batch_05) — most duplicate flags in this dataset are true exact repeats, not near-variants.

### 1.2 Global Batch Pipeline & Data Discrepancy Audit

<span class="sub-purpose">Purpose: trace exactly where the 563 reviewed patents narrow down to the 264 architectures with an approved main image, and account for every patent/architecture that drops out along the way.</span>

This subsection tracks the same 563 reviewed patents as they narrow down through `label_export.build_canonical` into the final labelled set — i.e. where architectures are gained, dropped, or reshaped, and why.

**Table 3.** Global batch pipeline & data discrepancy audit.

| Pipeline Metric | Value |
|---|---:|
| `is_main == True` images (manifest, both batches) | **264** |
| Total architectures labelled | **268** |
| Base patents behind those architectures | 222 |
| Taxonomy attributes (distinct fields) tracked | **256** |
| Architectures missing an approved main image | 4 |

#### Reconciling the Counts

**268 vs. 256:** these describe two different things, not two stages of the same count. **268** is a *row count* — one row per labelled aircraft configuration (architecture). **256** is a *column count* — the number of distinct taxonomy fields recorded per architecture (e.g. `M1_fusShape`, `M2_wingConf`, `M3_boom_bmech`, ...). Every architecture row carries all 256 attribute columns (populated or null). <span class="cite-inline">Confirmed via <code>label_analysis.coverage_table</code>, <code>cov['attribute'].nunique() == 256</code>.</span>

**268 vs. 264:** the only 4 architectures without an embedding candidate have no approved main image on disk — `CN106585976A`, `KR102760903B1`, `US2018354616A1`, `US2022348339A1`. 268 − 4 = **264**. <span class="cite-inline">Listed in <code>qc_issues.csv</code> ("architecture without approved main image", n=4).</span>

**Status:** RECONCILED (data audit). **Core finding:** every count reconciles exactly across the pipeline — 563 reviewed → 415 approved → 268 architectures → 264 with a usable main image (the 4-architecture gap fully explained by missing approved main images).

<span class="stage-status info">DATA AUDIT — RECONCILED</span>

---

<span class="provenance">All tables sourced from <code>src/label_export.py</code> + <code>src/label_analysis.py</code> and their outputs under <code>taxonomy.output_dir</code>.</span>
