# eVTOL-Embedding-Extraction

Model-agnostic workstream: turns labelled eVTOL patent figures into standardized
embeddings across multiple vision pipelines (frozen DINOv2, DINOv2+registers,
SigLIP, SAM-crops+SigLIP, style-normalized and fine-tuned variants), plus the
label pivot/QC and data augmentation that feed extraction. It does not do any
joint label+embedding analysis (probes, clustering, benchmark tables) — that
lives in the sibling repo `eVTOL-Embedding-Evaluation`, which only *consumes*
this repo's outputs.

## Scope

- **Labels** (`10a`/`10b`): pivot the labelling wizard's long-format export
  into an analysis-ready table (`labels_v1.parquet`), and audit it — batch
  reconciliation, coverage, QC issues. Label-only, no embeddings.
- **Augmentation** (`11`): dataset-variant generation (option A: writes new
  image sets to disk; each extraction notebook picks raw or an augmented
  variant via config).
- **Extraction** (`12a`, `12b`, ...): one notebook per vision pipeline. Loads
  a model, computes per-figure embeddings, runs QC (NaN/Inf/dead-vector
  checks, cosine-similarity sanity, layer/pooling selection where
  applicable), and saves in the standardized format below.

## Layout

```
eVTOL-Embedding-Extraction/
  README.md
  .gitignore
  requirements.txt
  config.yaml                      # self-contained config (fill in EDIT-ME paths)
  src/
    __init__.py
    config_loader.py               # find + load config.yaml, resolve paths
    data.py                        # figure discovery (Stage-02 manifests)
    label_export.py                # wizard long-format -> wide canonical label table
    label_analysis.py              # label-only QC / coverage / batch reconciliation
    augmentation.py                # dataset-variant generation
    embeddings.py                  # frozen DINOv2 extraction + per-figure storage + QC
  notebooks/
    10a_label_export.ipynb         # thin: build_canonical -> labels_v1.parquet
    10b_label_analysis.ipynb       # thin: coverage/QC/batch-reconciliation report
    11_augmentation.ipynb          # thin: dataset-variant generation
    12a_extract_dinov2_frozen.ipynb   # thin: extract + QC for one pipeline
    Archive/                       # superseded notebook snapshots
  docs/
    10b_label_analysis_report.md   # label-only batch/duplicate/coverage report
    12a_qc_report.md                # embedding QC + layer/pooling selection report
    figs/                          # QC figures (cosine-similarity histograms)
  outputs/
    embeddings/<pipeline_name>/    # emb_<variant>.npy + metadata.parquet + manifest.json
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml` and replace every `EDIT-ME`. Paths are resolved relative to
this folder unless given as absolute paths. An optional `.env` at the folder
root can define `DRIVE_PATH` for use in path values (e.g.
`image_dir: "$DRIVE_PATH/figures"`).

### Filename → patent_id convention

`src/data.py` parses `patent_id` from each figure filename using a documented,
**configurable** regex (see `FIGURE_FILENAME_REGEX`). When you run
`list_figures`, it prints a sample of parses — **confirm these are correct** (or
adjust the regex) before trusting the mapping.

## Run

1. `notebooks/10a_label_export.ipynb` — pivots the wizard's reviewed export
   into `labels_v1.parquet` + a column dictionary.
2. `notebooks/10b_label_analysis.ipynb` — label-only coverage/QC/batch report.
3. `notebooks/11_augmentation.ipynb` — (optional) generate augmented dataset
   variants.
4. `notebooks/12a_extract_dinov2_frozen.ipynb` (and future `12b`, `12c`, ...)
   — discover figures, compute embeddings, run QC, save under
   `outputs/embeddings/<pipeline_name>/`.

## Embedding output format (standardized across pipelines)

```
outputs/embeddings/<pipeline_name>/
  emb_<variant>.npy      # one per (layer,pooling) or just one for pipelines
                          # without that axis, row-aligned across figures
  metadata.parquet       # figure_id, patent_id, arch_index, image_path
  manifest.json          # pipeline name, model checkpoint/version,
                          # extraction config, date, git commit
```

`eVTOL-Embedding-Evaluation`'s `registry.py` reads `manifest.json` to resolve
a pipeline name to this folder — no special-casing per pipeline.

## Notes

- Architecture facts (num layers, hidden dim, register tokens) are **detected**
  from `model.config`, not hardcoded: base DINOv2 has 0 register tokens,
  `-with-registers` has 4, so `n_prefix = 1 (CLS) + num_register_tokens`.
- Embeddings are stored **per figure**, not aggregated. Patent-level
  aggregation is a separate, swappable step (kept in `embeddings.py`).
