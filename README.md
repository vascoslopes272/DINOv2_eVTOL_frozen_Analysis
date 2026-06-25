# evtol_frozen_baseline

Standalone, self-contained **Stage 0** workstream: a preliminary analysis using a
**frozen, off-the-shelf DINOv2** as a fixed feature extractor over eVTOL patent
figures. No fine-tuning, no dependency on any other folder in the workspace.

The goal of Stage 0 is purely a **sanity / QC gate** on the embeddings before any
downstream taxonomy analysis: confirm the model loads, the architecture is
detected correctly, and the per-figure embeddings are clean (no NaN/Inf, not
degenerate).

## Layout

```
evtol_frozen_baseline/
  README.md
  .gitignore
  requirements.txt
  config.yaml                      # self-contained config (fill in EDIT-ME paths)
  src/
    __init__.py
    config_loader.py               # find + load config.yaml, resolve paths
    data.py                        # figure discovery + Excel label loading (ALL label logic here)
    embeddings.py                  # frozen DINOv2 extraction + per-figure storage + QC
  notebooks/
    10_qc_embeddings.ipynb         # thin: import/call/display -> qc_embedding_report.csv
  outputs/
    analysis/                      # generated artifacts (gitignored except .gitkeep)
      embeddings/                  #   emb_layer{L}_{pool}.npy + metadata.parquet + model_info.json
      qc_embedding_report.csv      #   the Stage-0 report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `config.yaml` and replace every `EDIT-ME`:

- `paths.image_dir` — folder of per-figure PNGs (scanned recursively)
- `paths.labels_excel` — the labelled PatSeer Excel
- `data.patent_id_col` — the Excel column holding the patent id
- `data.label_cols` — taxonomy axes (e.g. `lift_arch, wing_type, rotor_count, propulsion, shrouded_open`)
- `data.confound_cols` — e.g. `applicant, filing_year, jurisdiction`

Paths are resolved relative to this folder unless given as absolute paths. An
optional `.env` at the folder root can define `DRIVE_PATH` for use in path
values (e.g. `image_dir: "$DRIVE_PATH/figures"`).

### Filename → patent_id convention

`src/data.py` parses `patent_id` from each figure filename using a documented,
**configurable** regex (see `FIGURE_FILENAME_REGEX`). When you run
`list_figures`, it prints a sample of parses — **confirm these are correct** (or
adjust the regex) before trusting the mapping. `figure_type` here is the figure
*marker token* from the filename, not a semantic view type.

## Run

Open `notebooks/10_qc_embeddings.ipynb` and run all cells. It will:

1. discover figures and print the parse preview,
2. compute per-figure embeddings for each `(layer, pooling)` and store them under
   `outputs/analysis/embeddings/`,
3. write `outputs/analysis/qc_embedding_report.csv`,
4. show two inspection-only histograms, and
5. print **PASS/FAIL** for Stage 0.

## Notes

- Architecture facts (num layers, hidden dim, register tokens) are **detected**
  from `model.config`, not hardcoded: base DINOv2 has 0 register tokens,
  `-with-registers` has 4, so `n_prefix = 1 (CLS) + num_register_tokens`.
- Embeddings are stored **per figure**, not aggregated. Patent-level aggregation
  (`aggregate_to_patent`) currently implements `mean`; `canonical` and
  `per_figure_type` are stubs to be decided empirically in Stage 5 and are
  swappable via `analysis.aggregation.method` in `config.yaml`.
