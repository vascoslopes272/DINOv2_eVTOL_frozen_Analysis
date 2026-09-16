# eVTOL-Embedding-Extraction

Model-agnostic workstream: turns labelled eVTOL patent figures into standardized
embeddings across multiple vision pipelines (frozen DINOv2, DINOv2+registers,
SigLIP, SAM-crops+SigLIP, style-normalized and fine-tuned variants), plus the
label pivot/QC and data augmentation that feed extraction. It does not do any
joint label+embedding analysis (probes, clustering, benchmark tables) — that
lives in the sibling repo `eVTOL-Embedding-Evaluation`, which only *consumes*
this repo's outputs.

## Scope

Stage 2 of the pipeline (stage 0 is labelling in `Patent-Labelling-Tools`,
stage 1 and stage 3 are in `eVTOL-Visual-Evaluation`). `notebooks/` holds
exactly three notebooks, run in order:

| Notebook | What it does | Writes (under `paths.pipeline_root`) |
|---|---|---|
| `20_figure_selection` | **Chooses the figures from the labels only**, without opening any image. It applies fixed gates (approved, domain gate, D1/D2 duplicates, whole-aircraft only), records the selection funnel, and builds one named figure set per strategy (main, all, hover, cruise, per-perspective, line-only, …). It also writes a coverage matrix and a select/report split of the aircraft. | `selection/` |
| `21_image_processing` | Rotates the selected raw crops (`1639_LABELLED/joined/approved_images`) by `rotation_deg`, resizes them with the aspect ratio kept, and pads them to a square, once per size (224, 518). | `processed/<size>/` |
| `22_embedding_extraction` | Embeds the **union** of all sets once for each size, with frozen DINOv2 (every layer × pooling). A set is just a subset of these rows. | `embeddings/<tag>/` |

The embeddings are evaluated by
`eVTOL-Visual-Evaluation/embedding_evaluation/notebooks/30_embedding_evaluation.ipynb`
(metrics, figures, report). The reasons behind each selection rule are in
`eVTOL-Visual-Evaluation/docs/embedding_evaluation/SELECTION_DECISIONS.md`.

## Layout

```
eVTOL-Embedding-Extraction/
  README.md  requirements.txt  .gitignore
  config.yaml                      # paths + selection / processing / extraction blocks
  src/
    config_loader.py               # find + load config.yaml, resolve paths
    figure_selection.py            # notebook 20: candidates, gates, figure sets, coverage, split
    image_processing.py            # notebook 21: rotate, resize, pad
    embeddings.py                  # notebook 22: frozen DINOv2 extraction (+ run_extraction)
    data.py  label_export.py  label_analysis.py   # legacy (archived notebooks)
    contrastive_finetune.py  sam_test.py          # legacy / diagnostics
  notebooks/
    20_figure_selection.ipynb
    21_image_processing.ipynb
    22_embedding_extraction.ipynb
  archive/notebooks/               # superseded notebooks (10a/10b/01 labels, 12a, old 10/11, SAM test)
  docs/                            # older hand-written reports (superseded)
```

On disk (`paths.pipeline_root`):

```
selection/   candidates.csv  funnel.csv  aircraft.parquet  coverage.csv
             sets/<name>.csv  sets_union.csv  selection_summary.json
processed/<size>/<batch>/<patent>/*.png   + manifest.csv
embeddings/<tag>/   emb_layer{L}_{pooling}.npy  metadata.parquet  model_info.json  manifest.json
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

1. `notebooks/20_figure_selection.ipynb`: rules come from `selection:` in `config.yaml`.
2. `notebooks/21_image_processing.ipynb`: settings come from `processing:`.
3. `notebooks/22_embedding_extraction.ipynb`: needs the GPU; settings come from `extraction:` and `analysis:`.
4. In `eVTOL-Visual-Evaluation`, run `embedding_evaluation/notebooks/30_embedding_evaluation.ipynb`.

After changing the selection rules, re-run 20 → 21 → 22. Notebooks 21 and 22 skip work that is already done.

## Embedding output format

```
<pipeline_root>/embeddings/<tag>/          # tag = extraction.tag, e.g. dinov2-large_518
  emb_layer{L}_{pooling}.npy   # L2-normalised, row-aligned with metadata
  metadata.parquet             # figure_uid, aircraft_uid, patent_id
  model_info.json              # detected architecture facts
  manifest.json                # model, input size, layers/pooling, git commit,
                               # the selection summary the figures came from
```

## Notes

- Architecture facts (num layers, hidden dim, register tokens) are **detected**
  from `model.config`, not hardcoded: base DINOv2 has 0 register tokens,
  `-with-registers` has 4, so `n_prefix = 1 (CLS) + num_register_tokens`.
- Embeddings are stored **per figure**, not aggregated. Patent-level
  aggregation is a separate, swappable step (kept in `embeddings.py`).
