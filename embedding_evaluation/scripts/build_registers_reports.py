#!/usr/bin/env python3
"""Registers twins of the two embedding reports: the same reports on dinov2-with-registers.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/build_registers_reports.py [patents] [evtolnews] [umap3d]
        [--reuse]    reuse the cached protocol results of the twin (its own tables/_results.pkl)
        [--no-pdf]

The base reports (build_embedding_reports.py, tag dinov2-large_518) are left untouched: this
script imports that builder, sets ``embedding_reports.TAG`` to the registers tag (extracted on
2026-09-30 by eVTOL-Embedding-Extraction/scripts/registers_experiment.py, not re-extracted
here) and redirects every output of the loaded sources to new locations:

    patents    docs/embedding_evaluation/PATENT_EMBEDDING_ANALYSIS_REGISTERS.md + .pdf
               (figs/patent_embedding_registers/), tables in
               1639_LABELLED/3_embedding_evaluation/embedding_protocol_registers/
    evtolnews  EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis_registers/
               EVTOLNEWS_EMBEDDING_ANALYSIS_REGISTERS.md + .pdf, tables/ and figs/ beside it
               (private: it shows directory photos); the patent <-> photo matching of this model
               is computed into its tables/ (matching.csv, matching_ranks.csv)
    umap3d     1639_LABELLED/3_embedding_evaluation/umap_3d/PATENT_UMAP_3D_REGISTERS.html
               (src/patent_3d.write on the registers tag; its own cache/<tag>/ and separability csv)

The author compares each twin with its base report to decide whether to switch the primary model.

The logic lives in src/ since 2026-09-30 (notebook 32 calls it there); this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import registers_reports as M  # noqa: E402

if __name__ == "__main__":
    M.main(sys.argv[1:])
