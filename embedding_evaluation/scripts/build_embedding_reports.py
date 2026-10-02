#!/usr/bin/env python3
"""Build the two embedding reports (patent figures, evtol.news photos) with one protocol.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/build_embedding_reports.py [patents] [evtolnews]
        [--reuse]    reuse the cached protocol results (tables/_results.pkl)
        [--no-pdf]

Outputs
    patents    docs/embedding_evaluation/PATENT_EMBEDDING_ANALYSIS.md + .pdf (figs/patent_embedding/),
               tables in 1639_LABELLED/3_embedding_evaluation/embedding_protocol/
    evtolnews  EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis/EVTOLNEWS_EMBEDDING_ANALYSIS.md + .pdf
               (private: it shows directory photos), tables in .../embedding_analysis/tables/

Section 1 (which images, and the check of each step) is written here per source;
Sections 2-8 come from src/embedding_reports.protocol_sections and are the same
text for both sources.

Twin reports on another checkpoint: scripts/build_registers_reports.py imports this
script, sets ``embedding_reports.TAG`` and ``TWIN`` (below) and redirects every output;
with ``TWIN = None`` (the default) this script builds the base reports as before.

The logic lives in src/embedding_reports_build.py since 2026-09-30; this is a thin wrapper (its names are
re-exported, but set TWIN on src.embedding_reports_build itself).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embedding_reports_build as _B  # noqa: E402
from src.embedding_reports_build import *  # noqa: E402,F401,F403

if __name__ == "__main__":
    _B.main(sys.argv[1:])
