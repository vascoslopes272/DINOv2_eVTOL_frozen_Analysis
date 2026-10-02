#!/usr/bin/env python3
"""Build the DINOv2 analysis in the Labelling Analysis brief format: Markdown, PDF and Word.

    /home/vasco/anaconda3/envs/Finetune/bin/python embedding_evaluation/scripts/build_analysis_brief.py [md] [pdf] [docx]

``md``   (Finetune env: umap, sklearn) src/analysis_brief.py writes
         docs/embedding_evaluation/analysis_document/ANALYSIS_BRIEF.md and its stamped figures in figs_brief/
         (photo figures in EVTOLNEWS_DS/3_embedding_evaluation/analysis_brief_figs/, private).
``pdf``  the root scripts/build_styled_md_pdf.py --compact, the brief's stylesheet, in two passes: the
         first finds the page of every heading, the second prints the index with those pages.
``docx`` labeling_evaluation/scripts/clean_word.py (which runs export_brief_docx.py: pandoc, bordered
         answer/methods boxes, compact styles, A4, figure height cap), then a LibreOffice page count.
The pdf and docx steps need python-markdown / python-docx, which live in the base env; a Finetune run
hands them to /home/vasco/anaconda3/bin/python. With no step named, all three run.

Output: "Embedding Analysis brief/current/Embedding Analysis — Visual Learning (private).docx" and .pdf
(private: © evtol.news photographs). ANALYSIS.md and its figures are never touched.

The logic lives in src/ since 2026-09-30 (notebook 32 calls it there); this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import brief_build as M  # noqa: E402

if __name__ == "__main__":
    M.main(sys.argv[1:])
