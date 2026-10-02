#!/usr/bin/env python3
"""Build the SHORT DINOv2 analysis (src/analysis_simple.py): Markdown, PDF and Word.

    /home/vasco/anaconda3/envs/Finetune/bin/python embedding_evaluation/scripts/build_analysis_simple.py [md] [pdf] [docx]

``md`` needs the Finetune env (umap, sklearn); ``pdf`` and ``docx`` need python-markdown / python-docx (base env),
so a Finetune run hands them to /home/vasco/anaconda3/bin/python. With no step named, all three run.
Output: "Embedding Analysis brief/current/Embedding Analysis — Visual Learning (private).docx" and .pdf; the Word
step fails if LibreOffice counts more than 20 pages. The 30-page brief is build_analysis_brief.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis_simple as M  # noqa: E402

if __name__ == "__main__":
    M.main(sys.argv[1:])
