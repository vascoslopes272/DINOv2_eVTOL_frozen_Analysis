#!/usr/bin/env python3
"""Build the rotatable 3-D UMAP page of the patent-figure embeddings.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/build_patent_3d.py

Output
    1639_LABELLED/3_embedding_evaluation/umap_3d/PATENT_UMAP_3D.html (+ separability_2d_3d.csv)

The photo twin is EVTOLNEWS_UMAP_3D.html, built by scripts/evtolnews_advisor_report.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import patent_3d  # noqa: E402

if __name__ == "__main__":
    print(patent_3d.write())
