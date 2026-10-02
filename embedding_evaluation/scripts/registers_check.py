#!/usr/bin/env python3
"""Do the taxonomy scores move under dinov2-with-registers? (chapter plan N2)

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/registers_check.py

Reads the base (dinov2-large_518) and the registers (dinov2-reg-large_518) embeddings of both
sources — the registers tag is written by eVTOL-Embedding-Extraction/scripts/
registers_experiment.py — and compares the protocol's primary score (kNN-5 balanced accuracy,
maker held out) on the primary image rule of each source: patents G1 12-class and parent
5-class (exp3 main figure), photos 5-class (hero image). Every matrix is scored; the primary
L24 CLS comparison also gets a paired bootstrap of the difference (same aircraft, same
neighbour protocol, only the model changes).

Outputs: 1639_LABELLED/3_embedding_evaluation/registers_check/{registers_knn.csv,
registers_paired.csv}; the attention side of N2 is registers_experiment.py's share step.

The logic lives in src/ since 2026-09-30 (notebook 32 calls it there); this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import registers_check as M  # noqa: E402

if __name__ == "__main__":
    M.main()
