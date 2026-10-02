#!/usr/bin/env python3
"""S1(a): held-out (split-half) choice of model x matrix (see src/heldout_selection.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/heldout_selection.py

Candidates: {registers, base} x {L18, L22, L24} x {CLS, patch mean} = 12, on the
primary image rule of each source (patents: exp3 main figure; photos: hero image).
Label sets: patents G1 12-class, patents parent 5-class, photos 5-class.
The model is fixed to registers by the author; the base rows are the comparison, and
the matrix is decided on the 6 registers candidates by ``heldout_selection.decide``.

Outputs: 1639_LABELLED/3_embedding_evaluation/setup_selection/
    s1a_half_scores.csv       every split x half x candidate x label set
    s1a_selections.csv        every choice (12 candidates and registers-only)
    s1a_candidates.csv        selection frequency + fixed held-out score + full-sample score
    s1a_labels.csv            chosen candidate's held-out score, optimism gap
    s1a_decision.csv          the decision rule's table (registers-only) -> the matrix

The logic lives in src/ since 2026-09-30 (notebook 32 calls it there); this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import heldout_selection as H  # noqa: E402

if __name__ == "__main__":
    H.run_s1a()
