#!/usr/bin/env python3
"""Side-by-side CLS attention, base dinov2-large vs dinov2-with-registers (decision figure, N2).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/registers_attention_figure.py

Three patent figures and three photos (different classes), each shown with its L18 / L22 / L24
CLS-attention overlay under both models, from the arrays that registers_experiment.py wrote.
Output: 1639_LABELLED/3_embedding_evaluation/registers_check/attention_base_vs_registers.png

The logic lives in src/ since 2026-09-30 (notebook 32 calls it there); this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import registers_attention_figure as M  # noqa: E402

if __name__ == "__main__":
    M.main()
