#!/usr/bin/env python3
"""dinov2-with-registers check (chapter plan N2): extraction + attention + artifact share.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/registers_experiment.py [extract] [attention] [share]

No step given runs all three, in that order. Taxonomy scores on the new embeddings are
computed by the evaluation pillar: eVTOL-Visual-Evaluation/embedding_evaluation/
scripts/registers_check.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import registers_experiment as RX  # noqa: E402
from src.config_loader import load_config  # noqa: E402

if __name__ == "__main__":
    steps = [a for a in sys.argv[1:] if not a.startswith("-")] or ["extract", "attention", "share"]
    cfg = load_config()
    for step in steps:
        {"extract": RX.run_extract, "attention": RX.run_attention, "share": RX.run_share}[step](cfg)
