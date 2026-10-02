#!/usr/bin/env python3
"""Thickening controls (grey, contrast), S1(b) over 5 preparations, grey-with-grey fairness
and SETUP_SELECTION_addendum.md (see src/preprocess_controls_eval.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_controls_check.py
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_controls_check.py report

Needs the registers embeddings of the controls (eVTOL-Embedding-Extraction/scripts/
preprocess_controls.py make process extract ink montage) and the S1(a) decision.
``report`` only rebuilds the Markdown from the CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocess_controls_eval as C  # noqa: E402


def main(argv: list[str]) -> None:
    if "report" not in argv:
        print(f"[controls] matrix {C.run()}")
    C.write_addendum()


if __name__ == "__main__":
    main(sys.argv[1:])
