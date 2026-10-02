#!/usr/bin/env python3
"""S2 / S3 evaluation, S1(b) preparation choice, fairness, and SETUP_SELECTION.md
(see src/preprocess_variants_eval.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_variants_check.py
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_variants_check.py report

Needs scripts/heldout_selection.py (S1(a) tables) and the registers embeddings of the
variants (eVTOL-Embedding-Extraction/scripts/preprocess_variants.py). ``report`` only
rebuilds the Markdown from the CSVs.

Outputs: 1639_LABELLED/3_embedding_evaluation/setup_selection/
    s2s3_knn.csv, s2s3_paired.csv            every condition, paired differences
    s1b_{scores,selections,candidates,labels,decision}.csv
    fairness.csv, fairness_class_counts.csv
    SETUP_SELECTION.md
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocess_variants_eval as V  # noqa: E402


def main(argv: list[str]) -> None:
    if "report" not in argv:
        res = V.run()
        print(f"[check] matrix {res['key']}, preparation choices {res['choices']}")
    V.write_report()


if __name__ == "__main__":
    main(sys.argv[1:])
