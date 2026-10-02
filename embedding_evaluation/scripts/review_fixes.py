#!/usr/bin/env python3
"""The CPU-side review fixes of the DINOv2 analysis document (review of 2026-09-30).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/review_fixes.py ITEM [ITEM ...] [--quick] [--out DIR]

ITEM: m3 m4 m6 m11 s2_12 s2_5 s2_ph s3 s4 s6 s78 s9 s10 s11 s11_umap figures index all
(``all`` = every number item; ``figures`` and ``index`` read the CSVs the items wrote).
--quick runs every item with small counts into a scratch folder (smoke test; never into review_fixes/).

Outputs: 1639_LABELLED/3_embedding_evaluation/review_fixes/ (CSVs, figs/*.png, INDEX.md). Code:
src/review_fixes.py. Nothing outside that folder is written.

The runner lives in src/review_fixes.py (run_items) since 2026-09-30; this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import review_fixes as RF  # noqa: E402

ITEMS, SUFFIX = RF.ITEMS, RF.SUFFIX


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else None
    items = [a for a in argv if not a.startswith("--") and (out is None or a != str(out))]
    print(f"-> {RF.run_items(items, quick=quick, out=out)}")


if __name__ == "__main__":
    main(sys.argv[1:])
