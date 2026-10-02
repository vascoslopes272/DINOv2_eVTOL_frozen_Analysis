#!/usr/bin/env python3
"""Does the drawings' architecture signal survive removing the view? (chapter plan N6)

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/view_erasure_check.py

For the base (dinov2-large_518) and the registers (dinov2-reg-large_518) embeddings, L24 CLS:

  1. fit a linear view erasure (src/view_erasure.py) on the whole-aircraft figures of the 663
     analysis aircraft, using ONLY their view4 labels: mean projection, then INLP rounds while
     the maker-grouped view probe is more than 0.05 above chance (at most 10 rounds);
  2. erase every figure, re-normalise, rebuild the "exp3 main figure" aircraft vectors from the
     erased figures (embedding_reports.rule_matrix) and re-score G1 12-class and parent 5-class
     kNN-5 (maker held out), paired bootstrap of erased - original;
  3. controls: the same number of RANDOM orthonormal directions removed instead (once), and a
     variance-matched random subspace (random directions inside the top principal components,
     as much variance as the view directions); the variance each removed subspace carried; the
     view probe when the erasure is fitted on half of the makers and tested on the other half;
     the view kNN and the T6 view / architecture separation before and after; the architecture
     kNN on the aircraft whose main figure is a Perspective view (view held constant by design);
     how often an aircraft's 5 neighbours share its main figure's view.

Output: 1639_LABELLED/3_embedding_evaluation/view_erasure/view_erasure.csv (long format, every
number); the write-up is VIEW_ERASURE.md in the same folder.

The logic lives in src/ since 2026-09-30 (notebook 32 calls it there); this is a thin wrapper.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import view_erasure_check as M  # noqa: E402

if __name__ == "__main__":
    M.main()
