"""Thin wrapper: build the Future-work DINOv2 figure, its points CSV and the scores table.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/thesis_dino_figure.py [--out DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import thesis_dino_figure as T  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=T.OUT_DEFAULT)
    a = ap.parse_args()
    R = T.run(a.out)
    print(R["table"].drop(columns=["class_counts"]).to_string(index=False))
    print(R["meta"])
