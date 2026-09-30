"""Run the evtol.news crop ablation step by step (see src/photo_crop_experiment.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/photo_crop_experiment.py detect process extract
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/photo_crop_experiment.py montage evaluate

``detect --per-class N`` runs on a stratified sample instead of the whole
``main`` set. Each step reads what the previous one wrote under
``<evtolnews.root>/{2_embedding_extraction,3_embedding_evaluation}/crop_experiment/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import photo_crop_experiment as X  # noqa: E402
from src.config_loader import load_config  # noqa: E402

STEPS = ["detect", "process", "extract", "montage", "evaluate"]


def main(argv: list[str]) -> None:
    per_class = 0
    if "--per-class" in argv:
        i = argv.index("--per-class")
        per_class = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    cfg = load_config()
    for s in argv or STEPS:
        print(f"==== {s}", flush=True)
        if s == "detect":
            X.run_detect(cfg, per_class=per_class)
        elif s in STEPS:
            getattr(X, f"run_{s}")(cfg)
        else:
            raise SystemExit(f"unknown step {s!r}; steps: {STEPS}")


if __name__ == "__main__":
    main(sys.argv[1:])
