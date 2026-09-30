"""Run the preparation variants step by step (see src/preprocess_variants.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_variants.py make process extract
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_variants.py montage
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_variants.py --variants grey make process extract

Variants default to all (thick1, thick2, grey); ``montage`` needs thick1 and thick2
processed. Each step reads what the previous one wrote under
``<tree>/2_embedding_extraction/preprocess_variants/<variant>/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocess_variants as X  # noqa: E402
from src.config_loader import load_config  # noqa: E402

STEPS = ["make", "process", "extract", "check", "montage"]


def main(argv: list[str]) -> None:
    variants = X.VARIANTS
    if "--variants" in argv:
        i = argv.index("--variants")
        variants = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    cfg = load_config()
    for s in argv or STEPS:
        if s not in STEPS:
            raise SystemExit(f"unknown step {s!r}; steps: {STEPS}")
        if s == "montage":
            print("==== montage", flush=True)
            X.run_montage(cfg)
            continue
        for v in variants:
            print(f"==== {s} {v}", flush=True)
            getattr(X, "check" if s == "check" else f"run_{s}")(cfg, v)


if __name__ == "__main__":
    main(sys.argv[1:])
