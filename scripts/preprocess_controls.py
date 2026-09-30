"""Run the thickening controls step by step (see src/preprocess_controls.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_controls.py make process extract
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/preprocess_controls.py ink montage

Variants run in the order grey, contrast (contrast is made from grey's processed
518 px inputs, so ``make contrast`` needs ``process grey``); every step loops
make -> process per variant before the next variant. ``ink`` and ``montage`` need
all five preparations processed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocess_controls as X  # noqa: E402
from src.config_loader import load_config  # noqa: E402

STEPS = ["make", "process", "extract", "check", "ink", "montage"]


def main(argv: list[str]) -> None:
    variants = X.CONTROLS
    if "--variants" in argv:
        i = argv.index("--variants")
        variants = argv[i + 1].split(",")
        argv = argv[:i] + argv[i + 2:]
    cfg = load_config()
    steps = argv or STEPS
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"unknown step(s) {bad}; steps: {STEPS}")
    per_variant = [s for s in steps if s in ("make", "process", "extract", "check")]
    for v in variants:            # contrast needs grey processed first
        for s in per_variant:
            print(f"==== {s} {v}", flush=True)
            getattr(X, "check" if s == "check" else f"run_{s}")(cfg, v)
    for s in steps:
        if s in ("ink", "montage"):
            print(f"==== {s}", flush=True)
            getattr(X, f"run_{s}")(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
