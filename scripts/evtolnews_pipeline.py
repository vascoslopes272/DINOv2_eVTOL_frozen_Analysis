"""Build the evtol.news photo set step by step (see src/evtolnews.py).

    python scripts/evtolnews_pipeline.py index pages parse images
    python scripts/evtolnews_pipeline.py filter sets process extract attention

Each step reads what the previous one wrote under ``evtolnews.root``; steps
already done are skipped where the data is on disk (pages, images, crops).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evtolnews as E  # noqa: E402
from src.config_loader import load_config  # noqa: E402

STEPS = ["index", "pages", "parse", "images", "filter", "sets", "process", "extract", "attention"]


def main(steps: list[str]) -> None:
    cfg = load_config()
    en = cfg["evtolnews"]
    for s in steps:
        print(f"==== {s}", flush=True)
        if s == "index":
            E.build_index(cfg)
        elif s == "pages":
            E.crawl_pages(cfg, en["crawl_workers"], en["crawl_pause_s"])
        elif s == "parse":
            E.parse_pages(cfg)
        elif s == "images":
            E.download_images(cfg, en["crawl_workers"], en["crawl_pause_s"])
        elif s in ("filter", "sets", "process", "extract", "attention"):
            from src import evtolnews_sets as S
            getattr(S, f"run_{s}")(cfg)
        else:
            raise SystemExit(f"unknown step {s!r}; steps: {STEPS}")


if __name__ == "__main__":
    main(sys.argv[1:] or STEPS[:4])
