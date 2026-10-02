"""Evaluate the evtol.news photo embeddings (see src/evtolnews_eval.py).

    python scripts/evtolnews_eval.py links            # candidate patent<->page links, to review
    python scripts/evtolnews_eval.py metrics class parent matching report
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embedding_metrics as em  # noqa: E402
from src import evtolnews_eval as ev  # noqa: E402
from src.config_loader import load_config  # noqa: E402

STEPS = ["metrics", "class", "parent", "matching", "report"]


def main(steps: list[str]) -> None:
    cfg = load_config()
    en = cfg["evtolnews"]
    for s in steps:
        print(f"==== {s}", flush=True)
        if s == "metrics":
            ev.run_label_free(cfg)
        elif s == "links":
            ev.build_links(cfg)
        elif s == "class":
            ev.run_class_metrics(cfg, en["embeddings"], en["sets"])
        elif s == "parent":
            print(ev.parent_check(cfg).agrees.value_counts())
        elif s == "matching":
            print(ev.run_matching(cfg, en["embeddings"]).to_string())
        elif s == "report":
            print(ev.write_report(cfg))
        else:
            raise SystemExit(f"unknown step {s!r}; steps: links + {STEPS}")


if __name__ == "__main__":
    main(sys.argv[1:] or STEPS)
