"""Build the evtol.news advisor report (figures, markdown, PDF). See src/evtolnews_advisor.py.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/evtolnews_advisor_report.py          # brief (default)
    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/evtolnews_advisor_report.py --full   # full report
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evtolnews_advisor as A  # noqa: E402
from src.config_loader import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    full = "--full" in sys.argv
    md = A.write(cfg) if full else A.write_brief(cfg)
    print(md)
    print(A.render_pdf(cfg, md))
