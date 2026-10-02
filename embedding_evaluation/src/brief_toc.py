"""The front-page index of the analysis brief, with page numbers from a second PDF pass.

Light on purpose (standard library only): the md step (Finetune env) writes the index with a dash for
every page, and the pdf step (base python) fills the pages after scanning the first-pass PDF, the way
the Labelling Analysis brief does (render_summary.scan_toc_pages, sm_index.index_md).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HEAD_RX = re.compile(r"^(##|###) ((?:\d+(?:\.\d+)?) — .*|The answer to RQ4.*|Appendix [A-Z] — .*)$", flags=re.M)


def toc_md(toc: Sequence[Tuple[int, str]], pages: Dict[str, str] | None = None) -> str:
    pages = pages or {}
    rows = []
    for level, head in toc:
        cls = "toc-l0" if level == 0 else "toc-l1"
        rows.append(f'<div class="toc-row {cls}"><span class="toc-t">{head}</span>'
                    f'<span class="toc-d"></span><span class="toc-p">{pages.get(head, "—")}</span></div>')
    return ('<!-- TOC -->\n<div class="toc">\n<div class="toc-h">Contents</div>\n' + "\n".join(rows)
            + "\n</div>\n<!-- /TOC -->")


def toc_entries(md: str) -> List[Tuple[int, str]]:
    return [(0 if m.group(1) == "##" else 1, m.group(2)) for m in HEAD_RX.finditer(md)]


def set_toc_pages(md: str, pages: Dict[str, str]) -> str:
    return re.sub(r"<!-- TOC -->.*?<!-- /TOC -->", lambda _: toc_md(toc_entries(md), pages), md, flags=re.S)


def _snippet(head: str) -> str:
    h = " ".join(head.split())
    if h.startswith("Appendix "):              # the PDF prints the appendix letter as an eyebrow above the title
        h = h.split(" — ", 1)[1]
    return h if len(h) <= 60 else h[:60].rsplit(" ", 1)[0]


def scan_pages(pdf: Path, md: str) -> Dict[str, str]:
    """Page of every index entry: the first page (after the index page) whose text holds the heading."""
    info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    n = int(next(l for l in info.splitlines() if l.startswith("Pages:")).split()[-1])
    left = {head: _snippet(head) for _, head in toc_entries(md)}
    found: Dict[str, str] = {}
    for i in range(2, n + 1):
        if not left:
            break
        txt = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), str(pdf), "-"],
                             capture_output=True, text=True).stdout
        norm = " ".join(txt.split())
        for head, snip in list(left.items()):
            if snip in norm:
                found[head] = str(i)
                del left[head]
    if left:
        print("TOC: no page found for", "; ".join(left))
    return found
