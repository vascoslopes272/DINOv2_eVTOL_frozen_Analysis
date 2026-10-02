"""Build the DINOv2 analysis in the Labelling Analysis brief format: Markdown, PDF and Word.

    /home/vasco/anaconda3/envs/Finetune/bin/python embedding_evaluation/scripts/build_analysis_brief.py [md] [pdf] [docx]

``md``   (Finetune env: umap, sklearn) src/analysis_brief.py writes
         docs/embedding_evaluation/analysis_document/ANALYSIS_BRIEF.md and its stamped figures in figs_brief/
         (photo figures in EVTOLNEWS_DS/3_embedding_evaluation/analysis_brief_figs/, private).
``pdf``  the root scripts/build_styled_md_pdf.py --compact, the brief's stylesheet, in two passes: the
         first finds the page of every heading, the second prints the index with those pages.
``docx`` labeling_evaluation/scripts/clean_word.py (which runs export_brief_docx.py: pandoc, bordered
         answer/methods boxes, compact styles, A4, figure height cap), then a LibreOffice page count.
The pdf and docx steps need python-markdown / python-docx, which live in the base env; a Finetune run
hands them to /home/vasco/anaconda3/bin/python. With no step named, all three run.

Output: "Embedding Analysis brief/current/Embedding Analysis — Visual Learning (private).docx" and .pdf
(private: © evtol.news photographs). ANALYSIS.md and its figures are never touched.

Moved from scripts/build_analysis_brief.py on 2026-09-30 (notebook 32 drives it); that script is now a thin
wrapper with the same command line.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PILLAR = Path(__file__).resolve().parent.parent
REPO = PILLAR.parent
sys.path.insert(0, str(PILLAR))

BASE_PY = "/home/vasco/anaconda3/bin/python"
DATA = Path("/mnt/storage_11tb/Drive_files_to_syncronize/3 - Images DataSets & Labelling Outputs")
OUT_DIR = REPO / "docs/embedding_evaluation/analysis_document"
MD = OUT_DIR / "ANALYSIS_BRIEF.md"
CURRENT = DATA / "Embedding Analysis brief/current"
STEM = "Embedding Analysis — Visual Learning (private)"
DOCX, PDF = CURRENT / f"{STEM}.docx", CURRENT / f"{STEM}.pdf"
TITLE = "Visual Learning: DINOv2 embeddings of patent drawings and photographs"
PDF_BUILDER = REPO / "scripts/build_styled_md_pdf.py"
CLEAN_WORD = REPO / "labeling_evaluation/scripts/clean_word.py"
SOFFICE = Path("/home/vasco/.claude/skills/synced/f9481964-6bf2-406f-a63c-ceb6de3d8bd7_791092c0-338c-4d2f-"
               "b11e-f48f47809742/docx/scripts/office/soffice.py")


def md_step() -> None:
    from . import analysis_brief as AB
    path, info = AB.write_markdown()
    (OUT_DIR / "_brief_build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    print("wrote", path)
    for n, p in info["figs"]:
        print(f"  Figure {n}: {p}")


def _pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return int(next(l for l in out.splitlines() if l.startswith("Pages:")).split()[-1])


def pdf_step() -> None:
    from . import brief_toc as BT
    CURRENT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        first = Path(td) / "pass1.pdf"
        subprocess.run([sys.executable, str(PDF_BUILDER), str(MD), str(first), "--compact"], check=True)
        md = MD.read_text(encoding="utf-8")
        MD.write_text(BT.set_toc_pages(md, BT.scan_pages(first, md)), encoding="utf-8")
    subprocess.run([sys.executable, str(PDF_BUILDER), str(MD), str(PDF), "--compact"], check=True)
    print(f"wrote {PDF}  ({_pages(PDF)} pages)")


def docx_step() -> None:
    CURRENT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(CLEAN_WORD), str(MD), str(DOCX), "--title", TITLE])
    if r.returncode != 0:
        raise SystemExit("the Word export failed")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", td,
                        str(DOCX)], check=True, capture_output=True)
        out = next(Path(td).glob("*.pdf"))
        print(f"wrote {DOCX}  (LibreOffice: {_pages(out)} pages)")


def main(argv: list[str]) -> None:
    steps = [a for a in argv if not a.startswith("--")] or ["md", "pdf", "docx"]
    if "md" in steps:
        md_step()
    rest = [s for s in ("pdf", "docx") if s in steps]
    if not rest:
        return
    try:
        import docx  # noqa: F401
        import markdown  # noqa: F401
    except ImportError:
        subprocess.run([BASE_PY, str(PILLAR / "scripts/build_analysis_brief.py")] + rest, check=True)
        return
    if "pdf" in rest:
        pdf_step()
    if "docx" in rest:
        docx_step()


