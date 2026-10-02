#!/usr/bin/env python3
"""Build the DINOv2 analysis document (Markdown + figures, then one private Word file).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/build_analysis_document.py [md] [docx]
        [--breaks]     start every numbered section on a new page
        [--pdf]        also render the Word file to PDF with LibreOffice (next to the scratch copy)

``md`` (Finetune env: umap, sklearn) writes docs/embedding_evaluation/analysis_document/ANALYSIS.md,
its chart figures in figs/, and the photo figures in EVTOLNEWS_DS/3_embedding_evaluation/
analysis_document_figs/ (src/analysis_document.py). ``docx`` needs python-docx, which lives in the base
env, so the Finetune run hands that step to /home/vasco/anaconda3/bin/python. The Word conversion
reuses scripts/build_review_docx.py (to_markdown, _write) unchanged, then sets A4 pages and the
document styles.

Output: "Embedding Analysis brief/current/Embedding Analysis — Visual Learning (private).docx"
(private: © evtol.news photos).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

PILLAR = Path(__file__).resolve().parent.parent
REPO = PILLAR.parent
sys.path.insert(0, str(PILLAR))

BASE_PY = "/home/vasco/anaconda3/bin/python"
DATA = Path("/mnt/storage_11tb/Drive_files_to_syncronize/3 - Images DataSets & Labelling Outputs")
OUT_DIR = REPO / "docs/embedding_evaluation/analysis_document"
DOCX = DATA / "Embedding Analysis brief/current/Embedding Analysis — Visual Learning (private).docx"
TITLE = "Visual Learning: DINOv2 embeddings of patent drawings and photographs"
SUBTITLE = "Analysis. Private: contains © evtol.news photographs"


def md_step() -> None:
    from src import analysis_document as AD
    path, info = AD.write_markdown()
    (OUT_DIR / "_build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    print("wrote", path)
    for n, p, origin in info["figs"]:
        print(f"  Figure {n}: {origin}: {p}")


def _review_docx():
    spec = importlib.util.spec_from_file_location("build_review_docx", REPO / "scripts/build_review_docx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _style(doc, name, size=None, italic=None, bold=None, color=None, keep_next=None, space_after=None,
           space_before=None):
    from docx.shared import Pt, RGBColor
    try:
        st = doc.styles[name]
    except KeyError:
        return
    if size is not None:
        st.font.size = Pt(size)
    if italic is not None:
        st.font.italic = italic
    if bold is not None:
        st.font.bold = bold
    if color is not None:
        st.font.color.rgb = RGBColor.from_string(color)
    pf = st.paragraph_format
    if keep_next is not None:
        pf.keep_with_next = keep_next
    if space_after is not None:
        pf.space_after = Pt(space_after)
    if space_before is not None:
        pf.space_before = Pt(space_before)


def docx_step(breaks: bool) -> Path:
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.shared import Mm
    BR = _review_docx()
    text = (OUT_DIR / "_sections.md").read_text(encoding="utf-8")
    md = BR.to_markdown(text, OUT_DIR)
    parts = md.split("<<SECTION>>")
    md = (BR.PAGE_BREAK if breaks else "\n\n").join(parts)
    out = BR._write(md, DOCX, TITLE, SUBTITLE, toc=True)
    doc = Document(out)
    for s in doc.sections:
        s.page_width, s.page_height = Mm(210), Mm(297)
        s.left_margin = s.right_margin = Mm(20)
        s.top_margin = s.bottom_margin = Mm(18)
    for name in ("Normal", "Body Text", "First Paragraph", "Compact"):
        _style(doc, name, size=11, space_after=6)
    _style(doc, "Body Text", space_before=2)
    _style(doc, "Title", size=20)
    _style(doc, "Subtitle", size=12)
    _style(doc, "Heading 1", size=15, space_before=14, space_after=6, keep_next=True)
    _style(doc, "Heading 2", size=12, space_before=10, space_after=4, keep_next=True)
    _style(doc, "Captioned Figure", keep_next=True, space_after=0)
    _style(doc, "Image Caption", size=9.5, italic=False, bold=True, keep_next=True, space_after=1, space_before=2)
    _style(doc, "Figure Source", size=8.5, color="5A5A5A", space_after=8)
    _style(doc, "Table Title", size=9, keep_next=True, space_before=6, space_after=2)
    for t in doc.tables:                  # keep each table on one page: every row but the last keeps with the next
        rows = t.rows
        for r in list(rows)[:-1]:
            for cell in r.cells:
                for par in cell.paragraphs:
                    par.paragraph_format.keep_with_next = True
        for r in rows:
            trPr = r._tr.get_or_add_trPr()
            cs = OxmlElement("w:cantSplit")
            trPr.append(cs)
    doc.save(out)
    return out


def main(argv: list[str]) -> None:
    steps = [a for a in argv if not a.startswith("--")] or ["md", "docx"]
    breaks = "--breaks" in argv
    if "md" in steps:
        md_step()
    if "docx" in steps:
        try:
            import docx  # noqa: F401
            print("wrote", docx_step(breaks))
        except ImportError:
            subprocess.run([BASE_PY, __file__, "docx"] + (["--breaks"] if breaks else []), check=True)


if __name__ == "__main__":
    main(sys.argv[1:])
