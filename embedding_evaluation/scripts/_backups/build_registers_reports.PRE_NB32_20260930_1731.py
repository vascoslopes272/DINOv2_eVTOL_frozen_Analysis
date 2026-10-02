#!/usr/bin/env python3
"""Registers twins of the two embedding reports: the same reports on dinov2-with-registers.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/build_registers_reports.py [patents] [evtolnews] [umap3d]
        [--reuse]    reuse the cached protocol results of the twin (its own tables/_results.pkl)
        [--no-pdf]

The base reports (build_embedding_reports.py, tag dinov2-large_518) are left untouched: this
script imports that builder, sets ``embedding_reports.TAG`` to the registers tag (extracted on
2026-09-30 by eVTOL-Embedding-Extraction/scripts/registers_experiment.py, not re-extracted
here) and redirects every output of the loaded sources to new locations:

    patents    docs/embedding_evaluation/PATENT_EMBEDDING_ANALYSIS_REGISTERS.md + .pdf
               (figs/patent_embedding_registers/), tables in
               1639_LABELLED/3_embedding_evaluation/embedding_protocol_registers/
    evtolnews  EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis_registers/
               EVTOLNEWS_EMBEDDING_ANALYSIS_REGISTERS.md + .pdf, tables/ and figs/ beside it
               (private: it shows directory photos); the patent <-> photo matching of this model
               is computed into its tables/ (matching.csv, matching_ranks.csv)
    umap3d     1639_LABELLED/3_embedding_evaluation/umap_3d/PATENT_UMAP_3D_REGISTERS.html
               (src/patent_3d.write on the registers tag; its own cache/<tag>/ and separability csv)

The author compares each twin with its base report to decide whether to switch the primary model.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PILLAR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PILLAR))

from src import embedding_reports as ER  # noqa: E402
from src.embedding_reports import EVN, PAT, REPO  # noqa: E402

REG_TAG = "dinov2-reg-large_518"

PAT_DOC_DIR = REPO / "docs/embedding_evaluation"
PAT_TABLES = PAT / "3_embedding_evaluation/embedding_protocol_registers"
EVN_DIR = EVN / "3_embedding_evaluation/embedding_analysis_registers"
EVN_TABLES = EVN_DIR / "tables"

# every base output this script must never write to
BASE_OUTPUTS = [PAT_DOC_DIR / "PATENT_EMBEDDING_ANALYSIS.md", PAT_DOC_DIR / "figs/patent_embedding",
                PAT / "3_embedding_evaluation/embedding_protocol", EVN / "3_embedding_evaluation/embedding_analysis",
                EVN / "3_embedding_evaluation/matching.csv"]


def load_builder():
    """scripts/build_embedding_reports.py as a module (it is a script, not a package member)."""
    spec = importlib.util.spec_from_file_location("build_embedding_reports",
                                                  PILLAR / "scripts/build_embedding_reports.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def retarget(S) -> None:
    """Point every output of a loaded Source at the twin's folders."""
    if S.key == "patents":
        S.doc_dir, S.doc_name = PAT_DOC_DIR, "PATENT_EMBEDDING_ANALYSIS_REGISTERS.md"
        S.table_dir, S.fig_sub = PAT_TABLES, "figs/patent_embedding_registers"
    else:
        S.doc_dir, S.doc_name = EVN_DIR, "EVTOLNEWS_EMBEDDING_ANALYSIS_REGISTERS.md"
        S.table_dir, S.fig_sub = EVN_TABLES, "figs"
    outs = [(S.doc_dir / S.doc_name).resolve(), (S.doc_dir / S.fig_sub).resolve(), S.table_dir.resolve()]
    for o in outs:
        for b in BASE_OUTPUTS:
            b = b.resolve()
            if o == b or b in o.parents:
                raise SystemExit(f"refusing to write into a base output: {o}")


def matching(out: Path, reuse: bool) -> Path:
    """Patent <-> photo matching with the registers embeddings, written to the twin's tables folder.

    Runs evtolnews_eval.run_matching unchanged, with three redirections: the links are read
    before its output folder is redirected; its output folder is the twin's; the patent
    embeddings are read from 1639_LABELLED (patent_reference/ holds a byte-identical copy of the
    base tag only, checked 2026-09-30)."""
    f = out / "matching.csv"
    if reuse and f.exists():
        return f
    from src import embedding_metrics as em
    from src import evtolnews_eval as EE
    from src.config_loader import load_config
    cfg = load_config()
    links = EE.load_links(cfg)
    out.mkdir(parents=True, exist_ok=True)
    saved = EE.out_dir, EE.load_links, EE.load_patent
    try:
        EE.out_dir = lambda cfg: out
        EE.load_links = lambda cfg: links
        EE.load_patent = lambda cfg, tag: {
            "result": em.load_embeddings(PAT / "2_embedding_extraction/embeddings" / tag),
            "sets": em.load_sets(cfg)}
        EE.run_matching(cfg, [REG_TAG])
    finally:
        EE.out_dir, EE.load_links, EE.load_patent = saved
    return f


def main(argv: list[str]) -> None:
    which = [a for a in argv if not a.startswith("--")] or ["patents", "evtolnews", "umap3d"]
    reuse, pdf = "--reuse" in argv, "--no-pdf" not in argv
    ER.TAG = REG_TAG                      # both loaders and every model name read this
    BR = load_builder()
    BR.TWIN = {
        "base_doc": {"patents": "eVTOL-Visual-Evaluation/docs/embedding_evaluation/PATENT_EMBEDDING_ANALYSIS.pdf",
                     "evtolnews": "EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis/"
                                  "EVTOLNEWS_EMBEDDING_ANALYSIS.pdf"},
        "base_tables": {"patents": PAT / "3_embedding_evaluation/embedding_protocol",
                        "evtolnews": EVN / "3_embedding_evaluation/embedding_analysis/tables"},
        "companion_doc": {"patents": "EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis_registers/"
                                     "EVTOLNEWS_EMBEDDING_ANALYSIS_REGISTERS.pdf",
                          "evtolnews": "eVTOL-Visual-Evaluation/docs/embedding_evaluation/"
                                       "PATENT_EMBEDDING_ANALYSIS_REGISTERS.pdf"},
        "companion_table": {"patents": EVN_TABLES / "t_main.csv", "evtolnews": PAT_TABLES / "t_main_5classes.csv"},
        "matching": EVN_TABLES / "matching.csv",
        "script": "build_registers_reports.py",
    }
    for w in which:
        if w == "umap3d":
            from src import patent_3d
            print("wrote", patent_3d.write(REG_TAG, "PATENT_UMAP_3D_REGISTERS.html"), flush=True)
            continue
        S = ER.load_patents() if w == "patents" else ER.load_evtolnews()
        retarget(S)
        if w == "evtolnews":
            print("[evtolnews] matching,", matching(EVN_TABLES, reuse), flush=True)
        BR.build(S, reuse=reuse, pdf=pdf)


if __name__ == "__main__":
    main(sys.argv[1:])
