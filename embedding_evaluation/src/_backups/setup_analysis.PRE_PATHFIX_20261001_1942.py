"""Notebook 32's steps: the setup selection, the review-round checks and the analysis brief, in order.

Each step is a call into src/ with the files it reads (inputs) and the files it writes (outputs). A step
runs only when an output is missing or older than its newest input, or when forced; otherwise it is
skipped and says so. Inputs are data, never code: after a code change, force the step.

    from src import setup_analysis as SA
    SA.run("registers_check", force=False)

Steps, in order (notebook 32 has one cell per step):

    registers_check      registers against base, every matrix (registers_check/)
    heldout_selection    S1(a): split-half choice of model x layer x token (setup_selection/s1a_*)
    preprocess_variants  S1(b), S2, S3: thick1 / thick2 drawings, greyscale photos, fairness (s2s3_*, s1b_*)
    preprocess_controls  grey and contrast drawing controls, 5-candidate S1(b), grey-with-grey (s2_controls_*)
    view_erasure         the linear view erasure and its controls (view_erasure/)
    review_fixes         the review round of 2026-09-30, item by item (review_fixes/), then its figures and index
    registers_reports    the registers twins of the two embedding reports (slow; optional)
    patent_3d            the rotatable 3-D page of the patent embeddings, base model (umap_3d/)
    patent_3d_registers  the same page on the registers model
    attention_figure     base against registers attention, the decision figure (registers_check/)
    class_evolution      the directory classes across the development years, brief 4.5 (class_evolution/)
    umap_by_window       the maps window by window, brief 4.5: centres CSVs and the two 3-D pages with a window slider
    analysis_brief       ANALYSIS_BRIEF.md, then its PDF and Word files ("Embedding Analysis brief/current/")

Upstream (eVTOL-Embedding-Extraction, not run here): the embeddings of both tags, the preprocessing
variants and controls, the crop embeddings, the attention arrays and registers_share.csv.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

from . import embedding_reports as ER

PAT, EVN, REPO = ER.PAT, ER.EVN, ER.REPO
PE, EE = PAT / "3_embedding_evaluation", EVN / "3_embedding_evaluation"
SEL, REGCHK, VEDIR, RFD = PE / "setup_selection", PE / "registers_check", PE / "view_erasure", PE / "review_fixes"
CEDIR = PE / "class_evolution"
CROP, ATT, U3D = EE / "crop_experiment", EVN / "2_embedding_extraction/attention", PE / "umap_3d"
BASE, REG = "dinov2-large_518", "dinov2-reg-large_518"
DOCS = REPO / "docs/embedding_evaluation"
BRIEF_MD = DOCS / "analysis_document/ANALYSIS_BRIEF.md"
CURRENT = ER.DATA / "Embedding Analysis brief/current"
BRIEF_STEM = "Embedding Analysis — Visual Learning (private)"


def emb(root: Path, tag: str) -> Path:
    return root / "2_embedding_extraction/embeddings" / tag


EMB = [emb(PAT, BASE), emb(PAT, REG), emb(EVN, BASE), emb(EVN, REG)]
LABELS = [PAT / "2_embedding_extraction/view_state_experiments/exp3_main.csv",
          PAT / "2_embedding_extraction/view_state_experiments/aircraft_common.csv",
          EVN / "2_embedding_extraction/selection/sets/main.csv"]
VARIANTS = ([PAT / "2_embedding_extraction/preprocess_variants" / v / "embeddings" / REG
             for v in ("thick1", "thick2", "grey", "contrast")]
            + [EVN / "2_embedding_extraction/preprocess_variants/grey/embeddings" / REG])
CROP_EMB = [EVN / "2_embedding_extraction/crop_experiment/embeddings" / t for t in (BASE, REG)]
ATT_FILES = [ATT / t / f"{n}_{k}" for t in (BASE, REG) for n in ("patent", "photo") for k in ("cls.npy", "meta.csv")]


# ── freshness ────────────────────────────────────────────────────────────────
def _files(paths: Iterable[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += [q for q in p.rglob("*") if q.is_file() and "_archive" not in q.parts and "_backups" not in q.parts]
        elif p.exists():
            out.append(p)
    return out


def status(inputs: Sequence[Path], outputs: Sequence[Path]) -> Tuple[bool, str]:
    """(needs to run, why)."""
    missing_in = [str(p) for p in inputs if not Path(p).exists()]
    if missing_in:
        return True, f"missing input {missing_in[0]}"
    missing_out = [p for p in outputs if not Path(p).exists()]
    if missing_out:
        return True, f"missing output {missing_out[0].name}"
    fin, fout = _files(inputs), _files(outputs)
    newest_in = max((f.stat().st_mtime for f in fin), default=0.0)
    oldest_out = min(f.stat().st_mtime for f in fout)
    if oldest_out < newest_in:
        src = max(fin, key=lambda f: f.stat().st_mtime)
        return True, f"input newer than output ({src.name})"
    return False, "outputs newer than inputs"


def md5(p: Path) -> str:
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def snapshot(paths: Sequence[Path]) -> Dict[str, str]:
    return {str(p): md5(p) for p in paths if Path(p).exists()}


KEY_FILES = [REGCHK / "registers_knn.csv", REGCHK / "registers_paired.csv", SEL / "s1a_candidates.csv",
             SEL / "s1a_decision.csv", SEL / "s2s3_knn.csv", SEL / "s2_controls_knn.csv", SEL / "s1b_v2_decision.csv",
             VEDIR / "view_erasure.csv", RFD / "S4_paired_contrasts.csv", RFD / "M3_pool_size_and_mix.csv",
             RFD / "S9_attention_extras.csv", RFD / "S8_joint_grid_labels.csv", CEDIR / "ce_convergence.csv",
             CEDIR / "ce_temporal_knn.csv", CEDIR / "ce_drift.csv", CEDIR / "umap_by_window_centroids.csv",
             CEDIR / "umap_by_window_separation.csv", BRIEF_MD]


# ── the steps ────────────────────────────────────────────────────────────────
def _registers_check():
    from . import registers_check
    registers_check.main()


def _heldout():
    from . import heldout_selection
    heldout_selection.run_s1a()


def _variants():
    from . import preprocess_variants_eval as V
    V.run()
    V.write_report()


def _controls():
    from . import preprocess_controls_eval as C
    C.run()
    C.write_addendum()


def _view_erasure():
    from . import view_erasure_check
    view_erasure_check.main()


def _patent_3d():
    from . import patent_3d
    print(patent_3d.write())


def _patent_3d_registers():
    from . import patent_3d
    print(patent_3d.write(REG, "PATENT_UMAP_3D_REGISTERS.html"))


def _attention():
    from . import registers_attention_figure
    registers_attention_figure.main()


def _class_evolution():
    from . import class_evolution
    class_evolution.run()


def _umap_by_window():
    from . import umap_by_window
    umap_by_window.run()


def _twins(reuse: bool = True):
    from . import registers_reports
    registers_reports.main(["patents", "evtolnews"] + (["--reuse"] if reuse else []))


def _brief(reuse: bool = True):
    from . import brief_build
    saved = os.environ.get("BRIEF_REUSE")
    os.environ["BRIEF_REUSE"] = "1" if reuse else "0"
    try:
        brief_build.main(["md", "pdf", "docx"])
    finally:
        if saved is None:
            os.environ.pop("BRIEF_REUSE", None)
        else:
            os.environ["BRIEF_REUSE"] = saved


S1A = [SEL / f"s1a_{n}.csv" for n in ("half_scores", "selections", "candidates", "labels", "decision")]
S2S3 = [SEL / f"{n}.csv" for n in ("s2s3_knn", "s2s3_paired", "fairness", "fairness_class_counts", "s1b_scores",
                                   "s1b_selections", "s1b_candidates", "s1b_labels", "s1b_decision")] + [SEL / "SETUP_SELECTION.md"]
CONTROLS = [SEL / f"{n}.csv" for n in ("s2_controls_knn", "s2_controls_paired", "s2_controls_halves", "s1b_v2_candidates",
                                       "s1b_v2_decision", "s1b_v2_labels", "fairness_grey_grey")] + [SEL / "SETUP_SELECTION_addendum.md"]
VE_OUT = [VEDIR / "view_erasure.csv"]
UPSTREAM_RF = EMB + LABELS + VARIANTS + CROP_EMB + S1A + S2S3 + CONTROLS + VE_OUT

#: review-fix items: the tables each writes, and anything it reads beyond UPSTREAM_RF
RF_ITEMS: Dict[str, Tuple[List[str], List[Path]]] = {
    "m3": (["M3_draws", "M3_per_class_pool_share", "M3_pool_size_and_mix"], []),
    "m4": (["M4a_heldout_architecture_by_split", "M4a_heldout_architecture_halves", "M4a_heldout_view_probe",
            "M4b_control_200_draws", "M4b_control_draws", "M4c_erased_scores"], []),
    "m6": (["M6_balanced_recall", "M9_recall_12_wilson", "M9_recall_5class_wilson", "M9_wrong_predictions"], []),
    "m11": (["M11_ari_like_for_like", "M11_ari_ratios", "M11_counts", "M11_linked_99_vs_94"], ATT_FILES),
    "s2_12": (["S2_probe_12", "S2_probe_shuffles_12"], []),
    "s2_5": (["S2_probe_5", "S2_probe_shuffles_5"], []),
    "s2_ph": (["S2_probe_photos", "S2_probe_shuffles_photos"], []),
    "s3": (["S3_image_rules_paired", "S3_image_rules_split_half", "S3_k_sensitivity"], []),
    "s4": (["S4_paired_contrasts"], []),
    "s6": (["S6_matching_intervals", "S6_matching_queries"], []),
    "s78": (["S7_best_heldout_criterion", "S7_heldout_differences", "S7_own_pick_per_source",
             "S8_halves_missing_classes", "S8_joint_grid_candidates", "S8_joint_grid_labels", "S8_seeds",
             "S8_table12_corrected"], []),
    "s9": (["S9_attention_extras"], ATT_FILES),
    "s10": (["S10_audit_wilson", "S10_kappa", "S10_kappa_by_type"], []),
    "s11": (["S11_bootstrap_draws", "S11_d_units", "S11_view_ratio"], []),
    "s11_umap": (["S11_umap_figures_by_view"], []),
}
RF_FIG_READS = ["S4_paired_contrasts", "M3_pool_size_and_mix", "M3_per_class_pool_share", "M9_recall_12_wilson",
                "M6_balanced_recall", "S8_table12_corrected", "S8_joint_grid_labels", "M4c_erased_scores",
                "M4b_control_200_draws", "S11_umap_figures_by_view"]
RF_FIGS = ["FA_forest_paired_contrasts", "FB_recall_vs_pool_share", "FC_optimism_dumbbell", "FD_class_counts",
           "FE_recall_12_wilson", "FF_view_erasure_control", "FG_umap_figures_by_view"]


def rf_csv(names: Sequence[str]) -> List[Path]:
    return [RFD / f"{n}.csv" for n in names]


RF_ALL = rf_csv([n for outs, _ in RF_ITEMS.values() for n in outs])
CE_OUT = [CEDIR / f"{n}.csv" for n in ("ce_coverage", "ce_counts", "ce_convergence", "ce_median_distance",
                                       "ce_temporal_knn", "ce_drift", "ce_shares")]
CE_IN = [emb(PAT, REG), emb(EVN, REG)] + LABELS + [
    EVN / "0_source/years/aircraft_dev_year.csv", PAT / "0_labelling/inputs/identity/aircraft_identity_ALL.xlsx"]
UW_CSV = [CEDIR / f"umap_by_window_{n}.csv" for n in ("coords_drawings", "coords_photos", "centroids", "separation")]
UW_OUT = UW_CSV + [U3D / "PATENT_UMAP_3D_BY_WINDOW.html", EE / "umap_3d/EVTOLNEWS_UMAP_3D_BY_WINDOW.html"]
BRIEF_OUT = [BRIEF_MD, CURRENT / f"{BRIEF_STEM}.docx", CURRENT / f"{BRIEF_STEM}.pdf"]
BRIEF_IN = (RF_ALL + S1A + S2S3 + CONTROLS + VE_OUT + [REGCHK / "registers_knn.csv", REGCHK / "registers_paired.csv"]
            + ATT_FILES + [ATT / "registers_share.csv", CROP / "knn_original_vs_cropped_registers.csv",
                           CROP / "knn_paired_difference.csv", PE / "embedding_protocol_registers",
                           EE / "embedding_analysis_registers/tables"] + CE_OUT + UW_CSV)

STEPS: Dict[str, Tuple[Callable, List[Path], List[Path]]] = {
    "registers_check": (_registers_check, EMB + LABELS, [REGCHK / "registers_knn.csv", REGCHK / "registers_paired.csv"]),
    "heldout_selection": (_heldout, EMB + LABELS, S1A),
    "preprocess_variants": (_variants, VARIANTS[:2] + VARIANTS[4:] + EMB + [SEL / "s1a_decision.csv"], S2S3),
    "preprocess_controls": (_controls, VARIANTS + EMB + [SEL / "s1a_decision.csv"], CONTROLS),
    "view_erasure": (_view_erasure, EMB[:2] + LABELS, VE_OUT),
    "patent_3d": (_patent_3d, [emb(PAT, BASE)] + LABELS, [U3D / "PATENT_UMAP_3D.html"]),
    "patent_3d_registers": (_patent_3d_registers, [emb(PAT, REG)] + LABELS, [U3D / "PATENT_UMAP_3D_REGISTERS.html"]),
    "attention_figure": (_attention, ATT_FILES, [REGCHK / "attention_base_vs_registers.png"]),
    "class_evolution": (_class_evolution, CE_IN, CE_OUT),
    "umap_by_window": (_umap_by_window, CE_IN, UW_OUT),
}


def run(name: str, force: bool = False) -> str:
    """Run one step of STEPS unless its outputs are newer than its inputs; returns 'ran' or 'skipped'."""
    fn, inputs, outputs = STEPS[name]
    need, why = status(inputs, outputs)
    if not (force or need):
        print(f"[{name}] skipped: {why}")
        return "skipped"
    print(f"[{name}] running: {'forced' if force else why}", flush=True)
    fn()
    return "ran"


def run_review_fixes(force: bool = False, items: Sequence[str] | None = None) -> Dict[str, str]:
    """Every review-fix item whose tables are missing or older than their inputs, then the figures and the index
    when anything they read changed."""
    from . import review_fixes as RF
    done: Dict[str, str] = {}
    for it in items or list(RF_ITEMS):
        outs, extra = RF_ITEMS[it]
        need, why = status(UPSTREAM_RF + extra, rf_csv(outs))
        if force or need:
            print(f"[review_fixes:{it}] running: {'forced' if force else why}", flush=True)
            RF.run_items([it])
            done[it] = "ran"
        else:
            print(f"[review_fixes:{it}] skipped: {why}")
            done[it] = "skipped"
    figs = [RFD / "figs" / f"{f}.png" for f in RF_FIGS]
    need, why = status(rf_csv(RF_FIG_READS), figs)
    if force or need:
        print(f"[review_fixes:figures] running: {'forced' if force else why}", flush=True)
        RF.run_items(["figures"])
    else:
        print(f"[review_fixes:figures] skipped: {why}")
    need, why = status(RF_ALL, [RFD / "INDEX.md"])
    if force or need:
        print(f"[review_fixes:index] running: {'forced' if force else why}", flush=True)
        RF.run_items(["index"])
    else:
        print(f"[review_fixes:index] skipped: {why}")
    return done


def run_twins(enabled: bool, force: bool = False, reuse: bool = True) -> str:
    """The registers twins of the two embedding reports: slow, so they run only when ``enabled``."""
    outs = [DOCS / "PATENT_EMBEDDING_ANALYSIS_REGISTERS.md",
            EE / "embedding_analysis_registers/EVTOLNEWS_EMBEDDING_ANALYSIS_REGISTERS.md"]
    if not enabled:
        print("[registers_reports] not enabled (RUN_TWINS = False)")
        return "disabled"
    need, why = status(EMB + LABELS, outs)
    if not (force or need):
        print(f"[registers_reports] skipped: {why}")
        return "skipped"
    print(f"[registers_reports] running: {'forced' if force else why}", flush=True)
    _twins(reuse)
    return "ran"


def run_brief(force: bool = False, reuse: bool = True) -> str:
    """The analysis brief: Markdown, PDF and Word (the Word step prints its LibreOffice page count)."""
    need, why = status(BRIEF_IN, BRIEF_OUT)
    if not (force or need):
        print(f"[analysis_brief] skipped: {why}")
        return "skipped"
    print(f"[analysis_brief] running: {'forced' if force else why}", flush=True)
    _brief(reuse)
    return "ran"
