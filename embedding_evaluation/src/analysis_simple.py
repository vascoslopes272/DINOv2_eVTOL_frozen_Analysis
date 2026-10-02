"""The DINOv2 analysis as a SHORT, plain document for a reader who has never seen embeddings (2026-10-01).

Why (author's review of the 30-page brief, 42 comments): the brief went deep into tuning before saying what
the variables are; it used jargon (kNN, k, maker, Holm, probe, split-half, post hoc, kappa) without a worked
example; it had no overview of the pipeline and the tests. The author uses DINOv2 as a tool: he wants the
basics, the few settings that change the result, and proof that the result is robust, in short.

Structure (approved 2026-10-01):
    1  the pipeline on one page (diagram) and the index of the tests
    2  the words used (glossary, with a worked example of the 5-neighbour vote)
    3  the data: patent drawings and evtol.news photos, step by step, who did each step
    4  the few settings that change the result (one table) and the attention problem (one picture)
    5  drawings: can the class be read; the view; the class survives the view's removal
    6  photos: the same, plus background and colour
    7  drawings against photos: the class comparison and the fair pool
    8  change over time (maps by window, links and QR codes to the 3-D pages)
    9  conclusions, ONE robustness table, what the author still needs to review

Every number is read from the saved results (``analysis_document.load``, ``analysis_brief.load_review``, the
class_evolution CSVs); nothing is typed. One small check is computed here because no saved table held it: the
5-neighbour vote with the same-company rule switched off (``same_company_check``, written once to
review_fixes/N1_same_company_rule.csv). The 30-page brief (``analysis_brief``) is left untouched and stays
buildable.

Output: docs/embedding_evaluation/analysis_document/ANALYSIS_SIMPLE.md, its new figures in figs_simple/, then
"Embedding Analysis brief/current/Embedding Analysis — Visual Learning (private).docx" and .pdf.
Run: ``python embedding_evaluation/scripts/build_analysis_simple.py [md] [pdf] [docx]`` or notebook 32.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

PILLAR = Path(__file__).resolve().parent.parent
if str(PILLAR) not in sys.path:
    sys.path.insert(0, str(PILLAR))

from src.embedding_reports import REPO, DATA  # noqa: E402  (light: the md step imports the heavy modules)

OUT_DIR = REPO / "docs/embedding_evaluation/analysis_document"
MD_PATH = OUT_DIR / "ANALYSIS_SIMPLE.md"
FIGS = OUT_DIR / "figs_simple"
CURRENT = DATA / "Embedding Analysis brief/current"
STEM = "Embedding Analysis — Visual Learning (private)"
DOCX, PDF = CURRENT / f"{STEM}.docx", CURRENT / f"{STEM}.pdf"
TITLE = "Visual Learning: DINOv2 embeddings of patent drawings and photographs"
BASE_PY = "/home/vasco/anaconda3/bin/python"
PDF_BUILDER = REPO / "scripts/build_styled_md_pdf.py"
CLEAN_WORD = REPO / "labeling_evaluation/scripts/clean_word.py"
SOFFICE = Path("/home/vasco/.claude/skills/synced/f9481964-6bf2-406f-a63c-ceb6de3d8bd7_791092c0-338c-4d2f-"
               "b11e-f48f47809742/docx/scripts/office/soffice.py")
PAGE_CAP = 20
PAT, EVN = DATA / "1639_LABELLED", DATA / "EVTOLNEWS_DS"
SAME_CO = PAT / "3_embedding_evaluation/review_fixes/N1_same_company_rule.csv"
CE_DIR = PAT / "3_embedding_evaluation/class_evolution"
PAGES_3D = [("Drawings, 3-D map", PAT / "3_embedding_evaluation/umap_3d/PATENT_UMAP_3D_REGISTERS.html"),
            ("Drawings by period, 3-D map", PAT / "3_embedding_evaluation/umap_3d/PATENT_UMAP_3D_BY_WINDOW.html"),
            ("Photos with drawings, 3-D map", EVN / "3_embedding_evaluation/advisor_report/EVTOLNEWS_UMAP_3D.html"),
            ("Photos by period, 3-D map", EVN / "3_embedding_evaluation/umap_3d/EVTOLNEWS_UMAP_3D_BY_WINDOW.html")]
#: words the author asked never to see again (2026-10-01 review); checked on the finished text
BANNED = [r"\bmakers?\b", r"\bk\b", r"\bHolm\b", r"\bprobe\b", r"post hoc", r"κ", r"\bkappa\b", r"\bkNN",
          r"split-half", r"\boptimism\b", r"\bHopkins\b", r"\bARI\b"]


# ── the one new check ────────────────────────────────────────────────────────
def same_company_check(force: bool = False) -> pd.DataFrame:
    """The 5-neighbour vote with and without the same-company rule (registers, layer 24 CLS, primary image).
    Without the rule an aircraft may take neighbours from its own assignee / developer (never itself). The
    difference (with minus without) is bootstrapped over whole companies, as every paired contrast is."""
    if SAME_CO.exists() and not force:
        return pd.read_csv(SAME_CO)
    from . import embedding_protocol as P
    from . import review_fixes as RF
    rows = []
    for lab, (X, y, mk) in RF.label_sets(RF.REG).items():
        y = np.asarray(y, dtype=object)
        p_rule = RF.predict(X, y, mk)
        p_free = P.vote(P.neighbours(X, np.arange(len(y)), P.K), y)
        same = P.neighbours(X, np.arange(len(y)), P.K)
        mk_ = np.asarray(mk, dtype=object)
        share_same = float((mk_[same] == mk_[:, None]).mean())
        b = RF.paired_boot(y, p_free, p_rule, mk)       # with the rule minus without
        rows.append({"label_set": lab, "n_aircraft": len(y), "with_rule": RF.bal(y, p_rule),
                     "without_rule": RF.bal(y, p_free), "diff_with_minus_without": b["diff"],
                     "ci_lo_company": b["ci_lo"], "ci_hi_company": b["ci_hi"],
                     "neighbours_from_same_company_without_rule": share_same,
                     "note": "registers L24 CLS, primary image rule; without the rule = only the aircraft itself is "
                             "excluded; interval = 1 000 resamples of whole companies (seed 42)"})
    t = pd.DataFrame(rows)
    SAME_CO.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(SAME_CO, index=False)
    return t


# ── new figures ──────────────────────────────────────────────────────────────
def _save(AB, fig, name: str, note: str) -> Path:
    AB.note(name, note)
    return AB._stamped_save(fig, FIGS / name)


def fig_pipeline(AB, ER, nP: int, nH: int) -> Path:
    def n_(x):
        return f"{int(x):,}".replace(",", " ")
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    ax.axis("off")
    blue, grey = "#e8f0fb", "#f3f2ee"

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc,
                                    ec=ER.INK, lw=0.8))
        ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="top", fontsize=7.6, weight="bold", color=ER.INK)
        ax.text(x + w / 2, y + h - 5.6, body, ha="center", va="top", fontsize=6.5, color=ER.INK, linespacing=1.25)

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=0.9,
                                     color=ER.INK))
    box(0.5, 18.5, 18, 14.5, "Patent drawings", f"{nP} aircraft\none main drawing\nper aircraft", grey)
    box(0.5, 1.0, 18, 14.5, "evtol.news\nphotos", f"\n{n_(nH)} aircraft\none hero image\nper aircraft", grey)
    box(22.5, 9, 16, 16.5, "Preparation", "pad to a white\nsquare, resize\nto 518 px", grey)
    box(43.5, 9, 17, 16.5, "DINOv2", "with registers,\nfrozen: used as\nit comes, never\ntrained here", blue)
    box(65.5, 9, 15, 16.5, "One vector", "1 024 numbers\nper aircraft\n(layer 24,\nsummary token)", blue)
    box(85.5, 9, 14, 16.5, "Tests", "class read by\nthe 5 nearest\naircraft; checks\n(index below)", grey)
    arrow(19.3, 25.5, 22.2, 19.5)
    arrow(19.3, 8.5, 22.2, 14.5)
    for x0, x1 in ((39.3, 43.2), (61.3, 65.2), (81.3, 85.2)):
        arrow(x0, 17.3, x1, 17.3)
    ax.text(92.5, 6.6, "labels used only\nto score: the\nauthor's 12 types,\nthe directory's\n5 classes",
            ha="center", va="top", fontsize=5.9, color=ER.MUTED)
    return _save(AB, fig, "s_pipeline.png", "Source: the analysis pipeline of this document; counts from the "
                 "analysis sets (drawings: the author's labelling; photos: evtol.news crawl of 2026-09-17)")


def knn_example(C: Dict, P) -> Dict:
    """One worked example of the 5-neighbour vote on the drawings (12 types), chosen by a fixed rule: the first
    aircraft (in id order) whose main figure is a Perspective view, whose vote is right with exactly 3 of its 5
    neighbours in its own type, and whose 5 neighbours come from 5 different assignees."""
    S = C["S_pat"]
    X = S.rules[S.primary_rule][(24, "cls")]
    y = np.asarray(S.y, dtype=object)
    mk = np.asarray(S.makers, dtype=object)
    nn = P.neighbours(X, mk, P.K)
    pred = P.vote(nn, y)
    Xn = P.l2(X)
    ft = S.extra["figure_table"].set_index("fig_id")
    pick = S.extra["picks"][S.primary_rule].set_index("aircraft_id").fig_id
    ids = S.aircraft.aircraft_id.to_numpy()
    for i in np.argsort(ids):
        f = pick.get(ids[i])
        if f is None or ft.loc[f, "view4"] != "Perspective":
            continue
        same = int((y[nn[i]] == y[i]).sum())
        if pred[i] == y[i] and same == 3 and len(set(mk[nn[i]])) == 5 and len(set(y[nn[i]])) >= 2:
            rows = [{"aircraft": ids[j], "class": y[j], "company": S.aircraft.company.iloc[j],
                     "similarity": float(Xn[i] @ Xn[j]), "path": ft.loc[pick[ids[j]], "processed_path_518"]}
                    for j in nn[i]]
            return {"query": ids[i], "class": y[i], "company": S.aircraft.company.iloc[i],
                    "path": ft.loc[f, "processed_path_518"], "neighbours": rows, "vote": pred[i], "n_same": same}
    raise ValueError("no aircraft fits the example rule")


def fig_knn_example(AB, ER, disp, ex: Dict, name12: Dict[str, str]) -> Path:
    fig, axes = plt.subplots(1, 6, figsize=(7.6, 2.05))
    items = [("hidden class", ex["path"], f"{ex['query'].split('_')[0]}\n(answer: {disp(ex['class'])})", ER.INK)]
    for r, nb in enumerate(ex["neighbours"], 1):
        col = "#1a7f37" if nb["class"] == ex["class"] else "#b3261e"
        items.append((f"neighbour {r}", nb["path"],
                      f"{disp(nb['class'])}, {name12[nb['class']]}\nsimilarity {nb['similarity']:.2f}", col))
    for ax, (head, path, lab, col) in zip(axes, items):
        ax.imshow(Image.open(path).convert("L"), cmap="gray", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(col)
            s.set_linewidth(1.6 if head != "hidden class" else 0.6)
        ax.set_title(head, fontsize=7.2, color=ER.INK)
        ax.set_xlabel(lab, fontsize=6.2, color=col)
    fig.tight_layout(w_pad=0.4)
    return _save(AB, fig, "s_knn_example.png",
                 "Source: patent drawings, the author's main figure of each aircraft, as the model receives it; "
                 "DINOv2 with registers, layer 24 summary token; neighbours from other assignees only; green = "
                 "same type as the hidden aircraft, red = another type")


def fig_recall12(AB, ER, disp, rec: pd.DataFrame) -> Path:
    r = rec.set_index("g1").loc[ER.G1_ORDER]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    yv = np.arange(len(r))[::-1]
    for yi, (c, row) in zip(yv, r.iterrows()):
        small = bool(row.under_20)
        ax.barh(yi, row.knn_recall, color=ER.CLASS_COLOR[row.parent], alpha=0.35 if small else 0.9,
                hatch="////" if small else None, edgecolor=ER.CLASS_COLOR[row.parent], lw=0.6)
        ax.text(row.knn_recall + 0.012, yi, f"{int(row.knn_correct)} of {int(row.n_aircraft)}", va="center",
                fontsize=6.6, color=ER.INK)
    ax.axvline(1 / len(r), color=ER.MUTED, ls="--", lw=0.9)
    ax.text(1 / len(r) + 0.005, len(r) - 0.35, "chance, 1/12", fontsize=6.4, color=ER.MUTED, va="bottom")
    ax.set_yticks(yv, [f"{disp(c)}  {row.g1_name if c != 'TR' else 'Tilt Propulsor'} ({row.parent})"
                       for c, row in r.iterrows()], fontsize=6.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of the type's aircraft whose type the 5-neighbour vote names correctly", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_ylim(-0.7, len(r) - 0.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    AB.note("s_recall12.png", [(0, "Source: DINOv2 with registers, layer 24 summary token; 663 patent-drawing "
                                   "aircraft, main figure; the author's 12 types, colour = the parent class; hatched "
                                   "= fewer than 20 aircraft, too few to judge")])
    return AB._stamped_save(fig, FIGS / "s_recall12.png")


def fig_view_removal(AB, ER, esc: pd.DataFrame) -> Path:
    e = esc[esc.model == "registers"].set_index(["label_set", "vectors"])
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for ax, (lab, word, chance) in zip(axes, (("G1 12-class", "12 types", 1 / 12), ("parent 5-class", "5 classes", 0.2))):
        rows = [("as built", "original", "#2a78d6"), ("view removed", "view erasure", "#6c4fb6"),
                ("random removal\nof the same size", "variance-matched random", "#b9b7ae")]
        for i, (t, v, col) in enumerate(rows):
            r = e.loc[(lab, v)]
            ax.bar(i, r.bal_acc, color=col, width=0.62)
            ax.errorbar(i, r.bal_acc, yerr=[[r.bal_acc - r.ci_lo], [r.ci_hi - r.bal_acc]], color=ER.INK, lw=0.8,
                        capsize=2.5)
            ax.text(i, r.ci_hi + 0.012, f"{r.bal_acc:.3f}", ha="center", fontsize=6.6, color=ER.INK)
        ax.axhline(chance, color=ER.MUTED, ls="--", lw=0.9)
        ax.text(1.01, chance, f"chance\n{'1/12' if chance < 0.1 else '1/5'}", ha="left", va="center", fontsize=6.3,
                transform=ax.get_yaxis_transform(),
                color=ER.MUTED)
        ax.set_xticks(range(3), [t for t, _, _ in rows], fontsize=6.6)
        ax.set_title(f"drawings, {word}", fontsize=8, loc="left")
        ax.set_ylim(0, 0.62 if chance > 0.1 else 0.36)
        ax.tick_params(axis="y", labelsize=6.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("score (balanced accuracy)", fontsize=7)
    fig.tight_layout(w_pad=1.6)
    AB.note("s_view_removal.png", "Source: DINOv2 with registers, layer 24 summary token; 663 patent-drawing aircraft, "
                                  "main figure; view removed = the 3 directions that separate the 4 drawing views taken "
                                  "out of every vector; random removal = 3 random directions of the same size; bars: 95 % "
                                  "interval")
    return AB._stamped_save(fig, FIGS / "s_view_removal.png")


def fig_class_compare(AB, ER, r5: pd.DataFrame) -> Path:
    d = r5[r5.label_set == "patents parent 5-class"].set_index("class").loc[ER.CLASSES5]
    h = r5[r5.label_set == "photos 5-class"].set_index("class").loc[ER.CLASSES5]
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    yv = np.arange(5)[::-1]
    for yi, c in zip(yv, ER.CLASSES5):
        for off, src, row, alpha in ((0.19, "drawings", d.loc[c], 0.45), (-0.19, "photos", h.loc[c], 0.95)):
            ax.barh(yi + off, row.knn_recall, height=0.36, color=ER.CLASS_COLOR[c], alpha=alpha,
                    hatch="////" if src == "drawings" else None, edgecolor=ER.CLASS_COLOR[c], lw=0.5)
            ax.text(row.knn_recall + 0.01, yi + off, f"{src}: {int(row.knn_correct)} of {int(row.n_aircraft)}",
                    va="center", fontsize=6.2, color=ER.INK)
    ax.axvline(0.2, color=ER.MUTED, ls="--", lw=0.9)
    ax.text(0.205, 4.55, "chance, 1/5", fontsize=6.3, color=ER.MUTED)
    ax.set_yticks(yv, [f"{c}  {ER.CLASS5_NAME[c]}" for c in ER.CLASSES5], fontsize=7)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-0.6, 4.75)
    ax.set_xlabel("share of the class's aircraft whose class the 5-neighbour vote names correctly", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    AB.note("s_class_compare.png", [(0, "Source: DINOv2 with registers, layer 24 summary token; drawings: 663 aircraft, "
                                        "the author's 12 types folded to the 5 directory classes (hatched); photos: "
                                        "1 157 evtol.news aircraft, hero image, the directory's own class (solid)")])
    return AB._stamped_save(fig, FIGS / "s_class_compare.png")


def fig_qr(AB, ER) -> Path:
    import segno
    fig, axes = plt.subplots(1, len(PAGES_3D), figsize=(7.2, 2.3))
    for ax, (label, path) in zip(axes, PAGES_3D):
        buf = io.BytesIO()
        segno.make(path.as_uri(), error="l").save(buf, kind="png", scale=6, border=2)
        buf.seek(0)
        ax.imshow(Image.open(buf).convert("L"), cmap="gray", interpolation="nearest")
        ax.axis("off")
        ax.set_title(label, fontsize=7.2)
    fig.tight_layout(w_pad=0.8)
    AB.note("s_qr_3d.png", "Source: the rotatable 3-D maps of this analysis; each code holds the file link of the page "
                           "in the synced Drive folder, so it opens only on a computer that has that folder")
    return AB._stamped_save(fig, FIGS / "s_qr_3d.png")


# ── the document ─────────────────────────────────────────────────────────────
def build(C: Dict) -> Tuple[str, Dict]:
    from . import analysis_brief as AB
    from . import analysis_document as AD
    from . import embedding_protocol as P
    from . import embedding_reports as ER
    from . import umap_by_window as UW
    from . import class_evolution as CE
    from .analysis_document import ci, disp, f2, n_, pc, sg

    class Doc(AB.BDoc):
        def section(self, num: str, title: str, *intro: str) -> None:
            head = f"{num} — {title}"
            self.toc.append((0, head))
            self.L += ['<div class="keep" markdown="1">', "", f"## {head}", ""]
            self.p(*intro)
            self.L += ["</div>", ""]

        def sub(self, title: str) -> None:
            self.L += [f"#### {title}", ""]

        def figure(self, path: Path, caption: str, takeaway: str, width: int = 100) -> None:
            path = Path(path)
            if path.parent == AB.PRIVB:
                rel = f"figs_brief/private/{path.name}"
            else:
                rel = str(path.relative_to(OUT_DIR))
            self.nfig += 1
            cap = f"Figure {self.nfig} — {caption[:1].upper()}{caption[1:]}"
            self.L += [f'![{cap}]({rel}){{: width="{width}%" }}', "", f"*{cap}*", ""]
            if takeaway:
                self.L += [AB._takeaway_line(takeaway), ""]
            self.L += [""]
            self.figs.append((self.nfig, str(path)))

        def inshort(self, lead: str, items: Sequence[str] = ()) -> None:
            self.answer(lead, items, head="In short")

    D = Doc()
    info: Dict = {}
    X = AB.load_review()
    FIGS.mkdir(parents=True, exist_ok=True)
    S_pat, S_evn = C["S_pat"], C["S_evn"]
    Rp, Rh = C["R_pat"], C["R_evn"]
    mp, m5, mh = AD.main_row(Rp), AD.main_row(Rp, "ev5"), AD.main_row(Rh)
    nP, nH = len(S_pat.aircraft), len(S_evn.aircraft)
    ft = S_pat.extra["figure_table"]
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(S_pat.aircraft.aircraft_id))]
    nW, nPat = len(wv), S_pat.aircraft.patent_id.nunique()
    cnt12 = S_pat.aircraft.label.value_counts()
    L12, L5, P5 = "patents G1 12-class", "patents parent 5-class", "photos 5-class"
    s4 = X["S4_paired_contrasts"]

    def cst(prefix: str, lab: str) -> pd.Series:
        m = s4[s4.contrast.str.startswith(prefix) & (s4.label_set == lab)]
        if len(m) != 1:
            raise KeyError((prefix, lab, len(m)))
        return m.iloc[0]

    def nd(lo, hi) -> str:
        return "no difference detected" if lo <= 0 <= hi else ("higher" if lo > 0 else "lower")

    def dci(d, lo, hi) -> str:
        return f"{sg(d)} {ci(lo, hi, 3, True)}"

    def times(v, chance) -> str:
        return f"{v / chance:.1f}"
    name12 = {r.g1: ("Tilt Propulsor" if r.g1 == "TR" else r.g1_name) for r in X["M9_recall_12_wilson"].itertuples()}
    fun = C["funnel"]

    def fstep(prefix):
        return int(fun[fun.step.str.startswith(prefix)].removed.sum())
    n_file = int(fun.remaining.iloc[0])
    rk = C["reg_knn"]
    base12 = rk[(rk.tag == AD.BASE_TAG) & (rk.source == "patents") & (rk.label == "G1 12-class")
                & (rk.matrix == "L24 CLS")].bal_acc.iloc[0]
    reg = {lab: cst("registers minus base", lab) for lab in (L12, L5, P5)}
    sh = C["share"].set_index(["model", "source", "layer"])
    s9 = X["S9_attention_extras"]
    s9a = s9[s9.image_set == "all"].set_index(["model", "source", "layer"])
    pad_b = {s: float(s9a.loc[("base", s, 24)].argmax_in_pad_share) for s in ("patent", "photo")}
    pad_r = {s: float(s9a.loc[("registers", s, 24)].argmax_in_pad_share) for s in ("patent", "photo")}
    att_b = float(sh.loc[("base", "patent", 24)].max_patch_share_median)
    att_bh = float(sh.loc[("base", "photo", 24)].max_patch_share_median)
    att_r = max(float(sh.loc[("registers", s, 24)].max_patch_share_median) for s in ("patent", "photo"))
    dec = C["s1a_decision"].set_index("candidate")
    own = X["S7_own_pick_per_source"]
    own_ph = own[(own.choice == "layer and token") & (own.source == "photos")].set_index("candidate")
    hd = X["S7_heldout_differences"].set_index(["label_set", "contrast"])
    l24_l22_ph = hd.loc[(P5, "registers L24 CLS minus L22 CLS (held out, same halves)")]
    rr = X["S3_image_rules_paired"]
    rr12 = rr[rr.label_set == L12].set_index("contrast")
    rph = rr[rr.label_set == P5].set_index("contrast")
    e1, e2, e4 = (rr12.loc[f"{x} minus exp3 main figure"] for x in ("exp1 view-first", "exp2 state-first",
                                                                     "exp4 view average"))
    ph_mean, ph_rand = rph.loc["mean of all images minus hero image"], rph.loc["random other image minus hero image"]
    kc = X["s2_controls_knn"].set_index(["label", "condition"])
    prep12 = {c: float(kc.loc[(L12, c)].bal_acc) for c in ("plain", "grey", "thick1", "thick2", "contrast")}
    g_ph = cst("grey minus colour", P5)
    crop = cst("cropped minus original, photos (registers)", P5)
    ks = X["S3_k_sensitivity"].astype({"k": str}).set_index(["label_set", "k"])
    k3, k10 = ks.loc[(L12, "3")], ks.loc[(L12, "10")]
    rec12 = X["M9_recall_12_wilson"]
    r12 = rec12.set_index("g1")
    small = [c for c in ER.G1_ORDER if bool(r12.loc[c, "under_20"])]
    zero = [c for c in ER.G1_ORDER if int(r12.loc[c, "knn_correct"]) == 0]
    big = r12[~r12.under_20].sort_values("knn_recall", ascending=False)
    r5 = X["M9_recall_5class_wilson"]
    r5h = r5[r5.label_set == P5].set_index("class")
    r5d = r5[r5.label_set == L5].set_index("class")
    esc = X["M4c_erased_scores"]
    e_ = esc[esc.model == "registers"].set_index(["label_set", "vectors"])
    er12, er5 = e_.loc[("G1 12-class", "view erasure")], e_.loc[("parent 5-class", "view erasure")]
    ctl = X["M4b_control_200_draws"]
    ctl = ctl[ctl.control == "variance-matched random directions"].set_index(["model", "subset", "label_set"])
    c12 = ctl.loc[("registers", "all", L12)]
    vr = X["S11_view_ratio"].set_index("measure")
    vk = vr.loc["view kNN-5 balanced accuracy (figures, maker held out)"]
    m3 = X["M3_pool_size_and_mix"].set_index("row")
    m3_full = m3.loc["difference: photos full set minus drawings"]
    m3_size = m3.loc["difference: photos size only minus drawings"]
    m3_mix = m3.loc["difference: photos size + mix (matched) minus drawings"]
    s_size = m3.loc["photos, size only: subsets of 663 at the photos' own class mix"]
    s_mix = m3.loc["photos, size + mix: subsets of 663 at the drawings' class counts"]
    cc_d = dict(x.split() for x in m3.loc["drawings (parent 5-class, exp3 main figure)"].class_counts.split("|"))
    cc_h = dict(x.split() for x in m3.loc["photos, full set (hero image)"].class_counts.split("|"))
    auw = X["S10_audit_wilson"].set_index("sample")
    sc = same_company_check()
    sc = sc.set_index("label_set")
    im = C["img_dec"]
    n_img = len(im)
    hard = im[im.hard_rule != ""].hard_rule.value_counts()
    n_keep = int((im.auto == "keep").sum())
    n_sdrop = int(((im.auto == "drop") & (im.hard_rule == "")).sum())
    n_rev = int((im.auto == "review").sum())
    n_manual = int((im.manual.astype(str) != "").sum())
    fsum = json.loads((EVN / "1_filter/filter_summary.json").read_text())
    n_pages = len(C["pages"])
    yrs = pd.read_csv(EVN / "0_source/years/aircraft_dev_year.csv", keep_default_na=False)
    ystat = yrs.dev_year_status.value_counts()
    cov = pd.read_csv(CE_DIR / "ce_coverage.csv").set_index("measure").aircraft
    tk = pd.read_csv(CE_DIR / "ce_temporal_knn.csv").set_index("class")
    drift = pd.read_csv(CE_DIR / "ce_drift.csv")
    conv = pd.read_csv(CE_DIR / "ce_convergence.csv").set_index(["source", "class"])
    sep = pd.read_csv(CE_DIR / "umap_by_window_separation.csv")
    e3 = pd.read_csv(PAT / "2_embedding_extraction/view_state_experiments/exp3_main.csv", keep_default_na=False)
    n_fallback = int(e3.rule_applied.str.startswith("FALLBACK").sum())
    conv_types = sorted(set(wv[wv.g1_group == "variant"].g1_code), key=ER.G1_ORDER.index)
    n_both = int(((wv.g1_group == "variant") & (wv.state4 == "Both")).sum())
    n_dupfile = fstep("same image file")
    chance12, chance5 = 1 / 12, 1 / 5

    # ═══ title ═══
    D.add(f"# {TITLE}", "", f"<!-- Date: {dt.date.today().isoformat()} -->", "",
          "Short version for using DINOv2 as a tool: what was done, what was found, and how robust it is. Private: "
          "contains evtol.news photographs, © their owners.", "", "---", "")

    # ═══ 1 pipeline ═══
    D.section("1", "The pipeline on one page",
              "**The question.** Does a picture alone carry the architecture of an aircraft? A general-purpose vision "
              "model, DINOv2, that has never seen the codebook or been trained on aircraft, turns each image into a "
              "list of numbers. If aircraft of the same class get similar numbers, the class can be read back from "
              "the picture. The same model and the same tests are applied to two image sources: the patent drawings "
              "of the author's labelling, and the photographs of the evtol.news directory.")
    f = fig_pipeline(AB, ER, nP, nH)
    D.figure(f, "the pipeline: from one image per aircraft to one vector per aircraft, then the tests.",
             "**Every test reads the same vectors: one per aircraft, from one frozen model.** The labels are never "
             "shown to the model; they are used only to score what the model did.", width=100)
    D.p("**The index of the tests.** The setup checks come first, because they fix the settings every later test "
        "uses; then the tests on each source, the comparison of the two, and the change over time.")
    idx = [("Setup", "S1 attention check", "Does the model waste its attention on blank background?", "4"),
           ("Setup", "S2 layer and token", "Which internal reading of the model holds the class best?", "4"),
           ("Setup", "S3 image preparation", "Should drawings or photos be changed before the model reads them?", "4"),
           ("Setup", "S4 image per aircraft", "Which image should stand for each aircraft?", "3, 4"),
           ("Drawings", "T1 valid vectors", "Is every vector usable (no empty, broken or duplicate rows)?", "5"),
           ("Drawings", "T2 class read", "Do the 5 most similar aircraft of other assignees name the right class?", "5"),
           ("Drawings", "T3 class by class", "Which classes are read well, and which rest on too few aircraft?", "5"),
           ("Drawings", "T4 what else", "Does the drawing view shape the vectors more than the class does?", "5"),
           ("Drawings", "T5 view removed", "Is the class still read once the view is taken out of the vectors?", "5"),
           ("Photos", "T1 to T3, T6", "The same tests; does the background or the colour matter?", "6"),
           ("Both", "T7 fair comparison", "Do photos carry more of the class than drawings, at equal pool size and "
                                          "class mix?", "7"),
           ("Both", "T8 over time", "Do the classes change their appearance across the years?", "8"),
           ("All", "robustness", "Does any single choice decide the result?", "9")]
    D.table(pd.DataFrame(idx, columns=["phase", "test", "what it asks", "section"]),
            "The tests of this document, in the order they are run.", "", right=["section"])

    # ═══ 2 glossary ═══
    D.section("2", "The words used in this document",
              "The terms below carry the whole document. Each is explained once here.")
    D.p(f"**Embedding (vector).** The model turns an image into a list of {n_(1024)} numbers, its embedding or "
        "vector. Two images that look alike to the model get similar lists. The model is **frozen**: it is used "
        "exactly as published and never trained on these images.",
        "**Layer and token.** The model reads an image in 24 steps, called layers. It cuts the image into small "
        "square tiles of 14 by 14 pixels, called patches (a 518-pixel image has 37 by 37 of them), and keeps one "
        "extra summary token (called CLS) that gathers the whole image. The vector used here is the summary token "
        "after the last layer, layer 24 (why: section 4).",
        "**Similarity.** How alike two vectors are, measured by the angle between them (cosine similarity): 1 "
        "means they point the same way, 0 that they have nothing in common. Only the direction counts, not the "
        "size of the numbers.")
    ex = knn_example(C, P)
    fk = fig_knn_example(AB, ER, disp, ex, name12)
    nbs = ex["neighbours"]
    votes = pd.Series([n["class"] for n in nbs]).value_counts()
    vote_txt = ", ".join(f"{v} {disp(c)}" for c, v in votes.items())
    D.p("**The 5-neighbour vote, the main test.** Take one aircraft and hide its class. Find the 5 aircraft whose "
        "vectors are most similar to it, taking them only from other assignees. Each of the 5 votes for its own "
        "class; the class with most votes is the answer (a tie goes to the class of the most similar one). The "
        "answer is right or wrong. This is repeated for every aircraft. 5 is a small, common number of voters, "
        f"fixed before any result was seen; 3 or 10 voters give the same score within its interval (section 9). "
        f"Figure {D.fignum()} is a real case: the hidden aircraft is a {name12[ex['class']]} "
        f"({disp(ex['class'])}) of patent {ex['query'].split('_')[0]}; its 5 nearest aircraft vote {vote_txt}, so "
        f"the vote says {name12[ex['vote']]}: right. The class is not always one of the 12 types: the same vote is "
        "also run on the 5 broader classes, which group the types (section 3).")
    D.figure(fk, "the 5-neighbour vote on one drawing: the hidden aircraft and its 5 most similar aircraft of other "
                 "assignees, with their types and similarities.",
             f"**{int(ex['n_same'])} of the 5 neighbours share the hidden type, so the vote names it correctly.** "
             "The model never saw a label: it only placed similar drawings close together.", width=100)
    D.p("**The score (balanced accuracy).** For each class, the share of its aircraft whose class the vote names "
        "correctly; then the average of these shares over the classes. Every class counts the same, so a large "
        "class cannot hide a class that is never read.",
        f"**Chance level, and how to read a score.** A blind guess is right once in 12 on the 12 types "
        f"({f2(chance12, 3)}) and once in 5 on the 5 classes ({f2(chance5, 2)}). A score is read against that level: "
        f"{f2(mp.knn_bal_acc, 3)} on 12 types is about {times(mp.knn_bal_acc, chance12)} times chance. As a check, "
        f"the same vote with the class labels shuffled at random stays at or below {f2(mp.knn_chance_p95, 2)} in "
        "95 of 100 shuffles.",
        "**The same-company rule.** Aircraft of the same company are never compared with each other: the neighbours "
        "of a patent aircraft always come from other assignees, those of a photographed aircraft from other "
        "developers. A company's drawing style or photo studio therefore cannot pass for architecture.",
        "**The 95 % interval.** The score is recomputed on 1 000 random resamples of the aircraft, and the middle "
        "95 % of those scores is the interval, written [low, high]. When two settings are compared, whole companies "
        "are resampled, so the interval allows for one company's aircraft being alike. A difference whose interval "
        "includes zero is read as **no difference detected**.",
        "**The map (UMAP).** A drawing of the 1 024-number vectors on a flat page (or in 3-D), placing similar "
        "aircraft close together. Its axes have no unit. A map is a picture to look at, never a test.")

    # ═══ 3 data ═══
    D.section("3", "The data",
              "Two sources, each reduced to one image and one vector per aircraft.")
    D.sub("3.1 Patent drawings")
    D.p(f"The drawings come from the author's labelling of the eVTOL patents. Of the {n_(n_file)} figures on file, "
        f"the author approved {n_(n_file - fstep('figure not approved'))}; {n_(fstep('domain gate'))} more leave "
        f"because their aircraft is similar to a drone or not electric, {n_(fstep('duplicate patent'))} because "
        f"their patent duplicates another, {n_dupfile} because it repeats an image file of another figure row, and "
        f"{n_(fstep('not a whole-aircraft'))} because they show a part. **{n_(nW)} whole-aircraft drawings of "
        f"{n_(nP)} aircraft from {n_(nPat)} patents remain.** Every aircraft carries the author's type, one of the "
        "codebook's 12, and its parent class, one of 5 (Table 2).")
    rows = []
    for c5 in ER.CLASSES5:
        types = [c for c in ER.G1_ORDER if ER.PARENT[c] == c5]
        rows.append((f"{c5} {ER.CLASS5_NAME[c5]}", ", ".join(f"{disp(c)} {name12[c]} ({int(cnt12.get(c, 0))})"
                                                             for c in types),
                     n_(sum(int(cnt12.get(c, 0)) for c in types)), n_(int(cc_h[c5]))))
    D.table(pd.DataFrame(rows, columns=["class (directory)", "codebook types under it (drawing aircraft)",
                                        "drawing aircraft", "photo aircraft"]),
            "The 5 classes, the 12 types they group, and the aircraft of each source.",
            f"**Two types, Lift + Cruise (SLC) and Tilt Propulsor (TP), hold "
            f"{pc((cnt12.get('SLC', 0) + cnt12.get('TR', 0)) / nP)} of the drawing aircraft, while "
            f"{len(small)} types have fewer than 20 ({', '.join(disp(c) for c in small)}).** TP is printed for the "
            "wizard's TR (Tilt Propulsor).", right=["drawing aircraft", "photo aircraft"])
    D.p(f"**One drawing per aircraft, by fixed priorities.** An aircraft has a median of two whole-aircraft "
        f"drawings, so one must be chosen. (1) The figure the author marked as **main** in the wizard; this "
        f"decides for {n_(nP - n_fallback)} of the {n_(nP)} aircraft. (2) If no single whole-aircraft main exists "
        f"({n_fallback} aircraft), the best **view**, in the order Perspective > Plan (top or bottom) > Side > "
        "Front or Rear; within the same view, the best **flight state**, in the order Both > Cruise > Hover > "
        "Other > Missing; then the lowest figure number. Both, the moving part drawn in both positions, is the best "
        f"state and comes first (author ruling of 2026-10-01). The flight state only matters for the 5 convertible "
        f"types ({', '.join(disp(c) for c in conv_types)}); every other type has one state. The other "
        "ways of choosing (view first, flight state first, an average of views) are compared in section 4.",
        "**What the model receives.** The chosen drawing is rotated by the angle recorded in the labelling, padded "
        "to a white square and resized to 518 by 518 pixels, black lines on white, numerals left in (Figure "
        f"{D.fignum()}).")
    AB.note("f_examples_patents.png", "Source: patent drawings, the author's main figure as the model receives it; "
                                      "the first aircraft of each type whose main figure is a Perspective view; label: "
                                      "the author's type (parent class); code under each drawing: the aircraft")
    with AB._into_brief():
        fex = AD.fig_examples_patents(C)
    D.figure(fex, "one main drawing per type, as the model receives it (518 px, padded white square).",
             "**The model sees black line-work on white; nothing in the image names the type.**", width=100)

    import shutil
    fph = AB.PRIVB / "f02_examples.png"
    shutil.copyfile(AD.EE / "embedding_analysis_registers/figs/f02_examples.png", fph)
    D.sub("3.2 evtol.news photos, step by step")
    D.p(f"The photos come from the World eVTOL Aircraft Directory of the Vertical Flight Society (evtol.news). "
        f"Table 3 lists each step, who or what did it, and whether the author still has to review it.")
    steps = [
        ("1 crawl", f"{n_(n_pages)} aircraft pages and {n_(n_img)} images downloaded (2026-09-17)", "script",
         "no"),
        ("2 class", "the class is the directory list the page is on (5 lists)", "script", "no"),
        ("3 fixed rules", f"drop {int(hard.get('too_small', 0))} images under 200 px, "
                          f"{int(hard.get('same_file_on_other_page', 0))} files shared with another page, "
                          f"{int(hard.get('other_aircraft_page', 0))} images of another aircraft", "script", "no"),
        ("4 whole-aircraft filter", f"an AI image model (SigLIP) scores whether each image shows a whole aircraft: "
                                    f"keep at 0.80 or more ({n_(n_keep)}), drop under 0.35 ({n_(n_sdrop)}), the "
                                    f"{n_rev} in between held back, never decided", "SigLIP + script",
         f"optional: the {n_rev} held-back images"),
        ("5 filter audit", f"a sample of {int(auw.loc['main'].n)} hero images, {int(auw.loc['siglip_drop'].n)} dropped "
                           f"and {int(auw.loc['review_band'].n)} held-back images judged by eye: "
                           f"{int(auw.loc['main']['count'])} of {int(auw.loc['main'].n)} heroes show the whole aircraft",
         "Claude, from contact sheets that showed the score", "**yes**: not blind, not yet checked"),
        ("6 hero image", f"the page's first kept image stands for the aircraft: {n_(nH)} aircraft", "script",
         "no (covered by 5)"),
        ("7 style guess", "photo, render or drawing; used only to leave line drawings out of one check", "SigLIP",
         "no (unreliable, kept as a score)"),
        ("8 crop to the aircraft", "the aircraft cut out of its background, for the background check only",
         "AI detector (OWLv2)", "no"),
        ("9 development year", f"two independent Claude readers pick the earliest dated sentence about the aircraft "
                               f"on its page; a script checks it: agreed for {n_(int(ystat.get('agreed', 0)))}, "
                               f"no date for {n_(int(ystat.get('not_stated', 0)))}, disagreement for "
                               f"{n_(int(ystat.get('conflict', 0)))} of the {n_(n_pages)} pages",
         "Claude (2 readers) + script", f"**yes**: the {int(ystat.get('conflict', 0))} disagreements, and a spot "
                                        "check of agreed years")]
    D.table(pd.DataFrame(steps, columns=["step", "what", "done by", "author review needed"]),
            "The evtol.news photo pipeline: each step, who or what did it, and what is left to review.",
            f"**{n_(nH)} aircraft enter, one hero image each; only the filter audit and the years wait for the "
            f"author.** No image was decided by hand ({n_manual} manual rulings).")
    D.figure(fph, f"one hero image per directory class, as the model receives it before padding. {AB.CREDIT}.",
             "**A hero image shows the whole aircraft, a photograph or a rendering, often in a scene.**", width=100)

    # ═══ 4 settings ═══
    D.section("4", "The few settings that change the result",
              "A frozen model leaves few choices. Table 4 lists the ones that can move the score: what each is, the "
              "value chosen, why, and one check of how much it matters. Everything else was fixed in advance. "
              "A choice made on the scores was made on half of the assignees and confirmed on the other half: "
              "the assignees were split at random into two halves 200 times, the best value was picked on one half "
              "and checked on the other, and the halves were swapped, giving 400 choices. A choice that wins most "
              "of the 400 does not depend on which companies happened to be in the data.")
    set_rows = [
        ("model", "DINOv2-large as published, or its version with 4 extra 'register' tokens",
         "with registers", f"the base model parks about a third of its attention on one blank patch (Figure "
                           f"{D.fignum()})",
         f"base {f2(base12, 3)}, registers {f2(mp.knn_bal_acc, 3)} on the 12 types: "
         f"{dci(reg[L12]['diff'], reg[L12].ci_lo_maker, reg[L12].ci_hi_maker)}; photos "
         f"{sg(reg[P5]['diff'])}, {nd(reg[P5].ci_lo_maker, reg[P5].ci_hi_maker)}"),
        ("layer and token", "which of the 24 steps is read, and the summary token or the average of the patches",
         "layer 24, summary token", "chosen most often on half of the assignees and confirmed on the other half",
         f"chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of 400 choices; runner-up layer 22 summary "
         f"{pc(dec.loc['registers L22 CLS'].selection_freq)}; on photos the two tie "
         f"({sg(l24_l22_ph.mean_diff)})"),
        ("image size", "pixels per side given to the model", "518 px", "the model's largest training size; keeps "
                                                                      "thin lines", "not tested: fixed in advance"),
        ("image per aircraft", "which drawing or photo stands for the aircraft",
         "drawings: the main figure; photos: the hero image", "one representative image per aircraft, chosen by "
                                                               "the source, fixed in advance",
         f"drawings: view first {f2(e1.score_rule, 3)}, flight state first {f2(e2.score_rule, 3)}, view average "
         f"{f2(e4.score_rule, 3)}, all below the main figure's {f2(e1.score_ref, 3)}; photos: average of all images "
         f"{sg(ph_mean['diff'])}, {nd(ph_mean.ci_lo_maker, ph_mean.ci_hi_maker)}"),
        ("drawing preparation", "the lines as filed, or changed: made about 1 or 2 pixels thicker per side, "
                                "made solid black (darker), or turned grey", "as filed",
         "no change tested helps",
         f"greyscale {f2(prep12['grey'], 3)} (same as {f2(prep12['plain'], 3)}); thicker "
         f"{f2(prep12['thick1'], 3)} and {f2(prep12['thick2'], 3)}; darker {f2(prep12['contrast'], 3)}")]
    D.table(pd.DataFrame(set_rows, columns=["setting", "what it is", "chosen", "why", "check (score on the 12 types "
                                                                                   "unless stated)"]),
            "The settings that change the result.",
            "**Every chosen value scores best or ties; the model and the image per aircraft matter most, and no "
            "change to the drawings helps.**")
    D.p(f"**The attention problem, in plain words.** Inside the model, the summary token decides how much to look "
        f"at each patch. In the base model, at layers 22 and 24, one single patch, usually a blank one in the white "
        f"padding at the corner, takes about {f2(att_b, 2)} of all the attention on drawings ({f2(att_bh, 2)} on "
        f"photos): the pale square in the top-left corner of Figure {D.fignum()}. That patch lies in the blank "
        f"padding for {pc(pad_b['patent'])} of the drawings and {pc(pad_b['photo'])} of the photos. The model uses it as a "
        "scratch pad, a known defect of this model family. The registers version adds 4 extra tokens that take this "
        f"role, so at layer 24 no patch takes more than {f2(att_r, 3)} and the blank padding is never the focus "
        f"({pc(pad_r['patent'])}). The switch was decided after seeing the scores of both models; it was then "
        "confirmed on half of the assignees.")
    AB.note("f_attention_base_vs_registers.png",
            "Source: DINOv2-L and DINOv2-L with registers, frozen, 518 px; summary-token (CLS) attention, mean over "
            "heads, each map scaled to its own 99th percentile; 2 drawings and 2 photos, first aircraft of each "
            f"class in both runs. {AB.CREDIT}")
    with AB._into_brief():
        fat = AD.fig_attention_models(C)
    D.figure(fat, "where the summary token looks: base model (columns 2 to 4) against the registers version (columns "
                  "5 to 7), layers 18, 22 and 24, on the same images; bright = more attention.",
             "**Base layers 22 and 24 put a large share of attention on one pale corner patch of the blank pad; "
             "with registers the attention stays on the aircraft.**", width=100)

    # ═══ 5 drawings ═══
    D.section("5", "Drawings: can the class be read?",
              f"All {n_(nP)} aircraft have a valid vector (no empty, broken or duplicate rows). The vote reads the "
              f"12 types at **{f2(mp.knn_bal_acc, 3)} {ci(mp.knn_ci_lo, mp.knn_ci_hi)}**, about "
              f"{times(mp.knn_bal_acc, chance12)} times chance ({f2(chance12, 3)}), and the 5 classes at "
              f"**{f2(m5.knn_bal_acc, 3)} {ci(m5.knn_ci_lo, m5.knn_ci_hi)}**, about {times(m5.knn_bal_acc, chance5)} "
              f"times chance ({f2(chance5, 2)}). The class is in the drawings, weakly.")
    fr = fig_recall12(AB, ER, disp, rec12)
    bal12 = X["M6_balanced_recall"].set_index("g1")
    lifted = [c for c in zero if bal12.loc[c, "knn_recall_mean"] > bal12.loc[c, "shuffled_recall_p95"]]
    ptc_line = (f"{', '.join(disp(c) for c in lifted)}: the zero is partly the type's small share of the pool; with "
                "every type equally present it is read above chance." if lifted else "")
    D.figure(fr, "how often each type is read correctly, with its number of aircraft.",
             f"**The large types are read best: {disp(big.index[0])} {f2(big.knn_recall.iloc[0], 2)} and "
             f"{disp(big.index[1])} {f2(big.knn_recall.iloc[1], 2)}. {', '.join(disp(c) for c in zero)} are never "
             f"read; {', '.join(disp(c) for c in small)} have fewer than 20 aircraft, too few to judge.** "
             + ptc_line, width=88)
    D.p(f"**What else the model sees: the view.** The drawing view (Perspective, Plan, Side, Front or Rear) is the "
        f"strongest other signal. The same vote names the view of a drawing at {f2(vk.estimate, 2)}, against "
        f"{f2(1 / 4, 2)} by chance with 4 views: the model sorts drawings by how they are drawn "
        f"before what they show. When the computer is asked to split the aircraft into 12 groups on its own, the "
        f"groups barely match the types (agreement {f2(mp.t5_ari, 2)}, where 0 is chance and 1 a perfect match).",
        f"**The class survives the removal of the view.** The 3 directions of the vectors that separate the 4 views "
        f"were taken out of every vector. The score falls to {f2(er12.bal_acc, 3)} on the 12 types and "
        f"{f2(er5.bal_acc, 3)} on the 5 classes, still well above chance (Figure {D.fignum()}). Removing 3 random "
        f"directions of the same size costs about as much ({sg(c12.control_mean)} on average over 200 tries, "
        f"against {sg(c12.erasure_diff)} for the view): the class information is spread through the vectors, not "
        "hidden in the view.")
    fv = fig_view_removal(AB, ER, esc)
    D.figure(fv, "the score before and after the view is removed, and after a random removal of the same size.",
             "**The class is still read well above chance without the view.**", width=88)
    D.inshort(f"Patent drawings carry the class, weakly: {f2(mp.knn_bal_acc, 3)} on 12 types and "
              f"{f2(m5.knn_bal_acc, 3)} on 5 classes, 2 to 3 times chance, and it survives the removal of the view.")

    # ═══ 6 photos ═══
    D.section("6", "Photos: can the class be read?",
              f"All {n_(nH)} aircraft have a valid vector. The vote reads the directory's 5 classes at "
              f"**{f2(mh.knn_bal_acc, 3)} {ci(mh.knn_ci_lo, mh.knn_ci_hi)}**, about {times(mh.knn_bal_acc, chance5)} "
              f"times chance ({f2(chance5, 2)}).")
    rows = [(f"{c} {ER.CLASS5_NAME[c]}", n_(int(r5h.loc[c].n_aircraft)), f"{int(r5h.loc[c].knn_correct)} "
             f"({f2(r5h.loc[c].knn_recall, 2)})") for c in ER.CLASSES5]
    D.table(pd.DataFrame(rows, columns=["class", "aircraft", "read correctly (share)"]),
            "How often each directory class is read correctly from its hero photo.",
            f"**{ER.CLASS5_NAME[r5h.knn_recall.idxmax()]} is read best ({f2(r5h.knn_recall.max(), 2)}), "
            f"{ER.CLASS5_NAME[r5h.knn_recall.idxmin()]} worst ({f2(r5h.knn_recall.min(), 2)}); every class is above "
            "chance.**", right=["aircraft", "read correctly (share)"])
    D.p(f"**Background and colour do not drive the score.** Cutting each aircraft out of its background (AI "
        f"detector OWLv2) changes the score by {dci(crop['diff'], crop.ci_lo_maker, crop.ci_hi_maker)}, and "
        f"greyscale photos by {dci(g_ph['diff'], g_ph.ci_lo_maker, g_ph.ci_hi_maker)}: no difference detected in "
        f"either, so the photos are used as they come, in colour and uncropped. Using the average of all of an aircraft's photos instead of the hero image gives "
        f"{sg(ph_mean['diff'])}, no difference detected; a random other photo of the page scores "
        f"{sg(ph_rand['diff'])} lower.")
    D.inshort(f"Photos carry the class moderately: {f2(mh.knn_bal_acc, 3)} on the 5 classes, about "
              f"{times(mh.knn_bal_acc, chance5)} times chance, whatever the background or colour.")

    # ═══ 7 drawings vs photos ═══
    D.section("7", "Drawings against photos",
              "The two sources are compared on the 5 classes they share: the author's 12 types folded to their "
              "parent class (Table 2) against the directory's own class. The sources hold different aircraft and "
              "a different mix of classes, so the comparison is repeated with the photo pool made like the drawing "
              f"pool: the same size ({n_(nP)} aircraft), then also the same number of aircraft per class.")
    d5 = f2(m3.loc['drawings (parent 5-class, exp3 main figure)'].estimate, 3)
    rows = [("full sets", d5,
             f2(m3.loc["photos, full set (hero image)"].estimate, 3), dci(m3_full.estimate, m3_full.ci_lo, m3_full.ci_hi),
             nd(m3_full.ci_lo, m3_full.ci_hi)),
            (f"photos cut to {nP} aircraft, their own class mix", d5, f2(s_size.estimate, 3),
             dci(m3_size.estimate, m3_size.ci_lo, m3_size.ci_hi), nd(m3_size.ci_lo, m3_size.ci_hi)),
            (f"photos cut to {nP} aircraft with the drawings' class mix (fair pool)", d5, f2(s_mix.estimate, 3),
             dci(m3_mix.estimate, m3_mix.ci_lo, m3_mix.ci_hi), nd(m3_mix.ci_lo, m3_mix.ci_hi))]
    D.table(pd.DataFrame(rows, columns=["comparison", "drawings", "photos", "photos minus drawings", "reading"]),
            "Drawings against photos on the 5 shared classes.",
            f"**Photos score higher on the full sets ({sg(m3_full.estimate)}); in a fair pool the difference shrinks "
            f"to {sg(m3_mix.estimate)}, no difference detected.** The photo lead comes mostly from the class mix: "
            f"Wingless multicopters are {pc(int(cc_d['WM']) / nP)} of the drawings and "
            f"{pc(int(cc_h['WM']) / nH)} of the photos.")
    fcc = fig_class_compare(AB, ER, r5)
    D.figure(fcc, "how often each class is read correctly from drawings (hatched) and from photos (solid).",
             f"**A class is read well when it is common in its pool:** Vectored Thrust, "
             f"{pc(int(cc_d['VT']) / nP)} of the drawings, is read better from drawings "
             f"({f2(r5d.loc['VT'].knn_recall, 2)} against {f2(r5h.loc['VT'].knn_recall, 2)}); Wingless, common "
             f"among the photos, better from photos ({f2(r5h.loc['WM'].knn_recall, 2)} against "
             f"{f2(r5d.loc['WM'].knn_recall, 2)}).", width=88)

    D.inshort(f"In a fair pool, drawings and photos read the 5 classes alike ({sg(m3_mix.estimate)}, no difference "
              "detected); the photos' lead on the full sets comes from their class mix.")

    # ═══ 8 time ═══
    tka = tk.loc["all (balanced accuracy)"]
    moved = drift[drift.reading.str.startswith("moved")]
    moved_vt = moved[moved["class"] == "VT"]
    pooled_h = conv.loc[("photos", "pooled")]
    pooled_d = conv.loc[("drawings", "pooled")]
    sp = sep.set_index(["source", "window"])
    D.section("8", "Change over time",
              f"Each drawing is dated by its patent's priority year, each photographed aircraft by its development "
              f"year (step 9 of Table 3; only the {n_(int(cov['agreed year']))} agreed years of the {n_(nH)} "
              f"aircraft, {n_(int(cov['not stated']))} with no date and {n_(int(cov['conflict (not ruled)']))} "
              "disagreements left out). The years are grouped in windows: up to 2011, 2012-15, 2016-19, 2020-23 "
              "and 2024-26 (partial).")
    D.p(f"**The classes keep their appearance.** The {n_(int(tka.n_queries))} photographed aircraft of 2020-26 are "
        f"read as well from neighbours of 2019 or earlier ({f2(tka.past, 2)}) as from neighbours of their own "
        f"period ({f2(tka.same_period, 2)}): {dci(tka.diff_same_minus_past, tka.diff_ci_lo, tka.diff_ci_hi)}, no "
        "difference detected. Recent designs look like the earlier ones of their class.",
        f"**One class moves: Vectored Thrust.** The centre of each class (the average of its vectors) was compared "
        f"between 2016-19 and 2020-23 against the moves that random reshuffles of the years produce. Vectored "
        f"Thrust's centre moves beyond that range in both sources "
        f"(only {_join([f'{pc(r.p, 1)} of the reshuffles in the {r.source}' for r in moved_vt.itertuples()])} move "
        f"it as far), a move strong enough to stand even allowing for the {int(drift.tested.sum())} class tests made. "
        + (f"Wingless also moves in the drawings ({pc(moved[(moved['class'] == 'WM')].p.iloc[0], 1)} of the "
           "reshuffles move it as far), but not beyond that allowance. "
           if ((moved["class"] == "WM") & (moved.source == "drawings")).any() else "")
        + "Vectored Thrust groups the most different machines (tilting rotors, wings, ducts), so a change in "
        "which of them are proposed would move its centre. Pooled over the classes, photographed designs sit "
        f"slightly closer to their class the later they are (correlation {f2(pooled_h.rho, 2)} "
        f"{ci(pooled_h.ci_lo, pooled_h.ci_hi, 2, True)}); the drawings show no such trend ({f2(pooled_d.rho, 2)} "
        f"{ci(pooled_d.ci_lo, pooled_d.ci_hi, 2, True)}).")
    U = UW.load_saved()
    n_dated = int(cov["agreed year"])
    for src, word, extra in (("drawings", "drawing", f"{n_(nP)} patent-drawing aircraft (main figure), parent class; "
                                                     "windows by the priority year of the patent"),
                             ("photos", "photo", f"{n_(nH)} evtol.news aircraft, hero image, all in the fit; the "
                                                 f"{n_(n_dated)} with an agreed development year in the panels")):
        AB.note(f"f_umap_by_window_{src}.png",
                f"Source: DINOv2-L with registers, frozen, 518 px; layer 24 summary token; {extra}; 2-D UMAP (15 "
                "neighbours, min. distance 0.1, cosine, seed 42), one fit on all aircraft; centre = mean map position "
                "of the class in the window; axes: UMAP dimensions, no unit")
        fu = AB.fig_umap_by_window(U, src)
        D.figure(fu, f"the {word} map, one panel per period: each period's aircraft in class colour, all others in "
                     "grey; the large marker is the class centre in that period, the line its earlier centres.",
                 "", width=90)
    D.inshort(f"The classes keep their appearance across the years; only Vectored Thrust's centre moves, in both "
              "sources.")
    D.p(f"**The 3-D maps.** The maps can be turned and zoomed in 3-D, with a slider for the periods. The links "
        "below open the pages on a computer that has the synced Drive folder; the codes in Figure "
        f"{D.fignum()} hold the same links. They are file links: to open them on a phone, the pages must first be "
        "shared from Drive.")
    D.bullets([f"[{label}]({path.as_uri()})" for label, path in PAGES_3D])
    fq = fig_qr(AB, ER)
    D.figure(fq, "codes for the four 3-D maps.", "", width=100)

    # ═══ 9 conclusions ═══
    sc12, sc5, sch = sc.loc[L12], sc.loc[L5], sc.loc[P5]
    D.section("9", "Conclusions, robustness and what is left to review")
    D.bullets([
        f"**A frozen general-purpose model reads the architecture from one picture.** Drawings: "
        f"{f2(mp.knn_bal_acc, 3)} on 12 types and {f2(m5.knn_bal_acc, 3)} on 5 classes; photos: "
        f"{f2(mh.knn_bal_acc, 3)} on 5 classes; all 2 to 3 times chance.",
        "**Drawings are first sorted by how they are drawn (the view), then by what they show**; the class survives "
        "the removal of the view.",
        f"**No medium difference is detected once the pools are made alike**: the photo lead "
        f"({sg(m3_full.estimate)}) shrinks to {sg(m3_mix.estimate)} with the drawings' class mix.",
        "**Large classes are read best**; types with fewer than 20 aircraft cannot be judged.",
        "**The classes keep their appearance over time**; only Vectored Thrust's centre moves, in both sources."])
    rob = [
        ("registers against base model", f"{dci(reg[L12]['diff'], reg[L12].ci_lo_maker, reg[L12].ci_hi_maker)} on "
                                         f"12 types; photos {sg(reg[P5]['diff'])}",
         "registers better on drawings; no difference detected on photos"),
        ("layer choice made on other assignees", f"layer 24 summary chosen in "
                                                 f"{pc(dec.loc['registers L24 CLS'].selection_freq)} of 400 choices",
         "stable"),
        ("number of voters (3 or 10 instead of 5)", f"{f2(k3.bal_acc, 3)} and {f2(k10.bal_acc, 3)} against "
                                                    f"{f2(mp.knn_bal_acc, 3)}", "no difference detected"),
        ("image per aircraft", f"other drawing rules {f2(min(e1.score_rule, e2.score_rule, e4.score_rule), 3)} to "
                               f"{f2(max(e1.score_rule, e2.score_rule, e4.score_rule), 3)}; photo average "
                               f"{sg(ph_mean['diff'])}", "main figure best; the hero photo is enough"),
        ("photo background cut away", dci(crop["diff"], crop.ci_lo_maker, crop.ci_hi_maker), "no difference detected"),
        ("colour removed", f"photos {sg(g_ph['diff'])}; drawings {sg(prep12['grey'] - prep12['plain'])}",
         "no difference detected"),
        ("view removed from the drawings", f"{f2(er12.bal_acc, 3)} on 12 types, chance {f2(chance12, 3)}",
         "class survives"),
        ("fair pool (same size and class mix)", dci(m3_mix.estimate, m3_mix.ci_lo, m3_mix.ci_hi),
         "no medium difference detected"),
        ("same-company rule switched off", f"{f2(sc12.without_rule, 3)} against {f2(sc12.with_rule, 3)} on 12 types; "
                                           f"photos {f2(sch.without_rule, 3)} against {f2(sch.with_rule, 3)}; "
                                           f"{pc(sc12.neighbours_from_same_company_without_rule)} of drawing "
                                           "neighbours then come from the same assignee",
         "the rule is needed: a company's aircraft look alike" if sc12.without_rule - sc12.with_rule > 0.02 else
         "the rule changes little")]
    D.table(pd.DataFrame(rob, columns=["check", "result", "verdict"]),
            "Robustness: each check, its result and the verdict.",
            "**No single choice decides the result.**")
    D.p("**What the author still needs to review.**")
    D.bullets([
        f"**Photo filter audit.** The audit (Table 3, step 5) was judged by Claude from sheets that showed the filter "
        f"score, so it is not blind; the author should re-check it, and may decide the {n_rev} held-back images.",
        f"**Development years.** The {int(ystat.get('conflict', 0))} pages where the two Claude readers disagree "
        "wait for the author's ruling, and a spot check of agreed years is advised; section 8 uses agreed years only.",
        "**Rotated drawings.** Some drawings may reach the model rotated by 90°, because the rotation labels of the "
        "wizard are not reliable. This is a known limitation of every drawing result here; it is not yet fixed.",
        f"**One duplicate drawing file.** The author plans to delete one image so the drawing set goes from "
        f"{n_(int(fun[fun.step.str.startswith('duplicate patent')].remaining.iloc[0]))} to "
        f"{n_(int(fun[fun.step.str.startswith('same image file')].remaining.iloc[0]))} figure rows; the "
        "pipeline already leaves that repeated file out."])
    D.p("**References.** Oquab et al. (2023), DINOv2, arXiv:2304.07193. Darcet et al. (2024), Vision transformers "
        "need registers, ICLR 2024. Zhai et al. (2023), Sigmoid loss for language image pre-training (SigLIP), "
        "ICCV 2023. Minderer et al. (2023), Scaling open-vocabulary object detection (OWLv2), NeurIPS 2023. Cover "
        "and Hart (1967), Nearest neighbor pattern classification. Brodersen et al. (2010), The balanced accuracy "
        "and its posterior distribution. Efron (1979), Bootstrap methods. McInnes et al. (2018), UMAP, "
        "arXiv:1802.03426.")
    info.update({"figs": D.figs, "ntab": D.ntab, "example": {k: v for k, v in ex.items() if k != "neighbours"}})
    return D.text(), info


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def check_text(text: str) -> None:
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)(\{[^}]*\})?", "", text)
    body = re.sub(r"\]\(file://[^)]*\)", "]", body)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    prose = re.sub(r"^#{2,4} [\d.]+ — ", "", body, flags=re.M)
    prose = re.sub(r"\*(?:Figure|Table) \d+ — ", "", prose)
    prose = re.sub(r"\*\*(?:Figure|Table) \d+ — ", "", prose)
    if "—" in prose:
        bad = [l[:100] for l in prose.split("\n") if "—" in l]
        raise ValueError(f"em dash in prose: {bad[:3]}")
    hits = [(rx, m.group(0)) for rx in BANNED for m in re.finditer(rx, prose)]
    if hits:
        raise ValueError(f"banned words: {hits[:6]}")
    out = re.findall(r"[\w/.-]+\.(?:csv|py|pkl|md|json)\b|/mnt/", prose)
    if out:
        raise ValueError(f"pointer out of the document: {sorted(set(out))[:5]}")


def write_markdown() -> Tuple[Path, Dict]:
    from . import analysis_document as AD
    from . import analysis_brief as AB
    AB.FIGB.mkdir(parents=True, exist_ok=True)
    AB.PRIVB.mkdir(parents=True, exist_ok=True)
    if not AB.PRIV_LINK.exists():
        os.symlink(AB.PRIVB, AB.PRIV_LINK)
    C = AD.load()
    text, info = build(C)
    if AB._PENDING:
        raise ValueError(f"source notes registered for figures never drawn: {sorted(AB._PENDING)}")
    check_text(text)
    MD_PATH.write_text(text, encoding="utf-8")
    (OUT_DIR / "_simple_build_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    return MD_PATH, info


# ── build: md (Finetune env), pdf and docx (base env) ───────────────────────
def _pages(pdf: Path) -> int:
    out = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout
    return int(next(l for l in out.splitlines() if l.startswith("Pages:")).split()[-1])


def pdf_step() -> None:
    CURRENT.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(PDF_BUILDER), str(MD_PATH), str(PDF), "--compact"], check=True)
    print(f"wrote {PDF}  ({_pages(PDF)} pages)")


def docx_step() -> int:
    CURRENT.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(CLEAN_WORD), str(MD_PATH), str(DOCX), "--title", TITLE])
    if r.returncode != 0:
        raise SystemExit("the Word export failed")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([sys.executable, str(SOFFICE), "--headless", "--convert-to", "pdf", "--outdir", td,
                        str(DOCX)], check=True, capture_output=True)
        n = _pages(next(Path(td).glob("*.pdf")))
    print(f"wrote {DOCX}  (LibreOffice: {n} pages, cap {PAGE_CAP})")
    if n > PAGE_CAP:
        raise SystemExit(f"the Word file has {n} pages, over the cap of {PAGE_CAP}")
    return n


def main(argv: List[str]) -> None:
    steps = [a for a in argv if not a.startswith("--")] or ["md", "pdf", "docx"]
    if "md" in steps:
        path, info = write_markdown()
        print("wrote", path)
        for n, p in info["figs"]:
            print(f"  Figure {n}: {p}")
    rest = [s for s in ("pdf", "docx") if s in steps]
    if not rest:
        return
    try:
        import docx  # noqa: F401
        import markdown  # noqa: F401
    except ImportError:
        subprocess.run([BASE_PY, str(PILLAR / "scripts/build_analysis_simple.py")] + rest, check=True)
        return
    if "pdf" in rest:
        pdf_step()
    if "docx" in rest:
        docx_step()
