"""The DINOv2 analysis in the format of the Labelling Analysis brief.

Same analyses and numbers as ``analysis_document.py`` (ANALYSIS.md), reorganised to the author's brief
standard (1639_LABELLED/1_labelling_analysis/labelling_analysis/BRIEF_REVISION_RULES.md, sections 2, 3,
4 and 7): chapters that open on a neutral question, sub-questions with a Method in five labelled parts
(Objective, Data and unit of analysis, Procedure, Scientific grounds, Assumptions and limits) and a
Results and discussion part where every figure and table carries one takeaway, one methods box at the
head of chapter 1, an Answer box closing every chapter, and a closing section on what the results rest
on and what is still open.

Every number is read from the saved results through ``analysis_document.load`` and the same
expressions ``analysis_document.build`` uses; nothing is typed. ``analysis_document`` itself is not
changed: its figure functions are called with their output folders and their ``_save`` redirected
(``_into_brief``), so the figures the reviewers of ANALYSIS.md may open are never overwritten.

Figures carry their source note INSIDE the image (rule 4 of the brief standard): one note per panel
for multi-panel charts, one note under an image grid. They go to
``docs/embedding_evaluation/analysis_document/figs_brief/``; the figures that show evtol.news photos go
to the private ``EVTOLNEWS_DS/3_embedding_evaluation/analysis_brief_figs/`` and are reached through the
symlink ``figs_brief/private`` (the markdown and Word exporters need paths without spaces).

Review round of 2026-09-30 (docs/embedding_evaluation/analysis_document/review/CONSOLIDATED_FLAGS.md): the
numbers of the review fixes are read from 1639_LABELLED/3_embedding_evaluation/review_fixes/ (``load_review``),
the setup addendum (grey and contrast controls) and the registers crop test; the review figures are redrawn for
the brief by ``rf_figure`` (source note without file names, no how-to-read line: that lives in the Method).
Paired contrasts quote the maker-resampled interval. BRIEF_APPENDIX=0 drops Appendix A (page cap).

Output: docs/embedding_evaluation/analysis_document/ANALYSIS_BRIEF.md.
Run: embedding_evaluation/scripts/build_analysis_brief.py (Finetune env for md; pdf and docx under the
base python).
"""

from __future__ import annotations

import os
import re
import shutil
import textwrap
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402
from PIL import Image  # noqa: E402

from . import analysis_document as AD  # noqa: E402
from . import embedding_protocol as P  # noqa: E402
from . import embedding_reports as ER  # noqa: E402
from .analysis_document import ci, disp, f2, n_, pc, sg  # noqa: E402
from .brief_toc import toc_entries, toc_md  # noqa: E402

OUT_DIR = AD.OUT_DIR
MD_PATH = OUT_DIR / "ANALYSIS_BRIEF.md"
FIGB = OUT_DIR / "figs_brief"
PRIVB = AD.EE / "analysis_brief_figs"
PRIV_LINK = FIGB / "private"
TITLE = "Visual Learning: DINOv2 embeddings of patent drawings and photographs"
CREDIT = "Photo: evtol.news directory, © owners"
MODEL = "DINOv2-L with registers, frozen, 518 px"
MUTED = ER.MUTED
DASH_OK = [re.compile(r"^#{2,3} (?:\d+(?:\.\d+)?|Appendix [A-Z]) — "),
           re.compile(r"\b(?:Figure|Table) \d+ — "),
           re.compile(r'^<div class="toc-row')]


# ── in-image source notes ────────────────────────────────────────────────────
_PENDING: Dict[str, object] = {}          # file name -> figure note (str) or [(axes index, note), ...]


def _wrap(text: str, width_in: float, fs: float) -> str:
    per_line = max(38, int(width_in * 72 / (fs * 0.52)))
    return "\n".join(textwrap.fill(t, per_line) for t in text.split("\n"))


def _panel_note(ax, text: str, fs: float = 6.3) -> None:
    """A source note hung at the foot of one panel, under its lowest tick label or axis label,
    measured at draw time so it never meets a tick (the brief's ``la_figures._panel_note``)."""
    fig = ax.figure
    fig.canvas.draw()
    width_in = ax.get_window_extent().width / fig.dpi

    def foot(rend):
        hb = ax.get_window_extent(rend)
        tb = ax.xaxis.get_tightbbox(rend)
        y0 = min(hb.y0, tb.y0) if tb is not None else hb.y0
        return Bbox([[hb.x0, y0], [hb.x1, y0]])
    ax.annotate(_wrap(text, width_in, fs), xy=(0, 0), xycoords=foot, xytext=(0, -4), textcoords="offset points",
                ha="left", va="top", fontsize=fs, color=MUTED, annotation_clip=False)


def _fig_note(fig, text: str, fs: float = 6.3) -> None:
    """One note under everything drawn (an image grid is one panel)."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_figwidth(), fig.get_figheight()
    fig.text(max(bb.x0 / w, 0.0), bb.y0 / h - 0.05 / h, _wrap(text, min(bb.width, 7.0), fs), fontsize=fs,
             color=MUTED, ha="left", va="top")


def _stamped_save(fig, path) -> Path:
    path = Path(path)
    spec = _PENDING.pop(path.name, None)
    if spec is None:
        raise KeyError(f"no source note registered for {path.name}")
    if isinstance(spec, str):
        _fig_note(fig, spec)
    else:
        for i, text in spec:
            _panel_note(fig.axes[i], text)
    path.parent.mkdir(parents=True, exist_ok=True)
    AD._check_codes(fig)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06, dpi=200)
    plt.close(fig)
    return path


@contextmanager
def _into_brief():
    """analysis_document's figure functions, writing stamped copies into the brief folders."""
    saved = (AD._save, AD.FIG_DIR, AD.PRIV_DIR)
    AD._save, AD.FIG_DIR, AD.PRIV_DIR = _stamped_save, FIGB, PRIVB
    try:
        yield
    finally:
        AD._save, AD.FIG_DIR, AD.PRIV_DIR = saved


def note(name: str, spec) -> None:
    _PENDING[name] = spec


def fig_attention_source(C: Dict, source: str, per_class: int = 2, seed: int = P.SEED) -> Path:
    """analysis_document.fig_attention_panel split by source: the random draws are replayed in the
    same order (drawings then photos, class by class, seed 42), so the aircraft are the same as in
    the combined panel; only the requested source is drawn."""
    rng = np.random.default_rng(seed)
    pa = C["S_pat"].aircraft.set_index("aircraft_id")
    pho = C["S_evn"].aircraft.set_index("aircraft_id")["label"]
    ap, mp = AD._att("patent", AD.REG_TAG)
    ah, mh = AD._att("photo", AD.REG_TAG)
    mp = mp[mp.aircraft_uid.isin(pa.index)].copy()
    mp["c5"] = mp.aircraft_uid.map(pa["label5"])
    mh = mh[mh.aircraft_uid.isin(pho.index)].copy()
    mh["c5"] = mh.aircraft_uid.map(pho)
    picks = []
    for c in ER.CLASSES5:
        for src, meta, arr in [("drawing", mp, ap), ("photo", mh, ah)]:
            cand = meta[meta.c5 == c]
            take = cand.iloc[np.sort(rng.choice(len(cand), per_class, replace=False))]
            if src == source:
                picks.append((c, take, arr))
    fig, axes = plt.subplots(5, 2 * per_class, figsize=(5.0, 6.2))
    for i, (c, take, arr) in enumerate(picks):
        col = 0
        for r in take.itertuples():
            img = np.asarray(Image.open(r.path).convert("L"))
            lab = disp(pa.loc[r.aircraft_uid, "label"]) if source == "drawing" else c
            for j, L in ((0, 18), (2, 24)):
                ax = axes[i, col]
                AD._overlay(ax, img, arr[r.Index, j])
                if i == 0:
                    ax.set_title(f"L{L}", fontsize=8.5)
                if j == 0:
                    ax.set_xlabel(lab, fontsize=8, color=ER.INK)
                col += 1
        axes[i, 0].set_ylabel(f"{c}\n{ER.CLASS5_NAME[c]}".replace(" (Multicopter)", "").replace(" / PFD", ""),
                              fontsize=8.5, rotation=0, ha="right", va="center")
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    name = f"f_attention_{'drawings' if source == 'drawing' else 'photos'}.png"
    return _stamped_save(fig, (FIGB if source == "drawing" else PRIVB) / name)


def _takeaway_line(t: str) -> str:
    """An italic takeaway line. A takeaway that ends in bold would close on '***', which both markdown
    readers misparse (a stray '**' was printed); underscores for the italic keep it unambiguous."""
    return f"_Takeaway: {t}_" if t.rstrip().endswith("**") else f"*Takeaway: {t}*"


# ── the document ─────────────────────────────────────────────────────────────
class BDoc:
    """Markdown in the brief's shape: numbered figures and tables, each with one takeaway."""

    def __init__(self):
        self.L: List[str] = []
        self.nfig = 0
        self.ntab = 0
        self.toc: List[Tuple[int, str]] = []
        self.figs: List[Tuple[int, str]] = []

    def add(self, *lines: str) -> None:
        self.L += list(lines)

    def p(self, *paras: str) -> None:
        for s in paras:
            self.L += [s, ""]

    def bullets(self, items: Sequence[str]) -> None:
        self.L += [f"- {s}" for s in items] + [""]

    def chapter(self, num: str, q: str, *intro: str) -> None:
        head = f"{num} — {q}"
        self.toc.append((0, head))
        self.L += ['<div class="keep chapter-start" markdown="1">', "", f"## {head}", ""]
        self.p(*intro)
        self.L += ["</div>", ""]

    def closing(self, head: str) -> None:
        self.toc.append((0, head))
        self.L += ['<div class="keep chapter-start" markdown="1">', "", f"## {head}", "", "</div>", ""]

    def method(self, num: str, q: str, objective: str, data: str, proc: str, grounds: str,
               limits: str = "") -> None:
        head = f"{num} — {q}"
        self.toc.append((1, head))
        self.L += ['<div class="keep" markdown="1">', "", f"### {head}", "", "#### Method", "",
                   f"**Objective.** {objective}", "", "</div>", ""]
        self.p(f"**Data and unit of analysis.** {data}", f"**Procedure.** {proc}",
               f"**Scientific grounds.** {grounds}")
        if limits:
            self.p(f"**Assumptions and limits.** {limits}")
        self.L += ['<div class="keep" markdown="1">', "", "#### Results and discussion", "", "</div>", ""]

    def fignum(self, k: int = 1) -> int:
        return self.nfig + k

    def tabnum(self, k: int = 1) -> int:
        return self.ntab + k

    def figure(self, path: Path, caption: str, takeaway: str, width: int = 100) -> None:
        self.nfig += 1
        path = Path(path)
        rel = f"figs_brief/private/{path.name}" if path.parent == PRIVB else str(path.relative_to(OUT_DIR))
        cap = f"Figure {self.nfig} — {caption[:1].upper()}{caption[1:]}"
        self.L += [f'![{cap}]({rel}){{: width="{width}%" }}', "", f"*{cap}*", "", _takeaway_line(takeaway), "", ""]
        self.figs.append((self.nfig, str(path)))

    def table(self, df: pd.DataFrame, caption: str, takeaway: str, right: Sequence[str] | None = None) -> None:
        self.ntab += 1
        cols = [str(c) for c in df.columns]
        rows = [[str(v) for v in r] for r in df.itertuples(index=False)]
        if right is None:
            right = [c for i, c in enumerate(cols) if i > 0 and all(
                re.fullmatch(r"[−+\-\d.,% \[\]/]+|n/a|", r[i] or "") for r in rows)]
        sep = ["---:" if c in right else ":---" for c in cols]
        self.L += [f"**Table {self.ntab} — {caption}**", "", "| " + " | ".join(cols) + " |",
                   "|" + "|".join(sep) + "|"]
        self.L += ["| " + " | ".join(r) + " |" for r in rows] + [""]
        if takeaway:
            self.L += [_takeaway_line(takeaway), ""]
        self.L += [""]

    def answer(self, lead: str, items: Sequence[str], head: str = "Answer") -> None:
        self.L += ['<div class="answer-box" markdown="1">', "", f"#### {head}", "", f"**{lead}**", ""]
        self.bullets(items)
        self.L += ["</div>", "", ""]

    def text(self) -> str:
        return "\n".join(self.L)


def md_df(rows, cols) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols)


# ── the review round of 2026-09-30: its saved numbers and figures ────────────
RF_DIR = AD.PE / "review_fixes"
RF_TABLES = ["M3_pool_size_and_mix", "M3_per_class_pool_share", "M4a_heldout_view_probe", "M4b_control_200_draws",
             "M4c_erased_scores", "M6_balanced_recall", "M9_recall_12_wilson", "M9_recall_5class_wilson",
             "M9_wrong_predictions", "M11_ari_like_for_like", "M11_linked_99_vs_94", "S2_probe_12", "S2_probe_5",
             "S2_probe_photos", "S3_k_sensitivity", "S3_image_rules_paired", "S3_image_rules_split_half",
             "S4_paired_contrasts", "S6_matching_intervals", "S7_best_heldout_criterion", "S7_heldout_differences",
             "S7_own_pick_per_source", "S8_halves_missing_classes", "S8_joint_grid_labels", "S8_seeds",
             "S8_table12_corrected", "S9_attention_extras", "S10_audit_wilson", "S10_kappa", "S10_kappa_by_type",
             "S11_d_units", "S11_view_ratio"]
SEL_EXTRA = ["s1b_v2_candidates", "s1b_v2_decision", "s1b_v2_labels", "s2_controls_knn", "s2_controls_halves",
             "fairness_grey_grey"]
GT_FILE = AD.PAT / "0_labelling/inputs/text_architecture/architecture_ground_truth.csv"


def load_review() -> Dict[str, pd.DataFrame]:
    """Every table of the review round the brief reads (review_fixes/, the setup addendum, the registers crop)."""
    X = {n: pd.read_csv(RF_DIR / f"{n}.csv", keep_default_na=False, na_values=[""]) for n in RF_TABLES}
    X.update({n: pd.read_csv(AD.SEL / f"{n}.csv") for n in SEL_EXTRA})
    X["crop_reg_knn"] = pd.read_csv(AD.CROP / "knn_original_vs_cropped_registers.csv")
    X["gt_rows"] = pd.DataFrame({"n": [len(pd.read_csv(GT_FILE, keep_default_na=False))]})
    return X


def _clean_source(s: str) -> str:
    """The review figures name their CSVs; the brief points nowhere outside itself."""
    s = re.sub(r";?\s*review_fixes\s+[A-Za-z0-9_]+\.csv(?:,\s*[A-Za-z0-9_]+\.csv)*\.?", ".", s)
    return re.sub(r"\.\s*\.", ".", re.sub(r"\s{2,}", " ", s)).strip()


def _rf_stamp(fig, path, source: str, read: str) -> Path:
    """review_fixes._stamp_save for the brief: the source note only (how to read lives in the Method), no file
    names, saved into figs_brief/."""
    from matplotlib.text import Text
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    x0, y0 = bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight()
    chars = max(int(min(bb.width, 7.0) * 15.5), 60)
    fig.text(x0, y0 - 0.012, "\n".join(textwrap.wrap("Source: " + _clean_source(source), chars)), fontsize=7,
             color=MUTED, ha="left", va="top")
    fig.canvas.draw()
    for t in fig.findobj(Text):
        s = t.get_text()
        if re.search(r"\bTR\b", s) or "—" in s or re.search(r"\.csv\b", s):
            raise ValueError(f"figure text breaks the print rules: {s!r}")
    out = FIGB / Path(path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def rf_figure(name: str) -> Path:
    """One review figure, redrawn for the brief. The forest leaves out the single-seed variance-matched control
    rows: the brief shows the 200-draw control in the erasure figure instead."""
    import tempfile
    from . import review_fixes as RF
    saved = RF._stamp_save
    RF._stamp_save = _rf_stamp
    try:
        if name != "fig_forest":
            return getattr(RF, name)(RF_DIR)
        with tempfile.TemporaryDirectory() as td:
            s4 = pd.read_csv(RF_DIR / "S4_paired_contrasts.csv", keep_default_na=False, na_values=[""])
            s4[s4.group != "erasure control"].to_csv(Path(td) / "S4_paired_contrasts.csv", index=False)
            shutil.copyfile(RF_DIR / "M3_pool_size_and_mix.csv", Path(td) / "M3_pool_size_and_mix.csv")
            return RF.fig_forest(Path(td), width=3.3)
    finally:
        RF._stamp_save = saved


def _join(items: Sequence[str]) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _short_contrast(r) -> str:
    """A paired contrast of the review table in a few words: 'thick2 on the 12 types'."""
    name = r.contrast.split(",")[0].replace(" minus plain", "").replace(" minus original", "").replace(
        "contrast only", "contrast").replace("view erased", "the view erasure")
    lab = {"patents G1 12-class": "the 12 types", "patents parent 5-class": "the 5 classes",
           "photos 5-class": "the photos"}[r.label_set]
    return f"{name}{' (Perspective only)' if 'Perspective' in r.contrast else ''} on {lab}"


PREPS = ["plain", "grey", "contrast", "thick1", "thick2"]


def fig_preparations(rows=("TR", "SLC", "HB")) -> Path:
    """The five drawing preparations as the model receives them: the plain input with its detail window, then
    the same 128 px detail under each preparation (the montage aircraft of the setup record, seed 42)."""
    pick = AD.thick_picks()
    pick = pick[pick.g1.isin(rows)].set_index("g1").loc[list(rows)].reset_index()
    root = AD.PAT / "2_embedding_extraction"
    m0 = pd.read_csv(root / "processed/518/manifest.csv")
    paths = {"plain": dict(zip(m0.figure_uid, m0.path))}
    for v in PREPS[1:]:
        m = pd.read_csv(root / "preprocess_variants" / v / "processed/518/manifest.csv")
        paths[v] = dict(zip(m.figure_uid, m.path))
    fig, axes = plt.subplots(len(pick), 1 + len(PREPS), figsize=(9.0, 1.55 * len(pick)))
    for i, r in enumerate(pick.itertuples()):
        ims = {c: Image.open(paths[c][r.fig_id]).convert("RGB") for c in PREPS}
        x0, y0 = AD._ink_window(ims["plain"])
        AD._thumb(axes[i, 0], ims["plain"], "plain, whole input" if i == 0 else "")
        axes[i, 0].add_patch(plt.Rectangle((x0, y0), 128, 128, fill=False, edgecolor="#c92f26", lw=0.8))
        for j, c in enumerate(PREPS):
            AD._thumb(axes[i, 1 + j], ims[c].crop((x0, y0, x0 + 128, y0 + 128)).resize((256, 256), Image.Resampling.NEAREST),
                      f"{c}, detail" if i == 0 else "")
        axes[i, 0].set_ylabel(f"{disp(r.g1)}\n{r.aircraft_id}", fontsize=7.5, rotation=0, ha="right", va="center")
    fig.tight_layout(w_pad=0.3)
    return _stamped_save(fig, FIGB / "f_preparations.png")


# ── the text ─────────────────────────────────────────────────────────────────
def build(C: Dict) -> Tuple[str, Dict]:
    D = BDoc()
    info: Dict = {}
    X = C["rf"] if "rf" in C else load_review()
    S_pat, S_evn = C["S_pat"], C["S_evn"]
    Rp, Rh, Bp, Bh = C["R_pat"], C["R_evn"], C["B_pat"], C["B_evn"]
    mp, mh, m5 = AD.main_row(Rp), AD.main_row(Rh), AD.main_row(Rp, "ev5")
    nP, nH = len(S_pat.aircraft), len(S_evn.aircraft)
    ft = S_pat.extra["figure_table"]
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(S_pat.aircraft.aircraft_id))]
    nW = len(wv)
    n_pat_patents = S_pat.aircraft.patent_id.nunique()
    cnt12 = S_pat.aircraft.label.value_counts()
    cnt5h = S_evn.aircraft.label.value_counts()
    small = [c for c in ER.G1_ORDER if cnt12.get(c, 0) < 20]
    a = S_pat.aircraft
    gt = a.arch_gt != ""
    from sklearn.metrics import cohen_kappa_score
    kap = cohen_kappa_score(a.label[gt], a.arch_gt[gt])
    agree, gt_n = int((a.label == a.arch_gt).sum()), int(gt.sum())
    n_excl = len(pd.read_csv(AD.PAT / "2_embedding_extraction/view_state_experiments/excluded_aircraft.csv"))
    rk = C["reg_knn"]

    def rkv(tag, src, lab):
        return rk[(rk.tag == tag) & (rk.source == src) & (rk.label == lab) & (rk.matrix == "L24 CLS")].iloc[0]
    dec = C["s1a_decision"].set_index("candidate")
    _c = C["s1a_candidates"]
    reg_only = _c[_c.pool == "registers only"]
    l22f = reg_only[reg_only.candidate == "registers L22 CLS"].set_index("label").selection_freq
    s1a_all = _c[_c.pool == "all 12"]
    reg_share = s1a_all[s1a_all.candidate.str.startswith("registers")].groupby("label").selection_freq.sum()
    lab_reg = C["s1a_labels"][C["s1a_labels"].pool == "registers only"].set_index("label")
    lab_all = C["s1a_labels"][C["s1a_labels"].pool == "all 12"].set_index("label")
    half = C["s1a_half_scores"]
    hn = half.groupby("label").n_aircraft.agg(["min", "max"])
    sh = C["share"].set_index(["model", "source", "layer"])
    mt = C["match"]
    mt = mt[(mt.images == "no line drawings") & (mt.layer == 24) & (mt.pooling == "cls")].set_index("direction")
    mpp, mph = mt.loc["patent->photo"], mt.loc["photo->patent"]
    ex4 = S_pat.extra["picks"]["exp4 view average"]
    nslots = ex4.groupby("aircraft_id").size()
    single = {k: v.set_index("aircraft_id").fig_id for k, v in S_pat.extra["picks"].items() if k != "exp4 view average"}

    def same(x, y):
        return float((single[x].reindex(a.aircraft_id) == single[y].reindex(a.aircraft_id)).mean())
    br = Rp["ev"]["by_rule"]
    br = br[br.matrix == "L24 CLS"].set_index("rule")
    brh = Rh["ev"]["by_rule"]
    brh = brh[brh.matrix == "L24 CLS"].set_index("rule")
    conf_p = Rp["confound"].set_index("matrix")
    conf_h = Rh["confound"].set_index("matrix")
    lay_p = Rp["layer"].set_index("matrix")
    cb, ch = conf_p.loc["L24 CLS"], conf_h.loc["L24 CLS"]
    cache_f = OUT_DIR / "_brief_cache.pkl"
    cache = None
    if os.environ.get("BRIEF_REUSE") == "1" and cache_f.exists():
        import pickle
        cache = pickle.loads(cache_f.read_bytes())       # written by the last full run of this module
    cmx = cache["class_mix"] if cache else AD.class_mix_matched(C)
    info["class_mix"] = cmx
    e4 = cache["exp4"] if cache else AD.exp4_subset(C)
    info["exp4"] = e4
    im, au = C["img_dec"], C["audit"]
    kept = im[im.keep.astype(str) == "True"]
    n_rev = int((im.auto == "review").sum())
    pch = C["parent_check"]
    n_link, n_agree = len(pch), int(pch.agrees.sum())
    fun = C["funnel"]
    n_file = int(fun.remaining.iloc[0])
    n_notappr = int(fun[fun.step == "figure not approved"].removed.iloc[0])
    n_gate = int(fun[fun.step.str.startswith("domain gate")].removed.sum())
    n_dup = int(fun[fun.step.str.startswith("duplicate patent")].removed.sum())
    n_part = int(fun[fun.step == "not a whole-aircraft figure"].removed.iloc[0])
    conv = AD.CONVERTIBLE
    var = wv[wv.g1_code.isin(conv)]
    pc_ = var[(var.view4 == "Perspective") & (var.state4 == "Cruise")].aircraft_id.nunique()
    ph_ = var[(var.view4 == "Perspective") & (var.state4 == "Hover")].aircraft_id.nunique()
    n_var = var.aircraft_id.nunique()
    n_both = int((var.state4 == "Both").sum())
    sm1 = pd.read_csv(AD.PAT / "2_embedding_extraction/preprocess_variants/thick1/sources_manifest.csv",
                      keep_default_na=False)
    sm2 = pd.read_csv(AD.PAT / "2_embedding_extraction/preprocess_variants/thick2/sources_manifest.csv",
                      keep_default_na=False)
    n_col = int((pd.to_numeric(sm1.src_max_chroma) > 8).sum())
    rh = Rh["ev"]["recall"]
    rh = rh[rh.matrix == "L24 CLS"].iloc[0]
    rbest = max(ER.CLASSES5, key=lambda c: float(rh[c]))
    rworst = min(ER.CLASSES5, key=lambda c: float(rh[c]))
    T, BT = AD.REG_TAG, AD.BASE_TAG
    ga, pa_ = "architecture kNN-5, G1 12-class", "architecture kNN-5, parent 5-class"
    ve = lambda tag, sec, var_, meas: AD.ve_get(C, tag, sec, var_, meas)  # noqa: E731
    vp_o = ve(T, "view probe (aircraft main figures)", "original", "probe_bal_acc").value
    vp_e = ve(T, "view probe (aircraft main figures)", "view erasure", "probe_bal_acc").value
    vk_e = ve(T, "view kNN-5 (figures, maker held out)", "view erasure", "bal_acc")
    e12, e5 = ve(BT, ga, "view erasure", "bal_acc").value, ve(BT, pa_, "view erasure", "bal_acc").value
    r12, r5 = ve(T, ga, "view erasure", "bal_acc").value, ve(T, pa_, "view erasure", "bal_acc").value
    n_status = int(S_evn.aircraft.status.isin(["built", "concept"]).sum())
    L12, L5, P5 = "patents G1 12-class", "patents parent 5-class", "photos 5-class"
    WORD = {L12: "drawings, 12 types", L5: "drawings, 5 classes", P5: "photos, 5 classes"}

    def nd(lo, hi) -> str:                         # the reading of an interval of a difference
        return "no difference detected" if lo <= 0 <= hi else ("above zero" if lo > 0 else "below zero")

    # review round: paired contrasts with the maker-resampled interval as the primary one
    s4 = X["S4_paired_contrasts"]

    def cst(prefix: str, lab: str | None = None) -> pd.Series:
        m = s4[s4.contrast.str.startswith(prefix)]
        if lab is not None:
            m = m[m.label_set == lab]
        if len(m) != 1:
            raise KeyError((prefix, lab, len(m)))
        return m.iloc[0]

    def mci(r) -> str:
        return f"{sg(r['diff'])} {ci(r.ci_lo_maker, r.ci_hi_maker, 3, True)}"

    def mnd(r) -> str:
        return nd(r.ci_lo_maker, r.ci_hi_maker)
    n_con, n_holm = len(s4), int((s4.p_maker_holm < 0.05).sum())
    flips = s4[((s4.ci_hi_aircraft < 0) | (s4.ci_lo_aircraft > 0)) & (s4.ci_lo_maker <= 0) & (s4.ci_hi_maker >= 0)]
    reg = {lab: cst("registers minus base", lab) for lab in (L12, L5, P5)}
    base_sc = {L12: rkv(BT, "patents", "G1 12-class").bal_acc, L5: rkv(BT, "patents", "parent 5-class").bal_acc,
               P5: rkv(BT, "photos", "5-class").bal_acc}
    # attention extras (both models on the same images)
    s9 = X["S9_attention_extras"]
    s9a = s9[s9.image_set == "all"].set_index(["model", "source", "layer"])
    pad = {(m_, s_): float(s9a.loc[(m_, s_, 24)].argmax_in_pad_share) for m_ in ("base", "registers")
           for s_ in ("patent", "photo")}
    n_att_p = int(sh.loc[("registers", "patent", 24)].n_images)
    n_att_pb = int(sh.loc[("base", "patent", 24)].n_images)
    reg_max = max(sh.loc[("registers", s, L)].max_patch_share_median for s in ("patent", "photo") for L in (18, 22, 24))
    reg_lo = min(sh.loc[("registers", s, L)].prefix_share_median for s in ("patent", "photo") for L in (18, 24))
    reg_hi = max(sh.loc[("registers", s, L)].prefix_share_median for s in ("patent", "photo") for L in (18, 24))
    # selection
    t12 = X["S8_table12_corrected"]
    t12 = t12[t12.pool == "registers only"].set_index("label_set")
    jg = X["S8_joint_grid_labels"].set_index("label_set")
    own = X["S7_own_pick_per_source"]

    def own_pick(choice, source, rank):
        r = own[(own.choice == choice) & (own.source == source) & (own["rank"] == rank)].iloc[0]
        return r.candidate.replace("registers ", ""), float(r.mean_selection_freq)
    hd = X["S7_heldout_differences"].set_index(["label_set", "contrast"])
    l24_l22_ph = hd.loc[(P5, "registers L24 CLS minus L22 CLS (held out, same halves)")]
    seeds = X["S8_seeds"]
    seed_list = sorted(int(s) for s in seeds.seed.unique() if int(s) != P.SEED)
    seed_same = all(seeds[(seeds.seed == s_) & (seeds.rule.str.startswith("S1(a)"))].decision_winner_registers_pool.eq(
        "registers L24 CLS").all() for s_ in seeds.seed.unique())
    miss = X["S8_halves_missing_classes"]
    miss12 = miss[(miss.label_set == L12) & miss.halves_missing_any_class.notna()].iloc[0]
    best_same = bool((X["S7_best_heldout_criterion"].by_selection_frequency ==
                      X["S7_best_heldout_criterion"].by_best_mean_heldout_score).all())
    head12 = mp.knn_bal_acc - jg.loc[L12, "joint_optimism_gap"]
    # preparation
    kc = X["s2_controls_knn"].set_index(["label", "condition"])
    hv = X["s2_controls_halves"].set_index(["label", "comparison"])
    v2c = X["s1b_v2_candidates"].set_index(["label", "candidate"])
    v2d = X["s1b_v2_decision"].set_index("candidate")
    plain_grey = float(v2d.loc["plain"].selection_freq + v2d.loc["grey"].selection_freq)
    same_pred = {lab: float(pd.read_csv(AD.SEL / "s2_controls_paired.csv").set_index(["label", "comparison"]).loc[
        (lab, "grey - plain")].agree) for lab in (L12, L5)}
    g_ph = cst("grey minus colour", P5)
    s1b = C["s1b_decision"].set_index("candidate")
    # the drawings
    rec12 = X["M9_recall_12_wilson"].set_index("g1")
    bal12 = X["M6_balanced_recall"].set_index("g1")
    wrong = X["M9_wrong_predictions"].set_index("measure")
    w_share = float(wrong.loc["share of wrong predictions that name SLC or TP"].value)
    w_pool = float(wrong.loc["share of the aircraft pool that is SLC or TP"].value)
    w_n = int(wrong.loc["wrong predictions (kNN-5, L24 CLS, 12 classes)"].value)
    w_k = int(wrong.loc["wrong predictions that name SLC or TP"].value)
    zero = [c for c in ER.G1_ORDER if int(rec12.loc[c, "knn_correct"]) == 0]
    above = [c for c in zero if bal12.loc[c, "knn_recall_mean"] > bal12.loc[c, "shuffled_recall_p95"]]
    notabove = [c for c in zero if c not in above]
    pr = {lab: X[f"S2_probe_{s}"].iloc[0] for lab, s in ((L12, "12"), (L5, "5"), (P5, "photos"))}
    ks = X["S3_k_sensitivity"].set_index(["label_set", "k"])
    kby = X["S10_kappa_by_type"].set_index("g1_wizard")
    ptc_dis = int(kby.loc["PTC", "disagreements"]) if "PTC" in kby.index else 0
    vr = X["S11_view_ratio"].set_index("measure")
    du = X["S11_d_units"].set_index("unit")
    d_air = du[du.index.str.startswith("aircraft")].d.iloc[0]
    d_fig = du[du.index.str.startswith("all whole-aircraft")].d.iloc[0]
    d_nonmain = du[du.index.str.startswith("non-main")].d.iloc[0]
    d_views = {v: du[du.index.str.contains(f"one view only: {v}")].d.iloc[0] for v in ("Perspective", "Plan", "Side", "Front/Rear")}
    vk = vr.loc["view kNN-5 balanced accuracy (figures, maker held out)"]
    vprobe = X["M4a_heldout_view_probe"].set_index(["model", "fit", "check"])
    ho = {m_: vprobe.loc[(m_, "held out")].iloc[0] for m_ in ("registers", "base")}
    ctl = X["M4b_control_200_draws"]
    ctl = ctl[(ctl.control == "variance-matched random directions")].set_index(["model", "subset", "label_set"])
    c12, c5 = ctl.loc[("registers", "all", L12)], ctl.loc[("registers", "all", L5)]
    esc = X["M4c_erased_scores"].set_index(["model", "label_set", "vectors"])
    er12, er5 = esc.loc[("registers", "G1 12-class", "view erasure")], esc.loc[("registers", "parent 5-class", "view erasure")]
    rules_p = X["S3_image_rules_paired"]
    rsh = X["S3_image_rules_split_half"].set_index(["label", "candidate"])
    ari = X["M11_ari_like_for_like"]
    ari_d5 = float(ari[(ari.model == "registers") & ari.comparison.str.startswith("drawings, k-means 5")].ari.iloc[0])
    ari_p5 = float(ari[(ari.model == "registers") & ari.comparison.str.startswith("photos")].ari.iloc[0])
    m3 = X["M3_pool_size_and_mix"].set_index("row")
    m3_full = m3.loc["difference: photos full set minus drawings"]
    m3_size = m3.loc["difference: photos size only minus drawings"]
    m3_mix = m3.loc["difference: photos size + mix (matched) minus drawings"]
    m3_dmix = m3.loc["decomposition: class mix at fixed size (size only minus size + mix)"]
    m3_dsize = m3.loc["decomposition: pool size (full set minus size only)"]
    s_size = m3.loc["photos, size only: subsets of 663 at the photos' own class mix"]
    s_mix = m3.loc["photos, size + mix: subsets of 663 at the drawings' class counts"]
    gg = X["fairness_grey_grey"].set_index(["drawings", "photos"]).loc[("grey", "grey")]
    crop = cst("cropped minus original, photos (registers)", P5)
    crop_b = cst("cropped minus original, photos (base)", P5)
    ck = X["crop_reg_knn"]
    ck = ck[ck.matrix == "L24 CLS"].set_index("condition")
    mi = X["S6_matching_intervals"].set_index(["model", "direction", "page_status"])
    mr_ = mi.loc[("registers", "patent to photo", "all")]
    mb_ = mi.loc[("base", "patent to photo", "all")]
    lk = X["M11_linked_99_vs_94"]
    n_gate_l = int(lk.explanation.str.startswith("not in the patent analysis set").sum())
    n_line_l = int(lk.explanation.str.contains("line drawings").sum())
    auw = X["S10_audit_wilson"].set_index("sample")
    kt = X["S10_kappa"]
    k_rec = kt[kt.origin != "computed"].iloc[0]
    k_fz_all = kt[kt.labels.str.contains("figure_label_frozen")].iloc[0]
    k_fz_set = kt[kt.labels.str.startswith("frozen image labels (2")].iloc[0]
    n_gt_rows = int(X["gt_rows"].n.iloc[0])

    def wil(sample) -> str:
        r = auw.loc[sample]
        return f"{int(r['count'])} of {int(r.n)} [{r.wilson_lo:.2f}, {r.wilson_hi:.2f}]"

    # ═══ title, index ═══
    import datetime as dt
    D.add(f"# {TITLE}", "", f"<!-- Date: {dt.date.today().isoformat()} -->", "",
          "The analysis for research question RQ4. Private: contains evtol.news "
          "photographs, © their owners.", "", "@@TOC@@", "", "---", "")

    # ═══ 1 ═══
    D.chapter("1", "The images and the labels: which drawings and photographs enter the analysis, "
                   "and does the way they are chosen decide the result?")
    D.add("#### Method common to every chapter", "")
    D.p("**What is asked (RQ4).** Whether the figure alone carries the architecture: whether a frozen vision "
        "model, DINOv2 with registers (Oquab et al., 2023; Darcet et al., 2024), never trained on aircraft or on "
        "the codebook, places aircraft of the same architecture class near each other, judged against the human "
        "labels. Two image sources are read with the same model and the same protocol: the patent drawings of the "
        "labelled dataset, and the images of the World eVTOL Aircraft Directory (evtol.news, Vertical Flight "
        "Society), called photographs below.",
        f"**Unit of analysis: the aircraft.** One patent can disclose several aircraft and one directory page "
        f"shows one, so every source is reduced to one vector per aircraft: {n_(nP)} drawing aircraft and "
        f"{n_(nH)} photo aircraft. **The maker is held out of every comparison**: the applicant's company group "
        "for a patent (the first applicant for an individual inventor), the maker named on the page for a photo. "
        "A company's house style can therefore never count as architecture.",
        "**The embedding.** The model turns an image into a vector of 1 024 numbers. It is read at one layer "
        "(18, 22 or 24 of the model's 24) and through one token: the CLS token, the summary token the model "
        "learns, or the mean of the patch tokens. One layer with one token is called a **matrix**; six are "
        "compared, and chapter 2 chooses one.")
    D.add('<div class="methods-box" markdown="1">', "", "##### How to read the scores", "",
          "*Every score, interval and chance level in the document is read as set out here.*", "")
    D.p("**The primary score: kNN-5 balanced accuracy.** Each aircraft's class is predicted by the vote of its "
        "five nearest aircraft, never from its own maker (**maker held out**), nearness being the cosine distance "
        "d(u, v) = 1 − u·v / (‖u‖ ‖v‖) (nearest-neighbour rule, Cover and Hart, 1967); a tie goes to the class "
        "of the nearer neighbour; k = 5 and the cosine distance were fixed in advance. The score is the "
        "**balanced accuracy**, BA = (1/K) Σ_c recall_c, with recall_c = (aircraft of class c predicted as c) / "
        "(aircraft of class c), over the K classes (Brodersen et al., 2010). Every class weighs the same, so a "
        "large class cannot carry the score, and guessing scores 1/K: 0.083 on 12 classes, 0.20 on 5.",
        f"**The 95 % intervals.** A score's interval resamples the aircraft with replacement {n_(P.N_BOOT)} "
        "times on the same predictions and keeps the middle 95 % (bootstrap, Efron, 1979). A **paired contrast** "
        "(two setups on the same aircraft) resamples whole makers, with all their aircraft, for both setups: "
        "this maker interval is the primary one, slightly wider than the aircraft one because a maker's aircraft "
        "resemble each other. Beside it stands the share of the 400 held-out halves (below) in which the "
        "difference has the same sign. Between the two sources, whose aircraft differ, each set is resampled on "
        "its own. A share of a small count (a recall, an audit) gets the Wilson score interval (Wilson, 1927). "
        "**A difference whose interval takes in zero is read as no difference detected.**",
        f"**Many contrasts, read together.** The document makes {n_con} paired contrasts; after a Holm "
        f"correction over the {n_con} (Holm, 1979) {'none survives' if n_holm == 0 else f'{n_holm} survive'} at "
        "0.05, so they are read as convergent evidence: by their direction, their intervals and their per-half "
        "consistency, not one by one.",
        f"**Chance, as the same neighbours reach it.** The labels are shuffled {P.N_SHUFFLE} times and the vote "
        "rerun; the 95th percentile of the shuffled scores is the **shuffled level**. A score above it is one "
        "that chance almost never produces with these neighbours.",
        f"**The linear probe.** A logistic regression reads every dimension of the vector, trained on four fifths "
        "of the makers and tested on the fifth (five folds grouped by maker), scored by the same balanced "
        "accuracy; L2 penalty of the default strength (C = 1), lbfgs solver, balanced class weights, C not "
        "tuned. Its chance is the same probe on 50 "
        "shuffled label sets. The neighbours use the overall distance only, so a probe above the kNN score means "
        "the architecture is in the vector without deciding which aircraft are nearest.",
        "**The split-half selection rule** (the author's own definition). A choice made on the labels must not "
        "be scored on the aircraft that made it. The makers are split at random into two halves, 200 times "
        "(fixed seed 42, a rerun gives the same numbers), and every aircraft follows its maker. Each candidate "
        "is scored on half A with neighbours from A only, the best is scored on half B, and the halves are "
        "swapped: 400 choices per label set. Reported: how often each candidate is chosen (its **selection "
        "frequency**), its held-out score, and the **optimism gap** = in-sample score of the winner − its "
        "held-out score. The candidate with the highest selection frequency, averaged over the label sets of a "
        "source and then over the two sources, is chosen.",
        f"**Separation, clusters and what else is encoded.** Separation compares the mean cosine distance "
        f"between aircraft of different classes with that between aircraft of the same class, over pairs of "
        f"different makers: ratio = mean between / mean within, and d = (mean between − mean within) / pooled "
        f"standard deviation; p is the share of {P.N_PERM} label shuffles with a ratio as large. The same d for "
        "a label that is not the architecture (the view of a drawing, the maturity of a photographed aircraft), "
        "divided by the architecture's d, is the **d ratio**: which label moves the vectors more. Clusters: "
        "k-means with as many clusters as classes, compared with the labels by the adjusted Rand index, "
        "ARI = (RI − E[RI]) / (max RI − E[RI]), where RI is the share of aircraft pairs on which the two "
        "partitions agree; 0 is a chance partition, 1 the labels themselves (Hubert and Arabie, 1985).",
        "**The maps.** UMAP (McInnes, Healy and Melville, 2018) draws each 1 024-number vector in two "
        "dimensions so that similar vectors sit close. The axes have no unit, and a map is a picture, never a "
        "test.")
    D.add("**The seven tests of the protocol**, run with the same code and parameters on both sources:", "",
          "| test | what it asks | measure |", "|:---|:---|:---|",
          "| T1 integrity | is every vector valid? | no NaN, infinite, all-zero or duplicate rows |",
          "| T2 structure | do the vectors have structure at all? | variance on the first principal component "
          "against a random matrix of the same shape; Hopkins statistic (0.5 random, near 1 clumped) |",
          "| T3 separation | are same-class aircraft closer? | ratio, d and p, as above |",
          "| T4 prediction | how often is the class recovered? | kNN-5 balanced accuracy (primary), interval, "
          "shuffled level; probe |",
          "| T5 clusters | do the classes form clusters? | k-means against the labels, ARI |",
          "| T6 what else | does another label move the vectors more? | the d ratio |",
          "| T7 image choice | does the image rule matter? | T4 under every image rule |", "",
          "</div>", "", "")

    # 1.1
    D.method(
        "1.1", "Which drawings and photographs enter the analysis, and with which labels?",
        "Which images of each source reach the model, which label each aircraft carries, and how far that "
        "label can be trusted.",
        "Drawings: the figures of the labelled dataset of eVTOL patents, counted as figures until each aircraft "
        "is reduced to one (1.2). A drawing aircraft carries its **G1 type**, one of the codebook's 12 "
        "architecture types, and the type's parent among 5 classes: **VT** Vectored Thrust, **LC** Lift + "
        "Cruise, **WM** Wingless (Multicopter), **ER** Electric Rotorcraft, **HB** Hover Bikes / PFD (the "
        "directory's own name, personal flying devices). **One labeller, the author, assigned every G1 type.** "
        "Photographs: the images of the directory pages (crawl of 2026-09-17); a photo aircraft carries the "
        "directory's own class, one of the same 5.",
        f"Figure {D.fignum()} follows each source from the files on disk to the analysis set, each image counted "
        "under the first rule that removed it. A drawing enters when the labeller approved it, its aircraft "
        "passes the domain gate (aircraft flagged as similar to a UAV, or not electric, leave) and its patent "
        "is not a duplicate of another; part figures leave, and so do the "
        f"{n_excl} aircraft marked unclassifiable. A photo passes fixed rules against small files and files "
        "shared with another page, then a zero-shot SigLIP score p(whole aircraft) that keeps an image at "
        f"p ≥ 0.80, drops it below 0.35 and holds back the review band between; the page's first kept image, "
        f"its **hero image**, stands for the aircraft. Figure {D.fignum(2)} counts the classes, Table "
        f"{D.tabnum()} the label checks, Figures {D.fignum(3)} and {D.fignum(4)} show one image per class.",
        "The G1 label is compared with the author's whole-patent reading of the architecture by Cohen's "
        "κ = (p_o − p_e) / (1 − p_e), the observed agreement p_o corrected for the agreement p_e two readings "
        "would reach by chance (Cohen, 1960). The photo filter is checked on a random sample of hero, dropped and "
        "review-band images, each share with its Wilson interval.",
        "κ is not an independent agreement: the same person made both readings, the whole-patent one after the "
        "wizard labels were corrected. The audit judge was the AI assistant of the pipeline, working from sheets "
        "that showed the filter score: neither blind nor yet re-checked by the author. The photo label is the "
        f"directory's class, not the codebook's: it agrees with the G1 parent for {n_agree} of the {n_link} "
        f"aircraft present in both ({pc(n_agree / n_link)}).")
    note("f_funnels.png", [(0, "Source: patent drawings of the labelled dataset; labeller's approval, domain "
                               "gate and duplicate rulings"),
                           (1, "Source: evtol.news directory pages (crawl 2026-09-17); fixed rules and a "
                               "zero-shot SigLIP filter")])
    with _into_brief():
        f1 = AD.fig_funnels(C)
    D.figure(f1, "from the files on disk to the analysis sets: (i) patent drawings, (ii) directory photographs; "
                 "the blue bar is the analysis set.",
             f"**{n_(nW)} whole-aircraft drawings of {n_(nP)} aircraft from {n_(n_pat_patents)} patents, and "
             f"{n_(nH)} hero images, one per directory aircraft, enter the analysis.** (i) The labeller's approval "
             f"removes most drawings ({n_(n_notappr)} of {n_(n_file)} figures on file); the domain gate removes "
             f"{n_gate} more, the duplicate rule {n_dup} and the part figures {n_part}. (ii) The filter keeps "
             f"{n_(len(kept))} of {n_(len(im))} images, holding back the {n_rev} of the review band.")
    shares = cnt12.sort_values(ascending=False)
    n_tiny = int((cnt12 < 20).sum())
    fcc = rf_figure("fig_class_counts")
    D.figure(fcc, "aircraft per G1 type, coloured by the parent class (left), and per shared class in the two "
                  "sources (right).",
             f"**Two types hold {pc((shares.iloc[0] + shares.iloc[1]) / nP)} of the drawing aircraft, "
             f"{AD.G1_NAME[shares.index[0]]} ({shares.iloc[0]}) and {AD.G1_NAME[shares.index[1]]} "
             f"({shares.iloc[1]}), while {n_tiny} types have fewer than 20.** The two sources have different class "
             f"mixes: Vectored Thrust is {pc(cnt12[[c for c in ER.G1_ORDER if ER.PARENT[c] == 'VT']].sum() / nP)} of "
             f"the drawings and {pc(cnt5h['VT'] / nH)} of the photos, Wingless {pc(cnt12.get('MR', 0) / nP)} "
             f"against {pc(cnt5h['WM'] / nH)}.")
    dis = kby[kby.disagreements > 0].sort_values("disagreements", ascending=False)
    D.table(md_df(
        [("G1 label against the whole-patent reading (current labels)", n_(gt_n), f"{agree} ({pc(agree / gt_n)})",
          f"κ {kap:.3f}")] +
        [(f"frozen labels of 2026-09-15 against the same reading (analysis set)", n_(int(k_fz_set.n)),
          f"{int(k_fz_set.agree)} ({pc(k_fz_set.agree_share)})", f"κ {k_fz_set.kappa:.3f}"),
         (f"frozen labels, whole ground-truth file ({n_(n_gt_rows)} rows)", n_(int(k_fz_all.n)),
          f"{int(k_fz_all.agree)} ({pc(k_fz_all.agree_share)})", f"κ {k_fz_all.kappa:.3f}"),
         ("frozen labels, project record of 2026-09-17", n_(int(k_rec.n)), f"({pc(float(k_rec.agree_share))})",
          f"κ {float(k_rec.kappa):.3f}")] +
        [("hero images that show the whole aircraft", n_(int(auw.loc['main'].n)), wil("main"), "Wilson 95 %"),
         ("images the filter dropped that were a whole vehicle", n_(int(auw.loc['siglip_drop'].n)), wil("siglip_drop"),
          "Wilson 95 %"),
         ("review-band images that would have been usable", n_(int(auw.loc['review_band'].n)), wil("review_band"),
          "Wilson 95 %")],
        ["check", "n", "agree / count", "measure"]),
        "How far the labels and the photo filter can be trusted.",
        f"**The current G1 labels agree with the author's whole-patent reading for {agree} of {gt_n} aircraft "
        f"(κ {kap:.2f}), and the labels frozen before the last corrections for κ {float(k_rec.kappa):.3f}.** The "
        f"disagreements concentrate in "
        + ", ".join(f"{disp(c)} ({int(r.disagreements)} of {int(r.n_aircraft)}: the reading says "
                    f"{r.whole_patent_reading_says.replace('TR', 'TP')})" for c, r in dis.head(2).iterrows())
        + f". {int(auw.loc['main']['count'])} of {int(auw.loc['main'].n)} audited hero images show the whole "
        "aircraft.", right=["n"])
    note("f_examples_patents.png", "Source: patent drawings, labeller's main figure as the model receives it; "
                                   "the first aircraft of each type whose main figure is a Perspective view; label: "
                                   "human G1 type (parent class); code under each drawing: the aircraft")
    with _into_brief():
        f2_ = AD.fig_examples_patents(C)
    D.figure(f2_, "one main figure per G1 type, as the model receives it (518 px, padded square).",
             "**The model receives the drawing as filed: black line-work on a white square, with reference "
             "numerals and figure labels left in.** Nothing in the image names the type; the type is what the "
             "labeller read from the drawing and the patent.", width=100)
    PRIVB.mkdir(parents=True, exist_ok=True)
    f3 = PRIVB / "f02_examples.png"
    shutil.copyfile(AD.EE / "embedding_analysis_registers/figs/f02_examples.png", f3)
    D.figure(f3, f"one hero image per directory class (built aircraft, highest filter score). {CREDIT}.",
             "**A hero image shows the whole aircraft, a photograph or a rendering, often in a scene.** Hover "
             "Bikes / PFD is the directory's name for its fifth class, which holds the G1 types HB and PFV.",
             width=100)

    # 1.2
    sens_rows, inside_all, rng_ = [], True, {}
    for src, R in (("drawings", Rp), ("photos", Rh)):
        s = R["sens"]
        s = s[s.matrix == "L24 CLS"]
        base_ = s[s.variant == "as built"].iloc[0]
        vals = []
        for r in s.itertuples():
            sens_rows.append((src, r.variant, n_(r.n_aircraft), f"{r.knn_bal_acc:.3f}", ci(r.ci_lo, r.ci_hi)))
            if r.variant != "as built":
                inside_all &= bool(base_.ci_lo <= r.knn_bal_acc <= base_.ci_hi)
            vals.append(r.knn_bal_acc)
        rng_[src] = (min(vals), max(vals))
    D.method(
        "1.2", "How does each aircraft become one image and one vector, and does the choice of images decide "
               "the result?",
        "An aircraft has a median of two whole-aircraft drawings, so a rule must pick the image that becomes its "
        "vector. The question is which rules exist, how often they agree, and whether any selection step moves "
        "the primary score.",
        f"The {n_(nP)} drawing aircraft and their {n_(nW)} figures; for the flight state, the figures of the "
        f"{len(conv)} convertible types (TP; CVT, Combined Vectored Thrust; TW, Tilt Wing; DS, Deflected "
        "Slipstream; SRW, Stopped/Slowed Rotor Wing), whose shape changes between hover and cruise; the "
        f"{n_(nH)} photo aircraft for the sensitivity check.",
        f"Four rules were fixed before any score was read (Table {D.tabnum()}): **exp1 view-first** sorts on "
        "the view (Perspective > Plan > Side > Front/Rear), then on the flight state; **exp2 state-first** sorts "
        "on the flight state (Cruise > Both > Hover > Other), then on the view; **exp3 main figure** takes the "
        "figure the labeller marked as main, the primary rule; **exp4 view average** averages the vectors of "
        "the best figure in each of Perspective, Plan and Side. The flight state is recorded per figure as "
        "Cruise, Hover, or Both when the moving part is drawn in both positions. The chosen image is rotated "
        "by its labelled angle, padded to a white square and resized to 518 px, 37 × 37 patches of 14 px, for "
        "drawings and photos alike. Each selection step is then made stricter, or its measured errors removed, "
        f"and the primary score recomputed (Table {D.tabnum(2)}).",
        f"Cruise is the canonical state because more convertible aircraft have a Perspective figure in Cruise "
        f"({pc_} aircraft) than in Hover ({ph_}) among the {n_var}; Both is rare ({n_both} figures). A selection "
        "step decides the result only if a variant leaves the interval of the set as built.",
        "The variants are subsets of the set as built, so their intervals are not independent of it.")
    D.table(md_df([
        ("exp1 view-first", "best view, then best flight state", pc(same("exp1 view-first", "exp3 main figure"), 1)),
        ("exp2 state-first", "best flight state, then best view", pc(same("exp2 state-first", "exp3 main figure"), 1)),
        ("exp3 main figure", "the figure the labeller marked as main (primary rule)", "100.0 %"),
        ("exp4 view average", "mean of the best figure of each of Perspective, Plan and Side", "n/a"),
    ], ["rule", "which figure", "same figure as exp3"]),
        f"The four image rules. exp4 averages 2 figures for {int((nslots == 2).sum())} aircraft and 3 for "
        f"{int((nslots == 3).sum())}, and is one figure for the other {int((nslots == 1).sum())}.",
        f"**The flight state barely moves the choice: exp1 and exp2 differ on "
        f"{int(round((1 - same('exp1 view-first', 'exp2 state-first')) * nP))} aircraft only.** The labeller's "
        f"main figure differs from the view-first pick on "
        f"{int(round((1 - same('exp1 view-first', 'exp3 main figure')) * nP))}.", right=["same figure as exp3"])
    D.table(md_df(sens_rows, ["source", "variant", "aircraft", "kNN-5", "95 % interval"]),
            "Sensitivity of the primary score to the selection (registers, layer 24 CLS, maker held out).",
            (f"**Every variant stays inside the interval of the set as built: no selection step decides the "
             f"result.** The drawing variants span {rng_['drawings'][0]:.3f} to {rng_['drawings'][1]:.3f}, the "
             f"photo variants {rng_['photos'][0]:.3f} to {rng_['photos'][1]:.3f}." if inside_all else
             "**At least one variant leaves the interval of the set as built.**"))
    D.answer(
        f"{n_(nP)} drawing aircraft and {n_(nH)} photo aircraft enter, each as one image and one vector; the "
        "labels agree closely with the author's own whole-patent reading, and no selection step moves the "
        "primary score outside its interval.",
        [f"**Two sources, one unit.** {n_(nW)} whole-aircraft drawings from {n_(n_pat_patents)} patents, and "
         "one hero image per directory page, each reduced to one vector per aircraft.",
         f"**One labeller.** κ {kap:.2f} against the author's whole-patent reading, {float(k_rec.kappa):.3f} for "
         "the frozen labels; no second labeller has read the drawings.",
         f"**Two label schemes.** The directory class agrees with the G1 parent for {n_agree} of {n_link} "
         "aircraft present in both.",
         "**The selection does not decide the result.** Stricter filters, label-checked aircraft and "
         "Perspective-only aircraft all score inside the interval of the set as built."])

    # ═══ 2 ═══
    D.chapter("2", "The setup: which DINOv2 checkpoint, layer, token and image preparation read the images, "
                   "and how are they chosen?",
              "A frozen model leaves few choices, but each can move the score. They fall into three kinds: "
              "**fixed** choices stated in advance (resolution, pad, rotation, one image per aircraft, k); "
              "**label-selected** choices, made on the labels and therefore chosen by the split-half rule (layer, "
              "token, each source's preparation); and the **model**, a property of the checkpoint that a "
              "label-free diagnostic can decide (Table " + f"{D.tabnum()}).")
    D.add("#### How the model was reached", "")
    D.add(f"1. **Planned:** the base DINOv2-large, the model of the extraction pipeline ({base_sc[L12]:.3f} on "
          "the drawings' 12 types).",
          f"2. **Checked, then switched:** the registers check was run for the attention artifact of the base "
          f"model (2.1) and also returned higher scores ({sg(reg[L12]['diff'])} and {sg(reg[L5]['diff'])} on the "
          "drawings); the switch was decided after those scores had been seen, so it is post hoc on the scores.",
          f"3. **Confirmed:** the split-half rule, run over all 12 candidates of both models, picks a registers "
          f"candidate in {pc(reg_share.min(), 1)} to {pc(reg_share.max(), 1)} of the held-out choices.", "")
    D.table(md_df([
        ("model", "DINOv2-large; DINOv2-large with registers", "model property; label-free diagnostic, label-confirmed",
         "attention check (2.1); split-half rule over all 12 candidates", "with registers"),
        ("resolution", "518 px", "fixed (author ruling)", "the model's largest training size; that it keeps thin "
         "lines is a rationale, not a measurement", "518 px"),
        ("pad and rotation", "white square pad; rotation by the labelled angle", "fixed, not selected",
         "the pad matches the paper of a drawing", "as stated"),
        ("image per aircraft", "drawings: the labeller's main figure; photos: the hero image",
         "fixed before the setup round", "one representative image per aircraft on both sides, chosen by the "
         "source; the primary rule since 2026-09-22 (3.3, 4.3)", "main figure; hero image"),
        ("neighbours", "k = 5, cosine distance", "fixed in advance", "k = 3 and 10 as sensitivity (3.1)", "k = 5"),
        ("layer and token", "layer 18, 22, 24; CLS or patch mean", "label-selected", "split-half rule (2.2)",
         "layer 24 CLS"),
        ("readout for both sources", "one layer and token, or one per source", "author ruling",
         "one readout keeps the comparison fair; each source's own pick is reported (2.2)", "one"),
        ("drawing preparation", "plain; thick1; thick2; grey and contrast as controls (added post hoc)",
         "label-selected", "split-half rule (2.3)", "plain"),
        ("photo preparation", "colour; greyscale", "label-selected", "split-half rule (2.3): a tie, the default "
         "is kept", "colour"),
    ], ["choice", "candidates", "kind", "how decided", "carried forward"]),
        "The setup: every choice, how it was decided, and what chapters 3 to 5 use.",
        "**One model and one readout for both sources, the images left as they come; only the layer, token and "
        "preparation are chosen on the labels, and they are tested on makers that did not choose them.**", right=[])

    # 2.1
    D.method(
        "2.1", "Does the registers checkpoint remove the attention artifact, and does it read the images better?",
        "Whether the base DINOv2 collapses its late attention onto blank background, whether the registers "
        "checkpoint removes the collapse, and whether it scores higher on the same aircraft.",
        f"Attention, per image, the unit at which attention exists: the {n_att_p} drawings of the patent main "
        f"set (the {n_(nP)} aircraft plus the {n_excl} marked unclassifiable, which are never scored) and the "
        f"{n_(nH)} hero images, both models on the same images. Scores: the {n_(nP)} drawing and {n_(nH)} photo "
        "aircraft, same processed images and code for both models.",
        f"Table {D.tabnum()} gives the collapse measure, the median share of the CLS token's attention on the "
        "single largest patch, per layer, and at layer 24 its 90th percentile, the share renormalised over the "
        "patches, the share the prefix tokens (CLS and registers) take, and the share of images whose largest "
        f"patch lies in the white pad. Figure {D.fignum()} maps four images. The scores of the two models are the "
        f"first rows of the contrast figure of 2.4 (Figure {D.fignum(5)}).",
        "DINOv2 without registers parks high-norm artifact tokens in low-information background patches, and "
        "the registers checkpoint adds four register tokens that absorb them (Darcet et al., 2024). The map is "
        "a = softmax(q_CLS · Kᵀ / √d_h), averaged over the heads; share = max_p a_p.",
        "The switch was decided after the scores were seen (post hoc). Part of the advantage does not survive "
        "the removal of the view (3.2).")

    def srow(m_, s_, lab):
        r24 = s9a.loc[(m_, s_, 24)]
        return (lab, *[f"{sh.loc[(m_, s_, L)].max_patch_share_median:.3f}" for L in (18, 22, 24)],
                f"{r24.max_patch_share_p90:.3f}", f"{r24.renorm_max_share_median:.3f}",
                f"{r24.prefix_share_median:.3f}", pc(r24.argmax_in_pad_share, 1))
    D.table(md_df([srow("base", "patent", "base, drawings"), srow("registers", "patent", "registers, drawings"),
                   srow("base", "photo", "base, photos"), srow("registers", "photo", "registers, photos")],
                  ["model, source", "L18", "L22", "L24", "L24 p90", "L24 renormalised", "L24 prefix tokens",
                   "largest patch in the pad"]),
            f"The attention collapse: median share of the CLS attention on the single largest patch ({n_att_p} "
            f"drawings, {n_(nH)} photos, both models on the same images).",
            f"**The base model's late attention collapses onto one patch, {sh.loc[('base', 'patent', 24)].max_patch_share_median:.3f} "
            f"of it at layer 24 on drawings and {sh.loc[('base', 'photo', 24)].max_patch_share_median:.3f} on photos, "
            f"and that patch lies in the white pad for {pc(pad[('base', 'patent')], 1)} of the drawings and "
            f"{pc(pad[('base', 'photo')], 1)} of the photos; with registers the share stays at or below "
            f"{reg_max:.3f} and never lands in the pad.** The collapse is a property of the model, not of patent "
            "drawings: it hits both sources, on the pad added to both.")
    note("f_attention_base_vs_registers.png",
         "Source: DINOv2-L and DINOv2-L with registers, frozen, 518 px; CLS attention, mean over heads, each map "
         f"scaled to its own 99th percentile; 2 drawings and 2 photos, first aircraft of each class in both runs. "
         f"{CREDIT}")
    with _into_brief():
        f4 = AD.fig_attention_models(C)
    D.figure(f4, "where the CLS token looks: base (columns 2 to 4) against registers (columns 5 to 7), layers 18, "
                 "22 and 24, on the same images; bright = attention, each map scaled on its own.",
             f"**In base layers 22 and 24 one patch of the pad, the pale square in the top-left corner, holds about "
             f"{sh.loc[('base', 'patent', 24)].max_patch_share_median:.1f} of the attention, and dots of the white "
             "field light up; with registers neither appears.** Each map is scaled to its own 99th percentile, so "
             "the corner patch is clipped to the brightest colour and the rest of a base map looks brighter than "
             "it is.", width=92)
    D.bullets([
        f"**Scores.** Registers minus base, makers resampled: {mci(reg[L12])} on the drawings' 12 types, "
        f"{mci(reg[L5])} on their 5 classes, {mci(reg[P5])} on the photos ({mnd(reg[P5])}); registers score higher "
        f"in {pc(min(r.share_of_halves_same_sign for r in reg.values()))} to "
        f"{pc(max(r.share_of_halves_same_sign for r in reg.values()))} of the halves, by "
        f"{sg(min(r.halves_mean_diff for r in reg.values()))} to {sg(max(r.halves_mean_diff for r in reg.values()))} "
        "on average.",
        f"**A reusable label-free test.** If a late layer puts more than about 0.1 of the CLS attention on one "
        "patch, or its largest patch in the pad, use a registers checkpoint; the test needs no label, and the "
        "labels then confirm it here."])

    # 2.2
    lwin = {lab: reg_only[reg_only.label == lab].sort_values("selection_freq").candidate.iloc[-1].replace("registers ", "")
            for lab in reg_only.label.unique()}
    all_l24 = all(v == "L24 CLS" for v in lwin.values())
    op_d1, op_d1f = own_pick("layer and token", "patents", 1)
    op_p1, op_p1f = own_pick("layer and token", "photos", 1)
    op_p2, op_p2f = own_pick("layer and token", "photos", 2)
    D.method(
        "2.2", "Which layer and which token carry the architecture?",
        "Which of the six matrices (layers 18, 22 and 24, read through the CLS token or the patch mean) is "
        "carried forward.",
        f"The six matrices of the registers model; three label sets: drawings on the 12 types, drawings on the 5 "
        f"classes, photos on the 5 classes. A half holds {hn.loc[L12, 'min']} to {hn.loc[L12, 'max']} drawing "
        f"aircraft and {hn.loc[P5, 'min']} to {hn.loc[P5, 'max']} photo aircraft.",
        f"The split-half rule picks the best matrix on one half and scores it on the other, 400 times per label "
        f"set; Figure {D.fignum()} gives how often each matrix is chosen, Table {D.tabnum()} the decision. "
        f"Checks: the rule rerun with seeds {' and '.join(map(str, seed_list))}; each source's own pick; the "
        "best mean held-out score as the criterion instead of the selection frequency.",
        "The selection frequency says how stable a choice is across makers; the held-out score, what it reaches "
        "on makers that did not choose it. The last layer's CLS token is the one DINOv2 is trained to make a "
        "global summary of the image (Oquab et al., 2023).",
        f"Halves are unbalanced, and {int(miss12.halves_missing_any_class)} of the 400 drawing halves miss a rare "
        f"type ({miss12.halves_missing_each_class}). A held-out score sits below the full-sample one because its "
        "neighbour pool is about half the size; only held-out scores compare with each other.")
    note("f_selection.png", [(0, "Source: registers model; 400 held-out choices; drawings, main figure, 12 types"),
                             (1, "Source: registers model; 400 held-out choices; drawings, main figure, 5 classes"),
                             (2, "Source: registers model; 400 held-out choices; photos, hero image, 5 classes")])
    with _into_brief():
        f5 = AD.fig_selection(C)
    D.figure(f5, "how often each layer and token is chosen by the split-half rule, per label set (registers "
                 "model); blue: the matrix carried forward.",
             ("**Layer 24 CLS is chosen most often on every label set" if all_l24 else
              "**Layer 24 CLS is not the first choice on every label set") +
             f"; its runner-up, layer 22 CLS, is chosen most on the photos ({pc(l22f[P5])}, against "
             f"{pc(min(l22f[L12], l22f[L5]))} to {pc(max(l22f[L12], l22f[L5]))} on the drawings).**")
    D.table(md_df([(c.replace("registers ", ""), pc(r.selection_freq, 1), f"{r.fixed_heldout_mean:.3f}")
                   for c, r in dec.iterrows()], ["matrix", "mean selection frequency", "mean held-out score"]),
            "The decision over the six registers matrices, averaged over the three label sets.",
            f"**Layer 24 CLS is chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of the choices and "
            f"layer 22 CLS in {pc(dec.loc['registers L22 CLS'].selection_freq)}; the patch-token mean is almost "
            "never chosen.**")
    D.bullets([
        f"**Each source's own pick.** Drawings alone pick {op_d1} in {pc(op_d1f)} of the choices; photos alone "
        f"pick {op_p1} in {pc(op_p1f)} and {op_p2} in {pc(op_p2f)}, a tie: on the same halves L24 CLS minus "
        f"L22 CLS averages {sg(l24_l22_ph.mean_diff)}, positive in {pc(l24_l22_ph.share_of_halves_positive)}. "
        "One readout for both sources is the author's ruling; the photos lose nothing measurable by it.",
        f"**Stable.** Seeds {' and '.join(map(str, seed_list))} "
        + ("keep the same winner" if seed_same else "change the winner")
        + ("; the best-score criterion picks the same setup at every step" if best_same else
           "; the best-score criterion disagrees at one step")
        + f"; the protocol's in-sample layer rule picks layer 24 CLS on {pc(lay_p.loc['L24 CLS'].wins_on_subsets)} "
        "of 200 subsets of 80 % of the drawing aircraft."])

    # 2.3
    D.method(
        "2.3", "Should the drawings or the photographs be transformed before the model reads them?",
        "Whether changing how the strokes of a drawing reach the model, or removing the colour of a photograph, "
        "helps the model read the architecture, and which part of a change does the work.",
        f"The {n_(nP)} drawing aircraft (main figure) and the {n_(nH)} photo aircraft (hero image), registers "
        f"model, layer 24 CLS. {n_col} of the {n_(len(sm1))} whole-aircraft drawings carry colour (the "
        f"{n_(len(sm1))} are the {n_(nW)} figures of the {n_(nP)} aircraft and the {len(sm1) - nW} of the "
        f"{n_excl} unclassifiable aircraft).",
        "**thick1** and **thick2** widen every stroke with a minimum filter on the original drawing, by about 1 "
        "and 2 input pixels per side (radius in original pixels: median "
        f"{int(sm1.radius.median())} and {int(sm2.radius.median())}); they also make the drawing grey and darken "
        "faint strokes. Two controls separate these three changes: **grey** removes the colour only; "
        "**contrast** turns the darker half of each image's ink solid black at the model input, so the strokes "
        f"darken without widening. Figure {D.fignum()} shows the five inputs; Table {D.tabnum()} their scores "
        "and the split-half choice over the five; the paired differences are rows of the contrast figure of 2.4.",
        "Paired differences with makers resampled and the per-half sign, as in the methods box. The ink mass "
        "(mean darkness of the input) measures how much black each preparation adds.",
        "The grey and contrast controls were added after the review of the first results (post hoc). contrast is "
        "a hard threshold, so it also removes the anti-aliasing of the darker strokes: it tests strokes "
        "rendered differently at the same footprint, not darkness alone. A gentler change was not tested.")
    note("f_preparations.png", "Source: patent drawings, labeller's main figure, 3 aircraft of the setup montage "
                               "(seed 42); 518 px model inputs; red square: the 128 px detail shown under each "
                               "preparation")
    f6 = fig_preparations()
    D.figure(f6, "the plain input with its detail window (left), and the same detail under the five preparations.",
             "**grey is indistinguishable from plain; contrast keeps the footprint but turns half of each stroke "
             "solid and jagged; thick1 and thick2 merge thin propeller blades and numerals and fill hatching.**",
             width=100)
    D.table(md_df([(c, {"plain": "nothing", "grey": "colour removed", "contrast": "strokes darkened in place",
                        "thick1": "strokes widened about 1 px per side", "thick2": "strokes widened about 2 px per side"}[c],
                    f"{kc.loc[(L12, c)].bal_acc:.3f}", f"{kc.loc[(L5, c)].bal_acc:.3f}",
                    pc(v2d.loc[c].selection_freq, 1), f"{v2d.loc[c].fixed_heldout_mean:.3f}") for c in PREPS],
                  ["preparation", "what changes", "kNN-5, 12 types", "kNN-5, 5 classes", "chosen (held out)",
                   "mean held-out score"]),
            "The five drawing preparations (registers, layer 24 CLS, maker held out); the split-half rule chooses "
            "among the five, frequencies and held-out scores averaged over the two drawing label sets.",
            f"**Greyscale alone equals plain: the predictions are identical for {pc(min(same_pred.values()))} to "
            f"{pc(max(same_pred.values()))} of the aircraft. Darkening the strokes in place lowers the score about "
            "as much as thick2, so the loss comes from changing how the strokes are rendered, not from dilation or "
            "from colour.** Plain is chosen in "
            f"{pc(v2d.loc['plain'].selection_freq, 1)} of the choices and plain or grey, which predict alike, in "
            f"{pc(plain_grey, 1)}.")
    t2 = [cst("thick2 minus plain", lab) for lab in (L12, L5)]
    co = [cst("contrast only minus plain", lab) for lab in (L12, L5)]
    incl = [f"{n} on the {'12 types' if lab == L12 else '5 classes'}" for n, rr in (("thick2", t2), ("contrast", co))
            for lab, r in zip((L12, L5), rr) if r.ci_lo_maker <= 0 <= r.ci_hi_maker]
    D.bullets([
        f"**How firm.** With makers resampled, thick2 minus plain is {mci(t2[0])} and {mci(t2[1])}, contrast "
        f"minus plain {mci(co[0])} and {mci(co[1])}"
        + (f"; the intervals of {_join(incl)} include 0, so the per-half shares carry the evidence: thick2 "
           f"is lower than plain in {pc(hv.loc[(L5, 'thick2 - plain')].share_lower)} to "
           f"{pc(hv.loc[(L12, 'thick2 - plain')].share_lower)} of the halves, contrast in "
           f"{pc(hv.loc[(L12, 'contrast - plain')].share_lower)}." if incl else "."),
        f"**Recipe.** Keep the lines as drawn: no rendering change tested improves the score; a gentler change is "
        "untested.",
        f"**Photos.** Greyscale minus colour is {mci(g_ph)} ({mnd(g_ph)}), same sign in "
        f"{pc(g_ph.share_of_halves_same_sign)} of the halves; the rule picks colour in "
        f"{pc(s1b.loc['colour'].selection_freq)} of the choices and grey in {pc(s1b.loc['grey'].selection_freq)}: "
        "a tie, so colour is kept as the default, not as a better choice."])

    # 2.4
    D.method(
        "2.4", "How much do the label-selected choices flatter the scores, and how firm is every paired contrast?",
        "How much of each score is owed to having chosen the setup on the same labels, and how firm each paired "
        "contrast of the document is once whole makers are resampled.",
        "The 400 held-out choices of each label set, for three pools of candidates: the six registers matrices; "
        "all 12 matrices of both models; and the joint grid of the six matrices with each source's preparations "
        f"(18 candidates on the drawings, 12 on the photos). The {n_con} paired contrasts of chapters 2 to 5.",
        f"Figure {D.fignum()} sets, per label set and pool, the in-sample score of the winner of a half, its "
        "score on the other half, the held-out score of layer 24 CLS kept fixed, and the full-sample score. "
        f"Figure {D.fignum(2)} draws every paired contrast: its difference, its interval with aircraft resampled "
        "(thin line) and with makers resampled (pale bar), and the share of halves with the same sign; three "
        "rows compare the two sources, whose aircraft differ (chapter 5).",
        "gap = in-sample score of the winner − its held-out score: the amount by which choosing on the labels "
        "inflates a score. Maker bootstrap and Holm correction as in the methods box.",
        "A held-out score uses about half the neighbour pool, so it sits below the full-sample score for a reason "
        "that has nothing to do with optimism.")
    frc = rf_figure("fig_optimism")
    D.figure(frc, "the optimism of the label-selected choices, per label set and pool of candidates.",
             f"**The drawing headline is {mp.knn_bal_acc:.3f} on the full sample, about {head12:.2f} after the "
             f"selection optimism, and {base_sc[L12]:.3f} for the base model, the model planned before any score was "
             f"read.** The joint-grid optimism is {jg.loc[L12, 'joint_optimism_gap']:.3f}, "
             f"{jg.loc[L5, 'joint_optimism_gap']:.3f} and {jg.loc[P5, 'joint_optimism_gap']:.3f} on the three label "
             f"sets ({f2(lab_all.optimism_gap.min(), 3)} to {f2(lab_all.optimism_gap.max(), 3)} when both models "
             f"compete), and layer 24 CLS kept fixed scores {t12.loc[L12, 'fixed_L24_CLS_heldout']:.3f}, "
             f"{t12.loc[L5, 'fixed_L24_CLS_heldout']:.3f} and {t12.loc[P5, 'fixed_L24_CLS_heldout']:.3f} held out.")
    fo = rf_figure("fig_forest")
    flip_txt = _join([_short_contrast(r) for _, r in flips.iterrows()])
    D.figure(fo, "every paired contrast of the document, with the aircraft and the maker intervals and the share "
                 "of halves with the same sign.",
             f"**Resampling whole makers widens the intervals a little and moves {len(flips)} readings to no "
             f"difference detected ({flip_txt}); none of the {n_con} survives a Holm correction, so the contrasts "
             "are read together, by their direction and their consistency across halves.**", width=100)
    D.answer(
        "DINOv2 with registers, at 518 px, read at layer 24 through the CLS token, on the images as they come: "
        "the checkpoint because its base twin collapses onto the blank pad, the rest by a held-out rule whose "
        "optimism is a few hundredths.",
        [f"**Registers.** A label-free diagnostic (the largest patch in the pad for "
         f"{pc(pad[('base', 'patent')], 1)} of the base model's drawings, {pc(pad[('registers', 'patent')], 1)} "
         f"with registers), adopted after the scores were seen, confirmed held out ({pc(reg_share.min(), 1)} to "
         f"{pc(reg_share.max(), 1)} of the choices).",
         f"**Layer 24 CLS.** Chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of the held-out "
         "choices; on the photos alone layer 22 CLS ties with it.",
         "**Plain drawings, colour photos.** Every rendering change tested lowers the drawing score or leaves it "
         "unchanged; greyscale changes nothing on either source.",
         f"**Three numbers for the drawings.** {mp.knn_bal_acc:.3f} measured, about {head12:.2f} after the "
         f"optimism, {base_sc[L12]:.3f} for the model planned first."])

    # ═══ 3 ═══
    D.chapter("3", "The drawings: do patent drawings carry the architecture, and what else does the model read "
                   "in them?",
              f"Chapters 1 and 2 fixed the images and the setup. This chapter reads the {n_(nP)} drawing aircraft "
              "with them: how well they predict the type (3.1), what organises the embedding (3.2), whether the "
              "choice of drawing matters (3.3) and where the model looks (3.4).")
    m = Rp["ev"]["main"]
    D.method(
        "3.1", "How well do the drawings predict the architecture type?",
        "Whether the drawing vectors carry the architecture: whether aircraft of the same type sit closer than "
        "aircraft of different types, and how often a type is recovered from the nearest aircraft.",
        f"The {n_(nP)} drawing aircraft, one main figure each, labelled with the 12 types and, folded, with the "
        "5 parent classes; all six matrices, layer 24 CLS being the one carried forward.",
        f"Soundness (T1, T2) and separation (T3) first. Figure {D.fignum()} gives, per matrix, the kNN-5 "
        f"balanced accuracy, the shuffled level and the probe; Figure {D.fignum(2)} the recall per type with its "
        "Wilson interval, and the same recall when every type weighs the same in the pool: 20 aircraft per type "
        "(all of a smaller type), 200 draws, neighbours inside the draw, against the same draws with shuffled "
        f"labels; Figure {D.fignum(3)} which types are confused with which. k = 3 and k = 10 are rerun as a "
        "sensitivity check.",
        "As in the methods box; Wilson (1927) for a recall on n aircraft; guessing scores 1/12 on the types.",
        f"{', '.join(f'{disp(c)} {int(cnt12[c])}' for c in sorted(small, key=lambda c: cnt12[c]))} aircraft: "
        f"these recalls rest on a handful of aircraft, and their types are greyed. PTC is also the type whose "
        f"label most often disagrees with the whole-patent reading ({ptc_dis} of {int(cnt12['PTC'])}).")
    D.bullets([
        f"**Soundness.** No matrix has an invalid row. On its first principal component every matrix carries "
        f"{m.t2_pc1_vs_random.min():.0f} to {m.t2_pc1_vs_random.max():.0f} times the variance of a random matrix, "
        f"and the Hopkins statistic lies between {m.t2_hopkins.min():.2f} and {m.t2_hopkins.max():.2f}: the "
        "vectors clump, as a continuum rather than as separate islands.",
        f"**Separation.** Same-type aircraft sit closer than different-type ones in all 6 matrices "
        f"(p ≤ {max(0.001, m.t3_p.max()):.3f}), but by little: at layer 24 CLS the ratio is {mp.t3_ratio:.3f} "
        f"and d {d_air:.2f} {ci(vr.loc['architecture d (aircraft, main figure)'].ci_lo, vr.loc['architecture d (aircraft, main figure)'].ci_hi)}."])
    note("pat_f04_prediction.png", f"Source: {MODEL}; {n_(nP)} drawing aircraft, labeller's main figure; human "
                                   "G1 type, 12 classes; maker held out")
    with _into_brief():
        F = AD.DocFigs(FIGB, "pat")
        ER.fig_prediction(F, S_pat, Rp)
    p12 = pr[L12]
    D.figure(F.items["f04_prediction"]["path"], "the architecture read from the drawings, per layer and token "
                                                "(12 types); blue: the matrix carried forward.",
             f"**At layer 24 CLS the drawings recover the 12 types at {f2(mp.knn_bal_acc)} "
             f"{ci(mp.knn_ci_lo, mp.knn_ci_hi)} against a shuffled level of {f2(mp.knn_chance_p95)}, and the 5 "
             f"classes at {f2(m5.knn_bal_acc)} {ci(m5.knn_ci_lo, m5.knn_ci_hi)} against {f2(m5.knn_chance_p95)}: "
             f"the architecture is present, and weak.** The probe reads {f2(p12.probe_bal_acc)} "
             f"{ci(p12.ci_lo_maker, p12.ci_hi_maker)} (makers resampled; shuffled {f2(p12.shuffle_p95)}), "
             f"{sg(p12.probe_minus_knn, 2)} {ci(p12.diff_ci_lo_maker, p12.diff_ci_hi_maker, 2, True)} above the "
             "neighbours: the architecture is in the vector, but it does not decide which aircraft are nearest. "
             f"k = 3 and k = 10 give {ks.loc[(L12, '3')].bal_acc:.3f} and {ks.loc[(L12, '10')].bal_acc:.3f} "
             f"({nd(ks.loc[(L12, '3 minus 5')].ci_lo, ks.loc[(L12, '3 minus 5')].ci_hi)} against k = 5).", width=80)
    fe = rf_figure("fig_recall12")
    zero_txt = ", ".join(f"{disp(c)} 0 of {int(rec12.loc[c, 'n_aircraft'])}" for c in zero)
    D.figure(fe, "recall per type: full sample with its Wilson interval (bars, under 20 aircraft greyed), and with "
                 "every type weighing the same in the pool (diamonds).",
             f"**Three types are never recovered in the full sample ({zero_txt}), but with class size equalised "
             + (", ".join(f"{disp(c)} rises to {bal12.loc[c, 'knn_recall_mean']:.2f} "
                          f"{ci(bal12.loc[c, 'knn_recall_lo'], bal12.loc[c, 'knn_recall_hi'])}, above its shuffled "
                          f"level of {bal12.loc[c, 'shuffled_recall_p95']:.2f}" for c in above) or "none rises above chance")
             + ": its zero was an effect of class size.** "
             + (f"{' and '.join(disp(c) for c in notabove)} stay at or below chance, on "
                f"{' and '.join(str(int(cnt12[c])) for c in notabove)} aircraft. " if notabove else "")
             + f"Among the types with 20 aircraft or more, TP ({rec12.loc['TR', 'knn_recall']:.2f}) and Lift + "
             f"Cruise ({rec12.loc['SLC', 'knn_recall']:.2f}) are read best.", width=88)
    note("pat_confusion.png", f"Source: {MODEL}; layer 24 CLS; {n_(nP)} drawing aircraft, main figure; "
                              "12 types; maker held out")
    with _into_brief():
        fcf = AD.fig_confusion(S_pat, Rp, FIGB / "pat_confusion.png", (5.4, 4.3))
    D.figure(fcf, "which types are confused with which: rows are the true type (aircraft in brackets), columns "
                  "the type kNN-5 predicts; the framed diagonal is correct.",
             f"**{pc(w_share, 1)} of the {w_n} wrong predictions ({w_k}) name Lift + Cruise or TP, which hold "
             f"{pc(w_pool, 1)} of the pool: the two largest types draw the errors, somewhat more than their share.**",
             width=60)

    # 3.2
    D.method(
        "3.2", "What organises the drawing embedding: the architecture, or the way the aircraft is drawn?",
        "Whether the types form clusters of their own, whether the view of the drawing moves the vectors more "
        "than the architecture, and whether the architecture survives when the view is removed.",
        f"The {n_(nP)} aircraft for clusters and scores; all {n_(nW)} whole-aircraft figures for the view, the "
        "unit at which a view exists; the view group of each figure (Perspective, Plan, Side, Front/Rear).",
        f"k-means with 12 clusters against the types (T5); the figures mapped and coloured by view "
        f"(Figure {D.fignum()}); T3's d for the view and for the architecture on the figures (T6, Figure "
        f"{D.fignum(2)}), the d ratio with a maker bootstrap; then a linear erasure: the three directions that "
        "separate the four view means are fitted on the figures and projected out. It is checked in sample, "
        "and fitted on half of the makers and tested on the other half (200 splits). A **variance-matched "
        "control** removes three random directions of the same variance, over 200 draws "
        f"(Figure {D.fignum(3)}).",
        "The erasure is x ↦ (I − Q Qᵀ) x, then renormalised, with Q an orthonormal basis of the differences "
        "between the view means (the author's own implementation). A loss inside the control's range is not "
        "specific to the view.",
        "The erasure is linear only; a non-linear view signal survives it.")
    D.bullets([f"**No clusters of their own.** k-means with 12 clusters matches the types with an ARI of "
               f"{mp.t5_ari:.3f} at layer 24 CLS, and no matrix exceeds {m.t5_ari.max():.3f}."])
    fg = rf_figure("fig_umap_view")
    D.figure(fg, f"the {n_(nW)} figure embeddings in two dimensions, coloured by the view of the drawing.",
             f"**The figures group by view: a view is recovered from the five nearest figures of other makers at "
             f"{vk.estimate:.2f} {ci(vk.ci_lo, vk.ci_hi)} against a shuffled level of {vk.chance:.2f}.**", width=62)
    note("pat_f09_confound.png", f"Source: {MODEL}; all {n_(nW)} whole-aircraft figures of the {n_(nP)} "
                                 "aircraft; view group from the labelling; same-maker pairs left out")
    with _into_brief():
        ER.fig_confound(F, S_pat, Rp)
    vrr = vr.loc["view d / architecture d (figures)"]
    D.figure(F.items["f09_confound"]["path"], "how far the view and the architecture move the vectors, per "
                                              "matrix: d, in standard deviations.",
             f"**The view moves the vectors more than the architecture in "
             f"{'all 6' if (conf_p.conf_over_arch > 1).all() else int((conf_p.conf_over_arch > 1).sum())} "
             f"matrices: at layer 24 CLS the d ratio is {vrr.estimate:.2f} {ci(vrr.ci_lo, vrr.ci_hi)}.** The "
             f"architecture d is {d_fig:.2f} here, per figure, against {d_air:.2f} per aircraft: the figures "
             f"add the non-main ones (d {d_nonmain:.2f}) and pairs in different views; within one view it is "
             f"{d_views['Perspective']:.2f} (Perspective), {d_views['Plan']:.2f} (Plan), {d_views['Side']:.2f} "
             f"(Side) and {d_views['Front/Rear']:.2f} (Front/Rear).", width=80)
    fff = rf_figure("fig_erasure")
    D.figure(fff, "the architecture score before and after the view erasure, both models, with the "
                  "variance-matched control over 200 draws.",
             f"**Removing the view costs {abs(c12.erasure_diff):.3f} on the 12 types and {abs(c5.erasure_diff):.3f} "
             f"on the 5 classes, and a random removal of the same variance costs {sg(c12.control_mean)} "
             f"{ci(c12.control_lo, c12.control_hi, 3, True)} and {sg(c5.control_mean)} "
             f"{ci(c5.control_lo, c5.control_hi, 3, True)}: the loss is not specific to the view.** The erased "
             f"scores, {er12.bal_acc:.3f} {ci(er12.ci_lo, er12.ci_hi)} and {er5.bal_acc:.3f} "
             f"{ci(er5.ci_lo, er5.ci_hi)}, stay above chance ({er12.chance_p95:.2f} and {er5.chance_p95:.2f}).",
             width=86)
    D.bullets([
        f"**In sample and held out.** Fitted and tested on the same figures, the erasure takes the linear view "
        f"probe on the main figures from {vp_o:.2f} to {vp_e:.2f} (chance 0.25). Fitted on half of the makers, it "
        f"takes the probe on the other half from {ho['registers'].before:.2f} to {ho['registers'].after:.2f}: "
        f"{pc(ho['registers'].share_of_above_chance_removed)} of the above-chance view signal removed (base "
        f"model {pc(ho['base'].share_of_above_chance_removed)}). The view kNN still reaches {vk_e.value:.2f}.",
        "**Why the view dominates yet its removal costs no more than chance.** Removing any three high-variance "
        "directions costs about as much as removing the view's three, so the view directions carry no more "
        "architecture than random ones: the reading consistent with this is an architecture spread through the "
        "dominant directions that the view also uses.",
        f"**Base model.** The same erasure leaves it unchanged, and after it the two models score alike "
        f"(12 types: registers {r12:.3f}, base {e12:.3f}; 5 classes: {r5:.3f} against {e5:.3f}): part of the "
        "registers advantage sits in the directions the view also uses."])

    # 3.3
    rp12 = rules_p[rules_p.label_set == L12].set_index("contrast")
    D.method(
        "3.3", "Does the choice of drawing change the result?",
        "Whether the image rule of 1.2 changes the architecture score, and whether averaging several views adds "
        "architecture.",
        f"The {n_(nP)} aircraft under each of the four rules; for the averaging, the {e4['n']} aircraft whose "
        "exp4 vector averages two or three views.",
        f"Table {D.tabnum()} gives the score under each rule (T7), each rule minus the main figure with makers "
        "resampled and its per-half sign, and how often the split-half rule would choose each rule.",
        "Intervals and split-half rule as in the methods box.",
        f"The main figure was fixed as the rule before the setup round; the exp4 subset comparison was run after "
        "the rule scores had been seen (post hoc).")
    rows = []
    for r in S_pat.rules:
        if r == "exp3 main figure":
            dcell, scell = "reference", "n/a"
        else:
            q = rp12.loc[f"{r} minus exp3 main figure"]
            dcell = f"{sg(q['diff'])} {ci(q.ci_lo_maker, q.ci_hi_maker, 3, True)}"
            scell = pc(q.share_of_halves_same_sign)
        rows.append((r, f"{br.loc[r].knn_bal_acc:.3f} {ci(br.loc[r].ci_lo, br.loc[r].ci_hi)}", dcell, scell,
                     pc(rsh.loc[(L12, r)].selection_freq), pc(rsh.loc[(L5, r)].selection_freq)))
    D.table(md_df(rows, ["image rule", "kNN-5, 12 types", "minus exp3 [makers]", "same sign in halves",
                         "chosen, 12 types", "chosen, 5 classes"]),
            "The architecture score under each image rule (registers, layer 24 CLS, maker held out).",
            f"**The main figure scores highest and the other rules are lower with intervals that exclude zero, "
            f"but the split-half rule would pick it in only {pc(rsh.loc[(L12, 'exp3 main figure')].selection_freq)} "
            f"and {pc(rsh.loc[(L5, 'exp3 main figure')].selection_freq)} of the choices: its lead is not stable "
            "across makers.** The rules span "
            f"{f2(br.knn_bal_acc.min())} to {f2(br.knn_bal_acc.max())}. On the {e4['n']} aircraft with two or "
            f"three views, exp4 minus view-first is {sg(e4['diff'])} {ci(e4['lo'], e4['hi'], 3, True)}, "
            f"{nd(e4['lo'], e4['hi'])}: averaging views adds no measurable architecture.")

    # 3.4
    l18 = dec.loc["registers L18 CLS"]
    D.method(
        "3.4", "Where does the model look on a drawing?",
        "Which parts of a drawing the CLS token of the chosen model reads: a qualitative check on what the "
        "scores measure.",
        f"Two drawing aircraft per parent class drawn at random (seed 42) from the main figures of the "
        f"{n_(nP)} aircraft.",
        f"Figure {D.fignum()} overlays the CLS attention at layers 18 and 24 on each drawing, read by eye. Layer "
        f"24 is the readout scored; layer 18, the layer the plan first named, scores "
        f"{AD.main_row(Rp, matrix='L18 CLS').knn_bal_acc:.3f} on the 12 types against "
        f"{mp.knn_bal_acc:.3f} and is chosen in {pc(l18.selection_freq)} of the held-out choices against "
        f"{pc(dec.loc['registers L24 CLS'].selection_freq)}, so its maps show what the model sees, not what it "
        "scores.",
        "Each map is scaled to its own 99th percentile, so brightness compares places within a map, never "
        "across maps.",
        "Ten aircraft read by eye are an illustration, not a measurement.")
    note("f_attention_drawings.png", "Source: DINOv2-L with registers, CLS attention at layers 18 and 24, mean "
                                     "over heads, each map scaled to its own 99th percentile; patent drawings, main "
                                     "figure, 2 aircraft per class drawn at random (seed 42); label under each pair: "
                                     "the G1 type")
    fa_d = fig_attention_source(C, "drawing")
    D.figure(fa_d, "CLS attention of the registers model on drawings, layers 18 and 24, two aircraft per parent "
                   "class; bright = attention within each map.",
             "**Layer 18 traces the line-work almost everywhere; layer 24 gathers in spots on the aircraft, often "
             "on propulsors, hubs and the fuselage, parts every type has, so no map singles out the part that "
             f"decides the type.** The register tokens take {reg_lo:.2f} to {reg_hi:.2f} of the attention, and the "
             f"median share on the single largest patch stays at or below {reg_max:.3f}.", width=66)
    D.answer(
        f"Patent drawings carry the architecture, weakly: a frozen model recovers the 12 types at "
        f"{f2(mp.knn_bal_acc)} against a shuffled level of {f2(mp.knn_chance_p95)}, but the embedding is "
        "organised first by how an aircraft is drawn, not by what it is.",
        [f"**Present and weak.** 12 types at {f2(mp.knn_bal_acc)} {ci(mp.knn_ci_lo, mp.knn_ci_hi)}, 5 classes "
         f"at {f2(m5.knn_bal_acc)}; the probe reads {f2(p12.probe_bal_acc)}.",
         f"**The large types are read best; a zero can be class size.** {zero_txt} in the full sample, but "
         + (", ".join(f"{disp(c)} is recovered above chance once class size is equalised" for c in above) or
            "none recovers with class size equalised") + ".",
         f"**The view comes first.** The d ratio is {vrr.estimate:.2f} {ci(vrr.ci_lo, vrr.ci_hi)}, the view is "
         f"recovered at {vk.estimate:.2f}, and the types form no clusters (ARI {mp.t5_ari:.3f}).",
         "**The architecture survives the view's removal.** The erasure costs no more than a random removal of "
         "the same variance, and every erased score stays above chance.",
         "**No single decisive part.** The image rules span "
         f"{f2(br.knn_bal_acc.min())} to {f2(br.knn_bal_acc.max())}, and the late CLS token spreads over parts "
         "every type has."])

    # ═══ 4 ═══
    D.chapter("4", "The photographs: do photographs of the aircraft carry the architecture, and does the choice "
                   "of image matter?",
              f"The same setup and protocol, on the {n_(nH)} directory aircraft and their 5 classes: prediction "
              "(4.1), what organises the embedding (4.2), the choice of image (4.3) and where the model looks "
              "(4.4). Whether the scene around the aircraft matters is a fairness question, answered with the "
              "other controls in chapter 5.")
    mm = Rh["ev"]["main"]
    s1r = C["s1a_candidates"]
    lab_l22 = float(s1r[(s1r.pool == "registers only") & (s1r.label == P5) &
                        (s1r.candidate == "registers L22 CLS")].selection_freq.iloc[0])
    r5h = X["M9_recall_5class_wilson"]
    r5h = r5h[r5h.label_set == P5].set_index("class")
    D.method(
        "4.1", "How well do photographs predict the directory class?",
        "Whether the photo vectors carry the architecture, read the same way as the drawings in 3.1.",
        f"The {n_(nH)} photo aircraft, hero image, labelled with the directory class; all six matrices.",
        f"Soundness and separation first; Figure {D.fignum()} gives, per matrix, the kNN-5 balanced accuracy, "
        f"the shuffled level and the probe; Figure {D.fignum(2)} which classes are confused with which.",
        "As in the methods box; guessing scores 1/5.",
        f"The classes range from {int(cnt5h.min())} ({cnt5h.idxmin()}) to {int(cnt5h.max())} ({cnt5h.idxmax()}) "
        "aircraft.")
    D.bullets([
        f"**Soundness and separation.** No invalid row, and the first principal component carries "
        f"{mm.t2_pc1_vs_random.min():.0f} to {mm.t2_pc1_vs_random.max():.0f} times the random variance. Same-class "
        f"photos sit closer in all 6 matrices, with a larger effect than on the drawings: ratio "
        f"{mh.t3_ratio:.3f} and d {mh.t3_d:.2f} at layer 24 CLS (drawings: d {d_air:.2f})."])
    note("pho_f04_prediction.png", f"Source: {MODEL}; {n_(nH)} evtol.news aircraft, hero image; directory class, "
                                   "5 classes; maker held out")
    with _into_brief():
        Fh = AD.DocFigs(FIGB, "pho")
        ER.fig_prediction(Fh, S_evn, Rh)
    pp = pr[P5]
    D.figure(Fh.items["f04_prediction"]["path"], "the architecture read from the photographs, per layer and token "
                                                 "(5 classes); blue: the matrix carried forward.",
             f"**At layer 24 CLS the photos recover the 5 classes at {f2(mh.knn_bal_acc)} "
             f"{ci(mh.knn_ci_lo, mh.knn_ci_hi)} against a shuffled level of {f2(mh.knn_chance_p95)}, and the probe "
             f"at {f2(pp.probe_bal_acc)} {ci(pp.ci_lo_maker, pp.ci_hi_maker)} (shuffled {f2(pp.shuffle_p95)}).** "
             f"Layer 22 CLS scores within that interval ({f2(AD.main_row(Rh, matrix='L22 CLS').knn_bal_acc)}), "
             f"which is why the split-half rule chose it in {pc(lab_l22)} of the photo choices.", width=80)
    note("pho_confusion.png", f"Source: {MODEL}; layer 24 CLS; {n_(nH)} evtol.news aircraft, hero image; "
                              "maker held out")
    with _into_brief():
        fpc = AD.fig_confusion(S_evn, Rh, FIGB / "pho_confusion.png", (3.9, 3.0))
    D.figure(fpc, "which directory classes are confused with which: rows are the true class (aircraft in "
                  "brackets), columns the predicted one.",
             f"**{ER.CLASS5_NAME[rbest]} is read best, {r5h.loc[rbest, 'knn_recall']:.2f} "
             f"{ci(r5h.loc[rbest, 'wilson_lo'], r5h.loc[rbest, 'wilson_hi'])}, and {ER.CLASS5_NAME[rworst]} worst, "
             f"{r5h.loc[rworst, 'knn_recall']:.2f} {ci(r5h.loc[rworst, 'wilson_lo'], r5h.loc[rworst, 'wilson_hi'])}.**",
             width=50)

    # 4.2
    D.method(
        "4.2", "What organises the photo embedding: the architecture, or the maturity of the aircraft?",
        "Whether the classes form clusters, and whether the maturity of the aircraft, built or concept, moves "
        "the vectors more than its class.",
        f"The {n_(nH)} aircraft for clusters; the {n_status} aircraft whose page states built or concept for the "
        "maturity label.",
        f"k-means with 5 clusters against the classes (T5), with the map of Figure {D.fignum()}; T3's d for the "
        "maturity and for the class, per matrix (T6).",
        "ARI and d as in the methods box.",
        "The maturity is the page's own statement.")
    note("pho_f07_umap.png", f"Source: {MODEL}; layer 24 CLS; {n_(nH)} evtol.news aircraft, hero image; UMAP "
                             "(15 neighbours, min. distance 0.1, cosine, seed 42); colour: directory class")
    with _into_brief():
        ER.fig_umap(Fh, S_evn, Rh)
    D.figure(Fh.items["f07_umap"]["path"], "the photo embeddings in two dimensions, one point per aircraft, "
                                           "coloured by the directory class.",
             f"**The classes form weak clusters: k-means matches them with an ARI of {ari_p5:.3f} at layer 24 CLS, "
             f"{ari_p5 / ari_d5:.1f} times the drawings' {ari_d5:.3f} on the same 5 classes.**", width=60)
    D.p(f"At layer 24 CLS the maturity moves the vectors less than the class (d {ch.conf_d:.2f} against "
        f"{ch.arch_d:.2f}, a ratio of {ch.conf_over_arch:.2f}), but in {int((conf_h.conf_over_arch > 1).sum())} "
        "of the 6 matrices the maturity dominates: the order depends on the layer, and the matrix carried "
        "forward is one where the class leads.")

    # 4.3
    rph = rules_p[rules_p.label_set == P5].set_index("contrast")
    D.method(
        "4.3", "Does the choice of photograph change the result?",
        "Whether another image of the page would change the photo score.",
        f"The {n_(nH)} photo aircraft under three rules: the hero image (the fixed rule), one random other "
        "kept image of the page, and the mean of the vectors of all kept images.",
        f"Table {D.tabnum()} gives the score under each rule, each rule minus the hero image with makers "
        "resampled and its per-half sign, and how often the split-half rule would choose each.",
        "As in the methods box.")
    rows = []
    for r in ("hero image", "random other image", "mean of all images"):
        if r == "hero image":
            dcell, scell = "reference", "n/a"
        else:
            q = rph.loc[f"{r} minus hero image"]
            dcell, scell = f"{sg(q['diff'])} {ci(q.ci_lo_maker, q.ci_hi_maker, 3, True)}", pc(q.share_of_halves_same_sign)
        rows.append((r, f"{brh.loc[r].knn_bal_acc:.3f} {ci(brh.loc[r].ci_lo, brh.loc[r].ci_hi)}", dcell, scell,
                     pc(rsh.loc[(P5, r)].selection_freq)))
    mq = rph.loc["mean of all images minus hero image"]
    D.table(md_df(rows, ["image rule", "kNN-5, 5 classes", "minus hero [makers]", "same sign in halves",
                         "chosen (held out)"]),
            "The photo score under each image rule (registers, layer 24 CLS, maker held out).",
            f"**The mean of all images would be chosen in {pc(rsh.loc[(P5, 'mean of all images')].selection_freq)} "
            f"of the held-out choices against {pc(rsh.loc[(P5, 'hero image')].selection_freq)} for the hero image, "
            f"but its gain, {mci(mq)}, is {mnd(mq)}; a random other image scores lower.** The hero image is kept: "
            "one representative image per aircraft on both sides.")

    # 4.4
    D.method(
        "4.4", "Where does the model look on a photograph?",
        "The same qualitative check as 3.4, on photographs, where the aircraft sits in a scene.",
        f"Two photo aircraft per class drawn at random (seed 42) from the {n_(nH)} hero images.",
        f"Figure {D.fignum()} overlays the CLS attention at layers 18 and 24 on each photograph, read by eye.",
        "As in 3.4: each map scaled on its own.",
        "Ten aircraft read by eye are an illustration, not a measurement.")
    note("f_attention_photos.png", "Source: DINOv2-L with registers, CLS attention at layers 18 and 24, mean over "
                                   "heads, each map scaled to its own 99th percentile; evtol.news hero images, 2 "
                                   f"aircraft per class drawn at random (seed 42). {CREDIT}")
    fa_p = fig_attention_source(C, "photo")
    D.figure(fa_p, f"CLS attention of the registers model on photographs, layers 18 and 24, two aircraft per "
                   f"class; bright = attention within each map. {CREDIT}.",
             "**Layer 18 covers the airframe; layer 24 gathers in spots on the aircraft, with weak spots in the "
             "sky. In one wingless photo the people in front of the aircraft draw as much attention as the "
             "aircraft itself: a kept hero image can still be a scene.**", width=66)
    D.answer(
        f"Photographs carry the architecture moderately: the 5 classes at {f2(mh.knn_bal_acc)} against a "
        f"shuffled level of {f2(mh.knn_chance_p95)}, a result the choice of image does not change.",
        [f"**Moderate.** kNN-5 {f2(mh.knn_bal_acc)} {ci(mh.knn_ci_lo, mh.knn_ci_hi)}, probe {f2(pp.probe_bal_acc)}; "
         f"{ER.CLASS5_NAME[rbest]} is read best, {ER.CLASS5_NAME[rworst]} worst.",
         f"**Weak clusters, stronger than on drawings.** ARI {ari_p5:.3f} against {ari_d5:.3f} on the same 5 "
         f"classes; separation d {mh.t3_d:.2f} against {d_air:.2f}.",
         "**The class leads the maturity only in some matrices**, the one carried forward among them; in "
         f"{int((conf_h.conf_over_arch > 1).sum())} of 6 whether the aircraft was built moves the vectors more.",
         f"**The image rule does not decide it.** The mean of all images scores {mci(mq)} against the hero image, "
         f"{mnd(mq)}."])

    # ═══ 5 ═══
    D.chapter("5", "Drawings against photographs: is the comparison fair, which source carries more of the "
                   "architecture, and can one aircraft be found in both?",
              "The two sources are read with the same model, readout and protocol on the 5 classes they share, "
              "each drawing type folded into its parent. Before the scores are compared, the comparison itself "
              "must be fair. Three things differ besides the medium:",
              f"- **The labels.** The photo class is the directory's, the drawing class the codebook parent: they "
              f"agree for {n_agree} of the {n_link} aircraft present in both ({pc(n_agree / n_link)}), the floor "
              "below which no source comparison can be read.\n"
              f"- **The pool.** {n_(nH)} photo aircraft against {n_(nP)} drawings, with different class mixes "
              "(chapter 1).\n"
              "- **The image.** Colour and scenery on the photos, line-work on white for the drawings; the photos "
              "also carry the aircraft's maturity, which moves the photo vectors more than the class in "
              f"{int((conf_h.conf_over_arch > 1).sum())} of 6 matrices (4.2), a signal a patent drawing does not "
              "have.")
    D.method(
        "5.1", "Once pool, class mix, colour and background are controlled, do photographs carry more architecture "
               "than drawings?",
        "Whether the medium, drawing or photograph, carries more of the architecture on the shared 5 classes, "
        "once each difference other than the medium has been controlled.",
        f"The {n_(nP)} drawing aircraft with their type folded to its parent, and the {n_(nH)} photo aircraft; "
        "the two sets hold different aircraft.",
        f"Table {D.tabnum()} compares the sources on the full sets, then controls one difference at a time: "
        f"**size**, 200 random photo subsets of {n_(nP)} at the photos' own class mix; **size and mix**, 200 "
        f"subsets with exactly the drawings' class counts; **colour**, both sources in greyscale; "
        "**background**, each hero image cropped to the aircraft by a zero-shot OWLv2 detector and re-embedded, "
        f"against the uncropped image. Figure {D.fignum()} sets each class's recall against its share of the "
        "neighbour pool.",
        "The difference of two independent sets resamples each on its own (1 000 draws); a subset score "
        "combines the 200 draws with a bootstrap of each draw's aircraft. A neighbour vote depends on its pool: "
        "a class that is thin in the pool is hard to recover whatever the medium.",
        "The size and mix controls were designed after the full-set comparison had been seen, the greyscale and "
        "registers crop checks after the review (post hoc).")
    fr = [("full sets (photos in colour, drawings plain)", f"{m5.knn_bal_acc:.3f}", f"{mh.knn_bal_acc:.3f}",
           f"{sg(m3_full.estimate)} {ci(m3_full.ci_lo, m3_full.ci_hi, 3, True)}", nd(m3_full.ci_lo, m3_full.ci_hi)),
          (f"same size ({n_(nP)} photos), photos' own class mix", f"{m5.knn_bal_acc:.3f}", f"{s_size.estimate:.3f}",
           f"{sg(m3_size.estimate)} {ci(m3_size.ci_lo, m3_size.ci_hi, 3, True)}", nd(m3_size.ci_lo, m3_size.ci_hi)),
          (f"same size and the drawings' class mix", f"{m5.knn_bal_acc:.3f}", f"{s_mix.estimate:.3f}",
           f"{sg(m3_mix.estimate)} {ci(m3_mix.ci_lo, m3_mix.ci_hi, 3, True)}", nd(m3_mix.ci_lo, m3_mix.ci_hi)),
          ("the class mix alone, at fixed size (photos only)", "n/a", "n/a",
           f"{sg(m3_dmix.estimate)} {ci(m3_dmix.ci_lo, m3_dmix.ci_hi, 3, True)}", nd(m3_dmix.ci_lo, m3_dmix.ci_hi)),
          ("both sources in greyscale", f"{gg.drawings_bal_acc:.3f}", f"{gg.photos_bal_acc:.3f}",
           f"{sg(gg.diff_photos_minus_drawings)} {ci(gg.ci_lo, gg.ci_hi, 3, True)}", nd(gg.ci_lo, gg.ci_hi)),
          ("photos cropped to the aircraft, minus uncropped (paired, makers)", "n/a",
           f"{ck.loc['cropped'].knn_bal_acc:.3f}", mci(crop), mnd(crop))]
    D.table(md_df(fr, ["comparison", "drawings", "photos", "difference [95 % CI]", "reading"]),
            f"Photographs against drawings on the shared 5 classes, one control at a time (registers, layer 24 CLS, "
            "kNN-5, maker held out). The difference is photos minus drawings unless the row says otherwise.",
            f"**Photographs score higher at equal pool size ({sg(m3_size.estimate)}); once the photo pool also has "
            f"the drawings' class mix, no difference is detected ({sg(m3_mix.estimate)}), though the comparison is "
            "too imprecise to rule out a medium effect of the size found at full scale.** The class mix alone "
            f"moves the photo score by {sg(m3_dmix.estimate)}, {nd(m3_dmix.ci_lo, m3_dmix.ci_hi)}, so it is the "
            "remaining candidate, not a demonstrated cause. Colour is not the cause (the lead survives with both "
            f"sources in grey), and neither is the background (cropping moves the photo score by "
            f"{sg(crop['diff'])}; on the base model {sg(crop_b['diff'])}).", right=["drawings", "photos"])
    ffb = rf_figure("fig_recall_vs_pool")
    pcs = X["M3_per_class_pool_share"].set_index("class")
    D.figure(ffb, "recall per class against the class's share of the neighbour pool: the shared 5 classes in "
                  "both sources (left), the 12 G1 types of the drawings (right).",
             f"**A class is read well largely when it fills the pool.** Wingless multicopters, "
             f"{pc(pcs.loc['WM', 'pool_share_drawings'])} of the drawing pool and "
             f"{pc(pcs.loc['WM', 'pool_share_photos'])} of the photo pool, are read at "
             f"{pcs.loc['WM', 'recall_drawings']:.2f} from drawings and {pcs.loc['WM', 'recall_photos']:.2f} from "
             f"photos, and at {pcs.loc['WM', 'recall_matched']:.2f} once the photos have the drawings' mix; Vectored "
             f"Thrust goes the other way ({pcs.loc['VT', 'recall_drawings']:.2f} drawings, "
             f"{pcs.loc['VT', 'recall_photos']:.2f} photos, {pcs.loc['VT', 'recall_matched']:.2f} matched).",
             width=92)

    # 5.2
    s6 = {st: mi.loc[("registers", "patent to photo", st)] for st in ("built", "concept")}
    D.method(
        "5.2", "Can the same aircraft be found in both sources?",
        "Whether the model links a patent drawing to a photograph of the same aircraft, and the reverse: the "
        "strictest form of the fairness question, one aircraft in two media.",
        f"The {int(mpp.queries)} patent aircraft that have a directory page, as queries against the "
        f"{n_(int(mpp.gallery))} photo aircraft whose hero image is not a line drawing, and those photo aircraft "
        f"against the {n_(int(mph.gallery))} aircraft of the patent main set. Of the {n_link} linked aircraft of "
        f"1.1, {n_gate_l} leave through the domain gate (similar to a UAV) and {n_line_l} whose page shows only a "
        "line drawing.",
        f"Each query ranks the gallery by cosine distance. Table {D.tabnum()} gives the share of queries whose "
        "own page is in the top 10 and the top 50, with bootstrap intervals over the queries, by page status and "
        f"for the base model, and the median rank; Figure {D.fignum()} the share within the top k for every k. "
        "Line drawings leave the gallery because some pages reproduce the patent figure itself.",
        "Against a random ranking: P(own page in the top k) = 1 − Π_{i<k} (G − n₊ − i) / (G − i), averaged "
        "over the queries, with G the gallery size and n₊ the query's own pages (the author's own baseline).",
        f"The {int(mpp.queries)} linked aircraft are well-known programmes with a public product, which a "
        "directory page exists for: a biased subset of the patent aircraft.")

    def mrow(lab, r):
        return (lab, int(r.queries), f"{pc(r.top10, 1)} {ci(100 * r.top10_boot_lo, 100 * r.top10_boot_hi, 1)}",
                f"{pc(r.top50, 1)} {ci(100 * r.top50_boot_lo, 100 * r.top50_boot_hi, 1)}", f"{r.median_rank:g}")
    D.table(md_df([mrow("patent → photo, registers", mr_),
                   mrow("patent → photo, built aircraft", s6["built"]),
                   mrow("patent → photo, concept aircraft", s6["concept"]),
                   mrow("photo → patent, registers", mi.loc[("registers", "photo to patent", "all")]),
                   mrow("patent → photo, base model", mb_)],
                  ["direction", "queries", "top 10, % [95 % CI]", "top 50, % [95 % CI]", "median rank"]),
            f"Same-aircraft matching (layer 24 CLS, no line drawings in the gallery); at random the top 10 and top "
            f"50 hold the own page for {pc(mr_.random_top10, 1)} and {pc(mr_.random_top50, 1)} of the patent "
            "queries.",
            f"**The own page is in the top 50 for {pc(mr_.top50, 1)} of the patent queries "
            f"{ci(100 * mr_.top50_boot_lo, 100 * mr_.top50_boot_hi, 1)}, against {pc(mr_.random_top50, 1)} at "
            f"random and {pc(mb_.top50, 1)} with the base model.** Concept aircraft find their page in "
            f"{pc(s6['concept'].top50, 1)} of the queries and built ones in {pc(s6['built'].top50, 1)}; "
            + ("the two intervals overlap, so no difference is claimed." if s6['built'].top50_boot_hi >= s6['concept'].top50_boot_lo
               else "the two intervals do not overlap."))
    note("f_matching.png", f"Source: {MODEL}; layer 24 CLS; {int(mpp.queries)} patent main figures as queries "
                           f"against {n_(int(mpp.gallery))} evtol.news aircraft without line drawings")
    with _into_brief():
        f17, _mr = AD.fig_matching(C)
    D.figure(f17, "how often a patent aircraft finds its own directory page within the top k photographs, against "
                  "a random ranking.",
             f"**The link is well above chance at every shortlist size but far from a retrieval tool: the median "
             f"rank is {mpp.median_rank:g}.**", width=70)
    D.answer(
        "Photographs score higher at equal pool size; once the photo pool also has the drawings' class mix, no "
        "difference is detected, though the comparison is too imprecise to rule out a medium effect of the size "
        "found at full scale.",
        [f"**Full sets.** Photos {mh.knn_bal_acc:.3f} against drawings {m5.knn_bal_acc:.3f}, "
         f"{sg(m3_full.estimate)} {ci(m3_full.ci_lo, m3_full.ci_hi, 3, True)}.",
         f"**Not the pool size, not the colour, not the background.** Equal size {sg(m3_size.estimate)}, both "
         f"grey {sg(gg.diff_photos_minus_drawings)}, cropping {sg(crop['diff'])} ({mnd(crop)}).",
         f"**The class mix is the remaining candidate.** With the drawings' mix {sg(m3_mix.estimate)} "
         f"{ci(m3_mix.ci_lo, m3_mix.ci_hi, 3, True)}, {nd(m3_mix.ci_lo, m3_mix.ci_hi)}; the mix alone "
         f"{sg(m3_dmix.estimate)}, {nd(m3_dmix.ci_lo, m3_dmix.ci_hi)}.",
         f"**One aircraft in two media.** {pc(mr_.top50)} of the linked patent aircraft find their own page in the "
         f"top 50 (random {pc(mr_.random_top50, 1)}), on a subset of well-known programmes."])

    # ═══ closing ═══
    D.closing("The answer to RQ4, what it rests on, and what is still open")
    D.answer(
        "The figure alone carries the architecture: weakly in patent drawings, moderately in photographs as they "
        "come, and no difference between the two media is detected once the photo pool has the drawings' size "
        "and class mix.",
        [f"**Drawings.** A frozen general-purpose model, never trained on aircraft or on the codebook, recovers "
         f"the 12 types from one drawing per aircraft at {mp.knn_bal_acc:.3f} (shuffled level "
         f"{f2(mp.knn_chance_p95)}), about {head12:.2f} after the selection optimism and {base_sc[L12]:.3f} with "
         f"the base model planned first; the 5 classes at {f2(m5.knn_bal_acc)}.",
         f"**Photographs.** {f2(mh.knn_bal_acc)} on the 5 classes; the lead over the drawings survives equal pool "
         "size and greyscale, cropping to the aircraft changes nothing detectable, and the lead is no longer "
         "detected with the drawings' class mix.",
         "**It holds** with the maker held out, under every image rule, without the photo backgrounds, in "
         "greyscale, and after the view of the drawing is linearly removed; these contrasts are read together, "
         "none of them surviving a correction for their number.",
         f"**The drawing embedding is organised first by how the aircraft is drawn**, second by what it is: the "
         f"d ratio is {vrr.estimate:.2f}, and the types form no clusters of their own.",
         "**The human reading stays the reference.** The architecture is in the vector (the probe reads more "
         "than the neighbours) but does not decide which aircraft are nearest."], head="Answer to RQ4")
    D.p("**What DINOv2 contributes to the thesis.**")
    D.bullets([
        "**A test of the taxonomy by an observer that has never seen it.** The types it recovers best are the "
        "large ones; " + (f"{', '.join(disp(c) for c in above)}, never recovered in the full sample, is recovered "
                          "above chance once class size is equalised, " if above else "")
        + (f"while {' and '.join(disp(c) for c in notabove)} rest on too few aircraft to say." if notabove else ""),
        "**A link from the patent record to the directory** of built and proposed aircraft, above chance.",
        "**A labelled benchmark of whole-aircraft drawings with a fixed protocol**, on which a fine-tuned or "
        "aircraft-specific model can be measured."])
    D.p("**A reusable recipe for patent-drawing studies.**")
    D.add(f"1. Test the checkpoint before reading any score: if a late layer puts more than about 0.1 of the CLS "
          f"attention on one patch, or its largest patch in the pad, use a registers checkpoint (base model: in "
          f"the pad for {pc(pad[('base', 'patent')], 1)} of the drawings; registers: "
          f"{pc(pad[('registers', 'patent')], 1)}).",
          "2. Feed the largest native input size (518 px) on a white pad, and keep the lines as drawn: no "
          "rendering change tested improves the score; a gentler change is untested.",
          "3. Choose the layer and token with the split-half rule over makers, and report the optimism gap beside "
          "the score.",
          "4. Hold the maker out of every neighbour, resample makers for every interval, and test the view as a "
          "confound before reading any class result.", "")
    D.p("**What the results rest on.**")
    D.bullets([
        f"**One labeller.** The G1 labels and the whole-patent reading are the author's (κ {kap:.2f}; "
        f"{float(k_rec.kappa):.3f} for the frozen labels); no second labeller is recorded.",
        f"**Small and unbalanced classes.** "
        f"{', '.join(f'{disp(c)} {int(cnt12[c])}' for c in sorted(small, key=lambda c: cnt12[c]))} aircraft; the "
        "two sources differ in class mix, and the matched comparison is wide.",
        f"**A linear view erasure only.** Held out it removes {pc(ho['registers'].share_of_above_chance_removed)} "
        f"of the view signal, and the view kNN stays at {vk_e.value:.2f} after it.",
        f"**Post-hoc steps.** The registers switch after its scores were seen; the size and mix controls after "
        "the photo lead was seen; the grey and contrast drawing controls, the grey-with-grey comparison and the "
        "registers crop after the review; the exp4 subset after the rule scores.",
        f"**Many contrasts.** {n_con} paired contrasts, none surviving a Holm correction: the evidence is their "
        "agreement, not any one of them.",
        f"**A photo filter audit that is not blind**, judged by the pipeline's AI assistant; the {n_rev} "
        "review-band images were never decided by hand.",
        f"**A biased matching subset.** The {int(mpp.queries)} linked aircraft are well-known programmes.",
        "**One frozen model.** A fine-tuned model, or one trained on technical drawings, is outside the scope."])
    D.p("**Questions still open.** Each is a choice for the chapter and can be settled with the results already "
        "held.")
    D.bullets([
        f"**Does same-aircraft matching earn its own section?** It is above chance ({pc(mr_.top50)} in the top 50) "
        "but far from reliable; it could shrink to one paragraph.",
        "**Does the exp4 question stay?** Averaging views is a null; it may leave the chapter.",
        "**A second labeller?** A blind second reading of a sample of drawings would give an independent κ."])
    D.p("**References**")
    D.p("Brodersen, K. H., Ong, C. S., Stephan, K. E. and Buhmann, J. M. (2010). The balanced accuracy and its "
        "posterior distribution. In *Proceedings of the 20th International Conference on Pattern Recognition*, "
        "3121-3124.",
        "Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological "
        "Measurement*, 20(1), 37-46.",
        "Cover, T. M. and Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on "
        "Information Theory*, 13(1), 21-27.",
        "Darcet, T., Oquab, M., Mairal, J. and Bojanowski, P. (2024). Vision transformers need registers. In "
        "*International Conference on Learning Representations (ICLR 2024)*.",
        "Efron, B. (1979). Bootstrap methods: another look at the jackknife. *The Annals of Statistics*, 7(1), "
        "1-26.",
        "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of "
        "Statistics*, 6(2), 65-70.",
        "Hubert, L. and Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.",
        "McInnes, L., Healy, J. and Melville, J. (2018). UMAP: Uniform manifold approximation and projection for "
        "dimension reduction. arXiv:1802.03426.",
        "Oquab, M., Darcet, T., Moutakanni, T. et al. (2023). DINOv2: Learning robust visual features without "
        "supervision. arXiv:2304.07193.",
        "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of "
        "the American Statistical Association*, 22(158), 209-212.")

    # ═══ appendix ═══
    if os.environ.get("BRIEF_APPENDIX", "1") == "1":
        D.add('<div class="keep chapter-start" markdown="1">', "", "## Appendix A — Every layer and token under "
              "both models", "", "</div>", "")
        D.p("Every matrix under both models, maker held out: kNN-5 balanced accuracy with the registers model's "
            "interval and shuffled level, and the linear probe. Base appears here only as the comparison; every "
            "chapter uses the registers model.")

        def grid(Rb, Rr, key, caption):
            mb, mr = Rb[key]["main"].set_index("matrix"), Rr[key]["main"].set_index("matrix")
            rows = [(mx, f"{mb.loc[mx].knn_bal_acc:.3f}", f"{mr.loc[mx].knn_bal_acc:.3f}",
                     ci(mr.loc[mx].knn_ci_lo, mr.loc[mx].knn_ci_hi), f"{mr.loc[mx].knn_chance_p95:.3f}",
                     f"{mb.loc[mx].probe_bal_acc:.3f}", f"{mr.loc[mx].probe_bal_acc:.3f}") for mx in mr.index]
            up = int((mr.knn_bal_acc > mb.knn_bal_acc.reindex(mr.index)).sum())
            top = mr.knn_bal_acc.idxmax()
            D.table(md_df(rows, ["matrix", "kNN-5 base", "kNN-5 registers", "95 % interval", "shuffled level",
                                 "probe base", "probe registers"]), caption,
                    f"**Registers score higher than base in {up} of the 6 matrices, and {top} is the best matrix "
                    "under registers.**")
        grid(Bp, Rp, "ev", f"Patent drawings, 12 types ({n_(nP)} aircraft, main figure).")
        grid(Bp, Rp, "ev5", f"Patent drawings, 5 parent classes ({n_(nP)} aircraft, main figure).")
        grid(Bh, Rh, "ev", f"Photographs, 5 directory classes ({n_(nH)} aircraft, hero image).")

    text = D.text()
    text = text.replace("@@TOC@@", toc_md(toc_entries(text)))
    info.update({"figs": D.figs, "ntab": D.ntab, "toc": D.toc})
    return text, info


def check_text(text: str) -> None:
    """No em dash outside the headings, captions and index rows that mirror the brief; no pointer out."""
    bad = []
    for line in text.split("\n"):
        stripped = line
        for rx in DASH_OK:
            stripped = rx.sub("", stripped)
        if "—" in stripped and not line.startswith("<div class=\"toc-row"):
            bad.append(line[:120])
    if bad:
        raise ValueError(f"em dash in prose: {bad[:3]}")
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # image links are paths by necessity
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    hits = re.findall(r"[\w/.-]+\.(?:csv|py|pkl|md|html|json)\b|/mnt/|reference pack|notebook \d+", body)
    if hits:
        raise ValueError(f"pointer out of the document: {sorted(set(hits))[:5]}")


def write_markdown() -> Tuple[Path, Dict]:
    FIGB.mkdir(parents=True, exist_ok=True)
    PRIVB.mkdir(parents=True, exist_ok=True)
    if not PRIV_LINK.exists():
        os.symlink(PRIVB, PRIV_LINK)
    C = AD.load()
    text, info = build(C)
    # the two slow computations, kept so that a rebuild with BRIEF_REUSE=1 need not redo them
    import pickle
    (OUT_DIR / "_brief_cache.pkl").write_bytes(pickle.dumps({"class_mix": info["class_mix"], "exp4": info["exp4"]}))
    if _PENDING:
        raise ValueError(f"source notes registered for figures never drawn: {sorted(_PENDING)}")
    check_text(text)
    MD_PATH.write_text(text, encoding="utf-8")
    return MD_PATH, info
