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


# ── the text ─────────────────────────────────────────────────────────────────
def build(C: Dict) -> Tuple[str, Dict]:
    D = BDoc()
    info: Dict = {}
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
    pv = C["reg_paired"].set_index(["source", "label"])
    rk = C["reg_knn"]

    def rkv(tag, src, lab):
        return rk[(rk.tag == tag) & (rk.source == src) & (rk.label == lab) & (rk.matrix == "L24 CLS")].iloc[0]
    dec = C["s1a_decision"].set_index("candidate")
    _c = C["s1a_candidates"]
    reg_only = _c[_c.pool == "registers only"]
    l22f = reg_only[reg_only.candidate == "registers L22 CLS"].set_index("label").selection_freq
    s1a_all = _c[_c.pool == "all 12"]
    reg_share = s1a_all[s1a_all.candidate.str.startswith("registers")].groupby("label").selection_freq.sum()
    s1b = C["s1b_decision"].set_index("candidate")
    s1bc = C["s1b_candidates"].set_index(["label", "candidate"])
    pair = C["s2s3_paired"].set_index(["label", "comparison"])
    s2k = C["s2s3_knn"].set_index(["label", "condition"])
    fair = C["fairness"].set_index("photos")
    lab_reg = C["s1a_labels"][C["s1a_labels"].pool == "registers only"].set_index("label")
    lab_all = C["s1a_labels"][C["s1a_labels"].pool == "all 12"].set_index("label")
    lab_b = C["s1b_labels"].set_index("label")
    half = C["s1a_half_scores"]
    hs = half[half.candidate.isin(["registers L24 CLS", "base L24 CLS"])].pivot_table(
        index=["split", "half", "label"], columns="candidate", values="score").reset_index()
    hs["d"] = hs["registers L24 CLS"] - hs["base L24 CLS"]
    hsd = hs.groupby("label").agg(mean=("d", "mean"))
    hn = half.groupby("label").n_aircraft.agg(["min", "max"])
    sh = C["share"].set_index(["model", "source", "layer"])
    mt = C["match"]
    mt = mt[(mt.images == "no line drawings") & (mt.layer == 24) & (mt.pooling == "cls")].set_index("direction")
    mtb = C["match_base"]
    mtb = mtb[(mtb.embedding == AD.BASE_TAG) & (mtb.images == "no line drawings") & (mtb.layer == 24) &
              (mtb.pooling == "cls")].set_index("direction")
    mpp, mph = mt.loc["patent->photo"], mt.loc["photo->patent"]
    cm = C["crop_man"]
    n_fb = int((cm.status != "ok").sum())
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
    fc = AD.five_class(C)
    fci = fc.set_index("class")
    cache_f = OUT_DIR / "_brief_cache.pkl"
    cache = None
    if os.environ.get("BRIEF_REUSE") == "1" and cache_f.exists():
        import pickle
        cache = pickle.loads(cache_f.read_bytes())       # written by the last full run of this module
    cmx = cache["class_mix"] if cache else AD.class_mix_matched(C)
    info["class_mix"] = cmx
    dA, pA = fair.loc["colour"].drawings_bal_acc, fair.loc["colour"].photos_bal_acc
    cmx_inside = cmx["lo"] <= dA <= cmx["hi"]
    pin = cmx["probe_lo"] <= m5.probe_bal_acc <= cmx["probe_hi"]
    e4 = cache["exp4"] if cache else AD.exp4_subset(C)
    info["exp4"] = e4
    im, au = C["img_dec"], C["audit"]
    kept = im[im.keep.astype(str) == "True"]

    def audit(stratum):
        s = au[au.stratum == stratum]
        return int((s.verdict == "whole_aircraft").sum()), len(s)
    (k_m, n_m), (k_s, n_s), (k_r, n_r) = audit("main"), audit("siglip_drop"), audit("review_band")
    n_rev = int((im.auto == "review").sum())
    n_hard = int((im.hard_rule == "").sum())
    pch = C["parent_check"]
    fun = C["funnel"]
    n_file = int(fun.remaining.iloc[0])
    n_notappr = int(fun[fun.step == "figure not approved"].removed.iloc[0])
    n_gate = int(fun[fun.step.str.startswith("domain gate")].removed.sum())
    n_dup = int(fun[fun.step.str.startswith("duplicate patent")].removed.sum())
    n_part = int(fun[fun.step == "not a whole-aircraft figure"].removed.iloc[0])
    conv = AD.CONVERTIBLE
    var = wv[wv.g1_code.isin(conv)]
    ctab = pd.crosstab(var.g1_code, var.state4).reindex(conv).fillna(0).astype(int)
    ctab = ctab[[c for c in ["Cruise", "Hover", "Both", "Other"] if c in ctab.columns]]
    pc_ = var[(var.view4 == "Perspective") & (var.state4 == "Cruise")].aircraft_id.nunique()
    ph_ = var[(var.view4 == "Perspective") & (var.state4 == "Hover")].aircraft_id.nunique()
    n_var = var.aircraft_id.nunique()
    sm1 = pd.read_csv(AD.PAT / "2_embedding_extraction/preprocess_variants/thick1/sources_manifest.csv",
                      keep_default_na=False)
    sm2 = pd.read_csv(AD.PAT / "2_embedding_extraction/preprocess_variants/thick2/sources_manifest.csv",
                      keep_default_na=False)
    n_col = int((pd.to_numeric(sm1.src_max_chroma) > 8).sum())
    rec = Rp["ev"]["recall"]
    rec = rec[rec.matrix == "L24 CLS"].iloc[0]
    cl = ER.G1_ORDER
    best = rec[cl].astype(float).sort_values(ascending=False)
    zero = [disp(c) for c in cl if float(rec[c]) == 0]
    big = [c for c in best.index if cnt12[c] >= 20]
    smallr = [c for c in best.index if cnt12[c] < 20 and float(rec[c]) >= 0.3]
    pred = Rp["ev"]["pred"][AD.KEY]
    wrong = pred != S_pat.y
    top2 = list(cnt12.sort_values(ascending=False).index[:2])
    into = float(np.isin(pred[wrong], top2).mean())
    rh = Rh["ev"]["recall"]
    rh = rh[rh.matrix == "L24 CLS"].iloc[0]
    rbest = max(ER.CLASSES5, key=lambda c: float(rh[c]))
    rworst = min(ER.CLASSES5, key=lambda c: float(rh[c]))
    T, BT = AD.REG_TAG, AD.BASE_TAG
    ga, pa_ = "architecture kNN-5, G1 12-class", "architecture kNN-5, parent 5-class"
    ve = lambda tag, sec, var_, meas: AD.ve_get(C, tag, sec, var_, meas)  # noqa: E731
    vp_o = ve(T, "view probe (aircraft main figures)", "original", "probe_bal_acc").value
    vp_e = ve(T, "view probe (aircraft main figures)", "view erasure", "probe_bal_acc").value
    vk_o = ve(T, "view kNN-5 (figures, maker held out)", "original", "bal_acc")
    vk_e = ve(T, "view kNN-5 (figures, maker held out)", "view erasure", "bal_acc")
    d12 = ve(T, ga, "view erasure - original", "paired_diff")
    d5 = ve(T, pa_, "view erasure - original", "paired_diff")
    b12 = ve(BT, ga, "view erasure - original", "paired_diff")
    b5 = ve(BT, pa_, "view erasure - original", "paired_diff")
    e12, e5 = ve(BT, ga, "view erasure", "bal_acc").value, ve(BT, pa_, "view erasure", "bal_acc").value
    r12, r5 = ve(T, ga, "view erasure", "bal_acc").value, ve(T, pa_, "view erasure", "bal_acc").value
    ck = C["crop_knn"].set_index(["matrix", "condition"])
    cd = C["crop_diff"].set_index("matrix")
    att_max = max(sh.loc[("registers", s, L)].max_patch_share_median for s in ("patent", "photo") for L in (18, 24))
    reg_lo = min(sh.loc[("registers", s, L)].prefix_share_median for s in ("patent", "photo") for L in (18, 24))
    reg_hi = max(sh.loc[("registers", s, L)].prefix_share_median for s in ("patent", "photo") for L in (18, 24))
    n_status = int(S_evn.aircraft.status.isin(["built", "concept"]).sum())

    def nd(lo, hi) -> str:                         # the reading of an interval of a difference
        return "no difference detected" if lo <= 0 <= hi else ("above zero" if lo > 0 else "below zero")

    def s12(x) -> str:
        return f"{f2(x, 3)}"

    # ═══ title, index ═══
    import datetime as dt
    # the one printed line under the title: Word's exporter takes the first paragraph as its subtitle, so
    # without it the index block would print there as text
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
        "five nearest aircraft, never from its own maker, nearness being the cosine distance "
        "d(u, v) = 1 − u·v / (‖u‖ ‖v‖) (nearest-neighbour rule, Cover and Hart, 1967); a tie goes to the class "
        "of the nearer neighbour. The score is the **balanced accuracy**, BA = (1/K) Σ_c recall_c, with "
        "recall_c = (aircraft of class c predicted as c) / (aircraft of class c), over the K classes (Brodersen "
        "et al., 2010). Every class weighs the same, so a large class cannot carry the score, and guessing "
        "scores 1/K: 0.083 on 12 classes, 0.20 on 5.",
        f"**The 95 % interval.** The aircraft are resampled with replacement {n_(P.N_BOOT)} times and the score "
        "recomputed on the same predictions; the interval is the middle 95 % of those scores (bootstrap, Efron, "
        "1979). A difference between two setups on the same aircraft resamples the same aircraft for both "
        "(paired); between the two sources, whose aircraft differ, each set is resampled on its own. **A "
        "difference whose interval takes in zero is read as no difference detected.**",
        f"**Chance, as the same neighbours reach it.** The labels are shuffled {P.N_SHUFFLE} times and the vote "
        "rerun; the 95th percentile of the shuffled scores is the **shuffled level**. A score above it is one "
        "that chance almost never produces with these neighbours.",
        "**The linear probe.** A logistic regression reads every dimension of the vector, trained on four fifths "
        "of the makers and tested on the fifth (five folds grouped by maker), scored by the same balanced "
        "accuracy. The neighbours use the overall distance only, so a probe above the kNN score means the "
        "architecture is in the vector without deciding which aircraft are nearest.",
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
        "divided by the architecture's d, says which label moves the vectors more. Clusters: k-means with as "
        "many clusters as classes, compared with the labels by the adjusted Rand index, ARI = (RI − E[RI]) / "
        "(max RI − E[RI]), where RI is the share of aircraft pairs on which the two partitions agree; 0 is a "
        "chance partition, 1 the labels themselves (Hubert and Arabie, 1985).",
        "**The maps.** UMAP (McInnes, Healy and Melville, 2018) draws each aircraft's 1 024 numbers in two "
        "dimensions so that similar vectors sit close. The axes have no unit, and a map is a picture, never a "
        "test.")
    D.add("**The seven tests of the protocol**, run with the same code and parameters on both sources, so a "
          "number in one source means what the same number means in the other:", "",
          "| test | what it asks | measure |", "|:---|:---|:---|",
          "| T1 integrity | is every vector valid? | no NaN, infinite, all-zero or duplicate rows |",
          "| T2 structure | do the vectors have structure at all? | variance on the first principal component "
          "against a random matrix of the same shape; Hopkins statistic (0.5 random, near 1 clumped) |",
          "| T3 separation | are same-class aircraft closer? | ratio, d and p, as above |",
          "| T4 prediction | how often is the class recovered? | kNN-5 balanced accuracy (primary), interval, "
          "shuffled level; probe |",
          "| T5 clusters | do the classes form clusters? | k-means against the labels, ARI |",
          "| T6 what else | does another label move the vectors more? | its d ÷ the architecture's d |",
          "| T7 image choice | does the image rule matter? | T4 under every image rule |", "",
          "</div>", "", "")

    # 1.1
    D.method(
        "1.1", "Which drawings and photographs enter the analysis, and with which labels?",
        "Which images of each source reach the model, which label each aircraft carries, and how far that "
        "label can be trusted.",
        "Drawings: the figures of the labelled dataset of eVTOL patents, counted as figures until each aircraft "
        f"is reduced to one (1.2). Photographs: the images of the directory pages (crawl of 2026-09-17), counted "
        "as images until each page is reduced to one. A drawing aircraft carries its human **G1 type**, one of "
        "the codebook's 12 architecture types (Table 1), and the type's parent among 5 classes: **VT** Vectored "
        "Thrust, **LC** Lift + Cruise, **WM** Wingless (Multicopter), **ER** Electric Rotorcraft, **HB** Hover "
        "Bikes / PFD (personal flying devices). A photo aircraft carries the directory's own class, one of the same 5 "
        f"(VT {cnt5h['VT']}, LC {cnt5h['LC']}, WM {cnt5h['WM']}, ER {cnt5h['ER']}, HB {cnt5h['HB']}).",
        f"Figure {D.fignum()} follows each source from the files on disk to the analysis set, each image counted "
        "under the first rule that removed it. A drawing enters when the labeller approved it, its aircraft "
        "passes the domain gate (aircraft flagged as similar to a UAV, or not electric, leave) and its patent "
        "is not a duplicate of another; part and detail figures leave, and so do the "
        f"{n_excl} aircraft marked unclassifiable. A photo passes fixed rules against small files and files "
        "shared with another page, then a zero-shot SigLIP score p(whole aircraft) that keeps an image at "
        f"p ≥ 0.80, drops it below 0.35 and holds back the review band between. The page's first kept image, "
        f"its **hero image**, stands for the aircraft. Figures {D.fignum(2)} and {D.fignum(3)} show one image "
        "per class.",
        "The drawing label is checked against the independent whole-patent reading of the architecture by "
        "Cohen's κ = (p_o − p_e) / (1 − p_e), the observed agreement p_o corrected for the agreement p_e two "
        "independent readings would reach by chance (Cohen, 1960). The photo filter is checked on a random "
        "sample of hero, dropped and review-band images.",
        "The audit judge was the AI assistant used for the pipeline, working from contact sheets that showed "
        "the filter score, so the audit is neither blind nor yet re-checked by the author. The photo label is "
        "the directory's class, not the codebook's.")
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
             f"{n_(len(kept))} of {n_(len(im))} images, holding back the {n_rev} of the review band; one per page "
             "is read.")
    shares = cnt12.sort_values(ascending=False)
    n_tiny = int((cnt12 <= 10).sum())
    D.table(md_df([(disp(c), AD.G1_NAME[c], ER.PARENT[c], int(cnt12.get(c, 0))) for c in ER.G1_ORDER],
                  ["G1 code", "architecture type", "parent class", "aircraft"]),
            f"The 12 architecture types (G1) of the codebook, their parent class and the drawing aircraft per "
            f"type ({n_(nP)} aircraft). TP, Tilt Propulsor, is the display name of the type the labelling tool "
            "records as TR.",
            f"**Two types hold {pc((shares.iloc[0] + shares.iloc[1]) / nP)} of the drawing aircraft, "
            f"{AD.G1_NAME[shares.index[0]]} ({shares.iloc[0]}) and {AD.G1_NAME[shares.index[1]]} "
            f"({shares.iloc[1]}), while {n_tiny} types have ten aircraft or fewer.** The label agrees with the "
            f"whole-patent reading for {agree} of {gt_n} aircraft (κ {kap:.2f}).")
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
             f"**A hero image shows the whole aircraft, a photograph or a rendering, often in a scene: {k_m} of "
             f"{n_m} audited hero images show the whole aircraft.** Of the images the filter dropped, {k_s} of "
             f"{n_s} audited were a whole vehicle, and {k_r} of {n_r} review-band images would have been usable. "
             f"On the {len(pch)} patent aircraft that have a directory page, the directory class agrees with the "
             f"parent of the human G1 type for {int(pch.agrees.sum())}.", width=100)

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
        "vector. The question is which rules exist, how often they agree, and whether this or any other "
        "selection step moves the primary score.",
        f"The {n_(nP)} drawing aircraft and their {n_(nW)} figures; for the flight state, the figures of the "
        f"{len(conv)} convertible types (TP; CVT, Combined Vectored Thrust; TW, Tilt Wing; DS, Deflected "
        "Slipstream; SRW, Stopped/Slowed Rotor Wing), whose shape changes between hover and cruise; the "
        f"{n_(nH)} photo aircraft for the sensitivity check.",
        f"Four rules were fixed before any score was read (Table {D.tabnum()}): **exp1 view-first** sorts on "
        "the view (Perspective > Plan > Side > Front/Rear), then on the flight state; **exp2 state-first** sorts "
        "on the flight state (Cruise > Both > Hover > Other), then on the view; **exp3 main figure** takes the "
        "figure the labeller marked as main, the primary rule; **exp4 view average** averages the vectors of "
        "the best figure in each of Perspective, Plan and Side. The flight state is recorded per figure as "
        f"Cruise, Hover, or Both when the moving part is drawn in both positions (Table {D.tabnum(2)}). The chosen "
        "image is rotated by its labelled angle, padded to a white square and resized to 518 px, 37 × 37 "
        "patches of 14 px, for drawings and photos alike. Each selection step is then made stricter, or its "
        f"measured errors removed, and the primary score recomputed (Table {D.tabnum(3)}).",
        f"Cruise is the canonical state because more convertible aircraft have a Perspective figure in Cruise "
        f"({pc_}) than in Hover ({ph_}) among the {n_var}. The white pad matches the paper of a drawing, so the "
        "square format adds no content to it. A selection step decides the result only if a variant leaves the "
        "interval of the set as built (methods box).",
        "The variants are subsets of the set as built, so their intervals are not independent of it.")
    D.table(md_df([
        ("exp1 view-first", "best view, then best flight state", pc(same("exp1 view-first", "exp3 main figure"), 1)),
        ("exp2 state-first", "best flight state, then best view", pc(same("exp2 state-first", "exp3 main figure"), 1)),
        ("exp3 main figure", "the figure the labeller marked as main (primary rule)", "100.0 %"),
        ("exp4 view average", "mean of the best figure of each of Perspective, Plan and Side", "n/a"),
    ], ["rule", "which figure", "same figure as exp3"]),
        f"The four image rules. exp1 and exp2 pick the same figure for "
        f"{pc(same('exp1 view-first', 'exp2 state-first'), 1)} of the aircraft; exp4 averages 2 figures for "
        f"{int((nslots == 2).sum())} aircraft and 3 for {int((nslots == 3).sum())}, and is one figure for the "
        f"other {int((nslots == 1).sum())}.",
        f"**The flight state barely moves the choice: exp1 and exp2 differ on "
        f"{int(round((1 - same('exp1 view-first', 'exp2 state-first')) * nP))} aircraft only.** The labeller's "
        f"main figure differs from the view-first pick on "
        f"{int(round((1 - same('exp1 view-first', 'exp3 main figure')) * nP))}.", right=["same figure as exp3"])
    big3 = [c for c in ctab.index if ctab.loc[c].sum() >= 50]
    ct = ctab.copy()
    ct.insert(0, "type", [disp(c) for c in ct.index])
    D.table(ct.reset_index(drop=True), "Flight state of the whole-aircraft drawings of the convertible types.",
            f"**In the three large convertible types ({', '.join(disp(c) for c in big3)}) Cruise and Hover are "
            f"drawn about equally often, and Both is rare everywhere (at most {int(ctab['Both'].max())} figures "
            f"per type)**, so the preference for Cruise decides little.")
    D.table(md_df(sens_rows, ["source", "variant", "aircraft", "kNN-5", "95 % interval"]),
            "Sensitivity of the primary score to the selection (registers, layer 24 CLS, maker held out).",
            (f"**Every variant stays inside the interval of the set as built: no selection step decides the "
             f"result.** The drawing variants span {rng_['drawings'][0]:.3f} to {rng_['drawings'][1]:.3f}, the "
             f"photo variants {rng_['photos'][0]:.3f} to {rng_['photos'][1]:.3f}." if inside_all else
             "**At least one variant leaves the interval of the set as built.**"))
    D.answer(
        f"{n_(nP)} drawing aircraft and {n_(nH)} photo aircraft enter, each as one image and one vector, with "
        f"labels that agree with an independent reading, and no selection step moves the primary score outside "
        "its interval.",
        [f"**Two sources, one unit.** {n_(nW)} whole-aircraft drawings from {n_(n_pat_patents)} patents, and "
         f"one hero image per directory page, each reduced to one vector per aircraft.",
         f"**The labels hold up.** The drawing type agrees with the whole-patent reading for {agree} of {gt_n} "
         f"aircraft (κ {kap:.2f}); the directory class agrees with the parent type for {int(pch.agrees.sum())} "
         f"of {len(pch)} aircraft present in both.",
         f"**The image rule is fixed before any score.** The labeller's main figure is the primary rule; the "
         f"flight state changes the pick for {int(round((1 - same('exp1 view-first', 'exp2 state-first')) * nP))} "
         "aircraft only.",
         "**The selection does not decide the result.** Stricter filters, label-checked aircraft and "
         "Perspective-only aircraft all score inside the interval of the set as built.",
         f"**The photo filter audit is not blind.** {k_m} of {n_m} audited hero images show the whole aircraft, "
         "judged by the pipeline's AI assistant from sheets that showed the filter score."])

    # ═══ 2 ═══
    D.chapter("2", "The setup: which DINOv2 checkpoint, layer, token and image preparation read the images, "
                   "and how are they chosen?",
              "A frozen model leaves few choices, but each can move the score. They fall into two kinds:",
              "- **Image-type choices** follow from what patent drawings are, thin black lines on a large white "
              "field, so they transfer to any patent-drawing study: the checkpoint and the input size.\n"
              "- **Label-selected choices** are made on the labels and are therefore chosen by the split-half "
              "rule (methods box): the layer, the token and each source's image preparation.",
              f"A half holds {hn.loc['patents G1 12-class', 'min']} to {hn.loc['patents G1 12-class', 'max']} "
              f"drawing aircraft and {hn.loc['photos 5-class', 'min']} to {hn.loc['photos 5-class', 'max']} photo "
              "aircraft, so its neighbour pool is about half the full one: held-out scores sit below the "
              "full-sample scores of chapters 3 to 5, and only held-out scores compare with each other.")
    D.table(md_df([
        ("model", "DINOv2-large; DINOv2-large with registers", "image type", "the attention artifact (2.1); the held-out rule confirms"),
        ("resolution", "518 px", "image type", "the model's largest training size; not selected"),
        ("layer", "18, 22, 24", "label-selected", "split-half rule (2.2)"),
        ("token", "CLS; mean of the patch tokens", "label-selected", "split-half rule (2.2)"),
        ("drawing preparation", "plain; thick1; thick2", "label-selected", "split-half rule (2.3)"),
        ("photo preparation", "colour; greyscale", "label-selected", "split-half rule (2.3)"),
    ], ["choice", "candidates, fixed in advance", "kind", "how chosen"]), "The candidate grid.",
        "**Two choices follow from the images alone; four are chosen on the labels, and are therefore tested on "
        "makers that did not choose them.**", right=[])

    # 2.1
    rows = []
    reg_rows = []
    words = {"drawings, 12 types": "the drawings' 12 types", "drawings, 5 classes": "the drawings' 5 classes",
             "photos, 5 classes": "the photos' 5 classes"}
    for lab_s, src, lab_k, lab_sel in [("drawings, 12 types", "patents", "G1 12-class", "patents G1 12-class"),
                                       ("drawings, 5 classes", "patents", "parent 5-class", "patents parent 5-class"),
                                       ("photos, 5 classes", "photos", "5-class", "photos 5-class")]:
        pr = pv.loc[(src, lab_k)]
        reg_rows.append((words[lab_s], pr))
        rows.append((lab_s, f"{rkv(BT, src, lab_k).bal_acc:.3f}", f"{rkv(T, src, lab_k).bal_acc:.3f}",
                     f"{sg(pr['diff'])} {ci(pr.ci_lo, pr.ci_hi, 3, True)}", pc(reg_share[lab_sel], 1)))
    D.method(
        "2.1", "Does the registers checkpoint read the images better than the base model?",
        "Whether DINOv2 with registers should replace the base DINOv2: whether it removes an attention artifact "
        "on blank background, and whether it scores higher on the same aircraft.",
        f"Attention: every drawing ({int(sh.loc[('registers', 'patent', 24)].n_images)}) and photo "
        f"({n_(int(sh.loc[('registers', 'photo', 24)].n_images))}) under both models, per image, the unit at "
        f"which attention exists. Scores: the {n_(nP)} drawing and {n_(nH)} photo aircraft, the same processed "
        "images and the same code for both models.",
        f"Table {D.tabnum()} gives the collapse measure per layer: the median share of the CLS token's attention "
        f"that falls on the single largest patch. Figure {D.fignum()} shows the maps on four images. Table "
        f"{D.tabnum(2)} compares the two models' scores at layer 24 CLS on the same aircraft, and gives the share "
        "of the 400 held-out choices that pick a registers candidate when all 12 candidates (2 models × 6 "
        "matrices) compete.",
        "DINOv2 without registers parks high-norm artifact tokens in low-information background patches, and "
        "the registers checkpoint adds four register tokens that absorb this mass (Darcet et al., 2024); a "
        "patent drawing is mostly white background. The attention map is the CLS query against every patch key, "
        "averaged over the heads. Paired differences and the held-out share follow the methods box.",
        "The registers checkpoint entered the candidate set after the base model's collapse had been seen, so "
        "testing it is a post-hoc step; the held-out rule then confirms it. Part of its advantage does not "
        "survive the removal of the view (3.2).")
    D.table(md_df([(f"L{L}", f"{sh.loc[('base', 'patent', L)].max_patch_share_median:.3f}",
                    f"{sh.loc[('registers', 'patent', L)].max_patch_share_median:.3f}",
                    f"{sh.loc[('base', 'photo', L)].max_patch_share_median:.3f}",
                    f"{sh.loc[('registers', 'photo', L)].max_patch_share_median:.3f}") for L in (18, 22, 24)],
                  ["layer", "drawings, base", "drawings, registers", "photos, base", "photos, registers"]),
            "Median share of the CLS attention that falls on the single largest patch (the collapse measure).",
            f"**In the base model the late-layer attention collapses onto one patch, "
            f"{sh.loc[('base', 'patent', 24)].max_patch_share_median:.3f} of it at layer 24 on drawings and "
            f"{sh.loc[('base', 'photo', 24)].max_patch_share_median:.3f} on photos; with registers it stays at or "
            f"below {max(sh.loc[('registers', s, L)].max_patch_share_median for s in ('patent', 'photo') for L in (18, 22, 24)):.3f}.**")
    note("f_attention_base_vs_registers.png",
         "Source: DINOv2-L and DINOv2-L with registers, frozen, 518 px; CLS attention, mean over heads, each map "
         f"scaled to its own 99th percentile; 2 drawings and 2 photos, first aircraft of each class in both runs. "
         f"{CREDIT}")
    with _into_brief():
        f4 = AD.fig_attention_models(C)
    D.figure(f4, "where the CLS token looks: base (columns 2 to 4) against registers (columns 5 to 7), layers 18, "
                 "22 and 24, on the same images; bright = attention.",
             "**In base layers 22 and 24 the attention gathers on the padding patch in the top-left corner and on "
             "dots of the white field; with registers the late layers stay on the aircraft.**", width=92)
    exc = [lab for lab, pr in reg_rows if pr.ci_lo > 0]
    nod = [lab for lab, pr in reg_rows if pr.ci_lo <= 0 <= pr.ci_hi]
    D.table(md_df(rows, ["label set", "base", "registers", "paired difference [95 % CI]",
                         "registers chosen (held out)"]),
            "Registers against base on the same aircraft: kNN-5 balanced accuracy, layer 24 CLS, maker held out; "
            "the last column is the share of the 400 held-out choices that pick a registers candidate when all "
            "12 compete.",
            f"**Registers score higher on all three label sets; the gain is detected on "
            f"{' and on '.join(exc) if exc else 'no label set'}"
            + (f", while on {' and on '.join(nod)} no difference is detected" if nod else "") + ".** "
            f"Held out, a registers candidate wins {pc(reg_share.min(), 1)} to {pc(reg_share.max(), 1)} of the "
            f"choices; on the same halves registers minus base averages "
            f"{sg(hsd.loc['patents G1 12-class', 'mean'])} (drawings, 12 types) to "
            f"{sg(hsd.loc['patents parent 5-class', 'mean'])} (drawings, 5 classes). Patent-to-photo matching "
            f"improves too: own page in the top 50 for {pc(mtb.loc['patent->photo', 'R@50'])} of the queries "
            f"with base, {pc(mpp['R@50'])} with registers (5.2).")

    # 2.2
    lwin = {lab: reg_only[reg_only.label == lab].sort_values("selection_freq").candidate.iloc[-1].replace("registers ", "")
            for lab in reg_only.label.unique()}
    all_l24 = all(v == "L24 CLS" for v in lwin.values())
    D.method(
        "2.2", "Which layer and which token carry the architecture?",
        "Which of the six matrices (layers 18, 22 and 24, read through the CLS token or the patch mean) is "
        "carried forward.",
        "The six matrices of the registers model; the aircraft of each half; three label sets: drawings on the "
        "12 types, drawings on the 5 classes, photos on the 5 classes.",
        f"The split-half rule picks the best matrix on one half and scores it on the other, 400 times per label "
        f"set; Figure {D.fignum()} gives how often each matrix is chosen and Table {D.tabnum()} the decision, "
        "frequencies averaged over the three label sets. As a check, the protocol's in-sample layer rule (the "
        "highest score averaged over the image rules) is rerun on 200 random subsets of 80 % of the drawing "
        "aircraft.",
        "The selection frequency says how stable a choice is across makers; the held-out score, what it reaches "
        "on makers that did not choose it. The last layer's CLS token is the one DINOv2 is trained to make a "
        "global summary of the image (Oquab et al., 2023), so the choice has an expected answer to be tested "
        "against.")
    note("f_selection.png", [(0, "Source: registers model; 400 held-out choices; drawings, main figure, 12 types"),
                             (1, "Source: registers model; 400 held-out choices; drawings, main figure, 5 classes"),
                             (2, "Source: registers model; 400 held-out choices; photos, hero image, 5 classes")])
    with _into_brief():
        f5 = AD.fig_selection(C)
    D.figure(f5, "how often each layer and token is chosen by the split-half rule, per label set (registers "
                 "model); blue: the matrix carried forward.",
             ("**Layer 24 CLS is chosen most often on every label set" if all_l24 else
              "**Layer 24 CLS is not the first choice on every label set") +
             f"; its runner-up, layer 22 CLS, is chosen most on the photos ({pc(l22f['photos 5-class'])}, "
             f"against {pc(min(l22f['patents G1 12-class'], l22f['patents parent 5-class']))} to "
             f"{pc(max(l22f['patents G1 12-class'], l22f['patents parent 5-class']))} on the drawings).**")
    D.table(md_df([(c.replace("registers ", ""), pc(r.selection_freq, 1), f"{r.fixed_heldout_mean:.3f}")
                   for c, r in dec.iterrows()], ["matrix", "mean selection frequency", "mean held-out score"]),
            "The decision over the six registers matrices, frequencies and held-out scores averaged over the "
            "three label sets.",
            f"**Layer 24 CLS is chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of the choices and "
            f"layer 22 CLS in {pc(dec.loc['registers L22 CLS'].selection_freq)}; the patch-token mean is almost "
            f"never chosen.** The in-sample layer rule agrees: it picks layer 24 CLS on "
            f"{pc(lay_p.loc['L24 CLS'].wins_on_subsets)} of the 80 % subsets.")

    # 2.3
    D.method(
        "2.3", "Should the drawings or the photographs be transformed before the model reads them?",
        "Whether thickening the thin lines of a drawing, or removing the colour of a photograph, helps the model "
        "read the architecture.",
        f"The {n_(nP)} drawing aircraft (main figure) and the {n_(nH)} photo aircraft (hero image), registers "
        "model, layer 24 CLS.",
        "After the resize to 518 px a fine stroke can shrink to a grey pixel, so line thickening is a candidate: "
        "a minimum filter on the original drawing whose radius makes **thick1** grow each side of a stroke by "
        "about 1 px at the model input (about 2 px of width) and **thick2** by about 2 px per side (radius in "
        f"original pixels: median {int(sm1.radius.median())} and {int(sm2.radius.median())}). Figure "
        f"{D.fignum()} shows the three inputs. Table {D.tabnum()} compares each variant, and greyscale photos, "
        "with the untransformed image on the same aircraft, and gives how often the split-half rule chooses the "
        "variant.",
        "Paired bootstrap differences and the held-out share, as in the methods box. Greyscale also tests "
        "whether any photo advantage over drawings is colour (5.1).",
        "A gentler thickening, below 1 px per side, was not in the candidate set fixed in advance. "
        f"{n_col} of the {n_(len(sm1))} source drawings carry colour, which the thick variants also remove.")
    note("f_thickening.png", "Source: patent drawings, labeller's main figure, 3 aircraft of the setup montage "
                             "(seed 42); 518 px model inputs; red square: the detail window shown on the right")
    with _into_brief():
        f6, _tinfo = AD.fig_thickening(C)
    D.figure(f6, "plain, thick1 and thick2 as the model receives them (left), and the same 128 px detail of each "
                 "(right).",
             "**Thickening merges thin propeller blades and reference numerals, and turns hatching and shading "
             "into solid black: it fills the fine detail that separates the types.**", width=88)
    rows = []
    thick_nd = []
    for lab_s, lab in (("drawings, 12 types", "patents G1 12-class"), ("drawings, 5 classes", "patents parent 5-class")):
        for v in ("thick1", "thick2"):
            r = pair.loc[(lab, f"{v} - plain")]
            rows.append((lab_s, v, f"{s2k.loc[(lab, 'plain')].bal_acc:.3f}", f"{s2k.loc[(lab, v)].bal_acc:.3f}",
                         f"{sg(r['diff'])} {ci(r.ci_lo, r.ci_hi, 3, True)}", pc(s1bc.loc[(lab, v)].selection_freq)))
            thick_nd.append((v, r.ci_hi < 0))
    g = pair.loc[("photos 5-class", "grey - colour")]
    rows.append(("photos, 5 classes", "greyscale", f"{s2k.loc[('photos 5-class', 'colour')].bal_acc:.3f}",
                 f"{s2k.loc[('photos 5-class', 'grey')].bal_acc:.3f}", f"{sg(g['diff'])} {ci(g.ci_lo, g.ci_hi, 3, True)}",
                 pc(s1bc.loc[("photos 5-class", "grey")].selection_freq)))
    t2below = all(b for v, b in thick_nd if v == "thick2")
    D.table(md_df(rows, ["label set", "variant", "untransformed", "variant score", "difference [95 % CI]",
                         "variant chosen (held out)"]),
            "Each image preparation against the untransformed image, paired on the same aircraft (registers, "
            "layer 24 CLS).",
            "**Thickening lowers the score in every row" + (", and thick2's interval lies below zero on both label "
                                                              "sets" if t2below else "")
            + f"; greyscale photos score as colour ones ({sg(g['diff'])}, {nd(g.ci_lo, g.ci_hi)}).** The "
            f"untransformed image is kept on both sources: plain drawings in {pc(s1b.loc['plain'].selection_freq)} "
            f"of the choices, colour photos in {pc(s1b.loc['colour'].selection_freq)}, colour as the untransformed "
            "image, not as a better one.")

    # 2.4
    rows = []
    for lab, lab_s in (("patents G1 12-class", "drawings, 12 types"), ("patents parent 5-class", "drawings, 5 classes"),
                       ("photos 5-class", "photos, 5 classes")):
        r, b = lab_reg.loc[lab], lab_b.loc[lab]
        rows.append((lab_s, f"{r.most_chosen.replace('registers ', '')}: {r.chosen_heldout_mean:.3f}",
                     f"{r.winner_in_sample_mean:.3f}", f"{r.optimism_gap:.3f}",
                     f"{lab_all.loc[lab].optimism_gap:.3f}", f"{b.most_chosen}: {b.optimism_gap:.3f}"))
    D.method(
        "2.4", "How much do the label-selected choices flatter the scores, and which setup is carried forward?",
        "How much of each score is owed to having chosen the setup on the same labels, and which setup every "
        "later chapter uses.",
        "The 400 held-out choices of each label set, among the registers candidates and, as a check, among all "
        "12 candidates of both models.",
        f"Table {D.tabnum()} sets the held-out score of the chosen candidate beside the in-sample score of the "
        f"winner, per label set; Table {D.tabnum(2)} lists the setup carried forward, with the reason for each "
        "choice.",
        "gap = in-sample score of the winner − its held-out score (methods box): the amount by which choosing on "
        "the labels inflates a score.")
    D.table(md_df(rows, ["label set", "held-out score of the chosen", "in-sample score of the winner",
                         "gap, layer and token", "gap, both models", "gap, preparation"]),
            "The optimism of each label-selected choice.",
            f"**Choosing on the labels inflates a score by a few hundredths at most: {f2(lab_reg.optimism_gap.min(), 3)} "
            f"to {f2(lab_reg.optimism_gap.max(), 3)} for the layer and token, {f2(lab_all.optimism_gap.min(), 3)} "
            f"to {f2(lab_all.optimism_gap.max(), 3)} when both models compete.** That is small against the "
            f"intervals of chapters 3 to 5, which are {mp.knn_ci_hi - mp.knn_ci_lo:.2f} wide on the drawings' "
            f"12 types and {mh.knn_ci_hi - mh.knn_ci_lo:.2f} on the photos.")
    D.table(md_df([
        ("model", "DINOv2 with registers", f"no attention collapse on blank background; chosen in "
         f"{pc(reg_share.min(), 1)} to {pc(reg_share.max(), 1)} of held-out choices", "image type"),
        ("resolution", "518 px", "the largest size the model was trained on; keeps thin lines", "image type"),
        ("layer and token", "layer 24, CLS", f"chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of "
         f"held-out choices (next: layer 22 CLS, {pc(dec.loc['registers L22 CLS'].selection_freq)})", "label-selected"),
        ("drawings", "plain", "thickening fills fine detail and lowers the score", "label-selected"),
        ("photos", "colour", f"greyscale scores the same ({sg(g['diff'])}, {nd(g.ci_lo, g.ci_hi)})", "label-selected"),
    ], ["choice", "value", "reason", "kind"]), "The setup carried forward to chapters 3 to 5.",
        "**One model and one readout for both sources, with each source's image left as it comes.**", right=[])
    D.answer(
        "DINOv2 with registers, at 518 px, read at layer 24 through the CLS token, on untransformed images: the "
        "checkpoint because it does not collapse onto blank background, the rest by a held-out rule whose "
        "optimism is a few hundredths.",
        [f"**Registers.** The base model's late attention collapses onto one padding patch "
         f"({sh.loc[('base', 'patent', 24)].max_patch_share_median:.3f} of it at layer 24 on drawings); registers "
         f"remove the collapse and score higher on all three label sets ({sg(reg_rows[0][1]['diff'])} on the "
         f"drawings' 12 types).",
         f"**Layer 24 CLS.** Chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of the held-out "
         f"choices; layer 22 CLS is the runner-up, mostly on the photos.",
         f"**Untransformed images.** Thickening lowers the drawing score; greyscale changes nothing on the "
         f"photos ({nd(g.ci_lo, g.ci_hi)}).",
         f"**Small optimism.** Choosing on the labels adds {f2(lab_reg.optimism_gap.min(), 3)} to "
         f"{f2(lab_all.optimism_gap.max(), 3)} to a score."])

    # ═══ 3 ═══
    D.chapter("3", "The drawings: do patent drawings carry the architecture, and what else does the model read "
                   "in them?",
              f"Chapters 1 and 2 fixed the images and the setup. This chapter reads the {n_(nP)} drawing aircraft "
              "with them, in four steps:",
              "- **3.1: how well the drawings predict the type** (T1 to T5 of the protocol).\n"
              "- **3.2: what organises the drawing embedding**: clusters, the view of the drawing, and the "
              "architecture once the view is removed (T5, T6).\n"
              "- **3.3: whether the choice of drawing matters** (T7).\n"
              "- **3.4: where the model looks on a drawing.**")
    m = Rp["ev"]["main"]
    D.method(
        "3.1", "How well do the drawings predict the architecture type?",
        "Whether the drawing vectors carry the architecture: whether aircraft of the same type sit closer than "
        "aircraft of different types, and how often a type is recovered from the nearest aircraft.",
        f"The {n_(nP)} drawing aircraft, one main figure each, labelled with the 12 types and, folded, with the "
        "5 parent classes; all six matrices, layer 24 CLS being the one carried forward.",
        f"Soundness (T1, T2) and separation (T3) are read first. Figure {D.fignum()} gives, per matrix, the "
        "kNN-5 balanced accuracy (bars, with the interval), the shuffled level (dashed) and the probe "
        f"(diamonds); Table {D.tabnum()} the recall per type; Figure {D.fignum(2)} which types are confused "
        "with which (rows: the true type, columns: the predicted one).",
        "Balanced accuracy, bootstrap interval, shuffled level and maker-grouped probe as in the methods box; "
        "guessing scores 1/12 on the types and 1/5 on the classes.",
        f"{', '.join(f'{disp(c)} {int(cnt12[c])}' for c in sorted(small, key=lambda c: cnt12[c]))} aircraft: "
        "the recalls of these types rest on a handful of aircraft.")
    D.bullets([
        f"**Soundness.** No matrix has an invalid row. On its first principal component every matrix carries "
        f"{m.t2_pc1_vs_random.min():.0f} to {m.t2_pc1_vs_random.max():.0f} times the variance of a random matrix, "
        f"and the Hopkins statistic lies between {m.t2_hopkins.min():.2f} and {m.t2_hopkins.max():.2f}: the "
        "vectors clump, as a continuum rather than as separate islands.",
        f"**Separation.** Same-type aircraft sit closer than different-type ones in all 6 matrices "
        f"(p ≤ {max(0.001, m.t3_p.max()):.3f}), but by little: at layer 24 CLS the ratio is {mp.t3_ratio:.3f} "
        f"and d {mp.t3_d:.2f}. On average the types overlap heavily."])
    note("pat_f04_prediction.png", f"Source: {MODEL}; {n_(nP)} drawing aircraft, labeller's main figure; human "
                                   "G1 type, 12 classes; maker held out")
    with _into_brief():
        F = AD.DocFigs(FIGB, "pat")
        ER.fig_prediction(F, S_pat, Rp)
    D.figure(F.items["f04_prediction"]["path"], "the architecture read from the drawings, per layer and token "
                                                "(12 types); blue: the matrix carried forward.",
             f"**At layer 24 CLS the drawings recover the 12 types at {f2(mp.knn_bal_acc)} "
             f"{ci(mp.knn_ci_lo, mp.knn_ci_hi)} against a shuffled level of {f2(mp.knn_chance_p95)}, and the 5 "
             f"classes at {f2(m5.knn_bal_acc)} {ci(m5.knn_ci_lo, m5.knn_ci_hi)} against {f2(m5.knn_chance_p95)}: "
             f"the architecture is present, and weak.** A type is recovered for {f2(mp.knn_bal_acc)} of its "
             f"aircraft on average, fewer than one in four. The probe reaches {f2(mp.probe_bal_acc)}, well above "
             "the neighbours: the architecture is in the vector, but it does not decide which aircraft are "
             "nearest.", width=80)
    hlf = (len(cl) + 1) // 2
    rows = []
    for i in range(hlf):
        c1 = cl[i]
        c2 = cl[i + hlf] if i + hlf < len(cl) else None
        rows.append((disp(c1), int(cnt12[c1]), f"{rec[c1]:.2f}",
                     disp(c2) if c2 else "", int(cnt12[c2]) if c2 else "", f"{rec[c2]:.2f}" if c2 else ""))
    D.table(md_df(rows, ["type", "aircraft", "recall", "type ", "aircraft ", "recall "]),
            "Recall per type (layer 24 CLS, kNN-5, maker held out).",
            f"**Among the types with 20 or more aircraft, {disp(big[0])} ({float(rec[big[0]]):.2f}) and "
            f"{AD.G1_NAME[big[1]]} ({float(rec[big[1]]):.2f}) are read best; {', '.join(zero)} are never "
            "recovered.**"
            + (f" {', '.join(f'{disp(c)} scores {float(rec[c]):.2f}, but on {int(cnt12[c])} aircraft' for c in smallr)}."
               if smallr else ""))
    note("pat_confusion.png", f"Source: {MODEL}; layer 24 CLS; {n_(nP)} drawing aircraft, main figure; "
                              "12 types; maker held out")
    with _into_brief():
        fcf = AD.fig_confusion(S_pat, Rp, FIGB / "pat_confusion.png", (5.4, 4.3))
    D.figure(fcf, "which types are confused with which: rows are the true type (aircraft in brackets), columns "
                  "the type kNN-5 predicts; the framed diagonal is correct.",
             f"**{pc(into)} of the wrong predictions fall into {AD.G1_NAME[top2[0]]} or {disp(top2[1])}, the two "
             "largest types, which is what a neighbour vote does when a type is thin in the pool.**", width=64)

    # 3.2
    D.method(
        "3.2", "What organises the drawing embedding: the architecture, or the way the aircraft is drawn?",
        "Whether the types form clusters of their own, whether the view of the drawing moves the vectors more "
        "than the architecture, and whether the architecture survives when the view is removed.",
        f"The {n_(nP)} aircraft for clusters and scores; all {n_(nW)} whole-aircraft figures for the view, the "
        "unit at which a view exists; the view group of each figure (Perspective, Plan, Side, Front/Rear).",
        f"k-means with 12 clusters against the types (T5), with the map of Figure {D.fignum()}; T3's d for the "
        f"view and for the architecture on the figures (T6, Figure {D.fignum(2)}); then a linear erasure: the "
        "three directions that separate the four view means are fitted on the figures and projected out, and the "
        f"aircraft vectors rebuilt (Table {D.tabnum()}). A **variance-matched control** removes three random "
        "directions of the same variance, and the Perspective-only rows hold the view constant by design.",
        "The erasure is the orthogonal projection x ↦ (I − Q Qᵀ) x, then renormalised, with Q an orthonormal "
        "basis of the differences between the view means (the author's own implementation). A loss that the "
        "random removal also produces is not specific to the view. ARI and d as in the methods box.",
        "The erasure is linear only, and exact only on the figures it was fitted on; a non-linear view signal "
        "survives it.")
    note("pat_f07_umap.png", f"Source: {MODEL}; layer 24 CLS; {n_(nP)} drawing aircraft, main figure; UMAP "
                             "(15 neighbours, min. distance 0.1, cosine, seed 42); colour: parent class")
    with _into_brief():
        ER.fig_umap(F, S_pat, Rp)
    D.figure(F.items["f07_umap"]["path"], "the drawing embeddings in two dimensions, one point per aircraft, "
                                          "coloured by the parent class.",
             f"**The types form no clusters of their own: k-means with 12 clusters matches them with an ARI of "
             f"{mp.t5_ari:.3f} at layer 24 CLS, and no matrix exceeds {m.t5_ari.max():.3f}.** The classes overlap "
             "across most of the map.", width=66)
    note("pat_f09_confound.png", f"Source: {MODEL}; all {n_(nW)} whole-aircraft figures of the {n_(nP)} "
                                 "aircraft; view group from the labelling; same-maker pairs left out")
    with _into_brief():
        ER.fig_confound(F, S_pat, Rp)
    D.figure(F.items["f09_confound"]["path"], "how far the view and the architecture move the vectors, per "
                                              "matrix: d, in standard deviations.",
             f"**The view moves the vectors more than the architecture in "
             f"{'all 6' if (conf_p.conf_over_arch > 1).all() else int((conf_p.conf_over_arch > 1).sum())} "
             f"matrices: at layer 24 CLS the view's d is {cb.conf_d:.2f} against {cb.arch_d:.2f}, a ratio of "
             f"{cb.conf_over_arch:.2f}, and at the earlier layers the ratio reaches "
             f"{conf_p.conf_over_arch.max():.1f}.** The model encodes first how the aircraft is drawn.", width=84)

    def vrow(lbl, sec):
        o = ve(T, sec, "original", "bal_acc")
        e = ve(T, sec, "view erasure", "bal_acc")
        d = ve(T, sec, "view erasure - original", "paired_diff")
        v = ve(T, sec, "variance-matched random - original", "paired_diff")
        return (lbl, int(o.n), f"{o.value:.3f}", f"{e.value:.3f}", f"{sg(d.value)} {ci(d.ci_lo, d.ci_hi, 3, True)}",
                f"{sg(v.value)} {ci(v.ci_lo, v.ci_hi, 3, True)}")
    D.table(md_df([vrow("12 types", ga), vrow("5 classes", pa_),
                   vrow("12 types, Perspective main figures only", ga + ", Perspective main figures only"),
                   vrow("5 classes, Perspective only", pa_ + ", Perspective main figures only")],
                  ["label set", "aircraft", "original", "view erased", "paired Δ [95 % CI]", "variance-matched Δ"]),
            "The architecture before and after the view erasure (registers, layer 24 CLS, kNN-5, maker held out).",
            f"**Removing the view costs {abs(d12.value):.3f} on the 12 types and {abs(d5.value):.3f} on the 5 "
            "classes, but a random removal of the same variance costs about as much, and so does the erasure "
            "when every aircraft is drawn in Perspective: the loss is not specific to the view, and every erased "
            "score stays far above chance.**")
    D.bullets([
        f"**The linear view signal is gone, a non-linear one is not.** The linear view probe on the main figures "
        f"falls from {vp_o:.2f} to {vp_e:.2f} (chance 0.25); the view kNN falls only from {vk_o.value:.2f} to "
        f"{vk_e.value:.2f}, against its shuffled level of {vk_e.chance_p95:.2f}.",
        f"**The base model is unchanged by the same erasure** ({sg(b12.value)} {ci(b12.ci_lo, b12.ci_hi, 3, True)} "
        f"and {sg(b5.value)} {ci(b5.ci_lo, b5.ci_hi, 3, True)}, no difference detected), and after it the two "
        f"models score alike (12 types: registers {r12:.3f}, base {e12:.3f}; 5 classes: {r5:.3f} against "
        f"{e5:.3f}). Part of the registers advantage of 2.1 sits in the dominant directions the view also uses."])

    # 3.3
    D.method(
        "3.3", "Does the choice of drawing change the result?",
        "Whether the image rule of 1.2 changes the architecture score, and whether averaging several views adds "
        "architecture.",
        f"The {n_(nP)} aircraft under each of the four rules; for the averaging, the {e4['n']} aircraft whose "
        "exp4 vector averages two or three views, the only aircraft on which exp4 differs from one figure.",
        f"Table {D.tabnum()} gives the score under each rule (T7). On the {e4['n']} aircraft, the predictions of "
        "exp1, exp3 and exp4 are read from the full neighbour pool and compared.",
        f"Intervals as in the methods box; the exp4 comparison is a paired bootstrap of exp4 minus exp1 "
        f"({n_(P.N_BOOT)} resamples) on the same aircraft.",
        "The subset comparison was run after the rule scores had been seen (post hoc).")
    D.table(md_df([(r, f"{br.loc[r].knn_bal_acc:.3f}", ci(br.loc[r].ci_lo, br.loc[r].ci_hi)) for r in S_pat.rules],
                  ["image rule", "kNN-5", "95 % interval"]),
            "The architecture score under each image rule (registers, layer 24 CLS, 12 types, maker held out).",
            f"**The rules span {f2(br.knn_bal_acc.min())} to {f2(br.knn_bal_acc.max())} and the labeller's main "
            "figure scores highest; the rule does not change the conclusion.** On the "
            f"{e4['n']} aircraft with two or three views, exp4 is right for {pc(e4['acc']['exp4 view average'])} "
            f"against {pc(e4['acc']['exp1 view-first'])} for the view-first figure and "
            f"{pc(e4['acc']['exp3 main figure'])} for the main figure; exp4 minus view-first is "
            f"{sg(e4['diff'])} {ci(e4['lo'], e4['hi'], 3, True)}, {nd(e4['lo'], e4['hi'])}: averaging views "
            "adds no measurable architecture.")

    # 3.4
    D.method(
        "3.4", "Where does the model look on a drawing?",
        "Which parts of a drawing the CLS token of the chosen model reads: a qualitative check on what the "
        "scores measure.",
        f"Two drawing aircraft per parent class drawn at random (seed 42) from the main figures of the "
        f"{n_(nP)} aircraft; for the summary shares, the attention of every drawing and photo.",
        f"Figure {D.fignum()} overlays the CLS attention at layers 18 and 24 on each drawing, read by eye. The "
        "median share of the attention on the single largest patch, and the share the register tokens take, "
        "say whether any background patch dominates.",
        "The map is the CLS query against every patch key, averaged over the heads, each map scaled to its own "
        "99th percentile.",
        "Ten aircraft read by eye are an illustration, not a measurement.")
    note("f_attention_drawings.png", "Source: DINOv2-L with registers, CLS attention at layers 18 and 24, mean "
                                     "over heads; patent drawings, main figure, 2 aircraft per class drawn at random "
                                     "(seed 42); label under each pair: the G1 type")
    fa_d = fig_attention_source(C, "drawing")
    D.figure(fa_d, "CLS attention of the registers model on drawings, layers 18 and 24, two aircraft per parent "
                   "class; bright = attention.",
             "**Layer 18 traces the line-work almost everywhere, outline, propulsors and sometimes the figure "
             "label alike; layer 24 gathers in spots on the aircraft, often on propulsors, hubs and the fuselage, "
             "with weak spots on the white pad. No map settles on one part that names the architecture.** Across "
             f"all images the median share on the single largest patch stays at or below {att_max:.3f} at layers "
             f"18 and 24, and the register tokens take {reg_lo:.2f} to {reg_hi:.2f} of the attention, so no "
             "background patch dominates.", width=68)
    D.answer(
        f"Patent drawings carry the architecture, weakly: a frozen model recovers the 12 types at "
        f"{f2(mp.knn_bal_acc)} against a shuffled level of {f2(mp.knn_chance_p95)}, but the embedding is "
        "organised first by how an aircraft is drawn, not by what it is.",
        [f"**Present and weak.** 12 types at {f2(mp.knn_bal_acc)} {ci(mp.knn_ci_lo, mp.knn_ci_hi)}, 5 classes "
         f"at {f2(m5.knn_bal_acc)}; the probe reads {f2(mp.probe_bal_acc)}, so the architecture is in the vector "
         "without deciding which aircraft are nearest.",
         f"**The large types are read, the rare ones not.** {disp(big[0])} and {AD.G1_NAME[big[1]]} lead; "
         f"{', '.join(zero)} are never recovered, and most errors fall into the two largest types.",
         f"**The view comes first.** The view moves the vectors {f2(cb.conf_over_arch)} times more than the "
         f"architecture at layer 24 CLS, and the types form no clusters (ARI {mp.t5_ari:.3f}).",
         "**The architecture survives the view's removal.** The erasure costs a few hundredths, no more than a "
         "random removal of the same variance, and every erased score stays far above chance.",
         "**Neither the image rule nor the attention points to one decisive part.** The rules span "
         f"{f2(br.knn_bal_acc.min())} to {f2(br.knn_bal_acc.max())}; the late CLS token summarises the whole "
         "shape and the way it is drawn, not the arrangement of lift and thrust."])

    # ═══ 4 ═══
    D.chapter("4", "The photographs: do photographs of the aircraft carry the architecture, and does the scene "
                   "around the aircraft matter?",
              f"The same setup and protocol, on the {n_(nH)} directory aircraft and their 5 classes:",
              "- **4.1: how well the photographs predict the class.**\n"
              "- **4.2: what organises the photo embedding**: clusters, and the maturity of the aircraft.\n"
              "- **4.3: whether the background and the choice of image matter.**\n"
              "- **4.4: where the model looks on a photograph.**")
    mm = Rh["ev"]["main"]
    s1r = C["s1a_candidates"]
    lab_l22 = float(s1r[(s1r.pool == "registers only") & (s1r.label == "photos 5-class") &
                        (s1r.candidate == "registers L22 CLS")].selection_freq.iloc[0])
    D.method(
        "4.1", "How well do photographs predict the directory class?",
        "Whether the photo vectors carry the architecture, read the same way as the drawings in 3.1.",
        f"The {n_(nH)} photo aircraft, hero image, labelled with the directory class; all six matrices.",
        f"Soundness and separation first; Figure {D.fignum()} gives, per matrix, the kNN-5 balanced accuracy, "
        f"the shuffled level and the probe; Figure {D.fignum(2)} which classes are confused with which.",
        "As in the methods box; guessing scores 1/5.",
        f"The photo label is the directory's class, which agrees with the codebook parent for "
        f"{int(pch.agrees.sum())} of {len(pch)} linked aircraft. The classes range from {int(cnt5h.min())} "
        f"({cnt5h.idxmin()}) to {int(cnt5h.max())} ({cnt5h.idxmax()}) aircraft.")
    D.bullets([
        f"**Soundness and separation.** No invalid row, and the first principal component carries "
        f"{mm.t2_pc1_vs_random.min():.0f} to {mm.t2_pc1_vs_random.max():.0f} times the random variance. Same-class "
        f"photos sit closer in all 6 matrices, with a larger effect than on the drawings: ratio "
        f"{mh.t3_ratio:.3f} and d {mh.t3_d:.2f} at layer 24 CLS (drawings: d {mp.t3_d:.2f})."])
    note("pho_f04_prediction.png", f"Source: {MODEL}; {n_(nH)} evtol.news aircraft, hero image; directory class, "
                                   "5 classes; maker held out")
    with _into_brief():
        Fh = AD.DocFigs(FIGB, "pho")
        ER.fig_prediction(Fh, S_evn, Rh)
    D.figure(Fh.items["f04_prediction"]["path"], "the architecture read from the photographs, per layer and token "
                                                 "(5 classes); blue: the matrix carried forward.",
             f"**At layer 24 CLS the photos recover the 5 classes at {f2(mh.knn_bal_acc)} "
             f"{ci(mh.knn_ci_lo, mh.knn_ci_hi)} against a shuffled level of {f2(mh.knn_chance_p95)}, and the probe "
             f"at {f2(mh.probe_bal_acc)}.** Layer 22 CLS scores within that interval "
             f"({f2(AD.main_row(Rh, matrix='L22 CLS').knn_bal_acc)}), which is why the split-half rule chose it in "
             f"{pc(lab_l22)} of the photo choices.", width=80)
    note("pho_confusion.png", f"Source: {MODEL}; layer 24 CLS; {n_(nH)} evtol.news aircraft, hero image; "
                              "maker held out")
    with _into_brief():
        fpc = AD.fig_confusion(S_evn, Rh, FIGB / "pho_confusion.png", (3.9, 3.0))
    D.figure(fpc, "which directory classes are confused with which: rows are the true class (aircraft in "
                  "brackets), columns the predicted one.",
             f"**{ER.CLASS5_NAME[rbest]} is read best (recall {float(rh[rbest]):.2f}) and {ER.CLASS5_NAME[rworst]} "
             f"worst ({float(rh[rworst]):.2f}).**", width=50)

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
             f"**The classes form weak clusters: k-means matches them with an ARI of {mh.t5_ari:.3f} at layer 24 "
             f"CLS, {mh.t5_ari / mp.t5_ari:.0f} times the drawings' {mp.t5_ari:.3f}.**", width=66)
    D.p(f"At layer 24 CLS the maturity moves the vectors less than the class (d {ch.conf_d:.2f} against "
        f"{ch.arch_d:.2f}, a ratio of {ch.conf_over_arch:.2f}), but in {int((conf_h.conf_over_arch > 1).sum())} "
        "of the 6 matrices the maturity dominates: the order depends on the layer, and the matrix carried "
        "forward is one where the class leads.")

    # 4.3
    D.method(
        "4.3", "Do the background and the choice of image change the photo result?",
        "Whether the photo score measures the aircraft or the scenery around it (sky, city, hangar), and whether "
        "another image rule would change it.",
        f"The {n_(len(cm))} hero images; for the crop test the base model (DINOv2-large), because the test ran "
        "before the registers model was adopted: it is the one result of this document on base.",
        f"Each hero image is cropped to the aircraft with a zero-shot OWLv2 detector ({n_fb} of {n_(len(cm))}, "
        f"{pc(n_fb / len(cm), 1)}, kept whole because the box was weak or tiny), re-processed and re-embedded, "
        f"and compared with the uncropped image (Table {D.tabnum()}). The image rule is varied among the hero "
        "image, one random other image of the page and the mean of all kept images.",
        "Paired bootstrap difference on the same aircraft, as in the methods box.")
    D.table(md_df([(mx, f"{ck.loc[(mx, 'original')].knn_bal_acc:.3f}", f"{ck.loc[(mx, 'cropped')].knn_bal_acc:.3f}",
                    f"{sg(cd.loc[mx].diff_cropped_minus_original)} {ci(cd.loc[mx].ci_lo, cd.loc[mx].ci_hi, 3, True)}")
                   for mx in ("L24 CLS", "L24 patch mean")],
                  ["matrix", "original", "cropped", "paired difference [95 % CI]"]),
            f"Cropping the photos to the aircraft (base model, {n_(len(cm))} aircraft, kNN-5, maker held out).",
            f"**Cropping moves the layer 24 CLS score by {sg(cd.loc['L24 CLS'].diff_cropped_minus_original)}, "
            f"{nd(cd.loc['L24 CLS'].ci_lo, cd.loc['L24 CLS'].ci_hi)}: the photo result measures the aircraft, not "
            "the scenery.** Only the patch mean, which averages over background patches, gains "
            f"({sg(cd.loc['L24 patch mean'].diff_cropped_minus_original)}, interval "
            f"{nd(cd.loc['L24 patch mean'].ci_lo, cd.loc['L24 patch mean'].ci_hi)}).")
    D.p(f"The image rule matters as little as on the drawings: hero image {f2(brh.loc['hero image'].knn_bal_acc, 3)}, "
        f"one random other image {f2(brh.loc['random other image'].knn_bal_acc, 3)}, the mean of all kept images "
        f"{f2(brh.loc['mean of all images'].knn_bal_acc, 3)}, against an interval about "
        f"{mh.knn_ci_hi - mh.knn_ci_lo:.2f} wide for one rule.")

    # 4.4
    D.method(
        "4.4", "Where does the model look on a photograph?",
        "The same qualitative check as 3.4, on photographs, where the aircraft sits in a scene.",
        f"Two photo aircraft per class drawn at random (seed 42) from the {n_(nH)} hero images.",
        f"Figure {D.fignum()} overlays the CLS attention at layers 18 and 24 on each photograph, read by eye.",
        "As in 3.4.",
        "Ten aircraft read by eye are an illustration, not a measurement.")
    note("f_attention_photos.png", "Source: DINOv2-L with registers, CLS attention at layers 18 and 24, mean over "
                                   "heads; evtol.news hero images, 2 aircraft per class drawn at random (seed 42). "
                                   f"{CREDIT}")
    fa_p = fig_attention_source(C, "photo")
    D.figure(fa_p, f"CLS attention of the registers model on photographs, layers 18 and 24, two aircraft per "
                   f"class; bright = attention. {CREDIT}.",
             "**Layer 18 covers the airframe; layer 24 gathers in spots on the aircraft, with weak spots in the "
             "sky. In one wingless photo the people in front of the aircraft draw as much attention as the "
             "aircraft itself: a kept hero image can still be a scene.**", width=68)
    D.answer(
        f"Photographs carry the architecture moderately: the 5 classes at {f2(mh.knn_bal_acc)} against a "
        f"shuffled level of {f2(mh.knn_chance_p95)}, a result that depends neither on the background, nor on the "
        "colour, nor on the image rule.",
        [f"**Moderate.** kNN-5 {f2(mh.knn_bal_acc)} {ci(mh.knn_ci_lo, mh.knn_ci_hi)}, probe {f2(mh.probe_bal_acc)}; "
         f"{ER.CLASS5_NAME[rbest]} is read best, {ER.CLASS5_NAME[rworst]} worst.",
         f"**Weak clusters, stronger than on drawings.** ARI {mh.t5_ari:.3f} against {mp.t5_ari:.3f}; separation "
         f"d {mh.t3_d:.2f} against {mp.t3_d:.2f}.",
         "**The class leads the maturity only in the matrix carried forward**; in "
         f"{int((conf_h.conf_over_arch > 1).sum())} of 6 matrices whether the aircraft was built moves the "
         "vectors more.",
         f"**The aircraft, not the scene.** Cropping to the aircraft changes nothing detectable "
         f"({sg(cd.loc['L24 CLS'].diff_cropped_minus_original)}), and the image rule moves the score by a few "
         "hundredths."])

    # ═══ 5 ═══
    D.chapter("5", "Drawings against photographs: which source carries more of the architecture, and can one "
                   "aircraft be found in both?",
              "The two sources are compared with the same model, readout and protocol, on the 5 classes they "
              "share (each drawing type folds into its parent class): 5.1 compares the scores, 5.2 asks whether "
              "the model links a drawing to a photograph of the same aircraft.")
    D.method(
        "5.1", "Do photographs carry more architecture than drawings on the classes they share?",
        "Whether the medium, drawing or photograph, carries more of the architecture on the shared 5 classes.",
        f"The {n_(nP)} drawing aircraft with their type folded to its parent, and the {n_(nH)} photo aircraft; "
        "the two sets hold different aircraft.",
        f"Table {D.tabnum()} compares the sources with the photos in colour and in greyscale, and against "
        f"{cmx['draws']} random photo subsets drawn with exactly the drawings' class counts ({cmx['n']} aircraft "
        f"each; the probe on the first {cmx['probe_draws']}), so that pool size and class mix match. Figure "
        f"{D.fignum()} gives the balanced accuracy and the recall per class of the three.",
        "The interval of a difference comes from independent bootstrap resamples of each set (methods box); the "
        "range of the matched photos is the 2.5th to 97.5th percentile of the draws. A neighbour vote depends on "
        "its pool: a class that is thin in the pool is hard to recover whatever the medium, and balanced "
        "accuracy weights every class equally but cannot make a thin class easy; the matched subsets remove "
        "that difference.",
        "The matching was designed after the full-set comparison had been seen (post hoc).")
    fr = [("photos in colour, all", f"{fair.loc['colour'].drawings_bal_acc:.3f}", f"{fair.loc['colour'].photos_bal_acc:.3f}",
           f"difference {sg(fair.loc['colour'].diff_photos_minus_drawings)} "
           f"{ci(fair.loc['colour'].ci_lo, fair.loc['colour'].ci_hi, 3, True)}"),
          ("photos in greyscale, all", f"{fair.loc['grey'].drawings_bal_acc:.3f}", f"{fair.loc['grey'].photos_bal_acc:.3f}",
           f"difference {sg(fair.loc['grey'].diff_photos_minus_drawings)} "
           f"{ci(fair.loc['grey'].ci_lo, fair.loc['grey'].ci_hi, 3, True)}"),
          ("photos with the drawings' class mix", f"{dA:.3f}", f"{cmx['mean']:.3f}",
           f"draws {cmx['lo']:.3f} to {cmx['hi']:.3f}"),
          ("probe, photos all", f"{m5.probe_bal_acc:.3f}", f"{mh.probe_bal_acc:.3f}", "n/a"),
          ("probe, photos with the drawings' class mix", f"{m5.probe_bal_acc:.3f}", f"{cmx['probe_mean']:.3f}",
           f"draws {cmx['probe_lo']:.3f} to {cmx['probe_hi']:.3f}")]
    D.table(md_df(fr, ["comparison", "drawings", "photos", "interval"]),
            f"Photographs against drawings on the shared 5 classes (registers, layer 24 CLS, kNN-5 unless "
            f"marked probe; {n_(nP)} drawing aircraft, {n_(nH)} photo aircraft).",
            (f"**On all photos the photos lead by {sg(pA - dA, 2)}, in greyscale too; with the drawings' size and "
             f"class mix they score {cmx['mean']:.3f}, and the drawings' {dA:.3f} lies inside the range of the "
             "draws: no difference detected, and most of the photo lead comes from the larger, more balanced "
             "pool.**" if cmx_inside else
             f"**The photos lead by {sg(pA - dA, 2)}, and still lead with the drawings' class mix "
             f"({cmx['mean']:.3f}, the drawings' {dA:.3f} outside the range of the draws).**")
            + (f" The probe tells the same story ({cmx['probe_mean']:.3f} matched against {m5.probe_bal_acc:.3f})."
               if pin == cmx_inside else " The probe disagrees."), right=["drawings", "photos"])
    vt_up = cmx["recall"]["VT"] > fci.loc["VT", "rec_photo"]
    wm_down = cmx["recall"]["WM"] < fci.loc["WM", "rec_photo"]
    note("f_five_class.png", f"Source: {MODEL}; layer 24 CLS; kNN-5, maker held out; drawings: {n_(nP)} aircraft, "
                             f"type folded to its parent; photos: {n_(nH)} hero images; matched: {cmx['draws']} "
                             "photo subsets with the drawings' class counts")
    with _into_brief():
        f16 = AD.fig_five_class(C, fc, cmx)
    D.figure(f16, "the shared 5 classes, drawings against photographs: balanced accuracy and recall per class; "
                  "under each class, the aircraft in drawings | photos; dashed line: guessing, 1/5.",
             f"**" + ("A class is read well largely when it dominates the pool. " if vt_up and wm_down else "")
             + f"Wingless multicopters, {fci.loc['WM', 'n_draw']} drawings but {fci.loc['WM', 'n_photo']} photos, "
             f"are read at {fci.loc['WM', 'rec_draw']:.2f} from drawings and {fci.loc['WM', 'rec_photo']:.2f} from "
             f"photos, and at {cmx['recall']['WM']:.2f} once the photos have the drawings' mix; Vectored Thrust, "
             f"{fci.loc['VT', 'n_draw']} of {n_(nP)} drawings, goes the other way ({fci.loc['VT', 'rec_draw']:.2f} "
             f"drawings, {fci.loc['VT', 'rec_photo']:.2f} photos, {cmx['recall']['VT']:.2f} matched).**", width=88)

    # 5.2
    D.method(
        "5.2", "Can the same aircraft be found in both sources?",
        "Whether the model links a patent drawing to a photograph of the same aircraft, and the reverse.",
        f"The {int(mpp.queries)} patent aircraft that have a directory page, as queries against the "
        f"{n_(int(mpp.gallery))} photo aircraft whose hero image is not a line drawing, and those photo aircraft "
        f"against the {n_(int(mph.gallery))} patent aircraft.",
        f"Each query ranks the gallery by cosine distance. Table {D.tabnum()} gives the share of queries whose "
        f"own page is ranked first, in the top 10 and in the top 50, and the median rank; Figure {D.fignum()} "
        "the share within the top k for every k, beside a random ranking. Line drawings are removed from the "
        "gallery because some directory pages reproduce the patent figure itself.",
        "Random ranking: P(own page in the top k) = 1 − Π_{i<k} (G − n₊ − i) / (G − i), averaged over the "
        "queries, with G the gallery size and n₊ the query's own pages (the author's own baseline).",
        f"{int(mpp.queries)} queries, all aircraft with a public product.")
    D.table(md_df([("patent → photo", int(mpp.queries), n_(int(mpp.gallery)), pc(mpp['R@1'], 1), pc(mpp['R@10'], 1),
                    pc(mpp['R@50'], 1), f"{pc(mpp['random_R@10'], 1)} / {pc(mpp['random_R@50'], 1)}", f"{mpp.median_rank:g}"),
                   ("photo → patent", int(mph.queries), n_(int(mph.gallery)), pc(mph['R@1'], 1), pc(mph['R@10'], 1),
                    pc(mph['R@50'], 1), f"{pc(mph['random_R@10'], 1)} / {pc(mph['random_R@50'], 1)}", f"{mph.median_rank:g}")],
                  ["direction", "queries", "gallery", "top 1", "top 10", "top 50", "random top 10 / 50", "median rank"]),
            "Same-aircraft matching (registers, layer 24 CLS, no line drawings in the gallery).",
            f"**The right page is in the top 10 for {pc(mpp['R@10'])} of the patent queries and in the top 50 for "
            f"{pc(mpp['R@50'])}, against {pc(mpp['random_R@10'], 1)} and {pc(mpp['random_R@50'], 1)} at random.**")
    note("f_matching.png", f"Source: {MODEL}; layer 24 CLS; {int(mpp.queries)} patent main figures as queries "
                           f"against {n_(int(mpp.gallery))} evtol.news aircraft without line drawings")
    with _into_brief():
        f17, _mr = AD.fig_matching(C)
    D.figure(f17, "how often a patent aircraft finds its own directory page within the top k photographs, against "
                  "a random ranking.",
             f"**The link is well above chance at every shortlist size but far from a retrieval tool: the median "
             f"rank is {mpp.median_rank:g}, and the medium separates the sources more than the aircraft joins "
             "them.**", width=78)
    D.answer(
        ("The photographs score higher only because their pool is larger and more balanced: with the drawings' "
         "size and class mix they score as the drawings do, and one aircraft is linked across the sources above "
         "chance but not reliably." if cmx_inside else
         "The photographs carry more architecture than the drawings, also with the drawings' class mix; one "
         "aircraft is linked across the sources above chance but not reliably."),
        [f"**The raw lead.** Photos {pA:.3f} against drawings {dA:.3f} ({sg(pA - dA, 2)}), the same in greyscale: "
         "whatever the lead is, it is not colour.",
         f"**Matched.** With the drawings' class counts the photos score {cmx['mean']:.3f} "
         f"({cmx['lo']:.3f} to {cmx['hi']:.3f} over the draws); the probe agrees "
         f"({cmx['probe_mean']:.3f} against {m5.probe_bal_acc:.3f}). The comparison was designed post hoc.",
         f"**A class is read well when it dominates the pool**, whatever the medium: Vectored Thrust in the "
         f"drawings, Wingless multicopters in the photos.",
         f"**Matching.** {pc(mpp['R@50'])} of patent aircraft find their own page in the top 50 (random "
         f"{pc(mpp['random_R@50'], 1)}); median rank {mpp.median_rank:g}."])

    # ═══ closing ═══
    D.closing("The answer to RQ4, what it rests on, and what is still open")
    mix_clause = (", an advantage that shrinks to nothing detectable once the photo pool has the drawings' size and "
                  "class mix") if cmx_inside else ""
    D.answer(
        f"The figure alone carries the architecture: weakly in patent drawings, and moderately in photographs as "
        f"they come{mix_clause}.",
        [f"**Drawings.** A frozen general-purpose model, never trained on aircraft or on the codebook, recovers "
         f"the 12 types from one drawing per aircraft at {f2(mp.knn_bal_acc)} against a shuffled level of "
         f"{f2(mp.knn_chance_p95)}, and the 5 classes at {f2(m5.knn_bal_acc)}; photographs reach "
         f"{f2(mh.knn_bal_acc)} on the 5 classes.",
         "**It holds** with the maker held out, under every image rule, without the photo backgrounds, in "
         "greyscale, and after the view of the drawing is linearly removed.",
         f"**The drawing embedding is organised first by how the aircraft is drawn**, only second by what it is: "
         f"the view separates the vectors {f2(cb.conf_over_arch)} times more than the architecture, and the "
         "types form no clusters of their own.",
         "**The human reading stays the reference.** The architecture is in the vector (the probe reads more "
         "than the neighbours) but does not decide which aircraft are nearest; the embedding offers a second, "
         "independent and label-free view of the same aircraft."], head="Answer to RQ4")
    D.p("**What DINOv2 contributes to the thesis.**")
    D.bullets([
        f"**A test of the taxonomy by an observer that has never seen it.** The types it recovers best are the "
        f"large ones; the types it never recovers ({', '.join(zero)}) are defined by how the propulsion moves or "
        "acts, which a single silhouette need not show.",
        "**A link from the patent record to the directory** of built and proposed aircraft, above chance.",
        "**A labelled benchmark of whole-aircraft drawings with a fixed protocol**, on which a fine-tuned or "
        "aircraft-specific model can be measured."])
    D.p("**A reusable recipe for patent-drawing studies.**")
    D.add("1. Use a registers checkpoint: drawings are mostly blank background, and a model without registers "
          "parks its artifact tokens there.",
          "2. Feed the largest native input size (518 px) on a white pad, and leave the lines as drawn.",
          "3. Choose the layer and token with the split-half rule over makers, and report the optimism gap beside "
          "the score.",
          "4. Hold the maker out of every neighbour, and test the view as a confound before reading any class "
          "result.", "")
    D.p("**What the results rest on.**")
    D.bullets([
        f"**A linear view erasure only.** The view kNN stays at {vk_e.value:.2f} after it, so a non-linear view "
        "component survives; the Perspective-only rows hold the view constant, but for one view only.",
        f"**Small and unbalanced classes.** "
        f"{', '.join(f'{disp(c)} {int(cnt12[c])}' for c in sorted(small, key=lambda c: cnt12[c]))} aircraft: these "
        f"recalls rest on a handful of aircraft. The photo classes range from {int(cnt5h.min())} "
        f"({cnt5h.idxmin()}) to {int(cnt5h.max())} ({cnt5h.idxmax()}), and the two sources differ in class mix.",
        f"**A photo filter audit that is not blind.** The sheets showed the filter score, the judge was the "
        f"pipeline's AI assistant, and the author has not yet re-checked the {len(au)} verdicts; the {n_rev} "
        "review-band images were never decided by hand.",
        "**An untested gentler thickening.** The smallest candidate grew each side of a stroke by about 1 px; a "
        "sub-pixel variant was not in the candidate set fixed in advance.",
        f"**The labels.** The photo label is the directory's own class ({int(pch.agrees.sum())} of {len(pch)} "
        f"agree with the codebook parent). The drawing labels are human labels, checked against the whole-patent "
        f"reading (κ {kap:.2f}); no second-labeller agreement is recorded.",
        "**Post-hoc steps.** The registers checkpoint was brought in after the base model's attention collapse "
        "was seen, the class-mix matching after the photo lead was seen, the exp4 subset after the rule scores; "
        "the crop test ran on the base model.",
        "**One frozen model.** The analysis measures what a general-purpose model sees without training; a "
        "fine-tuned model, or one trained on technical drawings, is outside its scope."])
    D.p("**Questions still open.** Each is a choice for the chapter, and each can be settled with the results "
        "already held.")
    D.bullets([
        f"**Does same-aircraft matching earn its own section?** It is above chance ({pc(mpp['R@50'])} in the top "
        "50) but far from reliable; it could shrink to one paragraph.",
        f"**Which drawing score leads?** {f2(mp.knn_bal_acc, 3)} as measured, with the erasure as a "
        f"qualification, or {f2(r12, 3)} after the view erasure, where the two models tie "
        f"({f2(mp.knn_bal_acc - r12, 3)} of the score does not survive it).",
        "**Does the exp4 question stay?** Averaging views is a null; it may leave the chapter.",
        "**One attention layer or two?** The plan named layer 18; with registers layer 24 is readable too, and "
        "Figures of 3.4 and 4.4 show both.",
        (f"**Which photo comparison is the headline?** The full set ({sg(pA - dA, 2)}) with the matching as its "
         f"qualification, or the matched comparison ({cmx['mean']:.3f} against {dA:.3f})?" if cmx_inside else
         "**Is the photo lead the headline?** It survives the class-mix matching."),
        "**One labeller or several?** Whether the drawing labels come from one labeller, and whether a "
        "second-labeller check is planned."])
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
        "Hubert, L. and Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.",
        "McInnes, L., Healy, J. and Melville, J. (2018). UMAP: Uniform manifold approximation and projection for "
        "dimension reduction. arXiv:1802.03426.",
        "Oquab, M., Darcet, T., Moutakanni, T. et al. (2023). DINOv2: Learning robust visual features without "
        "supervision. arXiv:2304.07193.")

    # ═══ appendix ═══
    D.add('<div class="keep chapter-start" markdown="1">', "", "## Appendix A — Every layer and token under "
          "both models", "", "</div>", "")
    D.p("Every matrix under both models, maker held out: kNN-5 balanced accuracy with the registers model's "
        "interval and shuffled level, and the linear probe. Base appears here only as the comparison; every "
        "chapter uses the registers model. Rotatable 3-D maps of both embeddings accompany the document as "
        "separate files.")

    def grid(Rb, Rr, key, caption):
        mb, mr = Rb[key]["main"].set_index("matrix"), Rr[key]["main"].set_index("matrix")
        rows = [(mx, f"{mb.loc[mx].knn_bal_acc:.3f}", f"{mr.loc[mx].knn_bal_acc:.3f}",
                 ci(mr.loc[mx].knn_ci_lo, mr.loc[mx].knn_ci_hi), f"{mr.loc[mx].knn_chance_p95:.3f}",
                 f"{mb.loc[mx].probe_bal_acc:.3f}", f"{mr.loc[mx].probe_bal_acc:.3f}") for mx in mr.index]
        up = int((mr.knn_bal_acc > mb.knn_bal_acc.reindex(mr.index)).sum())
        top = mr.knn_bal_acc.idxmax()
        D.table(md_df(rows, ["matrix", "kNN-5 base", "kNN-5 registers", "95 % interval", "shuffled level",
                             "probe base", "probe registers"]), caption,
                f"**Registers score higher than base in {up} of the 6 matrices, and {top} is the best matrix under "
                "registers.**")
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
