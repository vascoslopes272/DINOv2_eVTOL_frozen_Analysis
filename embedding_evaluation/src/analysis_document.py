"""The DINOv2 analysis document: one ~30-page Word file, from the setup decisions to the conclusions.

Author's approved outline of 2026-09-30 (title "Visual Learning: DINOv2 embeddings of patent drawings
and photographs", sections 0 to 10 and an appendix). Every number is read from saved results, never
typed:

    registers twins   1639_LABELLED/3_embedding_evaluation/embedding_protocol_registers/_results.pkl
                      EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis_registers/tables/_results.pkl
    base (only §2 and the appendix)   the same files of the base reports
    setup selection   1639_LABELLED/3_embedding_evaluation/setup_selection/*.csv
    registers check   .../registers_check/*.csv, EVTOLNEWS_DS/2_embedding_extraction/attention/
    view erasure      .../view_erasure/view_erasure.csv
    crop test         EVTOLNEWS_DS/3_embedding_evaluation/crop_experiment/*.csv (base model)
    matching          .../embedding_analysis_registers/tables/matching*.csv

Three small computations are new and run here from the saved embeddings (L24 CLS, registers):
the exp4-slot comparison on the aircraft with two or three view slots, the class-mix-matched
photo score of §6, and the random-ranking curve of the matching figure.

Printed class names follow the display-rename layer (labeling_evaluation/src/dataset_facts/
display_names.py): the class TR prints as TP (Tilt Propulsor); the data keep TR. Only that class
rule is applied here, never the whole text rename (it would also rewrite "level" and "rotor").

Outputs: docs/embedding_evaluation/analysis_document/ANALYSIS.md + figs/ (charts and patent
drawings); figures showing evtol.news photos go to EVTOLNEWS_DS/3_embedding_evaluation/
analysis_document_figs/ and are referenced by absolute path (private, © owners).
Run: scripts/build_analysis_document.py (Finetune env; the Word step runs under the base env).
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from . import embedding_protocol as P  # noqa: E402
from . import embedding_reports as ER  # noqa: E402

REG_TAG, BASE_TAG = "dinov2-reg-large_518", "dinov2-large_518"
PAT, EVN, REPO, DATA = ER.PAT, ER.EVN, ER.REPO, ER.DATA
PE, EE = PAT / "3_embedding_evaluation", EVN / "3_embedding_evaluation"
T_PAT_REG, T_PAT_BASE = PE / "embedding_protocol_registers", PE / "embedding_protocol"
T_EVN_REG, T_EVN_BASE = EE / "embedding_analysis_registers/tables", EE / "embedding_analysis/tables"
SEL, REGCHK, VE, CROP = PE / "setup_selection", PE / "registers_check", PE / "view_erasure", EE / "crop_experiment"
ATT = EVN / "2_embedding_extraction/attention"
OUT_DIR = REPO / "docs/embedding_evaluation/analysis_document"
FIG_DIR = OUT_DIR / "figs"
PRIV_DIR = EE / "analysis_document_figs"
MD_PATH = OUT_DIR / "ANALYSIS.md"
PHOTO_CREDIT = "Photo: evtol.news directory, © owners."
PAGES_3D = [("Patent drawings, registers model", PE / "umap_3d/PATENT_UMAP_3D_REGISTERS.html"),
            ("evtol.news photos with the patents in one joint map", EE / "advisor_report/EVTOLNEWS_UMAP_3D.html")]

KEY = (24, "cls")
DISPLAY = {"TR": "TP"}                         # display_names.py: TR prints as TP, Tilt Propulsor
G1_NAME = {"TR": "Tilt Propulsor", "CVT": "Combined vectored thrust", "TW": "Tilt Wing", "TB": "Tilt Body",
           "PTC": "Pitch-to-Cruise", "DS": "Deflected Slipstream", "SLC": "Lift + Cruise",
           "SRW": "Stopped/Slowed Rotor Wing", "MR": "Multirotor", "RC": "Rotorcraft", "HB": "Hoverbike",
           "PFV": "Personal Flying Vehicle"}
CONVERTIBLE = ["TR", "CVT", "TW", "DS", "SRW"]
INK, MUTED, FAINT, ACCENT = ER.INK, ER.MUTED, ER.FAINT, ER.ACCENT


def disp(c: str) -> str:
    return DISPLAY.get(c, c)


# ── number formatting ────────────────────────────────────────────────────────
def f2(v: float, d: int = 2) -> str:
    return f"{v:.{d}f}".replace("-", "−")


def sg(v: float, d: int = 3) -> str:
    """Signed difference: +0.017, −0.043."""
    return ("+" if v >= 0 else "−") + f"{abs(v):.{d}f}"


def ci(lo: float, hi: float, d: int = 2, signed: bool = False) -> str:
    fmt = (lambda v: sg(v, d)) if signed else (lambda v: f2(v, d))
    return f"[{fmt(lo)}, {fmt(hi)}]"


def pc(v: float, d: int = 0) -> str:
    return f"{100 * v:.{d}f} %"


def n_(x) -> str:
    return f"{int(x):,}".replace(",", " ")


# ── the document ─────────────────────────────────────────────────────────────
class Doc:
    """Markdown with numbered tables and figures; each figure carries a Source and a How-to-read line."""

    def __init__(self):
        self.L: List[str] = []
        self.nfig = 0
        self.ntab = 0
        self.fig_log: List[Tuple[int, str, str]] = []   # (number, path, origin)

    def p(self, *paras: str) -> None:
        for s in paras:
            self.L += [s, ""]

    def h1(self, s: str) -> None:
        self.L += ["<<SECTION>>", f"# {s}", ""]

    def h2(self, s: str) -> None:
        self.L += [f"## {s}", ""]

    def table(self, df: pd.DataFrame, caption: str, right: List[str] | None = None) -> None:
        self.ntab += 1
        cols = [str(c) for c in df.columns]
        rows = [[str(v) for v in r] for r in df.itertuples(index=False)]
        right = set(right if right is not None else
                    [c for c in cols[1:] if all(re.fullmatch(r"[−+\-\d.,% \[\]]+|n/a|—?", r[cols.index(c)] or "")
                                                 for r in rows)])
        width = [max(9, min(36, max([len(c)] + [len(r[i]) for r in rows]))) for i, c in enumerate(cols)]
        sep = ["-" * (w - 1) + ":" if c in right else "-" * w for c, w in zip(cols, width)]
        self.L += ['::: {custom-style="Table Title"}', f"**Table {self.ntab}.** {caption}", ":::", "",
                   "| " + " | ".join(cols) + " |", "|" + "|".join(sep) + "|"]
        self.L += ["| " + " | ".join(r) + " |" for r in rows] + [""]

    def figure(self, path: Path, title: str, source: str, read: str, width: int = 100,
               origin: str = "reused", private: bool = False) -> None:
        self.nfig += 1
        path = Path(path)
        try:
            ref = str(path.relative_to(OUT_DIR))
        except ValueError:
            ref = f"<{path}>"
        cap = f"Figure {self.nfig}. {title}".replace("[", "(").replace("]", ")")
        src = f"Source: {source}" + (f" {PHOTO_CREDIT}" if private else "")
        self.L += [f"![{cap}]({ref}){{width={width}%}}", "",
                   '::: {custom-style="Figure Source"}', src, "", f"How to read: {read}", ":::", ""]
        self.fig_log.append((self.nfig, str(path), origin))

    def text(self) -> str:
        return "\n".join(self.L)


def md_df(rows: List[Tuple], cols: List[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=cols)


# ── loading ──────────────────────────────────────────────────────────────────
def _pkl(p: Path) -> Dict:
    return pickle.loads((p / "_results.pkl").read_bytes())


def load() -> Dict:
    ER.TAG = REG_TAG                                   # both loaders and every model name read this
    S_pat, S_evn = ER.load_patents(), ER.load_evtolnews()
    S_pat.table_dir, S_evn.table_dir = T_PAT_REG, T_EVN_REG     # cached UMAP coordinates of the twins
    C = {"S_pat": S_pat, "S_evn": S_evn,
         "R_pat": _pkl(T_PAT_REG), "R_evn": _pkl(T_EVN_REG),
         "B_pat": _pkl(T_PAT_BASE), "B_evn": _pkl(T_EVN_BASE)}
    for name in ["s1a_candidates", "s1a_decision", "s1a_labels", "s1a_half_scores", "s1b_candidates",
                 "s1b_decision", "s1b_labels", "s2s3_knn", "s2s3_paired", "fairness", "fairness_class_counts"]:
        C[name] = pd.read_csv(SEL / f"{name}.csv")
    C["reg_knn"] = pd.read_csv(REGCHK / "registers_knn.csv")
    C["reg_paired"] = pd.read_csv(REGCHK / "registers_paired.csv")
    C["share"] = pd.read_csv(ATT / "registers_share.csv")
    C["ve"] = pd.read_csv(VE / "view_erasure.csv")
    C["crop_knn"] = pd.read_csv(CROP / "knn_original_vs_cropped.csv")
    C["crop_diff"] = pd.read_csv(CROP / "knn_paired_difference.csv")
    C["crop_man"] = pd.read_csv(EVN / "2_embedding_extraction/crop_experiment/crop_manifest.csv")
    C["match"] = pd.read_csv(T_EVN_REG / "matching.csv")
    C["match_ranks"] = pd.read_csv(T_EVN_REG / "matching_ranks.csv")
    C["match_base"] = pd.read_csv(EE / "matching.csv")
    C["funnel"] = pd.read_csv(PAT / "2_embedding_extraction/selection/funnel.csv", keep_default_na=False)
    C["img_dec"] = pd.read_csv(EVN / "1_filter/image_decisions.csv", keep_default_na=False)
    C["audit"] = pd.read_csv(EVN / "1_filter/audit_2026-09-22/audit_sample.csv", keep_default_na=False)
    C["parent_check"] = pd.read_csv(EE / "parent_check.csv", keep_default_na=False)
    C["pages"] = pd.read_csv(EVN / "0_source/aircraft.csv", keep_default_na=False)
    return C


def ve_get(C: Dict, tag: str, section: str, variant: str, measure: str) -> pd.Series:
    v = C["ve"]
    r = v[(v.tag == tag) & (v.section == section) & (v.variant == variant) & (v.measure == measure)]
    if len(r) != 1:
        raise KeyError((tag, section, variant, measure, len(r)))
    return r.iloc[0]


def main_row(R: Dict, key: str = "ev", matrix: str = "L24 CLS") -> pd.Series:
    m = R[key]["main"]
    return m[m.matrix == matrix].iloc[0]


# ── figures ──────────────────────────────────────────────────────────────────
def _check_codes(fig) -> None:
    """No printed TR: the display layer prints the class as TP."""
    fig.canvas.draw()
    for t in fig.findobj(Text):
        if re.search(r"\bTR\b", t.get_text()):
            raise ValueError(f"figure prints 'TR': {t.get_text()!r}")


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_codes(fig)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    return path


class DocFigs(ER.Figs):
    """ER.Figs without the text baked into the image (the document prints Source and How to read)."""

    def __init__(self, out: Path, prefix: str):
        super().__init__(out)
        self.prefix = prefix

    def save(self, fig, key, title, source, read=""):
        fig.tight_layout()
        path = _save(fig, self.dir / f"{self.prefix}_{key}.png")
        self.items[key] = {"title": title, "source": source, "read": read, "path": path}


def fig_funnels(C: Dict) -> Path:
    fun, im = C["funnel"], C["img_dec"]
    S = C["S_pat"]
    ft = S.extra["figure_table"]
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(S.aircraft.aircraft_id))]
    pat = [(r.step if r.step == "figures on file" else f"minus {r.step}", int(r.remaining))
           for r in fun.itertuples() if r.removed or r.step == "figures on file"]
    pat.append(("with a G1 type (analysis set)", len(wv)))
    kept = im[im.keep.astype(str) == "True"]
    pho = [("images on the pages", len(im)), ("after the hard rules", int((im.hard_rule == "").sum())),
           ("after the SigLIP drop (p < 0.35)", int(((im.hard_rule == "") & (im.auto != "drop")).sum())),
           ("kept (review band held back)", len(kept)), ("hero images = aircraft analysed", len(C["S_evn"].aircraft))]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.1), gridspec_kw={"width_ratios": [1.15, 1]})
    for ax, steps, unit, title in [(axes[0], pat, "figures remaining", "Patent drawings"),
                                   (axes[1], pho, "images remaining", "evtol.news photos")]:
        labels, vals = zip(*steps)
        y = np.arange(len(steps))[::-1]
        ax.barh(y, vals, color=FAINT, height=0.62)
        ax.barh(y[-1:], vals[-1:], color=ACCENT, height=0.62)
        for yi, v in zip(y, vals):
            ax.text(v, yi, "  " + n_(v), va="center", fontsize=7, color=INK)
        ax.set_yticks(y, [l.replace("duplicate patent (D1/D2, points at its original)", "duplicate patent (D1/D2)")
                          for l in labels], fontsize=7)
        ax.set_xlabel(unit)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, FIG_DIR / "f_funnels.png")


def _thumb(ax, img, title: str = "", sub: str = "") -> None:
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(FAINT)
    if title:
        ax.set_title(title, fontsize=8.5, loc="left")
    if sub:
        ax.set_xlabel(sub, fontsize=5.8, color=MUTED)


def fig_examples_patents(C: Dict) -> Path:
    S = C["S_pat"]
    ex3 = S.extra["picks"]["exp3 main figure"].set_index("aircraft_id")
    a = S.aircraft.set_index("aircraft_id")
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 4.1))
    for ax, g1 in zip(axes.ravel(), S.classes):
        ids = [i for i in a.index[a.label == g1] if ex3.loc[i, "view4"] == "Perspective"] or list(a.index[a.label == g1])
        i = sorted(ids)[0]
        f = ex3.loc[i, "fig_id"]
        img = Image.open(S.extra["processed"].get(f, ex3.loc[i, "image_path"])).convert("RGB")
        _thumb(ax, img, f"{disp(g1)} ({ER.PARENT[g1]})", i)
    fig.tight_layout()
    return _save(fig, FIG_DIR / "f_examples_patents.png")


def fig_selection(C: Dict) -> Path:
    c = C["s1a_candidates"]
    c = c[c.pool == "registers only"]
    order = [P.mname(k) for k in P.MATRICES]
    labels = [("patents G1 12-class", "Drawings, 12 G1 classes"), ("patents parent 5-class", "Drawings, 5 parent classes"),
              ("photos 5-class", "Photos, 5 directory classes")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.5), sharey=True)
    for ax, (lab, title) in zip(axes, labels):
        d = c[c.label == lab].assign(m=lambda x: x.candidate.str.replace("registers ", "")).set_index("m").reindex(order)
        y = np.arange(len(order))[::-1]
        col = [ACCENT if m == "L24 CLS" else FAINT for m in order]
        ax.barh(y, d.selection_freq * 100, color=col, height=0.62)
        for yi, v in zip(y, d.selection_freq * 100):
            ax.text(v + 1.5, yi, f"{v:.0f} %", va="center", fontsize=7, color=INK)
        ax.set_yticks(y, order, fontsize=7.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of 400 held-out selections")
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return _save(fig, FIG_DIR / "f_selection.png")


def _ink_window(img: Image.Image, win: int = 128, step: int = 8) -> tuple:
    """Top-left corner of the ``win`` square with the most dark pixels (same rule as the setup montage)."""
    a = (np.asarray(img.convert("L")) < 128).astype(np.int32)
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, xy = -1, (0, 0)
    for y in range(0, a.shape[0] - win + 1, step):
        for x in range(0, a.shape[1] - win + 1, step):
            s = ii[y + win, x + win] - ii[y, x + win] - ii[y + win, x] + ii[y, x]
            if s > best:
                best, xy = s, (x, y)
    return xy


def thick_picks(n: int = 8, seed: int = P.SEED) -> pd.DataFrame:
    """The montage aircraft of the setup record (eVTOL-Embedding-Extraction preprocess_variants.run_montage)."""
    V = PAT / "2_embedding_extraction/view_state_experiments"
    main = pd.read_csv(V / "exp3_main.csv", keep_default_na=False)
    g1 = pd.read_csv(V / "aircraft_common.csv", keep_default_na=False).set_index("aircraft_id")["g1_code"]
    main["g1"] = main.aircraft_id.map(g1)
    rng = np.random.default_rng(seed)
    classes = main.g1.value_counts().index[:n].tolist()
    return pd.DataFrame([main[main.g1 == c].iloc[rng.integers((main.g1 == c).sum())] for c in classes])


def fig_thickening(C: Dict, rows=("TR", "SLC", "HB")) -> Tuple[Path, pd.DataFrame]:
    pick = thick_picks()
    pick = pick[pick.g1.isin(rows)].set_index("g1").loc[list(rows)].reset_index()
    root = PAT / "2_embedding_extraction"
    m0 = pd.read_csv(root / "processed/518/manifest.csv")
    paths = {"plain": dict(zip(m0.figure_uid, m0.path))}
    rad = {}
    for v in ("thick1", "thick2"):
        m = pd.read_csv(root / "preprocess_variants" / v / "processed/518/manifest.csv")
        paths[v] = dict(zip(m.figure_uid, m.path))
        s = pd.read_csv(root / "preprocess_variants" / v / "sources_manifest.csv", keep_default_na=False)
        rad[v] = dict(zip(s.figure_uid, s.radius))
    cols = ["plain", "thick1", "thick2"]
    fig, axes = plt.subplots(len(pick), 6, figsize=(9.0, 1.65 * len(pick)))
    for i, r in enumerate(pick.itertuples()):
        ims = [Image.open(paths[c][r.fig_id]).convert("RGB") for c in cols]
        x0, y0 = _ink_window(ims[0])
        for j, (c, im) in enumerate(zip(cols, ims)):
            ax = axes[i, j]
            _thumb(ax, im, c if i == 0 else "")
            ax.add_patch(plt.Rectangle((x0, y0), 128, 128, fill=False, edgecolor="#c92f26", lw=0.8))
            _thumb(axes[i, 3 + j], im.crop((x0, y0, x0 + 128, y0 + 128)).resize((256, 256), Image.Resampling.NEAREST),
                   f"{c}, detail" if i == 0 else "")
        axes[i, 0].set_ylabel(f"{disp(r.g1)}\n{r.aircraft_id}", fontsize=8, rotation=0, ha="right", va="center")
    fig.tight_layout()
    info = pd.DataFrame({"g1": [disp(g) for g in pick.g1], "aircraft": pick.aircraft_id,
                         "r1": [rad["thick1"][f] for f in pick.fig_id], "r2": [rad["thick2"][f] for f in pick.fig_id]})
    return _save(fig, FIG_DIR / "f_thickening.png"), info


def _overlay(ax, img: np.ndarray, att: np.ndarray | None) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if att is not None:
        a = np.asarray(Image.fromarray(att.astype(np.float32)).resize(img.shape[::-1], Image.BILINEAR), dtype=np.float32)
        hi = np.quantile(a, 0.99) or 1.0
        v = np.clip(a / hi, 0, 1)
        rgba = plt.get_cmap("magma")(v)
        rgba[..., 3] = 0.85 * np.sqrt(v)          # low attention stays transparent, so the image reads through
        ax.imshow(rgba)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _att(name: str, tag: str):
    d = ATT / tag
    return np.load(d / f"{name}_cls.npy").astype(np.float32), pd.read_csv(d / f"{name}_meta.csv")


def fig_attention_models(C: Dict) -> Path:
    """Base against registers, layers 18/22/24: the check figure of registers_attention_figure.py,
    redrawn with the printed class names (first aircraft of each wanted class present in both runs)."""
    pat_cls = C["S_pat"].aircraft.set_index("aircraft_id")["label"]
    pho_cls = C["S_evn"].aircraft.set_index("aircraft_id")["label"]
    rows = []
    for name, want, cls_of in [("patent", ["TR", "MR"], pat_cls), ("photo", ["VT", "WM"], pho_cls)]:
        _, mb = _att(name, BASE_TAG)
        _, mr = _att(name, REG_TAG)
        common = mb[mb.figure_uid.isin(set(mr.figure_uid))]
        for c in want:
            sub = common[common.aircraft_uid.map(cls_of).eq(c) & common.path.map(lambda p: Path(p).exists())]
            r = sub.iloc[0]
            rows.append((name, r.figure_uid, r.path, c))
    arrs = {(n, t): _att(n, t) for n in ("patent", "photo") for t in (BASE_TAG, REG_TAG)}
    fig, axes = plt.subplots(len(rows), 7, figsize=(9.0, 1.4 * len(rows)))
    for i, (name, uid, path, c) in enumerate(rows):
        img = np.asarray(Image.open(path).convert("L"))
        _overlay(axes[i, 0], img, None)
        axes[i, 0].set_ylabel(f"{disp(c)}\n{'drawing' if name == 'patent' else 'photo'}", fontsize=8.5, rotation=0,
                              ha="right", va="center")
        for m, tag in enumerate((BASE_TAG, REG_TAG)):
            arr, meta = arrs[(name, tag)]
            row = meta.index[meta.figure_uid == uid][0]
            for j, L in enumerate((18, 22, 24)):
                ax = axes[i, 1 + 3 * m + j]
                _overlay(ax, img, arr[row, j])
                if i == 0:
                    ax.set_title(f"{'base' if m == 0 else 'registers'}\nL{L}", fontsize=8.5)
    axes[0, 0].set_title("input", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, PRIV_DIR / "f_attention_base_vs_registers.png")


def fig_attention_panel(C: Dict, per_class: int = 2, seed: int = P.SEED) -> Tuple[Path, List[str]]:
    """Registers CLS attention, L18 and L24, per parent class: aircraft drawn at random (seed 42)."""
    rng = np.random.default_rng(seed)
    pa = C["S_pat"].aircraft.set_index("aircraft_id")
    pho = C["S_evn"].aircraft.set_index("aircraft_id")["label"]
    ap, mp = _att("patent", REG_TAG)
    ah, mh = _att("photo", REG_TAG)
    mp = mp[mp.aircraft_uid.isin(pa.index)].copy()
    mp["c5"] = mp.aircraft_uid.map(pa["label5"])
    mh = mh[mh.aircraft_uid.isin(pho.index)].copy()
    mh["c5"] = mh.aircraft_uid.map(pho)
    fig, axes = plt.subplots(5, 4 * per_class, figsize=(9.2, 6.4))
    names = []
    for i, c in enumerate(ER.CLASSES5):
        cols = 0
        for src, meta, arr in [("drawing", mp, ap), ("photo", mh, ah)]:
            cand = meta[meta.c5 == c]
            take = cand.iloc[np.sort(rng.choice(len(cand), per_class, replace=False))]
            for r in take.itertuples():
                img = np.asarray(Image.open(r.path).convert("L"))
                lab = (f"{disp(pa.loc[r.aircraft_uid, 'label'])}" if src == "drawing" else c)
                for j, L in ((0, 18), (2, 24)):
                    ax = axes[i, cols]
                    _overlay(ax, img, arr[r.Index, j])
                    if i == 0:
                        ax.set_title(f"{src}\nL{L}", fontsize=8.5)
                    if j == 0:
                        ax.set_xlabel(lab, fontsize=8, color=INK)
                    cols += 1
                names.append(f"{c}: {r.aircraft_uid} ({src})")
        axes[i, 0].set_ylabel(f"{c}\n{ER.CLASS5_NAME[c]}".replace(" (Multicopter)", "").replace(" / PFD", ""), fontsize=8.5, rotation=0,
                              ha="right", va="center")
    fig.tight_layout(w_pad=0.2, h_pad=0.4)
    return _save(fig, PRIV_DIR / "f_attention_panel.png"), names


def fig_confusion(S, R: Dict, path: Path, size: Tuple[float, float]) -> Path:
    pred = R["ev"]["pred"][KEY]
    y, cl = S.y, S.classes
    M = pd.crosstab(pd.Categorical(y, cl), pd.Categorical(pred, cl), dropna=False).to_numpy()
    sh = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
    n = len(cl)
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(sh, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="white" if sh[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), [disp(c) for c in cl], fontsize=7)
    ax.set_yticks(range(n), [f"{disp(c)} ({M[i].sum()})" for i, c in enumerate(cl)], fontsize=7)
    ax.set_xlabel("predicted (kNN-5, maker held out)")
    ax.set_ylabel("true class (aircraft)")
    ax.spines[:].set_visible(False)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.8))
    fig.tight_layout()
    return _save(fig, path)


def five_class(C: Dict) -> pd.DataFrame:
    rp = C["R_pat"]["ev5"]["recall"]
    rp = rp[rp.matrix == "L24 CLS"].iloc[0]
    rh = C["R_evn"]["ev"]["recall"]
    rh = rh[rh.matrix == "L24 CLS"].iloc[0]
    cc = C["fairness_class_counts"].set_index("class")
    return pd.DataFrame({"class": ER.CLASSES5, "n_draw": [int(cc.loc[c, "drawings"]) for c in ER.CLASSES5],
                         "n_photo": [int(cc.loc[c, "photos"]) for c in ER.CLASSES5],
                         "rec_draw": [float(rp[c]) for c in ER.CLASSES5], "rec_photo": [float(rh[c]) for c in ER.CLASSES5]})


def fig_five_class(C: Dict, fc: pd.DataFrame, cmx: Dict) -> Path:
    from matplotlib.patches import Patch
    bd, bh = main_row(C["R_pat"], "ev5"), main_row(C["R_evn"])
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    x = np.arange(len(fc) + 1)
    w = 0.26
    groups = [(INK, bd.knn_bal_acc, bh.knn_bal_acc, cmx["mean"])] + \
        [(ER.CLASS_COLOR[r["class"]], r.rec_draw, r.rec_photo, cmx["recall"][r["class"]]) for _, r in fc.iterrows()]
    for xi, (col, d, h, m) in zip(x, groups):
        ax.bar(xi - w, d, w * 0.92, color="white", edgecolor=col, hatch="////", lw=0.8)
        ax.bar(xi, h, w * 0.92, color=col)
        ax.bar(xi + w, m, w * 0.92, color=col, alpha=0.4)
        for dx, v in ((-w, d), (0, h), (w, m)):
            ax.text(xi + dx, v + 0.015, f"{v:.2f}", ha="center", fontsize=6, color=INK)
    ax.errorbar([x[0] + w], [cmx["mean"]], yerr=[[cmx["mean"] - cmx["lo"]], [cmx["hi"] - cmx["mean"]]],
                fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
    ax.axhline(0.2, color=MUTED, lw=1, ls="--")
    ax.set_xticks(x, ["balanced\naccuracy"] + [f"{r['class']}\n{r.n_draw} | {r.n_photo}" for _, r in fc.iterrows()],
                  fontsize=7)
    ax.set_ylabel("recall (kNN-5, maker held out)")
    ax.set_ylim(0, 1.05)
    ax.legend(handles=[Patch(facecolor="white", edgecolor=INK, hatch="////", label="patent drawings"),
                       Patch(facecolor=INK, label="photos, all"),
                       Patch(facecolor=INK, alpha=0.4, label="photos, drawings' class mix")],
              fontsize=7, loc="upper left", ncol=3)
    fig.tight_layout()
    return _save(fig, FIG_DIR / "f_five_class.png")


def matching_curve(C: Dict) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    r = C["match_ranks"]
    r = r[(r.images == "no line drawings") & (r.direction == "patent->photo") & (r.layer == 24) & (r.pooling == "cls")]
    ks = np.unique(np.round(np.logspace(0, np.log10(int(r.gallery.iloc[0])), 200)).astype(int))
    found = np.array([(r["rank"] <= k).mean() for k in ks])
    # a random ranking: P(at least one of the query's own pages in the top k), averaged over the queries
    G = int(r.gallery.iloc[0])

    def p_rand(npos: int, k: int) -> float:
        if k >= G:
            return 1.0
        return 1.0 - np.exp(sum(np.log(G - npos - i) - np.log(G - i) for i in range(k)))
    rand = np.array([np.mean([p_rand(int(p), int(k)) for p in r.n_positive]) for k in ks])
    return r, ks, found, rand


def fig_matching(C: Dict) -> Tuple[Path, pd.DataFrame]:
    r, ks, found, rand = matching_curve(C)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    ax.plot(ks, found * 100, color=ACCENT, lw=2, label=f"registers, L24 CLS ({len(r)} patent aircraft)")
    ax.plot(ks, rand * 100, color=MUTED, lw=1.5, ls="--", label="random ranking")
    for k in (10, 50):
        v = float((r["rank"] <= k).mean())
        ax.plot([k], [v * 100], "o", color=ACCENT, ms=5)
        ax.annotate(f"top {k}: {v * 100:.0f} %", (k, v * 100), xytext=(6, -12), textcoords="offset points",
                    fontsize=7, color=INK)
    ax.set_xscale("log")
    ax.set_xlabel(f"k: size of the shortlist (of {int(r.gallery.iloc[0])} photo aircraft, log scale)")
    ax.set_ylabel("% of patent aircraft whose\nown page is in the top k")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    return _save(fig, FIG_DIR / "f_matching.png"), r


# ── the three small new computations ─────────────────────────────────────────
def exp4_subset(C: Dict, n_boot: int = 1000) -> Dict:
    """exp1, exp3 and exp4 against each other on the aircraft whose exp4 vector averages 2 or 3 slots:
    the only aircraft where exp4 differs from one figure. Predictions from the full neighbour pool
    (maker held out), accuracy on the subset, paired bootstrap of exp4 minus exp1."""
    S = C["S_pat"]
    y, mk = S.y, S.makers
    ex4 = S.extra["picks"]["exp4 view average"]
    ns = ex4.groupby("aircraft_id").size().reindex(S.aircraft.aircraft_id).to_numpy()
    sub = ns >= 2
    pred = {r: P.vote(P.neighbours(S.rules[r][KEY], mk), y) for r in ("exp1 view-first", "exp3 main figure",
                                                                     "exp4 view average")}
    ok = {r: (p == y)[sub].astype(float) for r, p in pred.items()}
    rng = np.random.default_rng(P.SEED)
    d = ok["exp4 view average"] - ok["exp1 view-first"]
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return {"n": int(sub.sum()), "acc": {r: float(v.mean()) for r, v in ok.items()}, "diff": float(d.mean()),
            "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975))}


def class_mix_matched(C: Dict, n_draw: int = 200, n_probe: int = 200) -> Dict:
    """Photo kNN-5 (maker held out, L24 CLS) on random photo subsets with the drawings' class counts:
    the neighbour pool then has the drawings' size and class mix. Per-class recall is averaged over the
    draws; the linear probe (folds by maker) runs on the first ``n_probe`` draws."""
    from sklearn.metrics import recall_score
    S = C["S_evn"]
    X, y, mk = S.rules["hero image"][KEY], S.y, S.makers
    cc = C["fairness_class_counts"].set_index("class")["drawings"]
    rng = np.random.default_rng(P.SEED)
    idx = {c: np.flatnonzero(y == c) for c in ER.CLASSES5}
    Sm = P.l2(X) @ P.l2(X).T
    out, recs, probes = [], [], []
    for d in range(n_draw):
        i = np.concatenate([rng.choice(idx[c], int(cc[c]), replace=False) for c in ER.CLASSES5])
        p = P.vote(P.neighbours(None, mk[i], S=Sm[np.ix_(i, i)]), y[i])
        out.append(balanced_accuracy_score(y[i], p))
        recs.append(recall_score(y[i], p, labels=ER.CLASSES5, average=None, zero_division=0))
        if d < n_probe:
            probes.append(P.probe(X[i], y[i], mk[i])["bal_acc"])
    out, probes = np.array(out), np.array(probes)
    return {"mean": float(out.mean()), "lo": float(np.quantile(out, 0.025)), "hi": float(np.quantile(out, 0.975)),
            "n": int(cc.sum()), "draws": n_draw, "recall": dict(zip(ER.CLASSES5, np.mean(recs, axis=0))),
            "probe_mean": float(probes.mean()), "probe_lo": float(np.quantile(probes, 0.025)),
            "probe_hi": float(np.quantile(probes, 0.975)), "probe_draws": n_probe}


# ── the text ─────────────────────────────────────────────────────────────────
def build(C: Dict) -> Tuple[str, Dict]:
    D = Doc()
    info: Dict = {}
    S_pat, S_evn = C["S_pat"], C["S_evn"]
    Rp, Rh, Bp, Bh = C["R_pat"], C["R_evn"], C["B_pat"], C["B_evn"]
    mp, mh, m5 = main_row(Rp), main_row(Rh), main_row(Rp, "ev5")
    nP, nH = len(S_pat.aircraft), len(S_evn.aircraft)
    ft = S_pat.extra["figure_table"]
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(S_pat.aircraft.aircraft_id))]
    n_pat_patents = S_pat.aircraft.patent_id.nunique()
    cnt12 = S_pat.aircraft.label.value_counts()
    cnt5h = S_evn.aircraft.label.value_counts()
    small = [c for c in ER.G1_ORDER if cnt12.get(c, 0) < 20]
    a = S_pat.aircraft
    gt = a.arch_gt != ""
    from sklearn.metrics import cohen_kappa_score
    kap = cohen_kappa_score(a.label[gt], a.arch_gt[gt])
    agree, gt_n = int((a.label == a.arch_gt).sum()), int(gt.sum())
    n_excl = len(pd.read_csv(PAT / "2_embedding_extraction/view_state_experiments/excluded_aircraft.csv"))
    pv = C["reg_paired"].set_index(["source", "label"])
    rk = C["reg_knn"]
    rkv = lambda tag, src, lab: rk[(rk.tag == tag) & (rk.source == src) & (rk.label == lab) & (rk.matrix == "L24 CLS")].iloc[0]  # noqa: E731
    dec = C["s1a_decision"].set_index("candidate")
    _c = C["s1a_candidates"]
    l22f = _c[(_c.pool == "registers only") & (_c.candidate == "registers L22 CLS")].set_index("label").selection_freq
    s1a_all = C["s1a_candidates"][C["s1a_candidates"].pool == "all 12"]
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
    hsd = hs.groupby("label").agg(mean=("d", "mean"), higher=("d", lambda v: (v > 0).mean()))
    hn = half.groupby("label").n_aircraft.agg(["min", "max"])
    sh = C["share"].set_index(["model", "source", "layer"])
    mt = C["match"]
    mt = mt[(mt.images == "no line drawings") & (mt.layer == 24) & (mt.pooling == "cls")].set_index("direction")
    mtb = C["match_base"]
    mtb = mtb[(mtb.embedding == BASE_TAG) & (mtb.images == "no line drawings") & (mtb.layer == 24) &
              (mtb.pooling == "cls")].set_index("direction")
    m_pp = mt.loc["patent->photo"]
    cm = C["crop_man"]
    n_fb = int((cm.status != "ok").sum())
    ex4 = S_pat.extra["picks"]["exp4 view average"]
    nslots = ex4.groupby("aircraft_id").size()
    single = {k: v.set_index("aircraft_id").fig_id for k, v in S_pat.extra["picks"].items() if k != "exp4 view average"}
    same = lambda x, y: float((single[x].reindex(a.aircraft_id) == single[y].reindex(a.aircraft_id)).mean())  # noqa: E731
    br = Rp["ev"]["by_rule"]
    br = br[br.matrix == "L24 CLS"].set_index("rule")
    brh = Rh["ev"]["by_rule"]
    brh = brh[brh.matrix == "L24 CLS"].set_index("rule")
    conf_p = Rp["confound"].set_index("matrix")
    conf_h = Rh["confound"].set_index("matrix")
    lay_p = Rp["layer"].set_index("matrix")
    lay_h = Rh["layer"].set_index("matrix")
    TAGR = REG_TAG
    cmx = class_mix_matched(C)
    info["class_mix"] = cmx
    cmx_inside = cmx["lo"] <= fair.loc["colour"].drawings_bal_acc <= cmx["hi"]

    # ── title block (removed by to_markdown; the Word title comes from the metadata) ──
    D.L += ["# Visual Learning: DINOv2 embeddings of patent drawings and photographs", ""]

    # ═══ 0 Summary ═══
    D.h1("0. Summary")
    D.p("**Research question (RQ4, method).** Does the figure alone carry the architecture? The figure is read by a "
        "frozen vision model, DINOv2 with registers, on patent drawings and on real photographs, and the reading is "
        "judged against the human labels.",
        f"The drawings are {n_(len(wv))} whole-aircraft patent figures of {n_(nP)} aircraft, labelled with the "
        f"codebook's 12 G1 architecture types. The photographs are the hero images of {n_(nH)} aircraft of the "
        "evtol.news directory, labelled with the directory's 5 classes. One vector per aircraft, the maker held out "
        "of every comparison, and a kNN-5 balanced accuracy as the primary score make the two sources directly "
        "comparable on the 5 classes they share.")
    fc = five_class(C)
    D.table(md_df([
        ("Do the drawings carry the architecture?",
         f"Yes, weakly. 12 classes: kNN-5 {f2(mp.knn_bal_acc)} {ci(mp.knn_ci_lo, mp.knn_ci_hi)} against a "
         f"shuffled level of {f2(mp.knn_chance_p95)}; probe {f2(mp.probe_bal_acc)}", "§4"),
        ("Do the photos carry it?",
         f"Yes, moderately. 5 classes: kNN-5 {f2(mh.knn_bal_acc)} {ci(mh.knn_ci_lo, mh.knn_ci_hi)}, shuffled "
         f"{f2(mh.knn_chance_p95)}; probe {f2(mh.probe_bal_acc)}", "§5"),
        ("Which source carries more?",
         f"Photos score higher on the same 5 classes ({f2(fair.loc['colour'].photos_bal_acc)} against "
         f"{f2(fair.loc['colour'].drawings_bal_acc)}, also in greyscale)"
         + (f", but with the drawings' class mix the photos score {f2(cmx['mean'])}: the gap is mostly the pool"
            if cmx_inside else f", also with the drawings' class mix ({f2(cmx['mean'])})"), "§6"),
        ("What dominates the drawing embedding?",
         f"The view of the drawing (d ratio {f2(conf_p.loc['L24 CLS'].conf_over_arch)}); the architecture signal "
         "stays above chance once the view is removed", "§4"),
        ("Do the classes form clusters on their own?",
         f"No. k-means ARI {f2(mp.t5_ari, 3)} (drawings), {f2(mh.t5_ari, 3)} (photos)", "§4, §5"),
        ("Does the choice of image matter?",
         f"Little. Rules span {f2(br.knn_bal_acc.min())} to {f2(br.knn_bal_acc.max())} (drawings); cropping the "
         "photos changes nothing measurable", "§4, §5"),
        ("Can one aircraft be matched across sources?",
         f"Above chance, not reliably: own page in the top 50 for {pc(m_pp['R@50'])} of patent aircraft "
         f"(random {pc(m_pp['random_R@50'], 1)})", "§6"),
    ], ["question", "short answer", "where"]), "The questions and their short answers (registers model, 518 px, "
        "layer 24 CLS, maker held out).", right=[])
    D.table(md_df([
        ("model", "DINOv2 with registers (large), frozen", "image type"),
        ("resolution", "518 px, white-padded square", "image type"),
        ("layer and token", "layer 24, CLS", "label-selected"),
        ("drawing preparation", "plain (no line thickening)", "label-selected"),
        ("photo preparation", "colour", "label-selected"),
    ], ["choice", "value", "type"]), "The chosen setup (§2 gives the reason for each row).", right=[])

    # ═══ 1 Data ═══
    D.h1("1. Data and images")
    D.p(f"Two image sources enter the analysis, and both reduce to **one vector per aircraft**. The patent drawings "
        f"come from the labelled corpus, the photographs from the evtol.news directory. The funnels in Figure "
        f"{D.nfig + 1} show "
        "how each source shrinks from the files on disk to the analysis set.")
    D.figure(fig_funnels(C), "From the files on disk to the analysis sets.",
             f"patent drawings: notebook 20 funnel (selection/funnel.csv) and notebook 23 figure table; photos: "
             f"1_filter/image_decisions.csv of the evtol.news crawl of 2026-09-17; each image counted under the first "
             f"rule that removed it.",
             "each bar is the number of images left after the step on its left; the blue bar is the analysis set.",
             origin="NEW")
    D.h2("1.1 Patent drawings")
    D.p(f"The analysis set holds **{n_(nP)} aircraft from {n_(n_pat_patents)} patents** and {n_(len(wv))} "
        f"whole-aircraft figures. A figure enters when the labeller approved it, its aircraft passes the domain gate "
        f"(UAV-similar, not electric or STOL-only aircraft leave) and its patent is not a D1/D2 duplicate. Part and "
        f"detail figures leave ({int(C['funnel'].removed.iloc[-1])} figures), and so do the "
        f"{n_excl} aircraft marked unclassifiable. The label is the human G1 type of the "
        f"codebook (Table {D.ntab + 1}). It agrees with the independent whole-patent reading for {agree} of {gt_n} aircraft "
        f"(κ {kap:.2f}).")
    g1t = md_df([(disp(c), G1_NAME[c], ER.PARENT[c], int(cnt12.get(c, 0))) for c in ER.G1_ORDER],
                ["G1", "architecture type", "parent (5-class)", "aircraft"])
    D.table(g1t, f"The 12 G1 types, their 5-class parent and the aircraft per type ({n_(nP)} aircraft). TP is the "
                 "wizard's TR, printed under the display name Tilt Propulsor.")
    D.figure(fig_examples_patents(C), "One main figure per G1 type, as the model receives it (518 px, padded square).",
             f"patent drawings, rule exp3 main figure; the alphabetically first aircraft of each type whose main figure "
             f"is a Perspective view; label: G1 type (its 5-class parent); n = 12 of {n_(nP)} aircraft.",
             "the label above each drawing is the human G1 type and its parent class; the code below is the aircraft.",
             origin="regenerated (TP label)")
    D.h2("1.2 One vector per aircraft, and the flight state")
    ct = wv[wv.g1_code.isin(CONVERTIBLE)]
    ctab = pd.crosstab(ct.g1_code, ct.state4).reindex(CONVERTIBLE).fillna(0).astype(int)
    var = wv[wv.g1_code.isin(CONVERTIBLE)]
    pc_ = var[(var.view4 == "Perspective") & (var.state4 == "Cruise")].aircraft_id.nunique()
    ph_ = var[(var.view4 == "Perspective") & (var.state4 == "Hover")].aircraft_id.nunique()
    n_var = var.aircraft_id.nunique()
    D.p("An aircraft has a median of two whole-aircraft figures, so a rule must decide which figure becomes its "
        "vector. Four rules were fixed before any score was read (notebook 23). Two of them sort on the **view** "
        "(Perspective > Plan > Side > Front/Rear) and on the **flight state**. The flight state matters only for the "
        f"{len(CONVERTIBLE)} convertible types ({', '.join(disp(c) for c in CONVERTIBLE)}), whose shape changes "
        "between hover and cruise. The wizard records it per figure as Cruise, Hover, or **Both** when the moving part "
        "is drawn in both positions (label added on 2026-09-19). The state-first rule prefers Cruise as the canonical "
        f"state, because more convertible aircraft have a Perspective figure in Cruise ({pc_}) than in Hover "
        f"({ph_}) among the {n_var}.")
    D.table(md_df([
        ("exp1 view-first", "best view, then best flight state", f"{pc(same('exp1 view-first', 'exp3 main figure'), 1)}"),
        ("exp2 state-first", "best flight state (Cruise > Both > Hover > Other), then best view",
         f"{pc(same('exp2 state-first', 'exp3 main figure'), 1)}"),
        ("exp3 main figure", "the figure the labeller marked as main (primary rule)", "100.0 %"),
        ("exp4 view average", "mean of the best figure in each of Perspective, Plan and Side", "n/a"),
    ], ["rule", "which figure", "same figure as exp3"]),
        f"The four image rules. exp1 and exp2 pick the same figure for {pc(same('exp1 view-first', 'exp2 state-first'), 1)} "
        f"of the aircraft; exp4 averages 2 slots for {int((nslots == 2).sum())} aircraft and 3 for "
        f"{int((nslots == 3).sum())}, and equals one figure for the other {int((nslots == 1).sum())}.")
    ctab = ctab[[c for c in ["Cruise", "Hover", "Both", "Other"] if c in ctab.columns]]
    ctab.insert(0, "G1", [disp(c) for c in ctab.index])
    D.table(ctab.reset_index(drop=True), "Flight state of the whole-aircraft figures of the convertible types.")
    D.p(f"The flight state barely moves the vector choice: exp1 and exp2 differ on "
        f"{int(round((1 - same('exp1 view-first', 'exp2 state-first')) * nP))} aircraft only. The labeller's main "
        f"figure differs from the view-first pick on {int(round((1 - same('exp1 view-first', 'exp3 main figure')) * nP))}.")
    D.h2("1.3 Photographs")
    im, au = C["img_dec"], C["audit"]
    kept = im[im.keep.astype(str) == "True"]
    k_m = int(((au.stratum == "main") & (au.verdict == "whole_aircraft")).sum())
    n_m = int((au.stratum == "main").sum())
    k_s = int(((au.stratum == "siglip_drop") & (au.verdict == "whole_aircraft")).sum())
    n_s = int((au.stratum == "siglip_drop").sum())
    k_r = int(((au.stratum == "review_band") & (au.verdict == "whole_aircraft")).sum())
    n_r = int((au.stratum == "review_band").sum())
    n_rev = int((im.auto == "review").sum())
    pch = C["parent_check"]
    D.p(f"The directory lists {n_(len(C['pages']))} aircraft pages carrying {n_(len(im))} images. Fixed rules drop "
        "small files and files shared with another page. A zero-shot SigLIP score p(whole aircraft) then keeps an image "
        f"at p ≥ 0.80 and drops it below 0.35. The {n_rev} images in between (the review band) are held back. "
        f"{n_(len(kept))} images survive, and the page's first kept image, its **hero image**, represents the "
        f"aircraft: **{n_(nH)} aircraft**. The label is the directory class of the page (VT {cnt5h['VT']}, "
        f"LC {cnt5h['LC']}, WM {cnt5h['WM']}, ER {cnt5h['ER']}, HB {cnt5h['HB']}). On the {len(pch)} patent aircraft "
        f"that have a directory page, the parent of the human G1 type matches the directory class for "
        f"{int(pch.agrees.sum())}.",
        f"The filter was audited on a random sample: {k_m} of {n_m} hero images show the whole aircraft, {k_s} of "
        f"{n_s} dropped images were a whole vehicle, and {k_r} of {n_r} review-band images would have been usable. The "
        "audit judge was the AI assistant used for the pipeline, working from contact sheets that showed the filter "
        "score, so the audit is neither blind nor yet re-checked by the author (§9).")
    D.figure(EE / "embedding_analysis_registers/figs/f02_examples.png",
             "One hero image per directory class (built aircraft, highest filter score).",
             f"evtol.news photos, hero images; 5 of {n_(nH)} aircraft (figure of the registers photo report).",
             "the heading is the directory class; the caption under each photo is the directory page title.",
             origin="reused", private=True)
    D.h2("1.4 Processing, and whether the selection decides the result")
    D.p("Both sources pass through the same processing code (notebook 21). A drawing is rotated by its labelled angle. "
        "Every image is padded to a white square and resized to 518 px, which gives 37 × 37 patches of 14 px. The "
        "white pad matches the paper background of a drawing, so the square format adds no new content to it.",
        "Each selection step was then made stricter, or its measured errors removed, and the primary score "
        f"recomputed (Table {D.ntab + 1}). Every variant stays inside the interval of the set as built, so no selection step "
        "decides the result.")
    rows = []
    for src, R in (("drawings", Rp), ("photos", Rh)):
        s = R["sens"]
        s = s[s.matrix == "L24 CLS"]
        for r in s.itertuples():
            rows.append((src, r.variant, int(r.n_aircraft), f"{r.knn_bal_acc:.3f}", ci(r.ci_lo, r.ci_hi)))
    D.table(md_df(rows, ["source", "variant", "aircraft", "kNN-5", "95 % interval"]),
            "Sensitivity of the primary score to the selection (registers, L24 CLS, maker held out).")

    # ═══ 2 Setup selection ═══
    D.h1("2. Setup selection")
    D.p("A frozen model leaves few choices, but each one can move the score: which checkpoint, which input size, "
        "which layer and token, and how each source's images are prepared. The choices fall into two kinds. "
        "**Image-type choices** follow from what patent drawings are (thin black lines on a large white field), so "
        "they transfer to any other patent-drawing study. **Label-selected choices** are made on the labels, and a "
        "choice made on labels must not be scored on the aircraft that made it.")
    D.table(md_df([
        ("model", "dinov2-large; dinov2-with-registers-large", "image type", "fixed on the attention artifact; held-out rule confirms"),
        ("resolution", "518 px", "image type", "the model's largest training size; not selected"),
        ("layer", "18, 22, 24", "label-selected", "split-half rule"),
        ("token", "CLS; mean of the patch tokens", "label-selected", "split-half rule"),
        ("drawing preparation", "plain; thick1; thick2", "label-selected", "split-half rule"),
        ("photo preparation", "colour; greyscale", "label-selected", "split-half rule"),
    ], ["choice", "candidates, fixed in advance", "type", "how chosen"]), "The candidate grid.", right=[])
    D.h2("2.1 The split-half rule")
    D.p("Every label-selected choice uses one rule. The makers are split at random into two halves, 200 times (seed "
        "42), and every aircraft follows its maker. Each candidate is scored on half A with neighbours from half A "
        "only. The best candidate on A is then scored on half B, and the same is done with the halves swapped. This "
        "gives 400 choices per label set. The rule reports how often each candidate is chosen, its held-out score, and "
        "the **optimism gap**: how much the winner's in-sample score exceeds its held-out score. The decision is the "
        "highest mean selection frequency, averaged over the label sets within a source and then over the two "
        "sources.",
        f"A half holds {hn.loc['patents G1 12-class', 'min']} to {hn.loc['patents G1 12-class', 'max']} drawing "
        f"aircraft and {hn.loc['photos 5-class', 'min']} to {hn.loc['photos 5-class', 'max']} photo aircraft, so the "
        "neighbour pool is about half the full one. Held-out scores therefore sit below the full-sample scores of "
        "§4 to §6, and only scores in the same column compare.")
    D.h2("2.2 Model: registers against base")
    D.p("DINOv2 without registers parks high-norm artifact tokens in low-information background patches (Darcet et "
        "al., 2024). A patent drawing is mostly white background, and in the base model the late-layer CLS attention "
        f"collapsed onto a single padding patch (Table {D.ntab + 1}, Figure {D.nfig + 1}). The registers checkpoint adds four register tokens "
        "that absorb this mass, and was run on the same processed images with the same code.")
    D.table(md_df([(f"L{L}", f"{sh.loc[('base', 'patent', L)].max_patch_share_median:.3f}",
                    f"{sh.loc[('registers', 'patent', L)].max_patch_share_median:.3f}",
                    f"{sh.loc[('base', 'photo', L)].max_patch_share_median:.3f}",
                    f"{sh.loc[('registers', 'photo', L)].max_patch_share_median:.3f}") for L in (18, 22, 24)],
                  ["layer", "drawings, base", "drawings, registers", "photos, base", "photos, registers"]),
            "Median share of the CLS attention that falls on the single largest patch (the collapse measure). "
            f"Registers: {int(sh.loc[('registers', 'patent', 24)].n_images)} drawings, "
            f"{int(sh.loc[('registers', 'photo', 24)].n_images)} photos.")
    D.figure(fig_attention_models(C), "Where the CLS token looks: base (columns 2 to 4) against registers (columns 5 "
             "to 7), layers 18, 22 and 24, same images.",
             "DINOv2-large and DINOv2-large with registers, frozen, 518 px; CLS query against every patch key, mean "
             "over heads, each map scaled to its own 99th percentile; 2 patent drawings and 2 photos, the first "
             "aircraft of each wanted class in both runs (as in registers_attention_figure.py).",
             "bright = where the CLS token looks. In base L22 and L24 the pale square in the top-left corner is the "
             "padding patch that collects the attention (clipped at the 99th percentile), and the dots on the white "
             "field are further background patches; with registers the late layers stay on the aircraft.",
             origin="regenerated (TP label, 4 of the 6 rows)", private=True)
    rows = []
    for lab_s, src, lab_k, lab_sel in [("drawings, 12 classes", "patents", "G1 12-class", "patents G1 12-class"),
                                       ("drawings, 5 classes", "patents", "parent 5-class", "patents parent 5-class"),
                                       ("photos, 5 classes", "photos", "5-class", "photos 5-class")]:
        pr = pv.loc[(src, lab_k)]
        rows.append((lab_s, f"{rkv(BASE_TAG, src, lab_k).bal_acc:.3f}", f"{rkv(TAGR, src, lab_k).bal_acc:.3f}",
                     f"{sg(pr['diff'])} {ci(pr.ci_lo, pr.ci_hi, 3, True)}", pc(reg_share[lab_sel], 1)))
    D.table(md_df(rows, ["label set", "base", "registers", "paired difference [95 % CI]", "registers chosen (held out)"]),
            "Registers against base on the same aircraft: kNN-5 balanced accuracy, L24 CLS, maker held out; the last "
            "column is the share of the 400 held-out selections that pick a registers candidate when all 12 compete.")
    D.p(f"The registers model scores higher on all three label sets. The paired interval excludes zero on the drawings "
        f"and touches it on the photos. Held out, a registers candidate wins {pc(reg_share.min(), 1)} to "
        f"{pc(reg_share.max(), 1)} of the selections, and on the same halves registers minus base at L24 CLS averages "
        f"{sg(hsd.loc['patents G1 12-class', 'mean'])} (drawings, 12 classes) to "
        f"{sg(hsd.loc['patents parent 5-class', 'mean'])} (drawings, 5 classes). Patent-to-photo matching improves too "
        f"(own page in the top 50: {pc(mtb.loc['patent->photo', 'R@50'])} base, {pc(m_pp['R@50'])} registers). "
        "The model choice was fixed in advance on the attention artifact, and the held-out rule confirms it. One "
        "qualification comes from §4.4: once the view of the drawing is removed, the two models score alike.")
    D.h2("2.3 Layer and token")
    D.figure(fig_selection(C), "How often each layer and token is chosen by the split-half rule (registers model).",
             "DINOv2-large with registers, frozen, 518 px; 6 matrices (layers 18/22/24 × CLS / patch mean); 200 "
             f"maker splits × 2 directions; drawings: exp3 main figure, halves of {hn.loc['patents G1 12-class', 'min']} "
             f"to {hn.loc['patents G1 12-class', 'max']} aircraft; photos: hero image, halves of "
             f"{hn.loc['photos 5-class', 'min']} to {hn.loc['photos 5-class', 'max']} aircraft.",
             "each bar is the share of the 400 choices that picked this matrix as best on one half; blue = the "
             "matrix carried forward.", origin="NEW")
    D.table(md_df([(c.replace("registers ", ""), pc(r.selection_freq, 1), f"{r.fixed_heldout_mean:.3f}")
                   for c, r in dec.iterrows()], ["matrix", "mean selection frequency", "mean held-out score"]),
            "Decision over the 6 registers matrices (frequencies averaged over the three label sets).")
    D.p(f"Layer 24 CLS is chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of the selections. The runner-up, "
        f"layer 22 CLS, in {pc(dec.loc['registers L22 CLS'].selection_freq)}, most often on the photos "
        f"({pc(l22f['photos 5-class'])}, against {pc(min(l22f['patents G1 12-class'], l22f['patents parent 5-class']))} "
        f"to {pc(max(l22f['patents G1 12-class'], l22f['patents parent 5-class']))} on the drawings). The "
        "patch-token mean is almost never chosen. The last layer's CLS token is the one DINOv2 is trained to make a "
        "global summary of the image, so the choice is also the expected one.")
    D.h2("2.4 Drawing preparation")
    sm1 = pd.read_csv(PAT / "2_embedding_extraction/preprocess_variants/thick1/sources_manifest.csv", keep_default_na=False)
    sm2 = pd.read_csv(PAT / "2_embedding_extraction/preprocess_variants/thick2/sources_manifest.csv", keep_default_na=False)
    n_col = int((pd.to_numeric(sm1.src_max_chroma) > 8).sum())
    fpath, tinfo = fig_thickening(C)
    D.p("Patent lines are thin, and after the resize to 518 px a fine stroke can shrink to a grey pixel. Line "
        "thickening was therefore a candidate: a minimum filter on the original drawing whose radius is chosen so "
        "that **thick1 grows each side of a stroke by about 1 px at the model input, about 2 px of width**, and thick2 "
        f"by about 2 px per side (radius in original pixels: thick1 median {int(sm1.radius.median())}, thick2 median "
        f"{int(sm2.radius.median())}). A gentler variant, below 1 px per side, was not in the candidate set fixed in "
        "advance.")
    D.figure(fpath, "Plain, thick1 and thick2 as the model receives them (left), and the same 128 px detail of each "
             "(right).",
             f"patent drawings, exp3 main figures of the aircraft in the setup montage (seed 42), {len(tinfo)} of the "
             "8 rows; 518 px model inputs; the red square marks the detail window.",
             "compare each row from left to right: thickening merges thin propeller blades and reference numerals, and "
             "turns hatching and shading into solid black.", origin="regenerated (TP label, 3 of the 8 rows)")
    rows = []
    for lab_s, lab in (("drawings, 12 classes", "patents G1 12-class"), ("drawings, 5 classes", "patents parent 5-class")):
        for v in ("thick1", "thick2"):
            r = pair.loc[(lab, f"{v} - plain")]
            rows.append((lab_s, v, f"{s2k.loc[(lab, 'plain')].bal_acc:.3f}", f"{s2k.loc[(lab, v)].bal_acc:.3f}",
                         f"{sg(r['diff'])} {ci(r.ci_lo, r.ci_hi, 3, True)}", pc(s1bc.loc[(lab, v)].selection_freq)))
    D.table(md_df(rows, ["label set", "variant", "plain", "variant score", "difference [95 % CI]", "chosen (held out)"]),
            "Line thickening against the plain drawing, paired on the same aircraft (registers, L24 CLS, main figure).")
    D.p(f"Thickening lowers the score in every row, and the thick2 interval excludes zero on both label sets. The "
        f"split-half rule keeps the plain drawing in {pc(s1b.loc['plain'].selection_freq)} of the selections. The "
        f"montage shows why: thickening fills the fine detail that separates the types. {n_col} of the "
        f"{len(sm1)} source drawings carry colour, which the thick variants also remove.")
    D.h2("2.5 Photo preparation")
    g = pair.loc[("photos 5-class", "grey - colour")]
    D.p(f"Greyscale photos score {f2(s2k.loc[('photos 5-class', 'grey')].bal_acc, 3)} against "
        f"{f2(s2k.loc[('photos 5-class', 'colour')].bal_acc, 3)} in colour, a paired difference of {sg(g['diff'])} "
        f"{ci(g.ci_lo, g.ci_hi, 3, True)}, and the rule keeps colour in {pc(s1b.loc['colour'].selection_freq)} of the "
        "selections. Colour is kept as the untransformed image, not as a better one. The greyscale run doubles as the "
        "fairness check of §6.")
    D.h2("2.6 Optimism gap and the chosen setup")
    rows = []
    for lab, lab_s in (("patents G1 12-class", "drawings, 12 classes"), ("patents parent 5-class", "drawings, 5 classes"),
                       ("photos 5-class", "photos, 5 classes")):
        r, b = lab_reg.loc[lab], lab_b.loc[lab]
        rows.append((lab_s, f"{r.most_chosen.replace('registers ', '')}: {r.chosen_heldout_mean:.3f}",
                     f"{r.winner_in_sample_mean:.3f}", f"{r.optimism_gap:.3f}",
                     f"{b.most_chosen}: {b.optimism_gap:.3f}"))
    D.table(md_df(rows, ["label set", "held-out score of the chosen", "in-sample score of the winner", "gap, layer+token",
                         "gap, preparation"]), "Optimism of each label-selected choice (registers candidates only).")
    D.p(f"The gap is {f2(lab_reg.optimism_gap.min(), 3)} to {f2(lab_reg.optimism_gap.max(), 3)} for the layer and "
        f"token, and {f2(lab_all.optimism_gap.min(), 3)} to {f2(lab_all.optimism_gap.max(), 3)} when both models "
        "compete. Choosing on the labels therefore inflates the scores by a few hundredths at most. That is small "
        "against the intervals of §4 to §6, and it is the honest size of the selection optimism.")
    D.table(md_df([
        ("model", "DINOv2 with registers", "no attention collapse on blank drawing background; registers chosen in "
         f"{pc(reg_share.min(), 1)} to {pc(reg_share.max(), 1)} of held-out selections", "image type"),
        ("resolution", "518 px", "the largest size the model was trained on; keeps thin lines", "image type"),
        ("layer + token", "layer 24, CLS", f"chosen in {pc(dec.loc['registers L24 CLS'].selection_freq)} of held-out "
         f"selections (next: layer 22 CLS, {pc(dec.loc['registers L22 CLS'].selection_freq)})", "label-selected"),
        ("drawings", "plain", "thickening fills fine detail and lowers the score (thick2: interval below zero)",
         "label-selected"),
        ("photos", "colour", f"greyscale scores the same ({sg(g['diff'])}, inside the interval)", "label-selected"),
    ], ["choice", "value", "reason", "type"]), "The chosen setup.", right=[])

    # ═══ 3 Method ═══
    D.h1("3. Evaluation method")
    D.p("The unit of analysis is the **aircraft**: one patent can disclose several aircraft, and one directory page "
        "shows one. Every test that compares aircraft leaves out pairs and neighbours of the **same maker**, so a "
        "company's house style can never count as architecture. The maker is the applicant's company group for "
        "patents (the first applicant for individual inventors) and the maker named on the directory page for photos.",
        "The **primary score** is the kNN-5 balanced accuracy: the class of each aircraft is predicted by the "
        "majority class of its five nearest aircraft by cosine distance, and balanced accuracy is the mean recall "
        "over the classes, so a large class cannot carry it. Its 95 % interval is a bootstrap over the aircraft "
        "(1 000 resamples). The chance level actually reached by the same neighbours is the 95th percentile of 200 "
        "label shuffles. A **linear probe** (logistic regression, 5 folds grouped by maker) reads the same vectors "
        "with every dimension weighted, and bounds what the neighbours miss. Paired differences use 1 000 paired "
        "bootstrap resamples of the same aircraft.",
        "The same code (embedding_protocol.py) runs on both sources with the same parameters, so a number in one "
        "source means what the same number means in the other.")
    D.table(md_df([
        ("T1 integrity", "no NaN, infinite, all-zero or duplicate rows"),
        ("T2 structure", "variance on PC1 against a random matrix; effective dimensions; Hopkins clustering tendency"),
        ("T3 separation", "mean distance between classes ÷ within classes, Cohen's d, p from 999 label permutations"),
        ("T4 prediction", "kNN-5 balanced accuracy (primary), bootstrap interval, shuffled chance; probe"),
        ("T5 clusters", "k-means with as many clusters as classes against the labels (ARI, NMI)"),
        ("T6 what else", "T3's d for a non-architecture label ÷ T3's d for the architecture"),
        ("T7 image choice", "T4 under every image rule of the source"),
    ], ["test", "what it measures"]), "The seven tests of the protocol.", right=[])

    # ═══ 4 Patents ═══
    D.h1("4. Results: patent drawings")
    m = Rp["ev"]["main"]
    D.p(f"**Soundness (T1, T2).** No matrix has an invalid row. On its first principal component every matrix carries "
        f"{m.t2_pc1_vs_random.min():.0f} to {m.t2_pc1_vs_random.max():.0f} times the variance of a random matrix of "
        f"the same shape, and Hopkins lies between {m.t2_hopkins.min():.2f} and {m.t2_hopkins.max():.2f}: the vectors "
        "clump, but as a continuum and not as separate islands.",
        f"**Separation (T3).** Same-class aircraft sit closer than different-class aircraft in all 6 matrices "
        f"(p ≤ {max(0.001, m.t3_p.max()):.3f}), but by little: at L24 CLS the between/within ratio is "
        f"{mp.t3_ratio:.3f} and Cohen's d {mp.t3_d:.2f}. On average the classes overlap heavily.")
    D.h2("4.1 Reading the architecture (T4)")
    rows = []
    m5all = Rp["ev5"]["main"]
    for (_, r), (_, r5) in zip(m.iterrows(), m5all.iterrows()):
        rows.append((r.matrix, f"{r.knn_bal_acc:.3f}", ci(r.knn_ci_lo, r.knn_ci_hi), f"{r.knn_chance_p95:.3f}",
                     f"{r.probe_bal_acc:.3f}", f"{r5.knn_bal_acc:.3f}", f"{r5.probe_bal_acc:.3f}"))
    D.table(md_df(rows, ["matrix", "kNN-5, 12", "95 % interval", "shuffled p95", "probe, 12", "kNN-5, 5", "probe, 5"]),
            f"Prediction of the architecture from the drawings ({nP} aircraft, exp3 main figure, maker held out). "
            "12 = G1 types (chance 1/12); 5 = parent classes (chance 1/5).")
    F = DocFigs(FIG_DIR, "pat")
    ER.fig_prediction(F, S_pat, Rp)
    it = F.items["f04_prediction"]
    D.figure(it["path"], "Architecture read from the drawings, per layer and token (blue: the matrix carried forward).",
             f"DINOv2-large with registers, frozen, 518 px; layers 18/22/24 × CLS / patch mean; exp3 main figure; "
             f"{nP} aircraft; label: G1 type, 12 classes.",
             "bars = kNN-5 balanced accuracy with 95 % interval; diamonds = linear probe; dashed line = shuffled-label "
             "chance (95th percentile). Balanced accuracy is the mean recall over the classes.", width=80, origin="NEW render (no baked text)")
    D.p(f"At L24 CLS the drawings reach {f2(mp.knn_bal_acc)} {ci(mp.knn_ci_lo, mp.knn_ci_hi)} on 12 classes against a "
        f"shuffled level of {f2(mp.knn_chance_p95)}, and {f2(m5.knn_bal_acc)} {ci(m5.knn_ci_lo, m5.knn_ci_hi)} on the "
        f"5 parent classes against {f2(m5.knn_chance_p95)}. The architecture is present, and weakly: on 12 classes "
        f"a type is recovered for {f2(mp.knn_bal_acc)} of its aircraft on average, fewer than one in four. The probe reaches "
        f"{f2(mp.probe_bal_acc)}, well above the neighbours, so the architecture is in the vector but it is not what "
        f"decides which aircraft are nearest. The reports' own layer rule, run in-sample on 200 random 80 % subsets, "
        f"picks L24 CLS in {pc(lay_p.loc['L24 CLS'].wins_on_subsets)} of them, in line with the split-half rule.")
    rec = Rp["ev"]["recall"]
    rec = rec[rec.matrix == "L24 CLS"].iloc[0]
    cl = ER.G1_ORDER
    hlf = (len(cl) + 1) // 2
    rows = []
    for i in range(hlf):
        c1 = cl[i]
        c2 = cl[i + hlf] if i + hlf < len(cl) else None
        rows.append((disp(c1), int(cnt12[c1]), f"{rec[c1]:.2f}",
                     disp(c2) if c2 else "", int(cnt12[c2]) if c2 else "", f"{rec[c2]:.2f}" if c2 else ""))
    D.table(md_df(rows, ["class", "aircraft", "recall", "class ", "aircraft ", "recall "]),
            "Recall per G1 type (L24 CLS, kNN-5, maker held out).")
    best3 = rec[cl].astype(float).sort_values(ascending=False)
    zero = [disp(c) for c in cl if float(rec[c]) == 0]
    D.figure(fig_confusion(S_pat, Rp, FIG_DIR / "pat_confusion.png", (5.4, 4.3)), "Which G1 types are confused with which.",
             f"DINOv2-large with registers, frozen, 518 px; layer 24, CLS; exp3 main figure; {nP} aircraft; 12 G1 types.",
             "rows = true class (aircraft in brackets); columns = the class kNN-5 predicts; numbers = aircraft; the "
             "framed diagonal is correct; darker = larger share of the row.", width=72, origin="regenerated (TP label)")
    big = [c for c in best3.index if cnt12[c] >= 20]
    smallr = [c for c in best3.index if cnt12[c] < 20 and float(rec[c]) >= 0.3]
    pred = Rp["ev"]["pred"][KEY]
    wrong = pred != S_pat.y
    top2 = [c for c, _ in cnt12.sort_values(ascending=False).items()][:2]
    into = float(np.isin(pred[wrong], top2).mean())
    D.p(f"Among the types with 20 or more aircraft, {disp(big[0])} ({float(rec[big[0]]):.2f}) and {disp(big[1])} "
        f"({float(rec[big[1]]):.2f}) are read best"
        + (f". {', '.join(f'{disp(c)} ({float(rec[c]):.2f} on {int(cnt12[c])} aircraft)' for c in smallr)} "
           "scores high, but on very few aircraft" if smallr else "")
        + f". {', '.join(zero)} are never recovered. {pc(into)} of the wrong predictions fall into "
        f"{disp(top2[0])} or {disp(top2[1])}, the two largest classes, which is what a neighbour vote does when a "
        "class is thin in the pool.")
    D.h2("4.2 Clusters without labels (T5)")
    F.dir = FIG_DIR
    ER.fig_umap(F, S_pat, Rp)
    D.p(f"A k-means partition with 12 clusters matches the G1 types with an ARI of {mp.t5_ari:.3f} (0 = chance) at "
        f"L24 CLS, and no matrix exceeds {m.t5_ari.max():.3f}. The space is not organised by architecture first: "
        "whatever structure it has, the classes do not form their own clusters.")
    D.figure(F.items["f07_umap"]["path"], "The drawing embeddings in two dimensions, coloured by the 5-class parent.",
             f"DINOv2-large with registers, frozen, 518 px; layer 24, CLS; exp3 main figure; {nP} aircraft; UMAP "
             "n_neighbors 15, min_dist 0.1, cosine, seed 42.",
             "each point is one aircraft; only closeness is meaningful, the axes have no unit. The rotatable 3-D "
             "version is linked in the appendix.", width=72, origin="NEW render (no baked text)")
    D.h2("4.3 What else the embedding encodes: the view (T6)")
    cb = conf_p.loc["L24 CLS"]
    ER.fig_confound(F, S_pat, Rp)
    D.p(f"The same separation test, run once with the architecture and once with the **view group** of the drawing "
        f"(Perspective, Plan, Side, Front/Rear) on all {n_(len(wv))} whole-aircraft figures, shows the view moving the "
        f"vectors more than the architecture in all 6 matrices. At L24 CLS the view d is {cb.conf_d:.2f} against "
        f"{cb.arch_d:.2f} for the architecture, a ratio of {cb.conf_over_arch:.2f}; at the earlier layers the ratio "
        f"reaches {conf_p.conf_over_arch.max():.1f}. What the drawing's viewpoint is, the model encodes first.")
    D.figure(F.items["f09_confound"]["path"], "The view of the drawing against the architecture.",
             f"DINOv2-large with registers, frozen, 518 px; 6 matrices; all {n_(len(wv))} whole-aircraft figures of "
             f"the {nP} aircraft; same-maker pairs left out.",
             "d = how much further apart two figures of different labels sit than two of the same label, in standard "
             "deviations; a grey bar taller than the black one means the view moves the vectors more than the "
             "architecture.", width=85, origin="NEW render (no baked text)")
    D.h2("4.4 Does the architecture survive removing the view?")
    T = TAGR
    ga = "architecture kNN-5, G1 12-class"
    pa_ = "architecture kNN-5, parent 5-class"

    def vrow(lbl, sec):
        o = ve_get(C, T, sec, "original", "bal_acc")
        e = ve_get(C, T, sec, "view erasure", "bal_acc")
        d = ve_get(C, T, sec, "view erasure - original", "paired_diff")
        v = ve_get(C, T, sec, "variance-matched random - original", "paired_diff")
        return (lbl, int(o.n), f"{o.value:.3f}", f"{e.value:.3f}", f"{sg(d.value)} {ci(d.ci_lo, d.ci_hi, 3, True)}",
                f"{sg(v.value)} {ci(v.ci_lo, v.ci_hi, 3, True)}")
    vp_o = ve_get(C, T, "view probe (aircraft main figures)", "original", "probe_bal_acc").value
    vp_e = ve_get(C, T, "view probe (aircraft main figures)", "view erasure", "probe_bal_acc").value
    vk_e = ve_get(C, T, "view kNN-5 (figures, maker held out)", "view erasure", "bal_acc")
    vk_o = ve_get(C, T, "view kNN-5 (figures, maker held out)", "original", "bal_acc")
    b12 = ve_get(C, BASE_TAG, ga, "view erasure - original", "paired_diff")
    b5 = ve_get(C, BASE_TAG, pa_, "view erasure - original", "paired_diff")
    e12 = ve_get(C, BASE_TAG, ga, "view erasure", "bal_acc").value
    e5 = ve_get(C, BASE_TAG, pa_, "view erasure", "bal_acc").value
    r12 = ve_get(C, T, ga, "view erasure", "bal_acc").value
    r5 = ve_get(C, T, pa_, "view erasure", "bal_acc").value
    D.p(f"A linear erasure answers whether the architecture signal only rides on the view. The three directions that "
        f"separate the four view means are fitted on the {n_(len(wv))} figures and projected out, and the aircraft "
        f"vectors are rebuilt. The linear view probe on the main figures falls from {vp_o:.2f} to {vp_e:.2f} "
        "(chance 0.25). A **variance-matched control** removes three random directions of the same variance, so a loss "
        "that is not view-specific shows up there too. The Perspective-only rows hold the view constant by design.")
    D.table(md_df([vrow("G1, 12 classes", ga), vrow("parent, 5 classes", pa_),
                   vrow("G1, Perspective main figures only", ga + ", Perspective main figures only"),
                   vrow("parent, Perspective only", pa_ + ", Perspective main figures only")],
                  ["label set", "aircraft", "original", "view erased", "paired Δ [95 % CI]", "variance-matched Δ"]),
            "Architecture before and after the view erasure (registers, L24 CLS, kNN-5, maker held out).")
    D.p(f"With the registers model the erasure costs {abs(ve_get(C, T, ga, 'view erasure - original', 'paired_diff').value):.3f} "
        f"(12 classes) and {abs(ve_get(C, T, pa_, 'view erasure - original', 'paired_diff').value):.3f} (5 classes), "
        "and both intervals lie below zero. The loss is not specific to the view: the variance-matched removal costs "
        "about as much, and the same loss appears when every aircraft is drawn in Perspective, where view cannot be "
        "what the neighbours match on. Every erased score stays far above chance. The **base model is unchanged** by "
        f"the same erasure ({sg(b12.value)} {ci(b12.ci_lo, b12.ci_hi, 3, True)} and {sg(b5.value)} "
        f"{ci(b5.ci_lo, b5.ci_hi, 3, True)}).",
        f"After erasure the two models score alike (12 classes: registers {r12:.3f}, base {e12:.3f}; 5 classes: "
        f"{r5:.3f} against {e5:.3f}). Part of the registers advantage therefore sits in the dominant, high-variance "
        "directions the view also uses. The erasure is linear only: a non-linear view signal survives, since the view "
        f"kNN falls from {vk_o.value:.2f} to {vk_e.value:.2f} and stays well above its chance of "
        f"{vk_e.chance_p95:.2f}.")
    D.h2("4.5 Does the choice of image matter? (T7)")
    e4 = exp4_subset(C)
    info["exp4"] = e4
    D.table(md_df([(r, f"{br.loc[r].knn_bal_acc:.3f}", ci(br.loc[r].ci_lo, br.loc[r].ci_hi)) for r in S_pat.rules],
                  ["image rule", "kNN-5", "95 % interval"]), "The architecture score under each image rule (registers, "
            "L24 CLS, 12 classes, maker held out).")
    D.p(f"The rules span {f2(br.knn_bal_acc.min())} to {f2(br.knn_bal_acc.max())}, and every interval overlaps the "
        "others. The labeller's main figure scores highest, and the rule does not change the conclusion. Averaging "
        f"the view slots (exp4) matters only for the {e4['n']} aircraft with two or three slots. On those aircraft, "
        f"read from the full neighbour pool, exp4 is right for {pc(e4['acc']['exp4 view average'])} against "
        f"{pc(e4['acc']['exp1 view-first'])} for the view-first figure and {pc(e4['acc']['exp3 main figure'])} for the "
        f"main figure, a paired difference to view-first of {sg(e4['diff'])} {ci(e4['lo'], e4['hi'], 3, True)}. "
        "Averaging several views does not add a measurable architecture signal; the slots sub-question stays a null.")

    # ═══ 5 Photos ═══
    D.h1("5. Results: photographs")
    mm = Rh["ev"]["main"]
    D.p(f"**Soundness and separation (T1 to T3).** No invalid row, and PC1 carries {mm.t2_pc1_vs_random.min():.0f} to "
        f"{mm.t2_pc1_vs_random.max():.0f} times the random variance. Same-class photos sit closer in all 6 matrices, "
        f"with a larger effect than on the drawings: ratio {mh.t3_ratio:.3f} and d {mh.t3_d:.2f} at L24 CLS "
        f"(drawings: d {mp.t3_d:.2f}).")
    D.h2("5.1 Reading the architecture (T4)")
    D.table(md_df([(r.matrix, f"{r.knn_bal_acc:.3f}", ci(r.knn_ci_lo, r.knn_ci_hi), f"{r.knn_chance_p95:.3f}",
                    f"{r.probe_bal_acc:.3f}") for r in mm.itertuples()],
                  ["matrix", "kNN-5", "95 % interval", "shuffled p95", "probe"]),
            f"Prediction of the directory class from the photos ({n_(nH)} aircraft, hero image, maker held out, "
            "chance 1/5).")
    Fh = DocFigs(FIG_DIR, "pho")
    ER.fig_prediction(Fh, S_evn, Rh)
    D.figure(Fh.items["f04_prediction"]["path"], "Architecture read from the photos, per layer and token (blue: the "
             "matrix carried forward).",
             f"DINOv2-large with registers, frozen, 518 px; layers 18/22/24 × CLS / patch mean; hero image; "
             f"{n_(nH)} aircraft; label: directory class, 5 classes.",
             "bars = kNN-5 balanced accuracy with 95 % interval; diamonds = linear probe; dashed line = shuffled-label "
             "chance.", width=80, origin="NEW render (no baked text)")
    rh = Rh["ev"]["recall"]
    rh = rh[rh.matrix == "L24 CLS"].iloc[0]
    rbest = max(ER.CLASSES5, key=lambda c: float(rh[c]))
    rworst = min(ER.CLASSES5, key=lambda c: float(rh[c]))
    s1r = C["s1a_candidates"]
    lab_l22 = float(s1r[(s1r.pool == "registers only") & (s1r.label == "photos 5-class") &
                        (s1r.candidate == "registers L22 CLS")].selection_freq.iloc[0])
    D.p(f"At L24 CLS the photos reach {f2(mh.knn_bal_acc)} {ci(mh.knn_ci_lo, mh.knn_ci_hi)} against a shuffled level "
        f"of {f2(mh.knn_chance_p95)}, and the probe {f2(mh.probe_bal_acc)}. L22 CLS scores within the interval "
        f"({f2(main_row(Rh, matrix='L22 CLS').knn_bal_acc)}), which is why the split-half rule chose it for the photos "
        f"in {pc(lab_l22)} of the selections. {ER.CLASS5_NAME[rbest]} is read best (recall {float(rh[rbest]):.2f}) "
        f"and {ER.CLASS5_NAME[rworst]} worst ({float(rh[rworst]):.2f}).")
    D.figure(fig_confusion(S_evn, Rh, FIG_DIR / "pho_confusion.png", (3.9, 3.0)), "Which directory classes are confused "
             "with which.",
             f"DINOv2-large with registers, frozen, 518 px; layer 24, CLS; hero image; {n_(nH)} aircraft.",
             "rows = true class (aircraft in brackets); columns = the predicted class; numbers = aircraft; darker = "
             "larger share of the row.", width=55, origin="NEW render (no baked text)")
    D.h2("5.2 Clusters and what else is encoded (T5, T6)")
    ch = conf_h.loc["L24 CLS"]
    ER.fig_umap(Fh, S_evn, Rh)
    D.p(f"k-means with 5 clusters matches the directory classes with an ARI of {mh.t5_ari:.3f} at L24 CLS: weak, but "
        f"{mh.t5_ari / mp.t5_ari:.0f} times the drawings' {mp.t5_ari:.3f}. For T6 the non-architecture label is the **maturity status** "
        f"given on the page (built or concept, {int(S_evn.aircraft.status.isin(['built', 'concept']).sum())} "
        f"aircraft). At L24 CLS the status d is {ch.conf_d:.2f} against {ch.arch_d:.2f} for the architecture, a "
        f"ratio of {ch.conf_over_arch:.2f}: in the selected matrix the architecture moves the photo vectors more than "
        f"whether the aircraft was built. In {int((conf_h.conf_over_arch > 1).sum())} of the 6 matrices the status "
        "dominates, so the order depends on the layer.")
    D.figure(Fh.items["f07_umap"]["path"], "The photo embeddings in two dimensions, coloured by the directory class.",
             f"DINOv2-large with registers, frozen, 518 px; layer 24, CLS; hero image; {n_(nH)} aircraft; UMAP "
             "n_neighbors 15, min_dist 0.1, cosine, seed 42.",
             "each point is one aircraft; only closeness is meaningful, the axes have no unit.", width=72,
             origin="NEW render (no baked text)")
    D.h2("5.3 Background and image choice")
    ck = C["crop_knn"].set_index(["matrix", "condition"])
    cd = C["crop_diff"].set_index("matrix")
    D.p(f"Photographs carry broad backgrounds (sky, city, hangar) that drawings do not. Each hero image was cropped to "
        f"the aircraft with a zero-shot OWLv2 detector ({n_fb} of {len(cm)} images, {pc(n_fb / len(cm), 1)}, kept "
        "whole because the box was weak or tiny), re-processed and re-embedded. **This test ran on the base model** "
        "(dinov2-large), before the registers model was adopted; it is the one result of this document on base.")
    D.table(md_df([(mx, f"{ck.loc[(mx, 'original')].knn_bal_acc:.3f}", f"{ck.loc[(mx, 'cropped')].knn_bal_acc:.3f}",
                    f"{sg(cd.loc[mx].diff_cropped_minus_original)} {ci(cd.loc[mx].ci_lo, cd.loc[mx].ci_hi, 3, True)}")
                   for mx in ("L24 CLS", "L24 patch mean")],
                  ["matrix", "original", "cropped", "paired difference [95 % CI]"]),
            f"Cropping the photos to the aircraft (base model, {n_(len(cm))} aircraft, kNN-5, maker held out).")
    D.p(f"At L24 CLS cropping moves the score by {sg(cd.loc['L24 CLS'].diff_cropped_minus_original)}, inside its "
        "interval: the photo result measures the aircraft and not the scenery. Only the patch mean, which averages "
        "over background patches, gains a little from the crop.",
        f"The image rule matters as little as on the drawings: hero image {f2(brh.loc['hero image'].knn_bal_acc, 3)}, "
        f"one random other image {f2(brh.loc['random other image'].knn_bal_acc, 3)}, the mean of all kept images "
        f"{f2(brh.loc['mean of all images'].knn_bal_acc, 3)}, against an interval about "
        f"{mh.knn_ci_hi - mh.knn_ci_lo:.2f} wide for one rule.")

    # ═══ 6 Drawings vs photos ═══
    D.h1("6. Drawings against photographs")
    D.p("The comparison uses the same model, the same readout, the same protocol and the 5 classes the two sources "
        "share: each G1 type folds into its parent. The aircraft are different in the two sets, so the interval of a "
        "difference comes from independent bootstrap resamples of each set.")
    rows = [(f"photos {p}", f"{r.drawings_bal_acc:.3f}", f"{r.photos_bal_acc:.3f}",
             f"{sg(r.diff_photos_minus_drawings)} {ci(r.ci_lo, r.ci_hi, 3, True)}") for p, r in fair.iterrows()]
    D.table(md_df(rows, ["comparison", "drawings (plain)", "photos", "photos − drawings [95 % CI]"]),
            f"Photos against drawings on the shared 5 classes (registers, L24 CLS, kNN-5; {nP} drawing aircraft, "
            f"{n_(nH)} photo aircraft).")
    fci = fc.set_index("class")
    dA, pA = fair.loc["colour"].drawings_bal_acc, fair.loc["colour"].photos_bal_acc
    D.p(f"The photos lead by {sg(fair.loc['colour'].diff_photos_minus_drawings, 2)}, and by "
        f"{sg(fair.loc['grey'].diff_photos_minus_drawings, 2)} in greyscale: whatever the photo advantage is, it is "
        f"not colour. The probes point the same way ({f2(m5.probe_bal_acc)} for drawings, {f2(mh.probe_bal_acc)} for "
        "photos).",
        "The two sets differ in size and in class mix, and a neighbour vote depends on its pool: a class that is thin "
        "in the pool is hard to recover, whatever the medium. Balanced accuracy weights every class equally, but it "
        f"cannot make a thin class easy. To size this, the photo score was recomputed on {cmx['draws']} random photo "
        f"subsets with exactly the drawings' class counts ({cmx['n']} aircraft each; the probe on the first "
        f"{cmx['probe_draws']}), so that pool size and class mix match (Table {D.ntab + 1}).")
    rows = [(c, int(fci.loc[c, "n_draw"]), f"{fci.loc[c, 'rec_draw']:.2f}", int(fci.loc[c, "n_photo"]),
             f"{fci.loc[c, 'rec_photo']:.2f}", f"{cmx['recall'][c]:.2f}") for c in ER.CLASSES5]
    rows.append(("balanced accuracy", nP, f"{dA:.3f}", nH, f"{pA:.3f}",
                 f"{cmx['mean']:.3f} [{cmx['lo']:.3f}, {cmx['hi']:.3f}]"))
    rows.append(("probe", nP, f"{m5.probe_bal_acc:.3f}", nH, f"{mh.probe_bal_acc:.3f}",
                 f"{cmx['probe_mean']:.3f} [{cmx['probe_lo']:.3f}, {cmx['probe_hi']:.3f}]"))
    D.table(md_df(rows, ["class", "drawings", "recall", "photos", "recall ", "photos, drawings' mix"]),
            "Recall per class on the shared 5 classes: drawings, all photos, and photo subsets with the drawings' "
            "class counts (registers, L24 CLS, kNN-5, maker held out; last column: mean over the draws, [2.5, 97.5] "
            "percentiles of the draws).")
    inside = cmx["lo"] <= dA <= cmx["hi"]
    pin = cmx["probe_lo"] <= m5.probe_bal_acc <= cmx["probe_hi"]
    D.p((f"Matched in size and class mix, the photos score {cmx['mean']:.3f}, and the drawings' {dA:.3f} lies inside "
         f"the range of the draws ({cmx['lo']:.3f} to {cmx['hi']:.3f}). Most of the {sg(pA - dA, 2)} advantage of the "
         "full photo set therefore comes from its larger and more balanced pool, and on this evidence the medium "
         "itself is not shown to carry more architecture." if inside else
         f"Matched in size and class mix, the photos still score {cmx['mean']:.3f}, and the drawings' {dA:.3f} falls "
         f"outside the range of the draws ({cmx['lo']:.3f} to {cmx['hi']:.3f}): the photo advantage survives the "
         "matching.")
        + (f" The probe tells the same story ({cmx['probe_mean']:.3f} matched against {m5.probe_bal_acc:.3f})."
           if pin == inside else
           f" The probe disagrees: matched photos {cmx['probe_mean']:.3f} "
           f"[{cmx['probe_lo']:.3f}, {cmx['probe_hi']:.3f}] against {m5.probe_bal_acc:.3f} for the drawings."))
    D.figure(fig_five_class(C, fc, cmx), "The shared 5 classes, drawings against photos: balanced accuracy and recall "
             "per class.",
             f"DINOv2-large with registers, frozen, 518 px; layer 24, CLS; kNN-5, maker held out; drawings: exp3 main "
             f"figure, {nP} aircraft, G1 folded to its parent; photos: hero image, {n_(nH)} aircraft; matched: "
             f"{cmx['draws']} photo subsets of {cmx['n']} aircraft with the drawings' class counts.",
             "hatched = patent drawings, solid = all photos, pale = photos with the drawings' class mix (whisker: "
             "range of the draws); class colours as in the maps; under each class the aircraft in drawings | photos; "
             "dashed line = chance 1/5.", width=85, origin="NEW")
    D.p(f"The per-class view shows where the pool matters. Wingless multicopters are {fci.loc['WM', 'n_draw']} "
        f"drawings but {fci.loc['WM', 'n_photo']} photos. The photos read them at {fci.loc['WM', 'rec_photo']:.2f} "
        f"and the drawings at {fci.loc['WM', 'rec_draw']:.2f}, and with the drawings' mix the photos fall to "
        f"{cmx['recall']['WM']:.2f}. Vectored thrust, the dominant drawing class ({fci.loc['VT', 'n_draw']} of {nP}), "
        f"goes the other way: the drawings read it at {fci.loc['VT', 'rec_draw']:.2f} against "
        f"{fci.loc['VT', 'rec_photo']:.2f} for all photos, and the matched photos "
        f"{'rise' if cmx['recall']['VT'] > fci.loc['VT', 'rec_photo'] else 'move'} to {cmx['recall']['VT']:.2f}."
        + (" In both sources a class is read well largely when it dominates the pool."
           if cmx['recall']['VT'] > fci.loc['VT', 'rec_photo'] and cmx['recall']['WM'] < fci.loc['WM', 'rec_photo']
           else ""))
    D.h2("6.1 The same aircraft in both sources")
    fp, mr = fig_matching(C)
    mpp = mt.loc["patent->photo"]
    mph = mt.loc["photo->patent"]
    D.p(f"For the {int(mpp.queries)} patent aircraft that have a directory page, the main figure queried all "
        f"{n_(int(mpp.gallery))} photo aircraft. Line drawings were removed from the gallery because some directory "
        "pages reproduce the patent figure itself.")
    D.table(md_df([("patent → photo", int(mpp.queries), int(mpp.gallery), pc(mpp['R@1'], 1), pc(mpp['R@10'], 1),
                    pc(mpp['R@50'], 1), f"{pc(mpp['random_R@10'], 1)} / {pc(mpp['random_R@50'], 1)}", f"{mpp.median_rank:g}"),
                   ("photo → patent", int(mph.queries), int(mph.gallery), pc(mph['R@1'], 1), pc(mph['R@10'], 1),
                    pc(mph['R@50'], 1), f"{pc(mph['random_R@10'], 1)} / {pc(mph['random_R@50'], 1)}", f"{mph.median_rank:g}")],
                  ["direction", "queries", "gallery", "top 1", "top 10", "top 50", "random top 10 / 50", "median rank"]),
            "Same-aircraft matching (registers, L24 CLS, no line drawings in the gallery).")
    D.figure(fp, "How often a patent aircraft finds its own directory page within the top k photos.",
             f"DINOv2-large with registers, frozen, 518 px; layer 24, CLS; {int(mpp.queries)} patent main figures "
             f"as queries against {n_(int(mpp.gallery))} photo aircraft without line drawings.",
             "the blue curve is the share of queries whose own page is ranked within the top k; the dashed curve is a "
             "random ranking with the same number of own pages per query.", width=80, origin="NEW")
    D.p(f"The model links a drawing to a photograph of the same aircraft well above chance: the right page is in the "
        f"top 10 for {pc(mpp['R@10'])} of the queries and in the top 50 for {pc(mpp['R@50'])}, against "
        f"{pc(mpp['random_R@10'], 1)} and {pc(mpp['random_R@50'], 1)} at random. It is far from a retrieval tool: the "
        f"median rank is {mpp.median_rank:g}, and the medium separates the sources more than the aircraft joins them.")

    # ═══ 7 Attention ═══
    D.h1("7. Where the model looks")
    fpa, names = fig_attention_panel(C)
    info["attention_panel"] = names
    D.p("The attention maps show where the CLS token of the registers model reads each image. They are a qualitative "
        "check, not a test: two aircraft per class and source were drawn at random, and the maps are read by eye. "
        f"Across all images, the median share of the CLS attention on the single largest patch stays at or below "
        f"{max(sh.loc[('registers', s, L)].max_patch_share_median for s in ('patent', 'photo') for L in (18, 24)):.3f} "
        "at layers 18 and 24, and the register tokens take "
        f"{min(sh.loc[('registers', s, L)].prefix_share_median for s in ('patent', 'photo') for L in (18, 24)):.2f} to "
        f"{max(sh.loc[('registers', s, L)].prefix_share_median for s in ('patent', 'photo') for L in (18, 24)):.2f} of "
        "it, so no single background patch dominates.")
    D.figure(fpa, "CLS attention of the registers model at layers 18 and 24, two aircraft per class and source.",
             f"DINOv2-large with registers, frozen, 518 px; CLS query against every patch key, mean over heads, each "
             f"map scaled to its own 99th percentile; drawings: main figure of {len(C['S_pat'].aircraft)} aircraft, "
             f"photos: hero image of {n_(nH)} aircraft; 2 per class and source drawn at random (seed 42).",
             "bright = where the CLS token looks; each drawing is labelled with its G1 type, each photo with its "
             "directory class; rows are the 5 parent classes.", origin="NEW", private=True)
    info["fig_att_no"] = D.nfig
    D.p("Read by eye, layer 18 traces the line-work of the drawings almost everywhere, outline, propulsors and "
        "sometimes the figure label alike, and covers the airframe of the photos. Layer 24 is sparser: its weight "
        "gathers in spots on the aircraft, often on propulsors, hubs and the fuselage, with a scatter of weak spots on "
        "the white pad or the sky. In one wingless photo the people in front of the aircraft draw as much attention as "
        "the aircraft itself, a reminder that a kept hero image can still be a scene. On neither source does the "
        "attention settle on one part that names the architecture. That is consistent with the modest kNN scores: "
        "the late CLS token summarises the whole shape and the way it is drawn, not the arrangement of lift and "
        "thrust. These readings rest on 20 aircraft and are not a measurement.")

    # ═══ 8 Conclusions ═══
    D.h1("8. Conclusions")
    mix_clause = (", an advantage that shrinks to nothing measurable once the photo pool has the drawings' size and "
                  "class mix") if cmx_inside else ""
    D.p(f"**RQ4.** The figure alone carries the architecture: weakly in patent drawings, and moderately in the "
        f"photographs as they come{mix_clause}. "
        f"A frozen general-purpose vision model, never trained on aircraft or on the codebook, recovers the 12 G1 "
        f"types from one drawing per aircraft at {f2(mp.knn_bal_acc)} against a chance level of "
        f"{f2(mp.knn_chance_p95)}, and the 5 parent classes at {f2(m5.knn_bal_acc)} from drawings and "
        f"{f2(mh.knn_bal_acc)} from photographs. The result holds with the maker held out, under every image rule, "
        "without the photo backgrounds, in greyscale, and after the view of the drawing is linearly removed.",
        f"The drawing embedding is organised first by **how** the aircraft is drawn and only second by **what** it "
        f"is: the view separates the vectors {f2(cb.conf_over_arch)} times more than the architecture, and the classes "
        "form no clusters of their own. The architecture is present in the vector (the probe reads more than the "
        "neighbours) but it does not decide which aircraft are nearest. The codebook's human reading therefore stays "
        "the reference for the architecture; the embedding offers a second, independent and label-free view of the "
        "same corpus.",
        "**What DINOv2 contributes to the thesis.** It tests the taxonomy against an observer that has never seen it: "
        "the types the model recovers best are the large ones, and the types it never recovers "
        f"({', '.join(zero)}) are defined by how the propulsion moves or acts, which a single silhouette need not "
        "show. It links the patent record "
        "to the directory of built and proposed aircraft above chance. And it delivers a labelled benchmark of "
        "whole-aircraft drawings with a fixed protocol, on which a fine-tuned or aircraft-specific model can be "
        "measured.",
        "**A reusable recipe for patent-drawing studies.** Use a registers checkpoint, because drawings are mostly "
        "blank background and a model without registers parks its artifact tokens there. Feed the largest native "
        "input size (518 px) on a white pad, and leave the lines as drawn. Choose the layer and token with the "
        "split-half rule over makers and report the optimism gap beside the score. Hold the maker out of every "
        "neighbour, and test the view as a confound before reading any class result.")

    # ═══ 9 Limitations ═══
    D.h1("9. Limitations")
    lim = [
        f"**Linear view erasure only.** The erasure removes the linear view signal. The view kNN stays at "
        f"{vk_e.value:.2f} after it, so a non-linear view component survives. The Perspective-only rows hold the view "
        "constant, but only for one view.",
        f"**Small and unbalanced classes.** {', '.join(f'{disp(c)} {int(cnt12[c])}' for c in sorted(small, key=lambda c: cnt12[c]))} "
        f"aircraft: these G1 recalls rest on a handful of aircraft. The photo classes range from "
        f"{int(cnt5h.min())} ({cnt5h.idxmin()}) to {int(cnt5h.max())} ({cnt5h.idxmax()}), and the two sources differ in "
        "class mix (§6).",
        "**The photo filter audit is not blind.** The sheets showed the filter score, the judge was the AI assistant of "
        f"the pipeline, and the author has not yet re-checked the {len(au)} verdicts. The {n_rev} review-band images "
        "were never decided by hand.",
        "**A gentler line thickening is untested.** The smallest candidate grew each stroke side by about 1 px. A "
        "sub-pixel variant was not in the candidate set fixed in advance.",
        f"**Labels.** The photo label is the directory's own class, which agrees with the codebook parent for "
        f"{int(pch.agrees.sum())} of {len(pch)} linked aircraft. The drawing labels are human labels from the wizard, "
        f"checked against the whole-patent reading (κ {kap:.2f}); the embedding reports record no second-labeller "
        "agreement.",
        "**One frozen model.** The analysis measures what a general-purpose model sees without training. A fine-tuned "
        "model, or a model trained on technical drawings, is outside the scope of this chapter.",
    ]
    D.L += [f"- {s}" for s in lim] + [""]

    # ═══ 10 Open questions ═══
    D.h1("10. Open questions for the author")
    oq = [
        f"**Matching depth (N5).** The matching result is above chance ({pc(mpp['R@50'])} in the top 50) but far "
        "from reliable. Keep §6.1 as a section of the chapter, or reduce it to one paragraph?",
        f"**Which registers number is the headline?** The drawings' 12-class score is {f2(mp.knn_bal_acc, 3)}, but "
        f"{f2(mp.knn_bal_acc - r12, 3)} of it does not survive the view erasure, where the two models tie. Quote "
        f"{f2(mp.knn_bal_acc, 3)} with the erasure as a qualification (as here), or lead with the erased score?",
        "**The exp4 slots sub-question.** It stays a null (§4.5). The chapter plan keeps it for now. Drop it from the "
        "chapter?",
        "**Attention layer for the chapter figure (N4).** The plan named layer 18. With the registers model layer 24 "
        f"is readable too. Figure {info['fig_att_no']} shows both: keep both, or one?",
        (f"**The photo advantage and the class mix.** With the drawings' size and class mix the photos score "
         f"{cmx['mean']:.3f} against the drawings' {fair.loc['colour'].drawings_bal_acc:.3f} (§6). REVIEW_SUMMARY "
         "finding 3 (photos carry the architecture better) needs rewording: keep the full-set comparison as the "
         "headline with this as its qualification, or make the matched comparison the headline?" if cmx_inside else
         "**The photo advantage** survives the class-mix matching (§6); keep finding 3 as it stands?"),
        "**Labeller wording.** The reports do not say whether the G1 labels come from one labeller. Should §9 state "
        "it, and is a second-labeller check planned?",
    ]
    D.L += [f"- {s}" for s in oq] + [""]

    # ═══ Appendix ═══
    D.h1("Appendix: full grids and the 3-D pages")
    D.p("Every matrix under both models, kNN-5 balanced accuracy and linear probe, maker held out. Base appears here "
        "only as the comparison. All text above uses the registers model. Every number in this document is computed "
        "by embedding_evaluation/src/analysis_document.py (run by scripts/build_analysis_document.py) from the saved "
        "results listed in its docstring.")

    def grid(Rb, Rr, key, caption):
        mb, mr = Rb[key]["main"].set_index("matrix"), Rr[key]["main"].set_index("matrix")
        rows = [(mx, f"{mb.loc[mx].knn_bal_acc:.3f}", f"{mr.loc[mx].knn_bal_acc:.3f}", f"{mb.loc[mx].probe_bal_acc:.3f}",
                 f"{mr.loc[mx].probe_bal_acc:.3f}") for mx in mr.index]
        D.table(md_df(rows, ["matrix", "kNN-5 base", "kNN-5 registers", "probe base", "probe registers"]), caption)
    grid(Bp, Rp, "ev", f"Patent drawings, 12 G1 classes ({nP} aircraft, exp3 main figure).")
    grid(Bp, Rp, "ev5", f"Patent drawings, 5 parent classes ({nP} aircraft).")
    grid(Bh, Rh, "ev", f"Photos, 5 directory classes ({n_(nH)} aircraft, hero image).")
    from urllib.parse import quote
    D.p("**Rotatable 3-D maps** (single HTML files; open in a browser):")
    D.L += [f"- {name}: [{p.name}](file://{quote(str(p))})  \n  `{p}`" for name, p in PAGES_3D] + [""]
    return D.text(), {"figs": D.fig_log, "ntab": D.ntab, **info}


def write_markdown() -> Tuple[Path, Dict]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PRIV_DIR.mkdir(parents=True, exist_ok=True)
    C = load()
    text, info = build(C)
    if "—" in text.replace("<<SECTION>>", ""):
        bad = [l for l in text.split("\n") if "—" in l]
        raise ValueError(f"em dash in the text: {bad[:3]}")
    MD_PATH.write_text(text.replace("<<SECTION>>\n", ""), encoding="utf-8")
    (OUT_DIR / "_sections.md").write_text(text, encoding="utf-8")      # with the section markers, for the Word step
    return MD_PATH, info
