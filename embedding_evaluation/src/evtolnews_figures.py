"""Figures for the evtol.news advisor report (src/evtolnews_advisor.py).

Every figure is a PNG in ``<evtolnews.root>/3_embedding_evaluation/advisor_report/figs/`` and
carries its own source statement printed along the bottom edge: model, extraction image size,
layer, pooling, set and population. The same statement is repeated in the caption, so a figure
copied out of the document still says what it shows.

Colour is fixed per entity for the whole document and was checked with the dataviz palette
validator (all-pairs, light surface):

    classes   VT blue · LC violet · WM green · ER yellow · HB magenta (+ one marker shape each)
    sources   photos = dark ink · patents = grey ink (never a class hue)
    clusters  orange · aqua · ink, each with its own marker and a C1/C2/C3 label

Runs in the Finetune environment (umap-learn): /home/vasco/anaconda3/envs/Finetune/bin/python.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from PIL import Image  # noqa: E402

from . import embedding_metrics as em  # noqa: E402
from . import evtolnews_eval as ev  # noqa: E402

CLASSES = ev.CLASSES
CLASS_COLOR = {"VT": "#2a78d6", "LC": "#4a3aa7", "WM": "#008300", "ER": "#eda100", "HB": "#e87ba4"}
CLASS_MARK = {"VT": "o", "LC": "s", "WM": "^", "ER": "D", "HB": "v"}
CLASS_SHORT = {"VT": "Vectored Thrust", "LC": "Lift + Cruise", "WM": "Wingless (Multicopter)",
               "ER": "Electric Rotorcraft", "HB": "Hover Bikes / PFD"}
INK = {"photo": "#1f1f1e", "patent": "#a3a29d"}
CLUSTER = [("#eb6834", "o"), ("#1baf7a", "s"), ("#52514e", "^")]
TEXT, MUTED, FAINT, GRID = "#0b0b0b", "#52514e", "#c9c8c3", "#ecebe7"
MODEL = "DINOv2-large, frozen"
REF = ("dinov2-large_518", 24, "cls")
MATRICES = [(18, "cls"), (18, "mean_patch"), (22, "cls"), (22, "mean_patch"), (24, "cls"), (24, "mean_patch")]
# well-known aircraft shown as the "kept" example of their class (fallback: highest filter score)
EXAMPLE_PREFERRED = {"VT": "joby-aviation-s4-production-prototype", "WM": "volocopter-2x"}
# directory images seen to be line drawings although SigLIP scores them as renders
NOT_A_PHOTO = ["/kitty-hawk-cora/"]
SET_DESC = {"main": "set main (the page's first kept image, one per aircraft)",
            "all": "set all (every kept image)"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2, "axes.titlesize": 8.8, "axes.titleweight": "bold",
    "axes.labelsize": 8.2, "axes.edgecolor": "#b9b8b3", "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.fontsize": 7.5, "figure.facecolor": "white", "savefig.dpi": 200,
})


# ── source statements ───────────────────────────────────────────────────────
def size_of(tag: str) -> int:
    return int(tag.rsplit("_", 1)[1])


def pool_name(pool: str) -> str:
    return "CLS token" if pool == "cls" else "mean of the patch tokens"


def mat_name(layer: int, pool: str) -> str:
    return f"L{layer} {'CLS' if pool == 'cls' else 'patch mean'}"


def src(source: str, tag: str | None = None, layer: int | None = None, pool: str | None = None,
        set_name: str | None = None, n: str = "", extra: str = "") -> str:
    """One-line statement of what a figure was computed from."""
    who = {"photo": "evtol.news photos", "patent": "patent figures (1639_LABELLED)",
           "both": "evtol.news photos and patent figures", "data": "evtol.news directory crawl"}[source]
    parts = [who]
    if tag:
        parts.append(f"{MODEL}, notebook 22 extraction at {size_of(tag)} px")
    if layer is not None:
        parts.append(f"layer {layer}, {pool_name(pool)}")
    elif pool == "all":
        parts.append("layers 18/22/24 x CLS / patch mean")
    if set_name:
        parts.append(SET_DESC.get(set_name, f"set {set_name}"))
    if n:
        parts.append(n)
    if extra:
        parts.append(extra)
    return "; ".join(parts)



# How to read each figure whose axes are not self-explanatory; printed under the figure and in its caption.
_UMAP = ("each point is one aircraft (its main image). UMAP reduces each 1 024-number embedding to 2 numbers "
         "so that aircraft with similar embeddings sit close together. The two axes are these 2 numbers: "
         "they have no unit, no physical meaning and an arbitrary orientation. Only closeness is meaningful; "
         "positions on the axes and distances between far-apart groups are not")
READ = {
    "f07_cosine": "x = cosine similarity between the embeddings of two images (1 = identical); each curve is "
                  "the distribution over image pairs",
    "f08_pca": "x = principal component, in order of the variance it explains; y = share of the variance "
               "(log scale); dashed = a random matrix of the same shape",
    "f09_distance": "x = cosine distance between the embeddings of two images; filled = a random matrix of the "
                    "same shape",
    "f10_dendrogram": "each leaf at the bottom is one aircraft; branches join aircraft whose embeddings are close; "
                      "the height of a join (Ward distance) measures how different the two groups are; the "
                      "strip gives each leaf's directory class",
    "f11_umap_clusters": "left: " + _UMAP + ". Right: for each directory class, the share of its aircraft in "
                         "each label-free cluster",
    "f12_umap_class": _UMAP,
    "f13_umap_facets": _UMAP + ". Each panel highlights one class on the same map",
    "f14_umap_matrices": _UMAP + ". Each panel is a separate UMAP fit, so the maps are not aligned with each other",
    "f15_umap_photo_patent": _UMAP + ". The two panels are separate UMAP fits and cannot be compared point by point",
    "f16_umap_joint": _UMAP + ". One UMAP fit on both sources: left coloured by source, right by class",
    "f17_contingency": "rows = directory class; columns = the five k-means clusters found without labels; "
                       "numbers = aircraft; darker = larger share of the row",
    "f21_confusion": "rows = true class; columns = the class kNN predicts; numbers = aircraft; the diagonal is "
                     "correct; darker = larger share of the row",
    "f23_parent": "rows = the human topType of the patent aircraft; columns = the directory class of its page; "
                  "numbers = aircraft; outlined = the parent the codebook implies",
    "f24_recall_at_k": "x = k, how many of the top-ranked aircraft are looked at (log scale); y = share of queries "
                       "whose right aircraft is among them",
    "f27_attention_classes": "red opacity = how much the CLS token attends to each 14 x 14 px patch, scaled per "
                             "image; the image is shown in grey underneath",
    "f28_attention_pairs": "red opacity = how much the CLS token attends to each 14 x 14 px patch, scaled per "
                           "image; the image is shown in grey underneath",
    "f29_attention_collapse": "left: median share of the CLS attention held by the single most attended patch; "
                              "right: share of images whose most attended patch lies on the white padding",
}


class Figs:
    """Collects figures: PNG on disk + the record the document needs."""

    def __init__(self, out: Path):
        self.dir = out / "figs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.items: Dict[str, Dict[str, str]] = {}

    def save(self, fig, key: str, title: str, source: str, note: str = "", bottom: float = 0.035) -> None:
        read = READ.get(key, "")
        _stamp(fig, source, read)
        fig.tight_layout(rect=(0, bottom + (0.03 if read else 0), 1, 1))
        fig.savefig(self.dir / f"{key}.png", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        self.items[key] = {"file": f"figs/{key}.png", "title": title, "source": source, "note": note,
                           "read": read}


def _stamp(fig, source: str, read: str = "") -> None:
    """Source line (and the how-to-read line) along the figure's lower edge."""
    import textwrap
    w = int(fig.get_figwidth() * 23)
    lines = textwrap.wrap("Source: " + source, w)
    if read:
        lines += textwrap.wrap("How to read: " + read, w)
    fig.text(0.005, 0.004, "\n".join(lines), fontsize=5.8, color=MUTED, ha="left", va="bottom")


# ── data ────────────────────────────────────────────────────────────────────
class Data:
    """Everything the figures read, loaded once."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.root = ev.root(cfg)
        self.out = ev.out_dir(cfg)
        self.bundles = {(s, t): (ev.load_photo(cfg, t) if s == "photo" else ev.load_patent(cfg, t))
                        for s in ("photo", "patent") for t in cfg["evtolnews"]["embeddings"]}
        self.frames: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self.aircraft = pd.read_csv(self.root / "0_source/aircraft.csv", keep_default_na=False)
        self.images = pd.read_csv(self.root / "1_filter/image_decisions.csv", keep_default_na=False)
        self.cm = pd.read_csv(self.out / "class_metrics.csv")
        self.match = pd.read_csv(self.out / "matching.csv")
        self.ranks = pd.read_csv(self.out / "matching_ranks.csv")
        self.parent = pd.read_csv(self.out / "parent_check.csv", keep_default_na=False)
        self.funnel = pd.read_csv(self.root / "2_embedding_extraction/selection/funnel.csv",
                                  keep_default_na=False)

    def frame(self, source: str, tag: str, set_name: str = "main") -> Dict[str, Any]:
        k = (source, tag, set_name)
        if k not in self.frames:
            self.frames[k] = ev.labelled_frame(self.bundles[(source, tag)], source, set_name)
        return self.frames[k]

    def X(self, source: str, tag: str, layer: int, pool: str, set_name: str = "main") -> np.ndarray:
        return em._l2(self.frame(source, tag, set_name)["sub"]["arrays"][(layer, pool)].astype(np.float64))

    def plot_data(self, source: str, tag: str, set_name: str = "main") -> Dict[str, np.ndarray]:
        d = "metrics" if source == "photo" else "metrics_patent_reference"
        with np.load(self.out / d / tag / set_name / "plot_data.npz") as z:
            return {k: z[k] for k in z.files}

    def table(self, source: str, tag: str, name: str, set_name: str = "main") -> pd.DataFrame:
        d = "metrics" if source == "photo" else "metrics_patent_reference"
        return pd.read_csv(self.out / d / tag / set_name / name)

    def umap(self, key: str, X: np.ndarray) -> np.ndarray:
        f = self.out / "advisor_report" / "cache" / f"umap_{key}.npy"
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists() and np.load(f).shape[0] == len(X):
            return np.load(f)
        import umap
        emb = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                        random_state=int(self.cfg.get("seed", 42))).fit_transform(X)
        np.save(f, emb)
        return emb


def _fmt_int(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ── scatter helpers ─────────────────────────────────────────────────────────
def _scatter_classes(ax, emb: np.ndarray, y: np.ndarray, s: float = 7, alpha: float = 0.75,
                     labels: bool = True) -> None:
    ax.scatter(emb[:, 0], emb[:, 1], s=s * 0.6, c=FAINT, lw=0, alpha=0.35, zorder=1)
    placed: List[Tuple[float, float]] = []
    for c in CLASSES:
        m = y == c
        if not m.any():
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=s, c=CLASS_COLOR[c], marker=CLASS_MARK[c], lw=0.25,
                   edgecolors="white", alpha=alpha, zorder=2, label=f"{c} ({m.sum()})")
        if labels:
            mx, my = np.median(emb[m], axis=0)
            span = np.ptp(emb, axis=0)
            for px, py in placed:          # nudge a label off one already placed nearby
                if abs(mx - px) < 0.07 * span[0] and abs(my - py) < 0.05 * span[1]:
                    my = py - 0.06 * span[1]
            placed.append((mx, my))
            ax.text(mx, my, c, fontsize=8, weight="bold", color=TEXT, ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=CLASS_COLOR[c], lw=0.8, alpha=0.9))
    _umap_axes(ax)


def _umap_axes(ax, small: bool = False) -> None:
    """UMAP axes: labelled, no ticks (the values carry no meaning)."""
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    fs = 5.8 if small else 7
    ax.set_xlabel("UMAP dimension 1 (no unit)", fontsize=fs, color=MUTED, labelpad=2)
    ax.set_ylabel("UMAP dimension 2 (no unit)", fontsize=fs, color=MUTED, labelpad=2)


def _class_legend(ax, y: np.ndarray | None = None, loc: str = "best", ncol: int = 1,
                  anchor: Tuple[float, float] | None = None) -> None:
    h = [Line2D([], [], marker=CLASS_MARK[c], ls="", mfc=CLASS_COLOR[c], mec="white", ms=6,
                label=f"{c}  {CLASS_SHORT[c]}" + (f"  ({(y == c).sum()})" if y is not None else ""))
         for c in CLASSES]
    ax.legend(handles=h, loc=loc, ncol=ncol, handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=anchor)


# ── figures: schemes ────────────────────────────────────────────────────────
def fig_pipeline(F: Figs, D: Data) -> None:
    im = D.images
    kept = int((im.keep.astype(str) == "True").sum())
    n_ac = im[im.keep.astype(str) == "True"].slug_key.nunique()
    pa = D.frame("patent", REF[0])["df"]
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 40); ax.axis("off")
    top = [("evtol.news World eVTOL\nAircraft Directory", f"{_fmt_int(len(D.aircraft))} aircraft pages\n5 classes"),
           ("crawl", f"{_fmt_int(len(im))} images, name,\nmaker, status, credit"),
           ("whole-aircraft filter", f"SigLIP + hard rules\n{_fmt_int(kept)} kept, {_fmt_int(n_ac)} aircraft"),
           ("figure sets", "main · all\nbuilt · concept"),
           ("notebook 21", "resize + white pad\n224 / 518 px"),
           ("notebook 22", "DINOv2-large, frozen\nL18/L22/L24 x CLS/patch")]
    w, h, y0 = 14.2, 11, 26
    xs = [0.5 + i * 16.6 for i in range(len(top))]

    def box(x, y, bw, bh, title, sub, fc="white", ec=INK["photo"]):
        ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.25,rounding_size=1.0",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + bw / 2, y + bh - 1.6, title, ha="center", va="top", fontsize=7.4, weight="bold")
        ax.text(x + bw / 2, y + 1.4, sub, ha="center", va="bottom", fontsize=6.3, color=MUTED)

    def arrow(p0, p1):
        ax.annotate("", p1, p0, arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9))

    for x, (t, sub) in zip(xs, top):
        box(x, y0, w, h, t, sub)
    for a_, b_ in zip(xs[:-1], xs[1:]):
        arrow((a_ + w + 0.3, y0 + h / 2), (b_ - 0.3, y0 + h / 2))
    # patent branch: its processed figures enter the same notebook 22 model
    px = xs[3]
    box(px, 3, w * 2 + 2.4, 11, "patent figures, 1639_LABELLED",
        f"{_fmt_int(len(pa))} main figures of {_fmt_int(pa.aircraft_uid.nunique())} aircraft, "
        "processed by notebook 21\ncodebook topType mapped to its parent class", fc="#f4f4f2", ec=INK["patent"])
    arrow((px + w * 2 + 2.7, 11), (xs[5] + 2, y0 - 0.3))
    ax.text(px + w * 2 + 5.6, 20.3, "same model", fontsize=6.2, color=MUTED, rotation=38)
    # evaluation, below notebook 22
    ex = xs[5]
    box(ex, 3, w, 11, "evaluation", "A-D label-free\n5 classes, matching", fc="white", ec=INK["photo"])
    arrow((ex + w / 2 + 3, y0 - 0.3), (ex + w / 2 + 3, 14.3))
    F.save(fig, "f01_pipeline", "Processing chain of the evtol.news photo set, beside the patent figures",
           src("data", extra="scripts/evtolnews_pipeline.py (index, pages, parse, images, filter, sets, "
                             "process, extract) and scripts/evtolnews_eval.py"))


def fig_taxonomy(F: Figs, D: Data) -> None:
    pa = D.frame("patent", REF[0])["df"]
    tt = D.bundles[("patent", REF[0])]["sets"]["main"]["topType"].value_counts()
    ph = D.frame("photo", REF[0])["df"]
    children = {c: [t for t, p in ev.TOPTYPE_PARENT.items() if p == c] for c in CLASSES}
    fig, ax = plt.subplots(figsize=(10.5, 3.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")
    # each family gets max(children, 2) slots of width; children centred in their family's span
    slots = {c: max(len(children[c]), 2) for c in CLASSES}
    unit = (99 - 15) / sum(slots.values())
    xs, spans, x = {}, {}, 15.0
    for c in CLASSES:
        spans[c] = (x, x + slots[c] * unit)
        n = len(children[c])
        for i, k in enumerate(children[c]):
            xs[k] = x + (i + 0.5) * slots[c] * unit / n
        x += slots[c] * unit
    for c in CLASSES:
        cx = float(np.mean(spans[c]))
        bw = spans[c][1] - spans[c][0] - 1.4
        ax.add_patch(FancyBboxPatch((cx - bw / 2, 27), bw, 11, boxstyle="round,pad=0.3,rounding_size=1.0",
                                    fc=CLASS_COLOR[c], ec="none"))
        ax.text(cx, 36.2, c, ha="center", va="top", fontsize=10, weight="bold", color="white")
        ax.text(cx, 31.8, CLASS_SHORT[c], ha="center", va="top", fontsize=6.4, color="white")
        ax.text(cx, 28.2, f"{(ph.cls == c).sum()} directory aircraft", ha="center", va="bottom",
                fontsize=6.0, color="white")
        for k in children[c]:
            ax.plot([cx, xs[k]], [27, 16.2], color=CLASS_COLOR[c], lw=1.0)
            ax.text(xs[k], 15.4, k, ha="center", va="top", fontsize=8, weight="bold", color=TEXT,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=CLASS_COLOR[c], lw=0.9))
            ax.text(xs[k], 9.6, f"{int(tt.get(k, 0))}", ha="center", va="top", fontsize=6.6, color=MUTED)
    ax.text(0.3, 36, "evtol.news\ndirectory class (5)", fontsize=7, color=MUTED, va="top")
    ax.text(0.3, 15.6, "codebook\ntopType (12)", fontsize=7, color=MUTED, va="top")
    ax.text(0.3, 9.6, "patent aircraft\nper topType", fontsize=6.4, color=MUTED, va="top")
    ax.text(50, 1.0, "Mapping = the codebook's own grouping (Vectored Thrust / Independent Thrust / Wingless / "
            "Other); the directory splits Electric Rotorcraft out of Wingless.\nSRW sits under Lift + Cruise "
            "in the codebook, while the directory files slowed rotors as rotorcraft (Section 6).",
            ha="center", va="bottom", fontsize=6.2, color=MUTED)
    F.save(fig, "f02_taxonomy", "The five directory classes and the codebook's twelve topTypes beneath them",
           src("both", set_name="main", n=f"{len(ph)} directory aircraft, {len(pa)} patent aircraft"))


# ── figures: data ───────────────────────────────────────────────────────────
def fig_funnel(F: Figs, D: Data) -> None:
    f = D.funnel.copy()
    names = {"": "kept (whole aircraft)", "siglip": "SigLIP: not a whole aircraft",
             "other_aircraft_page": "file of another aircraft's page",
             "same_file_on_other_page": "same file already on another page",
             "too_small": "smaller than 200 px", "manual": "removed by hand"}
    f["label"] = f["excluded_by"].map(names).fillna(f["excluded_by"])
    f = f.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    cols = [INK["photo"] if e == "" else INK["patent"] for e in f["excluded_by"]]
    ax.barh(f["label"], f["figures"], color=cols, height=0.62)
    for i, (v, a) in enumerate(zip(f["figures"], f["aircraft"])):
        ax.text(v + 25, i, f"{_fmt_int(int(v))} images · {int(a)} aircraft", va="center", fontsize=7, color=TEXT)
    ax.set_xlim(0, f["figures"].max() * 1.45)
    ax.set_xlabel("images")
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    F.save(fig, "f03_funnel", "What happened to the images found on the directory pages",
           src("data", extra=f"{_fmt_int(len(D.images))} images on {_fmt_int(len(D.aircraft))} pages; "
                             "each image counted under the first rule that removed it"))


def fig_classes(F: Figs, D: Data) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), sharey=False)
    for ax, source in zip(axes, ("photo", "patent")):
        y = D.frame(source, REF[0])["df"]["cls"]
        n = y.value_counts().reindex(CLASSES).fillna(0).astype(int)
        ax.bar(CLASSES, n, color=[CLASS_COLOR[c] for c in CLASSES], width=0.66)
        for i, v in enumerate(n):
            ax.text(i, v + n.max() * 0.02, f"{v}\n{v / n.sum():.0%}", ha="center", va="bottom", fontsize=6.8)
        ax.set_ylim(0, n.max() * 1.28)
        ax.set_title(("evtol.news directory aircraft" if source == "photo" else
                      "patent aircraft, topType mapped to its parent") + f" (n = {n.sum()})")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("aircraft")
    F.save(fig, "f04_classes", "Aircraft per class in the two populations",
           src("both", set_name="main", extra="only aircraft with a kept image (photos) or an approved main figure (patents)"))


def _credit(D: Data, local_path: str) -> str:
    r = D.images[D.images.local_path == local_path]
    if not len(r):
        return ""
    r = r.iloc[0]
    c = str(r.credit).strip("() ").replace("Image credit:", "").replace("Photo credit:", "").strip()
    return c or f"evtol.news/{r.slug_key}"


def _thumb(ax, path: str, title: str, sub: str = "", edge: str = FAINT) -> None:
    img = Image.open(path).convert("RGB")
    img.thumbnail((520, 520))
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(1.6)
    ax.set_title(title, fontsize=7.4, pad=3)
    if sub:
        ax.set_xlabel(sub, fontsize=6.0, color=MUTED, labelpad=2)


def fig_examples(F: Figs, D: Data) -> List[str]:
    """Kept images (one per class) and removed ones. Returns the credits shown."""
    im = D.images.copy()
    im["keep_b"] = im.keep.astype(str) == "True"
    for c in ("p_aircraft", "p_graphic", "p_interior", "p_people", "p_part"):
        im[c] = pd.to_numeric(im[c], errors="coerce")
    ph = D.frame("photo", REF[0])["df"]
    main = im[im.local_path.isin(ph.approved_copy_path)].merge(
        ph[["approved_copy_path", "cls", "status_group"]], left_on="local_path", right_on="approved_copy_path")
    picks, credits = [], []
    for c in CLASSES:
        pref = main[main.slug_key == EXAMPLE_PREFERRED.get(c, "")]
        g = main[(main.cls == c) & (main.status_group == "built")].sort_values("p_aircraft", ascending=False)
        g = pref if len(pref) else (g if len(g) else main[main.cls == c].sort_values("p_aircraft", ascending=False))
        r = g.iloc[0]
        picks.append((r.local_path, f"kept · {c}", r.slug_key[:34], CLASS_COLOR[c]))
    dropped = im[~im.keep_b & (im.hard_rule == "") & (im.local_path != "")]
    for col, word in (("p_graphic", "logo"), ("p_interior", "interior"), ("p_people", "people"),
                      ("p_part", "part close-up"), ("p_graphic", "diagram")):
        g = dropped.sort_values(col, ascending=False)
        if word == "logo":
            g = g[g.top_prompt.str.contains("logo|brand|title", case=False)]
        if word == "diagram":
            g = g[g.top_prompt.str.contains("diagram|infographic|specification", case=False)]
        g = g[~g.local_path.isin([p[0] for p in picks])]
        if len(g):
            r = g.iloc[0]
            picks.append((r.local_path, f"removed · {word}", f"p(aircraft) {r.p_aircraft:.2f}", INK["patent"]))
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.6))
    for ax, (p, t, s, e) in zip(axes.ravel(), picks):
        _thumb(ax, p, t, s, e)
        credits.append(f"{t}: {_credit(D, p)}")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    F.save(fig, "f05_examples", "Examples: the highest-scoring kept image of a built aircraft per class (top) "
           "and images the filter removed (bottom)",
           src("photo", extra="SigLIP ViT-SO400M-14-384 zero-shot filter; images (c) their owners, "
                              "reproduced for internal review only"))
    return credits


def fig_siglip_hist(F: Figs, D: Data, keep_p: float = 0.80, drop_p: float = 0.35) -> None:
    p = pd.to_numeric(D.images.p_aircraft, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(6.8, 2.4))
    ax.hist(p, bins=np.linspace(0, 1, 51), color=INK["photo"], rwidth=0.9)
    ax.set_yscale("log")
    for x, t in ((drop_p, "drop below"), (keep_p, "keep above")):
        ax.axvline(x, color=MUTED, ls="--", lw=0.9)
        ax.text(x + 0.01, ax.get_ylim()[1] * 0.5, f"{t} {x:.2f}", fontsize=7, color=MUTED)
    ax.set_xlabel("p(whole aircraft), SigLIP zero-shot, summed over the aircraft prompts")
    ax.set_ylabel("images (log)")
    F.save(fig, "f06_siglip", "Filter score of every downloaded image",
           src("photo", extra=f"{_fmt_int(len(p))} readable images; between the two lines = review band "
                              "(held back unless kept by hand)"))


# ── figures: stage A/B ──────────────────────────────────────────────────────
def _grid6(figsize=(10.5, 4.6)):
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    order = [(18, "cls"), (22, "cls"), (24, "cls"), (18, "mean_patch"), (22, "mean_patch"), (24, "mean_patch")]
    return fig, dict(zip(order, axes.ravel()))


def fig_cosine(F: Figs, D: Data, tag: str = REF[0]) -> None:
    pp, pa = D.plot_data("photo", tag), D.plot_data("patent", tag)
    fig, axes = _grid6()
    for (L, pool), ax in axes.items():
        t = f"L{L}_{pool}"
        bins = np.linspace(min(pp[f"cosine__{t}"].min(), pa[f"cosine__{t}"].min()), 1, 60)
        for p, s in ((pa, "patent"), (pp, "photo")):
            ax.hist(p[f"cosine__{t}"], bins=bins, density=True, histtype="step", lw=1.6, color=INK[s],
                    label="patent figures" if s == "patent" else "evtol.news photos")
        ax.set_title(mat_name(L, pool)); ax.set_yticks([])
        ax.set_xlabel("pairwise cosine similarity")
    axes[(18, "cls")].legend(loc="upper left")
    F.save(fig, "f07_cosine", "Stage A: spread of pairwise cosine similarity, photos against patent figures",
           src("both", tag, pool="all", set_name="main",
               n="512 images sampled per matrix for the pairs"))


def fig_pca(F: Figs, D: Data, tag: str = REF[0]) -> None:
    pp, pa = D.plot_data("photo", tag), D.plot_data("patent", tag)
    fig, axes = _grid6()
    for (L, pool), ax in axes.items():
        t = f"L{L}_{pool}"
        ax.plot(pp[f"pca_real__{t}"], color=INK["photo"], lw=1.6, label="photos")
        ax.plot(pa[f"pca_real__{t}"], color=INK["patent"], lw=1.6, label="patent figures")
        ax.plot(pp[f"pca_random__{t}"], color=MUTED, lw=1.0, ls="--", label="random, same shape")
        r1, r2 = pp[f"pca_real__{t}"][0] / pp[f"pca_random__{t}"][0], pa[f"pca_real__{t}"][0] / pa[f"pca_random__{t}"][0]
        ax.set_yscale("log"); ax.set_title(f"{mat_name(L, pool)}: PC1 x{r1:.0f} photos, x{r2:.0f} patents")
        ax.set_xlabel("principal component"); ax.set_ylabel("explained variance ratio")
    axes[(18, "cls")].legend(loc="upper right")
    F.save(fig, "f08_pca", "Stage B: PCA spectrum against a random matrix of the same shape",
           src("both", tag, pool="all", set_name="main"))


def fig_dist(F: Figs, D: Data, tag: str = REF[0]) -> None:
    pp = D.plot_data("photo", tag)
    fig, axes = _grid6()
    for (L, pool), ax in axes.items():
        t = f"L{L}_{pool}"
        ax.hist(pp[f"dist_random__{t}"], bins=40, density=True, color=FAINT, label="random, same shape")
        ax.hist(pp[f"dist_real__{t}"], bins=40, density=True, histtype="step", lw=1.6, color=INK["photo"],
                label="photos")
        ax.set_title(mat_name(L, pool)); ax.set_yticks([]); ax.set_xlabel("pairwise cosine distance")
    axes[(18, "cls")].legend(loc="upper left")
    F.save(fig, "f09_distance", "Stage B: pairwise cosine distance of the photo embeddings against random",
           src("photo", tag, pool="all", set_name="main"))


# ── figures: stage C ────────────────────────────────────────────────────────
def fig_dendrogram(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> None:
    from scipy.cluster.hierarchy import dendrogram

    pp = D.plot_data("photo", tag)
    t = f"L{L}_{pool}"
    y = D.frame("photo", tag)["df"]["cls"].to_numpy()
    fig = plt.figure(figsize=(10.5, 3.8))
    ax = fig.add_axes([0.05, 0.3, 0.93, 0.62])
    dn = dendrogram(pp[f"linkage__{t}"], ax=ax, no_labels=True, color_threshold=float(pp[f"cut__{t}"]),
                    link_color_func=lambda _: "#8a8984")
    ax.axhline(float(pp[f"cut__{t}"]), color=MUTED, ls="--", lw=0.8)
    ax.set_ylabel("Ward distance"); ax.set_xticks([])
    strip = fig.add_axes([0.05, 0.2, 0.93, 0.06])
    leaves = np.array(dn["leaves"])
    for i, c in enumerate(y[leaves]):
        strip.axvspan(i, i + 1, color=CLASS_COLOR.get(c, FAINT), lw=0)
    strip.set_xlim(0, len(leaves)); strip.axis("off")
    strip.text(-4, 0.5, "class", ha="right", va="center", fontsize=7, color=MUTED, transform=strip.transData)
    lax = fig.add_axes([0.05, 0.05, 0.93, 0.1]); lax.axis("off")
    _class_legend(lax, y, loc="center", ncol=5)
    _stamp(fig, src("photo", tag, L, pool, "main", f"{len(y)} aircraft; Ward linkage on PCA-90% (notebook 30), "
           "dashed line = cut at the best k-means k"), READ["f10_dendrogram"])
    fig.savefig(F.dir / "f10_dendrogram.png", bbox_inches="tight", pad_inches=0.06); plt.close(fig)
    F.items["f10_dendrogram"] = {"file": "figs/f10_dendrogram.png",
                                 "title": "Stage C: Ward dendrogram of the photo embeddings, leaves coloured by directory class",
                                 "source": src("photo", tag, L, pool, "main", f"{len(y)} aircraft"), "note": "",
                                 "read": READ["f10_dendrogram"]}


def fig_umap_clusters(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    X = D.X("photo", tag, L, pool)
    y = D.frame("photo", tag)["df"]["cls"].to_numpy()
    emb = D.umap(f"photo_{tag}_L{L}_{pool}_main", X)
    k = int(D.table("photo", tag, "C_clustering_summary.csv").set_index(["layer", "pooling"]).loc[(L, pool), "best_kmeans_k"])
    Z = PCA(n_components=0.9, random_state=42).fit_transform(X)
    lab = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Z)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    ax = axes[0]
    ax.scatter(emb[:, 0], emb[:, 1], s=4, c=FAINT, lw=0)
    for j in range(k):
        col, mk = CLUSTER[j % 3]
        m = lab == j
        ax.scatter(emb[m, 0], emb[m, 1], s=7, c=col, marker=mk, lw=0, alpha=0.8, label=f"C{j + 1} ({m.sum()})")
        mx, my = np.median(emb[m], axis=0)
        ax.text(mx, my, f"C{j + 1}", fontsize=8.5, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col, lw=0.9))
    _umap_axes(ax); ax.legend(loc="lower right", markerscale=1.6)
    ax.set_title(f"k-means, k = {k} (best silhouette, notebook 30)")
    ct = pd.crosstab(pd.Series(lab + 1, name="cluster").map(lambda v: f"C{v}"), pd.Series(y, name="class"))
    ct = ct.reindex(columns=CLASSES, fill_value=0)
    share = ct.div(ct.sum(axis=0), axis=1)
    ax2 = axes[1]
    bottom = np.zeros(len(CLASSES))
    for j, cl in enumerate(ct.index):
        col = CLUSTER[j % 3][0]
        ax2.bar(CLASSES, share.loc[cl], bottom=bottom, color=col, width=0.62, edgecolor="white", lw=1.2, label=cl)
        for i, v in enumerate(share.loc[cl]):
            if v > 0.08:
                ax2.text(i, bottom[i] + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=6.8,
                         color="white" if j != 1 else TEXT)
        bottom += share.loc[cl].to_numpy()
    ax2.set_ylabel("share of the class's aircraft"); ax2.set_ylim(0, 1)
    ax2.set_title("where each directory class falls among the clusters")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    F.save(fig, "f11_umap_clusters", "Stage C: the label-free k-means partition on the UMAP map, and how "
           "the directory classes spread over it",
           src("photo", tag, L, pool, "main", f"{len(y)} aircraft; k-means on PCA-90% of the matrix; "
               "UMAP n_neighbors 15, min_dist 0.1, cosine"))
    return share


def fig_umap_class(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> None:
    X = D.X("photo", tag, L, pool)
    y = D.frame("photo", tag)["df"]["cls"].to_numpy()
    emb = D.umap(f"photo_{tag}_L{L}_{pool}_main", X)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    _scatter_classes(ax, emb, y, s=10)
    _class_legend(ax, y, loc="upper center", ncol=3, anchor=(0.5, -0.06))
    ax.set_title(f"UMAP of the photo embeddings, {mat_name(L, pool)}, coloured by directory class")
    F.save(fig, "f12_umap_class", "Stage C: UMAP of the photo embeddings coloured by the directory class",
           src("photo", tag, L, pool, "main", f"{len(y)} aircraft; UMAP n_neighbors 15, min_dist 0.1, "
               "cosine, seed 42; labels mark each class's median position"))
    fig, axes = plt.subplots(1, 5, figsize=(11.5, 2.7))
    for ax, c in zip(axes, CLASSES):
        ax.scatter(emb[:, 0], emb[:, 1], s=2.5, c=FAINT, lw=0)
        m = y == c
        ax.scatter(emb[m, 0], emb[m, 1], s=6, c=CLASS_COLOR[c], marker=CLASS_MARK[c], lw=0)
        ax.set_title(f"{c} ({m.sum()})", color=TEXT)
        _umap_axes(ax, small=True)
    F.save(fig, "f13_umap_facets", "Stage C: the same UMAP map, one class highlighted per panel",
           src("photo", tag, L, pool, "main", f"{len(y)} aircraft"))


def fig_umap_matrices(F: Figs, D: Data, tag: str = REF[0]) -> None:
    y = D.frame("photo", tag)["df"]["cls"].to_numpy()
    fig, axes = _grid6(figsize=(10.5, 6.4))
    for (L, pool), ax in axes.items():
        emb = D.umap(f"photo_{tag}_L{L}_{pool}_main", D.X("photo", tag, L, pool))
        _scatter_classes(ax, emb, y, s=4, labels=True)
        ax.set_title(mat_name(L, pool))
    F.save(fig, "f14_umap_matrices", "Stage C: UMAP of the photo embeddings for every layer and pooling, "
           "coloured by directory class",
           src("photo", tag, pool="all", set_name="main", n=f"{len(y)} aircraft; one UMAP fit per matrix"))


def fig_umap_patent(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    for ax, source in zip(axes, ("photo", "patent")):
        y = D.frame(source, tag)["df"]["cls"].to_numpy()
        emb = D.umap(f"{source}_{tag}_L{L}_{pool}_main", D.X(source, tag, L, pool))
        _scatter_classes(ax, emb, y, s=8)
        ax.set_title(("evtol.news photos" if source == "photo" else "patent figures, parent of topType")
                     + f" (n = {len(y)})")
    fig.legend(handles=[Line2D([], [], marker=CLASS_MARK[c], ls="", mfc=CLASS_COLOR[c], mec="white", ms=6,
                               label=f"{c}  {CLASS_SHORT[c]}") for c in CLASSES],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.035))
    F.save(fig, "f15_umap_photo_patent", "Stage C: the same model and matrix on photos and on patent "
           "figures, each coloured by its five-class label",
           src("both", tag, L, pool, "main", "separate UMAP fits"), bottom=0.1)


def fig_umap_joint(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> None:
    Xp, Xa = D.X("photo", tag, L, pool), D.X("patent", tag, L, pool)
    emb = D.umap(f"joint_{tag}_L{L}_{pool}_main", np.vstack([Xp, Xa]))
    n = len(Xp)
    yp = D.frame("photo", tag)["df"]["cls"].to_numpy()
    ya = D.frame("patent", tag)["df"]["cls"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5))
    ax = axes[0]
    ax.scatter(emb[:n, 0], emb[:n, 1], s=6, c=INK["photo"], lw=0, alpha=0.7, label=f"photos ({n})")
    ax.scatter(emb[n:, 0], emb[n:, 1], s=6, c=INK["patent"], marker="s", lw=0, alpha=0.8,
               label=f"patent figures ({len(Xa)})")
    ax.legend(loc="lower left", markerscale=1.8); _umap_axes(ax)
    ax.set_title("one map for both sources: coloured by source")
    ax = axes[1]
    _scatter_classes(ax, emb, np.concatenate([yp, ya]), s=6)
    ax.set_title("the same map, coloured by five-class label")
    F.save(fig, "f16_umap_joint", "Stage C: photos and patent figures embedded together",
           src("both", tag, L, pool, "main", "one UMAP fit on both sources"))


def fig_contingency(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> None:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for ax, source in zip(axes, ("photo", "patent")):
        X = D.X(source, tag, L, pool)
        y = D.frame(source, tag)["df"]["cls"].to_numpy()
        lab = KMeans(n_clusters=5, n_init=10, random_state=42).fit_predict(X)
        ct = pd.crosstab(y, lab).reindex(CLASSES, fill_value=0)
        # clusters ordered by the class they hold most of
        order = sorted(ct.columns, key=lambda j: (CLASSES.index(ct[j].idxmax()), -ct[j].max()))
        ct = ct[order]
        share = ct.div(ct.sum(axis=1).clip(lower=1), axis=0)
        ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(share.shape[0]):
            for j in range(share.shape[1]):
                v = share.iat[i, j]
                ax.text(j, i, f"{ct.iat[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else TEXT)
        ax.set_yticks(range(5), CLASSES); ax.set_xticks(range(5), [f"K{j + 1}" for j in range(5)])
        ax.set_xlabel("k-means cluster (k = 5)")
        ax.set_title(f"{'photos' if source == 'photo' else 'patent figures'}: ARI {adjusted_rand_score(y, lab):.2f}")
        for sp in ax.spines.values():
            sp.set_visible(False)
    F.save(fig, "f17_contingency", "Stage C: five k-means clusters against the five classes "
           "(cell = aircraft, shade = share of the row)",
           src("both", tag, L, pool, "main", "k-means on the L2-normalised matrix, k fixed to 5"))


def fig_intra_inter(F: Figs, D: Data, tag: str = REF[0]) -> pd.DataFrame:
    """Inter- over intra-class cosine distance, raw (all pairs) and class-balanced (mean of class blocks)."""
    rows = []
    for source in ("photo", "patent"):
        y = D.frame(source, tag)["df"]["cls"].to_numpy()
        same = y[:, None] == y[None, :]
        off = ~np.eye(len(y), dtype=bool)
        for L, pool in MATRICES:
            X = D.X(source, tag, L, pool)
            Dm = 1 - X @ X.T
            blk = np.array([[Dm[np.ix_(y == a, y == b)][(off[np.ix_(y == a, y == b)])].mean() for b in CLASSES]
                            for a in CLASSES])
            rows.append({"source": source, "matrix": mat_name(L, pool), "layer": L, "pooling": pool,
                         "raw": Dm[~same].mean() / Dm[same & off].mean(),
                         "balanced": blk[~np.eye(5, dtype=bool)].mean() / np.diag(blk).mean()})
    t = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.6, 2.8))
    x = np.arange(len(MATRICES))
    for i, source in enumerate(("photo", "patent")):
        g = t[t.source == source]
        ax.bar(x + (i - 0.5) * 0.36, g.balanced, width=0.34, color=INK[source],
               label="evtol.news photos" if source == "photo" else "patent figures")
        for xi, v in zip(x, g.balanced):
            ax.text(xi + (i - 0.5) * 0.36, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=6.4)
    ax.axhline(1, color=MUTED, ls="--", lw=0.8)
    ax.text(len(MATRICES) - 0.5, 1.0015, "1 = classes no further apart than within a class", ha="right",
            va="bottom", fontsize=6.5, color=MUTED)
    ax.set_xticks(x, [mat_name(L, p) for L, p in MATRICES])
    ax.set_ylim(0.99, t.balanced.max() * 1.02)
    ax.set_ylabel("inter / intra-class\ncosine distance")
    ax.legend(loc="upper left")
    F.save(fig, "f18_intra_inter", "Stage C: how much further apart different classes sit than the same class "
           "(class-balanced: every class pair weighs the same)",
           src("both", tag, pool="all", set_name="main", n="mean distance of each class block, then averaged"))
    return t


# ── figures: supervised ─────────────────────────────────────────────────────
def fig_knn(F: Figs, D: Data, tag: str = REF[0]) -> None:
    c = D.cm[(D.cm.label == "class") & (D.cm.set == "main") & (D.cm.embedding == tag)]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.0), sharey=True)
    x = np.arange(len(MATRICES))
    for ax, (col, name) in zip(axes, (("knn5_bal_acc_maker_out", "kNN-5, the aircraft's maker held out"),
                                      ("probe_bal_acc_maker_out", "logistic probe, folds grouped by maker"))):
        for i, source in enumerate(("photo", "patent")):
            g = c[c.source == source].set_index(["layer", "pooling"]).reindex(MATRICES)
            ax.bar(x + (i - 0.5) * 0.36, g[col], width=0.34, color=INK[source],
                   label="evtol.news photos" if source == "photo" else "patent figures")
            for xi, v in zip(x, g[col]):
                ax.text(xi + (i - 0.5) * 0.36, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6.3)
        ax.axhline(0.2, color=MUTED, ls="--", lw=0.9)
        ax.text(len(MATRICES) - 0.55, 0.212, "chance 0.20", fontsize=6.6, color=MUTED, ha="right")
        ax.set_xticks(x, [mat_name(L, p) for L, p in MATRICES], fontsize=6.8)
        ax.set_title(name); ax.set_ylim(0, 0.8)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("balanced accuracy, 5 classes"); axes[0].legend(loc="upper left")
    F.save(fig, "f19_knn_probe", "Five-class accuracy per layer and pooling, photos against patent figures",
           src("both", tag, pool="all", set_name="main",
               n="photos: directory class; patents: parent of the human topType"))


def fig_recall(F: Figs, D: Data) -> None:
    conf = ev.confusion_tables(D.cfg)
    fig, ax = plt.subplots(figsize=(7.6, 2.8))
    x = np.arange(len(CLASSES))
    for i, source in enumerate(("photo", "patent")):
        r = conf[source].reindex(CLASSES)
        bars = ax.bar(x + (i - 0.5) * 0.38, r.recall, width=0.36, color=[CLASS_COLOR[c] for c in CLASSES],
                      hatch="////" if source == "patent" else None, edgecolor="white", lw=0.8,
                      alpha=1.0 if source == "photo" else 0.55)
        for b, v, n in zip(bars, r.recall, r.n):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}\nn={n}", ha="center", va="bottom", fontsize=6.1)
    ax.axhline(0.2, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(x, [f"{c}\n{CLASS_SHORT[c]}" for c in CLASSES], fontsize=7)
    ax.set_ylim(0, 1.05); ax.set_ylabel("recall (kNN-5, maker held out)")
    ax.legend(handles=[Patch(fc=MUTED, label="evtol.news photos (solid)"),
                       Patch(fc=MUTED, alpha=0.55, hatch="////", ec="white", label="patent figures (hatched)")],
              loc="upper right")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    F.save(fig, "f20_recall", "Share of each class recognised, photos against patent figures",
           src("both", REF[0], REF[1], REF[2], "main"))


def fig_confusion(F: Figs, D: Data) -> None:
    conf = ev.confusion_tables(D.cfg)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for ax, source in zip(axes, ("photo", "patent")):
        t = conf[source][CLASSES]
        share = t.div(t.sum(axis=1).clip(lower=1), axis=0)
        ax.imshow(share.to_numpy(), cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{t.iat[i, j]}", ha="center", va="center", fontsize=7,
                        color="white" if share.iat[i, j] > 0.55 else TEXT)
        ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(5), CLASSES)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("evtol.news photos" if source == "photo" else "patent figures")
        for sp in ax.spines.values():
            sp.set_visible(False)
    F.save(fig, "f21_confusion", "Confusion of the five classes (cell = aircraft, shade = share of the true class)",
           src("both", REF[0], REF[1], REF[2], "main", "kNN-5 with the aircraft's maker held out"))


def fig_maturity(F: Figs, D: Data, tag: str = REF[0]) -> None:
    c = D.cm[(D.cm.set == "main") & (D.cm.embedding == tag) & (D.cm.source == "photo")]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    x = np.arange(len(MATRICES))
    for i, (lab, chance, name) in enumerate((("class", 0.20, "architecture (5 classes)"),
                                             ("status", 1 / 3, "maturity: built / concept / unknown"))):
        g = c[c.label == lab].set_index(["layer", "pooling"]).reindex(MATRICES)
        lift = g.knn5_bal_acc / chance
        ax.bar(x + (i - 0.5) * 0.36, lift, width=0.34, color=INK["photo"] if i == 0 else INK["patent"], label=name)
        for xi, v, a in zip(x, lift, g.knn5_bal_acc):
            ax.text(xi + (i - 0.5) * 0.36, v + 0.04, f"{a:.2f}", ha="center", va="bottom", fontsize=6.2)
    ax.axhline(1, color=MUTED, ls="--", lw=0.8)
    ax.set_xticks(x, [mat_name(L, p) for L, p in MATRICES], fontsize=6.8)
    ax.set_ylabel("kNN-5 accuracy / chance"); ax.legend(loc="upper left")
    ax.set_ylim(0, 3.6); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    F.save(fig, "f22_maturity", "What the photo space encodes besides architecture: the directory's maturity status",
           src("photo", tag, pool="all", set_name="main",
               n="numbers on the bars = balanced accuracy; bar height = accuracy over chance"))


# ── figures: taxonomy check ─────────────────────────────────────────────────
def fig_parent(F: Figs, D: Data) -> None:
    p = D.parent.copy()
    # an aircraft whose pages span two classes sits under the one that matches, when one does
    p["shown_class"] = [par if par in str(dc).split("|") else mode
                        for par, dc, mode in zip(p.parent_of_topType, p.directory_classes, p.directory_class)]
    tt = [t for t in ev.TOPTYPE_PARENT if t in set(p.topType)]
    ct = pd.crosstab(p.topType, p.shown_class).reindex(index=tt, columns=CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.imshow(ct.to_numpy() > 0, cmap="Greys", vmin=0, vmax=6, aspect="auto")
    for i, t in enumerate(ct.index):
        for j, c in enumerate(CLASSES):
            v = ct.iat[i, j]
            if ev.TOPTYPE_PARENT.get(t) == c:
                ax.add_patch(plt.Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=CLASS_COLOR[c], lw=2))
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        weight="bold" if ev.TOPTYPE_PARENT.get(t) == c else "normal",
                        color=TEXT if ev.TOPTYPE_PARENT.get(t) == c else "#b3261e")
    ax.set_xticks(range(5), CLASSES); ax.set_yticks(range(len(tt)), tt)
    ax.set_xlabel("directory class of the linked page(s)"); ax.set_ylabel("codebook topType (human label)")
    for sp in ax.spines.values():
        sp.set_visible(False)
    n_ok = int(p.agrees.astype(str).eq("True").sum())
    ax.set_title(f"{n_ok} / {len(p)} aircraft in the codebook's parent class (outlined)")
    F.save(fig, "f23_parent", "Codebook topType against the directory class, on the patent aircraft "
           "linked to a directory page",
           src("data", extra=f"{len(p)} named patent aircraft linked by name, maker or the author's "
                             "NAME_DECISIONS ruling; outlined cell = the parent the codebook implies; "
                             "red numbers = disagreements; an aircraft whose pages span two classes is "
                             "shown under the matching one"))


# ── figures: matching ───────────────────────────────────────────────────────
def _recall_curve(r: pd.DataFrame, ks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rk = r["rank"].to_numpy()
    obs = np.array([(rk <= k).mean() for k in ks])
    rnd = np.array([np.mean([1 - np.prod([(g - p - m) / (g - m) for m in range(int(k))])
                             for g, p in zip(r.gallery, r.n_positive)]) for k in ks])
    return obs, rnd


def fig_recall_at_k(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> None:
    ks = np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.2), sharey=True)
    for ax, direction in zip(axes, ("patent->photo", "photo->patent")):
        for images, col, ls in (("all images", INK["patent"], "-"), ("no line drawings", INK["photo"], "-")):
            r = D.ranks[(D.ranks.embedding == tag) & (D.ranks.layer == L) & (D.ranks.pooling == pool)
                        & (D.ranks.direction == direction) & (D.ranks.images == images)]
            obs, rnd = _recall_curve(r, ks)
            ax.plot(ks, obs, color=col, lw=1.8, marker="o", ms=3,
                    label=f"{images} ({len(r)} queries)")
        ax.plot(ks, rnd, color=MUTED, ls="--", lw=1.0, label="random ranking")
        ax.set_xscale("log"); ax.set_xticks([1, 5, 10, 50, 100, 200], ["1", "5", "10", "50", "100", "200"])
        ax.set_xlabel("k (top-k of the ranked gallery)")
        gal = int(r.gallery.iloc[0])
        ax.set_title(("patent figure -> directory pages" if direction == "patent->photo"
                      else "directory photo -> patent aircraft") + f" (gallery {gal})")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_ylim(0, 0.8)
    axes[0].set_ylabel("recall@k (right aircraft family found)")
    axes[0].legend(loc="upper left")
    F.save(fig, "f24_recall_at_k", "Same-aircraft matching: how often the right aircraft appears in the top k",
           src("both", tag, L, pool, "main (queries) against all (gallery)",
               "99 named patent aircraft of 62 families; gallery images pooled per aircraft by max cosine"))


def fig_matching_matrices(F: Figs, D: Data) -> None:
    m = D.match[(D.match.images == "no line drawings")]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 2.9), sharey=True)
    x = np.arange(len(MATRICES))
    for ax, direction in zip(axes, ("patent->photo", "photo->patent")):
        for i, tag in enumerate(D.cfg["evtolnews"]["embeddings"]):
            g = m[(m.direction == direction) & (m.embedding == tag)].set_index(["layer", "pooling"]).reindex(MATRICES)
            ax.bar(x + (i - 0.5) * 0.36, g["R@10"], width=0.34, color=INK["patent"] if i == 0 else INK["photo"],
                   label=f"{size_of(tag)} px")
            for xi, v in zip(x, g["R@10"]):
                ax.text(xi + (i - 0.5) * 0.36, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=6.2)
        rnd = m[m.direction == direction]["random_R@10"].mean()
        ax.axhline(rnd, color=MUTED, ls="--", lw=0.8)
        ax.text(len(MATRICES) - 0.55, rnd + 0.006, f"random {rnd:.3f}", fontsize=6.4, color=MUTED, ha="right")
        ax.set_xticks(x, [mat_name(L, p) for L, p in MATRICES], fontsize=6.8)
        ax.set_title("patent -> directory" if direction == "patent->photo" else "directory -> patent")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("recall@10"); axes[0].legend(loc="upper left")
    F.save(fig, "f25_matching_matrices", "Same-aircraft matching per layer, pooling and image size (recall@10)",
           src("both", "dinov2-large_518", pool="all", set_name="main (queries) against all (gallery)",
               n="224 px and 518 px extractions; directory line drawings excluded from the gallery"))


def fig_pairs(F: Figs, D: Data, tag: str = REF[0], L: int = REF[1], pool: str = REF[2]) -> List[str]:
    """Patent figure beside its best-matching image of the right family, including the copies."""
    lk = ev.load_links(D.cfg)
    fam = lk.set_index("aircraft_uid").family.to_dict()
    fs: Dict[str, set] = {}
    for a, s in lk.set_index("aircraft_uid").slugs.items():
        fs.setdefault(fam[a], set()).update(s)
    ga = ev._frame(D.bundles[("photo", tag)]["result"], D.bundles[("photo", tag)]["sets"]["all"])
    qa = ev._frame(D.bundles[("patent", tag)]["result"], D.bundles[("patent", tag)]["sets"]["main"])
    GP = em._l2(ga["sub"]["arrays"][(L, pool)].astype(float))
    QA = em._l2(qa["sub"]["arrays"][(L, pool)].astype(float))
    r = D.ranks[(D.ranks.embedding == tag) & (D.ranks.layer == L) & (D.ranks.pooling == pool)
                & (D.ranks.direction == "patent->photo") & (D.ranks.images == "all images")].set_index("query")
    rows = []
    for i, a in enumerate(qa["df"].aircraft_uid):
        if a not in fam or a not in r.index:
            continue
        m = ga["df"].aircraft_uid.isin(fs[fam[a]]).to_numpy()
        if not m.any():
            continue
        s = GP @ QA[i]
        j = np.flatnonzero(m)[np.argmax(s[m])]
        rows.append(dict(query=a, family=fam[a], cos=float(s[j]), rank=int(r.loc[a, "rank"]),
                         style=ga["df"]["style"].iat[j], p_drawing=float(ga["df"]["style_drawing"].iat[j]),
                         photo=ga["df"].approved_copy_path.iat[j],
                         patent=qa["df"].approved_copy_path.iat[i]))
    d = pd.DataFrame(rows)
    copies = d[d["style"] == "drawing"].sort_values("cos", ascending=False).head(2)
    # clearly not a line drawing; plus images checked by eye that SigLIP scores as renders
    real = d[(d["p_drawing"] < 0.2) & ~d["photo"].str.contains("|".join(NOT_A_PHOTO))]
    genuine = real.sort_values(["rank", "cos"], ascending=[True, False]).head(3)
    miss = real.sort_values("rank", ascending=False).head(1)
    pick = pd.concat([copies.assign(kind="page shows a patent drawing"),
                      genuine.assign(kind="drawing -> photo / render"),
                      miss.assign(kind="a miss: linked page, very different image")])
    fig, axes = plt.subplots(len(pick), 2, figsize=(6.6, 2.05 * len(pick)))
    credits = []
    for (ax1, ax2), row in zip(axes, pick.itertuples()):
        _thumb(ax1, row.patent, f"{row.family}: patent {row.query.split('_')[0]}", "", INK["patent"])
        _thumb(ax2, row.photo, f"{row.kind}", f"rank {row.rank} of {int(r.gallery.iloc[0])}, cosine {row.cos:.2f}",
               INK["photo"])
        credits.append(f"{row.family}: {_credit(D, row.photo)}")
    D.pair_pick = pick
    F.save(fig, "f26_pairs", "Same-aircraft matching, examples: patent figure (left) and the most similar "
           "image of the right aircraft family (right)",
           src("both", tag, L, pool, "main (patent) against all (directory)",
               "images (c) their owners, internal review only"))
    return credits


def fig_method_scheme(F: Figs, D: Data) -> None:
    """One scheme: every step, what it is for, its variables and how each was chosen."""
    im = D.images
    kept = int((im.keep.astype(str) == "True").sum())
    n_ac = im[im.keep.astype(str) == "True"].slug_key.nunique()
    steps = [
        ("1", "Collect", "an independent image set of real eVTOL aircraft, labelled by an expert body",
         f"source: evtol.news World eVTOL Aircraft Directory, {_fmt_int(len(D.aircraft))} aircraft. "
         "Label: the directory's 5 classes, read from its index lists.",
         "the only public directory that classifies every known design; its classes match the "
         "codebook's own parent grouping of the 12 topTypes"),
        ("2", "Filter", "keep only images that show the page's aircraft as a whole",
         "SigLIP p(whole aircraft): keep at 0.80 or more, drop under 0.35. Rules: side of at least "
         "200 px, no file shared with another page or aircraft.",
         f"thresholds set by inspecting contact sheets at both borders; {_fmt_int(len(im))} images "
         f"gave {_fmt_int(kept)} kept, on {_fmt_int(n_ac)} aircraft"),
        ("3", "Select", "one comparable image per aircraft",
         "set main = the page's first kept image (its lead image); set all = every kept image",
         "mirrors the patent pipeline's main and all sets, so both sources are compared like for like"),
        ("4", "Embed", "the same representation as the patent figures",
         "DINOv2-large, frozen; 224 and 518 px; layers 18, 22, 24; CLS token or mean of the patch tokens",
         "identical to the patent study (notebooks 21-22); 518 px is DINOv2's native high resolution; "
         "reference matrix 518 px, L24, CLS = best on steps 5 and 6"),
        ("5", "Test classes", "do the architecture classes show in the embeddings?",
         "kNN with k = 5, the aircraft and then its whole maker held out; logistic probe; balanced "
         "accuracy; chance from shuffled labels",
         "holding the maker out stops a company's house style from carrying the answer; balanced "
         "accuracy because the classes are unbalanced"),
        ("6", "Match", "can a patent figure find its real aircraft among the photos?",
         "99 named patent aircraft linked to their pages; each directory aircraft scored by its most "
         "similar image; directory line drawings removed",
         "links from the author's own naming rulings; drawings removed because some pages show the "
         "patent drawing itself; a random ranking is the baseline"),
        ("7", "Inspect", "which parts of the image the model draws on",
         "CLS-token attention at layers 18, 22, 24; mean of the 16 heads; main images at 518 px",
         "rebuilt from each block's query and key weights (full attention output would need about "
         "23 GB); layer 18 shown, because from layer 22 on the CLS token attends mostly to the padding"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 12.8))
    n = len(steps)
    ax.set_xlim(0, 100); ax.set_ylim(0, n * 10 + 4); ax.axis("off")
    cols = [(13, 21, "intended"), (37.5, 33, "variables"), (69, 34, "how chosen")]
    for x, _, t in cols:
        ax.text(x, n * 10 + 2.6, t.upper(), fontsize=9.5, weight="bold", color=MUTED, va="center")
    for i, (num, name, goal, var, why) in enumerate(steps):
        y = (n - 1 - i) * 10 + 1
        ax.add_patch(FancyBboxPatch((0.5, y + 0.6), 10.5, 7.6, boxstyle="round,pad=0.2,rounding_size=1.0",
                                    fc=INK["photo"], ec="none"))
        ax.text(5.75, y + 5.6, num, ha="center", va="center", fontsize=17, weight="bold", color="white")
        ax.text(5.75, y + 2.6, name, ha="center", va="center", fontsize=10, weight="bold", color="white")
        if i < n - 1:
            ax.annotate("", (5.75, y - 1.2), (5.75, y + 0.5), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
        ax.add_patch(FancyBboxPatch((12.2, y + 0.6), 87.3, 7.6, boxstyle="round,pad=0.2,rounding_size=0.8",
                                    fc="#f6f6f4", ec=GRID, lw=0.8))
        for (x, w, _), text, kw in zip(cols, (goal, var, why),
                                       ({"weight": "bold", "color": TEXT, "fontsize": 9.6},
                                        {"color": TEXT, "fontsize": 9.0}, {"color": "#3a3936", "fontsize": 9.0})):
            ax.text(x, y + 4.4, "\n".join(_wrap(text, w)), va="center", linespacing=1.35, **kw)
    F.save(fig, "f00_method", "Method in seven steps: what each step is for, its settings, and why they were chosen",
           src("data", extra="scripts/evtolnews_pipeline.py and scripts/evtolnews_eval.py"))


def _wrap(text: str, width: int) -> List[str]:
    import textwrap
    return textwrap.wrap(text, width)


# ── figures: attention (CLS token, extraction repo's attention/<tag>/) ──────
ATTN_RED = "#e34948"


def _attention(D: Data, source: str, tag: str = REF[0]) -> Tuple[np.ndarray, pd.DataFrame]:
    d = D.root / "2_embedding_extraction" / "attention" / tag
    return np.load(d / f"{source}_cls.npy").astype(np.float32), pd.read_csv(d / f"{source}_meta.csv")


def _overlay(ax, path: str, amap: np.ndarray, title: str = "", edge: str = FAINT) -> None:
    """Grey image with the attention map as one red hue whose opacity follows the weight."""
    img = Image.open(path).convert("L").convert("RGB")
    m = np.clip(amap / max(np.percentile(amap, 99.5), 1e-9), 0, 1)
    m = np.array(Image.fromarray(m.astype(np.float32)).resize(img.size, Image.BILINEAR))
    rgba = np.zeros((*m.shape, 4), dtype=np.float32)
    rgba[..., :3] = matplotlib.colors.to_rgb(ATTN_RED)
    rgba[..., 3] = np.clip(m, 0, 1) ** 0.9 * 0.85
    ax.imshow(img); ax.imshow(rgba)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(1.4)
    if title:
        ax.set_title(title, fontsize=7.2, pad=3)


def _plain(ax, path: str, title: str = "", edge: str = FAINT) -> None:
    ax.imshow(Image.open(path).convert("RGB"))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(edge); sp.set_linewidth(1.4)
    if title:
        ax.set_title(title, fontsize=7.2, pad=3)


def fig_attention_classes(F: Figs, D: Data, L: int = 18, tag: str = REF[0]) -> None:
    a, m = _attention(D, "photo", tag)
    j = [18, 22, 24].index(L)
    ph = D.frame("photo", tag)["df"]
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.5))
    for k, c in enumerate(CLASSES):
        pref = EXAMPLE_PREFERRED.get(c)
        cand = m[m.aircraft_uid == pref] if pref else m.iloc[0:0]
        if not len(cand):
            g = ph[(ph.cls == c) & (ph.status_group == "built")]
            g = g if len(g) else ph[ph.cls == c]
            g = g.assign(p=pd.to_numeric(g.p_aircraft, errors="coerce")).sort_values("p", ascending=False)
            cand = m[m.aircraft_uid == g.aircraft_uid.iloc[0]]
        i = cand.index[0]
        _plain(axes[0, k], m.path[i], f"{c} · {m.aircraft_uid[i][:28]}", CLASS_COLOR[c])
        _overlay(axes[1, k], m.path[i], a[i, j], f"CLS attention, layer {L}", CLASS_COLOR[c])
    F.save(fig, "f27_attention_classes", f"Where the model looks: CLS-token attention at layer {L} on one "
           "photo per class", src("photo", tag, L, "cls", "main", "red opacity = attention weight, scaled "
           "to each image's 99.5th percentile; images (c) their owners, internal review only"))


def fig_attention_pairs(F: Figs, D: Data, L: int = 18, tag: str = REF[0]) -> None:
    ap, mp = _attention(D, "photo", tag)
    aa, ma = _attention(D, "patent", tag)
    j = [18, 22, 24].index(L)
    pick = D.pair_pick
    rows = pick[pick.kind.str.startswith("drawing")].copy()
    photo_ac = D.bundles[("photo", tag)]["sets"]["all"].set_index("approved_copy_path")["aircraft_uid"]
    rows["page"] = rows.photo.map(photo_ac)
    rows = rows[rows["query"].isin(set(ma.aircraft_uid)) & rows["page"].isin(set(mp.aircraft_uid))]
    fig, axes = plt.subplots(len(rows), 4, figsize=(9.6, 2.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(rows.iterrows()):
        ia = ma.index[ma.aircraft_uid == row["query"]][0]
        ip = mp.index[mp.aircraft_uid == row["page"]][0]
        _plain(axes[r, 0], ma.path[ia], f"{row['family']}: patent figure", INK["patent"])
        _overlay(axes[r, 1], ma.path[ia], aa[ia, j], f"attention, layer {L}", INK["patent"])
        _plain(axes[r, 2], mp.path[ip], "directory main image", INK["photo"])
        _overlay(axes[r, 3], mp.path[ip], ap[ip, j], f"attention, layer {L}", INK["photo"])
    F.save(fig, "f28_attention_pairs", f"Same aircraft, two media: CLS-token attention at layer {L} on the "
           "patent figure and on the directory's main image", src("both", tag, L, "cls", "main",
           "matched pairs of Section 7; images (c) their owners, internal review only"))


def fig_attention_collapse(F: Figs, D: Data, tag: str = REF[0]) -> pd.DataFrame:
    """Share of CLS attention on the strongest cell, and how often that cell is padding."""
    P = Path(D.cfg["paths"]["pipeline_root"])
    rows = []
    for source in ("photo", "patent"):
        a, m = _attention(D, source, tag)
        man_root = D.root / "2_embedding_extraction" if source == "photo" else P
        man = pd.read_csv(man_root / "processed" / str(size_of(tag)) / "manifest.csv")
        m = m.merge(man[["figure_uid", "orig_w", "orig_h"]], on="figure_uid", how="left")
        S = size_of(tag); g = S // 14
        sc = S / np.maximum(m.orig_w, m.orig_h)
        w, h = (m.orig_w * sc).round().to_numpy(), (m.orig_h * sc).round().to_numpy()
        x0, y0 = (S - w) // 2, (S - h) // 2
        for j, L in enumerate([18, 22, 24]):
            flat = a[:, j].reshape(len(a), -1)
            idx = flat.argmax(1)
            cx, cy = (idx % g) * 14 + 7, (idx // g) * 14 + 7
            pad = ~((cx >= x0) & (cx < x0 + w) & (cy >= y0) & (cy < y0 + h))
            rows.append({"source": source, "layer": L, "top_cell_share": float(np.median(flat.max(1))),
                         "top_cell_on_padding": float(pad.mean()), "n": len(a)})
    t = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.6))
    x = np.arange(3)
    for ax, col, lab, fmt in ((axes[0], "top_cell_share", "median share of CLS attention\non the strongest cell", "{:.0%}"),
                              (axes[1], "top_cell_on_padding", "images whose strongest cell\nlies on the white padding", "{:.0%}")):
        for i, source in enumerate(("photo", "patent")):
            g_ = t[t.source == source]
            ax.bar(x + (i - 0.5) * 0.36, g_[col], width=0.34, color=INK[source],
                   label="evtol.news photos" if source == "photo" else "patent figures")
            for xi, v in zip(x, g_[col]):
                ax.text(xi + (i - 0.5) * 0.36, v + 0.01, fmt.format(v), ha="center", va="bottom", fontsize=6.6)
        ax.set_xticks(x, ["layer 18", "layer 22", "layer 24"]); ax.set_ylabel(lab)
        ax.set_ylim(0, 1.1 if col == "top_cell_on_padding" else max(t[col]) * 1.35)
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    axes[0].legend(loc="upper left")
    F.save(fig, "f29_attention_collapse", "From layer 22 on, the CLS token reads the image through a few "
           "background tokens", src("both", tag, pool="cls", set_name="main", n=f"{t.n.iloc[0]} photos, "
           f"{t.n.iloc[-1]} patent figures; padding = the white border added to make the image square"))
    return t


def build_all(cfg: Dict[str, Any]) -> Tuple[Figs, Data, Dict[str, Any]]:
    D = Data(cfg)
    F = Figs(ev.out_dir(cfg) / "advisor_report")
    extra: Dict[str, Any] = {}
    fig_method_scheme(F, D)
    fig_pipeline(F, D)
    fig_taxonomy(F, D)
    fig_funnel(F, D)
    fig_classes(F, D)
    extra["example_credits"] = fig_examples(F, D)
    fig_siglip_hist(F, D)
    fig_cosine(F, D)
    fig_pca(F, D)
    fig_dist(F, D)
    fig_dendrogram(F, D)
    extra["cluster_share"] = fig_umap_clusters(F, D)
    fig_umap_class(F, D)
    fig_umap_matrices(F, D)
    fig_umap_patent(F, D)
    fig_umap_joint(F, D)
    fig_contingency(F, D)
    extra["intra_inter"] = fig_intra_inter(F, D)
    fig_knn(F, D)
    fig_recall(F, D)
    fig_confusion(F, D)
    fig_maturity(F, D)
    fig_parent(F, D)
    fig_recall_at_k(F, D)
    fig_matching_matrices(F, D)
    extra["pair_credits"] = fig_pairs(F, D)
    extra["pair_pick"] = D.pair_pick
    fig_attention_classes(F, D)
    fig_attention_pairs(F, D)
    extra["attention_collapse"] = fig_attention_collapse(F, D)
    (F.dir.parent / "figures.json").write_text(json.dumps(F.items, indent=1))
    return F, D, extra
