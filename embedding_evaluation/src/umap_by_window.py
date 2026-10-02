"""The embedding maps window by window: brief 4.5, "The maps, window by window" (added 2026-09-30).

Design (fixed before any map was looked at; the brief's Method states it as written here).

Data. The same aircraft, vectors and years as ``class_evolution`` (its loaders are reused, so the n are the
same): the 663 drawing aircraft (registers, exp3 main figure, layer 24 CLS) on their parent classes, dated by
their patent's priority year; the 1 157 evtol.news hero images (registers, layer 24 CLS), 949 of them dated by
the development year both language-model readers agreed on. Windows as in the Labelling Analysis: <= 2011,
2012-15, 2016-19, 2020-23, 2024-26 (partial).

Maps. ONE UMAP fit per source on all its aircraft (the undated photo aircraft are in the fit but in no window
panel), with the parameters and seed of the document's maps (15 neighbours, min. distance 0.1, cosine, seed
42); the 2-D fits are the document's own cached coordinates (``embedding_reports._umap``). Never a fit per
window: a separate fit puts each window on its own arbitrary axes, so positions could not be compared across
panels. The 3-D pages use one 3-D fit per source with the same parameters (the patent fit is the cache of
``patent_3d``'s registers page).

Per window: the window's aircraft in class colour, every other aircraft in light grey; each class's centre in
that window = the mean of its aircraft's map coordinates (the map counterpart of test 3's centroid, which is
also a mean); the track of the centres over the windows so far. A class with fewer than 15 aircraft in a
window (``class_evolution.MIN_N``) has a hollow, faded centre that is never joined to the track. The map only
illustrates: the drift is tested in the full 1 024-number space (``class_evolution`` test 3).

Separation by window (the optional check). Within each window, the protocol's separation ratio
(``embedding_protocol.separation``: mean cosine distance between classes over mean distance within a class,
pairs of the same maker left out), with a 95 % interval from 1 000 resamples of whole makers (seed 42), makers
resampled jointly over the windows so that the difference 2020-23 minus 2016-19 (the pair of test 3) has its
own interval. A trend is claimed only if that interval excludes zero. The ratio also moves with the class mix
of the window, which changes over time.

Outputs: 1639_LABELLED/3_embedding_evaluation/class_evolution/umap_by_window_{coords_drawings, coords_photos,
centroids, separation}.csv; 1639_LABELLED/3_embedding_evaluation/umap_3d/PATENT_UMAP_3D_BY_WINDOW.html and
EVTOLNEWS_DS/3_embedding_evaluation/umap_3d/EVTOLNEWS_UMAP_3D_BY_WINDOW.html (private). The 2-D figures are
drawn by ``draw`` when the brief is built (analysis_brief, figures in figs_brief/).
Run: ``from src import umap_by_window as UW; UW.run()`` (notebook 32, step "umap_by_window").
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from . import class_evolution as CE
from . import embedding_protocol as P
from . import embedding_reports as ER

REG, KEY = CE.REG, CE.KEY
OUT = CE.OUT
T_PAT_REG = ER.PAT / "3_embedding_evaluation" / "embedding_protocol_registers"
T_EVN_REG = ER.EVN / "3_embedding_evaluation" / "embedding_analysis_registers" / "tables"
U3D_PAT = ER.PAT / "3_embedding_evaluation" / "umap_3d"
U3D_EVN = ER.EVN / "3_embedding_evaluation" / "umap_3d"
HTML_PAT = U3D_PAT / "PATENT_UMAP_3D_BY_WINDOW.html"
HTML_EVN = U3D_EVN / "EVTOLNEWS_UMAP_3D_BY_WINDOW.html"
SOURCES = ("drawings", "photos")
CSV = {"coords_drawings": OUT / "umap_by_window_coords_drawings.csv",
       "coords_photos": OUT / "umap_by_window_coords_photos.csv",
       "centroids": OUT / "umap_by_window_centroids.csv",
       "separation": OUT / "umap_by_window_separation.csv"}
OUTPUTS = list(CSV.values()) + [HTML_PAT, HTML_EVN]
WIN = CE.WIN_NAMES
MIN_N, N_BOOT, SEED = CE.MIN_N, CE.N_BOOT, CE.SEED
CLASSES = ER.CLASSES5
SYMBOL5 = {"VT": "circle", "LC": "square", "WM": "diamond", "ER": "cross", "HB": "x"}
OPEN5 = {"VT": "circle-open", "LC": "square-open", "WM": "diamond-open", "ER": "cross", "HB": "x"}
PLOTLY_JS = Path("/home/vasco/anaconda3/lib/python3.13/site-packages/plotly/package_data/plotly.min.js")
WIN_TICK = {"<= 2011": "≤ 2011", "2012-15": "2012-15", "2016-19": "2016-19", "2020-23": "2020-23",
            "2024-26 (partial)": "2024-26 (partial)"}
UMAP_PARAMS = "UMAP 15 neighbours, min. distance 0.1, cosine, seed 42, one fit on all aircraft"


# ── data ─────────────────────────────────────────────────────────────────────
def load(source: str) -> Tuple[pd.DataFrame, np.ndarray, ER.Source]:
    """Every aircraft of the source (the UMAP fit set) with its class, maker, name and year: the years and the
    dated set come from class_evolution's loaders, so the n are those of 4.5."""
    ER.TAG = REG
    if source == "drawings":
        S = ER.load_patents()
        S.table_dir = T_PAT_REG
        dated, _ = CE.load_drawings()
        a = S.aircraft[["aircraft_id", "label5", "maker"]].rename(columns={"label5": "label"}).copy()
        a["name"] = a.aircraft_id.map(S.extra["acp"]["aircraft_name"]).fillna("").astype(str)
        X = S.rules[S.primary_rule][KEY]
    else:
        S = ER.load_evtolnews()
        S.table_dir = T_EVN_REG
        dated, _, _ = CE.load_photos()
        a = S.aircraft[["aircraft_id", "label", "maker", "title"]].rename(columns={"title": "name"}).copy()
        X = S.rules[S.primary_rule][KEY]
    yr = dated.set_index("aircraft_id")["year"]
    a["year"] = a.aircraft_id.map(yr)
    a["window"] = a.year.map(CE.window_of)
    return a.reset_index(drop=True), X, S


def umap2(S: ER.Source, X: np.ndarray) -> np.ndarray:
    """The document's own 2-D map (same function, same cache file)."""
    return ER._umap(S, X, f"{S.primary_rule}_{KEY[0]}_{KEY[1]}".replace(" ", "_"))


def umap3(source: str, X: np.ndarray) -> np.ndarray:
    from . import patent_3d
    if source == "drawings":           # patent_3d's registers page: the same fit, reused
        cache = U3D_PAT / "cache" / REG
        key = f"exp3_{KEY[0]}_{KEY[1]}"
    else:
        cache = U3D_EVN / "cache"
        key = f"hero_image_{KEY[0]}_{KEY[1]}"
    cache.mkdir(parents=True, exist_ok=True)
    return patent_3d._umap(X, 3, key, cache)


# ── centres ──────────────────────────────────────────────────────────────────
def centroids(a: pd.DataFrame, E: np.ndarray, source: str, space: str) -> pd.DataFrame:
    """Mean map position of each class in each window (and n; small = fewer than MIN_N aircraft)."""
    rows = []
    for k, w in enumerate(WIN):
        for c in CLASSES:
            m = ((a.window == w) & (a.label == c)).to_numpy()
            r = {"source": source, "space": space, "window": w, "window_idx": k, "class": c, "n": int(m.sum()),
                 "n_window": int((a.window == w).sum()), "small": bool(m.sum() < MIN_N)}
            if m.any():
                for j, ax in enumerate("xyz"[:E.shape[1]]):
                    r[ax] = float(E[m, j].mean())
            rows.append(r)
    return pd.DataFrame(rows)


# ── the optional check: separation within each window ───────────────────────
def separation(a: pd.DataFrame, X: np.ndarray, source: str) -> pd.DataFrame:
    """The protocol's separation ratio within each window, with a joint maker bootstrap over the windows and
    the difference 2020-23 minus 2016-19."""
    d = a[a.window.notna()].reset_index()
    Xd = P.l2(X[d["index"].to_numpy()])
    makers = d.maker.to_numpy(dtype=object)
    keys, inv = np.unique(makers, return_inverse=True)
    rng = np.random.default_rng(SEED)
    Wm = np.stack([np.bincount(rng.integers(0, len(keys), len(keys)), minlength=len(keys))
                   for _ in range(N_BOOT)]).astype(float)
    Wa = Wm[:, inv]                                           # weight of each aircraft in each resample
    obs, boot = {}, {}
    for w in WIN:
        idx = np.flatnonzero((d.window == w).to_numpy())
        lab, mk = d.label.to_numpy()[idx], makers[idx]
        D = 1.0 - Xd[idx] @ Xd[idx].T
        diffm = mk[:, None] != mk[None, :]
        same = (lab[:, None] == lab[None, :]) & diffm
        betw = (lab[:, None] != lab[None, :]) & diffm

        def ratio(W):
            W = np.atleast_2d(W)
            nb, db = ((W @ (D * betw)) * W).sum(1), ((W @ betw) * W).sum(1)
            nw, dw = ((W @ (D * same)) * W).sum(1), ((W @ same) * W).sum(1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return (nb / db) / (nw / dw)
        obs[w] = float(ratio(np.ones(len(idx)))[0])
        chk = P.separation(Xd[idx], lab, mk, n_perm=0)["ratio"]
        assert abs(chk - obs[w]) < 1e-9, (w, chk, obs[w])
        boot[w] = ratio(Wa[:, idx])
    rows = []
    for w in WIN:
        n = int((d.window == w).sum())
        lo, hi = np.nanquantile(boot[w], [0.025, 0.975])
        rows.append({"source": source, "window": w, "n_aircraft": n, "n_makers": int(d[d.window == w].maker.nunique()),
                     "ratio": obs[w], "ci_lo": float(lo), "ci_hi": float(hi)})
    a0, a1 = CE.DRIFT
    diff = boot[a1] - boot[a0]
    lo, hi = np.nanquantile(diff, [0.025, 0.975])
    rows.append({"source": source, "window": f"{a1} minus {a0}", "n_aircraft": np.nan, "n_makers": np.nan,
                 "ratio": obs[a1] - obs[a0], "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(rows)


# ── the 2-D small multiples ──────────────────────────────────────────────────
def draw(coords: pd.DataFrame, cents: pd.DataFrame, source: str):
    """One panel per window (3 + 2) and a key: window aircraft in class colour, the rest grey, each class's
    centre in the window and its track so far. Returns the matplotlib figure (the caller saves it)."""
    import matplotlib.pyplot as plt
    from matplotlib import patheffects as pe
    from matplotlib.lines import Line2D
    ce = cents[(cents.source == source) & (cents.space == "2d")]
    xy = coords[["x", "y"]].to_numpy()
    pad = 0.04 * (xy.max(0) - xy.min(0))
    lo, hi = xy.min(0) - pad, xy.max(0) + pad
    fig, axes = plt.subplots(2, 3, figsize=(7.3, 5.05))
    halo = [pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()]
    for k, w in enumerate(WIN):
        ax = axes.flat[k]
        m = (coords.window == w).to_numpy()
        ax.scatter(xy[~m, 0], xy[~m, 1], s=2.2, color=ER.FAINT, alpha=0.55, linewidths=0, zorder=1)
        for c in CLASSES:
            mc = m & (coords.label == c).to_numpy()
            ax.scatter(xy[mc, 0], xy[mc, 1], s=6, marker=ER.CLASS_MARK[c], color=ER.CLASS_COLOR[c], alpha=0.6,
                       linewidths=0, zorder=2)
        for c in CLASSES:
            t = ce[(ce["class"] == c) & (ce.window_idx <= k) & (ce.n > 0)].sort_values("window_idx")
            col = ER.CLASS_COLOR[c]
            for (_, r0), (_, r1) in zip(t.iloc[:-1].iterrows(), t.iloc[1:].iterrows()):
                if r1.window_idx == r0.window_idx + 1 and not r0.small and not r1.small:
                    ax.plot([r0.x, r1.x], [r0.y, r1.y], color=col, lw=1.5, zorder=4, path_effects=halo,
                            solid_capstyle="round")
            for r in t.itertuples():
                cur = r.window_idx == k
                ms = 8.5 if cur else 4.2
                if r.small:
                    ax.plot(r.x, r.y, marker=ER.CLASS_MARK[c], ms=ms, mfc="white", mec=col, mew=1.1, alpha=0.6,
                            zorder=5)
                else:
                    ax.plot(r.x, r.y, marker=ER.CLASS_MARK[c], ms=ms, mfc=col, mec="white" if not cur else ER.INK,
                            mew=0.9 if cur else 0.6, zorder=6 if cur else 5)
        ax.set_xlim(lo[0], hi[0]), ax.set_ylim(lo[1], hi[1])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(ER.FAINT), s.set_linewidth(0.6)
        ax.set_title(f"{WIN_TICK[w]}, n = {int(m.sum())}", fontsize=8, loc="left", fontweight="bold")
    ax = axes.flat[5]
    ax.axis("off")
    hand = [Line2D([], [], color=ER.CLASS_COLOR[c], marker=ER.CLASS_MARK[c], ms=5.5, lw=0,
                   label=f"{c} {ER.CLASS5_NAME[c].replace(' (Multicopter)', '').replace(' / PFD', '')}")
            for c in CLASSES]
    hand += [Line2D([], [], color=ER.FAINT, marker="o", ms=3.5, lw=0, label="aircraft of other windows"
                    + (" or undated" if source == "photos" else "")),
             Line2D([], [], color=ER.MUTED, marker="o", ms=8, mec=ER.INK, mew=0.9, lw=0,
                    label="class centre in the window"),
             Line2D([], [], color=ER.MUTED, marker="o", ms=4.2, mec="white", lw=1.5,
                    label="centres of earlier windows,\njoined in time order"),
             Line2D([], [], color=ER.MUTED, marker="o", ms=7, mfc="white", mew=1.1, lw=0, alpha=0.6,
                    label=f"centre of fewer than {MIN_N}\naircraft (not joined)")]
    ax.legend(handles=hand, loc="center left", fontsize=6.8, handletextpad=0.5, labelspacing=0.55,
              borderaxespad=0)
    fig.tight_layout(w_pad=0.4, h_pad=0.9)
    return fig


def map_moves(cents: pd.DataFrame, coords: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For the text: each class's 2-D centre move between the two windows of test 3, as a share of the map's
    spread (median distance of all aircraft to the map's centre); NaN when either window is small."""
    a0, a1 = CE.DRIFT
    rows = []
    for s in SOURCES:
        xy = coords[s][["x", "y"]].to_numpy()
        spread = float(np.median(np.linalg.norm(xy - xy.mean(0), axis=1)))
        ce = cents[(cents.source == s) & (cents.space == "2d")].set_index(["class", "window"])
        for c in CLASSES:
            r0, r1 = ce.loc[(c, a0)], ce.loc[(c, a1)]
            ok = not (r0.small or r1.small)
            mv = float(np.hypot(r1.x - r0.x, r1.y - r0.y)) / spread if ok else np.nan
            rows.append({"source": s, "class": c, "move_share_of_spread": mv})
    return pd.DataFrame(rows)


# ── the 3-D pages with a window slider ───────────────────────────────────────
def _fade(hexcol: str, t: float = 0.55) -> str:
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * t) for v in (r, g, b))


def _n(v: int) -> str:
    return f"{int(v):,}".replace(",", " ")


def _hover(a: pd.DataFrame, source: str) -> List[str]:
    yl = "priority year" if source == "drawings" else "development year"
    out = []
    for r in a.itertuples():
        y = f"{yl} {int(r.year)}" if pd.notna(r.year) else "no agreed year"
        nm = f"<br>{html.escape(str(r.name))}" if str(r.name) and str(r.name) != str(r.aircraft_id) else ""
        out.append(f"{html.escape(str(r.aircraft_id))}{nm}<br>{r.label} {html.escape(ER.CLASS5_NAME[r.label])}"
                   f"<br>{y}")
    return out


def _frame_data(a: pd.DataFrame, E: np.ndarray, ce: pd.DataFrame, text: List[str], k: int | None) -> List[Dict]:
    """The 16 traces of one state: grey others, 5 class point sets, 5 track lines, 5 centre marker sets.
    ``k`` = window index, None = all windows."""
    rnd = lambda v: np.round(v, 3).tolist()  # noqa: E731
    T = np.array(text, dtype=object)
    inw = a.window.notna().to_numpy() if k is None else (a.window == WIN[k]).to_numpy()
    tr = [{"x": rnd(E[~inw, 0]), "y": rnd(E[~inw, 1]), "z": rnd(E[~inw, 2]), "text": T[~inw].tolist()}]
    for c in CLASSES:
        m = inw & (a.label == c).to_numpy()
        tr.append({"x": rnd(E[m, 0]), "y": rnd(E[m, 1]), "z": rnd(E[m, 2]), "text": T[m].tolist()})
    last = len(WIN) - 1 if k is None else k
    lines, marks = [], []
    for c in CLASSES:
        t = ce[(ce["class"] == c) & (ce.window_idx <= last) & (ce.n > 0)].sort_values("window_idx")
        lx, ly, lz = [], [], []
        for (_, r0), (_, r1) in zip(t.iloc[:-1].iterrows(), t.iloc[1:].iterrows()):
            if r1.window_idx == r0.window_idx + 1 and not r0.small and not r1.small:
                lx += [r0.x, r1.x, None]
                ly += [r0.y, r1.y, None]
                lz += [r0.z, r1.z, None]
        lines.append({"x": lx, "y": ly, "z": lz})
        col = ER.CLASS_COLOR[c]
        cur = [(k is not None and r.window_idx == k) for r in t.itertuples()]
        marks.append({"x": rnd(t.x.to_numpy()), "y": rnd(t.y.to_numpy()), "z": rnd(t.z.to_numpy()),
                      "text": [f"{c} {html.escape(ER.CLASS5_NAME[c])}<br>centre of {WIN_TICK[r.window]}, "
                               f"{r.n} aircraft" + (f" (fewer than {MIN_N}, not joined)" if r.small else "")
                               for r in t.itertuples()],
                      "marker": {"size": [15 if cu else (11 if k is None else 8) for cu in cur],
                                 "color": [_fade(col) if r.small else col for r in t.itertuples()],
                                 "symbol": [OPEN5[c] if r.small else SYMBOL5[c] for r in t.itertuples()],
                                 "line": {"width": 1, "color": "#1f1f1e"}}})
    return tr + lines + marks


def write_page(source: str, a: pd.DataFrame, E: np.ndarray, cents: pd.DataFrame, path: Path) -> Path:
    ce = cents[(cents.source == source) & (cents.space == "3d")]
    text = _hover(a, source)
    base = []
    base.append({"type": "scatter3d", "mode": "markers", "name": "other aircraft", "hoverinfo": "text",
                 "marker": {"size": 2.2, "color": "#c9c8c2", "opacity": 0.35, "line": {"width": 0}}})
    for c in CLASSES:
        base.append({"type": "scatter3d", "mode": "markers", "name": f"{c} {ER.CLASS5_NAME[c]}", "hoverinfo": "text",
                     "legendgroup": c,
                     "marker": {"size": 2.4 if c in ("ER", "HB") else 3.2, "color": ER.CLASS_COLOR[c],
                                "symbol": SYMBOL5[c], "opacity": 0.7,
                                "line": {"width": 0}}})
    for c in CLASSES:
        base.append({"type": "scatter3d", "mode": "lines", "name": f"{c} track", "showlegend": False,
                     "legendgroup": c, "hoverinfo": "skip", "line": {"width": 5, "color": ER.CLASS_COLOR[c]}})
    for c in CLASSES:
        base.append({"type": "scatter3d", "mode": "markers", "name": f"{c} centres", "showlegend": False,
                     "legendgroup": c, "hoverinfo": "text"})
    states = [(w, i) for i, w in enumerate(WIN)] + [("all windows", None)]
    frames = []
    for name, k in states:
        n = int(a.window.notna().sum() if k is None else (a.window == WIN[k]).sum())
        lab = WIN_TICK[name] if k is not None else "all windows"
        frames.append({"name": lab, "data": _frame_data(a, E, ce, text, k),
                       "layout": {"title": {"text": f"{lab}: {_n(n)} aircraft in class colour", "x": 0.01,
                                            "font": {"size": 14}}}})
    data = [{**b, **d} for b, d in zip(base, frames[-1]["data"])]
    steps = [{"label": f["name"], "method": "animate",
              "args": [[f["name"]], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                                     "transition": {"duration": 0}}]} for f in frames]
    layout = {"margin": {"l": 0, "r": 0, "t": 36, "b": 0}, "title": frames[-1]["layout"]["title"],
              "legend": {"itemsizing": "constant", "x": 0.01, "y": 0.93, "bgcolor": "rgba(255,255,255,0.85)"},
              "scene": {**{ax: {"title": {"text": f"UMAP {j} (no unit)"}, "showticklabels": False}
                           for ax, j in (("xaxis", 1), ("yaxis", 2), ("zaxis", 3))}, "uirevision": "keep"},
              "uirevision": "keep", "paper_bgcolor": "#ffffff",
              "sliders": [{"active": len(frames) - 1, "steps": steps, "x": 0.05, "len": 0.9, "y": 0.02,
                           "currentvalue": {"prefix": "Window: ", "font": {"size": 13}}, "pad": {"t": 10}}]}
    if source == "drawings":
        h1 = "Patent drawings in 3-D, window by window"
        src = (f"{ER.model_short(REG)}, 518 px, layer 24 CLS; {_n(len(a))} patent-drawing aircraft (exp3 main figure), "
               "parent class of the G1 type folded to 5 classes; windows by the priority year of the aircraft's "
               f"patent; 3-D {UMAP_PARAMS} (the fit of the registers 3-D page)")
        yrs = "Years are the priority year of the aircraft's patent."
    else:
        h1 = "evtol.news photographs in 3-D, window by window"
        nd = int(a.window.notna().sum())
        src = (f"{ER.model_short(REG)}, 518 px, layer 24 CLS; {_n(len(a))} evtol.news aircraft, hero image, directory "
               f"class; windows by the development year both language-model readers agreed on ({_n(nd)} of "
               f"{_n(len(a))} aircraft; the others are in the fit and always grey); 3-D {UMAP_PARAMS}")
        yrs = ("Development years were read from the directory pages by two language-model readers and are not "
               "yet reviewed by the author.")
    read = ("each point is one aircraft. UMAP reduces each 1 024-number embedding to 3 numbers so that aircraft "
            "with similar embeddings sit close together; the axes have no unit and no meaning, only closeness "
            "counts. The slider picks a development window: its aircraft are in class colour, all others faint "
            "grey. The large marker of each class is the mean position of its aircraft in that window; the line "
            f"joins the centres of the earlier windows in time order. A hollow or pale centre rests on fewer than "
            f"{MIN_N} aircraft and is not joined. One fit covers all windows, so positions compare across the "
            "slider. Drag to rotate, scroll to zoom, click a class in the legend to hide it.")
    js = (f"Plotly.newPlot('v0', {json.dumps(data)}, {json.dumps(layout)}, "
          "{responsive: true, displaylogo: false}).then(function(gd) { Plotly.addFrames(gd, "
          f"{json.dumps(frames)}); }});")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Embeddings by window, 3-D</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: #0b0b0b; background: #ffffff;
       margin: 0 auto; max-width: 1100px; padding: 16px; }}
h1 {{ font-size: 20px; margin: 4px 0 6px; }}
.plot {{ width: 100%; height: 700px; border: 1px solid #ecebe7; }}
.src {{ font-size: 12px; color: #52514e; margin: 4px 0; line-height: 1.4; }}
.note {{ font-size: 13px; line-height: 1.5; max-width: 820px; }}
</style></head><body>
<h1>{h1}</h1>
<p class="note">Internal. Where each class sits in the embedding, window by window. {html.escape(yrs)}
The map is for looking: a UMAP map keeps neighbourhoods, not distances, so whether a class centre moves is
tested in the full embedding, not here.</p>
<div id="v0" class="plot"></div>
<p class="src"><b>Source:</b> {html.escape(src)}.</p>
<p class="src"><b>How to read:</b> {html.escape(read)}</p>
<script>{PLOTLY_JS.read_text(encoding="utf-8")}</script>
<script>{js}</script>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    return path


# ── run ──────────────────────────────────────────────────────────────────────
def run() -> Dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    cents, seps, out = [], [], {}
    for s in SOURCES:
        a, X, S = load(s)
        e2, e3 = umap2(S, X), umap3(s, X)
        co = a[["aircraft_id", "label", "maker", "year", "window"]].copy()
        co["x"], co["y"] = e2[:, 0], e2[:, 1]
        co.to_csv(CSV[f"coords_{s}"], index=False)
        out[f"coords_{s}"] = co
        c2, c3 = centroids(a, e2, s, "2d"), centroids(a, e3, s, "3d")
        cents += [c2, c3]
        seps.append(separation(a, X, s))
        write_page(s, a, e3, pd.concat([c2, c3]), HTML_PAT if s == "drawings" else HTML_EVN)
        print(f"[{s}] {len(a)} aircraft in the fit, {int(a.window.notna().sum())} dated; "
              + ", ".join(f"{WIN_TICK[w]} {int((a.window == w).sum())}" for w in WIN), flush=True)
    # rounded so that the last-bit noise of multithreaded BLAS sums never changes the files between runs
    out["centroids"] = pd.concat(cents, ignore_index=True).round(9)
    out["separation"] = pd.concat(seps, ignore_index=True).round(9)
    out["centroids"].to_csv(CSV["centroids"], index=False)
    out["separation"].to_csv(CSV["separation"], index=False)
    print("wrote", ", ".join(p.name for p in OUTPUTS))
    return out


def load_saved() -> Dict[str, pd.DataFrame]:
    return {k: pd.read_csv(p, keep_default_na=False, na_values=[""]) for k, p in CSV.items()}


if __name__ == "__main__":
    run()
