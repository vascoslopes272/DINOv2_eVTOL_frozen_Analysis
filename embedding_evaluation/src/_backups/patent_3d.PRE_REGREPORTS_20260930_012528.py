"""3-D UMAP of the patent-figure embeddings, as one self-contained HTML page you can rotate.

Three views, each with its source line and how to read it:
    main figure (exp3), coloured by the codebook's 12 G1 architecture classes
    the same points, coloured by the 5-class parent (the bridge to the photo page)
    view average (exp4), coloured by the 12 G1 classes

plus a table of how separable the classes are in the full embedding, in 3-D and in 2-D UMAP
(kNN-5 balanced accuracy with the maker held out), so the view can be checked against a number.

The photo twin is ``evtolnews_3d.py`` (EVTOLNEWS_UMAP_3D.html); this page reuses its skeleton.
plotly.js is inlined from the base environment's plotly package, so the page opens offline.
Writes ``1639_LABELLED/3_embedding_evaluation/umap_3d/PATENT_UMAP_3D.html``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, silhouette_score

from . import embedding_protocol as P
from . import embedding_reports as R

PLOTLY_JS = Path("/home/vasco/anaconda3/lib/python3.13/site-packages/plotly/package_data/plotly.min.js")

#: the layer rule's winner in both embedding reports (2026-09-22): layer 24, CLS token.
KEY: P.Key = (24, "cls")

#: 12-class palette in G1_ORDER. The seven atlas ``ARCH_COLOR`` hues are kept unchanged
#: (SLC, TR, CVT, TW, MR, TB, PTC); RC, HB and PFV are re-stepped and DS and SRW added so the
#: full twelve pass the dataviz validator (adjacent pairs in legend order, light surface,
#: validated 2026-09-30). Identity is never colour alone: every class also has a marker symbol,
#: the legend spells the name, and the hover names the class.
G1_COLOR = {"TR": "#eb6834", "CVT": "#1baf7a", "TW": "#eda100", "TB": "#008300",
            "PTC": "#4a3aa7", "DS": "#0f9b8e", "SLC": "#2a78d6", "SRW": "#97880f",
            "MR": "#e87ba4", "RC": "#c92f26", "HB": "#5a55d2", "PFV": "#a4551a"}
G1_SYMBOL = {"TR": "circle", "CVT": "square", "TW": "diamond", "TB": "cross", "PTC": "x",
             "DS": "circle-open", "SLC": "square-open", "SRW": "diamond-open",
             "MR": "circle", "RC": "square", "HB": "diamond", "PFV": "cross"}
#: copied from labeling_evaluation ``metrics.ARCH_NAMES`` (the pillars stay separate).
G1_NAME = {"SLC": "Lift + Cruise", "TR": "Tilt Rotor", "CVT": "Combined vectored thrust",
           "TW": "Tilt Wing", "MR": "Multirotor", "TB": "Tilt Body", "PTC": "Pitch-to-Cruise",
           "HB": "Hoverbike", "DS": "Deflected Slipstream", "PFV": "Personal Flying Vehicle",
           "SRW": "Stopped/Slowed Rotor Wing", "RC": "Rotorcraft"}
#: printed-only rename (display_names.py convention): the data keep the wizard code TR.
G1_DISPLAY = {"TR": ("TP", "Tilt Propulsor")}
SYMBOL5 = {"VT": "circle", "LC": "square", "WM": "diamond", "ER": "cross", "HB": "x"}

READ = ("each point is one aircraft (one vector chosen by the image rule). UMAP reduces each "
        "1 024-number embedding to 3 numbers so that aircraft with similar embeddings sit close "
        "together. The three axes are these 3 numbers: they have no unit, no physical meaning and "
        "an arbitrary orientation. Only closeness is meaningful. Drag to rotate, scroll to zoom, "
        "click a class in the legend to hide it, double-click to show it alone.")


def _umap(X: np.ndarray, n: int, key: str, cache: Path, seed: int = 42) -> np.ndarray:
    f = cache / f"umap{n}d_{key}.npy"
    if f.exists() and np.load(f).shape[0] == len(X):
        return np.load(f)
    import umap
    e = umap.UMAP(n_components=n, n_neighbors=15, min_dist=0.1, metric="cosine",
                  random_state=seed).fit_transform(X)
    np.save(f, e)
    return e


def _knn_bal(Z: np.ndarray, y: np.ndarray, makers: np.ndarray, cosine: bool) -> float:
    """kNN-5 balanced accuracy, maker held out — the protocol's neighbour rule, in the
    full space (cosine) or in a low-dimensional UMAP map (Euclidean)."""
    if cosine:
        nn = P.neighbours(Z, makers, P.K)
    else:
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        nn = P.neighbours(Z, makers, P.K, S=-D)
    return balanced_accuracy_score(y, P.vote(nn, y))


def separability(spaces: List[tuple], cache: Path) -> pd.DataFrame:
    rows = []
    for rule, label_name, X, y, makers, key in spaces:
        for name, Z, cos in ((f"full embedding (1 024 numbers)", X, True),
                             ("3-D UMAP", _umap(X, 3, key, cache), False),
                             ("2-D UMAP", _umap(X, 2, key, cache), False)):
            rows.append({"image rule": rule, "label": label_name, "space": name,
                         "kNN-5, maker held out": _knn_bal(Z, y, makers, cos),
                         "silhouette": float(silhouette_score(Z, y, metric="cosine" if cos else "euclidean"))})
    return pd.DataFrame(rows)


def _legend(c: str) -> str:
    code, name = G1_DISPLAY.get(c, (c, G1_NAME[c]))
    return f"{code} {name}"


def _traces(emb: np.ndarray, d: pd.DataFrame, col: str, classes: List[str],
            color: Dict[str, str], symbol: Dict[str, str], legend) -> List[Dict[str, Any]]:
    out = []
    for c in classes:
        m = (d[col] == c).to_numpy()
        if not m.any():
            continue
        sub, e = d[m], emb[m]
        hover = [f"{html.escape(str(i))}<br>{html.escape(str(n))}<br>{html.escape(str(k))} · {legend(c)}"
                 for i, n, k in zip(sub["aircraft_id"], sub["name"], sub["company"])]
        out.append({"type": "scatter3d", "mode": "markers", "name": f"{legend(c)} ({m.sum()})",
                    "x": e[:, 0].round(4).tolist(), "y": e[:, 1].round(4).tolist(), "z": e[:, 2].round(4).tolist(),
                    "text": hover, "hoverinfo": "text",
                    "marker": {"size": 3.2, "color": color[c], "symbol": symbol[c], "opacity": 0.85,
                               "line": {"width": 0}}})
    return out


def write() -> Path:
    S = R.load_patents()
    a = S.aircraft.copy()
    acp = S.extra["acp"]
    a["name"] = a.aircraft_id.map(acp["aircraft_name"]).fillna("")
    a["company"] = a["company"].replace("", "individual inventor")
    y12, y5, mk = S.y, a["label5"].to_numpy(dtype=object), S.makers
    classes12 = [c for c in R.G1_ORDER if c in set(y12)]

    out = R.PAT / "3_embedding_evaluation/umap_3d"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    X3 = S.rules["exp3 main figure"][KEY]
    X4 = S.rules["exp4 view average"][KEY]
    e3 = _umap(X3, 3, f"exp3_{KEY[0]}_{KEY[1]}", cache)
    e4 = _umap(X4, 3, f"exp4_{KEY[0]}_{KEY[1]}", cache)
    sep = separability([
        ("exp3 main figure", "G1 (12 classes)", X3, y12, mk, f"exp3_{KEY[0]}_{KEY[1]}"),
        ("exp3 main figure", "parent (5 classes)", X3, y5, mk, f"exp3_{KEY[0]}_{KEY[1]}"),
        ("exp4 view average", "G1 (12 classes)", X4, y12, mk, f"exp4_{KEY[0]}_{KEY[1]}"),
    ], cache)

    base = (f"dinov2-large, 518 px, {P.mname(KEY)}; {len(a)} aircraft, G1 labels of the master export; "
            "3-D UMAP n_neighbors 15, min_dist 0.1, cosine, seed 42")
    views = [
        ("Main figure (exp3), coloured by the 12 G1 architecture classes",
         _traces(e3, a, "label", classes12, G1_COLOR, G1_SYMBOL, _legend),
         f"{base}; one vector per aircraft = the figure the labeller marked as main"),
        ("The same points, coloured by the 5-class parent (the photo page's classes)",
         _traces(e3, a, "label5", R.CLASSES5, R.CLASS_COLOR, SYMBOL5,
                 lambda c: f"{c} {R.CLASS5_NAME[c]}"),
         f"{base}; same UMAP fit as view 1, G1 folded to its parent class"),
        ("View average (exp4), coloured by the 12 G1 architecture classes",
         _traces(e4, a, "label", classes12, G1_COLOR, G1_SYMBOL, _legend),
         f"{base}; one vector per aircraft = mean of the best figure in each of Perspective, Plan and Side"),
    ]
    sep_html = sep.to_html(index=False, float_format=lambda v: f"{v:.2f}", border=0, classes="sep")
    divs, scripts = [], []
    for i, (title, traces, source) in enumerate(views):
        divs.append(f'<section><h2>{i + 1}. {html.escape(title)}</h2><div id="v{i}" class="plot"></div>'
                    f'<p class="src"><b>Source:</b> {html.escape(source)}.</p>'
                    f'<p class="src"><b>How to read:</b> {html.escape(READ)}</p></section>')
        layout = {"margin": {"l": 0, "r": 0, "t": 10, "b": 0},
                  "legend": {"itemsizing": "constant", "x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
                  "scene": {ax: {"title": {"text": f"UMAP {k} (no unit)"}, "showticklabels": False}
                            for ax, k in (("xaxis", 1), ("yaxis", 2), ("zaxis", 3))},
                  "paper_bgcolor": "#ffffff"}
        scripts.append(f"Plotly.newPlot('v{i}', {json.dumps(traces)}, {json.dumps(layout)}, "
                       "{responsive: true, displaylogo: false});")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Patent-figure embeddings in 3-D</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: #0b0b0b; background: #ffffff;
       margin: 0 auto; max-width: 1100px; padding: 16px; }}
h1 {{ font-size: 20px; margin: 4px 0 6px; }} h2 {{ font-size: 15px; margin: 22px 0 4px; }}
.plot {{ width: 100%; height: 640px; border: 1px solid #ecebe7; }}
.src {{ font-size: 12px; color: #52514e; margin: 4px 0; line-height: 1.4; }}
table.sep {{ border-collapse: collapse; font-size: 13px; margin: 6px 0 4px; }}
table.sep th, table.sep td {{ padding: 4px 10px; border-bottom: 1px solid #ecebe7; text-align: left; }}
.note {{ font-size: 13px; line-height: 1.5; max-width: 820px; }}
</style></head><body>
<h1>DINOv2 embeddings of the patent figures in three dimensions</h1>
<p class="note">Internal. Patent figures from 1639_LABELLED, labels from the labelling wizard.
Hover over a point for the aircraft. Do the classes become more distinct in 3-D? The table measures
it: the share of aircraft whose 5 nearest neighbours (never from the same maker) vote for the right
class, averaged over the classes (chance {1 / len(classes12):.3f} for 12 classes, 0.20 for 5), in the
full embedding and in the 3-D and 2-D maps. The photo twin of this page is EVTOLNEWS_UMAP_3D.html.</p>
{sep_html}
<p class="src">UMAP maps are fitted without labels, but they can both blur real separation and
exaggerate local clumping (here the map rows score above the full-embedding row). Judge separation
by the full-embedding row; the maps are for looking, the numbers for deciding.</p>
{''.join(divs)}
<script>{PLOTLY_JS.read_text(encoding="utf-8")}</script>
<script>{''.join(scripts)}</script>
</body></html>"""
    f = out / "PATENT_UMAP_3D.html"
    f.write_text(page, encoding="utf-8")
    sep.to_csv(out / "separability_2d_3d.csv", index=False)
    return f
