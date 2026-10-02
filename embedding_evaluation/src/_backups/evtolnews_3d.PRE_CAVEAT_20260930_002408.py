"""3-D UMAP of the evtol.news photo embeddings, as one self-contained HTML page you can rotate.

Three views, each with its source line and how to read it:
    photos (1 157 aircraft), coloured by directory class
    patent figures (the same model), coloured by the parent of the codebook topType
    photos and patents in one fit, coloured by class, marker by source

plus a table of how separable the classes are in the full embedding, in 3-D and in 2-D UMAP
(kNN-5 balanced accuracy with the maker held out), so the view can be checked against a number.

plotly.js is inlined from the base environment's plotly package, so the page opens offline.
Writes ``<evtolnews.root>/3_embedding_evaluation/advisor_report/EVTOLNEWS_UMAP_3D.html``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, silhouette_score

from . import embedding_metrics as em
from . import evtolnews_eval as ev
from . import evtolnews_figures as FG

PLOTLY_JS = Path("/home/vasco/anaconda3/lib/python3.13/site-packages/plotly/package_data/plotly.min.js")
SYMBOL = {"VT": "circle", "LC": "square", "WM": "diamond", "ER": "cross", "HB": "x"}
READ = ("each point is one aircraft (its main image). UMAP reduces each 1 024-number embedding to 3 numbers so "
        "that aircraft with similar embeddings sit close together. The three axes are these 3 numbers: they have "
        "no unit, no physical meaning and an arbitrary orientation. Only closeness is meaningful. Drag to rotate, "
        "scroll to zoom, click a class in the legend to hide it, double-click to show it alone.")


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
    Z = em._l2(Z) if cosine else Z
    if cosine:
        nn = ev._neighbours(Z, makers, 10)
    else:  # Euclidean neighbours in the low-dimensional map
        D = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
        D[makers[:, None] == makers[None, :]] = np.inf
        nn = np.argsort(D, axis=1)[:, :10]
    return balanced_accuracy_score(y, ev._vote(nn, y, 5))


def separability(D: FG.Data, cache: Path, tag: str, L: int, pool: str) -> pd.DataFrame:
    rows = []
    for source in ("photo", "patent"):
        d = D.frame(source, tag)["df"]
        X = D.X(source, tag, L, pool)
        y, mk = d["cls"].to_numpy(), d["maker"].to_numpy()
        for name, Z, cos in (("full embedding (1 024 numbers)", X, True),
                             ("3-D UMAP", _umap(X, 3, f"{source}_{tag}_L{L}_{pool}", cache), False),
                             ("2-D UMAP", _umap(X, 2, f"{source}_{tag}_L{L}_{pool}", cache), False)):
            rows.append({"source": "photos" if source == "photo" else "patent figures", "space": name,
                         "kNN-5, maker held out": _knn_bal(Z, y, mk, cos),
                         "silhouette": float(silhouette_score(Z, y, metric="cosine" if cos else "euclidean"))})
    return pd.DataFrame(rows)


def _traces(emb: np.ndarray, d: pd.DataFrame, symbol_by: str | None = None) -> List[Dict[str, Any]]:
    out = []
    groups = [(c, d["cls"] == c) for c in ev.CLASSES]
    for c, m in groups:
        sub = d[m.to_numpy()]
        e = emb[m.to_numpy()]
        if not len(sub):
            continue
        hover = [f"{html.escape(str(t))}<br>{html.escape(str(k))}<br>class {c}"
                 for t, k in zip(sub["label"], sub["extra"])]
        sym = [SYMBOL[c]] * len(sub) if symbol_by is None else \
            ["circle" if s == "photo" else "diamond-open" for s in sub[symbol_by]]
        out.append({"type": "scatter3d", "mode": "markers", "name": f"{c} {FG.CLASS_SHORT[c]} ({len(sub)})",
                    "x": e[:, 0].round(4).tolist(), "y": e[:, 1].round(4).tolist(), "z": e[:, 2].round(4).tolist(),
                    "text": hover, "hoverinfo": "text",
                    "marker": {"size": 3.2, "color": FG.CLASS_COLOR[c], "symbol": sym, "opacity": 0.85,
                               "line": {"width": 0}}})
    return out


def write(cfg: Dict[str, Any]) -> Path:
    D = FG.Data(cfg)
    tag, L, pool = FG.REF
    out = ev.out_dir(cfg) / "advisor_report"
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    ac = D.aircraft.set_index("slug_key")

    ph = D.frame("photo", tag)["df"].copy()
    ph["label"] = ph["aircraft_uid"].map(ac["title"]).fillna(ph["aircraft_uid"])
    ph["extra"] = ph["aircraft_uid"].map(ac["company"]).fillna("") + " · " + ph["status_group"].astype(str)
    pa = D.frame("patent", tag)["df"].copy()
    pa["label"] = pa["aircraft_uid"] + " · " + pa["aircraft_name"].astype(str)
    pa["extra"] = pa["company"].astype(str) + " · topType " + pa["topType"].astype(str)

    Xp, Xa = D.X("photo", tag, L, pool), D.X("patent", tag, L, pool)
    ep = _umap(Xp, 3, f"photo_{tag}_L{L}_{pool}", cache)
    ea = _umap(Xa, 3, f"patent_{tag}_L{L}_{pool}", cache)
    ej = _umap(np.vstack([Xp, Xa]), 3, f"joint_{tag}_L{L}_{pool}", cache)
    joint = pd.concat([ph.assign(src="photo"), pa.assign(src="patent")], ignore_index=True)
    sep = separability(D, cache, tag, L, pool)

    base = FG.src
    views = [
        ("Photos, coloured by directory class", _traces(ep, ph),
         base("photo", tag, L, pool, "main", f"{len(ph)} aircraft; 3-D UMAP n_neighbors 15, min_dist 0.1, cosine, seed 42")),
        ("Patent figures, coloured by the parent of the codebook topType", _traces(ea, pa),
         base("patent", tag, L, pool, "main", f"{len(pa)} aircraft; same model and UMAP settings")),
        ("Photos and patent figures in one fit (circle = photo, open diamond = patent)", _traces(ej, joint, "src"),
         base("both", tag, L, pool, "main", "one 3-D UMAP fit on both sources")),
    ]
    sep_html = sep.to_html(index=False, float_format=lambda v: f"{v:.2f}", border=0, classes="sep")
    divs, scripts = [], []
    for i, (title, traces, source) in enumerate(views):
        divs.append(f'<section><h2>{i + 1}. {html.escape(title)}</h2><div id="v{i}" class="plot"></div>'
                    f'<p class="src"><b>Source:</b> {html.escape(source)}.</p>'
                    f'<p class="src"><b>How to read:</b> {html.escape(READ)}</p></section>')
        layout = {"margin": {"l": 0, "r": 0, "t": 10, "b": 0},
                  "legend": {"itemsizing": "constant", "x": 0.01, "y": 0.99, "bgcolor": "rgba(255,255,255,0.8)"},
                  "scene": {a: {"title": {"text": f"UMAP {k} (no unit)"}, "showticklabels": False}
                            for a, k in (("xaxis", 1), ("yaxis", 2), ("zaxis", 3))},
                  "paper_bgcolor": "#ffffff"}
        scripts.append(f"Plotly.newPlot('v{i}', {json.dumps(traces)}, {json.dumps(layout)}, "
                       "{responsive: true, displaylogo: false});")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eVTOL embeddings in 3-D</title>
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
<h1>DINOv2 embeddings of eVTOL aircraft in three dimensions</h1>
<p class="note">Internal. Photos from the evtol.news World eVTOL Aircraft Directory; patent figures from
1639_LABELLED. Hover over a point for the aircraft. Do the classes become more distinct in 3-D? The table
measures it: the share of aircraft whose 5 nearest neighbours (never from the same maker) vote for the right
class, averaged over the classes (chance 0.20), in the full embedding and in the 3-D and 2-D maps.</p>
{sep_html}
<p class="src">UMAP maps are fitted without labels; a map can separate the classes less than the full embedding,
never reveal separation the embedding does not hold.</p>
{''.join(divs)}
<script>{PLOTLY_JS.read_text(encoding="utf-8")}</script>
<script>{''.join(scripts)}</script>
</body></html>"""
    f = out / "EVTOLNEWS_UMAP_3D.html"
    f.write_text(page, encoding="utf-8")
    sep.to_csv(out / "separability_2d_3d.csv", index=False)
    return f
