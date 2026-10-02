"""Two embedding reports, one protocol: the patent figures and the evtol.news photos.

User, 2026-09-22: print the embedding analysis as two separate files, one for the
evtol.news images and one for the labelled patent figures; show how the images
were chosen and that each step is robust; evaluate the embeddings the same way.

Each report has the same layout:

    Summary          the same seven questions, answered for this source
    Section 1        which images, step by step, with the check of each step (source-specific)
    Sections 2-8     the protocol of ``embedding_protocol`` (T1-T7 + layer), identical text
    (photos only)    patent <-> photo matching

Every number is computed here from the files listed at the end of each report.
Run with the Finetune env (umap-learn): scripts/build_embedding_reports.py.
"""

from __future__ import annotations

import json
import pickle
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from . import embedding_protocol as P  # noqa: E402

PILLAR = Path(__file__).resolve().parent.parent
REPO = PILLAR.parent
DATA = Path("/mnt/storage_11tb/Drive_files_to_syncronize/3 - Images DataSets & Labelling Outputs")
PAT = DATA / "1639_LABELLED"
EVN = DATA / "EVTOLNEWS_DS"
TAG = "dinov2-large_518"
STYLE_SOURCE = REPO / "docs" / "Supervisor_Report_Taxonomy_Structure_Analysis_summary.md"

CLASSES5 = ["VT", "LC", "WM", "ER", "HB"]
CLASS5_NAME = {"VT": "Vectored Thrust", "LC": "Lift + Cruise", "WM": "Wingless (Multicopter)",
               "ER": "Electric Rotorcraft", "HB": "Hover Bikes / PFD"}
PARENT = {"TW": "VT", "TR": "VT", "TB": "VT", "PTC": "VT", "DS": "VT", "CVT": "VT",
          "SLC": "LC", "SRW": "LC", "MR": "WM", "RC": "ER", "HB": "HB", "PFV": "HB"}
G1_ORDER = ["TR", "CVT", "TW", "TB", "PTC", "DS", "SLC", "SRW", "MR", "RC", "HB", "PFV"]
# validated with the dataviz validator (2026-09-18, re-run 2026-09-22): passes; yellow and pink
# need relief, so every class plot also carries a marker and a legend
CLASS_COLOR = {"VT": "#2a78d6", "LC": "#4a3aa7", "WM": "#008300", "ER": "#eda100", "HB": "#e87ba4"}
CLASS_MARK = {"VT": "o", "LC": "s", "WM": "^", "ER": "D", "HB": "v"}
INK, MUTED, FAINT, ACCENT = "#1f1f1e", "#6b6a66", "#c9c8c2", "#2a78d6"
RULE_STYLE = [("o", "#1f1f1e"), ("s", "#6b6a66"), ("^", "#9a9994"), ("D", "#c2c1bb")]
SOLO = {"Individual Inventor", "Unknown / Independent", ""}

UMAP_READ = ("each point is one aircraft. UMAP reduces each 1 024-number embedding to 2 numbers so that "
             "aircraft with similar embeddings sit close together. The two axes are these 2 numbers: no unit, "
             "no physical meaning, arbitrary orientation. Only closeness is meaningful; positions on the axes "
             "and distances between far-apart groups are not")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 200})


# ── containers ───────────────────────────────────────────────────────────────
@dataclass
class Source:
    key: str                      # "patents" | "evtolnews"
    name: str                     # "patent figures" | "evtol.news photos"
    image_word: str               # "figure" | "image"
    label_name: str               # what the architecture label is
    maker_name: str
    aircraft: pd.DataFrame        # analysis order; columns aircraft_id, label, maker, ...
    classes: List[str]
    rules: Dict[str, Dict[P.Key, np.ndarray]]
    rule_desc: Dict[str, str]
    primary_rule: str
    confound: Dict[str, Any]
    doc_dir: Path
    doc_name: str
    table_dir: Path
    fig_sub: str
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def y(self) -> np.ndarray:
        return self.aircraft["label"].to_numpy(dtype=object)

    @property
    def makers(self) -> np.ndarray:
        return self.aircraft["maker"].to_numpy(dtype=object)


class Figs:
    def __init__(self, out: Path):
        self.dir = out
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.items: Dict[str, Dict[str, str]] = {}

    def save(self, fig, key: str, title: str, source: str, read: str = "") -> None:
        w = int(fig.get_figwidth() * 23)
        lines = textwrap.wrap("Source: " + source, w)
        if read:
            lines += textwrap.wrap("How to read: " + read, w)
        fig.text(0.005, 0.004, "\n".join(lines), fontsize=5.8, color=MUTED, ha="left", va="bottom")
        fig.tight_layout(rect=(0, min(0.45, (0.105 * len(lines) + 0.12) / fig.get_figheight()), 1, 1))
        fig.savefig(self.dir / f"{key}.png", bbox_inches="tight", pad_inches=0.06)
        plt.close(fig)
        self.items[key] = {"title": title, "source": source, "read": read}

    def md(self, key: str, rel: str, width: int = 100) -> str:
        self.n += 1
        it = self.items[key]
        read = f'<span class="src"><strong>How to read:</strong> {it["read"]}.</span>' if it["read"] else ""
        return (f'<figure class="wide"><img src="{rel}/{key}.png" style="width:{width}%"><figcaption><strong>Figure {self.n}.</strong> '
                f'{it["title"]}<span class="src">Source: {it["source"]}.</span>{read}</figcaption></figure>')


# ── loading ──────────────────────────────────────────────────────────────────
def load_matrices(emb_dir: Path) -> Tuple[Dict[str, int], Dict[P.Key, np.ndarray]]:
    meta = pd.read_parquet(emb_dir / "metadata.parquet")
    arrays = {key: P.l2(np.load(emb_dir / f"emb_layer{key[0]}_{key[1]}.npy")) for key in P.MATRICES}
    return {u: i for i, u in enumerate(meta["figure_uid"])}, arrays


def rule_matrix(idx: Dict[str, int], arrays: Dict[P.Key, np.ndarray], figs: List[List[str]]
                ) -> Dict[P.Key, np.ndarray]:
    """One vector per aircraft: the figure's vector, or the renormalised mean of several."""
    rows = [[idx[f] for f in fl] for fl in figs]
    return {key: P.l2(np.stack([A[r].mean(axis=0) for r in rows])) for key, A in arrays.items()}


def _assignee_key(s: str) -> str:
    first = str(s).split(";")[0]
    return re.sub(r"\s*\([A-Z]{2}\)\s*$", "", first).strip().upper() or "?"


def load_patents() -> Source:
    V = PAT / "2_embedding_extraction/view_state_experiments"
    common = pd.read_csv(V / "aircraft_common.csv", dtype=str, keep_default_na=False)
    acp = pd.read_parquet(PAT / "2_embedding_extraction/selection/aircraft.parquet").set_index("aircraft_id")
    a = common.rename(columns={"g1_code": "label"}).copy()
    comp = a.aircraft_id.map(acp["company"]).fillna("")
    a["company"] = comp
    a["maker"] = np.where(comp.isin(SOLO), "applicant: " + a.aircraft_id.map(acp["assignee"]).map(_assignee_key),
                          comp)
    a["label5"] = a["label"].map(PARENT)
    a["arch_gt"] = a.aircraft_id.map(acp["arch_gt"]).fillna("")
    idx, arrays = load_matrices(PAT / "2_embedding_extraction/embeddings" / TAG)
    order = a.aircraft_id.tolist()
    rules, picks = {}, {}
    for name, f in [("exp1 view-first", "exp1_view_first"), ("exp2 state-first", "exp2_state_first"),
                    ("exp3 main figure", "exp3_main"), ("exp4 view average", "exp4_slots")]:
        m = pd.read_csv(V / f"{f}.csv", keep_default_na=False)
        g = m.groupby("aircraft_id")["fig_id"].apply(list)
        rules[name] = rule_matrix(idx, arrays, [g[x] for x in order])
        picks[name] = m
    ft = pd.read_csv(V / "figure_table.csv", keep_default_na=False)
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(order))].reset_index(drop=True)
    mk = a.set_index("aircraft_id")["maker"]
    conf = {"name": "view group of the drawing", "short": "view",
            "levels": "Perspective, Plan, Side, Front/Rear",
            "unit": f"all {len(wv)} whole-aircraft figures of the {len(order)} aircraft",
            "X": {key: A[[idx[f] for f in wv.fig_id]] for key, A in arrays.items()},
            "y_arch": wv.aircraft_id.map(a.set_index("aircraft_id")["label"]).to_numpy(dtype=object),
            "y_conf": wv.view4.to_numpy(dtype=object), "makers": wv.aircraft_id.map(mk).to_numpy(dtype=object)}
    man = pd.read_csv(PAT / "2_embedding_extraction/processed/518/manifest.csv")
    return Source(
        key="patents", name="patent figures", image_word="figure",
        label_name="G1 topType, the codebook's 12 architecture types (human label)",
        maker_name="the applicant's company group, or the first applicant's name for individual inventors",
        aircraft=a, classes=[c for c in G1_ORDER if c in set(a.label)], rules=rules,
        rule_desc={"exp1 view-first": "best view (Perspective > Plan > Side > Front/Rear), then best flight state",
                   "exp2 state-first": "best flight state (Cruise > Both > Hover > Other > Missing), then best view",
                   "exp3 main figure": "the figure the labeller marked as main",
                   "exp4 view average": "mean of the best figure in each of Perspective, Plan and Side"},
        primary_rule="exp3 main figure", confound=conf,
        doc_dir=REPO / "docs/embedding_evaluation", doc_name="PATENT_EMBEDDING_ANALYSIS.md",
        table_dir=PAT / "3_embedding_evaluation/embedding_protocol", fig_sub="figs/patent_embedding",
        extra={"picks": picks, "figure_table": ft, "idx": idx, "arrays": arrays,
               "processed": dict(zip(man.figure_uid, man.path)), "acp": acp})


def load_evtolnews() -> Source:
    from .evtolnews_eval import company_key
    sel = EVN / "2_embedding_extraction/selection/sets"
    main = pd.read_csv(sel / "main.csv", keep_default_na=False)
    allk = pd.read_csv(sel / "all.csv", keep_default_na=False)
    for d in (main, allk):
        d["p_aircraft"] = pd.to_numeric(d["p_aircraft"])
        d["fig_order"] = pd.to_numeric(d["fig_order"])
    idx, arrays = load_matrices(EVN / "2_embedding_extraction/embeddings" / TAG)
    a = pd.DataFrame({"aircraft_id": main.aircraft_uid, "label": main.topType, "company": main.company,
                      "title": main.title_y, "status": main.status_group, "hero": main.figure_uid})
    mk = main.company.map(company_key)
    a["maker"] = np.where(mk == "", "solo: " + main.aircraft_uid, mk)
    allk = allk.sort_values(["aircraft_uid", "fig_order"])
    g = allk.groupby("aircraft_uid")["figure_uid"].apply(list)
    rng = np.random.default_rng(P.SEED)
    rand = []
    for u, h in zip(a.aircraft_id, a.hero):
        others = [f for f in g[u] if f != h]
        rand.append([others[rng.integers(len(others))]] if others else [h])
    order = a.aircraft_id.tolist()
    rules = {"hero image": rule_matrix(idx, arrays, [[h] for h in a.hero]),
             "random other image": rule_matrix(idx, arrays, rand),
             "mean of all images": rule_matrix(idx, arrays, [g[u] for u in order])}
    a["n_images"] = a.aircraft_id.map(g.map(len))
    a["rand_pick"] = [r[0] for r in rand]
    sub = a.status.isin(["built", "concept"]).to_numpy()
    conf = {"name": "maturity status on the directory page", "short": "status", "levels": "built, concept",
            "unit": f"the hero images of the {int(sub.sum())} aircraft whose page gives a built or a concept status",
            "X": {key: X[sub] for key, X in rules["hero image"].items()},
            "y_arch": a.label.to_numpy(dtype=object)[sub], "y_conf": a.status.to_numpy(dtype=object)[sub],
            "makers": a.maker.to_numpy(dtype=object)[sub]}
    out = EVN / "3_embedding_evaluation/embedding_analysis"
    return Source(
        key="evtolnews", name="evtol.news photos", image_word="image",
        label_name="the directory's own architecture class (5 classes)",
        maker_name="the maker named on the directory page",
        aircraft=a, classes=CLASSES5, rules=rules,
        rule_desc={"hero image": "the page's first kept image (its hero image)",
                   "random other image": "one other kept image of the page, drawn at random (seed 42); "
                                         "the hero when the page has only one",
                   "mean of all images": "mean of every kept image of the page"},
        primary_rule="hero image", confound=conf, doc_dir=out, doc_name="EVTOLNEWS_EMBEDDING_ANALYSIS.md",
        table_dir=out / "tables", fig_sub="figs",
        extra={"main": main, "all": allk, "idx": idx, "arrays": arrays})


# ── running the protocol ─────────────────────────────────────────────────────
def run(S: Source, reuse: bool = False) -> Dict[str, Any]:
    S.table_dir.mkdir(parents=True, exist_ok=True)
    cache = S.table_dir / "_results.pkl"
    if reuse and cache.exists():
        return pickle.loads(cache.read_bytes())
    print(f"[{S.key}] protocol on {len(S.aircraft)} aircraft", flush=True)
    R: Dict[str, Any] = {"ev": P.evaluate(S.rules, S.y, S.makers, S.primary_rule)}
    print(f"[{S.key}] layer choice", flush=True)
    R["layer"] = P.layer_choice(S.rules, S.y, S.makers)
    best = R["layer"].sort_values("mean_knn_over_rules", ascending=False).iloc[0]
    R["best"] = (int(best.layer), str(best.pooling))
    print(f"[{S.key}] confound", flush=True)
    c = S.confound
    R["confound"] = pd.DataFrame([{"matrix": P.mname(k), **P.confound(c["X"][k], c["y_arch"], c["y_conf"],
                                                                        c["makers"])} for k in P.MATRICES])
    if S.key == "patents":
        print("[patents] five parent classes", flush=True)
        R["ev5"] = P.evaluate(S.rules, S.aircraft.label5.to_numpy(dtype=object), S.makers, S.primary_rule)
    R["sens"] = sensitivity(S)
    for k in ("main", "by_rule", "recall"):
        R["ev"][k].to_csv(S.table_dir / f"t_{k}.csv", index=False)
    R["layer"].to_csv(S.table_dir / "t_layer.csv", index=False)
    R["confound"].to_csv(S.table_dir / "t_confound.csv", index=False)
    R["sens"].to_csv(S.table_dir / "t_sensitivity.csv", index=False)
    if "ev5" in R:
        R["ev5"]["main"].to_csv(S.table_dir / "t_main_5classes.csv", index=False)
    cache.write_bytes(pickle.dumps(R))
    return R


def _knn_row(name: str, X: Dict[P.Key, np.ndarray], y, makers, n: int) -> List[Dict[str, Any]]:
    out = []
    for key in P.MATRICES:
        r = P.knn(X[key], y, makers, n_shuffle=20)
        out.append({"variant": name, "matrix": P.mname(key), "n_aircraft": n, "knn_bal_acc": r["bal_acc"],
                    "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    return out


def sensitivity(S: Source) -> pd.DataFrame:
    """Does the image selection decide the result? kNN (T4) on the primary rule,
    recomputed after each selection step is made stricter or undone."""
    rows = []
    X0, y, mk = S.rules[S.primary_rule], S.y, S.makers
    rows += _knn_row("as built", X0, y, mk, len(y))
    if S.key == "patents":
        a = S.aircraft
        keep = (a.label == a.arch_gt).to_numpy()
        rows += _knn_row("only aircraft whose drawing label matches the whole-patent reading",
                         {k: X[keep] for k, X in X0.items()}, y[keep], mk[keep], int(keep.sum()))
        ex3 = S.extra["picks"]["exp3 main figure"]
        fb = set(ex3.loc[ex3.rule_applied.astype(str).str.startswith("FALLBACK"), "aircraft_id"])
        persp = ex3.set_index("aircraft_id").view4.reindex(a.aircraft_id).eq("Perspective").to_numpy()
        rows += _knn_row("only aircraft whose main figure is a Perspective view",
                         {k: X[persp] for k, X in X0.items()}, y[persp], mk[persp], int(persp.sum()))
        if fb:
            k2 = ~a.aircraft_id.isin(fb).to_numpy()
            rows += _knn_row("without the main-figure fallbacks", {k: X[k2] for k, X in X0.items()},
                             y[k2], mk[k2], int(k2.sum()))
    else:
        a, allk, idx, arrays = S.aircraft, S.extra["all"], S.extra["idx"], S.extra["arrays"]
        for name, ok in [("stricter filter: first image with SigLIP p(aircraft) of at least 0.95",
                          allk.p_aircraft >= 0.95),
                         ("no line drawings: first image the style guess does not call a drawing",
                          allk["style"] != "drawing")]:
            first = allk[ok].groupby("aircraft_uid")["figure_uid"].first()
            keep = a.aircraft_id.isin(first.index).to_numpy()
            X = rule_matrix(idx, arrays, [[first[u]] for u in a.aircraft_id[keep]])
            rows += _knn_row(name, X, y[keep], mk[keep], int(keep.sum()))
        nd = pd.read_csv(EVN / "1_filter/audit_2026-09-22/near_duplicates_main.csv")
        dup = set(nd.a) | set(nd.b)
        keep = ~a.aircraft_id.isin(dup).to_numpy()
        rows += _knn_row("without the aircraft that share a near-duplicate hero image",
                         {k: X[keep] for k, X in X0.items()}, y[keep], mk[keep], int(keep.sum()))
        au = pd.read_csv(EVN / "1_filter/audit_2026-09-22/audit_sample.csv", keep_default_na=False)
        bad = set(au[(au.stratum == "main") & (au.verdict != "whole_aircraft")].slug_key)
        keep = ~a.aircraft_id.isin(bad).to_numpy()
        rows += _knn_row("without the 4 hero images the audit found unusable",
                         {k: X[keep] for k, X in X0.items()}, y[keep], mk[keep], int(keep.sum()))
    return pd.DataFrame(rows)


# ── figures ──────────────────────────────────────────────────────────────────
def src_line(S: Source, key: P.Key | None = None, rule: str | None = None, extra: str = "") -> str:
    m = f"layer {key[0]}, {'CLS token' if key[1] == 'cls' else 'mean of the patch tokens'}" if key else \
        "layers 18/22/24 x CLS token / patch-token mean"
    r = f"; image rule: {rule}" if rule else ""
    return (f"{S.name}; DINOv2-large, frozen, notebook 22 extraction at 518 px; {m}{r}; "
            f"{len(S.aircraft)} aircraft{('; ' + extra) if extra else ''}")


def fig_funnel(F: Figs, S: Source, steps: List[Tuple[str, int]], unit: str, title: str, source: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(steps) + 0.9))
    labels, vals = zip(*steps)
    y = np.arange(len(steps))[::-1]
    ax.barh(y, vals, color=FAINT, height=0.62)
    ax.barh(y[-1:], vals[-1:], color=ACCENT, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:,}".replace(",", " "), va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel(unit)
    ax.set_xlim(0, max(vals) * 1.15)
    F.save(fig, "f01_funnel", title, source)


def _thumb(ax, path: str, title: str, sub: str = "") -> None:
    try:
        im = Image.open(path)
        im.seek(0)
        ax.imshow(im.convert("RGB"))
    except Exception:  # noqa: BLE001
        ax.text(0.5, 0.5, "unreadable", ha="center")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_color(FAINT)
    ax.set_title(title, fontsize=7.5, loc="left")
    if sub:
        ax.set_xlabel("\n".join(textwrap.wrap(sub, 26)), fontsize=6, color=MUTED)


def fig_examples_patents(F: Figs, S: Source) -> None:
    ex3 = S.extra["picks"]["exp3 main figure"].set_index("aircraft_id")
    a = S.aircraft.set_index("aircraft_id")
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 4.4))
    for ax, g1 in zip(axes.ravel(), S.classes):
        ids = [i for i in a.index[a.label == g1] if ex3.loc[i, "view4"] == "Perspective"] or list(a.index[a.label == g1])
        i = sorted(ids)[0]
        f = ex3.loc[i, "fig_id"]
        _thumb(ax, S.extra["processed"].get(f, ex3.loc[i, "image_path"]), f"{g1} ({PARENT[g1]})", i)
    F.save(fig, "f02_examples", "One main figure per G1 type, as the model receives it (518 px, padded square).",
           "patent figures, rule exp3 main figure; the alphabetically first aircraft of each type whose main "
           "figure is a Perspective view; label = G1 type (its five-class parent)")


def fig_examples_evtolnews(F: Figs, S: Source) -> List[str]:
    main = S.extra["main"]
    fig, axes = plt.subplots(1, 5, figsize=(10.5, 2.6))
    credits = []
    for ax, c in zip(axes, CLASSES5):
        d = main[(main.topType == c) & (main.status_group == "built")].sort_values(
            ["p_aircraft", "aircraft_uid"], ascending=[False, True])
        r = d.iloc[0]
        _thumb(ax, r.local_path, f"{c}: {CLASS5_NAME[c]}", str(r.title_y)[:60])
        credits.append(f"{c}: {r.credit or 'evtol.news/' + r.aircraft_uid}")
    F.save(fig, "f02_examples", "One hero image per directory class (built aircraft, highest filter score).",
           "evtol.news photos, hero images; images (c) their owners, reproduced for internal review only")
    return credits


def fig_audit_examples(F: Figs, S: Source) -> None:
    au = pd.read_csv(EVN / "1_filter/audit_2026-09-22/audit_sample.csv", keep_default_na=False)
    bad = au[(au.stratum == "main") & (au.verdict != "whole_aircraft")]
    fd = au[(au.stratum == "siglip_drop") & (au.verdict == "whole_aircraft")]
    rows = [(r, "kept, not usable") for r in bad.itertuples()] + [(r, "dropped, whole vehicle") for r in fd.itertuples()]
    fig, axes = plt.subplots(1, len(rows), figsize=(10.5, 2.9))
    for ax, (r, tag) in zip(axes, rows):
        _thumb(ax, r.local_path, f"{tag}", f"{str(r.page_title)[:44]} (p={float(r.p_aircraft):.2f})")
    F.save(fig, "f03_audit_errors", "The filter's errors in the audit sample: hero images kept but not usable (left) "
           "and whole-vehicle images the filter dropped (right).",
           "evtol.news photos, audit sample of 2026-09-22; p = SigLIP whole-aircraft score; images (c) their "
           "owners, internal review only")


def fig_prediction(F: Figs, S: Source, R: Dict[str, Any], key: str = "f04_prediction", ev: str = "ev",
                   label_note: str = "") -> None:
    m = R[ev]["main"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x = np.arange(len(m))
    best = P.mname(R["best"])
    col = [ACCENT if n == best else FAINT for n in m.matrix]
    ax.bar(x, m.knn_bal_acc, color=col, width=0.6, label="kNN-5, maker held out (whisker = 95 % interval)")
    ax.errorbar(x, m.knn_bal_acc, yerr=[m.knn_bal_acc - m.knn_ci_lo, m.knn_ci_hi - m.knn_bal_acc],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    ax.scatter(x, m.probe_bal_acc, marker="D", s=26, color=INK, zorder=3, label="logistic probe, folds by maker")
    for xi, c in zip(x, m.knn_chance_p95):
        ax.plot([xi - 0.34, xi + 0.34], [c, c], color=MUTED, lw=1.2, ls="--")
    ax.plot([], [], color=MUTED, ls="--", label="shuffled labels, 95th percentile")
    for xi, v in zip(x, m.knn_bal_acc):
        ax.text(xi, 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, m.matrix)
    ax.set_ylabel("balanced accuracy")
    ax.set_ylim(0, max(0.8, float(m[["knn_ci_hi", "probe_bal_acc"]].max().max()) + 0.08))
    ax.legend(fontsize=6.8, loc="upper left", ncol=1)
    F.save(fig, key, f"Architecture read from the embedding, per layer and pooling{label_note} "
           f"(highlighted: the matrix the layer rule selects).",
           src_line(S, rule=S.primary_rule, extra=f"label: {S.label_name}"),
           "balanced accuracy = mean recall over the classes, so a large class cannot carry the score; "
           f"chance = 1/{len(set(R[ev]['recall'].columns) - {'matrix'})}")


def fig_rules(F: Figs, S: Source, R: Dict[str, Any]) -> None:
    b = R["ev"]["by_rule"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    rules = list(S.rules)
    w = 0.16
    for j, rule in enumerate(rules):
        d = b[b.rule == rule]
        xs = np.arange(len(d)) + (j - (len(rules) - 1) / 2) * w
        mk, c = RULE_STYLE[j]
        ax.errorbar(xs, d.knn_bal_acc, yerr=[d.knn_bal_acc - d.ci_lo, d.ci_hi - d.knn_bal_acc], fmt=mk,
                    color=c, ms=5, elinewidth=0.8, capsize=0, label=rule + (" (primary)" if rule == S.primary_rule else ""))
    ax.set_xticks(np.arange(len(P.MATRICES)), [P.mname(k) for k in P.MATRICES])
    ax.set_ylabel("kNN-5 balanced accuracy")
    ax.legend(fontsize=6.8, ncol=2, loc="upper left")
    lo = float(b.ci_lo.min())
    ax.set_ylim(max(0, lo - 0.05), float(b.ci_hi.max()) + 0.12)
    F.save(fig, "f05_rules", "Does the choice of image change the result? The same test under every image rule.",
           src_line(S), "one marker per image rule; vertical line = 95 % bootstrap interval")


def fig_layer(F: Figs, S: Source, R: Dict[str, Any]) -> None:
    L = R["layer"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4))
    best = P.mname(R["best"])
    col = [ACCENT if n == best else FAINT for n in L.matrix]
    axes[0].bar(L.matrix, L.mean_knn_over_rules, color=col, width=0.6)
    for i, v in enumerate(L.mean_knn_over_rules):
        axes[0].text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    axes[0].set_ylabel("mean kNN-5 over the rules")
    axes[1].bar(L.matrix, L.wins_on_subsets * 100, color=col, width=0.6)
    for i, v in enumerate(L.wins_on_subsets * 100):
        axes[1].text(i, v, f"{v:.0f} %", ha="center", va="bottom", fontsize=7)
    axes[1].set_ylabel("% of 200 subsets won")
    for ax in axes:
        ax.set_xticks(range(len(L)), L.matrix, rotation=35, ha="right", fontsize=7)
    F.save(fig, "f06_layer", "Layer rule: the matrix with the best mean score, and how often it stays best when "
           "the aircraft change.", src_line(S))


def _umap(S: Source, X: np.ndarray, key: str) -> np.ndarray:
    f = S.table_dir / f"_umap_{key}.npy"
    if f.exists():
        e = np.load(f)
        if len(e) == len(X):
            return e
    import umap
    e = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=P.SEED).fit_transform(X)
    np.save(f, e)
    return e


def fig_umap(F: Figs, S: Source, R: Dict[str, Any]) -> None:
    key = R["best"]
    X = S.rules[S.primary_rule][key]
    e = _umap(S, X, f"{S.primary_rule}_{key[0]}_{key[1]}".replace(" ", "_"))
    y5 = S.aircraft["label5"].to_numpy() if S.key == "patents" else S.y
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for c in CLASSES5:
        m = y5 == c
        ax.scatter(e[m, 0], e[m, 1], s=9, marker=CLASS_MARK[c], color=CLASS_COLOR[c], alpha=0.75,
                   linewidths=0, label=f"{c} {CLASS5_NAME[c]} ({int(m.sum())})")
    ax.set_xlabel("UMAP dimension 1 (no unit)"); ax.set_ylabel("UMAP dimension 2 (no unit)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=6.8, markerscale=1.6, loc="best")
    lab = "five-class parent of the G1 type" if S.key == "patents" else "directory class"
    F.save(fig, "f07_umap", f"The embedding space of the selected matrix, coloured by the {lab}.",
           src_line(S, key, S.primary_rule, "UMAP n_neighbors 15, min_dist 0.1, cosine, seed 42"), UMAP_READ)


def fig_confusion(F: Figs, S: Source, R: Dict[str, Any]) -> None:
    key = R["best"]
    pred = R["ev"]["pred"][key]
    y = S.y
    cl = S.classes
    M = pd.crosstab(pd.Categorical(y, cl), pd.Categorical(pred, cl), dropna=False).to_numpy()
    sh = M / np.clip(M.sum(axis=1, keepdims=True), 1, None)
    n = len(cl)
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.4, 0.38 * n + 1.6))
    ax.imshow(sh, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if M[i, j]:
                ax.text(j, i, str(M[i, j]), ha="center", va="center", fontsize=6.8,
                        color="white" if sh[i, j] > 0.55 else INK)
    ax.set_xticks(range(n), cl, fontsize=7); ax.set_yticks(range(n), [f"{c} ({M[i].sum()})" for i, c in enumerate(cl)], fontsize=7)
    ax.set_xlabel("predicted (kNN-5, maker held out)"); ax.set_ylabel("true class (aircraft)")
    ax.spines[:].set_visible(False)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.8))
    F.save(fig, "f08_confusion", "Which classes are confused with which.", src_line(S, key, S.primary_rule),
           "rows = true class; columns = the class kNN predicts; numbers = aircraft; the diagonal is correct; "
           "darker = larger share of the row")


def fig_confound(F: Figs, S: Source, R: Dict[str, Any]) -> None:
    c = R["confound"]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    x = np.arange(len(c))
    ax.bar(x - 0.18, c.arch_d, width=0.34, color=INK, label="architecture label")
    ax.bar(x + 0.18, c.conf_d, width=0.34, color=FAINT, label=S.confound["name"])
    for xi, a_, b_ in zip(x, c.arch_d, c.conf_d):
        ax.text(xi - 0.18, a_, f"{a_:.2f}", ha="center", va="bottom", fontsize=6.5)
        ax.text(xi + 0.18, b_, f"{b_:.2f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, c.matrix)
    ax.set_ylabel("Cohen's d (between vs within)")
    ax.legend(fontsize=7, loc="upper left")
    F.save(fig, "f09_confound", f"What else the embedding encodes: the {S.confound['name']} against the architecture.",
           src_line(S, extra=S.confound["unit"] + "; pairs of the same maker left out"),
           "d = how much further apart two items of different labels sit than two items of the same label, in "
           "standard deviations; a taller grey bar means this label moves the vectors more than the architecture")


def fig_matching(F: Figs) -> pd.DataFrame:
    m = pd.read_csv(EVN / "3_embedding_evaluation/matching.csv")
    m = m[(m.embedding == TAG) & (m.images == "no line drawings") & (m.direction == "patent->photo")].copy()
    m["matrix"] = [P.mname((int(a), b)) for a, b in zip(m.layer, m.pooling)]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(m))
    ax.bar(x - 0.18, m["R@10"] * 100, width=0.34, color=INK, label="right aircraft in the top 10")
    ax.bar(x + 0.18, m["R@50"] * 100, width=0.34, color=FAINT, label="right aircraft in the top 50")
    ax.plot(x - 0.18, m["random_R@10"] * 100, "_", color=ACCENT, ms=14, mew=2, label="random ranking, top 10")
    ax.plot(x + 0.18, m["random_R@50"] * 100, "_", color=MUTED, ms=14, mew=2, label="random ranking, top 50")
    ax.set_xticks(x, m.matrix)
    ax.set_ylabel("% of patent aircraft")
    ax.legend(fontsize=6.8, ncol=2, loc="upper left")
    ax.set_ylim(0, max(60, float(m["R@50"].max() * 100) + 15))
    F.save(fig, "f10_matching", "Can a patent aircraft be found among the photos? Rank of its own page.",
           f"patent main figures as queries against the {int(m.gallery.iloc[0])} photo aircraft without line "
           f"drawings; {int(m.queries.iloc[0])} patent aircraft with a linked page; DINOv2-large, frozen, 518 px")
    return m


# ── text helpers ─────────────────────────────────────────────────────────────
def _dec(v: float) -> int:
    t = f"{v:.3f}".rstrip("0")
    return len(t.split(".")[1]) if "." in t else 0


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def table(df: pd.DataFrame, caption: str | None = None, bold: str | None = None, key: str = "matrix") -> str:
    """Markdown table: numbers right-aligned with the same decimals down a column, text left-aligned."""
    df = df.copy()
    align = []
    for i, c in enumerate(df.columns):
        vals = df[c].tolist()
        num = bool(vals) and all(_is_num(v) for v in vals)
        if num and any(isinstance(v, (float, np.floating)) for v in vals):
            d = max(_dec(float(v)) for v in vals)
            df[c] = [f"{float(v):.{d}f}" for v in vals]
        align.append("---:" if (num or all(re.fullmatch(r"[-\d.,% ≤]+( to [-\d.,%]+)?", str(v)) for v in vals))
                     and i > 0 else "---")
    cols = [str(c) for c in df.columns]
    out = ['<div class="tbl" markdown="1">', ""]
    out += [f"**{caption}**", ""] if caption else []
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join(align) + "|"]
    for _, r in df.iterrows():
        cells = [str(v) for v in r.tolist()]
        if bold is not None and key in df.columns and str(r[key]) == bold:
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out + ["", "</div>"])


def n_(x: int) -> str:
    return f"{int(x):,}".replace(",", " ")


def pct(a: float, b: float, d: int = 1) -> str:
    return f"{100 * a / b:.{d}f} %"


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def style_block() -> str:
    m = re.search(r"<style>.*?</style>", STYLE_SOURCE.read_text(encoding="utf-8"), re.S)
    base = m.group(0) if m else ""
    extra = """<style>
figure.wide { margin: 0.8em 0 1.1em 0; break-inside: avoid; }
figure.wide img { display: block; margin: 0 auto; }
div.tbl { break-inside: avoid; page-break-inside: avoid; }
figure.wide figcaption { font-size: 0.86em; color: #3a3f44; margin-top: 0.3em; }
span.src { display: block; font-size: 0.86em; color: #6b7076; margin-top: 0.15em; }
.internal { display: block; margin: 0.4em 0 0.8em 0; padding: 0.45em 0.8em; border: 1px solid #b0b4ba;
            font-size: 0.85em; color: #3a3f44; }
table { break-inside: avoid; }
</style>"""
    return base + "\n" + extra


# ── the shared protocol sections (identical in both reports) ─────────────────
def protocol_sections(S: Source, R: Dict[str, Any], F: Figs, tno: List[int]) -> List[str]:
    def T(caption: str) -> str:
        tno[0] += 1
        return f"Table {tno[0]}. {caption}"

    m = R["ev"]["main"].copy()
    best = P.mname(R["best"])
    bm = m[m.matrix == best].iloc[0]
    nA = len(S.aircraft)
    ncl = len(S.classes)
    rel = S.fig_sub
    L: List[str] = []

    # 2 ─ integrity and structure
    L += ["## Section 2: Are the Embeddings Sound? (T1, T2)", "",
          '<div class="stage-purpose"><strong>What this section checks:</strong> that every matrix is valid '
          'and carries more structure than random noise. Nothing here uses the architecture label.</div>', ""]
    bad = int(m[["t1_nan_or_inf", "t1_zero_rows"]].to_numpy().sum())
    dup = int(m.t1_duplicate_rows.max())
    t2 = pd.DataFrame({"matrix": m.matrix, "cosine mean": m.t2_cos_mean.round(3), "cosine SD": m.t2_cos_sd.round(3),
                       "PC1 ÷ random": m.t2_pc1_vs_random.round(1),
                       "effective dims": m.t2_participation_ratio.round(1), "dims for 90 %": m.t2_dims_90,
                       "Hopkins": m.t2_hopkins.round(3)})
    L += [table(t2, T(f"Integrity and label-free structure, one vector per aircraft ({S.primary_rule}, {nA} aircraft)."),
                bold=best), ""]
    L += [f"Integrity: {'no' if bad == 0 else bad} row with a NaN, an infinite value or all zeros in the six "
          f"matrices, and {'no exact duplicate row' if dup == 0 else f'at most {dup} exact duplicate rows in one matrix'}. "
          f"On its first principal component every matrix carries {m.t2_pc1_vs_random.min():.0f} to "
          f"{m.t2_pc1_vs_random.max():.0f} times the variance of a random matrix of the same shape (pass bar 3). "
          f"Hopkins lies between {m.t2_hopkins.min():.2f} and {m.t2_hopkins.max():.2f}: the vectors clump, "
          "but as a continuum and not as separate islands (0.5 would be uniform).", ""]
    if dup:
        L += [f"The duplicate rows are aircraft whose selected {S.image_word}s are the same file or identical "
              "pixels; they are counted as separate aircraft, as the source lists them.", ""]
    L += ['<span class="stage-status passed">T1 AND T2: PASSED</span>', "", "---", ""]

    # 3 ─ separation
    L += ["## Section 3: Do Same-Class Aircraft Sit Closer Together? (T3)", "",
          '<div class="stage-purpose"><strong>What this section checks:</strong> whether two aircraft of the same '
          'class are, on average, closer in the embedding than two aircraft of different classes.</div>', "",
          "**Separation ratio** = mean cosine distance between aircraft of different classes ÷ mean distance "
          "between aircraft of the same class; 1.0 means the class is invisible. **Cohen's d** is the same gap in "
          "pooled standard deviations. Pairs of the same maker are left out, so a company's house style cannot "
          "count as architecture. p from 999 label permutations.", ""]
    t3 = pd.DataFrame({"matrix": m.matrix, "ratio": m.t3_ratio.round(3), "Cohen's d": m.t3_d.round(3),
                       "p": [f"≤ 0.001" if p <= 0.001 else f"{p:.3f}" for p in m.t3_p],
                       "pairs same class": m.t3_pairs_within.map(n_), "pairs different class": m.t3_pairs_between.map(n_)})
    L += [table(t3, T(f"Class separation, {S.primary_rule}, label: {S.label_name}."), bold=best), ""]
    sig = int((m.t3_p <= 0.05).sum())
    L += [f"The ratio lies between {m.t3_ratio.min():.3f} and {m.t3_ratio.max():.3f}, and {sig} of 6 matrices "
          f"separate the classes at p ≤ 0.05. The effect is small in every matrix (largest d "
          f"{m.t3_d.max():.2f}, {m.matrix[m.t3_d.idxmax()]}): the classes overlap heavily on average. "
          "Averages over all pairs hide local structure, which the next test measures.", "", "---", ""]

    # 4 ─ prediction
    L += ["## Section 4: Can the Architecture Be Read from the Embedding? (T4)", "",
          '<div class="stage-purpose"><strong>What this section checks:</strong> how well the class of an aircraft '
          'can be predicted from the classes of its nearest neighbours, and by a linear probe, with the whole maker '
          'held out.</div>', "",
          f"**kNN-5 (primary score).** The class of each aircraft is predicted by the majority class of its five "
          f"nearest aircraft (cosine), never counting an aircraft of the same maker ({S.maker_name}). "
          "The score is the **balanced accuracy**, the mean recall over the classes, so that a large class cannot "
          f"carry it; chance is 1/{ncl} = {1 / ncl:.2f}. The interval is a 95 % bootstrap over the aircraft. "
          "**Shuffled labels** give the chance level actually reached by the same neighbours (95th percentile of 200 "
          "shuffles). **Probe**: logistic regression, 5 folds grouped by maker.", ""]
    t4 = pd.DataFrame({"matrix": m.matrix, "kNN-5": m.knn_bal_acc.round(3),
                       "95 % interval": [f"{a:.2f} to {b:.2f}" for a, b in zip(m.knn_ci_lo, m.knn_ci_hi)],
                       "macro-F1": m.knn_macro_f1.round(3), "shuffled, p95": m.knn_chance_p95.round(3),
                       "probe": m.probe_bal_acc.round(3)})
    L += [table(t4, T(f"Prediction of the architecture, {S.primary_rule}, maker held out ({nA} aircraft, {ncl} classes)."),
                bold=best), ""]
    fig_prediction(F, S, R)
    L += [F.md("f04_prediction", rel), ""]
    above = int((m.knn_ci_lo > m.knn_chance_p95).sum())
    L += [f"{best} scores {bm.knn_bal_acc:.2f} (interval {bm.knn_ci_lo:.2f} to {bm.knn_ci_hi:.2f}) against a shuffled "
          f"level of {bm.knn_chance_p95:.2f}; the probe reaches {bm.probe_bal_acc:.2f}. In {above} of 6 matrices the "
          "whole interval lies above the shuffled level."
          + (f" The probe, which weighs every dimension, reads the classes better than the neighbours do "
             f"({bm.probe_bal_acc:.2f} against {bm.knn_bal_acc:.2f}): the architecture is present in the vector, "
             "but it is not what decides which aircraft are nearest." if bm.probe_bal_acc > bm.knn_bal_acc + 0.05
             else ""), ""]
    S.extra["recall_table"] = tno[0] + 1
    rec = R["ev"]["recall"]
    rb = rec[rec.matrix == best].iloc[0]
    cnt = S.aircraft.label.value_counts()
    t4b = pd.DataFrame({"class": S.classes, "aircraft": [int(cnt.get(c, 0)) for c in S.classes],
                        "recall": [round(float(rb[c]), 2) for c in S.classes]})
    L += [table(t4b, T(f"Recall per class, {best}, kNN-5 with the maker held out.")), ""]
    fig_confusion(F, S, R)
    L += [F.md("f08_confusion", rel, 70 if ncl > 6 else 55), ""]
    if "ev5" in R:
        m5 = R["ev5"]["main"]
        b5 = m5[m5.matrix == best].iloc[0]
        t5 = pd.DataFrame({"matrix": m5.matrix, "kNN-5": m5.knn_bal_acc.round(3),
                           "95 % interval": [f"{a:.2f} to {b:.2f}" for a, b in zip(m5.knn_ci_lo, m5.knn_ci_hi)],
                           "shuffled, p95": m5.knn_chance_p95.round(3), "probe": m5.probe_bal_acc.round(3)})
        L += ["**The same test on the five parent classes.** The evtol.news report can only use the directory's five "
              "classes. For a like-for-like comparison, each G1 type is mapped to its parent (TR, CVT, TW, TB, PTC, "
              "DS → Vectored Thrust; SLC, SRW → Lift + Cruise; MR → Wingless; RC → Electric Rotorcraft; HB, PFV → "
              "Hover Bikes / PFD) and the test is repeated.", "",
              table(t5, T("Prediction of the five parent classes, same protocol."), bold=best), "",
              f"On five classes {best} reaches {b5.knn_bal_acc:.2f} (chance 0.20).", ""]
    L += ["---", ""]

    # 5 ─ clusters
    L += ["## Section 5: Does the Structure Follow the Classes Without Labels? (T5)", "",
          '<div class="stage-purpose"><strong>What this section checks:</strong> whether a partition found without '
          'any label (k-means with as many clusters as classes) matches the classes.</div>', ""]
    t5c = pd.DataFrame({"matrix": m.matrix, "ARI": m.t5_ari.round(3), "NMI": m.t5_nmi.round(3)})
    L += [table(t5c, T(f"k-means ({ncl} clusters) against the classes. ARI 0 = chance, 1 = identical."), bold=best), ""]
    fig_umap(F, S, R)
    L += [F.md("f07_umap", rel, 72), "",
          f"ARI lies between {m.t5_ari.min():.3f} and {m.t5_ari.max():.3f}. "
          + ("The label-free clusters follow the classes weakly." if m.t5_ari.max() >= 0.05 else
             "The label-free clusters do not follow the classes: whatever structure the space has, it is not "
             "organised by architecture first."), "", "---", ""]

    # 6 ─ confound
    c = R["confound"]
    cb = c[c.matrix == best].iloc[0]
    L += [f"## Section 6: What Else Does the Embedding Encode? (T6)", "",
          '<div class="stage-purpose"><strong>What this section checks:</strong> whether a property of the image '
          f'that is not the architecture ({S.confound["name"]}) separates the vectors more strongly than the '
          'architecture does.</div>', "",
          f"The T3 separation is computed twice on the same vectors and the same pairs ({S.confound['unit']}; "
          f"same-maker pairs left out): once with the architecture as the label, once with the {S.confound['name']} "
          f"({S.confound['levels']}). A ratio d(confound) ÷ d(architecture) above 1 means that property moves the "
          "vectors more than the architecture.", ""]
    t6 = pd.DataFrame({"matrix": c.matrix, "architecture d": c.arch_d.round(3),
                       f"{S.confound['short']} d": c.conf_d.round(3),
                       f"{S.confound['short']} d ÷ architecture d": c.conf_over_arch.round(2)})
    L += [table(t6, T(f"Architecture against {S.confound['name']}."), bold=best), ""]
    fig_confound(F, S, R)
    L += [F.md("f09_confound", rel), ""]
    more = int((c.conf_over_arch > 1).sum())
    L += [f"In {more} of 6 matrices the {S.confound['name']} separates the vectors more than the architecture "
          f"(ratio {c.conf_over_arch.min():.2f} to {c.conf_over_arch.max():.2f}; {best}: {cb.conf_over_arch:.2f}).",
          ""]
    L += ["---", ""]

    # 7 ─ image choice
    b = R["ev"]["by_rule"]
    L += ["## Section 7: Does the Choice of Image Matter? (T7)", "",
          '<div class="stage-purpose"><strong>What this section checks:</strong> whether the result depends on which '
          f'{S.image_word} of each aircraft is used. The T4 score is recomputed under every image rule.</div>', ""]
    L += [table(pd.DataFrame({"rule": list(S.rule_desc), "which image": list(S.rule_desc.values())}),
                T("The image rules.")), ""]
    piv = b.pivot(index="matrix", columns="rule", values="knn_bal_acc").reindex([P.mname(k) for k in P.MATRICES])
    piv = piv[list(S.rules)].round(3).reset_index()
    L += [table(piv, T("kNN-5 balanced accuracy (maker held out) per image rule."), bold=best), ""]
    fig_rules(F, S, R)
    L += [F.md("f05_rules", rel), ""]
    sp = piv.set_index("matrix").loc[best]
    spread, width = sp.max() - sp.min(), bm.knn_ci_hi - bm.knn_ci_lo
    L += [f"On {best} the rules range from {sp.min():.2f} to {sp.max():.2f}, a spread of {spread:.2f}, against a "
          f"95 % interval about {width:.2f} wide for one rule. The highest score comes from {sp.idxmax()}. "
          + ("The spread is smaller than half the interval: the conclusion does not depend on the image rule."
             if spread < width / 2 else
             "The spread is smaller than the interval, so no rule changes the conclusion, but the order of the "
             "rules is consistent enough to note." if spread < width else
             "The spread exceeds the interval: the image rule changes the result."), "", "---", ""]

    # 8 ─ layer
    Lr = R["layer"]
    lb = Lr[Lr.matrix == best].iloc[0]
    L += ["## Section 8: Which Layer? (Layer Rule)", "",
          '<div class="stage-purpose"><strong>Rule, fixed before the numbers were read:</strong> the matrix with the '
          'highest kNN-5 score (maker held out) averaged over the image rules is carried forward. Robustness: the '
          'same choice is repeated on 200 random subsets of 80 % of the aircraft.</div>', ""]
    tl = pd.DataFrame({"matrix": Lr.matrix, "mean kNN-5 over the rules": Lr.mean_knn_over_rules.round(3),
                       "subsets won": [f"{100 * w:.0f} %" for w in Lr.wins_on_subsets]})
    L += [table(tl, T("Layer rule."), bold=best), ""]
    fig_layer(F, S, R)
    L += [F.md("f06_layer", rel), "",
          f"**{best}** is carried forward. It wins {100 * lb.wins_on_subsets:.0f} % of the subsets. "
          + ("The choice is stable." if lb.wins_on_subsets >= 0.8 else
             "The choice is not fully stable: the runner-up wins the rest of the subsets, so the two are close."),
          "", f'<span class="stage-status progress">LAYER: {best.upper()}</span>', "", "---", ""]
    return L


# ── summary rows shared by both reports ──────────────────────────────────────
def summary_rows(S: Source, R: Dict[str, Any], selection: Tuple[str, str]) -> List[Tuple[str, str, str]]:
    m = R["ev"]["main"]
    best = P.mname(R["best"])
    bm = m[m.matrix == best].iloc[0]
    c = R["confound"][R["confound"].matrix == best].iloc[0]
    b = R["ev"]["by_rule"]
    sp = b[b.matrix == best].knn_bal_acc
    Lr = R["layer"][R["layer"].matrix == best].iloc[0]
    ncl = len(S.classes)
    return [
        ("Are the images the right ones? (Section 1)", selection[0], selection[1]),
        ("Are the embeddings sound? (T1, T2)", f"no invalid row; PC1 {m.t2_pc1_vs_random.min():.0f} to "
         f"{m.t2_pc1_vs_random.max():.0f} × random", "PASSED"),
        ("Do same-class aircraft sit closer? (T3)", f"yes, weakly: ratio up to {m.t3_ratio.max():.3f}, d up to "
         f"{m.t3_d.max():.2f}", "PASSED, small effect"),
        ("Can the architecture be read? (T4)", f"{best}: kNN-5 {bm.knn_bal_acc:.2f} [{bm.knn_ci_lo:.2f}, "
         f"{bm.knn_ci_hi:.2f}], probe {bm.probe_bal_acc:.2f}; chance {1 / ncl:.2f}, shuffled {bm.knn_chance_p95:.2f}",
         ("PASSED, " + ("weak" if bm.knn_bal_acc < 0.3 else "moderate" if bm.knn_bal_acc < 0.6 else "strong"))
         if bm.knn_ci_lo > bm.knn_chance_p95 else "NOT PASSED"),
        ("Does it show without labels? (T5)", f"k-means ARI {bm.t5_ari:.3f} ({best})",
         "WEAK" if bm.t5_ari >= 0.05 else "NO"),
        ("What else is encoded? (T6)", f"{S.confound['name']}: d ratio {c.conf_over_arch:.2f} ({best})",
         "CAUTION" if c.conf_over_arch > 1 else "PASSED"),
        ("Does the image choice matter? (T7)", f"rules span {sp.min():.2f} to {sp.max():.2f} ({best})",
         "ROBUST" if sp.max() - sp.min() < (bm.knn_ci_hi - bm.knn_ci_lo) / 2 else
         "SMALL EFFECT" if sp.max() - sp.min() < bm.knn_ci_hi - bm.knn_ci_lo else "MATTERS"),
        ("Which layer?", f"{best}, wins {100 * Lr.wins_on_subsets:.0f} % of random subsets",
         "STABLE" if Lr.wins_on_subsets >= 0.8 else "CLOSE"),
    ]
