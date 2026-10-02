"""Figures + markdown report for the embedding evaluation (notebook 30_embedding_evaluation).

Presentation only — every number comes from ``embedding_metrics.py``
(``<paths.metrics_dir>/<tag>/<set>/``) and from the extraction repo's figure
selection (``<paths.pipeline_root>/selection/``).

Report layout:
    0  what went in: selection funnel, figure sets, coverage by topType
    1  set comparison: best layer x pooling of every set, per embedding run
    A–D full detail (tables + figures) for one reference set and run
       (``report.detail_set`` / ``report.detail_embedding``), and the
       cls vs mean_patch rationale

Writes ``<paths.embedding_report_dir>/embedding_evaluation_report.md`` plus
``figs/``. The taxonomy-alignment side is not covered yet.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = {
    "qc": "A_qc_report.csv",
    "structure": "B_structure_vs_random.csv",
    "sweep": "C_clustering_sweep.csv",
    "clusters": "C_clustering_summary.csv",
    "summary_table": "D_layer_pooling_summary.csv",
}


def report_dir(cfg: Dict[str, Any]) -> Path:
    d = Path(cfg["paths"]["embedding_report_dir"])
    (d / "figs").mkdir(parents=True, exist_ok=True)
    return d


def load_metrics(cfg: Dict[str, Any], emb_tag: str, set_name: str) -> Dict[str, Any]:
    """Every table + plot series of one (embedding run, set)."""
    src = Path(cfg["paths"]["metrics_dir"]) / emb_tag / set_name
    if not (src / "summary.json").exists():
        raise FileNotFoundError(f"no summary.json in {src} — run the metrics section first.")
    m = {name: pd.read_csv(src / f) for name, f in TABLES.items()}
    m["summary"] = json.loads((src / "summary.json").read_text(encoding="utf-8"))
    with np.load(src / "plot_data.npz") as z:
        m["plot"] = {k: z[k] for k in z.files}
    m["source"] = src
    return m


# ── figures ──────────────────────────────────────────────────────────────────

def _title(t: str) -> str:
    L, pool = t[1:].split("_", 1)
    return f"L{L} {pool}"


def load_selection(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = Path(cfg["paths"]["pipeline_root"]) / "selection"
    return {"funnel": pd.read_csv(d / "funnel.csv"),
            "coverage": pd.read_csv(d / "coverage.csv", keep_default_na=False, na_values=[""]),
            "summary": json.loads((d / "selection_summary.json").read_text()),
            "source": d}


def load_comparison(cfg: Dict[str, Any]) -> pd.DataFrame:
    root = Path(cfg["paths"]["metrics_dir"])
    parts = [pd.read_csv(f) for f in sorted(root.glob("*/set_comparison.csv"))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fig_prefix(m: Dict[str, Any]) -> str:
    return f"{m['summary']['embedding_tag']}/{m['summary']['set']}"


def draw_figures(m: Dict[str, Any], cfg: Dict[str, Any]) -> List[Path]:
    """One PNG per (figure kind, matrix): cosine, pcarand, basecomp, dendrogram."""
    from scipy.cluster.hierarchy import dendrogram

    figs = report_dir(cfg) / "figs" / fig_prefix(m)
    figs.mkdir(parents=True, exist_ok=True)
    p = m["plot"]
    written: List[Path] = []

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(figs / name, dpi=150)
        plt.close(fig)
        written.append(figs / name)

    for t in m["summary"]["matrices"]:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(p[f"cosine__{t}"], bins=50, color="tab:blue", alpha=0.85)
        ax.set_title(f"{_title(t)} — pairwise cosine", fontsize=10)
        ax.set_xlabel("cosine similarity"); ax.set_ylabel("pairs")
        _save(fig, f"cosine_{t}.png")

        ev, ev_r = p[f"pca_real__{t}"], p[f"pca_random__{t}"]
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(ev, label="real"); ax.plot(ev_r, label="random", ls="--")
        ax.set_yscale("log"); ax.set_xlabel("component"); ax.set_ylabel("explained var ratio")
        ax.set_title(f"{_title(t)} — PC1 ×{ev[0] / ev_r[0]:.1f}", fontsize=10)
        ax.legend(fontsize=8)
        _save(fig, f"pcarand_{t}.png")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.hist(p[f"dist_real__{t}"], bins=40, alpha=0.7, density=True, label="real")
        ax.hist(p[f"dist_random__{t}"], bins=40, alpha=0.5, density=True, label="random")
        ax.set_xlabel("cosine distance"); ax.set_title(_title(t), fontsize=10)
        ax.legend(fontsize=8)
        _save(fig, f"basecomp_{t}.png")

        cut = float(p[f"cut__{t}"])
        fig, ax = plt.subplots(figsize=(10, 3.5))
        dendrogram(p[f"linkage__{t}"], ax=ax, no_labels=True, color_threshold=cut)
        ax.axhline(cut, ls="--", c="gray", lw=1)
        row = m["clusters"].set_index(["layer", "pooling"]).loc[
            (int(t[1:].split("_", 1)[0]), t.split("_", 1)[1])]
        ax.set_title(f"Ward dendrogram — {_title(t)} (PCA-{row.pca_dims}), "
                     f"cut at k={row.best_kmeans_k}")
        _save(fig, f"dendrogram_{t}.png")
    return written


# ── text ─────────────────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fig_grid(prefix: str, matrices: List[str], folder: str = "") -> str:
    return "\n".join(f"![{_title(t)}](figs/{folder}/{prefix}_{t}.png)" for t in matrices)


def pooling_rationale(summary: pd.DataFrame) -> List[str]:
    """Plain-language cls vs mean_patch comparison, one bullet per layer,
    written from the numbers (answers "why mean_patch and not cls")."""
    out = []
    for L, g in summary.groupby("layer"):
        g = g.set_index("pooling")
        if not {"cls", "mean_patch"} <= set(g.index):
            continue
        c, mp = g.loc["cls"], g.loc["mean_patch"]
        out.append(
            f"- **Layer {L}:** cls has cosine mean {c.cos_mean:.3f} (std {c.cos_std:.3f}), "
            f"effective dim {c.participation_ratio}, PC1 ×{c.pc1_ratio_vs_random}, "
            f"silhouette {c.best_kmeans_silhouette}, stability {c.bootstrap_ari}; "
            f"mean_patch has cosine mean {mp.cos_mean:.3f} (std {mp.cos_std:.3f}), "
            f"effective dim {mp.participation_ratio}, PC1 ×{mp.pc1_ratio_vs_random}, "
            f"silhouette {mp.best_kmeans_silhouette}, stability {mp.bootstrap_ari}."
        )
    out += [
        "",
        "How the two poolings differ. `cls` is one token that the final DINOv2 "
        "blocks train to summarise the whole image. Its geometry changes a lot "
        "with depth, and at the last block it becomes unstable (see the layer-24 "
        "cosine spread). `mean_patch` averages all patch tokens. That average is "
        "steadier across layers and puts more variance into a few dominant "
        "directions (higher PC1 ratio, lower effective dim). A line drawing is "
        "mostly empty background, so the average is probably pulled towards the "
        "background too, which would explain why its cosine values sit close to 1.",
        "",
        *_pooling_verdict(summary),
        "",
        "Whether the pooling also tracks design attributes better is a taxonomy "
        "test against the labels, which this notebook does not run yet.",
    ]
    return out


def _pooling_verdict(summary: pd.DataFrame) -> List[str]:
    wide = summary.pivot(index="layer", columns="pooling", values="rank_sum")
    if not {"cls", "mean_patch"} <= set(wide.columns):
        return []
    mp_layers = [int(L) for L in wide.index if wide.loc[L, "mean_patch"] > wide.loc[L, "cls"]]
    n = len(wide)
    if len(mp_layers) == n:
        head = f"**In this run, mean_patch ranks above cls at all {n} layers.**"
        tail = " This is the label-free reason to prefer mean_patch."
    elif not mp_layers:
        head = f"**In this run, cls ranks above mean_patch at all {n} layers.**"
        tail = " Neither pooling is a clear label-free winner."
    else:
        head = (f"**In this run, mean_patch ranks above cls at layers {mp_layers} "
                f"only ({len(mp_layers)}/{n}).**")
        tail = " Neither pooling is a clear label-free winner."
    return [head + tail + " For the per-criterion detail, compare the PC1 ratio, "
            "Hopkins, silhouette and HDBSCAN noise of the two poolings at each "
            "layer in Table D1."]


def _selection_section(sel: Dict[str, Any]) -> List[str]:
    f, cov, ss = sel["funnel"], sel["coverage"], sel["summary"]
    sets = list(ss["sets"])
    by_type = cov.groupby(cov["topType"].fillna("∅"))
    ct = pd.DataFrame({"aircraft": by_type.size()})
    for s in sets:
        ct[s] = by_type[s].apply(lambda x: int((x > 0).sum()))
    ct = ct.sort_values("aircraft", ascending=False).reset_index()
    sizes = pd.DataFrame([{"set": k, **v, "rule": json.dumps(ss["rules"]["strategies"][k])}
                          for k, v in ss["sets"].items()])
    f = f.assign(aircraft_remaining=f["aircraft_remaining"].map(
        lambda v: "" if pd.isna(v) else str(int(v))))
    skipped = [k for k in ss["rules"]["strategies"] if k not in ss["sets"]]
    return [
        "## 0. What went in: figure selection",
        "",
        "The figures come from the extraction repo's `20_figure_selection`, which "
        f"chose them from the labels only (run {ss['generated']}; source "
        f"`{sel['source']}`). The reasons behind each rule are in "
        "[SELECTION_DECISIONS.md](SELECTION_DECISIONS.md).",
        "",
        "**Table 0.1.** Selection funnel. Each figure is counted under the "
        "first gate that removed it.",
        "",
        _md_table(f),
        "",
        "**Table 0.2.** Figure sets. Each set holds at most one figure per "
        "aircraft, except `all`.",
        "",
        _md_table(sizes),
        "",
        *([f"Strategies configured but not built yet: {', '.join(skipped)}.", ""] if skipped else []),
        f"Eligible aircraft: {ss['eligible_aircraft']}, split into "
        f"{ss['split'].get('select', 0)} for choosing the strategy (`select`) and "
        f"{ss['split'].get('report', 0)} for reporting it (`report`). "
        f"Only {ss['aircraft_in_every_set']} aircraft appear in every set, so "
        "strategies should be compared pair by pair on the aircraft two sets share.",
        "",
        "**Table 0.3.** Aircraft covered by each set, by topType.",
        "",
        _md_table(ct),
        "",
    ]


def _comparison_section(comp: pd.DataFrame) -> List[str]:
    if comp.empty:
        return []
    best = (comp.sort_values("rank_sum", ascending=False)
            .groupby(["embedding", "set"]).head(1)
            .sort_values(["embedding", "set"]))
    cols = ["embedding", "set", "n_figures", "n_aircraft", "layer", "pooling",
            "pc1_ratio_vs_random", "hopkins", "best_kmeans_silhouette",
            "bootstrap_ari", "hdbscan_noise_frac"]
    return [
        "## 1. Set comparison (label-free)",
        "",
        "**Table 1.1.** For each embedding run and set, the best-ranked layer × "
        "pooling. These numbers describe the geometry of each set only. Sets "
        "differ in size and in which aircraft they cover, so they are not a "
        "verdict on which images are best. That needs the comparison against "
        "the labels on shared aircraft.",
        "",
        _md_table(best[cols]),
        "",
    ]


def write_report(m: Dict[str, Any], cfg: Dict[str, Any],
                 sel: Dict[str, Any] | None = None,
                 comp: pd.DataFrame | None = None) -> Path:
    """Write ``embedding_evaluation_report.md``: selection, set comparison,
    then sections A–D for the reference (run, set) ``m``."""
    s = m["summary"]
    info, prm, thr, v = s["model"], s["params"], s["thresholds"], s["verdicts"]
    mats = s["matrices"]
    folder = fig_prefix(m)
    qc, structure, clusters, summary = m["qc"], m["structure"], m["clusters"], m["summary_table"]
    layers = sorted({int(t[1:].split("_", 1)[0]) for t in mats})
    pools = sorted({t.split("_", 1)[1] for t in mats})

    qc_cols = ["layer", "pooling", "cos_mean", "cos_std", "cos_p05", "cos_p95",
               "exact_duplicate_count", "near_dead_dims", "nan_count", "inf_count",
               "all_zero_count"]
    st_cols = ["layer", "pooling", "pc1_ratio_vs_random", "participation_ratio",
               "dims_for_90pct", "hopkins", "cos_dist_mean_real", "cos_dist_mean_random"]
    cl_cols = ["layer", "pooling", "pca_dims", "best_kmeans_k", "best_kmeans_silhouette",
               "best_agglo_k", "best_agglo_silhouette", "bootstrap_ari",
               "hdbscan_clusters", "hdbscan_noise_frac"]
    higher = ", ".join(s["rank_criteria"]["higher_better"])
    lower = ", ".join(s["rank_criteria"]["lower_better"])

    md = [
        "# Embedding Evaluation Report",
        "",
        f"*Generated {date.today().isoformat()} by "
        "`embedding_evaluation/notebooks/30_embedding_evaluation.ipynb`. Do not edit "
        "by hand. Re-run the notebook instead.*",
        "",
        f"**Model:** `{info['model_name']}` (frozen, {info['variant']}, "
        f"{info['num_hidden_layers']} blocks, hidden dim {info['hidden_dim']})  ",
        f"**Embeddings:** `{s['embeddings_dir']}`  ",
        f"**Metrics:** `{Path(cfg['paths']['metrics_dir'])}`",
        "",
        "This report covers only the embeddings: which figures went in, "
        "integrity, structure compared with random noise, and clustering "
        "without labels. The comparison against the taxonomy labels is not "
        "part of it yet.",
        "",
        *(_selection_section(sel) if sel else []),
        *(_comparison_section(comp) if comp is not None else []),
        f"## Detail: set `{s['set']}`, run `{s['embedding_tag']}`",
        "",
        f"This set holds {s['n_figures']} figures of {s['n_aircraft']} aircraft "
        f"from {s['n_patents']} patents. Matrices: layers {layers} × pooling {pools}.",
        "",
        "### Verdicts",
        "",
        "| Section | Test | Result |",
        "|---|---|---|",
        f"| A | Integrity: zero NaN, Inf or all-zero rows | **{'PASS' if v['A_integrity'] else 'FAIL'}** |",
        f"| B | PC1 ≥ {thr['pc1_ratio']:g}× random, every matrix "
        f"| **{'PASS' if v['B_structure_all_pass'] else 'FAIL'}** |",
        f"| C | Silhouette > {thr['silhouette']} and bootstrap ARI > {thr['bootstrap_ari']} "
        f"on at least one matrix | **{'PASS' if v['C_clustering_n_pass'] else 'WEAK'}** "
        f"({v['C_clustering_n_pass']}/{len(mats)} matrices) |",
        f"| D | Top-ranked matrix on these label-free criteria | **{_title(v['D_top_matrix'])}** |",
        "",
        "### A. Integrity QC",
        "",
        "Each row checks one layer × pooling matrix. The cosine statistics come "
        f"from all pairs among {prm['cosine_sample_size']} randomly sampled "
        "figures. A value of 1.0 means two embeddings are identical.",
        "",
        "**Table A1.** Integrity and cosine-similarity statistics.",
        "",
        _md_table(qc[qc_cols], "{:.4f}"),
        "",
        "How to read the cosine statistics. If the values crowd near 1.0, the "
        "figures all look alike to the model, which leaves little room for "
        "design differences to show. A wider spread leaves more room, but the "
        "matrix is also more sensitive to noise.",
        "",
        _fig_grid("cosine", mats, folder),
        "",
        "### B. Global structure vs. random noise",
        "",
        "Each matrix is compared with a random Gaussian matrix of the same "
        "shape, L2-normalised. PC1 ratio is the variance of the real first "
        "principal component divided by the random one. Participation ratio is "
        "the effective number of dimensions. Hopkins is computed on "
        f"PCA-{prm['hopkins_pca_dims']}: 0.5 means the points are spread "
        "uniformly, and values near 1 mean they form clumps.",
        "",
        "**Table B1.** PCA structure and clustering tendency.",
        "",
        _md_table(structure[st_cols]),
        "",
        "**PCA spectrum, real vs. random:**",
        "",
        _fig_grid("pcarand", mats, folder),
        "",
        "**Pairwise cosine distance, real vs. random:**",
        "",
        _fig_grid("basecomp", mats, folder),
        "",
        f"**Verdict:** {'every matrix clears' if v['B_structure_all_pass'] else 'NOT every matrix clears'} "
        f"the {thr['pc1_ratio']:g}× threshold (PC1 ratio "
        f"{structure['pc1_ratio_vs_random'].min():.1f}–{structure['pc1_ratio_vs_random'].max():.1f}×). "
        f"Hopkins ranges {structure['hopkins'].min():.2f}–{structure['hopkins'].max():.2f}. "
        "Hopkins well above 0.5 but below about 0.8 means the points form a "
        "smooth continuum. They do not split into sharply separated islands.",
        "",
        "### C. Unsupervised clustering (label-free)",
        "",
        "The same procedure runs on every matrix. Each matrix is reduced with "
        "PCA to the number of dimensions that keeps 90% of the variance. "
        "k-means and ward-agglomerative clustering are then swept over "
        f"k = {prm['cluster_k_sweep']}. HDBSCAN is run once, with min cluster "
        f"size {prm['hdbscan_min_cluster']}. For stability, the best k-means "
        f"partition is refit on {prm['bootstrap_n']} bootstrap resamples, and "
        "the table reports the mean ARI.",
        "",
        "**Table C1.** Best partition per matrix.",
        "",
        _md_table(clusters[cl_cols]),
        "",
        "**Ward dendrograms,** each cut at the best k-means k:",
        "",
        _fig_grid("dendrogram", mats, folder),
        "",
        f"The full sweep is in `{m['source'] / TABLES['sweep']}`.",
        "",
        "### D. Layer × pooling selection",
        "",
        "**Table D1.** All label-free criteria side by side, sorted by rank sum. "
        f"Higher is better for {higher}. Lower is better for {lower}. The cosine "
        "columns are shown for context but are not ranked, because a wide "
        "spread can be either signal or noise.",
        "",
        _md_table(summary),
        "",
        "#### cls vs. mean_patch",
        "",
        *pooling_rationale(summary),
        "",
    ]
    path = report_dir(cfg) / "embedding_evaluation_report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    print("[report] wrote", path)
    return path
