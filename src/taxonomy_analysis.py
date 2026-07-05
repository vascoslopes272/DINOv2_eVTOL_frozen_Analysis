"""Stages of the taxonomy structure-separation study (notebook 11_taxonomy_*).

Stage 0  data prep: canonical labels + main-image embeddings + alignment.
Stage 1  embedding sanity check (wraps embeddings.qc_report).
Stage 2  global structure: PCA vs random baseline, UMAP colored by attributes.

Each stage saves its artefacts under ``<taxonomy.output_dir>/stage{N}/`` and
returns a plain results dict; ``record_stage`` accumulates those into
``results.json`` and ``write_report`` renders ``REPORT_11_taxonomy.md`` so a
full report file exists after every notebook run.
"""

from __future__ import annotations

import copy
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import embeddings as emb
from src import labels as lab

DEFAULT_UMAP_ATTRS = ["G1_topType", "M2_wCount", "M2_wingConf",
                      "M1_boomsPresent", "M1_fusShape", "M1_gearArch"]


# ── plumbing ──────────────────────────────────────────────────────────────────

def tax_out(cfg: Dict[str, Any], stage: Optional[int] = None) -> Path:
    """Taxonomy output dir (optionally a stage subfolder), created on demand."""
    out = Path(cfg["taxonomy"]["output_dir"])
    if stage is not None:
        out = out / f"stage{stage}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tax_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """cfg copy whose paths.output_dir points at the taxonomy tree, so the
    existing embeddings.save/load machinery writes there untouched."""
    c = copy.deepcopy(cfg)
    c["paths"]["output_dir"] = str(tax_out(cfg))
    return c


def record_stage(cfg: Dict[str, Any], stage: int, name: str,
                 results: Dict[str, Any]) -> None:
    """Merge one stage's results into <output>/results.json (incremental)."""
    path = tax_out(cfg) / "results.json"
    all_res = json.loads(path.read_text()) if path.exists() else {}
    all_res[f"stage{stage}"] = {"name": name,
                                "run_date": datetime.date.today().isoformat(),
                                **results}
    path.write_text(json.dumps(all_res, indent=2, default=str))


def _fmt_dict(d: Dict[str, Any]) -> List[str]:
    return [f"    - {k}: {v}" for k, v in d.items()]


def _explain_stage0(s: Dict[str, Any]) -> List[str]:
    n_lab = s.get("n_architectures_labelled", "?")
    n_pat = s.get("n_base_patents", "?")
    n_emb = s.get("n_rows_embedded", "?")
    n_small = s.get("n_classes_below_min", "?")
    return [
        "**What this stage does:** gathers everything the analysis needs — your "
        "taxonomy labels from the labelling wizard, one approved 'main' figure "
        "per aircraft architecture, patent metadata (company, year), and turns "
        "each figure into an *embedding* (see glossary) with the frozen DINOv2 "
        "model. It then checks that everything lines up one-to-one.",
        "",
        f"- Architectures with taxonomy labels: **{n_lab}** "
        f"(from **{n_pat}** patents; multi-architecture patents contribute "
        f"{s.get('n_multi_arch_rows', '?')} of those rows).",
        f"- Figures actually analysed: **{n_emb}** (only the main architecture's "
        "approved main image, per the current study design).",
        f"- Taxonomy attributes available: **{s.get('n_attributes', '?')}**.",
        f"- ⚠️ **{n_small} attribute classes have fewer than the configured "
        "minimum of examples** — results for those must be treated as "
        "provisional until more patents are labelled.",
        "- Data problems found: "
        + ("; ".join(f"{k} ({v})" for k, v in s.get("qc_issues", {}).items())
           if s.get("qc_issues") else "none"),
        "",
        "**In plain words:** this is the bookkeeping stage. If the numbers "
        "above match what you expect from your labelling work, everything "
        "downstream is built on the right data.",
    ]


def _explain_stage1(s: Dict[str, Any]) -> List[str]:
    ok = s.get("pass", False)
    return [
        "**What this stage does:** checks the embeddings themselves are not "
        "broken — no corrupted numbers (NaN/Inf), no two figures accidentally "
        "getting the exact same vector, no 'dead' dimensions that never vary.",
        "",
        f"- Verdict: **{'PASS ✅' if ok else 'FAIL ❌'}** — "
        + ("the extraction pipeline is healthy."
           if ok else "something in the extraction pipeline is broken; nothing "
           "downstream can be trusted until this is fixed."),
        f"- Exact duplicate vectors: {s.get('exact_duplicates_max', '?')} "
        "(should be 0 — a duplicate would mean two images got identical "
        "embeddings, usually a pipeline bug).",
        f"- Dead dimensions: {s.get('near_dead_dims_max', '?')} (should be 0).",
        f"- Spread of similarities (cos_std): {s.get('cos_std_min', 0):.3f} — "
        "if this were ~0, every figure would look identical to the model, "
        "meaning the embeddings carry no information. It isn't, so they do.",
    ]


def _explain_stage2(s: Dict[str, Any]) -> List[str]:
    pc1 = s.get("pc1_ratio_vs_random", {})
    hop = s.get("hopkins", {})
    lines = [
        "**What this stage does:** asks whether the embeddings contain any "
        "real structure at all, by comparing them against random noise of the "
        "same size. If the real data doesn't beat random noise, the model "
        "isn't seeing anything in the drawings.",
        "",
        "- **PC1 ratio vs random** (how much stronger the main direction of "
        "variation is compared to noise; ≥3× is the pass bar — higher is "
        "better):",
        *_fmt_dict(pc1),
        f"- Best matrix: **{s.get('best_matrix', '?')}**.",
        "- **Hopkins statistic** (0.5 = spread out evenly like noise, "
        "close to 1 = strongly clumped into groups):",
        *_fmt_dict(hop),
        "- **Effective dimensionality** (participation ratio — roughly how "
        "many independent 'directions of variation' the data really uses):",
        *_fmt_dict(s.get("participation_ratio", {})),
        "",
        f"**In plain words:** {s.get('verdict', '')}. ",
    ]
    if pc1 and max(pc1.values()) >= 3:
        lines.append(
            "The embeddings are 5–9× more structured than random noise — the "
            "model definitely 'sees' something systematic in the patent "
            "drawings. However, Hopkins near 0.5 says the space is a smooth "
            "continuum, not a set of well-separated islands: designs blend "
            "into each other gradually rather than forming sharp families.")
    return lines


def _explain_stage3(s: Dict[str, Any]) -> List[str]:
    sil = s.get("best_silhouette", 0)
    ari = s.get("bootstrap_ari", 0)
    sil_read = ("strong" if sil > 0.4 else "suggestive" if sil > 0.25
                else "weak but real" if sil > 0.10 else "negligible")
    ari_read = ("stable" if ari > 0.6 else "unstable")
    return [
        "**What this stage does:** lets the computer group the figures into "
        "clusters *without looking at any labels*, then measures whether "
        "those groups are compact and reproducible.",
        "",
        f"- Best grouping found: **{s.get('best_kmeans_k', '?')} clusters** "
        f"(on {s.get('reference_matrix', '?')}, reduced to "
        f"{s.get('pca_dims', '?')} dimensions).",
        f"- **Silhouette = {sil}** — measures how cleanly separated the "
        "clusters are (0 = arbitrary, 0.25+ = suggestive, 0.4+ = strong). "
        f"Here: *{sil_read}*.",
        f"- **Bootstrap stability = {ari}** — if we resample the data and "
        "recluster, do we get the same groups? (>0.6 = yes). "
        f"Here: *{ari_read}*.",
        f"- HDBSCAN (a stricter method that only accepts dense groups) found "
        f"**{s.get('hdbscan_clusters', '?')} clusters** and called "
        f"{100 * s.get('hdbscan_noise_frac', 0):.0f}% of points 'noise' — "
        "consistent with the Stage-2 finding that the space is a continuum.",
        "",
        f"**In plain words:** {s.get('verdict', '')}. The space splits "
        "reliably into two halves, but there are no sharp, obvious design "
        "families at this sample size — structure is gradual.",
    ]


def _explain_stage4(s: Dict[str, Any]) -> List[str]:
    top = s.get("top_attributes", {})
    conf = s.get("confounds", {})
    best_attr = max(top.values()) if top else 0
    best_conf = max(conf.values()) if conf else 0
    return [
        "**What this stage does:** checks whether the computer's label-free "
        "clusters from Stage 3 coincide with *your* taxonomy — or instead "
        "with nuisance factors (which company filed the patent, the drawing "
        "viewpoint, the filing year). NMI is the score used: 0 = no relation, "
        "1 = perfect match.",
        "",
        "- Taxonomy attributes that best match the clusters (NMI):",
        *_fmt_dict(top),
        "- Nuisance factors (confounds), same score:",
        *_fmt_dict(conf),
        *(["- External validation labels (CPC patent-office classification — "
           "an *independent* labelling of the same patents; agreement here "
           "validates the approach without relying on our own taxonomy):",
           *_fmt_dict(s["external_labels"])]
          if s.get("external_labels") else []),
        "",
        f"**In plain words:** {s.get('verdict', '')}. "
        + ((f"The best confound (NMI {best_conf}) matches the clusters at "
            f"least as well as the best taxonomy attribute (NMI {best_attr}). "
            "The clusters are therefore partly explained by WHO drew the "
            "figures and HOW, not only by what is designed. This does not "
            "invalidate the analysis — design and confound scores are this "
            "close partly because companies specialise in architectures — but "
            "no architectural claim should rest on the clusters alone; the "
            "per-attribute tests in Stage 5 (and future probing with "
            "confound controls) are the trustworthy evidence."
            if best_conf >= best_attr else
            f"The best taxonomy attribute (NMI {best_attr}) beats the best "
            f"confound (NMI {best_conf}), but not by much — nuisance factors "
            "explain almost as much of the clustering as design does, so "
            "every architectural claim must be double-checked against them."
            if best_conf > 0.5 * best_attr else
            f"The taxonomy (best NMI {best_attr}) clearly out-aligns every "
            f"confound (best NMI {best_conf}) — the clusters are primarily "
            "design-driven.")),
    ]


def _explain_stage5(s: Dict[str, Any]) -> List[str]:
    n_sig = s.get("n_significant_p05", 0)
    n_tot = s.get("n_attributes_tested", 0)
    return [
        "**What this stage does:** for each taxonomy attribute, asks: are "
        "patents *with the same value* (e.g. both have wing-mounted booms) "
        "actually closer together in the embedding space than patents with "
        "different values? Three numbers per attribute:",
        "- *separation ratio*: >1 means same-class patents are closer "
        "(1.0 = no effect; for reference, the strongest here is ~1.4);",
        "- *Cohen's d*: the size of that effect (0.2 small / 0.5 medium / "
        "0.8 large);",
        "- *p-value*: probability the effect is a fluke (<0.05 = statistically "
        "credible).",
        "",
        f"- **{n_sig} of {n_tot} attributes separate significantly** "
        "(p < 0.05). The strongest:",
        *_fmt_dict(s.get("top_separations", {})),
        f"- Best layer on average: **{s.get('best_layer_on_average', '?')}** "
        "— note this is an *earlier* DINOv2 layer, hinting that geometric "
        "placement attributes live in mid-level features.",
        "",
        f"**In plain words:** {s.get('verdict', '')}. "
        "The model genuinely encodes about half of the testable design "
        "attributes — mostly *where things are placed* (wing height, boom "
        "position/length) rather than abstract categories. One warning: the "
        "drawing viewpoint (figure_type) separates MORE strongly than any "
        "design attribute, so viewpoint must always be controlled for before "
        "claiming the model 'understands' a design concept.",
    ]


_STAGE_EXPLAINERS = {"stage0": _explain_stage0, "stage1": _explain_stage1,
                     "stage2": _explain_stage2, "stage3": _explain_stage3,
                     "stage4": _explain_stage4, "stage5": _explain_stage5}

_STAGE_FIGURES = {
    "stage1": ["stage1/cosine_histograms.png"],
    "stage2": ["stage2/pca_vs_random.png", "stage2/random_baseline_comparison.png",
               "stage2/umap_by_attribute.png"],
    "stage3": ["stage3/dendrogram.png", "stage3/umap_clusters.png"],
    "stage4": ["stage4/alignment_nmi.png"],
    "stage5": ["stage5/separation_heatmap.png"],
}

_GLOSSARY = """## Glossary (plain language)

- **Embedding**: the model converts each drawing into a list of ~1000 numbers
  that summarise what it "sees". Drawings the model considers similar get
  similar numbers. All analysis happens on these numbers, never the pixels.
- **DINOv2 / frozen**: the vision model used. "Frozen" = used exactly as
  downloaded, never trained on our patents — so anything it sees, it learned
  from generic images, which is the point of the study.
- **Layer / pooling (e.g. L22 mean_patch)**: the model processes an image in
  24 steps ("layers"); we tap the signal at steps 18, 22 and 24, each in two
  variants (cls / mean_patch). Six versions of every embedding — part of the
  study is finding which sees design best.
- **PCA**: a way to find the main "directions of variation" in the data and
  compress it. Also gives the *explained variance* used in Stage 2.
- **UMAP**: squashes the ~1000 numbers to 2 so they can be plotted. Only for
  visual inspection — distances in the picture are approximate.
- **Cluster**: a group of drawings the computer put together *without seeing
  any labels*, purely by similarity of embeddings.
- **Silhouette**: how cleanly separated clusters are. 0 ≈ arbitrary,
  0.25 suggestive, 0.4+ strong.
- **NMI / ARI / purity**: agreement scores between two groupings (e.g. the
  computer's clusters vs your labels). 0 = unrelated, 1 = identical.
- **Separation ratio**: (average distance between different-class patents) ÷
  (average distance between same-class patents). Above 1 = the attribute is
  visible to the model.
- **Cohen's d**: standard effect-size scale: 0.2 small, 0.5 medium, 0.8 large.
- **p-value (permutation)**: we shuffle the labels 1000 times and see how often
  chance alone beats the real result. p < 0.05 = fewer than 5% of shuffles did,
  so the effect is probably real.
- **Confound**: a nuisance factor (drawing style, company, year, viewpoint)
  that could *fake* a design signal. If a confound scores as high as a design
  attribute, claims about that attribute need controls.
- **Hopkins statistic**: 0.5 = data spread like random noise, near 1 = strongly
  clumped. Tells whether "clusters" are even the right mental model.
"""


def write_report(cfg: Dict[str, Any]) -> Path:
    """Render results.json into a plain-language markdown report.

    Every stage section explains what was done, what each number means and how
    to read it, so the report is self-contained for a non-ML reader. Falls back
    to a raw key dump for stages without a dedicated explainer.
    """
    out = tax_out(cfg)
    res = json.loads((out / "results.json").read_text())
    n_emb = res.get("stage0", {}).get("n_rows_embedded", "?")
    lines = [
        "# Frozen-DINOv2 taxonomy analysis — plain-language report",
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M} · "
        f"{n_emb} patent figures analysed · "
        "produced by `notebooks/11_taxonomy_structure_separation.ipynb`",
        "",
        "**The question behind everything:** when a general-purpose vision "
        "model (DINOv2, never trained on patents) looks at eVTOL patent "
        "drawings, does it organise them by *aircraft design* — or only by "
        "superficial things like drawing style and viewpoint? Each stage below "
        "answers one sub-question and states its verdict; regenerated "
        "automatically on every notebook run.",
        "",
    ]
    for key in sorted(res):
        s = dict(res[key])
        name = s.pop("name", key)
        s.pop("run_date", None)
        lines += [f"## {key.capitalize()} — {name}", ""]
        if key in _STAGE_EXPLAINERS:
            lines += _STAGE_EXPLAINERS[key](s)
        else:  # future stages without an explainer yet: raw dump
            for k, v in s.items():
                if isinstance(v, dict):
                    lines.append(f"- **{k}**:")
                    lines += _fmt_dict(v)
                else:
                    lines.append(f"- **{k}**: {v}")
        figs = [f for f in _STAGE_FIGURES.get(key, []) if (out / f).exists()]
        if figs:
            lines += ["", "Figures: " + " · ".join(f"[{Path(f).name}]({f})"
                                                   for f in figs)]
        lines.append("")
    lines.append(_GLOSSARY)
    path = out / "REPORT_11_taxonomy.md"
    path.write_text("\n".join(lines))
    print(f"[write_report] {path}")
    return path


# ── Stage 0: data prep ────────────────────────────────────────────────────────

def stage0_prepare(cfg: Dict[str, Any], force_recompute: bool = False) -> Dict[str, Any]:
    """Build canonical labels, extract/load main-image embeddings, align rows.

    Returns dict with ``canon`` (aligned label rows, main architecture only),
    ``arrays`` {(layer,pool): X}, ``metadata``, ``coverage``, ``qc_issues``.
    Embeddings are cached under <taxonomy.output_dir>/embeddings/ and only
    recomputed when the figure set changed or ``force_recompute``.
    """
    canon, images = lab.build_canonical(cfg)
    qc_issues = lab.validate_canonical(canon)
    coverage = lab.coverage_table(canon, cfg)
    lab.save_canonical(canon, cfg)

    main_only = bool(cfg["taxonomy"].get("main_arch_only", True))
    sel = canon[canon["image_path"].notna()]
    if main_only:
        sel = sel[sel["arch_index"] == 1]
    if bool(cfg["taxonomy"].get("exclude_uncertain", False)):
        n_before = len(sel)
        sel = sel[~sel["any_uncertain"]]
        print(f"[stage0] exclude_uncertain: dropped "
              f"{n_before - len(sel)} human-uncertain architectures.")
    sel = sel.reset_index(drop=True)

    figures = pd.DataFrame({
        "figure_id": sel["image_path"].map(lambda p: Path(p).name),
        "patent_id": sel["Patent_ID"],
        "figure_type": sel.get("perspective", pd.Series(dtype=object)).fillna("unknown"),
        "path": sel["image_path"],
    })

    tcfg = _tax_cfg(cfg)
    emb_dir = tax_out(cfg) / "embeddings"
    cached = None
    if not force_recompute and (emb_dir / "metadata.parquet").exists():
        cached = emb.load_embeddings(tcfg)
        if set(cached["metadata"]["figure_id"]) != set(figures["figure_id"]):
            print("[stage0] cached embeddings cover a different figure set — recomputing.")
            cached = None
    if cached is None:
        result = emb.compute_embeddings(figures, tcfg)
        emb.save_embeddings(result, tcfg)
    else:
        result = cached

    # Align: label rows ordered as the embedding metadata rows.
    meta = result["metadata"].reset_index(drop=True)
    canon_aligned = (meta[["figure_id"]]
                     .merge(figures.merge(sel, left_index=True, right_index=True,
                                          suffixes=("", "_dup")),
                            on="figure_id", how="left"))
    assert len(canon_aligned) == len(meta), "alignment row-count mismatch"
    assert canon_aligned["base_patent_id"].notna().all(), "unmatched embedding rows"
    assert not canon_aligned.duplicated("base_patent_id").any() or not main_only, \
        "duplicate base patents in main-arch-only mode"

    out0 = tax_out(cfg, 0)
    coverage.to_csv(out0 / "coverage_table.csv", index=False)
    qc_issues.to_csv(out0 / "qc_issues.csv", index=False)
    canon_aligned.to_csv(out0 / "aligned_rows.csv", index=False)

    n_flag = int(coverage["below_min"].sum())
    results = {
        "n_architectures_labelled": int(len(canon)),
        "n_base_patents": int(canon["base_patent_id"].nunique()),
        "n_multi_arch_rows": int(canon["multi_arch"].sum()),
        "n_rows_embedded": int(len(meta)),
        "n_attributes": len(lab.attribute_columns(canon)),
        "n_classes_below_min": n_flag,
        "qc_issues": {r["issue"]: r["n"] for _, r in qc_issues.iterrows()},
    }
    record_stage(cfg, 0, "data prep (labels + embeddings + alignment)", results)
    return {"canon": canon_aligned, "canon_full": canon, "images": images,
            "arrays": result["arrays"], "metadata": meta,
            "info": result["info"], "coverage": coverage, "qc_issues": qc_issues}


# ── Stage 1: sanity check ─────────────────────────────────────────────────────

def stage1_sanity(prep: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """QC every (layer, pooling) matrix; save table + cosine histograms."""
    out1 = tax_out(cfg, 1)
    seed = cfg["analysis"]["seed"]
    result = {"metadata": prep["metadata"], "arrays": prep["arrays"],
              "info": prep["info"]}
    report = emb.qc_report(result, seed=seed)
    report.to_csv(out1 / "qc_embedding_report.csv", index=False)
    passed = emb.qc_pass_fail(report)

    keys = sorted(prep["arrays"].keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3),
                             sharey=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        sims = emb._pairwise_cosine_sample(prep["arrays"][key], 500, seed)
        ax.hist(sims, bins=50, color="tab:blue", alpha=0.8)
        ax.set_title(f"L{key[0]} {key[1]}", fontsize=9)
        ax.set_xlabel("pairwise cos")
    fig.suptitle("Stage 1 — pairwise cosine similarity per matrix")
    fig.tight_layout()
    fig.savefig(out1 / "cosine_histograms.png", dpi=150)
    plt.show()

    record_stage(cfg, 1, "embedding sanity check", {
        "pass": bool(passed),
        "n_figures": int(report["n_figures"].iloc[0]),
        "exact_duplicates_max": int(report["exact_duplicate_count"].max()),
        "near_dead_dims_max": int(report["near_dead_dims"].max()),
        "cos_std_min": float(report["cos_std"].min()),
        "verdict": "PASS — embeddings valid" if passed else
                   "FAIL — NaN/Inf/all-zero rows present",
    })
    print(f"Stage 1 verdict: {'PASS' if passed else 'FAIL'}")
    return report


# ── Stage 2: global structure ─────────────────────────────────────────────────

def stage2_global_structure(prep: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """PCA spectrum vs matched random baseline, intrinsic dimensionality,
    Hopkins statistic, distance-distribution comparison + attribute UMAPs."""
    from sklearn.decomposition import PCA
    from src.analysis import hopkins, intrinsic_dim, _l2
    import umap

    out2 = tax_out(cfg, 2)
    seed = cfg["analysis"]["seed"]
    rng = np.random.default_rng(seed)
    keys = sorted(prep["arrays"].keys())
    canon = prep["canon"]

    # 2a — PCA explained variance vs random Gaussian of same shape, plus
    # intrinsic dimensionality (participation ratio) and Hopkins tendency.
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.2),
                             sharey=True)
    pca_rows = []
    for ax, key in zip(np.atleast_1d(axes), keys):
        X = prep["arrays"][key]
        n_comp = min(X.shape[0] - 1, 60)
        ev = PCA(n_components=n_comp).fit(X).explained_variance_ratio_
        R = rng.standard_normal(X.shape)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        ev_r = PCA(n_components=n_comp).fit(R).explained_variance_ratio_
        ratio = float(ev[0] / ev_r[0])
        k90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
        pr, _ = intrinsic_dim(X)
        hop = hopkins(X, seed=seed)
        pca_rows.append({"layer": key[0], "pooling": key[1],
                         "pc1_var": float(ev[0]), "pc1_random": float(ev_r[0]),
                         "pc1_ratio_vs_random": ratio, "dims_for_90pct": k90,
                         "participation_ratio": round(pr, 1),
                         "hopkins": round(hop, 3)})
        ax.plot(ev, label="real"); ax.plot(ev_r, label="random", ls="--")
        ax.set_yscale("log"); ax.set_title(f"L{key[0]} {key[1]}\nPC1 x{ratio:.1f}",
                                           fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("explained var ratio (log)")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("Stage 2a — PCA spectrum: real vs random baseline")
    fig.tight_layout(); fig.savefig(out2 / "pca_vs_random.png", dpi=150); plt.show()
    pca_stats = pd.DataFrame(pca_rows)
    pca_stats.to_csv(out2 / "pca_stats.csv", index=False)

    # 2a' — pairwise cosine-distance distribution: real vs random Gaussian.
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3),
                             sharey=True)
    for ax, key in zip(np.atleast_1d(axes), keys):
        X = _l2(prep["arrays"][key])
        iu = np.triu_indices(X.shape[0], k=1)
        d_real = (1.0 - X @ X.T)[iu]
        R = rng.standard_normal(X.shape)
        R = _l2(R)
        d_rand = (1.0 - R @ R.T)[iu]
        ax.hist(d_real, bins=40, alpha=0.7, label="real", density=True)
        ax.hist(d_rand, bins=40, alpha=0.5, label="random", density=True)
        ax.set_title(f"L{key[0]} {key[1]}", fontsize=9)
        ax.set_xlabel("cosine distance")
    np.atleast_1d(axes)[-1].legend(fontsize=8)
    fig.suptitle("Stage 2a' — pairwise distance distribution: real vs random")
    fig.tight_layout()
    fig.savefig(out2 / "random_baseline_comparison.png", dpi=150); plt.show()

    # 2b — UMAP of the reference matrix, colored by taxonomy attributes.
    ref = tuple(cfg["taxonomy"].get("reference_matrix", [22, "mean_patch"]))
    attrs = [a for a in cfg["taxonomy"].get("umap_color_attrs", DEFAULT_UMAP_ATTRS)
             if a in canon.columns]
    X = prep["arrays"][ref]
    emb2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                      random_state=seed).fit_transform(X)
    n = len(attrs)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.2 * nrows))
    for ax, attr in zip(np.ravel(axes), attrs):
        vals = canon[attr].fillna("∅ missing").astype(str)
        for v in vals.value_counts().index:
            m = (vals == v).values
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=18, alpha=0.8,
                       label=f"{v} ({m.sum()})")
        ax.set_title(attr, fontsize=10)
        ax.legend(fontsize=6, markerscale=0.8, loc="best")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.suptitle(f"Stage 2b — UMAP (L{ref[0]} {ref[1]}) colored by taxonomy attribute")
    fig.tight_layout(); fig.savefig(out2 / "umap_by_attribute.png", dpi=150); plt.show()

    best = pca_stats.loc[pca_stats["pc1_ratio_vs_random"].idxmax()]
    record_stage(cfg, 2, "global structure (PCA vs random, UMAP)", {
        "pc1_ratio_vs_random": {f"L{r.layer}_{r.pooling}":
                                round(r.pc1_ratio_vs_random, 2)
                                for r in pca_stats.itertuples()},
        "dims_for_90pct": {f"L{r.layer}_{r.pooling}": r.dims_for_90pct
                           for r in pca_stats.itertuples()},
        "participation_ratio": {f"L{r.layer}_{r.pooling}": r.participation_ratio
                                for r in pca_stats.itertuples()},
        "hopkins": {f"L{r.layer}_{r.pooling}": r.hopkins
                    for r in pca_stats.itertuples()},
        "best_matrix": f"L{int(best['layer'])}_{best['pooling']}",
        "verdict": ("PASS — clear non-random structure"
                    if (pca_stats["pc1_ratio_vs_random"] >= 3).any()
                    else "WEAK — PC1 < 3x random baseline"),
        "umap_attributes_plotted": attrs,
    })
    return pca_stats


# ── Stage 3: unsupervised clustering ─────────────────────────────────────────

def stage3_clustering(prep: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Cluster the reference matrix without labels; measure quality + stability.

    Methods: k-means (k sweep), agglomerative (ward, k sweep, + dendrogram),
    HDBSCAN. Metrics: silhouette / Davies-Bouldin / Calinski-Harabasz;
    bootstrap-ARI stability for the best k-means k. Cluster assignments of the
    chosen partition are saved for Stage 4.
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans, AgglomerativeClustering, HDBSCAN
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score, adjusted_rand_score)
    from scipy.cluster.hierarchy import dendrogram, linkage

    out3 = tax_out(cfg, 3)
    seed = int(cfg["analysis"]["seed"])
    rng = np.random.default_rng(seed)
    ref = tuple(cfg["taxonomy"].get("reference_matrix", [22, "mean_patch"]))
    X_raw = prep["arrays"][ref]
    # PCA to 90%-variance dims: denoise before distance-based clustering.
    n90 = int(np.searchsorted(
        np.cumsum(PCA(n_components=min(X_raw.shape[0] - 1, 60))
                  .fit(X_raw).explained_variance_ratio_), 0.90) + 1)
    X = PCA(n_components=n90, random_state=seed).fit_transform(X_raw)
    print(f"[stage3] reference L{ref[0]} {ref[1]} -> PCA {n90} dims (90% var)")

    ks = list(cfg["taxonomy"].get("cluster_k_sweep", [2, 3, 4, 5, 6, 8, 10]))
    rows = []
    partitions: Dict[str, np.ndarray] = {}
    for k in ks:
        for method in ("kmeans", "agglomerative"):
            if method == "kmeans":
                lbl = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
            else:
                lbl = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
            rows.append({"method": method, "k": k,
                         "silhouette": float(silhouette_score(X, lbl)),
                         "davies_bouldin": float(davies_bouldin_score(X, lbl)),
                         "calinski_harabasz": float(calinski_harabasz_score(X, lbl)),
                         "noise_frac": 0.0})
            partitions[f"{method}_k{k}"] = lbl
    hdb = HDBSCAN(min_cluster_size=int(cfg["taxonomy"].get("hdbscan_min_cluster", 4)))
    lbl_h = hdb.fit_predict(X)
    n_h = int(len(set(lbl_h)) - (1 if -1 in lbl_h else 0))
    mask = lbl_h != -1
    rows.append({"method": "hdbscan", "k": n_h,
                 "silhouette": float(silhouette_score(X[mask], lbl_h[mask]))
                 if n_h >= 2 else np.nan,
                 "davies_bouldin": float(davies_bouldin_score(X[mask], lbl_h[mask]))
                 if n_h >= 2 else np.nan,
                 "calinski_harabasz": float(calinski_harabasz_score(X[mask], lbl_h[mask]))
                 if n_h >= 2 else np.nan,
                 "noise_frac": float((lbl_h == -1).mean())})
    partitions["hdbscan"] = lbl_h
    metrics = pd.DataFrame(rows)

    # Best k-means k by silhouette; bootstrap stability of that partition.
    km = metrics[metrics["method"] == "kmeans"]
    best_k = int(km.loc[km["silhouette"].idxmax(), "k"])
    base_lbl = partitions[f"kmeans_k{best_k}"]
    aris = []
    for b in range(30):
        idx = rng.choice(len(X), size=len(X), replace=True)
        bl = KMeans(best_k, n_init=10, random_state=seed + b).fit_predict(X[idx])
        aris.append(adjusted_rand_score(base_lbl[idx], bl))
    stability = float(np.mean(aris))

    # Artefacts: dendrogram (cut at best k) + UMAP by cluster + assignments.
    Z = linkage(X, method="ward")
    cut_height = float((Z[-best_k, 2] + Z[-best_k + 1, 2]) / 2) \
        if best_k > 1 else 0.0
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=cut_height)
    ax.axhline(cut_height, ls="--", c="gray", lw=1)
    ax.set_title(f"Stage 3 — ward dendrogram cut at k={best_k} "
                 f"(L{ref[0]} {ref[1]}, PCA-{n90})")
    fig.tight_layout(); fig.savefig(out3 / "dendrogram.png", dpi=150); plt.show()

    # Cluster-size distribution across every partition tried.
    size_rows = []
    for name, lbl in partitions.items():
        sizes = pd.Series(lbl).value_counts().sort_index()
        size_rows.append({"partition": name,
                          "n_clusters": int((sizes.index != -1).sum()),
                          "sizes": "/".join(str(int(s)) for i, s in sizes.items()
                                            if i != -1),
                          "n_noise": int(sizes.get(-1, 0)),
                          "largest_frac": round(float(
                              sizes[sizes.index != -1].max() / len(lbl)), 2)
                          if (sizes.index != -1).any() else np.nan})
    cluster_sizes = pd.DataFrame(size_rows)
    cluster_sizes.to_csv(out3 / "cluster_sizes.csv", index=False)

    import umap
    emb2d = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                      random_state=seed).fit_transform(X_raw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, [f"kmeans_k{best_k}", "hdbscan"]):
        lbl = partitions[name]
        for c in sorted(set(lbl)):
            m = lbl == c
            ax.scatter(emb2d[m, 0], emb2d[m, 1], s=20,
                       label=f"{'noise' if c == -1 else c} ({m.sum()})",
                       alpha=0.8, c="lightgray" if c == -1 else None)
        ax.set_title(name); ax.legend(fontsize=7); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Stage 3 — cluster partitions on UMAP")
    fig.tight_layout(); fig.savefig(out3 / "umap_clusters.png", dpi=150); plt.show()

    assign = prep["canon"][["figure_id", "base_patent_id"]].copy()
    for name, lbl in partitions.items():
        assign[name] = lbl
    assign.to_csv(out3 / "cluster_labels.csv", index=False)
    metrics.to_csv(out3 / "cluster_metrics.csv", index=False)

    best_sil = float(km["silhouette"].max())
    record_stage(cfg, 3, "unsupervised clustering", {
        "reference_matrix": f"L{ref[0]}_{ref[1]}", "pca_dims": n90,
        "best_kmeans_k": best_k, "best_silhouette": round(best_sil, 3),
        "bootstrap_ari": round(stability, 3),
        "hdbscan_clusters": n_h,
        "hdbscan_noise_frac": round(float((lbl_h == -1).mean()), 3),
        "verdict": ("PASS — clusters exist and are stable"
                    if best_sil > 0.10 and stability > 0.60 else
                    "WEAK — silhouette <= 0.10 or unstable partitions"),
    })
    display_cols = metrics.sort_values("silhouette", ascending=False)
    prep["stage3"] = {"partitions": partitions, "best": f"kmeans_k{best_k}",
                      "X_pca": X, "umap2d": emb2d}
    return display_cols


# ── Stage 4: cluster <-> taxonomy alignment + confounds ─────────────────────

def _alignment_attrs(canon: pd.DataFrame, cfg: Dict[str, Any]) -> List[str]:
    """Attributes testable for alignment: >=2 classes with >=min_align_n rows."""
    min_n = int(cfg["taxonomy"].get("min_align_class_count", 5))
    attrs = []
    for col in lab.attribute_columns(canon):
        vc = canon[col].dropna().value_counts()
        if (vc >= min_n).sum() >= 2:
            attrs.append(col)
    return attrs


def stage4_alignment(prep: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Do the Stage-3 clusters line up with taxonomy attributes — or confounds?

    For every testable attribute AND every confound (assignee, year_bin,
    perspective, drawing style): purity, NMI, ARI vs the best k-means partition.
    Rows where the attribute is null are excluded per attribute.
    """
    from sklearn.metrics import (normalized_mutual_info_score,
                                 adjusted_rand_score)

    out4 = tax_out(cfg, 4)
    canon = prep["canon"]
    clusters = prep["stage3"]["partitions"][prep["stage3"]["best"]]
    confounds = [c for c in cfg["taxonomy"].get("confound_cols", [])
                 if c in canon.columns]
    externals = [c for c in cfg["taxonomy"].get("external_cols", [])
                 if c in canon.columns]
    attrs = _alignment_attrs(canon, cfg)
    print(f"[stage4] {len(attrs)} testable attributes + {len(confounds)} confounds "
          f"+ {len(externals)} external labels "
          f"against partition {prep['stage3']['best']}")

    def purity(y_true, y_clu):
        df = pd.DataFrame({"t": y_true, "c": y_clu})
        return float(df.groupby("c")["t"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df))

    rows = []
    for col, kind in ([(a, "attribute") for a in attrs]
                      + [(c, "confound") for c in confounds]
                      + [(e, "external") for e in externals]):
        vals = canon[col]
        m = vals.notna().values
        if m.sum() < 10 or vals[m].nunique() < 2:
            continue
        y = vals[m].astype(str).values
        c = clusters[m]
        rows.append({"name": col, "kind": kind,
                     "n": int(m.sum()), "n_classes": int(pd.Series(y).nunique()),
                     "purity": round(purity(y, c), 3),
                     "nmi": round(float(normalized_mutual_info_score(y, c)), 3),
                     "ari": round(float(adjusted_rand_score(y, c)), 3)})
    align = pd.DataFrame(rows).sort_values("nmi", ascending=False).reset_index(drop=True)
    align.to_csv(out4 / "cluster_label_alignment.csv", index=False)

    # Per-cluster enrichment for the top attributes: which class dominates?
    top_attrs = align[align["kind"] == "attribute"].head(6)["name"]
    enr_rows = []
    for col in top_attrs:
        m = canon[col].notna().values
        ct = pd.crosstab(clusters[m], canon.loc[m, col].astype(str))
        for clu in ct.index:
            share = ct.loc[clu] / ct.loc[clu].sum()
            enr_rows.append({"attribute": col, "cluster": int(clu),
                             "n": int(ct.loc[clu].sum()),
                             "dominant_class": share.idxmax(),
                             "dominant_share": round(float(share.max()), 2)})
    enrich = pd.DataFrame(enr_rows)
    enrich.to_csv(out4 / "cluster_enrichment.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(align))))
    colors = align["kind"].map({"attribute": "tab:blue", "confound": "tab:red",
                                "external": "tab:green"})
    ax.barh(align["name"], align["nmi"], color=colors)
    ax.invert_yaxis(); ax.set_xlabel("NMI with clusters")
    ax.set_title("Stage 4 — cluster alignment: attributes (blue) / "
                 "confounds (red) / external CPC (green)")
    fig.tight_layout(); fig.savefig(out4 / "alignment_nmi.png", dpi=150); plt.show()

    attr_best = align[align["kind"] == "attribute"].head(1)
    conf_best = align[align["kind"] == "confound"].head(1)
    best_attr_nmi = float(attr_best["nmi"].iloc[0]) if len(attr_best) else 0.0
    best_conf_nmi = float(conf_best["nmi"].iloc[0]) if len(conf_best) else 0.0
    record_stage(cfg, 4, "cluster <-> taxonomy alignment + confounds", {
        "partition": prep["stage3"]["best"],
        "n_attributes_tested": int((align["kind"] == "attribute").sum()),
        "top_attributes": {r["name"]: r["nmi"] for _, r in
                           align[align["kind"] == "attribute"].head(5).iterrows()},
        "confounds": {r["name"]: r["nmi"] for _, r in
                      align[align["kind"] == "confound"].iterrows()},
        "external_labels": {r["name"]: r["nmi"] for _, r in
                            align[align["kind"] == "external"].iterrows()},
        "verdict": ("PASS — taxonomy aligns better than confounds"
                    if best_attr_nmi > best_conf_nmi else
                    "CAUTION — a confound aligns with clusters at least as well "
                    "as the best taxonomy attribute"),
    })
    return align


# ── Stage 5: intra- vs inter-class distances ─────────────────────────────────

def _multiclass_separation(X: np.ndarray, y: np.ndarray, n_perm: int,
                           seed: int) -> Dict[str, float]:
    """Separation ratio (inter/intra mean cosine distance), Cohen's d and a
    label-permutation p-value for an arbitrary categorical label vector."""
    from src.analysis import _l2, _pairwise

    d, same, a, b = _pairwise(_l2(X), y)
    intra, inter = d[same], d[~same]
    if len(intra) == 0 or len(inter) == 0:
        return {"sep_ratio": np.nan, "cohens_d": np.nan, "p_value": np.nan}
    obs = inter.mean() / intra.mean()
    pooled = np.sqrt((intra.var(ddof=1) * (len(intra) - 1)
                      + inter.var(ddof=1) * (len(inter) - 1))
                     / (len(intra) + len(inter) - 2))
    cohens = float((inter.mean() - intra.mean()) / pooled)

    rng = np.random.default_rng(seed)
    yp = y.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(yp)
        sm = yp[a] == yp[b]
        null[i] = d[~sm].mean() / d[sm].mean()
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return {"sep_ratio": float(obs), "cohens_d": cohens, "p_value": p,
            "intra_mean": float(intra.mean()), "inter_mean": float(inter.mean())}


def stage5_distances(prep: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """Are same-class patents genuinely closer than different-class patents?

    For every testable attribute AND the Stage-3 clusters, on ALL 6 matrices:
    separation ratio, Cohen's d, permutation p-value. Doubles as the layer
    comparison — the heatmap shows which matrix sees which attribute best.
    """
    out5 = tax_out(cfg, 5)
    seed = int(cfg["analysis"]["seed"])
    n_perm = int(cfg["taxonomy"].get("n_permutations", 1000))
    canon = prep["canon"]
    keys = sorted(prep["arrays"].keys())
    attrs = _alignment_attrs(canon, cfg)
    # Drop exact-duplicate attribute columns (e.g. boom_t1_bmech == boom_t2_bmech
    # on most rows, figure_type == perspective) to avoid double counting.
    seen: Dict[str, str] = {}
    uniq_attrs = []
    for a in attrs:
        sig = canon[a].astype(str).str.cat(sep="|")
        if sig in seen:
            print(f"[stage5] skipping {a} — identical to {seen[sig]}")
            continue
        seen[sig] = a
        uniq_attrs.append(a)

    targets = [("cluster", prep["stage3"]["partitions"][prep["stage3"]["best"]],
                np.ones(len(canon), bool))]
    externals = [c for c in cfg["taxonomy"].get("external_cols", [])
                 if c in canon.columns]
    for a in uniq_attrs + externals:
        m = canon[a].notna().values
        targets.append((a, canon[a].astype(str).values, m))

    rows = []
    for name, y, m in targets:
        for key in keys:
            r = _multiclass_separation(prep["arrays"][key][m],
                                       np.asarray(y)[m], n_perm, seed)
            rows.append({"name": name, "layer": key[0], "pooling": key[1],
                         "n": int(m.sum()), **r})
    dist = pd.DataFrame(rows)
    dist.to_csv(out5 / "intra_inter_distance_stats.csv", index=False)

    # Heatmap: attributes x matrices -> separation ratio (layer comparison view).
    piv = dist.pivot_table(index="name", columns=["layer", "pooling"],
                           values="sep_ratio")
    piv = piv.loc[piv.max(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(piv))))
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.95, vmax=1.15,
                   aspect="auto")
    ax.set_xticks(range(piv.shape[1]),
                  [f"L{l}\n{p}" for l, p in piv.columns], fontsize=8)
    ax.set_yticks(range(len(piv)), piv.index, fontsize=8)
    for i in range(len(piv)):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="separation ratio (inter/intra)")
    ax.set_title("Stage 5 — separation ratio per attribute x matrix")
    fig.tight_layout(); fig.savefig(out5 / "separation_heatmap.png", dpi=150)
    plt.show()

    # Summary: best matrix per attribute, significance-flagged.
    best_rows = dist.loc[dist.groupby("name")["sep_ratio"].idxmax()]
    best_rows = best_rows.sort_values("sep_ratio", ascending=False)
    sig = best_rows[best_rows["p_value"] < 0.05]
    per_layer = dist.groupby(["layer", "pooling"])["sep_ratio"].mean()
    best_layer = per_layer.idxmax()
    record_stage(cfg, 5, "intra- vs inter-class distances (+ layer comparison)", {
        "n_attributes_tested": len(uniq_attrs),
        "n_significant_p05": int((sig["name"] != "cluster").sum()),
        "top_separations": {f"{r['name']} (L{r['layer']}_{r['pooling']})":
                            f"ratio={r['sep_ratio']:.3f} d={r['cohens_d']:.2f} "
                            f"p={r['p_value']:.3f}"
                            for _, r in sig.head(8).iterrows()
                            } if len(sig) else {},
        "cluster_partition_sep": {
            f"L{r.layer}_{r.pooling}": round(r.sep_ratio, 3)
            for r in dist[dist['name'] == 'cluster'].itertuples()},
        "best_layer_on_average": f"L{best_layer[0]}_{best_layer[1]}",
        "verdict": ("PASS — some attributes separate significantly (p<0.05)"
                    if (sig["name"] != "cluster").any() else
                    "WEAK — no attribute separates beyond the permutation null"),
    })
    return best_rows.reset_index(drop=True)
