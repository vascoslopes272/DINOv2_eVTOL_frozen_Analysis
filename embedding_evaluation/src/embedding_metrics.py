"""Label-free metrics on the extracted embeddings (notebook 30_embedding_evaluation).

Reads what eVTOL-Embedding-Extraction wrote under its ``paths.pipeline_root``
(``paths.pipeline_root`` here points at the same folder):

    selection/sets/<set>.csv          notebook 20 — which figures form each set
    embeddings/<tag>/emb_*.npy        notebook 22 — one row per figure
    embeddings/<tag>/metadata.parquet (figure_uid, aircraft_uid, patent_id)

and computes, for every embedding run x figure set, the embedding part of the
supervisor report (``docs/Supervisor_Report_Taxonomy_Structure_Analysis_summary``):

    A  integrity QC           (report §2)   NaN/Inf/zero rows, duplicates, dead
                                             dims, pairwise-cosine spread
    B  structure vs random    (report §3)   PC1 ratio vs a matched random matrix,
                                             participation ratio, dims for 90%
                                             variance, Hopkins
    C  unsupervised clusters  (report §1.3/§4.2, label-free columns)
                                             k-means / ward silhouette sweep,
                                             HDBSCAN noise, bootstrap ARI — on
                                             ALL layer x pooling matrices
    D  layer x pooling summary with a rank sum over the label-free criteria

Written to ``<paths.metrics_dir>/<tag>/<set>/`` (A_–D_*.csv, plot_data.npz,
summary.json) and ``<paths.metrics_dir>/<tag>/set_comparison.csv``;
``embedding_report.py`` turns them into figures + the report. No taxonomy
labels are read here. Methods mirror ``structure_clustering.py`` / ``probes.py``
so the numbers are comparable with the supervisor report.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

Key = Tuple[int, str]

PC1_PASS_RATIO = 3.0        # real PC1 must beat the random baseline by this factor
SILHOUETTE_PASS = 0.10      # label-free cluster gate (same as structure_clustering stage 3)
STABILITY_PASS = 0.60       # bootstrap ARI gate (same as structure_clustering stage 3)
PAIRWISE_SAMPLE_SIZE = 512  # rows sampled for the pairwise-cosine statistics
DEAD_DIM_THRESH = 1e-8
RANK_HIGHER_BETTER = ["pc1_ratio_vs_random", "hopkins", "best_kmeans_silhouette", "bootstrap_ari"]
RANK_LOWER_BETTER = ["hdbscan_noise_frac"]


def _params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    e = cfg.get("metrics", {}) or {}
    return {
        "cluster_k_sweep": list(e.get("cluster_k_sweep", [2, 3, 4, 5, 6, 8, 10])),
        "hdbscan_min_cluster": int(e.get("hdbscan_min_cluster", 4)),
        "bootstrap_n": int(e.get("bootstrap_n", 30)),
        "cosine_sample_size": PAIRWISE_SAMPLE_SIZE,
        "hopkins_pca_dims": 10,
        "aircraft_scope": e.get("aircraft_scope", "own"),
        "max_plot_pairs": int(e.get("max_plot_pairs", 50000)),
        "seed": int(cfg.get("seed", 42)),
    }


def tag(key: Key) -> str:
    return f"L{key[0]}_{key[1]}"


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def metrics_root(cfg: Dict[str, Any], emb_tag: str) -> Path:
    return Path(cfg["paths"]["metrics_dir"]) / emb_tag


def output_dir(cfg: Dict[str, Any], emb_tag: str, set_name: str) -> Path:
    out = metrics_root(cfg, emb_tag) / set_name
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── inputs (the extraction repo's on-disk format) ────────────────────────────

def load_embeddings(emb_dir: Path) -> Dict[str, Any]:
    """``{"metadata", "info", "arrays": {(layer, pooling): X}}`` from one run folder."""
    emb_dir = Path(emb_dir)
    arrays = {}
    for f in sorted(emb_dir.glob("emb_layer*_*.npy")):
        layer, pooling = f.stem.removeprefix("emb_layer").split("_", 1)
        arrays[(int(layer), pooling)] = np.load(f)
    return {"metadata": pd.read_parquet(emb_dir / "metadata.parquet"),
            "info": json.loads((emb_dir / "model_info.json").read_text()),
            "arrays": arrays}


def load_sets(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    d = Path(cfg["paths"]["pipeline_root"]) / "selection" / "sets"
    return {p.stem: pd.read_csv(p, keep_default_na=False, na_values=[""])
            for p in sorted(d.glob("*.csv"))}


def subset(result: Dict[str, Any], figure_uids) -> Dict[str, Any]:
    """The rows of an embedding run that belong to a figure set (set order kept)."""
    meta = result["metadata"].reset_index(drop=True)
    pos = pd.Series(meta.index, index=meta["figure_uid"])
    missing = [u for u in figure_uids if u not in pos.index]
    if missing:
        raise KeyError(f"{len(missing)} set figures have no embedding, e.g. {missing[:3]} — "
                       "re-run the extraction repo's 22_embedding_extraction after "
                       "changing the selection.")
    idx = pos.loc[list(figure_uids)].to_numpy()
    return {"metadata": meta.iloc[idx].reset_index(drop=True), "info": result["info"],
            "arrays": {k: v[idx] for k, v in result["arrays"].items()}}


# ── metric primitives (same as probes.py) ────────────────────────────────────

def hopkins(X: np.ndarray, n_pca: int = 10, m: Optional[int] = None, seed: int = 42) -> float:
    """Hopkins clustering tendency on a PCA-reduced view (0.5=uniform, ~1=clumpy)."""
    Xr = PCA(n_components=min(n_pca, X.shape[0] - 1, X.shape[1]), random_state=seed).fit_transform(X)
    rng = np.random.default_rng(seed)
    n, d = Xr.shape
    m = m or max(5, int(0.1 * n))
    nn = NearestNeighbors(n_neighbors=2).fit(Xr)
    rand = rng.uniform(Xr.min(0), Xr.max(0), size=(m, d))
    u = NearestNeighbors(n_neighbors=1).fit(Xr).kneighbors(rand)[0].ravel()
    idx = rng.choice(n, m, replace=False)
    w = nn.kneighbors(Xr[idx])[0][:, 1]
    return float(u.sum() / (u.sum() + w.sum()))


def participation_ratio(X: np.ndarray) -> float:
    """(Σλ)² / Σλ² — the effective number of dimensions."""
    ev = PCA().fit(X).explained_variance_
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def _dims_for(X: np.ndarray, var_target: float = 0.90) -> int:
    # Full spectrum: capping the PCA (structure_clustering used 60 components)
    # makes the count saturate at cap+1 whenever 90% is not reached inside the cap.
    ev = PCA().fit(X).explained_variance_ratio_
    return int(np.searchsorted(np.cumsum(ev), var_target) + 1)


def _sample_pairs(d: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return d if len(d) <= k else d[rng.choice(len(d), k, replace=False)]


def _pairwise_cosine_sample(arr: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Upper-triangle pairwise cosine of a random row sample."""
    rng = np.random.default_rng(seed)
    k = min(sample_size, arr.shape[0])
    sub = _l2(arr[rng.choice(arr.shape[0], size=k, replace=False)].astype(np.float64))
    return (sub @ sub.T)[np.triu_indices(k, k=1)]


# ── A: integrity QC ──────────────────────────────────────────────────────────

def qc_report(result: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """Integrity + cosine statistics, one row per (layer, pooling)."""
    rows = []
    for (L, pool), arr in sorted(result["arrays"].items()):
        dim_var = arr.var(axis=0)
        sims = _pairwise_cosine_sample(arr, PAIRWISE_SAMPLE_SIZE, seed)
        rows.append({
            "layer": L, "pooling": pool, "emb_dim": int(arr.shape[1]),
            "n_figures": int(arr.shape[0]),
            "nan_count": int(np.isnan(arr).sum()), "inf_count": int(np.isinf(arr).sum()),
            "all_zero_count": int((~arr.any(axis=1)).sum()),
            "exact_duplicate_count": int(arr.shape[0] - np.unique(arr, axis=0).shape[0]),
            "near_dead_dims": int((dim_var < DEAD_DIM_THRESH).sum()),
            "dim_var_mean": float(dim_var.mean()),
            "cos_mean": float(sims.mean()), "cos_std": float(sims.std()),
            "cos_p05": float(np.percentile(sims, 5)), "cos_p50": float(np.percentile(sims, 50)),
            "cos_p95": float(np.percentile(sims, 95)),
        })
    return pd.DataFrame(rows)


def qc_pass_fail(report: pd.DataFrame) -> bool:
    """Hard gate: every matrix has 0 NaN, 0 Inf, 0 all-zero rows."""
    return bool((report[["nan_count", "inf_count", "all_zero_count"]] == 0).all().all())


def section_a_qc(result: Dict[str, Any], cfg: Dict[str, Any], out: Path,
                 plot: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, bool]:
    """QC table + the hard gate. Stores each matrix's pairwise-cosine sample in
    ``plot`` as ``cosine__<tag>``."""
    seed = _params(cfg)["seed"]
    report = qc_report(result, seed=seed)
    report.to_csv(out / "A_qc_report.csv", index=False)
    for key, arr in sorted(result["arrays"].items()):
        plot[f"cosine__{tag(key)}"] = _pairwise_cosine_sample(
            arr, PAIRWISE_SAMPLE_SIZE, seed).astype(np.float32)
    return report, qc_pass_fail(report)


# ── B: global structure vs random ────────────────────────────────────────────

def section_b_structure(arrays: Dict[Key, np.ndarray], cfg: Dict[str, Any], out_dir: Path,
                        plot: Dict[str, np.ndarray]) -> pd.DataFrame:
    """PCA spectrum and pairwise distances vs a random matrix of identical shape,
    plus effective dimensionality and Hopkins. Stores ``pca_real``/``pca_random``
    spectra and sampled ``dist_real``/``dist_random`` distances in ``plot``."""
    p = _params(cfg)
    seed = p["seed"]
    rng = np.random.default_rng(seed)
    rows = []
    for key in sorted(arrays):
        X = arrays[key]
        n_comp = min(X.shape[0] - 1, 60)
        R = _l2(rng.standard_normal(X.shape))
        ev = PCA(n_components=n_comp).fit(X).explained_variance_ratio_
        ev_r = PCA(n_components=n_comp).fit(R).explained_variance_ratio_
        ratio = float(ev[0] / ev_r[0])

        Xn = _l2(X)
        iu = np.triu_indices(X.shape[0], k=1)
        d_real = (1.0 - Xn @ Xn.T)[iu]
        d_rand = (1.0 - R @ R.T)[iu]

        t = tag(key)
        plot[f"pca_real__{t}"] = ev
        plot[f"pca_random__{t}"] = ev_r
        plot[f"dist_real__{t}"] = _sample_pairs(d_real, p["max_plot_pairs"], rng).astype(np.float32)
        plot[f"dist_random__{t}"] = _sample_pairs(d_rand, p["max_plot_pairs"], rng).astype(np.float32)

        rows.append({
            "layer": key[0], "pooling": key[1],
            "pc1_var": float(ev[0]), "pc1_var_random": float(ev_r[0]),
            "pc1_ratio_vs_random": round(ratio, 2),
            "participation_ratio": round(participation_ratio(X), 1),
            "dims_for_90pct": _dims_for(X),
            "hopkins": round(hopkins(X, seed=seed), 3),
            "cos_dist_mean_real": float(d_real.mean()),
            "cos_dist_mean_random": float(d_rand.mean()),
            "pass": ratio >= PC1_PASS_RATIO,
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "B_structure_vs_random.csv", index=False)
    return out


# ── C: label-free clustering on every matrix ─────────────────────────────────

def _cluster_one(X: np.ndarray, p: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    seed = p["seed"]
    rows: List[Dict[str, Any]] = []
    for k in p["cluster_k_sweep"]:
        if k >= len(X):
            continue
        for method in ("kmeans", "agglomerative"):
            if method == "kmeans":
                lbl = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
            else:
                lbl = AgglomerativeClustering(k, linkage="ward").fit_predict(X)
            rows.append({"method": method, "k": k, "noise_frac": 0.0,
                         "silhouette": float(silhouette_score(X, lbl)),
                         "sizes": "/".join(map(str, np.bincount(lbl)))})

    lbl_h = HDBSCAN(min_cluster_size=p["hdbscan_min_cluster"]).fit_predict(X)
    mask = lbl_h != -1
    n_h = int(len(set(lbl_h[mask])))
    rows.append({"method": "hdbscan", "k": n_h, "noise_frac": float((~mask).mean()),
                 "silhouette": float(silhouette_score(X[mask], lbl_h[mask])) if n_h >= 2 else np.nan,
                 "sizes": "/".join(map(str, np.bincount(lbl_h[mask]))) if n_h else ""})

    best = max((r for r in rows if r["method"] == "kmeans"), key=lambda r: r["silhouette"])
    ag = max((r for r in rows if r["method"] == "agglomerative"), key=lambda r: r["silhouette"])
    best_k = best["k"]
    base = KMeans(best_k, n_init=10, random_state=seed).fit_predict(X)
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(p["bootstrap_n"]):
        idx = rng.choice(len(X), size=len(X), replace=True)
        bl = KMeans(best_k, n_init=10, random_state=seed + b).fit_predict(X[idx])
        aris.append(adjusted_rand_score(base[idx], bl))
    stability = float(np.mean(aris))

    summary = {
        "best_kmeans_k": best_k, "best_kmeans_silhouette": round(best["silhouette"], 3),
        "best_agglo_k": ag["k"], "best_agglo_silhouette": round(ag["silhouette"], 3),
        "bootstrap_ari": round(stability, 3),
        "hdbscan_clusters": n_h, "hdbscan_noise_frac": round(float((~mask).mean()), 3),
        "pass": best["silhouette"] > SILHOUETTE_PASS and stability > STABILITY_PASS,
    }
    return rows, summary


def section_c_clustering(arrays: Dict[Key, np.ndarray], cfg: Dict[str, Any], out: Path,
                         plot: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(full sweep, one row per matrix). Each matrix is first reduced to its
    90%-variance PCA dims. Stores the ward linkage (``linkage``) and the cut
    height of the best k (``cut``) in ``plot``."""
    from scipy.cluster.hierarchy import linkage

    p = _params(cfg)
    sweep, summ = [], []
    for key in sorted(arrays):
        n90 = _dims_for(arrays[key])
        X = PCA(n_components=n90, random_state=p["seed"]).fit_transform(arrays[key])
        rows, s = _cluster_one(X, p)
        sweep += [{"layer": key[0], "pooling": key[1], "pca_dims": n90, **r} for r in rows]
        summ.append({"layer": key[0], "pooling": key[1], "pca_dims": n90, **s})

        Z = linkage(X, method="ward")
        k = s["best_kmeans_k"]
        plot[f"linkage__{tag(key)}"] = Z
        plot[f"cut__{tag(key)}"] = np.array((Z[-k, 2] + Z[-k + 1, 2]) / 2 if k > 1 else 0.0)
        print(f"  [clustering] {tag(key)}: PCA-{n90} k={k} sil={s['best_kmeans_silhouette']} "
              f"ARI={s['bootstrap_ari']} hdbscan noise={s['hdbscan_noise_frac']}")

    sweep_df, summ_df = pd.DataFrame(sweep), pd.DataFrame(summ)
    sweep_df.to_csv(out / "C_clustering_sweep.csv", index=False)
    summ_df.to_csv(out / "C_clustering_summary.csv", index=False)
    return sweep_df, summ_df


# ── D: layer x pooling summary ───────────────────────────────────────────────

def section_d_summary(qc: pd.DataFrame, structure: pd.DataFrame,
                      clusters: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Every label-free criterion side by side plus a rank sum. cos_mean/cos_std
    stay descriptive only: a wide spread can be signal or noise (layer 24 cls)."""
    on = ["layer", "pooling"]
    s = (qc[on + ["cos_mean", "cos_std"]]
         .merge(structure[on + ["pc1_ratio_vs_random", "participation_ratio",
                                "dims_for_90pct", "hopkins"]], on=on)
         .merge(clusters[on + ["best_kmeans_k", "best_kmeans_silhouette",
                               "best_agglo_silhouette", "bootstrap_ari",
                               "hdbscan_noise_frac"]], on=on))
    ranks = pd.concat([s[c].rank(ascending=True) for c in RANK_HIGHER_BETTER]
                      + [s[c].rank(ascending=False) for c in RANK_LOWER_BETTER], axis=1)
    s["rank_sum"] = ranks.sum(axis=1)
    s = s.sort_values("rank_sum", ascending=False).reset_index(drop=True)
    s.to_csv(out / "D_layer_pooling_summary.csv", index=False)
    return s


# ── persist + run ────────────────────────────────────────────────────────────

def save(result: Dict[str, Any], set_name: str, emb_tag: str, qc_pass: bool,
         structure: pd.DataFrame, clusters: pd.DataFrame, summary: pd.DataFrame,
         plot: Dict[str, np.ndarray], cfg: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """Write ``plot_data.npz`` and ``summary.json`` next to the CSV tables."""
    np.savez_compressed(out / "plot_data.npz", **plot)
    meta = result["metadata"]
    facts = {
        "generated": date.today().isoformat(),
        "producer": "eVTOL-Visual-Evaluation/embedding_evaluation/notebooks/30_embedding_evaluation.ipynb",
        "set": set_name,
        "embedding_tag": emb_tag,
        "embeddings_dir": str(Path(cfg["paths"]["pipeline_root"]) / "embeddings" / emb_tag),
        "model": result["info"],
        "n_figures": int(len(meta)),
        "n_aircraft": int(meta["aircraft_uid"].nunique()),
        "n_patents": int(meta["patent_id"].nunique()),
        "matrices": [tag(k) for k in sorted(result["arrays"])],
        "params": _params(cfg),
        "thresholds": {"pc1_ratio": PC1_PASS_RATIO, "silhouette": SILHOUETTE_PASS,
                       "bootstrap_ari": STABILITY_PASS},
        "rank_criteria": {"higher_better": RANK_HIGHER_BETTER,
                          "lower_better": RANK_LOWER_BETTER},
        "verdicts": {
            "A_integrity": bool(qc_pass),
            "B_structure_all_pass": bool(structure["pass"].all()),
            "C_clustering_n_pass": int(clusters["pass"].sum()),
            "D_top_matrix": f"L{summary.iloc[0]['layer']}_{summary.iloc[0]['pooling']}",
        },
    }
    (out / "summary.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")
    return facts


def evaluate_set(result: Dict[str, Any], set_name: str, emb_tag: str,
                 cfg: Dict[str, Any]) -> pd.DataFrame:
    """Sections A–D for one set of one embedding run; returns table D."""
    out = output_dir(cfg, emb_tag, set_name)
    plot: Dict[str, np.ndarray] = {}
    qc, qc_pass = section_a_qc(result, cfg, out, plot)
    structure = section_b_structure(result["arrays"], cfg, out, plot)
    _, clusters = section_c_clustering(result["arrays"], cfg, out, plot)
    summary = section_d_summary(qc, structure, clusters, out)
    save(result, set_name, emb_tag, qc_pass, structure, clusters, summary, plot, cfg, out)
    return summary


def run_all(cfg: Dict[str, Any], sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Every configured embedding tag x every set; writes set_comparison.csv per tag."""
    m = cfg.get("metrics", {}) or {}
    sets = {k: sets[k] for k in (m.get("sets") or list(sets))}
    if m.get("aircraft_scope", "own") == "common":
        common = set.intersection(*(set(df["aircraft_uid"]) for df in sets.values()))
        sets = {k: df[df["aircraft_uid"].isin(common)] for k, df in sets.items()}
        print(f"[metrics] aircraft_scope=common: {len(common)} aircraft in every set")
    emb_root = Path(cfg["paths"]["pipeline_root"]) / "embeddings"
    rows = []
    for emb_tag in m.get("embeddings", []):
        if not (emb_root / emb_tag / "metadata.parquet").exists():
            print(f"[metrics] {emb_tag}: no embeddings — skipped "
                  "(run the extraction repo's 22_embedding_extraction first)")
            continue
        full = load_embeddings(emb_root / emb_tag)
        per_tag = []
        for set_name, df in sets.items():
            print(f"[metrics] {emb_tag} / {set_name}: {len(df)} figures, "
                  f"{df['aircraft_uid'].nunique()} aircraft")
            d = evaluate_set(subset(full, df["figure_uid"]), set_name, emb_tag, cfg)
            per_tag.append(d.assign(set=set_name, n_figures=len(df),
                                    n_aircraft=df["aircraft_uid"].nunique()))
        comp = pd.concat(per_tag, ignore_index=True)
        comp.insert(0, "embedding", emb_tag)
        comp.to_csv(metrics_root(cfg, emb_tag) / "set_comparison.csv", index=False)
        rows.append(comp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
