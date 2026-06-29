"""Stage-1 structure & separation analysis.

Operates on the per-figure embeddings produced by :mod:`src.embeddings`
(``load_embeddings`` -> dict with ``arrays`` keyed by ``(layer, pooling)`` and a
row-aligned ``metadata`` table). Everything here is label-aware geometry; the
notebook ``11_structure_separation.ipynb`` only imports, calls, and displays.

The question throughout: does frozen DINOv2 feature space "know" shrouded vs
open? We answer it four ways that must agree —

  * structure_report   : intrinsic dim, Hopkins, separation+probe per matrix
  * permutation_separation : the backbone significance test (with a null)
  * confound_presence  : are applicant/year skewed BY the label? (if not, harmless)
  * confound_decodability : can the embedding READ applicant/year?

A confound only biases the design result when it is BOTH present AND decodable.

Statistical discipline (N is small, dim is large -> p >> n):
  * never an in-sample probe score — always StratifiedKFold + a permutation p
  * probe on PCA-50 (the data's intrinsic dim is ~15-30), not raw 1024-D
  * Hopkins on PCA-10 (Hopkins in raw high-D is meaningless)
  * separation significance via label-shuffling, which needs no independence
    assumption on the (dependent) pairwise distances
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, roc_auc_score, silhouette_score
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    permutation_test_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Labels & helpers
# ---------------------------------------------------------------------------
def binary_labels(metadata: pd.DataFrame, positive_token: str = "SHR") -> np.ndarray:
    """Row-aligned 0/1 labels parsed from the filename token (_SHR_ / _OPN_).

    1 where ``positive_token`` (default ``"SHR"`` = shrouded) is in ``figure_id``.
    The shrouded/open label lives in the filename, not the Excel.
    """
    tok = f"_{positive_token}_"
    return np.array([1 if tok in f else 0 for f in metadata["figure_id"]], dtype=int)


def _l2(X: np.ndarray) -> np.ndarray:
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def hopkins(X: np.ndarray, n_pca: int = 10, m: Optional[int] = None, seed: int = 42) -> float:
    """Hopkins clustering tendency on a PCA-reduced view (0.5=uniform, ~1=clumpy).

    Reduced to ``n_pca`` dims first because Hopkins is meaningless in raw high-D
    (every point looks isolated). ``m`` sample points default to 10% of n.
    """
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


def intrinsic_dim(X: np.ndarray, var_target: float = 0.90) -> Tuple[float, int]:
    """Return (participation ratio, #PCs to reach ``var_target`` variance).

    Participation ratio = (Σλ)² / Σλ² — the effective number of dimensions.
    """
    pca = PCA().fit(X)
    ev = pca.explained_variance_
    pr = float((ev.sum() ** 2) / (ev ** 2).sum())
    n_var = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_target) + 1)
    return pr, n_var


# ---------------------------------------------------------------------------
# Separation (the backbone)
# ---------------------------------------------------------------------------
def _pairwise(X: np.ndarray, yb: np.ndarray):
    """Upper-triangle cosine distances + same/positive masks (X is L2-normalized)."""
    D = 1.0 - X @ X.T
    iu = np.triu_indices(len(yb), k=1)
    d = D[iu]
    a, b = iu
    same = yb[a] == yb[b]
    return d, same, a, b


def separation_ratio(X: np.ndarray, yb: np.ndarray) -> float:
    """between-class / within-class mean cosine distance. >1 => labels separate."""
    d, same, _, _ = _pairwise(_l2(X), yb)
    return float(d[~same].mean() / d[same].mean())


def permutation_separation(
    X: np.ndarray, yb: np.ndarray, n_perm: int = 10000, seed: int = 42
) -> Dict[str, Any]:
    """Deep separation test: observed ratio vs a label-shuffled null.

    Shuffling the labels (keeping the same points and 42/42 balance) realizes the
    "labels carry no geometry" null exactly, with no independence assumption on
    the dependent pairwise distances. Returns observed/null/p/z, the per-class
    within distances, the between distance, and a descriptive effect size.
    """
    X = _l2(X)
    d, same, a, b = _pairwise(X, yb)

    def ratio(y):
        sm = y[a] == y[b]
        return d[~sm].mean() / d[sm].mean()

    obs = ratio(yb)
    w_pos = float(d[same & (yb[a] == 1)].mean())
    w_neg = float(d[same & (yb[a] == 0)].mean())
    betw = float(d[~same].mean())

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    yp = yb.copy()
    for i in range(n_perm):
        rng.shuffle(yp)
        null[i] = ratio(yp)
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    z = float((obs - null.mean()) / null.std())
    effect = float((betw - 0.5 * (w_pos + w_neg)) / d.std())
    return {
        "observed": float(obs),
        "null": null,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": p,
        "z_score": z,
        "within_pos": w_pos,
        "within_neg": w_neg,
        "between": betw,
        "effect_size": effect,
        "n_perm": n_perm,
    }


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def linear_probe(
    X: np.ndarray, y: np.ndarray, n_pca: int = 50, n_perm: int = 300, seed: int = 42
) -> Dict[str, float]:
    """Cross-validated logistic probe on PCA-``n_pca``: AUC (binary) + acc + perm p."""
    Xp = PCA(n_components=min(n_pca, X.shape[0] - 1), random_state=seed).fit_transform(_l2(X))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    auc = (
        float(cross_val_score(pipe, Xp, y, cv=cv, scoring="roc_auc").mean())
        if len(np.unique(y)) == 2
        else float("nan")
    )
    acc, _, p = permutation_test_score(
        pipe, Xp, y, cv=cv, scoring="accuracy", n_permutations=n_perm, random_state=seed
    )
    return {"auc": auc, "accuracy": float(acc), "p_value": float(p)}


def regression_probe(
    X: np.ndarray, yv: np.ndarray, n_pca: int = 50, n_perm: int = 300, seed: int = 42
) -> Dict[str, float]:
    """Ridge probe for a continuous target (e.g. filing_year): CV R² + perm p.

    A negative R² means the embedding cannot predict the target (worse than the
    mean) — i.e. the information is not linearly present.
    """
    m = ~np.isnan(yv)
    Xp = PCA(n_components=min(n_pca, m.sum() - 1), random_state=seed).fit_transform(_l2(X[m]))
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    cv = KFold(5, shuffle=True, random_state=seed)
    r2, _, p = permutation_test_score(
        pipe, Xp, yv[m], cv=cv, scoring="r2", n_permutations=n_perm, random_state=seed
    )
    return {"r2": float(r2), "p_value": float(p)}


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (the clean way to report at large N)
# ---------------------------------------------------------------------------
# A CI carries three facts a floored p-value cannot: realness (does it exclude
# the chance level?), magnitude (where does it sit?), precision (how wide?).
# Chance levels: sep_ratio -> 1.0, AUC -> 0.5. If the CI excludes that, "real".
def _percentiles(boots: List[float], ci: float) -> Tuple[float, float]:
    lo = (100 - ci) / 2
    return float(np.percentile(boots, lo)), float(np.percentile(boots, 100 - lo))


def sep_ratio_ci(
    X: np.ndarray, yb: np.ndarray, n_boot: int = 2000, ci: float = 95.0, seed: int = 42
) -> Dict[str, float]:
    """Bootstrap CI for the between/within separation ratio.

    Stratified resampling (with replacement, within each class) preserves the
    class sizes so no bootstrap sample degenerates to a single class. Returns the
    point estimate plus ``[lo, hi]`` and whether the interval excludes 1.0.
    """
    X = _l2(X)
    yb = np.asarray(yb)
    pos, neg = np.where(yb == 1)[0], np.where(yb == 0)[0]
    rng = np.random.default_rng(seed)
    point = separation_ratio(X, yb)

    def _ratio_nozero(Xs, ys):
        # mask exact-zero distances: those are duplicate-point artifacts from
        # resampling, which would otherwise deflate the within-class mean.
        d, same, _, _ = _pairwise(Xs, ys)
        nz = d > 1e-12
        return d[(~same) & nz].mean() / d[same & nz].mean()

    boots: List[float] = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos, size=len(pos), replace=True),
            rng.choice(neg, size=len(neg), replace=True),
        ])
        boots.append(_ratio_nozero(X[idx], yb[idx]))
    lo, hi = _percentiles(boots, ci)
    return {"point": float(point), "lo": lo, "hi": hi, "ci": ci, "excludes_null": bool(lo > 1.0)}


def auc_ci(
    X: np.ndarray, y: np.ndarray, n_pca: int = 50, n_boot: int = 2000,
    ci: float = 95.0, seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI for the cross-validated probe AUC.

    Computes out-of-fold predicted probabilities once (so scoring is honest), then
    bootstraps the (label, score) pairs to get the AUC's sampling interval. Chance
    level is 0.5; ``excludes_null`` is True when the whole interval is above it.
    """
    Xp = PCA(n_components=min(n_pca, X.shape[0] - 1), random_state=seed).fit_transform(_l2(X))
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    scores = cross_val_predict(pipe, Xp, y, cv=cv, method="predict_proba")[:, 1]
    y = np.asarray(y)
    point = roc_auc_score(y, scores)
    rng = np.random.default_rng(seed)
    n = len(y)
    boots: List[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:  # need both classes to define AUC
            continue
        boots.append(roc_auc_score(y[idx], scores[idx]))
    lo, hi = _percentiles(boots, ci)
    return {"point": float(point), "lo": lo, "hi": hi, "ci": ci, "excludes_null": bool(lo > 0.5)}


# ---------------------------------------------------------------------------
# Per-matrix structure report
# ---------------------------------------------------------------------------
def structure_report(
    arrays: Dict[Tuple[int, str], np.ndarray],
    yb: np.ndarray,
    n_perm_sep: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """One row per (layer, pooling): intrinsic dim, Hopkins, separation, probe, silhouette."""
    rows: List[Dict[str, Any]] = []
    for (L, pool), X in sorted(arrays.items()):
        Xn = _l2(X)
        pr, n90 = intrinsic_dim(Xn)
        sep = permutation_separation(Xn, yb, n_perm=n_perm_sep, seed=seed)
        probe = linear_probe(Xn, yb, seed=seed)
        rows.append(
            {
                "layer": L,
                "pooling": pool,
                "partic_dim": round(pr, 1),
                "pcs_90": n90,
                "hopkins": round(hopkins(Xn, seed=seed), 2),
                "sep_ratio": round(sep["observed"], 3),
                "p_sep": round(sep["p_value"], 4),
                "silhouette": round(float(silhouette_score(Xn, yb, metric="cosine")), 3),
                "probe_auc": round(probe["auc"], 3),
                "probe_acc": round(probe["accuracy"], 3),
                "p_probe": round(probe["p_value"], 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Confounds
# ---------------------------------------------------------------------------
def derive_confounds(metadata: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Join Excel confounds to embedding rows and derive analysis-ready axes.

    ``labels`` is the patent_id-indexed frame from ``data.load_labels`` (raw Excel
    columns named by ``config.data.confound_cols``). Returns a row-aligned frame
    with: ``filing_year`` (int), ``is_bell`` (0/1 Bell/Textron assignee),
    ``assignee_country``, ``publication_country``. Missing joins -> NaN.
    """
    j = metadata.copy()
    j["patent_id"] = j["patent_id"].astype("string").str.strip()
    j = j.join(labels, on="patent_id")
    out = pd.DataFrame(index=metadata.index)
    if "Filing/Application Date" in j:
        out["filing_year"] = pd.to_datetime(j["Filing/Application Date"], errors="coerce").dt.year
    if "Assignee" in j:
        ass = j["Assignee"].astype("string").fillna("UNKNOWN")
        out["is_bell"] = ass.str.contains("BELL|TEXTRON", case=False, na=False).astype(int)
    if "Assignee Country" in j:
        out["assignee_country"] = j["Assignee Country"].astype("string").fillna("??")
    if "Publication Country" in j:
        out["publication_country"] = j["Publication Country"].astype("string").fillna("??")
    return out.reset_index(drop=True)


def confound_presence(yb: np.ndarray, confounds: pd.DataFrame) -> pd.DataFrame:
    """Is each confound SKEWED by the shrouded/open label? (if not, it can't bias.)

    Continuous (filing_year): Welch t-test of the confound by class.
    Binary (is_bell): chi-square of the 2x2 table. Returns a tidy verdict frame.
    """
    rows = []
    yb = np.asarray(yb)
    if "filing_year" in confounds:
        yr = confounds["filing_year"].to_numpy(dtype=float)
        m = ~np.isnan(yr)
        t, p = stats.ttest_ind(yr[m & (yb == 1)], yr[m & (yb == 0)])
        rows.append({
            "confound": "filing_year",
            "test": "t-test",
            "stat_pos": round(float(yr[m & (yb == 1)].mean()), 1),
            "stat_neg": round(float(yr[m & (yb == 0)].mean()), 1),
            "p_value": round(float(p), 3),
            "verdict": "SKEWED" if p < 0.05 else "balanced",
        })
    if "is_bell" in confounds:
        bell = confounds["is_bell"].to_numpy()
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(bell, yb))
        rows.append({
            "confound": "is_bell",
            "test": "chi2",
            "stat_pos": round(float(yb[bell == 1].mean()), 2),   # shrouded-rate among bell
            "stat_neg": round(float(yb[bell == 0].mean()), 2),   # shrouded-rate among non-bell
            "p_value": round(float(p), 3),
            "verdict": "SKEWED" if p < 0.05 else "balanced",
        })
    return pd.DataFrame(rows)


def confound_decodability(
    arrays: Dict[Tuple[int, str], np.ndarray],
    yb: np.ndarray,
    confounds: pd.DataFrame,
    matrices: Optional[List[Tuple[int, str]]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Can the embedding READ each confound? Compare to the design probe.

    For each requested matrix: design AUC (shrouded/open), is_bell AUC, and
    filing_year R². If a confound is decodable AND skewed (see confound_presence)
    it threatens the design result; decodable-but-balanced is a parallel signal,
    not a confound.
    """
    matrices = matrices or sorted(arrays.keys())
    rows = []
    for L, pool in matrices:
        X = arrays[(L, pool)]
        rec = {"layer": L, "pooling": pool}
        d = linear_probe(X, yb, seed=seed)
        rec["design_auc"] = round(d["auc"], 3)
        rec["design_p"] = round(d["p_value"], 4)
        if "is_bell" in confounds:
            b = linear_probe(X, confounds["is_bell"].to_numpy(), seed=seed)
            rec["bell_auc"] = round(b["auc"], 3)
            rec["bell_p"] = round(b["p_value"], 4)
        if "filing_year" in confounds:
            r = regression_probe(X, confounds["filing_year"].to_numpy(dtype=float), seed=seed)
            rec["year_r2"] = round(r["r2"], 3)
            rec["year_p"] = round(r["p_value"], 4)
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-class configuration taxonomy (tilt-rotor / tilt-wing / lift+cruise / ...)
# The REAL thesis target. sep_ratio / permutation_separation / hopkins /
# silhouette / intrinsic_dim already generalise to >2 classes; these add the
# multinomial probe and the class-vs-class views.
# ---------------------------------------------------------------------------
def multiclass_labels(
    metadata: pd.DataFrame, labels: pd.DataFrame, col: str
) -> pd.Series:
    """Row-aligned configuration labels from an Excel column joined by patent_id.

    ``labels`` is the patent_id-indexed frame from ``data.load_labels`` (add the
    config column to ``config.data.label_cols`` first). Returns a Series aligned
    to ``metadata`` rows, with NaN where the patent is not yet labelled. Downstream
    functions drop NaN and rare classes themselves.
    """
    j = metadata.copy()
    j["patent_id"] = j["patent_id"].astype("string").str.strip()
    s = j.join(labels, on="patent_id")[col].astype("string").str.strip()
    s = s.replace({"": pd.NA})
    return pd.Series(s.values, name=col)


def _prep_multiclass(X, y, min_per_class: int):
    """Drop unlabelled rows + classes with < min_per_class samples; encode to int."""
    y = pd.Series(y).astype("object")
    mask = y.notna().to_numpy()
    Xs, ys = _l2(np.asarray(X))[mask], y[mask].to_numpy()
    vc = pd.Series(ys).value_counts()
    keep_classes = vc[vc >= min_per_class].index
    keep = np.isin(ys, keep_classes)
    Xs, ys = Xs[keep], ys[keep]
    enc = LabelEncoder().fit(ys)
    return Xs, enc.transform(ys), list(enc.classes_)


def multiclass_probe(
    X: np.ndarray, y, n_pca: int = 50, n_perm: int = 300,
    min_per_class: int = 4, seed: int = 42,
) -> Dict[str, Any]:
    """Multinomial logistic probe: macro-F1, balanced acc, macro 1-vs-rest AUC + perm p.

    Chance balanced-accuracy = 1 / n_classes. Folds auto-capped at the smallest
    class size (min 2). Rare classes (< ``min_per_class``) are dropped with the
    count reported back in ``n``.
    """
    Xs, yi, classes = _prep_multiclass(X, y, min_per_class)
    k = len(classes)
    if k < 2:
        raise ValueError(f"need >=2 classes with >={min_per_class} samples; got {k}.")
    n_splits = int(min(5, np.bincount(yi).min()))
    Xp = PCA(n_components=min(n_pca, Xs.shape[0] - 1), random_state=seed).fit_transform(Xs)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    f1 = float(cross_val_score(pipe, Xp, yi, cv=cv, scoring="f1_macro").mean())
    proba = cross_val_predict(pipe, Xp, yi, cv=cv, method="predict_proba")
    auc = float(
        roc_auc_score(yi, proba[:, 1]) if k == 2
        else roc_auc_score(yi, proba, multi_class="ovr", average="macro")
    )
    bal, _, p = permutation_test_score(
        pipe, Xp, yi, cv=cv, scoring="balanced_accuracy",
        n_permutations=n_perm, random_state=seed,
    )
    return {
        "n": int(Xs.shape[0]), "n_classes": k, "classes": classes,
        "macro_f1": round(f1, 3), "balanced_acc": round(float(bal), 3),
        "macro_auc": round(auc, 3), "chance": round(1.0 / k, 3),
        "p_value": round(float(p), 4), "n_splits": n_splits,
    }


def class_similarity_matrix(X: np.ndarray, y, min_per_class: int = 2) -> pd.DataFrame:
    """Mean pairwise cosine similarity WITHIN and BETWEEN classes (k x k frame).

    Diagonal = how tight each class is; high off-diagonal = two configurations are
    entangled in embedding space (the model can't tell them apart). Unsupervised
    view — no classifier involved.
    """
    Xs, yi, classes = _prep_multiclass(X, y, min_per_class)
    S = Xs @ Xs.T
    k = len(classes)
    M = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            bi, bj = yi == i, yi == j
            block = S[np.ix_(bi, bj)]
            if i == j:  # within: exclude self-similarity diagonal
                iu = np.triu_indices(bi.sum(), k=1)
                M[i, j] = block[iu].mean() if len(iu[0]) else np.nan
            else:
                M[i, j] = block.mean()
    return pd.DataFrame(M, index=classes, columns=classes).round(3)


def class_confusion_matrix(
    X: np.ndarray, y, n_pca: int = 50, min_per_class: int = 4,
    normalize: bool = True, seed: int = 42,
) -> pd.DataFrame:
    """Row-normalised cross-validated confusion matrix of the multinomial probe.

    Row = true class, column = predicted class. Off-diagonal mass shows WHICH
    configurations get mistaken for which — the most thesis-relevant output.
    """
    Xs, yi, classes = _prep_multiclass(X, y, min_per_class)
    n_splits = int(min(5, np.bincount(yi).min()))
    Xp = PCA(n_components=min(n_pca, Xs.shape[0] - 1), random_state=seed).fit_transform(Xs)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, Xp, yi, cv=cv)
    cm = confusion_matrix(yi, pred).astype(float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    return pd.DataFrame(cm.round(3), index=classes, columns=classes)
