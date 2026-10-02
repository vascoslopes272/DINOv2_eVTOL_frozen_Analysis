"""Linear view erasure: does the architecture signal survive removing the drawing view? (chapter plan N6)

The protocol's T6 found that the view group of a patent drawing (Perspective, Plan, Side,
Front/Rear) separates the figure embeddings more than the architecture does (L24 CLS: view d /
architecture d = 2.2). This module removes the view from the embeddings with a LINEAR erasure
fitted on the view labels only (the architecture labels are never seen), so the architecture
scores can be recomputed on vectors that no longer carry the view.

    1. mean projection   project out the span of the view-class mean differences (rank <= 3)
    2. INLP              while a linear view probe (maker-grouped folds, embedding_protocol.probe)
                         is more than ``TOL`` above chance: fit a logistic view classifier on all
                         figures, project out its weight directions, repeat (at most ``MAX_ITER``)

The erasure is an orthogonal projection x -> (I - Q Q^T) x followed by re-normalisation
(embedding_protocol.l2), Q = orthonormal basis of every removed direction. Metrics come from
embedding_protocol; only the projection and its bookkeeping live here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from . import embedding_protocol as P

TOL = 0.05        # "near chance": view probe balanced accuracy within TOL of 1 / number of views
MAX_ITER = 10     # INLP rounds after the mean projection


def orthonormal(D: np.ndarray, Q: np.ndarray | None = None, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal basis (columns) of span(D's columns) with the part already in span(Q) removed."""
    D = np.asarray(D, dtype=np.float64)
    if Q is not None and Q.shape[1]:
        D = D - Q @ (Q.T @ D)
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    return U[:, s > tol * max(s.max(initial=0.0), 1.0)]


def project(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Remove span(Q) from every row and re-normalise."""
    X = np.asarray(X, dtype=np.float64)
    return P.l2(X - (X @ Q) @ Q.T) if Q.shape[1] else P.l2(X)


def mean_directions(X: np.ndarray, y: Sequence) -> np.ndarray:
    """Class-mean differences (columns): mean of each class minus the mean of the first class."""
    Xn, y = P.l2(X), np.asarray(y, dtype=object)
    mu = np.stack([Xn[y == c].mean(axis=0) for c in sorted(set(y))])
    return (mu[1:] - mu[0]).T


def classifier_directions(X: np.ndarray, y: Sequence, seed: int = P.SEED) -> np.ndarray:
    """Weight vectors (columns) of a logistic view classifier fitted on all rows; same settings
    as embedding_protocol.probe."""
    clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced", random_state=seed)
    clf.fit(P.l2(X), np.asarray(y, dtype=object))
    return clf.coef_.T


def fit(X: np.ndarray, y: Sequence, makers: Sequence, tol: float = TOL, max_iter: int = MAX_ITER,
        seed: int = P.SEED) -> Dict[str, Any]:
    """Fit the erasure on figure vectors ``X`` with view labels ``y``. Returns the basis ``Q`` of
    the removed directions and ``history``: one row per step with the view probe on the erased
    vectors (step 0 = nothing removed)."""
    y = np.asarray(y, dtype=object)
    chance = 1.0 / len(set(y))
    Q = np.zeros((X.shape[1], 0))
    hist: List[Dict[str, Any]] = []

    def log(step: str) -> float:
        pr = P.probe(project(X, Q), y, makers, seed)
        hist.append({"step": step, "n_dirs": int(Q.shape[1]), "probe_bal_acc": pr["bal_acc"],
                     "probe_macro_f1": pr["macro_f1"]})
        return pr["bal_acc"]

    log("original")
    Q = np.hstack([Q, orthonormal(mean_directions(X, y), Q)])
    acc = log("mean projection")
    it = 0
    while acc > chance + tol and it < max_iter:
        it += 1
        Q = np.hstack([Q, orthonormal(classifier_directions(project(X, Q), y, seed), Q)])
        acc = log(f"INLP {it}")
    return {"Q": Q, "history": hist, "chance": chance, "converged": bool(acc <= chance + tol)}


def random_basis(dim: int, rank: int, seed: int = P.SEED) -> np.ndarray:
    """``rank`` random orthonormal directions (the control erasure)."""
    return np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, rank)))[0]


def random_matched(X: np.ndarray, rank: int, share: float, seed: int = P.SEED) -> Dict[str, Any]:
    """``rank`` random orthonormal directions drawn inside the span of the top ``m`` principal
    components of ``X`` (no labels), ``m`` chosen so that the expected centred variance share
    (rank / m x variance of the top m) is closest to ``share``: a control that removes as much
    variance as the view erasure does, but in directions not chosen by the view."""
    Xn = P.l2(X)
    _, s, Vt = np.linalg.svd(Xn - Xn.mean(axis=0), full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    ms = np.arange(rank, len(cum) + 1)
    m = int(ms[np.argmin(np.abs(rank / ms * cum[ms - 1] - share))])
    R = np.linalg.qr(np.random.default_rng(seed).standard_normal((m, rank)))[0]
    return {"Q": Vt[:m].T @ R, "m": m}


def variance_share(X: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """Share of the (unit-norm) vectors' variance that lies in span(Q): centred = of the total
    variance around the mean; uncentred = of the squared norm (what cosine distances see)."""
    Xn = P.l2(X)
    Xc = Xn - Xn.mean(axis=0)
    return {"centred": float(((Xc @ Q) ** 2).sum() / (Xc ** 2).sum()),
            "uncentred": float(((Xn @ Q) ** 2).sum() / (Xn ** 2).sum())}


def paired(y: Sequence, p0: np.ndarray, p1: np.ndarray, n_boot: int = P.N_BOOT,
           seed: int = P.SEED) -> Dict[str, float]:
    """Paired bootstrap of balanced accuracy (p1 - p0) on the same aircraft (the resamples that
    miss a class are skipped, as in embedding_protocol.knn)."""
    y = np.asarray(y, dtype=object)
    rng, diffs = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == len(set(y)):
            diffs.append(balanced_accuracy_score(y[i], p1[i]) - balanced_accuracy_score(y[i], p0[i]))
    return {"diff": float(balanced_accuracy_score(y, p1) - balanced_accuracy_score(y, p0)),
            "ci_lo": float(np.quantile(diffs, 0.025)), "ci_hi": float(np.quantile(diffs, 0.975)),
            "agree": float((p0 == p1).mean())}
