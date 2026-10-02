"""One evaluation protocol for every image source (patent figures, evtol.news photos).

Both embedding reports call these functions with the same parameters, so a
number in one report means exactly what the number in the same table of the
other report means (user, 2026-09-22: "evaluating the embeddings shall be kind
of the same thing"). Only the images, the labels and the confound differ.

Unit = the aircraft: one vector per aircraft, chosen by an image rule.
Every test that compares aircraft leaves out pairs or neighbours of the SAME
MAKER, so a company's house style can never count as architecture.

    T1 integrity     NaN / Inf / all-zero rows, exact duplicate rows
    T2 structure     cosine mean and SD, PC1 / random, participation ratio,
                     dimensions for 90 % of the variance, Hopkins
    T3 separation    mean between-class / mean within-class cosine distance,
                     Cohen's d, label-permutation p; same-maker pairs left out
    T4 prediction    kNN-5 balanced accuracy, maker held out (PRIMARY SCORE),
                     95 % bootstrap interval, shuffled-label chance;
                     logistic probe on folds grouped by maker
    T5 clusters      k-means with k = number of classes against the labels (ARI, NMI)
    T6 confound      T3's d for a non-architecture label / T3's d for the architecture
    T7 image choice  T4 under every image rule of the source
    layer rule       the matrix with the highest T4 score averaged over the image
                     rules; how often it wins on random 80 % subsets of the aircraft
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, f1_score,
                             normalized_mutual_info_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold

from .embedding_metrics import _dims_for, hopkins, participation_ratio

Key = Tuple[int, str]
MATRICES: List[Key] = [(18, "cls"), (18, "mean_patch"), (22, "cls"), (22, "mean_patch"),
                       (24, "cls"), (24, "mean_patch")]
SEED = 42
K = 5                 # neighbours of the kNN vote
N_PERM = 999          # label permutations for T3
N_SHUFFLE = 200       # shuffled labels for the kNN chance level
N_BOOT = 1000         # bootstrap resamples of the kNN predictions (interval)
N_SUB = 200           # random 80 % subsets for the layer-choice stability
SUB_FRAC = 0.8


def mname(key: Key) -> str:
    return f"L{key[0]} {'CLS' if key[1] == 'cls' else 'patch mean'}"


def l2(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


# ── T1 / T2 ──────────────────────────────────────────────────────────────────
def integrity(X: np.ndarray) -> Dict[str, int]:
    X = np.asarray(X, dtype=np.float64)
    return {"rows": int(len(X)), "nan_or_inf": int((~np.isfinite(X)).any(axis=1).sum()),
            "zero_rows": int((np.abs(X).sum(axis=1) == 0).sum()),
            "duplicate_rows": int(len(X) - len(np.unique(np.round(X, 6), axis=0)))}


def structure(X: np.ndarray, seed: int = SEED) -> Dict[str, float]:
    Xn = l2(X)
    rng = np.random.default_rng(seed)
    S = Xn @ Xn.T
    iu = np.triu_indices(len(Xn), 1)
    cos = S[iu]
    n_comp = min(len(Xn) - 1, 60)
    R = l2(rng.standard_normal(Xn.shape))
    ev = PCA(n_components=n_comp).fit(Xn).explained_variance_ratio_
    ev_r = PCA(n_components=n_comp).fit(R).explained_variance_ratio_
    return {"cos_mean": float(cos.mean()), "cos_sd": float(cos.std()),
            "pc1_vs_random": float(ev[0] / ev_r[0]),
            "participation_ratio": float(participation_ratio(Xn)),
            "dims_90": int(_dims_for(Xn)), "hopkins": float(hopkins(Xn, seed=seed))}


# ── T3 ───────────────────────────────────────────────────────────────────────
def separation(X: np.ndarray, y: Sequence, groups: Sequence, n_perm: int = N_PERM,
               seed: int = SEED) -> Dict[str, float]:
    """Are same-label aircraft closer than different-label ones? Cosine distance
    over all pairs from DIFFERENT groups (makers). Ratio = mean(between) /
    mean(within); d = the same gap in pooled standard deviations; p from
    ``n_perm`` label permutations (one-sided, ratio as large or larger)."""
    Xn = l2(X)
    y, g = np.asarray(y), np.asarray(groups)
    a, b = np.triu_indices(len(Xn), 1)
    keep = g[a] != g[b]
    a, b = a[keep], b[keep]
    d = 1.0 - np.einsum("ij,ij->i", Xn[a], Xn[b])
    codes = pd.factorize(y)[0]
    same = codes[a] == codes[b]
    intra, inter = d[same], d[~same]
    obs = inter.mean() / intra.mean()
    pooled = np.sqrt((intra.var(ddof=1) * (len(intra) - 1) + inter.var(ddof=1) * (len(inter) - 1))
                     / (len(intra) + len(inter) - 2))
    rng = np.random.default_rng(seed)
    cp = codes.copy()
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(cp)
        s = cp[a] == cp[b]
        ge += d[~s].mean() / d[s].mean() >= obs
    return {"ratio": float(obs), "d": float((inter.mean() - intra.mean()) / pooled),
            "p": float((ge + 1) / (n_perm + 1)), "pairs_within": int(same.sum()),
            "pairs_between": int((~same).sum())}


# ── T4 ───────────────────────────────────────────────────────────────────────
def neighbours(X: np.ndarray, groups: Sequence, k: int = K, S: np.ndarray | None = None) -> np.ndarray:
    """The ``k`` nearest aircraft (cosine) of each aircraft, never from its own group.
    ``S`` = a precomputed cosine-similarity matrix of the same rows (it is copied)."""
    if S is None:
        Xn = l2(X)
        S = Xn @ Xn.T
    else:
        S = S.copy()
    g = pd.factorize(np.asarray(groups))[0]
    S[g[:, None] == g[None, :]] = -np.inf
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    order = np.take_along_axis(-S, nn, axis=1).argsort(axis=1)
    return np.take_along_axis(nn, order, axis=1)


def vote(nn: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Majority label of the neighbours; a tie goes to the label of the nearer one."""
    k = nn.shape[1]
    out = np.empty(len(nn), dtype=object)
    for i, row in enumerate(nn):
        v: Dict[str, float] = {}
        for r, j in enumerate(row):
            v[y[j]] = v.get(y[j], 0.0) + 1.0 + 1e-3 * (k - r)
        out[i] = max(v, key=v.get)
    return out


def knn(X: np.ndarray, y: Sequence, makers: Sequence, k: int = K, seed: int = SEED,
        n_shuffle: int = N_SHUFFLE, n_boot: int = N_BOOT) -> Dict[str, object]:
    y = np.asarray(y, dtype=object)
    nn = neighbours(X, makers, k)
    pred = vote(nn, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i])) == len(set(y)):
            boot.append(balanced_accuracy_score(y[i], pred[i]))
    shuf = [balanced_accuracy_score(yp, vote(nn, yp)) for yp in (rng.permutation(y) for _ in range(n_shuffle))]
    labels = sorted(set(y))
    rec = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    return {"bal_acc": float(balanced_accuracy_score(y, pred)),
            "ci_lo": float(np.quantile(boot, 0.025)), "ci_hi": float(np.quantile(boot, 0.975)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "chance_mean": float(np.mean(shuf)), "chance_p95": float(np.quantile(shuf, 0.95)),
            "recall": dict(zip(labels, map(float, rec))), "pred": pred}


def probe(X: np.ndarray, y: Sequence, makers: Sequence, seed: int = SEED) -> Dict[str, float]:
    """Logistic regression, 5 folds grouped by maker (no maker in train and test)."""
    Xn, y = l2(X), np.asarray(y, dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(Xn, y, np.asarray(makers)):
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        clf.fit(Xn[tr], y[tr])
        pred[te] = clf.predict(Xn[te])
    return {"bal_acc": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro"))}


# ── T5 ───────────────────────────────────────────────────────────────────────
def clusters(X: np.ndarray, y: Sequence, seed: int = SEED) -> Dict[str, float]:
    y = np.asarray(y)
    km = KMeans(n_clusters=len(set(y)), n_init=10, random_state=seed).fit_predict(l2(X))
    return {"ari": float(adjusted_rand_score(y, km)), "nmi": float(normalized_mutual_info_score(y, km))}


# ── the whole protocol on one source ────────────────────────────────────────
def evaluate(rules: Dict[str, Dict[Key, np.ndarray]], y: Sequence, makers: Sequence,
             primary_rule: str, n_perm: int = N_PERM, seed: int = SEED
             ) -> Dict[str, pd.DataFrame]:
    """T1-T5 and T7 for every matrix. ``rules`` = {rule: {matrix key: X}}, rows
    in the same aircraft order as ``y`` and ``makers``. T1, T2, T3, T5 and the
    probe run on the primary rule; kNN (T4) runs on every rule (T7)."""
    y, makers = np.asarray(y, dtype=object), np.asarray(makers, dtype=object)
    rows_main, rows_rule, recalls, preds = [], [], [], {}
    for key in MATRICES:
        X = rules[primary_rule][key]
        kn = knn(X, y, makers, seed=seed)
        preds[key] = kn.pop("pred")
        recalls.append({"matrix": mname(key), **kn.pop("recall")})
        sep = separation(X, y, makers, n_perm, seed)
        pr = probe(X, y, makers, seed)
        cl = clusters(X, y, seed)
        rows_main.append({"matrix": mname(key), "layer": key[0], "pooling": key[1],
                          **{f"t1_{k}": v for k, v in integrity(X).items()},
                          **{f"t2_{k}": v for k, v in structure(X, seed).items()},
                          **{f"t3_{k}": v for k, v in sep.items()},
                          **{f"knn_{k}": v for k, v in kn.items()},
                          "probe_bal_acc": pr["bal_acc"], "probe_macro_f1": pr["macro_f1"],
                          "t5_ari": cl["ari"], "t5_nmi": cl["nmi"]})
        for rule, mats in rules.items():
            if rule == primary_rule:
                r = kn
            else:
                r = knn(mats[key], y, makers, seed=seed, n_shuffle=20)
                r.pop("pred"); r.pop("recall")
            rows_rule.append({"matrix": mname(key), "layer": key[0], "pooling": key[1], "rule": rule,
                              "knn_bal_acc": r["bal_acc"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    main = pd.DataFrame(rows_main)
    by_rule = pd.DataFrame(rows_rule)
    return {"main": main, "by_rule": by_rule, "recall": pd.DataFrame(recalls), "pred": preds}


def layer_choice(rules: Dict[str, Dict[Key, np.ndarray]], y: Sequence, makers: Sequence,
                 n_sub: int = N_SUB, frac: float = SUB_FRAC, seed: int = SEED) -> pd.DataFrame:
    """Mean kNN score over the image rules per matrix, and how often each matrix
    has the best mean on random ``frac`` subsets of the aircraft (the choice is
    only robust if the winner keeps winning when the aircraft change)."""
    y, makers = np.asarray(y, dtype=object), np.asarray(makers, dtype=object)
    sims = {(rule, key): l2(mats[key]) @ l2(mats[key]).T for rule, mats in rules.items() for key in MATRICES}
    preds = {rk: vote(neighbours(None, makers, S=S), y) for rk, S in sims.items()}
    rng = np.random.default_rng(seed)
    wins = {key: 0 for key in MATRICES}
    n = len(y)
    for _ in range(n_sub):
        i = rng.choice(n, int(frac * n), replace=False)
        # neighbours are recomputed inside the subset: the left-out aircraft cannot vote
        score = {}
        for key in MATRICES:
            s = []
            for rule in rules:
                p = vote(neighbours(None, makers[i], S=sims[(rule, key)][np.ix_(i, i)]), y[i])
                s.append(balanced_accuracy_score(y[i], p))
            score[key] = np.mean(s)
        wins[max(score, key=score.get)] += 1
    full = {key: np.mean([balanced_accuracy_score(y, preds[(r, key)]) for r in rules]) for key in MATRICES}
    return pd.DataFrame({"matrix": [mname(k) for k in MATRICES], "layer": [k[0] for k in MATRICES],
                         "pooling": [k[1] for k in MATRICES],
                         "mean_knn_over_rules": [full[k] for k in MATRICES],
                         "wins_on_subsets": [wins[k] / n_sub for k in MATRICES]})


def confound(X: np.ndarray, y_arch: Sequence, y_conf: Sequence, makers: Sequence,
             n_perm: int = 199, seed: int = SEED) -> Dict[str, float]:
    """T6: does a non-architecture label (view, maturity, ...) separate the
    vectors more than the architecture does? Same pairs for both labels."""
    a = separation(X, y_arch, makers, n_perm, seed)
    c = separation(X, y_conf, makers, n_perm, seed)
    return {"arch_ratio": a["ratio"], "arch_d": a["d"], "arch_p": a["p"],
            "conf_ratio": c["ratio"], "conf_d": c["d"], "conf_p": c["p"],
            "conf_over_arch": c["d"] / a["d"] if a["d"] > 0 else np.nan}
