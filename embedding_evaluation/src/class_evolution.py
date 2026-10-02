"""Do the evtol.news directory classes change in appearance across the development years?

Sub-question 4.5 of the analysis brief (added 2026-09-30). The design below was fixed before any result
was computed; the brief's Method states it as written here.

Data. The evtol.news hero images (registers model, 518 px, layer 24 CLS), each aircraft dated by its
development year (EVTOLNEWS_DS/0_source/years/aircraft_dev_year.csv): ONLY ``dev_year_status == "agreed"``,
the year both language-model readers gave; "not_stated" and "conflict" aircraft leave (the author has not
ruled the conflicts). Windows as in the Labelling Analysis: <= 2011, 2012-15, 2016-19, 2020-23,
2024-26 (partial). The maker is held out everywhere. For comparison, the patent drawings (primary rule,
exp3 main figure) with their five parent classes, dated by the priority year of the aircraft's patent
(``priority_year`` of the identity record, the field the Labelling Analysis uses; ``app_year`` is the
filing year and is later for 43 % of the drawing aircraft).

Tests.

1. convergence  For each aircraft, the cosine distance to its class centroid, the centroid being the
                renormalised mean of the class's dated aircraft of OTHER makers (the aircraft itself and its
                maker's other aircraft left out). Spearman rho between that distance and the year, per class
                and pooled (pooled: each distance minus its class median, so that a change in the class mix
                over time cannot make a trend). Interval: 1 000 bootstrap resamples of whole makers (seed 42),
                distances held fixed. Negative rho = the class's designs grow more alike over time. Also the
                median distance per class and window, with a maker-bootstrap interval, for the figure.
2. temporal kNN Queries = the dated aircraft of 2020-26. Past: neighbours from the aircraft of <= 2019 only;
                same period: neighbours from 2020-26 only (the query and its maker never vote). kNN-5 vote
                (embedding_protocol.neighbours / vote). The two pools are matched class by class, each class
                subsampled to the smaller of its two counts, 200 draws (seed 42), so size and class mix are
                equal. Scores = balanced accuracy averaged over the draws; score interval = aircraft bootstrap
                (1 000); difference same period minus past = maker bootstrap of the queries (1 000, one draw
                per resample); shuffled chance = past-pool labels shuffled 200 times.
3. drift        Cosine distance between the class centroid of 2016-19 and of 2020-23, against a permutation
                band: window labels shuffled within the class, 999 times (seed 42); p = (1 + #perm >= obs) /
                (1 + 999). A class with fewer than MIN_N aircraft in either window is not tested.
4. drawings     Tests 1 and 3 on the drawings' parent classes by priority year.

Outputs (numbers only, no images): 1639_LABELLED/3_embedding_evaluation/class_evolution/ce_*.csv.
Run: ``from src import class_evolution as CE; CE.run()`` (notebook 32, step "class_evolution").
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score, recall_score

from . import embedding_protocol as P
from . import embedding_reports as ER

REG = "dinov2-reg-large_518"
KEY: P.Key = (24, "cls")
OUT = ER.PAT / "3_embedding_evaluation" / "class_evolution"
YEARS = ER.EVN / "0_source" / "years" / "aircraft_dev_year.csv"
IDENTITY = ER.PAT / "0_labelling" / "inputs" / "identity" / "aircraft_identity_ALL.xlsx"
WINDOWS: List[Tuple[str, int, int]] = [("<= 2011", 0, 2011), ("2012-15", 2012, 2015), ("2016-19", 2016, 2019),
                                       ("2020-23", 2020, 2023), ("2024-26 (partial)", 2024, 2100)]
WIN_NAMES = [w[0] for w in WINDOWS]
PAST_END, PRESENT_START = 2019, 2020
DRIFT = ("2016-19", "2020-23")
MIN_N = 15
N_BOOT, N_PERM, N_DRAW, N_SHUF = 1000, 999, 200, 200
SEED = P.SEED
CLASSES = ER.CLASSES5
FILES = ["ce_coverage", "ce_counts", "ce_convergence", "ce_median_distance", "ce_temporal_knn", "ce_drift",
         "ce_shares"]


def window_of(y: float) -> str | None:
    for name, lo, hi in WINDOWS:
        if pd.notna(y) and lo <= y <= hi:
            return name
    return None


# ── data ─────────────────────────────────────────────────────────────────────
def load_photos() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    """The hero-image aircraft with an agreed development year: frame (aircraft_id, label, maker, year,
    window), their L24 CLS vectors, and the coverage counts."""
    ER.TAG = REG
    S = ER.load_evtolnews()
    y = pd.read_csv(YEARS, keep_default_na=False).set_index("slug_key")
    a = S.aircraft[["aircraft_id", "label", "maker"]].copy()
    st = a.aircraft_id.map(y["dev_year_status"]).fillna("missing")
    cov = {"hero-image aircraft": len(a), "agreed year": int((st == "agreed").sum()),
           "not stated": int((st == "not_stated").sum()), "conflict (not ruled)": int((st == "conflict").sum()),
           "absent from the year file": int((st == "missing").sum())}
    keep = (st == "agreed").to_numpy()
    a["year"] = pd.to_numeric(a.aircraft_id.map(y["dev_year"]), errors="coerce")
    a = a[keep].reset_index(drop=True)
    X = S.rules["hero image"][KEY][keep]
    a["window"] = a.year.map(window_of)
    return a, P.l2(X), cov


def load_drawings() -> Tuple[pd.DataFrame, np.ndarray]:
    """The drawing aircraft (exp3 main figure) with their parent class and the priority year of their patent."""
    ER.TAG = REG
    S = ER.load_patents()
    idn = pd.read_excel(IDENTITY).drop_duplicates("patent_id").set_index("patent_id")
    a = S.aircraft[["aircraft_id", "patent_id", "label5", "maker"]].rename(columns={"label5": "label"}).copy()
    a["year"] = pd.to_numeric(a.patent_id.map(idn["priority_year"]), errors="coerce")
    ok = a.year.notna().to_numpy()
    a = a[ok].reset_index(drop=True)
    a["window"] = a.year.map(window_of)
    return a, P.l2(S.rules[S.primary_rule][KEY][ok])


# ── test 1: convergence ──────────────────────────────────────────────────────
def centroid_distance(a: pd.DataFrame, X: np.ndarray) -> np.ndarray:
    """Cosine distance of each aircraft to its class centroid built from the class's aircraft of other makers."""
    d = np.full(len(a), np.nan)
    lab, mk = a.label.to_numpy(), a.maker.to_numpy()
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        tot = X[idx].sum(axis=0)
        by_maker: Dict[str, np.ndarray] = {}
        for i in idx:
            by_maker[mk[i]] = by_maker.get(mk[i], 0) + X[i]
        for i in idx:
            cen = tot - by_maker[mk[i]]
            nrm = np.linalg.norm(cen)
            if nrm > 0:
                d[i] = 1.0 - float(X[i] @ cen) / nrm
    return d


def _maker_boot(a: pd.DataFrame, n_boot: int, seed: int):
    """Row indices of ``n_boot`` resamples of whole makers."""
    rng = np.random.default_rng(seed)
    groups = a.groupby("maker").indices
    keys = list(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        yield np.concatenate([groups[keys[j]] for j in pick])


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    return float(spearmanr(x, y).statistic)


def convergence(a: pd.DataFrame, source: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    a = a[a.dist.notna()].reset_index(drop=True)
    a["dist_c"] = a.dist - a.groupby("label").dist.transform("median")
    rows, med = [], []
    for c in CLASSES + ["pooled"]:
        s = a if c == "pooled" else a[a.label == c].reset_index(drop=True)
        col = "dist_c" if c == "pooled" else "dist"
        obs = _rho(s[col].to_numpy(), s.year.to_numpy())
        boot = [_rho(s[col].to_numpy()[i], s.year.to_numpy()[i]) for i in _maker_boot(s, N_BOOT, SEED)]
        boot = np.array([b for b in boot if np.isfinite(b)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append({"source": source, "class": c, "n_aircraft": len(s), "n_makers": s.maker.nunique(),
                     "rho": obs, "ci_lo": lo, "ci_hi": hi,
                     "reading": "no trend detected" if lo <= 0 <= hi else
                     ("converging (distance falls with the year)" if hi < 0 else "diverging (distance rises with the year)")})
        if c == "pooled":
            continue
        for w in WIN_NAMES:
            sw = s[s.window == w].reset_index(drop=True)
            if len(sw) == 0:
                med.append({"source": source, "class": c, "window": w, "n_aircraft": 0})
                continue
            bm = [float(np.median(sw.dist.to_numpy()[i])) for i in _maker_boot(sw, N_BOOT, SEED)]
            med.append({"source": source, "class": c, "window": w, "n_aircraft": len(sw),
                        "median_distance": float(sw.dist.median()), "ci_lo": float(np.quantile(bm, 0.025)),
                        "ci_hi": float(np.quantile(bm, 0.975)), "small": len(sw) < MIN_N})
    return pd.DataFrame(rows), pd.DataFrame(med)


# ── test 2: temporal kNN ─────────────────────────────────────────────────────
def _vote_from(S: np.ndarray, q: np.ndarray, pool: np.ndarray, mk: np.ndarray, y: np.ndarray) -> np.ndarray:
    """kNN-5 vote of each query from ``pool`` only, never from the query's maker (embedding_protocol)."""
    sub = S[np.ix_(q, pool)].copy()
    sub[mk[q][:, None] == mk[pool][None, :]] = -np.inf
    nn = np.argpartition(-sub, P.K, axis=1)[:, :P.K]
    order = np.take_along_axis(-sub, nn, axis=1).argsort(axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    return P.vote(pool[nn], y)


def temporal_knn(a: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:
    y, mk = a.label.to_numpy(dtype=object), a.maker.to_numpy(dtype=object)
    S = X @ X.T
    past = np.flatnonzero(a.year.to_numpy() <= PAST_END)
    now = np.flatnonzero(a.year.to_numpy() >= PRESENT_START)
    q = now
    rng = np.random.default_rng(SEED)
    cap = {c: min(int((y[past] == c).sum()), int((y[now] == c).sum())) for c in CLASSES}
    preds_p, preds_s = [], []
    for _ in range(N_DRAW):
        pp = np.concatenate([rng.choice(past[y[past] == c], cap[c], replace=False) for c in CLASSES])
        ps = np.concatenate([rng.choice(now[y[now] == c], cap[c], replace=False) for c in CLASSES])
        preds_p.append(_vote_from(S, q, pp, mk, y))
        preds_s.append(_vote_from(S, q, ps, mk, y))
    yq = y[q]
    labels = [c for c in CLASSES if (yq == c).any()]

    def ba(yt, pr):
        return balanced_accuracy_score(yt, pr)
    sc_p = np.array([ba(yq, p) for p in preds_p])
    sc_s = np.array([ba(yq, p) for p in preds_s])
    rec_p = np.array([recall_score(yq, p, labels=labels, average=None, zero_division=0) for p in preds_p])
    rec_s = np.array([recall_score(yq, p, labels=labels, average=None, zero_division=0) for p in preds_s])
    # intervals: scores by an aircraft bootstrap, the difference by a maker bootstrap, one draw per resample
    n = len(q)
    boot_p, boot_s = [], []
    for b in range(N_BOOT):
        i = rng.integers(0, n, n)
        if len(set(yq[i])) < len(labels):
            continue
        j = b % N_DRAW
        boot_p.append(ba(yq[i], preds_p[j][i]))
        boot_s.append(ba(yq[i], preds_s[j][i]))
    qa = a.iloc[q].reset_index(drop=True)
    diff, dcls = [], {c: [] for c in labels}
    for b, i in enumerate(_maker_boot(qa, N_BOOT, SEED + 1)):
        j = b % N_DRAW
        if len(set(yq[i])) < len(labels):
            continue
        diff.append(ba(yq[i], preds_s[j][i]) - ba(yq[i], preds_p[j][i]))
        for c in labels:
            m = yq[i] == c
            dcls[c].append(float((preds_s[j][i][m] == c).mean() - (preds_p[j][i][m] == c).mean()))
    # chance: the past pool's labels shuffled (whole past pool, no subsampling)
    shuf = []
    for _ in range(N_SHUF):
        yy = y.copy()
        yy[past] = rng.permutation(y[past])
        shuf.append(ba(yq, _vote_from(S, q, past, mk, yy)))
    rows = [{"class": "all (balanced accuracy)", "n_queries": n, "pool_per_draw": int(sum(cap.values())),
             "past": float(sc_p.mean()), "past_ci_lo": float(np.quantile(boot_p, 0.025)),
             "past_ci_hi": float(np.quantile(boot_p, 0.975)), "same_period": float(sc_s.mean()),
             "same_ci_lo": float(np.quantile(boot_s, 0.025)), "same_ci_hi": float(np.quantile(boot_s, 0.975)),
             "diff_same_minus_past": float(sc_s.mean() - sc_p.mean()),
             "diff_ci_lo": float(np.quantile(diff, 0.025)), "diff_ci_hi": float(np.quantile(diff, 0.975)),
             "shuffled_p95": float(np.quantile(shuf, 0.95)), "shuffled_mean": float(np.mean(shuf))}]
    for k, c in enumerate(labels):
        rows.append({"class": c, "n_queries": int((yq == c).sum()), "pool_per_draw": cap[c],
                     "past": float(rec_p[:, k].mean()), "same_period": float(rec_s[:, k].mean()),
                     "diff_same_minus_past": float(rec_s[:, k].mean() - rec_p[:, k].mean()),
                     "diff_ci_lo": float(np.quantile(dcls[c], 0.025)), "diff_ci_hi": float(np.quantile(dcls[c], 0.975)),
                     "past_pool_full": int((y[past] == c).sum()), "same_pool_full": int((y[now] == c).sum())})
    out = pd.DataFrame(rows)
    out["reading"] = ["no difference detected" if lo <= 0 <= hi else
                      ("same period higher: the class has moved" if lo > 0 else "past higher")
                      for lo, hi in zip(out.diff_ci_lo, out.diff_ci_hi)]
    return out


# ── test 3: centroid drift ───────────────────────────────────────────────────
def _cdist(A: np.ndarray, B: np.ndarray) -> float:
    ca, cb = P.l2(A.mean(axis=0, keepdims=True))[0], P.l2(B.mean(axis=0, keepdims=True))[0]
    return float(1.0 - ca @ cb)


def drift(a: pd.DataFrame, X: np.ndarray, source: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for c in CLASSES:
        i1 = np.flatnonzero((a.label == c).to_numpy() & (a.window == DRIFT[0]).to_numpy())
        i2 = np.flatnonzero((a.label == c).to_numpy() & (a.window == DRIFT[1]).to_numpy())
        r = {"source": source, "class": c, "n_2016_19": len(i1), "n_2020_23": len(i2)}
        if min(len(i1), len(i2)) < MIN_N:
            rows.append({**r, "tested": False, "reading": f"not tested (fewer than {MIN_N} aircraft in a window)"})
            continue
        obs = _cdist(X[i1], X[i2])
        both = np.concatenate([i1, i2])
        perm = []
        for _ in range(N_PERM):
            p = rng.permutation(both)
            perm.append(_cdist(X[p[:len(i1)]], X[p[len(i1):]]))
        perm = np.array(perm)
        pval = (1 + int((perm >= obs).sum())) / (1 + N_PERM)
        rows.append({**r, "tested": True, "distance": obs, "perm_median": float(np.median(perm)),
                     "perm_p95": float(np.quantile(perm, 0.95)), "excess": obs - float(np.median(perm)), "p": pval,
                     "reading": "moved (above the permutation band)" if pval < 0.05 else "no drift detected"})
    return pd.DataFrame(rows)


def holm(d: pd.DataFrame) -> pd.DataFrame:
    """Holm-adjusted p over every tested drift row, both sources (added after the results were seen)."""
    d = d.copy()
    t = d[d.tested.astype(bool)].sort_values("p")
    m, run_max, adj = len(t), 0.0, {}
    for k, (i, p) in enumerate(t.p.items()):
        run_max = max(run_max, min(1.0, (m - k) * p))
        adj[i] = run_max
    d["p_holm"] = pd.Series(adj)
    return d


# ── counts and context ───────────────────────────────────────────────────────
def counts(a: pd.DataFrame, source: str) -> pd.DataFrame:
    t = pd.crosstab(a.label, a.window).reindex(index=CLASSES, columns=WIN_NAMES, fill_value=0)
    return t.reset_index().melt(id_vars="label", var_name="window", value_name="n_aircraft").rename(
        columns={"label": "class"}).assign(source=source)


def shares(a: pd.DataFrame) -> pd.DataFrame:
    t = pd.crosstab(a.window, a.label).reindex(index=WIN_NAMES, columns=CLASSES, fill_value=0)
    rows = []
    for w in WIN_NAMES:
        n = int(t.loc[w].sum())
        for c in CLASSES:
            k = int(t.loc[w, c])
            lo, hi = ER.wilson(k, n) if n else (np.nan, np.nan)
            rows.append({"window": w, "class": c, "n_aircraft": k, "window_total": n, "share": k / n if n else np.nan,
                         "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(rows)


def run() -> Dict[str, pd.DataFrame]:
    OUT.mkdir(parents=True, exist_ok=True)
    ph, Xh, cov = load_photos()
    dr, Xd = load_drawings()
    ph["dist"] = centroid_distance(ph, Xh)
    dr["dist"] = centroid_distance(dr, Xd)
    conv_h, med_h = convergence(ph, "photos")
    conv_d, med_d = convergence(dr, "drawings")
    T = {"ce_coverage": pd.DataFrame([{"measure": k, "aircraft": v} for k, v in cov.items()]
                                     + [{"measure": "drawing aircraft with a priority year", "aircraft": len(dr)}]),
         "ce_counts": pd.concat([counts(ph, "photos"), counts(dr, "drawings")], ignore_index=True),
         "ce_convergence": pd.concat([conv_h, conv_d], ignore_index=True),
         "ce_median_distance": pd.concat([med_h, med_d], ignore_index=True),
         "ce_temporal_knn": temporal_knn(ph, Xh),
         "ce_drift": holm(pd.concat([drift(ph, Xh, "photos"), drift(dr, Xd, "drawings")], ignore_index=True)),
         "ce_shares": shares(ph)}
    for k, df in T.items():
        df.to_csv(OUT / f"{k}.csv", index=False)
    print("wrote", ", ".join(f"{k}.csv" for k in T), "to", OUT)
    return T


def load() -> Dict[str, pd.DataFrame]:
    return {k: pd.read_csv(OUT / f"{k}.csv") for k in FILES}


if __name__ == "__main__":
    run()
