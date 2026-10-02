"""Review fixes: the CPU-side numbers and figures asked for by the review of the DINOv2 analysis document.

Review of 2026-09-30 (docs/embedding_evaluation/analysis_document/review/CONSOLIDATED_FLAGS.md). Every
function here answers one flag with a new number computed from the saved embeddings and results; the
metrics are the protocol's own (embedding_protocol: neighbours, vote, knn, probe, separation, l2), the
split-half rule is heldout_selection's, the erasure is view_erasure's. Nothing here overwrites an existing
result: every output goes to 1639_LABELLED/3_embedding_evaluation/review_fixes/.

    M3   pool size and class mix of the photo advantage: size-only and size+mix subsets, intervals that
         resample both sides
    M4   view erasure: held-out table, held-out architecture effect over maker splits, the
         variance-matched control over 200 draws, erased scores with intervals and chance
    M6   size-balanced per-class recall (12 G1 types) and the probe's per-class recall
    M9   Wilson intervals on the full-sample per-class recall; the share of wrong predictions in SLC / TP
    M11  count reconciliation (99 / 94 linked aircraft, 665 / 663, 1 545 / 1 541, 677 / 665, ARI)
    S2   probe with bootstrap interval, maker-clustered interval, fold-seed spread and label-shuffle chance
    S3   k = 3 / 5 / 10 and paired differences between the image rules (+ the split-half rule over rules)
    S4   maker-clustered paired bootstrap of every paired contrast, per-half sign consistency, Holm
    S6   matching top-10 / top-50 with bootstrap intervals, by the status of the linked page; base row
    S7/S8  each source's own pick, halves missing a class, seeds 7 and 123, corrected Table 12, the
         joint 6 x 3 grid, the best-held-out-score criterion
    S9   attention: renormalised max-patch share, p90, prefix share, argmax in the pad; base on the 665
    S10  Wilson intervals of the photo-filter audit; kappa breakdown and the frozen-label kappa
    S11  bootstrap interval of the view / architecture d ratio; per-aircraft against per-figure d;
         the view kNN with chance; the figure embeddings in 2-D coloured by view

Run: scripts/review_fixes.py (Finetune env). Printed class names follow the display layer: TR prints as TP.
"""

from __future__ import annotations

import json
import pickle
import textwrap
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, recall_score

from . import embedding_protocol as P
from . import embedding_reports as ER
from . import heldout_selection as H
from . import view_erasure as VE

REG, BASE = "dinov2-reg-large_518", "dinov2-large_518"
KEY: P.Key = (24, "cls")
PE, EE = ER.PAT / "3_embedding_evaluation", ER.EVN / "3_embedding_evaluation"
OUT = PE / "review_fixes"
SEL, VEDIR, REGCHK, CROP = PE / "setup_selection", PE / "view_erasure", PE / "registers_check", EE / "crop_experiment"
T_PAT_REG, T_PAT_BASE = PE / "embedding_protocol_registers", PE / "embedding_protocol"
T_EVN_REG, T_EVN_BASE = EE / "embedding_analysis_registers/tables", EE / "embedding_analysis/tables"
ATT = ER.EVN / "2_embedding_extraction/attention"
C5, G1 = ER.CLASSES5, ER.G1_ORDER
LAB12, LAB5, PHO5 = "patents G1 12-class", "patents parent 5-class", "photos 5-class"
LAB_SHORT = {LAB12: "drawings, 12 G1 classes", LAB5: "drawings, 5 parent classes", PHO5: "photos, 5 classes"}
SOURCE_OF_LABEL = {LAB12: "patents", LAB5: "patents", PHO5: "photos"}
N_BOOT, SEED = P.N_BOOT, P.SEED
G1_NAME = {"TR": "Tilt Propulsor", "CVT": "Combined vectored thrust", "TW": "Tilt Wing", "TB": "Tilt Body",
           "PTC": "Pitch-to-Cruise", "DS": "Deflected Slipstream", "SLC": "Lift + Cruise",
           "SRW": "Stopped/Slowed Rotor Wing", "MR": "Multirotor", "RC": "Rotorcraft", "HB": "Hoverbike",
           "PFV": "Personal Flying Vehicle"}


def disp(c: str) -> str:
    """Display layer (labeling_evaluation display_names.py): the class TR prints as TP."""
    return {"TR": "TP"}.get(c, c)


# ── loading ──────────────────────────────────────────────────────────────────
_SRC: Dict[str, Tuple[ER.Source, ER.Source]] = {}


def sources(tag: str = REG) -> Tuple[ER.Source, ER.Source]:
    """(patents, photos) loaded under ``tag``; cached. The loaders read the module constant ER.TAG."""
    if tag not in _SRC:
        old = ER.TAG
        ER.TAG = tag
        try:
            _SRC[tag] = (ER.load_patents(), ER.load_evtolnews())
        finally:
            ER.TAG = old
    return _SRC[tag]


def label_sets(tag: str = REG, key: P.Key = KEY) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """{label set: (X, y, makers)} on the primary image rule of each source."""
    Sp, Sh = sources(tag)
    Xp, Xh = Sp.rules[Sp.primary_rule][key], Sh.rules[Sh.primary_rule][key]
    return {LAB12: (Xp, Sp.y, Sp.makers),
            LAB5: (Xp, Sp.aircraft["label5"].to_numpy(dtype=object), Sp.makers),
            PHO5: (Xh, Sh.y, Sh.makers)}


def saved_results(which: str) -> Dict:
    d = {"pat_reg": T_PAT_REG, "pat_base": T_PAT_BASE, "evn_reg": T_EVN_REG, "evn_base": T_EVN_BASE}[which]
    return pickle.loads((d / "_results.pkl").read_bytes())


def saved_main(label: str, tag: str = REG, matrix: str = "L24 CLS") -> pd.Series:
    """The saved protocol row (kNN, interval, chance, probe, ARI) of a label set."""
    R = saved_results(("pat_" if label != PHO5 else "evn_") + ("reg" if tag == REG else "base"))
    m = R["ev5" if label == LAB5 else "ev"]["main"]
    return m[m.matrix == matrix].iloc[0]


# ── primitives (all metric work is the protocol's) ───────────────────────────
def predict(X: np.ndarray, y: Sequence, makers: Sequence, k: int = P.K) -> np.ndarray:
    """kNN-k vote, maker held out (embedding_protocol.neighbours + vote)."""
    return P.vote(P.neighbours(X, makers, k), np.asarray(y, dtype=object))


def predict_sub(S: np.ndarray, i: np.ndarray, y: np.ndarray, makers: np.ndarray) -> np.ndarray:
    """kNN-5 inside the subset ``i`` only: its neighbours come from the subset (as in the split-half rule)."""
    return P.vote(P.neighbours(None, makers[i], S=S[np.ix_(i, i)]), y[i])


def bal(y, p) -> float:
    return float(balanced_accuracy_score(np.asarray(y, dtype=object), np.asarray(p, dtype=object)))


def wilson(k: int, n: int) -> Tuple[float, float]:
    if not n:
        return np.nan, np.nan
    lo, hi = ER.wilson(int(k), int(n))
    return max(0.0, lo), min(1.0, hi)


def _members(makers: np.ndarray) -> List[np.ndarray]:
    codes = pd.factorize(np.asarray(makers, dtype=object))[0]
    return [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]


def resample(rng, n: int, members: List[np.ndarray] | None) -> np.ndarray:
    """One bootstrap resample: aircraft (``members`` None) or whole makers with all their aircraft."""
    if members is None:
        return rng.integers(0, n, n)
    return np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])


def boot_scores(y, preds: Dict[str, np.ndarray], makers=None, n_boot: int = N_BOOT, seed: int = SEED
                ) -> Dict[str, np.ndarray]:
    """Balanced accuracy of several prediction vectors on the SAME resamples (resamples that miss a
    class are drawn again, as in preprocess_variants_eval._boot_diff)."""
    y = np.asarray(y, dtype=object)
    members = _members(makers) if makers is not None else None
    rng, k = np.random.default_rng(seed), len(set(y))
    out: Dict[str, List[float]] = {n: [] for n in preds}
    tries = 0
    while len(next(iter(out.values()))) < n_boot and tries < 50 * n_boot:
        tries += 1
        i = resample(rng, len(y), members)
        if len(set(y[i])) != k:
            continue
        for n, p in preds.items():
            out[n].append(balanced_accuracy_score(y[i], p[i]))
    return {n: np.asarray(v) for n, v in out.items()}


def paired_boot(y, p0, p1, makers=None, n_boot: int = N_BOOT, seed: int = SEED) -> Dict[str, float]:
    """(p1 - p0) balanced accuracy, bootstrap over aircraft (makers None) or over makers."""
    b = boot_scores(y, {"a": p0, "b": p1}, makers, n_boot, seed)
    d = b["b"] - b["a"]
    return {"diff": bal(y, p1) - bal(y, p0), "ci_lo": float(np.quantile(d, 0.025)),
            "ci_hi": float(np.quantile(d, 0.975)),
            "p_two_sided": float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))), "n_resamples": len(d)}


def interval(v: Sequence[float]) -> Tuple[float, float]:
    v = np.asarray(v, dtype=float)
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def probe_predictions(X: np.ndarray, y: Sequence, makers: Sequence, seed: int = SEED) -> np.ndarray:
    """Out-of-fold predictions of embedding_protocol.probe: the same folds (StratifiedGroupKFold, 5,
    shuffled, random_state = seed) and the same classifier (LogisticRegression C = 1.0, L2, lbfgs,
    class_weight balanced, max_iter 4000) on the same l2-normalised vectors. P.probe returns only the
    scores; the per-class recall needs the predictions. Callers check the score against P.probe's."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    Xn, y = P.l2(X), np.asarray(y, dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(Xn, y, np.asarray(makers)):
        clf = LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced")
        clf.fit(Xn[tr], y[tr])
        pred[te] = clf.predict(Xn[te])
    return pred


# ═════════════════════════════════════════════════════════════════════════════
# M3: is the photo advantage the pool?
# ═════════════════════════════════════════════════════════════════════════════
def _counts_like(y: np.ndarray, n: int) -> Dict[str, int]:
    """Class counts summing to ``n`` in the proportions of ``y`` (largest remainder)."""
    share = pd.Series(y).value_counts(normalize=True).reindex(C5).fillna(0)
    raw = share * n
    base = np.floor(raw).astype(int)
    rest = n - int(base.sum())
    for c in (raw - base).sort_values(ascending=False).index[:rest]:
        base[c] += 1
    return {c: int(base[c]) for c in C5}


def m3_pool(n_draw: int = 200, n_boot: int = N_BOOT) -> Dict[str, pd.DataFrame]:
    from .preprocess_variants_eval import _boot_diff
    Sp, Sh = sources(REG)
    Xd, yd, md = label_sets()[LAB5]
    Xh, yh, mh = label_sets()[PHO5]
    pdr, ph = predict(Xd, yd, md), predict(Xh, yh, mh)
    Sm = P.l2(Xh) @ P.l2(Xh).T
    n_d = len(yd)
    idx = {c: np.flatnonzero(yh == c) for c in C5}
    counts = {"size + mix": {c: int((yd == c).sum()) for c in C5}, "size only": _counts_like(yh, n_d)}
    # size + mix: the same draws as analysis_document.class_mix_matched (same rng, same calls)
    seeds = {"size + mix": SEED, "size only": SEED + 1}
    draws: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for name, cc in counts.items():
        rng = np.random.default_rng(seeds[name])
        draws[name] = []
        for _ in range(n_draw):
            i = np.concatenate([rng.choice(idx[c], cc[c], replace=False) for c in C5])
            draws[name].append((i, predict_sub(Sm, i, yh, mh)))
    score = {n: np.array([bal(yh[i], p) for i, p in d]) for n, d in draws.items()}
    rec = {n: np.array([recall_score(yh[i], p, labels=C5, average=None, zero_division=0) for i, p in d])
           for n, d in draws.items()}

    def both_sides(name: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """Draw x aircraft bootstrap for the photos, aircraft bootstrap for the drawings."""
        rng = np.random.default_rng(seed)
        sp, diff = [], []
        while len(diff) < n_boot:
            i, p = draws[name][rng.integers(n_draw)]
            j = rng.integers(0, len(i), len(i))
            if len(set(yh[i][j])) != len(C5):
                continue
            k = rng.integers(0, n_d, n_d)
            if len(set(yd[k])) != len(C5):
                continue
            s = balanced_accuracy_score(yh[i][j], p[j])
            sp.append(s)
            diff.append(s - balanced_accuracy_score(yd[k], pdr[k]))
        return np.asarray(sp), np.asarray(diff)

    sd, sh = bal(yd, pdr), bal(yh, ph)
    m_d, m_h = saved_main(LAB5), saved_main(PHO5)
    assert abs(sd - m_d.knn_bal_acc) < 1e-12 and abs(sh - m_h.knn_bal_acc) < 1e-12
    full = _boot_diff(yd, pdr, None, np.random.default_rng(SEED), y2=yh, pb2=ph)
    fair = pd.read_csv(SEL / "fairness.csv").set_index("photos").loc["colour"]
    assert abs(np.quantile(full, 0.025) - fair.ci_lo) < 1e-12, "full-set interval differs from fairness.csv"
    bs = {n: both_sides(n, SEED + 10 + j) for j, n in enumerate(draws)}
    # mix effect at fixed size: size-only minus size+mix, both drawn independently
    rng = np.random.default_rng(SEED + 20)
    mix = []
    while len(mix) < n_boot:
        a = draws["size only"][rng.integers(n_draw)]
        b = draws["size + mix"][rng.integers(n_draw)]
        ja, jb = rng.integers(0, n_d, n_d), rng.integers(0, n_d, n_d)
        if len(set(yh[a[0]][ja])) != 5 or len(set(yh[b[0]][jb])) != 5:
            continue
        mix.append(balanced_accuracy_score(yh[a[0]][ja], a[1][ja]) - balanced_accuracy_score(yh[b[0]][jb], b[1][jb]))
    rows = [
        {"row": "drawings (parent 5-class, exp3 main figure)", "kind": "score", "n_aircraft": n_d,
         "class_counts": "|".join(f"{c} {int((yd == c).sum())}" for c in C5), "estimate": sd,
         "ci_lo": m_d.knn_ci_lo, "ci_hi": m_d.knn_ci_hi, "draw_lo": np.nan, "draw_hi": np.nan,
         "interval_method": "bootstrap over aircraft, 1000 (saved protocol result)", "chance_p95": m_d.knn_chance_p95},
        {"row": "photos, full set (hero image)", "kind": "score", "n_aircraft": len(yh),
         "class_counts": "|".join(f"{c} {int((yh == c).sum())}" for c in C5), "estimate": sh,
         "ci_lo": m_h.knn_ci_lo, "ci_hi": m_h.knn_ci_hi, "draw_lo": np.nan, "draw_hi": np.nan,
         "interval_method": "bootstrap over aircraft, 1000 (saved protocol result)", "chance_p95": m_h.knn_chance_p95}]
    for name, lab in (("size only", "photos, size only: subsets of 663 at the photos' own class mix"),
                      ("size + mix", "photos, size + mix: subsets of 663 at the drawings' class counts")):
        lo, hi = interval(bs[name][0])
        dlo, dhi = interval(score[name])
        rows.append({"row": lab, "kind": "score", "n_aircraft": n_d,
                     "class_counts": "|".join(f"{c} {counts[name][c]}" for c in C5),
                     "estimate": float(score[name].mean()), "ci_lo": lo, "ci_hi": hi, "draw_lo": dlo, "draw_hi": dhi,
                     "interval_method": f"{n_draw} draws x bootstrap over the draw's aircraft, {n_boot}; draw_lo/hi = "
                                        "2.5-97.5 % of the draw scores alone", "chance_p95": np.nan})
    rows.append({"row": "difference: photos full set minus drawings", "kind": "difference", "n_aircraft": np.nan,
                 "class_counts": "", "estimate": sh - sd, "ci_lo": float(np.quantile(full, 0.025)),
                 "ci_hi": float(np.quantile(full, 0.975)), "draw_lo": np.nan, "draw_hi": np.nan,
                 "interval_method": "independent bootstrap of the two sets, 1000 (reproduces setup_selection/fairness.csv)",
                 "chance_p95": np.nan})
    for name, lab in (("size only", "difference: photos size only minus drawings"),
                      ("size + mix", "difference: photos size + mix (matched) minus drawings")):
        lo, hi = interval(bs[name][1])
        rows.append({"row": lab, "kind": "difference", "n_aircraft": np.nan, "class_counts": "",
                     "estimate": float(score[name].mean() - sd), "ci_lo": lo, "ci_hi": hi,
                     "draw_lo": np.nan, "draw_hi": np.nan,
                     "interval_method": f"photos: {n_draw} draws x bootstrap over the draw's aircraft; drawings: "
                                        f"bootstrap over aircraft; {n_boot} paired draws of the two", "chance_p95": np.nan})
    lo, hi = interval(sh - score["size only"])
    rows.append({"row": "decomposition: pool size (full set minus size only)", "kind": "difference",
                 "n_aircraft": np.nan, "class_counts": "", "estimate": float(sh - score["size only"].mean()),
                 "ci_lo": lo, "ci_hi": hi, "draw_lo": np.nan, "draw_hi": np.nan,
                 "interval_method": "2.5-97.5 % over the size-only draws (the draws are subsets of the full set)",
                 "chance_p95": np.nan})
    lo, hi = interval(mix)
    rows.append({"row": "decomposition: class mix at fixed size (size only minus size + mix)", "kind": "difference",
                 "n_aircraft": np.nan, "class_counts": "",
                 "estimate": float(score["size only"].mean() - score["size + mix"].mean()),
                 "ci_lo": lo, "ci_hi": hi, "draw_lo": np.nan, "draw_hi": np.nan,
                 "interval_method": "independent draw x aircraft bootstrap of both, 1000", "chance_p95": np.nan})
    table = pd.DataFrame(rows)
    table.insert(0, "flag", "M3")
    # per class, with the class's share of the neighbour pool (F-B)
    pc = []
    rd = recall_score(yd, pdr, labels=C5, average=None, zero_division=0)
    rh = recall_score(yh, ph, labels=C5, average=None, zero_division=0)
    for j, c in enumerate(C5):
        nd_, nh_ = int((yd == c).sum()), int((yh == c).sum())
        kd, kh = int(((yd == c) & (pdr == c)).sum()), int(((yh == c) & (ph == c)).sum())
        row = {"flag": "M3", "class": c, "class_name": ER.CLASS5_NAME[c],
               "n_drawings": nd_, "pool_share_drawings": nd_ / n_d, "recall_drawings": rd[j],
               "recall_drawings_lo": wilson(kd, nd_)[0], "recall_drawings_hi": wilson(kd, nd_)[1],
               "n_photos": nh_, "pool_share_photos": nh_ / len(yh), "recall_photos": rh[j],
               "recall_photos_lo": wilson(kh, nh_)[0], "recall_photos_hi": wilson(kh, nh_)[1]}
        for name, tag in (("size only", "size_only"), ("size + mix", "matched")):
            row[f"n_{tag}"] = counts[name][c]
            row[f"pool_share_{tag}"] = counts[name][c] / n_d
            row[f"recall_{tag}"] = float(rec[name][:, j].mean())
            row[f"recall_{tag}_lo"], row[f"recall_{tag}_hi"] = interval(rec[name][:, j])
        pc.append(row)
    per_draw = pd.DataFrame([{"flag": "M3", "subset": n, "draw": d, "bal_acc": s}
                             for n, v in score.items() for d, s in enumerate(v)])
    return {"M3_pool_size_and_mix": table, "M3_per_class_pool_share": pd.DataFrame(pc), "M3_draws": per_draw}


# ═════════════════════════════════════════════════════════════════════════════
# M4: view erasure
# ═════════════════════════════════════════════════════════════════════════════
def erasure_setup(tag: str) -> Dict[str, Any]:
    """The erasure of scripts/view_erasure_check.py: the mean projection (the step that converged; INLP added no
    direction, VIEW_ERASURE.md), the aircraft vectors of the main figure, and the SVD the variance-matched
    control draws from (view_erasure.random_matched, computed once here and checked against it)."""
    Sp = sources(tag)[0]
    c = Sp.confound
    Xf, yv, mf = c["X"][KEY], c["y_conf"], c["makers"]
    X0 = Sp.rules[Sp.primary_rule][KEY]
    Q = VE.orthonormal(VE.mean_directions(Xf, yv))
    ve = pd.read_csv(VEDIR / "view_erasure.csv")
    nd = ve[(ve.tag == tag) & (ve.section == "erasure") & (ve.measure == "n_dirs_removed")].value.iloc[0]
    assert Q.shape[1] == int(nd), (Q.shape, nd)
    share = VE.variance_share(Xf, Q)["centred"]
    Xn = P.l2(Xf)
    _, s, Vt = np.linalg.svd(Xn - Xn.mean(axis=0), full_matrices=False)
    cum = np.cumsum(s ** 2) / (s ** 2).sum()
    rank = Q.shape[1]
    ms = np.arange(rank, len(cum) + 1)
    m = int(ms[np.argmin(np.abs(rank / ms * cum[ms - 1] - share))])

    def matched(seed: int) -> np.ndarray:
        R = np.linalg.qr(np.random.default_rng(seed).standard_normal((m, rank)))[0]
        return Vt[:m].T @ R

    ref = VE.random_matched(Xf, rank, share, seed=SEED)
    assert ref["m"] == m and np.allclose(ref["Q"], matched(SEED)), "variance-matched control differs"
    ex3 = Sp.extra["picks"][Sp.primary_rule].set_index("aircraft_id")
    view_ac = ex3.view4.reindex(Sp.aircraft.aircraft_id).to_numpy(dtype=object)
    return {"S": Sp, "Xf": Xf, "yv": yv, "mf": mf, "X0": X0, "Q": Q, "share": share, "m": m, "matched": matched,
            "XE": VE.project(X0, Q), "persp": view_ac == "Perspective", "view_ac": view_ac,
            "labels": {LAB12: Sp.y, LAB5: Sp.aircraft["label5"].to_numpy(dtype=object)}, "makers": Sp.makers}


def m4_erasure(n_ctrl: int = 200, n_split: int = 200) -> Dict[str, pd.DataFrame]:
    ve = pd.read_csv(VEDIR / "view_erasure.csv")
    # (a) the held-out check already saved, beside the in-sample ones
    rows = []
    for tag in (BASE, REG):
        v = ve[ve.tag == tag]
        for sec, unit in (("view probe (figures)", "figures, erasure fitted and tested on the same 1 541"),
                          ("view probe (aircraft main figures)", "663 main figures, erasure fitted on all figures"),
                          ("view probe, erasure fitted on other makers (figures)",
                           "figures of half of the makers; erasure fitted on the other half (GroupShuffleSplit, seed 42)")):
            b = v[(v.section == sec) & (v.variant == "original")].iloc[0]
            a = v[(v.section == sec) & (v.variant == "view erasure")].iloc[0]
            rows.append({"flag": "M4", "model": "registers" if tag == REG else "base", "check": sec, "unit": unit,
                         "n": int(b.n), "before": b.value, "after": a.value, "chance": b.chance,
                         "share_of_above_chance_removed": (b.value - a.value) / (b.value - b.chance),
                         "fit": "held out" if "other makers" in sec else "in sample"})
    t_ho = pd.DataFrame(rows)
    # (c) erased scores with intervals and chance
    arch = ve[ve.section.str.startswith("architecture") & (ve.measure == "bal_acc")].copy()
    arch = arch.assign(flag="M4", model=arch.tag.map({REG: "registers", BASE: "base"}),
                       label_set=arch.section.str.replace("architecture kNN-5, ", "", regex=False))
    t_sc = arch[["flag", "model", "label_set", "variant", "n", "value", "ci_lo", "ci_hi", "chance_p95"]].rename(
        columns={"value": "bal_acc", "variant": "vectors"})
    t_sc["note"] = "saved in view_erasure.csv; interval = bootstrap over aircraft (1000); chance = 95th percentile of 50 label shuffles"

    ctrl_rows, ctrl_sum, ho_rows = [], [], []
    for tag in (BASE, REG):
        E = erasure_setup(tag)
        mk = E["makers"]
        subsets = {"all": np.ones(len(mk), bool), "Perspective main figures only": E["persp"]}
        for sub, msk in subsets.items():
            for lab, y in E["labels"].items():
                y_, mk_ = y[msk], mk[msk]
                p0 = predict(E["X0"][msk], y_, mk_)
                pe = predict(E["XE"][msk], y_, mk_)
                d_er = bal(y_, pe) - bal(y_, p0)
                dm, dr = [], []
                for s in range(n_ctrl):
                    XC = VE.project(E["X0"][msk], E["matched"](s))
                    dm.append(bal(y_, predict(XC, y_, mk_)) - bal(y_, p0))
                    XR = VE.project(E["X0"][msk], VE.random_basis(E["X0"].shape[1], E["Q"].shape[1], s))
                    dr.append(bal(y_, predict(XR, y_, mk_)) - bal(y_, p0))
                    ctrl_rows.append({"flag": "M4", "model": "registers" if tag == REG else "base", "subset": sub,
                                      "label_set": lab, "seed": s, "diff_variance_matched": dm[-1],
                                      "diff_random_directions": dr[-1]})
                for name, v in (("variance-matched random directions", dm), ("random directions", dr)):
                    lo, hi = interval(v)
                    ctrl_sum.append({"flag": "M4", "model": "registers" if tag == REG else "base", "subset": sub,
                                     "label_set": lab, "n_aircraft": int(msk.sum()), "control": name,
                                     "draws": n_ctrl, "erasure_diff": d_er, "control_mean": float(np.mean(v)),
                                     "control_lo": lo, "control_hi": hi,
                                     "share_of_draws_as_costly_as_erasure": float(np.mean(np.asarray(v) <= d_er)),
                                     "seed42_control_diff": v[SEED] if n_ctrl > SEED else np.nan,
                                     "top_pcs_sampled": E["m"] if name.startswith("variance") else np.nan})
        # held out: the erasure fitted on the figures of one half of the makers, applied to the other half
        y12, y5 = E["labels"][LAB12], E["labels"][LAB5]
        Sm0 = E["X0"] @ E["X0"].T
        for s, a in enumerate(H.maker_splits(mk, n_split, SEED)):
            for fit_half, test in (("A", ~a), ("B", a)):
                fit_makers = set(mk[~test])
                fm = np.isin(E["mf"], list(fit_makers))
                tm = ~fm
                Qh = VE.orthonormal(VE.mean_directions(E["Xf"][fm], E["yv"][fm]))
                Qi = VE.orthonormal(VE.mean_directions(E["Xf"][tm], E["yv"][tm]))
                i = np.flatnonzero(test)
                vt = E["view_ac"][i]
                p_same = sum((vt == v).mean() ** 2 for v in set(vt))
                nnb = P.neighbours(None, mk[i], S=Sm0[np.ix_(i, i)])
                out = {"before": nnb}
                for nm, Qx in (("held out", Qh), ("in sample", Qi)):
                    Xt = VE.project(E["X0"][i], Qx)
                    out[nm] = P.neighbours(Xt, mk[i])
                row = {"flag": "M4", "model": "registers" if tag == REG else "base", "split": s,
                       "fitted_on_half": fit_half, "n_test_aircraft": len(i), "n_fit_figures": int(fm.sum()),
                       "view_share_chance": p_same}
                for nm, nn in out.items():
                    row[f"view_share_{nm.replace(' ', '_')}"] = float((vt[nn] == vt[:, None]).mean())
                    for lab, y in ((LAB12, y12), (LAB5, y5)):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            row[f"{'g1' if lab == LAB12 else 'parent'}_{nm.replace(' ', '_')}"] = bal(y[i], P.vote(nn, y[i]))
                ho_rows.append(row)
    ho = pd.DataFrame(ho_rows)
    hs = []
    for (model,), g in ho.groupby(["model"]):
        for lab in ("g1", "parent"):
            for nm in ("held_out", "in_sample"):
                d = g[f"{lab}_{nm}"] - g[f"{lab}_before"]
                lo, hi = interval(d)
                hs.append({"flag": "M4", "model": model, "label_set": LAB12 if lab == "g1" else LAB5,
                           "erasure_fit": nm.replace("_", " "), "halves": len(d), "mean_diff": float(d.mean()),
                           "halves_lo": lo, "halves_hi": hi, "share_of_halves_lower": float((d < 0).mean())})
        for nm in ("held_out", "in_sample"):
            d = g[f"view_share_{nm}"] - g["view_share_before"]
            lo, hi = interval(d)
            hs.append({"flag": "M4", "model": model, "label_set": "view: share of the 5 neighbours sharing the main-figure view",
                       "erasure_fit": nm.replace("_", " "), "halves": len(d), "mean_diff": float(d.mean()),
                       "halves_lo": lo, "halves_hi": hi, "share_of_halves_lower": float((d < 0).mean()),
                       "mean_before": float(g.view_share_before.mean()), "mean_after": float(g[f"view_share_{nm}"].mean()),
                       "mean_chance": float(g.view_share_chance.mean())})
    return {"M4a_heldout_view_probe": t_ho, "M4a_heldout_architecture_by_split": pd.DataFrame(hs),
            "M4a_heldout_architecture_halves": ho, "M4b_control_200_draws": pd.DataFrame(ctrl_sum),
            "M4b_control_draws": pd.DataFrame(ctrl_rows), "M4c_erased_scores": t_sc}


# ═════════════════════════════════════════════════════════════════════════════
# M6 / M9: per-class recall
# ═════════════════════════════════════════════════════════════════════════════
def m6_per_class(n_draw: int = 200, per_class: int = 20, n_probe_draws: int = 50) -> Dict[str, pd.DataFrame]:
    X, y, mk = label_sets()[LAB12]
    p = predict(X, y, mk)
    main = saved_main(LAB12)
    assert abs(bal(y, p) - main.knn_bal_acc) < 1e-12
    pp = probe_predictions(X, y, mk)
    assert abs(bal(y, pp) - main.probe_bal_acc) < 1e-9, (bal(y, pp), main.probe_bal_acc)
    n = len(y)
    rows = []
    for c in G1:
        m = y == c
        k, kp = int((p[m] == c).sum()), int((pp[m] == c).sum())
        lo, hi = wilson(k, m.sum())
        plo, phi = wilson(kp, m.sum())
        rows.append({"flag": "M9", "g1": c, "g1_display": disp(c), "g1_name": G1_NAME[c], "parent": ER.PARENT[c],
                     "n_aircraft": int(m.sum()), "pool_share": m.sum() / n, "under_20": bool(m.sum() < 20),
                     "knn_correct": k, "knn_recall": k / m.sum(), "knn_wilson_lo": lo, "knn_wilson_hi": hi,
                     "probe_correct": kp, "probe_recall": kp / m.sum(), "probe_wilson_lo": plo, "probe_wilson_hi": phi,
                     "predicted_as_this_class": int((p == c).sum())})
    full = pd.DataFrame(rows)
    wrong = p != y
    into = {c: int((p[wrong] == c).sum()) for c in G1}
    top2 = {"SLC", "TR"}
    w_rows = [{"flag": "M9", "measure": "wrong predictions (kNN-5, L24 CLS, 12 classes)", "value": int(wrong.sum()),
               "n": n, "note": ""},
              {"flag": "M9", "measure": "wrong predictions that name SLC or TP", "value": int(sum(into[c] for c in top2)),
               "n": int(wrong.sum()), "note": ""},
              {"flag": "M9", "measure": "share of wrong predictions that name SLC or TP",
               "value": sum(into[c] for c in top2) / wrong.sum(), "n": int(wrong.sum()),
               "note": "reference: SLC and TP are this share of the aircraft pool: "
                       f"{((y == 'SLC') | (y == 'TR')).mean():.3f}"},
              {"flag": "M9", "measure": "share of the aircraft pool that is SLC or TP", "value": float(((y == "SLC") | (y == "TR")).mean()),
               "n": n, "note": ""}]
    w_rows += [{"flag": "M9", "measure": f"wrong predictions that name {disp(c)}", "value": into[c],
                "n": int(wrong.sum()), "note": ""} for c in G1]
    # 5-class recall with Wilson (drawings and photos; review C7)
    r5 = []
    for lab in (LAB5, PHO5):
        X5, y5, m5 = label_sets()[lab]
        p5 = predict(X5, y5, m5)
        for c in C5:
            m = y5 == c
            k = int((p5[m] == c).sum())
            lo, hi = wilson(k, m.sum())
            r5.append({"flag": "M9/C7", "label_set": lab, "class": c, "class_name": ER.CLASS5_NAME[c],
                       "n_aircraft": int(m.sum()), "knn_correct": k, "knn_recall": k / m.sum(),
                       "wilson_lo": lo, "wilson_hi": hi})
    # M6: class-balanced subsamples, neighbours only inside the subsample, maker held out
    Sm = P.l2(X) @ P.l2(X).T
    idx = {c: np.flatnonzero(y == c) for c in G1}
    take = {c: min(len(idx[c]), per_class) for c in G1}
    rng = np.random.default_rng(SEED)
    rec, rec_sh, ba, ba_sh, prec = [], [], [], [], []
    for d in range(n_draw):
        i = np.concatenate([rng.choice(idx[c], take[c], replace=False) for c in G1])
        nn = P.neighbours(None, mk[i], S=Sm[np.ix_(i, i)])
        pi = P.vote(nn, y[i])
        ys = rng.permutation(y[i])
        ps = P.vote(nn, ys)
        rec.append(recall_score(y[i], pi, labels=G1, average=None, zero_division=0))
        rec_sh.append(recall_score(ys, ps, labels=G1, average=None, zero_division=0))
        ba.append(bal(y[i], pi))
        ba_sh.append(bal(ys, ps))
        if d < n_probe_draws:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pr = probe_predictions(X[i], y[i], mk[i])
            prec.append(recall_score(y[i], pr, labels=G1, average=None, zero_division=0))
    rec, rec_sh, prec = np.array(rec), np.array(rec_sh), np.array(prec) if prec else None
    brows = []
    for j, c in enumerate(G1):
        lo, hi = interval(rec[:, j])
        brow = {"flag": "M6", "g1": c, "g1_display": disp(c), "parent": ER.PARENT[c], "n_aircraft": len(idx[c]),
                "per_draw": take[c], "draws": n_draw, "knn_recall_mean": float(rec[:, j].mean()),
                "knn_recall_lo": lo, "knn_recall_hi": hi, "share_of_draws_at_zero": float((rec[:, j] == 0).mean()),
                "shuffled_recall_mean": float(rec_sh[:, j].mean()),
                "shuffled_recall_p95": float(np.quantile(rec_sh[:, j], 0.95)),
                "full_sample_knn_recall": float(full.set_index("g1").loc[c, "knn_recall"])}
        if prec is not None:
            plo, phi = interval(prec[:, j])
            brow.update({"probe_draws": len(prec), "probe_recall_mean": float(prec[:, j].mean()),
                         "probe_recall_lo": plo, "probe_recall_hi": phi})
        brows.append(brow)
    lo, hi = interval(ba)
    slo, shi = interval(ba_sh)
    brows.append({"flag": "M6", "g1": "balanced accuracy", "g1_display": "balanced accuracy", "parent": "",
                  "n_aircraft": n, "per_draw": sum(take.values()), "draws": n_draw,
                  "knn_recall_mean": float(np.mean(ba)), "knn_recall_lo": lo, "knn_recall_hi": hi,
                  "share_of_draws_at_zero": np.nan, "shuffled_recall_mean": float(np.mean(ba_sh)),
                  "shuffled_recall_p95": float(np.quantile(ba_sh, 0.95)), "full_sample_knn_recall": bal(y, p)})
    return {"M9_recall_12_wilson": full, "M9_wrong_predictions": pd.DataFrame(w_rows),
            "M9_recall_5class_wilson": pd.DataFrame(r5), "M6_balanced_recall": pd.DataFrame(brows)}


# ═════════════════════════════════════════════════════════════════════════════
# M11: count reconciliation
# ═════════════════════════════════════════════════════════════════════════════
def m11_counts() -> Dict[str, pd.DataFrame]:
    from .config_loader import load_config
    from .evtolnews_eval import load_links
    Sp, Sh = sources(REG)
    rows = []
    links = load_links(load_config())
    pc = pd.read_csv(EE / "parent_check.csv", keep_default_na=False)
    ranks = pd.read_csv(T_EVN_REG / "matching_ranks.csv")
    main_pat = pd.read_csv(ER.PAT / "2_embedding_extraction/selection/sets/main.csv", keep_default_na=False)
    ana = set(Sp.aircraft.aircraft_id)
    allk = Sh.extra["all"]
    at = pd.read_csv(ER.PAT / "0_labelling/outputs/tables/aircraft_table.csv", dtype=str, keep_default_na=False)
    at = at.set_index("aircraft_id")
    q = {im: set(ranks[(ranks.images == im) & (ranks.layer == 24) & (ranks.pooling == "cls") &
                       (ranks.direction == "patent->photo")]["query"]) for im in ("all images", "no line drawings")}
    for a in sorted(set(links.aircraft_uid)):
        if a in q["no line drawings"]:
            continue
        lk = links[links.aircraft_uid == a].iloc[0]
        imgs = allk[allk.aircraft_uid.isin(lk.slugs)]
        if a not in set(main_pat.aircraft_uid):
            why = (f"not in the patent analysis set: domain gate (uav_final = {at.loc[a, 'uav_final'] if a in at.index else '?'})"
                   if a in at.index else "not in the patent analysis set")
        elif a not in q["all images"]:
            why = "in the analysis set, but no page of its family in the photo gallery"
        else:
            why = (f"its directory page(s) carry only line drawings ({', '.join(imgs['style'])}), removed from the "
                   "no-line-drawing gallery")
        rows.append({"flag": "M11", "item": "99 linked aircraft vs 94 matching queries", "id": a, "family": lk.family,
                     "count_a": len(links), "count_b": len(q["no line drawings"]), "explanation": why})
    t_link = pd.DataFrame(rows)
    # 665 vs 663; 1 545 vs 1 541; 677 vs 665
    exc = pd.read_csv(ER.PAT / "2_embedding_extraction/view_state_experiments/excluded_aircraft.csv", keep_default_na=False)
    ft = Sp.extra["figure_table"]
    wv_all = ft[ft.scope == "whole_vehicle"]
    wv = wv_all[wv_all.aircraft_id.isin(ana)]
    extra_fig = wv_all[~wv_all.aircraft_id.isin(ana)]
    mb = pd.read_csv(ATT / BASE / "patent_meta.csv")
    mr = pd.read_csv(ATT / REG / "patent_meta.csv")
    only_b = mb[~mb.figure_uid.isin(set(mr.figure_uid))]
    gal = ranks[(ranks.images == "no line drawings") & (ranks.layer == 24) & (ranks.pooling == "cls") &
                (ranks.direction == "photo->patent")].gallery.iloc[0]
    rec = []
    for r in exc.itertuples():
        rec.append({"flag": "M11", "item": "665 (patent main set: attention run, photo-to-patent gallery) vs 663 (analysis)",
                    "id": r.aircraft_id, "count_a": len(set(main_pat.aircraft_uid)), "count_b": len(ana),
                    "explanation": f"marked unclassifiable, excluded from the analysis: {r.reason}"})
    rec.append({"flag": "M11", "item": "photo-to-patent gallery size", "id": "", "count_a": int(gal),
                "count_b": len(ana), "explanation": "the gallery is the patent main set, which still holds the 2 "
                                                    "unclassifiable aircraft; their figures are never a positive"})
    for r in extra_fig.itertuples():
        rec.append({"flag": "M11", "item": "1 545 whole-aircraft figures (thickening source manifest, figure table) vs 1 541 (analysis)",
                    "id": r.fig_id, "count_a": len(wv_all), "count_b": len(wv),
                    "explanation": f"whole-aircraft figure of {r.aircraft_id}, one of the 2 unclassifiable aircraft"})
    rec.append({"flag": "M11", "item": "part figures", "id": "", "count_a": int((ft.scope == "part").sum()),
                "count_b": 0, "explanation": "part / detail figures leave before the whole-aircraft count; not the source of the 4"})
    only_r = mr[~mr.figure_uid.isin(set(mb.figure_uid))]
    reg_ac = set(mr.aircraft_uid)
    for r in only_b.itertuples():
        rec.append({"flag": "M11", "item": "base attention 677 images vs registers 665 (image only in the base run)",
                    "id": r.figure_uid, "count_a": len(mb), "count_b": len(mr),
                    "explanation": ("the aircraft's main figure changed between the two runs"
                                    if r.aircraft_uid in reg_ac else
                                    "aircraft not in the current patent main set (the base attention ran on an earlier "
                                    "main set, before later domain-gate and duplicate rulings)")})
    for r in only_r.itertuples():
        rec.append({"flag": "M11", "item": "base attention 677 images vs registers 665 (image only in the registers run)",
                    "id": r.figure_uid, "count_a": len(mb), "count_b": len(mr),
                    "explanation": "current main figure of an aircraft whose main figure changed after the base run"})
    rec.append({"flag": "M11", "item": "images in both attention runs (base and registers recomputed on these)", "id": "",
                "count_a": len(mb), "count_b": int(mb.figure_uid.isin(set(mr.figure_uid)).sum()),
                "explanation": "see S9_attention_extras.csv, rows image_set = 'common'"})
    t_cnt = pd.DataFrame(rec)
    # ARI like for like
    ari = []
    for lab, name, k in ((LAB12, "drawings, k-means 12 clusters vs 12 G1 types", 12),
                         (LAB5, "drawings, k-means 5 clusters vs 5 parent classes", 5),
                         (PHO5, "photos, k-means 5 clusters vs 5 directory classes", 5)):
        for tag in (REG, BASE):
            m = saved_main(lab, tag)
            ari.append({"flag": "M11", "comparison": name, "model": "registers" if tag == REG else "base",
                        "clusters": k, "ari": m.t5_ari, "nmi": m.t5_nmi})
    ari = pd.DataFrame(ari)
    r = ari[ari.model == "registers"].set_index("comparison").ari
    ratios = pd.DataFrame([
        {"flag": "M11", "ratio": "photos 5v5 / drawings 12v12 (as printed in the document)",
         "value": r.iloc[2] / r.iloc[0]},
        {"flag": "M11", "ratio": "photos 5v5 / drawings 5v5 (like for like)", "value": r.iloc[2] / r.iloc[1]}])
    return {"M11_linked_99_vs_94": t_link, "M11_counts": t_cnt, "M11_ari_like_for_like": ari, "M11_ari_ratios": ratios}


# ═════════════════════════════════════════════════════════════════════════════
# S2: the probe with interval and chance
# ═════════════════════════════════════════════════════════════════════════════
def s2_probe(labels: Sequence[str] = (LAB12, LAB5, PHO5), n_shuffle: int = 50, fold_seeds=(0, 1, 2, 3, 4),
             n_boot: int = N_BOOT) -> Dict[str, pd.DataFrame]:
    rows, shuf_rows = [], []
    for lab in labels:
        X, y, mk = label_sets()[lab]
        main = saved_main(lab)
        pp = probe_predictions(X, y, mk)
        assert abs(bal(y, pp) - main.probe_bal_acc) < 1e-9
        pk = predict(X, y, mk)
        a = boot_scores(y, {"probe": pp, "knn": pk}, None, n_boot)
        m = boot_scores(y, {"probe": pp, "knn": pk}, mk, n_boot)
        rng = np.random.default_rng(SEED)
        sh = []
        for s in range(n_shuffle):
            ys = rng.permutation(y)
            sh.append(P.probe(X, ys, mk)["bal_acc"])
            shuf_rows.append({"flag": "S2", "label_set": lab, "shuffle": s, "probe_bal_acc": sh[-1]})
        fs = [P.probe(X, y, mk, seed=f)["bal_acc"] for f in fold_seeds]
        rows.append({"flag": "S2", "label_set": lab, "n_aircraft": len(y), "n_makers": len(set(mk)),
                     "probe_bal_acc": bal(y, pp),
                     "ci_lo_aircraft": interval(a["probe"])[0], "ci_hi_aircraft": interval(a["probe"])[1],
                     "ci_lo_maker": interval(m["probe"])[0], "ci_hi_maker": interval(m["probe"])[1],
                     "fold_seed_min": min(fs), "fold_seed_max": max(fs), "fold_seeds": "|".join(map(str, fold_seeds)),
                     "shuffle_mean": float(np.mean(sh)), "shuffle_p95": float(np.quantile(sh, 0.95)),
                     "shuffles": n_shuffle, "knn_bal_acc": bal(y, pk),
                     "probe_minus_knn": bal(y, pp) - bal(y, pk),
                     "diff_ci_lo_aircraft": interval(a["probe"] - a["knn"])[0],
                     "diff_ci_hi_aircraft": interval(a["probe"] - a["knn"])[1],
                     "diff_ci_lo_maker": interval(m["probe"] - m["knn"])[0],
                     "diff_ci_hi_maker": interval(m["probe"] - m["knn"])[1],
                     "regularisation": "LogisticRegression, L2 penalty, C = 1.0 (sklearn default strength), lbfgs, "
                                       "class_weight balanced, max_iter 4000, on l2-normalised vectors; 5 folds "
                                       "StratifiedGroupKFold by maker, shuffle, random_state 42; C not tuned",
                     "interval_note": "bootstrap of the out-of-fold predictions (fits held fixed); fold-seed range = "
                                      "the probe refitted with other fold assignments"})
    return {"S2_probe": pd.DataFrame(rows), "S2_probe_shuffles": pd.DataFrame(shuf_rows)}


# ═════════════════════════════════════════════════════════════════════════════
# S3: k sensitivity and the image rules
# ═════════════════════════════════════════════════════════════════════════════
def s3_k_and_rules(ks=(3, 5, 10)) -> Dict[str, pd.DataFrame]:
    rows = []
    for lab, (X, y, mk) in label_sets().items():
        preds = {}
        for k in ks:
            r = P.knn(X, y, mk, k=k)
            preds[k] = r["pred"]
            rows.append({"flag": "S3", "label_set": lab, "k": k, "bal_acc": r["bal_acc"], "ci_lo": r["ci_lo"],
                         "ci_hi": r["ci_hi"], "chance_mean": r["chance_mean"], "chance_p95": r["chance_p95"]})
        for k in ks:
            if k == P.K:
                continue
            d = VE.paired(y, preds[P.K], preds[k])
            rows.append({"flag": "S3", "label_set": lab, "k": f"{k} minus 5", "bal_acc": d["diff"],
                         "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "chance_mean": np.nan, "chance_p95": np.nan})
    t_k = pd.DataFrame(rows)
    Sp, Sh = sources(REG)
    rr, sel_rows = [], []
    for src, S, labs, ref in (("patents", Sp, {LAB12: Sp.y, LAB5: Sp.aircraft["label5"].to_numpy(dtype=object)},
                               "exp3 main figure"),
                              ("photos", Sh, {PHO5: Sh.y}, "hero image")):
        mk = S.makers
        cands = {r: X[KEY] for r, X in S.rules.items()}
        sc = H.half_scores(cands, labs, mk)
        sel = H.select(sc)
        pc, pl = H.summarise(sc, sel)
        piv = sc.pivot_table(index=["split", "half", "label"], columns="candidate", values="score").reset_index()
        for lab, y in labs.items():
            p_ref = predict(cands[ref], y, mk)
            for r, X in cands.items():
                if r == ref:
                    continue
                p = predict(X, y, mk)
                a = VE.paired(y, p_ref, p)
                m = paired_boot(y, p_ref, p, mk)
                dh = (piv[piv.label == lab][r] - piv[piv.label == lab][ref])
                rr.append({"flag": "S3", "source": src, "label_set": lab, "contrast": f"{r} minus {ref}",
                           "score_ref": bal(y, p_ref), "score_rule": bal(y, p), "diff": a["diff"],
                           "ci_lo_aircraft": a["ci_lo"], "ci_hi_aircraft": a["ci_hi"],
                           "ci_lo_maker": m["ci_lo"], "ci_hi_maker": m["ci_hi"],
                           "halves_mean_diff": float(dh.mean()),
                           "share_of_halves_same_sign": float((np.sign(dh) == np.sign(a["diff"])).mean()),
                           "agree": a["agree"]})
        sel_rows.append(pc.assign(flag="S3", source=src))
    return {"S3_k_sensitivity": t_k, "S3_image_rules_paired": pd.DataFrame(rr),
            "S3_image_rules_split_half": pd.concat(sel_rows, ignore_index=True)}


# ═════════════════════════════════════════════════════════════════════════════
# S4: every paired contrast, bootstrap by maker, per-half consistency, Holm
# ═════════════════════════════════════════════════════════════════════════════
def _half_from_saved(scores: pd.DataFrame, lab: str, a: str, b: str) -> pd.Series:
    t = scores[scores.label == lab].pivot_table(index=["split", "half"], columns="candidate", values="score")
    return t[b] - t[a]


def contrasts(include_gpu: bool = True) -> List[Dict[str, Any]]:
    """Every paired contrast of the document: (name, group, label set, y, makers, p0, p1, per-half diffs, saved CI)."""
    from . import preprocess_variants_eval as PVE
    out: List[Dict[str, Any]] = []
    half_a = pd.read_csv(SEL / "s1a_half_scores.csv")
    half_b = pd.read_csv(SEL / "s1b_scores.csv")
    rp = pd.read_csv(REGCHK / "registers_paired.csv").set_index(["source", "label"])
    for lab in (LAB12, LAB5, PHO5):
        Xr, y, mk = label_sets(REG)[lab]
        Xb, _, _ = label_sets(BASE)[lab]
        s = rp.loc[("patents" if lab != PHO5 else "photos",
                    {LAB12: "G1 12-class", LAB5: "parent 5-class", PHO5: "5-class"}[lab])]
        out.append({"name": f"registers minus base, {LAB_SHORT[lab]}", "group": "model", "label_set": lab,
                    "y": y, "makers": mk, "p0": predict(Xb, y, mk), "p1": predict(Xr, y, mk),
                    "halves": _half_from_saved(half_a, lab, "base L24 CLS", "registers L24 CLS"),
                    "halves_source": "setup_selection/s1a_half_scores.csv",
                    "saved": (s["diff"], s.ci_lo, s.ci_hi), "saved_source": "registers_check/registers_paired.csv"})
    D = PVE.load()
    sp = pd.read_csv(SEL / "s2s3_paired.csv").set_index(["label", "comparison"])
    for lab in (LAB12, LAB5):
        y, mk = np.asarray(D["patents"]["labels"][lab], dtype=object), D["patents"]["makers"]
        p0 = predict(D["patents"]["conds"]["plain"][KEY], y, mk)
        for v in ("thick1", "thick2"):
            s = sp.loc[(lab, f"{v} - plain")]
            out.append({"name": f"{v} minus plain, {LAB_SHORT[lab]}", "group": "drawing preparation", "label_set": lab,
                        "y": y, "makers": mk, "p0": p0, "p1": predict(D["patents"]["conds"][v][KEY], y, mk),
                        "halves": _half_from_saved(half_b, lab, "plain", v), "halves_source": "setup_selection/s1b_scores.csv",
                        "saved": (s["diff"], s.ci_lo, s.ci_hi), "saved_source": "setup_selection/s2s3_paired.csv"})
    # the GPU run's drawing controls (plain grey, contrast only), when their embeddings exist
    ctl_csv = SEL / "s2_controls_paired.csv"
    Sp_ = sources(REG)[0]
    figs = Sp_.extra["picks"][Sp_.primary_rule].groupby("aircraft_id")["fig_id"].apply(list)
    figs = [figs[a] for a in Sp_.aircraft.aircraft_id]
    for v in ("grey", "contrast"):
        d = PVE.variant_dir(ER.PAT, v)
        if not include_gpu or not (d / "metadata.parquet").exists():
            continue
        idx, arrays = ER.load_matrices(d)
        Xv = ER.rule_matrix(idx, {KEY: arrays[KEY]}, figs)[KEY]
        sv = pd.read_csv(ctl_csv).set_index(["label", "comparison"]) if ctl_csv.exists() else None
        for lab in (LAB12, LAB5):
            y, mk = np.asarray(D["patents"]["labels"][lab], dtype=object), D["patents"]["makers"]
            X0 = D["patents"]["conds"]["plain"][KEY]
            sc = H.half_scores({"plain": X0, v: Xv}, {lab: y}, mk)
            sd = sv.loc[(lab, f"{v} - plain")] if sv is not None and (lab, f"{v} - plain") in sv.index else None
            out.append({"name": f"{'plain grey' if v == 'grey' else 'contrast only'} minus plain, {LAB_SHORT[lab]}",
                        "group": "drawing preparation", "label_set": lab, "y": y, "makers": mk,
                        "p0": predict(X0, y, mk), "p1": predict(Xv, y, mk),
                        "halves": _half_from_saved(sc, lab, "plain", v), "halves_source": "computed here",
                        "saved": (sd["diff"], sd.ci_lo, sd.ci_hi) if sd is not None else (np.nan,) * 3,
                        "saved_source": "setup_selection/s2_controls_paired.csv (GPU run)" if sd is not None else ""})
    y, mk = np.asarray(D["photos"]["labels"][PHO5], dtype=object), D["photos"]["makers"]
    s = sp.loc[(PHO5, "grey - colour")]
    out.append({"name": "grey minus colour, photos", "group": "photo preparation", "label_set": PHO5, "y": y,
                "makers": mk, "p0": predict(D["photos"]["conds"]["colour"][KEY], y, mk),
                "p1": predict(D["photos"]["conds"]["grey"][KEY], y, mk),
                "halves": _half_from_saved(half_b, PHO5, "colour", "grey"), "halves_source": "setup_selection/s1b_scores.csv",
                "saved": (s["diff"], s.ci_lo, s.ci_hi), "saved_source": "setup_selection/s2s3_paired.csv"})
    # crop: base (saved), and registers when the GPU run has written its embeddings
    Shb = sources(BASE)[1]
    crop_root = ER.EVN / "2_embedding_extraction/crop_experiment/embeddings"
    cd = pd.read_csv(CROP / "knn_paired_difference.csv").set_index("matrix").loc["L24 CLS"]
    cdr_f = CROP / "knn_paired_difference_registers.csv"
    cdr = pd.read_csv(cdr_f).set_index("matrix").loc["L24 CLS"] if cdr_f.exists() else None
    for tag, model in ((BASE, "base"), (REG, "registers")):
        d = crop_root / tag
        if not (d / "metadata.parquet").exists() or (tag == REG and not include_gpu):
            continue
        S = sources(tag)[1]
        idx, arrays = ER.load_matrices(d)
        if not all(h in idx for h in S.aircraft.hero):
            continue
        Xc = ER.rule_matrix(idx, {KEY: arrays[KEY]}, [[h] for h in S.aircraft.hero])[KEY]
        X0 = S.rules[S.primary_rule][KEY]
        y, mk = S.y, S.makers
        sc = H.half_scores({"original": X0, "cropped": Xc}, {PHO5: y}, mk)
        out.append({"name": f"cropped minus original, photos ({model})", "group": "photo background",
                    "label_set": PHO5, "y": y, "makers": mk, "p0": predict(X0, y, mk), "p1": predict(Xc, y, mk),
                    "halves": _half_from_saved(sc, PHO5, "original", "cropped"), "halves_source": "computed here",
                    "saved": ((cd.diff_cropped_minus_original, cd.ci_lo, cd.ci_hi) if tag == BASE else
                              ((cdr.diff_cropped_minus_original, cdr.ci_lo, cdr.ci_hi) if cdr is not None else (np.nan,) * 3)),
                    "saved_source": ("crop_experiment/knn_paired_difference.csv" if tag == BASE else
                                     "crop_experiment/knn_paired_difference_registers.csv (GPU run)" if cdr is not None else "")})
    # view erasure and its control, all aircraft and Perspective only
    ve = pd.read_csv(VEDIR / "view_erasure.csv")
    E = erasure_setup(REG)
    XC = VE.project(E["X0"], E["matched"](SEED))
    for sub, msk in (("all", np.ones(len(E["makers"]), bool)), ("Perspective only", E["persp"])):
        for lab, y in E["labels"].items():
            y_, mk_ = y[msk], E["makers"][msk]
            X0, XE, XCm = E["X0"][msk], E["XE"][msk], XC[msk]
            sc = H.half_scores({"original": X0, "erased": XE, "control": XCm}, {lab: y_}, mk_)
            sec = f"architecture kNN-5, {'G1 12-class' if lab == LAB12 else 'parent 5-class'}" + \
                  (", Perspective main figures only" if sub != "all" else "")
            for nm, X1, var in (("view erased", XE, "view erasure - original"),
                                ("variance-matched control (seed 42)", XCm, "variance-matched random - original")):
                s = ve[(ve.tag == REG) & (ve.section == sec) & (ve.variant == var)].iloc[0]
                out.append({"name": f"{nm} minus original, {LAB_SHORT[lab]}" + (", Perspective only" if sub != "all" else ""),
                            "group": "view erasure" if nm == "view erased" else "erasure control",
                            "label_set": lab, "y": y_, "makers": mk_, "p0": predict(X0, y_, mk_),
                            "p1": predict(X1, y_, mk_),
                            "halves": _half_from_saved(sc, lab, "original", "erased" if nm == "view erased" else "control"),
                            "halves_source": "computed here (erasure fitted on all figures)",
                            "saved": (s.value, s.ci_lo, s.ci_hi), "saved_source": "view_erasure/view_erasure.csv"})
    # image rules (12 classes)
    Sp = sources(REG)[0]
    y, mk = Sp.y, Sp.makers
    cands = {r: X[KEY] for r, X in Sp.rules.items()}
    sc = H.half_scores(cands, {LAB12: y}, mk)
    br = saved_results("pat_reg")["ev"]["by_rule"]
    p3 = predict(cands["exp3 main figure"], y, mk)
    for r in ("exp1 view-first", "exp2 state-first", "exp4 view average"):
        out.append({"name": f"{r.split()[0]} minus exp3, drawings, 12 G1 classes", "group": "image rule",
                    "label_set": LAB12, "y": y, "makers": mk, "p0": p3, "p1": predict(cands[r], y, mk),
                    "halves": _half_from_saved(sc, LAB12, "exp3 main figure", r), "halves_source": "computed here",
                    "saved": (np.nan,) * 3, "saved_source": "(no paired interval in the document)"})
    return out


def s4_paired() -> Dict[str, pd.DataFrame]:
    rows = []
    for c in contrasts():
        y = np.asarray(c["y"], dtype=object)
        a = VE.paired(y, c["p0"], c["p1"])
        m = paired_boot(y, c["p0"], c["p1"], c["makers"])
        h = c["halves"]
        rows.append({"flag": "S4", "contrast": c["name"], "group": c["group"], "label_set": c["label_set"],
                     "n_aircraft": len(y), "n_makers": len(set(c["makers"])), "diff": a["diff"],
                     "ci_lo_aircraft": a["ci_lo"], "ci_hi_aircraft": a["ci_hi"],
                     "ci_lo_maker": m["ci_lo"], "ci_hi_maker": m["ci_hi"], "p_maker_two_sided": m["p_two_sided"],
                     "halves": len(h), "halves_mean_diff": float(h.mean()),
                     "share_of_halves_same_sign": float((np.sign(h) == np.sign(a["diff"])).mean()),
                     "share_of_halves_positive": float((h > 0).mean()), "halves_source": c["halves_source"],
                     "saved_diff": c["saved"][0], "saved_ci_lo": c["saved"][1], "saved_ci_hi": c["saved"][2],
                     "saved_source": c["saved_source"], "agree": a["agree"]})
    t = pd.DataFrame(rows)
    # Holm over the family of contrasts the document reads as tests (maker-level p)
    order = t.p_maker_two_sided.sort_values().index
    m, run = len(t), 0.0
    holm = pd.Series(index=t.index, dtype=float)
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * t.loc[i, "p_maker_two_sided"]))
        holm[i] = run
    t["p_maker_holm"] = holm
    t["note"] = ("aircraft CI = paired bootstrap over aircraft (view_erasure.paired, seed 42); maker CI = the same "
                 "with whole makers resampled; predictions held fixed in both; halves = 200 maker splits x 2, "
                 "neighbours inside the half; Holm over all rows of this table")
    return {"S4_paired_contrasts": t}


# ═════════════════════════════════════════════════════════════════════════════
# S6: matching intervals and the status of the linked page
# ═════════════════════════════════════════════════════════════════════════════
def s6_matching(n_boot: int = N_BOOT) -> Dict[str, pd.DataFrame]:
    from .config_loader import load_config
    from .evtolnews_eval import load_links
    Sh = sources(REG)[1]
    status = Sh.aircraft.set_index("aircraft_id").status
    links = load_links(load_config())
    fam_slugs: Dict[str, set] = {}
    for r in links.itertuples():
        fam_slugs.setdefault(r.family, set()).update(r.slugs)
    allk = Sh.extra["all"]
    gallery = set(allk[allk["style"] != "drawing"].aircraft_uid)
    rows, per_q = [], []
    for model, f, tag in (("registers", T_EVN_REG / "matching_ranks.csv", REG),
                          ("base", EE / "matching_ranks.csv", BASE)):
        r = pd.read_csv(f)
        r = r[(r.embedding == tag) & (r.images == "no line drawings") & (r.layer == 24) & (r.pooling == "cls")]
        mt = pd.read_csv(f.parent / "matching.csv")
        mt = mt[(mt.embedding == tag) & (mt.images == "no line drawings") & (mt.layer == 24) & (mt.pooling == "cls")]
        def fam_status(fam: str) -> str:
            st = {status.get(s, "") for s in fam_slugs[fam] & gallery}
            return "built" if "built" in st else ("concept" if "concept" in st else "unknown")

        for direction, g in r.groupby("direction"):
            g = g.copy()
            g["page_status"] = [fam_status(fm) if direction == "patent->photo" else status.get(qq, "")
                                for fm, qq in zip(g.family, g["query"])]
            g["page_status"] = g.page_status.replace({"": "unknown"})
            if model == "registers":
                per_q.append(g.assign(flag="S6", model=model))
            rnd = mt[mt.direction == direction].iloc[0]
            for st, gg in [("all", g)] + list(g.groupby("page_status")):
                rk = gg["rank"].to_numpy()
                rng = np.random.default_rng(SEED)
                bs = np.array([[(rk[i] <= k).mean() for k in (10, 50)]
                               for i in (rng.integers(0, len(rk), len(rk)) for _ in range(n_boot))])
                row = {"flag": "S6", "model": model, "direction": direction.replace("->", " to "),
                       "page_status": st, "queries": len(rk), "gallery": int(gg.gallery.iloc[0]),
                       "median_rank": float(np.median(rk))}
                for j, k in enumerate((10, 50)):
                    hit = int((rk <= k).sum())
                    lo, hi = interval(bs[:, j])
                    wl, wh = wilson(hit, len(rk))
                    row.update({f"top{k}": hit / len(rk), f"top{k}_boot_lo": lo, f"top{k}_boot_hi": hi,
                                f"top{k}_wilson_lo": wl, f"top{k}_wilson_hi": wh,
                                f"random_top{k}": float(rnd[f"random_R@{k}"]) if st == "all" else np.nan})
                rows.append(row)
    t = pd.DataFrame(rows)
    t["note"] = ("bootstrap over the queries (1000, seed 42) and Wilson; page_status: patent to photo = built if any "
                 "of the family's gallery pages is built, else concept if any is concept; photo to patent = the "
                 "query page's own status; random = expected share under a random ranking (matching.csv)")
    return {"S6_matching_intervals": t,
            "S6_matching_queries": pd.concat(per_q, ignore_index=True)[
                ["flag", "model", "direction", "query", "family", "rank", "n_positive", "gallery", "page_status"]]}


# ═════════════════════════════════════════════════════════════════════════════
# S7 / S8: the split-half rule
# ═════════════════════════════════════════════════════════════════════════════
def _grid_s1a() -> Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]]:
    """The S1(a) candidates of scripts/heldout_selection.py: {source: (candidates, labels, makers)}."""
    out: Dict[str, Tuple[Dict, Dict, np.ndarray]] = {}
    for model, tag in (("registers", REG), ("base", BASE)):
        Sp, Sh = sources(tag)
        for src, S, labs in (("patents", Sp, {LAB12: Sp.y, LAB5: Sp.aircraft["label5"].to_numpy(dtype=object)}),
                             ("photos", Sh, {PHO5: Sh.y})):
            if src not in out:
                out[src] = ({}, labs, S.makers)
            out[src][0].update({f"{model} {P.mname(k)}": S.rules[S.primary_rule][k] for k in P.MATRICES})
    return out


def _run_rule(grid, seed: int, pools: Dict[str, List[str] | None]) -> Dict[str, pd.DataFrame]:
    scores = pd.concat([H.half_scores(c, l, m, seed=seed) for c, l, m in grid.values()], ignore_index=True)
    pcs, pls, sels = [], [], []
    for pool, subset in pools.items():
        sel = H.select(scores, subset)
        pc, pl = H.summarise(scores, sel, subset)
        pcs.append(pc.assign(pool=pool)), pls.append(pl.assign(pool=pool)), sels.append(sel.assign(pool=pool))
    return {"scores": scores, "per_c": pd.concat(pcs, ignore_index=True), "per_l": pd.concat(pls, ignore_index=True),
            "sel": pd.concat(sels, ignore_index=True)}


def s78_selection(seeds=(42, 7, 123)) -> Dict[str, pd.DataFrame]:
    from . import preprocess_variants_eval as PVE
    a_c = pd.read_csv(SEL / "s1a_candidates.csv")
    a_l = pd.read_csv(SEL / "s1a_labels.csv")
    a_h = pd.read_csv(SEL / "s1a_half_scores.csv")
    b_c, b_l = pd.read_csv(SEL / "s1b_candidates.csv"), pd.read_csv(SEL / "s1b_labels.csv")
    out: Dict[str, pd.DataFrame] = {}
    # (a) each source's own pick
    own = []
    for pool in ("registers only", "all 12"):
        t = a_c[a_c["pool"] == pool].assign(source=lambda d: d.label.map(SOURCE_OF_LABEL))
        for src, g in t.groupby("source"):
            f = g.groupby("candidate", sort=False)[["selection_freq", "fixed_heldout_mean"]].mean() \
                .sort_values(["selection_freq", "fixed_heldout_mean"], ascending=False)
            for rank, (c, r) in enumerate(f.head(3).iterrows()):
                own.append({"flag": "S7", "choice": "layer and token" if pool == "registers only" else "model x layer x token",
                            "pool": pool, "source": src, "rank": rank + 1, "candidate": c,
                            "mean_selection_freq": r.selection_freq, "mean_fixed_heldout": r.fixed_heldout_mean})
    for src, g in b_c.assign(source=lambda d: d.label.map(SOURCE_OF_LABEL)).groupby("source"):
        f = g.groupby("candidate", sort=False)[["selection_freq", "fixed_heldout_mean"]].mean() \
            .sort_values(["selection_freq", "fixed_heldout_mean"], ascending=False)
        for rank, (c, r) in enumerate(f.iterrows()):
            own.append({"flag": "S7", "choice": "preparation", "pool": "preparations", "source": src, "rank": rank + 1,
                        "candidate": c, "mean_selection_freq": r.selection_freq, "mean_fixed_heldout": r.fixed_heldout_mean})
    out["S7_own_pick_per_source"] = pd.DataFrame(own)
    # L24 vs L22 CLS held out, each label set (equivalence reading)
    eq = []
    for lab in (LAB12, LAB5, PHO5):
        d = _half_from_saved(a_h, lab, "registers L22 CLS", "registers L24 CLS")
        eq.append({"flag": "S7", "label_set": lab, "contrast": "registers L24 CLS minus L22 CLS (held out, same halves)",
                   "halves": len(d), "mean_diff": float(d.mean()), "halves_lo": interval(d)[0], "halves_hi": interval(d)[1],
                   "share_of_halves_positive": float((d > 0).mean())})
    b_s = pd.read_csv(SEL / "s1b_scores.csv")
    for lab, a, b in ((LAB12, "plain", "thick1"), (LAB12, "plain", "thick2"), (LAB5, "plain", "thick1"),
                      (LAB5, "plain", "thick2"), (PHO5, "colour", "grey")):
        d = _half_from_saved(b_s, lab, a, b)
        eq.append({"flag": "S7", "label_set": lab, "contrast": f"{b} minus {a} (held out, same halves)",
                   "halves": len(d), "mean_diff": float(d.mean()), "halves_lo": interval(d)[0], "halves_hi": interval(d)[1],
                   "share_of_halves_positive": float((d > 0).mean())})
    out["S7_heldout_differences"] = pd.DataFrame(eq)
    # (b) halves missing a class; Monte Carlo SE
    Sp, Sh = sources(REG)
    miss = []
    for src, S, labs in (("patents", Sp, {LAB12: Sp.y, LAB5: Sp.aircraft["label5"].to_numpy(dtype=object)}),
                         ("photos", Sh, {PHO5: Sh.y})):
        for s, a in enumerate(H.maker_splits(S.makers, H.R_SPLITS, SEED)):
            for half, m in (("A", a), ("B", ~a)):
                for lab, y in labs.items():
                    gone = sorted(set(y) - set(y[m]))
                    miss.append({"flag": "S8", "label_set": lab, "split": s, "half": half, "n_aircraft": int(m.sum()),
                                 "classes_present": len(set(y[m])), "classes_missing": "|".join(disp(c) for c in gone)})
    miss = pd.DataFrame(miss)
    ms = []
    for lab, g in miss.groupby("label_set", sort=False):
        per = pd.Series([c for v in g.classes_missing if v for c in v.split("|")]).value_counts()
        ms.append({"flag": "S8", "label_set": lab, "halves": len(g), "halves_missing_any_class": int((g.classes_missing != "").sum()),
                   "share": float((g.classes_missing != "").mean()), "min_aircraft_in_half": int(g.n_aircraft.min()),
                   "max_aircraft_in_half": int(g.n_aircraft.max()),
                   "halves_missing_each_class": "; ".join(f"{c} {n}" for c, n in per.items())})
    sel = pd.read_csv(SEL / "s1a_selections.csv")
    for pool in ("registers only", "all 12"):
        for lab in (LAB12, LAB5, PHO5):
            g = sel[(sel["pool"] == pool) & (sel.label == lab)]
            w = (g.chosen == "registers L24 CLS").astype(float)
            per_split = w.groupby(g.split).mean()
            p = w.mean()
            ms.append({"flag": "S8", "label_set": lab, "halves": len(g), "pool": pool,
                       "l24_cls_selection_freq": p, "mc_se_naive": float(np.sqrt(p * (1 - p) / len(g))),
                       "mc_se_split_clustered": float(per_split.std(ddof=1) / np.sqrt(len(per_split))),
                       "both_halves_of_a_split_agree": float(g.groupby("split").chosen.nunique().eq(1).mean())})
    out["S8_halves_missing_classes"] = pd.DataFrame(ms)
    # (c) seeds
    grid = _grid_s1a()
    reg = [f"registers {P.mname(k)}" for k in P.MATRICES]
    D = PVE.load()
    seed_rows = []
    for seed in seeds:
        R = _run_rule(grid, seed, {"all 12": None, "registers only": reg})
        pc = R["per_c"]
        win, dec = H.decide(pc[pc["pool"] == "registers only"], SOURCE_OF_LABEL)
        if seed == SEED:   # the saved run, reproduced
            chk = pc.merge(a_c, on=["label", "candidate", "pool"], suffixes=("", "_saved"))
            assert np.allclose(chk.selection_freq, chk.selection_freq_saved), "seed 42 does not reproduce s1a"
        for lab in (LAB12, LAB5, PHO5):
            for pool in ("registers only", "all 12"):
                t = pc[(pc["pool"] == pool) & (pc.label == lab)].set_index("candidate")
                l_ = R["per_l"][(R["per_l"]["pool"] == pool) & (R["per_l"].label == lab)].iloc[0]
                seed_rows.append({"flag": "S8", "rule": "S1(a) layer and token", "seed": seed, "pool": pool,
                                  "label_set": lab, "most_chosen": t.selection_freq.idxmax(),
                                  "freq_registers_L24_CLS": t.loc["registers L24 CLS", "selection_freq"],
                                  "freq_registers_L22_CLS": t.loc["registers L22 CLS", "selection_freq"],
                                  "freq_registers_any": t[t.index.str.startswith("registers")].selection_freq.sum(),
                                  "optimism_gap": l_.optimism_gap, "best_fixed": l_.best_fixed,
                                  "decision_winner_registers_pool": win,
                                  "decision_freq": float(dec.set_index("candidate").loc[win, "selection_freq"])})
        for src in ("patents", "photos"):
            d = D[src]
            cands = {c: X[KEY] for c, X in d["conds"].items()}
            sc = H.half_scores(cands, d["labels"], d["makers"], seed=seed)
            sl = H.select(sc)
            pcb, plb = H.summarise(sc, sl)
            wb, db = H.decide(pcb, {k: v for k, v in SOURCE_OF_LABEL.items() if v == src})
            for lab in d["labels"]:
                t = pcb[pcb.label == lab].set_index("candidate")
                seed_rows.append({"flag": "S8", "rule": "S1(b) preparation", "seed": seed, "pool": "preparations",
                                  "label_set": lab, "most_chosen": t.selection_freq.idxmax(),
                                  "freq_plain_or_colour": t.loc["plain" if src == "patents" else "colour", "selection_freq"],
                                  "optimism_gap": plb[plb.label == lab].optimism_gap.iloc[0],
                                  "best_fixed": plb[plb.label == lab].best_fixed.iloc[0],
                                  "decision_winner_registers_pool": wb,
                                  "decision_freq": float(db.set_index("candidate").loc[wb, "selection_freq"])})
    out["S8_seeds"] = pd.DataFrame(seed_rows)
    # (d) corrected Table 12
    t12 = []
    for pool in ("registers only", "all 12"):
        for lab in (LAB12, LAB5, PHO5):
            l_ = a_l[(a_l["pool"] == pool) & (a_l.label == lab)].iloc[0]
            f = a_c[(a_c["pool"] == pool) & (a_c.label == lab)].set_index("candidate").loc["registers L24 CLS"]
            prep = "plain" if lab != PHO5 else "colour"
            bl = b_l[b_l.label == lab].iloc[0]
            bc = b_c[(b_c.label == lab) & (b_c.candidate == prep)].iloc[0]
            t12.append({"flag": "S8", "pool": pool, "label_set": lab,
                        "fixed_L24_CLS_heldout": f.fixed_heldout_mean, "fixed_L24_CLS_lo": f.fixed_lo,
                        "fixed_L24_CLS_hi": f.fixed_hi, "mean_heldout_of_winners": l_.chosen_heldout_mean,
                        "winner_in_sample_mean": l_.winner_in_sample_mean, "optimism_gap_layer": l_.optimism_gap,
                        "full_sample_L24_CLS": f.full_sample, "preparation": prep,
                        "fixed_prep_heldout": bc.fixed_heldout_mean, "mean_heldout_of_prep_winners": bl.chosen_heldout_mean,
                        "optimism_gap_prep": bl.optimism_gap,
                        "note": "Table 12 printed the mean held-out score of the winners as the score of L24 CLS; "
                                "fixed_L24_CLS_heldout is L24 CLS's own held-out score"})
    out["S8_table12_corrected"] = pd.DataFrame(t12)
    # (e) joint grid: matrix x preparation through the rule
    jc_rows, jl_rows = [], []
    for src in ("patents", "photos"):
        d = D[src]
        cands = {f"{prep} {P.mname(k)}": Xs[k] for prep, Xs in d["conds"].items() for k in P.MATRICES}
        sc = H.half_scores(cands, d["labels"], d["makers"])
        sl = H.select(sc)
        pcj, plj = H.summarise(sc, sl)
        full = H.full_scores(cands, d["labels"], d["makers"])
        wj, dj = H.decide(pcj, {k: v for k, v in SOURCE_OF_LABEL.items() if v == src})
        jc_rows.append(pcj.merge(full, on=["label", "candidate"]).assign(flag="S8", source=src, decision_winner=wj))
        for r in plj.itertuples():
            seq = (a_l[(a_l["pool"] == "registers only") & (a_l.label == r.label)].optimism_gap.iloc[0]
                   + b_l[b_l.label == r.label].optimism_gap.iloc[0])
            jl_rows.append({"flag": "S8", "source": src, "label_set": r.label, "candidates": len(cands),
                            "most_chosen": r.most_chosen, "chosen_heldout_mean": r.chosen_heldout_mean,
                            "winner_in_sample_mean": r.winner_in_sample_mean, "joint_optimism_gap": r.optimism_gap,
                            "sequential_gaps_summed": seq, "best_fixed": r.best_fixed,
                            "best_fixed_heldout_mean": r.best_fixed_heldout_mean, "decision_winner": wj})
    out["S8_joint_grid_candidates"] = pd.concat(jc_rows, ignore_index=True)
    out["S8_joint_grid_labels"] = pd.DataFrame(jl_rows)
    # (f) the best-held-out-score criterion
    bh = []
    for nm, pc, sol in (("S1(a) registers only", a_c[a_c["pool"] == "registers only"], SOURCE_OF_LABEL),
                        ("S1(a) all 12", a_c[a_c["pool"] == "all 12"], SOURCE_OF_LABEL)):
        t = pc.assign(source=pc.label.map(sol))
        agg = t.groupby(["source", "candidate"])[["selection_freq", "fixed_heldout_mean"]].mean().groupby("candidate").mean()
        bh.append({"flag": "S7", "choice": nm, "by_selection_frequency": agg.selection_freq.idxmax(),
                   "by_best_mean_heldout_score": agg.fixed_heldout_mean.idxmax(),
                   "best_fixed_per_label": "; ".join(f"{LAB_SHORT[r.label]}: {r.best_fixed}" for r in
                                                     a_l[a_l["pool"] == nm.split(") ")[1]].itertuples())})
    for src, cands in (("patents", ["plain", "thick1", "thick2"]), ("photos", ["colour", "grey"])):
        t = b_c[b_c.candidate.isin(cands)]
        agg = t.groupby("candidate")[["selection_freq", "fixed_heldout_mean"]].mean()
        bh.append({"flag": "S7", "choice": f"S1(b) preparation, {src}", "by_selection_frequency": agg.selection_freq.idxmax(),
                   "by_best_mean_heldout_score": agg.fixed_heldout_mean.idxmax(),
                   "best_fixed_per_label": "; ".join(f"{LAB_SHORT[r.label]}: {r.best_fixed}" for r in
                                                     b_l[b_l.label.map(SOURCE_OF_LABEL) == src].itertuples())})
    for src, g in out["S8_joint_grid_candidates"].groupby("source"):
        agg = g.groupby("candidate")[["selection_freq", "fixed_heldout_mean"]].mean()
        bh.append({"flag": "S7", "choice": f"joint grid (matrix x preparation), {src}",
                   "by_selection_frequency": agg.selection_freq.idxmax(),
                   "by_best_mean_heldout_score": agg.fixed_heldout_mean.idxmax(),
                   "best_fixed_per_label": "; ".join(f"{LAB_SHORT[r['label_set']]}: {r['best_fixed']}" for r in jl_rows
                                                     if r["source"] == src)})
    out["S7_best_heldout_criterion"] = pd.DataFrame(bh)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# S9: attention extras
# ═════════════════════════════════════════════════════════════════════════════
def _pad_and_white(meta: pd.DataFrame, man: pd.DataFrame, g: int = 37, px: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Per image, (pad mask, white mask) over the g x g patch grid. Pad = patches wholly outside the pasted
    content (image_processing.process_image: rotate with expand, scale the long side to 518, paste centred on a
    white square). White = patches whose every pixel is >= 250 in every channel of the 518 input."""
    from PIL import Image
    size = g * px
    mi = man.set_index("figure_uid")
    pads, whites, how = [], [], []
    cols = np.arange(g) * px
    for r in meta.itertuples():
        a = np.asarray(Image.open(r.path).convert("RGB")) if Path(r.path).exists() else None
        if r.figure_uid in mi.index:
            w, h = int(mi.loc[r.figure_uid, "orig_w"]), int(mi.loc[r.figure_uid, "orig_h"])
            rot = int(mi.loc[r.figure_uid, "rotation_deg"]) % 360
            if rot:
                t = np.deg2rad(rot)
                w, h = (int(round(abs(w * np.cos(t)) + abs(h * np.sin(t)))),
                        int(round(abs(w * np.sin(t)) + abs(h * np.cos(t)))))
            s = size / max(w, h)
            nw, nh = max(1, round(w * s)), max(1, round(h * s))
            x0, y0 = (size - nw) // 2, (size - nh) // 2
            x1, y1 = x0 + nw, y0 + nh
            how.append("geometry")
        elif a is not None:
            # not in the current manifest (an image of the earlier base run): the pad is the pure-white
            # (255) border, found from the image; a drawing's own white margin counts as pad here
            full = (a == 255).all(axis=2)
            rows_c, cols_c = np.flatnonzero(~full.all(axis=1)), np.flatnonzero(~full.all(axis=0))
            y0, y1 = (rows_c.min(), rows_c.max() + 1) if len(rows_c) else (0, size)
            x0, x1 = (cols_c.min(), cols_c.max() + 1) if len(cols_c) else (0, size)
            how.append("white border of the image")
        else:
            x0, y0, x1, y1 = 0, 0, size, size
            how.append("unknown")
        cx = (cols + px <= x0) | (cols >= x1)
        cy = (cols + px <= y0) | (cols >= y1)
        pads.append(cy[:, None] | cx[None, :])
        whites.append((a >= 250).all(axis=2).reshape(g, px, g, px).all(axis=(1, 3)) if a is not None
                      else np.zeros((g, g), bool))
    return np.array(pads), np.array(whites), np.array(how)


def s9_attention() -> Dict[str, pd.DataFrame]:
    mans = {"patent": pd.read_csv(ER.PAT / "2_embedding_extraction/processed/518/manifest.csv"),
            "photo": pd.read_csv(ER.EVN / "2_embedding_extraction/processed/518/manifest.csv")}
    rows = []
    metas = {(t, n): pd.read_csv(ATT / t / f"{n}_meta.csv") for t in (BASE, REG) for n in ("patent", "photo")}
    for name in ("patent", "photo"):
        common = set(metas[(BASE, name)].figure_uid) & set(metas[(REG, name)].figure_uid)
        for tag in (BASE, REG):
            meta = metas[(tag, name)]
            arr = np.load(ATT / tag / f"{name}_cls.npy").astype(np.float32)
            pad, white, how = _pad_and_white(meta, mans[name])
            n, nl = arr.shape[0], arr.shape[1]
            flat = arr.reshape(n, nl, -1)
            for set_name, keep in (("all", np.ones(n, bool)), ("common", meta.figure_uid.isin(common).to_numpy())):
                for j, L in enumerate((18, 22, 24)):
                    f = flat[keep, j]
                    mx, tot = f.max(axis=1), f.sum(axis=1)
                    am = f.argmax(axis=1)
                    pk, wk = pad[keep].reshape(keep.sum(), -1), white[keep].reshape(keep.sum(), -1)
                    rows.append({"flag": "S9", "model": "registers" if tag == REG else "base", "source": name,
                                 "image_set": set_name, "layer": L, "n_images": int(keep.sum()),
                                 "max_patch_share_median": float(np.median(mx)), "max_patch_share_p90": float(np.quantile(mx, 0.9)),
                                 "renorm_max_share_median": float(np.median(mx / tot)),
                                 "renorm_max_share_p90": float(np.quantile(mx / tot, 0.9)),
                                 "prefix_share_median": float(np.median(1 - tot)), "prefix_share_p90": float(np.quantile(1 - tot, 0.9)),
                                 "argmax_in_pad_share": float(pk[np.arange(len(am)), am].mean()),
                                 "pad_patch_share_mean": float(pk.mean()),
                                 "images_with_any_pad_share": float(pk.any(axis=1).mean()),
                                 "argmax_in_white_patch_share": float(wk[np.arange(len(am)), am].mean()),
                                 "white_patch_share_mean": float(wk.mean()),
                                 "pad_from_white_border": int((how[keep] == "white border of the image").sum()),
                                 "pad_unknown": int((how[keep] == "unknown").sum())})
    t = pd.DataFrame(rows)
    t["note"] = ("attention = CLS query against every patch key, softmax over all tokens, mean over heads (saved arrays); "
                 "max_patch_share = largest patch / whole softmax; renorm = largest patch / sum over patches; prefix = "
                 "1 - sum over patches (CLS + register tokens); pad = patch wholly outside the pasted image; white = every "
                 "pixel >= 250; *_share_mean = expected argmax-in-pad / white share if the argmax fell on a random patch; "
                 "image_set common = the images both models ran on")
    return {"S9_attention_extras": t}


# ═════════════════════════════════════════════════════════════════════════════
# S10: audit intervals and the kappa breakdown
# ═════════════════════════════════════════════════════════════════════════════
def s10_audit_kappa() -> Dict[str, pd.DataFrame]:
    au = pd.read_csv(ER.EVN / "1_filter/audit_2026-09-22/audit_sample.csv", keep_default_na=False)
    rows = []
    for st, what in (("main", "hero images that show the whole aircraft"),
                     ("siglip_drop", "SigLIP-dropped images that were a whole vehicle"),
                     ("review_band", "review-band images that would have been usable")):
        g = au[au.stratum == st]
        k = int((g.verdict == "whole_aircraft").sum())
        lo, hi = wilson(k, len(g))
        rows.append({"flag": "S10", "sample": st, "what": what, "n": len(g), "count": k, "share": k / len(g),
                     "wilson_lo": lo, "wilson_hi": hi, "judge": "AI assistant from contact sheets showing the filter score"})
    t_au = pd.DataFrame(rows)
    Sp = sources(REG)[0]
    a = Sp.aircraft
    gt = a.arch_gt != ""
    lab, ref = a.label[gt], a.arch_gt[gt]
    rows = []
    for c in G1:
        m = lab == c
        dis = ref[m][ref[m] != c].value_counts()
        rows.append({"flag": "S10", "g1_wizard": c, "g1_display": disp(c), "n_aircraft": int(m.sum()),
                     "disagreements": int((ref[m] != c).sum()), "agree_share": float((ref[m] == c).mean()) if m.sum() else np.nan,
                     "whole_patent_reading_says": "; ".join(f"{disp(k)} {v}" for k, v in dis.items()),
                     "as_reading_type_n": int((ref == c).sum())})
    t_k = pd.DataFrame(rows)
    fz = pd.read_csv(ER.PAT / "0_labelling/inputs/text_architecture/image_labels_frozen_20260915.csv",
                     dtype=str, keep_default_na=False).set_index("variant_id").image_label_frozen
    acp = Sp.extra["acp"]
    frozen = a.aircraft_id.map(acp["variant_id"]).map(fz).fillna("")
    fg = gt & frozen.isin(G1)
    gtf = pd.read_csv(ER.PAT / "0_labelling/inputs/text_architecture/architecture_ground_truth.csv",
                      dtype=str, keep_default_na=False)
    gtf = gtf[gtf.ground_truth.isin(G1) & gtf.figure_label_frozen.isin(G1)]
    kap = [{"flag": "S10", "labels": "wizard G1 (current, analysis set)", "against": "whole-patent reading (arch_gt)",
            "set": "analysis aircraft with a reading", "n": int(gt.sum()), "agree": int((lab == ref).sum()),
            "agree_share": float((lab == ref).mean()), "kappa": float(cohen_kappa_score(lab, ref)), "origin": "computed"},
           {"flag": "S10", "labels": "frozen image labels (2026-09-15)", "against": "whole-patent reading (arch_gt)",
            "set": "analysis aircraft with a reading and a frozen label", "n": int(fg.sum()),
            "agree": int((frozen[fg] == a.arch_gt[fg]).sum()), "agree_share": float((frozen[fg] == a.arch_gt[fg]).mean()),
            "kappa": float(cohen_kappa_score(frozen[fg], a.arch_gt[fg])), "origin": "computed"},
           {"flag": "S10", "labels": "frozen image labels (figure_label_frozen)", "against": "ground truth (whole-patent reading)",
            "set": "every row of architecture_ground_truth.csv with a type in both", "n": len(gtf),
            "agree": int((gtf.figure_label_frozen == gtf.ground_truth).sum()),
            "agree_share": float((gtf.figure_label_frozen == gtf.ground_truth).mean()),
            "kappa": float(cohen_kappa_score(gtf.figure_label_frozen, gtf.ground_truth)), "origin": "computed"},
           {"flag": "S10", "labels": "frozen image labels", "against": "ground truth", "set": "all aircraft rows incl. duplicates (n 803)",
            "n": 803, "agree": np.nan, "agree_share": 0.890, "kappa": 0.866,
            "origin": "from the project record of 2026-09-17 (GT pass), not recomputed: the file now holds 798 rows"}]
    return {"S10_audit_wilson": t_au, "S10_kappa_by_type": t_k, "S10_kappa": pd.DataFrame(kap)}


# ═════════════════════════════════════════════════════════════════════════════
# S11: the view claim
# ═════════════════════════════════════════════════════════════════════════════
def s11_view(n_boot: int = 500, n_boot_aircraft: int = 1000, umap_only: bool = False) -> Dict[str, pd.DataFrame]:
    Sp = sources(REG)[0]
    c = Sp.confound
    Xf, ya, yv, mf = c["X"][KEY], c["y_arch"], c["y_conf"], c["makers"]
    out: Dict[str, pd.DataFrame] = {}
    ft = Sp.extra["figure_table"]
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(Sp.aircraft.aircraft_id))].reset_index(drop=True)
    import umap
    e = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(Xf)
    out["S11_umap_figures_by_view"] = pd.DataFrame({"flag": "S11", "fig_id": wv.fig_id, "aircraft_id": wv.aircraft_id,
                                                    "view4": yv, "g1": ya, "g1_display": [disp(v) for v in ya],
                                                    "is_main": wv.is_main, "umap_1": e[:, 0], "umap_2": e[:, 1]})
    if umap_only:
        return out
    X0, y, mk = Sp.rules[Sp.primary_rule][KEY], Sp.y, Sp.makers
    rows = []

    def sep(X, yy, gg, what, unit):
        r = P.separation(X, yy, gg, n_perm=0)
        rows.append({"flag": "S11", "measure": what, "unit": unit, "n": len(yy), "d": r["d"], "ratio": r["ratio"],
                     "pairs_within": r["pairs_within"], "pairs_between": r["pairs_between"]})
        return r["d"]

    d_ac = sep(X0, y, mk, "architecture d", "aircraft (main figure = one vector per aircraft), 663")
    d_fa = sep(Xf, ya, mf, "architecture d", "all whole-aircraft figures, 1 541")
    d_fv = sep(Xf, yv, mf, "view d", "all whole-aircraft figures, 1 541")
    main = wv.is_main.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    sep(Xf[main], ya[main], mf[main], "architecture d", f"main figures only, as figures ({int(main.sum())})")
    sep(Xf[~main], ya[~main], mf[~main], "architecture d", f"non-main figures only ({int((~main).sum())})")
    for v in ("Perspective", "Side", "Plan", "FrontRear"):
        m = yv == v
        if m.sum() > 50:
            sep(Xf[m], ya[m], mf[m], "architecture d",
                f"figures of one view only: {v.replace('FrontRear', 'Front/Rear')} ({int(m.sum())})")
    t = pd.DataFrame(rows)
    t["view_over_arch_figures"] = d_fv / d_fa
    out["S11_d_units"] = t
    # bootstrap by maker: the ratio on figures, and the aircraft-level architecture d
    rng = np.random.default_rng(SEED)
    memf = _members(mf)
    bs = []
    for b in range(n_boot):
        i = resample(rng, len(mf), memf)
        a_ = P.separation(Xf[i], ya[i], mf[i], n_perm=0)["d"]
        v_ = P.separation(Xf[i], yv[i], mf[i], n_perm=0)["d"]
        bs.append({"flag": "S11", "boot": b, "unit": "figures", "arch_d": a_, "view_d": v_, "ratio": v_ / a_})
    rng = np.random.default_rng(SEED)
    mema = _members(mk)
    for b in range(n_boot_aircraft):
        i = resample(rng, len(mk), mema)
        bs.append({"flag": "S11", "boot": b, "unit": "aircraft", "arch_d": P.separation(X0[i], y[i], mk[i], n_perm=0)["d"],
                   "view_d": np.nan, "ratio": np.nan})
    bs = pd.DataFrame(bs)
    out["S11_bootstrap_draws"] = bs
    f = bs[bs.unit == "figures"]
    ac = bs[bs.unit == "aircraft"]
    ve = pd.read_csv(VEDIR / "view_erasure.csv")
    vk = ve[(ve.tag == REG) & (ve.section == "view kNN-5 (figures, maker held out)") & (ve.variant == "original")].iloc[0]
    summ = [{"flag": "S11", "measure": "view d / architecture d (figures)", "estimate": d_fv / d_fa,
             "ci_lo": interval(f.ratio)[0], "ci_hi": interval(f.ratio)[1], "chance": np.nan,
             "method": f"bootstrap over makers ({n_boot}), all figures of a drawn maker, same-maker pairs left out"},
            {"flag": "S11", "measure": "architecture d (figures)", "estimate": d_fa, "ci_lo": interval(f.arch_d)[0],
             "ci_hi": interval(f.arch_d)[1], "chance": 0.0, "method": f"bootstrap over makers ({n_boot})"},
            {"flag": "S11", "measure": "view d (figures)", "estimate": d_fv, "ci_lo": interval(f.view_d)[0],
             "ci_hi": interval(f.view_d)[1], "chance": 0.0, "method": f"bootstrap over makers ({n_boot})"},
            {"flag": "S11", "measure": "architecture d (aircraft, main figure)", "estimate": d_ac,
             "ci_lo": interval(ac.arch_d)[0], "ci_hi": interval(ac.arch_d)[1], "chance": 0.0,
             "method": f"bootstrap over makers ({n_boot_aircraft})"},
            {"flag": "S11", "measure": "view kNN-5 balanced accuracy (figures, maker held out)", "estimate": vk.value,
             "ci_lo": vk.ci_lo, "ci_hi": vk.ci_hi, "chance": vk.chance_p95,
             "method": "saved in view_erasure.csv; chance = 95th percentile of 50 label shuffles; 4 views, 1/4 = 0.25"}]
    out["S11_view_ratio"] = pd.DataFrame(summ)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# output helpers
# ═════════════════════════════════════════════════════════════════════════════
def write(tables: Dict[str, pd.DataFrame], out: Path = OUT) -> List[Path]:
    """Write each table as <name>.csv; refuse to overwrite anything outside the review_fixes folder."""
    out.mkdir(parents=True, exist_ok=True)
    assert out.resolve().name == "review_fixes"
    paths = []
    for name, df in tables.items():
        p = out / f"{name}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


# ═════════════════════════════════════════════════════════════════════════════
# figures (PNG, 200 dpi, A4 text width, Source and How-to-read printed inside the image)
# ═════════════════════════════════════════════════════════════════════════════
INK, MUTED, FAINT, ACCENT = ER.INK, ER.MUTED, ER.FAINT, ER.ACCENT
VIEW_STYLE = {"Perspective": ("#2a78d6", "o"), "Plan": ("#eb6834", "s"), "Side": ("#1baf7a", "^"),
              "FrontRear": ("#4a3aa7", "D")}      # validated with the dataviz validator (light): passes
MODEL_NOTE = "DINOv2-large with registers, frozen, 518 px, layer 24 CLS"
G1_CODES = "TP Tilt Propulsor, CVT Combined vectored thrust, TW Tilt Wing, TB Tilt Body, PTC Pitch-to-Cruise, DS Deflected Slipstream, SLC Lift + Cruise, SRW Stopped/Slowed Rotor Wing, MR Multirotor, RC Rotorcraft, HB Hoverbike, PFV Personal Flying Vehicle"


def _stamp_save(fig, path: Path, source: str, read: str) -> Path:
    """Source / How-to-read lines hung under everything drawn (the la_figures stamp rule), then saved."""
    import re
    from matplotlib.text import Text
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    x0, y0 = bb.x0 / fig.get_figwidth(), bb.y0 / fig.get_figheight()
    chars = max(int(min(bb.width, 7.0) * 15.5), 60)
    lines = textwrap.wrap("Source: " + source, chars) + textwrap.wrap("How to read: " + read, chars)
    fig.text(x0, y0 - 0.012, "\n".join(lines), fontsize=7, color=MUTED, ha="left", va="top")
    fig.canvas.draw()
    for t in fig.findobj(Text):
        s = t.get_text()
        if re.search(r"\bTR\b", s) or "—" in s:
            raise ValueError(f"figure text breaks the print rules (TR or em dash): {s!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return path


def _place_labels(ax, pts: List[Tuple[float, float, str]], fs: float = 6.3) -> None:
    """Greedy label placement: each label goes right of its point, moved up in steps (with a thin leader)
    until it no longer overlaps an already placed label. Positions in axes fractions."""
    tr = ax.transData + ax.transAxes.inverted()
    placed: List[Tuple[float, float, float]] = []
    for x, y, t in sorted(pts, key=lambda p: (p[1], p[0])):
        fx, fy = tr.transform((x, y))
        w = 0.0105 * len(t) * fs / 6.3
        ly = fy + 0.01
        for _ in range(40):
            if all(not (abs(ly - py) < 0.045 and (fx + 0.015 < px + pw and px < fx + 0.015 + w)) for px, py, pw in placed):
                break
            ly += 0.02
        placed.append((fx + 0.015, ly, w))
        ax.annotate(t, (x, y), xytext=(fx + 0.015, ly), textcoords="axes fraction", fontsize=fs, color=INK, va="bottom",
                    arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.5) if ly - fy > 0.03 else None)


def _read(out: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(out / f"{name}.csv", keep_default_na=False, na_values=[""])


def fig_forest(out: Path, width: float = 5.2) -> Path:
    """F-A: every paired contrast of the document, with both intervals and the per-half sign share.
    ``width``: the axes' width in inches (the analysis brief draws it narrower so its labels print larger)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    s4 = _read(out, "S4_paired_contrasts")
    m3 = _read(out, "M3_pool_size_and_mix").set_index("row")
    groups = ["model", "drawing preparation", "photo preparation", "photo background", "view erasure",
              "erasure control", "image rule"]
    head = {"model": "Model: registers against base (L24 CLS)", "drawing preparation": "Drawing preparation",
            "photo preparation": "Photo preparation", "photo background": "Photo background: crop to the aircraft",
            "view erasure": "View erasure (registers)", "erasure control": "Variance-matched control, seed 42 (registers)",
            "image rule": "Image rule, drawings, 12 G1 classes", "pool": "Photos against drawings, 5 classes (independent sets)"}
    items: List[Tuple[str, Any]] = []
    for g in groups:
        sub = s4[s4.group == g]
        if len(sub):
            items.append(("head", head[g]))
            items += [("row", r) for r in sub.itertuples()]
    items.append(("head", head["pool"]))
    for key, lab in (("difference: photos full set minus drawings", "full photo set - drawings"),
                     ("difference: photos size only minus drawings", "size only (663, photos' mix) - drawings"),
                     ("difference: photos size + mix (matched) minus drawings", "size + mix (663, drawings' mix) - drawings")):
        items.append(("m3", (lab, m3.loc[key])))
    n = len(items)
    fig, ax = plt.subplots(figsize=(width, 0.22 * n + 0.9))
    ylab, yt = [], []
    for k, (kind, it) in enumerate(items):
        y = n - k
        if kind == "head":
            ax.text(-0.005, y, it, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                    fontweight="bold", color=INK)
            continue
        if kind == "row":
            r = it
            base_ = r.contrast.split(",")[0]
            for a, b_ in (("view erased minus original", "erased - original"),
                          ("variance-matched control (seed 42) minus original", "control - original"),
                          (" minus ", " - ")):
                base_ = base_.replace(a, b_)
            tags = [{LAB12: "drawings, 12", LAB5: "drawings, 5", PHO5: "photos, 5"}[r.label_set]]
            if "Perspective" in r.contrast:
                tags.append("Perspective")
            if "(base)" in r.contrast or "(registers)" in r.contrast:
                tags.append("base model" if "(base)" in r.contrast else "registers")
            name = f"{base_.replace(' photos (base)', '').replace(' photos (registers)', '')} ({', '.join(tags)})"
            ax.plot([r.ci_lo_maker, r.ci_hi_maker], [y, y], color=ACCENT, lw=3.2, alpha=0.35, solid_capstyle="butt")
            ax.plot([r.ci_lo_aircraft, r.ci_hi_aircraft], [y, y], color=INK, lw=1.0)
            ax.plot([r.diff], [y], "o", color=INK, ms=4)
            ax.text(1.01, y, f"{r['diff'] if isinstance(r, pd.Series) else r.diff:+.3f}   {r.share_of_halves_same_sign:.0%}",
                    transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.8, color=INK)
        else:
            lab, r = it
            name = lab
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=MUTED, lw=1.0, ls=(0, (2, 1)))
            ax.plot([r.estimate], [y], "o", mfc="white", mec=INK, ms=4.5)
            ax.text(1.01, y, f"{r.estimate:+.3f}   n/a", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=6.8, color=INK)
        yt.append(y)
        ylab.append("  " + name)
    ax.set_yticks(yt, ylab, fontsize=6.8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xlabel("difference in kNN-5 balanced accuracy (second condition better when > 0)")
    ax.text(1.01, n + 0.6, "diff   same sign\n          in halves", transform=ax.get_yaxis_transform(), ha="left",
            va="bottom", fontsize=6.5, color=MUTED)
    ax.grid(axis="x", color=FAINT, lw=0.5)
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.0, marker="o", ms=4, label="estimate, 95 % interval (aircraft resampled)"),
                       Line2D([], [], color=ACCENT, lw=3.2, alpha=0.35, label="95 % interval (makers resampled)"),
                       Line2D([], [], color=MUTED, lw=1.0, ls=(0, (2, 1)), marker="o", mfc="white", mec=INK, ms=4.5,
                              label="independent sets: 200 draws x bootstrap")],
              fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.07 - 0.3 / n), ncol=2)
    return _stamp_save(fig, out / "figs" / "FA_forest_paired_contrasts.png",
                       f"{MODEL_NOTE} unless the row says base; kNN-5 maker held out; drawings: exp3 main figure, 663 "
                       "aircraft; photos: hero image, 1 157 aircraft; paired bootstrap 1 000 resamples (seed 42); halves: "
                       "200 maker splits x 2, neighbours inside the half; review_fixes S4_paired_contrasts.csv, "
                       "M3_pool_size_and_mix.csv. In brackets: drawings, 12 = the 12 G1 types; drawings, 5 = their 5 parent classes; "
                       "photos, 5 = the 5 directory classes; plain grey, contrast only and the registers crop use the "
                       "embeddings of the GPU run of 2026-09-30.",
                       "each row is one contrast of the document: the dot is the difference of the two conditions on the "
                       "same aircraft, the thin line its interval when aircraft are resampled, the pale bar when whole makers are "
                       "resampled (the more conservative one); a bar that crosses 0 is no detected difference. The right column "
                       "gives the difference and the share of the 400 half-samples whose difference has the same sign. The "
                       "last three rows compare different aircraft, so they carry no half share.")


def fig_recall_vs_pool(out: Path) -> Path:
    """F-B: recall per class against the class's share of the neighbour pool."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pc = _read(out, "M3_per_class_pool_share")
    g12 = _read(out, "M9_recall_12_wilson")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    pts_l: List[Tuple[float, float, str]] = []
    series = [("drawings", "pool_share_drawings", "recall_drawings", "s", "white"),
              ("photos, all", "pool_share_photos", "recall_photos", "o", None),
              ("photos, drawings' mix", "pool_share_matched", "recall_matched", "^", None)]
    for name, xs, ys, mk, face in series:
        for r in pc.itertuples():
            col = ER.CLASS_COLOR[r._2 if False else r[pc.columns.get_loc("class") + 1]]
            x, y = getattr(r, xs), getattr(r, ys)
            if name == "photos, drawings' mix":
                ax.plot([x, x], [r.recall_matched_lo, r.recall_matched_hi], color=col, lw=0.8, alpha=0.7)
            ax.plot([x], [y], mk, ms=6, mfc=face or col, mec=col, mew=1.1, alpha=1 if face or name == "photos, all" else 0.6)
            pts_l.append((x, y, r[pc.columns.get_loc("class") + 1]))
    ax.axhline(0.2, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, 0.65), ax.set_ylim(0, 1)
    _place_labels(ax, pts_l)
    ax.set_xlabel("class's share of the neighbour pool")
    ax.set_ylabel("recall (kNN-5, maker held out)")
    ax.set_title("Shared 5 classes", loc="left")
    ax.set_xlim(0, 0.65), ax.set_ylim(0, 1)
    ax.legend(handles=[Line2D([], [], marker=m, ls="", mfc="white" if f else MUTED, mec=MUTED, ms=6, label=n)
                       for n, _, _, m, f in series], fontsize=6.5, loc="upper left")
    ax = axes[1]
    for r in g12.itertuples():
        col = ER.CLASS_COLOR[r.parent]
        ax.plot([r.pool_share] * 2, [r.knn_wilson_lo, r.knn_wilson_hi], color=col, lw=0.8, alpha=0.6)
        ax.plot([r.pool_share], [r.knn_recall], "o", ms=6, mfc="white" if r.under_20 else col, mec=col, mew=1.1)
    ax.axhline(1 / 12, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, 0.33), ax.set_ylim(0, 1)
    _place_labels(ax, [(r.pool_share, r.knn_recall, f"{r.g1_display} ({r.n_aircraft})") for r in g12.itertuples()])
    ax.set_xlabel("type's share of the neighbour pool")
    ax.set_title("Drawings, 12 G1 types", loc="left")
    ax.set_xlim(0, 0.33), ax.set_ylim(0, 1)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc=ER.CLASS_COLOR[c], mec=ER.CLASS_COLOR[c], ms=5, label=c)
                       for c in C5] + [Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=5, label="under 20 aircraft")],
              fontsize=6.3, loc="upper right", ncol=2, title="parent class", title_fontsize=6.3)
    return _stamp_save(fig, out / "figs" / "FB_recall_vs_pool_share.png",
                       f"{MODEL_NOTE}; kNN-5 maker held out; drawings: exp3 main figure, 663 aircraft (G1 folded to its "
                       "parent on the left); photos: hero image, 1 157 aircraft; drawings' mix = mean of 200 photo subsets of "
                       "663 with the drawings' class counts (whisker: 2.5 to 97.5 % of the draws); right: Wilson 95 % "
                       "whiskers; review_fixes M3_per_class_pool_share.csv, M9_recall_12_wilson.csv. Classes: VT Vectored "
                       "Thrust, LC Lift + Cruise, WM Wingless (Multicopter), ER Electric Rotorcraft, HB Hover Bikes; TP = Tilt Propulsor.",
                       "each point is one class; further right = the class fills more of the pool of possible neighbours. "
                       "A rising cloud means a class is read well largely because it dominates the pool. Dashed lines: chance "
                       "1/5 (left) and 1/12 (right).")


def fig_optimism(out: Path) -> Path:
    """F-C: in-sample winner, held-out chosen, fixed held-out and full-sample score, per label set."""
    import matplotlib.pyplot as plt
    t12 = _read(out, "S8_table12_corrected")
    jg = _read(out, "S8_joint_grid_labels")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6))
    marks = [("full sample, L24 CLS", "o", INK, "white"), ("in sample, winner of a half", "D", MUTED, MUTED),
             ("held out, winner of a half", "o", ACCENT, ACCENT), ("held out, L24 CLS fixed", "s", INK, INK)]
    for ax, lab in zip(axes, (LAB12, LAB5, PHO5)):
        rows = []
        for pool, nm in (("registers only", "layer + token\n(6 registers)"), ("all 12", "model + layer\n+ token (12)")):
            r = t12[(t12["pool"] == pool) & (t12.label_set == lab)].iloc[0]
            rows.append((nm, r.full_sample_L24_CLS, r.winner_in_sample_mean, r.mean_heldout_of_winners, r.fixed_L24_CLS_heldout))
        j = jg[jg.label_set == lab]
        if len(j):
            j = j.iloc[0]
            r0 = t12[(t12["pool"] == "registers only") & (t12.label_set == lab)].iloc[0]
            rows.append((f"layer + token\n+ preparation ({int(j.candidates)})", r0.full_sample_L24_CLS,
                         j.winner_in_sample_mean, j.chosen_heldout_mean, r0.fixed_L24_CLS_heldout))
        for k, (nm, full, ins, ho, fx) in enumerate(rows):
            y = len(rows) - k
            ax.plot([ho, ins], [y, y], color=FAINT, lw=2.2, zorder=1)
            for (lbl, m, ec, fc), v in zip(marks, (full, ins, ho, fx)):
                ax.plot([v], [y], m, ms=5.5, mec=ec, mfc=fc, zorder=2, label=lbl if k == 0 else None)
            ax.text(ins, y + 0.22, f"gap {ins - ho:.3f}", fontsize=6, color=INK, ha="center")
        ax.set_yticks(range(len(rows), 0, -1), [r[0] for r in rows], fontsize=6.3)
        ax.set_ylim(0.4, len(rows) + 0.6)
        ax.set_title(LAB_SHORT[lab][0].upper() + LAB_SHORT[lab][1:], loc="left", fontsize=7.8)
        ax.set_xlabel("kNN-5 balanced accuracy", fontsize=7)
        ax.grid(axis="x", color=FAINT, lw=0.5)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].legend(fontsize=6.3, loc="upper center", bbox_to_anchor=(1.7, -0.25), ncol=4)
    return _stamp_save(fig, out / "figs" / "FC_optimism_dumbbell.png",
                       "DINOv2-large with registers (and base in the 12-candidate pool), frozen, 518 px; split-half rule: 200 "
                       "maker splits (seed 42) x 2 directions; drawings: exp3 main figure, halves of 248 to 415 aircraft; photos: "
                       "hero image, halves of 496 to 661; full sample: 663 / 1 157 aircraft; review_fixes S8_table12_corrected.csv, "
                       "S8_joint_grid_labels.csv.",
                       "per row, the grey bar joins the in-sample score of the winner of a half to its score on the other half: "
                       "its length is the optimism gap (printed). The blue dot (held-out score of whatever won) and the black "
                       "square (held-out score of L24 CLS kept fixed) differ because L24 CLS does not win every half; the open "
                       "circle is the full-sample score quoted in the results, on a pool twice as large, so it sits to the right "
                       "of every half score.")


def fig_class_counts(out: Path) -> Path:
    """F-D: aircraft per G1 type coloured by parent; the 5 shared classes in both sources."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    g12 = _read(out, "M9_recall_12_wilson")
    pc = _read(out, "M3_per_class_pool_share")
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    x = np.arange(len(g12))
    ax.bar(x, g12.n_aircraft, width=0.72, color=[ER.CLASS_COLOR[p] for p in g12.parent])
    for xi, v in zip(x, g12.n_aircraft):
        ax.text(xi, v + 3, str(v), ha="center", fontsize=6.5, color=INK)
    ax.axhline(20, color=MUTED, lw=0.8, ls="--")
    ax.annotate("20", (1.0, 20), xycoords=("axes fraction", "data"), xytext=(2, 0), textcoords="offset points",
                va="center", fontsize=6.3, color=MUTED, annotation_clip=False)
    ax.set_xticks(x, g12.g1_display, fontsize=7)
    ax.set_ylabel("aircraft")
    ax.set_title("Drawings, 12 G1 types (663 aircraft)", loc="left")
    ax.legend(handles=[Patch(color=ER.CLASS_COLOR[c], label=c) for c in C5], fontsize=6.3, title="parent class",
              title_fontsize=6.3, loc="upper right", ncol=2)
    ax = axes[1]
    x = np.arange(len(pc))
    for xi, r in zip(x, pc.itertuples()):
        c = r[pc.columns.get_loc("class") + 1]
        ax.bar(xi - 0.19, r.n_drawings, 0.36, color="white", edgecolor=ER.CLASS_COLOR[c], hatch="////", lw=0.8)
        ax.bar(xi + 0.19, r.n_photos, 0.36, color=ER.CLASS_COLOR[c])
        ax.text(xi - 0.19, r.n_drawings + 5, str(r.n_drawings), ha="center", fontsize=6, color=INK)
        ax.text(xi + 0.19, r.n_photos + 5, str(r.n_photos), ha="center", fontsize=6, color=INK)
    ax.set_xticks(x, pc["class"], fontsize=7)
    ax.set_title("Shared 5 classes", loc="left")
    ax.legend(handles=[Patch(facecolor="white", edgecolor=INK, hatch="////", label="drawings (663)"),
                       Patch(facecolor=MUTED, label="photos (1 157)")], fontsize=6.3, loc="upper right")
    return _stamp_save(fig, out / "figs" / "FD_class_counts.png",
                       "analysis sets: patent drawings, 663 aircraft with a human G1 type (left; folded to the parent on the "
                       "right); evtol.news photos, 1 157 aircraft with the directory class; review_fixes "
                       "M9_recall_12_wilson.csv, M3_per_class_pool_share.csv. Codes: " + G1_CODES + "; parents "
                       "VT Vectored Thrust, LC Lift + Cruise, WM Wingless (Multicopter), ER Electric Rotorcraft, HB Hover Bikes.",
                       "bar height = aircraft per class; the dashed line marks 20 aircraft, below which a recall rests on a "
                       "handful of aircraft. Right: hatched = drawings, solid = photos; the two sources have different class mixes.")


def fig_recall12(out: Path) -> Path:
    """F-E: 12-class recall with n and Wilson whiskers, classes under 20 greyed; the balanced reading beside it."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    g12 = _read(out, "M9_recall_12_wilson")
    bal_ = _read(out, "M6_balanced_recall").set_index("g1")
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    x = np.arange(len(g12))
    cols = [FAINT if u else ER.CLASS_COLOR[p] for u, p in zip(g12.under_20, g12.parent)]
    ax.bar(x - 0.17, g12.knn_recall, 0.32, color=cols)
    ax.errorbar(x - 0.17, g12.knn_recall, yerr=[g12.knn_recall - g12.knn_wilson_lo, g12.knn_wilson_hi - g12.knn_recall],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=1.5)
    b = bal_.reindex(g12.g1)
    ax.errorbar(x + 0.17, b.knn_recall_mean, yerr=[b.knn_recall_mean - b.knn_recall_lo, b.knn_recall_hi - b.knn_recall_mean],
                fmt="D", ms=4, color=INK, mfc="white", elinewidth=0.8, capsize=1.5)
    ax.plot(x + 0.17, b.shuffled_recall_p95, "_", ms=9, color=MUTED, mew=1.2)
    for xi, r in zip(x, g12.itertuples()):
        ax.text(xi - 0.17, -0.07, f"{r.knn_correct}/{r.n_aircraft}", ha="center", fontsize=5.8, color=INK)
    ax.set_xticks(x, g12.g1_display, fontsize=7.5)
    ax.tick_params(axis="x", pad=10)
    ax.set_ylim(-0.1, 1.0)
    ax.axhline(0, color=MUTED, lw=0.6)
    ax.set_ylabel("recall (kNN-5, maker held out)")
    ax.legend(handles=[Patch(color=ER.CLASS_COLOR["VT"], label="all 663 aircraft, Wilson 95 % (colour = parent)"),
                       Patch(color=FAINT, label="class under 20 aircraft"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=4,
                              label="size-balanced: 20 per class, 200 draws, 2.5 to 97.5 %"),
                       Line2D([], [], marker="_", ls="", color=MUTED, ms=9, mew=1.2, label="balanced, shuffled labels, 95th percentile")],
              fontsize=6.3, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    return _stamp_save(fig, out / "figs" / "FE_recall_12_wilson.png",
                       f"{MODEL_NOTE}; exp3 main figure; 663 aircraft, 12 G1 types; bars: full sample with Wilson 95 % "
                       "intervals, correct / aircraft under each bar; diamonds: class-balanced subsamples (20 aircraft per type, "
                       "all of a type with fewer), neighbours only inside the subsample; review_fixes M9_recall_12_wilson.csv, "
                       "M6_balanced_recall.csv. Codes: " + G1_CODES + ".",
                       "bar = share of a type's aircraft whose 5 nearest aircraft of other makers vote for the right type; grey "
                       "bars rest on under 20 aircraft and their whiskers show how little they pin down. The diamond is the "
                       "same reading when every type has the same weight in the pool; above the grey tick it beats shuffled labels.")


def fig_erasure(out: Path) -> Path:
    """F-F: before / after the view erasure, and the variance-matched control over 200 draws."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sc = _read(out, "M4c_erased_scores")
    ctl = _read(out, "M4b_control_200_draws")
    ctl = ctl[ctl.control == "variance-matched random directions"]
    labs = [("G1 12-class", "all", LAB12, "G1, 12"), ("parent 5-class", "all", LAB5, "parent, 5"),
            ("G1 12-class, Perspective main figures only", "Perspective main figures only", LAB12, "G1, 12\nPerspective"),
            ("parent 5-class, Perspective main figures only", "Perspective main figures only", LAB5, "parent, 5\nPerspective")]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, model in zip(axes, ("registers", "base")):
        for k, (ls_, sub, lab, short) in enumerate(labs):
            g = sc[(sc.model == model) & (sc.label_set == ls_)].set_index("vectors")
            o, e = g.loc["original"], g.loc["view erasure"]
            c = ctl[(ctl.model == model) & (ctl.subset == sub) & (ctl.label_set == lab)].iloc[0]
            ax.plot([k - 0.15, k + 0.1], [o.bal_acc, e.bal_acc], color=FAINT, lw=1.5, zorder=1)
            ax.plot([k - 0.15], [o.bal_acc], "o", color=INK, ms=5)
            ax.plot([k + 0.1], [e.bal_acc], "o", color=ACCENT, ms=5)
            cm = o.bal_acc + c.control_mean
            ax.errorbar([k + 0.3], [cm], yerr=[[c.control_mean - c.control_lo], [c.control_hi - c.control_mean]],
                        fmt="^", ms=4.5, color=MUTED, elinewidth=0.8, capsize=1.5)
            ax.plot([k - 0.3, k + 0.4], [o.chance_p95] * 2, color=MUTED, lw=0.8, ls=":")
        ax.set_xticks(range(len(labs)), [l[3] for l in labs], fontsize=6.5)
        ax.set_title(f"{model.capitalize()} model", loc="left")
        ax.grid(axis="y", color=FAINT, lw=0.5)
    axes[0].set_ylabel("kNN-5 balanced accuracy")
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="", color=INK, ms=5, label="original"),
                            Line2D([], [], marker="o", ls="", color=ACCENT, ms=5, label="view erased (3 directions)"),
                            Line2D([], [], marker="^", ls="-", color=MUTED, ms=4.5, lw=0.8,
                                   label="variance-matched control: mean, 2.5 to 97.5 % of 200 draws"),
                            Line2D([], [], ls=":", color=MUTED, lw=0.8, label="shuffled-label chance (95th percentile)")],
                   fontsize=6.3, loc="upper center", bbox_to_anchor=(1.05, -0.2), ncol=2)
    return _stamp_save(fig, out / "figs" / "FF_view_erasure_control.png",
                       "DINOv2-large and DINOv2-large with registers, frozen, 518 px, layer 24 CLS; exp3 main figure; 663 aircraft "
                       "(477 with a Perspective main figure); erasure = 3 view-mean directions fitted on the 1 541 whole-aircraft "
                       "figures; control = 3 random directions inside the top principal components carrying the same variance, "
                       "200 seeds; review_fixes M4c_erased_scores.csv, M4b_control_200_draws.csv.",
                       "each pair of dots joins a score before (black) and after (blue) removing the view; the grey triangle "
                       "is where the score lands, on average, when the same amount of variance is removed in directions the "
                       "view did not choose. When the blue dot sits inside the grey whisker, the loss is not specific to the view.")


def fig_umap_view(out: Path) -> Path:
    """F-G: the 1 541 figure embeddings in 2-D, coloured by the view of the drawing."""
    import matplotlib.pyplot as plt
    u = _read(out, "S11_umap_figures_by_view")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for v, (col, mk) in VIEW_STYLE.items():
        m = u.view4 == v
        ax.scatter(u.umap_1[m], u.umap_2[m], s=8, marker=mk, color=col, alpha=0.7, linewidths=0,
                   label=f"{v.replace('FrontRear', 'Front/Rear')} ({int(m.sum())})")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel("UMAP dimension 1 (no unit)"), ax.set_ylabel("UMAP dimension 2 (no unit)")
    ax.legend(fontsize=7, markerscale=1.6, loc="best", title="view of the drawing", title_fontsize=7)
    return _stamp_save(fig, out / "figs" / "FG_umap_figures_by_view.png",
                       f"{MODEL_NOTE}; all 1 541 whole-aircraft patent figures of the 663 aircraft (not one per aircraft); "
                       "UMAP n_neighbors 15, min_dist 0.1, cosine, seed 42; review_fixes S11_umap_figures_by_view.csv.",
                       "each point is one figure. UMAP reduces each 1 024-number embedding to 2 numbers so that figures with "
                       "similar embeddings sit close together; the axes have no unit and no physical meaning, only closeness "
                       "is meaningful. Regions of one colour are figures grouped by how the aircraft is drawn (the view), "
                       "not by what it is.")


def make_figures(out: Path = OUT) -> List[Path]:
    paths = []
    for f in (fig_forest, fig_recall_vs_pool, fig_optimism, fig_class_counts, fig_recall12, fig_erasure, fig_umap_view):
        try:
            paths.append(f(out))
        except FileNotFoundError as e:
            print(f"[figures] {f.__name__} skipped: {e}")
    return paths


# ═════════════════════════════════════════════════════════════════════════════
# INDEX.md: every file, its flag, and the verdict sentences (all numbers read from the CSVs)
# ═════════════════════════════════════════════════════════════════════════════
FILES = {
    "M3_pool_size_and_mix": ("M3", "drawings, full photos, size-only and size+mix photo subsets (663), each with an interval; "
                                   "the photo-minus-drawing differences resampling both sides; the size / mix decomposition"),
    "M3_per_class_pool_share": ("M3, S15", "per shared class: counts, pool share and recall of drawings, all photos, size-only "
                                           "and matched photos (draw percentiles, Wilson for the full sets)"),
    "M3_draws": ("M3", "balanced accuracy of each of the 200 size-only and 200 size+mix photo subsets"),
    "M4a_heldout_view_probe": ("M4", "view probe before / after the erasure: in sample (figures, main figures) and fitted on "
                                     "other makers (the saved held-out check), with the share of above-chance signal removed"),
    "M4a_heldout_architecture_by_split": ("M4", "NEW held-out erasure: fitted on the figures of one half of the makers, applied "
                                                "to the other half; architecture kNN change and view-neighbour change, against "
                                                "the same erasure fitted in sample on that half; 200 splits x 2"),
    "M4a_heldout_architecture_halves": ("M4", "the per-half rows behind M4a_heldout_architecture_by_split"),
    "M4b_control_200_draws": ("M4", "variance-matched random-direction control (and plain random directions) over 200 seeds, "
                                    "both models, 4 label sets: mean, 2.5-97.5 %, share of draws as costly as the erasure"),
    "M4b_control_draws": ("M4", "every control draw"),
    "M4c_erased_scores": ("M4", "original / erased / control scores with bootstrap intervals and the shuffled chance (from view_erasure.csv)"),
    "M9_recall_12_wilson": ("M9, M6", "per G1 type: n, correct, kNN recall with Wilson interval, probe recall with Wilson "
                                      "interval, pool share, under-20 flag"),
    "M9_wrong_predictions": ("M9, C7", "where the 438 wrong predictions go; the SLC-or-TP share"),
    "M9_recall_5class_wilson": ("M9, C7", "per-class recall with Wilson interval for drawings (5 parents) and photos (5 classes)"),
    "M6_balanced_recall": ("M6", "size-balanced reading: 20 aircraft per type (all of smaller types), 200 draws, kNN recall "
                                 "mean and 2.5-97.5 %, shuffled-label chance, probe recall on the first 50 draws"),
    "M11_linked_99_vs_94": ("M11", "the 5 linked patent aircraft that are not matching queries, and why"),
    "M11_counts": ("M11", "665 vs 663, the gallery 665, 1 545 vs 1 541 figures, the images of the base and registers attention runs"),
    "M11_ari_like_for_like": ("M11", "k-means ARI / NMI for 12v12 and 5v5 on the drawings and 5v5 on the photos, both models"),
    "M11_ari_ratios": ("M11", "the printed ratio (5v5 over 12v12) against the like-for-like one (5v5 over 5v5)"),
    "S2_probe_12": ("S2", "probe, drawings 12 classes: bootstrap intervals (aircraft, makers), fold-seed range, 50-shuffle "
                          "chance, probe minus kNN with intervals, the regularisation"),
    "S2_probe_5": ("S2", "the same, drawings 5 parent classes"),
    "S2_probe_photos": ("S2", "the same, photos 5 classes"),
    "S2_probe_shuffles_12": ("S2", "the 50 shuffled-label probe scores (drawings, 12)"),
    "S2_probe_shuffles_5": ("S2", "the 50 shuffled-label probe scores (drawings, 5)"),
    "S2_probe_shuffles_photos": ("S2", "the 50 shuffled-label probe scores (photos)"),
    "S3_k_sensitivity": ("S3", "kNN balanced accuracy for k = 3, 5, 10 with interval and chance; paired k minus 5"),
    "S3_image_rules_paired": ("S3, S5", "every image rule minus the primary rule, paired, aircraft and maker intervals, per-half sign"),
    "S3_image_rules_split_half": ("S3, S5", "the split-half rule run over the image rules (selection frequency, fixed held-out score)"),
    "S4_paired_contrasts": ("S4", "every paired contrast: aircraft and maker bootstrap intervals, maker p and Holm, per-half "
                                  "mean and sign share, the saved interval beside it"),
    "S6_matching_intervals": ("S6, F6.12", "matching top-10 / top-50 with bootstrap and Wilson intervals, by page status, "
                                          "registers and base"),
    "S6_matching_queries": ("S6", "each registers query with its rank and the status of its linked page"),
    "S7_own_pick_per_source": ("S7", "each source's own pick under the split-half rule (layer and token, model, preparation)"),
    "S7_heldout_differences": ("S7", "held-out differences on the same halves: L24 vs L22 CLS, thick vs plain, grey vs colour"),
    "S7_best_heldout_criterion": ("S7", "does the best mean held-out score pick the same setup as the selection frequency?"),
    "S8_halves_missing_classes": ("S8", "halves missing one or more classes; Monte Carlo SE of the L24 CLS frequency; how often "
                                        "both halves of a split agree"),
    "S8_seeds": ("S8", "the split-half rule rerun with seeds 42 (reproduces the saved run), 7 and 123"),
    "S8_table12_corrected": ("S8", "corrected Table 12: fixed L24 CLS held-out score beside the mean held-out score of the winners"),
    "S8_joint_grid_candidates": ("S8", "joint grid, 6 matrices x preparations, through the split-half rule, per candidate"),
    "S8_joint_grid_labels": ("S8", "joint grid per label set: chosen, gap, against the sum of the sequential gaps"),
    "S9_attention_extras": ("S9, M11", "attention: max-patch share, renormalised share, p90, prefix share, argmax in the pad / "
                                       "on white; both models on all their images and on the images both ran on"),
    "S10_audit_wilson": ("S10", "photo-filter audit rows with Wilson intervals"),
    "S10_kappa_by_type": ("S10", "wizard G1 against the whole-patent reading, per G1 type: disagreements and what the reading says"),
    "S10_kappa": ("S10, M8", "kappa of the current wizard labels and of the frozen labels against the whole-patent reading"),
    "S11_d_units": ("S11", "architecture d per aircraft and per figure, main vs non-main figures, one view at a time; view d"),
    "S11_view_ratio": ("S11", "view d / architecture d with a maker bootstrap interval; the d values; the view kNN with chance"),
    "S11_bootstrap_draws": ("S11", "the bootstrap draws behind S11_view_ratio"),
    "S11_umap_figures_by_view": ("S11", "2-D UMAP coordinates of the 1 541 figures with view and G1 type"),
    "figs/FA_forest_paired_contrasts.png": ("M6, S4, S15", "F-A forest plot of every paired contrast, aircraft and maker "
                                                           "intervals, per-half sign share; matched and size-only photo rows"),
    "figs/FB_recall_vs_pool_share.png": ("M3, M6, S15", "F-B recall per class against the class's share of the pool: shared 5 "
                                                         "classes (drawings, photos, matched photos) and the 12 G1 types"),
    "figs/FC_optimism_dumbbell.png": ("M1, S8, S15", "F-C in-sample winner, held-out winner, fixed L24 CLS held out and full "
                                                     "sample, per label set and pool (replaces Table 12)"),
    "figs/FD_class_counts.png": ("S15, M9", "F-D aircraft per G1 type coloured by parent; the shared 5 classes in both sources"),
    "figs/FE_recall_12_wilson.png": ("M9, M6", "F-E 12-class recall with n and Wilson whiskers, under-20 greyed, balanced reading beside"),
    "figs/FF_view_erasure_control.png": ("M4", "F-F before / after the view erasure, both models, with the 200-draw control"),
    "figs/FG_umap_figures_by_view.png": ("S11", "F-G the 1 541 figure embeddings in 2-D coloured by view"),
}


def _ci_s(lo: float, hi: float) -> str:
    return f"[{lo:+.3f}, {hi:+.3f}]"


def _has0(lo: float, hi: float) -> bool:
    return lo <= 0 <= hi


def verdicts(out: Path) -> Dict[str, List[str]]:
    V: Dict[str, List[str]] = {}

    def opt(name):
        f = out / f"{name}.csv"
        return _read(out, name) if f.exists() else None

    m3 = opt("M3_pool_size_and_mix")
    if m3 is not None:
        t = m3.set_index("row")
        full = t.loc["difference: photos full set minus drawings"]
        so = t.loc["difference: photos size only minus drawings"]
        mt = t.loc["difference: photos size + mix (matched) minus drawings"]
        sz = t.loc["decomposition: pool size (full set minus size only)"]
        mx = t.loc["decomposition: class mix at fixed size (size only minus size + mix)"]
        word = lambda r: "no difference detected" if _has0(r.ci_lo, r.ci_hi) else ("photos higher" if r.estimate > 0 else "drawings higher")
        V["M3"] = [
            f"Matched (size + mix) photos minus drawings {mt.estimate:+.3f} {_ci_s(mt.ci_lo, mt.ci_hi)}: {word(mt)}.",
            f"Size only (663 photos at their own mix) minus drawings {so.estimate:+.3f} {_ci_s(so.ci_lo, so.ci_hi)}: {word(so)}; "
            f"full set minus drawings {full.estimate:+.3f} {_ci_s(full.ci_lo, full.ci_hi)}: {word(full)}.",
            f"Decomposition: pool size alone moves the photo score by {sz.estimate:+.3f} {_ci_s(sz.ci_lo, sz.ci_hi)}, the class mix "
            f"at fixed size by {mx.estimate:+.3f} {_ci_s(mx.ci_lo, mx.ci_hi)}. "
            + (("The advantage survives the size-only control and is no longer detected under the drawings' class mix, so "
                "pool size does not explain it; the class mix is the remaining candidate"
                + (", and its own effect at fixed size is detected." if not _has0(mx.ci_lo, mx.ci_hi) else
                   ", but its own effect at fixed size is not significant (interval includes 0), so it is not a demonstrated "
                   "cause: the matched comparison is too wide to settle it."))
               if not _has0(so.ci_lo, so.ci_hi) and _has0(mt.ci_lo, mt.ci_hi) else
               "Read the two differences together; the headline wording follows the matched row."),
            f"Headline wording: \"{'no difference detected' if _has0(mt.ci_lo, mt.ci_hi) else 'the photos carry more'}\" once the "
            "photo pool has the drawings' size and class mix."]
    ho, ctl = opt("M4a_heldout_architecture_by_split"), opt("M4b_control_200_draws")
    vp = opt("M4a_heldout_view_probe")
    if ho is not None and ctl is not None and vp is not None:
        L = []
        for model in ("registers", "base"):
            v = vp[(vp.model == model) & (vp.fit == "held out")].iloc[0]
            L.append(f"{model.capitalize()}: fitted on half of the makers, the erasure takes the view probe from {v.before:.3f} to "
                     f"{v.after:.3f} (chance {v.chance:.2f}), {v.share_of_above_chance_removed:.0%} of the above-chance signal; "
                     "in sample it removes all of it.")
            for lab, short in ((LAB12, "G1 12"), (LAB5, "parent 5")):
                h = ho[(ho.model == model) & (ho.label_set == lab)].set_index("erasure_fit")
                c = ctl[(ctl.model == model) & (ctl.subset == "all") & (ctl.label_set == lab) &
                        (ctl.control == "variance-matched random directions")].iloc[0]
                inside = c.control_lo <= c.erasure_diff <= c.control_hi
                L.append(f"{model} {short}: full-sample erasure {c.erasure_diff:+.3f}; variance-matched control over {int(c.draws)} "
                         f"draws {c.control_mean:+.3f} {_ci_s(c.control_lo, c.control_hi)} "
                         f"({c.share_of_draws_as_costly_as_erasure:.0%} of draws cost as much or more): "
                         + ("the erasure's change is inside the control range: no view-specific effect. " if inside else
                            ("the erasure costs more than the control, so the loss is view-specific. " if c.erasure_diff < c.control_lo
                             else "the erasure costs less than the control. "))
                         + f"Held out (200 splits x 2) the erasure changes the half score by {h.loc['held out', 'mean_diff']:+.3f} "
                           f"on average ({h.loc['held out', 'share_of_halves_lower']:.0%} of halves lower), in sample on the same "
                           f"halves by {h.loc['in sample', 'mean_diff']:+.3f}.")
        V["M4"] = L
    b, g = opt("M6_balanced_recall"), opt("M9_recall_12_wilson")
    if b is not None and g is not None:
        bb, gg = b.set_index("g1"), g.set_index("g1")
        L = []
        rec = []
        for c in ("PTC", "DS", "SRW"):
            r, f = bb.loc[c], gg.loc[c]
            above = r.knn_recall_mean > r.shuffled_recall_p95
            rec.append(above)
            L.append(f"{c}: full sample {int(f.knn_correct)} of {int(f.n_aircraft)} (Wilson {_ci_s(f.knn_wilson_lo, f.knn_wilson_hi).replace('+', '')}); "
                     f"size-balanced {r.knn_recall_mean:.2f} [{r.knn_recall_lo:.2f}, {r.knn_recall_hi:.2f}] against a shuffled 95th "
                     f"percentile of {r.shuffled_recall_p95:.2f}; probe {f.probe_recall:.2f} ({int(f.probe_correct)} of {int(f.n_aircraft)}).")
        rc = bb.loc["RC"]
        L.append(f"For comparison RC (9 aircraft): balanced {rc.knn_recall_mean:.2f}; TB (25): {bb.loc['TB'].knn_recall_mean:.2f}.")
        if all(rec):
            v = "all three are recovered above chance once the classes weigh the same: the zeros are a class-size effect, and the 'propulsion movement' explanation is not supported"
        elif not any(rec):
            v = "all three stay at or below chance when balanced: class size does not explain the zeros, so the explanation is not refuted (it stays untested as a cause)"
        else:
            up = [c for c, a in zip(("PTC", "DS", "SRW"), rec) if a]
            down = [c for c, a in zip(("PTC", "DS", "SRW"), rec) if not a]
            v = (f"{', '.join(up)} {'rises' if len(up) == 1 else 'rise'} above chance once every type weighs the same in "
                 f"the pool (its zero was a class-size effect); {', '.join(down)} {'stays' if len(down) == 1 else 'stay'} at or "
                 "below chance, but rest on 5 and 9 aircraft. The 'propulsion movement' explanation is not supported as a "
                 "statement about all three and should be cut or limited to the smallest types, with their counts")
        L.append("Verdict: " + v + ".")
        w = opt("M9_wrong_predictions")
        if w is not None:
            s = w.set_index("measure")
            L.append(f"Wrong predictions naming SLC or TP: {int(s.loc['wrong predictions that name SLC or TP', 'value'])} of "
                     f"{int(s.loc['wrong predictions (kNN-5, L24 CLS, 12 classes)', 'value'])} = "
                     f"{s.loc['share of wrong predictions that name SLC or TP', 'value']:.1%} (the two classes are "
                     f"{s.loc['share of the aircraft pool that is SLC or TP', 'value']:.1%} of the pool).")
        V["M6 / M9"] = L
    s2 = [opt(f"S2_probe{suf}") for suf in ("_12", "_5", "_photos")]
    s2 = [x for x in s2 if x is not None]
    if s2:
        L = []
        for x in s2:
            r = x.iloc[0]
            L.append(f"{LAB_SHORT[r.label_set]}: probe {r.probe_bal_acc:.3f}, 95 % {_ci_s(r.ci_lo_aircraft, r.ci_hi_aircraft).replace('+', '')} "
                     f"(aircraft) / {_ci_s(r.ci_lo_maker, r.ci_hi_maker).replace('+', '')} (makers), fold seeds "
                     f"{r.fold_seed_min:.3f} to {r.fold_seed_max:.3f}, shuffled labels mean {r.shuffle_mean:.3f} (95th percentile "
                     f"{r.shuffle_p95:.3f}, {int(r.shuffles)} shuffles); probe minus kNN {r.probe_minus_knn:+.3f} "
                     f"{_ci_s(r.diff_ci_lo_maker, r.diff_ci_hi_maker)} (makers)"
                     + (": the probe reads more than the neighbours." if r.diff_ci_lo_maker > 0 else ": no difference detected."))
        L.append("Regularisation: " + s2[0].regularisation.iloc[0] + ".")
        V["S2"] = L
    k, rr = opt("S3_k_sensitivity"), opt("S3_image_rules_paired")
    if k is not None and rr is not None:
        L = []
        for lab, g_ in k.groupby("label_set", sort=False):
            ks = g_[g_.k.astype(str).isin(["3", "5", "10"])]
            ks = ks.set_index(ks.k.astype(str))
            ds = g_[g_.k.astype(str).str.contains("minus")]
            L.append(f"{LAB_SHORT[lab]}: k 3 / 5 / 10 = {ks.loc['3', 'bal_acc']:.3f} / {ks.loc['5', 'bal_acc']:.3f} / "
                     f"{ks.loc['10', 'bal_acc']:.3f}; " + "; ".join(f"{d.k} {d.bal_acc:+.3f} {_ci_s(d.ci_lo, d.ci_hi)}" for d in ds.itertuples())
                     + (": k does not change the reading." if all(_has0(d.ci_lo, d.ci_hi) for d in ds.itertuples())
                        else ": at least one k differs from 5 (interval excludes 0)."))
        for r in rr.itertuples():
            L.append(f"{LAB_SHORT[r.label_set]}, {r.contrast}: {r.diff:+.3f}, aircraft {_ci_s(r.ci_lo_aircraft, r.ci_hi_aircraft)}, "
                     f"makers {_ci_s(r.ci_lo_maker, r.ci_hi_maker)}, same sign in {r.share_of_halves_same_sign:.0%} of halves.")
        sh = opt("S3_image_rules_split_half")
        if sh is not None:
            for lab, g_ in sh.groupby("label", sort=False):
                top = g_.sort_values("selection_freq", ascending=False).iloc[0]
                L.append(f"Split-half rule over the image rules, {LAB_SHORT[lab]}: most chosen {top.candidate} "
                         f"({top.selection_freq:.0%}); " + ", ".join(f"{r.candidate} {r.selection_freq:.0%}" for r in g_.itertuples()) + ".")
        V["S3"] = L
    s4 = opt("S4_paired_contrasts")
    if s4 is not None:
        lost = s4[~s4.apply(lambda r: _has0(r.ci_lo_aircraft, r.ci_hi_aircraft), axis=1)
                  & s4.apply(lambda r: _has0(r.ci_lo_maker, r.ci_hi_maker), axis=1)]
        holm = s4[s4.p_maker_holm < 0.05]
        V["S4"] = [f"{len(s4)} paired contrasts. Intervals that exclude 0 when aircraft are resampled but include it when makers "
                   f"are resampled: {len(lost)} ({'; '.join(lost.contrast) if len(lost) else 'none'}).",
                   f"Surviving Holm at 0.05 on the maker-bootstrap p: {len(holm)} ({'; '.join(holm.contrast) if len(holm) else 'none'})."]
    s6 = opt("S6_matching_intervals")
    if s6 is not None:
        a = s6[(s6.model == "registers") & (s6.direction == "patent to photo")].set_index("page_status")
        bse = s6[(s6.model == "base") & (s6.direction == "patent to photo")].set_index("page_status")
        L = [f"Registers, patent to photo: top 10 {a.loc['all', 'top10']:.1%} [{a.loc['all', 'top10_boot_lo']:.1%}, "
             f"{a.loc['all', 'top10_boot_hi']:.1%}], top 50 {a.loc['all', 'top50']:.1%} [{a.loc['all', 'top50_boot_lo']:.1%}, "
             f"{a.loc['all', 'top50_boot_hi']:.1%}] (random {a.loc['all', 'random_top10']:.1%} / {a.loc['all', 'random_top50']:.1%}); "
             f"base top 50 {bse.loc['all', 'top50']:.1%} [{bse.loc['all', 'top50_boot_lo']:.1%}, {bse.loc['all', 'top50_boot_hi']:.1%}]."]
        for st in ("built", "concept", "unknown"):
            if st in a.index:
                L.append(f"Linked page {st} ({int(a.loc[st, 'queries'])} queries): top 50 {a.loc[st, 'top50']:.1%} "
                         f"[{a.loc[st, 'top50_boot_lo']:.1%}, {a.loc[st, 'top50_boot_hi']:.1%}].")
        V["S6"] = L
    own, seeds, miss, t12, jg, bh, eq = (opt(n) for n in ("S7_own_pick_per_source", "S8_seeds", "S8_halves_missing_classes",
                                                           "S8_table12_corrected", "S8_joint_grid_labels",
                                                           "S7_best_heldout_criterion", "S7_heldout_differences"))
    if own is not None and seeds is not None:
        L = []
        for (ch, src), g_ in own.groupby(["choice", "source"], sort=False):
            L.append(f"Own pick, {ch}, {src}: " + ", ".join(f"{r.candidate} {r.mean_selection_freq:.0%}" for r in g_.itertuples()) + ".")
        if eq is not None:
            for r in eq.itertuples():
                L.append(f"{LAB_SHORT[r.label_set]}, {r.contrast}: {r.mean_diff:+.3f}, positive in {r.share_of_halves_positive:.0%} of halves.")
        a = seeds[(seeds.rule == "S1(a) layer and token") & (seeds["pool"] == "registers only")]
        winners = a.groupby("seed").decision_winner_registers_pool.first()
        L.append("Seeds: the S1(a) decision picks " + ", ".join(f"{w} (seed {s})" for s, w in winners.items())
                 + (": the winner does not change." if winners.nunique() == 1 else ": the winner CHANGES."))
        bsd = seeds[seeds.rule == "S1(b) preparation"].groupby(["seed"]).decision_winner_registers_pool.apply(lambda v: "/".join(v.unique()))
        L.append("S1(b) preparation winners per seed: " + ", ".join(f"seed {s}: {w}" for s, w in bsd.items()) + ".")
        for r in a[a.label_set == LAB12].itertuples():
            L.append(f"Seed {r.seed}, drawings 12: L24 CLS chosen {r.freq_registers_L24_CLS:.1%}, L22 CLS {r.freq_registers_L22_CLS:.1%}.")
        if miss is not None:
            mm = miss[miss.halves_missing_any_class.notna()]
            for r in mm.itertuples():
                L.append(f"{LAB_SHORT[r.label_set]}: {int(r.halves_missing_any_class)} of {int(r.halves)} halves miss at least one "
                         f"class ({r.halves_missing_each_class if isinstance(r.halves_missing_each_class, str) and r.halves_missing_each_class else 'none'}); halves hold {int(r.min_aircraft_in_half)} to "
                         f"{int(r.max_aircraft_in_half)} aircraft.")
            se = miss[miss.l24_cls_selection_freq.notna() & (miss["pool"] == "registers only")]
            for r in se.itertuples():
                L.append(f"Monte Carlo SE of the L24 CLS frequency ({LAB_SHORT[r.label_set]}, {r.l24_cls_selection_freq:.1%}): "
                         f"{r.mc_se_naive:.3f} naive, {r.mc_se_split_clustered:.3f} with the two halves of a split as one cluster; "
                         f"both halves agree in {r.both_halves_of_a_split_agree:.0%} of splits.")
        if t12 is not None:
            for r in t12[t12["pool"] == "registers only"].itertuples():
                L.append(f"Corrected Table 12, {LAB_SHORT[r.label_set]}: L24 CLS held out (fixed) {r.fixed_L24_CLS_heldout:.3f}; "
                         f"mean held-out score of the winners {r.mean_heldout_of_winners:.3f}; in sample {r.winner_in_sample_mean:.3f}; "
                         f"gap {r.optimism_gap_layer:.3f}; {r.preparation} held out {r.fixed_prep_heldout:.3f}, gap {r.optimism_gap_prep:.3f}.")
        if jg is not None:
            for r in jg.itertuples():
                L.append(f"Joint grid ({int(r.candidates)} candidates), {LAB_SHORT[r.label_set]}: most chosen {r.most_chosen}, decision "
                         f"{r.decision_winner}; joint optimism gap {r.joint_optimism_gap:.3f} against {r.sequential_gaps_summed:.3f} "
                         "for the two sequential gaps summed.")
        if bh is not None:
            same = (bh.by_selection_frequency == bh.by_best_mean_heldout_score)
            L.append("Best-held-out-score criterion: " + "; ".join(
                f"{r.choice}: {r.by_best_mean_heldout_score}" + (" (same)" if s else f" (frequency picks {r.by_selection_frequency})")
                for r, s in zip(bh.itertuples(), same)) + ".")
        V["S7 / S8"] = L
    vr, du = opt("S11_view_ratio"), opt("S11_d_units")
    if vr is not None and du is not None:
        t = vr.set_index("measure")
        r = t.loc["view d / architecture d (figures)"]
        L = [f"View d / architecture d on the figures: {r.estimate:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] (maker bootstrap)"
             + (": the view dominates." if r.ci_lo > 1 else ": the interval reaches 1.")]
        a_ = t.loc["architecture d (aircraft, main figure)"]
        f_ = t.loc["architecture d (figures)"]
        L.append(f"Architecture d is {a_.estimate:.2f} [{a_.ci_lo:.2f}, {a_.ci_hi:.2f}] per aircraft (one main figure each) and "
                 f"{f_.estimate:.2f} [{f_.ci_lo:.2f}, {f_.ci_hi:.2f}] per figure (all 1 541).")
        sub = du[du.measure == "architecture d"]
        L.append("Per unit: " + "; ".join(f"{u.unit}: {u.d:.3f}" for u in sub.itertuples()) + ".")
        g_ = {u.unit.split(" (")[0].split(", as")[0]: u.d for u in sub.itertuples()}
        nm = next((v for k, v in g_.items() if k.startswith("non-main")), np.nan)
        mn = next((v for k, v in g_.items() if k.startswith("main figures only")), np.nan)
        L.append(f"Why 0.28 against 0.21: the per-aircraft d uses one main figure per aircraft ({mn:.2f} when the same main "
                 f"figures are scored as figures); the per-figure d adds the non-main figures, whose architecture d is only "
                 f"{nm:.2f}, and pairs drawn in different views. Within one view the architecture d is "
                 + ", ".join(f"{k.split(': ')[1]} {v:.2f}" for k, v in sorted(
                     ((k, v) for k, v in g_.items() if k.startswith("figures of one view")), key=lambda kv: -kv[1]))
                 + ": near or above the per-aircraft value for the informative views, far below it for the weakest. The two "
                   "numbers measure different units, and the per-figure one is diluted by weaker figures and by the view.")
        vk = t.loc["view kNN-5 balanced accuracy (figures, maker held out)"]
        L.append(f"View kNN-5 on the figures {vk.estimate:.3f} [{vk.ci_lo:.3f}, {vk.ci_hi:.3f}] against a shuffled 95th "
                 f"percentile of {vk.chance:.3f}.")
        V["S11"] = L
    s9 = opt("S9_attention_extras")
    if s9 is not None:
        L = []
        for (model, src), g_ in s9[(s9.image_set == "common") & (s9.layer == 24)].groupby(["model", "source"]):
            r = g_.iloc[0]
            L.append(f"{model} {src} L24 (images both ran on, {int(r.n_images)}): max patch {r.max_patch_share_median:.3f} "
                     f"(p90 {r.max_patch_share_p90:.3f}), renormalised over patches {r.renorm_max_share_median:.3f} "
                     f"(p90 {r.renorm_max_share_p90:.3f}), prefix {r.prefix_share_median:.3f}; largest patch in the pad for "
                     f"{r.argmax_in_pad_share:.1%} of images (pad is {r.pad_patch_share_mean:.1%} of patches), on a white patch for "
                     f"{r.argmax_in_white_patch_share:.1%}.")
        V["S9"] = L
    au, kp = opt("S10_audit_wilson"), opt("S10_kappa")
    if au is not None and kp is not None:
        L = [f"{r.what}: {int(r['count'] if isinstance(r, pd.Series) else r.count)} of {int(r.n)}, Wilson [{r.wilson_lo:.2f}, {r.wilson_hi:.2f}]."
             for _, r in au.iterrows()]
        L += [f"kappa {r.kappa:.3f} ({r.labels} against {r.against}, n {int(r.n)}; {r.origin})." for r in kp.itertuples()]
        V["S10"] = L
    lk, ari = opt("M11_linked_99_vs_94"), opt("M11_ari_ratios")
    if lk is not None:
        L = [f"99 vs 94: " + "; ".join(f"{r.id} ({r.family}): {r.explanation}" for r in lk.itertuples()) + "."]
        if ari is not None:
            L += [f"ARI {r.ratio}: {r.value:.1f}" for r in ari.itertuples()]
        V["M11"] = L
    return V


def _base_attention_note() -> str:
    """What the S9 'common' rows cover: the base patent attention was recomputed on the current main set
    (2026-09-30, 665 images), so both models now share every image; before that the base run held 677."""
    nb = len(pd.read_csv(ATT / BASE / "patent_meta.csv"))
    nr = len(pd.read_csv(ATT / REG / "patent_meta.csv"))
    if nb == nr:
        return (f"- Base attention: recomputed on the current main set, so both models are read on the same {nb} patent "
                "images (S9, image_set all = common).")
    return (f"- Base attention: the base run holds {nb} patent images, the registers run {nr}; both models are compared "
            "on the images they share (S9, image_set common).")


def write_index(out: Path = OUT) -> Path:
    V = verdicts(out)
    L = ["# Review fixes: CPU-side numbers and figures (2026-09-30)", "",
         "Answers to the review of the DINOv2 analysis document (docs/embedding_evaluation/analysis_document/review/"
         "CONSOLIDATED_FLAGS.md). Every number is computed by embedding_evaluation/src/review_fixes.py (run: "
         "scripts/review_fixes.py) from the saved embeddings and results, with the protocol's own functions; nothing "
         "outside this folder was written. Model: DINOv2-large with registers, 518 px, layer 24 CLS, kNN-5 with the maker "
         "held out, unless a row says otherwise. Drawings: exp3 main figure; photos: hero image. Seed 42 throughout unless stated.",
         "", "## Verdicts", ""]
    for k, lines in V.items():
        L.append(f"**{k}**")
        L.append("")
        L += [f"- {s}" for s in lines]
        L.append("")
    L += ["## Files", "", "| file | flag | what it holds |", "|---|---|---|"]
    present = sorted(p.relative_to(out).as_posix() for p in list(out.glob("*.csv")) + list((out / "figs").glob("*.png")))
    for f in present:
        key = f[:-4] if f.endswith(".csv") else f
        flag, what = FILES.get(key, ("", "(not described)"))
        L.append(f"| {f} | {flag} | {what} |")
    L += ["", "## Not done here", "",
          "- M5 (crop test on the registers model) and S1 (plain-grey and contrast-only drawing controls) were extracted "
          "by the GPU run, not here. Their embeddings are read here only to add the maker-level interval and the "
          "per-half sign to S4 and Figure F-A (rows 'cropped minus original, photos (registers)', 'plain grey minus "
          "plain', 'contrast only minus plain'); their own write-ups are crop_experiment/RESULTS_registers.md and "
          "setup_selection/SETUP_SELECTION_addendum.md. A row is absent from S4 when its embeddings were missing at build time.",
          "- The frozen-label kappa of the record (0.866, n 803) cannot be recomputed exactly: architecture_ground_truth.csv "
          "now holds 798 rows (6 GT rows of aircraft that became D1 duplicates were removed on 2026-09-17). S10_kappa gives "
          "the recomputed values on the current file and on the analysis set beside the recorded one.",
          _base_attention_note(),
          "- Paired bootstraps hold the predictions fixed (as in the document); the per-half columns carry the pool variability.",
          ""]
    txt = "\n".join(L)
    assert "—" not in txt, "em dash in INDEX.md"
    p = out / "INDEX.md"
    p.write_text(txt, encoding="utf-8")
    return p


# ── the item runner (moved from scripts/review_fixes.py on 2026-09-30; notebook 32 calls it) ─────────────
QUICK_OUT = Path("/tmp/claude-1006/-home-vasco-Vasco-Workspace-Tese-Vasco-Lnx-WSpaces/"
                 "2d3d33f7-3c1c-421a-a475-05015c55ac0b/scratchpad/review_fixes_quick")
ITEMS = {
    "m3": (m3_pool, {"n_draw": 200, "n_boot": 1000}, {"n_draw": 10, "n_boot": 50}),
    "m4": (m4_erasure, {"n_ctrl": 200, "n_split": 200}, {"n_ctrl": 3, "n_split": 3}),
    "m6": (m6_per_class, {"n_draw": 200, "per_class": 20, "n_probe_draws": 50},
           {"n_draw": 5, "per_class": 20, "n_probe_draws": 1}),
    "m11": (m11_counts, {}, {}),
    "s2_12": (lambda **kw: s2_probe([LAB12], **kw), {}, {"n_shuffle": 1, "fold_seeds": (0,), "n_boot": 50}),
    "s2_5": (lambda **kw: s2_probe([LAB5], **kw), {}, {"n_shuffle": 1, "fold_seeds": (0,), "n_boot": 50}),
    "s2_ph": (lambda **kw: s2_probe([PHO5], **kw), {}, {"n_shuffle": 1, "fold_seeds": (0,), "n_boot": 50}),
    "s3": (s3_k_and_rules, {}, {}),
    "s4": (s4_paired, {}, {}),
    "s6": (s6_matching, {}, {"n_boot": 50}),
    "s78": (s78_selection, {}, {"seeds": (42,)}),
    "s9": (s9_attention, {}, {}),
    "s10": (s10_audit_kappa, {}, {}),
    "s11": (s11_view, {"n_boot": 500, "n_boot_aircraft": 1000}, {"n_boot": 2, "n_boot_aircraft": 5}),
    "s11_umap": (lambda **kw: s11_view(umap_only=True), {}, {}),
}
SUFFIX = {"s2_12": "_12", "s2_5": "_5", "s2_ph": "_photos"}



def run_items(items: Sequence[str], quick: bool = False, out: Path | None = None) -> Path:
    """Run review-fix items (keys of ITEMS, plus 'figures' and 'index'; 'all' = every number item) and write
    their CSVs into ``out`` (default OUT; a quick run writes to a scratch folder, never into OUT)."""
    import time
    out = Path(out) if out else (QUICK_OUT if quick else OUT)
    items = list(items)
    if "all" in items:
        items = [k for k in ITEMS if k != "s11_umap"] + [i for i in items if i in ("figures", "index")]
    for it in items:
        if it in ("figures", "index"):
            continue
        fn, full, small = ITEMS[it]
        t = time.time()
        print(f"[review_fixes] {it} ...", flush=True)
        tables = fn(**(small if quick else full))
        if it in SUFFIX:
            tables = {k + SUFFIX[it]: v for k, v in tables.items()}
        if it == "s11_umap":
            tables = {k: v for k, v in tables.items() if k == "S11_umap_figures_by_view"}
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        print(f"[review_fixes] {it} done in {time.time() - t:.0f} s: {', '.join(tables)}", flush=True)
    if "figures" in items:
        for p in make_figures(out):
            print(f"[review_fixes] figure {p}", flush=True)
    if "index" in items:
        print(f"[review_fixes] index {write_index(out)}", flush=True)
    return out
