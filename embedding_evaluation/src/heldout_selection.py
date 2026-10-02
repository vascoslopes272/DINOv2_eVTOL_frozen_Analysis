"""Held-out (split-half) selection of the embedding setup (author request 2026-09-30).

The protocol's ``layer_choice`` picks the matrix on the same aircraft it then
reports on, so the reported score of the winner is optimistic. This module
replaces it with a split-half rule that another study can repeat:

    1. R = 200 random splits of the MAKERS into two halves (default_rng(42));
       every aircraft follows its maker, so no maker is on both sides.
    2. On half A, score every candidate: kNN-5 balanced accuracy, neighbours
       drawn only from half A, the aircraft's own maker held out
       (``embedding_protocol.neighbours`` / ``vote`` on the half's rows).
    3. Choose the best candidate on A; score THAT candidate on half B
       (neighbours only within B). Then the swap: choose on B, score on A.
    4. Report the selection frequency of each candidate, the mean held-out
       score of the chosen candidate, the held-out score of each fixed
       candidate, and the optimism gap (score of the winner on the half it was
       chosen on minus its score on the other half).

Decision rule (fixed before the numbers were seen): the chosen candidate is the
one with the highest selection frequency, averaged over the label sets of a
source and then over the sources the choice applies to (every source weighs the
same); a tie goes to the higher mean held-out score.

A candidate is anything that gives one vector per aircraft in the same aircraft
order: a (model, matrix) pair for S1(a), a preparation for S1(b).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import embedding_protocol as P

R_SPLITS = 200


def maker_splits(makers: Sequence, r: int = R_SPLITS, seed: int = P.SEED) -> List[np.ndarray]:
    """``r`` boolean masks of half A; the makers are shuffled and cut in two."""
    makers = np.asarray(makers, dtype=object)
    uniq = np.array(sorted(set(makers)), dtype=object)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(r):
        a = set(rng.permutation(uniq)[: len(uniq) // 2])
        out.append(np.array([m in a for m in makers]))
    return out


def half_scores(cands: Dict[str, np.ndarray], labels: Dict[str, Sequence], makers: Sequence,
                r: int = R_SPLITS, seed: int = P.SEED) -> pd.DataFrame:
    """kNN-5 balanced accuracy of every candidate on each half of every split.

    ``cands`` = {candidate: aircraft x dim}; ``labels`` = {label set: labels},
    all in the same aircraft order as ``makers``. One row per split x half x
    candidate x label set."""
    makers = np.asarray(makers, dtype=object)
    ys = {k: np.asarray(v, dtype=object) for k, v in labels.items()}
    sims = {c: P.l2(X) @ P.l2(X).T for c, X in cands.items()}
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # a rare class can be absent from a half
        for s, a in enumerate(maker_splits(makers, r, seed)):
            for half, m in (("A", a), ("B", ~a)):
                i = np.flatnonzero(m)
                for c, S in sims.items():
                    nn = P.neighbours(None, makers[i], S=S[np.ix_(i, i)])
                    for lab, y in ys.items():
                        rows.append({"split": s, "half": half, "candidate": c, "label": lab,
                                     "score": balanced_accuracy_score(y[i], P.vote(nn, y[i])),
                                     "n_aircraft": len(i)})
    return pd.DataFrame(rows)


def select(scores: pd.DataFrame, candidates: Sequence[str] | None = None) -> pd.DataFrame:
    """Choose on one half, score on the other, both directions of every split.
    ``candidates`` restricts the choice to a subset (candidate order breaks ties)."""
    sc = scores if candidates is None else scores[scores.candidate.isin(candidates)]
    order = list(dict.fromkeys(candidates or sc.candidate))
    rows = []
    for (lab, s), g in sc.groupby(["label", "split"], sort=False):
        t = g.pivot(index="candidate", columns="half", values="score").reindex(order)
        for on, off in (("A", "B"), ("B", "A")):
            c = t[on].idxmax()
            rows.append({"label": lab, "split": s, "chosen_on": on, "chosen": c,
                         "in_sample": float(t.loc[c, on]), "held_out": float(t.loc[c, off]),
                         "best_on_other_half": float(t[off].max())})
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame, sel: pd.DataFrame, candidates: Sequence[str] | None = None
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per candidate, per label set) tables.

    per candidate: selection frequency; held-out score as a FIXED candidate
    (its mean over all halves, no selection involved); 2.5-97.5 % of that
    score over the halves.
    per label set: mean held-out score of the chosen candidate and its
    2.5-97.5 % over the selections; mean in-sample score of the winner; the
    optimism gap; the best fixed candidate for reference."""
    sc = scores if candidates is None else scores[scores.candidate.isin(candidates)]
    order = list(dict.fromkeys(candidates or sc.candidate))
    per_c = []
    for lab, g in sc.groupby("label", sort=False):
        s = sel[sel.label == lab]
        freq = s.chosen.value_counts(normalize=True)
        for c in order:
            v = g[g.candidate == c].score
            per_c.append({"label": lab, "candidate": c, "selection_freq": float(freq.get(c, 0.0)),
                          "fixed_heldout_mean": float(v.mean()),
                          "fixed_lo": float(v.quantile(0.025)), "fixed_hi": float(v.quantile(0.975))})
    per_c = pd.DataFrame(per_c)
    per_l = []
    for lab, s in sel.groupby("label", sort=False):
        pc = per_c[per_c.label == lab].set_index("candidate")
        best_fixed = pc.fixed_heldout_mean.idxmax()
        per_l.append({"label": lab, "selections": len(s),
                      "most_chosen": s.chosen.value_counts().idxmax(),
                      "chosen_heldout_mean": float(s.held_out.mean()),
                      "chosen_lo": float(s.held_out.quantile(0.025)),
                      "chosen_hi": float(s.held_out.quantile(0.975)),
                      "winner_in_sample_mean": float(s.in_sample.mean()),
                      "optimism_gap": float((s.in_sample - s.held_out).mean()),
                      "best_fixed": best_fixed,
                      "best_fixed_heldout_mean": float(pc.loc[best_fixed, "fixed_heldout_mean"])})
    return per_c, pd.DataFrame(per_l)


def decide(per_c: pd.DataFrame, source_of_label: Dict[str, str]) -> tuple[str, pd.DataFrame]:
    """The decision rule: mean selection frequency over a source's label sets,
    then over the sources; tie -> higher mean held-out score (same averaging)."""
    t = per_c[per_c.label.isin(source_of_label)].assign(source=lambda d: d.label.map(source_of_label))
    by_src = t.groupby(["source", "candidate"], sort=False)[["selection_freq", "fixed_heldout_mean"]].mean()
    agg = by_src.groupby("candidate", sort=False).mean()
    order = list(dict.fromkeys(t.candidate))
    agg = agg.reindex(order).sort_values(["selection_freq", "fixed_heldout_mean"], ascending=False,
                                         kind="stable")
    return str(agg.index[0]), agg.reset_index()


def full_scores(cands: Dict[str, np.ndarray], labels: Dict[str, Sequence], makers: Sequence
                ) -> pd.DataFrame:
    """The in-sample score on all aircraft (what the old layer rule looked at), for reference."""
    rows = []
    for c, X in cands.items():
        nn = P.neighbours(X, makers)
        for lab, y in labels.items():
            y = np.asarray(y, dtype=object)
            rows.append({"label": lab, "candidate": c,
                         "full_sample": balanced_accuracy_score(y, P.vote(nn, y))})
    return pd.DataFrame(rows)


# ── S1(a) runner (moved from scripts/heldout_selection.py on 2026-09-30; notebook 32 calls it) ──────────
from . import embedding_reports as ER  # noqa: E402

TAGS = {"registers": "dinov2-reg-large_518", "base": "dinov2-large_518"}
OUT = ER.PAT / "3_embedding_evaluation/setup_selection"
SOURCE_OF_LABEL = {"patents G1 12-class": "patents", "patents parent 5-class": "patents",
                   "photos 5-class": "photos"}


def cand(model: str, key: P.Key) -> str:
    return f"{model} {P.mname(key)}"


def grid():
    """{source: (candidates, labels, makers)} over both models, primary rule."""
    out, order = {}, {}
    for model, tag in TAGS.items():
        ER.TAG = tag  # noqa: the loaders read the module constant
        for S in (ER.load_patents(), ER.load_evtolnews()):
            src = "patents" if S.key == "patents" else "photos"
            ids = S.aircraft.aircraft_id.tolist()
            assert order.setdefault(src, ids) == ids, f"{src}: aircraft order differs between tags"
            if src not in out:
                labels = ({"patents G1 12-class": S.y, "patents parent 5-class":
                           S.aircraft["label5"].to_numpy(dtype=object)} if src == "patents"
                          else {"photos 5-class": S.y})
                out[src] = ({}, labels, S.makers)
            X = S.rules[S.primary_rule]
            out[src][0].update({cand(model, k): X[k] for k in P.MATRICES})
    return out


def run_s1a() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reg = [cand("registers", k) for k in P.MATRICES]
    scores, full = [], []
    for src, (cands, labels, makers) in grid().items():
        print(f"[S1a] {src}: {len(makers)} aircraft, {len(set(makers))} makers, {len(cands)} candidates",
              flush=True)
        scores.append(H.half_scores(cands, labels, makers))
        full.append(H.full_scores(cands, labels, makers))
    scores, full = pd.concat(scores, ignore_index=True), pd.concat(full, ignore_index=True)
    scores.to_csv(OUT / "s1a_half_scores.csv", index=False)

    sels, pcs, pls = [], [], []
    for pool, subset in (("all 12", None), ("registers only", reg)):
        sel = H.select(scores, subset)
        pc, pl = H.summarise(scores, sel, subset)
        sels.append(sel.assign(pool=pool))
        pcs.append(pc.assign(pool=pool))
        pls.append(pl.assign(pool=pool))
    sel, pc, pl = (pd.concat(x, ignore_index=True) for x in (sels, pcs, pls))
    pc = pc.merge(full, on=["label", "candidate"], how="left")
    sel.to_csv(OUT / "s1a_selections.csv", index=False)
    pc.to_csv(OUT / "s1a_candidates.csv", index=False)
    pl.to_csv(OUT / "s1a_labels.csv", index=False)
    winner, dec = H.decide(pc[pc.pool == "registers only"], SOURCE_OF_LABEL)
    dec.assign(chosen=dec.candidate == winner).to_csv(OUT / "s1a_decision.csv", index=False)

    fmt = lambda v: f"{v:.3f}"  # noqa: E731
    print(pc.to_string(index=False, float_format=fmt))
    print()
    print(pl.to_string(index=False, float_format=fmt))
    print()
    print(dec.to_string(index=False, float_format=fmt))
    print(f"\n[S1a] matrix chosen (registers only): {winner}\n-> {OUT}")
