#!/usr/bin/env python3
"""S1(a): held-out (split-half) choice of model x matrix (see src/heldout_selection.py).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/heldout_selection.py

Candidates: {registers, base} x {L18, L22, L24} x {CLS, patch mean} = 12, on the
primary image rule of each source (patents: exp3 main figure; photos: hero image).
Label sets: patents G1 12-class, patents parent 5-class, photos 5-class.
The model is fixed to registers by the author; the base rows are the comparison, and
the matrix is decided on the 6 registers candidates by ``heldout_selection.decide``.

Outputs: 1639_LABELLED/3_embedding_evaluation/setup_selection/
    s1a_half_scores.csv       every split x half x candidate x label set
    s1a_selections.csv        every choice (12 candidates and registers-only)
    s1a_candidates.csv        selection frequency + fixed held-out score + full-sample score
    s1a_labels.csv            chosen candidate's held-out score, optimism gap
    s1a_decision.csv          the decision rule's table (registers-only) -> the matrix
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embedding_protocol as P  # noqa: E402
from src import embedding_reports as ER  # noqa: E402
from src import heldout_selection as H  # noqa: E402

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
        ER.TAG = tag
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


def main() -> None:
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


if __name__ == "__main__":
    main()
