"""Do the taxonomy scores move under dinov2-with-registers? (chapter plan N2)

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/registers_check.py

Reads the base (dinov2-large_518) and the registers (dinov2-reg-large_518) embeddings of both
sources — the registers tag is written by eVTOL-Embedding-Extraction/scripts/
registers_experiment.py — and compares the protocol's primary score (kNN-5 balanced accuracy,
maker held out) on the primary image rule of each source: patents G1 12-class and parent
5-class (exp3 main figure), photos 5-class (hero image). Every matrix is scored; the primary
L24 CLS comparison also gets a paired bootstrap of the difference (same aircraft, same
neighbour protocol, only the model changes).

Outputs: 1639_LABELLED/3_embedding_evaluation/registers_check/{registers_knn.csv,
registers_paired.csv}; the attention side of N2 is registers_experiment.py's share step.

Moved from scripts/registers_check.py on 2026-09-30 (notebook 32 drives it); that script is now a thin
wrapper with the same command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


from . import embedding_protocol as P  # noqa: E402
from . import embedding_reports as ER  # noqa: E402

BASE_TAG, REG_TAG = "dinov2-large_518", "dinov2-reg-large_518"
KEY: P.Key = (24, "cls")
PAIRED_ORDER = [("patents", "parent 5-class"), ("patents", "G1 12-class"), ("photos", "5-class")]


def comparisons(tag: str):
    """[(source, label, y, makers, {key: X})] for one embedding tag."""
    ER.TAG = tag  # both loaders read the module constant; every other path is shared
    S = ER.load_patents()
    Xp = S.rules[S.primary_rule]
    out = [("patents", "G1 12-class", S.y, S.makers, Xp),
           ("patents", "parent 5-class", S.aircraft["label5"].to_numpy(dtype=object), S.makers, Xp)]
    S = ER.load_evtolnews()
    out.append(("photos", "5-class", S.y, S.makers, S.rules[S.primary_rule]))
    return out


def main() -> None:
    rows, primary = [], {}
    for tag in (BASE_TAG, REG_TAG):
        for source, label, y, makers, X in comparisons(tag):
            for key in P.MATRICES:
                r = P.knn(X[key], y, makers, n_shuffle=50)
                rows.append({"tag": tag, "source": source, "label": label, "matrix": P.mname(key),
                             "bal_acc": r["bal_acc"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                             "chance_p95": r["chance_p95"]})
                if key == KEY:
                    primary[(tag, source, label)] = (np.asarray(y, dtype=object), r["pred"])

    paired, rng = [], np.random.default_rng(P.SEED)
    # one RNG stream for the three label sets, consumed in a FIXED order: the set this loop once iterated
    # followed Python's hash seed, so reruns moved the intervals in the third decimal. PAIRED_ORDER is the
    # order that reproduces the saved registers_paired.csv of 2026-09-30 00:49 (checked 2026-09-30).
    for (source, label) in PAIRED_ORDER:
        y, pb = primary[(BASE_TAG, source, label)]
        _, pr = primary[(REG_TAG, source, label)]
        diffs = []
        for _ in range(P.N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) == len(set(y)):
                diffs.append(balanced_accuracy_score(y[i], pr[i]) - balanced_accuracy_score(y[i], pb[i]))
        paired.append({"source": source, "label": label, "matrix": P.mname(KEY),
                       "diff": balanced_accuracy_score(y, pr) - balanced_accuracy_score(y, pb),
                       "ci_lo": float(np.quantile(diffs, 0.025)), "ci_hi": float(np.quantile(diffs, 0.975)),
                       "agree": float((pb == pr).mean())})

    out = ER.PAT / "3_embedding_evaluation/registers_check"
    out.mkdir(parents=True, exist_ok=True)
    knn = pd.DataFrame(rows)
    pr = pd.DataFrame(paired).sort_values(["source", "label"])
    knn.to_csv(out / "registers_knn.csv", index=False)
    pr.to_csv(out / "registers_paired.csv", index=False)
    fmt = lambda v: f"{v:.3f}"
    print(knn[knn.matrix == P.mname(KEY)].to_string(index=False, float_format=fmt))
    print()
    print(pr.to_string(index=False, float_format=fmt))
    print(f"\n-> {out}")



run = main
