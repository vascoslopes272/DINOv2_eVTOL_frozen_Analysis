#!/usr/bin/env python3
"""Does the drawings' architecture signal survive removing the view? (chapter plan N6)

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/view_erasure_check.py

For the base (dinov2-large_518) and the registers (dinov2-reg-large_518) embeddings, L24 CLS:

  1. fit a linear view erasure (src/view_erasure.py) on the whole-aircraft figures of the 663
     analysis aircraft, using ONLY their view4 labels: mean projection, then INLP rounds while
     the maker-grouped view probe is more than 0.05 above chance (at most 10 rounds);
  2. erase every figure, re-normalise, rebuild the "exp3 main figure" aircraft vectors from the
     erased figures (embedding_reports.rule_matrix) and re-score G1 12-class and parent 5-class
     kNN-5 (maker held out), paired bootstrap of erased - original;
  3. controls: the same number of RANDOM orthonormal directions removed instead (once), and a
     variance-matched random subspace (random directions inside the top principal components,
     as much variance as the view directions); the variance each removed subspace carried; the
     view probe when the erasure is fitted on half of the makers and tested on the other half;
     the view kNN and the T6 view / architecture separation before and after; the architecture
     kNN on the aircraft whose main figure is a Perspective view (view held constant by design);
     how often an aircraft's 5 neighbours share its main figure's view.

Output: 1639_LABELLED/3_embedding_evaluation/view_erasure/view_erasure.csv (long format, every
number); the write-up is VIEW_ERASURE.md in the same folder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embedding_protocol as P  # noqa: E402
from src import embedding_reports as ER  # noqa: E402
from src import view_erasure as VE  # noqa: E402

TAGS = ("dinov2-large_518", "dinov2-reg-large_518")
KEY: P.Key = (24, "cls")
N_SHUFFLE = 50
ORIG, ERASED, RAND, RAND_V = "original", "view erasure", "random directions", "variance-matched random"


def row(tag: str, section: str, variant: str, measure: str, value: float, **kw: Any) -> Dict[str, Any]:
    return {"tag": tag, "matrix": P.mname(KEY), "section": section, "variant": variant,
            "measure": measure, "value": value, **kw}


def heldout(X: np.ndarray, y: np.ndarray, makers: np.ndarray) -> Dict[str, Any]:
    """Fit the erasure on the figures of half of the makers, test the view probe on the other half."""
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=P.SEED).split(X, y, makers))
    E = VE.fit(X[tr], y[tr], makers[tr])
    return {"n_test": len(te), "n_dirs": E["Q"].shape[1],
            "before": P.probe(X[te], y[te], makers[te])["bal_acc"],
            "after": P.probe(VE.project(X[te], E["Q"]), y[te], makers[te])["bal_acc"]}


def arch_knn(tag: str, section: str, XA: Dict[str, np.ndarray], y: np.ndarray, makers: np.ndarray
             ) -> List[Dict[str, Any]]:
    """kNN-5 on every variant, and the paired difference of each erasure against the original."""
    rows, pred = [], {}
    for v, X in XA.items():
        print(f"[{tag}] {section}: {v}", flush=True)
        r = P.knn(X, y, makers, n_shuffle=N_SHUFFLE)
        pred[v] = r["pred"]
        rows.append(row(tag, section, v, "bal_acc", r["bal_acc"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                        chance_p95=r["chance_p95"], n=len(y)))
    for v in XA:
        if v != ORIG:
            d = VE.paired(y, pred[ORIG], pred[v])
            rows.append(row(tag, section, f"{v} - original", "paired_diff", d["diff"], ci_lo=d["ci_lo"],
                            ci_hi=d["ci_hi"], agree=d["agree"], n=len(y)))
    return rows


def one_tag(tag: str) -> List[Dict[str, Any]]:
    ER.TAG = tag  # the loader reads the module constant
    S = ER.load_patents()
    rows: List[Dict[str, Any]] = []
    c = S.confound                       # the whole-aircraft figures of the analysis aircraft
    Xf, yv, mf, ya = c["X"][KEY], c["y_conf"], c["makers"], c["y_arch"]
    nf = len(yv)

    print(f"[{tag}] fitting the view erasure on {nf} figures", flush=True)
    E = VE.fit(Xf, yv, mf)
    Q = E["Q"]
    share = VE.variance_share(Xf, Q)["centred"]
    RM = VE.random_matched(Xf, Q.shape[1], share)
    B = {ERASED: Q, RAND: VE.random_basis(Xf.shape[1], Q.shape[1]), RAND_V: RM["Q"]}
    for h in E["history"]:
        rows.append(row(tag, "view probe, erasure steps (figures)", h["step"], "probe_bal_acc",
                        h["probe_bal_acc"], n=nf, n_dirs=h["n_dirs"], chance=E["chance"]))
    rows.append(row(tag, "erasure", ERASED, "n_dirs_removed", Q.shape[1], converged=E["converged"],
                    chance=E["chance"]))
    rows.append(row(tag, "erasure", RAND_V, "top_pcs_sampled", RM["m"], n_dirs=Q.shape[1]))

    # aircraft vectors rebuilt from the erased figure matrix (same code path as the loader)
    idx, arrays = S.extra["idx"], S.extra["arrays"]
    m = S.extra["picks"][S.primary_rule]
    g = m.groupby("aircraft_id")["fig_id"].apply(list)
    figs = [g[x] for x in S.aircraft.aircraft_id]
    X0 = ER.rule_matrix(idx, {KEY: arrays[KEY]}, figs)[KEY]
    assert np.allclose(X0, S.rules[S.primary_rule][KEY])
    XA = {ORIG: X0, **{v: ER.rule_matrix(idx, {KEY: VE.project(arrays[KEY], b)}, figs)[KEY] for v, b in B.items()}}
    # one main figure per aircraft, so this equals projecting the aircraft vectors directly
    assert np.allclose(XA[ERASED], VE.project(X0, Q))
    XF = {ORIG: P.l2(Xf), **{v: VE.project(Xf, b) for v, b in B.items()}}

    # did the erasure work? linear probe, kNN and T6 separation of the view, before and after
    yv_ac = m.set_index("aircraft_id").view4.reindex(S.aircraft.aircraft_id).to_numpy(dtype=object)
    p_same = sum((yv_ac == v).mean() ** 2 for v in set(yv_ac))
    for v in XF:
        print(f"[{tag}] view checks: {v}", flush=True)
        pr = P.probe(XF[v], yv, mf)
        rows.append(row(tag, "view probe (figures)", v, "probe_bal_acc", pr["bal_acc"], n=nf,
                        chance=E["chance"]))
        kn = P.knn(XF[v], yv, mf, n_shuffle=N_SHUFFLE)
        rows.append(row(tag, "view kNN-5 (figures, maker held out)", v, "bal_acc", kn["bal_acc"],
                        ci_lo=kn["ci_lo"], ci_hi=kn["ci_hi"], chance_p95=kn["chance_p95"], n=nf))
        cf = P.confound(XF[v], ya, yv, mf)
        for k in ("arch_d", "conf_d", "conf_over_arch"):
            rows.append(row(tag, "T6 separation (figures)", v, k.replace("conf", "view"), cf[k], n=nf))
        pa = P.probe(XA[v], yv_ac, S.makers)
        rows.append(row(tag, "view probe (aircraft main figures)", v, "probe_bal_acc", pa["bal_acc"],
                        n=len(yv_ac), chance=E["chance"]))
        nn = P.neighbours(XA[v], S.makers)
        rows.append(row(tag, "neighbours sharing the main figure's view (aircraft)", v, "share",
                        float((yv_ac[nn] == yv_ac[:, None]).mean()), chance=p_same, n=len(yv_ac)))
    for name, b in B.items():
        for unit, X in (("figures", Xf), ("aircraft", X0)):
            for k, val in VE.variance_share(X, b).items():
                rows.append(row(tag, f"variance in removed directions ({unit})", name, f"share_{k}", val,
                                n_dirs=b.shape[1]))

    print(f"[{tag}] held-out makers check", flush=True)
    ho = heldout(Xf, yv, mf)
    for k, v in (("before", ORIG), ("after", ERASED)):
        rows.append(row(tag, "view probe, erasure fitted on other makers (figures)", v, "probe_bal_acc",
                        ho[k], n=ho["n_test"], n_dirs=ho["n_dirs"], chance=E["chance"]))

    # the architecture: kNN on the aircraft vectors, paired against the original
    persp = yv_ac == "Perspective"
    for label, y in (("G1 12-class", S.y), ("parent 5-class", S.aircraft["label5"].to_numpy(dtype=object))):
        rows += arch_knn(tag, f"architecture kNN-5, {label}", XA, y, S.makers)
        rows += arch_knn(tag, f"architecture kNN-5, {label}, Perspective main figures only",
                         {v: X[persp] for v, X in XA.items()}, y[persp], S.makers[persp])
    return rows


def main() -> None:
    rows = [r for tag in TAGS for r in one_tag(tag)]
    df = pd.DataFrame(rows)
    out = ER.PAT / "3_embedding_evaluation/view_erasure"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "view_erasure.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(df.drop(columns=["matrix"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
