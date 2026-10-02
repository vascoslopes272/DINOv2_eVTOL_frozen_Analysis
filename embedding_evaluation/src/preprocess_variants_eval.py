"""Evaluation of the preparation variants and the setup-selection report (author request 2026-09-30).

Reads the registers embeddings of the prepared copies written by
eVTOL-Embedding-Extraction/src/preprocess_variants.py:

    drawings  plain | thick1 | thick2   (each stroke side moved out by about 1 / 2 px at the 518 px input)
    photos    colour | grey

and scores them with the protocol's primary score (kNN-5 balanced accuracy, maker held
out) on the primary image rule of each source (patents: exp3 main figure; photos: hero
image), on the same aircraft, at the matrix chosen by S1(a) and at L24 CLS if different.

    paired     variant minus plain (colour), paired bootstrap (1 000 resamples, seed 42,
               the same resamples score both conditions)
    S1(b)      the split-half rule of ``heldout_selection`` over the preparations
    fairness   drawings (chosen preparation) vs photos colour vs photos grey on the shared
               5 classes; difference photo minus drawing by independent bootstrap of the
               two sets (different aircraft, so no pairing)

``write_report`` assembles SETUP_SELECTION.md from these tables and the S1(a) tables
written by scripts/heldout_selection.py. Every number in it is read from a CSV here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import embedding_protocol as P
from . import embedding_reports as ER
from . import heldout_selection as H

REG_TAG = "dinov2-reg-large_518"
OUT = ER.PAT / "3_embedding_evaluation/setup_selection"
DRAW = ["plain", "thick1", "thick2"]
PHOTO = ["colour", "grey"]
L24 = (24, "cls")
SOURCE_OF_LABEL = {"patents G1 12-class": "patents", "patents parent 5-class": "patents",
                   "photos 5-class": "photos"}


def variant_dir(tree: Path, variant: str) -> Path:
    return tree / "2_embedding_extraction/preprocess_variants" / variant / "embeddings" / REG_TAG


def parse_key(name: str) -> P.Key:
    """'registers L24 CLS' -> (24, 'cls')."""
    for k in P.MATRICES:
        if name.endswith(P.mname(k)):
            return k
    raise ValueError(name)


def load() -> Dict[str, Dict]:
    """{source: {"conds": {condition: {key: X}}, "labels": {...}, "makers": ...}} on the primary rule."""
    ER.TAG = REG_TAG
    S = ER.load_patents()
    order = S.aircraft.aircraft_id.tolist()
    g = S.extra["picks"][S.primary_rule].groupby("aircraft_id")["fig_id"].apply(list)
    figs = [g[a] for a in order]
    conds = {"plain": S.rules[S.primary_rule]}
    for v in DRAW[1:]:
        idx, arrays = ER.load_matrices(variant_dir(ER.PAT, v))
        conds[v] = ER.rule_matrix(idx, arrays, figs)
    check = ER.rule_matrix(S.extra["idx"], S.extra["arrays"], figs)
    assert all(np.allclose(check[k], conds["plain"][k]) for k in P.MATRICES), "plain rule rebuilt differently"
    pat = {"conds": conds, "makers": S.makers,
           "labels": {"patents G1 12-class": S.y,
                      "patents parent 5-class": S.aircraft["label5"].to_numpy(dtype=object)}}
    S = ER.load_evtolnews()
    hero = [[h] for h in S.aircraft.hero]
    idx, arrays = ER.load_matrices(variant_dir(ER.EVN, "grey"))
    pho = {"conds": {"colour": S.rules[S.primary_rule], "grey": ER.rule_matrix(idx, arrays, hero)},
           "makers": S.makers, "labels": {"photos 5-class": S.y}}
    return {"patents": pat, "photos": pho}


def _boot_diff(y, pa, pb, rng, y2=None, pb2=None) -> np.ndarray:
    """Bootstrap of balanced accuracy (b) - (a). Paired when ``y2`` is None (one resample
    scores both), otherwise independent resamples of the two sets (a on y, b on y2)."""
    out: List[float] = []
    while len(out) < P.N_BOOT:
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) != len(set(y)):
            continue
        if y2 is None:
            out.append(balanced_accuracy_score(y[i], pb[i]) - balanced_accuracy_score(y[i], pa[i]))
            continue
        j = rng.integers(0, len(y2), len(y2))
        if len(set(y2[j])) != len(set(y2)):
            continue
        out.append(balanced_accuracy_score(y2[j], pb2[j]) - balanced_accuracy_score(y[i], pa[i]))
    return np.asarray(out)


def paired(D: Dict[str, Dict], keys: List[P.Key]) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """kNN of every condition and the paired difference of each variant against plain / colour."""
    rows, drows, preds = [], [], {}
    for src, d in D.items():
        base = "plain" if src == "patents" else "colour"
        for lab, y in d["labels"].items():
            y = np.asarray(y, dtype=object)
            for key in keys:
                for cond, X in d["conds"].items():
                    r = P.knn(X[key], y, d["makers"], n_shuffle=50)
                    preds[(lab, key, cond)] = r["pred"]
                    rows.append({"source": src, "label": lab, "matrix": P.mname(key), "condition": cond,
                                 "n_aircraft": len(y), "bal_acc": r["bal_acc"], "ci_lo": r["ci_lo"],
                                 "ci_hi": r["ci_hi"], "chance_p95": r["chance_p95"]})
                for cond in d["conds"]:
                    if cond == base:
                        continue
                    pa, pb = preds[(lab, key, base)], preds[(lab, key, cond)]
                    diffs = _boot_diff(y, pa, pb, np.random.default_rng(P.SEED))
                    drows.append({"source": src, "label": lab, "matrix": P.mname(key),
                                  "comparison": f"{cond} - {base}",
                                  "diff": balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa),
                                  "ci_lo": float(np.quantile(diffs, 0.025)),
                                  "ci_hi": float(np.quantile(diffs, 0.975)),
                                  "agree": float((pa == pb).mean())})
    return pd.DataFrame(rows), pd.DataFrame(drows), preds


def s1b(D: Dict[str, Dict], key: P.Key) -> Dict[str, pd.DataFrame]:
    """The split-half rule over the preparations, model = registers, matrix = ``key``."""
    scores, choices, sels, pcs, pls = [], {}, [], [], []
    for src, d in D.items():   # per source: each source has its own candidates and makers
        cands = {c: X[key] for c, X in d["conds"].items()}
        sc = H.half_scores(cands, d["labels"], d["makers"])
        sel = H.select(sc)
        pc, pl = H.summarise(sc, sel)
        pcs.append(pc.merge(H.full_scores(cands, d["labels"], d["makers"]), on=["label", "candidate"]))
        scores.append(sc), sels.append(sel), pls.append(pl)
    scores, sel, pc, pl = (pd.concat(x, ignore_index=True) for x in (scores, sels, pcs, pls))
    dec = []
    for src, cands in (("patents", DRAW), ("photos", PHOTO)):
        w, t = H.decide(pc[pc.candidate.isin(cands)], {k: v for k, v in SOURCE_OF_LABEL.items() if v == src})
        choices[src] = w
        dec.append(t.assign(source=src, chosen=t.candidate == w))
    return {"scores": scores, "selections": sel, "candidates": pc, "labels": pl,
            "decision": pd.concat(dec, ignore_index=True), "choices": choices}


def fairness(D: Dict[str, Dict], preds: Dict, draw_prep: str, keys: List[P.Key]) -> pd.DataFrame:
    """Drawings (chosen preparation) vs photos colour / grey on the shared 5 classes."""
    y_d = np.asarray(D["patents"]["labels"]["patents parent 5-class"], dtype=object)
    y_p = np.asarray(D["photos"]["labels"]["photos 5-class"], dtype=object)
    rows = []
    for key in keys:
        pd_ = preds[("patents parent 5-class", key, draw_prep)]
        ba_d = balanced_accuracy_score(y_d, pd_)
        for cond in PHOTO:
            pp = preds[("photos 5-class", key, cond)]
            diffs = _boot_diff(y_d, pd_, None, np.random.default_rng(P.SEED), y2=y_p, pb2=pp)
            ba_p = balanced_accuracy_score(y_p, pp)
            rows.append({"matrix": P.mname(key), "drawings": draw_prep, "photos": cond,
                         "n_drawings": len(y_d), "n_photos": len(y_p),
                         "drawings_bal_acc": ba_d, "photos_bal_acc": ba_p,
                         "diff_photos_minus_drawings": ba_p - ba_d,
                         "ci_lo": float(np.quantile(diffs, 0.025)), "ci_hi": float(np.quantile(diffs, 0.975))})
    return pd.DataFrame(rows)


def run() -> Dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    dec = pd.read_csv(OUT / "s1a_decision.csv")
    key = parse_key(dec.loc[dec.chosen, "candidate"].iloc[0])
    keys = [key] + ([L24] if key != L24 else [])
    D = load()
    knn, diff, preds = paired(D, keys)
    knn.to_csv(OUT / "s2s3_knn.csv", index=False)
    diff.to_csv(OUT / "s2s3_paired.csv", index=False)
    B = s1b(D, key)
    for k in ("scores", "selections", "candidates", "labels", "decision"):
        B[k].to_csv(OUT / f"s1b_{k}.csv", index=False)
    fair = fairness(D, preds, B["choices"]["patents"], keys)
    fair.to_csv(OUT / "fairness.csv", index=False)
    counts = pd.concat([pd.Series(D["patents"]["labels"]["patents parent 5-class"]).value_counts().rename("drawings"),
                        pd.Series(D["photos"]["labels"]["photos 5-class"]).value_counts().rename("photos")],
                       axis=1).reindex(ER.CLASSES5).fillna(0).astype(int)
    counts.rename_axis("class").reset_index().to_csv(OUT / "fairness_class_counts.csv", index=False)
    return {"key": key, "choices": B["choices"]}


# ── the report ───────────────────────────────────────────────────────────────
def _f(v: float) -> str:
    return f"{v:.3f}"


def _ci(v: float, lo: float, hi: float) -> str:
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)])


def write_report() -> Path:
    r = lambda n: pd.read_csv(OUT / n, keep_default_na=False)  # noqa: E731
    a_c, a_l, a_d, a_h = r("s1a_candidates.csv"), r("s1a_labels.csv"), r("s1a_decision.csv"), r("s1a_half_scores.csv")
    knn, diff = r("s2s3_knn.csv"), r("s2s3_paired.csv")
    b_c, b_l, b_d = r("s1b_candidates.csv"), r("s1b_labels.csv"), r("s1b_decision.csv")
    fair, counts = r("fairness.csv"), r("fairness_class_counts.csv")
    labs = list(SOURCE_OF_LABEL)
    short = {"patents G1 12-class": "drawings G1-12", "patents parent 5-class": "drawings 5-class",
             "photos 5-class": "photos 5-class"}
    winner = a_d.loc[a_d.chosen.astype(str) == "True", "candidate"].iloc[0]
    wkey = parse_key(winner)
    choice = {s: b_d.loc[(b_d.source == s) & (b_d.chosen.astype(str) == "True"), "candidate"].iloc[0]
              for s in ("patents", "photos")}
    n_air = a_h.groupby("label").n_aircraft.agg(["min", "max"])
    R = int(a_h.split.max()) + 1

    # S1(a) table: all 12 candidates, selection frequency (12-candidate pool) and fixed held-out score
    t = a_c[a_c["pool"] == "all 12"]
    rows = []
    for c in dict.fromkeys(t.candidate):
        row = {"candidate": c}
        for lab in labs:
            x = t[(t.candidate == c) & (t.label == lab)].iloc[0]
            row[f"{short[lab]} chosen"] = f"{x.selection_freq:.0%}"
            row[f"{short[lab]} held-out"] = _f(x.fixed_heldout_mean)
        rows.append(row)
    s1a_tab = _md(pd.DataFrame(rows))
    lt = []
    for pool in ("all 12", "registers only"):
        for lab in labs:
            x = a_l[(a_l["pool"] == pool) & (a_l.label == lab)].iloc[0]
            lt.append({"pool": pool, "label set": short[lab], "most chosen": x.most_chosen,
                       "held-out score of the chosen [2.5-97.5 %]":
                           f"{x.chosen_heldout_mean:.3f} [{x.chosen_lo:.3f}, {x.chosen_hi:.3f}]",
                       "in-sample score of the winner": _f(x.winner_in_sample_mean),
                       "optimism gap": _f(x.optimism_gap)})
    s1a_lab = _md(pd.DataFrame(lt))
    # registers vs base on the same halves, at the chosen matrix
    rb = []
    for lab in labs:
        h = a_h[a_h.label == lab].pivot_table(index=["split", "half"], columns="candidate", values="score")
        d = h[f"registers {P.mname(wkey)}"] - h[f"base {P.mname(wkey)}"]
        rb.append({"label set": short[lab], "halves": len(d), "mean (registers - base)": f"{d.mean():+.3f}",
                   "halves where registers is higher": f"{(d > 0).mean():.0%}"})
    rb_tab = _md(pd.DataFrame(rb))
    dec_tab = _md(a_d.assign(selection_freq=a_d.selection_freq.map(lambda v: f"{v:.1%}"),
                             fixed_heldout_mean=a_d.fixed_heldout_mean.map(_f))
                  [["candidate", "selection_freq", "fixed_heldout_mean"]]
                  .rename(columns={"selection_freq": "mean selection frequency",
                                   "fixed_heldout_mean": "mean held-out score"}))

    def ptab(src: str) -> str:
        k = knn[knn.source == src]
        rows = []
        for (lab, m), g in k.groupby(["label", "matrix"], sort=False):
            base = "plain" if src == "patents" else "colour"
            b = g[g.condition == base].iloc[0]
            for x in g[g.condition != base].itertuples():
                dd = diff[(diff.label == lab) & (diff.matrix == m) & (diff.comparison == f"{x.condition} - {base}")].iloc[0]
                rows.append({"label set": short[lab], "matrix": m, base: _f(b.bal_acc), "variant": x.condition,
                             "variant score": _f(x.bal_acc),
                             "difference [95 % CI]": _ci(dd["diff"], dd.ci_lo, dd.ci_hi),
                             "same prediction": f"{dd.agree:.0%}"})
        return _md(pd.DataFrame(rows))

    bt = []
    for x in b_c.itertuples():
        bt.append({"label set": short[x.label], "preparation": x.candidate,
                   "chosen": f"{x.selection_freq:.0%}", "held-out": _f(x.fixed_heldout_mean),
                   "full sample": _f(x.full_sample)})
    b_tab = _md(pd.DataFrame(bt))
    bl = _md(pd.DataFrame([{"label set": short[x.label], "most chosen": x.most_chosen,
                            "held-out score of the chosen": _f(x.chosen_heldout_mean),
                            "optimism gap": _f(x.optimism_gap)} for x in b_l.itertuples()]))
    ft = _md(pd.DataFrame([{"matrix": x.matrix, f"drawings ({x.drawings})": _f(x.drawings_bal_acc),
                            "photos": x.photos, "photos score": _f(x.photos_bal_acc),
                            "photos - drawings [95 % CI]": _ci(x.diff_photos_minus_drawings, x.ci_lo, x.ci_hi)}
                           for x in fair.itertuples()]))
    ct = _md(counts)

    # the chosen setup, each line from the tables
    reg_best = {lab: a_c[(a_c["pool"] == "all 12") & (a_c.label == lab)].set_index("candidate") for lab in labs}
    freq_reg = {lab: reg_best[lab].loc[reg_best[lab].index.str.startswith("registers"), "selection_freq"].sum()
                for lab in labs}
    fw = fair[fair.matrix == P.mname(wkey)].set_index("photos")
    grey_survives = fw.loc["grey", "ci_lo"] > 0
    dd = diff[diff.matrix == P.mname(wkey)].set_index(["label", "comparison"])

    def dline(lab, comp):
        x = dd.loc[(lab, comp)]
        return f"{x['diff']:+.3f} [{x.ci_lo:+.3f}, {x.ci_hi:+.3f}]"

    def bsel(src, c):
        return f"{b_d.loc[(b_d.source == src) & (b_d.candidate == c), 'selection_freq'].iloc[0]:.0%}"

    # the prepared sources: radius used, and how many drawings carried colour (thickening also greys them)
    sm = {v: pd.read_csv(ER.PAT / "2_embedding_extraction/preprocess_variants" / v / "sources_manifest.csv")
          for v in DRAW[1:]}
    n_col = int((sm["thick1"].src_max_chroma > 8).sum())
    rad = {v: m.radius.describe() for v, m in sm.items()}
    r0 = int((sm["thick1"].radius == 0).sum())

    setup = [
        ("Model", "facebook/dinov2-with-registers-large",
         "fixed in advance on the attention artifacts of the base model (image-type-justified); held out, the "
         f"registers candidates are chosen in {', '.join(f'{freq_reg[l]:.0%}' for l in labs)} of the "
         f"selections ({', '.join(short[l] for l in labs)}) when both models compete"),
        ("Resolution", "518 px", "fixed in advance: DINOv2's native high-resolution input, 37 x 37 patches "
         "(image-type-justified; not selected on labels)"),
        ("Layer", str(wkey[0]), f"label-selected by the split-half rule: {winner} has the highest mean "
         f"selection frequency ({a_d.loc[a_d.candidate == winner, 'selection_freq'].iloc[0]:.0%})"),
        ("Token", "CLS" if wkey[1] == "cls" else "patch mean", "label-selected with the layer (same rule, same candidate)"),
        ("Drawing preparation", choice["patents"], "label-selected by the split-half rule over plain / thick1 / "
         f"thick2: mean selection frequency {bsel('patents', choice['patents'])}; paired, thick2 minus plain is "
         f"{dline('patents G1 12-class', 'thick2 - plain')} (G1-12)"),
        ("Photo preparation", choice["photos"], "label-selected by the split-half rule over colour / grey: "
         f"selection frequency {bsel('photos', choice['photos'])}; paired, grey minus colour is "
         f"{dline('photos 5-class', 'grey - colour')}, so colour is kept as the untransformed image, not as a better one"
         if dd.loc[('photos 5-class', 'grey - colour')].ci_lo < 0 < dd.loc[('photos 5-class', 'grey - colour')].ci_hi
         else f"selection frequency {bsel('photos', choice['photos'])}"),
    ]
    setup_tab = _md(pd.DataFrame(setup, columns=["choice", "value", "reason"]))

    md = f"""# Setup selection for the DINOv2 chapter

Every number below is computed by `embedding_evaluation/scripts/heldout_selection.py` and
`scripts/preprocess_variants_check.py`; the CSVs beside this file hold the full tables.

## Question

Which layer, token and image preparation should the chapter use, chosen by a rule that does not
report its score on the aircraft it chose on, so another study can repeat the choice?

## Candidate sets (fixed before scoring)

- Model: dinov2-with-registers-large (the chapter model) and dinov2-large (comparison row only). Resolution 518 px.
- Matrix: layer 18 / 22 / 24 x CLS / patch mean (6 per model).
- Drawing preparation: plain; thick1, thick2 (greyscale, then a minimum filter on the original image of radius
  round(k / s) px, s = 518 / long side, so each side of a stroke moves out by about k = 1 or 2 px at the model
  input and a stroke gets about 2k px wider).
- Photo preparation: colour; grey (greyscale, back to RGB).
- Score: kNN-5 balanced accuracy, maker held out, primary image rule (drawings: main figure; photos: hero image).

## Split-half rule

1. {R} random splits of the makers into two halves (seed 42); every aircraft follows its maker.
2. Every candidate is scored on half A with neighbours from half A only (maker held out).
3. The best candidate on A is scored on half B; then the same with A and B swapped ({2 * R} choices).
4. Reported: selection frequency, held-out score of the chosen and of each fixed candidate, optimism gap.
5. Decision: highest mean selection frequency (label sets averaged within a source, sources averaged); tie to the higher held-out score.

A half holds {n_air.loc[labs[0], 'min']} to {n_air.loc[labs[0], 'max']} drawings aircraft and {n_air.loc[labs[2], 'min']} to {n_air.loc[labs[2], 'max']} photo aircraft, so the neighbour pool is about half the full one and
held-out scores sit below full-sample scores; only scores in the same column compare.

## S1(a) model x matrix

Selection frequency when all 12 candidates compete, and the mean held-out score of each candidate as a fixed choice:

{s1a_tab}

{s1a_lab}

Registers minus base at {P.mname(wkey)}, on the same halves:

{rb_tab}

Decision on the 6 registers matrices:

{dec_tab}

## S2 drawing line thickening (registers, main figure, paired on the same aircraft)

{ptab("patents")}

Radius in original pixels: thick1 median {rad['thick1']['50%']:.0f} (range {rad['thick1']['min']:.0f} to {rad['thick1']['max']:.0f},
{r0} small figures left unfiltered), thick2 median {rad['thick2']['50%']:.0f} (range {rad['thick2']['min']:.0f} to {rad['thick2']['max']:.0f}).
{n_col} of the {len(sm['thick1'])} source drawings carry colour (channel spread above 8), so for them the
thick variants also remove colour.

![plain, thick1, thick2 as the model receives them, with a 128 px crop of each](montage_thickening.png)

## S3 greyscale photos (registers, hero image, paired on the same aircraft)

{ptab("photos")}

Difference = variant minus plain (colour); 95 % CI from {P.N_BOOT:} paired bootstrap resamples (seed 42).

## S1(b) preparation choice (registers, {P.mname(wkey)})

{b_tab}

{bl}

## Fairness: drawings vs photos on the shared 5 classes

{ft}

Different aircraft in the two sets, so the interval comes from independent bootstrap resamples of each set.
The photo advantage {"survives" if grey_survives else "does not survive"} greyscale at {P.mname(wkey)}
(interval of photos grey minus drawings {"above" if grey_survives else "not above"} zero). Aircraft per class:

{ct}

## Chosen setup

{setup_tab}

"""
    dst = OUT / "SETUP_SELECTION.md"
    dst.write_text(md, encoding="utf-8")
    print(f"-> {dst}")
    return dst
