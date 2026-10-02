"""Controls for the drawing thickening test and the grey-with-grey fairness check (review flags S1, S13; 2026-09-30).

Reads the registers embeddings of the two control preparations written by
eVTOL-Embedding-Extraction/src/preprocess_controls.py beside the three of
``preprocess_variants_eval``:

    plain | grey | contrast | thick1 | thick2      (drawings, main figure per aircraft)

grey = greyscale only; contrast = grey plus a pointwise levels curve at the model input
(the darker half of each image's ink to black, footprint unchanged). Same protocol as
``preprocess_variants_eval``: kNN-5 balanced accuracy, maker held out, L24 CLS (the S1(a)
choice), paired bootstrap 1 000 resamples, seed 42, the same resamples for every comparison.

    paired     every variant minus plain; thick1 / thick2 / contrast minus grey (colour held
               equal); thick1 / thick2 minus contrast
    halves     per-half sign consistency on the split-half rule's 200 maker splits x 2 halves:
               share of halves where the variant scores below plain (and below grey)
    S1(b) v2   the split-half preparation choice over the 5 candidates
    fairness   drawings grey vs photos grey on the shared 5 classes (independent bootstraps)

Outputs (new files only; SETUP_SELECTION.md and its CSVs are not touched), in
1639_LABELLED/3_embedding_evaluation/setup_selection/:

    s2_controls_knn.csv, s2_controls_paired.csv, s2_controls_halves.csv
    s1b_v2_{scores,selections,candidates,labels,decision}.csv
    fairness_grey_grey.csv
    SETUP_SELECTION_addendum.md
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
from . import preprocess_variants_eval as V

OUT = V.OUT
DRAW = ["plain", "grey", "contrast", "thick1", "thick2"]
LABELS = ["patents G1 12-class", "patents parent 5-class"]
SHORT = {"patents G1 12-class": "G1 12-class", "patents parent 5-class": "parent 5-class"}
COMPARISONS = [("grey", "plain"), ("contrast", "plain"), ("thick1", "plain"), ("thick2", "plain"),
               ("contrast", "grey"), ("thick1", "grey"), ("thick2", "grey"),
               ("thick1", "contrast"), ("thick2", "contrast")]


def load() -> Dict[str, Dict]:
    D = V.load()                                  # plain / thick1 / thick2, colour / grey photos
    S = ER.load_patents()                         # ER.TAG is the registers tag after V.load
    order = S.aircraft.aircraft_id.tolist()
    g = S.extra["picks"][S.primary_rule].groupby("aircraft_id")["fig_id"].apply(list)
    figs = [g[a] for a in order]
    for v in ("grey", "contrast"):
        idx, arrays = ER.load_matrices(V.variant_dir(ER.PAT, v))
        D["patents"]["conds"][v] = ER.rule_matrix(idx, arrays, figs)
    D["patents"]["conds"] = {c: D["patents"]["conds"][c] for c in DRAW}
    return D


def paired(D: Dict[str, Dict], key: P.Key) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    d = D["patents"]
    rows, drows, preds = [], [], {}
    for lab in LABELS:
        y = np.asarray(d["labels"][lab], dtype=object)
        for cond, X in d["conds"].items():
            r = P.knn(X[key], y, d["makers"], n_shuffle=50)
            preds[(lab, cond)] = r["pred"]
            rows.append({"label": lab, "matrix": P.mname(key), "condition": cond, "n_aircraft": len(y),
                         "bal_acc": r["bal_acc"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "chance_p95": r["chance_p95"]})
        for b, a in COMPARISONS:
            pa, pb = preds[(lab, a)], preds[(lab, b)]
            diffs = V._boot_diff(y, pa, pb, np.random.default_rng(P.SEED))
            drows.append({"label": lab, "matrix": P.mname(key), "comparison": f"{b} - {a}",
                          "diff": balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa),
                          "ci_lo": float(np.quantile(diffs, 0.025)),
                          "ci_hi": float(np.quantile(diffs, 0.975)),
                          "agree": float((pa == pb).mean())})
    return pd.DataFrame(rows), pd.DataFrame(drows), preds


def s1b(D: Dict[str, Dict], key: P.Key) -> Dict[str, pd.DataFrame]:
    d = D["patents"]
    cands = {c: X[key] for c, X in d["conds"].items()}
    labels = {lab: d["labels"][lab] for lab in LABELS}
    sc = H.half_scores(cands, labels, d["makers"])
    sel = H.select(sc)
    pc, pl = H.summarise(sc, sel)
    pc = pc.merge(H.full_scores(cands, labels, d["makers"]), on=["label", "candidate"])
    w, t = H.decide(pc, {lab: "patents" for lab in LABELS})
    return {"scores": sc, "selections": sel, "candidates": pc, "labels": pl,
            "decision": t.assign(source="patents", chosen=t.candidate == w)}


def halves(scores: pd.DataFrame) -> pd.DataFrame:
    """Share of the 400 halves where ``b`` scores below ``a``, per comparison and label set."""
    rows = []
    for lab, g in scores.groupby("label", sort=False):
        h = g.pivot_table(index=["split", "half"], columns="candidate", values="score")
        for b, a in COMPARISONS:
            dd = h[b] - h[a]
            rows.append({"label": lab, "comparison": f"{b} - {a}", "halves": len(dd),
                         "mean_diff": float(dd.mean()), "share_lower": float((dd < 0).mean()),
                         "share_higher": float((dd > 0).mean()), "share_equal": float((dd == 0).mean())})
    return pd.DataFrame(rows)


def fairness_grey(D: Dict[str, Dict], preds: Dict, key: P.Key) -> pd.DataFrame:
    """Drawings vs photos on the shared 5 classes, grey with grey (and the colour pair for reference)."""
    y_d = np.asarray(D["patents"]["labels"]["patents parent 5-class"], dtype=object)
    y_p = np.asarray(D["photos"]["labels"]["photos 5-class"], dtype=object)
    ph = D["photos"]
    rows = []
    for dcond, pcond in (("grey", "grey"), ("plain", "colour"), ("contrast", "grey")):
        pd_ = preds[("patents parent 5-class", dcond)]
        pp = P.knn(ph["conds"][pcond][key], y_p, ph["makers"], n_shuffle=50)["pred"]
        diffs = V._boot_diff(y_d, pd_, None, np.random.default_rng(P.SEED), y2=y_p, pb2=pp)
        ba_d, ba_p = balanced_accuracy_score(y_d, pd_), balanced_accuracy_score(y_p, pp)
        rows.append({"matrix": P.mname(key), "drawings": dcond, "photos": pcond,
                     "n_drawings": len(y_d), "n_photos": len(y_p),
                     "drawings_bal_acc": ba_d, "photos_bal_acc": ba_p,
                     "diff_photos_minus_drawings": ba_p - ba_d,
                     "ci_lo": float(np.quantile(diffs, 0.025)), "ci_hi": float(np.quantile(diffs, 0.975))})
    return pd.DataFrame(rows)


def run() -> P.Key:
    dec = pd.read_csv(OUT / "s1a_decision.csv")
    key = V.parse_key(dec.loc[dec.chosen, "candidate"].iloc[0])
    D = load()
    knn, diff, preds = paired(D, key)
    knn.to_csv(OUT / "s2_controls_knn.csv", index=False)
    diff.to_csv(OUT / "s2_controls_paired.csv", index=False)
    B = s1b(D, key)
    for k in ("scores", "selections", "candidates", "labels", "decision"):
        B[k].to_csv(OUT / f"s1b_v2_{k}.csv", index=False)
    halves(B["scores"]).to_csv(OUT / "s2_controls_halves.csv", index=False)
    fairness_grey(D, preds, key).to_csv(OUT / "fairness_grey_grey.csv", index=False)
    print(diff.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    return key


# ── the addendum ─────────────────────────────────────────────────────────────
def _ci(v: float, lo: float, hi: float) -> str:
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _n(v: int) -> str:
    return f"{v:,}".replace(",", " ")


def write_addendum() -> Path:
    r = lambda n: pd.read_csv(OUT / n, keep_default_na=False)  # noqa: E731
    knn, diff, hv = r("s2_controls_knn.csv"), r("s2_controls_paired.csv"), r("s2_controls_halves.csv")
    bc, bl, bd = r("s1b_v2_candidates.csv"), r("s1b_v2_labels.csv"), r("s1b_v2_decision.csv")
    fair, ink = r("fairness_grey_grey.csv"), r("s2_controls_ink.csv")
    sm = {v: pd.read_csv(ER.PAT / "2_embedding_extraction/preprocess_variants" / v / "sources_manifest.csv",
                         keep_default_na=False) for v in ("grey", "contrast")}
    matrix = knn.matrix.iloc[0]
    n_col = int((sm["grey"].src_max_chroma > 8).sum())
    thr = sm["contrast"].threshold.astype(int)
    n_left = int((thr < 0).sum())
    frac_changed = (sm["contrast"].n_changed / sm["contrast"].n_ink.clip(lower=1))[thr >= 0]
    D = diff.set_index(["label", "comparison"])
    HV = hv.set_index(["label", "comparison"])

    def tab(comps: List[str]) -> str:
        head = ("| label set | comparison | difference [95 % CI] | halves where the first is lower | "
                "same prediction |\n|---|---|---|---|---|")
        lines = [head]
        for lab in LABELS:
            for c in comps:
                x, h = D.loc[(lab, c)], HV.loc[(lab, c)]
                lines.append(f"| {SHORT[lab]} | {c} | {_ci(x['diff'], x.ci_lo, x.ci_hi)} | "
                             f"{h.share_lower:.0%} | {x.agree:.0%} |")
        return "\n".join(lines)

    kt = ["| preparation | " + " | ".join(f"{SHORT[l]} kNN [95 % CI]" for l in LABELS) + " |",
          "|---|" + "---|" * len(LABELS)]
    for c in DRAW:
        cells = []
        for lab in LABELS:
            x = knn[(knn.label == lab) & (knn.condition == c)].iloc[0]
            cells.append(f"{x.bal_acc:.3f} [{x.ci_lo:.3f}, {x.ci_hi:.3f}]")
        kt.append(f"| {c} | " + " | ".join(cells) + " |")
    kt = "\n".join(kt)

    it = ["| preparation | ink mass (mean darkness) | footprint (share of pixels below 250) | "
          "solid share (below 128) |", "|---|---|---|---|"]
    for x in ink.itertuples():
        it.append(f"| {x.preparation} | {x.ink_mass_mean:.4f} | {x.footprint_mean:.4f} | {x.solid_share_mean:.4f} |")
    it = "\n".join(it)

    bt = ["| label set | preparation | chosen | held-out (fixed) | full sample |", "|---|---|---|---|---|"]
    for x in bc.itertuples():
        bt.append(f"| {SHORT[x.label]} | {x.candidate} | {x.selection_freq:.1%} | "
                  f"{x.fixed_heldout_mean:.3f} | {x.full_sample:.3f} |")
    bt = "\n".join(bt)
    dt = ["| preparation | mean selection frequency | mean held-out score | chosen |", "|---|---|---|---|"]
    for x in bd.itertuples():
        dt.append(f"| {x.candidate} | {x.selection_freq:.1%} | {x.fixed_heldout_mean:.3f} | {x.chosen} |")
    dt = "\n".join(dt)
    choice = bd.loc[bd.chosen.astype(str) == "True", "candidate"].iloc[0]
    pg_share = float(bd.set_index("candidate").loc[["plain", "grey"], "selection_freq"].sum())
    agree_pg = " and ".join(f"{D.loc[(l, 'grey - plain')].agree:.0%}" for l in LABELS)
    I = ink.set_index("preparation")
    ink_c = I.loc["contrast", "ink_mass_mean"] - I.loc["grey", "ink_mass_mean"]
    ink_t1 = I.loc["thick1", "ink_mass_mean"] - I.loc["grey", "ink_mass_mean"]
    fp_t1 = I.loc["thick1", "footprint_mean"] - I.loc["grey", "footprint_mean"]
    gap = "; ".join(f"{SHORT[x.label]}: most chosen {x.most_chosen}, optimism gap {x.optimism_gap:.3f}"
                    for x in bl.itertuples())

    ft = ["| drawings | photos | drawings kNN | photos kNN | photos - drawings [95 % CI] |", "|---|---|---|---|---|"]
    for x in fair.itertuples():
        ft.append(f"| {x.drawings} | {x.photos} | {x.drawings_bal_acc:.3f} | {x.photos_bal_acc:.3f} | "
                  f"{_ci(x.diff_photos_minus_drawings, x.ci_lo, x.ci_hi)} |")
    ft = "\n".join(ft)

    # verdicts, each read from the tables
    def excl(lab, c):   # interval below zero
        return D.loc[(lab, c)].ci_hi < 0

    def inside(lab, c):
        x = D.loc[(lab, c)]
        return x.ci_lo <= 0 <= x.ci_hi

    def lower_share(c):
        v = [HV.loc[(lab, c)].share_lower for lab in LABELS]
        return min(v), max(v)

    def rng_txt(lo, hi):
        return f"{lo:.0%}" if round(lo, 2) == round(hi, 2) else f"{lo:.0%} to {hi:.0%}"

    v = []
    g_in = all(inside(l, "grey - plain") for l in LABELS)
    lo, hi = lower_share("grey - plain")
    v.append(f"Colour loss: grey minus plain is {' and '.join(_ci(D.loc[(l, 'grey - plain')]['diff'], D.loc[(l, 'grey - plain')].ci_lo, D.loc[(l, 'grey - plain')].ci_hi) for l in LABELS)} "
             f"(G1 12-class, parent 5-class); grey is lower than plain in {rng_txt(lo, hi)} of halves. "
             + ("Removing colour alone does not move the score measurably." if g_in
                else "Removing colour alone moves the score."))
    c_in = all(inside(l, "contrast - plain") for l in LABELS)
    lo, hi = lower_share("contrast - plain")
    v.append(f"Darkening in place (contrast): contrast minus plain is {' and '.join(_ci(D.loc[(l, 'contrast - plain')]['diff'], D.loc[(l, 'contrast - plain')].ci_lo, D.loc[(l, 'contrast - plain')].ci_hi) for l in LABELS)}; "
             f"contrast is lower than plain in {rng_txt(lo, hi)} of halves. "
             + ("Darkening the strokes without widening them does not move the score measurably." if c_in
                else "Darkening the strokes without widening them moves the score."))
    for t in ("thick1", "thick2"):
        lo, hi = lower_share(f"{t} - grey")
        ex = [excl(l, f"{t} - grey") for l in LABELS]
        v.append(f"Dilation, colour held equal: {t} minus grey is "
                 f"{' and '.join(_ci(D.loc[(l, f'{t} - grey')]['diff'], D.loc[(l, f'{t} - grey')].ci_lo, D.loc[(l, f'{t} - grey')].ci_hi) for l in LABELS)}; "
                 f"{t} is lower than grey in {rng_txt(lo, hi)} of halves"
                 + (" (interval below zero on both label sets)." if all(ex)
                    else " (interval below zero on one label set)." if any(ex) else " (intervals include zero)."))
    for t in ("thick1", "thick2"):
        c = f"{t} - contrast"
        lo, hi = lower_share(c)
        v.append(f"Widening beyond darkening: {c} is "
                 f"{' and '.join(_ci(D.loc[(l, c)]['diff'], D.loc[(l, c)].ci_lo, D.loc[(l, c)].ci_hi) for l in LABELS)}; "
                 f"{t} is lower than contrast in {rng_txt(lo, hi)} of halves"
                 + (" (intervals include zero)." if all(inside(l, c) for l in LABELS) else "."))
    colour_none = g_in
    dark_hurts = (not c_in) and all(HV.loc[(l, "contrast - plain")].share_lower >= 0.75 for l in LABELS)
    widen_none = all(inside(l, f"{t} - contrast") for l in LABELS for t in ("thick1", "thick2"))
    t2_hurts = any(excl(l, "thick2 - grey") for l in LABELS) and \
        all(HV.loc[(l, "thick2 - grey")].share_lower >= 0.75 for l in LABELS)
    t1_hurts = any(excl(l, "thick1 - grey") for l in LABELS)
    if colour_none and dark_hurts and widen_none:
        verdict = ("Changing how the strokes are rendered at the model input lowers the score, whether they are "
                   "darkened in place or widened; colour removal plays no part, and widening adds no measurable loss "
                   "beyond darkening, so the loss cannot be attributed to dilation as such"
                   + ("; with colour held equal, thick2 (2 input px per side) lowers the score and thick1 (1 px) does "
                      "not measurably" if t2_hurts and not t1_hurts else "")
                   + ". Keep the lines as drawn; a gentler change is untested.")
    elif colour_none and t2_hurts and not dark_hurts:
        verdict = ("Dilation of two input pixels per side lowers the score with colour and darkening accounted for"
                   + ("; one pixel per side also does." if t1_hurts else "; one pixel per side is not measurably lower."))
    else:
        verdict = "The controls do not isolate a single cause (see the tables)."
    fg = fair[(fair.drawings == "grey") & (fair.photos == "grey")].iloc[0]
    fair_line = (f"Grey drawings against grey photos: photos minus drawings "
                 f"{_ci(fg.diff_photos_minus_drawings, fg.ci_lo, fg.ci_hi)}; the photo advantage "
                 + ("survives with both sources in greyscale." if fg.ci_lo > 0
                    else "is not distinguishable from zero with both sources in greyscale."))

    md = f"""# Setup selection addendum: thickening controls and grey-with-grey fairness

Review flags S1 and S13 (2026-09-30). Computed by `embedding_evaluation/scripts/preprocess_controls_check.py`
from the embeddings written by `eVTOL-Embedding-Extraction/scripts/preprocess_controls.py`; every number is
read from the CSVs beside this file. SETUP_SELECTION.md and its CSVs are unchanged.

## Why

The thick variants change three things at once: they greyscale the drawing ({n_col} of the
{_n(len(sm['grey']))} drawings carry colour, channel spread above 8), they darken strokes that reach the model
input as faint grey, and they widen the strokes. Two controls, through the same pipeline (source copy,
`process_all` at 518 px, registers extraction), separate them.

## The two controls

- **grey**: the original, transparency composited over white, converted to greyscale and back to RGB.
  No spatial filter.
- **contrast**: grey, then a pointwise levels curve on the grey 518 px model input. Per image, with v the
  grey level (0 to 255): ink = pixels with v below 250; t = the median of v over the ink pixels, clipped to
  [64, 224]; v becomes 0 where v is at most t, and is unchanged elsewhere. The darker half of each image's
  ink turns solid black; the lighter half (the anti-aliased fringe) and the paper keep their values, so the
  set of non-white pixels is exactly the grey input's: darker strokes, not wider ones. The curve runs at
  the model input because most source drawings are bi-level scans whose strokes only turn grey when the
  resize to 518 px averages them. Threshold: median {int(thr[thr >= 0].median())} (range {int(thr[thr >= 0].min())}
  to {int(thr.max())}); {n_left} images with fewer than 50 ink pixels left as grey; per image, a median
  {frac_changed.median():.0%} of the ink pixels turn black.

Ink at the 518 px input, mean over all {_n(int(ink.n_figures.iloc[0]))} drawings:

{it}

contrast adds {ink_c:.4f} of ink mass to grey with the footprint unchanged; thick1 adds {ink_t1:.4f} and grows
the footprint by {fp_t1:.4f}. The curve is a hard threshold, so besides darkening it also removes the
anti-aliasing of the darker half of the ink (jagged, bi-level strokes; halftone and hatching turn into
black dots, see the montage): it tests "strokes rendered differently, same footprint", not a pure change
of darkness.

## kNN per preparation (registers, {matrix}, main figure, maker held out)

{kt}

## Paired differences against plain

{tab(['grey - plain', 'contrast - plain', 'thick1 - plain', 'thick2 - plain'])}

## Dilation with colour held equal, and widening against darkening in place

{tab(['thick1 - grey', 'thick2 - grey', 'contrast - grey', 'thick1 - contrast', 'thick2 - contrast'])}

Difference = first minus second; 95 % CI from {_n(P.N_BOOT)} paired bootstrap resamples (seed 42), the same
resamples for every comparison. "Halves where the first is lower" = share of the {int(hv.halves.iloc[0])} halves
(200 maker splits, seed 42) where the first preparation scores below the second, neighbours within the half.

## S1(b) v2: preparation choice over the 5 candidates (registers, {matrix})

{bt}

{dt}

Chosen: **{choice}**. {gap}. plain and grey give the same prediction for {agree_pg} of aircraft, so they
split the votes that either would take alone: together they are chosen in {pg_share:.1%} of the selections.

## Fairness, grey with grey (shared 5 classes, independent bootstraps)

{ft}

## Verdicts

{chr(10).join('- ' + s for s in v)}
- **{verdict}**
- {fair_line}

![plain, grey, contrast, thick1, thick2 as the model receives them, with a 128 px detail of each](montage_thickening_v2.png)

Montage rows: the same 8 figures as montage_thickening.png (the most frequent G1 classes, one random main
figure each, seed 42); each caption sits under its row.
"""
    dst = OUT / "SETUP_SELECTION_addendum.md"
    dst.write_text(md, encoding="utf-8")
    print(f"-> {dst}")
    return dst
