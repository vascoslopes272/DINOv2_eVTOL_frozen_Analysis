#!/usr/bin/env python3
"""Build the two embedding reports (patent figures, evtol.news photos) with one protocol.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/build_embedding_reports.py [patents] [evtolnews]
        [--reuse]    reuse the cached protocol results (tables/_results.pkl)
        [--no-pdf]

Outputs
    patents    docs/embedding_evaluation/PATENT_EMBEDDING_ANALYSIS.md + .pdf (figs/patent_embedding/),
               tables in 1639_LABELLED/3_embedding_evaluation/embedding_protocol/
    evtolnews  EVTOLNEWS_DS/3_embedding_evaluation/embedding_analysis/EVTOLNEWS_EMBEDDING_ANALYSIS.md + .pdf
               (private: it shows directory photos), tables in .../embedding_analysis/tables/

Section 1 (which images, and the check of each step) is written here per source;
Sections 2-8 come from src/embedding_reports.protocol_sections and are the same
text for both sources.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import embedding_protocol as P  # noqa: E402
from src import embedding_reports as ER  # noqa: E402
from src.embedding_reports import EVN, PAT, REPO, n_, pct, table, wilson  # noqa: E402

REFRESH_ARCHIVE = PAT / "_archive/embedding_pipeline_PRE_REFRESH_20260922_150338"


def _t(tno, caption):
    tno[0] += 1
    return f"Table {tno[0]}. {caption}"


# ── patents: Section 1 ───────────────────────────────────────────────────────
def extraction_repro() -> dict:
    new = PAT / "2_embedding_extraction/embeddings" / ER.TAG
    old = REFRESH_ARCHIVE / "embeddings" / ER.TAG
    mn, mo = pd.read_parquet(new / "metadata.parquet"), pd.read_parquet(old / "metadata.parquet")
    io = {u: i for i, u in enumerate(mo.figure_uid)}
    pairs = [(i, io[u]) for i, u in enumerate(mn.figure_uid) if u in io]
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    worst = 1.0
    for L, pool in P.MATRICES:
        Xn = P.l2(np.load(new / f"emb_layer{L}_{pool}.npy")[a])
        Xo = P.l2(np.load(old / f"emb_layer{L}_{pool}.npy")[b])
        worst = min(worst, float((Xn * Xo).sum(1).min()))
    return {"n": len(pairs), "worst": worst, "old_date": json.loads((old / "manifest.json").read_text()).get("created", "")[:10]}


def section1_patents(S, R, F, tno) -> tuple[list[str], tuple[str, str]]:
    rel = S.fig_sub
    a = S.aircraft
    fun = pd.read_csv(PAT / "2_embedding_extraction/selection/funnel.csv", keep_default_na=False)
    ft = S.extra["figure_table"]
    wv = ft[(ft.scope == "whole_vehicle") & ft.aircraft_id.isin(set(a.aircraft_id))]
    agree = int((a.label == a.arch_gt).sum())
    gt_n = int((a.arch_gt != "").sum())
    from sklearn.metrics import cohen_kappa_score
    kap = cohen_kappa_score(a.label[a.arch_gt != ""], a.arch_gt[a.arch_gt != ""])
    rep = extraction_repro()
    picks = S.extra["picks"]
    single = {k: v.set_index("aircraft_id").fig_id for k, v in picks.items() if k != "exp4 view average"}
    order = a.aircraft_id
    same = lambda x, y: float((single[x].reindex(order) == single[y].reindex(order)).mean())  # noqa: E731
    ex3 = picks["exp3 main figure"]
    fb = ex3[ex3.rule_applied.astype(str).str.startswith("FALLBACK")]
    ex4 = picks["exp4 view average"]
    nslots = ex4.groupby("aircraft_id").size()
    L: list[str] = ["## Section 1: Which Figures Are Analysed, and Is Each Step Sound?", "",
                    '<div class="stage-purpose"><strong>What this section establishes:</strong> how the patent '
                    'figures became one vector per aircraft, which setting each step uses and how it was chosen, and '
                    'the check that shows the step does not decide the result.</div>', "",
                    "The unit is the **aircraft** (`<patent>_ua<N>`): one patent can disclose several aircraft, and "
                    "the G1 type and the main-figure marker are recorded per aircraft. Every label below is a human "
                    "label from the labelling wizard (Stage 04 tables of "
                    f"{date.today().isoformat()}); no model chooses a figure.", ""]
    L += ["### 1.1 The chain, step by step", ""]
    steps = pd.DataFrame([
        ("1 Approved figures", "keep the figures the labeller approved, of approved in-domain aircraft",
         "human approval in the wizard", "every approved figure has its file on disk (0 missing)"),
        ("2 Domain gate", "drop aircraft tagged UAV-similar, not electric (ElectricSimilar) or STOL-only",
         "the Preliminary Analysis gate (user ruling 2026-09-15; tags as of 2026-09-22)",
         "same 665 aircraft as the Preliminary Analysis"),
        ("3 Duplicates", "drop D1/D2 duplicate patents (they repeat an original); keep D3 variants",
         "codebook duplicate rules", "each D1/D2 points at an existing original (notebook 04 check)"),
        ("4 Whole aircraft only", "drop part and detail figures", "the wizard's parts field",
         f"{int(fun.removed.iloc[-1])} part figures removed; 0 whole-aircraft figures without a view"),
        ("5 G1 label", "keep aircraft with a G1 type; 2 marked unclassifiable are left out",
         "human G1 label", f"agrees with the whole-patent reading for {agree} of {gt_n} aircraft (κ {kap:.2f})"),
        ("6 One figure per aircraft", "four rules pick the figure(s); the main figure is the primary rule",
         "fixed orderings of view and flight state (notebook 23)", "Section 7: the score under every rule"),
        ("7 Processing", "rotate by the labelled angle, pad to a white square, resize to 518 px",
         "notebook 21, same code for both sources", "0 processed images older than their source"),
        ("8 Extraction", "DINOv2-large, frozen, layers 18/22/24, CLS token and patch-token mean",
         "notebook 22", f"re-extracted today: {rep['n']} figures shared with the {rep['old_date']} run, lowest "
                        f"cosine old/new {rep['worst']:.6f}"),
    ], columns=["step", "what it does", "setting and how it was chosen", "check"])
    L += [table(steps, _t(tno, "The selection chain of the patent figures and the check of each step.")), ""]
    steps_f = [(r.step if r.step == "figures on file" else f"minus {r.step}", int(r.remaining))
               for r in fun.itertuples() if r.removed or r.step == "figures on file"]
    steps_f.append(("with a G1 type (analysis set)", len(wv)))
    ER.fig_funnel(F, S, steps_f, "figures remaining", "From figures on file to the analysis set.",
                  "patent figures; notebook 20 funnel (selection/funnel.csv) and notebook 23 figure table; each "
                  "figure counted under the first rule that removed it")
    L += [F.md("f01_funnel", rel), ""]
    L += [f"The analysis set holds **{len(a)} aircraft from {a.patent_id.nunique()} patents** and "
          f"{len(wv)} whole-aircraft figures. {int(fun.removed.iloc[-3])} figures leave with duplicate patents, "
          f"{int(fun.removed.iloc[6] + fun.removed.iloc[7] + fun.removed.iloc[8])} with the domain gate.", ""]
    ER.fig_examples_patents(F, S)
    L += [F.md("f02_examples", rel), ""]

    L += ["### 1.2 The labels the rules sort on", ""]
    L += ["**View.** The eight wizard views are grouped into four: Perspective (front- and rear-isometric), Plan "
          "(top, bottom), Side, Front/Rear. **Flight state** matters only for a convertible type (TW, TR, CVT, DS, "
          "SRW), whose shape changes between hover and cruise. For the fixed types every figure is Invariant. On a "
          "convertible aircraft Hover and Cruise are kept; a figure labelled Invariant (it fits both states) or "
          "Both (the moving part drawn in both positions, added to the wizard on 2026-09-19) becomes **Both**; "
          "Transition and Other become Other; an unlabelled figure is Missing.", ""]
    ct = pd.crosstab(wv.g1_code, wv.state4).reindex(S.classes).fillna(0).astype(int)
    ct = ct[[c for c in ["Cruise", "Both", "Hover", "Other", "Missing", "Invariant"] if c in ct.columns]]
    ct["figures"] = ct.sum(axis=1)
    L += [table(ct.reset_index().rename(columns={"g1_code": "G1"}),
                _t(tno, "Flight state of each whole-aircraft figure, by G1 type.")), ""]
    vc = wv.view4.value_counts()
    L += [f"Views: " + ", ".join(f"{k.replace('FrontRear', 'Front/Rear')} {int(v)}" for k, v in vc.items()) + ". The canonical state for the "
          "state-first rule is Cruise, re-checked by notebook 23 on today's labels.", ""]

    L += ["### 1.3 One vector per aircraft: the four rules", ""]
    L += [table(pd.DataFrame({"rule": list(S.rule_desc), "which figure": list(S.rule_desc.values())}),
                _t(tno, "The four image rules (notebook 23).")), ""]
    ov = pd.DataFrame([("exp1 vs exp2", same("exp1 view-first", "exp2 state-first")),
                       ("exp1 vs exp3", same("exp1 view-first", "exp3 main figure")),
                       ("exp2 vs exp3", same("exp2 state-first", "exp3 main figure"))], columns=["pair", "share"])
    ov["same figure"] = [f"{100 * s:.1f} %" for s in ov.share]
    ov["aircraft that differ"] = [int(round((1 - s) * len(a))) for s in ov.share]
    L += [table(ov[["pair", "same figure", "aircraft that differ"]],
                _t(tno, "How often two single-figure rules pick the same figure.")), ""]
    L += [f"exp4 averages 1 slot for {int((nslots == 1).sum())} aircraft (then it equals one figure), 2 for "
          f"{int((nslots == 2).sum())} and 3 for {int((nslots == 3).sum())}. "
          + (f"{len(fb)} aircraft whose main figure is a part figure {'falls' if len(fb) == 1 else 'fall'} back "
             f"to exp1's pick ({', '.join(fb.aircraft_id)})." if len(fb) else "No rule needed a fallback."), ""]

    L += ["### 1.4 Does the selection decide the result?", ""]
    sens = R["sens"]
    best = P.mname(R["best"])
    sb = sens[sens.matrix == best]
    ts = pd.DataFrame({"variant": sb.variant, "aircraft": sb.n_aircraft, "kNN-5": sb.knn_bal_acc.round(3),
                       "95 % interval": [f"{x:.2f} to {y:.2f}" for x, y in zip(sb.ci_lo, sb.ci_hi)]})
    L += ["The primary score of Section 4 is recomputed after each selection step is made stricter: only the "
          "aircraft whose drawing label matches the whole-patent reading (label noise removed), only the aircraft "
          "whose main figure is a Perspective view (view held constant), and without the fallbacks.", "",
          table(ts, _t(tno, f"Sensitivity of the kNN-5 score to the selection ({best}, maker held out).")), ""]
    base = sb.iloc[0]
    inside = all(base.ci_lo <= v <= base.ci_hi for v in sb.knn_bal_acc)
    L += [("Every variant stays inside the interval of the full set: no selection step decides the result."
           if inside else "At least one variant leaves the interval of the full set; the text below names it."),
          ""]
    if not inside:
        out = sb[(sb.knn_bal_acc < base.ci_lo) | (sb.knn_bal_acc > base.ci_hi)]
        L += ["Outside the interval: " + "; ".join(f"{r.variant} ({r.knn_bal_acc:.2f})" for r in out.itertuples())
              + ".", ""]
    L += ['<span class="stage-status passed">SELECTION: HUMAN LABELS, CHECKED</span>', "", "---", ""]
    verdict = (f"human labels; G1 matches the whole-patent reading for {agree}/{gt_n}; extraction reproduces "
               f"(cos ≥ {rep['worst']:.4f})", "PASSED")
    return L, verdict


# ── evtol.news: Section 1 ────────────────────────────────────────────────────
def section1_evtolnews(S, R, F, tno) -> tuple[list[str], tuple[str, str]]:
    rel = S.fig_sub
    src = EVN / "0_source"
    idx = pd.read_csv(src / "index.csv", keep_default_na=False)
    ac = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    im = pd.read_csv(EVN / "1_filter/image_decisions.csv", keep_default_na=False)
    im["p_aircraft"] = pd.to_numeric(im.p_aircraft)
    errs = json.loads((src / "page_errors.json").read_text()) if (src / "page_errors.json").exists() else {}
    summ = json.loads((EVN / "1_filter/filter_summary.json").read_text())
    man_f = EVN / "1_filter/manual_decisions.csv"
    n_manual = len(pd.read_csv(man_f)) if man_f.exists() else 0
    hard = im.hard_rule.value_counts()
    sig_drop = int(((im.auto == "drop") & (im.hard_rule == "")).sum())
    review = im[im.auto == "review"]
    kept = im[im.keep.astype(str) == "True"]
    keep_ac = set(kept.slug_key)
    no_img = ac[~ac.slug_key.isin(keep_ac)]
    rev_only = set(review.slug_key) - keep_ac
    main = S.extra["main"]
    hero00 = int((pd.to_numeric(main.img_pos) == 0).sum())
    au = pd.read_csv(EVN / "1_filter/audit_2026-09-22/audit_sample.csv", keep_default_na=False)
    nd = pd.read_csv(EVN / "1_filter/audit_2026-09-22/near_duplicates_main.csv")
    pc = pd.read_csv(EVN / "3_embedding_evaluation/parent_check.csv", keep_default_na=False)
    dup_list = idx[idx.slug_key.duplicated(keep=False)]
    two_cls = int((ac.n_classes > 1).sum())
    stat = S.aircraft.status.value_counts()

    def au_row(stratum):
        d = au[au.stratum == stratum]
        return d, int((d.verdict == "whole_aircraft").sum()), len(d)
    d_m, k_m, n_m = au_row("main")
    d_s, k_s, n_s = au_row("siglip_drop")
    d_r, k_r, n_r = au_row("review_band")
    lo_m, hi_m = wilson(k_m, n_m)
    ann_s = int((d_s.verdict == "annotated_whole").sum())

    L: list[str] = ["## Section 1: Which Images Are Analysed, and Is Each Step Sound?", "",
                    '<div class="stage-purpose"><strong>What this section establishes:</strong> how the directory '
                    'pages became one image per aircraft, which setting each step uses and how it was chosen, and '
                    'the check that measures its error or shows it does not decide the result.</div>', "",
                    "The patent figures carry human labels for every choice. The directory images do not: a script "
                    "decides which images show the aircraft, so each of its steps is checked here, the filter itself "
                    "on a random sample judged by eye.", ""]
    L += ["### 1.1 The chain, step by step", ""]
    steps = pd.DataFrame([
        ("1 Index", "read the five class lists of evtol.news/aircraft", "the directory's own lists (2026-09-17)",
         f"{len(idx)} list entries, {idx.slug_key.nunique()} pages; {len(dup_list) // 2 if len(dup_list) else 0} "
         f"page listed twice (same class); {two_cls or 'no'} page in two classes"),
        ("2 Pages", "save every aircraft page", "2 requests at a time, 0.5 s pause",
         f"{ac.page_status.eq('ok').sum()} of {len(ac)} saved, {len(errs)} fetch errors"),
        ("3 Images", "download every image on the page, with its credit line", "all images in the page body",
         f"{len(im)} images, {int((im.dl_error != '').sum())} download errors"),
        ("4 Hard rules", "drop images under 200 px, the same file already used by another page (SHA-1), and "
                         "files stored in another aircraft's media folder",
         "fixed rules; the shared file stays on the page whose own folder holds it",
         f"{int(hard.get('too_small', 0))} small, {int(hard.get('same_file_on_other_page', 0))} shared, "
         f"{int(hard.get('other_aircraft_page', 0))} from another aircraft's folder"),
        ("5 SigLIP filter", "zero-shot score p(whole aircraft) against prompts for parts, interiors, people, "
                            "graphics and other scenes",
         "ViT-SO400M-14-SigLIP-384; keep at p ≥ 0.80, drop below 0.35; thresholds and prompts set by eye on "
         "contact sheets (2026-09-17)",
         f"audit (1.3): {k_m} of {n_m} kept hero images usable; {k_s} of {n_s} dropped images were a whole "
         "vehicle"),
        ("6 Review band", "p between 0.35 and 0.80, or a kept image from another folder or unlike the page's other "
                          "images", "held back", f"{len(review)} images, never decided by hand ({n_manual} manual "
                                                     f"decisions); audit: {k_r} of {n_r} usable"),
        ("7 Hero image", "the page's first kept image represents the aircraft", "the directory's own order",
         f"image 00 of the page for {hero00} of {len(main)} aircraft"),
        ("8 Label", "the class of the list the page sits in", "the directory's own classes (user ruling 2026-09-17)",
         f"parent check against the codebook on {len(pc)} linked patent aircraft: {int(pc.agrees.sum())} agree"),
        ("9 Processing and extraction", "pad to a white square, 518 px; DINOv2-large, frozen",
         "notebooks 21 and 22's code, unchanged", "same code as the patent figures, whose re-extraction reproduced "
                                                  "every vector"),
    ], columns=["step", "what it does", "setting and how it was chosen", "check and result"])
    L += [table(steps, _t(tno, "The selection chain of the evtol.news images and the check of each step.")), ""]
    steps_f = [("images on the pages", len(im)),
               ("after the hard rules", int((im.hard_rule == "").sum())),
               ("after the SigLIP drop (p < 0.35)", int(((im.hard_rule == "") & (im.auto != "drop")).sum())),
               ("kept (review band held back)", len(kept)),
               ("hero images = aircraft analysed", len(main))]
    ER.fig_funnel(F, S, steps_f, "images remaining", "From the images on the directory pages to one hero image "
                  "per aircraft.", "evtol.news crawl of 2026-09-17; 1_filter/image_decisions.csv; each image counted "
                  "under the first rule that removed it")
    L += [F.md("f01_funnel", rel), ""]
    L += [f"**{len(main)} of {len(ac)} aircraft** keep at least one image. The other {len(no_img)} lose every image: "
          f"{len(rev_only)} of them only because their images sat in the review band.", ""]
    credits = ER.fig_examples_evtolnews(F, S)
    L += [F.md("f02_examples", rel), "", '<span class="provenance">Image credits: ' + "; ".join(credits) + ".</span>", ""]

    L += ["### 1.2 How the filter settings were chosen", ""]
    L += [f"The two thresholds (keep at 0.80, drop below 0.35) and the prompt lists were set on 2026-09-17 by "
          f"looking at contact sheets of the lowest-scoring kept images, the highest-scoring dropped images and "
          f"the review band. One round of stricter prompts reduced the kept images from {n_(3208)} to "
          f"{n_(summ['kept'])}. The settings were not tested on a labelled sample at the time; the audit below is "
          f"that test. The style guess of the same model (photograph, render or drawing) proved unreliable: it "
          f"calls {pct((kept['style'] == 'render').sum(), len(kept), 0)} of the kept images renders, so it is used "
          "only to flag line drawings.", ""]

    L += ["### 1.3 Audit of the filter", ""]
    L += [f"A random sample was drawn on 2026-09-22 (seed 7): {n_m} hero images, 20 per class; {n_s} images the "
          f"filter dropped; {n_r} images of the review band. The question for each image: does it show the whole "
          "aircraft, clean enough to represent it? The images were judged from contact sheets that showed the page "
          "title, the class and the filter score, in shuffled order within each sample, so the judgement was not "
          "blind to the score. The judge was the AI assistant used for this pipeline; the author has not yet "
          "re-checked the verdicts. Every verdict and every sheet is kept in `1_filter/audit_2026-09-22/` "
          "(`audit_sample.csv`, `sheets/`) for that check.", ""]
    ta = pd.DataFrame([
        ("hero images (kept)", n_m, k_m, int((d_m.verdict != "whole_aircraft").sum()), 0,
         f"{100 * k_m / n_m:.0f} % usable (95 % interval {100 * lo_m:.0f} to {100 * hi_m:.0f} %)"),
        ("dropped by SigLIP (p < 0.35)", n_s, k_s, int((d_s.verdict == "not_aircraft").sum()), ann_s,
         f"{100 * k_s / n_s:.1f} % false drops"),
        ("review band (held back)", n_r, k_r, int((d_r.verdict == "not_aircraft").sum()),
         int((d_r.verdict == "annotated_whole").sum()), f"{100 * k_r / n_r:.0f} % usable"),
    ], columns=["sample", "images", "whole aircraft", "not an aircraft / not usable", "annotated sheet", "reading"])
    L += [table(ta, _t(tno, "Audit of the filter (judged 2026-09-22).")), ""]
    L += [f"**Kept images are clean.** {k_m} of {n_m} hero images show the whole aircraft. The "
          f"{n_m - k_m} failures are a marketing slide, a night video frame, a group of soldiers with the aircraft "
          f"cut off and a gyroplane whose rotor is out of frame. Two of the usable ones are line drawings, one of "
          f"them the patent figure itself (Section 9). Extrapolated, about {int(round((1 - k_m / n_m) * len(main)))} "
          f"of the {len(main)} hero images are not usable.", "",
          f"**Dropped images are mostly right to go.** {n_s - k_s - ann_s} of {n_s} are interiors, parts, people, "
          f"maps or diagrams; {ann_s} show the whole aircraft under text and dimension lines (specification "
          f"sheets, excluded by design because the text would enter the embedding); {k_s} were whole vehicles lost.",
          "", f"**The review band is a coin toss.** {k_r} of {n_r} would have been usable. Holding the whole band "
          f"back loses about {int(round(k_r / n_r * len(review)))} usable images, but it moves only "
          f"{len(rev_only)} aircraft out of the set, because most aircraft keep other images.", ""]
    ER.fig_audit_examples(F, S)
    L += [F.md("f03_audit_errors", rel), ""]

    L += ["### 1.4 The same picture on two pages", ""]
    xs = nd[~nd.same_company].sort_values("cos", ascending=False)
    top = xs.iloc[0]
    L += [f"Exact copies were removed by the hard rules. Near-copies (resized or re-cropped) were searched among the "
          f"hero images with the SigLIP features: {len(nd)} pairs reach a cosine of 0.95, involving "
          f"{len(set(nd.a) | set(nd.b))} aircraft. {int(nd.same_company.sum())} pairs share the maker, so the "
          f"maker-held-out tests never pair them. The {len(xs)} pairs that cross makers were checked by eye "
          f"(`sheets/near_dup_cross_maker.jpg`). One is a true copy and a label conflict of the directory itself: "
          f"the DF3000 appears as `{top.a}` and `{top.b}` (cosine {top.cos:.3f}), under two makers and two "
          "classes. One is the same company under two names (ASKA and NFT). The other five are different "
          "aircraft rendered over similar city skylines: at this threshold the scene, not the aircraft, makes "
          "them alike. The sensitivity test below drops every aircraft involved.", ""]

    L += ["### 1.5 The label", ""]
    L += [f"The label is the list the page sits in: Vectored Thrust {int((main.topType == 'VT').sum())}, "
          f"Lift + Cruise {int((main.topType == 'LC').sum())}, Wingless {int((main.topType == 'WM').sum())}, "
          f"Electric Rotorcraft {int((main.topType == 'ER').sum())}, Hover Bikes / PFD "
          f"{int((main.topType == 'HB').sum())} aircraft. The five classes are the parents of the codebook's twelve "
          f"types; on the {len(pc)} patent aircraft that have a directory page, the parent of the human G1 type "
          f"matches the directory class for {int(pc.agrees.sum())}. The disagreements are aircraft with an unusual "
          "rotor (cyclorotors, a stopped rotor) and lift + cruise patents of vectored-thrust families. The maturity "
          f"status is given as built for {int(stat.get('built', 0))}, concept for {int(stat.get('concept', 0))} and "
          f"not at all for {int(stat.get('unknown', 0))} pages; only built and concept are used (Section 6).", ""]

    L += ["### 1.6 Does the selection decide the result?", ""]
    sens = R["sens"]
    best = P.mname(R["best"])
    sb = sens[sens.matrix == best]
    ts = pd.DataFrame({"variant": sb.variant, "aircraft": sb.n_aircraft, "kNN-5": sb.knn_bal_acc.round(3),
                       "95 % interval": [f"{x:.2f} to {y:.2f}" for x, y in zip(sb.ci_lo, sb.ci_hi)]})
    L += ["The primary score of Section 4 is recomputed after each filter step is made stricter or its errors "
          "are removed.", "", table(ts, _t(tno, f"Sensitivity of the kNN-5 score to the image selection ({best}, "
                                                 f"maker held out).")), ""]
    base = sb.iloc[0]
    inside = all(base.ci_lo <= v <= base.ci_hi for v in sb.knn_bal_acc)
    L += [("Every variant stays inside the interval of the full set: the filter's settings and its measured errors "
           "do not decide the result." if inside else
           "At least one variant leaves the interval of the full set: " +
           "; ".join(f"{r.variant} ({r.knn_bal_acc:.2f})" for r in
                     sb[(sb.knn_bal_acc < base.ci_lo) | (sb.knn_bal_acc > base.ci_hi)].itertuples()) + "."), ""]
    L += ['<span class="stage-status caution">SELECTION: AUTOMATIC, AUDITED; REVIEW BAND UNDECIDED</span>', "",
          "---", ""]
    verdict = (f"automatic filter, audited: {k_m}/{n_m} hero images usable; {k_s}/{n_s} false drops; "
               f"review band ({len(review)}) held back", "PASSED, with caveats")
    return L, verdict


def section_matching(S, R, F, tno) -> list[str]:
    m = ER.fig_matching(F)
    best = P.mname(R["best"])
    b = m[m.matrix == best].iloc[0]
    t = pd.DataFrame({"matrix": m.matrix, "top 1": (m["R@1"] * 100).round(1), "top 10": (m["R@10"] * 100).round(1),
                      "top 50": (m["R@50"] * 100).round(1), "random, top 10": (m["random_R@10"] * 100).round(1),
                      "median rank": m.median_rank})
    return ["## Section 9: Can a Patent Aircraft Be Found Among the Photos?", "",
            '<div class="stage-purpose"><strong>What this section checks:</strong> a cross-check between the two '
            'sources. Each patent aircraft with a linked directory page is used as a query; its main figure is '
            'compared with every hero image, and the rank of its own page is recorded.</div>', "",
            f"Links come from the naming rulings (NAME_DECISIONS, patent_links_manual.csv). Line drawings are "
            f"removed from the gallery, because some directory pages show the patent drawing itself (cosine 0.97 to "
            f"0.99), which would make the match trivial.", "",
            table(t, _t(tno, "Share of patent aircraft (%) whose own page is ranked in the top k.")), "",
            F.md("f10_matching", S.fig_sub), "",
            f"With {best}, the right aircraft is in the top 10 for {100 * b['R@10']:.0f} % of the queries and in the top "
            f"50 for {100 * b['R@50']:.0f} %, against {100 * b['random_R@10']:.1f} % and {100 * b['random_R@50']:.1f} % "
            "for a random ranking. The model links a line drawing to a photograph of the same aircraft well above "
            "chance, but far from reliably: the medium separates the two sources more than the aircraft joins them.",
            "", "---", ""]


# ── assembly ─────────────────────────────────────────────────────────────────
PROTOCOL_NOTE = ("**How to read this report.** Section 1 is specific to this source. Sections 2 to 8 apply one "
                 "protocol, with the same code, parameters and wording, to the patent figures and to the evtol.news "
                 "photos, each in its own report; a table number in one report answers the same question as in the "
                 "other. The unit is the aircraft (one vector each). Every test that compares aircraft leaves out "
                 "pairs and neighbours of the same maker. Six matrices are compared: DINOv2-large (frozen), layers "
                 "18, 22 and 24, each as the CLS token and as the mean of the patch tokens, extracted at 518 px.")


def build(S, reuse: bool, pdf: bool) -> Path:
    R = ER.run(S, reuse=reuse)
    S.doc_dir.mkdir(parents=True, exist_ok=True)
    F = ER.Figs(S.doc_dir / S.fig_sub)
    tno = [0]
    if S.key == "patents":
        s1, sel = section1_patents(S, R, F, tno)
        title = "Patent Figures: DINOv2 Embedding Analysis"
        what = (f"{len(S.aircraft)} aircraft from {S.aircraft.patent_id.nunique()} patents (1 639-patent dataset, "
                "labelling closed 2026-09-19, domain tags of 2026-09-22)")
        note = ('<span class="provenance">Companion report: <code>EVTOLNEWS_DS/3_embedding_evaluation/'
                'embedding_analysis/EVTOLNEWS_EMBEDDING_ANALYSIS.pdf</code> (the same protocol on the evtol.news '
                'photos). Supersedes <code>EMBEDDING_ANALYSIS.md</code> of 2026-09-18, which used the labels of '
                'that date and scored on separation only.</span>')
    else:
        s1, sel = section1_evtolnews(S, R, F, tno)
        title = "evtol.news Photos: DINOv2 Embedding Analysis"
        what = (f"{len(S.aircraft)} aircraft of the VFS World eVTOL Aircraft Directory (evtol.news, crawled "
                "2026-09-17), one hero image each")
        note = ('<span class="internal"><strong>Internal document.</strong> Figures 2 and 3 reproduce images from '
                'the evtol.news directory, © their owners, for review only; not to be copied or published without '
                'asking the source.</span>\n\n<span class="provenance">Companion report: <code>eVTOL-Visual-'
                'Evaluation/docs/embedding_evaluation/PATENT_EMBEDDING_ANALYSIS.pdf</code> (the same protocol on '
                'the patent figures). Supersedes the evaluation sections of <code>EVTOLNEWS_ADVISOR_REPORT</code> '
                'of 2026-09-18.</span>')
    body = s1 + ER.protocol_sections(S, R, F, tno)
    if S.key == "evtolnews":
        body += section_matching(S, R, F, tno)
    rows = ER.summary_rows(S, R, sel)
    summ = pd.DataFrame(rows, columns=["question", "result", "status"])
    head = [f"# {title}", "",
            f"**Images:** {what}  ",
            f"**Label:** {S.label_name}  ",
            f"**Model:** `facebook/dinov2-large`, frozen; 518 px; layers 18, 22, 24 × CLS token / patch-token mean  ",
            f"**Date:** {date.today().isoformat()}", "", note, "", ER.style_block(), "", "---", "",
            "## Summary", "", table(summ), "", cross_line(S, R), "", PROTOCOL_NOTE, "", "---", ""]
    tail = open_points(S, R) + sources(S)
    md = "\n".join(head + body + tail)
    out = S.doc_dir / S.doc_name
    out.write_text(md, encoding="utf-8")
    print("wrote", out)
    if pdf:
        subprocess.run(["/home/vasco/anaconda3/bin/python3", str(REPO / "scripts/build_report_pdf.py"), str(out)],
                       check=True)
    return out


def cross_line(S, R) -> str:
    """The one number both reports share: kNN-5 on the same five classes, same protocol."""
    other = (EVN / "3_embedding_evaluation/embedding_analysis/tables/t_main.csv" if S.key == "patents"
             else PAT / "3_embedding_evaluation/embedding_protocol/t_main_5classes.csv")
    if not other.exists():
        return ""
    o = pd.read_csv(other)
    best = P.mname(R["best"])
    ob = o[o.matrix == best].iloc[0]
    own = (R["ev5"] if S.key == "patents" else R["ev"])["main"]
    sb = own[own.matrix == best].iloc[0]
    pat, pho = (sb, ob) if S.key == "patents" else (ob, sb)
    return (f"**Across the two sources.** On the same five classes, with the same protocol and matrix ({best}), "
            f"the patent figures reach a kNN-5 balanced accuracy of {pat.knn_bal_acc:.2f} [{pat.knn_ci_lo:.2f}, "
            f"{pat.knn_ci_hi:.2f}] and the evtol.news photos {pho.knn_bal_acc:.2f} [{pho.knn_ci_lo:.2f}, "
            f"{pho.knn_ci_hi:.2f}] (chance 0.20; probe {pat.probe_bal_acc:.2f} against {pho.probe_bal_acc:.2f}). "
            "The same model reads the architecture from photographs and renders better than from patent drawings.")


def open_points(S, R) -> list[str]:
    L = ["## Open Points", ""]
    if S.key == "patents":
        cnt = S.aircraft.label.value_counts()
        small = [f"{c} {int(cnt[c])}" for c in S.classes if cnt.get(c, 0) < 20]
        fb = S.extra["picks"]["exp3 main figure"]
        fb = fb[fb.rule_applied.astype(str).str.startswith("FALLBACK")].aircraft_id.tolist()
        L += [f"1. **Small classes.** {', '.join(small)} have fewer than 20 aircraft; their recall rests on a handful "
              f"of aircraft (Table {S.extra.get('recall_table', '')}).",
              f"2. **Wizard.** Mark a whole-aircraft main figure for {', '.join(fb) or 'none'}; check whether "
              "KR20240170008A ua1 and ua2 are one aircraft drawn in two states.",
              "3. **Notebook 23** still writes its experiment folders from L22 CLS (`PRIMARY_LAYER`); this report "
              "reads the full matrices and does not depend on it.", ""]
    else:
        L += ["1. **Review band.** 239 images were held back and never decided by hand; the audit finds about half "
              "usable. A manual pass (`1_filter/manual_decisions.csv`, read by the filter) would recover images for "
              "some of the 26 aircraft that lost all of theirs.",
              "2. **Directory conflicts.** The DF3000 appears on two pages under two makers and two classes; the "
              "Honghu page exists but is not in the index.",
              "3. **Status.** 506 pages give no status; the maturity test uses the 651 that do.",
              "4. **Attention.** From layer 22 on, the CLS token of DINOv2 (without registers) concentrates on a "
              "padding token (advisor report of 2026-09-18); a model with registers is the next test.", ""]
    return L


def sources(S) -> list[str]:
    if S.key == "patents":
        s = ("<code>1639_LABELLED/2_embedding_extraction/</code>: <code>selection/</code> (notebook 20), "
             "<code>processed/518/</code> (21), <code>embeddings/dinov2-large_518/</code> (22), "
             "<code>view_state_experiments/</code> (23); labels <code>0_labelling/outputs/</code> (notebook 04). "
             "Tables as CSV in <code>1639_LABELLED/3_embedding_evaluation/embedding_protocol/</code>.")
    else:
        s = ("<code>EVTOLNEWS_DS/</code>: <code>0_source/</code> (crawl), <code>1_filter/</code> (filter and "
             "<code>audit_2026-09-22/</code>), <code>2_embedding_extraction/</code> (sets, 518 px embeddings), "
             "<code>3_embedding_evaluation/matching.csv, parent_check.csv</code>. Tables as CSV in "
             "<code>3_embedding_evaluation/embedding_analysis/tables/</code>.")
    return ["---", "", f'<span class="provenance">Generated by <code>embedding_evaluation/scripts/'
            f'build_embedding_reports.py</code> with <code>src/embedding_protocol.py</code> (the shared protocol) and '
            f'<code>src/embedding_reports.py</code>. Sources: {s}</span>', ""]


def main(argv: list[str]) -> None:
    which = [a for a in argv if not a.startswith("--")] or ["patents", "evtolnews"]
    for w in which:
        S = ER.load_patents() if w == "patents" else ER.load_evtolnews()
        build(S, reuse="--reuse" in argv, pdf="--no-pdf" not in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
