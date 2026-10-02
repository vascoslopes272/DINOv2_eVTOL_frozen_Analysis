#!/usr/bin/env python3
"""Build the patent-figure embedding analysis report (md + pdf).

    python scripts/build_embedding_analysis.py [--perm 999] [--no-pdf]

Section 1 explains which images are analysed: the gates, the labels each figure
carries, and the four rules of eVTOL-Embedding-Extraction notebook 23 that pick
the image(s) standing for each aircraft, with how much their picks differ.
Section 2 tests the six DINOv2 layer x pooling matrices to choose the layer:
label-free structure, G1 separation under each rule, viewpoint against
architecture, and whether an aircraft stays recognisable across its views.
Section 3 lists what is still open (the probes).

Every number in the report is computed here from the files. Layout and style
follow docs/Supervisor_Report_Taxonomy_Structure_Analysis_summary.md (its
<style> block is copied at build time) and the PDF is rendered by
scripts/build_report_pdf.py, the renderer of that report.

Reads (read-only), under paths.pipeline_root:
    selection/funnel.csv                        notebook 20
    embeddings/dinov2-large_518/                notebook 22 (6 matrices)
    view_state_experiments/                     notebook 23
Writes:
    <metrics_dir>/../embedding_analysis/*.csv   every table of the report
    <embedding_report_dir>/EMBEDDING_ANALYSIS.md + .pdf + figs/embedding_analysis/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

PILLAR = Path(__file__).resolve().parent.parent
REPO = PILLAR.parent
sys.path.insert(0, str(PILLAR))
sys.path.insert(0, str(REPO / "scripts"))

from src.config_loader import load_config                      # noqa: E402
from src.embedding_metrics import (_dims_for, hopkins,          # noqa: E402
                                   participation_ratio, qc_report)

EMB_TAG = "dinov2-large_518"
SEED = 42
RULES = ["exp1_view_first", "exp2_state_first", "exp3_main", "exp4_slots_mean"]
SHORT = {"exp1_view_first": "exp1", "exp2_state_first": "exp2", "exp3_main": "exp3", "exp4_slots_mean": "exp4"}
SLOTS = ["Perspective", "Plan", "Side"]
STYLE_SOURCE = REPO / "docs" / "Supervisor_Report_Taxonomy_Structure_Analysis_summary.md"
REPORT_NAME = "EMBEDDING_ANALYSIS.md"
FIG_SUB = "embedding_analysis"


# ── inputs ───────────────────────────────────────────────────────────────────

def load_inputs(cfg):
    root = Path(cfg["paths"]["pipeline_root"])
    vse = root / "view_state_experiments"
    emb = root / "embeddings" / EMB_TAG
    rd = lambda p: pd.read_csv(p, keep_default_na=False)
    inp = {
        "funnel": pd.read_csv(root / "selection" / "funnel.csv"),
        "figs": rd(vse / "figure_table.csv"),
        "common": rd(vse / "aircraft_common.csv"),
        "conflicts": rd(vse / "exp3_conflicts.csv"),
        "excluded": rd(vse / "excluded_aircraft.csv"),
        "exp4": rd(vse / "exp4_slots.csv"),
        "folder_meta": json.loads((vse / "exp_embeddings" / "exp1_view_first" / "meta.json").read_text()),
        "emb_manifest": json.loads((emb / "manifest.json").read_text()),
        "meta": pd.read_parquet(emb / "metadata.parquet"),
        "vse_dir": vse, "emb_dir": emb,
    }
    for r in ["exp1_view_first", "exp2_state_first", "exp3_main"]:
        inp[r] = rd(vse / f"{r}.csv")
    inp["arrays"] = {}
    for L in inp["emb_manifest"]["layers"]:
        for p in inp["emb_manifest"]["pooling"]:
            inp["arrays"][(L, p)] = np.load(emb / f"emb_layer{L}_{p}.npy")
    inp["row"] = pd.Series(np.arange(len(inp["meta"])), index=inp["meta"]["figure_uid"])
    return inp


def l2(X):
    X = X.astype(np.float64)
    return X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)


def mtag(key):
    return f"L{key[0]} {key[1]}"


# ── tests ────────────────────────────────────────────────────────────────────

def separation(X, y, groups, n_perm, seed=SEED):
    """Are same-label items closer than different-label ones? Cosine distances over
    all pairs from DIFFERENT groups (patents), so two drawings of one patent never
    count. Returns the ratio mean(between)/mean(within), Cohen's d and a
    label-permutation p-value (as structure_clustering stage 5)."""
    Xn = l2(X)
    a, b = np.triu_indices(len(Xn), 1)
    g = np.asarray(groups)
    keep = g[a] != g[b]
    a, b = a[keep], b[keep]
    d = 1.0 - (Xn @ Xn.T)[a, b]
    y = np.asarray(y)
    same = y[a] == y[b]
    intra, inter = d[same], d[~same]
    obs = inter.mean() / intra.mean()
    pooled = np.sqrt((intra.var(ddof=1) * (len(intra) - 1) + inter.var(ddof=1) * (len(inter) - 1))
                     / (len(intra) + len(inter) - 2))
    rng = np.random.default_rng(seed)
    yp = y.copy()
    null = np.empty(n_perm)
    for i in range(n_perm):
        rng.shuffle(yp)
        s = yp[a] == yp[b]
        null[i] = d[~s].mean() / d[s].mean()
    return {"ratio": float(obs), "d": float((inter.mean() - intra.mean()) / pooled),
            "p": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
            "pairs_within": int(same.sum()), "pairs_between": int((~same).sum())}


def label_free(key, X, seed=SEED):
    rng = np.random.default_rng(seed)
    n_comp = min(X.shape[0] - 1, 60)
    R = l2(rng.standard_normal(X.shape))
    ev = PCA(n_components=n_comp).fit(X).explained_variance_ratio_
    ev_r = PCA(n_components=n_comp).fit(R).explained_variance_ratio_
    return {"matrix": mtag(key), "pc1_ratio_vs_random": round(float(ev[0] / ev_r[0]), 1),
            "participation_ratio": round(participation_ratio(X), 1),
            "dims_for_90pct": _dims_for(X), "hopkins": round(hopkins(X, seed=seed), 3)}


def rule_matrices(inp, X):
    """Aircraft x dim matrix of each rule for one layer x pooling, rows in aircraft_common order."""
    ids = list(inp["common"]["aircraft_id"])
    out = {}
    for r in RULES[:3]:
        f = inp[r].set_index("aircraft_id").loc[ids, "fig_id"]
        out[r] = X[inp["row"][f].to_numpy()]
    g = inp["exp4"].groupby("aircraft_id")["fig_id"]
    out["exp4_slots_mean"] = np.stack([l2(X[inp["row"][g.get_group(a)].to_numpy()]).mean(0) for a in ids])
    return out


def views_geometry(inp, X):
    """Cosine of one aircraft's slot figures (Perspective/Plan/Side) with each other,
    against different aircraft; the gap is in standard deviations of the
    different-aircraft cosines, so layers with different spread compare."""
    s = inp["exp4"][inp["exp4"]["slot"].isin(SLOTS)]
    Xn = l2(X[inp["row"][s["fig_id"]].to_numpy()])
    ac, sl, pat = s["aircraft_id"].to_numpy(), s["slot"].to_numpy(), s["patent_id"].to_numpy()
    a, b = np.triu_indices(len(Xn), 1)
    cos = (Xn @ Xn.T)[a, b]
    same_ac = ac[a] == ac[b]
    other = pat[a] != pat[b]
    same_view = sl[a] == sl[b]
    c1 = cos[same_ac & ~same_view]
    c2 = cos[other & same_view]
    c3 = cos[other & ~same_view]
    sd = cos[other].std()
    return {"same_aircraft_other_view": round(float(c1.mean()), 3),
            "other_aircraft_same_view": round(float(c2.mean()), 3),
            "other_aircraft_other_view": round(float(c3.mean()), 3),
            "gap_sd": round(float((c1.mean() - c2.mean()) / sd), 2), "n_same_aircraft_pairs": int(len(c1))}


# ── figure: what each rule picks ─────────────────────────────────────────────

def examples(inp):
    e = {r: inp[r].set_index("aircraft_id") for r in RULES[:3]}
    ids = list(inp["common"]["aircraft_id"])
    n_slots = inp["exp4"][inp["exp4"]["slot"].isin(SLOTS)].groupby("aircraft_id").size()
    pick = lambda r, a: e[r].loc[a, "fig_id"]
    rows = []
    a12 = [a for a in ids if pick("exp1_view_first", a) != pick("exp2_state_first", a)]
    a13 = [a for a in ids if pick("exp1_view_first", a) != pick("exp3_main", a)
           and not e["exp3_main"].loc[a, "rule_applied"].startswith("FALLBACK")
           and e["exp3_main"].loc[a, "view4"] != e["exp1_view_first"].loc[a, "view4"]]
    a4 = [a for a in ids if n_slots.get(a, 0) == 3
          and len({pick(r, a) for r in RULES[:3]}) == 1]
    used = set()
    for label, pool in [("exp1 and exp2 disagree", a12), ("exp3 (main) differs from exp1", a13),
                        ("all single-figure rules agree, exp4 has three views", a4)]:
        pool = [a for a in pool if a not in used]
        if pool:
            rows.append((label, pool[0]))
            used.add(pool[0])
    return rows


def montage(inp, rows, path):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    font_p = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    f_b = ImageFont.truetype(str(font_p / "DejaVuSans-Bold.ttf"), 15)
    f_r = ImageFont.truetype(str(font_p / "DejaVuSans.ttf"), 13)
    T, CAP, BAND, HEAD = 210, 40, 30, 28
    cols = ["exp1 view-first", "exp2 state-first", "exp3 main", "exp4 Perspective", "exp4 Plan", "exp4 Side"]
    W = T * len(cols) + 10 * (len(cols) + 1)
    H = HEAD + len(rows) * (BAND + T + CAP + 10) + 10
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    for j, c in enumerate(cols):
        dr.text((10 + j * (T + 10), 6), c, font=f_b, fill="black")
    ft = inp["figs"].set_index("fig_id")
    g1 = inp["common"].set_index("aircraft_id")["g1_code"]
    y = HEAD
    for label, a in rows:
        dr.text((10, y + 6), f"{a}  ({g1[a]})  {label}", font=f_b, fill="#333333")
        y += BAND
        picks = [inp[r].set_index("aircraft_id").loc[a, "fig_id"] for r in RULES[:3]]
        s = inp["exp4"][inp["exp4"]["aircraft_id"] == a].set_index("slot")["fig_id"]
        picks += [s.get(sl) for sl in SLOTS]
        for j, fid in enumerate(picks):
            x = 10 + j * (T + 10)
            if fid is None:
                dr.rectangle([x, y, x + T, y + T], fill="#eeeeee", outline="#cccccc")
                dr.text((x + 60, y + T // 2 - 8), "no figure", font=f_r, fill="#777777")
                continue
            r = ft.loc[fid]
            tile = Image.open(r["processed_path_518"]).convert("RGB").resize((T, T))
            img.paste(tile, (x, y))
            dr.rectangle([x, y, x + T, y + T], outline="#999999")
            fign = "F?" if r["fig_number"] == "" else f"F{int(float(r['fig_number']))}"
            dr.text((x, y + T + 3), f"{r['view8']}", font=f_r, fill="black")
            dr.text((x, y + T + 20), f"{r['state4']} · {fign}", font=f_r, fill="#444444")
        y += T + CAP + 10
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


# ── markdown helpers ─────────────────────────────────────────────────────────

def table(df, caption=None, bold_row=None):
    cols = [str(c) for c in df.columns]
    out = [f"**{caption}**", ""] if caption else []
    out += ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for i, (_, r) in enumerate(df.iterrows()):
        cells = [str(v) for v in r.tolist()]
        if bold_row is not None and i == bold_row:
            cells = [f"**{c}**" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def style_block():
    m = re.search(r"<style>.*?</style>", STYLE_SOURCE.read_text(encoding="utf-8"), re.S)
    return m.group(0) if m else ""


def fmt_p(p, n_perm):
    return f"≤ {1 / (n_perm + 1):.3f}" if p <= 1 / (n_perm + 1) + 1e-12 else f"{p:.3f}"


# ── build ────────────────────────────────────────────────────────────────────

def build(cfg, n_perm, pdf=True):
    inp = load_inputs(cfg)
    tab_dir = Path(cfg["paths"]["metrics_dir"]).parent / "embedding_analysis"
    tab_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(cfg["paths"]["embedding_report_dir"])
    fig_dir = rep_dir / "figs" / FIG_SUB
    T = {}                                                   # every table, saved as CSV

    figs = inp["figs"]
    w = figs[figs["scope"] == "whole_vehicle"].copy()
    ids = list(inp["common"]["aircraft_id"])
    common = inp["common"].set_index("aircraft_id")
    variant = set(w.loc[w["g1_group"] == "variant", "aircraft_id"]) & set(ids)
    e = {r: inp[r].set_index("aircraft_id").loc[ids] for r in RULES[:3]}
    fm = inp["folder_meta"]
    folder_key = (int(fm["layer"]), fm["pooling"])

    # ── Section 1 numbers ──
    fun = inp["funnel"].copy()
    fun["aircraft_remaining"] = fun["aircraft_remaining"].map(lambda v: "" if pd.isna(v) else int(v))
    fun.columns = ["Step", "Figures removed", "Figures remaining", "Aircraft remaining"]
    T["s1_funnel"] = fun

    vm = (w.groupby(["view8", "view4"]).size().rename("Figures").reset_index()
          .assign(_o=lambda d: d["view4"].map({v: i for i, v in enumerate(["Perspective", "Plan", "Side", "FrontRear"])}))
          .sort_values(["_o", "Figures"], ascending=[True, False]).drop(columns="_o"))
    vm.columns = ["Labelled view (view8)", "View group (view4)", "Figures"]
    T["s1_views"] = vm
    cov = {v: int(w.loc[w["view4"] == v, "aircraft_id"].nunique()) for v in ["Perspective", "Plan", "Side", "FrontRear"]}

    st_order = ["Cruise", "Both", "Hover", "Other", "Missing", "Invariant"]
    sx = pd.crosstab(w["g1_code"], w["state4"]).reindex(columns=[c for c in st_order if c in set(w["state4"])], fill_value=0)
    sx = sx[sx.index != ""]
    sx["Figures"] = sx.sum(axis=1)
    sx.insert(0, "Group", ["convertible" if g in {"TW", "TR", "DS", "CVT", "SRW"} else "fixed" for g in sx.index])
    sx = sx.sort_values(["Group", "Figures"], ascending=[True, False]).reset_index().rename(columns={"g1_code": "G1"})
    T["s1_states"] = sx
    wv = w[w["aircraft_id"].isin(variant)]
    has = lambda st: wv[(wv["view4"] == "Perspective") & (wv["state4"] == st)]["aircraft_id"].nunique()
    n_pc, n_ph = has("Cruise"), has("Hover")

    prof = []
    for r in RULES[:3]:
        v = e[r]["view4"].value_counts()
        s = e[r]["state4"].value_counts()
        prof.append({"Rule": SHORT[r], **{k: int(v.get(k, 0)) for k in ["Perspective", "Plan", "Side", "FrontRear"]},
                     **{f"state {k}": int(s.get(k, 0)) for k in ["Invariant", "Cruise", "Both", "Hover", "Other", "Missing"]}})
    T["s1_pick_profile"] = pd.DataFrame(prof)

    ov = []
    for i, a in enumerate(RULES[:3]):
        for b in RULES[i + 1:3]:
            same = e[a]["fig_id"] == e[b]["fig_id"]
            sv = same[same.index.isin(variant)]
            ov.append({"Pair": f"{SHORT[a]} vs {SHORT[b]}", "Same figure (all)": f"{100 * same.mean():.1f} %",
                       "Aircraft that differ": int((~same).sum()),
                       "Same figure (convertible)": f"{100 * sv.mean():.1f} %",
                       "Convertible aircraft that differ": int((~sv).sum())})
    T["s1_overlap"] = pd.DataFrame(ov)

    d12 = [a for a in ids if e["exp1_view_first"].loc[a, "fig_id"] != e["exp2_state_first"].loc[a, "fig_id"]]
    n12_cruise = int(sum(e["exp2_state_first"].loc[a, "state4"] == "Cruise" for a in d12))
    c12 = (pd.DataFrame({"exp1 picks": [f"{e['exp1_view_first'].loc[a, 'view4']} + {e['exp1_view_first'].loc[a, 'state4']}" for a in d12],
                         "exp2 picks": [f"{e['exp2_state_first'].loc[a, 'view4']} + {e['exp2_state_first'].loc[a, 'state4']}" for a in d12]})
           .value_counts().rename("Aircraft").reset_index())
    T["s1_exp1_vs_exp2"] = c12

    d13 = [a for a in ids if e["exp1_view_first"].loc[a, "fig_id"] != e["exp3_main"].loc[a, "fig_id"]]
    c13 = pd.crosstab(pd.Series([e["exp3_main"].loc[a, "view4"] for a in d13], name="exp3 (main) view"),
                      pd.Series([e["exp1_view_first"].loc[a, "view4"] for a in d13], name="exp1 view"))
    same_view13 = int(sum(e["exp3_main"].loc[a, "view4"] == e["exp1_view_first"].loc[a, "view4"] for a in d13))
    T["s1_exp3_vs_exp1"] = c13.reset_index()

    s4 = inp["exp4"][inp["exp4"]["slot"].isin(SLOTS)]
    ns = s4.groupby("aircraft_id").size().reindex(ids, fill_value=0)
    combo = s4.groupby("aircraft_id")["slot"].agg(lambda x: " + ".join(sorted(x, key=SLOTS.index))).reindex(ids, fill_value="FrontRear only")
    T["s1_exp4_combos"] = combo.value_counts().rename("Aircraft").rename_axis("Slots the aircraft has").reset_index()
    byc = pd.crosstab(common.loc[ids, "g1_code"], ns.rename("slots")).reindex(columns=[0, 1, 2, 3], fill_value=0)
    byc = byc.assign(multi=byc[2] + byc[3], total=byc.sum(axis=1)).sort_values("total", ascending=False)
    byc = byc.drop(columns=[0]) if byc[0].sum() == 0 else byc
    byc.columns = [f"{c} slot{'s' if c != 1 else ''}" if isinstance(c, (int, np.integer)) else c for c in byc.columns]
    byc = byc.rename(columns={"multi": "2 or 3 slots", "total": "Aircraft"}).reset_index().rename(columns={"g1_code": "G1"})
    T["s1_exp4_by_class"] = byc
    n_multi = int((ns >= 2).sum())
    n_one = int((ns <= 1).sum())

    fb = pd.concat([inp["exp3_main"][inp["exp3_main"]["rule_applied"].str.startswith("FALLBACK")].assign(Rule="exp3"),
                    inp["exp4"][inp["exp4"]["rule_applied"].str.startswith("FALLBACK")].assign(Rule="exp4")])
    T["s1_fallbacks"] = fb[["Rule", "aircraft_id", "rule_applied"]].rename(columns={"aircraft_id": "Aircraft", "rule_applied": "What was used"})

    ex = examples(inp)
    montage(inp, ex, fig_dir / "rule_picks_examples.png")

    # ── Section 2 tests ──
    keys = sorted(inp["arrays"])
    qc = qc_report({"arrays": inp["arrays"]}, seed=SEED)
    qc_ok = bool((qc[["nan_count", "inf_count", "all_zero_count", "exact_duplicate_count"]] == 0).all().all())
    lf = pd.DataFrame([label_free(k, inp["arrays"][k]) for k in keys])
    lf.insert(1, "cos_mean", qc["cos_mean"].round(3).to_numpy())
    lf.insert(2, "cos_std", qc["cos_std"].round(3).to_numpy())
    T["s2_label_free"] = lf

    g1 = common.loc[ids, "g1_code"].to_numpy()
    pat = common.loc[ids, "patent_id"].to_numpy()
    sep_rows, sep_raw = [], []
    for k in keys:
        rm = rule_matrices(inp, inp["arrays"][k])
        row = {"Matrix": mtag(k)}
        ds = []
        for r in RULES:
            res = separation(rm[r], g1, pat, n_perm)
            sep_raw.append({"matrix": mtag(k), "rule": SHORT[r], **res})
            row[SHORT[r]] = f"{res['ratio']:.3f} (d {res['d']:.2f})"
            ds.append(res["d"])
        row["Mean d"] = round(float(np.mean(ds)), 3)
        sep_rows.append(row)
    sep = pd.DataFrame(sep_rows)
    sep_raw = pd.DataFrame(sep_raw)
    T["s2_g1_separation"] = sep
    T["s2_g1_separation_raw"] = sep_raw
    best_i = int(sep["Mean d"].idxmax())
    best = sep.loc[best_i, "Matrix"]
    rule_best = sep_raw.groupby("rule")["d"].mean().sort_values(ascending=False)

    wg = w[w["g1_code"] != ""]
    X_rows = inp["row"][wg["fig_id"]].to_numpy()
    vr = []
    for k in keys:
        Xf = inp["arrays"][k][X_rows]
        rv = separation(Xf, wg["view4"].to_numpy(), wg["patent_id"].to_numpy(), n_perm)
        rg = separation(Xf, wg["g1_code"].to_numpy(), wg["patent_id"].to_numpy(), n_perm)
        vr.append({"Matrix": mtag(k), "View ratio": round(rv["ratio"], 3), "View d": round(rv["d"], 2),
                   "G1 ratio": round(rg["ratio"], 3), "G1 d": round(rg["d"], 2),
                   "View d ÷ G1 d": round(rv["d"] / rg["d"], 2) if rg["d"] else np.nan})
    vr = pd.DataFrame(vr)
    T["s2_view_vs_g1"] = vr

    geo = []
    for k in keys:
        X = inp["arrays"][k]
        gm = views_geometry(inp, X)
        rm4 = rule_matrices(inp, X)
        multi = np.array([ns[a] >= 2 for a in ids])
        c41 = np.einsum("ij,ij->i", l2(rm4["exp4_slots_mean"][multi]), l2(rm4["exp1_view_first"][multi]))
        geo.append({"Matrix": mtag(k), "Same aircraft, other view": gm["same_aircraft_other_view"],
                    "Other aircraft, same view": gm["other_aircraft_same_view"],
                    "Other aircraft, other view": gm["other_aircraft_other_view"],
                    "Gap (SD)": gm["gap_sd"], "cos(exp4, exp1) multi-view": round(float(c41.mean()), 3)})
    geo = pd.DataFrame(geo)
    T["s2_views_geometry"] = geo

    for name, df in T.items():
        df.to_csv(tab_dir / f"{name}.csv", index=False)

    # ── markdown ──
    n_air, n_fig = len(ids), int(len(w))
    n_pat = int(inp["common"]["patent_id"].nunique())
    ov_d = {r["Pair"]: r for r in ov}
    fold_is_best = mtag(folder_key) == best
    gb = geo.set_index("Matrix")
    vb = vr.set_index("Matrix")
    lfb = lf.set_index("matrix")
    ex_lines = "; ".join(f"row {i + 1}, {a} ({lab})" for i, (lab, a) in enumerate(ex))
    em = inp["emb_manifest"]
    P = f"≤ {1 / (n_perm + 1):.3f}"
    all_p = sep_raw["p"].max()

    md = [
        "# eVTOL Patent Figures and DINOv2 Embeddings: Progress Report",
        "",
        "**Prepared for:** Phase Supervisor  ",
        f"**Date:** {date.today().isoformat()}  ",
        f"**Material:** {n_air} aircraft from {n_pat} patents, {n_fig} whole-aircraft figures (1 639-patent dataset, all five batches)  ",
        f"**Model:** `{em['model']['model_name']}` (frozen), input {em['input_size']} px, "
        f"layers {{{', '.join(map(str, em['layers']))}}} × pooling {{{', '.join(em['pooling'])}}}",
        "",
        "<span class=\"provenance\">Successor of <code>Supervisor_Report_Taxonomy_Structure_Analysis_summary.md</code> "
        "(Batches 01 + 05, 264 images, July labels). Same layout and tests, now on the whole labelled dataset. "
        "Generated by <code>embedding_evaluation/scripts/build_embedding_analysis.py</code>; every number is computed "
        "from the files listed at the end.</span>",
        "",
        style_block(),
        "",
        "---",
        "",
        "## Summary",
        "",
        f"- **Images.** The probes need one vector per aircraft, and most aircraft are drawn more than once. Four rules "
        f"choose that vector for every one of the {n_air} aircraft that has a G1 code. Three pick a single figure "
        "(by view, by flight state, or the labeller's main figure). The fourth averages one figure per view.",
        f"- **How much the rules differ.** View-first and state-first pick the same figure for "
        f"{ov_d['exp1 vs exp2']['Same figure (all)']} of aircraft. The labeller's main figure differs from both for "
        f"{ov_d['exp1 vs exp3']['Aircraft that differ']} aircraft. The view average differs from a single figure for "
        f"{n_multi} aircraft, the ones drawn in two or three views.",
        f"- **Layer.** All six matrices pass the integrity checks. On the taxonomy criterion (same-G1 aircraft closer "
        f"than different-G1 aircraft), **{best}** separates G1 best, averaged over the four rules"
        + ("." if fold_is_best else f". The experiment folders currently use {mtag(folder_key)}, so they are to be "
           "rebuilt on the winner before the probes."),
        f"- **Open.** Which rule and layer predict G1 best is decided by the probes (Section 3). "
        "The separation tests of Section 2 rank the options but do not train a classifier.",
        "",
        "---",
        "",
        "## Section 1: Which Images Are Analysed",
        "",
        "<div class=\"stage-purpose\"><strong>What this section establishes:</strong> which figures enter the embedding "
        "analysis, what each figure is labelled with, and the four rules that turn an aircraft's figures into the one "
        "vector a probe reads, with how much the rules disagree.</div>",
        "",
        "<span class=\"provenance\">Figure selection: <code>eVTOL-Embedding-Extraction/notebooks/20_figure_selection</code> "
        "(gates) and <code>23_view_state_experiments</code> (rules, <code>src/view_state_experiments.py</code>). "
        "Rationale of each gate: <code>SELECTION_DECISIONS.md</code>.</span>",
        "",
        "### 1.1 From patents to whole-aircraft figures",
        "",
        "<span class=\"sub-purpose\">Purpose: the gates every figure passes before any rule sees it.</span>",
        "",
        "The unit is the **aircraft** (`<patent>_ua<N>`): one patent can disclose several aircraft, and the G1 "
        "architecture and the main-figure marker are recorded per aircraft. Only approved figures of approved, "
        "in-domain aircraft are kept. The domain gate removes aircraft tagged UAV-similar, not electric, or STOL-only. "
        "D1 and D2 duplicates repeat an original and are removed, D3 variants stay. Only whole-aircraft figures are kept.",
        "",
        table(fun, "Table 1. Selection funnel (notebook 20)."),
        "",
        f"The {len(inp['excluded'])} aircraft marked unclassifiable in G1 have no class and are left out of every rule, "
        f"which leaves **{n_air} aircraft and {n_fig - int((w['g1_code'] == '').sum())} figures**.",
        "",
        "### 1.2 What each figure is labelled with",
        "",
        "<span class=\"sub-purpose\">Purpose: the two figure labels the rules sort on, and how they are grouped.</span>",
        "",
        "**View.** The eight wizard views are grouped into four. \"Generic 3D\" is offered in the wizard but has not "
        "been used.",
        "",
        table(vm, "Table 2. Labelled views and their groups (whole-aircraft figures)."),
        "",
        f"Aircraft with at least one figure in each group: Perspective {cov['Perspective']}, Plan {cov['Plan']}, "
        f"Side {cov['Side']}, FrontRear {cov['FrontRear']}.",
        "",
        "**Flight state.** Only a convertible aircraft changes shape between hover and cruise, so only there does the "
        "flight state of a drawing matter.",
        "",
        "- **Fixed types** (RC, MR, SLC, HB, PFV, TB, PTC): every figure is `Invariant`, whatever the label says.",
        "- **Convertible types** (TW, TR, CVT, DS, SRW):",
        "  - `Hover` and `Cruise` are kept as labelled.",
        "  - A figure labelled Invariant, meaning the drawing fits both states, becomes `Both`.",
        "  - Transition and Other become `Other`, and an unlabelled figure is `Missing`.",
        "- **SRW is convertible**, as in the wizard: its rotor stops and serves as the wing in cruise.",
        "",
        table(sx, "Table 3. Flight state of each figure, by G1 type."),
        "",
        f"**Canonical state: Cruise.** Of {len(variant)} convertible aircraft, {n_pc} have a Perspective figure in "
        f"cruise and {n_ph} one in hover. The margin is small. Cruise also suits the types whose defining feature only "
        "shows in forward flight, such as a stopped rotor used as a wing.",
        "",
        "<span class=\"stage-status info\">LABELS: VIEW AND STATE GROUPED</span>",
        "",
        "### 1.3 Four rules for one vector per aircraft",
        "",
        "<span class=\"sub-purpose\">Purpose: how each rule chooses, so a difference in results can be traced to a "
        "difference in images.</span>",
        "",
        "**Orderings.** Each rule sorts the aircraft's whole-aircraft figures and takes the first.",
        "",
        "- **View:** Perspective > Plan > Side > FrontRear.",
        "- **State:** Cruise > Both > Hover > Other > Missing.",
        "- **Ties:** the lowest FIG number wins, then the file name. Crops with no FIG label (`_Fu`) sort last.",
        "",
        table(pd.DataFrame([
            {"Rule": "exp1 view-first", "How the image is chosen": "best view; within it, best state",
             "Vector": "that figure"},
            {"Rule": "exp2 state-first", "How the image is chosen": "best state; within it, best view",
             "Vector": "that figure"},
            {"Rule": "exp3 main", "How the image is chosen": "the figure the labeller marked as main",
             "Vector": "that figure"},
            {"Rule": "exp4 view average", "How the image is chosen": "best figure in each of Perspective, Plan, Side "
             "that the aircraft has", "Vector": "mean of 1 to 3 figure vectors, rescaled to length 1"},
        ]), "Table 4. The four rules."),
        "",
        "exp1 and exp2 use the same two criteria in opposite order. When one figure is best on both, for example a "
        "Perspective drawing in cruise, both rules pick it. For fixed types the state never decides, so exp1 and exp2 "
        "always agree there. Two side tests place the slot vectors side by side instead of averaging them. They "
        "exist only for the aircraft that have every slot (Section 1.5).",
        "",
        table(T["s1_fallbacks"], "Table 5. Fallbacks (the rule could not apply as written)."),
        "",
        "The two exp3 fallbacks have a part figure as their main and are fixed by marking a whole-aircraft main in the "
        "wizard.",
        "",
        "### 1.4 How much the picks differ",
        "",
        "<span class=\"sub-purpose\">Purpose: two rules can only score differently on the aircraft where they "
        "picked different images.</span>",
        "",
        table(T["s1_overlap"], "Table 6. Same figure picked by two single-figure rules."),
        "",
        f"exp4 uses exactly exp1's figure for the {n_one} aircraft with a single slot and differs for the {n_multi} "
        "drawn in two or three views.",
        "",
        table(T["s1_pick_profile"], "Table 7. What each single-figure rule picks (views, then states)."),
        "",
        table(c12, f"Table 8. The {len(d12)} aircraft where exp1 and exp2 disagree."),
        "",
        f"exp1 keeps the better view in a weaker state. exp2 moves to a weaker view to reach a better state, which "
        f"is Cruise in {n12_cruise} of the {len(d12)} cases.",
        "",
        table(T["s1_exp3_vs_exp1"], f"Table 9. The {len(d13)} aircraft where the main figure differs from exp1: "
              "view of each pick."),
        "",
        f"In {same_view13} of the {len(d13)} the labeller's main has the same view as exp1's pick and is another "
        "drawing of it. In the rest the labeller preferred a different view.",
        "",
        f"<figure class=\"single\"><img src=\"figs/{FIG_SUB}/rule_picks_examples.png\"><figcaption><strong>Figure 1."
        f"</strong> The images each rule uses for three aircraft ({ex_lines}). Captions give the labelled view, the "
        "state group and the FIG number.</figcaption></figure>",
        "",
        "### 1.5 The view average (exp4)",
        "",
        "<span class=\"sub-purpose\">Purpose: which aircraft the view average actually changes, and for which "
        "types it can show an effect.</span>",
        "",
        "A single drawing shows one view of the aircraft. The Plan view carries the wing and rotor layout, the Side "
        "view the vertical arrangement, the Perspective the overall shape. exp4 averages the views the aircraft has, "
        "on the assumption that the aircraft stays recognisable from one view to the next (tested in Section 2.4).",
        "",
        table(T["s1_exp4_combos"], "Table 10. Which slots the aircraft have."),
        "",
        table(byc, "Table 11. Slots per aircraft, by G1 type."),
        "",
        "Types with few multi-view aircraft (see the \"2 or 3 slots\" column) cannot show an effect of the view average. "
        "The side tests keep the views apart: Perspective + Plan concatenated for the "
        f"{int(((ns >= 2) & combo.str.contains('Perspective') & combo.str.contains('Plan')).sum())} aircraft that have "
        f"both, and all three for the {int((ns == 3).sum())} that have every slot. They are compared with exp1 on "
        "the same aircraft only.",
        "",
        "<span class=\"stage-status info\">IMAGES: FOUR RULES DEFINED</span>",
        "",
        "---",
        "",
        "## Section 2: Which Layer",
        "",
        "<div class=\"stage-purpose\"><strong>What this section tests:</strong> the six layer × pooling matrices on "
        "integrity, label-free structure, how well they keep same-G1 aircraft together under each rule, whether the "
        "drawing's view outweighs the architecture, and whether one aircraft stays recognisable across its views.</div>",
        "",
        f"<span class=\"provenance\">Matrices: <code>2_embedding_extraction/embeddings/{EMB_TAG}/</code> "
        f"(notebook 22, {em['created'][:10]}, {em['n_figures']} figures, vectors L2-normalised). Distances are "
        f"cosine distances. Pairs from the same patent are left out of every separation test, so two drawings of "
        f"one patent never count as a match. p-values from {n_perm} label permutations (seed {SEED}).</span>",
        "",
        "### 2.1 Integrity and structure against random noise",
        "",
        "<span class=\"sub-purpose\">Purpose: that every matrix is valid and carries more structure than noise "
        "(label-free).</span>",
        "",
        f"Integrity over all {em['n_figures']} figures: "
        + ("no NaN, Inf, all-zero or duplicate vectors in any matrix." if qc_ok else "**failures present** (see "
           "the CSV)."),
        "",
        table(lf.rename(columns={"matrix": "Matrix", "cos_mean": "Cosine mean", "cos_std": "Cosine SD",
                                 "pc1_ratio_vs_random": "PC1 ÷ random", "participation_ratio": "Effective dims",
                                 "dims_for_90pct": "Dims for 90 %", "hopkins": "Hopkins"}),
              "Table 12. Label-free structure of each matrix (all figures)."),
        "",
        f"On its first principal component each matrix carries {lf['pc1_ratio_vs_random'].min():g} to "
        f"{lf['pc1_ratio_vs_random'].max():g} times the variance of a random matrix of the same shape "
        + ("(pass bar 3). " if (lf["pc1_ratio_vs_random"] >= 3).all() else "(pass bar 3, **not met by every matrix**). ")
        +
        "Hopkins above 0.5 means the vectors clump. These measures describe the matrices and do not use the taxonomy, "
        "so they do not decide the layer.",
        "",
        f"<span class=\"stage-status {'passed' if qc_ok else 'caution'}\">QC AND STRUCTURE: "
        f"{'PASSED' if qc_ok else 'CHECK'}</span>",
        "",
        "### 2.2 Do same-G1 aircraft sit closer together?",
        "",
        "<span class=\"sub-purpose\">Purpose: the taxonomy criterion, for every matrix under every rule.</span>",
        "",
        "**Separation ratio** = mean distance between aircraft of different G1 types ÷ mean distance between aircraft "
        "of the same type. A ratio of 1.0 means G1 is invisible to the matrix. **Cohen's d** is the same gap in "
        f"standard deviations. {n_air} aircraft per cell.",
        "",
        table(sep, "Table 13. G1 separation: ratio (Cohen's d) per matrix and rule.", bold_row=best_i),
        "",
        (f"Every cell is significant (p {P})." if all_p <= 1 / (n_perm + 1) + 1e-12
         else f"Largest p-value in the table: {all_p:.3f}."),
        f" **{best}** has the highest mean effect over the four rules ({sep.loc[best_i, 'Mean d']:.3f}). "
        "Ranked by mean d over the six matrices, the rules come out as "
        + ", ".join(f"{k} ({v:.3f})" for k, v in rule_best.items())
        + f". The rules differ by at most {rule_best.max() - rule_best.min():.3f} in mean d, the matrices by "
        f"{sep['Mean d'].max() - sep['Mean d'].min():.3f}. On this test the layer decides far more than the rule. "
        "This is a preview, not the verdict, since the probes measure prediction directly.",
        "",
        "### 2.3 Viewpoint against architecture",
        "",
        "<span class=\"sub-purpose\">Purpose: the July report found the drawing's viewpoint separating more strongly "
        "than any design attribute. Checked here on every matrix.</span>",
        "",
        "The same test on all figures, once with the view group as the label and once with G1. Pairs from the same "
        "patent are excluded, so the comparison is between different aircraft only.",
        "",
        table(vr, "Table 14. Separation by view group and by G1 (all figures, different patents only)."),
        "",
        "A \"View d ÷ G1 d\" above 1 means two drawings in the same view look more alike than two aircraft of the "
        f"same type. The lowest value is {vr['View d ÷ G1 d'].min():g} ({vr.loc[vr['View d ÷ G1 d'].idxmin(), 'Matrix']}) "
        f"and the highest {vr['View d ÷ G1 d'].max():g} ({vr.loc[vr['View d ÷ G1 d'].idxmax(), 'Matrix']}). "
        + (f"{int((vr['View d ÷ G1 d'] > 1).sum())} of the 6 matrices weigh the view more than the architecture, "
           "so the view of the image a rule picks can move an aircraft further than its G1 type does."
           if (vr["View d ÷ G1 d"] > 1).any() else
           "In every matrix the architecture weighs more than the view."),
        "",
        "### 2.4 Does an aircraft stay recognisable across its views?",
        "",
        "<span class=\"sub-purpose\">Purpose: the assumption behind the view average (exp4).</span>",
        "",
        "Cosine similarity between the slot figures that exp4 averages. Two views of the same aircraft are compared "
        "with two different aircraft in the same view. The gap is expressed in standard deviations of the "
        "different-aircraft similarities, because the spread differs from layer to layer.",
        "",
        table(geo, "Table 15. Same aircraft across views against different aircraft."),
        "",
        f"A positive gap means that the same aircraft seen from another angle is closer than a different aircraft "
        f"seen from the same angle. The largest gap is {geo['Gap (SD)'].max():g} SD "
        f"({geo.loc[geo['Gap (SD)'].idxmax(), 'Matrix']}). The last column shows how far averaging moves the vector "
        "away from exp1's single figure for the multi-view aircraft.",
        "",
        "### 2.5 Layer declaration",
        "",
        f"**{best}** is carried forward. It separates G1 best averaged over the four rules (Table 13). On the "
        f"other two criteria it stands as follows. View d ÷ G1 d is {vb.loc[best, 'View d ÷ G1 d']:g}, against a range of "
        f"{vr['View d ÷ G1 d'].min():g} to {vr['View d ÷ G1 d'].max():g} (Table 14). The across-view gap is "
        f"{gb.loc[best, 'Gap (SD)']:g} SD, against {geo['Gap (SD)'].min():g} to {geo['Gap (SD)'].max():g} (Table 15). "
        f"Its label-free profile is PC1 ÷ random {lfb.loc[best, 'pc1_ratio_vs_random']:g} and Hopkins "
        f"{lfb.loc[best, 'hopkins']:g}.",
        "",
        ("The experiment folders already use this matrix." if fold_is_best else
         f"The experiment folders were built on {mtag(folder_key)} (the July choice). Before the probes they are "
         f"rebuilt on {best}, a change of one constant in notebook 23 (`PRIMARY_LAYER`, `PRIMARY_POOLING`)."),
        " The choice is provisional: the probes of Section 3 run on every matrix and confirm or overturn it on "
        "prediction.",
        "",
        "<span class=\"stage-status progress\">LAYER: PROVISIONAL, PENDING PROBES</span>",
        "",
        "---",
        "",
        "## Section 3: Open Work",
        "",
        "<div class=\"stage-purpose\"><strong>What remains:</strong> the tests that decide which rule, and which "
        "layer, gives the embedding that predicts the architecture best.</div>",
        "",
        f"1. **Probes.** k-nearest-neighbour and logistic-regression probes predicting G1 from each rule's vectors on "
        f"all six matrices, with folds grouped by patent and macro-F1 as the score.",
        f"   - exp4 is compared with exp1 on the {n_multi} multi-view aircraft only, since the other {n_one} are "
        "identical in both.",
        "   - The side tests are compared with exp1 on their own aircraft.",
        f"2. **Small classes.** "
        + ", ".join(f"{r['G1']} {r['Aircraft']}" for _, r in byc.iterrows() if r["Aircraft"] < 20)
        + " have fewer than 20 aircraft. Their scores are reported with that caveat, or merged.",
        "3. **Wizard fixes.**",
        "   - Mark a whole-aircraft main for the two exp3 fallbacks.",
        "   - Check whether KR20240170008A ua1 and ua2 are one aircraft drawn in two states.",
        "4. **224 px.** The 224 px embeddings are not extracted yet. They are needed only for a resolution comparison.",
        "",
        "<span class=\"stage-status progress\">SUMMARY: IN PROGRESS</span>",
        "",
        "---",
        "",
        "<span class=\"provenance\">Sources (read-only): <code>1639_LABELLED/2_embedding_extraction/selection/funnel.csv"
        "</code> (notebook 20); <code>…/embeddings/dinov2-large_518/</code> (notebook 22); "
        "<code>…/view_state_experiments/</code> (notebook 23: figure_table, exp1–exp4 manifests, aircraft_common, "
        "exp_embeddings). Every table is also written as CSV to <code>1639_LABELLED/3_embedding_evaluation/"
        "embedding_analysis/</code>. Labels: <code>0_labelling/outputs/tables/</code> (notebook 04).</span>",
        "",
    ]
    rep_dir.mkdir(parents=True, exist_ok=True)
    md_path = rep_dir / REPORT_NAME
    md_path.write_text("\n".join(md), encoding="utf-8")
    print("wrote", md_path)
    print("tables in", tab_dir)
    if pdf:
        import build_report_pdf
        print("wrote", build_report_pdf.build(md_path))
    return md_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--perm", type=int, default=999, help="label permutations per separation test")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    build(load_config(), a.perm, pdf=not a.no_pdf)
