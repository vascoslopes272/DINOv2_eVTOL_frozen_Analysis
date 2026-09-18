"""View / flight-state figure-choice experiments (notebook 23_view_state_experiments).

Which figure of an aircraft should represent it in the embedding space? Four
deterministic rules are compared on EVERY aircraft that has a G1 code (user,
2026-09-18: "all the aircraft compared", not only those with a full slot set):

    exp1_view_first   best view, then best flight state
    exp2_state_first  best flight state, then best view
    exp3_main         the figure the labeller marked as main (exp1's pick when the
                      main is not a whole-vehicle figure — flagged)
    exp4_slots_mean   one figure per view slot (Perspective, Plan, Side), the slot
                      vectors averaged over the slots the aircraft HAS
Side tests on the subsets that have the slots: exp4_slots_2 / exp4_slots_3, the
slot vectors concatenated in fixed order (no slot is ever imputed).

Unit = the AIRCRAFT (``aircraft_id = <patent>_ua<N>``): G1 topType and the main
marker are stored per aircraft (user ruling 2026-09-17). ``patent_id`` is kept
as a column. Only labels are read and only existing embedding rows are sliced;
nothing is sampled and no model is run here.

Input: notebook 20's gates (``figure_selection``, imported, not changed) over
Stage 04's tables, and notebook 22's embeddings ``embeddings/<tag>/``.
Output: ``<paths.pipeline_root>/view_state_experiments/``::

    figure_table.csv          one row per figure (gates passed, scope kept as a column)
    excluded_aircraft.csv     eligible aircraft left out of the manifests, with the reason
    exp1_view_first.csv  exp2_state_first.csv  exp3_main.csv  exp3_conflicts.csv  exp4_slots.csv
    aircraft_common.csv       every aircraft with a G1 code (the comparison set)
    VIEW_STATE_REPORT.md
    exp_embeddings/<exp>/     X.npy  aircraft_ids.npy  figs.csv  meta.json  y.csv
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src import figure_selection as fs

OUT_SUBDIR = "view_state_experiments"

# ── mappings ─────────────────────────────────────────────────────────────────
# Wizard T2 "per" ids (AC 8-value list; Generic 3D is offered but unused so far).
VIEW4 = {
    "Top": "Plan", "Bottom/Down": "Plan",
    "Front": "FrontRear", "Back": "FrontRear",
    "Side": "Side",
    "Front-Isometric": "Perspective", "Rear-Isometric": "Perspective", "Generic 3D": "Perspective",
}
VIEW_ORDER = ["Perspective", "Plan", "Side", "FrontRear"]
SLOTS = ["Perspective", "Plan", "Side"]            # exp4 concatenation order

# G1 topType ids as the wizard stores them. The brief's "TP" is the wizard's TR
# (Tilt Rotor; TP was recoded to TR). PTC joins the invariant set (user, 2026-09-17).
INVARIANT = {"RC", "MR", "SLC", "HB", "PFV", "TB", "SRW", "PTC"}
VARIANT = {"TW", "TR", "DS", "CVT"}

# Labelled acState -> state4 for a VARIANT aircraft. Notebook 04 writes the wizard's
# display name 'Invariant' (stored id 'HoverCruise', recoded since 2026-09-18);
# Ground/Unclear/NonApplicable are retired ids that old records may still hold.
# A blank label is "Missing" (never imputed).
STATE_VARIANT = {
    "Hover": "Hover", "Cruise": "Cruise",
    "Transition": "Other", "Other": "Other", "Invariant": "Other",
    "Ground": "Other", "Unclear": "Other", "NonApplicable": "Other",
}
REAL_STATES = ("Hover", "Cruise")

# Decided by decide_canonical() (Step 3.2, run of 2026-09-17): of 337 variant
# aircraft, 212 have >=1 Perspective+Cruise figure and 207 >=1 Perspective+Hover.
# Notebook 23 asserts the data still agree with this constant.
CANONICAL_STATE = "Cruise"

# One matrix for every experiment folder (notebook 22's run). L22 CLS had the
# best mean taxonomy separation in the July frozen-DINOv2 analysis
# (REPORT_11_taxonomy.md, stage 5).
EMB_TAG = "dinov2-large_518"
PRIMARY_LAYER = 22
PRIMARY_POOLING = "cls"

_FIG_RE = re.compile(r"_F([0-9A-Za-z]+)\.(?:png|jpe?g)$")


def out_dir(cfg: Dict[str, Any]) -> Path:
    d = Path(cfg["paths"]["pipeline_root"]) / OUT_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── Step 1: figure table ─────────────────────────────────────────────────────

def parse_fig_number(image_file: str, fig_key: str) -> float:
    """Leading integer of the crop's ``_F<label>`` suffix (``F1A`` -> 1);
    ``Fu`` (unknown) falls back to a numeric ``fig_key``; else NaN."""
    m = _FIG_RE.search(str(image_file))
    if m:
        d = re.match(r"\d+", m.group(1))
        if d:
            return float(d.group(0))
    k = str(fig_key).strip()
    return float(k) if k.isdigit() else np.nan


def load_family_map(cfg: Dict[str, Any]) -> pd.DataFrame:
    p = Path(cfg["paths"]["labelled_root"]) / "0_labelling" / "inputs" / "reference" / "family_map.csv"
    fm = pd.read_csv(p, dtype=str, keep_default_na=False)
    return fm[["canonical_pub_number", "family_id"]].rename(columns={"canonical_pub_number": "patent_id"})


def build_candidates(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(aircraft rows, candidates with notebook 20's gates, funnel), from Stage 04's tables."""
    master, figures = fs.load_master(cfg)
    c, funnel = fs.apply_gates(fs.build_candidates(master, figures), cfg)
    return master, c, funnel


def build_figure_table(c: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """One row per figure that passed every gate except the scope gate."""
    keep = c["excluded_by"].isin(["", "not a whole-aircraft figure"])
    f = c[keep].copy()
    missing = sorted({"per", "acState", "topType", "is_main", "parts"} - set(f.columns))
    if missing:
        raise KeyError(f"figure candidates lack columns {missing}")
    first_part = f["parts"].fillna("").astype(str).str.split("|").str[0]
    proc = Path(cfg["paths"]["pipeline_root"]) / "processed" / "518"
    t = pd.DataFrame({
        "aircraft_id": f["aircraft_uid"],
        "patent_id": f["patent_id"],
        "fig_id": f["figure_uid"],
        "image_path": f["approved_copy_path"],
        "processed_path_518": [str(proc / u) for u in f["figure_uid"]],
        "scope": np.where(first_part == fs.WHOLE_VEHICLE, "whole_vehicle", "part"),
        "parts": f["parts"].fillna(""),
        "view8": f["per"].fillna(""),
        "state_raw": f["acState"].fillna(""),
        "g1_code": f["topType"].fillna(""),
        "is_main": f["is_main"].astype(bool),
        "fig_number": [parse_fig_number(a, b) for a, b in zip(f["image_file"].fillna(""), f["fig_key"].fillna(""))],
        "fig_order": f["fig_order"],
        "batch": f["batch"],
    })
    t = t.merge(load_family_map(cfg), on="patent_id", how="left", validate="m:1")
    return t.sort_values(["aircraft_id", "fig_id"]).reset_index(drop=True)


# ── Step 2: derived fields ───────────────────────────────────────────────────

def view4_of(view8: str) -> str:
    if view8 not in VIEW4:
        raise ValueError(f"view {view8!r} has no view4 mapping")
    return VIEW4[view8]


def state4_of(g1_code: str, state_raw: str) -> str:
    """Invariant types are overwritten; variant types map their label; no G1 -> 'NoG1'."""
    if g1_code in INVARIANT:
        return "Invariant"
    if g1_code in VARIANT:
        if state_raw == "":
            return "Missing"
        if state_raw not in STATE_VARIANT:
            raise ValueError(f"acState {state_raw!r} has no state4 mapping")
        return STATE_VARIANT[state_raw]
    if g1_code == "":
        return "NoG1"
    raise ValueError(f"G1 code {g1_code!r} is in neither the INVARIANT nor the VARIANT set")


def add_derived(t: pd.DataFrame) -> pd.DataFrame:
    """view4 / state4 / g1_group. A blank view is tolerated only on part figures
    (they leave at the scope gate)."""
    t = t.copy()
    blank = (t["view8"] == "") & (t["scope"] == "whole_vehicle")
    if blank.any():
        raise ValueError(f"whole-vehicle figures without a view: {t.loc[blank, 'fig_id'].tolist()}")
    t["view4"] = t["view8"].map(lambda v: view4_of(v) if v else "")
    t["state4"] = [state4_of(g, s) for g, s in zip(t["g1_code"], t["state_raw"])]
    t["g1_group"] = np.select([t["g1_code"].isin(INVARIANT), t["g1_code"].isin(VARIANT)],
                              ["invariant", "variant"], "none")
    return t


def whole_vehicle(t: pd.DataFrame) -> pd.DataFrame:
    return t[t["scope"] == "whole_vehicle"].reset_index(drop=True)


# ── Step 3: coverage ─────────────────────────────────────────────────────────

def _has(t: pd.DataFrame, view4: str, state4: str | None = None) -> pd.Series:
    m = t["view4"] == view4
    if state4 is not None:
        m &= t["state4"] == state4
    return m.groupby(t["aircraft_id"]).any()


def coverage_stats(w: pd.DataFrame, lost_aircraft: List[str]) -> Dict[str, Any]:
    """Numbers of Step 3 on the whole-vehicle table ``w``."""
    n_fig = w.groupby("aircraft_id").size()
    var = w[w["g1_group"] == "variant"]
    ph, pc = _has(var, "Perspective", "Hover"), _has(var, "Perspective", "Cruise")
    per, plan, side = _has(w, "Perspective"), _has(w, "Plan"), _has(w, "Side")
    n = int(n_fig.size)
    return {
        "aircraft": n,
        "figures": int(len(w)),
        "fig_per_aircraft": {"min": int(n_fig.min()), "median": float(n_fig.median()),
                             "max": int(n_fig.max()),
                             "distribution": {int(k): int(v) for k, v in n_fig.value_counts().sort_index().items()}},
        "variant_aircraft": int(ph.size),
        "variant_persp_hover": int(ph.sum()),
        "variant_persp_cruise": int(pc.sum()),
        "variant_persp_both": int((ph & pc).sum()),
        "variant_persp_neither": int((~ph & ~pc).sum()),
        "slot2_aircraft": int((per & plan).sum()),
        "slot2_pct": round(float(100 * (per & plan).sum() / n), 1),
        "slot3_aircraft": int((per & plan & side).sum()),
        "slot3_pct": round(float(100 * (per & plan & side).sum() / n), 1),
        "no_whole_vehicle_figure": sorted(lost_aircraft),
    }


def decide_canonical(stats: Dict[str, Any]) -> str:
    h, c = stats["variant_persp_hover"], stats["variant_persp_cruise"]
    if h == c:
        raise ValueError(f"Perspective+Hover and Perspective+Cruise tie ({h}) — decide CANONICAL_STATE by hand")
    return "Hover" if h > c else "Cruise"


def domain_aircraft(master: pd.DataFrame, cfg: Dict[str, Any]) -> set:
    """Aircraft that pass notebook 20's aircraft-level gates: approved, primary
    (no D1/D2), no domain-gate tag."""
    s = cfg["selection"]
    tags = master["edgeTags"].fillna("").astype(str).map(lambda v: set(v.split("|")) - {""})
    gated = tags.map(lambda t: bool(t & set(s.get("gate_tags", []))))
    ok = fs._truthy(master["is_approved"]) & fs._truthy(master["is_primary"]) & ~gated
    for t in s.get("exclude_dup_types", []):
        ok &= pd.to_numeric(master["dup_type"], errors="coerce") != t
    if s.get("exclude_uncertain", False):
        ok &= ~fs._truthy(master["t1_humanUncertain"])
    return set(master.loc[ok, "aircraft_id"])


def aircraft_without_whole_vehicle(domain: set, t: pd.DataFrame) -> List[str]:
    """In-domain aircraft with no usable whole-vehicle figure."""
    have = set(t.loc[t["scope"] == "whole_vehicle", "aircraft_id"])
    return sorted(domain - have)


# ── Step 4: manifests ────────────────────────────────────────────────────────

MANIFEST_COLS = ["aircraft_id", "patent_id", "fig_id", "image_path", "slot", "view4", "state4", "rule_applied"]


def state_rank(state4: pd.Series, canonical: str = CANONICAL_STATE) -> pd.Series:
    """canonical > other real state > Other > Missing. 'Invariant' ranks with the
    canonical state: an invariant aircraft's figures all carry it, so it never competes."""
    other_real = [s for s in REAL_STATES if s != canonical]
    order = {canonical: 0, "Invariant": 0, **{s: 1 for s in other_real}, "Other": 2, "Missing": 3}
    bad = sorted(set(state4) - set(order))
    if bad:
        raise ValueError(f"state4 values without a rank: {bad}")
    return state4.map(order)


def _ranked(w: pd.DataFrame, keys: List[str], canonical: str) -> pd.DataFrame:
    r = w.assign(_view=w["view4"].map({v: i for i, v in enumerate(VIEW_ORDER)}),
                 _state=state_rank(w["state4"], canonical),
                 _fign=w["fig_number"].fillna(np.inf))
    return r.sort_values(["aircraft_id", *keys, "_fign", "fig_id"], kind="mergesort")


def _describe(row: pd.Series, first: str, n: int) -> str:
    fign = "?" if pd.isna(row["fig_number"]) else int(row["fig_number"])
    return (f"{first}: view4={row['view4']} (rank {int(row['_view'])}), "
            f"state4={row['state4']} (rank {int(row['_state'])}), fig_number={fign}, of {n} figures")


def select_single(w: pd.DataFrame, first: str, canonical: str = CANONICAL_STATE) -> pd.DataFrame:
    """exp1 (first='view') or exp2 (first='state'): one figure per aircraft."""
    keys = {"view": ["_view", "_state"], "state": ["_state", "_view"]}[first]
    r = _ranked(w, keys, canonical)
    n = r.groupby("aircraft_id").size()
    top = r.drop_duplicates("aircraft_id").copy()
    top["slot"] = "single"
    top["rule_applied"] = [_describe(row, f"{first}-first", n[row["aircraft_id"]]) for _, row in top.iterrows()]
    return top[MANIFEST_COLS].reset_index(drop=True)


def select_main(w: pd.DataFrame, t_all: pd.DataFrame,
                 fallback: pd.DataFrame | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """exp3: the human-marked main figure; aircraft with 0 or >1 mains are listed in
    the conflicts. With ``fallback`` (exp1's manifest) they take its pick, flagged in
    rule_applied, so exp3 covers every aircraft. ``t_all`` (scope column included)
    tells whether a missing main was a part figure."""
    mains = w[w["is_main"]]
    k = mains.groupby("aircraft_id").size().reindex(w["aircraft_id"].unique(), fill_value=0)
    ok = mains[mains["aircraft_id"].isin(k[k == 1].index)].copy()
    ok["slot"] = "single"
    ok["rule_applied"] = "human main"
    rows = []
    for aid, cnt in k[k != 1].items():
        part_mains = t_all[(t_all["aircraft_id"] == aid) & t_all["is_main"] & (t_all["scope"] != "whole_vehicle")]
        reason = ("more than one main" if cnt > 1 else
                  "main figure is not whole-vehicle" if len(part_mains) else
                  "no main figure among eligible figures")
        rows.append({"aircraft_id": aid, "patent_id": w.loc[w["aircraft_id"] == aid, "patent_id"].iat[0],
                     "n_main": int(cnt), "reason": reason,
                     "main_fig_ids": "|".join(mains.loc[mains["aircraft_id"] == aid, "fig_id"].tolist()
                                              or part_mains["fig_id"].tolist())})
    conflicts = pd.DataFrame(rows, columns=["aircraft_id", "patent_id", "n_main", "reason", "main_fig_ids"])
    out = ok[MANIFEST_COLS]
    if fallback is not None and len(conflicts):
        fb = fallback[fallback["aircraft_id"].isin(conflicts["aircraft_id"])].copy()
        why = conflicts.set_index("aircraft_id")["reason"]
        fb["rule_applied"] = [f"FALLBACK ({why[a]}): exp1's pick" for a in fb["aircraft_id"]]
        out = pd.concat([out, fb[MANIFEST_COLS]])
    return out.sort_values("aircraft_id").reset_index(drop=True), conflicts


def select_slots(w: pd.DataFrame, canonical: str = CANONICAL_STATE) -> pd.DataFrame:
    """exp4: per slot (Perspective, Plan, Side) the best state, then lowest fig_number.
    An aircraft with none of the three slots gets its best FrontRear figure as its
    only slot, so exp4_slots_mean covers every aircraft."""
    has_slot = w["view4"].isin(SLOTS).groupby(w["aircraft_id"]).transform("any")
    pool = w[w["view4"].isin(SLOTS) | (~has_slot & (w["view4"] == "FrontRear"))]
    r = _ranked(pool, ["_view", "_state"], canonical)
    n = r.groupby(["aircraft_id", "view4"]).size()
    top = r.drop_duplicates(["aircraft_id", "view4"]).copy()
    top["slot"] = top["view4"]
    top["rule_applied"] = [
        ("FALLBACK (no Perspective/Plan/Side figure): " if row["view4"] not in SLOTS else "")
        + f"slot {row['view4']}: state4={row['state4']} (rank {int(row['_state'])}), "
        f"fig_number={'?' if pd.isna(row['fig_number']) else int(row['fig_number'])}, "
        f"of {n[(row['aircraft_id'], row['view4'])]} figures"
        for _, row in top.iterrows()]
    top["_slot"] = top["slot"].map({s: i for i, s in enumerate(VIEW_ORDER)})
    return top.sort_values(["aircraft_id", "_slot"])[MANIFEST_COLS].reset_index(drop=True)


def slot_complete(exp4: pd.DataFrame, slots: List[str]) -> set:
    have = exp4.groupby("aircraft_id")["slot"].agg(set)
    return {a for a, s in have.items() if set(slots) <= s}


def common_aircraft(exp1, exp2, exp3, exp4) -> List[str]:
    """Every aircraft all four experiments cover (with exp3's fallback and exp4's
    slot mean, that is every aircraft with a G1 code)."""
    return sorted(set(exp1["aircraft_id"]) & set(exp2["aircraft_id"])
                  & set(exp3["aircraft_id"]) & set(exp4["aircraft_id"]))


# ── Step 5: overlap ──────────────────────────────────────────────────────────

def overlap(manifests: Dict[str, pd.DataFrame], ids: List[str], variant_ids: set) -> pd.DataFrame:
    """Share of aircraft (in ``ids``) where two single-figure experiments pick the same figure."""
    pick = {k: m.set_index("aircraft_id")["fig_id"].reindex(ids) for k, m in manifests.items()}
    names = list(pick)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            same = pick[a] == pick[b]
            var = same[same.index.isin(variant_ids)]
            rows.append({"pair": f"{a} vs {b}",
                         "all_n": int(same.size), "all_same_pct": round(100 * same.mean(), 1),
                         "variant_n": int(var.size), "variant_same_pct": round(100 * var.mean(), 1)})
    return pd.DataFrame(rows)


# ── Step 6/7: embedding folders + ground truth ───────────────────────────────

def load_primary_matrix(cfg: Dict[str, Any]) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    d = Path(cfg["paths"]["pipeline_root"]) / "embeddings" / EMB_TAG
    f = d / f"emb_layer{PRIMARY_LAYER}_{PRIMARY_POOLING}.npy"
    for p in (f, d / "metadata.parquet", d / "manifest.json"):
        if not p.exists():
            raise FileNotFoundError(f"{p} is missing — run notebook 22 for size 518 first")
    X = np.load(f)
    meta = pd.read_parquet(d / "metadata.parquet")
    if len(meta) != len(X) or not meta["figure_uid"].is_unique:
        raise ValueError(f"{d}: metadata rows do not index the matrix one-to-one")
    info = {"embeddings_dir": str(d), "matrix_file": f.name, "layer": PRIMARY_LAYER,
            "pooling": PRIMARY_POOLING, "manifest": json.loads((d / "manifest.json").read_text())}
    return X, meta, info


def missing_vectors(manifests: Dict[str, pd.DataFrame], meta: pd.DataFrame) -> pd.DataFrame:
    """Selected figures that have no row in the embedding metadata."""
    have = set(meta["figure_uid"])
    rows = [m.assign(experiment=k) for k, m in manifests.items()]
    allm = pd.concat(rows, ignore_index=True)
    return allm[~allm["fig_id"].isin(have)][["experiment", "aircraft_id", "fig_id"]]


def build_matrix(manifest: pd.DataFrame, ids: List[str], slots: List[str] | None,
                 X: np.ndarray, meta: pd.DataFrame,
                 mode: str = "concat") -> Tuple[np.ndarray, List[str], pd.DataFrame, List[str]]:
    """Rows in ``ids`` order. ``slots=None``: one figure per aircraft. ``mode='concat'``:
    the slot vectors concatenated in ``slots`` order, dropping aircraft that miss a
    slot. ``mode='mean'``: the mean of every slot vector the aircraft has in the
    manifest, L2-renormalised (keeps every aircraft; ``slots`` is ignored)."""
    row_of = pd.Series(np.arange(len(meta)), index=meta["figure_uid"])
    m = manifest[manifest["aircraft_id"].isin(ids)]
    if mode == "mean":
        g = m.groupby("aircraft_id")
        lacking = sorted(set(ids) - set(g.groups))
        if lacking:
            raise ValueError(f"aircraft without any slot figure: {lacking}")
        rows = []
        for a in ids:
            v = X[row_of[g.get_group(a)["fig_id"]].to_numpy()].mean(axis=0)
            rows.append(v / max(np.linalg.norm(v), 1e-12))
        grp = g[["slot", "fig_id"]].agg("|".join).reindex(ids)
        figs = pd.DataFrame({"aircraft_id": ids, "n_slots": g.size().reindex(ids).to_numpy(),
                             "slots": grp["slot"].to_numpy(), "fig_ids": grp["fig_id"].to_numpy()})
        return np.stack(rows).astype(np.float32), list(ids), figs, []
    if slots is None:
        m = m.set_index("aircraft_id").reindex(ids)
        if m["fig_id"].isna().any():
            raise ValueError(f"aircraft without a selected figure: {m.index[m['fig_id'].isna()].tolist()}")
        Xo = X[row_of[m["fig_id"]].to_numpy()]
        figs = m.reset_index()[["aircraft_id", "fig_id", "view4", "state4"]]
        return Xo.astype(np.float32), list(ids), figs, []
    wide = m.pivot(index="aircraft_id", columns="slot", values="fig_id").reindex(ids)
    complete = wide[slots].notna().all(axis=1)
    kept = [a for a in ids if complete[a]]
    dropped = [a for a in ids if not complete[a]]
    blocks = [X[row_of[wide.loc[kept, s]].to_numpy()] for s in slots]
    Xo = np.concatenate(blocks, axis=1) if kept else np.zeros((0, X.shape[1] * len(slots)), np.float32)
    figs = wide.loc[kept, slots].rename(columns=lambda s: f"fig_id_{s}").reset_index()
    figs.columns.name = None
    figs.insert(1, "fig_ids", figs[[f"fig_id_{s}" for s in slots]].agg("|".join, axis=1))
    return Xo.astype(np.float32), kept, figs, dropped


def ground_truth(ids: List[str], w: pd.DataFrame) -> pd.DataFrame:
    a = w.drop_duplicates("aircraft_id").set_index("aircraft_id")
    y = a.loc[ids, ["patent_id", "g1_code", "family_id"]].reset_index()
    if y["g1_code"].eq("").any():
        raise ValueError(f"aircraft without a G1 code: {y.loc[y['g1_code'].eq(''), 'aircraft_id'].tolist()}")
    return y


def excluded_no_g1(w: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Eligible aircraft with no G1 code (the wizard's G1 'unclassifiable' override)."""
    a = w.loc[w["g1_group"] == "none", ["aircraft_id", "patent_id"]].drop_duplicates()
    note = master.set_index("aircraft_id")["g1_quickNote"].fillna("")
    a["reason"] = ["no G1 topType (G1 quick override; note: " + note.get(x, "") + ")" for x in a["aircraft_id"]]
    return a.reset_index(drop=True)


def experiment_meta(name: str, manifest_file: str, slots: List[str] | None, emb: Dict[str, Any],
                    canonical: str, Xo: np.ndarray, dropped: List[str], mode: str = "concat") -> Dict[str, Any]:
    m = emb["manifest"]
    if mode == "mean":
        vectors = ("L2-normalised per figure; the aircraft's slot vectors (1-3 of Perspective/Plan/Side, "
                   "FrontRear only when it has none) averaged, then L2-renormalised")
    elif slots:
        vectors = "L2-normalised per figure; slot vectors concatenated, not renormalised"
    else:
        vectors = "L2-normalised per figure"
    return {
        "experiment": name,
        "generated": date.today().isoformat(),
        "producer": "eVTOL-Embedding-Extraction/notebooks/23_view_state_experiments.ipynb",
        "manifest_file": manifest_file,
        "unit": "aircraft (aircraft_id = <patent>_ua<N>)",
        "rows": int(Xo.shape[0]),
        "dim": int(Xo.shape[1]),
        "embedding_matrix": f"{emb['embeddings_dir']}/{emb['matrix_file']}",
        "embedding_tag": m["pipeline"],
        "model": m["model"]["model_name"],
        "input_size": m["input_size"],
        "layer": emb["layer"],
        "pooling": emb["pooling"],
        "vectors": vectors,
        "slots": slots if mode == "concat" else "available slots (mean)",
        "extraction_created": m["created"],
        "extraction_git_commit": m["git_commit"],
        "canonical_state": canonical,
        "restricted_to": "aircraft_common.csv" + (" with every slot present" if dropped else ""),
        "dropped_missing_slot": len(dropped),
        "row_order": "aircraft_ids.npy = figs.csv = y.csv (aircraft_common.csv order)",
    }


def write_experiment(folder: Path, Xo: np.ndarray, ids: List[str], figs: pd.DataFrame,
                     y: pd.DataFrame, meta: Dict[str, Any], dropped: List[str]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    assert list(y["aircraft_id"]) == list(ids) == list(figs["aircraft_id"]) and len(Xo) == len(ids)
    np.save(folder / "X.npy", Xo)
    np.save(folder / "aircraft_ids.npy", np.array(ids, dtype=str))
    figs.to_csv(folder / "figs.csv", index=False)
    y.to_csv(folder / "y.csv", index=False)
    if dropped:
        pd.DataFrame({"aircraft_id": dropped}).to_csv(folder / "dropped_ids.csv", index=False)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")


def class_distribution(y: pd.DataFrame, min_n: int = 20) -> pd.DataFrame:
    d = y["g1_code"].value_counts().rename_axis("g1_code").reset_index(name="aircraft")
    d["group"] = np.where(d["g1_code"].isin(INVARIANT), "invariant", "variant")
    d["below_20"] = d["aircraft"] < min_n
    return d


# ── report ───────────────────────────────────────────────────────────────────

def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines)


def write_report(path: Path, *, ov: pd.DataFrame, n_common: int, n_slots: Dict[int, int],
                 funnel: pd.DataFrame, scope_dropped: int, n_no_fig_number: int,
                 stats: Dict[str, Any], canonical: str,
                 ct_g1: pd.DataFrame, ct_view: pd.DataFrame, sizes: Dict[str, int],
                 excluded: pd.DataFrame, conflicts: pd.DataFrame, emb: Dict[str, Any],
                 classes: pd.DataFrame, folders: Dict[str, Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    s = stats
    dist = ", ".join(f"{k}: {v}" for k, v in s["fig_per_aircraft"]["distribution"].items())
    lost = ", ".join(s["no_whole_vehicle_figure"]) or "none"
    tables = Path(cfg["paths"]["labelled_root"]) / "0_labelling" / "outputs" / "tables"
    small = classes.loc[classes["below_20"], "g1_code"].tolist()
    L = [
        "# View / state figure-choice experiments",
        "",
        f"Generated {date.today().isoformat()} by `eVTOL-Embedding-Extraction/notebooks/23_view_state_experiments.ipynb` "
        "(`src/view_state_experiments.py`). Unit: the aircraft (`<patent>_ua<N>`).",
        "",
        "## Same figure chosen? (Step 5)",
        "",
        f"Share of aircraft for which two single-figure rules pick the same figure, on the **{n_common} aircraft** of "
        "the comparison set: every aircraft with a G1 code.",
        "",
        _md_table(ov),
        "",
        "Invariant aircraft carry one state on every figure, so exp1 and exp2 can only differ on variant aircraft. "
        "exp3 counts the labeller's main figure, except for the aircraft listed under exp3 conflicts, which take exp1's pick.",
        "",
        f"G1 classes with fewer than 20 aircraft in the comparison set: {len(small)} of {len(classes)} "
        f"({', '.join(small) or 'none'}).",
        "",
        "## Sources",
        "",
        f"* Labels: `{tables}/aircraft_table.csv` and `figure_table.csv` (Stage 04, built from the wizard record "
        "`0_labelling/inputs/record/reviewed_patents_Batch_ALL.xlsx`).",
        "* Figure fields: `parts` (scope; first value `Whole Vehicle Layout` = whole vehicle), `per` (view8), "
        "`acState` (state), `is_main` (main marker, per aircraft), `arch` (aircraft number). Aircraft field: `topType` (G1).",
        "* Family: `0_labelling/inputs/reference/family_map.csv` (`canonical_pub_number` → `family_id`).",
        f"* Embeddings: `{emb['embeddings_dir']}` (row index `metadata.parquet: figure_uid` = `fig_id` here).",
        "",
        "## Input and scope (Steps 1–2)",
        "",
        "Gates are notebook 20's (Stage 04 tables; domain gate on UAV/Electric/STOL-similar tags; D1/D2 out). "
        f"The scope gate removed **{scope_dropped}** part figures; everything below uses whole-vehicle figures only.",
        "",
        _md_table(funnel.fillna("")),
        "",
        "Mappings: view4 Top, Bottom/Down → Plan; Front, Back → FrontRear; Side → Side; Front-Isometric, "
        "Rear-Isometric, Generic 3D → Perspective. "
        "state4: invariant G1 codes (" + ", ".join(sorted(INVARIANT)) + ") → Invariant whatever the label; "
        "variant codes (" + ", ".join(sorted(VARIANT)) + ") map Hover → Hover, Cruise → Cruise, blank → Missing, "
        "anything else (Transition, Other, Invariant) → Other. "
        "The brief's TP is the wizard's TR; PTC is invariant (ruling 2026-09-17).",
        "",
        "### G1 code × state4 (figures)",
        "",
        _md_table(ct_g1.reset_index()),
        "",
        "### view4 × state4 (figures)",
        "",
        _md_table(ct_view.reset_index()),
        "",
        "## Coverage (Step 3)",
        "",
        f"1. **Figures per aircraft** ({s['aircraft']} aircraft, {s['figures']} figures): min {s['fig_per_aircraft']['min']}, "
        f"median {s['fig_per_aircraft']['median']:g}, max {s['fig_per_aircraft']['max']}. Distribution (figures: aircraft): {dist}.",
        f"2. **Variant aircraft** ({s['variant_aircraft']}): ≥1 Perspective+Hover {s['variant_persp_hover']}, "
        f"≥1 Perspective+Cruise {s['variant_persp_cruise']}, both {s['variant_persp_both']}, neither {s['variant_persp_neither']}. "
        f"**Canonical state = {canonical}.**",
        f"3. **Slot coverage**: 2-slot (Perspective + Plan) {s['slot2_aircraft']} aircraft = {s['slot2_pct']} %; "
        f"3-slot (+ Side) {s['slot3_aircraft']} = {s['slot3_pct']} %.",
        f"4. **Aircraft with no whole-vehicle figure** (excluded everywhere): {len(s['no_whole_vehicle_figure'])} — {lost}.",
        "",
        "## Manifests (Step 4)",
        "",
        _md_table(pd.DataFrame([{"file": k, "rows": v} for k, v in sizes.items()])),
        "",
        "Order rules — exp1: view (Perspective > Plan > Side > FrontRear), then state (canonical > other real state > "
        "Other > Missing), then lowest fig_number, then fig_id. exp2: state first, then view, same tie-break. "
        "exp3: the labeller's main figure (exp1's pick, flagged, for the conflicts below). "
        "exp4: per slot (Perspective, Plan, Side), state then fig_number; an aircraft with none of the three slots "
        "gets its best FrontRear figure (flagged). "
        "fig_number is the leading integer of the crop's `_F<label>` suffix (or a numeric fig_key); "
        f"{n_no_fig_number} whole-vehicle figures are `_Fu` crops with no FIG label, sort last and fall to the "
        "fig_id order (the crop name, which follows the drawing sheet).",
        "",
    ]
    if len(excluded):
        L += ["Eligible aircraft left out of every manifest:", "", _md_table(excluded), ""]
    L += ["exp3 conflicts (no single whole-vehicle main; these aircraft take exp1's pick in exp3 — "
          "the fix is to mark a whole-vehicle main in the wizard):", "",
          _md_table(conflicts) if len(conflicts) else "none", ""]
    m = emb["manifest"]
    L += [
        "## Embedding folders (Steps 6–7)",
        "",
        f"Primary matrix: `{emb['embeddings_dir']}/{emb['matrix_file']}` — DINOv2 {m['model']['model_name']}, "
        f"input {m['input_size']} px, layer {emb['layer']}, pooling {emb['pooling']}, L2-normalised per figure; "
        f"extracted {m['created']} over {m['n_figures']} figures (notebook 22's `run_extraction`, the processed/518 set). "
        "Every selected figure has a row; nothing was imputed.",
        "",
        _md_table(pd.DataFrame([{"folder": k, **v} for k, v in folders.items()])),
        "",
        "The four main folders (`exp1_view_first`, `exp2_state_first`, `exp3_main`, `exp4_slots_mean`) hold every "
        "aircraft of `aircraft_common.csv` in its row order. `exp4_slots_mean` averages the slot vectors the aircraft "
        "has and renormalises (slots per aircraft: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(n_slots.items())) + "). The side tests `exp4_slots_2` and `exp4_slots_3` concatenate the slot vectors "
        "[" + ", ".join(SLOTS) + "] without renormalising and keep only the aircraft that have every slot "
        "(the others are in `dropped_ids.csv`); compare them with the other rules on the same aircraft only.",
        "",
        "### G1 classes (comparison set)",
        "",
        _md_table(classes),
        "",
        "`family_id` comes from `0_labelling/inputs/reference/family_map.csv`; the dataset keeps one patent per family, "
        "so grouping by family equals grouping by patent (it keeps the aircraft of one patent together, nothing more).",
        "",
    ]
    path.write_text("\n".join(L), encoding="utf-8")
