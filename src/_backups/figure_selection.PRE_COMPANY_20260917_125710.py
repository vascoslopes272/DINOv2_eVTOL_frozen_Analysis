"""Label-driven figure selection (notebook 20_figure_selection).

The "pre-pre-processing": decides WHICH figures enter the embedding pipeline
using only the human labels — nothing here opens an image.

Input: Stage 04's tables in ``<paths.labelled_root>/0_labelling/outputs/tables/``
(``aircraft_table.csv``, one row per aircraft ``aircraft_id = <patent>_ua<N>``;
``figure_table.csv``, one row per figure). Layout of 2026-09-17.

1. candidates   every figure on file, with its figure labels (per, acState,
                acSty, parts, qualityFlag, is_main, rotation_deg) and its
                aircraft's labels (topType, duplicate status, edge tags).
2. gates        the same for every figure set, applied in order; each figure
                keeps the FIRST gate that removed it (``excluded_by``) — the
                selection funnel. The domain gate and the whole-aircraft rule
                mirror eVTOL-Visual-Evaluation/labeling_evaluation's loader
                (rulings 2026-09-15), so both stages analyse the same aircraft.
3. strategies   named figure sets built from the eligible figures — at most one
                figure per aircraft unless the strategy is ``all``.
4. coverage     aircraft x set matrix, plus a select/report split of the
                aircraft (stratified by topType) so the best strategy can be
                chosen on one half and reported on the other.

Output (``<paths.pipeline_root>/selection/``)::

    candidates.csv   funnel.csv   aircraft.parquet   coverage.csv
    sets/<name>.csv  sets_union.csv   selection_summary.json
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

WHOLE_VEHICLE = "Whole Vehicle Layout"
FIGURE_COLS = ["aircraft_id", "batch", "patent_id", "block", "fig_key", "arch", "image_file", "approved_copy_path", "status",
               "is_main", "per", "acState", "acSty", "acCol", "bgSty", "parts",
               "qualityFlag", "rotation_deg", "comment"]
AIRCRAFT_COLS = ["patent_id", "variant", "aircraft_name", "assignee", "app_year", "topType",
                 "is_approved", "is_primary", "dup_type", "same_aircraft_as", "labels_inherited_from",
                 "t1_humanUncertain", "edgeTags", "arch_gt", "arch_gt_visible"]


def selection_dir(cfg: Dict[str, Any]) -> Path:
    d = Path(cfg["paths"]["pipeline_root"]) / "selection"
    (d / "sets").mkdir(parents=True, exist_ok=True)
    return d


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().isin(["True", "true", "1", "1.0"])


# ── 1. candidates ────────────────────────────────────────────────────────────

def load_master(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """(aircraft rows, figure rows) from Stage 04. Option ids ``NA``/``None``
    are kept as text (read with keep_default_na=False)."""
    root = Path(cfg["paths"]["labelled_root"]) / "0_labelling" / "outputs" / "tables"   # 2026-09-17 layout
    master = pd.read_csv(root / "aircraft_table.csv", keep_default_na=False, na_values=[""], low_memory=False)
    figures = pd.read_csv(root / "figure_table.csv", keep_default_na=False, na_values=[""], low_memory=False)
    master["arch"] = pd.to_numeric(master["variant"], errors="coerce").astype("Int64")
    figures["arch"] = pd.to_numeric(figures["arch"], errors="coerce").astype("Int64")
    master["aircraft_uid"] = master["aircraft_id"]          # <patent>_ua<N> — the one key from stage 0 to stage 3
    return master, figures


def build_candidates(master: pd.DataFrame, figures: pd.DataFrame) -> pd.DataFrame:
    """One row per figure on file, joined to the aircraft row it draws."""
    f = figures[[c for c in FIGURE_COLS if c in figures.columns]].copy()
    # The wizard block name is the only unique figure key: some rows carry no
    # file name, fig_key is empty on older batches, and one record
    # (CN114684361A) has two figure rows pointing at the same file.
    # <aircraft_id>/<figure file> — the same path shape as 0_labelling/outputs/images/ and processed/<size>/
    f["figure_uid"] = f["aircraft_id"] + "/" + f["block"].astype(str).str.replace(r"^Image:\s*", "", regex=True)
    f = f.drop(columns=["aircraft_id"])
    a = master[AIRCRAFT_COLS + ["arch", "aircraft_uid"]].rename(columns={"variant": "variant_label"})
    c = f.merge(a, on=["patent_id", "arch"], how="left", validate="m:1")
    # patent-level verdict for figures that carry no arch (disapproved figures)
    pat_ok = master.groupby("patent_id")["is_approved"].agg(lambda s: _truthy(s).any())
    c["patent_approved"] = c["patent_id"].map(pat_ok).fillna(False).astype(bool)
    c["rotation_deg"] = pd.to_numeric(c["rotation_deg"], errors="coerce").fillna(0).astype(int)
    c["is_main"] = _truthy(c["is_main"])
    c["fig_order"] = c.groupby("patent_id").cumcount()
    assert c["figure_uid"].is_unique, "figure_uid is not unique"
    return c


# ── 2. gates ─────────────────────────────────────────────────────────────────

def apply_gates(c: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fill ``excluded_by`` (first failing gate) and ``eligible``; return the funnel."""
    s = cfg["selection"]
    gate_tags = list(s.get("gate_tags", []))
    tags = c["edgeTags"].fillna("").astype(str).map(lambda v: set(v.split("|")) - {""})
    similar = tags.map(lambda t: next((g for g in gate_tags if g in t), None))
    first_part = c["parts"].fillna("").astype(str).str.split("|").str[0]
    dup_type = pd.to_numeric(c["dup_type"], errors="coerce")

    gates: List[Tuple[str, pd.Series]] = [
        ("figure not approved", c["status"].astype(str).str.lower() != "approved"),
        ("image file missing", ~c["approved_copy_path"].fillna("").astype(str)
            .map(lambda p: bool(p) and Path(p).exists())),
        ("patent not approved", ~c["patent_approved"]),
        ("no aircraft row", c["aircraft_uid"].isna()),
        ("aircraft not approved", ~_truthy(c["is_approved"])),
        *[(f"domain gate: {g}", similar == g) for g in gate_tags],
        ("duplicate patent (D1/D2, points at its original)", ~_truthy(c["is_primary"])),
        *[(f"duplicate type D{int(t)} excluded", dup_type == t)
          for t in s.get("exclude_dup_types", [])],
    ]
    if s.get("exclude_uncertain", False):
        gates.append(("human-uncertain aircraft", _truthy(c["t1_humanUncertain"])))
    # two approved figure rows of one patent pointing at the same file: keep the
    # main one (else the first), so no image is embedded twice
    order = c.assign(_m=~c["is_main"]).sort_values(["_m", "fig_order"])
    same_file = order["approved_copy_path"].notna() & order.duplicated(
        ["patent_id", "approved_copy_path"])
    gates.append(("same image file as another figure row", same_file.reindex(c.index)))
    if s.get("whole_vehicle_only", True):
        gates.append(("not a whole-aircraft figure", first_part != WHOLE_VEHICLE))
    for q in s.get("exclude_quality", []):
        gates.append((f"quality flag: {q}", c["qualityFlag"].astype(str) == q))
    for st in s.get("exclude_styles", []):
        gates.append((f"drawing style: {st}", c["acSty"].astype(str) == st))

    c["excluded_by"] = ""
    rows = [{"step": "figures on file", "removed": 0, "remaining": len(c),
             "aircraft_remaining": np.nan}]
    for name, mask in gates:
        hit = mask.fillna(False).astype(bool) & (c["excluded_by"] == "")
        c.loc[hit, "excluded_by"] = name
        alive = c["excluded_by"] == ""
        rows.append({"step": name, "removed": int(hit.sum()), "remaining": int(alive.sum()),
                     "aircraft_remaining": int(c.loc[alive, "aircraft_uid"].nunique())})
    c["eligible"] = c["excluded_by"] == ""
    return c, pd.DataFrame(rows)


# ── 3. strategies ────────────────────────────────────────────────────────────

def _rank(df: pd.DataFrame, tie_break: List[str]) -> pd.DataFrame:
    """Order candidate figures of one aircraft: main first, then the
    perspective order, then clean quality, then patent figure order."""
    per_order = {p: i for i, p in enumerate(tie_break)}
    return df.assign(
        _main=~df["is_main"],
        _per=df["per"].map(per_order).fillna(len(per_order)),
        _q=df["qualityFlag"].astype(str) != "clean",
    ).sort_values(["aircraft_uid", "_main", "_per", "_q", "fig_order"])


def _one_per_aircraft(pool: pd.DataFrame, tie_break: List[str]) -> pd.DataFrame:
    return _rank(pool, tie_break).drop_duplicates("aircraft_uid").drop(columns=["_main", "_per", "_q"])


def build_set(elig: pd.DataFrame, name: str, spec: Dict[str, Any],
              tie_break: List[str]) -> pd.DataFrame | None:
    """Figures of one strategy. ``None`` when the strategy is not configured yet."""
    rule = spec["rule"]
    if rule == "all":
        out = elig.copy()
    elif rule == "main":
        out = _one_per_aircraft(elig, tie_break)
    elif rule in ("state", "perspective", "style"):
        col = {"state": "acState", "perspective": "per", "style": "acSty"}[rule]
        out = _one_per_aircraft(elig[elig[col].isin(spec["values"])], tie_break)
    elif rule == "state_by_type":
        table = {k: v for k, v in (spec.get("table") or {}).items() if v}
        if not table:
            print(f"[selection] {name}: table is empty — skipped. REMINDER: decide which "
                  f"flight state best shows each topType (selection.strategies.{name}.table).")
            return None
        want = elig["topType"].map(table)
        keep = [st in (w or []) for st, w in zip(elig["acState"], want)]
        out = _one_per_aircraft(elig[keep], tie_break)
    else:
        raise ValueError(f"strategy {name!r}: unknown rule {rule!r}")
    out = out.copy()
    out.insert(0, "set", name)
    return out


def build_sets(c: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    s = cfg["selection"]
    elig = c[c["eligible"]]
    sets = {}
    for name, spec in s["strategies"].items():
        out = build_set(elig, name, spec, list(s.get("tie_break_perspective", [])))
        if out is not None:
            sets[name] = out
    return sets


# ── 4. aircraft table, coverage, split ───────────────────────────────────────

def aircraft_table(master: pd.DataFrame, c: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Every eligible aircraft with ALL its labels, plus the select/report split."""
    uids = c.loc[c["eligible"], "aircraft_uid"].unique()
    a = master[master["aircraft_uid"].isin(uids)].copy()
    a["n_eligible_figures"] = a["aircraft_uid"].map(c[c["eligible"]].groupby("aircraft_uid").size())

    rng = np.random.default_rng(int(cfg["selection"].get("seed", 42)))
    a["split"] = ""
    for _, g in a.groupby(a["topType"].fillna("∅")):
        idx = rng.permutation(g.index.to_numpy())
        half = np.array(["select", "report"])[np.arange(len(idx)) % 2]
        a.loc[idx, "split"] = half
    # parquet needs one type per column: labels are ids/text, keep them as strings
    for col in a.columns:
        if a[col].dtype == object:
            a[col] = a[col].map(lambda v: None if pd.isna(v) else str(v))
    return a


def coverage(aircraft: pd.DataFrame, sets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cov = aircraft[["aircraft_uid", "topType", "split"]].copy()
    for name, df in sets.items():
        n = df.groupby("aircraft_uid").size()
        cov[name] = cov["aircraft_uid"].map(n).fillna(0).astype(int)
    return cov


def coverage_by_type(cov: pd.DataFrame, set_names: List[str]) -> pd.DataFrame:
    """Aircraft covered per set and topType (share of that type's aircraft)."""
    g = cov.groupby(cov["topType"].fillna("∅"))
    out = pd.DataFrame({"aircraft": g.size()})
    for s in set_names:
        out[s] = g[s].apply(lambda x: int((x > 0).sum()))
    return out.sort_values("aircraft", ascending=False)


# ── persist ──────────────────────────────────────────────────────────────────

def save(c: pd.DataFrame, funnel: pd.DataFrame, aircraft: pd.DataFrame,
         sets: Dict[str, pd.DataFrame], cov: pd.DataFrame, cfg: Dict[str, Any]) -> Path:
    d = selection_dir(cfg)
    c.to_csv(d / "candidates.csv", index=False)
    funnel.to_csv(d / "funnel.csv", index=False)
    aircraft.to_parquet(d / "aircraft.parquet", index=False)
    cov.to_csv(d / "coverage.csv", index=False)
    for old in (d / "sets").glob("*.csv"):
        old.unlink()
    for name, df in sets.items():
        df.to_csv(d / "sets" / f"{name}.csv", index=False)
    union = (pd.concat(sets.values()).drop(columns="set")
             .drop_duplicates("figure_uid").reset_index(drop=True))
    union.to_csv(d / "sets_union.csv", index=False)
    all_sets = set(sets)
    common = cov.loc[(cov[[s for s in all_sets]] > 0).all(axis=1), "aircraft_uid"]
    summary = {
        "generated": date.today().isoformat(),
        "producer": "eVTOL-Embedding-Extraction/notebooks/20_figure_selection.ipynb",
        "labelled_root": str(cfg["paths"]["labelled_root"]),
        "rules": cfg["selection"],
        "figures_on_file": int(len(c)),
        "eligible_figures": int(c["eligible"].sum()),
        "eligible_aircraft": int(len(aircraft)),
        "split": aircraft["split"].value_counts().to_dict(),
        "sets": {k: {"figures": int(len(v)), "aircraft": int(v["aircraft_uid"].nunique())}
                 for k, v in sets.items()},
        "aircraft_in_every_set": int(len(common)),
        "union_figures": int(len(union)),
    }
    (d / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                              encoding="utf-8")
    print(f"[selection] wrote {len(sets)} sets ({len(union)} unique figures) to {d}")
    return d


def load_sets(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    d = selection_dir(cfg) / "sets"
    return {p.stem: pd.read_csv(p, keep_default_na=False, na_values=[""])
            for p in sorted(d.glob("*.csv"))}
