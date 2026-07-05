"""Taxonomy-label ingestion from the HTML labelling wizard's long-format export.

The wizard exports ONE ROW PER (patent, section, sub-dimension, field, value):
columns ``Patent_ID, Section, Sub_Dimension, Field, Value, Source, Image_Path``.
Multi-architecture patents carry an ``_arch{N}`` suffix on ``Patent_ID``.

Section ``T2`` is per-IMAGE labelling (rows carry ``Image_Path``); every other
section (G1/M1/M2/M3/...) is per-ARCHITECTURE labelling. The per-image ``T2``
fields ``isMain`` / ``arch`` / ``status`` are how we pick the canonical "main"
figure of each architecture.

Everything here is read-only w.r.t. the raw xlsx (raw inputs are immutable);
outputs go under ``taxonomy.output_dir``.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ARCH_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_arch(?P<n>\d+)$")

# Wizard bookkeeping fields that are not design attributes -> excluded from the
# wide canonical table (they stay available in the raw long frame).
DEFAULT_DROP_FIELDS = ["dinoUnderstanding", "footAmbiguous"]
# ...plus anything matching these patterns (quick-override / uncertainty /
# free-text notes fields exist in every section with varying prefixes).
DROP_FIELD_RE = re.compile(
    r"(quickOverride|humanUncertain|quickCount|Notes?$|_notes$|otherNote)", re.I)


# ── basic loading / inspection ────────────────────────────────────────────────

def load_long(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Load the wizard long-format xlsx (path from ``taxonomy.wizard_excel``)."""
    path = Path(cfg["taxonomy"]["wizard_excel"])
    if not path.exists():
        raise FileNotFoundError(f"wizard export not found: {path}")
    df = pd.read_excel(path)
    expected = {"Patent_ID", "Section", "Sub_Dimension", "Field", "Value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"wizard export missing columns: {sorted(missing)}")
    df["_source_file"] = path.name
    return df


def inspect_exports(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Distinct (Section, Sub_Dimension, Field) triples with row counts."""
    long = load_long(cfg)
    tbl = (
        long.groupby(["Section", "Sub_Dimension", "Field"])
        .size().rename("n_rows").reset_index()
        .sort_values(["Section", "Sub_Dimension", "Field"])
    )
    print(f"{len(long)} rows | {long['Patent_ID'].nunique()} distinct Patent_IDs "
          f"| {len(tbl)} distinct (section, subdim, field) triples")
    return tbl


def parse_arch_id(pid: str) -> Tuple[str, int]:
    """``US..._arch2`` -> (``US...``, 2); unsuffixed ids are architecture 1."""
    m = ARCH_SUFFIX_RE.match(str(pid))
    if m:
        return m.group("base"), int(m.group("n"))
    return str(pid), 1


# ── per-image table (T2 section) ──────────────────────────────────────────────

def image_table(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot T2 rows to one row per image with its per-image fields.

    Adds ``base_patent_id``, ``arch_index`` (from the T2 ``arch`` field,
    default 1), ``is_main`` (bool) and ``approved`` (bool).
    """
    t2 = long[(long["Section"] == "T2") & long["Image_Path"].notna()]
    piv = t2.pivot_table(index=["Patent_ID", "Image_Path"], columns="Field",
                         values="Value", aggfunc="first").reset_index()
    piv.columns.name = None

    parsed = piv["Patent_ID"].map(parse_arch_id)
    piv["base_patent_id"] = parsed.map(lambda t: t[0])
    # arch: explicit T2 field wins; fall back to the Patent_ID suffix.
    arch_field = pd.to_numeric(piv.get("arch"), errors="coerce")
    piv["arch_index"] = arch_field.fillna(parsed.map(lambda t: t[1])).astype(int)
    piv["is_main"] = piv.get("isMain").astype(str).str.lower().isin(
        ["true", "1", "yes", "main"])
    piv["approved"] = piv.get("status").astype(str).str.lower().eq("approved")
    piv["image_exists"] = piv["Image_Path"].map(lambda p: Path(str(p)).exists())
    return piv


def select_main_images(images: pd.DataFrame) -> pd.DataFrame:
    """One canonical main image per (base_patent_id, arch_index).

    Keeps approved + isMain images that exist on disk. If an architecture has
    several mains, the first (sorted by filename) is kept and the surplus is
    reported. Returns columns base_patent_id, arch_index, image_path, plus the
    per-image style fields useful as confounds (per, acSty, acCol).
    """
    ok = images[images["is_main"] & images["approved"] & images["image_exists"]]
    ok = ok.sort_values(["base_patent_id", "arch_index", "Image_Path"])
    dups = ok.duplicated(["base_patent_id", "arch_index"], keep="first")
    if dups.any():
        print(f"[select_main_images] {int(dups.sum())} architectures have >1 "
              f"approved main image — keeping the first per architecture.")
    sel = ok[~dups]
    keep = ["base_patent_id", "arch_index", "Image_Path"]
    # Per-image style descriptors — hand-labelled confound material.
    for extra in ("per", "acSty", "acCol", "bgSty", "bgCol", "parts",
                  "acState", "qualityFlag"):
        if extra in sel.columns:
            keep.append(extra)
    sel = sel[keep].rename(columns={"Image_Path": "image_path",
                                    "per": "perspective"})
    n_missing_file = int((images["is_main"] & images["approved"]
                          & ~images["image_exists"]).sum())
    if n_missing_file:
        print(f"[select_main_images] WARNING: {n_missing_file} main images "
              f"missing on disk — excluded.")
    return sel.reset_index(drop=True)


def apply_processing_manifest(sel: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Swap wizard ``image_path`` (raw matched/ crop) for the Stage-02 518px file.

    Joins on the manifests written by Patent-Labelling-Tools'
    ``processor.process_batch`` (``paths.processing_manifest_glob``): the
    manifest's ``src_path`` equals the wizard export's ``Image_Path``, and
    ``dst_main``/``dst_all`` is the padded+resized PNG. Rows without a
    processed counterpart keep the raw path and are reported (that usually
    means 02_processing wasn't run for their batch yet).
    """
    import glob as _glob
    pattern = cfg["paths"].get("processing_manifest_glob")
    if not pattern:
        print("[apply_processing_manifest] paths.processing_manifest_glob not "
              "set — embeddings will use RAW matched/ crops (224px HF crop).")
        return sel
    files = sorted(_glob.glob(str(pattern)))
    if not files:
        print(f"[apply_processing_manifest] no manifest matches {pattern} — "
              f"embeddings will use RAW matched/ crops.")
        return sel

    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m = m[m["processed"] == True]                                # noqa: E712
    m["dst"] = m["dst_main"].fillna(m["dst_all"])
    mapping = dict(zip(m["src_path"].astype(str), m["dst"].astype(str)))

    sel = sel.copy()
    mapped = sel["image_path"].astype(str).map(mapping)
    n_hit = int(mapped.notna().sum())
    n_miss = int((sel["image_path"].notna() & mapped.isna()).sum())
    sel.loc[mapped.notna(), "image_path"] = mapped[mapped.notna()]
    print(f"[apply_processing_manifest] {n_hit} image paths -> processed 518px"
          f"{f'; {n_miss} NOT in any manifest (still raw!)' if n_miss else ''}")
    return sel


# ── wide canonical architecture table ─────────────────────────────────────────

TAXONOMY_SECTIONS = ["G1", "M1", "M2", "M3"]


def build_wide(long: pd.DataFrame, drop_fields: Optional[List[str]] = None) -> pd.DataFrame:
    """Pivot taxonomy-section rows to one row per architecture, columns ``{Section}_{Field}``.

    Only sections in ``TAXONOMY_SECTIONS`` are architecture attributes. META/T1
    (patent-level) and T2 (per-image) live on the UNSUFFIXED Patent_ID even for
    multi-arch patents, so including them would create phantom duplicate rows.
    """
    if drop_fields is None:
        drop_fields = DEFAULT_DROP_FIELDS
    lab = long[long["Section"].isin(TAXONOMY_SECTIONS)
               & ~long["Field"].isin(drop_fields)
               & ~long["Field"].astype(str).str.contains(DROP_FIELD_RE)].copy()
    lab["col"] = lab["Section"].astype(str) + "_" + lab["Field"].astype(str)

    dups = lab.duplicated(["Patent_ID", "col"], keep="last")
    if dups.any():
        print(f"[build_wide] {int(dups.sum())} duplicate (patent, field) rows "
              f"— keeping the last occurrence.")
        lab = lab[~dups]

    wide = lab.pivot(index="Patent_ID", columns="col", values="Value").reset_index()
    wide.columns.name = None

    parsed = wide["Patent_ID"].map(parse_arch_id)
    wide["base_patent_id"] = parsed.map(lambda t: t[0])
    wide["arch_index"] = parsed.map(lambda t: t[1])
    n_arch = wide.groupby("base_patent_id")["arch_index"].transform("nunique")
    wide["n_architectures"] = n_arch
    wide["multi_arch"] = n_arch > 1
    return wide


def join_metadata(wide: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Join assignee + priority year from the PatSeer export; add year_bin."""
    meta_path = Path(cfg["taxonomy"].get("metadata_excel")
                     or cfg["paths"]["labels_excel"])
    id_col = cfg["data"]["patent_id_col"]
    meta = pd.read_excel(meta_path, usecols=lambda c: c in
                         {id_col, "Assignee", "Assignee Country",
                          "Priority Year", "Publication/Issue Date",
                          "CPC Main Group (CPCG)"})
    meta = meta.rename(columns={id_col: "base_patent_id",
                                "Assignee": "assignee",
                                "Assignee Country": "assignee_country",
                                "Priority Year": "priority_year"})
    meta = meta.drop_duplicates("base_patent_id")
    # Primary CPC: first code in the semicolon list; subclass = first 4 chars
    # (e.g. B64C, H02K) — coarse enough to have usable class sizes.
    cpcg = meta.pop("CPC Main Group (CPCG)").astype(str)
    first = cpcg.str.split(";").str[0].str.strip()
    meta["cpc_first"] = first.where(first.ne("nan"))
    meta["cpc_subclass"] = meta["cpc_first"].str[:4]
    out = wide.merge(meta, on="base_patent_id", how="left")
    out["priority_year"] = pd.to_numeric(out["priority_year"], errors="coerce")

    n_bins = int(cfg["taxonomy"].get("year_bins", 3))
    valid = out["priority_year"].notna()
    out["year_bin"] = pd.NA
    if valid.sum() >= n_bins:
        out.loc[valid, "year_bin"] = pd.qcut(
            out.loc[valid, "priority_year"], q=n_bins, duplicates="drop"
        ).astype(str)
    n_unmatched = int(out["assignee"].isna().sum())
    if n_unmatched:
        print(f"[join_metadata] {n_unmatched}/{len(out)} architectures have no "
              f"PatSeer metadata match.")
    return out


def build_canonical(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Full label build. Returns ``(canonical, images)``.

    canonical: one row per architecture, wide attribute columns + metadata +
    ``image_path`` of its main figure (NaN when none approved yet) + provenance.
    images: the full per-image T2 table (for figure-style confounds later).
    """
    long = load_long(cfg)
    drop_fields = cfg["taxonomy"].get("drop_fields", DEFAULT_DROP_FIELDS)
    wide = build_wide(long, drop_fields)
    wide = join_metadata(wide, cfg)

    # Human-uncertainty flag: True when the labeller marked ANY section of this
    # architecture as uncertain (usable as a filter via taxonomy.exclude_uncertain).
    unc = long[long["Field"].astype(str).str.contains("humanUncertain", case=False)]
    unc_ids = set(unc.loc[unc["Value"].astype(str).str.lower()
                          .isin(["true", "1", "yes"]), "Patent_ID"])
    wide["any_uncertain"] = wide["Patent_ID"].isin(unc_ids)

    images = image_table(long)
    mains = select_main_images(images)
    mains = apply_processing_manifest(mains, cfg)
    canon = wide.merge(mains, on=["base_patent_id", "arch_index"], how="left")

    canon["label_source_file"] = long["_source_file"].iloc[0]
    canon["ingest_date"] = datetime.date.today().isoformat()

    n_img = int(canon["image_path"].notna().sum())
    print(f"[build_canonical] {len(canon)} architectures "
          f"({canon['base_patent_id'].nunique()} base patents, "
          f"{int(canon['multi_arch'].sum())} rows in multi-arch patents); "
          f"{n_img} have an approved main image.")
    return canon, images


# ── coverage / QC ─────────────────────────────────────────────────────────────

def attribute_columns(canon: pd.DataFrame) -> List[str]:
    """Wide attribute columns (Section_Field pattern), excluding bookkeeping."""
    non_attr = {"Patent_ID", "base_patent_id", "arch_index", "n_architectures",
                "multi_arch", "assignee", "assignee_country", "priority_year",
                "year_bin", "cpc_first", "cpc_subclass", "any_uncertain",
                "image_path", "figure_id", "figure_type", "path", "patent_id",
                "perspective", "acSty", "acCol", "bgSty", "bgCol", "parts",
                "acState", "qualityFlag",
                "label_source_file", "ingest_date", "Publication/Issue Date"}
    return [c for c in canon.columns if c not in non_attr]


def coverage_table(canon: pd.DataFrame, cfg: Dict[str, Any],
                   only_with_image: bool = True) -> pd.DataFrame:
    """attribute x class -> count, flagging classes below min_class_count."""
    min_n = int(cfg["taxonomy"].get("min_class_count", 15))
    df = canon[canon["image_path"].notna()] if only_with_image else canon
    rows = []
    for col in attribute_columns(canon):
        vc = df[col].dropna().value_counts()
        for val, n in vc.items():
            rows.append({"attribute": col, "class": str(val), "n": int(n),
                         "null_pct": round(100 * df[col].isna().mean(), 1),
                         "below_min": bool(n < min_n)})
    return pd.DataFrame(rows).sort_values(["attribute", "n"],
                                          ascending=[True, False])


def validate_canonical(canon: pd.DataFrame) -> pd.DataFrame:
    """QC issue list: duplicate arch keys, missing images, missing metadata."""
    issues = []
    dup = canon.duplicated(["base_patent_id", "arch_index"])
    if dup.any():
        issues.append({"issue": "duplicate (base_patent_id, arch_index)",
                       "n": int(dup.sum()),
                       "detail": ", ".join(canon.loc[dup, "Patent_ID"].head(10))})
    no_img = canon["image_path"].isna()
    if no_img.any():
        issues.append({"issue": "architecture without approved main image",
                       "n": int(no_img.sum()),
                       "detail": ", ".join(canon.loc[no_img, "Patent_ID"].head(10))})
    no_meta = canon["assignee"].isna()
    if no_meta.any():
        issues.append({"issue": "no PatSeer metadata match",
                       "n": int(no_meta.sum()),
                       "detail": ", ".join(canon.loc[no_meta, "base_patent_id"]
                                           .drop_duplicates().head(10))})
    return pd.DataFrame(issues, columns=["issue", "n", "detail"])


def save_canonical(canon: pd.DataFrame, cfg: Dict[str, Any]) -> Path:
    """Write labels parquet/csv + a column dictionary under taxonomy output."""
    out_dir = Path(cfg["taxonomy"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # Wizard columns mix types (True / 2 / "text") -> stringify object columns
    # so parquet accepts them; NaN stays NaN.
    pq = canon.copy()
    for col in pq.columns[pq.dtypes.eq(object)]:
        pq[col] = pq[col].map(lambda v: v if pd.isna(v) else str(v))
    pq.to_parquet(out_dir / "labels_v1.parquet", index=False)
    canon.to_csv(out_dir / "labels_v1.csv", index=False)

    dict_rows = []
    for col in canon.columns:
        vals = canon[col].dropna()
        dict_rows.append({
            "column": col, "dtype": str(canon[col].dtype),
            "null_pct": round(100 * canon[col].isna().mean(), 1),
            "n_unique": int(vals.nunique()),
            "observed_values": "; ".join(map(str, sorted(vals.astype(str).unique())[:25])),
        })
    pd.DataFrame(dict_rows).to_csv(out_dir / "labels_v1_dictionary.csv", index=False)
    print(f"[save_canonical] wrote labels_v1.parquet/.csv + dictionary to {out_dir}")
    return out_dir
