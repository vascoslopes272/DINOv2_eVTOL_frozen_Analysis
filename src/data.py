"""Data layer: figure discovery + label loading.

ALL Excel / label / filename-parsing logic lives here — never in the notebook.

Two public entry points:
    list_figures(cfg)  -> DataFrame[figure_id, patent_id, figure_type, path]
    load_labels(cfg)   -> DataFrame indexed by patent_id, columns = label + confound cols

Patent-id parsing convention
-----------------------------
Figure exports follow the fixed convention::

    {RecordNumber}_{CAT}_{CPC}_p{page:03d}_c{crop:02d}.png

where ``CAT`` is ``SHR`` (shrouded rotor) or ``OPN`` (open rotor), e.g.::

    US2022267016A1_SHR_B64C2720_p003_c01.png
        -> patent_id="US2022267016A1", shrouded_open="SHR",
           cpc="B64C2720", figure_type="p003_c01"

The convention is encoded in :data:`FIGURE_FILENAME_REGEX` below and is an
ASSUMPTION — :func:`list_figures` prints a sample of parses so you can confirm
(or adjust the regex) before trusting the mapping. ``figure_type`` here is the
page/crop marker parsed from the filename, NOT a semantic view type
(top/side/perspective), which is not recoverable from the filename alone.
"""

from __future__ import annotations

import glob
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Image extensions we treat as figures.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Convention: {RecordNumber}_{CAT}_{CPC}_p{page:03d}_c{crop:02d}
#   group "patent_id"      : the record number (e.g. "US2022267016A1")
#   group "shrouded_open"  : the category, "SHR" (shrouded) or "OPN" (open)
#   group "cpc"            : the CPC code (e.g. "B64C2720")
#   group "page" / "crop"  : zero-padded page and crop indices
FIGURE_FILENAME_REGEX = re.compile(
    r"^(?P<patent_id>[A-Z]{2}\w+?)"
    r"_(?P<shrouded_open>SHR|OPN)"
    r"_(?P<cpc>[A-Z0-9]+)"
    r"_p(?P<page>\d{3})"
    r"_c(?P<crop>\d{2})$",
    re.IGNORECASE,
)

# Number of sample parses to print for confirmation.
PARSE_PREVIEW_N = 12


def _is_edit_me(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "EDIT-ME"


def parse_patent_id(
    stem: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    """Parse a filename stem (no extension) into its four components.

    Convention: ``{RecordNumber}_{CAT}_{CPC}_p{page:03d}_c{crop:02d}``.

    Returns ``(patent_id, shrouded_open, cpc, figure_type)`` where:
        * ``patent_id``     - the record number, e.g. ``"US2022267016A1"``
        * ``shrouded_open`` - the category ``"SHR"`` or ``"OPN"`` (upper-cased)
        * ``cpc``           - the CPC code, e.g. ``"B64C2720"`` (upper-cased)
        * ``figure_type``   - the ``p###_c##`` page/crop marker

    If the stem does not match the convention, returns
    ``(stem, None, None, None)`` rather than raising, so a stray file never
    crashes a directory scan.
    """
    m = FIGURE_FILENAME_REGEX.match(stem)
    if not m:
        return stem, None, None, None

    patent_id = m.group("patent_id")
    shrouded_open = m.group("shrouded_open").upper()
    cpc = m.group("cpc").upper()
    figure_type = f"p{m.group('page')}_c{m.group('crop')}"

    return patent_id, shrouded_open, cpc, figure_type


def list_figures(cfg: Dict[str, Any], preview: bool = True) -> pd.DataFrame:
    """Recursively scan ``paths.image_dir`` and return a figure-level DataFrame.

    Columns: ``figure_id`` (filename), ``patent_id`` (parsed), ``figure_type``
    (parsed marker or None), ``path`` (absolute path as str).

    When ``preview`` is True, prints a sample of parses + summary so the
    filename convention can be confirmed.
    """
    image_dir = cfg["paths"].get("image_dir")
    if _is_edit_me(cfg.get("paths", {}).get("image_dir", "EDIT-ME")):
        raise ValueError(
            "paths.image_dir is still 'EDIT-ME' in config.yaml — fill it in first."
        )
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    rows = []
    for p in sorted(image_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        patent_id, shrouded_open, cpc, figure_type = parse_patent_id(p.stem)
        rows.append(
            {
                "figure_id": p.name,
                "patent_id": patent_id,
                "shrouded_open": shrouded_open,
                "cpc": cpc,
                "figure_type": figure_type,
                "path": str(p.resolve()),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "figure_id",
            "patent_id",
            "shrouded_open",
            "cpc",
            "figure_type",
            "path",
        ],
    )

    if preview:
        print(f"[list_figures] scanned: {image_dir}")
        print(f"[list_figures] found {len(df)} figures across "
              f"{df['patent_id'].nunique()} distinct patent_ids\n")
        print(f"[list_figures] CONFIRM the parse convention "
              f"(showing first {min(PARSE_PREVIEW_N, len(df))}):")
        if len(df):
            preview_df = df.head(PARSE_PREVIEW_N)[
                ["figure_id", "patent_id", "shrouded_open", "cpc", "figure_type"]
            ]
            with pd.option_context("display.max_colwidth", 80):
                print(preview_df.to_string(index=False))
        n_no_marker = int(df["figure_type"].isna().sum())
        if n_no_marker:
            print(f"\n[list_figures] note: {n_no_marker} files had no figure marker "
                  f"(figure_type=None) — patent_id = full stem for those.")
        print()

    return df


def resolve_duplicate_roots(cfg: Dict[str, Any]) -> Dict[str, str]:
    """patent_id -> duplicate-chain root patent_id, pooled across every
    Excel matching ``paths.reviewed_xlsx_glob`` (Patent-Labelling-Tools'
    02a_preprocessing output; the T1 ``duplicateId`` link, keyed at the base
    patent_id level — never arch-suffixed).

    A patent with no duplicateId row (or a blank one) maps to itself. Chains
    are resolved transitively (A dup-of B dup-of C -> C) and across batches,
    since a duplicate can reference a patent reviewed in a different batch.
    Cycle-safe: a chain that loops back on itself stops where it started.
    """
    pattern = cfg["paths"].get("reviewed_xlsx_glob")
    if not pattern or _is_edit_me(pattern):
        raise ValueError("paths.reviewed_xlsx_glob is not set in config.yaml")
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no reviewed xlsx matches {pattern} — run "
                                f"Patent-Labelling-Tools 02a_preprocessing first.")

    long = pd.concat([pd.read_excel(f, sheet_name="Review") for f in files], ignore_index=True)

    dup_rows = long.loc[
        (long["Field"] == "duplicateId") & long["Value"].notna()
        & (long["Value"].astype(str).str.strip() != "")
    ]
    dup_target = (
        dup_rows.assign(Patent_ID=dup_rows["Patent_ID"].astype(str))
        .set_index("Patent_ID")["Value"].astype(str).str.strip()
    )

    def _root(pid: str) -> str:
        seen = set()
        current = str(pid)
        while current not in seen:
            seen.add(current)
            target = dup_target.get(current)
            if target is None:
                return current
            current = target
        return current  # cycle guard — stop where it started

    all_ids = long["Patent_ID"].astype(str).unique()
    return {pid: _root(pid) for pid in all_ids}


def list_figures_manifest(cfg: Dict[str, Any], scope: str = "all",
                          preview: bool = True,
                          dedupe_architectures: bool = False) -> pd.DataFrame:
    """Figure table from Stage-02 processing manifests — no filename parsing.

    Reads every CSV matching ``paths.processing_manifest_glob`` (written by
    Patent-Labelling-Tools' ``processor.process_batch``: one row per wizard
    image, with the processed 518px path plus the reviewer's per-image labels).

    ``scope``: ``"all"`` = every approved processed figure (dst_all);
    ``"main"`` = only the flat canonical mains (dst_main).

    ``dedupe_architectures``: when True, drops every figure belonging to a
    patent that is a labelled duplicate (T1 ``duplicateId``, any
    duplicateType) of another patent, keeping only the chain ROOT's figures
    (see :func:`resolve_duplicate_roots`) — so an architecture filed under N
    near-identical patents contributes exactly one entry, not N.

    Returns the same core columns as :func:`list_figures` (``figure_id``,
    ``patent_id``, ``figure_type``, ``path``) so it drops into
    ``embeddings.compute_embeddings`` unchanged, plus ``arch``, ``is_main``
    and the style labels (``bg_sty``/``bg_col``/``ac_sty``/``ac_col``).
    """
    pattern = cfg["paths"].get("processing_manifest_glob")
    if not pattern or _is_edit_me(pattern):
        raise ValueError("paths.processing_manifest_glob is not set in config.yaml")
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no manifest matches {pattern} — run "
                                f"Patent-Labelling-Tools 02_processing first.")

    m = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    m = m[m["processed"] == True]                                # noqa: E712
    path_col = {"all": "dst_all", "main": "dst_main"}[scope]
    m = m[m[path_col].notna()]

    df = pd.DataFrame({
        "figure_id": m[path_col].map(lambda p: Path(str(p)).name),
        "patent_id": m["base_patent_id"],
        "figure_type": np.where(m["is_main"].astype(bool), "main", "other"),
        "path": m[path_col],
        "arch": m["arch"],
        "is_main": m["is_main"].astype(bool),
        "bg_sty": m.get("bg_sty"), "bg_col": m.get("bg_col"),
        "ac_sty": m.get("ac_sty"), "ac_col": m.get("ac_col"),
    }).reset_index(drop=True)

    missing = ~df["path"].map(lambda p: Path(str(p)).exists())
    if missing.any():
        print(f"[list_figures_manifest] WARNING: {int(missing.sum())} manifest "
              f"paths missing on disk — excluded.")
        df = df[~missing].reset_index(drop=True)

    if dedupe_architectures:
        roots = resolve_duplicate_roots(cfg)
        is_root = df["patent_id"].map(lambda p: roots.get(p, p) == p)
        n_dropped_patents = df.loc[~is_root, "patent_id"].nunique()
        if preview and n_dropped_patents:
            print(f"[list_figures_manifest] dedupe_architectures: dropped "
                  f"{int((~is_root).sum())} figure(s) belonging to "
                  f"{n_dropped_patents} duplicate patent(s) (kept each "
                  f"chain's original)")
        df = df[is_root].reset_index(drop=True)

    if preview:
        print(f"[list_figures_manifest] {len(files)} manifest(s), scope={scope}: "
              f"{len(df)} figures across {df['patent_id'].nunique()} patents "
              f"({int(df['is_main'].sum())} mains)")
    return df


def load_labels(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Read the labelled Excel and return a clean, patent_id-indexed DataFrame.

    The returned frame is indexed by ``patent_id`` and exposes exactly
    ``data.label_cols + data.confound_cols``. Requested columns that are absent
    in the sheet are created as all-NaN (with a warning) so downstream code does
    not crash while labelling is still in progress. Patents present in the sheet
    but not yet labelled simply carry NaN in the relevant cells.
    """
    data_cfg = cfg.get("data", {})
    paths = cfg.get("paths", {})

    labels_excel = paths.get("labels_excel")
    if _is_edit_me(labels_excel) or labels_excel is None:
        raise ValueError(
            "paths.labels_excel is still 'EDIT-ME' in config.yaml — fill it in first."
        )
    labels_excel = Path(labels_excel)
    if not labels_excel.exists():
        raise FileNotFoundError(f"labels_excel does not exist: {labels_excel}")

    id_col = data_cfg.get("patent_id_col")
    if _is_edit_me(id_col) or id_col is None:
        raise ValueError(
            "data.patent_id_col is still 'EDIT-ME' in config.yaml — fill it in first."
        )

    label_cols = list(data_cfg.get("label_cols", []) or [])
    confound_cols = list(data_cfg.get("confound_cols", []) or [])
    label_cols = [c for c in label_cols if not _is_edit_me(c)]
    confound_cols = [c for c in confound_cols if not _is_edit_me(c)]
    wanted = label_cols + confound_cols

    df = pd.read_excel(labels_excel, engine="openpyxl")

    if id_col not in df.columns:
        raise KeyError(
            f"patent_id_col '{id_col}' not found in {labels_excel.name}. "
            f"Available columns: {list(df.columns)}"
        )

    # Build the clean frame; create missing requested columns as NaN.
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        warnings.warn(
            f"load_labels: columns not in Excel (created as NaN): {missing}",
            stacklevel=2,
        )
    for c in missing:
        df[c] = pd.NA

    out = df[[id_col] + wanted].copy()
    out[id_col] = out[id_col].astype("string").str.strip()

    # De-duplicate patent ids (keep first), key the index.
    n_dupes = int(out[id_col].duplicated().sum())
    if n_dupes:
        warnings.warn(
            f"load_labels: {n_dupes} duplicate patent_id rows — keeping first.",
            stacklevel=2,
        )
        out = out.drop_duplicates(subset=[id_col], keep="first")

    out = out.set_index(id_col)
    out.index.name = "patent_id"
    return out


if __name__ == "__main__":
    # Quick visual confirmation of parse_patent_id on known stems.
    _examples = [
        "US2022267016A1_SHR_B64C2720_p003_c01",
        "EP3456789A1_OPN_B64C3902_p001_c02",
        "DE102024105440A1_SHR_B64U1020_p012_c00",
        "some_random_filename_that_doesnt_match",
    ]
    print("parse_patent_id self-test")
    print("=" * 70)
    for _stem in _examples:
        _patent_id, _shrouded_open, _cpc, _figure_type = parse_patent_id(_stem)
        print(f"stem          : {_stem}")
        print(f"  patent_id     : {_patent_id}")
        print(f"  shrouded_open : {_shrouded_open}")
        print(f"  cpc           : {_cpc}")
        print(f"  figure_type   : {_figure_type}")
        print("-" * 70)
