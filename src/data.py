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

import re
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
