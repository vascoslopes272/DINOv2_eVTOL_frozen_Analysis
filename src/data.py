"""Data layer: figure discovery + label loading.

ALL Excel / label / filename-parsing logic lives here — never in the notebook.

Two public entry points:
    list_figures(cfg)  -> DataFrame[figure_id, patent_id, figure_type, path]
    load_labels(cfg)   -> DataFrame indexed by patent_id, columns = label + confound cols

Patent-id parsing convention
-----------------------------
PatSeer figure exports are typically named ``<PATENT_ID><sep><figure marker>``,
e.g.::

    US20210107642A1_fig1.png      -> patent_id="US20210107642A1", figure_type="fig1"
    US-20210107642-A1-figure-3.png-> patent_id="US-20210107642-A1", figure_type="figure3"
    EP1234567B1_0003.png          -> patent_id="EP1234567B1",      figure_type="0003"
    WO2020012345A1.png            -> patent_id="WO2020012345A1",    figure_type=None

The convention is encoded in :data:`FIGURE_FILENAME_REGEX` below and is an
ASSUMPTION — :func:`list_figures` prints a sample of parses so you can confirm
(or adjust the regex) before trusting the mapping. ``figure_type`` here is the
figure *marker token* parsed from the filename, NOT a semantic view type
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

# Convention: patent id, then an optional separator + figure marker.
#   group "patent_id"  : everything up to the figure marker (lazy)
#   group "marker"     : the figure-indicator word (fig / figure / sheet / drw ...)
#   group "num"        : a number attached to the marker, OR a bare trailing number
FIGURE_FILENAME_REGEX = re.compile(
    r"^(?P<patent_id>.+?)"
    r"(?:"
    r"[ _\-.]*?(?P<marker>fig(?:ure)?|sheet|dr?w(?:g)?|drawing)[ _\-.]*(?P<num>\d+)?"
    r"|"
    r"[ _\-.]+(?P<trailing>\d+)"
    r")?$",
    re.IGNORECASE,
)

# Number of sample parses to print for confirmation.
PARSE_PREVIEW_N = 12


def _is_edit_me(value: Any) -> bool:
    return isinstance(value, str) and value.strip().upper() == "EDIT-ME"


def parse_patent_id(stem: str) -> Tuple[str, Optional[str]]:
    """Parse ``(patent_id, figure_type)`` from a filename stem (no extension).

    Returns the patent id and a normalized figure marker (e.g. ``"fig1"`` or
    ``"0003"``), or ``None`` for the marker if none is present.
    """
    m = FIGURE_FILENAME_REGEX.match(stem)
    if not m:
        return stem, None

    patent_id = m.group("patent_id").strip(" _-.")
    marker = m.group("marker")
    num = m.group("num")
    trailing = m.group("trailing")

    if marker is not None:
        figure_type = marker.lower() + (num or "")
    elif trailing is not None:
        figure_type = trailing
    else:
        figure_type = None

    return patent_id, figure_type


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
        patent_id, figure_type = parse_patent_id(p.stem)
        rows.append(
            {
                "figure_id": p.name,
                "patent_id": patent_id,
                "figure_type": figure_type,
                "path": str(p.resolve()),
            }
        )

    df = pd.DataFrame(rows, columns=["figure_id", "patent_id", "figure_type", "path"])

    if preview:
        print(f"[list_figures] scanned: {image_dir}")
        print(f"[list_figures] found {len(df)} figures across "
              f"{df['patent_id'].nunique()} distinct patent_ids\n")
        print(f"[list_figures] CONFIRM the parse convention "
              f"(showing first {min(PARSE_PREVIEW_N, len(df))}):")
        if len(df):
            preview_df = df.head(PARSE_PREVIEW_N)[["figure_id", "patent_id", "figure_type"]]
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
    from config_loader import load_config

    cfg = load_config()
    figs = list_figures(cfg)
    print(figs.head())
