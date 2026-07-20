"""Label-only QC, coverage, and batch-reconciliation stats.

Operates purely on the canonical label table from :mod:`src.label_export`
(``build_canonical``) — nothing here touches embeddings. This is the
"how many patents/architectures reviewed, approved, duplicated, and covered
per taxonomy class" audit that used to live inside the embedding-analysis
pipeline (Supervisor Report §1.1/1.2); it's label-only so it lives here,
next to ``label_export.py``, instead of in the joint analysis repo.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


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


def batch_summary(long: pd.DataFrame) -> pd.DataFrame:
    """Per-batch reviewed/approved/architecture counts (Supervisor Report Table 1).

    ``long`` is the pooled long-format frame from
    :func:`src.label_export.load_long` (has ``_source_file``).
    """
    t1 = long[long["Section"] == "T1"]
    rows = []
    for src, g in t1.groupby("_source_file"):
        approved = g[g["Field"] == "isApproved"]
        n_reviewed = g["Patent_ID"].nunique()
        n_approved = int((approved["Value"].astype(str).str.lower() == "true").sum())
        n_disapproved = int((approved["Value"].astype(str).str.lower() == "false").sum())
        rows.append({"batch": src, "n_reviewed": n_reviewed,
                     "n_approved": n_approved, "n_disapproved": n_disapproved,
                     "n_unresolved": n_reviewed - n_approved - n_disapproved})
    return pd.DataFrame(rows)


def duplicate_type_counts(long: pd.DataFrame) -> pd.DataFrame:
    """Per-batch counts of the ``duplicateType`` META field (Supervisor Report Table 2)."""
    dup = long[long["Field"] == "duplicateType"]
    return (dup.groupby(["_source_file", "Value"]).size()
            .rename("n").reset_index()
            .sort_values(["_source_file", "Value"]))
