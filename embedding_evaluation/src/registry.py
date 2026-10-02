"""Pipeline registry: resolves a pipeline name to its embeddings folder.

Expects ``paths.embeddings_root`` (set in config.yaml) to contain one
subfolder per extraction pipeline, each written by eVTOL-Embedding-Extraction
in the standardized format::

    <embeddings_root>/<pipeline_name>/
        emb_<variant>.npy      # one per (layer,pooling) or a single file
        metadata.parquet       # figure_id, patent_id, arch_index, image_path
        manifest.json          # pipeline name, model checkpoint/version,
                                # extraction config, date, git commit

No special-casing per pipeline: everything needed to load a pipeline's
embeddings is declared in its own ``manifest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def list_pipelines(cfg: Dict[str, Any]) -> List[str]:
    """Names of every pipeline folder under paths.embeddings_root."""
    root = Path(cfg["paths"]["embeddings_root"])
    return sorted(p.name for p in root.iterdir()
                 if p.is_dir() and (p / "manifest.json").exists())


def load_manifest(cfg: Dict[str, Any], pipeline: str) -> Dict[str, Any]:
    path = Path(cfg["paths"]["embeddings_root"]) / pipeline / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"no manifest.json for pipeline {pipeline!r} at {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_embeddings(cfg: Dict[str, Any], pipeline: str, variant: str | None = None
                    ) -> Dict[str, Any]:
    """Load one pipeline's embeddings + metadata.

    ``variant`` selects a specific ``emb_<variant>.npy`` (e.g. layer/pooling
    combo); if omitted and the pipeline has exactly one ``emb_*.npy`` file,
    that one is used.
    """
    folder = Path(cfg["paths"]["embeddings_root"]) / pipeline
    manifest = load_manifest(cfg, pipeline)
    metadata = pd.read_parquet(folder / "metadata.parquet")

    npy_files = sorted(folder.glob("emb_*.npy"))
    if variant is not None:
        target = folder / f"emb_{variant}.npy"
        if target not in npy_files:
            raise FileNotFoundError(f"no {target.name} for pipeline {pipeline!r}")
        npy_files = [target]
    elif len(npy_files) != 1:
        raise ValueError(
            f"pipeline {pipeline!r} has {len(npy_files)} emb_*.npy files — "
            f"pass `variant=` to select one: "
            f"{[f.stem.removeprefix('emb_') for f in npy_files]}")

    arrays = {f.stem.removeprefix("emb_"): np.load(f) for f in npy_files}
    return {"arrays": arrays, "metadata": metadata, "manifest": manifest}
