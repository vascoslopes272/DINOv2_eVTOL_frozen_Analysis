"""Frozen, off-the-shelf DINOv2 feature extraction.

The model is used strictly as a fixed feature extractor: ``model.eval()`` +
``torch.no_grad()``, no fine-tuning. Architecture details (number of transformer
blocks, hidden dim, number of register tokens) are DETECTED from ``model.config``
— never hardcoded — because they differ between ``facebook/dinov2-*`` (0 register
tokens) and ``facebook/dinov2-with-registers-*`` (4 register tokens).

hidden_states from a HF DINOv2 forward with ``output_hidden_states=True`` has
length ``num_hidden_layers + 1``: index 0 is the embedding output and index ``i``
is the output of transformer block ``i``. So ``layer 22 == hidden_states[22]``.

For each requested layer we produce two poolings, each L2-normalized (cosine
geometry):
    cls        = hidden_states[L][:, 0, :]
    mean_patch = hidden_states[L][:, n_prefix:, :].mean(dim=1)
where ``n_prefix = 1 (CLS) + num_register_tokens``.

Embeddings are stored PER FIGURE (not aggregated) as one ``.npy`` per
(layer, pooling) plus a metadata table, under ``<output_dir>/embeddings/``.
Patent-level aggregation is a separate, swappable step.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm


@dataclass
class ModelInfo:
    """Detected, not hardcoded, architecture facts. All logged at load time."""

    model_name: str
    variant: str                # "base" | "with-registers" (inferred)
    num_hidden_layers: int
    hidden_dim: int
    num_register_tokens: int
    n_prefix: int               # 1 (CLS) + num_register_tokens
    device: str


def load_model(cfg: Dict[str, Any]):
    """Load a frozen DINOv2 + its own image processor and detect architecture.

    Returns ``(model, processor, ModelInfo)``.
    """
    from transformers import AutoImageProcessor, AutoModel

    model_name = cfg["analysis"]["model"]
    device = cfg["analysis"].get("device", "cpu")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    conf = model.config
    num_hidden_layers = int(conf.num_hidden_layers)
    hidden_dim = int(conf.hidden_size)
    # base DINOv2 has no such attribute -> 0; -with-registers exposes 4.
    num_register_tokens = int(getattr(conf, "num_register_tokens", 0) or 0)
    n_prefix = 1 + num_register_tokens
    variant = "with-registers" if num_register_tokens > 0 else "base"

    info = ModelInfo(
        model_name=model_name,
        variant=variant,
        num_hidden_layers=num_hidden_layers,
        hidden_dim=hidden_dim,
        num_register_tokens=num_register_tokens,
        n_prefix=n_prefix,
        device=str(device),
    )

    print("[load_model] detected architecture:")
    for k, v in asdict(info).items():
        print(f"    {k:>22}: {v}")
    print(f"    hidden_states length per forward = {num_hidden_layers + 1} "
          f"(index 0 = embeddings, index i = block i output)\n")

    return model, processor, info


def _validate_layers(layers: List[int], info: ModelInfo) -> None:
    max_idx = info.num_hidden_layers  # hidden_states is 0..num_hidden_layers
    bad = [L for L in layers if L < 0 or L > max_idx]
    if bad:
        raise ValueError(
            f"Requested layers {bad} out of range. Valid hidden_states indices "
            f"are 0..{max_idx} for {info.model_name}."
        )


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def compute_embeddings(
    figures: pd.DataFrame,
    cfg: Dict[str, Any],
    model=None,
    processor=None,
    info: ModelInfo | None = None,
) -> Dict[str, Any]:
    """Compute per-figure embeddings for all (layer, pooling) combinations.

    ``figures`` is the DataFrame from ``data.list_figures`` (needs ``path``,
    ``figure_id``, ``patent_id``, ``figure_type``).

    Returns a dict::

        {
          "metadata": DataFrame(row -> figure_id/patent_id/figure_type),
          "info": ModelInfo,
          "arrays": { (layer, pooling): np.ndarray [n_figures, hidden_dim] },
        }

    Row order is identical across every array and the metadata table.
    """
    if model is None or processor is None or info is None:
        model, processor, info = load_model(cfg)

    layers: List[int] = list(cfg["analysis"]["layers"])
    poolings: List[str] = list(cfg["analysis"]["pooling"])
    batch_size: int = int(cfg["analysis"]["batch_size"])
    device = info.device
    _validate_layers(layers, info)

    unknown = [p for p in poolings if p not in ("cls", "mean_patch")]
    if unknown:
        raise ValueError(f"Unknown pooling(s) {unknown}; expected cls / mean_patch.")

    # Accumulators: one growing list of row-vectors per (layer, pooling).
    acc: Dict[tuple, List[np.ndarray]] = {
        (L, pool): [] for L in layers for pool in poolings
    }
    meta_rows: List[Dict[str, Any]] = []

    paths = figures["path"].tolist()
    n = len(paths)
    for start in tqdm(range(0, n, batch_size), desc="embedding", unit="batch"):
        batch_rows = figures.iloc[start:start + batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_rows["path"]]
        inputs = processor(images=images, return_tensors="pt").to(device)

        out = model(**inputs, output_hidden_states=True)
        hs = out.hidden_states  # tuple length num_hidden_layers + 1

        for L in layers:
            layer_h = hs[L]                       # [B, seq, hidden]
            for pool in poolings:
                if pool == "cls":
                    vec = layer_h[:, 0, :]
                else:  # mean_patch
                    vec = layer_h[:, info.n_prefix:, :].mean(dim=1)
                vec = _l2_normalize(vec).float().cpu().numpy()
                acc[(L, pool)].append(vec)

        for _, r in batch_rows.iterrows():
            meta_rows.append(
                {
                    "figure_id": r["figure_id"],
                    "patent_id": r["patent_id"],
                    "figure_type": r["figure_type"],
                }
            )

    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in acc.items()}
    metadata = pd.DataFrame(meta_rows)
    return {"metadata": metadata, "info": info, "arrays": arrays}


def save_embeddings(result: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    """Persist per-figure embeddings + metadata under ``<output_dir>/embeddings/``.

    Writes one ``emb_layer{L}_{pooling}.npy`` per (layer, pooling), a
    ``metadata.parquet`` (row -> patent_id/figure_id/figure_type), and a small
    ``model_info.json``. Returns the embeddings directory.
    """
    import json

    out_dir = Path(cfg["paths"]["output_dir"]) / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    for (L, pool), arr in result["arrays"].items():
        np.save(out_dir / f"emb_layer{L}_{pool}.npy", arr)

    result["metadata"].to_parquet(out_dir / "metadata.parquet", index=False)
    with open(out_dir / "model_info.json", "w", encoding="utf-8") as fh:
        json.dump(asdict(result["info"]), fh, indent=2)

    print(f"[save_embeddings] wrote {len(result['arrays'])} arrays + metadata to {out_dir}")
    return out_dir


def load_embeddings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Reload what :func:`save_embeddings` wrote. Returns the same dict shape."""
    import json

    out_dir = Path(cfg["paths"]["output_dir"]) / "embeddings"
    metadata = pd.read_parquet(out_dir / "metadata.parquet")
    with open(out_dir / "model_info.json", "r", encoding="utf-8") as fh:
        info = ModelInfo(**json.load(fh))

    arrays: Dict[tuple, np.ndarray] = {}
    for L in cfg["analysis"]["layers"]:
        for pool in cfg["analysis"]["pooling"]:
            f = out_dir / f"emb_layer{L}_{pool}.npy"
            if f.exists():
                arrays[(L, pool)] = np.load(f)
    return {"metadata": metadata, "info": info, "arrays": arrays}


# -----------------------------------------------------------------------------
# Patent-level aggregation (swappable via config; decided empirically in Stage 5)
# -----------------------------------------------------------------------------
def aggregate_to_patent(
    figure_embeddings: np.ndarray,
    metadata: pd.DataFrame,
    method: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-figure embeddings to one vector per patent.

    ``figure_embeddings`` is ``[n_figures, dim]`` aligned row-for-row with
    ``metadata`` (which must carry ``patent_id`` and, for some methods,
    ``figure_type``). Returns a DataFrame indexed by ``patent_id`` whose columns
    are the aggregated embedding dimensions.

    Only ``"mean"`` is implemented; the others are stubs we will fill in once we
    pick an aggregation empirically (Stage 5).
    """
    if figure_embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"row mismatch: {figure_embeddings.shape[0]} embeddings vs "
            f"{len(metadata)} metadata rows."
        )

    if method == "mean":
        dim = figure_embeddings.shape[1]
        emb_df = pd.DataFrame(
            figure_embeddings,
            columns=[f"d{i}" for i in range(dim)],
        )
        emb_df["patent_id"] = metadata["patent_id"].values
        agg = emb_df.groupby("patent_id").mean()
        # Re-normalize the mean to keep cosine geometry.
        norms = np.linalg.norm(agg.values, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        agg.loc[:, :] = agg.values / norms
        return agg

    if method == "canonical":
        # TODO(Stage 5): pick a single canonical/representative figure per patent
        # (e.g. medoid by cosine, or the designated primary figure) instead of
        # averaging. Should return one row per patent_id, same column shape.
        raise NotImplementedError("aggregation.method='canonical' not implemented yet")

    if method == "per_figure_type":
        # TODO(Stage 5): aggregate within figure_type first (e.g. mean per
        # figure_type) then combine across types — possibly returning a wider
        # frame or a dict keyed by figure_type. Design TBD.
        raise NotImplementedError("aggregation.method='per_figure_type' not implemented yet")

    raise ValueError(f"Unknown aggregation method: {method!r}")


# -----------------------------------------------------------------------------
# Stage-0 QC (called by notebooks/10_qc_embeddings.ipynb — keeps it thin)
# -----------------------------------------------------------------------------
DEAD_DIM_THRESH = 1e-8
PAIRWISE_SAMPLE_SIZE = 512


def _pairwise_cosine_sample(arr: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Upper-triangle pairwise cosine of a random row sample.

    Rows are L2-normalized at compute time, so cosine == dot product. Returns
    the flat vector of off-diagonal similarities.
    """
    n = arr.shape[0]
    rng = np.random.default_rng(seed)
    k = min(sample_size, n)
    idx = rng.choice(n, size=k, replace=False)
    sub = arr[idx].astype(np.float64)
    # Re-normalize defensively (handles any reloaded/altered arrays).
    norms = np.linalg.norm(sub, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sub = sub / norms
    sims = sub @ sub.T
    iu = np.triu_indices(k, k=1)
    return sims[iu]


def qc_report(
    result: Dict[str, Any],
    sample_size: int = PAIRWISE_SAMPLE_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """Build the Stage-0 QC table: one row per (layer, pooling).

    Returns a DataFrame; the notebook just displays it and writes it to CSV.
    """
    info: ModelInfo = result["info"]
    metadata: pd.DataFrame = result["metadata"]
    arrays: Dict[tuple, np.ndarray] = result["arrays"]

    # figures-per-patent stats (identical across arrays — computed once).
    per_patent = metadata.groupby("patent_id").size()
    fpp_min, fpp_med, fpp_max = (
        int(per_patent.min()),
        float(per_patent.median()),
        int(per_patent.max()),
    )

    rows: List[Dict[str, Any]] = []
    for (L, pool), arr in sorted(arrays.items()):
        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        all_zero_count = int((~arr.any(axis=1)).sum())
        # exact-duplicate rows
        n_unique = np.unique(arr, axis=0).shape[0]
        dup_count = int(arr.shape[0] - n_unique)

        dim_var = arr.var(axis=0)
        dead_dims = int((dim_var < DEAD_DIM_THRESH).sum())

        sims = _pairwise_cosine_sample(arr, sample_size, seed)

        rows.append(
            {
                "layer": L,
                "pooling": pool,
                "variant": info.variant,
                "n_layers": info.num_hidden_layers,
                "hidden_dim": info.hidden_dim,
                "num_register_tokens": info.num_register_tokens,
                "n_prefix": info.n_prefix,
                "emb_dim": int(arr.shape[1]),
                "n_figures": int(arr.shape[0]),
                "n_patents": int(metadata["patent_id"].nunique()),
                "fig_per_patent_min": fpp_min,
                "fig_per_patent_median": fpp_med,
                "fig_per_patent_max": fpp_max,
                "nan_count": nan_count,
                "inf_count": inf_count,
                "all_zero_count": all_zero_count,
                "exact_duplicate_count": dup_count,
                "dim_var_mean": float(dim_var.mean()),
                "dim_var_min": float(dim_var.min()),
                "dim_var_max": float(dim_var.max()),
                "near_dead_dims": dead_dims,
                "cos_mean": float(sims.mean()),
                "cos_std": float(sims.std()),
                "cos_p05": float(np.percentile(sims, 5)),
                "cos_p50": float(np.percentile(sims, 50)),
                "cos_p95": float(np.percentile(sims, 95)),
            }
        )

    return pd.DataFrame(rows)


def qc_pass_fail(report: pd.DataFrame) -> bool:
    """Stage-0 hard gate: every (layer, pooling) must have 0 NaN, 0 Inf, 0 all-zero."""
    return bool(
        (report["nan_count"] == 0).all()
        and (report["inf_count"] == 0).all()
        and (report["all_zero_count"] == 0).all()
    )
