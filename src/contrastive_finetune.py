"""Supervised-contrastive (NT-Xent) fine-tuning of DINOv2, style as nuisance.

Frozen DINOv2 (see :mod:`src.embeddings`) clusters patent figures by drawing
style (line drawing vs. rendered vs. draft, ...) rather than by architecture
class. This module unfreezes only the LAST FEW transformer blocks of the same
``analysis.model`` and fine-tunes with a supervised NT-Xent loss: a positive
pair is two architectures' main figures that share ``contrastive_finetune.
class_col`` (e.g. ``G1_topType``) but differ on ``contrastive_finetune.
style_cols`` (e.g. ``acSty``) — pulling same-class-different-style pairs
together so style stops dominating the embedding geometry.

Pair source is :func:`src.label_export.build_canonical`'s per-architecture
table (one row per (base_patent_id, arch_index) with the class label, style
tags and the Stage-02 processed image path already joined) — no separate
manifest join needed.

Training operates on the CLS token of the model's FINAL transformer block,
L2-normalized — the same geometry as ``embeddings.py``'s ``cls`` pooling at
``layer = num_hidden_layers``.

Three public entry points:
    load_pair_table(cfg)            -> DataFrame of class/style-eligible figures
    run_finetune(cfg)               -> trains, checkpoints, returns the fine-tuned (model, processor, info)
    save_finetuned_pipeline(...)    -> writes embeddings_root/<pipeline_name>/ in repo 3's registry.py format
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from . import embeddings
from . import label_export

# ---------------------------------------------------------------------------
# Pair source
# ---------------------------------------------------------------------------
def load_pair_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Class/style-eligible main-image figures, one row per architecture.

    Columns: ``base_patent_id``, ``arch_index``, ``path``, ``figure_id``,
    ``patent_id`` (= base_patent_id, kept for embeddings.compute_embeddings
    compatibility), ``figure_type`` (constant "main"), the class column and
    the style column(s) named exactly as in ``contrastive_finetune.class_col``
    / ``style_cols``.

    Drops rows missing an image, the class label, or any style column. Then
    drops classes with fewer than ``min_class_count`` remaining rows OR whose
    remaining rows carry only a single distinct style combination (no valid
    positive pair possible).
    """
    fcfg = cfg["contrastive_finetune"]
    class_col = fcfg["class_col"]
    style_cols: List[str] = list(fcfg["style_cols"])

    canon, _images = label_export.build_canonical(cfg)

    wanted = ["base_patent_id", "arch_index", "image_path", class_col] + style_cols
    missing = [c for c in wanted if c not in canon.columns]
    if missing:
        raise KeyError(
            f"load_pair_table: columns not found in build_canonical() output: "
            f"{missing}. Check contrastive_finetune.class_col/style_cols against "
            f"the wizard taxonomy fields."
        )

    df = canon[wanted].copy()
    df = df.rename(columns={"image_path": "path"})
    df = df.dropna(subset=["path", class_col] + style_cols)
    df = df[df["path"].map(lambda p: Path(str(p)).exists())]

    df["style_key"] = df[style_cols].astype(str).agg("|".join, axis=1)

    min_count = int(fcfg.get("min_class_count", 2))
    class_sizes = df.groupby(class_col).size()
    eligible_classes = class_sizes[class_sizes >= min_count].index
    n_dropped_small = int((~df[class_col].isin(eligible_classes)).sum())

    style_span = df.groupby(class_col)["style_key"].nunique()
    multi_style_classes = style_span[style_span >= 2].index
    eligible_classes = eligible_classes.intersection(multi_style_classes)

    n_before = len(df)
    df = df[df[class_col].isin(eligible_classes)].reset_index(drop=True)

    df["figure_id"] = df["path"].map(lambda p: Path(str(p)).name)
    df["patent_id"] = df["base_patent_id"]
    df["figure_type"] = "main"

    print(f"[load_pair_table] {n_before} main figures with class+style labels -> "
          f"{len(df)} kept ({n_dropped_small} dropped for class_col count < "
          f"{min_count}, rest dropped for having only one style within their "
          f"class); {df[class_col].nunique()} classes usable for pair sampling.")
    return df


# ---------------------------------------------------------------------------
# Positive-pair dataset
# ---------------------------------------------------------------------------
class PositivePairDataset(Dataset):
    """One item = (image_a, image_b): same class_col, different style_key.

    Length equals the number of eligible figures (each used as an anchor once
    per epoch); the partner is resampled fresh each call, so re-iterating the
    DataLoader across epochs sees different pairings.
    """

    def __init__(self, pair_table: pd.DataFrame, class_col: str, processor,
                 input_size: int, seed: int = 42):
        self.df = pair_table.reset_index(drop=True)
        self.class_col = class_col
        self.processor = processor
        self.input_size = input_size
        self.rng = np.random.default_rng(seed)

        self._by_class: Dict[Any, np.ndarray] = {
            cls: grp.index.to_numpy()
            for cls, grp in self.df.groupby(class_col)
        }

    def __len__(self) -> int:
        return len(self.df)

    def _sample_partner(self, anchor_idx: int) -> int:
        row = self.df.iloc[anchor_idx]
        candidates = self._by_class[row[self.class_col]]
        candidates = candidates[
            (candidates != anchor_idx)
            & (self.df.loc[candidates, "style_key"].to_numpy() != row["style_key"])
        ]
        return int(self.rng.choice(candidates))

    def _load(self, idx: int) -> torch.Tensor:
        path = self.df.iloc[idx]["path"]
        img = Image.open(path).convert("RGB")
        px = self.processor(images=[img], return_tensors="pt")["pixel_values"][0]
        return px

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        partner_idx = self._sample_partner(idx)
        return self._load(idx), self._load(partner_idx)


# ---------------------------------------------------------------------------
# Model: same loader as embeddings.py, then selectively unfrozen
# ---------------------------------------------------------------------------
def load_model_for_finetune(cfg: Dict[str, Any]):
    """Load ``analysis.model`` via ``embeddings.load_model`` (frozen), then
    unfreeze the last ``contrastive_finetune.unfreeze_last_n_blocks``
    transformer blocks plus the final layernorm. Everything else (embeddings,
    earlier blocks) stays frozen. Returns ``(model, processor, info)``.
    """
    finetune_cfg = {**cfg, "analysis": {**cfg["analysis"],
                                        "device": cfg["contrastive_finetune"]["device"]}}
    model, processor, info = embeddings.load_model(finetune_cfg)

    n_unfreeze = int(cfg["contrastive_finetune"]["unfreeze_last_n_blocks"])
    blocks = model.encoder.layer
    if n_unfreeze > len(blocks):
        raise ValueError(
            f"unfreeze_last_n_blocks={n_unfreeze} exceeds the model's "
            f"{len(blocks)} transformer blocks."
        )
    for block in blocks[len(blocks) - n_unfreeze:]:
        for p in block.parameters():
            p.requires_grad_(True)
    if hasattr(model, "layernorm"):
        for p in model.layernorm.parameters():
            p.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[load_model_for_finetune] unfroze last {n_unfreeze}/{len(blocks)} blocks "
          f"+ final layernorm: {n_trainable:,}/{n_total:,} params trainable "
          f"({100 * n_trainable / n_total:.1f}%)")

    return model, processor, info


# ---------------------------------------------------------------------------
# NT-Xent loss
# ---------------------------------------------------------------------------
def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Supervised NT-Xent for a batch of ``(a, b)`` positive pairs.

    ``z_a``/``z_b`` are ``[N, D]``, already L2-normalized. Interleaves to
    ``[2N, D]`` (a0, b0, a1, b1, ...) so each row's positive is its neighbor;
    every other row (including the row's own duplicate-view slot) is a
    negative. Standard SimCLR-style symmetric cross-entropy.
    """
    n = z_a.shape[0]
    z = torch.stack([z_a, z_b], dim=1).reshape(2 * n, -1)  # [2N, D]

    sim = z @ z.T / temperature                            # [2N, 2N]
    self_mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float("-inf"))

    targets = torch.arange(2 * n, device=z.device)
    targets = targets + torch.where(targets % 2 == 0, 1, -1)  # 0<->1, 2<->3, ...

    return F.cross_entropy(sim, targets)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{run_id}.log"

    logger = logging.getLogger(f"contrastive_finetune.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"log file: {log_path}")
    return logger


def _log_gpu_stats(logger: logging.Logger, device: str) -> None:
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return
    idx = torch.device(device).index or 0
    alloc = torch.cuda.memory_allocated(idx) / 1e9
    reserved = torch.cuda.memory_reserved(idx) / 1e9
    name = torch.cuda.get_device_name(idx)
    logger.info(f"  gpu={device} ({name}): {alloc:.2f} GB allocated, "
                f"{reserved:.2f} GB reserved")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_finetune(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train, checkpoint periodically, and return the fine-tuned model state.

    Returns ``{"model": model, "processor": processor, "info": info,
    "history": DataFrame[epoch, avg_loss, seconds]}``.
    """
    fcfg = cfg["contrastive_finetune"]
    device = fcfg["device"]
    output_dir = Path(fcfg["output_dir"])
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_logger(output_dir / "logs")

    torch.manual_seed(int(fcfg.get("seed", 42)))
    np.random.seed(int(fcfg.get("seed", 42)))

    logger.info(f"device={device} | analysis.model={cfg['analysis']['model']} | "
                f"unfreeze_last_n_blocks={fcfg['unfreeze_last_n_blocks']} | "
                f"lr={fcfg['learning_rate']} | batch_size={fcfg['batch_size']} | "
                f"temperature={fcfg['temperature']} | num_epochs={fcfg['num_epochs']}")

    pair_table = load_pair_table(cfg)
    model, processor, info = load_model_for_finetune(cfg)

    input_size = int(cfg["analysis"].get("input_size", 0) or 0)
    dataset = PositivePairDataset(pair_table, fcfg["class_col"], processor,
                                  input_size, seed=int(fcfg.get("seed", 42)))
    loader = DataLoader(dataset, batch_size=int(fcfg["batch_size"]), shuffle=True,
                        drop_last=True, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(fcfg["learning_rate"]))
    temperature = float(fcfg["temperature"])
    checkpoint_every = int(fcfg.get("checkpoint_every_n_epochs", 1))

    history: List[Dict[str, Any]] = []
    num_epochs = int(fcfg["num_epochs"])
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        epoch_losses: List[float] = []

        for img_a, img_b in loader:
            img_a = img_a.to(device)
            img_b = img_b.to(device)

            out_a = model(pixel_values=img_a, output_hidden_states=True)
            out_b = model(pixel_values=img_b, output_hidden_states=True)
            z_a = F.normalize(out_a.hidden_states[-1][:, 0, :], dim=-1)
            z_b = F.normalize(out_b.hidden_states[-1][:, 0, :], dim=-1)

            loss = nt_xent_loss(z_a, z_b, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.detach().cpu()))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        elapsed = time.time() - t0
        logger.info(f"epoch {epoch}/{num_epochs} | avg_loss={avg_loss:.4f} | "
                    f"{elapsed:.1f}s | {len(epoch_losses)} batches")
        _log_gpu_stats(logger, device)
        history.append({"epoch": epoch, "avg_loss": avg_loss, "seconds": elapsed})

        if epoch % checkpoint_every == 0 or epoch == num_epochs:
            ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), checkpoint_dir / "last.pt")
            logger.info(f"  checkpoint saved: {ckpt_path}")

    logger.info("training complete.")
    pd.DataFrame(history).to_csv(output_dir / "loss_history.csv", index=False)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    return {"model": model, "processor": processor, "info": info,
            "history": pd.DataFrame(history)}


# ---------------------------------------------------------------------------
# Save as a repo-3-consumable pipeline folder
# ---------------------------------------------------------------------------
def save_finetuned_pipeline(result: Dict[str, Any], cfg: Dict[str, Any]) -> Path:
    """Extract embeddings with the fine-tuned model and write
    ``embeddings_root/<output_pipeline_name>/`` in the format
    ``eVTOL-Embedding-Evaluation/src/registry.py`` expects: one
    ``emb_layer{L}_{pooling}.npy`` per (layer, pooling) in ``analysis.layers``
    x ``analysis.pooling``, a row-aligned ``metadata.parquet``, and a
    ``manifest.json`` (registry.py's ``load_manifest``/``load_embeddings``
    only require the folder + npy + parquet to exist; the manifest content
    below documents provenance, matching the header comment in
    ``registry.py``).
    """
    fcfg = cfg["contrastive_finetune"]
    figures = load_pair_table(cfg)[["figure_id", "patent_id", "figure_type", "path"]]

    embed_cfg = {**cfg, "analysis": {**cfg["analysis"], "device": fcfg["device"]}}
    embed_result = embeddings.compute_embeddings(
        figures, embed_cfg, model=result["model"], processor=result["processor"],
        info=result["info"],
    )

    embeddings_root = Path(cfg["paths"]["embeddings_root"])
    pipeline_dir = embeddings_root / fcfg["output_pipeline_name"]
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    for (L, pool), arr in embed_result["arrays"].items():
        np.save(pipeline_dir / f"emb_layer{L}_{pool}.npy", arr)
    embed_result["metadata"].to_parquet(pipeline_dir / "metadata.parquet", index=False)

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True,
        ).strip()
    except Exception:
        git_commit = None

    manifest = {
        "pipeline_name": fcfg["output_pipeline_name"],
        "base_model": cfg["analysis"]["model"],
        "fine_tune_method": "supervised_contrastive_nt_xent",
        "class_col": fcfg["class_col"],
        "style_cols": fcfg["style_cols"],
        "unfreeze_last_n_blocks": fcfg["unfreeze_last_n_blocks"],
        "learning_rate": fcfg["learning_rate"],
        "batch_size": fcfg["batch_size"],
        "temperature": fcfg["temperature"],
        "num_epochs": fcfg["num_epochs"],
        "seed": fcfg.get("seed", 42),
        "model_info": asdict(embed_result["info"]),
        "git_commit": git_commit,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(pipeline_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[save_finetuned_pipeline] wrote {len(embed_result['arrays'])} arrays + "
          f"metadata + manifest.json to {pipeline_dir}")
    return pipeline_dir
