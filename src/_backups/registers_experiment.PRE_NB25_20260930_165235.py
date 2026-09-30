"""dinov2-with-registers check (chapter plan N2; works beside the base extraction, touches nothing of it).

DINOv2 without registers parks high-norm artifact tokens in background patches of its late
layers (Darcet et al., 2024): in this pipeline the L22/L24 CLS attention collapses onto one
padding token (28-30 %), so only L18 attention maps were honest. The -with-registers variant
adds 4 register tokens that are supposed to absorb those artifacts. This experiment answers
two questions on the SAME images the base runs used:

    1. attention  - do the late-layer maps become presentable (artifact share drops)?
    2. embeddings - do the taxonomy scores move (checked by the evaluation pillar's
                    scripts/registers_check.py on the tags written here)?

Extraction reuses ``embeddings.run_extraction`` unchanged (it is already register-aware via
``ModelInfo.n_prefix``); only the model name and the tag differ. Outputs:

    1639_LABELLED/2_embedding_extraction/embeddings/dinov2-reg-large_518/     (patents)
    EVTOLNEWS_DS/2_embedding_extraction/embeddings/dinov2-reg-large_518/      (photos)
    EVTOLNEWS_DS/2_embedding_extraction/attention/dinov2-reg-large_518/       (both sources)
    EVTOLNEWS_DS/2_embedding_extraction/attention/registers_share.csv         (base vs reg)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from . import attention_maps as AM
from . import embeddings as EM
from . import evtolnews as E

REG_MODEL = "facebook/dinov2-with-registers-large"
REG_TAG = "dinov2-reg-large_{size}"


def reg_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {**cfg, "analysis": {**cfg["analysis"], "model": REG_MODEL},
            "extraction": {**cfg["extraction"], "tag": REG_TAG}}


def run_extract(cfg: Dict[str, Any], size: int = 518) -> None:
    """Registers embeddings for the patent figures and the photo sets, same manifests as base."""
    EM.run_extraction(reg_cfg(cfg), size)                      # patents
    EM.run_extraction(reg_cfg(E.pipeline_cfg(cfg)), size)      # photos


def run_attention(cfg: Dict[str, Any], size: int = 518) -> None:
    """Registers CLS attention on exactly the images of the base attention run (main sets)."""
    pcfg = E.pipeline_cfg(cfg)
    root = Path(pcfg["paths"]["pipeline_root"])
    rcfg = reg_cfg(pcfg)
    out = root / "attention" / REG_TAG.format(size=size)
    for name, proc_root in (("photo", root), ("patent", Path(cfg["paths"]["pipeline_root"]))):
        man = pd.read_csv(proc_root / "processed" / str(size) / "manifest.csv")
        main = pd.read_csv(proc_root / "selection" / "sets" / "main.csv", keep_default_na=False)
        man = man[man.figure_uid.isin(set(main.figure_uid))]
        AM.run(rcfg, man, out, name, size=size)


def run_share(cfg: Dict[str, Any], size: int = 518) -> Path:
    """Artifact-share table, base vs registers, per source and layer.

    ``cls_attention`` keeps the post-softmax attention of the patch tokens only, so per image
    the max patch cell is the collapse candidate and ``1 - grid.sum()`` is what the CLS spends
    on the prefix (CLS + register) tokens instead of the picture.
    """
    root = Path(E.pipeline_cfg(cfg)["paths"]["pipeline_root"]) / "attention"
    layers = (18, 22, 24)
    rows = []
    for model, tag_t in (("base", cfg["extraction"]["tag"]), ("registers", REG_TAG)):
        d = root / tag_t.format(size=size)
        for name in ("photo", "patent"):
            arr = np.load(d / f"{name}_cls.npy").astype(np.float32)   # n, layers, g, g
            flat = arr.reshape(len(arr), len(layers), -1)
            for j, L in enumerate(layers):
                mx, tot = flat[:, j].max(axis=1), flat[:, j].sum(axis=1)
                rows.append({"model": model, "source": name, "layer": L,
                             "n_images": len(arr),
                             "max_patch_share_median": float(np.median(mx)),
                             "max_patch_share_p90": float(np.quantile(mx, 0.9)),
                             "prefix_share_median": float(np.median(1 - tot))})
    out = root / "registers_share.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return out
