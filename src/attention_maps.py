"""CLS-token attention maps of the frozen DINOv2 (notebook 24; works on any processed manifest).

The attention of block L is rebuilt from the block's own weights: the block's input
(``hidden_states[L-1]``) goes through ``norm1`` and the query / key projections, and the CLS
query is scored against every token (softmax over all tokens, then the mean over the heads).
This needs only the hidden states the extraction already returns. Asking the model for
``output_attentions`` instead would hold all 24 blocks' maps at once, about 23 GB for a batch
of 8 images at 518 px.

DINOv2 without registers keeps a few high-norm tokens in background patches of its late
layers (Darcet et al., 2024, "Vision Transformers Need Registers"). They show as isolated
bright cells away from the aircraft and are a property of the model, not of the image.

Output: ``<out_dir>/<name>_cls.npy`` (n x layers x g x g, float16, g = size / 14) and
``<out_dir>/<name>_meta.csv`` (figure_uid, aircraft_uid, path; row order of the array).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from . import embeddings as EM


@torch.no_grad()
def cls_attention(paths: Sequence[str], cfg: Dict[str, Any], layers: Sequence[int], size: int,
                  device: str, batch_size: int = 8) -> np.ndarray:
    run_cfg = {**cfg, "analysis": {**cfg["analysis"], "input_size": int(size), "device": device}}
    model, processor, info = EM.load_model(run_cfg)
    g = size // 14
    out = np.zeros((len(paths), len(layers), g, g), dtype=np.float16)
    for s in tqdm(range(0, len(paths), batch_size), desc=f"attention {size}px", unit="batch"):
        imgs = [Image.open(p).convert("RGB") for p in paths[s:s + batch_size]]
        inputs = processor(images=imgs, return_tensors="pt").to(info.device)
        hs = model(**inputs, output_hidden_states=True).hidden_states
        for j, L in enumerate(layers):
            blk = model.encoder.layer[L - 1]
            att = blk.attention.attention
            x = blk.norm1(hs[L - 1])
            B, T, _ = x.shape
            nh = att.num_attention_heads
            dh = att.attention_head_size
            q = att.query(x[:, :1]).view(B, 1, nh, dh).transpose(1, 2)        # B, h, 1, d
            k = att.key(x).view(B, T, nh, dh).transpose(1, 2)                 # B, h, T, d
            a = torch.softmax((q @ k.transpose(-1, -2)) / dh ** 0.5, dim=-1)  # B, h, 1, T
            a = a.mean(dim=1)[:, 0, info.n_prefix:]                            # B, patches
            out[s:s + len(imgs), j] = a.reshape(B, g, g).float().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out


def run(cfg: Dict[str, Any], manifest: pd.DataFrame, out_dir: Path, name: str, size: int = 518,
        layers: Sequence[int] = (18, 22, 24), device: str | None = None, force: bool = False) -> Path:
    """Attention maps for every row of ``manifest`` (needs figure_uid, aircraft_uid, path)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_f, arr_f = out_dir / f"{name}_meta.csv", out_dir / f"{name}_cls.npy"
    man = manifest[["figure_uid", "aircraft_uid", "path"]].reset_index(drop=True)
    if not force and meta_f.exists() and arr_f.exists() and \
            pd.read_csv(meta_f).figure_uid.tolist() == man.figure_uid.tolist():
        print(f"[attention] {name}: up to date ({len(man)} images)")
        return arr_f
    dev = device or (cfg["analysis"].get("devices") or [cfg["analysis"]["device"]])[0]
    arr = cls_attention(man.path.tolist(), cfg, list(layers), size, dev)
    np.save(arr_f, arr)
    man.to_csv(meta_f, index=False)
    (out_dir / "attention_info.json").write_text(json.dumps(
        {"model": cfg["analysis"]["model"], "size": size, "layers": list(layers), "grid": size // 14,
         "map": "CLS query against every patch key, softmax over all tokens, mean over heads",
         "note": "DINOv2 without registers: isolated bright background cells are model artifacts"},
        indent=2))
    print(f"[attention] {name}: {len(man)} images -> {arr_f}")
    return arr_f
