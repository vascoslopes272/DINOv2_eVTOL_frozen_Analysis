"""Image pre-processing of the selected figures (notebook 21_image_processing).

Input: ``selection/sets_union.csv`` (notebook 20) — every figure any set uses,
read from Stage 04's raw copies (``approved_copy_path``, 1639_LABELLED/joined/
approved_images). Those copies are the wizard crops as drawn: not rotated, not
square, any size.

Per figure, per configured size:
    1. RGB, transparency composited over white
    2. rotate by the labelled ``rotation_deg`` (clockwise, as the wizard shows
       it — same convention as Patent-Labelling-Tools/src/processor.py)
    3. resize with aspect kept so the long side = size (LANCZOS)
    4. pad to size x size (``pad_fill``: white, or the median border colour)

Output: ``<paths.pipeline_root>/processed/<size>/<batch>/<patent>/<file>.png``
and ``processed/<size>/manifest.csv`` (figure_uid -> path + original size).
Existing files are skipped unless ``force``; the manifest is always rewritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm


def processed_dir(cfg: Dict[str, Any], size: int) -> Path:
    d = Path(cfg["paths"]["pipeline_root"]) / "processed" / str(size)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "P":
        img = img.convert("RGBA")
    if img.mode in ("RGBA", "LA"):
        base = Image.new("RGB", img.size, (255, 255, 255))
        base.paste(img, mask=img.getchannel("A"))
        return base
    return img.convert("RGB")


def _border_median(img: Image.Image) -> tuple:
    a = np.asarray(img, dtype=np.uint8)
    ring = np.vstack([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(ring, axis=0))


def process_image(src: Path, rotation_deg: int, size: int, pad_fill: str = "white",
                  sharpen: float = 1.0) -> Image.Image:
    img = _to_rgb(Image.open(src))
    if rotation_deg:
        img = img.rotate(-rotation_deg, expand=True, fillcolor=(255, 255, 255))
    w, h = img.size
    scale = size / max(w, h)
    if sharpen != 1.0 and scale < 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpen)
    new = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                     Image.Resampling.LANCZOS)
    fill = (255, 255, 255) if pad_fill == "white" else _border_median(new)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(new, ((size - new.width) // 2, (size - new.height) // 2))
    return canvas


def process_all(figures: pd.DataFrame, cfg: Dict[str, Any], size: int,
                force: bool = False) -> pd.DataFrame:
    p = cfg["processing"]
    out_root = processed_dir(cfg, size)
    rows = []
    for r in tqdm(figures.itertuples(index=False), total=len(figures), desc=f"{size}px"):
        src = Path(r.approved_copy_path)
        dst = out_root / r.batch / r.patent_id / (Path(r.image_file).stem + ".png")
        with Image.open(src) as im:
            orig_w, orig_h = im.size
        if force or not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            process_image(src, int(r.rotation_deg), size, p.get("pad_fill", "white"),
                          float(p.get("sharpen", 1.0))).save(dst)
        rows.append({"figure_uid": r.figure_uid, "aircraft_uid": r.aircraft_uid,
                     "patent_id": r.patent_id, "path": str(dst), "src": str(src),
                     "orig_w": orig_w, "orig_h": orig_h,
                     "rotation_deg": int(r.rotation_deg), "size": size,
                     "upscaled": max(orig_w, orig_h) < size})
    man = pd.DataFrame(rows)
    man.to_csv(out_root / "manifest.csv", index=False)
    print(f"[processing] {len(man)} figures at {size}px -> {out_root}")
    return man


def load_manifest(cfg: Dict[str, Any], size: int) -> pd.DataFrame:
    return pd.read_csv(processed_dir(cfg, size) / "manifest.csv")
