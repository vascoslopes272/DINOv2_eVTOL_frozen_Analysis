"""Preparation variants for the setup selection (author request 2026-09-30).

Question: does the preparation of the image, before the fixed resize-and-pad,
change the architecture signal of the chapter's model
(``facebook/dinov2-with-registers-large`` at 518 px)? Two candidate sets,
fixed in advance:

    drawings  plain | thick1 | thick2   (patent figures, every whole-aircraft figure of
                                         the processed-518 manifest)
    photos    colour | grey             (evtol.news hero image per aircraft, the main set)

thick<k>  on the ORIGINAL source image, before rotation and resize: greyscale,
          then a minimum filter (square window, dark strokes grow) of radius
          r = round(k / s) original pixels, s = 518 / max(width, height), so each
          side of a stroke moves out by about k px at the 518 px model input (the
          stroke gets about 2k px wider). r = 0 after rounding -> r = 1 when
          k / s >= 0.5, otherwise the image is left as it is. The radius used is
          recorded per image. Back to RGB.
grey      the original, transparency composited over white, to "L" and back to RGB.

The copies then go through the SAME ``image_processing.process_all`` (518) and the
SAME ``embeddings.run_extraction`` as the plain / colour run, with the registers
model (``registers_experiment.reg_cfg``). Pattern: ``photo_crop_experiment``
(variant table -> ``pipeline_root`` override -> process_all -> run_extraction).

Writes (private, like the rest of the tree)::

    <tree>/2_embedding_extraction/preprocess_variants/<variant>/
        sources/<aircraft_uid>/<stem>.png     the prepared originals
        sources_manifest.csv                  one row per figure (radius used, ...)
        processed/518/..., embeddings/dinov2-reg-large_518/
    1639_LABELLED/3_embedding_evaluation/setup_selection/montage_thickening.png

The evaluation is in the evaluation pillar (``src/preprocess_variants_eval.py``).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import evtolnews as E
from . import image_processing as IP
from .registers_experiment import REG_TAG, reg_cfg

SIZE = 518
SEED = 42
THICK = {"thick1": 1, "thick2": 2}          # variant -> stroke growth k at the model input (px)
PHOTO = {"grey"}
VARIANTS = list(THICK) + sorted(PHOTO)


# ── where things live ───────────────────────────────────────────────────────
def _tree(cfg: Dict[str, Any], variant: str) -> Path:
    """The data tree of the variant's source: the patent tree or the photo tree."""
    return E.root(cfg) if variant in PHOTO else Path(cfg["paths"]["pipeline_root"]).parent


def variant_root(cfg: Dict[str, Any], variant: str) -> Path:
    d = _tree(cfg, variant) / "2_embedding_extraction" / "preprocess_variants" / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_root(cfg: Dict[str, Any]) -> Path:
    d = Path(cfg["paths"]["pipeline_root"]).parent / "3_embedding_evaluation" / "setup_selection"
    d.mkdir(parents=True, exist_ok=True)
    return d


def variant_cfg(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """Registers model + tag, ``pipeline_root`` moved into the variant folder."""
    return reg_cfg({**cfg, "paths": {**cfg["paths"], "pipeline_root": variant_root(cfg, variant)}})


def plain_table(cfg: Dict[str, Any], variant: str) -> pd.DataFrame:
    """The figures of the plain (colour) run the variant copies, in the plain run's row order.

    drawings: the patent processed-518 manifest (every whole-aircraft figure);
    photos:   the main set (one hero image per aircraft).
    """
    if variant in PHOTO:
        sel = E.root(cfg) / "2_embedding_extraction" / "selection" / "sets" / "main.csv"
        m = pd.read_csv(sel, keep_default_na=False)
        return m[["figure_uid", "aircraft_uid", "patent_id", "approved_copy_path", "image_file",
                  "rotation_deg"]].reset_index(drop=True)
    man = pd.read_csv(Path(cfg["paths"]["pipeline_root"]) / "processed" / str(SIZE) / "manifest.csv")
    return pd.DataFrame({"figure_uid": man.figure_uid, "aircraft_uid": man.aircraft_uid,
                         "patent_id": man.patent_id, "approved_copy_path": man.src,
                         "image_file": man.src.map(lambda p: Path(p).name),
                         "rotation_deg": man.rotation_deg})


# ── make the prepared copies ────────────────────────────────────────────────
def thick_radius(w: int, h: int, k: int, size: int = SIZE) -> int:
    """Radius in original pixels so a stroke grows by about ``k`` px at ``size``."""
    ks = k / (size / max(w, h))
    r = int(np.floor(ks + 0.5))
    if r == 0 and ks >= 0.5:
        r = 1
    return r


def _prepare_one(args) -> Dict[str, Any]:
    from scipy.ndimage import minimum_filter

    src, dst, variant = args
    img = IP._to_rgb(Image.open(src))
    w, h = img.size
    a = np.asarray(img, dtype=np.int16)
    chroma = int(np.abs(a - a.mean(axis=2, keepdims=True)).max())  # 0 = the source was already grey
    grey = img.convert("L")
    row = {"orig_w": w, "orig_h": h, "src_max_chroma": chroma}
    if variant in THICK:
        k = THICK[variant]
        r = thick_radius(w, h, k)
        if r:
            grey = Image.fromarray(minimum_filter(np.asarray(grey), size=2 * r + 1, mode="nearest"))
        row.update({"k_model_px": k, "scale": SIZE / max(w, h), "k_over_s": k / (SIZE / max(w, h)),
                    "radius": r})
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    grey.convert("RGB").save(dst)
    return row


def run_make(cfg: Dict[str, Any], variant: str, workers: int = 12) -> pd.DataFrame:
    base = plain_table(cfg, variant)
    root = variant_root(cfg, variant)
    dst = [str(root / "sources" / str(a) / (Path(f).stem + ".png"))
           for a, f in zip(base.aircraft_uid, base.image_file)]
    jobs = list(zip(base.approved_copy_path, dst, [variant] * len(base)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_prepare_one, jobs, chunksize=8))
    man = pd.concat([base.rename(columns={"approved_copy_path": "orig_path"}),
                     pd.DataFrame(rows)], axis=1)
    man["approved_copy_path"] = dst
    man["image_file"] = [Path(d).name for d in dst]
    man.to_csv(root / "sources_manifest.csv", index=False)
    msg = f"[make {variant}] {len(man)} figures -> {root / 'sources'}"
    if variant in THICK:
        msg += f"; radius counts {man.radius.value_counts().sort_index().to_dict()}"
    print(msg, flush=True)
    return man


# ── process + extract (notebooks 21-22's code on the copies) ────────────────
def run_process(cfg: Dict[str, Any], variant: str) -> pd.DataFrame:
    man = pd.read_csv(variant_root(cfg, variant) / "sources_manifest.csv", keep_default_na=False)
    return IP.process_all(man, variant_cfg(cfg, variant), SIZE)


def run_extract(cfg: Dict[str, Any], variant: str) -> Path:
    from . import embeddings as EM

    out = EM.run_extraction(variant_cfg(cfg, variant), SIZE)
    check(cfg, variant)
    return out


def check(cfg: Dict[str, Any], variant: str) -> None:
    """The variant's figure_uid set and order must match the plain registers run."""
    tag = REG_TAG.format(size=SIZE)
    meta = pd.read_parquet(variant_root(cfg, variant) / "embeddings" / tag / "metadata.parquet")
    plain_root = (E.root(cfg) / "2_embedding_extraction" if variant in PHOTO
                  else Path(cfg["paths"]["pipeline_root"]))
    plain = pd.read_parquet(plain_root / "embeddings" / tag / "metadata.parquet").figure_uid.tolist()
    want = plain_table(cfg, variant).figure_uid.tolist()
    got = meta.figure_uid.tolist()
    assert got == want, f"{variant}: figure order differs from the plain table"
    if variant in PHOTO:   # the plain photo run embeds every kept image; the main set is a subset
        missing = set(got) - set(plain)
        assert not missing, f"{variant}: {len(missing)} figures missing from the plain run"
    else:
        assert got == plain, f"{variant}: figure_uid set/order differs from the plain registers run"
    print(f"[check {variant}] {len(got)} figures, set and order match the plain registers run", flush=True)


# ── montage: plain | thick1 | thick2 as the model receives them ─────────────
def _ink_window(img: Image.Image, win: int = 128, step: int = 8) -> tuple:
    """Top-left corner of the ``win`` square with the most dark pixels (strokes)."""
    a = (np.asarray(img.convert("L")) < 128).astype(np.int32)
    ii = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, xy = -1, (0, 0)
    for y in range(0, a.shape[0] - win + 1, step):
        for x in range(0, a.shape[1] - win + 1, step):
            s = ii[y + win, x + win] - ii[y, x + win] - ii[y + win, x] + ii[y, x]
            if s > best:
                best, xy = s, (x, y)
    return xy


def run_montage(cfg: Dict[str, Any], n: int = 8, full: int = 300, zoom: int = 2,
                win: int = 128, seed: int = SEED) -> Path:
    """``n`` main figures of different G1 classes; per row the three 518 px inputs
    (shown at ``full`` px) and the same ``win`` px crop of each, magnified ``zoom`` x."""
    pat = Path(cfg["paths"]["pipeline_root"])
    V = pat / "view_state_experiments"
    main = pd.read_csv(V / "exp3_main.csv", keep_default_na=False)
    g1 = pd.read_csv(V / "aircraft_common.csv", keep_default_na=False).set_index("aircraft_id")["g1_code"]
    main["g1"] = main.aircraft_id.map(g1)
    rng = np.random.default_rng(seed)
    classes = main.g1.value_counts().index[:n].tolist()          # the n most frequent classes
    pick = [main[main.g1 == c].iloc[rng.integers((main.g1 == c).sum())] for c in classes]
    m0 = pd.read_csv(pat / "processed" / str(SIZE) / "manifest.csv")
    paths = {"plain": dict(zip(m0.figure_uid, m0.path))}
    rad = {}
    for v in THICK:
        m = pd.read_csv(variant_root(cfg, v) / "processed" / str(SIZE) / "manifest.csv")
        paths[v] = dict(zip(m.figure_uid, m.path))
        s = pd.read_csv(variant_root(cfg, v) / "sources_manifest.csv", keep_default_na=False)
        rad[v] = dict(zip(s.figure_uid, s.radius))
    cols = ["plain"] + list(THICK)
    cz, head, lab = win * zoom, 22, 16
    W = 3 * (full + 6) + 3 * (cz + 6) + 8
    H = head + n * (max(full, cz) + lab + 8)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((j * (full + 6) + 4, 4), f"{c} (518 px input, shown at {full} px)", fill="black")
        d.text((3 * (full + 6) + j * (cz + 6) + 4, 4), f"{c} {win} px crop x{zoom}", fill="black")
    for i, r in enumerate(pick):
        y = head + i * (max(full, cz) + lab + 8)
        ims = [Image.open(paths[c][r.fig_id]).convert("RGB") for c in cols]
        x0, y0 = _ink_window(ims[0], win)
        for j, im in enumerate(ims):
            sheet.paste(im.resize((full, full), Image.Resampling.LANCZOS), (j * (full + 6), y))
            crop = im.crop((x0, y0, x0 + win, y0 + win)).resize((cz, cz), Image.Resampling.NEAREST)
            xc = 3 * (full + 6) + j * (cz + 6)
            sheet.paste(crop, (xc, y))
            d.rectangle([xc, y, xc + cz - 1, y + cz - 1], outline="#bbbbbb")
            fx = j * (full + 6)
            s = full / SIZE
            d.rectangle([fx + x0 * s, y + y0 * s, fx + (x0 + win) * s, y + (y0 + win) * s], outline="red")
        d.text((4, y + max(full, cz) + 2),
               f"{r.g1}  {r.fig_id}  radius (orig px): thick1={rad['thick1'][r.fig_id]}, "
               f"thick2={rad['thick2'][r.fig_id]}", fill="black")
    dst = eval_root(cfg) / "montage_thickening.png"
    sheet.save(dst)
    print(f"[montage] {len(pick)} figures ({', '.join(classes)}) -> {dst}", flush=True)
    return dst
