"""Controls for the drawing thickening test (review flag S1, 2026-09-30).

The thick variants of ``preprocess_variants`` change three things at once: they
greyscale the drawing (210 of the 1 545 drawings carry colour), they darken
strokes that reach the model input as faint anti-aliased grey, and they widen
the strokes. Two controls, through the SAME pipeline and the SAME registers
extraction, separate the first two from the third:

grey       the original, transparency composited over white, to "L" and back to
           RGB. No spatial filter. thick1 / thick2 minus grey = the dilation with
           colour held equal.
contrast   grey, then a pointwise levels curve applied AT THE MODEL INPUT SCALE:
           the curve runs on the grey 518 px model input (``processed/518`` of
           the grey variant: rotated, resized, padded), because most source
           drawings are bi-level scans (0 / 255) whose strokes only turn faint
           grey when the resize to 518 px averages them; a curve on the source
           would leave those unchanged. Per image, with v the grey level (0 to
           255) of the 518 px input:

               ink   = pixels with v < 250            (the image's stroke histogram)
               t     = median of v over the ink pixels, clipped to [64, 224]
               v'    = 0     if v <= t
                       v     otherwise                 (unchanged)

           The darker half of each image's ink turns solid black; the lighter
           half (the anti-aliased fringe) and the paper keep their values. No
           pixel at 250 or above changes, so the set of non-white pixels (the
           footprint of the strokes) is exactly the grey input's: darker, not
           wider. An image with fewer than 50 ink pixels is left as grey. The
           prepared 518 px squares are saved as the variant's sources with
           rotation 0 and go through ``process_all`` at 518 unchanged (the
           resize of a 518 px square to 518 px is the identity), then the SAME
           extraction.

Writes (private, like the rest of the tree)::

    1639_LABELLED/2_embedding_extraction/preprocess_variants/{grey,contrast}/
        sources/..., sources_manifest.csv, processed/518/..., embeddings/dinov2-reg-large_518/
    1639_LABELLED/3_embedding_evaluation/setup_selection/
        montage_thickening_v2.png, s2_controls_ink.csv

The drawing ``grey`` here lives under the patent tree; the photo ``grey`` of
``preprocess_variants`` lives under EVTOLNEWS_DS. Nothing of the plain, thick1,
thick2 or photo-grey runs is touched. Evaluation: the evaluation pillar's
``src/preprocess_controls_eval.py``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import image_processing as IP
from . import preprocess_variants as PV
from .registers_experiment import REG_TAG, reg_cfg

SIZE = PV.SIZE
SEED = PV.SEED
CONTROLS = ["grey", "contrast"]
INK_LEVEL = 250          # v below this = ink (stroke or its fringe)
T_CLIP = (64, 224)       # the threshold stays inside this range
MIN_INK = 50             # fewer ink pixels -> image left as grey

# display names (the brief's display-rename layer: TR prints as "TP, Tilt Propulsor")
G1_NAME = {"TR": "TP, Tilt Propulsor", "CVT": "CVT, Combined vectored thrust", "TW": "TW, Tilt Wing",
           "TB": "TB, Tilt Body", "PTC": "PTC, Pitch-to-Cruise", "DS": "DS, Deflected Slipstream",
           "SLC": "SLC, Lift + Cruise", "SRW": "SRW, Stopped/Slowed Rotor Wing", "MR": "MR, Multirotor",
           "RC": "RC, Rotorcraft", "HB": "HB, Hoverbike", "PFV": "PFV, Personal Flying Vehicle"}


def variant_root(cfg: Dict[str, Any], variant: str) -> Path:
    d = Path(cfg["paths"]["pipeline_root"]) / "preprocess_variants" / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def variant_cfg(cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
    return reg_cfg({**cfg, "paths": {**cfg["paths"], "pipeline_root": variant_root(cfg, variant)}})


# ── the curve ───────────────────────────────────────────────────────────────
def contrast_curve(v: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
    """The levels curve on one grey 518 px input (uint8 array)."""
    ink = v[v < INK_LEVEL]
    if ink.size < MIN_INK:
        return v.copy(), {"threshold": -1, "n_ink": int(ink.size), "n_changed": 0}
    t = int(np.clip(np.median(ink), *T_CLIP))
    out = v.copy()
    hit = v <= t
    out[hit] = 0
    return out, {"threshold": t, "n_ink": int(ink.size), "n_changed": int((hit & (v > 0)).sum())}


def _prepare_one(args) -> Dict[str, Any]:
    src, dst, variant = args
    img = IP._to_rgb(Image.open(src))
    w, h = img.size
    a = np.asarray(img, dtype=np.int16)
    row = {"orig_w": w, "orig_h": h,
           "src_max_chroma": int(np.abs(a - a.mean(axis=2, keepdims=True)).max())}
    grey = np.asarray(img.convert("L"))
    if variant == "contrast":
        grey, info = contrast_curve(grey)
        row.update(info)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grey).convert("RGB").save(dst)
    return row


def run_make(cfg: Dict[str, Any], variant: str, workers: int = 12) -> pd.DataFrame:
    """grey: from the original sources (the plain run's table). contrast: from the grey
    variant's processed 518 px inputs (rotation already applied, so rotation 0 here)."""
    base = PV.plain_table(cfg, "thick1")            # the drawing table, plain run order
    root = variant_root(cfg, variant)
    if variant == "contrast":
        gm = pd.read_csv(variant_root(cfg, "grey") / "processed" / str(SIZE) / "manifest.csv")
        assert gm.figure_uid.tolist() == base.figure_uid.tolist(), "grey processed order differs"
        srcs = gm.path.tolist()
        base = base.assign(rotation_deg=0, grey_rotation_deg=gm.rotation_deg.to_numpy())
    else:
        srcs = base.approved_copy_path.tolist()
    dst = [str(root / "sources" / str(a) / (Path(f).stem + ".png"))
           for a, f in zip(base.aircraft_uid, base.image_file)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_prepare_one, zip(srcs, dst, [variant] * len(base)), chunksize=8))
    man = pd.concat([base.rename(columns={"approved_copy_path": "orig_path"}),
                     pd.DataFrame(rows)], axis=1)
    man["prepared_from"] = srcs
    man["approved_copy_path"] = dst
    man["image_file"] = [Path(d).name for d in dst]
    man.to_csv(root / "sources_manifest.csv", index=False)
    msg = f"[make {variant}] {len(man)} figures -> {root / 'sources'}"
    if variant == "contrast":
        msg += (f"; threshold median {man.threshold[man.threshold >= 0].median():.0f}, "
                f"{int((man.threshold < 0).sum())} left as grey")
    print(msg, flush=True)
    return man


def run_process(cfg: Dict[str, Any], variant: str) -> pd.DataFrame:
    man = pd.read_csv(variant_root(cfg, variant) / "sources_manifest.csv", keep_default_na=False)
    out = IP.process_all(man, variant_cfg(cfg, variant), SIZE)
    if variant == "contrast":   # the resize of a 518 square to 518 must be the identity
        for s, p in zip(man.approved_copy_path.iloc[:20], out.path.iloc[:20]):
            assert np.array_equal(np.asarray(Image.open(s).convert("RGB")),
                                  np.asarray(Image.open(p).convert("RGB"))), p
    return out


def run_extract(cfg: Dict[str, Any], variant: str) -> Path:
    # resolve transformers' lazy imports once before the per-GPU threads (import race)
    from transformers import AutoImageProcessor, AutoModel  # noqa: F401

    from . import embeddings as EM

    out = EM.run_extraction(variant_cfg(cfg, variant), SIZE)
    check(cfg, variant)
    return out


def check(cfg: Dict[str, Any], variant: str) -> None:
    tag = REG_TAG.format(size=SIZE)
    meta = pd.read_parquet(variant_root(cfg, variant) / "embeddings" / tag / "metadata.parquet")
    plain = pd.read_parquet(Path(cfg["paths"]["pipeline_root"]) / "embeddings" / tag / "metadata.parquet")
    assert meta.figure_uid.tolist() == plain.figure_uid.tolist(), f"{variant}: order differs from plain"
    print(f"[check {variant}] {len(meta)} figures, set and order match the plain registers run", flush=True)


# ── ink at the model input (how much each preparation darkens / widens) ─────
def _ink_one(path: str) -> tuple:
    v = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return float((255 - v).mean() / 255), float((v < INK_LEVEL).mean()), float((v < 128).mean())


def run_ink(cfg: Dict[str, Any], workers: int = 12) -> Path:
    """Per preparation, over every drawing at the 518 px input: ink mass (mean darkness),
    footprint (share of pixels below 250) and solid share (below 128)."""
    pat = Path(cfg["paths"]["pipeline_root"])
    mans = {"plain": pat / "processed" / str(SIZE) / "manifest.csv"}
    for v in CONTROLS + list(PV.THICK):
        mans[v] = variant_root(cfg, v) / "processed" / str(SIZE) / "manifest.csv"
    rows = []
    for v, m in mans.items():
        paths = pd.read_csv(m).path.tolist()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            r = np.array(list(pool.map(_ink_one, paths, chunksize=16)))
        rows.append({"preparation": v, "n_figures": len(paths), "ink_mass_mean": r[:, 0].mean(),
                     "footprint_mean": r[:, 1].mean(), "solid_share_mean": r[:, 2].mean()})
    df = pd.DataFrame(rows)
    dst = PV.eval_root(cfg) / "s2_controls_ink.csv"
    df.to_csv(dst, index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)
    return dst


# ── montage: plain | grey | contrast | thick1 | thick2 ─────────────────────
def run_montage(cfg: Dict[str, Any], n: int = 8, full: int = 300, zoom: int = 2,
                win: int = 128, seed: int = SEED) -> Path:
    """``preprocess_variants.run_montage`` with the two controls as extra columns: the same
    ``n`` figures (same seed, same picks), the same 128 px detail window per row."""
    pat = Path(cfg["paths"]["pipeline_root"])
    V = pat / "view_state_experiments"
    main = pd.read_csv(V / "exp3_main.csv", keep_default_na=False)
    g1 = pd.read_csv(V / "aircraft_common.csv", keep_default_na=False).set_index("aircraft_id")["g1_code"]
    main["g1"] = main.aircraft_id.map(g1)
    rng = np.random.default_rng(seed)
    classes = main.g1.value_counts().index[:n].tolist()
    pick = [main[main.g1 == c].iloc[rng.integers((main.g1 == c).sum())] for c in classes]
    m0 = pd.read_csv(pat / "processed" / str(SIZE) / "manifest.csv")
    paths = {"plain": dict(zip(m0.figure_uid, m0.path))}
    cols = ["plain", "grey", "contrast"] + list(PV.THICK)
    for v in cols[1:]:
        m = pd.read_csv(variant_root(cfg, v) / "processed" / str(SIZE) / "manifest.csv")
        paths[v] = dict(zip(m.figure_uid, m.path))
    rad = {v: dict(zip(*pd.read_csv(variant_root(cfg, v) / "sources_manifest.csv",
                                    keep_default_na=False)[["figure_uid", "radius"]].T.values))
           for v in PV.THICK}
    thr = pd.read_csv(variant_root(cfg, "contrast") / "sources_manifest.csv", keep_default_na=False)
    thr = dict(zip(thr.figure_uid, thr.threshold))
    k = len(cols)
    cz, head, lab = win * zoom, 22, 16
    W = k * (full + 6) + k * (cz + 6) + 8
    H = head + n * (max(full, cz) + lab + 8)
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for j, c in enumerate(cols):
        d.text((j * (full + 6) + 4, 4), f"{c} (518 px input, shown at {full} px)", fill="black")
        d.text((k * (full + 6) + j * (cz + 6) + 4, 4), f"{c} {win} px crop x{zoom}", fill="black")
    for i, r in enumerate(pick):
        y = head + i * (max(full, cz) + lab + 8)
        ims = [Image.open(paths[c][r.fig_id]).convert("RGB") for c in cols]
        x0, y0 = PV._ink_window(ims[0], win)
        for j, im in enumerate(ims):
            sheet.paste(im.resize((full, full), Image.Resampling.LANCZOS), (j * (full + 6), y))
            crop = im.crop((x0, y0, x0 + win, y0 + win)).resize((cz, cz), Image.Resampling.NEAREST)
            xc = k * (full + 6) + j * (cz + 6)
            sheet.paste(crop, (xc, y))
            d.rectangle([xc, y, xc + cz - 1, y + cz - 1], outline="#bbbbbb")
            fx, s = j * (full + 6), full / SIZE
            d.rectangle([fx + x0 * s, y + y0 * s, fx + (x0 + win) * s, y + (y0 + win) * s], outline="red")
        d.text((4, y + max(full, cz) + 2),
               f"{G1_NAME.get(r.g1, r.g1)}  {r.fig_id}  radius (orig px): thick1={rad['thick1'][r.fig_id]}, "
               f"thick2={rad['thick2'][r.fig_id]}; contrast threshold={thr[r.fig_id]}", fill="black")
    dst = PV.eval_root(cfg) / "montage_thickening_v2.png"
    sheet.save(dst)
    print(f"[montage] {len(pick)} figures ({', '.join(classes)}) -> {dst}", flush=True)
    return dst
