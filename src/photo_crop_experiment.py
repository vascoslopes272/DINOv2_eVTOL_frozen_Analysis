"""Crop ablation on the evtol.news photos (user request 2026-09-30).

Question: does tight-cropping each photo to the aircraft — removing broad
backgrounds (sky, mountains, cityscape) — change the architecture-class signal
of the DINOv2 embeddings? The patent figures have no such backgrounds, so a
big shift would mean the photo numbers partly measure the scenery.

Unit and labels are the report's: the ``main`` set (one hero image per
aircraft), the directory's 5 classes as ``topType``, the maker held out of
every kNN vote (``evtolnews_eval.company_key``).

detect    zero-shot OWLv2 (``google/owlv2-base-patch16-ensemble``) on the
          ORIGINAL source image, prompts = aircraft words; the best-scoring box
          over all prompts, padded by ``MARGIN`` on each side, clamped to the
          image. Images are pre-padded to a white square so the box
          coordinates cannot depend on the processor's own padding. Below
          ``SCORE_MIN`` no box is trusted and the full image is kept
          (status ``low_score``); a box under ``TINY_FRAC`` of the image is
          not trusted either (status ``tiny_box``).
process   the crops through the SAME path as the originals
          (``image_processing.process_all``: resize long side to 518, pad to a
          white 518x518 square)
extract   the SAME extraction (``embeddings.run_extraction``, frozen
          dinov2-large, layers 18/22/24, cls + mean_patch)
montage   original | crop pairs, stratified over the 5 classes, to eyeball the
          detector (wrong object, truncated aircraft)
evaluate  ``embedding_protocol.knn`` (kNN-5 maker-held-out balanced accuracy,
          bootstrap interval) and ``embedding_protocol.probe`` on the SAME
          aircraft, original vs cropped, plus a paired bootstrap of the
          difference (same resamples for both conditions)

Writes (everything private, like the rest of the tree)::

    2_embedding_extraction/crop_experiment/
        crops/<aircraft_uid>/<file>.png     the padded detector crops
        crop_manifest.csv                   one row per aircraft: box, score, status
        detect_summary.json
        processed/518/..., embeddings/dinov2-large_518/   the patent-layout outputs
    3_embedding_evaluation/crop_experiment/
        montage_orig_vs_crop.png
        knn_original_vs_cropped.csv         every layer x pooling x condition
        knn_paired_difference.csv           cropped - original, paired bootstrap
        RESULTS.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from . import evtolnews as E
from . import evtolnews_sets as S

DETECTOR = "google/owlv2-base-patch16-ensemble"
PROMPTS = ["an aircraft", "a drone", "a helicopter", "an airplane",
           "a flying vehicle", "a person riding a hoverbike"]
SCORE_MIN = 0.15      # best box below -> keep the full image (status low_score)
TINY_FRAC = 0.01      # box under 1 % of the image -> not trusted (status tiny_box)
MARGIN = 0.10         # padding around the box, fraction of the box's own size
SIZE = 518            # the report's representation is layer 24 CLS at 518 px
SEED = 42
CLASSES = ["VT", "LC", "WM", "ER", "HB"]

EVAL_PILLAR = Path(__file__).resolve().parent.parent.parent / \
    "eVTOL-Visual-Evaluation" / "embedding_evaluation" / "src"


def exp_root(cfg: Dict[str, Any]) -> Path:
    d = E.root(cfg) / "2_embedding_extraction" / "crop_experiment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def eval_root(cfg: Dict[str, Any]) -> Path:
    d = E.root(cfg) / "3_embedding_evaluation" / "crop_experiment"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exp_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The patent config with ``pipeline_root`` moved into the experiment tree,
    so notebooks 21-22's code writes processed/ and embeddings/ here."""
    return {**cfg, "paths": {**cfg["paths"], "pipeline_root": exp_root(cfg)}}


def load_main(cfg: Dict[str, Any], per_class: int = 0, seed: int = SEED) -> pd.DataFrame:
    """The ``main`` set (one hero image per aircraft), optionally a stratified
    sample of ``per_class`` aircraft per directory class."""
    sel = E.root(cfg) / "2_embedding_extraction" / "selection" / "sets" / "main.csv"
    m = pd.read_csv(sel, keep_default_na=False)
    if per_class:
        rng = np.random.default_rng(seed)
        parts = [g.iloc[np.sort(rng.choice(len(g), min(per_class, len(g)), replace=False))]
                 for _, g in m.groupby("topType")]
        m = pd.concat(parts, ignore_index=True)
    return m.sort_values("aircraft_uid").reset_index(drop=True)


# ── detect ──────────────────────────────────────────────────────────────────
def _square(img: Image.Image) -> Image.Image:
    side = max(img.size)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(img, (0, 0))
    return canvas


def _detect_shard(main: pd.DataFrame, root: Path, dev: str, batch: int = 8
                  ) -> List[Dict[str, Any]]:
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    proc = Owlv2Processor.from_pretrained(DETECTOR)
    model = Owlv2ForObjectDetection.from_pretrained(DETECTOR).to(dev).eval()
    post = getattr(proc, "post_process_grounded_object_detection",
                   getattr(proc, "post_process_object_detection", None))

    rows: List[Dict[str, Any]] = []
    for i in range(0, len(main), batch):
        part = main.iloc[i:i + batch]
        imgs = [S._load_rgb(p) for p in part.approved_copy_path]
        sq = [_square(im) for im in imgs]
        inputs = proc(text=[PROMPTS] * len(sq), images=sq, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs)
        sizes = torch.tensor([im.size[::-1] for im in sq])
        res = post(out, threshold=0.0, target_sizes=sizes)
        for r, det, im in zip(part.itertuples(index=False), res, imgs):
            w, h = im.size
            if len(det["scores"]):
                j = int(det["scores"].argmax())
                score = float(det["scores"][j])
                prompt = PROMPTS[int(det["labels"][j])]
                x0, y0, x1, y1 = [float(v) for v in det["boxes"][j]]
                x0, y0 = max(0.0, x0), max(0.0, y0)          # clamp to the un-padded image
                x1, y1 = min(float(w), x1), min(float(h), y1)
            else:
                score, prompt, x0, y0, x1, y1 = 0.0, "", 0.0, 0.0, float(w), float(h)
            frac = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / (w * h)
            status = "ok"
            if score < SCORE_MIN:
                status = "low_score"
            elif frac < TINY_FRAC or x1 <= x0 or y1 <= y0:
                status = "tiny_box"
            if status == "ok":
                mx, my = MARGIN * (x1 - x0), MARGIN * (y1 - y0)
                cx0, cy0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
                cx1, cy1 = int(min(w, x1 + mx)), int(min(h, y1 + my))
            else:                                            # detector not trusted: full image
                cx0, cy0, cx1, cy1 = 0, 0, w, h
            dst = root / "crops" / str(r.aircraft_uid) / (Path(r.image_file).stem + ".png")
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.crop((cx0, cy0, cx1, cy1)).save(dst)
            rows.append({"figure_uid": r.figure_uid, "aircraft_uid": r.aircraft_uid,
                         "patent_id": r.patent_id, "topType": r.topType, "company": r.company,
                         "src": r.approved_copy_path, "crop_path": str(dst),
                         "image_file": Path(dst).name, "rotation_deg": 0,
                         "score": round(score, 4), "prompt": prompt, "status": status,
                         "x0": cx0, "y0": cy0, "x1": cx1, "y1": cy1, "w": w, "h": h,
                         "box_frac": round(frac, 4),
                         "crop_frac": round((cx1 - cx0) * (cy1 - cy0) / (w * h), 4)})
        if i % (batch * 10) == 0:
            print(f"[detect {dev}] {i}/{len(main)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return rows


def run_detect(cfg: Dict[str, Any], per_class: int = 0) -> pd.DataFrame:
    """Best OWLv2 box per hero image -> crop_manifest.csv + crops/<uid>/<file>.png.
    One detector per GPU (``analysis.devices``), like compute_embeddings_multi_gpu."""
    from concurrent.futures import ThreadPoolExecutor

    # resolve transformers' lazy imports ONCE before the threads: two shards
    # importing concurrently race the lazy module loader (seen 2026-09-30)
    from transformers import Owlv2ForObjectDetection, Owlv2Processor  # noqa: F401

    main = load_main(cfg, per_class)
    root = exp_root(cfg)
    devices = cfg["analysis"].get("devices") or [cfg["evtolnews"]["filter"].get("device", "cuda:0")]
    shards = np.array_split(np.arange(len(main)), len(devices))
    print(f"[detect] {len(main)} images across {devices}: {[len(s) for s in shards]}", flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        parts = list(pool.map(lambda a: _detect_shard(main.iloc[a[1]].reset_index(drop=True),
                                                      root, devices[a[0]]), enumerate(shards)))
    man = pd.DataFrame([r for p in parts for r in p])
    man.to_csv(root / "crop_manifest.csv", index=False)
    summ = {"detector": DETECTOR, "prompts": PROMPTS, "score_min": SCORE_MIN,
            "margin": MARGIN, "tiny_frac": TINY_FRAC, "aircraft": int(len(man)),
            "status": man.status.value_counts().to_dict(),
            "by_class": man.groupby("topType").size().to_dict(),
            "median_crop_frac": float(man[man.status == "ok"].crop_frac.median()),
            "created": datetime.now().isoformat(timespec="seconds")}
    (root / "detect_summary.json").write_text(json.dumps(summ, indent=2))
    print("[detect]", json.dumps(summ))
    return man


# ── process + extract (notebooks 21-22's code on the crops) ─────────────────
def run_process(cfg: Dict[str, Any]) -> None:
    from . import image_processing as IP

    man = pd.read_csv(exp_root(cfg) / "crop_manifest.csv", keep_default_na=False)
    figures = man.assign(approved_copy_path=man.crop_path)
    IP.process_all(figures, exp_cfg(cfg), SIZE)


def run_extract(cfg: Dict[str, Any]) -> None:
    from . import embeddings as EM

    EM.run_extraction(exp_cfg(cfg), SIZE)


# ── montage (original | crop pairs, to judge the detector by eye) ───────────
def run_montage(cfg: Dict[str, Any], per_class: int = 8, cell: int = 240,
                seed: int = SEED) -> Path:
    man = pd.read_csv(exp_root(cfg) / "crop_manifest.csv", keep_default_na=False)
    rng = np.random.default_rng(seed)
    parts = [g.iloc[np.sort(rng.choice(len(g), min(per_class, len(g)), replace=False))]
             for _, g in man.groupby("topType")]
    rows = pd.concat(parts, ignore_index=True)
    cols, pair_w, cell_h = 4, 2 * cell + 6, cell + 30
    n_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (pair_w + 8), n_rows * cell_h), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows.itertuples(index=False)):
        x, y = (k % cols) * (pair_w + 8), (k // cols) * cell_h
        for dx, p in ((0, r.src), (cell + 6, r.crop_path)):
            try:
                im = S._load_rgb(p)
                im.thumbnail((cell - 4, cell - 24))
                sheet.paste(im, (x + dx + (cell - im.width) // 2, y + (cell - 24 - im.height) // 2))
            except Exception:  # noqa: BLE001
                d.text((x + dx + 4, y + 4), "unreadable", fill="red")
        d.line([(x + cell + 2, y), (x + cell + 2, y + cell - 24)], fill="#bbbbbb")
        d.text((x + 2, y + cell - 20),
               f"{r.topType}  {r.aircraft_uid[:34]}", fill="black")
        d.text((x + 2, y + cell - 9),
               f"score={r.score:.2f} {r.status} crop={r.crop_frac:.2f} of image",
               fill="red" if r.status != "ok" else "black")
    dst = eval_root(cfg) / "montage_orig_vs_crop.png"
    sheet.save(dst)
    print(f"[montage] {len(rows)} pairs -> {dst}")
    return dst


# ── evaluate (the evaluation pillar's protocol, on the same aircraft) ───────
def _protocol_modules():
    """Import the evaluation pillar's ``src`` under its own package name (both
    repos call their package ``src``, so a plain sys.path insert would clash)."""
    import importlib
    import importlib.util
    import sys

    name = "evtol_visual_eval_src"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, EVAL_PILLAR / "__init__.py", submodule_search_locations=[str(EVAL_PILLAR)])
        pkg = importlib.util.module_from_spec(spec)
        sys.modules[name] = pkg
        spec.loader.exec_module(pkg)
    return (importlib.import_module(f"{name}.embedding_protocol"),
            importlib.import_module(f"{name}.evtolnews_eval"))


def _matrices(emb_dir: Path, P) -> tuple[Dict[str, int], Dict[Any, np.ndarray]]:
    meta = pd.read_parquet(emb_dir / "metadata.parquet")
    arrays = {key: P.l2(np.load(emb_dir / f"emb_layer{key[0]}_{key[1]}.npy"))
              for key in P.MATRICES}
    return {u: i for i, u in enumerate(meta["figure_uid"])}, arrays


def run_evaluate(cfg: Dict[str, Any]) -> pd.DataFrame:
    from sklearn.metrics import balanced_accuracy_score

    P, EE = _protocol_modules()
    root, out = exp_root(cfg), eval_root(cfg)
    man = pd.read_csv(root / "crop_manifest.csv", keep_default_na=False)
    tag = cfg["extraction"]["tag"].format(size=SIZE)
    oidx, oarr = _matrices(E.root(cfg) / "2_embedding_extraction" / "embeddings" / tag, P)
    cidx, carr = _matrices(root / "embeddings" / tag, P)

    y = man.topType.to_numpy(dtype=object)
    mk = man.company.map(EE.company_key)
    makers = np.where(mk == "", "solo: " + man.aircraft_uid, mk).astype(object)
    orows = [oidx[u] for u in man.figure_uid]
    crows = [cidx[u] for u in man.figure_uid]

    rows, preds = [], {}
    for key in P.MATRICES:
        for cond, X in (("original", oarr[key][orows]), ("cropped", carr[key][crows])):
            kn = P.knn(X, y, makers)
            preds[(key, cond)] = kn.pop("pred")
            rec = kn.pop("recall")
            pr = P.probe(X, y, makers) if key == (24, "cls") else {}
            rows.append({"matrix": P.mname(key), "layer": key[0], "pooling": key[1],
                         "condition": cond, **{f"knn_{k}": v for k, v in kn.items()},
                         "probe_bal_acc": pr.get("bal_acc", np.nan),
                         "probe_macro_f1": pr.get("macro_f1", np.nan),
                         **{f"recall_{c}": rec.get(c, np.nan) for c in CLASSES}})
            print(f"[evaluate] {P.mname(key)} {cond}: knn {kn['bal_acc']:.3f} "
                  f"[{kn['ci_lo']:.3f}, {kn['ci_hi']:.3f}]", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "knn_original_vs_cropped.csv", index=False)

    # paired bootstrap of the difference: the same resamples score both conditions
    rng = np.random.default_rng(P.SEED)
    drows = []
    for key in P.MATRICES:
        pa, pb = preds[(key, "original")], preds[(key, "cropped")]
        diffs = []
        while len(diffs) < P.N_BOOT:
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i])) == len(set(y)):
                diffs.append(balanced_accuracy_score(y[i], pb[i])
                             - balanced_accuracy_score(y[i], pa[i]))
        diffs = np.asarray(diffs)
        drows.append({"matrix": P.mname(key), "layer": key[0], "pooling": key[1],
                      "diff_cropped_minus_original":
                          float(balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa)),
                      "ci_lo": float(np.quantile(diffs, 0.025)),
                      "ci_hi": float(np.quantile(diffs, 0.975)),
                      "agree_frac": float((pa == pb).mean())})
    diff = pd.DataFrame(drows)
    diff.to_csv(out / "knn_paired_difference.csv", index=False)
    print(diff.to_string(index=False))
    return res
