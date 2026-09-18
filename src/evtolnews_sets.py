"""evtol.news photo set: image filter, figure sets, processing, extraction.

Follows ``src/evtolnews.py`` (crawl). The steps mirror notebooks 20-22, but the
selection cannot come from human labels, so a filter decides which downloaded
images show the page's aircraft as a whole:

filter   (``1_filter/``)
    hard rules   download failed, smaller than ``min_side_px``, the same file
                 (sha1) already used by an earlier page, or stored in ANOTHER
                 aircraft's media folder (the page shows a sibling/predecessor)
    SigLIP       zero-shot score against prompt groups: whole aircraft (any
                 style) vs part close-up / interior / people / graphics / other;
                 plus a style guess (photo / render / drawing)
    page check   cosine of each image to the page's other images (a low value
                 on a multi-image page = possibly a different aircraft)
    manual       ``1_filter/manual_decisions.csv`` (image_uid, keep, reason)
                 overrides everything; written after looking at the contact
                 sheets in ``1_filter/sheets/``
sets     (``2_embedding_extraction/selection/``, same files as notebook 20)
    main     the page's first kept image (its hero image)
    all      every kept image
    built    first kept image of an aircraft the directory lists as built
             (prototype / production / demonstrator)
    concept  first kept image of a concept design
    (the SigLIP photo/render guess proved unreliable and is kept as a score only)
process  notebook 21's ``process_all`` (no rotation) -> processed/<size>/
extract  notebook 22's ``run_extraction`` -> embeddings/<tag>/, plus the same
         model on the patents' processed figures -> patent_reference/<tag>/
         (a copy for the patent<->photo matching; the patent tree is not touched)
attention CLS-token attention maps (src/attention_maps.py) of both main sets at 518 px
         -> attention/<tag>/{photo,patent}_cls.npy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from . import evtolnews as E

PROMPTS: Dict[str, List[str]] = {
    "aircraft": [
        "a photo of an electric aircraft flying in the sky",
        "a photo of an aircraft standing on the ground",
        "a 3D rendering of a futuristic flying vehicle",
        "a computer rendering of an air taxi flying over a city",
        "a line drawing of an aircraft",
        "a photo of a passenger drone",
        "a photo of a small helicopter",
        "a photo of a person riding a hoverbike",
        "a photo of a flying car",
        "a scale model of an aircraft",
        "an aircraft inside a hangar or at an air show",
        "a three-view drawing of an airplane",
    ],
    "part": [
        "a close-up photo of a propeller or rotor blade",
        "a close-up of an electric motor",
        "a mechanical component or engine on a workbench",
        "a battery pack",
        "a close-up of an aircraft wing or landing gear",
    ],
    "interior": [
        "the interior of a cockpit with seats and screens",
        "the inside of a passenger cabin",
    ],
    "people": [
        "a portrait photo of a person",
        "a group of people posing for a photo",
        "people on stage at a conference",
        "a team photo of people standing in front of an aircraft",
    ],
    "graphic": [
        "a company logo",
        "a logo with text on a plain background",
        "a title card with large text",
        "a brand name written on a dark background",
        "a chart, graph or table",
        "a map",
        "a screenshot of a document with text",
        "an infographic with icons and text",
        "a labelled technical diagram with arrows and text annotations",
        "a specification sheet with text boxes around a picture of an aircraft",
        "a slide with hexagonal icons and marketing text",
        "a comic or cartoon illustration",
    ],
    "other": [
        "a building or a landing pad without aircraft",
        "a car driving on a road",
        "a boat on the water",
        "a city skyline",
        "a self-driving ground shuttle pod on a road",
        "a ship at sea",
        "a piece of furniture such as a seat",
    ],
}
STYLE_PROMPTS = {
    "photo": "a real photograph taken with a camera",
    "render": "a computer-generated 3D rendering",
    "drawing": "a black and white line drawing or sketch",
}
BUILT_WORDS = ("prototype", "production", "demonstrator", "proof of concept", "mockup",
               "flight test", "test aircraft")


def status_group(*values: str) -> str:
    """built / concept / unknown, from the directory's own status wording."""
    t = " ".join(str(v) for v in values).lower()
    if any(w in t for w in BUILT_WORDS):
        return "built"
    if "concept" in t:
        return "concept"
    return "unknown"


KEEP_P = 0.80     # p(aircraft) at or above -> keep
DROP_P = 0.35     # below -> drop; in between -> review
PAGE_SIM_FLAG = 0.55


def filter_dir(cfg: Dict[str, Any]) -> Path:
    d = E.root(cfg) / "1_filter"
    (d / "sheets").mkdir(parents=True, exist_ok=True)
    return d


def _load_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    img.seek(0)
    if img.mode in ("P", "LA", "RGBA"):
        img = img.convert("RGBA")
        base = Image.new("RGB", img.size, (255, 255, 255))
        base.paste(img, mask=img.getchannel("A"))
        return base
    return img.convert("RGB")


def _prompt_hash() -> str:
    import hashlib
    return hashlib.sha1(json.dumps([PROMPTS, STYLE_PROMPTS], sort_keys=True).encode()).hexdigest()[:12]


def siglip_run(paths: List[str], cfg: Dict[str, Any], feats: np.ndarray | None = None
               ) -> tuple[pd.DataFrame, np.ndarray]:
    """Zero-shot scores for every prompt group (+ a style guess, kept as a score only).

    ``feats`` are cached image features; only the text side is recomputed when the
    prompts change.
    """
    import open_clip
    import torch

    f = cfg["evtolnews"]["filter"]
    dev = f.get("device", "cuda:0")
    model, pre = open_clip.create_model_from_pretrained(f["model"], device=dev)
    tok = open_clip.get_tokenizer(f["model"])
    model.eval()
    groups = [g for g, ps in PROMPTS.items() for _ in ps]
    texts = [p for ps in PROMPTS.values() for p in ps]
    with torch.no_grad():
        T = model.encode_text(tok(texts).to(dev))
        T = T / T.norm(dim=-1, keepdim=True)
        S = model.encode_text(tok(list(STYLE_PROMPTS.values())).to(dev))
        S = S / S.norm(dim=-1, keepdim=True)
        if feats is None:
            chunks = []
            for i in range(0, len(paths), 32):
                batch = torch.stack([pre(_load_rgb(p)) for p in paths[i:i + 32]]).to(dev)
                x = model.encode_image(batch)
                chunks.append((x / x.norm(dim=-1, keepdim=True)).float().cpu())
                if i % 640 == 0:
                    print(f"[filter] siglip {i}/{len(paths)}", flush=True)
            X = torch.cat(chunks)
        else:
            X = torch.from_numpy(feats).float()
        scale = model.logit_scale.exp().float().cpu()
        p = (scale * X @ T.float().cpu().T).softmax(-1).numpy()
        ps = (scale * X @ S.float().cpu().T).softmax(-1).numpy()
    out = pd.DataFrame({f"p_{g}": p[:, [i for i, gg in enumerate(groups) if gg == g]].sum(1)
                        for g in PROMPTS})
    out["top_prompt"] = [texts[i] for i in p.argmax(1)]
    for j, k in enumerate(STYLE_PROMPTS):
        out[f"style_{k}"] = ps[:, j]
    out["style"] = [list(STYLE_PROMPTS)[i] for i in ps.argmax(1)]
    del model
    torch.cuda.empty_cache()
    return out, X.numpy()


def contact_sheet(rows: pd.DataFrame, dst: Path, cols: int = 8, cell: int = 220) -> None:
    from PIL import ImageDraw

    n = len(rows)
    h = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, h * (cell + 34)), "white")
    d = ImageDraw.Draw(sheet)
    for k, r in enumerate(rows.itertuples(index=False)):
        x, y = (k % cols) * cell, (k // cols) * (cell + 34)
        try:
            im = _load_rgb(r.local_path)
            im.thumbnail((cell - 6, cell - 6))
            sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
        except Exception:  # noqa: BLE001
            d.text((x + 5, y + 5), "unreadable", fill="red")
        d.text((x + 3, y + cell), f"{r.sheet_label}"[:34], fill="black")
        d.text((x + 3, y + cell + 12), f"p={r.p_aircraft:.2f} {r.style[:5]} {r.flag}"[:34],
               fill="red" if r.p_aircraft < KEEP_P else "black")
    sheet.save(dst, quality=85)


def run_filter(cfg: Dict[str, Any]) -> pd.DataFrame:
    src = E.source_dir(cfg)
    fd = filter_dir(cfg)
    im = pd.read_csv(src / "images.csv", keep_default_na=False)
    ac = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    min_side = int(cfg["evtolnews"]["filter"].get("min_side_px", 200))
    im["width"] = pd.to_numeric(im["width"], errors="coerce")
    im["height"] = pd.to_numeric(im["height"], errors="coerce")
    im["in_own_folder"] = im["in_own_folder"].astype(str) == "True"

    # hard rules, first match wins
    im["hard_rule"] = ""
    im.loc[im.sha1 == "", "hard_rule"] = "download_failed"
    small = im[["width", "height"]].min(axis=1) < min_side
    im.loc[(im.hard_rule == "") & small, "hard_rule"] = "too_small"
    im["_ord"] = np.arange(len(im))  # images.csv is in page order
    # the same file on several pages: keep it on the page whose own folder holds it
    im["_own_rank"] = (~im.in_own_folder).astype(int)
    first = (im[im.sha1 != ""].sort_values(["_own_rank", "_ord"])
             .drop_duplicates("sha1").set_index("sha1")["image_uid"])
    dup = (im.sha1 != "") & (im.sha1.map(first) != im.image_uid)
    im.loc[(im.hard_rule == "") & dup, "hard_rule"] = "same_file_on_other_page"
    im["dup_of"] = np.where(dup, im.sha1.map(first), "")
    # media folders: a page's own folder is the one most of its images sit in; an image
    # from a folder that IS another page's own folder shows that other aircraft
    main_folder = (im[im.media_folder != ""].groupby("slug_key").media_folder
                   .agg(lambda s: s.value_counts().idxmax()))
    im["page_folder"] = im.slug_key.map(main_folder).fillna("")
    owner = {f: k for k, f in main_folder.items()}
    im["folder_owner"] = im.media_folder.map(owner).fillna("")
    other_page = (im.media_folder != "") & (im.folder_owner != "") & (im.folder_owner != im.slug_key)
    im.loc[(im.hard_rule == "") & other_page, "hard_rule"] = "other_aircraft_page"
    im["other_folder"] = (im.media_folder != im.page_folder) & (im.hard_rule == "")
    im = im.drop(columns=["_own_rank"])

    # SigLIP on every readable image (cached)
    cache, fcache = fd / "siglip_scores.parquet", fd / "siglip_features.npy"
    state = fd / "siglip_cache.json"
    readable = im[im.sha1 != ""]
    st = json.loads(state.read_text()) if state.exists() else {}
    feats, sc = None, None
    if fcache.exists() and st.get("image_uids") == readable.image_uid.tolist():
        feats = np.load(fcache)          # images unchanged: reuse the features
        if cache.exists() and st.get("prompts") == _prompt_hash():
            sc = pd.read_parquet(cache)  # prompts unchanged too: reuse the scores
    if sc is None:
        sc, feats = siglip_run(readable.local_path.tolist(), cfg, feats)
        sc.insert(0, "image_uid", readable.image_uid.to_numpy())
        sc.to_parquet(cache, index=False)
        np.save(fcache, feats)
        state.write_text(json.dumps({"image_uids": readable.image_uid.tolist(),
                                     "prompts": _prompt_hash()}))
    im = im.merge(sc, on="image_uid", how="left")
    im["p_aircraft"] = im["p_aircraft"].fillna(0.0)
    im["style"] = im["style"].fillna("")

    # page check: similarity to the page's other aircraft-like images
    pos = {u: i for i, u in enumerate(sc.image_uid)}
    im["page_sim"] = np.nan
    for key, g in im[(im.hard_rule == "") & (im.p_aircraft >= DROP_P)].groupby("slug_key"):
        if len(g) < 2:
            continue
        F = feats[[pos[u] for u in g.image_uid]]
        C = F @ F.T
        np.fill_diagonal(C, np.nan)
        im.loc[g.index, "page_sim"] = np.nanmedian(C, axis=1)

    # decision
    im["auto"] = np.select(
        [im.hard_rule != "", im.p_aircraft >= KEEP_P, im.p_aircraft < DROP_P],
        ["drop", "keep", "drop"], "review")
    low_sim = (im.auto == "keep") & (im.page_sim < PAGE_SIM_FLAG)
    im.loc[low_sim | ((im.auto == "keep") & im.other_folder), "auto"] = "review"
    im["flag"] = np.where(im.hard_rule != "", im.hard_rule.str[:14],
                          np.where(im.other_folder, "otherfolder", np.where(low_sim, "lowsim", "")))
    man_f = fd / "manual_decisions.csv"
    im["manual"] = ""
    im["manual_reason"] = ""
    if man_f.exists():
        man = pd.read_csv(man_f, keep_default_na=False).drop_duplicates("image_uid", keep="last")
        m = im.image_uid.map(man.set_index("image_uid")["keep"].astype(str))
        im["manual"] = m.fillna("")
        im["manual_reason"] = im.image_uid.map(man.set_index("image_uid")["reason"]).fillna("")
    im["keep"] = np.where(im.manual.isin(["True", "1", "keep"]), True,
                          np.where(im.manual.isin(["False", "0", "drop"]), False, im.auto == "keep"))
    im["decided_by"] = np.where(im.manual != "", "manual", "auto:" + im.auto)
    im = im.drop(columns=["_ord"])
    im.to_csv(fd / "image_decisions.csv", index=False)

    # contact sheets: every review image, then samples of keeps and drops
    im["sheet_label"] = im.image_uid.str[-30:]
    rev = im[(im.auto == "review") & (im.manual == "")].sort_values("p_aircraft")
    for old in (fd / "sheets").glob("*.jpg"):
        old.unlink()
    for i in range(0, len(rev), 48):
        contact_sheet(rev.iloc[i:i + 48], fd / "sheets" / f"review_{i // 48:03d}.jpg")
    for name, part in [("keep_lowest", im[(im.auto == "keep")].nsmallest(96, "p_aircraft")),
                       ("drop_highest", im[(im.auto == "drop") & (im.hard_rule == "")].nlargest(96, "p_aircraft")),
                       ("keep_sample", im[im.auto == "keep"].sample(min(96, (im.auto == "keep").sum()), random_state=0)),
                       ("other_folder", im[im.hard_rule == "other_aircraft_folder"].head(96))]:
        for j in range(0, len(part), 48):
            contact_sheet(part.iloc[j:j + 48], fd / "sheets" / f"{name}_{j // 48}.jpg")
    summ = {"images": int(len(im)), "auto": im.auto.value_counts().to_dict(),
            "hard_rule": im.hard_rule.value_counts().to_dict(),
            "kept": int(im.keep.sum()), "manual": int((im.manual != "").sum()),
            "aircraft_with_kept_image": int(im[im.keep].slug_key.nunique()),
            "thresholds": {"keep_p": KEEP_P, "drop_p": DROP_P, "page_sim_flag": PAGE_SIM_FLAG}}
    (fd / "filter_summary.json").write_text(json.dumps(summ, indent=2))
    print("[filter]", json.dumps(summ))
    return im


# ── sets (notebook 20's files) ──────────────────────────────────────────────
def run_sets(cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    from . import figure_selection as FS

    pcfg = E.pipeline_cfg(cfg)
    src = E.source_dir(cfg)
    im = pd.read_csv(filter_dir(cfg) / "image_decisions.csv", keep_default_na=False)
    ac = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    im["keep"] = im["keep"].astype(str) == "True"
    c = im.merge(ac[["slug_key", "title", "model", "company", "location", "status", "list_status",
                     "evtol_class_main", "url"]], on="slug_key", how="left")
    c = c.assign(
        aircraft_uid=c.slug_key, patent_id=c.slug_key, figure_uid=c.slug_key + "/" +
        c.local_path.map(lambda p: Path(p).name if p else ""),
        image_file=c.local_path.map(lambda p: Path(p).name if p else ""),
        approved_copy_path=c.local_path, rotation_deg=0, topType=c.evtol_class_main,
        fig_order=pd.to_numeric(c.img_pos), eligible=c.keep,
        excluded_by=np.where(c.keep, "", np.where(c.manual.isin(["False", "0", "drop"]), "manual",
                                                  np.where(c.hard_rule != "", c.hard_rule, "siglip"))))
    funnel = (c.assign(n=1).groupby("excluded_by", sort=False)
              .agg(figures=("n", "size"), aircraft=("aircraft_uid", "nunique")).reset_index())
    c["status_group"] = [status_group(a, b) for a, b in zip(c.status, c.list_status)]
    elig = c[c.eligible].sort_values(["aircraft_uid", "fig_order"])
    sets = {
        "main": elig.drop_duplicates("aircraft_uid"),
        "all": elig,
        "built": elig[elig.status_group == "built"].drop_duplicates("aircraft_uid"),
        "concept": elig[elig.status_group == "concept"].drop_duplicates("aircraft_uid"),
    }
    sets = {k: v.assign(set=k)[["set"] + [x for x in v.columns if x != "set"]] for k, v in sets.items()}

    a = ac[ac.slug_key.isin(elig.aircraft_uid)].rename(columns={"slug_key": "aircraft_uid"}).copy()
    a["topType"] = a["evtol_class_main"]
    a["n_eligible_figures"] = a.aircraft_uid.map(elig.groupby("aircraft_uid").size())
    rng = np.random.default_rng(int(cfg["evtolnews"].get("selection_seed", 42)))
    a["split"] = ""
    for _, g in a.groupby("topType"):
        idx = rng.permutation(g.index.to_numpy())
        a.loc[idx, "split"] = np.array(["select", "report"])[np.arange(len(idx)) % 2]
    for col in a.columns:
        if a[col].dtype == object:
            a[col] = a[col].map(lambda v: None if pd.isna(v) else str(v))
    cov = FS.coverage(a, sets)
    d = FS.save(c, funnel, a, sets, cov, pcfg)
    s = json.loads((d / "selection_summary.json").read_text())
    s["producer"] = "eVTOL-Embedding-Extraction/scripts/evtolnews_pipeline.py sets"
    s["source"] = "https://evtol.news/aircraft (VFS World eVTOL Aircraft Directory)"
    s["rules"] = {"filter": json.loads((filter_dir(cfg) / "filter_summary.json").read_text())}
    s["by_class"] = FS.coverage_by_type(cov, list(sets)).reset_index().to_dict("records")
    (d / "selection_summary.json").write_text(json.dumps(s, indent=2, default=str))
    print(pd.DataFrame(s["by_class"]).to_string(index=False))
    return sets


# ── processing + extraction (notebooks 21-22's code) ────────────────────────
def run_process(cfg: Dict[str, Any]) -> None:
    from . import image_processing as IP

    pcfg = E.pipeline_cfg(cfg)
    union = pd.read_csv(Path(pcfg["paths"]["pipeline_root"]) / "selection" / "sets_union.csv",
                        keep_default_na=False)
    for size in cfg["processing"]["sizes"]:
        IP.process_all(union, pcfg, int(size))


def run_extract(cfg: Dict[str, Any]) -> None:
    from . import embeddings as EM

    pcfg = E.pipeline_cfg(cfg)
    root = Path(pcfg["paths"]["pipeline_root"])
    for size in cfg["extraction"]["sizes"]:
        EM.run_extraction(pcfg, int(size))
        # same model on the patents' processed figures (read-only copy for matching)
        man = Path(cfg["paths"]["pipeline_root"]) / "processed" / str(size) / "manifest.csv"
        tag = cfg["extraction"]["tag"].format(size=size)
        out = root / "patent_reference" / tag
        figs = pd.read_csv(man)
        if (out / "metadata.parquet").exists() and \
                set(pd.read_parquet(out / "metadata.parquet").figure_uid) == set(figs.figure_uid):
            print(f"[extract] patent_reference/{tag}: up to date")
            continue
        run_cfg = {**pcfg, "analysis": {**pcfg["analysis"], "input_size": int(size)}}
        devices = run_cfg["analysis"].get("devices") or [run_cfg["analysis"]["device"]]
        res = EM.compute_embeddings_multi_gpu(figs, run_cfg, devices)
        EM.save_embeddings(res, run_cfg, out)
        (out / "manifest.json").write_text(json.dumps(
            {"pipeline": tag, "input_size": int(size), "n_figures": int(len(figs)),
             "processed_manifest": str(man),
             "note": "copy of the patent figures embedded for the evtol.news matching; "
                     "the patent pipeline_root is not written"}, indent=2))


# ── attention maps (CLS token, main sets) ───────────────────────────────────
def run_attention(cfg: Dict[str, Any], size: int = 518) -> None:
    """CLS attention of the photos' and the patents' main images, for the report figures."""
    from . import attention_maps as AM

    pcfg = E.pipeline_cfg(cfg)
    root = Path(pcfg["paths"]["pipeline_root"])
    tag = cfg["extraction"]["tag"].format(size=size)
    out = root / "attention" / tag
    for name, proc_root in (("photo", root), ("patent", Path(cfg["paths"]["pipeline_root"]))):
        man = pd.read_csv(proc_root / "processed" / str(size) / "manifest.csv")
        main = pd.read_csv(proc_root / "selection" / "sets" / "main.csv", keep_default_na=False)
        man = man[man.figure_uid.isin(set(main.figure_uid))]
        AM.run(pcfg, man, out, name, size=size)
