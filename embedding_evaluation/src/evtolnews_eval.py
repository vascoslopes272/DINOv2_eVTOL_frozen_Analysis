"""Evaluation of the evtol.news photo embeddings (user request 2026-09-17).

The photo set is built by eVTOL-Embedding-Extraction (``scripts/evtolnews_pipeline.py``)
under ``evtolnews.root``; its ``2_embedding_extraction/`` has the patent layout, so
the label-free metrics of notebook 30 run on it unchanged. On top of that:

1. photos on their own — the directory's 5 classes (VT, LC, WM, ER, HB) as the
   label: kNN (the aircraft's own images, or its whole maker, held out), a
   grouped linear probe, silhouette and k-means ARI/NMI; the directory's status
   (built prototype vs concept design) as a confound. The same numbers on the patent
   figures, their codebook topType mapped to its parent class, are the reference.
2. same-aircraft matching — patent aircraft with a real name that has directory
   pages: where do those pages rank among all pages for the patent figure
   (patent -> photo), and where does the patent aircraft rank among all patent
   aircraft for the page's photos (photo -> patent)?

Writes ``<evtolnews.root>/3_embedding_evaluation/``::

    metrics/<tag>/<set>/...        notebook 30's A-D tables (label-free)
    patent_links.csv               patent aircraft <-> directory pages (reviewed)
    class_metrics.csv              question 1, every tag x set x layer x pooling
    matching.csv  matching_ranks.csv   question 2
    parent_check.csv               codebook topType x directory class on linked aircraft
    EVTOLNEWS_EVALUATION.md        the report (numbers only, no photos)
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, balanced_accuracy_score, f1_score,
                             normalized_mutual_info_score, silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold

from . import embedding_metrics as em

CLASSES = ["VT", "LC", "WM", "ER", "HB"]
CLASS_NAMES = {"VT": "Vectored Thrust", "LC": "Lift + Cruise", "WM": "Wingless (Multicopter)",
               "ER": "Electric Rotorcraft", "HB": "Hover Bikes / Personal Flying Devices"}
TOPTYPE_PARENT = {"TW": "VT", "TR": "VT", "TB": "VT", "PTC": "VT", "DS": "VT", "CVT": "VT",
                  "SLC": "LC", "SRW": "LC", "MR": "WM", "RC": "ER", "HB": "HB", "PFV": "HB"}
COMPANY_STOP = {"inc", "llc", "ltd", "gmbh", "co", "corp", "corporation", "company", "aviation",
                "aerospace", "aero", "technologies", "technology", "group", "the", "sas", "ag", "sa",
                "limited", "air", "mobility", "systems", "innovations", "industries", "motors",
                "motor", "aircraft", "international", "holdings", "labs", "lab", "design", "of",
                "and", "de", "srl", "bv", "pty", "plc", "kg", "se", "individual", "inventor",
                "unknown", "independent", "formerly", "now", "vehicles", "vehicle", "urban",
                "helicopters", "helicopter", "electric", "flying", "engineering", "university"}
MAX_COMPANY_PAGES = 12   # a family name that is just the maker: too many pages = no link
GENERIC_NAME = {"evtol", "vtol", "concept", "unnamed", "prototype", "demonstrator"}
KS = (1, 5, 10)
# matching galleries: every kept image, or without the line drawings — some directory
# pages reproduce the patent drawing itself, which makes patent->photo trivially easy
IMAGE_VARIANTS = ("all images", "no line drawings")


def root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["evtolnews"]["root"])


def out_dir(cfg: Dict[str, Any]) -> Path:
    d = root(cfg) / "3_embedding_evaluation"
    d.mkdir(parents=True, exist_ok=True)
    return d


def photo_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Notebook 30's config pointed at the photo tree."""
    return {**cfg, "paths": {**cfg["paths"],
                             "pipeline_root": root(cfg) / "2_embedding_extraction",
                             "metrics_dir": out_dir(cfg) / "metrics"}}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def company_tokens(s: str) -> set:
    return {t for t in _norm(s).split() if t not in COMPANY_STOP and len(t) > 1}


def company_key(s: str) -> str:
    toks = [t for t in _norm(s).split() if t not in COMPANY_STOP and len(t) > 1]
    return " ".join(toks[:2]) or _norm(s)


# ── inputs ──────────────────────────────────────────────────────────────────
def load_photo(cfg: Dict[str, Any], tag: str) -> Dict[str, Any]:
    pc = photo_cfg(cfg)
    res = em.load_embeddings(Path(pc["paths"]["pipeline_root"]) / "embeddings" / tag)
    return {"result": res, "sets": em.load_sets(pc)}


def load_patent(cfg: Dict[str, Any], tag: str) -> Dict[str, Any]:
    res = em.load_embeddings(root(cfg) / "2_embedding_extraction" / "patent_reference" / tag)
    return {"result": res, "sets": em.load_sets(cfg)}


def run_label_free(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Notebook 30's sections A-D on the photos, and on the patent copy (patent_reference/)."""
    en = cfg["evtolnews"]
    pc = photo_cfg(cfg)
    pc = {**pc, "metrics": {**pc["metrics"], "embeddings": en["embeddings"], "sets": []}}
    photos = em.run_all(pc, em.load_sets(pc)).assign(source="photo")
    rc = {**cfg, "paths": {**cfg["paths"], "metrics_dir": out_dir(cfg) / "metrics_patent_reference"}}
    rows = []
    for tag in en["embeddings"]:
        pa = load_patent(cfg, tag)
        for s in en["sets"]:
            f = _frame(pa["result"], pa["sets"][s])
            d = em.evaluate_set(f["sub"], s, tag, rc)
            rows.append(d.assign(embedding=tag, set=s, n_figures=len(f["df"]), source="patent"))
    return pd.concat([photos, *rows], ignore_index=True)


# ── 1. class structure ──────────────────────────────────────────────────────
def _neighbours(X: np.ndarray, groups: np.ndarray, kmax: int) -> np.ndarray:
    """Top-``kmax`` cosine neighbours of each row, never from its own group (-1 = none)."""
    S = X @ X.T
    S[groups[:, None] == groups[None, :]] = -np.inf
    nn = np.argpartition(-S, kmax, axis=1)[:, :kmax]
    order = np.take_along_axis(-S, nn, axis=1).argsort(axis=1)
    nn = np.take_along_axis(nn, order, axis=1)
    nn[~np.isfinite(np.take_along_axis(S, nn, axis=1))] = -1
    return nn


def _vote(nn: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Majority label of the first ``k`` neighbours; ties go to the nearer one."""
    pred = []
    for row in nn[:, :k]:
        votes: Dict[str, float] = {}
        for rank, j in enumerate(row[row >= 0]):
            votes[y[j]] = votes.get(y[j], 0) + 1 + 1e-3 * (k - rank)
        pred.append(max(votes, key=votes.get) if votes else "")
    return np.array(pred)


def class_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, makers: np.ndarray,
                 seed: int = 42, n_perm: int = 20) -> Dict[str, float]:
    X = em._l2(X.astype(np.float64))
    out: Dict[str, float] = {"n": len(y), "n_classes": len(set(y))}
    kmax = max(KS)
    nn_g, nn_m = _neighbours(X, groups, kmax), _neighbours(X, makers, kmax)
    for k in KS:
        p = _vote(nn_g, y, k)
        out[f"knn{k}_bal_acc"] = balanced_accuracy_score(y, p)
        out[f"knn{k}_macro_f1"] = f1_score(y, p, average="macro")
        out[f"knn{k}_bal_acc_maker_out"] = balanced_accuracy_score(y, _vote(nn_m, y, k))
    rng = np.random.default_rng(seed)
    perm = [balanced_accuracy_score(yp, _vote(nn_g, yp, 5))
            for yp in (rng.permutation(y) for _ in range(n_perm))]
    out["knn5_perm_mean"] = float(np.mean(perm))
    out["knn5_perm_p95"] = float(np.quantile(perm, 0.95))
    # linear probe, folds grouped by maker (no maker in train and test)
    n_splits = min(5, int(pd.Series(y).value_counts().min()))
    if n_splits >= 2:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        pred = np.empty(len(y), dtype=object)
        for tr, te in cv.split(X, y, makers):
            clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        out["probe_bal_acc_maker_out"] = balanced_accuracy_score(y, pred)
        out["probe_macro_f1_maker_out"] = f1_score(y, pred, average="macro")
    out["silhouette_cos"] = float(silhouette_score(X, y, metric="cosine"))
    km = KMeans(n_clusters=len(set(y)), n_init=10, random_state=seed).fit_predict(X)
    out["kmeans_ari"] = adjusted_rand_score(y, km)
    out["kmeans_nmi"] = normalized_mutual_info_score(y, km)
    out["chance_bal_acc"] = 1.0 / len(set(y))
    return out


def _frame(result: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    df = df[df["figure_uid"].isin(set(result["metadata"]["figure_uid"]))]
    return {"sub": em.subset(result, df["figure_uid"]), "df": df.reset_index(drop=True)}


def run_class_metrics(cfg: Dict[str, Any], tags: List[str], sets: List[str]) -> pd.DataFrame:
    rows = []
    for tag in tags:
        ph = load_photo(cfg, tag)
        pa = load_patent(cfg, tag)
        for s in sets:
            for source, bundle in (("photo", ph), ("patent", pa)):
                if s not in bundle["sets"]:
                    continue
                df = bundle["sets"][s].copy()
                if source == "patent":
                    df["cls"] = df["topType"].map(TOPTYPE_PARENT)
                    df["maker"] = df["company"].fillna("")
                    df.loc[df["maker"].isin(["Individual Inventor", "Unknown / Independent", ""]),
                           "maker"] = "solo:" + df["aircraft_uid"]
                else:
                    df["cls"] = df["topType"]
                    df["maker"] = df["company"].fillna("").map(company_key)
                    df.loc[df["maker"] == "", "maker"] = "solo:" + df["aircraft_uid"]
                df = df[df["cls"].isin(CLASSES)]
                f = _frame(bundle["result"], df)
                d = f["df"]
                for key, X in f["sub"]["arrays"].items():
                    r = class_scores(X, d["cls"].to_numpy(), d["aircraft_uid"].to_numpy(),
                                     d["maker"].to_numpy(), int(cfg.get("seed", 42)))
                    row = {"embedding": tag, "set": s, "source": source, "label": "class",
                           "layer": key[0], "pooling": key[1],
                           "n_aircraft": d["aircraft_uid"].nunique(), **r}
                    rows.append(row)
                    if source == "photo" and d["status_group"].nunique() > 1:
                        st = d["status_group"].to_numpy()
                        rs = class_scores(X, st, d["aircraft_uid"].to_numpy(), d["maker"].to_numpy(),
                                          int(cfg.get("seed", 42)), n_perm=5)
                        rows.append({**row, "label": "status", **rs})
                print(f"[class] {tag} {s} {source}: {len(d)} figures", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir(cfg) / "class_metrics.csv", index=False)
    return out


# ── 2. patent <-> directory links ───────────────────────────────────────────
def build_links(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Candidate directory pages for each named patent aircraft in the patent sets."""
    lab = Path(cfg["evtolnews"]["labelled_root"])
    at = pd.read_csv(lab / "0_labelling/outputs/tables/aircraft_table.csv", keep_default_na=False)
    nd = pd.read_csv(lab / "0_labelling/inputs/review_decisions/NAME_DECISIONS.csv", keep_default_na=False)
    pages = pd.read_csv(root(cfg) / "0_source/aircraft.csv", keep_default_na=False)
    union = pd.read_csv(Path(cfg["paths"]["pipeline_root"]) / "selection/sets_union.csv",
                        keep_default_na=False)
    at = at[at["aircraft_id"].isin(set(union["aircraft_uid"])) & (at["name_is_real"].astype(str) == "True")]
    url_of = {}
    for r in nd.itertuples(index=False):
        s = re.findall(r"evtol\.news/([A-Za-z0-9\-]+)", r.source_note)
        if s:
            url_of[r.patent_id] = [x.lower() for x in s]
    known = set(pages["slug_key"])
    mf = out_dir(cfg) / "patent_links_manual.csv"
    manual: Dict[str, List[str]] = {}
    if mf.exists():
        for m in pd.read_csv(mf, keep_default_na=False).itertuples(index=False):
            manual[m.key] = [x for x in str(m.slug_keys).split("|") if x in known]
    pages["_title"] = (pages["title"] + " " + pages["model"] + " " + pages["list_name"]).map(_norm)
    pages["_co"] = (pages["company"] + " " + pages["list_name"]).map(company_tokens)
    rows = []
    for r in at.itertuples(index=False):
        family = re.sub(r"\s+v\d+$", "", r.aircraft_name).strip()
        fam = _norm(family)
        co = company_tokens(r.company) | company_tokens(r.assignee)
        fam_toks = company_tokens(family) - GENERIC_NAME
        fam_is_company = bool(fam_toks) and fam_toks <= co
        by_name = pages[pages["_title"].map(lambda t: bool(fam) and re.search(r"\b" + re.escape(fam) + r"\b", t) is not None)]
        by_both = by_name[by_name["_co"].map(lambda c: bool(c & co))]
        by_co = pages[pages["_co"].map(lambda c: bool(c & co))]
        direct = [d for d in url_of.get(r.patent_id, []) if d in known]
        if direct:          # the user's own ruling in NAME_DECISIONS wins over any guess
            cand, how = pages.iloc[0:0], "ruling"
        elif len(by_both):
            cand, how = by_both, "name+company"
        elif len(by_name) and len(by_name) <= 3:
            cand, how = by_name, "name only"
        elif fam_is_company and 0 < len(by_co) <= MAX_COMPANY_PAGES:
            cand, how = by_co, "company (name = maker)"
        else:
            cand, how = pages.iloc[0:0], "none"
        slugs = manual.get(r.aircraft_id, manual.get(family,
                           list(dict.fromkeys(direct + cand["slug_key"].tolist()))))
        how = "manual" if (r.aircraft_id in manual or family in manual) else how
        rows.append({"aircraft_uid": r.aircraft_id, "patent_id": r.patent_id, "aircraft_name": r.aircraft_name,
                     "family": family, "company": r.company, "topType": r.topType,
                     "match": ("url+" if direct and how != "ruling" else "") + how, "n_pages": len(slugs),
                     "slug_keys": "|".join(slugs),
                     "page_titles": " | ".join(pages.set_index("slug_key").reindex(slugs)["title"].fillna("?")),
                     "page_classes": "|".join(pages.set_index("slug_key").reindex(slugs)["evtol_class_main"].fillna("?")),
                     "use": "", "note": ""})
    out = pd.DataFrame(rows)
    f = out_dir(cfg) / "patent_links_auto.csv"
    out.to_csv(f, index=False)
    print(f"[links] {len(out)} named aircraft; with pages: {(out.n_pages > 0).sum()} -> {f}")
    return out


def load_links(cfg: Dict[str, Any]) -> pd.DataFrame:
    """The reviewed links: ``patent_links.csv`` rows with use == yes."""
    f = out_dir(cfg) / "patent_links.csv"
    lk = pd.read_csv(f, keep_default_na=False)
    lk = lk[lk["use"].str.lower().isin(["yes", "y", "true", "1"])]
    lk["slugs"] = lk["slug_keys"].str.split("|")
    return lk


def parent_check(cfg: Dict[str, Any]) -> pd.DataFrame:
    lk = load_links(cfg)
    pages = pd.read_csv(root(cfg) / "0_source/aircraft.csv", keep_default_na=False).set_index("slug_key")
    rows = []
    for r in lk.itertuples(index=False):
        cls = pages.reindex(r.slugs)["evtol_class_main"].dropna()
        page_cls = cls.mode().iloc[0] if len(cls) else ""
        rows.append({"aircraft_uid": r.aircraft_uid, "family": r.family, "topType": r.topType,
                     "parent_of_topType": TOPTYPE_PARENT.get(r.topType, ""),
                     "directory_class": page_cls, "directory_classes": "|".join(sorted(set(cls))),
                     "agrees": TOPTYPE_PARENT.get(r.topType, "") in set(cls)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir(cfg) / "parent_check.csv", index=False)
    return out


# ── 3. matching ─────────────────────────────────────────────────────────────
def _rank_of(scores: pd.Series, positives: set) -> int:
    pos = scores[scores.index.isin(positives)]
    if not len(pos):
        return -1
    return int((scores[~scores.index.isin(positives)] > pos.max()).sum()) + 1


def run_matching(cfg: Dict[str, Any], tags: List[str], query_set: str = "main",
                 gallery_set: str = "all") -> pd.DataFrame:
    lk = load_links(cfg)
    rows, ranks = [], []
    for tag, images in [(t, v) for t in tags for v in IMAGE_VARIANTS]:
        ph, pa = load_photo(cfg, tag), load_patent(cfg, tag)
        keep = (lambda df: df[df["style"] != "drawing"]) if images == "no line drawings" else (lambda df: df)
        g_ph = _frame(ph["result"], keep(ph["sets"][gallery_set]))
        q_ph = _frame(ph["result"], keep(ph["sets"][query_set]))
        g_pa = _frame(pa["result"], pa["sets"][gallery_set])
        q_pa = _frame(pa["result"], pa["sets"][query_set])
        fam_of = lk.set_index("aircraft_uid")["family"].to_dict()
        slugs_of = lk.set_index("aircraft_uid")["slugs"].to_dict()
        fam_slugs: Dict[str, set] = {}
        for a, s in slugs_of.items():
            fam_slugs.setdefault(fam_of[a], set()).update(s)
        fam_aircraft = lk.groupby("family")["aircraft_uid"].apply(set).to_dict()
        page_set = set(g_ph["df"]["aircraft_uid"])
        for key in g_ph["sub"]["arrays"]:
            GP = em._l2(g_ph["sub"]["arrays"][key])
            QA = em._l2(q_pa["sub"]["arrays"][key])
            GA = em._l2(g_pa["sub"]["arrays"][key])
            QP = em._l2(q_ph["sub"]["arrays"][key])
            # patent -> photo: one query per linked patent aircraft (its main figure)
            qd = q_pa["df"]
            r1 = []
            for i, a in enumerate(qd["aircraft_uid"]):
                if a not in fam_of:
                    continue
                pos = fam_slugs[fam_of[a]] & page_set
                if not pos:
                    continue
                sc = pd.Series(GP @ QA[i]).groupby(g_ph["df"]["aircraft_uid"].to_numpy()).max()
                rk = _rank_of(sc, pos)
                r1.append(rk)
                ranks.append({"embedding": tag, "images": images, "layer": key[0], "pooling": key[1],
                              "direction": "patent->photo", "query": a, "family": fam_of[a],
                              "rank": rk, "n_positive": len(pos), "gallery": len(sc),
                              "top1": sc.idxmax()})
            # photo -> patent: one query per linked directory page (its main photo)
            pd_ = q_ph["df"]
            ga_ids = g_pa["df"]["aircraft_uid"].to_numpy()
            r2 = []
            for fam, slugs in fam_slugs.items():
                pos = fam_aircraft[fam] & set(ga_ids)
                if not pos:
                    continue
                for j in np.flatnonzero(pd_["aircraft_uid"].isin(slugs).to_numpy()):
                    sc = pd.Series(GA @ QP[j]).groupby(ga_ids).max()
                    rk = _rank_of(sc, pos)
                    r2.append(rk)
                    ranks.append({"embedding": tag, "images": images, "layer": key[0], "pooling": key[1],
                                  "direction": "photo->patent", "query": pd_["aircraft_uid"].iloc[j],
                                  "family": fam, "rank": rk, "n_positive": len(pos),
                                  "gallery": len(sc), "top1": sc.idxmax()})
            for direction, rr, gal in (("patent->photo", r1, len(page_set)),
                                       ("photo->patent", r2, len(set(ga_ids)))):
                rr = np.array([x for x in rr if x > 0])
                if not len(rr):
                    continue
                sub = [x for x in ranks if x["embedding"] == tag and x["images"] == images and x["layer"] == key[0]
                       and x["pooling"] == key[1] and x["direction"] == direction]
                exp = {k: float(np.mean([min(1.0, 1 - np.prod([(x["gallery"] - x["n_positive"] - m) /
                                                               (x["gallery"] - m) for m in range(k)]))
                                         for x in sub])) for k in (1, 5, 10, 50)}
                rows.append({"embedding": tag, "images": images, "layer": key[0], "pooling": key[1],
                             "direction": direction, "queries": len(rr), "gallery": gal,
                             **{f"R@{k}": float((rr <= k).mean()) for k in (1, 5, 10, 50)},
                             **{f"random_R@{k}": exp[k] for k in (1, 5, 10, 50)},
                             "median_rank": float(np.median(rr)), "mrr": float((1 / rr).mean())})
        print(f"[matching] {tag} / {images} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(out_dir(cfg) / "matching.csv", index=False)
    pd.DataFrame(ranks).to_csv(out_dir(cfg) / "matching_ranks.csv", index=False)
    return out


def labelled_frame(bundle: Dict[str, Any], source: str, set_name: str) -> Dict[str, Any]:
    """A set with its 5-class label (``cls``) and maker (``maker``), aligned to its embeddings.

    Photos carry the directory class; patents carry the codebook topType mapped to its parent.
    Aircraft of a lone inventor or an unknown maker are their own maker group.
    """
    df = bundle["sets"][set_name].copy()
    if source == "patent":
        df["cls"] = df["topType"].map(TOPTYPE_PARENT)
        mk = df["company"].fillna("")
        solo = mk.isin(["", "Individual Inventor", "Unknown / Independent"])
    else:
        df["cls"] = df["topType"]
        mk = df["company"].fillna("").map(company_key)
        solo = mk == ""
    df["maker"] = np.where(solo, "solo:" + df["aircraft_uid"], mk)
    df = df[df["cls"].isin(CLASSES)]
    return _frame(bundle["result"], df)


def confusion_tables(cfg: Dict[str, Any], tag: str = "dinov2-large_518", key=(24, "cls"),
                     set_name: str = "main") -> Dict[str, pd.DataFrame]:
    """kNN-5 (maker held out) confusion of the 5 classes, photos and patents; rows = true."""
    from sklearn.metrics import confusion_matrix

    out = {}
    for source, bundle in (("photo", load_photo(cfg, tag)), ("patent", load_patent(cfg, tag))):
        f = labelled_frame(bundle, source, set_name)
        d = f["df"]
        X = em._l2(f["sub"]["arrays"][key].astype(np.float64))
        y = d["cls"].to_numpy()
        pred = _vote(_neighbours(X, d["maker"].to_numpy(), max(KS)), y, 5)
        cmx = pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
        cmx["n"] = cmx[CLASSES].sum(axis=1)
        cmx["recall"] = (np.diag(cmx[CLASSES]) / cmx["n"].clip(lower=1)).round(2)
        cmx.index.name = "true"
        cmx.to_csv(out_dir(cfg) / f"confusion_{source}_{set_name}_{tag}_L{key[0]}_{key[1]}.csv")
        out[source] = cmx
    return out


# ── report ──────────────────────────────────────────────────────────────────
def _md(df: pd.DataFrame, fmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                       for v in r) + " |")
    return "\n".join(lines)


def _findings(cfg: Dict[str, Any], cm: pd.DataFrame) -> List[str]:
    """The headline numbers, read from the result tables (so a re-run keeps them true)."""
    d = out_dir(cfg)
    c = cm[(cm["label"] == "class") & (cm["set"] == "main")]
    best = lambda src: c[c["source"] == src].sort_values("knn5_bal_acc_maker_out").iloc[-1]
    ph, pa = best("photo"), best("patent")
    st = cm[(cm["label"] == "status") & (cm["set"] == "main")]["knn5_bal_acc"].max()
    L = ["## Findings", "",
         f"- **The 5 classes are in the photo embeddings.** Main image per aircraft, "
         f"{ph['embedding']} L{ph['layer']} {ph['pooling']}: kNN-5 balanced accuracy "
         f"**{ph['knn5_bal_acc']:.2f}** ({ph['knn5_bal_acc_maker_out']:.2f} with the maker held out), "
         f"linear probe {ph['probe_bal_acc_maker_out']:.2f}; chance 0.20 (shuffled-label 95th pct "
         f"{ph['knn5_perm_p95']:.2f}). The patent drawings, topType mapped to the same 5 parents: "
         f"{pa['knn5_bal_acc']:.2f} / {pa['knn5_bal_acc_maker_out']:.2f} / {pa['probe_bal_acc_maker_out']:.2f}. "
         f"Unsupervised k-means follows the classes on photos (ARI {ph['kmeans_ari']:.2f}) "
         f"and hardly on patents (ARI {pa['kmeans_ari']:.2f}).",
         f"- **But the photo space also encodes maturity** (built vs concept render): kNN-5 "
         f"{st:.2f} on 3 statuses, chance 0.33 — part of the structure is photo-vs-render, not architecture."]
    pc = d / "parent_check.csv"
    if pc.exists():
        p = pd.read_csv(pc)
        L.append(f"- **The 5 classes are the parents of the 12, with exceptions**: "
                 f"{int(p['agrees'].astype(str).eq('True').sum())} / {len(p)} linked aircraft agree. "
                 "The directory files unusual rotors (cyclorotor, slowed rotor, gyro) under Electric "
                 "Rotorcraft, and several SLC patents belong to families the directory lists as "
                 "Vectored Thrust — probably the patent shows another configuration than the aircraft "
                 "on the page (not checked; list in section 4).")
    mf = d / "matching.csv"
    if mf.exists():
        m = pd.read_csv(mf)
        h = m[(m["images"] == "no line drawings") & (m["direction"] == "patent->photo")]
        h = h.sort_values("mrr").iloc[-1]
        L.append(f"- **Patent ↔ photo matching works, modestly**: a patent aircraft's own directory page "
                 f"is in the top 10 of {int(h['gallery'])} for **{h['R@10']:.0%}** of {int(h['queries'])} "
                 f"named aircraft (random {h['random_R@10']:.1%}), top 50 for {h['R@50']:.0%} "
                 f"(random {h['random_R@50']:.1%}) — {h['embedding']} L{h['layer']} {h['pooling']}, "
                 "directory line drawings excluded.")
    return L + [""]


def write_report(cfg: Dict[str, Any]) -> Path:
    d = out_dir(cfg)
    src = root(cfg) / "0_source"
    sel = json.loads((root(cfg) / "2_embedding_extraction/selection/selection_summary.json").read_text())
    ac = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    im = pd.read_csv(root(cfg) / "1_filter/image_decisions.csv", keep_default_na=False)
    cm = pd.read_csv(d / "class_metrics.csv")
    L = [f"# evtol.news photo set — embedding evaluation",
         "", f"Generated {date.today().isoformat()} · private (images © their owners; nothing here "
         "is to be published before the source is asked).", "",
         "## 1. What went in", "",
         f"- Directory pages: **{len(ac)}** (index https://evtol.news/aircraft); "
         f"images on the pages: **{len(im)}**; kept after the whole-aircraft filter: "
         f"**{int((im['keep'].astype(str) == 'True').sum())}** on "
         f"**{im[im['keep'].astype(str) == 'True']['slug_key'].nunique()}** aircraft.",
         "", "Aircraft covered by each set (`built` / `concept` = the directory's own status; the "
         f"other {im[im['keep'].astype(str) == 'True']['slug_key'].nunique() - sum(r['built'] + r['concept'] for r in sel['by_class'])} "
         "aircraft have no status on their page or in the list, or a status that is neither, e.g. defunct):", "",
         _md(pd.DataFrame(sel["by_class"]).rename(columns={"topType": "class"}), "{}"), "",
         "Filter funnel (first rule that removed the image):", "",
         _md(pd.read_csv(root(cfg) / "2_embedding_extraction/selection/funnel.csv", keep_default_na=False), "{}"),
         ""]
    L[4:4] = _findings(cfg, cm)
    L += ["## 2. Label-free structure (notebook 30, sections A–D)", ""]
    for tag in cfg["evtolnews"]["embeddings"]:
        f = d / "metrics" / tag / "set_comparison.csv"
        if f.exists():
            sc = pd.read_csv(f).sort_values("rank_sum", ascending=False).groupby("set").head(1)
            L += [f"**{tag}** — best matrix per set:", "",
                  _md(sc[["set", "n_figures", "n_aircraft", "layer", "pooling", "pc1_ratio_vs_random",
                          "hopkins", "best_kmeans_silhouette", "bootstrap_ari", "hdbscan_noise_frac"]]), ""]
    L += ["## 3. Do the directory's 5 classes show in the embeddings?", "",
          "kNN = leave the aircraft's own images out (`maker_out`: leave its whole maker out); "
          "probe = logistic regression with maker-grouped folds; chance balanced accuracy = 0.20; "
          "`perm` = kNN-5 with shuffled labels. Patent rows use the codebook topType mapped to its parent class.", ""]
    keep = ["embedding", "set", "source", "layer", "pooling", "n", "n_aircraft", "knn1_bal_acc", "knn5_bal_acc",
            "knn5_bal_acc_maker_out", "knn5_perm_p95", "probe_bal_acc_maker_out", "silhouette_cos",
            "kmeans_ari", "kmeans_nmi"]
    c = cm[cm["label"] == "class"]
    best = c.sort_values("knn5_bal_acc_maker_out", ascending=False).groupby(["embedding", "set", "source"]).head(1)
    L += ["Best layer × pooling per run (by kNN-5, maker out):", "",
          _md(best[[k for k in keep if k in best]].sort_values(["embedding", "set", "source"])), ""]
    st = cm[cm["label"] == "status"]
    if len(st):
        bs = st.sort_values("knn5_bal_acc", ascending=False).groupby(["embedding", "set"]).head(1)
        L += ["Maturity confound — the same kNN predicting the directory's status "
              "(built prototype / concept design / unknown; 3 classes, chance 0.33), i.e. how much of the "
              "structure is \"real photo vs concept render\" rather than architecture:", "",
              _md(bs[["embedding", "set", "layer", "pooling", "n", "knn5_bal_acc", "kmeans_ari"]]), ""]
    conf = confusion_tables(cfg)
    L += ["Per-class recall, main set, dinov2-large_518 L24 CLS, kNN-5 with the maker held out "
          "(rows = true class):", ""]
    for source, t in conf.items():
        L += [f"*{source}*", "", _md(t.reset_index(), "{}"), ""]
    pc = d / "parent_check.csv"
    if pc.exists():
        p = pd.read_csv(pc, keep_default_na=False)
        L += ["## 4. Are the 5 classes the parents of the codebook's 12?", "",
              f"On the {len(p)} patent aircraft linked to directory pages, the codebook parent of the "
              f"human topType is among the pages' classes for **{int(p['agrees'].astype(str).eq('True').sum())}"
              f" / {len(p)}**.", "",
              _md(pd.crosstab(p["topType"], p["directory_class"]).reset_index(), "{}"), "",
              "Disagreements:", "",
              _md(p[p["agrees"].astype(str) != "True"][["aircraft_uid", "family", "topType",
                                                         "parent_of_topType", "directory_classes"]], "{}"), ""]
    mf = d / "matching.csv"
    if mf.exists():
        m = pd.read_csv(mf)
        L += ["## 5. Same-aircraft matching (patent ↔ photo)", "",
              "Query = the aircraft's main image; gallery = every kept image, pooled per aircraft "
              "(max cosine). A hit = any page/aircraft of the same family. `random_R@k` = the "
              "expected recall of a random ranking.", ""]
        L += ["`images = no line drawings` drops the 62 directory images that are line drawings: "
              "some pages reproduce the patent drawing itself (Leonardo, Bell Nexus: cosine 0.97–0.99 "
              "to the patent figure), which makes patent → photo trivially easy. That row is the honest "
              "drawing → photo/render number.", ""]
        bm = (m.sort_values("mrr", ascending=False)
              .groupby(["embedding", "images", "direction"]).head(1)
              .sort_values(["direction", "images", "embedding"]))
        L += [_md(bm.drop(columns=["gallery"])), ""]
    f = d / "EVTOLNEWS_EVALUATION.md"
    f.write_text("\n".join(L), encoding="utf-8")
    return f
