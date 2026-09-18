"""evtol.news World eVTOL Aircraft Directory as a second, photographic image source.

The Vertical Flight Society's directory (https://evtol.news/aircraft) lists every
known eVTOL design in five classes, each aircraft on its own page with photos or
renders, the maker, its location and a status (concept design / prototype / ...).
This module turns it into a data tree that the patent pipeline (notebooks 21-22,
evaluation notebook 30) can read by pointing ``pipeline_root`` at it.

The images are (c) their owners — each row of ``images.csv`` keeps the page's
credit line. They stay in the private data tree and are used for embeddings
only; nothing is published without asking the source (user ruling 2026-09-17).

Classes (the directory's own, used as the architecture label) and the parent
groups of the codebook's 12 topTypes they correspond to::

    VT  Vectored Thrust                      TW TR TB PTC DS CVT
    LC  Lift + Cruise                        SLC (SRW)
    WM  Wingless (Multicopter)               MR
    ER  Electric Rotorcraft                  RC
    HB  Hover Bikes/Personal Flying Devices  HB PFV

Layout under ``cfg["evtolnews"]["root"]``::

    0_source/index.html  index.csv          one row per directory list entry
    0_source/html/<slug>.html               raw aircraft pages
    0_source/pages.jsonl                    parsed page text (for later reading)
    0_source/aircraft.csv                   one row per aircraft page
    0_source/images.csv                     one row per image on a page
    0_source/images/<slug>/<nn>_<file>      downloaded images
    1_filter/                               whole-aircraft photo filter
    2_embedding_extraction/                 selection/ processed/ embeddings/ (patent layout)
    3_embedding_evaluation/                 metrics/
"""

from __future__ import annotations

import hashlib
import html as htmllib
import json
import re
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

BASE = "https://evtol.news/"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) eVTOL-thesis-research (non-commercial)"

SECTIONS = {  # heading on the index page -> class code
    "Vectored Thrust": "VT",
    "Hover Bikes/Personal Flying Devices": "HB",
    "Lift + Cruise": "LC",
    "Wingless (Multicopter)": "WM",
    "Electric Rotorcraft": "ER",
}
CLASS_NAMES = {v: k for k, v in SECTIONS.items()}
# media sub-folder "Aircraft Directory Images<suffix>" -> class code
FOLDER_CLASS = {"": "VT", "Lift plus Cruise": "LC", "Wingless (Multicopter)": "WM",
                "Electric Rotorcraft": "ER", "Hoverbikes": "HB"}
# codebook topType -> directory class (the codebook's own parent groups)
TOPTYPE_PARENT = {"TW": "VT", "TR": "VT", "TB": "VT", "PTC": "VT", "DS": "VT", "CVT": "VT",
                  "SLC": "LC", "SRW": "LC", "MR": "WM", "RC": "ER", "HB": "HB", "PFV": "HB"}
STATUSES = ["concept design", "prototype", "production model", "technology demonstrator",
            "defunct", "production prototype", "subscale", "full-scale"]


def root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["evtolnews"]["root"])


def source_dir(cfg: Dict[str, Any]) -> Path:
    d = root(cfg) / "0_source"
    d.mkdir(parents=True, exist_ok=True)
    return d


def pipeline_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """The patent config with ``pipeline_root`` moved to this tree (notebooks 21-22 code)."""
    return {**cfg, "paths": {**cfg["paths"],
                             "pipeline_root": root(cfg) / "2_embedding_extraction"}}


def _text(fragment: str) -> str:
    t = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    t = htmllib.unescape(re.sub(r"<[^>]+>", " ", t)).replace("\xa0", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", x).strip() for x in t.split("\n")).strip()


def _slug(href: str) -> Optional[str]:
    m = re.match(r"^(?:https?://(?:www\.)?evtol\.news)?/?([^?#]*?)/?$", href.strip(), re.I)
    if not m or not m.group(1) or "/" in m.group(1) or m.group(1).startswith("__"):
        return None
    if href.startswith("http") and "evtol.news" not in href.lower():
        return None
    return m.group(1)


def fetch(url: str, timeout: int = 40, retries: int = 3) -> bytes:
    url = urllib.parse.quote(url, safe=":/%?=&#+,;@!$'()*~")
    last: Exception | None = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 410):
                raise
            last = e
        except Exception as e:  # noqa: BLE001 — network errors are retried
            last = e
        time.sleep(2 * (k + 1))
    raise last  # type: ignore[misc]


# ── 1. index ────────────────────────────────────────────────────────────────
def parse_index(page: str) -> pd.DataFrame:
    """One row per ``<li>`` of the five class lists (an aircraft may sit in two)."""
    heads = [(m.start(), SECTIONS[k]) for k in SECTIONS
             for m in re.finditer(r"<strong>\s*" + re.escape(k).replace(r"\ ", r"\s+") + r"\s*</strong>", page)]
    heads.sort()
    end = page.find("Recent Pages")
    rows = []
    for i, (pos, code) in enumerate(heads):
        stop = heads[i + 1][0] if i + 1 < len(heads) else end
        block = page[pos:stop]
        for j, li in enumerate(re.findall(r"<li>(.*?)</li>", block, flags=re.S)):
            hrefs = [h for h in re.findall(r'href="([^"]+)"', li)]
            slugs = [s for s in (_slug(h) for h in hrefs) if s]
            if not slugs:
                continue
            anchors = " ".join(_text(a) for a in re.findall(r"<a[^>]*>(.*?)</a>", li, flags=re.S))
            tail = _text(re.sub(r"<a[^>]*>.*?</a>", "", li, flags=re.S))
            status = re.findall(r"\(([^)]*)\)", tail)
            rows.append({"slug": slugs[0], "slug_key": slugs[0].lower(), "evtol_class": code,
                         "list_pos": j, "list_name": re.sub(r"\s+", " ", anchors).strip(),
                         "list_status": status[-1].strip() if status else "",
                         "gofly": bool(re.search(r"-\s*I\b", tail)),
                         "url": BASE + slugs[0]})
    return pd.DataFrame(rows)


def build_index(cfg: Dict[str, Any]) -> pd.DataFrame:
    d = source_dir(cfg)
    page = fetch(BASE + "aircraft").decode("utf-8", "replace")
    (d / "index.html").write_text(page, encoding="utf-8")
    idx = parse_index(page)
    idx.to_csv(d / "index.csv", index=False)
    print(f"[index] {len(idx)} list entries, {idx.slug_key.nunique()} pages; per class:",
          idx.groupby("evtol_class").slug_key.nunique().to_dict())
    return idx


# ── 2. pages ────────────────────────────────────────────────────────────────
def crawl_pages(cfg: Dict[str, Any], workers: int = 2, pause: float = 0.5) -> None:
    """Save each aircraft page once (existing files are skipped)."""
    d = source_dir(cfg)
    hdir = d / "html"
    hdir.mkdir(exist_ok=True)
    idx = pd.read_csv(d / "index.csv", keep_default_na=False)
    todo = idx.drop_duplicates("slug_key")
    todo = todo[[not (hdir / f"{k}.html").exists() for k in todo.slug_key]]
    errors: Dict[str, str] = {}
    lock = threading.Lock()

    def one(slug: str, key: str) -> None:
        try:
            (hdir / f"{key}.html").write_bytes(fetch(BASE + slug))
        except Exception as e:  # noqa: BLE001
            with lock:
                errors[key] = str(e)
        time.sleep(pause)

    print(f"[pages] {len(todo)} to fetch")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo.slug, todo.slug_key))
    err_f = d / "page_errors.json"
    old = json.loads(err_f.read_text()) if err_f.exists() else {}
    old = {k: v for k, v in old.items() if not (hdir / f"{k}.html").exists()}
    old.update(errors)
    err_f.write_text(json.dumps(old, indent=1))
    print(f"[pages] done, {len(old)} errors")


def _content(page: str) -> str:
    a = page.find("<h1")
    b = page.find("Recent Pages", a)
    return page[a:b if b > 0 else len(page)]


def _media(src: str) -> Dict[str, str]:
    path = urllib.parse.unquote(urllib.parse.urlparse(urllib.parse.urljoin(BASE, src)).path)
    m = re.match(r"^/__media/Aircraft Directory Images ?([^/]*)/(.*)/([^/]+)$", path)
    if not m:
        return {"media_class": "", "media_folder": "/".join(path.split("/")[2:-1])}
    return {"media_class": FOLDER_CLASS.get(m.group(1).strip(), m.group(1).strip()),
            "media_folder": m.group(2)}


def parse_page(page: str, key: str) -> Dict[str, Any]:
    c = _content(page)
    title = _text((re.findall(r"<h1[^>]*>(.*?)</h1>", c, flags=re.S) or [""])[0])
    imgs, seen = [], set()
    tags = list(re.finditer(r"<img\b[^>]*>", c, flags=re.I))
    for i, m in enumerate(tags):
        attrs = dict((k.lower(), htmllib.unescape(v)) for k, v in
                     re.findall(r'(\w+)\s*=\s*"([^"]*)"', m.group(0)))
        src = attrs.get("src", "")
        if "__media/" not in src or "banners-and-ads" in src:
            continue
        url = urllib.parse.urljoin(BASE, src.strip())
        if url in seen:
            continue
        seen.add(url)
        nxt = tags[i + 1].start() if i + 1 < len(tags) else len(c)
        after = c[m.end():min(nxt, m.end() + 1500)]
        ems = [_text(e) for e in re.findall(r"<em>(.*?)</em>", after, flags=re.S)]
        credit = next((e for e in ems if re.search(r"credit|courtesy|source", e, re.I)), "")
        imgs.append({"slug_key": key, "img_pos": len(imgs), "src_url": url,
                     "alt": attrs.get("alt", ""), "title": attrs.get("title", ""),
                     "credit": credit, "caption": " | ".join(e for e in ems if e != credit)[:300],
                     **_media(src)})
    # header block: <p><strong>Model</strong> (status)<br/>Company<br/>Location<br/><a>site</a></p>
    head = {"model": "", "status": "", "company": "", "location": "", "website": ""}
    for p in re.findall(r"<p[^>]*>(.*?)</p>", c, flags=re.S):
        if "<strong" in p and len(re.findall(r"<br", p)) >= 2:
            lines = [x for x in _text(p).split("\n") if x]
            st = re.findall(r"\(([^)]*)\)", lines[0])
            head["model"] = re.sub(r"\s*\([^)]*\)\s*$", "", lines[0]).strip()
            head["status"] = st[-1] if st else ""
            rest = lines[1:]
            site = re.findall(r'href="([^"]+)"', p)
            if site and rest and re.search(r"www\.|https?://|\.\w{2,3}$", rest[-1]):
                head["website"] = rest[-1]
                rest = rest[:-1]
            head["company"] = rest[0] if rest else ""
            head["location"] = rest[-1] if len(rest) >= 2 else ""
            break
    text = _text(c)
    spec = {}
    for k in ["Aircraft type", "Piloting", "Capacity", "Cruise speed", "Range", "Flight time",
              "Propellers", "Electric motors", "Power source", "Fuselage", "Wings", "Windows",
              "Landing gear", "Safety features", "Tail"]:
        m = re.search(r"(?:^|\n)\s*" + re.escape(k) + r"s?\s*:\s*([^\n]{1,300})", text, re.I)
        spec["spec_" + k.lower().replace(" ", "_")] = m.group(1).strip() if m else ""
    return {"slug_key": key, "title": title, **head, **spec, "n_images": len(imgs),
            "images": imgs, "text": text[:30000]}


def parse_pages(cfg: Dict[str, Any]) -> pd.DataFrame:
    """aircraft.csv + images.csv (+ pages.jsonl) from the saved pages."""
    d = source_dir(cfg)
    idx = pd.read_csv(d / "index.csv", keep_default_na=False)
    classes = (idx.groupby("slug_key")
               .agg(slug=("slug", "first"), url=("url", "first"),
                    list_name=("list_name", "first"), list_status=("list_status", "first"),
                    evtol_class=("evtol_class", lambda s: "|".join(dict.fromkeys(s))),
                    list_pos=("list_pos", "first"))
               .reset_index())
    rows, imgs = [], []
    with open(d / "pages.jsonl", "w", encoding="utf-8") as fj:
        for r in classes.itertuples(index=False):
            f = d / "html" / f"{r.slug_key}.html"
            if not f.exists():
                rows.append({**r._asdict(), "page_status": "missing"})
                continue
            p = parse_page(f.read_text(encoding="utf-8", errors="replace"), r.slug_key)
            fj.write(json.dumps({k: p[k] for k in ("slug_key", "title", "text")}, ensure_ascii=False) + "\n")
            imgs.extend(p.pop("images"))
            p.pop("text")
            rows.append({**r._asdict(), **p, "page_status": "ok"})
    ac = pd.DataFrame(rows)
    ac["n_classes"] = ac["evtol_class"].str.count(r"\|") + 1
    ac["evtol_class_main"] = ac["evtol_class"].str.split("|").str[0]
    im = pd.DataFrame(imgs)
    # the page's own media folder = the folder of its first aircraft-directory image
    own = (im[im.media_class != ""].groupby("slug_key").media_folder.first())
    im["own_folder"] = im.slug_key.map(own).fillna("")
    im["in_own_folder"] = im.media_folder == im.own_folder
    im["image_uid"] = im.slug_key + "__" + im.img_pos.astype(int).map("{:02d}".format)
    ac.to_csv(d / "aircraft.csv", index=False)
    old = d / "images.csv"
    if old.exists():  # keep download columns of earlier runs
        prev = pd.read_csv(old, keep_default_na=False)
        keep = [c for c in prev.columns if c not in im.columns or c == "src_url"]
        im = im.merge(prev[keep].drop_duplicates("src_url"), on="src_url", how="left")
    im.to_csv(old, index=False)
    print(f"[parse] {len(ac)} aircraft pages ({(ac.page_status == 'ok').sum()} ok), {len(im)} images")
    return ac


# ── 3. images ───────────────────────────────────────────────────────────────
def download_images(cfg: Dict[str, Any], workers: int = 2, pause: float = 0.4) -> pd.DataFrame:
    from PIL import Image

    d = source_dir(cfg)
    im = pd.read_csv(d / "images.csv", keep_default_na=False)
    for col in ("local_path", "sha1", "bytes", "width", "height", "img_format", "dl_error"):
        if col not in im.columns:
            im[col] = ""
    lock = threading.Lock()

    def one(i: int) -> None:
        r = im.loc[i]
        name = re.sub(r"[^\w.\-]+", "_", Path(urllib.parse.unquote(urllib.parse.urlparse(r.src_url).path)).name)
        dst = d / "images" / r.slug_key / f"{int(r.img_pos):02d}_{name}"
        out: Dict[str, Any] = {"local_path": str(dst), "dl_error": ""}
        try:
            if not dst.exists():
                data = fetch(r.src_url)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)
                time.sleep(pause)
            data = dst.read_bytes()
            out["sha1"] = hashlib.sha1(data).hexdigest()
            out["bytes"] = len(data)
            with Image.open(dst) as pic:
                out["width"], out["height"], out["img_format"] = pic.size[0], pic.size[1], pic.format
        except Exception as e:  # noqa: BLE001
            out = {"local_path": "", "dl_error": str(e)[:200]}
        with lock:
            for k, v in out.items():
                im.at[i, k] = v

    todo = [i for i in im.index if not str(im.at[i, "sha1"])]
    print(f"[images] {len(todo)} to fetch/read")
    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(one, todo):
            done += 1
            if done % 250 == 0:
                im.to_csv(d / "images.csv", index=False)
                print(f"[images] {done}/{len(todo)}", flush=True)
    im.to_csv(d / "images.csv", index=False)
    print(f"[images] ok {(im.sha1 != '').sum()}, errors {(im.dl_error != '').sum()}")
    return im
