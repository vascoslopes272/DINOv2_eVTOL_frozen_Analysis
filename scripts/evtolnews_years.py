#!/usr/bin/env python3
"""Year of each evtol.news directory aircraft, read from the saved page text by keyword rules.

Nothing is inferred: every year kept comes with the sentence it was read from, the keyword
that typed it, and a link that opens the live page with that sentence highlighted
(a text fragment, ``#:~:text=``). The same input always gives the same output.

Input   EVTOLNEWS_DS/0_source/pages.jsonl  (page text, crawled 2026-09-17)
        EVTOLNEWS_DS/0_source/aircraft.csv (slug, name, model, company, status, class)
Output  EVTOLNEWS_DS/0_source/years/
        year_evidence.csv  one row per (page, sentence, year): kind, keyword, quote, link
        aircraft_years.csv one row per aircraft: year_rule (+ its quote and link),
                           year_first_flight, year_founded, year_first_mention, year_tier

Year kinds (prose sentences; first matching rule in this order wins)
  future       plan/expect/target/aim/will/by YYYY ... or a year after the crawl  -> never used
  founded      founded / established / incorporated / formed                      -> company, not aircraft
  dev_start    began/started developing|designing|work|program|project, conceived, the idea, designed in
  first_flight first flight / maiden flight / first flew / first hover / flew for the first time
  reveal       unveiled / revealed / debuted / introduced / announced / presented / showcased / exhibited
  flight       flight test / flew / flown / hovered / test flight
  source       article / reported / press release / interview (the date of a source, not an event)
  other        any other past year
  resource     an entry of the page's "Resources:" list (Article: ..., Video: ...); only its
               publication date (the year at the end of the line) is read, never the title words

year_rule is filled by the first tier that has evidence; year_tier says which:
  A  earliest development record naming the model: a dev_start / first_flight / reveal / flight
     sentence, or a Resources entry (its publication date), whose words name the model
  B  earliest dev_start / first_flight / reveal / flight sentence, model not named (can be another aircraft)
  C  earliest other dated sentence naming the model ("designed in 2019", "as of 2021 ...")
  D  earliest dated mention of anything except founded / future       -> weakest, company history
  -  no dated statement on the page
Every dated record is an upper bound of the year the development started, so the earliest one
is kept; A and B are event years, C and D are "the page already mentions it by" years.

Usage: python scripts/evtolnews_years.py [--ds <EVTOLNEWS_DS>]
"""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import quote

import pandas as pd

DS = Path("/mnt/storage_11tb/Drive_files_to_syncronize/3 - Images DataSets & Labelling Outputs/EVTOLNEWS_DS")
CRAWL_YEAR = 2026

# a year, not part of a model number such as J-2000 or VT-2030 (a year range 2023-2024 still counts)
YEAR = re.compile(r"(?<![A-Za-z]-)(?<![A-Za-z0-9])(19[5-9]\d|20[0-3]\d)(?![A-Za-z0-9])")
# abbreviations whose dot must not end a sentence
ABBR = re.compile(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Co|Corp|St|No|vs|approx|U\.S|U\.K|e\.g|i\.e)\.", re.I)

RULES = [  # (kind, pattern) — searched in the clause around the year
    ("future", r"\b(plan\w*|expect\w*|target\w*|aim\w*|hope\w*|intend\w*|will|would|could|project(ed|s)|schedul\w*|anticipat\w*|goal|forecast\w*|estimat\w*|should|may|might|by (the end of )?(early |mid-?|late )?(19|20)\d\d)\b"),
    ("founded", r"\b(co-?founded|founded|founding|established|incorporated|formed)\b"),
    ("dev_start", r"\b((began|begun|started|start|commenced|initiated|kicked off)\b[^.;]{0,40}\b(develop\w*|design\w*|work\w*|program\w*|project\w*|build\w*|construct\w*|research\w*)|(development|design|work|program|project|construction|research)\b[^.;]{0,30}\b(began|begun|started|commenced|kicked off|was launched|was initiated)|conceiv\w*|the idea|designed in|in development since|under development since)\b"),
    ("first_flight", r"\b(first (\w+ ){0,2}(flight|flew|hover\w*)|maiden (\w+ )?flight|flew for the first time|first took (to the )?(air|flight)|took off for the first time)\b"),
    ("reveal", r"\b(unveil\w*|reveal\w*|debut\w*|introduc\w*|announc\w*|present(ed|s|ing)|showcas\w*|exhibit\w*|display\w*|made public|first shown|premiere\w*|rolled out|roll-?out)\b"),
    ("flight", r"\b(flight[- ]test\w*|test flight\w*|flight trials?|flew|flown|hover(ed|ing) (test|flight)\w*|began fly\w*|started fly\w*)\b"),
    ("source", r"\b(article|reported|report|press release|interview\w*|according to|wrote|published|news)\b"),
]
RULES = [(k, re.compile(p, re.I)) for k, p in RULES]
RES_DATE = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\s*\W*$")
PUBLIC_KINDS = ("dev_start", "first_flight", "reveal", "flight")
STOP = {"the", "a", "an", "of", "and", "evtol", "vtol", "aircraft", "concept", "design", "air", "taxi", "drone",
        "prototype", "model", "version", "technology", "demonstrator", "production", "passenger", "cargo", "electric"}


def sentences(text):
    """(sentence, is_resource) pairs; lines are hard breaks, abbreviation dots are protected,
    every line after "Resources:" is one resource entry."""
    out, in_res = [], False
    for line in text.split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        if line.rstrip(":").strip().lower() in ("resources", "company insights"):
            in_res = line.lower().startswith("resources")
            continue
        if in_res:
            out.append((line, True))
            continue
        prot = ABBR.sub(lambda m: m.group(0).replace(".", "\x00"), line)
        for s in re.split(r"(?<=[.!?])\s+(?=[\"“A-Z0-9])", prot):
            s = s.replace("\x00", ".").strip()
            if s:
                out.append((s, False))
    return out


def kind_of(sent, m):
    """Type one year occurrence from the words of its clause (the text between the surrounding
    ';' or ',' boundaries is too narrow for 'In 2019, X unveiled ...', so use a character window)."""
    lo, hi = max(0, m.start() - 110), min(len(sent), m.end() + 50)
    window = sent[lo:hi]
    if int(m.group(1)) > CRAWL_YEAR:
        return "future", m.group(1)
    for kind, pat in RULES:
        hits = list(pat.finditer(window))
        if hits:
            # nearest keyword to the year wins the evidence label
            pos = m.start() - lo
            h = min(hits, key=lambda h: abs(h.start() - pos))
            return kind, h.group(0)
    return "other", ""


def model_tokens(row):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-]+", f"{row.get('model', '')} {row.get('list_name', '')}")
    toks = {w.lower() for w in words if w.lower() not in STOP and (len(w) >= 3 or re.search(r"\d", w) and len(w) >= 2)}
    comp = {w.lower() for w in re.findall(r"[A-Za-z0-9]+", str(row.get("company", "")))}
    return toks - comp  # the model's own words; empty when the name is only the company (then no sentence "names the model")


def fragment_link(url, sent):
    """Chrome/Edge text-fragment link that scrolls to and highlights the sentence on the live page."""
    enc = lambda s: quote(s, safe="").replace("-", "%2D")
    words = sent.split()
    if len(words) <= 12:
        return f"{url}#:~:text={enc(sent)}"
    return f"{url}#:~:text={enc(' '.join(words[:6]))},{enc(' '.join(words[-4:]))}"


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>evtol.news years</title><style>
:root{--bg:#fff;--ink:#1d1d1f;--mute:#6e6e73;--line:#e3e3e8;--A:#1a7f37;--B:#9a6700;--C:#0969da;--D:#8250df;--none:#cf222e}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--ink:#e8e8ea;--mute:#9a9aa0;--line:#333338}}
body{background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;margin:0 16px 40px}
header{position:sticky;top:0;background:var(--bg);padding:10px 0;border-bottom:1px solid var(--line);display:flex;gap:10px;flex-wrap:wrap;align-items:center}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--line);padding:6px;vertical-align:top;text-align:left}
th{font-weight:600;color:var(--mute);font-size:12px}.y{font-weight:700;font-size:16px}.q{max-width:700px}.m{color:var(--mute);font-size:12px}
.t{font-weight:700;padding:1px 6px;border-radius:4px;color:#fff}.tA{background:var(--A)}.tB{background:var(--B)}.tC{background:var(--C)}.tD{background:var(--D)}.t-{background:var(--none)}
button{font:inherit;padding:2px 8px;cursor:pointer}tr.ok td{background:color-mix(in srgb,var(--A) 12%,transparent)}tr.bad td{background:color-mix(in srgb,var(--none) 12%,transparent)}
</style></head><body><header><b>evtol.news: first public year per aircraft</b>
<select id=tier><option value="">all tiers</option><option>A</option><option>B</option><option>C</option><option>D</option><option value="-">none</option></select>
<input id=q placeholder="search name / company" size=28><span id=n class=m></span><button id=exp>export my checks (CSV)</button>
<span class=m>A event naming the model · B event, model not named · C earliest dated mention naming the model · D earliest dated mention of anything</span></header>
<table><thead><tr><th>aircraft</th><th>year</th><th>tier</th><th>evidence (click: opens the page at this sentence)</th><th>other years</th><th>check</th></tr></thead><tbody id=b></tbody></table>
<script>const R=__DATA__;const K='evtolnewsYears_v1';let C={};try{C=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function draw(){const t=tier.value,q=document.getElementById('q').value.toLowerCase();let h='',n=0;
for(const r of R){if(t&&r.year_tier!==t)continue;if(q&&!(r.list_name+' '+r.company).toLowerCase().includes(q))continue;n++;
const c=C[r.slug_key]||{};h+=`<tr class="${c.v||''}" data-s="${r.slug_key}"><td><a href="${esc(r.url)}" target=evtolnews>${esc(r.list_name)}</a><div class=m>${esc(r.company)} · ${esc(r.status)} · ${esc(r.evtol_class)}</div></td>
<td class=y>${r.year_rule??'—'}</td><td><span class="t t${r.year_tier}">${r.year_tier}</span><div class=m>${esc(r.year_kind)} ${esc(r.year_keyword)}</div></td>
<td class=q>${r.year_link?`<a href="${esc(r.year_link)}" target=evtolnews>${esc(r.year_quote)}</a>`:'<span class=m>no dated statement on the page</span>'}</td>
<td class=m>first flight ${r.year_first_flight??'—'}<br>founded ${r.year_founded??'—'}<br>first mention ${r.year_first_mention??'—'}</td>
<td><button data-v=ok>✓</button> <button data-v=bad>✗</button><br><input data-y placeholder="correct year" size=8 value="${esc(c.y||'')}"></td></tr>`}
b.innerHTML=h;n_.textContent=n+' aircraft'}
const n_=document.getElementById('n');tier.onchange=draw;document.getElementById('q').oninput=draw;
b.onclick=e=>{const v=e.target.dataset.v;if(!v)return;const s=e.target.closest('tr').dataset.s;C[s]={...(C[s]||{}),v:C[s]?.v===v?'':v};save();e.target.closest('tr').className=C[s].v};
b.onchange=e=>{if(!('y' in e.target.dataset))return;const s=e.target.closest('tr').dataset.s;C[s]={...(C[s]||{}),y:e.target.value};save()};
function save(){try{localStorage.setItem(K,JSON.stringify(C))}catch(e){}}
exp.onclick=()=>{const rows=[['slug_key','year_rule','year_tier','check','correct_year']];for(const r of R){const c=C[r.slug_key];if(c&&(c.v||c.y))rows.push([r.slug_key,r.year_rule??'',r.year_tier,c.v||'',c.y||''])}
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(x=>'"'+String(x).replace(/"/g,'""')+'"').join(',')).join('\\n')],{type:'text/csv'}));a.download='YEARS_CHECK.csv';a.click()};
draw();</script></body></html>"""


def write_review_page(res, path):
    cols = ["slug_key", "list_name", "company", "status", "evtol_class", "url", "year_rule", "year_tier",
            "year_kind", "year_keyword", "year_quote", "year_link", "year_first_flight", "year_founded", "year_first_mention"]
    data = json.loads(res[cols].to_json(orient="records"))
    path.write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default=str(DS))
    a = ap.parse_args()
    src = Path(a.ds) / "0_source"
    out = src / "years"
    out.mkdir(exist_ok=True)

    air = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    pages = {r["slug_key"]: r for r in map(json.loads, open(src / "pages.jsonl"))}

    toks_of = {r.slug_key: model_tokens(r) for _, r in air.iterrows()}
    ev = []
    for _, row in air.iterrows():
        text = pages.get(row.slug_key, {}).get("text", "")
        # words shared with the maker's other directory aircraft (Aptos Blue / Aptos Mini) do not
        # identify this one; keep only the distinctive words, all of them if nothing is distinctive
        sib = set().union(*[toks_of[k] for k in air.slug_key[(air.company == row.company) & (air.slug_key != row.slug_key)]])
        toks = (toks_of[row.slug_key] - sib) or toks_of[row.slug_key]
        toks_re = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(map(re.escape, sorted(toks, key=len, reverse=True))) + r")(?![A-Za-z0-9])", re.I) if toks else None
        for i, (s, is_res) in enumerate(sentences(text)):
            if is_res:  # publication date at the end of the entry, nothing else
                m = RES_DATE.search(s)
                found = [(m, "resource", "")] if m else []
            else:
                found = [(m, *kind_of(s, m)) for m in YEAR.finditer(s)]
            for m, kind, kw in found:
                ev.append(dict(slug_key=row.slug_key, list_name=row.list_name, year=int(m.group(1)), kind=kind,
                               keyword=kw, names_model=bool(toks_re and toks_re.search(s)),
                               sentence_no=i, quote=s, link=fragment_link(row.url, s)))
    ev = pd.DataFrame(ev).drop_duplicates(["slug_key", "year", "kind", "quote"])
    ev.to_csv(out / "year_evidence.csv", index=False)

    rows = []
    for _, row in air.iterrows():
        e = ev[ev.slug_key == row.slug_key] if len(ev) else ev
        past = e[~e.kind.isin(["future"])]
        pub = past[past.kind.isin(PUBLIC_KINDS)]
        prose = past[past.kind != "resource"]
        pub = prose[prose.kind.isin(PUBLIC_KINDS)]
        mention = past[past.kind != "founded"]
        public = pd.concat([pub[pub.names_model], past[(past.kind == "resource") & past.names_model]])
        tiers = [("A", public), ("B", pub), ("C", mention[mention.names_model]), ("D", mention)]
        pick, tier = None, "-"
        for t, pool in tiers:
            if len(pool):
                pick, tier = pool.sort_values(["year", "sentence_no"]).iloc[0], t
                break
        res_named = past[(past.kind == "resource") & past.names_model]
        ff = past[past.kind == "first_flight"]
        fo = past[past.kind == "founded"]
        rows.append(dict(
            slug_key=row.slug_key, list_name=row.list_name, company=row.company, status=row.status,
            evtol_class=row.evtol_class_main, url=row.url,
            year_rule=int(pick.year) if pick is not None else None,
            year_tier=tier,
            year_kind=pick.kind if pick is not None else "",
            year_keyword=pick.keyword if pick is not None else "",
            year_quote=pick.quote if pick is not None else "",
            year_link=pick.link if pick is not None else "",
            year_first_flight=int(ff.year.min()) if len(ff) else None,
            year_founded=int(fo.year.min()) if len(fo) else None,
            year_first_mention=int(mention.year.min()) if len(mention) else None,
            year_first_resource_naming_model=int(res_named.year.min()) if len(res_named) else None,
            n_year_sentences=int(e.quote.nunique()) if len(e) else 0,
        ))
    res = pd.DataFrame(rows)
    for c in ["year_rule", "year_first_flight", "year_founded", "year_first_mention",
              "year_first_resource_naming_model"]:
        res[c] = res[c].astype("Int64")
    res.to_csv(out / "aircraft_years.csv", index=False)

    write_review_page(res, out / "years_review.html")
    print(f"{len(res)} aircraft, {len(ev)} dated sentences -> {out}")
    print(res.year_tier.value_counts().sort_index().to_string())
    print("year_founded filled:", res.year_founded.notna().sum(), "| first_flight:", res.year_first_flight.notna().sum())


if __name__ == "__main__":
    main()
