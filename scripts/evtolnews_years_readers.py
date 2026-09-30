#!/usr/bin/env python3
"""Development-start year of each evtol.news aircraft by two independent readers over numbered evidence.

Step 1  packets  every dated sentence of every page gets a fixed id (E<page>.<nn>); the pages are cut
                 into packet files (years/readers/packet_NN.txt). A reader sees the packet only.
Step 2  (outside this script) two readers, independently, answer each page of a packet by picking ONE
                 evidence id — the prompt is years/readers/READER_PROMPT.md — and write
                 years/readers/reader_<R>_packet_NN.csv  (R = a | b).
Step 3  merge    checks every answer mechanically (the id exists on that page, the year is in that
                 sentence), measures the agreement of the two readers, and writes
                 years/aircraft_dev_year.csv: agreed year + its sentence + highlight link, or
                 status = conflict (both picks shown, for the author) / not_stated.

Definition read by the readers (READER_PROMPT.md): the earliest year the page documents for THIS
aircraft's development — design / program start, announcement or unveiling, first flight or flight
tests, or a Resources publication about it. Every such record is an upper bound of the start of the
development, so the earliest one is the estimate. Company founding, other models of the maker,
predecessors that are a different aircraft, and planned / future dates are excluded.

Usage: python scripts/evtolnews_years_readers.py packets [--per 80]
       python scripts/evtolnews_years_readers.py merge
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evtolnews_years import DS, YEAR, RES_DATE, sentences, fragment_link  # noqa: E402

PROMPT = """# Reader prompt — development-start year of evtol.news aircraft

You read packets of the evtol.news World eVTOL Aircraft Directory. Each page block starts with a line
`### P<nnnn> slug=<slug> | aircraft: ... | model: ... | company: ... | status: ... | type: ...`
then `other aircraft by this maker: ...`, then the page's dated sentences, one per line:
`E<nnnn>.<nn> <sentence>` — optionally preceded by `(context: <previous sentence>)` — and entries of the
page's Resources list marked `[Resources]` (only their publication date, at the end, counts).

For EVERY page answer from the packet text ONLY — never from your own knowledge of the aircraft:
pick the ONE evidence id whose year is the EARLIEST year the page documents for the development of
THIS aircraft (the one in the header):
  * counts: design or program start ("started designing in 2017", "the idea was conceived in 2011"),
    announcement / unveiling / exhibition, first flight / flight tests / prototype flying, a Resources
    article or video that is about this aircraft (its publication date), earlier versions or subscale
    prototypes OF THIS SAME aircraft, a dated statement showing it already existed ("as of 2021 ...").
  * never counts: the company's founding or history, a founder's biography, other models of the maker
    (see the list in the header), a predecessor that is a different aircraft, generic technology or
    industry facts, planned or future dates ("will fly in 2027", "certification by 2026").
  * if the sentence says "the company" / "it" / "the aircraft", use the context line and the header to
    decide whether it is this aircraft; if unclear, prefer an id that names the aircraft.
  * a year range ("2023-2024") counts as its first year.

Write one CSV row per page, header exactly
page_id,slug,evidence_id,year,basis,note
  basis: dev_start | announced | flight | publication | existed_by | not_stated
  evidence_id and year empty when not_stated (nothing dated is about this aircraft);
  note: short, only when ambiguous (e.g. "sentence may be about the S2, not the S4").
Quote every field with double quotes. One row for every page of the packet, none missing.
"""


def build_packets(per):
    src = DS / "0_source"
    out = src / "years" / "readers"
    out.mkdir(parents=True, exist_ok=True)
    (out / "READER_PROMPT.md").write_text(PROMPT)
    air = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    pages = {r["slug_key"]: r for r in map(json.loads, open(src / "pages.jsonl"))}
    blocks, ev = [], []
    for p, row in enumerate(air.itertuples(), 1):
        sib = sorted(air.list_name[(air.company == row.company) & (air.slug_key != row.slug_key)])
        lines = [f"### P{p:04d} slug={row.slug_key} | aircraft: {row.list_name} | model: {row.model} | "
                 f"company: {row.company} | status: {row.status} | type: {row.spec_aircraft_type[:160]}",
                 "other aircraft by this maker: " + ("; ".join(sib) if sib else "none listed")]
        prev, n = "", 0
        for s, is_res in sentences(pages.get(row.slug_key, {}).get("text", "")):
            has = RES_DATE.search(s) if is_res else YEAR.search(s)
            if has:
                n += 1
                eid = f"E{p:04d}.{n:02d}"
                ctx = "" if is_res or not prev else f"(context: {prev[:220]}) "
                lines.append(f"{eid} {'[Resources] ' if is_res else ''}{ctx}{s}")
                ev.append(dict(evidence_id=eid, page_id=f"P{p:04d}", slug_key=row.slug_key, is_resource=is_res,
                               years=" ".join(sorted({m.group(1) for m in YEAR.finditer(s)})) if not is_res else has.group(1),
                               sentence=s, link=fragment_link(row.url, s)))
            prev = s
        if n == 0:
            lines.append("(no dated sentence on this page)")
        blocks.append("\n".join(lines))
    pd.DataFrame(ev).to_csv(out / "evidence_ids.csv", index=False)
    for k in range(0, len(blocks), per):
        (out / f"packet_{k // per + 1:02d}.txt").write_text("\n\n".join(blocks[k:k + per]) + "\n")
    sizes = [len("\n\n".join(blocks[k:k + per])) for k in range(0, len(blocks), per)]
    print(f"{len(blocks)} pages, {len(ev)} evidence ids, {len(sizes)} packets of {per} pages "
          f"({min(sizes) // 1000}-{max(sizes) // 1000} k chars) -> {out}")

def read_answers(folder, reader):
    frames = []
    for f in sorted(folder.glob(f"reader_{reader}_packet_*.csv")):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        d["packet"] = f.stem.split("_")[-1]
        frames.append(d)
    return pd.concat(frames) if frames else pd.DataFrame(columns=["page_id", "slug", "evidence_id", "year", "basis", "note"])


def check(ans, evid):
    """Mechanical check of one reader: the id is on that page and its sentence carries that year."""
    ev = evid.set_index("evidence_id")
    bad = []
    for r in ans.itertuples():
        if r.basis == "not_stated":
            if r.evidence_id:
                bad.append((r.page_id, "not_stated with an id"))
            continue
        if r.evidence_id not in ev.index:
            bad.append((r.page_id, f"unknown id {r.evidence_id}"))
        elif ev.loc[r.evidence_id, "page_id"] != r.page_id:
            bad.append((r.page_id, f"id {r.evidence_id} is on another page"))
        elif r.year not in str(ev.loc[r.evidence_id, "years"]).split():
            bad.append((r.page_id, f"year {r.year} not in {r.evidence_id}"))
    return bad


def merge():
    src = DS / "0_source"
    yd, rd = src / "years", src / "years" / "readers"
    evid = pd.read_csv(rd / "evidence_ids.csv", dtype=str, keep_default_na=False)
    air = pd.read_csv(src / "aircraft.csv", keep_default_na=False)
    air["page_id"] = [f"P{i:04d}" for i in range(1, len(air) + 1)]
    rule = pd.read_csv(yd / "aircraft_years.csv", keep_default_na=False)[["slug_key", "year_rule", "year_tier"]]
    A, B = read_answers(rd, "a"), read_answers(rd, "b")
    report = []
    for name, ans in (("a", A), ("b", B)):
        bad = check(ans, evid)
        report.append(f"reader {name}: {len(ans)} answers, {len(bad)} failed the mechanical check")
        report += [f"  {p}: {why}" for p, why in bad]
        for p, _ in bad:  # a failed answer counts as no answer
            ans.loc[ans.page_id == p, ["evidence_id", "year", "basis"]] = ["", "", "failed_check"]
    m = air[["page_id", "slug_key", "list_name", "company", "status", "evtol_class_main", "url"]].rename(
        columns={"evtol_class_main": "evtol_class"})
    for name, ans in (("a", A), ("b", B)):
        m = m.merge(ans[["page_id", "evidence_id", "year", "basis", "note"]].add_suffix(f"_{name}")
                    .rename(columns={f"page_id_{name}": "page_id"}), on="page_id", how="left")
    m = m[m.basis_a.notna() & m.basis_b.notna()].fillna("")
    dec_f = yd / "DEV_YEAR_DECISIONS.csv"  # the author's rulings on conflicts (from dev_year_review.html)
    dec = pd.read_csv(dec_f, dtype=str, keep_default_na=False).set_index("slug_key") if dec_f.exists() else pd.DataFrame()
    rev_f = yd / "YEAR_REVIEW.csv"  # the author's review of every aircraft (from year_review.html) — final word
    rev = pd.read_csv(rev_f, dtype=str, keep_default_na=False).drop_duplicates("slug_key", keep="last").set_index("slug_key") \
        if rev_f.exists() else pd.DataFrame()
    if len(rev):
        bad = [k for k, d in rev.iterrows() if d.evidence_id and (d.evidence_id not in set(evid.evidence_id) or
               evid.set_index("evidence_id").loc[d.evidence_id, "slug_key"] != k or
               d.year not in str(evid.set_index("evidence_id").loc[d.evidence_id, "years"]).split())]
        report.append(f"author review: {len(rev)} aircraft, {len(bad)} failed the mechanical check {bad[:10]}")
    ev = evid.set_index("evidence_id")
    rows = []
    for r in m.itertuples():
        agree = r.year_a == r.year_b
        if r.slug_key in rev.index:
            d = rev.loc[r.slug_key]
            year, eid, status = d.year, d.evidence_id, f"author_{d.action}"
        elif r.slug_key in dec.index:
            d = dec.loc[r.slug_key]
            year, eid, status = d.year, d.get("evidence_id", ""), "author"
        elif agree:
            year, status = r.year_a, ("agreed" if r.year_a else "not_stated")
            ids = [i for i in (r.evidence_id_a, r.evidence_id_b) if i]
            eid = min(ids) if ids else ""
        else:
            year, eid, status = "", "", "conflict"
        rows.append(dict(slug_key=r.slug_key, list_name=r.list_name, company=r.company, status=r.status,
                         evtol_class=r.evtol_class, dev_year=year, dev_year_status=status, evidence_id=eid,
                         evidence=ev.loc[eid, "sentence"] if eid in ev.index else "",
                         evidence_link=ev.loc[eid, "link"] if eid in ev.index else "", url=r.url,
                         year_a=r.year_a, basis_a=r.basis_a, id_a=r.evidence_id_a, note_a=r.note_a,
                         year_b=r.year_b, basis_b=r.basis_b, id_b=r.evidence_id_b, note_b=r.note_b))
    res = pd.DataFrame(rows).merge(rule, on="slug_key", how="left")
    res.to_csv(yd / "aircraft_dev_year.csv", index=False)
    n = len(res)
    both = res[(res.year_a != "") & (res.year_b != "")]
    ya, yb = pd.to_numeric(both.year_a), pd.to_numeric(both.year_b)
    report += [f"pages read by both: {n}",
               f"readers agree on the year (incl. both not_stated): {(res.year_a == res.year_b).sum()} ({(res.year_a == res.year_b).mean():.1%})",
               f"  both gave a year: {len(both)}, same year {(ya == yb).sum()}, within 1 year {((ya - yb).abs() <= 1).sum()}",
               f"  one gave a year, the other not_stated: {((res.year_a == '') != (res.year_b == '')).sum()}",
               "status: " + res.dev_year_status.value_counts().to_dict().__repr__()]
    if len(rev):
        rr_ = res[res.dev_year_status.str.startswith("author_")]
        was_agreed = rr_[(rr_.year_a == rr_.year_b) & (rr_.year_a != "")]
        ok = (was_agreed.dev_year == was_agreed.year_a).sum()
        report += [f"author review status: {rr_.dev_year_status.value_counts().to_dict()}",
                   f"readers' agreed year confirmed by the author: {ok} / {len(was_agreed)}"
                   + (f" ({ok / len(was_agreed):.1%})" if len(was_agreed) else "")]
    ag = res[res.dev_year_status == "agreed"]
    rr = pd.to_numeric(ag.year_rule, errors="coerce")
    report.append(f"keyword rule vs agreed year: same {(rr == pd.to_numeric(ag.dev_year)).sum()} / {len(ag)}")
    (rd / "agreement.txt").write_text("\n".join(report) + "\n")
    write_conflict_page(res, yd / "dev_year_review.html")
    write_review(yd)
    print("\n".join(r for r in report if not r.startswith("  P")))


def write_conflict_page(res, path):
    from evtolnews_years import PAGE  # reuse the stylesheet of the years page
    style = PAGE[PAGE.index("<style>"):PAGE.index("</style>") + 8]
    data = json.loads(res.to_json(orient="records"))
    js = """const R=__DATA__;const K='evtolnewsDevYear_v1';let D={};try{D=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const EV=__EV__;
function cell(r,w){const id=r['id_'+w];if(!id)return `<span class=m>${esc(r['basis_'+w])}</span>`;const e=EV[id]||{};
return `<b>${esc(r['year_'+w])}</b> <span class=m>${esc(r['basis_'+w])}</span><br><a href="${esc(e.l)}" target=evtolnews>${esc(e.s)}</a>${r['note_'+w]?'<div class=m>'+esc(r['note_'+w])+'</div>':''}
<br><button data-w="${w}">use this</button>`}
function draw(){const f=st.value,q=document.getElementById('q').value.toLowerCase();let h='',n=0;
for(const r of R){if(f&&r.dev_year_status!==f)continue;if(q&&!(r.list_name+' '+r.company).toLowerCase().includes(q))continue;n++;const d=D[r.slug_key];
h+=`<tr data-s="${r.slug_key}" class="${d?'ok':''}"><td><a href="${esc(r.url)}" target=evtolnews>${esc(r.list_name)}</a><div class=m>${esc(r.company)} · ${esc(r.status)} · <b>${esc(r.evtol_class)}</b></div></td>
<td class=y>${d?esc(d.year)+' ✎':(r.dev_year||'—')}<div class=m>${esc(r.dev_year_status)}</div></td><td class=q>${cell(r,'a')}</td><td class=q>${cell(r,'b')}</td>
<td><input data-y size=6 placeholder="year" value="${esc(d?.year||'')}"><br><button data-none>not stated</button></td></tr>`}
b.innerHTML=h;nn.textContent=n+' aircraft'}
function save(){try{localStorage.setItem(K,JSON.stringify(D))}catch(e){}draw()}
b.onclick=e=>{const tr=e.target.closest('tr');if(!tr)return;const r=R.find(x=>x.slug_key===tr.dataset.s);
if(e.target.dataset.w){const w=e.target.dataset.w;D[r.slug_key]={year:r['year_'+w],evidence_id:r['id_'+w]};save()}
if('none' in e.target.dataset){D[r.slug_key]={year:'',evidence_id:''};save()}};
b.onchange=e=>{if(!('y' in e.target.dataset))return;const s=e.target.closest('tr').dataset.s;D[s]={year:e.target.value.trim(),evidence_id:''};save()};
st.onchange=draw;document.getElementById('q').oninput=draw;
exp.onclick=()=>{const rows=[['slug_key','year','evidence_id']];for(const[k,v]of Object.entries(D))rows.push([k,v.year,v.evidence_id]);
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.map(r=>r.map(x=>'"'+String(x??'').replace(/"/g,'""')+'"').join(',')).join('\\n')],{type:'text/csv'}));a.download='DEV_YEAR_DECISIONS.csv';a.click()};
draw();"""
    ev = pd.read_csv(path.parent / "readers" / "evidence_ids.csv", dtype=str, keep_default_na=False)
    used = set(res.id_a) | set(res.id_b)
    evm = {r.evidence_id: {"s": r.sentence, "l": r.link} for r in ev.itertuples() if r.evidence_id in used}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>evtol.news development year</title>{style}</head><body>
<header><b>evtol.news: development-start year</b><select id=st><option value=conflict>conflicts</option><option value="">all</option>
<option value=agreed>agreed</option><option value=not_stated>not stated</option><option value=author>author</option></select>
<input id=q placeholder="search name / company" size=28><span id=nn class=m></span><button id=exp>export decisions (DEV_YEAR_DECISIONS.csv)</button>
<span class=m>two independent readers picked the earliest dated record of each aircraft's development; click a sentence to see it highlighted on the page</span></header>
<table><thead><tr><th>aircraft</th><th>year</th><th>reader a</th><th>reader b</th><th>your ruling</th></tr></thead><tbody id=b></tbody></table>
<script>{js.replace("__DATA__", json.dumps(data, ensure_ascii=False)).replace("__EV__", json.dumps(evm, ensure_ascii=False)).replace("</", "<\\/")}</script></body></html>"""
    path.write_text(html)



def write_review(yd):
    """Author review page over ALL aircraft: the readers' result is only the pre-fill."""
    res = pd.read_csv(yd / "aircraft_dev_year.csv", dtype=str, keep_default_na=False)
    ev = pd.read_csv(yd / "readers" / "evidence_ids.csv", dtype=str, keep_default_na=False)
    by_page = {k: g for k, g in ev.groupby("slug_key")}
    agreed = res.dev_year_status.isin(["agreed"])
    data = []
    for r in res.itertuples():
        g = by_page.get(r.slug_key)
        sents = [] if g is None else [[e.evidence_id, e.years, e.is_resource == "True", e.sentence] for e in g.itertuples()]
        prop_status = r.dev_year_status if r.dev_year_status in ("agreed", "conflict", "not_stated") else \
            ("agreed" if r.year_a == r.year_b and r.year_a else "not_stated" if r.year_a == r.year_b else "conflict")
        data.append(dict(s=r.slug_key, n=r.list_name, co=r.company, st=r.status, c=r.evtol_class, u=r.url,
                         ps=prop_status, py=r.year_a if prop_status == "agreed" else "",
                         pid=min([i for i in (r.id_a, r.id_b) if i]) if prop_status == "agreed" and (r.id_a or r.id_b) else "",
                         ya=r.year_a, yb=r.year_b, ia=r.id_a, ib=r.id_b, na=r.note_a, nb=r.note_b, e=sents))
    tpl = (Path(__file__).parent / "evtolnews_year_review_template.html").read_text()
    out = yd / "year_review.html"
    out.write_text(tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")))
    print(f"review page: {len(data)} aircraft ({int(agreed.sum())} with an agreed proposal) -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["packets", "merge", "review"])
    ap.add_argument("--per", type=int, default=80)
    a = ap.parse_args()
    if a.step == "packets":
        build_packets(a.per)
    elif a.step == "merge":
        merge()
    else:
        write_review(DS / "0_source" / "years")


if __name__ == "__main__":
    main()
