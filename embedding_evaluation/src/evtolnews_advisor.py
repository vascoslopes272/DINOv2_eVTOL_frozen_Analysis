"""Advisor report on the evtol.news photo set: markdown + PDF, same layout as the supervisor report.

Structure follows ``docs/Supervisor_Report_Taxonomy_Structure_Analysis_summary.md`` (QC, structure
vs random, unsupervised clustering, alignment with the labels, next steps), run on the evtol.news
photographs and set against the directory's five classes, the patent figures and the codebook.
Every number is read from the evaluation tables; every figure comes from
``evtolnews_figures.py`` and states its source (model, image size, layer, pooling, set).

Written to ``<evtolnews.root>/3_embedding_evaluation/advisor_report/`` (private data tree, because
the document reproduces copyrighted directory images). Render::

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/evtolnews_advisor_report.py
"""

from __future__ import annotations

import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import evtolnews_eval as ev
from . import evtolnews_figures as FG

SUPERVISOR_MD = "docs/Supervisor_Report_Taxonomy_Structure_Analysis_summary.md"
EXTRA_CSS = """
figure.wide img { max-width: 100%; }
figure.half img { max-width: 62%; }
figure.full img { max-width: 100%; max-height: 8.6in; }
figure.small img { max-width: 50%; }
figure.w75 img { max-width: 75%; }
figcaption .src { display: block; font-style: italic; color: #7a7f86; margin-top: 0.15em; }
.internal { display: block; margin: 0.4em 0 1em 0; padding: 0.45em 0.8em; border: 1px solid #c9ccd1;
            font-size: 0.82em; color: #3a3f44; background: #fafbfc; }
"""


def _md(df: pd.DataFrame, fmt: Dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---:" if df[c].dtype.kind in "fi" else "---"
                                                         for c in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else fmt.get(c, "{:.3f}").format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _n(v: int) -> str:
    return f"{int(v):,}".replace(",", " ")


class Doc:
    def __init__(self, figs: Dict[str, Dict[str, str]]):
        self.figs, self.lines, self.k, self.t = figs, [], 0, 0

    def add(self, *lines: str) -> None:
        self.lines.extend(lines)

    def fig(self, key: str, width: str = "wide") -> None:
        self.k += 1
        f = self.figs[key]
        self.add(f'<figure class="{width}"><img src="{f["file"]}"><figcaption><strong>Figure {self.k}.</strong> '
                 f'{f["title"]}.<span class="src">Source: {f["source"]}.</span>'
                 + (f'<span class="src"><strong>How to read:</strong> {f["read"]}.</span>' if f.get("read") else "")
                 + '</figcaption></figure>', "")

    def table(self, title: str, df: pd.DataFrame, fmt: Dict[str, str] | None = None) -> None:
        self.t += 1
        self.add(f"**Table {self.t}.** {title}", "", _md(df, fmt), "")

    def badge(self, kind: str, text: str) -> None:
        self.add(f'<span class="stage-status {kind}">{text}</span>', "", "---", "")

    def purpose(self, text: str) -> None:
        self.add(f'<div class="stage-purpose"><strong>What this stage checks:</strong> {text}</div>', "")

    def prov(self, text: str) -> None:
        self.add(f'<span class="provenance">{text}</span>', "")


def _style_block(repo: Path) -> str:
    md = (repo / SUPERVISOR_MD).read_text(encoding="utf-8")
    m = re.search(r"<style>(.*?)</style>", md, flags=re.S)
    return "<style>" + (m.group(1) if m else "") + EXTRA_CSS + "</style>"


def write(cfg: Dict[str, Any]) -> Path:
    F, D, extra = FG.build_all(cfg)
    out = ev.out_dir(cfg) / "advisor_report"
    repo = Path(cfg["folder_root"]).parent
    tag, L, pool = FG.REF
    im, ac = D.images, D.aircraft
    kept = im[im.keep.astype(str) == "True"]
    n_ac_img = kept.slug_key.nunique()
    ph, pa = D.frame("photo", tag)["df"], D.frame("patent", tag)["df"]
    cm = D.cm
    cls_main = cm[(cm.label == "class") & (cm.set == "main")]
    best = lambda src_, tg=tag: cls_main[(cls_main.source == src_) & (cls_main.embedding == tg)] \
        .set_index(["layer", "pooling"]).loc[(L, pool)]
    bp, bt = best("photo"), best("patent")
    stat = cm[(cm.label == "status") & (cm.set == "main") & (cm.embedding == tag)].set_index(["layer", "pooling"]).loc[(L, pool)]
    conf = ev.confusion_tables(cfg)
    m = D.match
    hon = m[(m.images == "no line drawings") & (m.embedding == tag) & (m.layer == L) & (m.pooling == pool)].set_index("direction")
    allimg = m[(m.images == "all images") & (m.embedding == tag) & (m.layer == L) & (m.pooling == pool)].set_index("direction")
    p = D.parent
    n_agree = int(p.agrees.astype(str).eq("True").sum())
    n_style = int((kept["style"] == "drawing").sum())

    d = Doc(F.items)
    d.add(_style_block(repo), "",
          "# External Validation with the evtol.news Directory: Progress Report", "",
          "**Prepared for:** Thesis supervisor  ",
          f"**Model:** `facebook/dinov2-large`, frozen; layers {{18, 22, 24}} × pooling {{CLS token, mean of the "
          f"patch tokens}}; extraction notebook 22 at 224 px and 518 px  ",
          "**Images:** photographs and renders of real and concept eVTOL aircraft from the Vertical Flight "
          "Society's World eVTOL Aircraft Directory (evtol.news), set against the patent figures of "
          "`1639_LABELLED`  ",
          f"**Date:** {date.today().isoformat()}", "",
          '<span class="internal"><strong>Internal document.</strong> The example, attention and matched-pair figures reproduce images from the '
          "evtol.news directory, © their owners, for review only. They are not to be copied or published "
          "without asking the source. Every other figure shows numbers only or the author's patent figures."
          "</span>", "")
    d.prov(f"Data tree <code>{ev.root(cfg)}</code>. Code: <code>eVTOL-Embedding-Extraction/src/evtolnews.py</code>, "
           "<code>evtolnews_sets.py</code>; <code>eVTOL-Visual-Evaluation/embedding_evaluation/src/evtolnews_eval.py</code>, "
           "<code>evtolnews_figures.py</code>, <code>evtolnews_advisor.py</code>.")

    # ── summary ──────────────────────────────────────────────────────────
    d.add("## Summary", "",
          "The patent figures of the dataset are line drawings, and the embeddings of Stage 2 had so far "
          "been measured on them alone. That leaves open whether a weak architecture signal comes from "
          "the model or from the drawing medium. The evtol.news directory answers the question from the "
          "other side: it lists every known eVTOL design, sorts each one into one of five architecture "
          "classes chosen by an expert body, and shows it in photographs and renders. The same frozen "
          "model, the same layers and the same metrics were run on it.", "")
    summ = pd.DataFrame([
        {"Question": "Do the five directory classes show in the photo embeddings?",
         "Result": f"kNN-5 balanced accuracy {bp.knn5_bal_acc:.2f} ({bp.knn5_bal_acc_maker_out:.2f} with the maker held out), "
                   f"probe {bp.probe_bal_acc_maker_out:.2f}; chance 0.20",
         "Status": "PASSED, moderate"},
        {"Question": "Photos against patent figures, same classes",
         "Result": f"patents {bt.knn5_bal_acc:.2f} / {bt.knn5_bal_acc_maker_out:.2f} / {bt.probe_bal_acc_maker_out:.2f}; "
                   f"k-means follows the classes on photos (ARI {bp.kmeans_ari:.2f}) and hardly on patents ({bt.kmeans_ari:.2f})",
         "Status": "photos ahead"},
        {"Question": "What else the photo space encodes",
         "Result": f"maturity status (built / concept / unknown): kNN-5 {stat.knn5_bal_acc:.2f}, chance 0.33",
         "Status": "CAUTION"},
        {"Question": "Are the five classes the parents of the codebook's twelve?",
         "Result": f"{n_agree} / {len(p)} linked aircraft agree", "Status": "PASSED, with exceptions"},
        {"Question": "Can a patent aircraft be found among the photos?",
         "Result": f"right aircraft in the top 10 for {hon.loc['patent->photo', 'R@10']:.0%} "
                   f"(random {hon.loc['patent->photo', 'random_R@10']:.1%}), top 50 for {hon.loc['patent->photo', 'R@50']:.0%}",
         "Status": "CAUTION, modest"},
    ])
    d.add(_md(summ), "")
    d.add("**Reading guide.** Every figure prints its source along its lower edge and repeats it in the "
          "caption: which images, which extraction (notebook 22 and the image size), which layer, which "
          "pooling (the CLS token or the mean of the patch tokens) and which set. The reference matrix is "
          f"518 px, layer {L}, CLS token. It scores highest on the class test (Section 5) and on the "
          "matching (Section 7). The label-free ranking of Section 3 prefers other matrices, and that is "
          "reported where it applies.", "", "---", "")

    # ── 1. data ─────────────────────────────────────────────────────────
    d.add("## Section 1: The evtol.news Photo Set", "")
    d.purpose("what the directory holds, how its images were collected and filtered, and how its five "
              "classes relate to the codebook's twelve topTypes.")
    d.add("### 1.1 Source and collection", "",
          f"The directory index at `evtol.news/aircraft` lists **{_n(len(ac))} aircraft** in five lists, one "
          "per class. Every aircraft page gives the model, the maker, its location, a status (concept "
          "design, prototype, production model, technology demonstrator, defunct) and a specification list, "
          f"followed by one or more images. All pages were saved on 2026-09-17 with no fetch error, and "
          f"the **{_n(len(im))} images** on them were downloaded with the credit line of their page.", "")
    facts = pd.DataFrame([
        ("aircraft pages in the index", _n(len(ac))),
        ("images on those pages", _n(len(im))),
        ("images kept as whole-aircraft views", _n(len(kept))),
        ("aircraft with at least one kept image", _n(n_ac_img)),
        ("aircraft left without an image", _n(len(ac) - n_ac_img)),
        ("model, maker, location filled", f"{(ac.model != '').mean():.0%}, {(ac.company != '').mean():.0%}, "
                                         f"{(ac.location != '').mean():.0%}"),
        ("aircraft type line filled", f"{(ac.spec_aircraft_type != '').mean():.0%}"),
    ], columns=["Item", "Value"])
    d.table("What the directory crawl produced.", facts)
    d.fig("f01_pipeline")
    d.add("### 1.2 The label: five directory classes", "",
          "The directory sorts aircraft by what the thrusters do. **Vectored Thrust** uses any of its "
          "thrusters for both lift and cruise. **Lift + Cruise** has separate thrusters for each, without "
          "vectoring. **Wingless (Multicopter)** has no cruise thruster. **Electric Rotorcraft** relies on a "
          "rotor, as a helicopter or an autogyro does. **Hover Bikes / Personal Flying Devices** is a use "
          "class: a single person sits on a saddle or stands, and most of these are multicopters.", "",
          "The codebook groups its twelve topTypes in the same way (Vectored Thrust, Independent Thrust, "
          "Wingless, Other). The directory classes are therefore used as the parent level of the codebook, "
          "and each patent aircraft carries the parent of its human topType. Section 6 checks that "
          "assumption on the aircraft present in both sources.", "")
    d.fig("f02_taxonomy")
    d.fig("f04_classes")
    d.add(f"The two populations are balanced differently. Multicopters are {(ph.cls == 'WM').mean():.0%} of the "
          f"directory and {(pa.cls == 'WM').mean():.0%} of the patent aircraft, while Vectored Thrust is "
          f"{(ph.cls == 'VT').mean():.0%} and {(pa.cls == 'VT').mean():.0%}. Balanced accuracy is used throughout "
          "for this reason, and per-class numbers on the patent side rest on as few as "
          f"{int((pa.cls == 'ER').sum())} aircraft (Electric Rotorcraft).", "")
    d.add("### 1.3 Whole-aircraft filter", "",
          "A directory page mixes aircraft views with logos, cabin interiors, team photographs, component "
          "close-ups and diagrams, and some pages show a sibling or predecessor aircraft. Four rules act "
          "first: a failed download, an image under 200 px, a file already used by another page, and a file "
          "stored in another aircraft's media folder. SigLIP (ViT-SO400M-14-384) then scores every image "
          "against prompt groups for a whole aircraft (photograph, render or drawing) and for each kind of "
          "non-aircraft image. Images scoring at least 0.80 are kept, images under 0.35 are dropped, and "
          "the band between them was inspected on contact sheets and held back.", "")
    d.fig("f03_funnel")
    d.fig("f06_siglip")
    d.fig("f05_examples")
    d.add("Image credits, Figure 5: " + "; ".join(extra["example_credits"]) + ".", "")
    d.add(f"Two limits of the filter matter for what follows. The SigLIP style guess (photograph, render "
          f"or drawing) proved unreliable: it called nearly every image a render. The maturity status of "
          f"the directory replaces it as the confound in Section 5. And {_n(n_style)} kept images are line "
          "drawings, some of them the patent drawing itself (Section 7).", "")
    d.badge("info", f"DATA STAGE: {_n(n_ac_img)} AIRCRAFT, {_n(len(kept))} IMAGES")

    # ── 2. QC ───────────────────────────────────────────────────────────
    d.add("## Section 2: Embedding QC (Stage A)", "")
    d.purpose("that every embedding matrix is numerically sound, and how tightly the images crowd "
              "together, photos against patent figures.")
    qa = D.table("photo", tag, "A_qc_report.csv").assign(source="photos")
    qb = D.table("patent", tag, "A_qc_report.csv").assign(source="patents")
    q = pd.concat([qa, qb])
    q["matrix"] = [FG.mat_name(a, b) for a, b in zip(q.layer, q.pooling)]
    d.table(f"Integrity and cosine statistics, {FG.size_of(tag)} px, set main.",
            q[["source", "matrix", "cos_mean", "cos_std", "cos_p05", "cos_p95", "nan_count", "exact_duplicate_count",
               "near_dead_dims"]], {"cos_mean": "{:.3f}", "cos_std": "{:.3f}"})
    d.fig("f07_cosine")
    d.add("No matrix has a NaN, an infinite value or an all-zero row. The photo embeddings spread wider than "
          "the patent embeddings in every matrix, with a lower mean cosine: "
          f"{qa.set_index(['layer', 'pooling']).loc[(18, 'mean_patch'), 'cos_mean']:.2f} against "
          f"{qb.set_index(['layer', 'pooling']).loc[(18, 'mean_patch'), 'cos_mean']:.2f} at layer 18 patch mean. "
          "Line drawings on white paper look more alike to the model than photographs do.", "")
    d.badge("passed", "INTEGRITY QC: PASSED")

    # ── 3. structure ────────────────────────────────────────────────────
    d.add("## Section 3: Global Structure vs. Random Noise (Stage B)", "")
    d.purpose("whether the embeddings hold real structure, measured against a random matrix of the same shape.")
    ba = D.table("photo", tag, "B_structure_vs_random.csv").assign(source="photos")
    bb = D.table("patent", tag, "B_structure_vs_random.csv").assign(source="patents")
    b = pd.concat([ba, bb])
    b["matrix"] = [FG.mat_name(a, c) for a, c in zip(b.layer, b.pooling)]
    d.table(f"PCA structure and clustering tendency, {FG.size_of(tag)} px, set main. Hopkins 0.5 = uniform, "
            "values towards 1 = clumped.",
            b[["source", "matrix", "pc1_ratio_vs_random", "participation_ratio", "dims_for_90pct", "hopkins"]],
            {"pc1_ratio_vs_random": "{:.1f}", "participation_ratio": "{:.1f}", "hopkins": "{:.3f}"})
    d.fig("f08_pca")
    d.fig("f09_distance")
    d.add(f"Every matrix clears the 3× threshold by a wide margin: PC1 is {ba.pc1_ratio_vs_random.min():.0f} to "
          f"{ba.pc1_ratio_vs_random.max():.0f} times the random value on photos and {bb.pc1_ratio_vs_random.min():.0f} "
          f"to {bb.pc1_ratio_vs_random.max():.0f} times on patents. Hopkins sits between "
          f"{min(ba.hopkins.min(), bb.hopkins.min()):.2f} and {max(ba.hopkins.max(), bb.hopkins.max()):.2f}, "
          "a smooth continuum on both sources and no sharply separated islands, as on the patents before.", "")
    d.badge("passed", "REAL-VS-RANDOM TEST: PASSED ON BOTH SOURCES")

    # ── 4. clustering ───────────────────────────────────────────────────
    d.add("## Section 4: Unsupervised Clustering (Stage C)", "")
    d.purpose("whether groupings found without labels follow the architecture classes, on photos and on "
              "patent figures.")
    ca = D.table("photo", tag, "C_clustering_summary.csv").assign(source="photos")
    cb = D.table("patent", tag, "C_clustering_summary.csv").assign(source="patents")
    c = pd.concat([ca, cb])
    c["matrix"] = [FG.mat_name(a, e) for a, e in zip(c.layer, c.pooling)]
    d.table("Best label-free partition per matrix (k-means and ward swept over k = 2 to 10, HDBSCAN, "
            "30 bootstrap refits).",
            c[["source", "matrix", "pca_dims", "best_kmeans_k", "best_kmeans_silhouette", "bootstrap_ari",
               "hdbscan_clusters", "hdbscan_noise_frac"]],
            {"best_kmeans_silhouette": "{:.3f}", "bootstrap_ari": "{:.3f}", "hdbscan_noise_frac": "{:.2f}"})
    d.add("### 4.1 Hierarchical structure", "")
    d.fig("f10_dendrogram")
    from scipy.cluster.hierarchy import fcluster
    two = fcluster(D.plot_data("photo", tag)[f"linkage__L{L}_{pool}"], 2, criterion="maxclust")
    sp = pd.crosstab(two, ph.cls.to_numpy()).reindex(columns=ev.CLASSES, fill_value=0)
    sh = sp.div(sp.sum(axis=0), axis=1)
    wing = sh.loc[:, ["VT", "LC"]].sum(axis=1).idxmax()
    rot = [k for k in sh.index if k != wing][0]
    d.add(f"The first split of the dendrogram already follows architecture. One branch holds "
          f"{sh.loc[wing, 'LC']:.0%} of the Lift + Cruise and {sh.loc[wing, 'VT']:.0%} of the Vectored Thrust "
          f"aircraft; the other holds {sh.loc[rot, 'ER']:.0%} of the rotorcraft, {sh.loc[rot, 'WM']:.0%} of the "
          f"multicopters and {sh.loc[rot, 'HB']:.0%} of the hover bikes, with the remaining winged aircraft.", "")
    d.table("Share of each class's aircraft in the two branches of the first dendrogram split.",
            sh.rename(index={wing: "branch with the winged aircraft", rot: "other branch"}).reset_index(names="branch"),
            {c: "{:.0%}" for c in ev.CLASSES})
    d.add("### 4.2 Label-free clusters and the classes", "")
    d.fig("f11_umap_clusters")
    cs = extra["cluster_share"]
    parts = []
    for cl in cs.index:
        big = [f"{cs.loc[cl, c]:.0%} of {FG.CLASS_SHORT[c]}" for c in ev.CLASSES if cs.loc[cl, c] >= 0.5]
        parts.append(f"{cl} holds " + (", ".join(big) if big else "no majority of any class"))
    d.add(f"The best k-means partition of the reference matrix has {len(cs)} clusters, found without any "
          "label: " + "; ".join(parts) + ". The table below gives every share.", "")
    d.table("Share of each class's aircraft in each label-free cluster (columns sum to 100 %).",
            cs.reset_index(), {c: "{:.0%}" for c in ev.CLASSES})
    d.add("### 4.3 The photo space coloured by class", "")
    d.fig("f12_umap_class")
    d.fig("f13_umap_facets")
    d.fig("f14_umap_matrices")
    d.add("Classes appear as regions and not as islands, in line with Hopkins. The deeper layers separate "
          "them better than layer 18, and the CLS token better than the patch mean at layers 22 and 24.", "")
    d.add("### 4.4 Photos against patent figures", "")
    d.fig("f15_umap_photo_patent")
    d.fig("f16_umap_joint")
    d.add("On patent figures the same model spreads Vectored Thrust over the whole map, and the rarer "
          "classes sit at its edges. When both sources are embedded together they occupy almost separate "
          "regions: the model sees the medium (line drawing or photograph) before the aircraft. This gap "
          "is what the matching of Section 7 has to cross.", "")
    d.add("### 4.5 Cluster to class alignment", "")
    d.fig("f17_contingency")
    ii = extra["intra_inter"]
    d.add("### 4.6 Intra- against inter-class distance", "")
    d.fig("f18_intra_inter")
    order = [FG.mat_name(a, e) for a, e in FG.MATRICES]
    iit = pd.DataFrame({"matrix": order})
    for col, name in (("raw", "all pairs"), ("balanced", "class-balanced")):
        w = ii.pivot(index="matrix", columns="source", values=col).reindex(order)
        iit[f"photos, {name}"] = w["photo"].to_numpy()
        iit[f"patents, {name}"] = w["patent"].to_numpy()
    d.table("Mean inter-class over mean intra-class cosine distance (1.0 = no class structure). "
            "All pairs: every aircraft pair counts once, so frequent classes dominate. Class-balanced: "
            "each class block counts once.", iit, {c: "{:.3f}" for c in iit.columns if c != "matrix"})
    nb = int((iit["photos, class-balanced"] > iit["patents, class-balanced"]).sum())
    d.add(f"Different classes sit further apart than the same class on both sources. Measured on average "
          f"distances, the photos do not separate the classes better: the class-balanced margin is larger "
          f"on photos in only {nb} of 6 matrices. On patent figures the rarer classes (rotorcraft, hover "
          "bikes, multicopters) sit far from the winged mass, which widens the average margin, while "
          "Vectored Thrust and Lift + Cruise overlap inside it. The photo advantage is therefore local: it "
          "shows in the nearest neighbours (Section 5) and in the clusters (Figure 17, ARI "
          f"{bp.kmeans_ari:.2f} against {bt.kmeans_ari:.2f}), and not in the mean distances.", "")
    acf = extra["attention_collapse"].set_index(["source", "layer"])
    d.add("### 4.7 Where the model looks", "",
          "The CLS token builds the embedding. Its attention over the image patches is rebuilt for layers 18, "
          "22 and 24 from each block's own query and key weights and averaged over the 16 heads.", "")
    d.fig("f27_attention_classes")
    d.fig("f28_attention_pairs")
    d.fig("f29_attention_collapse")
    d.add(f"At layer 18 the attention follows the aircraft in both media, and on patent figures also the "
          f"reference numbers and the figure caption. From layer 22 on, one cell takes about "
          f"{acf.loc[('photo', 24), 'top_cell_share']:.0%} of the attention and lies on the white padding in "
          f"{acf.loc[('photo', 24), 'top_cell_on_padding']:.0%} of photos and {acf.loc[('patent', 24), 'top_cell_on_padding']:.0%} "
          "of patent figures, the high-norm token artifact of DINOv2 without registers.", "")
    d.badge("passed", "CLUSTERS FOLLOW THE CLASSES ON PHOTOS, NOT ON PATENTS; MEAN MARGINS SIMILAR")

    # ── 5. supervised ───────────────────────────────────────────────────
    d.add("## Section 5: Do the Classes Separate? (Label-Based Tests)", "")
    d.purpose("how well the five classes can be read from the frozen embeddings, with every aircraft and "
              "then its whole maker kept out of its own prediction.")
    d.add("Three tests run on every matrix. **kNN-5** predicts an aircraft's class from its five nearest "
          "neighbours, never from its own images; the harder version also excludes every aircraft of the "
          "same maker, so a company's house style cannot carry the answer. A **logistic probe** is trained "
          "on folds grouped by maker. **Shuffled labels** give the chance level: its 95th percentile is "
          f"{bp.knn5_perm_p95:.2f} against a nominal 0.20.", "")
    d.fig("f19_knn_probe")
    tb = cls_main[cls_main.embedding == tag].copy()
    tb["matrix"] = [FG.mat_name(a, e) for a, e in zip(tb.layer, tb.pooling)]
    wide = pd.DataFrame({"matrix": [FG.mat_name(a, e) for a, e in FG.MATRICES]})
    for source, lab in (("photo", "photos"), ("patent", "patents")):
        g = tb[tb.source == source].set_index("matrix").reindex(wide.matrix)
        wide[f"{lab} kNN-5"] = g.knn5_bal_acc.to_numpy()
        wide[f"{lab} kNN-5, maker out"] = g.knn5_bal_acc_maker_out.to_numpy()
        wide[f"{lab} probe"] = g.probe_bal_acc_maker_out.to_numpy()
    d.table(f"Five-class balanced accuracy, {FG.size_of(tag)} px, set main (chance 0.20). The 224 px run and "
            "the silhouette, ARI and NMI of every run are in class_metrics.csv.", wide,
            {c: "{:.2f}" for c in wide.columns if c != "matrix"})
    b224 = cls_main[(cls_main.embedding == "dinov2-large_224") & (cls_main.source == "photo")]
    b224 = b224.sort_values("knn5_bal_acc_maker_out").iloc[-1]
    d.add(f"At 224 px the best photo matrix is {FG.mat_name(b224.layer, b224.pooling)} with kNN-5 "
          f"{b224.knn5_bal_acc:.2f} ({b224.knn5_bal_acc_maker_out:.2f} maker out): the larger input helps "
          "a little and changes no conclusion.", "")
    d.fig("f20_recall")
    d.fig("f21_confusion")
    rc = pd.DataFrame({"class": ev.CLASSES,
                       "photos: recall": conf["photo"].reindex(ev.CLASSES).recall.to_numpy(),
                       "photos: n": conf["photo"].reindex(ev.CLASSES).n.to_numpy(),
                       "patents: recall": conf["patent"].reindex(ev.CLASSES).recall.to_numpy(),
                       "patents: n": conf["patent"].reindex(ev.CLASSES).n.to_numpy()})
    d.table("Recall per class, reference matrix, kNN-5 with the maker held out.", rc,
            {"photos: recall": "{:.2f}", "patents: recall": "{:.2f}"})
    d.add("Multicopters are recognised best on photos, and hover bikes worst, where most errors go to the "
          "multicopters they resemble. Lift + Cruise is taken for Vectored Thrust most often: both carry "
          "wings and several rotors, and the difference, whether a rotor tilts, is rarely visible in one "
          "still image. On patent figures Vectored Thrust dominates the predictions, which lifts its recall "
          "and lowers every other class.", "")
    d.fig("f22_maturity")
    d.add(f"The photo space also separates built aircraft from concept renders (kNN-5 {stat.knn5_bal_acc:.2f} "
          "on three statuses, chance 0.33). Part of the structure is the medium and the finish of the image, "
          "the same kind of confound the patent study found in drawing perspective and style.", "")
    d.badge("passed", "FIVE CLASSES READABLE FROM PHOTOS: PASSED (MODERATE)")

    # ── 6. parents ──────────────────────────────────────────────────────
    d.add("## Section 6: Directory Classes against the Codebook's Twelve Types", "")
    d.purpose("whether the directory's five classes really are the parents of the codebook's twelve topTypes, "
              "on the patent aircraft that also have a directory page.")
    d.add(f"Named patent aircraft were linked to directory pages by the author's own naming rulings, or by "
          f"name and maker. {len(p)} aircraft of {p.family.nunique()} families have at least one page. For each, the parent "
          "implied by the human topType is compared with the class the directory gives.", "")
    d.fig("f23_parent", "half")
    dis = p[p.agrees.astype(str) != "True"][["aircraft_uid", "family", "topType", "parent_of_topType",
                                             "directory_classes"]].copy()
    kind = {"ER": "unusual rotor filed as rotorcraft", "VT": "family listed as Vectored Thrust",
            "LC": "family listed as Lift + Cruise"}
    dis["pattern"] = [kind.get(str(x).split("|")[0], "other") if tt != "HB" else "hover bike listed as VT"
                      for x, tt in zip(dis.directory_classes, dis.topType)]
    d.table("The disagreements.", dis.sort_values(["pattern", "family"]))
    er = dis[dis.pattern == kind["ER"]]
    slc = dis[(dis.topType == "SLC") & (dis.pattern == kind["VT"])]
    fam_list = lambda g: ", ".join(f"{f}" + (f" ×{n}" if n > 1 else "") for f, n in g.family.value_counts().items())
    d.add(f"{n_agree} of {len(p)} agree, and the {len(dis)} disagreements fall into clear groups. The directory "
          f"files unusual rotor systems as Electric Rotorcraft ({len(er)}: {fam_list(er)}), among them the "
          f"slowed rotor the codebook calls SRW. {len(slc)} aircraft labelled SLC belong to families the "
          f"directory lists as Vectored Thrust ({fam_list(slc)}). The likely cause is that the patent shows "
          "another configuration than the aircraft on the page; this has not been checked yet.", "")
    d.badge("passed", f"PARENT RELATION HOLDS FOR {n_agree} / {len(p)}; SRW AND UNUSUAL ROTORS DIFFER")

    # ── 7. matching ─────────────────────────────────────────────────────
    d.add("## Section 7: Same-Aircraft Matching (Patent ↔ Photo)", "")
    d.purpose("whether a patent figure finds the photographs of its own aircraft, and the reverse, across "
              "the drawing-to-photo gap of Section 4.4.")
    d.add("Each named patent aircraft queries the directory with its main figure. Every directory aircraft is "
          "scored by its most similar kept image, and the rank of the right family is recorded. The reverse "
          "query starts from a page's main image and ranks the patent aircraft. A random ranking gives the "
          "baseline for each query, from the size of the gallery and the number of right answers in it.", "")
    d.fig("f26_pairs", "half")
    d.add("Image credits, Figure 26: " + "; ".join(extra["pair_credits"]) + ".", "")
    pk = extra["pair_pick"]
    cp = pk[pk.kind == "page shows a patent drawing"]
    gn = pk[pk.kind == "drawing -> photo / render"]
    d.add(f"Some directory pages reproduce the patent drawing itself ({', '.join(cp.family)}, cosine "
          f"{cp.cos.min():.2f} to {cp.cos.max():.2f}). Those make the task trivial, so every result is also given with the {_n(n_style)} "
          "directory line drawings removed from the gallery. That second number is the honest drawing-to-photo "
          "result. A few drawings escape the style score (a Cora three-view counts as a render), so it may "
          "still be slightly optimistic.", "")
    d.fig("f24_recall_at_k")
    d.fig("f25_matching_matrices")
    mt = m[(m.embedding == tag) & (m.layer == L) & (m.pooling == pool)][
        ["direction", "images", "queries", "gallery", "R@1", "R@10", "R@50", "random_R@10", "random_R@50",
         "median_rank"]]
    d.table(f"Matching at the reference matrix ({FG.size_of(tag)} px, layer {L}, CLS token).", mt,
            {"queries": "{:.0f}", "gallery": "{:.0f}", "median_rank": "{:.0f}"})
    d.add(f"With the drawings removed, the right aircraft is in the top 10 of {int(hon.loc['patent->photo', 'gallery'])} "
          f"for {hon.loc['patent->photo', 'R@10']:.0%} of the patent queries, about "
          f"{hon.loc['patent->photo', 'R@10'] / hon.loc['patent->photo', 'random_R@10']:.0f} times the random rate, "
          f"and in the top 50 for {hon.loc['patent->photo', 'R@50']:.0%}. The genuine matches in Figure 26 "
          f"({', '.join(gn.family)}, ranks {', '.join(str(r) for r in gn['rank'])}) show the model reaching across the medium when the drawing and the render "
          "share a clear silhouette. The miss shows a link to a page of the same maker that shows another "
          "aircraft. Matching can propose candidate names for patents; it cannot assign them.", "")
    d.badge("caution", "MATCHING: CLEARLY ABOVE CHANCE, TOO WEAK TO ASSIGN NAMES ALONE")

    # ── 8. next ─────────────────────────────────────────────────────────
    d.add("## Section 8: Progress and Next Steps", "",
          "**Established in this stage**", "",
          f"- A photo set of {_n(len(kept))} whole-aircraft images of {_n(n_ac_img)} directory aircraft, with "
          "the directory's own class, maker and status, built by the same pipeline as the patent figures.",
          "- The frozen DINOv2 embeddings read the five architecture classes from photographs at "
          f"{bp.knn5_bal_acc:.2f} balanced accuracy ({bp.knn5_bal_acc_maker_out:.2f} across makers), above the "
          f"patent figures ({bt.knn5_bal_acc:.2f}) and with label-free clusters that follow the classes.",
          "- The drawing medium, and not the model alone, limits what the patent embeddings show: photographs "
          "and drawings of the same classes occupy separate regions of one embedding space.",
          f"- The codebook's parent grouping agrees with an independent, industry-maintained classification "
          f"for {n_agree} of {len(p)} linked aircraft.", "",
          "**Open rulings for the author**", "",
          "- The eight SLC patents in families the directory lists as Vectored Thrust: patent configuration, "
          "wrong link, or label to revisit.",
          "- Where SRW belongs: Lift + Cruise by the codebook, rotorcraft by the directory.",
          "- Permission from the Vertical Flight Society before any directory image appears outside this "
          "internal document.", "",
          "**Next steps**", "",
          "- Label a sample of directory photos with the twelve topTypes, to test the fine level on photos.",
          "- Train the linear probe on photos and apply it to patent figures (the cross-domain test left "
          "out here).",
          "- Style-normalised or fine-tuned embeddings (the contrastive notebook) to close the gap between "
          "drawings and photographs, measured with the matching of Section 7.", "")
    md_path = out / "EVTOLNEWS_ADVISOR_REPORT.md"
    md_path.write_text("\n".join(d.lines), encoding="utf-8")
    return md_path


def render_pdf(cfg: Dict[str, Any], md_path: Path) -> Path:
    repo = Path(cfg["folder_root"]).parent
    subprocess.run(["/home/vasco/anaconda3/bin/python3", str(repo / "scripts/build_report_pdf.py"), str(md_path)],
                   check=True)
    return md_path.with_suffix(".pdf")


# ── brief: takeaways + method scheme + one figure per claim ─────────────────
def write_brief(cfg: Dict[str, Any]) -> Path:
    """The short version: five claims, the method scheme, one figure each (about 5 pages)."""
    F, D, extra = FG.build_all(cfg)
    out = ev.out_dir(cfg) / "advisor_report"
    repo = Path(cfg["folder_root"]).parent
    tag, L, pool = FG.REF
    kept = D.images[D.images.keep.astype(str) == "True"]
    cls_main = D.cm[(D.cm.label == "class") & (D.cm.set == "main") & (D.cm.embedding == tag)]
    at = lambda src_: cls_main[cls_main.source == src_].set_index(["layer", "pooling"]).loc[(L, pool)]
    bp, bt = at("photo"), at("patent")
    stat = D.cm[(D.cm.label == "status") & (D.cm.set == "main") & (D.cm.embedding == tag)] \
        .set_index(["layer", "pooling"]).loc[(L, pool)]
    conf = ev.confusion_tables(cfg)
    rp = conf["photo"]
    h = D.match[(D.match.images == "no line drawings") & (D.match.embedding == tag) & (D.match.layer == L)
                & (D.match.pooling == pool)].set_index("direction").loc["patent->photo"]
    p = D.parent
    n_agree = int(p.agrees.astype(str).eq("True").sum())
    lc_vt = int(rp.loc["LC", "VT"]); hb_wm = int(rp.loc["HB", "WM"])
    ac = extra["attention_collapse"].set_index(["source", "layer"])
    er = p[(p.agrees.astype(str) != "True") & p.directory_classes.str.contains("ER")]
    slc = p[(p.agrees.astype(str) != "True") & (p.topType == "SLC")]

    d = Doc(F.items)
    d.add(_style_block(repo), "",
          "# Real Aircraft Photographs as a Test of the Patent Embeddings", "",
          f"**Prepared for:** Thesis supervisor · **Date:** {date.today().isoformat()} · **Model:** DINOv2-large, "
          "frozen, the same extraction as the patent figures  ",
          f"**Data:** {_n(len(kept))} images of {_n(kept.slug_key.nunique())} aircraft from the evtol.news World eVTOL "
          "Aircraft Directory (Vertical Flight Society), against the patent figures of `1639_LABELLED`", "",
          '<span class="internal"><strong>Internal.</strong> Every figure prints its source: images, extraction '
          "(image size), layer, pooling (CLS token or patch mean) and set. The full report with every table is "
          "EVTOLNEWS_ADVISOR_REPORT.pdf.</span>", "",
          "## Takeaways", "",
          f"1. **Photographs of real eVTOL aircraft carry their architecture class.** The frozen model assigns "
          f"the right one of five classes with {bp.knn5_bal_acc:.2f} balanced accuracy, {bp.knn5_bal_acc_maker_out:.2f} "
          "when every aircraft of the same maker is hidden (chance 0.20).",
          f"2. **Patent drawings carry it less, because the model sees the medium first.** On patent figures the "
          f"same test gives {bt.knn5_bal_acc:.2f}; drawings and photographs fall into separate regions of one "
          "embedding space.",
          f"3. **The errors are what one still image cannot show.** Lift + Cruise is taken for Vectored Thrust "
          f"({lc_vt} of {int(rp.loc['LC', 'n'])}): whether a rotor tilts is rarely visible. Hover bikes are taken for "
          f"multicopters ({hb_wm} of {int(rp.loc['HB', 'n'])}), which most of them are.",
          f"4. **The directory's five classes are the parents of the codebook's twelve** for {n_agree} of "
          f"{len(p)} aircraft present in both sources.",
          f"5. **A patent figure can find its real aircraft, modestly.** The right aircraft is in the top 10 of "
          f"{int(h.gallery)} for {h['R@10']:.0%} of named patent aircraft (random {h['random_R@10']:.1%}).",
          f"6. **The model looks at the aircraft, and in drawings also at their labels.** At layer 18 the CLS "
          "attention sits on the propulsors, wings and fuselage in both media, and on patent figures also on "
          "the reference numbers and the figure caption. From layer 22 on it collapses onto a few background "
          f"tokens: the strongest cell holds {ac.loc[('photo', 24), 'top_cell_share']:.0%} of it and lies on the "
          f"white padding in {ac.loc[('photo', 24), 'top_cell_on_padding']:.0%} of photos.", "",
          )
    d.add("## Caution and open points", "",
          f"- The photo space also encodes maturity, a built prototype against a concept render (kNN-5 "
          f"{stat.knn5_bal_acc:.2f}, chance 0.33), and mean class distances are not larger on photos than on "
          "patents: the photo advantage is local (neighbours, clusters).",
          f"- Open: the {len(slc)} SLC patents in Vectored Thrust families; where SRW belongs.",
          "- The reference embedding (layer 24, CLS) draws partly on padding tokens, a known artifact of "
          "DINOv2 without registers. DINOv2 with registers, or padding with the image's border colour, is the "
          "direct test of how much this costs.",
          "- Next: label a sample of photos with the twelve topTypes; train on photos and test on patents; "
          "style-normalised embeddings to close the drawing-photo gap.",
          "- The directory images stay internal until the Vertical Flight Society is asked.", "")
    d.add('<div style="break-before: page;"></div>', "", "## Method in one scheme", "")
    d.fig("f00_method", "full")
    d.add('<div style="break-before: page;"></div>', "", "## Evidence", "")

    d.add("### 1. Photographs carry the architecture class", "")
    d.fig("f12_umap_class", "wide")
    d.add(f"Each class occupies its own region of the map. Balanced accuracy: kNN-5 {bp.knn5_bal_acc:.2f}, "
          f"{bp.knn5_bal_acc_maker_out:.2f} with the maker held out, logistic probe {bp.probe_bal_acc_maker_out:.2f}; "
          f"shuffled labels reach {bp.knn5_perm_p95:.2f} at most. Clusters found without labels follow the classes "
          f"(ARI {bp.kmeans_ari:.2f}).", "")

    d.add("### 2. Drawings carry it less: the model sees the medium first", "")
    d.fig("f16_umap_joint")
    d.add(f"Photos and patent drawings occupy almost separate regions, even for the same classes. On patents "
          f"kNN-5 falls to {bt.knn5_bal_acc:.2f} ({bt.knn5_bal_acc_maker_out:.2f} maker out) and the label-free "
          f"clusters stop following the classes (ARI {bt.kmeans_ari:.2f}). The weak architecture signal of the "
          "patent study comes from the drawing medium at least as much as from the model.", "")

    d.add("### 3. The errors follow what a still image hides", "")
    d.fig("f21_confusion")
    d.add("Rows are the true class. On photos the confusions sit where the directory's own definitions depend "
          "on motion or use: tilting rotors (Lift + Cruise against Vectored Thrust) and a single rider "
          "(hover bikes against multicopters). On patents nearly every prediction drifts to Vectored Thrust, "
          "the class that dominates the patent set.", "")

    d.add("### 4. The directory classes are the parents of the codebook's types", "")
    d.fig("f23_parent", "half")
    d.add(f"{n_agree} of {len(p)} agree. The directory files unusual rotors as rotorcraft ({len(er)}, among them "
          f"the codebook's SRW), and {len(slc)} SLC patents belong to families it lists as Vectored Thrust, "
          "probably because the patent shows another configuration (not checked yet).", "")

    d.add("### 5. Patent to photo matching works, but only as a suggestion", "")
    d.fig("f24_recall_at_k")
    d.add(f"Top 10: {h['R@10']:.0%} against {h['random_R@10']:.1%} at random; top 50: {h['R@50']:.0%} against "
          f"{h['random_R@50']:.1%}. The grey curve includes directory pages that reproduce the patent drawing "
          "itself, which makes matching trivial; the black curve excludes them and is the honest result.", "")

    d.add("### 6. The model looks at the aircraft, and in drawings also at their labels", "")
    d.fig("f29_attention_collapse", "half")
    d.add(f"Layer 18 spreads the attention (strongest cell {ac.loc[('photo', 18), 'top_cell_share']:.0%}). From "
          f"layer 22 on a single cell takes {ac.loc[('photo', 24), 'top_cell_share']:.0%} to "
          f"{ac.loc[('patent', 24), 'top_cell_share']:.0%} of it, on the white padding in "
          f"{ac.loc[('photo', 24), 'top_cell_on_padding']:.0%} of photos and {ac.loc[('patent', 24), 'top_cell_on_padding']:.0%} "
          "of patent figures. These are the high-norm tokens that DINOv2 without registers uses to store "
          "global information, so the maps below are taken at layer 18.", "")
    d.fig("f28_attention_pairs", "w75")
    d.add("At layer 18 the attention follows the propulsors, the wings and the fuselage in both media. On the "
          "patent figures it also lands on the reference numbers and on the figure caption (\"FIG. 1\"), "
          "text a photograph does not have. On the photographs part of it leaks to busy backgrounds.", "")

    path = out / "EVTOLNEWS_ADVISOR_BRIEF.md"
    path.write_text("\n".join(d.lines), encoding="utf-8")
    return path
