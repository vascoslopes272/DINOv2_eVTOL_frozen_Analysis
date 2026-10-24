"""Future-work figure of the dissertation: DINOv2 embeddings of VFS directory images vs patent figures.

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/thesis_dino_figure.py [--out DIR]
    (notebook 33_thesis_dino_figure.ipynb drives it)

Reads the saved registers embeddings (tag dinov2-reg-large_518, written by eVTOL-Embedding-Extraction
notebook 25 / scripts/registers_experiment.py) through the protocol's own loaders, so every score is the
protocol's (embedding_protocol.py): one vector per aircraft (patents: exp3 main figure; photos: hero image),
layer 24 CLS, kNN-5 balanced accuracy with the maker held out. The patent G1 types are mapped to the
directory's five classes (embedding_reports.PARENT) so the two panels score the same classes.

Writes into OUT (default: the hand-over folder for LaTeX_v15):
    Figures/Future_work/dino_embedding_vfs_vs_patents.{pdf,png}
    Figures/Future_work/dino_embedding_vfs_vs_patents_points.csv     (every plotted point)
    Data_for_writing/csv/dino_scores.csv                              (the scores table of NUMBERS_DINO.md)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from . import embedding_protocol as P
from . import embedding_reports as ER

REG_TAG = "dinov2-reg-large_518"
KEY: P.Key = (24, "cls")
UMAP_KW = dict(n_neighbors=15, min_dist=0.1, metric="cosine", n_components=2, random_state=P.SEED)
N_SIL_SHUFFLE = 200

_TS_DIR = (Path(__file__).resolve().parents[3] / "eVTOL-Visual-Evaluation" / "labeling_evaluation" / "notebooks")
OUT_DEFAULT = Path(__file__).resolve().parents[3] / "COPY_TO_OTHER_PC_2026-10-24_DINO"
FIG_SUB = Path("Figures/Future_work")
NAME = "dino_embedding_vfs_vs_patents"

# printed names (the dissertation's terms) and the patent codes inside each directory class (TR prints TP, TB prints TF)
CLASS_PRINT = {"VT": "Vectored Thrust", "LC": "Separate Lift + Cruise", "WM": "Multicopter",
               "ER": "Electric Rotorcraft", "HB": "Hoverbike"}
CLASS_CODES = {"VT": "TP, TW, CVT, TF, PTC, DS", "LC": "SLC, SRW", "WM": "MR", "ER": "RC", "HB": "HB, PFV"}
PRINT_CODE = {"TR": "TP", "TB": "TF"}
VFS_KEY = {"VT": "Vectored Thrust", "LC": "Lift + Cruise", "WM": "Wingless (Multicopter)",
           "ER": "Electric Rotorcraft", "HB": "Hover Bikes / PFD"}       # keys of thesis_style.VFS
DRAW_ORDER = ["VT", "LC", "WM", "HB", "ER"]                               # largest first, small classes on top


def _ts():
    sys.path.insert(0, str(_TS_DIR))
    import thesis_style as ts
    return ts


def sources() -> List[Dict[str, Any]]:
    """[{key, X, y5, makers, table}] for the photos (hero image) and the patents (exp3 main figure)."""
    ER.TAG = REG_TAG
    out = []
    S = ER.load_evtolnews()
    t = S.aircraft[["aircraft_id", "maker", "title", "hero", "label"]].rename(
        columns={"hero": "image_id", "label": "directory_class"})
    out.append({"key": "vfs", "X": S.rules[S.primary_rule][KEY], "y5": S.y, "makers": S.makers, "table": t,
                "image_rule": "hero image (first kept image of the directory page)"})
    S = ER.load_patents()
    m = S.extra["picks"]["exp3 main figure"].groupby("aircraft_id")["fig_id"].first()
    t = S.aircraft[["aircraft_id", "maker", "label", "label5"]].rename(columns={"label": "g1_code",
                                                                              "label5": "directory_class"})
    t["g1_print"] = t.g1_code.map(lambda c: PRINT_CODE.get(c, c))
    t["image_id"] = t.aircraft_id.map(m)
    out.append({"key": "patents", "X": S.rules[S.primary_rule][KEY], "y5": S.aircraft["label5"].to_numpy(dtype=object),
                "y12": S.y, "makers": S.makers, "table": t,
                "image_rule": "exp3 main figure (the figure the labeller marked as main)"})
    return out


def silhouette(X: np.ndarray, y: np.ndarray, seed: int = P.SEED) -> Dict[str, float]:
    """Cosine silhouette of the class labels in the full 1 024-d space, against shuffled labels."""
    D = np.clip(1.0 - P.l2(X) @ P.l2(X).T, 0, None)
    np.fill_diagonal(D, 0.0)
    obs = silhouette_score(D, y, metric="precomputed")
    rng = np.random.default_rng(seed)
    sh = [silhouette_score(D, rng.permutation(y), metric="precomputed") for _ in range(N_SIL_SHUFFLE)]
    return {"silhouette": float(obs), "sil_shuffle_mean": float(np.mean(sh)), "sil_shuffle_p95": float(np.quantile(sh, 0.95))}


def score(X, y, makers, label_set: str, source: str) -> Dict[str, Any]:
    k = P.knn(X, y, makers)
    pr = P.probe(X, y, makers)
    counts = pd.Series(y).value_counts()
    return {"source": source, "label_set": label_set, "n_images": len(y), "n_classes": int(counts.size),
            "n_makers": int(pd.Series(makers).nunique()),
            "class_counts": "; ".join(f"{c} {counts.get(c, 0)}" for c in counts.index),
            "knn5_bal_acc": k["bal_acc"], "interval_lo": k["ci_lo"], "interval_hi": k["ci_hi"],
            "chance_1_over_k": 1.0 / counts.size, "chance_shuffled_mean": k["chance_mean"],
            "chance_shuffled_p95": k["chance_p95"], "probe_bal_acc": pr["bal_acc"],
            **silhouette(X, np.asarray(y)), "_pred": k["pred"]}


def umap2(X: np.ndarray) -> np.ndarray:
    import umap
    return umap.UMAP(**UMAP_KW).fit_transform(P.l2(X))


def draw(points: pd.DataFrame, rows: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ts = _ts()
    ts.use()
    fig, axs = plt.subplots(1, 2, figsize=(ts.TEXTWIDTH, 3.0), layout="constrained")
    lab = {"vfs": "(a) VFS eVTOL directory images", "patents": "(b) patent figures"}
    for ax, src in zip(axs, ("vfs", "patents")):
        d = points[points.source == src]
        for c in DRAW_ORDER:
            e = d[d.directory_class == c]
            ax.scatter(e.umap_1, e.umap_2, s=5, c=ts.VFS[VFS_KEY[c]], marker=ER.CLASS_MARK[c], linewidths=0,
                       alpha=0.8, rasterized=False)
        r = rows[src]
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP dimension 1 (no unit)"); ax.set_ylabel("UMAP dimension 2 (no unit)")
        n = f"{r['n_images']:,}".replace(",", " ")      # Nimbus Sans has no thin space
        ts.panel_label(ax, f"{lab[src]}, n = {n} aircraft"
                       + f"\nkNN-5 balanced accuracy {r['knn5_bal_acc']:.2f} [{r['interval_lo']:.2f}, {r['interval_hi']:.2f}]"
                       + f"\nchance {r['chance_1_over_k']:.2f}")
    h = [Line2D([], [], ls="none", marker=ER.CLASS_MARK[c], ms=4.5, mfc=ts.VFS[VFS_KEY[c]], mec="none",
                label=f"{CLASS_PRINT[c]} ({CLASS_CODES[c]})") for c in ["VT", "LC", "WM", "ER", "HB"]]
    fig.legend(handles=h, loc="outside lower center", ncol=3, handletextpad=0.2, columnspacing=1.0)
    ts.save(fig, NAME, out_dir)
    plt.close(fig)


def run(out: Path = OUT_DEFAULT) -> Dict[str, Any]:
    t0 = time.time()
    fig_dir, csv_dir = out / FIG_SUB, out / "Data_for_writing" / "csv"
    fig_dir.mkdir(parents=True, exist_ok=True); csv_dir.mkdir(parents=True, exist_ok=True)
    rows, pts = {}, []
    for s in sources():
        r = score(s["X"], s["y5"], s["makers"], "5 directory classes", s["key"])
        rows[s["key"]] = r
        t = s["table"].copy()
        t.insert(0, "source", s["key"])
        t["image_rule"] = s["image_rule"]
        t["knn5_prediction"] = r.pop("_pred")
        t["knn5_correct"] = t.knn5_prediction == t.directory_class
        e = umap2(s["X"])
        t["umap_1"], t["umap_2"] = e[:, 0], e[:, 1]
        pts.append(t)
        if s["key"] == "patents":
            r12 = score(s["X"], s["y12"], s["makers"], "12 G1 types (codebook)", "patents")
            r12.pop("_pred")
            rows["patents_12"] = r12
    points = pd.concat(pts, ignore_index=True)
    points["class_printed"] = points.directory_class.map(CLASS_PRINT)
    points.to_csv(fig_dir / f"{NAME}_points.csv", index=False)
    table = pd.DataFrame(rows.values())
    table.to_csv(csv_dir / "dino_scores.csv", index=False)
    draw(points, rows, fig_dir)
    meta = {"tag": REG_TAG, "matrix": P.mname(KEY), "umap": UMAP_KW, "k": P.K, "n_boot": P.N_BOOT,
            "n_shuffle": P.N_SHUFFLE, "n_sil_shuffle": N_SIL_SHUFFLE, "seed": P.SEED,
            "run_seconds": round(time.time() - t0, 1)}
    (csv_dir / "dino_scores_run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"table": table, "points": points, "meta": meta}
