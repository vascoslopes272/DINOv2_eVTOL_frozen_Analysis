"""Side-by-side CLS attention, base dinov2-large vs dinov2-with-registers (decision figure, N2).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/registers_attention_figure.py

Three patent figures and three photos (different classes), each shown with its L18 / L22 / L24
CLS-attention overlay under both models, from the arrays that registers_experiment.py wrote.
Output: 1639_LABELLED/3_embedding_evaluation/registers_check/attention_base_vs_registers.png

Moved from scripts/registers_attention_figure.py on 2026-09-30 (notebook 32 drives it); that script is now a thin
wrapper with the same command line.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


from .embedding_reports import EVN, PAT  # noqa: E402

ATT = EVN / "2_embedding_extraction/attention"
BASE_TAG, REG_TAG = "dinov2-large_518", "dinov2-reg-large_518"
LAYERS = (18, 22, 24)
OUT = PAT / "3_embedding_evaluation/registers_check/attention_base_vs_registers.png"

#: wanted classes per source, first matching aircraft in both runs is taken
WANT_PHOTO = ["VT", "WM", "LC"]
WANT_PATENT = ["TR", "MR", "SLC"]


def load(name: str, tag: str):
    d = ATT / tag
    return np.load(d / f"{name}_cls.npy").astype(np.float32), pd.read_csv(d / f"{name}_meta.csv")


def pick(name: str, want, cls_of) -> list:
    """[(figure_uid, path, class)] — one per wanted class, present in both runs."""
    _, mb = load(name, BASE_TAG)
    _, mr = load(name, REG_TAG)
    common = mb[mb.figure_uid.isin(set(mr.figure_uid))]
    rows = []
    for c in want:
        sub = common[common.aircraft_uid.map(cls_of).eq(c) & common.path.map(lambda p: Path(p).exists())]
        if len(sub):
            r = sub.iloc[0]
            rows.append((r.figure_uid, r.path, c))
    return rows


def overlay(ax, img: np.ndarray, att: np.ndarray | None) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    if att is not None:
        a = Image.fromarray(att).resize(img.shape[::-1], Image.BILINEAR)
        a = np.asarray(a, dtype=np.float32)
        hi = np.quantile(a, 0.99) or 1.0
        ax.imshow(np.clip(a / hi, 0, 1), cmap="magma", alpha=0.55, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    photo_cls = pd.read_csv(EVN / "2_embedding_extraction/selection/sets/main.csv",
                            keep_default_na=False).set_index("aircraft_uid")["topType"]
    pat_cls = pd.read_csv(PAT / "2_embedding_extraction/view_state_experiments/aircraft_common.csv",
                          dtype=str, keep_default_na=False).set_index("aircraft_id")["g1_code"]
    picks = pick("patent", WANT_PATENT, pat_cls) + pick("photo", WANT_PHOTO, photo_cls)

    fig, axes = plt.subplots(len(picks), 1 + 2 * len(LAYERS), figsize=(11.5, 1.75 * len(picks)))
    for i, (uid, path, c) in enumerate(picks):
        img = np.asarray(Image.open(path).convert("L"))
        axes[i, 0].set_ylabel(f"{c}\n{uid[:18]}", fontsize=6.5, rotation=0, ha="right", va="center")
        overlay(axes[i, 0], img, None)
        for m, tag in enumerate((BASE_TAG, REG_TAG)):
            arr, meta = load("patent" if i < len(WANT_PATENT) else "photo", tag)
            row = meta.index[meta.figure_uid == uid][0]
            for j, L in enumerate(LAYERS):
                ax = axes[i, 1 + m * len(LAYERS) + j]
                overlay(ax, img, arr[row, j])
                if i == 0:
                    ax.set_title(f"L{L}\n{'base' if m == 0 else 'registers'}", fontsize=7.5)
    axes[0, 0].set_title("image", fontsize=7.5)
    fig.suptitle("CLS attention: dinov2-large (base) vs dinov2-with-registers — same images, layers 18/22/24",
                 fontsize=9.5, fontweight="bold")
    fig.text(0.005, 0.004,
             "Source: attention/{dinov2-large_518, dinov2-reg-large_518} (registers_experiment.py), 518 px, "
             "CLS query vs every patch key, mean over heads; each map normalised to its own 99th percentile.\n"
             "How to read: bright = where the CLS token looks. Base L22/L24 collapse onto one padding cell "
             "(the isolated bright dot away from the aircraft); with registers the late layers stay on the "
             "aircraft. Share table: registers_share.csv.",
             fontsize=5.8, color="#6b6a66", ha="left", va="bottom")
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.06)
    print(OUT)



run = main
