#!/usr/bin/env python3
"""The CPU-side review fixes of the DINOv2 analysis document (review of 2026-09-30).

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/review_fixes.py ITEM [ITEM ...] [--quick] [--out DIR]

ITEM: m3 m4 m6 m11 s2_12 s2_5 s2_ph s3 s4 s6 s78 s9 s10 s11 s11_umap figures index all
(``all`` = every number item; ``figures`` and ``index`` read the CSVs the items wrote).
--quick runs every item with small counts into a scratch folder (smoke test; never into review_fixes/).

Outputs: 1639_LABELLED/3_embedding_evaluation/review_fixes/ (CSVs, figs/*.png, INDEX.md). Code:
src/review_fixes.py. Nothing outside that folder is written.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import review_fixes as RF  # noqa: E402

ITEMS = {
    "m3": (RF.m3_pool, {"n_draw": 200, "n_boot": 1000}, {"n_draw": 10, "n_boot": 50}),
    "m4": (RF.m4_erasure, {"n_ctrl": 200, "n_split": 200}, {"n_ctrl": 3, "n_split": 3}),
    "m6": (RF.m6_per_class, {"n_draw": 200, "per_class": 20, "n_probe_draws": 50},
           {"n_draw": 5, "per_class": 20, "n_probe_draws": 1}),
    "m11": (RF.m11_counts, {}, {}),
    "s2_12": (lambda **kw: RF.s2_probe([RF.LAB12], **kw), {}, {"n_shuffle": 1, "fold_seeds": (0,), "n_boot": 50}),
    "s2_5": (lambda **kw: RF.s2_probe([RF.LAB5], **kw), {}, {"n_shuffle": 1, "fold_seeds": (0,), "n_boot": 50}),
    "s2_ph": (lambda **kw: RF.s2_probe([RF.PHO5], **kw), {}, {"n_shuffle": 1, "fold_seeds": (0,), "n_boot": 50}),
    "s3": (RF.s3_k_and_rules, {}, {}),
    "s4": (RF.s4_paired, {}, {}),
    "s6": (RF.s6_matching, {}, {"n_boot": 50}),
    "s78": (RF.s78_selection, {}, {"seeds": (42,)}),
    "s9": (RF.s9_attention, {}, {}),
    "s10": (RF.s10_audit_kappa, {}, {}),
    "s11": (RF.s11_view, {"n_boot": 500, "n_boot_aircraft": 1000}, {"n_boot": 2, "n_boot_aircraft": 5}),
    "s11_umap": (lambda **kw: RF.s11_view(umap_only=True), {}, {}),
}
SUFFIX = {"s2_12": "_12", "s2_5": "_5", "s2_ph": "_photos"}


def main(argv: list[str]) -> None:
    quick = "--quick" in argv
    out = RF.OUT
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    elif quick:
        out = Path("/tmp/claude-1006/-home-vasco-Vasco-Workspace-Tese-Vasco-Lnx-WSpaces/"
                   "2d3d33f7-3c1c-421a-a475-05015c55ac0b/scratchpad/review_fixes_quick")
    items = [a for a in argv if not a.startswith("--") and a != str(out)]
    if "all" in items:
        items = [k for k in ITEMS if k != "s11_umap"]
    for it in items:
        if it in ("figures", "index"):
            continue
        fn, full, small = ITEMS[it]
        t = time.time()
        print(f"[review_fixes] {it} ...", flush=True)
        tables = fn(**(small if quick else full))
        if it in SUFFIX:
            tables = {k + SUFFIX[it]: v for k, v in tables.items()}
        if it == "s11_umap":
            tables = {k: v for k, v in tables.items() if k == "S11_umap_figures_by_view"}
        out.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out / f"{name}.csv", index=False)
        print(f"[review_fixes] {it} done in {time.time() - t:.0f} s: {', '.join(tables)}", flush=True)
    if "figures" in items:
        for p in RF.make_figures(out):
            print(f"[review_fixes] figure {p}", flush=True)
    if "index" in items:
        print(f"[review_fixes] index {RF.write_index(out)}", flush=True)
    print(f"-> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
