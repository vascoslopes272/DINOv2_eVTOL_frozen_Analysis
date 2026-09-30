"""Crop ablation on the chosen model (review flag M5, 2026-09-30).

The crop test of ``src/photo_crop_experiment.py`` ran on the base model
(dinov2-large). This runner repeats it on the chapter's model
(``facebook/dinov2-with-registers-large``, tag ``dinov2-reg-large_518``) on the
SAME crops (``crop_experiment/processed/518``, already made) and the SAME 1 157
aircraft, touching nothing of the base run:

    extract    registers extraction of the processed crops
               -> crop_experiment/embeddings/dinov2-reg-large_518/
    evaluate   original (registers photo run) vs cropped, every matrix: kNN-5
               balanced accuracy maker held out, probe at L24 CLS, paired
               bootstrap of the difference (1 000, seed 42); per-half sign
               consistency over the 200 maker splits of ``heldout_selection``

    /home/vasco/anaconda3/envs/Finetune/bin/python scripts/photo_crop_registers.py extract evaluate

Writes beside the base files, new names only::

    3_embedding_evaluation/crop_experiment/
        knn_original_vs_cropped_registers.csv
        knn_paired_difference_registers.csv
        halves_original_vs_cropped_registers.csv
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import evtolnews as E  # noqa: E402
from src import photo_crop_experiment as X  # noqa: E402
from src.config_loader import load_config  # noqa: E402
from src.registers_experiment import REG_TAG, reg_cfg  # noqa: E402

STEPS = ["extract", "evaluate", "report"]


def run_extract(cfg) -> None:
    # resolve transformers' lazy imports ONCE before the per-GPU threads: two shards
    # importing concurrently race the lazy module loader (seen again 2026-09-30)
    from transformers import AutoImageProcessor, AutoModel  # noqa: F401

    from src import embeddings as EM

    EM.run_extraction(reg_cfg(X.exp_cfg(cfg)), X.SIZE)


def run_evaluate(cfg) -> pd.DataFrame:
    from sklearn.metrics import balanced_accuracy_score

    P, EE = X._protocol_modules()
    H = importlib.import_module("evtol_visual_eval_src.heldout_selection")
    root, out = X.exp_root(cfg), X.eval_root(cfg)
    man = pd.read_csv(root / "crop_manifest.csv", keep_default_na=False)
    tag = REG_TAG.format(size=X.SIZE)
    oidx, oarr = X._matrices(E.root(cfg) / "2_embedding_extraction" / "embeddings" / tag, P)
    cidx, carr = X._matrices(root / "embeddings" / tag, P)

    y = man.topType.to_numpy(dtype=object)
    mk = man.company.map(EE.company_key)
    makers = np.where(mk == "", "solo: " + man.aircraft_uid, mk).astype(object)
    orows = [oidx[u] for u in man.figure_uid]
    crows = [cidx[u] for u in man.figure_uid]

    rows, preds = [], {}
    for key in P.MATRICES:
        for cond, M in (("original", oarr[key][orows]), ("cropped", carr[key][crows])):
            kn = P.knn(M, y, makers)
            preds[(key, cond)] = kn.pop("pred")
            rec = kn.pop("recall")
            pr = P.probe(M, y, makers) if key == (24, "cls") else {}
            rows.append({"model": tag, "matrix": P.mname(key), "layer": key[0], "pooling": key[1],
                         "condition": cond, **{f"knn_{k}": v for k, v in kn.items()},
                         "probe_bal_acc": pr.get("bal_acc", np.nan),
                         "probe_macro_f1": pr.get("macro_f1", np.nan),
                         **{f"recall_{c}": rec.get(c, np.nan) for c in X.CLASSES}})
            print(f"[evaluate] {P.mname(key)} {cond}: knn {kn['bal_acc']:.3f} "
                  f"[{kn['ci_lo']:.3f}, {kn['ci_hi']:.3f}]", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / "knn_original_vs_cropped_registers.csv", index=False)

    # paired bootstrap of the difference: the same resamples score both conditions
    # (same generator sequence as the base run's run_evaluate)
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
        drows.append({"model": tag, "matrix": P.mname(key), "layer": key[0], "pooling": key[1],
                      "diff_cropped_minus_original":
                          float(balanced_accuracy_score(y, pb) - balanced_accuracy_score(y, pa)),
                      "ci_lo": float(np.quantile(diffs, 0.025)),
                      "ci_hi": float(np.quantile(diffs, 0.975)),
                      "agree_frac": float((pa == pb).mean())})
    diff = pd.DataFrame(drows)

    # per-half sign consistency: the split-half scores of heldout_selection (200 maker splits x 2 halves)
    hrows = []
    for key in P.MATRICES:
        sc = H.half_scores({"original": oarr[key][orows], "cropped": carr[key][crows]},
                           {"photos 5-class": y}, makers)
        h = sc.pivot_table(index=["split", "half"], columns="candidate", values="score")
        d = h["cropped"] - h["original"]
        hrows.append({"matrix": P.mname(key), "halves": len(d), "mean_diff_cropped_minus_original": d.mean(),
                      "share_cropped_higher": float((d > 0).mean()),
                      "share_cropped_lower": float((d < 0).mean()),
                      "share_equal": float((d == 0).mean())})
    halves = pd.DataFrame(hrows)
    halves.to_csv(out / "halves_original_vs_cropped_registers.csv", index=False)
    diff = diff.merge(halves[["matrix", "share_cropped_higher", "share_cropped_lower"]], on="matrix")
    diff.to_csv(out / "knn_paired_difference_registers.csv", index=False)
    print(diff.to_string(index=False))
    return res


def run_report(cfg) -> Path:
    """RESULTS_registers.md from the CSVs (registers) and the base run's CSVs, numbers read, never typed."""
    out = X.eval_root(cfg)
    k = pd.read_csv(out / "knn_original_vs_cropped_registers.csv")
    d = pd.read_csv(out / "knn_paired_difference_registers.csv")
    kb = pd.read_csv(out / "knn_original_vs_cropped.csv")
    db = pd.read_csv(out / "knn_paired_difference.csv")
    n = f"{len(pd.read_csv(X.exp_root(cfg) / 'crop_manifest.csv')):,}".replace(",", " ")

    def cell(df, m, c):
        x = df[(df.matrix == m) & (df.condition == c)].iloc[0]
        return x, f"{x.knn_bal_acc:.3f} [{x.knn_ci_lo:.3f}, {x.knn_ci_hi:.3f}]"

    lines = ["| representation | original kNN [95 % CI] | cropped kNN [95 % CI] | paired diff (crop - orig) [95 % CI] "
             "| halves where cropped is higher | base model paired diff [95 % CI] |",
             "|---|---|---|---|---|---|"]
    for r in d.itertuples():
        _, o = cell(k, r.matrix, "original")
        _, c = cell(k, r.matrix, "cropped")
        b = db[db.matrix == r.matrix].iloc[0]
        name = f"**{r.matrix}**" if r.matrix == "L24 CLS" else r.matrix
        lines.append(f"| {name} | {o} | {c} | {r.diff_cropped_minus_original:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] "
                     f"| {r.share_cropped_higher:.0%} | {b.diff_cropped_minus_original:+.3f} [{b.ci_lo:+.3f}, {b.ci_hi:+.3f}] |")
    p = d[d.matrix == "L24 CLS"].iloc[0]
    pb = db[db.matrix == "L24 CLS"].iloc[0]
    po, _ = cell(k, "L24 CLS", "original")
    pc, _ = cell(k, "L24 CLS", "cropped")
    bo, _ = cell(kb, "L24 CLS", "original")
    bc, _ = cell(kb, "L24 CLS", "cropped")
    inside = p.ci_lo <= 0 <= p.ci_hi
    base_inside = pb.ci_lo <= 0 <= pb.ci_hi
    same = inside == base_inside
    clear = d[(d.ci_lo > 0) | (d.ci_hi < 0)].matrix.tolist()
    verdict = (("Background removal does not change the architecture-class signal of the chosen model at L24 CLS: "
                f"the paired difference ({p.diff_cropped_minus_original:+.3f}) has a 95 % interval that includes zero. ")
               if inside else
               (f"On the chosen model, cropping moves L24 CLS by {p.diff_cropped_minus_original:+.3f} "
                f"[{p.ci_lo:+.3f}, {p.ci_hi:+.3f}]: the interval excludes zero. "))
    verdict += ("The verdict is the same as on the base model." if same
                else "The verdict differs from the base model's.")
    verdict += (f" Matrices whose interval excludes zero: {', '.join(clear)}." if clear
                else " No matrix has an interval that excludes zero.")
    md = f"""# Crop ablation on the chosen model (registers)

2026-09-30, review flag M5. Code: `eVTOL-Embedding-Extraction/scripts/photo_crop_registers.py`
(`extract evaluate report`, Finetune env). The base-model run and its files (RESULTS.md,
knn_original_vs_cropped.csv, knn_paired_difference.csv) are unchanged.

**What changed from RESULTS.md.** Only the model: `facebook/dinov2-with-registers-large`
(tag `dinov2-reg-large_518`). Same {n} aircraft (one hero image each), same OWLv2 crops
(`crop_experiment/processed/518`), same labels, makers and protocol: kNN-5 balanced accuracy, maker
held out, 1 000 bootstrap resamples (seed 42); the paired bootstrap scores both conditions on the same
resamples. The originals are the registers photo run (`EVTOLNEWS_DS/2_embedding_extraction/embeddings/dinov2-reg-large_518`).
"Halves where cropped is higher": share of the 400 halves (200 maker splits, seed 42, split-half rule of
`heldout_selection`) where cropped scores above original, neighbours within the half.

## Numbers (5 classes, maker held out, n = {n})

{chr(10).join(lines)}

Linear probe, L24 CLS, folds grouped by maker: original {po.probe_bal_acc:.3f}, cropped {pc.probe_bal_acc:.3f}
balanced accuracy (base model: {bo.probe_bal_acc:.3f}, {bc.probe_bal_acc:.3f}). kNN predictions agree on
{p.agree_frac:.1%} of aircraft at L24 CLS.

## Verdict

{verdict}

## Files

- `EVTOLNEWS_DS/2_embedding_extraction/crop_experiment/embeddings/dinov2-reg-large_518/`
- `EVTOLNEWS_DS/3_embedding_evaluation/crop_experiment/`: `knn_original_vs_cropped_registers.csv`,
  `knn_paired_difference_registers.csv`, `halves_original_vs_cropped_registers.csv`, this file
"""
    dst = out / "RESULTS_registers.md"
    dst.write_text(md, encoding="utf-8")
    print(f"-> {dst}")
    return dst


def main(argv: list[str]) -> None:
    cfg = load_config()
    for s in argv or STEPS:
        print(f"==== {s}", flush=True)
        if s not in STEPS:
            raise SystemExit(f"unknown step {s!r}; steps: {STEPS}")
        {"extract": run_extract, "evaluate": run_evaluate, "report": run_report}[s](cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
