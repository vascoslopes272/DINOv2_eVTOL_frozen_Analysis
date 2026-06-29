# Analysis guide — reading the frozen-DINOv2 eVTOL results

A single study reference for every metric in this workstream: what it means, how
it is computed (which function / which notebook cell), how to read it, and which
knob changes it. Code: [`src/analysis.py`](../src/analysis.py). Notebooks:
`10_qc_embeddings.ipynb` (Stage 0), `11_structure_separation.ipynb` (Stage 1).

---

## 0. Mental model — what we measure

DINOv2 turns each image into vectors. A 224×224 image is cut into **14×14-pixel
patches**; at 224 px that is a **16×16 grid = 256 patch tokens**, plus **1 CLS
token** (a learned "summary"). At each of the 24 layers, every token is a
**1024-number vector**. So one image at one layer = a 257×1024 block.

The full sequence = `n_prefix + 256 patches`, where
`n_prefix = 1 (CLS) + num_register_tokens`:
- plain `dinov2-large` (ours): `num_register_tokens = 0` → `n_prefix = 1` → 257 tokens.
- `dinov2-with-registers`: 4 registers → `n_prefix = 5` → 261 tokens.

To analyse, **pool** the 257 tokens into one vector per image:
- **CLS** = token 0 (trained as a global summary) — `layer_h[:, 0, :]`.
- **mean_patch** = average of the 256 patch tokens — `layer_h[:, n_prefix:, :].mean(1)`.

We do 3 layers (18/22/24) × 2 poolings = **6 matrices**, each 84×1024. Every test
asks the same thing of these matrices: *do the numbers separate shrouded vs open?*

**Labels live in the FILENAME token `_SHR_` / `_OPN_`**, not the folder
(`list_figures` scans recursively; `binary_labels` re-reads the token). To relabel
an image you must rename its token; moving folders alone changes nothing.

---

## 1. Stage 0 QC columns (`qc_embedding_report.csv`)

### Architecture facts (descriptive)
| Column | Meaning |
|---|---|
| `num_register_tokens` | extra scratchpad tokens (0 for base, 4 for -with-registers) |
| `n_prefix` | `1 + num_register_tokens`; how many non-patch tokens lead the sequence |
| `hidden_dim` | model's internal width (1024 for ViT-L), from `config.hidden_size` |
| `emb_dim` | width of the vector actually **saved** (= `hidden_dim` here; differs only if you project) |

### Health metrics (judged)
| Column | Computed as | Read it |
|---|---|---|
| `nan_count` / `inf_count` | `np.isnan/isinf(arr).sum()` | **must be 0** |
| `all_zero_count` | rows that are entirely zero | **must be 0** (dead image) |
| `exact_duplicate_count` | `n_rows − unique rows` (whole matrix) | >0 ⇒ duplicate image file |
| `dim_var_mean/min/max` | `arr.var(axis=0)` = variance of **each of the 1024 dims across images**, summarised | very low everywhere ⇒ embeddings barely move |
| `near_dead_dims` | count of **dimensions (columns)** with variance < 1e-8 | dims that are constant across all images (no info) |
| `cos_mean` / `cos_std` | mean / std of pairwise cosine similarity (all pairs of a ≤512 sample; here all 3,486) | the **spread**. ≈1.0 = crammed/degenerate (bad); **lower = more spread = good** |
| `cos_p05/p50/p95` | 5/50/95th percentiles of that similarity | distribution shape; `p95−p05` = how varied |

Hard gate (`qc_pass_fail`): `nan==0 AND inf==0 AND all_zero==0`. The "96% vs 44%
similar" story is the `cos_mean` column: layer 18 ≈ 0.96 (crammed), layer 24 ≈ 0.44
(spread) → later layers separate images more.

---

## 2. Stage 1 structure metrics (`structure_report`)

### Intrinsic dimension — `partic_dim`, `pcs_90` (`intrinsic_dim`)
PCA finds directions of greatest variance; each has eigenvalue λ = variance along it.
- `pcs_90` = how many directions summed reach 90% of total variance.
- `partic_dim` (participation ratio) `= (Σλ)² / Σλ²`: spread evenly over k dims → ≈k;
  one dim dominates → ≈1. A smooth "effective number of dimensions."

Per-matrix values (this data): 14.0, 16.3, 21.6, 20.1, **28.7**, 19.2 → "**~15–30**".
Capped at `min(n−1, 1024) = 83` by the **84-sample** size, so it is a *floor* estimate;
more images would give a more reliable (possibly higher) number. **Use:** sets how far
you can compress (a sensible `n_pca`) and diagnoses a rich vs collapsed representation.

### Hopkins — clustering tendency (`hopkins`, `n_pca=10`)
**Unsupervised**: "does the cloud clump at all?" 0.5 = uniform, →1 = clumpy. It is
nearest-neighbour based, and in high-D distances **concentrate** (everything ≈
equidistant), so it must be run on a **low-D PCA** (10) to be meaningful. 10 keeps the
dominant shape while avoiding distance-concentration; 30 starts re-admitting it.

### Hopkins vs the probe
- **Hopkins = unsupervised**: is there *any* structure? (ignores labels)
- **Probe = supervised**: does structure align with the *shrouded/open label*?
Clumping can exist that has nothing to do with your labels — you need both.

### The probe — `probe_auc`, `probe_acc`, `p_probe` (`linear_probe`, `n_pca=50`)
`PCA-50 → StandardScaler → LogisticRegression`, scored with **5-fold StratifiedKFold**
(scores on held-out images; without CV, 1024 dims separate 84 points by accident).
- `probe_auc` = mean ROC-AUC over folds. 0.5 = coin-flip, 1.0 = perfect.
- `p_probe` from `permutation_test_score`: fraction of 300 **shuffled-label** refits that
  beat the real accuracy. Probes need *signal capture* → 50 dims (covers intrinsic
  ~15–30 + margin) whereas Hopkins needs *robust geometry* → 10 dims.

---

## 3. Separation, the permutation test, p, z

### The "red line" — observed `sep_ratio` (computed once, true labels)
1. All 3,486 pairwise cosine distances (`D = 1 − X·Xᵀ`, vectors L2-normalised).
2. **between** = mean of the **1,764** shrouded↔open distances (= 42×42 cross pairs).
3. **within** = mean of the **1,722** same-class distances (= 2·C(42,2)).
4. `sep_ratio = between / within`. >1 ⇒ classes separate. (layer24_mean_patch = 1.057)

### The permutation null — the "blue histogram"
Each of the 10,000 blue values is the *same* `sep_ratio` with the **labels shuffled**
(same 84 points, same balance, random labels) → what chance produces (centred ≈ 1.0).
- **x-axis = sep_ratio value**, **y-axis = count of shuffles in that bin**, total area = 10,000.

### The p-value
`p = (#shuffles with ratio ≥ observed + 1) / (n_perm + 1)` = the **fraction of the blue
area at/right of the red line** = "how often does chance match or beat me." Tiny p
(red line far right) ⇒ not chance ⇒ **real**. Big p (red buried in blue) ⇒ could be
chance. Shuffling needs **no independence assumption** on the dependent pairwise
distances — that is why it is the honest test here (a plain t-test is not).

### The z-score (and "is z=8.5 bad?" — no)
`z = (observed − null_mean) / null_std` = how many blue-histogram SDs the red line sits
out. `null_mean` ≈ centre of blue, `null_std` ≈ width of blue. **High z = strong evidence
the effect is REAL, not that the result is bad.** Significance and size are different axes.

---

## 4. Significance ≠ effect size — and scaling to ~1,000 images
- **Significance** (p, z): "am I sure it's not chance?" — depends on effect **and** N.
- **Effect size** (`sep_ratio`, AUC, silhouette): "how big?" — ~independent of N.

At N=84 → 3,486 pairs → razor-thin null (std≈0.006) → a tiny shift (1.00→1.057) is 8.5
SDs out → p≈0.0002. **Lots of comparisons make a small effect rock-solid significant.**

At ~1,000 images (~500k pairs):
1. **Significance becomes automatic / uninformative** — p floors at `1/(n_perm+1)` for
   everything real; you still see it (a "real?" checkbox) but **rank by effect size**.
2. **Effect sizes stay ~the same, measured more precisely** (tight CIs). More data pins
   the effect down; it does not inflate it.
3. The probe gets more trustworthy (modest AUC rise, smaller fold variance).
4. **Add at scale:** `GroupKFold` by patent (multiple figures/patent → no leakage),
   the patent-aggregation decision, and **bootstrap CIs** (below). Subsample pairs for
   the permutation test (500k×10k is slow).

---

## 5. Bootstrap confidence intervals (the clean report at large N)
A CI carries three facts a floored p cannot: **real** (excludes the chance level?),
**big** (where it sits), **precise** (how wide).

**Bootstrap idea:** you measured AUC on one sample; to find its plausible range,
*fake* new samples by resampling your data **with replacement** N times, recompute the
metric, repeat ~2,000×, take the middle 95% (2.5–97.5th percentile).

- `auc_ci(X, y)` → out-of-fold probabilities once (honest), then bootstrap the
  (label, score) pairs. Chance = 0.5; `excludes_null` = lo > 0.5.
- `sep_ratio_ci(X, yb)` → **stratified** resample (within each class, preserves balance);
  exact-zero distances from duplicate points are masked (they would deflate `within`).
  Chance = 1.0; `excludes_null` = lo > 1.0.

Read `AUC = 0.72 [0.60, 0.83]` as: best estimate 0.72; 95% of resamples land in
[0.60, 0.83]; the interval is **above 0.5 → real**, sits at ~0.72 → **modest**, and is
**wide → only 84 images** (it will tighten as N grows). Prefer CIs over "p<0.0001" at
large N because p hides whether the true AUC is 0.51 or 0.95.

---

## 6. Confounds — "skewed" vs "decodable" (`confound_presence`, `confound_decodability`)
- **Skewed** = the confound is distributed **unequally across shrouded/open**
  (t-test for `filing_year`, chi² for `is_bell`). If balanced, it can't align with the label.
- **Decodable** = the **embedding can predict** the confound (probe AUC / R²).

A confound biases the result **only if BOTH**:

| | decodable | not decodable |
|---|---|---|
| **skewed** | ⚠️ CONFOUND | harmless (model can't use it) |
| **not skewed** | parallel signal — harmless ← *Bell here* | irrelevant |

Our result: applicant (Bell) is **decodable (AUC≤0.82) but balanced (p=0.80)** → model
reads drawing style, but it doesn't explain the design signal. Filing year is **balanced
(p=0.34) and not decodable (negative R²)** → no temporal leakage.

---

## 7. Knobs & honesty (avoid inflated numbers)
A **knob** = any free choice that affects the result but isn't the data:
`n_pca`, logistic `C`, Ridge `alpha`, **which layer/pooling you headline**, classifier type.

Trap: trying many knob values and reporting only the best = p-hacking → inflated, won't
replicate. Two honest options:
1. **Fix knobs in advance, report ALL 6 matrices** (what `structure_report` does). No cherry.
2. **Nested CV** if you must tune: an **outer** loop evaluates; an **inner** loop (on the
   outer *training* part only) picks the knob, never seeing the outer test fold. Report the
   outer score. Rule: *never let the data you score on influence the choices you make.*

`n_perm` is not a knob in this sense — it only sharpens p's resolution; raise it freely.
The real levers for *better* numbers are a better representation (layer/pooling/**fine-tuning**),
not refitting the probe on the same information.

---

## 8. Current headline result
> Frozen DINOv2-large carries a **real, highly significant, but small** shrouded-vs-open
> signal (best at **layer 22 cls / layer 24 mean_patch**): permutation p ≈ 0.0002 (z≈8.5),
> `sep_ratio` ≈ 1.06 with 95% CI excluding 1.0, probe **AUC ≈ 0.72 [0.60, 0.83]**. The
> signal is **genuine** — not explained by applicant drawing-style (decodable but balanced)
> or filing year (balanced, not decodable). N = 84, 1 figure/patent, balanced 42/42.

Next: scale to ~1,000 images (GroupKFold + bootstrap CIs), then **fine-tune** and remeasure
— does the weak signal become strong?
