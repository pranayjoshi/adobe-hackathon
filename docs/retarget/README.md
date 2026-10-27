# Learned retargeting baseline (v2)

The baseline is **engine + model**: the layout transformer in `engine/retarget` proposes
layouts (greedy sample plus nucleus samples, each raw and de-overlapped), they compete with the
template candidates, and `_prefer_learned` in `engine/pipeline.py` picks the model's layout
whenever it passes every hard check. Otherwise the engine's normal ranking stands.

Raw numbers: [`baseline-v2.json`](baseline-v2.json).

## Data and model

- Crello 5.1.0 layout columns only (CDLA-Permissive-2.0): 23,302 designs.
- Designer resize families (identical text, different aspect): 1,401 families, 56,610 pairs.
- Split by family hash: train 1,146 families / 46,935 pairs, val 94 / 4,463, test 161 / 5,212.
- Encoder-decoder transformer, about 4M parameters, one decoder token per element.
  It predicts w, h, x, y and font size as residual bins relative to a per-axis stretch.
- Weights: `engine/models/retarget.npz` (fp16, best val NLL at step 11,500). Inference is numpy only.
- Training code: `training/retarget` (`fetch_crello`, `build_dataset`, `train`, `export`, `evaluate`, `visual`).

## Geometry against the designer (500 held-out pairs, 161 families)

| Method | IoU | Center err | Size err | Font err | Order agree |
|---|---|---|---|---|---|
| Uniform fit | 0.336 | 0.062 | 0.502 | 0.229 | 0.949 |
| Per-axis stretch | 0.413 | 0.053 | 0.308 | 0.130 | 0.944 |
| Model, greedy | 0.411 | 0.053 | 0.323 | 0.140 | 0.948 |
| Model, oracle best of 8 | 0.453 | 0.052 | 0.295 | 0.127 | 0.945 |

On 22 extreme pairs (aspect change over 3×), IoU is 0.10 for stretch, 0.11 for greedy and 0.18 for the oracle best of 8.

## Rendered, full engine (50 unseen designs with images)

| Setup | IoU | Center err | Size err | Font err | Order agree | Passes hard checks |
|---|---|---|---|---|---|---|
| Engine before | 0.257 | 0.102 | 0.725 | 0.328 | 0.871 | 50 / 50 |
| **Engine + model (baseline)** | **0.366** | **0.071** | **0.434** | **0.201** | **0.924** | **50 / 50** |
| Model alone | 0.388 | 0.057 | 0.362 | 0.197 | 0.937 | 32 / 50 |

Engine + model uses a learned layout in 42 of 50 cases, with a median of about 11 s per design.
The model alone fails a hard check in 18 cases: collisions in 12, unsupported text in 3, contrast in 2, and one unreadable/invalid case.

## Known weak spots

The baseline is weakest on extreme portrait↔wide changes:

- **Skyscraper (4.5×):** text separates from its badge shape and gets tiny.
- **1:1 to 4:1 banner:** the copy shrinks instead of moving beside the artwork.
- **Inset panels:** an inset panel image keeps its aspect, which leaves bands.
- **Fallbacks:** in several 2.7× cases the learned layout is vetoed and the engine falls back to a letterboxed template.

Reproduce:

```
PYTHONPATH=. python training/retarget/evaluate.py --out work/retarget-runs/v2/test-eval.json
PYTHONPATH=. python training/retarget/visual.py
```

## Extreme-stretch fixes (v3 proposals, same weights)

These are geometry changes in `engine/retarget` only; the weights are unchanged. Raw numbers: [`extreme-fixes-v3.json`](extreme-fixes-v3.json).

- **Backings:** copy on a badge or bar (at most 25% of the canvas, with the copy filling at least 20% of it) moves and scales with it.
- **Rigid units:** they are scaled by the predicted font size, never below readable floors, and placed on the canvas as a whole.
- **Overlap removal:**
  - It is measured per analysis block, as the engine's collision check does, while units move together.
  - Disjoint pairs keep their order but may switch axis (stack becomes row) when that is clearly cheaper.
  - When space runs out, graphics shrink before copy.
- **Text:** fitting no longer shrinks multi-line text into a box predicted one line short.

| Engine + model, 50 rendered cases | IoU | Font err | Order agree | Learned picks | Valid |
|---|---|---|---|---|---|
| Baseline v2 | 0.366 | 0.201 | 0.924 | 42 | 50 / 50 |
| Extreme fixes | **0.375** | **0.173** | **0.928** | 45 | 50 / 50 |

The model-alone top layout was also scored on the 32 held-out image designs that the visual set does not use, to guard against tuning on the 50:

| | IoU (50 / 32) | Font err (50 / 32) | Fails a rule (50 / 32) |
|---|---|---|---|
| Baseline v2 | 0.380 / 0.281 | 0.247 / 0.240 | 14 / 8 |
| Extreme fixes | 0.384 / 0.275 | 0.140 / 0.159 | 6 / 5 |

A looser backing (half-canvas panels, any text share) bound too much: rule failures went from 6 to 11 on the 50 cases.

## Fallbacks and safe margins (v4 proposals, same weights)

Raw numbers: [`fallbacks-margins-v4.json`](fallbacks-margins-v4.json).

- **Copy-bound fallback:** the greedy layout also comes as `learned_0_bound`, where copy stays with any graphic it sat on, photos and panels included. It ranks after the tight greedy layout, so it is used only when that fails a check. This is what lets 04 (a portrait poster turned into a banner) use a learned layout.
- **Safe margins:** copy keeps 4% of the short side from the edges, which is the 5th percentile of Crello designers' text. Copy the source already placed nearer an edge keeps its proportional gap.
- **Decorations in overlap removal:** panels and strips of at least 1% of the canvas take part, so they keep their source overlap with photos.
- **New engine vetoes for learned layouts:**
  - content that collapsed to under half the canvas share the source used;
  - an element newly hidden under another, by more than 25% of its area.

  Learned layouts failing either check rank below every template.

| Engine + model, 50 rendered cases | IoU | Center err | Font err | Order agree | Learned picks | Valid |
|---|---|---|---|---|---|---|
| Baseline v2 | 0.366 | 0.071 | 0.201 | 0.924 | 42 | 50 / 50 |
| Extreme fixes (v3) | 0.375 | 0.066 | 0.173 | 0.928 | 45 | 50 / 50 |
| Fallbacks and margins (v4) | **0.378** | **0.064** | 0.174 | **0.931** | 46 | 50 / 50 |

Still weak:
- **33:** the full photo plus its panel cannot fit side by side without a crop, so it falls back to the letterboxed source frame.
- **32:** the 4.5× skyscraper ends up with small badge copy.
- **01:** grey bands remain because the inset panel image cannot stretch.
