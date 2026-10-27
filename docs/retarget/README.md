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
