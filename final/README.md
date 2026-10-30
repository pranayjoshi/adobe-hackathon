# Final engine

The submission engine. `resize(source, target_canvas) -> design`; run with `python -m final.main` (the Docker entry point).

```
our pipeline + merged floor plans ─┐
                                   ├─► strict gate ─► drawn by the official renderer ─► judge fitted on human votes ─► design
Gaurav's top layouts ─► comply ────┘
```

## Why this shape

The blind panel (3 judges, 200 tasks, 600 votes, κ = 0.73) and the rules point in different directions, and the engine has to satisfy both.

- **People preferred bold layouts.** Gaurav's engine won 46% of picks, merged 38%, ours 25%. A model fitted on the votes shows why: the size of the smallest text, how far the content spreads across the canvas, and the headline's size decide the vote. Our defect counts barely matter.
- **The rules reject much of what people liked.** RULES.md checks the JSON too: nothing off-canvas, images in proportion. Gaurav's layouts break those in 68% and 60% of a 200-task sample. The judges could not see that in a picture, and the official scorer can.

So no single engine is enough. The final engine generates broadly, makes every layout legal, then chooses as the people did.

## The stages

1. **Generate.** `engine/` with the merged engine's floor plans and growth rules (`merged/`), and Gaurav's top 6 layouts, the best of each strategy first (`vendor/gaurav`, in a helper process, because both are a package called `engine`).
2. **Comply** (`comply.py`). Foreign layouts are rebuilt on untouched copies of the source elements, with only position, size and font size taken from them. Bleeding edges are trimmed, images are fitted back to their proportions, and text is re-fitted under its original letter spacing.
3. **Gate** (`resize.py`). Only layouts failing the fewest structural checks stay. Then new text overflow, text printed over text, newly hidden elements and lost contrast are ruled out, as long as something survives.
4. **Judge** (`prefs.py`, `pixels.py`, `weights.json`). Survivors are rendered, and a linear model over type sizes, canvas use, our checks' counts and pixel measures picks one. Fit it with `python -m tools.final_fit`.

It never refuses: if everything fails, the least bad layout returns. If every generator fails, it returns a uniform scale.

## Decisions to know about

- **Letter spacing is never changed**, unlike `merged/`, because the FAQ allows only position, size and font size.
- **`FINAL_GAURAV=0`** removes Gaurav's layouts. `vendor/gaurav` is a copy of his engine at the commit in `vendor/gaurav/COMMIT`, so it only belongs in the submission if he is on the team or has agreed.
- **The model is weakly evidenced**: 854 preferences from one panel. Held out by design, it predicts which of two layouts people picked about 69% of the time.

## Results on held-out designs

152 tasks on the 76 catalogue designs left out of all fitting (`work/final-eval/report.json`):

| Engine | Passes every structural check | Mean seconds |
|---|---:|---:|
| **Final** | **145 / 152** | 15.3 |
| Final without Gaurav's layouts | 144 / 152 | 24.2 |
| Merged | 60 / 152 | 22.4 |
| Gaurav v12 | 6 / 152 | 2.9 |

The 7 remaining failures are hairlines under 1 px, from thin source rules on extreme canvases.

Only the Final row was re-timed after the speedups, four tasks at a time on an otherwise idle machine; the other rows ran under heavier load, so compare their seconds with care. Back to back on 51 of these tasks, the speedups took the final engine from 19.7 s to 15.7 s on average.

Five tasks still lose some text (7 of 616 texts draw under 3 px). Letter spacing is a fixed pixel amount that the rules do not let us change, so widely tracked type on a thin strip may have no size that fits: on a 90 px canvas, 30 px tracking across 18 letters needs 510 px.

People voted on these 152 tasks and picked the final engine's layout 60.5% of the time (merged 21.1%, Gaurav's 15.1%; effectively one judge, not blind). On 292 tasks from 146 designs used only once at the end, 282 passed every structural check, matching the 95.4% here, so there is no sign of overfitting.

## Testing

```bash
.venv/bin/python -m unittest tests.test_final_engine
.venv/bin/python -m tools.final_eval
```

`final_eval` runs the catalogue's designs that were left out of fitting, and writes a judge-ready folder and `report.json` to `work/final-eval`.
