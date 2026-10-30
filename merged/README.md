# Merged engine

Our engine with the strengths of Gaurav's built into it, choosing its final layout the way the blind judges did. It lives apart from `engine/` and edits nothing there.

```python
from merged.resize import resize            # resize(source, target_canvas) -> design
from merged.resize import resize_with_report  # the same, plus which candidate won and why
```

## Why

The blind review (`tools/blind_eval`) found that the two engines fail in opposite ways:

- **Ours** keeps phrases, text-on-panel units and photos intact. But 22% of its outputs shrink the whole design into the middle of the canvas, and it refuses 46 of 1,000 sizes.
- **Gaurav's** fills the canvas with bold layouts for each canvas shape. But it often splits phrases, lifts text off its panel and stretches photos.

Rather than choosing between two finished layouts, this engine merges the candidate layouts inside one pipeline, so Gaurav's kind of layout is checked by our guardrails.

The root cause, found by inspecting the ranked candidates: the uniform "Original arrangement" (the shrunken copy) was often the only candidate with zero defects. A plain shrink can't create a collision or a contrast loss, and our ranking puts defect count ahead of everything, wasted canvas included. Our templates also shrink background images to fit rather than crop them, so recomposed layouts started out letterboxed too.

## What it changes

1. **Floor plans join the candidate pool** (`floorplan.py`). This is a fresh implementation of Gaurav's idea on our analysis: one plan per canvas shape.
   - tall: one column
   - square: one column, or image beside the copy
   - wide: hero on its source side, copy column beside it
   - banner: one row, `logo | copy | hero | button`

   Each block is scaled to `s_top × k^speed(role)`, and a binary search finds the largest shrink factor `k` that fits. Headline, logo and button shrink slowest, then hero images, then subheads and body text, then badges and legal copy. Layout units are our blocks, with two relations made rigid: text and accents printed on the hero photo travel with it, and a small accent (a rule, an icon) stays with the block it touches.
2. **Template candidates cover-crop eligible backgrounds**, using the engine's own `cover_backdrops`, which its invariants already allow for bottom-layer canvas backgrounds.
3. **Ranking charges for wasted canvas** (`wasted_canvas`). A background that no longer reaches the canvas edges costs 2 defects. Content shrunk below 60% of its source share costs 1. Otherwise the ranking is the pipeline's own.
4. **No refusals.** If nothing passes every guardrail, the best structurally valid layout is returned. If nothing is structurally valid, the one failing the fewest checks is returned. `resize_with_report` says which case applied.
5. **Letter spacing shrinks with the font** (never grows, because text boxes were measured with the source spacing). `MERGED_SCALE_SPACING=0` turns this off.
6. **Text grows with the canvas** (`growth.py`). Going from a strip to a squarer canvas, text may wrap to grow instead of staying at its source size, and the largest allowed scale also follows the canvas area. `MERGED_GROWTH=0` turns this off.
7. **Floor plans get leeway** (`MERGED_FLOORPLAN_LEEWAY=0` turns this off).
   - They are excused the "moved away from the original" flags (text no longer on the part of the photo it began on), because recomposing is what they're for.
   - They win ties against the pipeline's own layouts.
   - Real problems still count in full: collisions, overflow, lost contrast, unreadable or overlapping text, and wasted canvas.
   - The leeway only applies when the floor plan's headline is at least 85% as big as the headline of the best other layout, so a plan wins by being bolder, never smaller.

8. **The final choice follows the judges** (`prefs.py`, `MERGED_PREFS=0` turns this off).
   - The first blind round (200 core tasks, 3 judges, κ = 0.73) picked Gaurav's engine 46% of the time, merged 38% and ours 25%. Gaurav's biggest lead was in wide strips (leaderboards 72%).
   - `tools/blind_eval/fit_prefs.py` fits a pairwise logistic model on those votes. Its features are measured from the finished layout: smallest text, headline size (against the canvas and against the source), letterboxing, empty canvas, distortion, text overlap and clipping, plus our own checks' defect counts. Held out by design, it predicts which of two layouts the judges picked 66% of the time.
   - What it learned: **text size and headline size decide the vote, then letterboxing. Our defect counts barely matter** (collisions even get a slightly positive weight, because bold layouts have more of them). Ranking by defect count first is what kept choosing small, safe layouts.
   - So a shortlist is scored by the model: the ranking's top 4, the best floor plan, and **Gaurav's own layout** (`gaurav_bridge.py`, which runs his engine in a helper process because both engines are a package called `engine`). Any image his engine distorted is first fitted back to its proportions inside its slot (position and size only).
   - Hard rules still veto: no distorted image, no text off the canvas. The fitted weights are in `prefs_weights.json` and are part of the study's merged fingerprint, as is Gaurav's commit.

**Guards on floor plans:**
- A plan is dropped if it would push any text below 8 px (2% of the short side on big canvases), or below its source size when that was smaller.
- When the photo's own text would get too small as one unit, a second variant lays the copy beside the photo instead (`floorplan_*_free`).

`install()` adds two run-time hooks to `engine.pipeline`: floor plans appended to `propose`, and cover-crop applied after `serialise`. It runs on whichever `engine` package is importable. The blind study runs it on the committed snapshot in `work/blind-eval/engines/pranay`.

## Rules to be aware of

- **Letter spacing.** Our engine treats only position, size and font size as changeable, following the FAQ ("updated position/size and, for text, optionally font size"). Scaling letter spacing goes slightly beyond that, so turn it off if the organisers read the FAQ strictly.
- **Images keep their aspect ratio.** Unlike Gaurav's engine, this one does not stretch frames or flat images to a new shape, because the brief treats distorted images as a hard fail.

## Tests

```bash
.venv/bin/python -m unittest tests.test_merged_engine
```
