# Chameleon: Design Guardrails and Evaluation Rubric

Prepared October 24, 2026. Companion to [chameleon-input-output-and-judging.md](chameleon-input-output-and-judging.md) and [chameleon-design-principles-for-training.md](chameleon-design-principles-for-training.md).

## Purpose

This document defines what a correct, good resized design looks like and how to check it. Use it to:

- gate every final output before it is written to `output/`;
- score candidates and engine versions on the benchmark;
- write prompts for an AI judge, and calibrate that judge against people;
- decide whether a change to the engine or a model is an improvement.

It does not describe the official scorer. The official rubric weights are unpublished. Everything here is our internal standard. It is built from the challenge rules plus established design guidance (sources at the end). When a threshold is our own choice rather than a published standard, it is labelled **Chameleon default**.

## How evaluation is layered

Evaluation runs in four tiers, in this order. A failure in a higher tier is never cancelled out by a good score in a lower one.

| Tier | Question | Outcome | Who checks |
|---|---|---|---|
| 0. Disqualifiers | Did we break a rule that ends the submission? | Pass / **reject** | Deterministic code |
| 1. Structural correctness | Is the document valid and honest? | Pass / fail per check | Deterministic code |
| 2. Measurable design quality | Does the render meet the design floors? | Per-dimension scores and defect lists | Code using the real renderer |
| 3. Perceptual quality | Would a designer accept it? | 1–10 per criterion plus pairwise preference | AI judge, calibrated against human reviewers |

Report every tier separately. Never fold Tier 0–1 results into one average with Tier 2–3.

---

## Tier 0 — Disqualifiers (hard gates)

Any one of these rejects the output. The engine must not return it, and a benchmark run counts it as a failure, whatever its other scores.

| ID | Guardrail | Check |
|---|---|---|
| D1 | **No regenerated pixels.** No asset's pixel content is regenerated, redrawn, inpainted, outpainted or replaced, including backgrounds and minor decorations. | Every image element's `asset_id` equals the source. No new asset files are written. No generative image call exists anywhere in the runtime path. |
| D2 | **No content rewrite in stage one.** Text `content` is byte-identical to the source. | String equality per text ID, including whitespace and `style_runs` ranges. |
| D3 | **No font-family change in stage one.** | `font_family` equality per text ID, and in every style run. |
| D4 | **No added or removed elements.** | The multiset of output IDs equals the source ID set. Each ID appears exactly once. |
| D5 | **No hiding.** No element is made effectively invisible to dodge a layout problem. | Opacity is unchanged from the source. The visible area inside the canvas is at least 50% of the element's own box (Chameleon default), unless the source element already bled. No element is fully covered by an opaque element it was not covered by in the source. |
| D6 | **Offline.** The runtime makes no network calls. | Run inside the Docker image with networking disabled. |

The image-retargeting literature includes methods that prune and then *repaint* an image to a new ratio (for example PruneRepaint, NeurIPS 2024). Those methods violate D1. Only cropping through `fit: "cover"` on a frame and uniform scaling are allowed.

## Tier 1 — Structural correctness

| ID | Check | Pass rule |
|---|---|---|
| S1 | Schema validity | Validates against `chameleon_design_schema_v0.9.json`. |
| S2 | Canvas | `canvas.width/height` equal the target exactly. |
| S3 | Type preserved | `type` is unchanged per ID. |
| S4 | Finite geometry | `x, y, width, height` are finite. Width and height are > 0. |
| S5 | Containment | Every element's rotated bounding box is inside the canvas. Exception: a source element that intentionally bled may bleed by the same proportion. The final bleed policy is still pending organizer guidance. |
| S6 | Foreground image proportions | For image elements other than the canvas background, `|log((w/h)_out / (w/h)_src)| ≤ 0.01` (about 1%), or `fit` is `contain`/`cover` so the pixels are not stretched. A `fill` foreground image must keep its aspect ratio. |
| S7 | Style preservation | `rotation`, `opacity`, `color`, `fill_color`, stroke, `text_align`, `letter_spacing`, `group_id` and `style_runs` are unchanged. `line_height` may change only if it stays a ratio of the font size. |
| S8 | Stacking order | The relative `z_index` order of every pair of overlapping elements is unchanged. |
| S9 | Only permitted edits | Diff the output against the source. The only changed fields are `x`, `y`, `width`, `height`, `font_size`, and `fit` on a canvas background (see `background_cover_allowed` in [engine/scene.py](../engine/scene.py)). |
| S10 | Renders | The official renderer produces a PNG of exactly the target size without errors. The PNG is rendered from the *exact* serialized JSON. |
| S11 | Diagnostics separation | Scores, candidates and debug metadata are not inside the design document. |

[engine/invariants.py](../engine/invariants.py) implements most of S1–S9. **Rule:** if no candidate passes Tier 0 and Tier 1, the run is logged as a structural failure. It must never be silently returned as success. The current last-resort path in [engine/pipeline.py](../engine/pipeline.py) must emit that flag.

---

## Tier 2 — Measurable design principles

Each dimension below has a principle, a measurable floor (*gate*), a scoring guide (*score*), and a check. Gates produce defects. Scores range from 0 to 100 and are used for ranking.

Measure everything on the **rendered output at target size**, using the real fonts and the renderer's wrapping. Geometry-only checks cannot prove text fits.

### 2.1 Legibility and text fit

**Principle.** Every piece of copy must be readable at the size it will be seen. A text box can lie inside the canvas while its glyphs are clipped.

| Check | Gate | Notes |
|---|---|---|
| Overflow | No text is clipped, and no word is wider than its box. Rendered text height ≤ box height + 0.5 px. | New or worsened overflow is a defect. Overflow already in the source is not (as in `judge_quality`). |
| Absolute size floor | headline ≥ 14 px, subhead ≥ 10 px, body ≥ 9 px, legal ≥ 7 px (Chameleon default, `FLOORS` in [engine/blocks.py](../engine/blocks.py)). | When the target area shrinks a lot, the floor is `min(FLOOR, src_size × clamp(√(area ratio), 0.5, 1))`. Any text below 7 px is always reported. |
| Comfortable size | Score against `IDEAL` in [engine/metrics.py](../engine/metrics.py): headline max(18 px, 7% of short side), body max(11 px, 3% of short side). Never require more than the source size. | Butterick recommends 15–25 px body text on screens. That figure assumes long-form reading, so treat it as the comfortable end for posters and the upper bound for banners. |
| Line length | Multi-line body copy should run 45–90 characters per line (Butterick). Lines over 90 characters are a warning. Widows (a one-word last line) are left to visual review and are not checked automatically. | Headlines are exempt from the minimum. |
| Line spacing | Keep the source's line-height ratio. If the source does not define it, the effective ratio should be 1.2–1.45 for body text (Butterick). | Tight leading on display type is acceptable. |
| Readability compromise | Dense tasks such as Level 3 (1080×1350 → 728×90) may not fit every line above its floor. The output must still preserve all copy (D2). Such outputs are flagged `relaxed_floors` and reported, not hidden. | Banner guidance favours short copy (5–10 words), but copy cuts are a finale feature, not a stage-one one. |

### 2.2 Contrast

**Principle.** Text and meaningful graphics must stand out from what is actually behind them in the render.

| Check | Gate | Source |
|---|---|---|
| Text contrast | ≥ 4.5:1 for normal text, ≥ 3:1 for large text (≥ 24 px regular or ≥ 18.66 px bold). Measured against the composited pixels below the text, not the canvas colour. | WCAG 2.2 SC 1.4.3 (AA) |
| Non-text contrast | Button shapes, badges and logos on a coloured field: ≥ 3:1 against adjacent colours. | WCAG 2.2 SC 1.4.11 |
| Regression rule | If the **source** already fails WCAG for a text, the output must not be worse. Flag when output contrast < 3 **and** < 0.6 × source contrast (current `contrast_regressions`). The contrast *score* counts only text that met AA in the source. A text counts as kept when its ratio is within 5% of the source's, because the same colours at a smaller size can fall out of WCAG's large-text band and colours are immutable. | Chameleon default. Brand designs often break AA on purpose, and we must not punish preserved intent. |
| Busy background | Text placed over a photo must sit in a low-texture area. Gray-level standard deviation behind the text ≤ the source's + 0.1 (`busyness` in [engine/quality.py](../engine/quality.py)). | Layout papers call this Readability, the image gradient behind text. Lower is better. |
| Over gradients and images | Test the lowest-contrast region, not the average. The current implementation samples a 12×4 grid and takes the 25th percentile. | WCAG guidance via WebAIM |

### 2.3 Hierarchy and scale

**Principle.** The eye should reach elements in order of importance. Importance comes from size, weight, colour and placement (NN/g).

| Check | Gate / score |
|---|---|
| Size order preserved | For every text pair whose source sizes differ by ≥ 15%, the larger source text stays ≥ the smaller one in the output. Any inversion is a warning. Three or more inversions, or any inversion involving the headline, is a defect. |
| Headline dominance | The primary headline (or hero image) is still the largest or most prominent element of its kind. |
| Scale contrast | Keep the source's headline-to-body size ratio within ±35% (Chameleon default). Keeping less than 65% of the source ratio is a **defect** (`hierarchy.scale`), and smaller losses reduce the hierarchy score in proportion. Collapsing a 3:1 ratio to 1.1:1 flattens hierarchy, even if every line is legible. A modular-scale ratio of about 1.25–1.5 between adjacent levels is a useful reference. |
| Few sizes | Do not introduce new distinct size levels. NN/g suggests about three sizes in a layout. When text shrinks, keep same-size texts in the source at the same size in the output. |
| Hero prominence | The hero image's share of canvas area should be ≥ 45% of its source share (current `prom`). A score below 0.5 is reported. |
| CTA visibility | Any CTA (a shape + text group) is still visible and distinguishable. Banner guidance says the lead message and CTA should both register in a "squint/blur test". |

### 2.4 Grouping (Gestalt)

**Principle.** Things that belong together look together, through proximity, common region and similarity. Unrelated things are separated by more space than related ones.

| Check | Gate |
|---|---|
| Rigid groups | Elements with a shared `group_id`, and inferred compound units (button = shape + label, badge = shape + text), keep their relative offsets. Translation and uniform scaling are allowed. Relative-position error ≤ 2% of the group size. |
| Common region | Text that sat inside a shape or card in the source still lies ≥ 95% inside it. |
| Proximity ratio | Within a group, the median gap must be smaller than the gap to the nearest non-member. Required ratio: inter-group gap ≥ 1.5 × intra-group gap (Chameleon default). This applies NN/g's label/field rule to design blocks. |
| Photo support | Text that sat on a large photo in the source stays on that photo (`unsupported_text`). It must also stay in roughly the same quiet region, with a relative shift ≤ 0.14 (`scene_displacement`). |
| No false merging | A full-canvas background is never treated as a group container for everything on top of it. |

### 2.5 Overlap and collisions

**Principle.** Overlap is intentional or it is a defect.

| Check | Gate |
|---|---|
| Unintended overlap | Two foreground blocks that did not intersect in the source must not intersect in the output by more than 8% of the smaller box (current `collisions`). Text-on-text overlap must be **0**. Text must also not be newly covered by an element of another group (a button or badge landing on copy). This counts once the other element covers more than 35% of the text's height or width; touching only the empty edge of the line box does not count. It is reported as `overlap.text`. |
| Intentional overlap | Overlaps that existed in the source (text on a photo, a badge on a product) may stay. Their contrast is re-checked under 2.2. |
| Occlusion of salient content | Text and shapes should not cover the salient region of a foreground image (a product or face). Score the average saliency under overlapping elements. Lower is better (layout-paper "Occlusion"). |
| Underlay | When a shape sits under text, the text should be fully contained by the shape (strict underlay). |

### 2.6 Alignment

**Principle.** Element edges and centres should share rows and columns. Near-misses look sloppy (O'Donovan et al., Haraguchi et al.).

| Check | Score |
|---|---|
| Shared lines | Count blocks whose left, centre or right x, or top, centre or bottom y, align with at least one other block within 1 px. More and larger alignment groups score higher. |
| Near-miss penalty | Offsets between 1 px and 1% of the canvas dimension are penalised. They read as errors, not choices. |
| Text alignment coherence | Stacked text blocks with the same `text_align` share that alignment edge. |
| Grid snapping | Prefer positions and gaps on a 4 px or 8 px increment (Material Design baseline grid). This is a tie-breaker, not a gate. |

Known limitation: AI judges are weak at detecting small alignment errors (Haraguchi et al., 2024). Alignment must therefore be checked in code, not only by the judge.

### 2.7 White space, margins and balance

**Principle.** Content needs breathing room, consistent margins and an even distribution of visual weight. Symmetry is not required (NN/g).

| Check | Gate / score |
|---|---|
| Outer margin | Foreground content keeps a margin of at least `max(8 px, 3% of the short side)` from every edge (Chameleon default). Backgrounds and intentional bleeds are exempt. Margins should be consistent, with left ≈ right or a deliberate asymmetry. |
| Platform safe zones | When the target matches a catalogue format with a known UI overlay, keep text, logo and CTA out of it. See the table below. |
| Coverage | The union of foreground boxes should cover 35–85% of the canvas (Chameleon default). Below 35% reads as empty, which the current code reports. Above 85% reads as cramped. |
| Even gaps | Vertical gaps between stacked text blocks of the same level should be uniform (coefficient of variation ≤ 0.25). |
| Balance | The area-weighted centroid of foreground blocks should lie near the canvas centre. Current score: `1 − min(1, 2.5 × normalised distance)`. Deliberate asymmetry copied from the source is not penalised. |
| No dead zones | No empty rectangle larger than 40% of the canvas unless the source had one (Chameleon default). Haraguchi et al. note that very large empty regions are undesirable even though white space is good. |
| Composition fit | When the aspect ratio changes (`|log(aspect_out/aspect_src)| ≥ 0.05`), foreground content should spread along the axis that grew. `long_axis_use` = the content bounding box's share of that axis divided by the same share in the source, capped at 1. Because it is relative to the source, a source that was sparse along that axis is not penalised, and same-aspect resizes are skipped. A letterbox (the source shrunk uniformly) scores `exp(−aspect change)`, e.g. 0.56 for square → 16:9. Score: space loses `100 × max(0, 0.85 − use)`; a `space.letterbox` warning is raised below 0.7 (Chameleon defaults). |

**Format safe zones (community conventions, not platform-published specs, so verify before relying on them):**

| Format | Size | Keep important content out of |
|---|---|---|
| Instagram Story / Reel cover | 1080×1920 | Top ~250 px and bottom ~250–340 px, sides ~65–90 px |
| Instagram Reel | 1080×1920 | Top ~108 px, bottom ~320 px, left ~60 px, right ~120 px |
| YouTube thumbnail | 1280×720 | Bottom-right corner (duration overlay) |
| YouTube channel art | 2560×1440 | Everything outside the centred 1235×338 area |
| Leaderboard | 728×90 | A 10 px margin on all sides (one publisher's banner spec) |

### 2.8 Image integrity and presentation

| Check | Gate |
|---|---|
| No distortion | S6. Logos and product shots are never non-uniformly scaled. |
| Crop safety | A `cover` crop must keep the salient subject. At least 90% of the salient bounding box of the source image must remain visible. |
| Minimum presence | A foreground image must not shrink below 25% of its source area share, unless the target area itself shrinks that much (Chameleon default). |
| Logo | The logo stays fully visible, undistorted and unoccluded, with a minimum height of 12 px on banners (Chameleon default). |

### 2.9 Fidelity to the source's intent

**Principle.** The resized design should read as *the same design*, with the same message order, the same focal point and the same style.

| Check | Score |
|---|---|
| Reading order | For block pairs separated along an axis in the source, keep the same relative order. The tolerance grows with aspect change: `tolerance = min(0.6, 0.4 × |log(aspect_out / aspect_src)|)` (current fidelity metric). |
| Focal point | The source's most prominent element is still the most prominent. |
| Palette and style | No new colours. Background treatment is unchanged (S7 enforces this). |
| Arrangement fit | The composition suits the target shape. A square layout shrunk to fit a 4:1 banner fails this even if every check above passes. The code proxy is composition fit in §2.7 (long-axis use); whether the rearrangement actually works is judged in Tier 3. |

### 2.10 Tier 2 aggregation

The rubric's overall Tier 2 score weights the nine dimensions as `WEIGHTS` in [engine/rubric.py](../engine/rubric.py): legibility 0.22, hierarchy 0.16, grouping 0.12, space 0.12, contrast 0.10, overlap 0.10, alignment 0.08, imagery 0.05, fidelity 0.05. The engine's older four-part score ([engine/metrics.py](../engine/metrics.py)) feeds several of these dimensions. Both are internal ranking aids, not the official weights.

1. **Gate first.** Any Tier 2 *defect* (overflow, text-on-text overlap, broken rigid group, unreadable headline, contrast regression) means the output is marked `quality_failed` (`judge_quality` returns `passed: false`). Such an output can still be returned if it is the best compliant option. Report it.
2. **Then score.** Use the 0–100 weighted score only to rank candidates that share the same gate status.
3. **Report the worst case.** For a benchmark, publish the mean, the 10th percentile and the defect rate per dimension. A good mean can hide failures on dense or extreme-aspect tasks.

---

## Tier 3 — AI judge rubric

The official evaluator scores the rendered PNG with an AI model. We run our own judge to approximate it, and calibrate that judge against people.

### Criteria and anchors

Score each criterion from 1 to 10 using these anchors: **10** flawless; **7** acceptable with minor issues; **4** obvious shortcomings; **1–2** broken. These anchors follow Haraguchi et al. (2024).

| Criterion | What a 10 looks like | What a 4 looks like |
|---|---|---|
| Legibility | All copy reads easily at the displayed size. No clipping, no text on busy areas. | Some copy needs effort to read, or a line is clipped or cramped. |
| Hierarchy | One clear entry point. The headline, then supporting copy, then the CTA read in order. | Two elements compete. Fine print is too prominent. |
| Grouping | Related items clearly belong together. Spacing separates units. | A label floats away from its button, or a caption drifts from its image. |
| Composition and balance | Uses the target shape well. Even weight, consistent margins, purposeful white space. | Content is crammed into one area, or large dead zones remain. Looks like the source shrunk to fit. |
| Imagery | Product and logo are clear, undistorted, not covered, and sensibly cropped. | The subject is cut off, too small, or covered by text. |
| Alignment and polish | Edges line up, gaps are consistent, no near-misses. | Several near-misses and uneven gaps. |
| Faithfulness | Clearly the same design and message, adapted rather than altered. | The message or focal point has changed. |

### Judge input

Give the judge the **source render** and the **output render**, each labelled with its size, plus the target format name (for example "YouTube Thumbnail, 1280×720"). Do not show internal scores, IDs or the engine's own diagnostics, because they bias the judge.

### Prompt template

```text
You are a senior graphic designer reviewing an automatic resize of a design.
Image A is the ORIGINAL ({src_w}×{src_h}). Image B is the RESIZED version for
"{format_name}" ({tgt_w}×{tgt_h}). The resize may only move and scale existing
elements and change font sizes. It may not change copy, fonts, colours or images.

Score Image B from 1 to 10 on each criterion: 10 = flawless, 7 = acceptable with
minor issues, 4 = obvious shortcomings, 1-2 = broken.
- legibility, hierarchy, grouping, composition, imagery, alignment, faithfulness
For each score, name the specific element or area that drove it.
Then list up to 3 concrete defects, most severe first.

Respond only with JSON:
{"scores": {"legibility": n, ...}, "reasons": {"legibility": "...", ...},
 "defects": ["..."], "overall": n}
```

### Pairwise mode (preferred for ranking)

To compare two candidates or two engine versions, show the source and both outputs. Ask "Which resize is better overall, or are they equivalent?" Allowed answers: `A`, `B`, `tie`, `unsure`.

Bias controls:

- **Position bias:** run each pair in both orders. A win counts only if the same output wins both times. Otherwise it is a tie.
- **Variance:** take 3–5 samples per item and use the median score or the majority vote. Haraguchi et al. found GPT-4o's run-to-run variance lower than human variance, but not zero.
- **Score-option bias:** keep the anchor wording fixed. Do not reorder criteria between runs.
- **Blindness:** strip engine names and version labels from images and filenames.
- **Self-preference:** if a VLM proposes layouts, do not use the same model as the judge.

### Known judge limits

- It misses small alignment and spacing differences (Haraguchi et al., 2024). Tier 2 code checks cover these.
- It can reward a pleasant image that hides a structural lie. Tier 0–1 always run first.
- It may assume missing copy was intended. Tell it the copy must be complete, and check D2/D4 in code.

### Calibration protocol

1. Build a **gold set** of 100–200 resize pairs. Cover every aspect bucket and density level, and include deliberate perturbations (x shifts, font-size changes, broken groups, low contrast).
2. Have ≥ 3 people score each item with the same rubric, and collect pairwise preferences.
3. Require judge–human agreement of Cohen's κ ≥ 0.6 (≥ 0.8 is strong) on pairwise outcomes, and Spearman ρ ≥ 0.5 per criterion. Report human–human agreement alongside as the ceiling.
4. Repeat calibration whenever the judge model, its prompt or the renderer changes.

---

## Benchmark construction

Overfitting to the three samples is the main risk. The hidden targets are real platform formats we have not seen.

| Axis | Buckets |
|---|---|
| Aspect change `|log(aspect_out/aspect_src)|` | ≈0 (same shape), small (< 0.3), large (0.3–1.0), extreme (> 1.0, e.g. square → 728×90) |
| Direction | Toward wide, toward tall |
| Area change | Upscale, similar, ≥ 4× smaller |
| Density | ≤ 5, 6–12, > 12 elements; text-heavy vs image-heavy |
| Structure | Photo background with overlaid copy, flat colour, explicit groups, rotated elements, missing asset |
| Target catalogue | Instagram Post 1080×1080, Instagram Portrait 1080×1350, Story 1080×1920, YouTube Thumbnail 1280×720, Facebook/LinkedIn link 1200×628, X post 1600×900, Leaderboard 728×90, Medium Rectangle 300×250, Skyscraper 160×600, Pinterest 1000×1500 |

Sources: the three supplied samples, held-out Crello families (never families used for training; see [engine/models/README.md](../engine/models/README.md)), and synthetic designs from [evaluation/synthetic.py](../evaluation/synthetic.py). Run with `evaluation/benchmark.py`, using pixels on for final numbers.

**Held-out Crello split.** The family is `metadata.source_id`, and the split is `training/export.py` `split_of`: `sha256(family) % 100`, with ≥ 85 as test. (This is not the dataset's own `metadata.split` field, which disagrees for 50 of the 60 local designs.) `training/export.py` skips test families. `training/perturb.py` still audits them but never writes them to `pairs.jsonl`. Ten of the 60 local designs in `output/crello/designs` are test families. Run:

```bash
python -m evaluation.benchmark --corpus output/crello/designs --split test --name crello-test --offline --formats 1080x1080,1080x1920,1280x720,1584x396,1000x1500,728x90,300x250,160x600
```

Latest run (10 families × 8 sizes = 80 cases, corpus hash `f7b73bf2a7aaec51`, `output/evaluation/crello-test`):

| | Value |
|---|---|
| Tier 0 failures | 0 |
| Tier 1 failures | S4 × 4 (all from one source, `45a507fb…`, at banner and small sizes) |
| Valid / quality passed | 76 / 54 of 80 |
| Tier 2 defects | grouping.photo 14, legibility.floor 13, contrast.regression 12, legibility.overflow 6, overlap.collision 1 |
| Tier 2 overall mean / p10 | 84.2 / 68.4 (85.7 / 70.5 without composition fit) |
| By aspect bucket (mean) | same 88.5 (7 cases), small 87.5 (9), large 86.5 (31), extreme 80.1 (33) |
| `space.letterbox` warnings | 23 of 80 |
| Median / p95 seconds | 1.6 / 5.9 |

The held-out set is harder than the samples (samples: overall 94.8, 38/40 quality passed). It surfaces contrast regressions and photo-grouping defects the samples barely show, and the engine still letterboxes many Crello sources at large and extreme aspect changes. Adding composition fit changed no defect count. It moved 7 of 80 picks away from letterbox-like templates, and the same-aspect bucket was unchanged.

## Required report per run

```text
engine_version, judge_version, renderer_version, corpus_hash
tier0_failures: count + ids             (must be 0)
tier1_failures: per check
tier2: per dimension mean / p10 / defect rate, sliced by aspect bucket
tier3: per criterion median, pairwise win/tie/loss vs previous release
runtime: median / p95 seconds per task, failures/timeouts
```

## Release gate

A build may be submitted only if all of these hold:

- [ ] Tier 0: zero failures on the full benchmark.
- [ ] Tier 1: zero failures, or every failure is a documented source bleed pending organizer policy.
- [ ] Tier 2: no new or worsened overflow; zero text-on-text overlap; zero broken rigid groups; headline above its floor on every non-extreme task.
- [ ] Tier 3: pairwise result vs the previous release is no worse (wins ≥ losses) overall *and* in every aspect bucket.
- [ ] The exact JSON returned is the JSON rendered, in the offline Docker image.
- [ ] Any model weights used are publicly accessible.

## Implementation

Every tier is code. The same functions gate the engine's own choice, run the benchmark and compute training rewards.

| Piece | Where | Command |
|---|---|---|
| Tier 0–2 checks | [engine/rubric.py](../engine/rubric.py) `evaluate()`; each pipeline candidate carries `rubric` | — |
| Engine ranking | [engine/pipeline.py](../engine/pipeline.py): candidates failing Tier 0–1 rank last; Tier 2 defects count as hard defects; `best()` prints `STRUCTURAL FAILURE` to stderr rather than returning silently | `python -m engine.main` |
| Benchmark | [evaluation/benchmark.py](../evaluation/benchmark.py): format catalogue ([engine/formats.py](../engine/formats.py)), aspect buckets, per-dimension mean / p10 / defect rate, D6 network guard | `python -m evaluation.benchmark --name NAME --render --offline` |
| Tier 3 judge | [evaluation/judge.py](../evaluation/judge.py): prompts and schemas in [engine/constitution.py](../engine/constitution.py), median of samples, swapped-order pairwise, Cohen's κ against human labels | `python -m evaluation.judge score RUN`, `... compare NEW OLD`, `... calibrate output/workspace/labels.jsonl` |
| Release gate | [evaluation/release_gate.py](../evaluation/release_gate.py), exits non-zero on any failed check | `python -m evaluation.release_gate RUN --baseline OLD` |
| Signal audit | [training/perturb.py](../training/perturb.py): how often each injected violation is flagged | `python -m training.perturb` |

The judge's `claude` backend uses the Anthropic API (`requirements-eval.txt`). It is for development evaluation only and never runs in the offline submission image. The `local` backend uses the optional llama.cpp model.

Signal audit on the three samples × five catalogue sizes (`--formats 1080x1080,1080x1920,1280x720,1600x900,1000x1500`; detection = rubric flags the injected violation; lower = the perturbed design scores below the clean one):

| Violation | Detection | Scored lower |
|---|---|---|
| Overlap, broken group, hierarchy swap, readability floor, safe zone, balance | 100% | 100% |
| Text overflow (small / medium / large) | 93% / 93% / 100% | 93% / 100% / 100% |
| Alignment shift (1 px–1% / 3% / 10%) | 100% / 67% / 60% | 93% / 93% / 100% |
| Letterbox (source shrunk into the target, n = 12) | 100% | 100% |

Letterboxing used to be the weakest code-level signal: 82% detected and 55% scored lower in the first audit. On the current engine without composition fit it is 83% and 92%. With composition fit (§2.7) it is 100% and 100%, and the mean overall drop is 9.5 points, up from 6.7. No other perturbation's scored-lower rate changed. Alignment/large detection fell from 67% to 60% (one case). Composition fit changes which clean layout the engine picks for two sample sizes, and the alignment check depends on that layout. Composition fit only measures whether content spreads along the axis that grew. Tier 3 still judges whether the rearrangement works.

## Sources

- W3C, *WCAG 2.2*, SC 1.4.3 Contrast (Minimum) and SC 1.4.11 Non-text Contrast; summaries via [WebAIM](https://webaim.org/articles/contrast/) and [Stark](https://www.getstark.co/wcag-explained/perceivable/distinguishable/contrast-minimum/).
- Nielsen Norman Group, [5 Principles of Visual Design in UX](https://www.nngroup.com/articles/principles-visual-design/) (scale, hierarchy, balance, contrast, Gestalt) and [The Principle of Common Region](https://www.nngroup.com/articles/common-region/).
- M. Butterick, [Practical Typography](https://practicaltypography.com/): point size, line spacing 120–145%, line length 45–90 characters.
- Material Design, [Spacing methods](https://material.io/design/layout/spacing-methods.html) (8 dp / 4 dp grid).
- D. Haraguchi et al., [Can GPTs Evaluate Graphic Design Based on Design Principles?](https://arxiv.org/html/2410.08885v1) (2024): alignment, overlap and white-space definitions, the 1–10 anchors, and judge failure modes.
- P. O'Donovan, A. Agarwala, A. Hertzmann, [Learning Layouts for Single-Page Graphic Designs](https://www.dgp.toronto.edu/~donovan/layout/) (2014).
- Layout metric definitions (Overlap, Alignment, Underlay, Readability, Occlusion): [LayoutDM](https://arxiv.org/pdf/2305.02567), [RALF](https://arxiv.org/pdf/2311.13602), [PosterLlama](https://arxiv.org/pdf/2404.00995).
- LLM-as-judge practice: [position-bias controls](https://www.metacto.com/blogs/llm-as-judge-production-evaluation), [calibration targets](https://futureagi.com/blog/llm-as-judge-best-practices-2026), [rubric score-option bias](https://arxiv.org/abs/2602.02219v1).
- Banner hierarchy and margins: [Linear Design](https://lineardesign.com/blog/banner-ad-design-best-practices-sizes-and-testing/), [University of Leeds banner guidelines](https://comms.leeds.ac.uk/wp-content/uploads/sites/7/2018/05/Digital_banner_guidelines_March17.pdf). Safe zones: [Instagram Story guide](https://www.inro.social/blog/instagram-story-size), [YouTube banner safe area](https://www.capcut.com/create/youtube-banner-safe-area-mobile-desktop-tv).
- Retargeting by repainting (forbidden here under D1): [PruneRepaint, NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93493).
- Challenge rules: [RULES.md](../RULES.md), [FAQ.md](../FAQ.md), [design schema](../chameleon_design_schema_v0.9.json).
