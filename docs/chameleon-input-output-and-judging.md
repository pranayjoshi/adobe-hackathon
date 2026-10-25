# Chameleon: Inputs, Outputs, Implementation, and Judging

Prepared October 24, 2026, from the supplied repository documentation and current implementation.

## Overview

Chameleon takes an editable design and a requested canvas size, then rearranges the original elements to suit that size. It returns an editable design document and a rendered preview.

The central challenge is adapting the composition when the aspect ratio changes. A square poster becoming a narrow banner needs a different arrangement to keep its text readable and its imagery recognizable.

This document distinguishes requirements stated in the supplied materials from engineering recommendations. The complete official scoring rubric and numerical weights were not found in the files reviewed.

## 1. Input contract

Each task is a JSON file with two main fields:

```json
{
  "source": {
    "canvas": {
      "width": 1080,
      "height": 1080
    },
    "elements": []
  },
  "target_canvas": {
    "width": 1600,
    "height": 400
  }
}
```

The empty `elements` array illustrates the structure only. A valid design under the supplied schema contains at least one element.

- `source.canvas` describes the original dimensions in pixels.
- `source.elements` contains the individual editable objects.
- `target_canvas` describes the required output dimensions in pixels.
- Sample task files also contain metadata such as `task_id`, `difficulty`, and `notes`. The engine function receives the source and target canvas.

Each task file represents one resize request. The competition runner processes every `*.json` file in `input/`.

### Element types

| Type | Information supplied | What the engine must determine |
|---|---|---|
| Text | Exact copy, font family, font size, color, position, dimensions, and optional formatting | Suitable placement, box width, wrapping, and font size |
| Image | Asset reference, position, dimensions, and fit mode | Suitable placement and scale while preserving visual integrity |
| Shape | Shape type, fill, stroke, position, and dimensions | How to move or resize it with related content |

Every element has these required common fields:

| Field | Meaning |
|---|---|
| `id` | Stable identifier that must be preserved |
| `type` | `text`, `image`, or `shape` |
| `x`, `y` | Position in pixels, measured from the canvas's upper-left corner |
| `width`, `height` | Element dimensions in pixels |

Type-specific required fields are:

- Text: `content`, `font_family`, `font_size`, and `color`.
- Image: `asset_id`.
- Shape: `shape_type` and `fill_color`.

Optional properties include:

| Property | Meaning |
|---|---|
| `rotation` | Clockwise rotation in degrees about the element's center |
| `opacity` | Transparency |
| `z_index` | Stacking order; higher values paint above lower values |
| `group_id` | Explicit membership in a designer-intended unit |
| `text_align` | Text alignment |
| `line_height`, `letter_spacing` | Text spacing |
| `max_lines` | Optional line limit |
| `style_runs` | Formatting applied to character ranges |
| `fit` | Image fitting mode: `contain`, `cover`, or `fill` |

### Semantic information is incomplete

The engine is not guaranteed labels such as “headline,” “logo,” or “product image.” Descriptive sample IDs should not become assumptions about hidden inputs. Treat IDs as identifiers and infer roles from geometry, typography, explicit groups, and appearance.

For example, a button can consist of a shape and a text label. A shared `group_id` says they belong together without identifying the group as a button.

### Supporting assets and fonts

- Images are stored in `brand_kit/assets/` and referenced by `asset_id`.
- Fonts are stored in `brand_kit/fonts/` and selected through font properties.
- The supplied structured input already contains text and individual objects, so OCR and image segmentation are unnecessary for the core challenge path.
- The brand kit is documented as placeholder material. The Level 3 reference to `solstice_holiday_badge_2025.png` is deliberately missing; do not manufacture or add a replacement.

See the [design schema](../chameleon_design_schema_v0.9.json), [FAQ](../FAQ.md), and [brand kit notes](../brand_kit/README.md).

## 2. Expected output

The engine function has this contract:

```python
resize(source, target_canvas) -> design
```

The returned design contains:

```json
{
  "canvas": {
    "width": 1600,
    "height": 400
  },
  "elements": []
}
```

Again, the empty array illustrates the document structure only. Actual output must contain every source element exactly once.

The output does not repeat the task's outer `source` and `target_canvas` wrapper. Its canvas must match the requested dimensions.

### Output files

| Artifact | Purpose |
|---|---|
| `output/<task>_output.json` | Editable resized design |
| `output/<task>_output.png` | Preview rendered from that design |

Both artifacts matter during evaluation. Structural correctness and visual quality must agree.

### Preservation and modification rules

| Property | Expected treatment |
|---|---|
| Element IDs and count | Preserve every original element exactly once |
| Element type | Preserve |
| Text content | Preserve exact copy for stage one |
| Font family | Preserve for stage one |
| Image assets | Preserve references and original pixel content |
| Position and dimensions | Change to fit the target |
| Text font size | May change to help text fit |
| Other styling | Conservatively preserve unless an explicit rule permits a change |
| Visibility | Keep elements present and meaningfully visible |
| Foreground image proportions | Preserve |

Conservative preservation includes rotation, opacity, stacking order, group membership, fills, and rich-text styling. This is an implementation policy where explicit permission to modify a field is absent, rather than a claim that every field has a separately stated official prohibition.

An image using `cover` can fill a differently shaped frame through cropping. Independently stretching a product image's width and height would distort it. Handle backgrounds and foreground imagery deliberately rather than applying the same transformation to everything.

**Regenerating, redrawing, or replacing any asset's pixel content is explicitly disqualifying**, including backgrounds and minor decorative elements.

Missing assets should be handled gracefully while retaining their references. Do not hide difficult content by deleting it, moving it outside the canvas, or making it effectively invisible.

### Correctness versus design quality

- **Correctness:** valid document, preserved elements and content, target dimensions, and legal geometry.
- **Design quality:** readable text, coherent grouping, sensible image placement, clear hierarchy, and balanced spacing.

Passing structural checks does not prove that a design is visually useful.

## 3. Sample tasks and difficulty

| Sample | Elements | Transformation | Main challenge |
|---|---:|---|---|
| Level 1 | 4 | 1080×1080 → 1080×1350 | Adjust spacing while maintaining balance |
| Level 2 | 11 | 1080×1080 → 1600×400 | Recompose a square design into a horizontal arrangement |
| Level 3 | 17 | 1080×1350 → 728×90 | Preserve dense content within a very short banner |

For Level 1, modest spacing changes may be sufficient. Level 2 can require moving imagery and text into adjacent columns or a banner row.

Level 3 presents a real capacity constraint: some content may not fit at comfortable reading sizes. The engine should return the best compliant compromise and report readability limitations through a separate diagnostic channel. Dropping or rewriting copy is not an acceptable stage-one workaround.

## 4. Efficient implementation approach

Use a staged layout search, with deterministic rules handling the core work.

### Step 1: Validate and snapshot the source

Check the input structure, unique IDs, finite geometry, target dimensions, and resource references. Keep an untouched copy of the original elements.

Apply only permitted geometry and font-size changes to that copy when producing the final document. This reduces accidental content or style changes.

### Step 2: Analyze the source once

Identify likely backgrounds, hero images, text hierarchy, groups, alignments, and intentional overlaps. Record confidence for inferred relationships.

Reuse this analysis when generating several target sizes for the same source.

### Step 3: Move meaningful groups

Treat a button's label and background as one unit. Keep badge components together. Allow flexible groups, such as a headline and paragraph, to reflow while preserving their relationship.

Grouping reduces independent placement decisions and prevents disconnected compositions. A background should not cause every overlapping element to merge into one group.

### Step 4: Generate a bounded set of different arrangements

Useful candidates include:

- A composition close to the source.
- A vertical stack.
- Image and text side by side.
- A horizontal banner arrangement.
- An anchored arrangement retaining important source positions.

Explore different structures before spending effort on small coordinate variations. The same arrangement can perform well at one aspect ratio and poorly at another.

### Step 5: Measure actual typography

Use the actual fonts and renderer-compatible wrapping to calculate line breaks and required text height for promising box widths and font sizes.

A text box can fit inside the canvas while its text is clipped. Geometry checks alone cannot establish text fit.

Cache measurements using text, font identity, size, width, and relevant styling as keys.

### Step 6: Validate and rank candidates cheaply

Before rendering, check IDs, content preservation, geometry, image proportions, and group relationships. Reject invalid candidates where feasible.

Rank valid candidates using readability, hierarchy, spacing, unintended overlap, and composition. Treat these as internal decision aids, not official judge scores.

### Step 7: Render a shortlist and repair specific failures

Render the strongest candidates with the supplied renderer. Inspect actual clipping, awkward wrapping, collisions, contrast, and image visibility.

Make bounded repairs such as widening a text box, adjusting gaps, or changing a group's orientation. Retain improvements and avoid unbounded retries.

### Step 8: Serialize and verify the final result

Apply the selected changes to the preserved source objects. Validate again, then render the exact serialized document that will be returned.

Keep diagnostics, scores, and candidate metadata outside the official design document.

### Practical efficiency measures

- Cache source analysis, text measurements, and asset loading.
- Use geometry checks before rendering.
- Render only the strongest candidates.
- Bound candidate counts and repair attempts.
- Profile actual bottlenecks before optimizing.
- Reuse the same engine for the competition runner and viewer.
- Keep the core runtime functional without external APIs.

## 5. Current repository capabilities and gaps

The implementation already includes:

- Source role and relationship analysis.
- Group-aware layout blocks.
- Renderer-based text measurement.
- Multiple candidate arrangements.
- Preservation checks and internal quality scoring.
- A viewer with ranked options and previews.

Relevant implementation files:

| File | Responsibility |
|---|---|
| [engine/resize.py](../engine/resize.py) | Competition function entry point |
| [engine/main.py](../engine/main.py) | Task loading, output writing, self-check, and rendering |
| [engine/analysis.py](../engine/analysis.py) | Source roles and relationships |
| [engine/blocks.py](../engine/blocks.py) | Layout units and groups |
| [engine/typography.py](../engine/typography.py) | Text measurement and fitting |
| [engine/layout.py](../engine/layout.py) | Layout proposals and placement |
| [engine/invariants.py](../engine/invariants.py) | Preservation and geometry checks |
| [engine/metrics.py](../engine/metrics.py) | Internal quality scoring |
| [engine/pipeline.py](../engine/pipeline.py) | Candidate generation, serialization, validation, and selection |
| [service/app.py](../service/app.py) | Viewer API and preview rendering |

The README lists a general CP-SAT geometry solver, render-based repair loop, VLM proposals, learned ranker, and Express add-on as not yet built.

The current pipeline prefers a valid candidate when one exists, but its last-resort path can return a candidate that failed validation when none passes. Compliance must therefore be measured explicitly rather than assumed.

### Recommended development priority

Benchmark and improve the existing engine before adding a model:

1. Establish preservation and rendering reliability.
2. Test the supplied samples and additional varied designs.
3. Record structural failures separately from visual failures.
4. Fix the most frequent layout and text-fit problems.
5. Add stronger geometry constraints or render-based repair where evidence supports them.
6. Consider a model only when a repeatable failure explains its role.

If role inference is consistently wrong, a local vision-language model may help propose relationships or arrangements. If good candidates exist but the wrong one wins, improve ranking. Neither addition should bypass preservation checks.

## 6. How the solution will be judged

### Documented evaluation layers

The supplied rules describe evaluation of both:

1. **Output JSON:** structural checks, including element identity, required fields, bounds, and image proportions.
2. **Rendered PNG:** the output design rendered through the official renderer and scored by an AI evaluator against a rubric.

A valid JSON alone is insufficient. A good-looking PNG cannot compensate for a broken or dishonest document.

### Structural checks

| Check | What it means for the solution |
|---|---|
| Element preservation | Every original ID appears exactly once; nothing is added or removed |
| Required fields | The design follows the schema |
| Target dimensions | The returned canvas matches the request |
| Canvas containment | Elements fit within the requested canvas, subject to any final clarified bounds policy |
| Image proportions | Logos and product imagery are not distorted |
| Content preservation | Preserve copy and font families for stage one |
| Asset preservation | Reuse supplied images without generating replacement pixels |

The supplied materials mention an intentionally bleeding decorative element in a sample. The precise final treatment of decorative bleed and rotated bounds should be confirmed from final organizer guidance; local self-check behavior is not authoritative.

### Visual quality

The rules explicitly discuss legibility and faithfulness to the original design's intent. The following questions are useful internal criteria, not a confirmed list of official rubric rows:

| Quality question | Example failure |
|---|---|
| Is the text readable? | Body copy becomes microscopic |
| Does the composition suit the target? | A square poster is compressed into a banner |
| Is hierarchy preserved? | Fine print becomes more prominent than the headline |
| Do related elements stay together? | A button label separates from its background |
| Is imagery presented well? | A product is obscured or a logo is distorted |
| Is the original intent retained? | All elements survive, but the promotional message becomes unclear |

### Speed and reliability

The supplied rules state that there is no hard stage-one execution-time limit. Faster execution is preferred when quality is comparable, and “speed and reliability” is identified as a rubric row.

This supports prioritizing output quality, while still ensuring practical and dependable execution.

### Offline execution and packaging

The documented stage-one Docker setup runs without network access. Dependencies and any required model weights must be packaged before runtime.

If the engine depends on fine-tuned or adapted model weights, those weights must be publicly accessible according to the supplied rules.

The rules mention an Apple M5 Pro reference machine, but explicitly leave exact hardware details provisional. Do not treat CPU, RAM, GPU, or disk limits as settled.

### Viewer requirement

The FAQ describes the viewer add-on as required but not scored for stage one. The current web viewer is a development capability; its existence alone does not establish completion of an Express add-on requirement.

### Stage one versus finale

For automated stage-one evaluation, treat text content and font family as fixed. The rules describe adaptive copy and font-family changes as potential finale differentiators for a live jury, rather than reliable stage-one scoring improvements.

The prohibition on regenerating image content remains an explicit rule.

## 7. Limits of local checks and scores

The starter self-check is a development aid, not the official scorer. It checks only a subset of requirements and emits warnings without failing the run.

For example, the runner's set-based ID comparison alone cannot detect duplicated copies of an element. The engine's separate invariant validation includes stronger duplicate and count checks.

Likewise, the engine's internal 0–100 quality scores are heuristics for candidate selection. They are not the official evaluator's scores.

Use varied test designs and visual review to establish whether higher internal scores correspond to better results.

## 8. Recommended evaluation priorities

1. **Zero preservation violations:** no lost, duplicated, rewritten, or replaced elements.
2. **Readable rendered output:** detect clipping, collisions, and excessively small text.
3. **Useful compositions:** test unfamiliar designs and substantial aspect-ratio changes.
4. **Reliable offline execution:** verify the packaged runner and actual renderer.
5. **Measured efficiency:** improve runtime once correctness and quality are established.

Record structural failures, renderer failures, visual problems, and runtime separately. A single average quality score can hide serious failures on dense or extreme-aspect-ratio tasks.

## 9. Unconfirmed details

The files reviewed do not provide a complete final official scoring specification. In particular, do not assume:

- Exact numerical weights for visual criteria.
- A published formula matching the engine's internal metrics.
- Final hardware resource limits.
- A complete final policy for decorative bleed and rotated bounds.
- That the supplied provisional materials supersede later organizer announcements.

## Sources

- [README](../README.md): runner contract, architecture, rendering, and current implementation status.
- [FAQ](../FAQ.md): inputs, sample tasks, resources, offline packaging, and viewer requirement.
- [Submission rules](../RULES.md): evaluation, preservation guidance, disqualification, performance, and provisional hardware.
- [Design schema](../chameleon_design_schema_v0.9.json): field-level input/output contract.
- [Brand kit notes](../brand_kit/README.md): placeholder resources and intentional missing asset.
- [End-to-end build plan](chameleon-end-to-end-build-plan.md): proposed engineering approach; not an official judging rubric.
- Current implementation files linked above: observed behavior and existing limitations.
