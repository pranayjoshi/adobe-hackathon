# Post-processing plan: constrained layout decoding with an appeal layer

This plan replaces the learned retargeter's sequential post-processing with one constrained optimisation. The current steps are: materialize, clamp, fit text, then de-overlap by shrinking. The optimisation is ranked by how good the result looks to people.

**Status (2026-10-27):** phases 1–4 are built behind `CHAMELEON_POSTPROC=solver`. The constrained decoder, strip mode and appeal-ranked selection are in; the visual judge is not. Gate numbers and gaps are in `postproc-results.md`; the human preference rounds are waiting for votes.

**The primary target is human preference.** Following the rules and resembling the designer are guardrails, not the goal.

Evidence comes from a full-dataset run (`work/retarget-scale/report.md`):
- all 23,294 runnable Crello designs, each retargeted to 3 formats (69,882 outputs);
- 15,959 comparisons against the designer's own resize (9,675 of them held out);
- 300 runs of the full engine;
- 12 ablations, each switching off one post-processing stage.

I spot-checked the headline numbers and both code-level root causes against the raw data and code, and they hold.

## 0. What success means

| Metric | Role | Target |
|---|---|---|
| **Human preference vs today's engine** (blind side-by-side, ties count half) | **primary** | ≥ 65% win rate, with the 95% interval's lower bound > 55% |
| Human preference vs the designer's own resize | primary, ceiling | ≥ 40% (a designer's hand-made version is the ceiling) |
| "Broken" votes (raters tag unreadable, crowded, cut-off, misaligned) | primary | ≤ 5% of outputs |
| Learned appeal score (validated against human votes, section 3.6) | proxy for every phase | improves at every phase |
| Fails a hard engine check | guardrail | ≤ 8% held-out, ≤ 20% extreme formats (today 31% / 51%) |
| IoU to the designer | guardrail | not below the raw model (0.310) |
| Latency | guardrail | median ≤ 15 s per design in the full engine (today 12.9 s) |

IoU is kept only as a sanity bar. It is a weak proxy for quality: a better layout can score lower, and matching the average designer gives average designs.

## 1. Where we are

| Held-out designer pairs (n=9,675) | IoU to designer | Font error | Fails a hard check |
|---|---|---|---|
| Per-axis stretch (no model) | 0.312 | 0.151 | not usable: distorts images |
| Model, raw geometry | 0.310 | 0.159 | 99.8% |
| Model + current post-processing | 0.288 | 0.208 | 31.0% (51% on extreme formats) |
| Best of 8 samples, chosen by IoU (oracle, upper bound) | 0.336 | 0.201 | 31% |

So post-processing does its validity job, but crudely. It removes 69 points of failures and costs 0.02 IoU and 0.05 font error along the way. It still leaves almost a third of outputs failing a check. Nothing in it looks at the rendered result.

Top failure modes (held-out / extreme-format panel), with the cause each traces to:

| Mode | Rate | Cause in today's code |
|---|---|---|
| Contrast regression | 10.3% / 20.6% | Copy is placed without knowing what is under it. Backing groups only cover ≤25% of the canvas. |
| Text below readable floor | 8.8% / 25.7% | `fit_text` can only shrink the font, and drops to `fs_min=1` when the copy doesn't fit. |
| Copy left its photo | 8.4% / 11.3% | Copy on a large photo moves independently of the photo. |
| New collision | 6.7% / 18.0% | De-overlap gives up after its shrink budget and leaves the overlap ("the engine vetoes it"). |
| New occlusion | 6.1% / 13.5% | Shapes that bled off the canvas get pulled inside it and land on top of other content. |
| Hairline below 1 px (vetoes the whole candidate) | 4.0% / 15.5% | Graphics are scaled by area with no minimum thickness. |
| Text overflow | 4.6% / 10.7% | Letter-spacing is an absolute pixel value the rules don't allow us to change, and fitting never re-wraps. Tracked text overflows in 15% of cases vs 0.4% for other text. |
| Alignment loss | 17% / 19% (designers: 10%) | Nothing in the pipeline keeps shared edges. |
| Under-fill | coverage 0.50 vs designer 0.62 | Graphics give up space first, and nothing grows content afterwards. |
| No re-arrangement | designers change the arrangement in 41% of large changes; we match 18% | Stack-to-row only happens when de-overlap switches axis. |

What designers do that the pipeline can't, all visible in the contact sheets:
- re-wrap copy into a different number of lines (e.g. 4 lines become 7 in a narrow column, letter-spacing kept);
- turn a stack into a row;
- keep column edges aligned;
- keep copy on its photo or panel with padding;
- fill the canvas, with median margins of 8.7% of the short side (12% on wide formats).

## 2. Design principles

1. **People decide.** Every design choice is judged by whether people prefer the result. Learned scorers and solver terms are proxies, and they are validated against human votes before they are trusted.
2. **One solver, not a pipeline of fixes.** Every stage today repairs the previous one and breaks something else (fitting creates every below-floor case; de-overlap adds 5 points of alignment loss). Rules, designer habits and aesthetic goals belong in one model, solved jointly.
3. **Valid by construction.** Every hard check the engine applies becomes a hard constraint, written in the engine's own definitions. A candidate either satisfies all of them or is reported infeasible, with the reason.
4. **Generate with the model, choose with appeal.** The solver finds the layouts the model rates most likely and that respect the rules and design principles. A learned appeal score and a rendered judge then choose among them.
5. **Everything measurable comes from data.** Margins, gaps, headline scale, padding, alignment and arrangement changes come from the train-split priors (`designer_priors.json`). Weights are tuned on train data, chosen on val and reported once on test.
6. **Only fields the rules allow us to change.** Position, size and font size, plus `fit: cover` for backgrounds only. Letter-spacing, z-order and styles stay unchanged, and nothing is cropped, dropped or recoloured.

## 3. The algorithm

**Overview:**
1. structure extraction;
2. seeds from the model;
3. a CP-SAT solve with hard rules, the model's likelihood, designer priors and aesthetic terms;
4. exact verification;
5. a learned appeal score ranks the valid candidates;
6. a rendered judge picks between the top two.

### 3.1 Structure extraction (once per source design, ~10–50 ms)

From the source design, its analysis and its layer images, build a structure graph:

- **Units.** Analysis blocks, `group_id` groups, and backing groups move rigidly with one scale variable each. Backing groups are copy on a badge, bar or panel, and become containment relations with a padding prior (median 0.51 font heights, q25 0.07).
- **Hosts.** Copy that sits on a large photo or panel (≥25% of the canvas) gets a host, chosen by painted coverage as `scene.source_photo_supports` does. It becomes a containment relation that can be released at a cost, not a rigid unit, which is why loose rigid binding failed.
- **Alignment sets.** Copy sharing a left, centre or right edge, or a top, middle or bottom edge, within 1% in the source. 73% of designer text shares an edge, and designers keep 90% of these alignments when they resize.
- **Reading order, arrangement and semantic groups.**
  - Pairwise above/left-of relations between copy blocks.
  - Whether the copy forms a stack, a row or a mix.
  - Which copy belongs together (headline with subhead, date with venue), from proximity and roles in the source.
- **Hierarchy.** Pairs of text whose font ratio is ≥1.25 in the source.
- **Overlap budget.** For every pair of blocks and large graphics, the source overlap fraction. Disjoint pairs must stay disjoint, and overlapping pairs may keep their source overlap. The same pairs are checked for occlusion (later-drawn covering earlier).
- **Text options.** For each text and each line count L = 1…L_max (L_max ≈ 2× its source lines + 2), break the content into L lines with the renderer's own measurement at a reference size. Record the coefficients:
  - width(L) = a_L · font + b_L, where b_L is the letter-spacing term (characters in the longest line × spacing);
  - height(L) = c_L · font.

  Both are linear in the font size, so tracking is modelled exactly. Each option also gets a **typography cost** (section 3.3, aesthetic term 6).
- **Thin elements.** Hairlines and rules get a minimum scale of 1/(thinner side) so they never fall below 1 px.
- **Visual weight per element.** This is precomputed from the layer pixels (`photo_analysis` maps): ink coverage × contrast against what sits under it in the source × saliency. It feeds visual balance. A dark photo and a pale shape of the same size no longer weigh the same.
- **Photo subjects.** Subject boxes per photo (`photo_analysis.analyze_asset`: saliency or the registered subject model). They are carried as affine functions of the photo's (x, y, s), so they move and scale with the photo.

### 3.2 Seeds (what the solver starts from)

- The model's greedy decode, plus 3–4 nucleus samples. Diversity is where the headroom lives: the best of 8 samples is +0.045 IoU.
- One arrangement seed per large change (|log aspect change| > 0.7). The model's own `TASK_ARRANGE` decode exists today but `propose` never uses it.

Each seed gives the solver a starting solution and its per-field target distributions.

### 3.3 The CP-SAT model (OR-Tools 9.15 is already installed)

The model works in integer pixels; scale is stored in per-mille.

**Variables**

- Per unit: x, y and scale s. All member boxes are linear in (x, y, s), and images keep their aspect automatically because one s scales both sides.
- Per text: a one-hot line-option choice, with width/height tied to font = s · f₀ through the linear (a_L, b_L, c_L) coefficients.
- Relation booleans, only for pairs whose order may change: text-text reading order and arrangement.
- Gap variables between consecutive copy in a stack or row (for spacing rhythm and grouping).

**Hard constraints.** Each is the engine's own check:

| Check | Constraint |
|---|---|
| Inside canvas (rotated extent) | bounds on x, y and x + w, y + h (rotation uses the rotated extent's coefficients) |
| Image proportions | implicit: a single scale per unit |
| Nothing hidden | s ≥ 1 / min(w₀, h₀) per element |
| Collisions (engine definition) | `AddNoOverlap2D` over intervals of source-disjoint blocks, with a gap; source-overlapping pairs get an overlap bound linearised from their source fraction |
| New occlusion | the same no-overlap and overlap bounds over later-drawn graphics ≥1% of the canvas |
| Overflow | text box ≥ the width/height of the chosen line option |
| Readable floors | font ≥ `FLOORS[role]` (the engine vetoes below them) |
| Copy margins | copy at least `margin_prior(format, q05)` from the edge; the source's bleed is kept where intended |
| Backing containment | copy inside its backing, with at least the prior padding at q25 |
| Hierarchy | font_a ≥ 1.1 · font_b for source pairs with ratio ≥ 1.25 |
| Cover background | background is `fit: cover` and fills the canvas, when `background_cover_allowed` |
| Photo subjects | copy that did not cover a subject in the source does not cover it now |

**Soft costs: fidelity and designer habits.** Everything is linear or piecewise-linear, so the model stays exact:

1. **Model likelihood.** For each field, the cost is −log p(bin) from the model's own distribution at the seed's decoding context, applied as a piecewise-linear table via `AddElement` over the binned residual. The temperature is calibrated on val. Confident fields are stiff and uncertain ones are flexible.
2. **Host release.** A penalty when copy leaves its host photo or panel, scaled by how much the source depended on it. It covers contrast and unsupported text, which are 18% of held-out outputs today.
3. **Alignment kept.** A per-set penalty when a source alignment set no longer shares its edge.
4. **Reading order and arrangement.** A cost for each flipped relation, of −log P(transition | aspect-change bucket) from the designer pairs. Turning a stack into a row is cheap for large portrait-to-banner changes and expensive for small ones.
5. **Designer priors.**
   - Content margins near the format's designer median.
   - Content-box share and grid coverage near the designer prior (fixing the 0.50 vs 0.62 gap).
   - Headline/short-side inside the format's q05–q95 band.
   - Text gaps near the gap/font prior.

**Soft costs: aesthetic terms.** These are design principles written as measurable, solver-friendly terms. Their weights are tuned against the appeal score (section 3.6), which is validated by human votes:

1. **Visual balance.** The weighted centre of visual mass, Σ mᵢ·centreᵢ / Σ mᵢ using the precomputed weights, stays near the format's optical centre. That is the designers' median content-centre offset per format, slightly above geometric centre on most formats. It is linear in positions, with masses fixed per seed scale.
2. **Grouping by proximity.** Gaps inside a semantic group (headline-subhead, date-venue) are smaller than gaps between groups, by a ratio from the designer priors (median within-group gap/font 0.81). This is a soft inequality on gap variables.
3. **Spacing rhythm.** Gaps between consecutive copy are snapped toward multiples of a base unit (half the body line height). A small integer variable per gap, with a cost for its residual, gives the consistent vertical rhythm designers use.
4. **Focal point and breathing room.**
   - The headline keeps clear space around it of at least half its line height.
   - Its centre is drawn toward the designers' headline positions for the format (a density prior, computed from train designs per format).
   - It is the copy element with the largest visual weight.
5. **Edge and grid discipline.** Rewards for copy left edges that coincide with each other or with the margin, and for centred copy sharing the canvas axis. A column grid suited to the format applies only where the source was grid-aligned.
6. **Typographic quality of the chosen line option** (precomputed per option):
   - Prefer breaks at phrase boundaries; avoid ending a line on an article, preposition or conjunction.
   - Avoid a single short word on the last line (a widow) when the text has 3+ words.
   - Avoid a ragged edge, measured as line-length variance, except on deliberate left-aligned body copy.
   - Uppercase tracked titles get balanced line lengths.
7. **Size contrast.** Headline/body font ratio inside the designer band (q25–q75: 1.24–2.74), so the hierarchy reads at a glance.
8. **Subject respect.** On top of the hard rule, copy close to a subject pays a cost that falls with distance. Copy settles into the photo's quiet regions (`photo_analysis.quiet_regions` already computes them).

**Solving**

- Hint each seed's current solution; the existing pipeline's output passes the reference-free checks in about 69% of held-out cases. The time limit is 150 ms per seed, with 8 workers.
- Pairs are generated lazily: start with the pairs that conflict at the seed and add any new conflicts the solution creates. This is the same loop as de-overlap today, but exact.
- On infeasibility, relax the soft-able constraints in priority order (host, alignment, margin prior, hierarchy) and record which one was relaxed. If it's still infeasible, return "rule-infeasible" with the reason, not a broken layout.
- For appeal, solve each seed under 2–3 weight profiles: balanced, faithful to the model, and bolder (stronger focal and fill terms). This yields diverse valid candidates (8–15 per design) for the appeal scorer to choose from.
- The spike (`work/retarget-scale/spike_cpsat.py`, synthetic random instances) found a feasible solution within 0.2 s for 8–40 units, including 15 units on 728×90.

### 3.4 Verify and repair (exact, after the solve)

- Re-measure every text with `typography.measure` at the chosen font and width. If it overflows, tighten that option's coefficient and re-solve warm; one or two rounds is expected.
- Run the engine's full checks (`invariants.validate`, `quality.diagnose`, `scene_defects`, `_new_occlusion`, `_content_share`) as the final gate. Nothing ships that the engine would veto.

### 3.5 Selection: appeal first

1. **Validity gate.** Only candidates that pass every engine check compete.
2. **Learned appeal score** (section 3.6) ranks them on the rendered image plus structural features. The IoU ranker from the previous version of this plan is dropped as a selector, and IoU stays a feature and a guardrail.
3. **Rendered judge.** The top two are compared side by side by the OpenJev judge (`engine/evaluators/openjev.py`, already wired into `generate`, off by default). If the judge is unavailable or undecided, the appeal score's choice stands. The judge's verdicts are logged for later agreement analysis against human votes.
4. **Pixel contrast check.** The engine's existing render verification runs on the final pick.

### 3.6 The learned appeal scorer

**What it predicts:** P(people prefer this design over an alternative for the same brief), trained pairwise, so it ranks candidates.

**Two models, built in order:**
1. **A features model (fast, interpretable, built first).** Gradient-boosted trees (sklearn `HistGradientBoosting`) on about 60 measured features:
   - visual balance, grid coverage, largest empty region and content centre;
   - margins vs the format prior, spacing rhythm residuals, grouping ratio;
   - alignment share, headline share of visual weight and clear space around it;
   - typography costs, size contrast;
   - subject overlap, contrast estimates;
   - the solver's own cost terms.
2. **A pixel model (built second, kept only if it beats the features model on human agreement).** A small image model scoring the rendered design at 384 px plus the target format. It uses the existing VLM/DPO training setup (`training/train_vlm.py`, `training/edit_dpo.py`, `training/reward.py`). Running a pretrained image backbone requires downloading weights; that is a decision for you (section 7).

**Training pairs** (all train split; val held out for selection):

| Source of pairs | Label | Volume |
|---|---|---|
| Designer original > the same design with one principle broken (`training/perturb.py`: misalign, crowd, shrink headline, break margins, unbalance, occlude, bad line breaks) | known-better | about 19,000 designs × 6 perturbations |
| Designer resize > our flagged outputs for the same pair (only where ours carries a hard or appeal flag, so "ours" is clearly worse) | known-better | about 10,000 |
| Our candidates against each other, labelled by the OpenJev judge | weak; down-weighted | as available |
| **Human votes** from the preference rounds (section 5) | gold; up-weighted, also used for calibration | a few hundred to a few thousand |

**Validation:**
- Agreement with held-out human votes is the key metric, with a target of ≥ 70%.
- Accuracy on held-out designer > perturbed pairs.
- A sanity check that it prefers designer originals over our best output most of the time.

A scorer that disagrees with people isn't used for selection.

**Guarding against gaming it:**
- The scorer only ranks valid candidates.
- Solver weights tuned against it are re-checked on human votes at each preference round.
- Retrain if agreement drops.

### 3.7 Engine integration

- `engine/retarget/propose.py` exposes `solve_candidates(source, target, A)`, which returns valid-by-construction candidates with their solver reasons and appeal scores.
- `_prefer_learned` keeps its veto role, and its choice among eligible learned candidates becomes "highest appeal" instead of "greedy first".
- Templates remain the fallback for rule-infeasible designs, and they compete on the same appeal score.
- The current `materialize` and `deoverlap` stay behind a flag until the new path wins the human preference gate.

## 4. Failure mode → fix

| Mode (held-out / panel) | Fixed by | Expected |
|---|---|---|
| Hairline < 1 px (4.0 / 15.5%) | minimum-scale constraint | ~0 where geometrically feasible |
| Overflow (4.6 / 10.7%), below floor (8.8 / 25.7%), small text, hierarchy flips (6.0 / 9.7%) | line options with exact tracking; hard floors and hierarchy | large drop; the remainder becomes explicit "rule-infeasible" on tiny canvases |
| Collision (6.7 / 18%), occlusion (6.1 / 13.5%) | `NoOverlap2D` and overlap bounds as hard constraints, re-arrangement instead of shrinking | ~0 for solved candidates |
| Unsupported text (8.4 / 11.3%), contrast (10.3 / 20.6%) | host containment with release cost; subject respect; final pixel check | about half (the bound candidate shows 2.2% unsupported when binding fits) |
| Alignment loss (17 / 19%) | alignment-set cost, edge discipline | toward the designer's 10% |
| Under-fill / content collapse (5.1 / 10.8%) | fill and coverage priors, scale freedom, bold profile | coverage toward 0.6 |
| No re-arrangement (18% match) | relation booleans with transition priors, arrangement seed | toward the designers' 41% where it helps |
| Unbalanced, cramped or flat-looking layouts (not measured today) | balance, grouping, rhythm, focal and typography terms; appeal-ranked selection | measured by the appeal score and human votes |
| Awkward line breaks, widows | typography cost per line option | measured by the appeal score and human votes |

## 5. Build phases and gates

Every phase is measured on:
- a fixed 2,000-pair held-out subset;
- a fixed 3,000-output panel subset;
- the 50 + 32 rendered sets.

The harness is `work/retarget-scale/` (checks in `common.py`, flags in `flags.py`), reused as-is.

### The human preference gate

- **A blind side-by-side voting page.** It will be published as a private claude.ai artifact with a shared vote store. Each screen shows the brief (source design and target size) and two renders in random left/right order, with no labels.
- **Raters choose** "left better", "right better" or "about the same". They can tick reason tags: hard to read, crowded, misaligned, unbalanced, empty, cut off, other.
- **Designer column.** It uses the **designer's own rendered preview** from Crello, which is a true render, not the reference rebuilt from source layers that misled the automated numbers.
- **Sample per round:** 120 held-out designs, stratified by aspect-change bucket and target format, each shown as two comparisons. That is "new vs today's engine" and "new vs designer".
- **At least 3 votes per pair**, from you plus anyone you invite.
- **Reporting:** win rate with a 95% bootstrap interval, broken-tag rate, and per-bucket breakdowns.

| Phase | Work | Gate to continue |
|---|---|---|
| **P0 Harness and baseline** | Freeze the eval subsets; add `engine_eligible(source, design)`; fetch designer previews for the eval set; build the voting page; **preference round 0**: today's engine vs designer (labels for scorer validation, plus the baseline gap) | baseline reproduced (hard 30.3% / IoU 0.291); ≥ 600 human votes collected |
| **P1 Quick wins in today's pipeline** (safety net) | hairline minimum scale; tracking-aware re-wrap before shrinking; alignment snapping after de-overlap; best-of-k with an alignment term | hard ≤ 20%, align loss ≤ 15%, no IoU loss |
| **P2 Solver core** | structure extraction; CP-SAT with hard constraints and model-likelihood cost (expose per-field logits from `infer.Model.sample`); lazy pairs; verify loop | 100% engine-eligible or explicitly infeasible; IoU ≥ 0.300; p90 ≤ 0.5 s per seed |
| **P3 Structure, priors and aesthetic terms** | hosts, alignment sets, arrangement relations and transition costs, designer priors, the 8 aesthetic terms, the arrangement seed, weight profiles | hard ≤ 8% held-out / ≤ 20% panel; IoU ≥ 0.310; align loss ≤ 12%; coverage ≥ 0.57; manual sheet review shows no new failure type |
| **P4 Appeal scorer** | features model trained on perturbation, designer-resize and round-0 pairs; then the pixel model if approved; validation as in 3.6 | ≥ 70% agreement with held-out human votes; beats the IoU ranker on that agreement |
| **P5 Tuning and selection** | solver weights and profiles tuned to maximise the appeal score at hard ≤ 8% (random/CMA search on 1,000 train designs); appeal-ranked selection plus the OpenJev top-two judge; **preference round 1**: new vs today's engine on val designs | win rate ≥ 60% vs today's engine; broken tags ≤ 8%; appeal agreement re-checked |
| **P6 Engine and final evaluation** | `solve_candidates` in the engine; full `generate` on the 300-pair sample and the 50 + 32 sets; **preference round 2 on the test split, reported once** | **≥ 65% vs today's engine (lower bound > 55%), ≥ 40% vs designer, broken ≤ 5%**; guardrails in section 0 hold |

## 6. Risks

- **Small, noisy human samples.** 120 designs × 3 votes gives a ±6–8 point interval on the win rate. Mitigations: stratified sampling, paired comparisons on the same design, and reporting intervals rather than point estimates. Add a round if a gate is within its interval.
- **Rater taste vs the audience.** Your judgement and your team's are the gold standard here. If the audience differs (e.g. competition judges), include a few of them in round 2.
- **Gaming the appeal score.** Mitigated by validity-only ranking, re-validation at every human round, and a held-out human set the tuning never sees.
- **Aesthetic terms fighting each other.** They are soft, with tuned weights, and validated by preference, not by their own values. The weight profiles keep diversity, so a single bad weighting can't dominate.
- **Text measurement under the linear approximation.** Kerning and wrap points are not exactly linear. The verify loop catches it; the cost is an occasional extra solve.
- **Solver time on 40+ element designs.** Lazy pairs, units instead of elements, and the hint keep it in check. A hard time limit falls back to the best feasible solution, or to today's pipeline. More candidates per design (profiles × seeds) are bounded by latency: the solver runs in parallel and the profiles share hints.
- **Joint vs marginal likelihood.** The decoder chains its fields, so unary costs from the seed's context are an approximation. Several seeds and appeal-ranked selection compensate.
- **Rectangles for non-rectangular shapes.** Diagonal panels and triangles are box-approximated, which is conservative. A later option is alpha-mask occupancy for large shapes.

## 7. Decisions outside post-processing (yours to make)

1. **Readable floors on tiny canvases.** The engine's floors (body 9 px, legal 7 px) are stricter than designers'. On canvases with a short side of 150 px or less, 49% of designs have text under 9 px. The plan keeps them as hard constraints so outputs pass the engine. Relaxing them for small formats is a rubric change.
2. **Cropping non-background photos.** Designers re-crop photos in 60–90% of resizes. The competition rules only allow `fit: cover` on backgrounds, so this plan never crops, and portrait photos in banners will stay narrow. If the rules allow cover on hero photos, the solver supports it with one line.
3. **Pixel appeal model weights.** The pixel scorer needs a pretrained image backbone downloaded once (a few hundred MB). The features model needs nothing new.
4. **Raters.** Who votes in the preference rounds (you, your team, invited judges), and how many votes per round you can commit to; at least 600 per round is planned.
5. **OpenJev server.** The rendered judge needs the local OpenJev server running (`CHAMELEON_OPENJEV_URL`). Without it, selection uses the appeal score alone.
6. **Retraining.** The raw model is roughly equal to stretch on IoU. The solver and appeal layer get the most out of whatever model they are given, but the larger long-term gain is a model trained with these constraints and preferences in the loop: hard examples from this run, TASK_ARRANGE data from transitions, and preference-weighted targets. That is a separate track.
