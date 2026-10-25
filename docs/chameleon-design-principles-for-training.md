# Chameleon: Design Principles for Model Training

Prepared October 24, 2026. This document is about building principles into models. Measuring outputs is covered in [chameleon-design-guardrails-and-eval-rubric.md](chameleon-design-guardrails-and-eval-rubric.md), the **Eval Rubric**. Check IDs below (D1, S6, §2.4, …) refer to that document.

## Purpose

Chameleon is mostly deterministic. Models help in four places:

| Model | File | Job | Status |
|---|---|---|---|
| Role classifier | [engine/learning.py](../engine/learning.py), `roles.json` | Label each element: headline, subhead, body, cta, badge, legal, logo, hero, image | Pilot: 62.7% accuracy against noisy labels; off by default |
| Layout retrieval / prior | [engine/retrieval.py](../engine/retrieval.py), `layouts.json` | Suggest arrangements seen in real designs at a similar aspect ratio | 638 layouts |
| Candidate ranker | `preference_score`, `ranker.json`; trained by [training/train_ranker.py](../training/train_ranker.py) | Choose the best of the valid candidates | Trainer ready; weights not deployed until reviewed human preferences exist |
| VLM proposer / critic | [engine/vlm.py](../engine/vlm.py), [engine/neural_proposals.py](../engine/neural_proposals.py) | Propose roles, groups or arrangements from the rendered source; critique renders | Experimental; no weights shipped |

This document explains how to make each model *want* the same things the Eval Rubric checks, and how to stop any model from learning to break the rules.

## Core principle: models propose, rules decide

1. **A restricted action space.** No model may output anything except: element roles and groups (as hints), and a layout *plan* of block positions, sizes and font sizes. The serializer applies these changes to an untouched copy of the source. That makes D1–D4 and S7/S9 impossible to violate through the model, by construction.
2. **The gates run after every model.** Every model output goes through Tier 0–1 validation and the Tier 2 defect checks. A model can rank or propose. It can never override a gate.
3. **Missing weights are neutral.** If a model is missing or incompatible, the deterministic path runs unchanged. This is the current behaviour, and it must stay that way.
4. **Never train on the eval set.** Benchmark families, the three public samples and gold-set items are kept out of all training data.

---

## 1. Principles as training signals

Each principle in the Eval Rubric becomes something a model can learn from. The table shows the principle, the kind of signal, and where it is used.

| Principle (Eval §) | Signal | Kind | Used by |
|---|---|---|---|
| No regeneration, no copy/font change (D1–D3) | Not learnable; enforced by the action space | Constraint | All |
| Every element exactly once (D4) | Plan must cover all block IDs; penalty for a missing or duplicate ID | Verifiable reward / parse check | VLM proposer |
| Inside canvas, no distortion (S5–S6) | Fraction of boxes inside; aspect error | Verifiable reward | VLM proposer, ranker features |
| Text fit (§2.1) | Rendered overflow = 0 using the real fonts | Verifiable reward | Proposer, ranker |
| Readable size (§2.1) | `min(1, fs/ideal)^1.5`, weighted by role | Dense reward | Proposer, ranker |
| Contrast (§2.2) | WCAG ratio on composited pixels; regression flag | Verifiable reward | Proposer, ranker |
| Hierarchy order (§2.3) | Share of source size orderings kept | Verifiable reward | Proposer, ranker |
| Hierarchy semantics (§2.3) | Role labels: headline vs subhead vs legal | Supervised label | Role classifier |
| Grouping (§2.4) | Rigid-group offset error; proximity ratio | Verifiable reward + supervised group labels | Proposer, role/group model |
| Unintended overlap (§2.5) | Overlap IoU between non-source-overlapping blocks | Verifiable reward | Proposer, ranker |
| Alignment (§2.6) | Aligned-edge count minus near-miss count | Dense reward | Proposer, ranker |
| White space / balance (§2.7) | Coverage in band, centroid distance, margin floor | Dense reward | Proposer, ranker |
| Image presentation (§2.8) | Salient-box retention under crop, area share | Verifiable reward | Proposer |
| Fidelity (§2.9) | Reading-order agreement with aspect tolerance | Dense reward | Proposer, ranker |
| "Looks designed" (Tier 3) | Pairwise human preference | Preference label | Ranker, DPO |

Every signal is computed by **the same code** the evaluator uses ([engine/metrics.py](../engine/metrics.py), [engine/quality.py](../engine/quality.py), [engine/judging.py](../engine/judging.py), [engine/invariants.py](../engine/invariants.py)). If training rewards and eval checks are coded separately, they will drift apart, and the model will learn to exploit the gap.

---

## 2. Data

### 2.1 Sources

| Source | Use | Licence / caution |
|---|---|---|
| [Crello](https://huggingface.co/datasets/cyberagent/crello) | Real multi-element designs with sizes, types and fonts | CDLA-Permissive-2.0; the artwork terms in the dataset card also apply |
| [DesignAsCode](https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data) | Machine-distilled group/theme labels | MIT declared. Labels are **noisy**: treat as weak supervision only |
| [DesignSense-10k](https://arxiv.org/abs/2602.23438) | 10,235 human preference pairs for layouts | Check the licence before use. Could pre-train a ranker; no importer is built yet |
| Our synthetic generator ([evaluation/synthetic.py](../evaluation/synthetic.py)) | Controlled densities and structures | Must not copy the public samples |
| Our own human labels (§3) | Preferences and role corrections | The highest-value data we have |

Record every dataset revision and hash in [engine/models/data-manifest.json](../engine/models/data-manifest.json).

### 2.2 Splits

- Split by **design family** (copies, translations and size variants of one design stay together). Never split by individual design.
- Use the same family partition across all models. The retrieval library already excludes the role classifier's validation families. Keep it that way.
- Hold out a **test** partition that no model, prompt or threshold is tuned on. The benchmark draws from it.

### 2.3 Multi-size families are the best supervision

Crello and similar sources often contain the *same* design at several sizes. When one exists, a designer's own resize is a positive example of the task:

- input: size A version + target size B;
- reference: the designer's size-B version (geometry and font sizes per matched element);
- match elements by content, asset and type, not by ID.

Only keep pairs where the matching is unambiguous, all elements match, and copy and assets are identical. That makes the pair a valid stage-one example.

### 2.4 Negatives built from the guardrails

Positives teach what is typical. Negatives teach what is wrong. For each source design, generate perturbed resizes. Each perturbation violates exactly one principle and is labelled with it:

| Perturbation | Violates | Severity levels |
|---|---|---|
| Shift one block's x by 1 px – 1% / 2–5% / 10% | Alignment (§2.6) | small / medium / large |
| Scale font up until it overflows | Text fit (§2.1) | 1 line / 2 lines / clipped word |
| Shrink body text below its floor | Readability (§2.1) | 0.9× / 0.7× / 0.5× floor |
| Swap headline and subhead sizes | Hierarchy (§2.3) | |
| Pull a button label 20–60 px off its shape | Grouping (§2.4) | |
| Move text over a busy region of a photo | Contrast (§2.2) | |
| Overlap two text blocks by 10–50% | Overlap (§2.5) | |
| Non-uniformly stretch a product image by 5–20% | Image integrity (S6) | |
| Uniformly shrink the source into the target (letterbox) | Composition (§2.7, Tier 3) | |
| Push content into one corner | Balance (§2.7) | |
| Place a CTA in a platform's UI overlay zone | Safe zone (§2.7) | |

This follows Haraguchi et al. (2024), who built judge data by perturbing x-coordinates and font sizes at small, medium and large levels. Keep the severity labels, because small perturbations are exactly where AI judges fail. Ranker training needs pairs at that level of difficulty.

### 2.5 Augmentation

- Re-target each source to every size in the target catalogue (Eval Rubric, benchmark section). Hidden targets are real platform formats.
- Jitter canvas size by ±5% around catalogue sizes so the model does not memorise exact dimensions.
- **Do not** augment by flipping images, recolouring or editing assets. Train on what the rules allow.
- Never treat element IDs as semantic features. Hidden inputs are not guaranteed meaningful IDs. Randomise IDs in training data.

---

## 3. Human labelling guidelines

Human preference labels are the only signal for "looks designed". They are expensive, so collect them carefully.

### 3.1 Task format

Each item shows the source render, then two resizes (A and B) at the same target size, with the format name. The annotator answers:

1. **Preference:** A / B / equivalent / both unacceptable.
2. **Main reason** (one choice): legibility, hierarchy, grouping, composition, imagery, alignment, faithfulness.
3. **Defects** in the loser (multi-select, from the same list), with optional click-to-mark.
4. For single-item rounds: 1–10 per criterion, using the Eval Rubric anchors (10 flawless / 7 minor issues / 4 obvious shortcomings / 1–2 broken).

### 3.2 Annotator instructions

> You are judging a resize of an existing design, not designing from scratch. The tool can only move and scale elements and change font sizes. Prefer the version a professional designer would ship for this format. In order of importance:
>
> 1. Can every piece of text be read at this size? Is anything cut off?
> 2. Is it obvious what to read first, second and last? Is the CTA findable?
> 3. Do things that belong together stay together?
> 4. Does the layout use this shape well, or does it look like the original squeezed in?
> 5. Are the product and logo clear, uncropped and undistorted?
> 6. Do edges line up and gaps look even?
> 7. Is it still recognisably the same design and message?
>
> Do not reward a version for being "more creative" if it changes the message or focal point. If both are broken, say so; don't pick the less bad one.

### 3.3 Quality control

- ≥ 3 annotators per item. Keep each item's agreement score.
- 10% of items are known-answer checks (a large perturbation vs its clean original). Drop annotators below 90% on them.
- Randomise A/B position. Hide which engine or model produced which.
- Report human–human Cohen's κ. That is the ceiling for any model or judge trained on these labels.

---

## 4. Training recipes

### 4.1 Role and group classifier

**Goal:** better role inference from geometry, typography, z-order and groups. Role errors cause many downstream layout errors (hierarchy weights, floors, hero detection).

- **Inputs:** the 16 `FEATURE_NAMES` in [engine/learning.py](../engine/learning.py), optionally plus neighbour features (relative position to the largest text, containment in a shape, shared `group_id`).
- **Labels:** weak labels from DesignAsCode/Crello, corrected by human review on a 1–2k element sample. Report accuracy on the **human-corrected** subset separately from accuracy on the weak labels.
- **Model:** keep the portable tree ensemble (JSON inference, no runtime dependencies). Use class weights so rare roles (cta, legal, logo) are not swamped by body text.
- **Principles built in:**
  - at most one primary headline, hero and logo per design (already enforced after inference);
  - text roles only for text elements, image roles only for images (already enforced);
  - abstain below the confidence gates (0.8 confidence, 0.3 margin) rather than guess.
- **Metric that matters:** the change in benchmark Tier 2/3 when the model is enabled, not classifier accuracy. Only enable it by default when the benchmark improves in every aspect bucket.

### 4.2 Candidate ranker

**Goal:** when several valid candidates exist, pick the one people prefer. Use this when good candidates exist but the wrong one wins.

- **Features:** `RANK_NAMES` in [engine/learning.py](../engine/learning.py): readability, hierarchy, composition, fidelity, coverage, fit, collision_free, contrast, plus the rubric's grouping, alignment, space and imagery scores, warning count and near-miss count.
- **Model:** a linear Bradley–Terry / pairwise logistic model on feature *differences*, `P(A ≻ B) = σ(w·(x_A − x_B))`. It is small, interpretable, publishable as JSON, and matches the existing `ranker.json` format (`weights`, `mean`, `scale`).
- **Data:** our human pairs (§3), plus DesignSense-10k pairs for pre-training if the licence allows, plus perturbation pairs (§2.4: clean ≻ perturbed). Down-weight synthetic pairs relative to human ones (e.g. 0.3×).
- **Constraints built in:**
  - train only on pairs where *both* candidates pass Tier 0–1. The ranker never sees invalid candidates, and the pipeline never offers it any;
  - constrain the weights on readability, fit, collision_free, contrast and grouping (`NONNEGATIVE`) to be ≥ 0, so the ranker can never prefer more defects;
  - check the learned weights. A negative readability weight means the data or the features are broken.
- **Acceptance:** pairwise accuracy on held-out human pairs ≥ human–human agreement − 5 points, and benchmark pairwise wins ≥ losses vs the deterministic ranking in every bucket.

### 4.3 VLM proposer: supervised fine-tuning

**Goal:** propose arrangements (and optionally roles and groups) for hard aspect changes, where the rule-based candidates are weak.

- **Base:** a small open VLM that runs locally within the reference-machine budget. Use LoRA. **The adapter must be published** (e.g. a public Hugging Face repo), per [RULES.md](../RULES.md). Weights must be bundled in the image; there are no runtime downloads.
- **Input:** the source render (letterboxed to 672×672 by `prepare_image` in [engine/vlm.py](../engine/vlm.py)) and the prompt from `plan_prompt` in [engine/neural_proposals.py](../engine/neural_proposals.py). The prompt contains the constitution, the target size and format name, and a block table (`block_table`: id, role, kind, member count, normalised source box, font size, character count, line count). Actual copy is never included, only its length. This prevents learning to rewrite. Runtime and training use exactly the same prompt and image preparation.
- **Output:** a JSON plan in a fixed schema:

```json
{
  "arrangement": "side_by_side | stack | banner_row | anchored | source_like",
  "groups": [
    {"id": "b3", "x": 0.04, "y": 0.10, "w": 0.42, "h": 0.80}
  ],
  "rationale": "hero left, headline+cta stacked right; logo top-right"
}
```

  This is `PLAN_SCHEMA` in [engine/constitution.py](../engine/constitution.py). Coordinates are normalised to the target. `apply_plan` turns a plan into a design. A lone text block takes the planned box and the largest font that fits it. Multi-element blocks scale uniformly into their box. The engine then repairs and validates the result at runtime. Rewards are computed *without* repair, so the model cannot learn to depend on it.
- **SFT targets:** designer resizes from multi-size families (§2.3), then the best engine candidate *as a plan*. Each candidate is converted with `design_to_plan`, rebuilt with `apply_plan` and rescored, and it is kept only if the rebuilt design passes every rubric gate with Tier 2 overall ≥ 85. Never use a target that failed any gate.
- **Loss masking:** compute loss only on plan tokens.

### 4.4 VLM proposer: preference and reward optimisation

After SFT, push the proposer toward plans that pass checks and that people prefer.

**Option A — DPO on ranked plans** (as in AesthetiQ, CVPR 2025). Take up to 8 gated engine candidates per input as plans. Rebuild each with `apply_plan` and score it with the GRPO reward below. Discard any that fail Tier 0–1. Pair the best clean plan with up to two plans whose reward is at least 0.05 lower. Once calibrated, the judge or ranker can replace the reward for ordering.

**Option B — GRPO with verifiable rewards** (as in AeSlides, 2026, which raised aspect-ratio compliance from 36% to 85% with ~5k prompts, and LaySPA's geometric + aesthetic dual rewards). The reward for a sampled plan:

```text
if plan fails to parse, misses/duplicates a block, or fails any Tier 0–1 check:
    R = -1
else:
    R = gate × (0.30·readability + 0.20·hierarchy + 0.30·composition + 0.20·fidelity) / 100
        + 0.15·alignment − 0.10·near_misses − 0.10·margin_violations
        + λ·ranker_or_judge_preference          (λ ≈ 0.2, only after calibration)
    gate = 0.25 if any Tier 2 defect else 1.0
```

This is `reward_from` in [training/reward.py](../training/reward.py). It imports `WEIGHTS` from [engine/metrics.py](../engine/metrics.py), so the two cannot drift apart. `grpo_reward` adapts it to TRL's reward-function signature.

**Reward-hacking risks and counter-measures:**

| Exploit the model will find | Counter-measure |
|---|---|
| Shrink all text to avoid overflow and overlap | Readability term with role floors; hierarchy presence term; the gate on unreadable headlines |
| Shrink everything into a small, safe, centred cluster | Coverage band (35–85%); minimum image presence (§2.8); the letterbox negative in DPO pairs |
| Push hard elements off-canvas or under others | Containment and D5 visibility as hard −1 |
| Copy the source layout and scale it (high fidelity, poor fit) | The fidelity tolerance grows with aspect change; Tier 3 composition; letterbox negatives |
| Exploit judge blind spots (fine alignment) | Code-based alignment terms; judge weight λ kept small |
| Overfit to catalogue sizes | Size jitter (§2.5); held-out formats in eval |
| Mode-collapse to one arrangement | Track the arrangement-type distribution per aspect bucket; add an entropy bonus or sample per arrangement |

### 4.5 VLM critic (optional)

A separate model, or the same base model with a separate adapter, that takes a render and returns the Eval Rubric's Tier 3 JSON. Train it on human single-item scores and defect marks (§3.1). Use it to target bounded repairs ("headline crowds CTA" → widen the gap), never to accept an output. It must not be the same model instance that proposes layouts, because of self-preference bias.

---

## 5. Design constitution for prompts

Put this text into every proposer and critic prompt, as the system prompt or as a preamble. It is short on purpose. The checks live in code.

```text
You adapt an existing design to a new canvas. You may only move and resize the
given blocks and scale their fonts. Never change, drop, or add copy, fonts,
colours, or images. Every block must appear exactly once, fully on canvas.

Priorities, in order:
1. Legible: all copy readable at target size; nothing clipped; text on calm areas.
2. Clear hierarchy: the headline (or hero) leads; supporting copy, then CTA follow.
   Keep the original size order. Use few size levels.
3. Grouped: keep units together (button = shape + label; caption with image).
   Space within a group is smaller than space between groups.
4. Fit the shape: re-arrange for the new aspect ratio (side-by-side for wide,
   stacked for tall) rather than shrinking the original into it.
5. Images intact: no stretching; crop only backgrounds; keep the subject visible.
6. Tidy: align edges, consistent margins and gaps, balanced visual weight,
   keep important content out of platform UI zones.
7. Faithful: same message, focal point and reading order where the shape allows.
```

---

## 6. Evaluating during training

- After each checkpoint, run the **full Eval Rubric** on the validation families and a fixed subset of the benchmark. Track Tier 0–1 failure rate, Tier 2 defect rate per dimension, Tier 3 pairwise result vs the deterministic engine, and runtime.
- Stop early if the Tier 0–1 failure rate on *raw* model plans (before repair) rises, even when post-repair scores improve. That means the model is relying on repair.
- Slice every metric by aspect bucket. Most of a model's value should come from the large and extreme buckets. If it only helps same-shape resizes, it is not worth the runtime.
- Keep a qualitative grid: the same 24 source × target pairs, rendered every checkpoint, for a human to look at.

## 7. Shipping a model

- [ ] Trained only on train-partition families. Manifest updated with dataset revisions and hashes.
- [ ] Weights bundled in the image. No runtime downloads. Runs offline in Docker.
- [ ] LoRA or other adapted weights published publicly, with the link in the README.
- [ ] Fits the reference-machine budget (provisional: Apple M5 Pro). Model inference time per task measured and reported.
- [ ] Disabled cleanly when weights are missing (neutral fallback tested).
- [ ] Benchmark release gate in the Eval Rubric passes with the model enabled, and is no worse than with it disabled in every aspect bucket.

## 8. Commands

The whole pipeline, from data to a trained adapter. Every dataset step only emits rows that pass the rubric gates.

```bash
# Signals and negatives (§2.4): ranker pairs plus a detection report per violation
python -m training.perturb --corpus output/crello/designs --out output/training/perturb
# Designer resizes (§2.3)
python -m training.families output/crello/designs --out output/training/families.jsonl
# VLM datasets (§4.3–4.4): sft / dpo / grpo jsonl + images + manifest; test families and public samples excluded
python -m training.export output/crello/designs --families output/training/families.jsonl --out output/training/vlm
# Validate a dataset without ML dependencies
python -m training.train_vlm sft output/training/vlm --dry-run
# Ranker (§4.2): human labels from the review workspace plus synthetic pairs
python -m training.train_ranker output/workspace/labels.jsonl --synthetic output/training/perturb/pairs.jsonl
# LoRA training (requirements-train.txt): SFT, then DPO or GRPO
python -m training.train_vlm sft  output/training/vlm --base Qwen/Qwen2.5-VL-3B-Instruct --out work/vlm-sft --merge
python -m training.train_vlm grpo output/training/vlm --base work/vlm-sft/merged --out work/vlm-grpo
```

State as of October 24, 2026:

- **Ranker.** The development fit on 606 synthetic pairs from 12 families scored 0.84 held-out synthetic accuracy. It is not deployable: there are no human labels yet.
- **Families.** The 60 local Crello designs contain no multi-size families, so `training.families` currently finds 0 pairs. The importer has to pull more Crello shards first.
- **LoRA trainer.** The code path was smoke-tested for SFT, DPO and GRPO with TRL 1.16 on `trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration`, a random-weight test model, for two steps each. No real model has been trained.
- **GRPO batching.** GRPO groups one prompt's generations per batch. Mixing prompt lengths mis-pads `mm_token_type_ids` for Qwen-VL in that TRL version.
- **GRPO on Apple MPS.** GRPO stalled in a GPU sync during the smoke run. Use `--cpu` there, or train GRPO on CUDA. The CPU smoke run finished, with loss 0 as expected: the random test model emits no valid plans, so every reward is -1 and there is no advantage signal.
- **Runtime use.** A trained adapter must be merged, converted to GGUF plus a projector, and published before the engine's llama.cpp backend can use it.

## Sources

- Preference alignment for layouts: [AesthetiQ (CVPR 2025)](https://arxiv.org/html/2503.00591v1); verifiable layout rewards: [AeSlides (2026)](https://arxiv.org/pdf/2604.22840), [LaySPA](https://arxiv.org/html/2509.16891v3); layout reasoning and the limits of holistic aesthetic rewards: [PosterCopilot](https://arxiv.org/pdf/2512.04082); human layout preference data: [DesignSense-10k](https://arxiv.org/abs/2602.23438).
- Perturbation-based judge data and AI-judge limits: [Haraguchi et al., 2024](https://arxiv.org/html/2410.08885v1). Energy-based layout principles: [O'Donovan et al., 2014](https://www.dgp.toronto.edu/~donovan/layout/).
- Content-aware layout metrics (readability, occlusion, underlay): [RALF](https://arxiv.org/pdf/2311.13602), [LayoutDM](https://arxiv.org/pdf/2305.02567), [PosterLlama](https://arxiv.org/pdf/2404.00995).
- Design principles: [NN/g visual design principles](https://www.nngroup.com/articles/principles-visual-design/), [NN/g common region](https://www.nngroup.com/articles/common-region/), [Butterick's Practical Typography](https://practicaltypography.com/), [Material spacing](https://material.io/design/layout/spacing-methods.html), WCAG 2.2 via [WebAIM](https://webaim.org/articles/contrast/).
- Judge bias and calibration: [position bias](https://www.metacto.com/blogs/llm-as-judge-production-evaluation), [calibration targets](https://futureagi.com/blog/llm-as-judge-best-practices-2026).
- Project rules and artifacts: [RULES.md](../RULES.md), [engine/models/README.md](../engine/models/README.md).
