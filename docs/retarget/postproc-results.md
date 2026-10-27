# Post-processing results

Results of the build in `post-processing-plan.md`: constrained decoding (CP-SAT), a strip mode, and appeal-ranked selection. It is switched on with `CHAMELEON_POSTPROC=solver`; without it the engine behaves as before. Tooling is in `tools/postproc/`.

## What was built

| Piece | Where | What it does |
|---|---|---|
| Constrained decoder | `engine/retarget/solver.py` | Places every unit (rigid group or free text) at once. Cost is a mode-centred fit to the model's distributions plus alignment, order, margin, balance, rhythm, typography and hosting terms. Hard constraints: canvas, no new overlap, containment, readable floors and text fit. A verify loop re-solves against the engine's own collision and occlusion checks, and a relaxation cascade handles infeasible cases. Warm-started from today's output, with a deterministic time budget. |
| Strip mode | `engine/retarget/strip.py`, solver `strip` weights | For aspect ≥ 4 or ≤ 1/4. Builds a reading-order composition along the long axis with role-share font scaling and decorations packed into free gaps. It is a candidate in its own right and the solver's warm start. |
| Appeal scorer | `engine/retarget/appeal.py`, `engine/models/appeal_features.joblib` | Gradient-boosted trees on 72 reference-free features. Trained on 30,000 train-split examples: the designer's resize vs the same resize with one principle broken. On held-out designs it ranks the designer above a flawed version 81% of the time. |
| Selection | `engine/pipeline.py` `_prefer_learned` | Among learned candidates that pass the engine's checks, the highest appeal score wins. Candidates: today's, solver "balanced", solver "bold", strip composer, copy-bound fallback. |
| Preference rounds | `tools/postproc/prefs/`, voting page | A blind side-by-side page with per-viewer random order that stores the winning arm and broken-tags per voter. |
| Synthetic strips | `tools/postproc/strips/synth.py` | Strip targets for train designs that pass the engine and score well, for a later model update. |

## Gate results (post-processing only, frozen subsets)

Held-out designer pairs (`ho`, 2,000 pairs, test/val splits):

| | Today | New | Plan target |
|---|---|---|---|
| **Passes the engine's checks** | 60.5% | **80.3%** | ≥ 92% |
| Alignment loss | 17.0% | 11.9% | |
| Text below readable floor | 9.2% | 4.5% | |
| Overflow | 4.2% | 2.7% | |
| Contrast regression | 9.6% | 4.9% | |
| Text unsupported by its backing | 8.1% | 2.8% | |
| Hidden (under 1 px) | 4.1% | 1.0% | |
| Hierarchy flip | 6.2% | 2.2% | |
| Collision | 6.8% | 9.0% | ↑ (accents/decorations) |
| IoU to designer | 0.291 | 0.281 | ≥ 0.310 (missed) |
| Font error | 0.208 | 0.194 | |
| Median post-processing time | 0.05 s | 14.6 s | |

Extreme strips (`strip`, 600: 300 held-out designs × one wide and one tall strip, e.g. 970×90, 90×800):

| | Today | New |
|---|---|---|
| **Passes the engine's checks** | 24.0% | **44.0%** |
| Text below readable floor | 36.7% | 14.8% |
| Hidden (under 1 px) | 22.7% | 6.2% |
| Hierarchy flip | 17.0% | 11.8% |
| Overflow | 16.0% | 12.3% |
| Collision | 30.5% | 32.8% |
| Occlusion | 17.5% | 20.0% |
| Median post-processing time | 0.09 s | 21.2 s |

Which candidate the appeal selection picked (held-out / strips):
- today's layout: 707 / 139;
- solver balanced: 632 / 136;
- solver bold: 367 / 106;
- copy-bound fallback: 287 / 29;
- strip composer: 6 / 190.

Times are from the harness at 8-way parallel load, so they overstate idle latency.

## What is not met, and why

- **IoU is 0.01 below today's** and still short of the raw model's 0.310. The solver's model term pulls toward the mode of each distribution, but alignment and margin terms and the verify loop move units off it. Appeal selection also does not reward designer agreement. The plan treats IoU as a sanity bar; the human rounds decide whether this matters.
- **Collisions rose** (held-out 6.8% → 9.0%). Most are small accents and decorations touching copy after the solver packs the layout tighter. The verify loop catches collisions only with checked layers.
- **Strips stay hard.** 56% still fail, mostly because some cases cannot satisfy the rules: long single words in 90-px-wide strips, and hairline decorations that must stay ≥ 1 px at a locked aspect. A model trained on strip data, from the synthetic set, is the next lever.
- **Latency is at the limit.** On a near-idle machine, the full engine on 8 test designs took a median 15.0 s (max 34.0 s), against 11.5 s (max 18.8 s) with post-processing off. The plan's guardrail is a 15 s median. Under heavy load it reached 67–71 s. The solver runs once per profile, so dropping the "bold" profile or verify rounds is the lever if the human rounds do not need them.
- **Visual judge.** No OpenJev server was available, so selection uses the appeal score alone, which the plan allows.

## Synthetic strips

From 2,000 train designs, each given one random strip format, 872 targets passed every engine check and were kept (`work/postproc/strips/synthetic.jsonl`). 540 came from the composer and 332 from the solver; every format has 69–154 targets. The other 1,128 had no candidate that passed. These are ready to add as extra (source, target) pairs in the next model update. They are not yet used.

## Human preference (primary target)

There are two voting pages, both private until shared: https://claude.ai/artifact/XnjGXRU4gr2rsWZ7FChGAK (round 0) and https://claude.ai/artifact/B8ZsoHd5VaGeLKgKRsUW3c (rounds 1 and strips). An artifact version holds at most 511 files, so the rounds are split across two pages.

| Round | Arms | Designs | Status |
|---|---|---|---|
| r0 | today's engine vs designer's own render | 101 test | live, no votes yet |
| r1 | new vs today's engine | 88 val | live, no votes yet |
| rs | new vs today's engine, strips | 60 | live, no votes yet |

The primary targets (≥ 65% vs today, ≥ 40% vs designer, ≤ 5% broken) cannot be judged until votes come in. The appeal scorer should also be re-validated against those votes (plan 3.6) before its ranking is trusted further.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python tools/postproc/harness.py --method final_w1 --subset ho --out work/postproc/runs/final-ho.jsonl
PYTHONPATH=. .venv/bin/python tools/postproc/summarize.py work/postproc/runs/current-ho.jsonl work/postproc/runs/final-ho.jsonl
```
