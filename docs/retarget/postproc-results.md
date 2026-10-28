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

## In the app (full engine, 2026-10-27)

The new post-processing is now **on by default**. `CHAMELEON_POSTPROC=off` (or the request option `postproc: off`) switches it off. The full engine's top pick, as the app shows it first, on 200 held-out pairs and 120 extreme strips. Both arms include the fixes below:

| | Held-out, off | Held-out, on | Strips, off | Strips, on |
|---|---|---|---|---|
| Passes every engine check | 88.5% | **91.0%** | 52.5% | **60.0%** |
| A learned layout is picked | 90.5% | 92.5% | 71.7% | 81.7% |
| Text below readable floor | 5.0% | 4.5% | 30.8% | 25.8% |
| Hidden element | 1.0% | 0.0% | 7.5% | 5.8% |
| Mean appeal score | 0.194 | **0.245** | 0.082 | **0.124** |
| Appeal clearly higher (on vs off) | | 112 vs 4 | | 60 vs 7 |
| IoU to designer | 0.290 | 0.278 | | |
| Median time per size (4 jobs in parallel) | 5.5 s | 9.1 s | 9.6 s | 18.4 s |

Fixes made while testing the real app path:
- **Warm renderer** (`engine/render_pool.py`). Each pixel check used to start a new renderer process, and about 1.3 s of every check was the renderer import. Warm worker processes give pixel-identical renders about 6.5× faster. They are replaced after a crash, a timeout or 40 renders. `CHAMELEON_RENDER_POOL=0` restores one process per render. Idle median per size: 15.0 s → 9.8 s with the solver, and 11.5 s → 6.1 s without it.
- **Solver deadline.** One 5 s budget is shared by both styles (`solver_budget`). The worst case went from 16 s to 5 s.
- **Background bleed.** A background photo that peeked out past the panel above it in the source was fitted exactly under that panel, which counts as deleting it. Every learned candidate then failed the rubric gate, and some sizes returned no layout at all. `keep_bleed` re-opens the layer inside the canvas.
- **Wrapping options.** The engine wraps lines without counting letter spacing but measures widths with it, so the valid widths are not monotonic. Bisection skipped the 3–4-line wrappings, and the solver called simple designs infeasible in narrow formats. It now scans a ladder of widths.

Not fixable by post-processing: copy that cannot be readable at all in the target, such as a letter-spaced word in a 160-px-wide strip. Letter spacing and rotation are protected fields.

## Synthetic strips

From 2,000 train designs, each given one random strip format, 872 targets passed every engine check and were kept (`work/postproc/strips/synthetic.jsonl`). 540 came from the composer and 332 from the solver; every format has 69–154 targets. The other 1,128 had no candidate that passed. These are ready to add as extra (source, target) pairs in the next model update. They are not yet used.

## Human preference (primary target)

There are three voting pages, all private until shared with raters as Contributors. An artifact version holds at most 511 files, so the rounds are split across pages.

| Page | Round | Arms | Designs | Votes (2026-10-28) |
|---|---|---|---|---|
| https://claude.ai/artifact/XnjGXRU4gr2rsWZ7FChGAK | r0 | today's engine vs designer | 101 test | 0 |
| https://claude.ai/artifact/B8ZsoHd5VaGeLKgKRsUW3c | r1 | new vs today's engine | 88 val | 0 |
| same | rs | new vs today's engine, strips | 60 | 1 (new won) |
| https://claude.ai/artifact/KYvR4M7jBYQosnWt5ukjPF | r2 | new vs today's engine | 101 test | 0 |
| same | r2d | new vs designer | 101 test | 0 |

How the rounds were made:
- **Every arm uses the same renderer.** The designer arm is the designer's own resize geometry, drawn with the source layers by our renderer (`tools/postproc/prefs/render_designer.py`). Crello's own previews use the true fonts and effects and would give that side away. Round 0 first used those previews; on 2026-10-28 its designer images were replaced with same-renderer ones before any vote.
- **"Today's engine" is the committed engine with post-processing off.** These renders were made before the solver became the default (rounds r0, r1, rs and r2 reuse them).
- **Round 2's new arm was rendered from commit b3d7463** in a clean checkout, without the strip adapter that is still in progress. It passed every engine check in 100 of 101 designs.
- Image file names on the round-2 page are hashed. The key is in `work/postproc/prefs/r2/opaque_names.json`.

The primary targets (≥ 65% vs today, ≥ 40% vs designer, ≤ 5% broken) cannot be judged until votes come in. Round 2 on the test split is the one to report.

## Weight tuning (P5)

`tools/postproc/tune.py` searched the solver's soft-cost weights on 60 train-split designer pairs. It tried 10 settings: the defaults plus 9 random ones within 3× of each weight. The objective was mean appeal, with layouts failing an engine check charged 0.3.

| Setting | Objective | Passes engine checks |
|---|---|---|
| best random (config 6) | 0.126 | 73.3% |
| defaults | 0.121 | 73.3% |
| next three | 0.109–0.113 | 70.0% |

The best setting beats the defaults by 0.005 ± 0.020 (paired, 95%) and is worse on 23 of 60 designs, better on 17. No setting is reliably better, so **the defaults are kept**. That also means the round-2 renders use the final weights. Raw results: `work/postproc/runs/tune.json`.

## Appeal scorer: realistic check

On held-out designs, the scorer ranks a designer's resize above the same resize with one principle broken 81% of the time. A harder check uses 120 val-split pairs (not in any tuning or voting set) and compares the designer's resize with today's output (`tools/postproc/appeal/validate.py`):

| | Result |
|---|---|
| Designer scored above today | 55.8% |
| … when today fails an engine check (51 pairs) | 66.7% |
| … when today passes (69 pairs) | 47.8% |
| Solver scored above today | 58.5% |
| Passes engine checks: today / solver | 57.5% / 61.9% |

This check is confounded: the designer layout fails the engine's checks in every case, because it is rebuilt from the source's layers with unmatched elements stretched. The scorer partly rewards passing checks. Even so, where today's output passes, the scorer cannot tell it from the designer's, so it should not be read as a measure of taste. It ranks candidates mostly by rule health plus coarse layout statistics. Human votes are the only real test of it. Until then, appeal-ranked selection should be read as "pick the cleanest valid candidate".

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python tools/postproc/harness.py --method final_w1 --subset ho --out work/postproc/runs/final-ho.jsonl
PYTHONPATH=. .venv/bin/python tools/postproc/summarize.py work/postproc/runs/current-ho.jsonl work/postproc/runs/final-ho.jsonl
```
