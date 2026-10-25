# Engine and product upgrade — October 24, 2026

This upgrade implements independent counterparts to the competitor capabilities reviewed in this chat. Repository presence and feature coverage do not establish equivalent visual quality; that requires a shared held-out evaluation and blind review.

| Capability | Implementation | State |
|---|---|---|
| Multiple arrangement and size variants | `engine/layout.py` | Active; bounded search, preserves an identity candidate |
| Measured glyphs and style-run widths | `engine/typography.py` | Active; mirrors the official converter's base-font wrapping and inclusive run handling |
| Panel/containment/alignment relations | `engine/analysis.py` | Active; flexible panel geometry follows its text |
| Group repair | `engine/repair.py` | Active; bounded collision moves, containment, text refitting |
| CP-SAT | `engine/constraints.py` | Active by default; fixed-size group placement, preserved internal transforms |
| Layout retrieval | `engine/retrieval.py`, `engine/models/layouts.json` | Active; 638 training-family geometry layouts, assignment and prominence guard |
| Pixel checks | `engine/quality.py`, `engine/render_worker.py` | Active; composited estimate and isolated official-renderer shortlist checks |
| Learned roles | `engine/learning.py`, `engine/models/roles.json` | Trained pilot, wired into generation, opt-in pending quality review |
| Preference ranker | `tools/calibrate.py`, `/api/preferences` | Human-label collection and held-out fitting implemented; no human-trained weights yet |
| Local VLM proposals | `engine/vlm.py`, `engine/vlm_worker.py` | Optional adapter/proposal path; no model or projector bundled, native inference unverified here |
| Model packaging | `requirements-vlm.txt`, `Dockerfile.vlm`, `tools/fetch_weights.py` | Explicit optional build-time installation with hash verification; container not built here |
| Reproducible evaluation | `evaluation/benchmark.py`, `evaluation/synthetic.py` | Per-case diagnostics, source families, platform/timing, output snapshots, HTML report |
| Data extraction | `tools/import_corpus.py`, `tools/import_crello.py`, `tools/bootstrap_roles.py` | Family splits, manifests, metadata conversion; approximate reconstruction explicitly flagged |
| Training | `tools/train_roles.py` | Portable model export, grouped holdout, reports; pilot trained on public weak labels |
| Job control | `service/app.py`, `service/workspace.py` | Single-worker queue, progress, cooperative cancellation between targets, bounded backlog |
| Developer interface | Viewer + review workspace | All-candidate inspection, traces, placement locks, saved history, blind A/B labels, evaluation browser |
| Express product prototype | `express_addon/` | Builds; browser previews and supported new-page application code; live Express unverified |

## Validation

- 23 regression/API tests passed during the upgrade, including preservation, opaque IDs, fixed placements, solver behavior, missing models, portable role inference, rich text, path boundaries, history, preferences, and jobs.
- The main React viewer and the separate Express panel build. The viewer completed a browser generation with three rendered results and no captured browser errors.
- An internal no-pixel sweep over three supplied source designs × 12 target formats returned 36/36 structurally valid results, with zero detected text-overflow, new group collision, or contrast-regression cases. This was repeated with the real retrieval library. The sweep is regression evidence on familiar examples, not an unseen-design or human-quality benchmark.
- A live service request produced three candidates with official-renderer contrast diagnostics. The competition runner generated all three sample JSON/PNG pairs; all outputs passed the supplied JSON schema.
- An additional 96-case generated-text stress sweep (eight sources × 12 sizes, pixels disabled) passed all structural checks with zero detected overflow, new collisions, or contrast regressions. This corpus is small and text-focused; it does not replace real-design evaluation.
- The role classifier's held-out accuracy is 62.74% against weak labels; do not compare that directly with a rival's different data, role taxonomy, or label mapping.

## Limits that remain material

The visual checks use sampled backgrounds and conservative rotated bounds, not exact glyph alpha-mask collision detection. CP-SAT moves fixed-size groups; text size alternatives come from the existing arrangement search rather than a single joint solver model. Learned roles are experimental. VLM integration requires testing against a compatible llama.cpp version and permitted weights. Preference fitting needs real human labels. The Express prototype handles a bounded structured-input workflow and rejects unsupported text styling; native extraction and complete style/group round trips remain future product work. Docker and live Express validation were not available evidence for this upgrade.

## Local commands

```sh
.venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8765
npm --prefix viewer run build
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m evaluation.benchmark --name samples --render
.venv/bin/python -m evaluation.synthetic --count 100
.venv/bin/python -m evaluation.benchmark --corpus output/synthetic --name stress
.venv/bin/python -m tools.import_corpus input output/curated
.venv/bin/python -m tools.build_library YOUR_REVIEWED_CORPUS
.venv/bin/python -m tools.bootstrap_roles --shards 2
.venv/bin/python -m tools.train_roles work/role-data/labels.jsonl
.venv/bin/python -m tools.calibrate output/workspace/labels.jsonl
npm --prefix express_addon run build
```

Open http://127.0.0.1:8765 for the built viewer. Use Review workspace for histories, preferences, and evaluation reports. Calibration writes a proposal under `output/calibration/`; review held-out performance before copying it to `engine/models/ranker.json`. Do not call weak synthetic labels human preferences or silently deploy a model because it is trained.
