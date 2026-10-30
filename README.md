# Chameleon — adaptive layout engine

Retargets structured designs while preserving every element, text string, font family, original asset reference, and non-layout property. The pipeline searches arrangements and scales, fits measured typography, repairs groups, optimizes placement with CP-SAT, retrieves training geometry, checks composited/rendered contrast, and returns ranked editable JSON candidates.

**Submission entry point:** `python -m final.main` (the Docker image), which combines every approach in this repository. See [final/README.md](final/README.md).

## Run locally

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
npm --prefix viewer install
npm --prefix viewer run build
.venv/bin/uvicorn service.app:app --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765**. For frontend development, `npm --prefix viewer run dev` starts port 5180 with the API proxy. The competition interface remains `python -m engine.main` and runs without network access.

The viewer also offers an experimental **Workflow** selector for one-design refinement, composition recovery, or a small shortlist. These modes keep a verified checkpoint, save edit history, and support canonical geometry corrections with undo/redo and revision-specific feedback. The optional joint solver chooses measured text configurations as well as positions. Legacy generation and placement remain defaults until independent visual evidence supports promotion. Local jobs, partial results and feedback persist in SQLite; generation runs in supervised worker processes.

Implementation scope, visual experiments and limitations are recorded in [the Codex execution report](docs/individual-update-plans/22-codex-execution-report.md). Human feedback can be exported with `python -m tools.export_preferences --out output/human-preferences/labels.jsonl`; saving feedback does not automatically train or deploy a model.

The viewer includes a [Crello browser](docs/crello-browser.md) with 60 locally imported validation designs, original layer images and fonts, preview comparison, search and filters. Rebuild it with `.venv/bin/python -m tools.build_crello_catalog --download --limit 60` and restart the service.

The viewer supports sample/custom JSON designs, custom sizes, headline/product emphasis, placement locks, all-candidate inspection, diagnostics, repair traces, job progress/cancellation, saved history, blind preference labels, and evaluation reports. Review workspace is available from the header.

## Engine and model status

| Component | Status |
|---|---|
| Arrangements, typography, preservation checks | Active |
| Group repair, CP-SAT placement, visual diagnostics | Active |
| Retrieval | Active; 638 geometry-only training-family layouts |
| Learned role suggestions | Trained 48-tree pilot, opt-in through viewer or `CHAMELEON_ROLE_MODEL=1` |
| Preference learning | Human-label collection and held-out fitting available; no human-trained ranker deployed |
| Local visual model | Optional isolated proposal adapter; requires compatible native library, permitted weights and projector |
| Express panel | Separate prototype in `express_addon/`; builds, live Adobe application unverified |

The role pilot was trained on 1,256 public designs joined to machine-distilled descriptions and achieves 62.74% accuracy against held-out weak labels. This does not measure human layout quality. Dataset revisions, hashes, split policy, and attribution are in `engine/models/`. Neural inference makes no automatic downloads. See [feature coverage and validation](docs/feature-upgrade-2026-10-24.md) for precise limits.

## Evaluation, data, and training

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m evaluation.benchmark --name samples --render
.venv/bin/python -m evaluation.synthetic --count 100
.venv/bin/python -m evaluation.benchmark --corpus output/synthetic --name stress
.venv/bin/python -m tools.import_corpus YOUR_JSON_CORPUS output/curated
.venv/bin/python -m tools.build_library YOUR_REVIEWED_TRAINING_CORPUS
.venv/bin/python -m tools.bootstrap_roles --shards 2
.venv/bin/python -m tools.train_roles work/role-data/labels.jsonl
.venv/bin/python -m tools.calibrate output/workspace/labels.jsonl --synthetic output/training/perturb/pairs.jsonl
```

### Guardrails, judging, and model training

The design rules in [docs/chameleon-design-guardrails-and-eval-rubric.md](docs/chameleon-design-guardrails-and-eval-rubric.md) are implemented in `engine/rubric.py`. After the first pass, `engine/refine.py` takes the three best distinct candidates and hill-climbs each against its rubric issues (margins, alignment, small text, collisions, gaps, groups, type order). It keeps a change only if the candidate ranks strictly higher, and it never trades in a release-blocking defect. Disable it with the viewer's *Refine best 3* toggle or `options={'refine': False}`; `refine_seconds` sets the time budget (default 12 s). Every candidate carries a tiered `rubric` report. Candidates failing Tier 0 (disqualifiers) or Tier 1 (structure) rank last, and Tier 2 design defects count against ranking.

```sh
.venv/bin/python -m evaluation.benchmark --name release --render --offline   # catalogue sizes, tiers, aspect buckets
.venv/bin/pip install -r requirements-eval.txt                               # Tier 3 judge (development only)
.venv/bin/python -m evaluation.judge compare output/evaluation/release output/evaluation/previous
.venv/bin/python -m evaluation.release_gate output/evaluation/release --baseline output/evaluation/previous
```

Training follows [docs/chameleon-design-principles-for-training.md](docs/chameleon-design-principles-for-training.md). Signals come from the same rubric code. See section 8 there for the full command sequence (`training.perturb`, `training.families`, `training.export`, `training.train_ranker`, `training.train_vlm`; ML dependencies in `requirements-train.txt`).

Runs live in ignored `output/`. The benchmark produces case snapshots, separate structural/text/collision/contrast counts, timing, and an HTML report. Human calibration exports a proposed ranker to `output/calibration/`; review the family-held-out results before deploying it as `engine/models/ranker.json`. Metadata-only Crello extraction is available through `python -m tools.import_crello --help`; it never substitutes images or silently claims faithful reconstruction.

For local VLM experiments, install `requirements-vlm.txt`, provide `CHAMELEON_VLM_MODEL` and `CHAMELEON_VLM_MMPROJ`, then enable `CHAMELEON_VLM=1`. This native backend was not validated with real model weights during the upgrade. `tools.fetch_weights` requires an HTTPS URL and SHA256 for explicit build-time downloads; `Dockerfile.vlm` is an optional packaging path. Confirm challenge model eligibility before submission. The default Dockerfile ships the offline deterministic engine and small local artifacts.

---

# Starter repo notes

# CHAMELEON — starter repo

This is the scaffolding for your engine. It handles input loading, output
writing, rendering, and Docker packaging — so the only file you need to
edit is **`engine/resize.py`**.

```
input:  a task JSON (source design + target canvas)
              |
              v
        engine/resize.py   <-- YOU IMPLEMENT THIS
              |
              v
output: <task>_output.json  (schema-valid resized design)
        <task>_output.png   (rendered preview)
```

Both the JSON and the PNG are what get evaluated — see `RULES.md`.

## Repo layout

```
engine/
  resize.py        <- implement your approach here
  main.py           <- orchestrator (input -> resize() -> output). Don't need to edit this.
renderer/           <- the real renderer (transformer-convertor + cr-renderer), already vendored
brand_kit/
  assets/           <- image files referenced by asset_id in the sample designs
  fonts/            <- font files matching the schema's font_family values
input/               <- drop task JSON files here for local testing (3 samples included)
output/              <- results land here
chameleon_design_schema_v0.9.json   <- full field-by-field spec for the design format
Dockerfile
RULES.md
```

## Quick start (no Docker needed for iterating on `resize()`)

```bash
pip install -r requirements.txt
python -m engine.main
```

This processes every `*.json` file in `input/` (the 3 sample designs are
already there — one per difficulty level) and writes an `_output.json` and
`_output.png` for each into `output/`.

The included baseline is a **uniform stretch** — deliberately the wrong
approach (see the main brief: "a naive program stretches the picture; a
good one moves the furniture"). It's there so the pipeline runs on day
one. Run it once, look at `output/*_output.png`, and you'll see exactly
why it's not good enough — that's the point. Replace the body of
`resize()` in `engine/resize.py`, not just its constants.

## What the self-check does (and doesn't do)

`main.py` runs a quick structural check after every task — same element
`id`s as the source, required fields present, nothing off-canvas. **This
is not the official scorer.** It exists to catch obvious mistakes before
you waste a submission on them. See `RULES.md` for what's actually
checked and graded.

A self-check warning does not fail the run — your JSON and PNG are still
written either way. Read the warning; decide whether it's a real problem
or, like the Level 2 sample's intentionally corner-bleeding decorative
circle, an expected one.

## The renderer

`main.py` calls `renderer/converter/convert.py`'s `render_design_to_png()`
(a vendored copy of `transformer-convertor`, wrapping `cr-renderer`) to
produce each task's PNG — already wired in, no setup needed. It reads
fonts/assets from `brand_kit/` and writes a real rendered preview, not a
placeholder. If you ever see plain labeled boxes instead of real text and
images in `output/*.png`, that means the import in `main.py` failed (e.g.
`cr-renderer` didn't install — check `requirements.txt` installed cleanly)
and it silently fell back to the box-and-label placeholder; don't judge
your layout quality by that fallback.

## Docker (how you'll actually be run)

```bash
docker build --platform linux/amd64 -t my-chameleon-engine .
docker run --rm --network none \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  my-chameleon-engine
```

`--network none` matches how stage-one scoring runs: everything your
engine needs must be installed at **build** time (add it to
`requirements.txt`), not fetched at run time. If you're using a locally
hosted model, bake its weights into the image during the build step.

## See also

- `chameleon_design_schema_v0.9.json` — full input/output field spec
- `RULES.md` — what's checked, what's graded, and what disqualifies a submission
- FAQ — for anything not covered above
