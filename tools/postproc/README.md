# Post-processing experiments

Tooling behind `docs/retarget/post-processing-plan.md` and `docs/retarget/postproc-results.md`.
It reads data under `work/` (git-ignored): the Crello layouts in `work/retarget-data`, the
fetched layer images and raw rows in `work/retarget-scale`, and it writes runs, frozen subsets
and preference-round renders to `work/postproc`.

| Script | What it does |
|---|---|
| `harness.py` | Gate runs on the frozen subsets (`ho` 2,000 held-out pairs, `panel` 3,000 extreme-format outputs, `strip` 600 extreme strips, `tune` train pairs). Each output gets the engine's own eligibility, reference-free checks, designer agreement and the appeal score. |
| `methods.py` | The post-processing variants compared: `current`, `solver_gated`, `solver_strip`, `final` (appeal-ranked over all candidates). |
| `quick.py` | Fast in-process solver evaluation for weight sweeps. |
| `summarize.py` | Gate summary of run files. |
| `appeal/` | Appeal scorer data (`build.py`), training (`train.py`) and features. The engine copy is `engine/retarget/appeal.py`. |
| `strips/synth.py` | Synthetic extreme-strip training targets (composer + solver, kept when they pass the engine and score well). |
| `prefs/` | Preference rounds: pair selection (`select_rounds.py`), designer previews (`fetch_previews.py`), source thumbnails (`prep_sources.py`), arm renders (`gen_round.py`), and the voting site (`build_site.py`, `site.html`). |
| `scale_common.py`, `scale_run.py` | The full-dataset evaluation's shared checks (from `work/retarget-scale`). |

Example: `PYTHONPATH=. .venv/bin/python tools/postproc/harness.py --method final --subset ho --out work/postproc/runs/final-ho.jsonl`
