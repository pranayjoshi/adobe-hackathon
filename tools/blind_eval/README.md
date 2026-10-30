# Blind judge: Pranay vs Gaurav vs Merged

Three people (Pranay, Gaurav, Kavya) compare three resize engines without knowing which is which: ours, Gaurav's, and the merged engine (`merged/`), which builds Gaurav's strengths into ours.

## The study

- **200 designs** from the Crello validation split (CDLA-Permissive-2.0), with their original fonts and image layers. Imported to `work/blind-eval/crello` and sampled by a seeded hash, so the set is stable.
- **5 target sizes per design, chosen from the 14 app presets** (`lite/app.py`). The picker is greedy farthest-point in (log aspect ratio, log area) space. It starts from the source's own shape, so each design gets the 5 sizes that differ most from the original and from each other. Sizes with the source's aspect ratio are skipped (a plain scale is not a design decision). A coverage penalty keeps the two most extreme ad sizes from taking every slot, so all 14 appear: 33 to 152 uses each.
- **1,000 tasks, three options each.** Each engine runs in its own interpreter: ours is a snapshot of `HEAD` in `work/blind-eval/engines/pranay`, and Gaurav's is his `main` checkout (`4d76410`, v12, since round 2). Gaurav's runs record his commit, so pulling his repository makes them stale and they are redone. Both use the same renderer and the same brand kit (fonts and assets), so neither one loses points to a missing font. Every output is rendered by the same renderer.
- **The merged engine** runs on the same snapshot of our engine with `merged/` added (see `merged/README.md`).
- When an engine refuses a size (ours raises `NoValidDesign` rather than emit a layout that fails its guardrails), the judges see "no output" in that slot.

The A/B/C order is shuffled for each task by a seeded draw. The tasks come in one seeded shuffled order, so the same size or design never comes several times in a row. When two engines produce the same picture (the merged engine often keeps our layout), a vote for it counts for both.

**Keeping pictures and names straight:**
- Image URLs are `/img/<task>/<letter>?v=<picture fingerprint>`. File names never reveal the engine, and a browser cache can never show an old picture under a new letter. The server refuses a fingerprint that doesn't match.
- Each render is stamped with the fingerprint of the layout it was drawn from, and is redrawn whenever that layout changes.
- Votes are stored by the engines they credit, so a rebuilt study never moves a past vote to another picture.
- The build ends with `verify`, which checks every option against its engine's run file (layout, picture, canvas and element ids) and re-renders a sample to confirm the pictures. It exits with an error on any mismatch.

## Rounds

- **Round 1** (Gaurav v10 at `ed11607`, merged without the judges' model): 600 votes on the core 200. Archived with its runs in `work/blind-eval/archive/study-v3-gaurav-v10`, and used to fit `merged/prefs_weights.json`:

  ```bash
  .venv/bin/python -m tools.blind_eval.fit_prefs work/blind-eval/archive/study-v3-gaurav-v10
  ```
- **Round 2** (Gaurav v12, merged choosing with the judges' model): a fresh vote database. Its core set is chosen again for the new merged version. The model was fitted on the same 200 designs, so a round on unseen designs is the clean test.

## Build

```bash
.venv/bin/python -m tools.build_crello_catalog --limit 280 --output work/blind-eval/crello
.venv/bin/python -m tools.blind_eval.build --gaurav ../chameleon-competitors/gaurav/chameleon
```

Every stage resumes and skips work already done. Each merged run records a fingerprint of `merged/*.py` and its switches (`MERGED_GROWTH`, `MERGED_SCALE_SPACING`), so a run from other code or settings counts as unfinished and is redone. `--publish` renders and publishes only the tasks every engine has finished. The app picks up a new `pairs.json` without a restart, so judging can start while the engines are still running.

## Judge

```bash
.venv/bin/uvicorn tools.blind_eval.app:app --host 0.0.0.0 --port 8790
```

Open `http://<this-mac's-LAN-IP>:8790` (it works from any machine on the same network). The page opens straight onto a task: the original design, then options A, B and C. Below them is one row for each judge. The three judges vote together on each task: **A best / B best / C best / no clear winner / all bad**. Once all three have voted, the page moves on to the next task that still needs votes.

- Keys: Pranay `Q W E R T`, Gaurav `A S D F G`, Kavya `Z X C V B`, in the order A best, B best, C best, no clear winner, all bad. Pressing the same choice again takes the vote back. `←` `→` move between tasks, and `N` goes to the next unfinished one.
- **Core set first.** Once every task is published, `build` chooses a core set of 200 tasks and saves it in `work/blind-eval/core.json`:
  - one size per design;
  - balanced across canvas shapes (about 40 each of extreme wide, wide, near square, tall and extreme tall);
  - preferring tasks where the merged picture differs from ours, since identical pictures say nothing about merged vs ours.

  The judge serves the core set first, with progress shown as "Core set: n / 200", and the Results tab can show the core set or all tasks. About 200 tasks gives roughly 150 decisive results, enough to detect a 60/40 preference. The core set is chosen again only if the merged engine's version changes.
- The optional note is saved with the next vote anyone casts.
- Blind mode is on whenever the page loads; it is never remembered, so a browser once used to check names cannot show them to the next judge. Turning it off shows each option's engine ("Pranay's engine", "Gaurav's engine", "Merged engine") under an orange banner, and every vote records whether it was cast blind.
- When two options show the same picture (the merged engine often keeps our layout), the second is marked "same picture as A". A vote for either credits both engines.
- The **Results** tab reveals the names. It shows how often each engine was picked as best, per judge, per target size and per source orientation, plus the 2-of-3 majority and Fleiss' κ (agreement is on the picture picked). It also links a CSV of every vote.

Votes are stored in `work/blind-eval/votes.sqlite`. Set `BLIND_EVAL_DIR` to serve a different study folder.
