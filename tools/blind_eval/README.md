# Blind judge: Pranay vs Gaurav vs Hybrid

Three people (Pranay, Gaurav, Kavya) compare three resize engines without knowing which is which: ours, Gaurav's, and the hybrid built from both (`hybrid/`).

## The study

- **200 designs** from the Crello validation split (CDLA-Permissive-2.0), with their original fonts and image layers. Imported to `work/blind-eval/crello` and sampled by a seeded hash, so the set is stable.
- **5 target sizes per design, chosen from the 14 app presets** (`lite/app.py`). The picker is greedy farthest-point in (log aspect ratio, log area) space. It starts from the source's own shape, so each design gets the 5 sizes that differ most from the original and from each other. Sizes with the source's aspect ratio are skipped (a plain scale is not a design decision). A coverage penalty keeps the two most extreme ad sizes from taking every slot, so all 14 appear: 33 to 152 uses each.
- **1,000 tasks, three options each.** Each engine runs in its own interpreter: ours is a snapshot of `HEAD` in `work/blind-eval/engines/pranay`, and Gaurav's is his `main` at `ed11607`. Both use the same renderer and the same brand kit (fonts and assets), so neither one loses points to a missing font. Every output is rendered by the same renderer.
- **The hybrid** is built from the two cached results with `hybrid.combine.choose`, the same code `hybrid.resize` runs live. It makes two repairs. Letter spacing shrinks with the font size, which fixes ours. Reshaped photos are cropped (`cover`) instead of stretched; flat colours and frames still stretch, which fixes Gaurav's. Then it keeps whichever layout has fewer defects, counting text collisions, text or photos lifted off their panel, split phrases, cut content, letterboxing, tiny text and sparse layouts.
- When an engine refuses a size (ours raises `NoValidDesign` rather than emit a layout that fails its guardrails), the judges see "no output" in that slot.

The A/B/C order is shuffled for each task by a seeded draw. Image URLs are `/img/<task>/A|B|C`, so file names never reveal the engine. The tasks come in one seeded shuffled order, so the same size or design never comes several times in a row. The hybrid often keeps one engine's layout unchanged. When two engines produce the same picture, a vote for it counts for both.

## Build

```bash
.venv/bin/python -m tools.build_crello_catalog --limit 280 --output work/blind-eval/crello
.venv/bin/python -m tools.blind_eval.build --gaurav ../chameleon-competitors/gaurav/chameleon
```

Every stage resumes and skips work already done. Engine runs and renders resume; the hybrid is rebuilt from the cached runs each time. To redraw the hybrid after changing `hybrid/combine.py`, delete `work/blind-eval/img/*__hybrid.png` first. `--publish` builds and publishes only the tasks both engines have finished. The app picks up a new `pairs.json` without a restart, so judging can start while the engines are still running.

## Judge

```bash
.venv/bin/uvicorn tools.blind_eval.app:app --host 0.0.0.0 --port 8790
```

Open `http://<this-mac's-LAN-IP>:8790` (it works from any machine on the same network). The page opens straight onto a task: the original design, then options A, B and C. Below them is one row for each judge. The three judges vote together on each task: **A best / B best / C best / no clear winner / all bad**. Once all three have voted, the page moves on to the next task that still needs votes.

- Keys: Pranay `Q W E R T`, Gaurav `A S D F G`, Kavya `Z X C V B`, in the order A best, B best, C best, no clear winner, all bad. Pressing the same choice again takes the vote back. `←` `→` move between tasks, and `N` goes to the next unfinished one.
- The optional note is saved with the next vote anyone casts.
- Blind mode is on by default. Turning it off shows the engine names. Every vote records whether it was cast blind.
- The **Results** tab reveals the names. It shows:
  - how often each engine was picked as best, per judge, per target size and per source orientation
  - the 2-of-3 majority and Fleiss' κ (agreement is on the picture picked)
  - how often the votes agree with the hybrid's choice of layout, which is the number to tune `hybrid/combine.py` against

  It also links a CSV of every vote.

Votes are stored in `work/blind-eval/votes.sqlite`. Set `BLIND_EVAL_DIR` to serve a different study folder.
