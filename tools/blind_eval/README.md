# Blind judge: Pranay vs Gaurav

Three people (Pranay, Gaurav, Kavya) judge the two resize engines side by side without knowing which is which.

## The study

- **200 designs** from the Crello validation split (CDLA-Permissive-2.0), with their original fonts and image layers. Imported to `work/blind-eval/crello` and sampled by a seeded hash, so the set is stable.
- **5 target sizes per design, chosen from the 14 app presets** (`lite/app.py`). The picker is greedy farthest-point in (log aspect ratio, log area) space. It starts from the source's own shape, so each design gets the 5 sizes that differ most from the original and from each other. Sizes with the source's aspect ratio are skipped (a plain scale is not a design decision). A coverage penalty keeps the two most extreme ad sizes from taking every slot, so all 14 appear: 33 to 152 uses each.
- **1,000 pairs.** Each engine runs in its own interpreter: ours is a snapshot of `HEAD` in `work/blind-eval/engines/pranay`, and Gaurav's is his `main` at `ed11607`. Both use the same renderer and the same brand kit (fonts and assets), so neither one loses points to a missing font. Every output is rendered by the same renderer.
- When an engine refuses a size (ours raises `NoValidDesign` rather than emit a layout that fails its guardrails), the judge sees "no output" on that side.

Left/right order is fixed for each pair by a seeded coin flip. Image URLs are `/img/<pair>/A|B`, so file names never reveal the engine. The pairs come in one seeded shuffled order, so the same size or design never comes several times in a row.

## Build

```bash
.venv/bin/python -m tools.build_crello_catalog --limit 280 --output work/blind-eval/crello
.venv/bin/python -m tools.blind_eval.build --gaurav ../chameleon-competitors/gaurav/chameleon
```

Every stage resumes and skips work already done. `--publish` renders and publishes only the pairs both engines have finished. The app picks up a new `pairs.json` without a restart, so judging can start while the engines are still running.

## Judge

```bash
.venv/bin/uvicorn tools.blind_eval.app:app --host 0.0.0.0 --port 8790
```

Open `http://<this-mac's-LAN-IP>:8790` (it works from any machine on the same network). The page opens straight onto a pair: the original design, then options A and B. Below them is one row for each judge. The three judges vote together on each pair: **A better / tie / both bad / B better**. Once all three have voted, the page moves on to the next pair that still needs votes.

- Keys: Pranay `Q W E R`, Gaurav `A S D F`, Kavya `Z X C V` (in the order A better, tie, both bad, B better). Pressing the same choice again takes the vote back. `←` `→` move between pairs, and `N` goes to the next unfinished pair.
- The optional note is saved with the next vote anyone casts.
- Blind mode is on by default. Turning it off shows the engine names. Every vote records whether it was cast blind.
- The **Results** tab reveals the names. It shows the win share per judge, per target size and per source orientation, the 2-of-3 majority, and Fleiss' κ. It also links a CSV of every vote.

Votes are stored in `work/blind-eval/votes.sqlite`. Set `BLIND_EVAL_DIR` to serve a different study folder.
