# Selective grouping and layout quality

Implemented 2026-10-24, following a review of Gaurav's engine. This is a new implementation within our engine, not a replacement with his code.

## Behavior

- Geometric overlap records a relationship; it no longer transitively merges content into a rigid block. Explicit schema groups and inferred small shape/label buttons retain their existing treatment.
- Large, visually uniform raster layers can be identified as panels rather than hero images. Panel fitting retains original pixels and image proportions. Images that cannot expand to fit surrounding content remain constrained, and resulting contrast failures are reported.
- Multi-column candidates offer an alternative to shrinking every item into a single column or banner row. The search tries two, three and four columns where space permits.
- An original-arrangement candidate uniformly contains the entire source geometry, including rotated bounds and bleeds. It retains internal positions and competes through the same gates as new layouts.
- Readability floors, worsened overflow, new collisions and contrast regressions form a separate quality gate. Structural preservation remains independently reported. Rank first considers preservation, then the number of quality defects, then the existing quality score. Defective candidates have capped scores and explicit review labels; they are still returned if no clean option exists.
- Existing source overflow is distinguished from worsened output overflow. This matters for Crello's tight text boxes. Pixel diagnostics still refine shortlisted candidates, with quality gates recomputed afterward.
- Composition includes occupied visible-box density as well as hull coverage. The UI calls the old fidelity component “Relative order”; it is not a pixel fidelity measurement. Source and output content, family names, asset references, styles and rotations remain immutable.

## Limits

The native invoice from the reported screenshot is no longer collapsed into an 18-element rigid block. Its extreme 1600×400 output still cannot be claimed as a successful redesign. The source uses a portrait raster background; preserving its proportions and keeping all layers inside the canvas leaves unused space. The invoice also needs richer table and region relationships for a true wide document layout. The engine explicitly marks the remaining unreadable output as needing review. No pixels, fonts or words are replaced to conceal that limitation.

The columns are general reading-order proposals, not a table reconstruction algorithm. Flat-image detection is deliberately conservative. These engineering gates and scores have not been calibrated against human preferences. Gaurav's published Crello benchmarks substitute assets/fonts, so they are not reported as a direct comparison against this original-resource collection.

## Evidence

`output/layout-upgrade/before.json` and `after.json` record three native Crello designs across banner, square and portrait formats, with PNGs, preservation results, diagnostics and chosen layouts. Scores before/after are not directly comparable because the scoring rules changed. The test suite covers panel separation, overlap without merging, explicit grouping, unreadable-type rejection, source immutability, candidate coverage and existing engine/service behavior.

Final verification: 38 automated tests pass and the production viewer builds. All nine tested native-resource outputs pass preservation; five pass the new quality gates and four remain explicitly flagged for review. Those counts are not a human visual quality benchmark. The problematic invoice banner remains one of the flagged cases.
