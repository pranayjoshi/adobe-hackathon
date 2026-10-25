# Testing Crello designs in Chameleon

Open http://127.0.0.1:8765 and click **Browse Crello** in the header or source panel.

1. Search by title, theme, source ID or font. Filter by format, shape or resource status, and use Next/Previous to browse.
2. Click a design to compare its dataset preview with its editable reconstruction. On smaller screens the comparison appears above the gallery.
3. Review the layer counts, fonts and conversion notes. **Resources ready** means the original layer images and requested font faces are installed; it does not certify matching pixels or good resize results.
4. Click **Use this design**, select output sizes in the source panel, and click **Generate**. The current target selection stays in place when switching designs.
5. Inspect ranked outputs, validation checks, overlays and the naive stretch comparison. Use the existing detail view to download generated JSON/PNGs. Open **Compare dataset preview** under the source title to revisit the original preview.
6. **Download design JSON** in the gallery exports the editable source and pinned provenance. Image and font files remain local resources; this JSON alone is not a portable artwork bundle.

## Installed collection

The initial collection contains the first 60 records from the first validation shard of `cyberagent/crello`, pinned to commit `7997e2f434ee4aa73cf4cdf22c5954cb175872e1`. This is a bounded test collection, not the entire dataset, and these validation records are not added to training or the retrieval library.

Each record includes the dataset preview, actual original non-text PNG layers, editable text, categorical font names decoded from the shard's own feature metadata, per-character color/bold/italic runs, alignment, capitalization, spacing, rotation, opacity and layering. Escaped newlines and HTML entities are decoded during import. Original non-text artwork is not regenerated. Native shapes and masks are represented using their dataset-provided raster layers; they are not editable vectors.

59 records have all requested resources. “Connection Quote Electronic Wires in Blue” requests Happy Monkey bold, which is absent from the font archive. It is visibly marked **Needs review**; the renderer uses its available regular face. Its font-family name stays unchanged. No exact visual fidelity is claimed: source text effects are not fully represented, line wrapping can change, and large source layers can extend past the canvas. Non-cropping background images are contained proportionally during resize, which may leave space around them.

## Rebuild or expand

From the repository root:

```sh
.venv/bin/python -m tools.build_crello_catalog --download --limit 60
```

`--limit` accepts 1–500 records from the same pinned shard. The first run downloads approximately 553 MB into `work/crello-cache`; subsequent runs reuse the cache. SHA-256 digests are verified before reading. Font archive loading forbids executable Python objects. Restart the service after importing to load new font faces, and refresh the browser. Rebuilding preserves installed fonts so previously saved runs can still resolve their resources.

Runtime files:

- `output/crello/catalog.json`: gallery metadata and rejected records.
- `output/crello/designs/`: editable sources with provenance.
- `output/crello/media/`: source and reconstructed previews.
- `brand_kit/assets/crello/`: original PNG layers, scoped by hashed design ID.
- `brand_kit/fonts/crello/` and `crello-fonts.json`: font bytes and explicit family/face mapping.

These downloaded resources are ignored by Git. The dataset is labeled CDLA-Permissive-2.0; retain its [license and source provenance](https://huggingface.co/datasets/cyberagent/crello/tree/7997e2f434ee4aa73cf4cdf22c5954cb175872e1). Font resources retain their original family metadata. Review applicable artwork/font terms before distributing a resource bundle.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests
npm --prefix viewer run build
.venv/bin/python -m tools.verify_crello_catalog
```

The last command expects the local service at port 8765. It tests four different designs at square, landscape and portrait sizes, verifies every returned PNG and checks that element IDs, text, fonts and asset references survive generation. Results, including compliance failures, are saved in `output/evaluation/crello-browser/results.json`. This is a functional integration check, not a visual quality benchmark or a claim that every Crello design works at every size.

Verified locally on 2026-10-24: 33 automated tests pass; the production viewer builds; all 60 source records and 120 preview URLs load; 12 resize runs (four sources × three shapes) return 36/36 structurally compliant candidates. Browser checks cover search, the missing-font filter, preview comparison, selection, job completion and the generated-layout detail view. These checks establish operation and preservation, not subjective design quality.
