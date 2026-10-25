# Photo-backed layout correction

The “Tools for Home Renovation in Blue” banner exposed two missing relationships: text must stay on its original photographic support, and its original clear area matters because the subject is embedded in the photo. Bounding-box contrast alone did not detect this: source contrast was already around 2:1, so a relative 40% regression threshold tolerated large visible damage.

The engine now detects unsupported text and excessive movement across its photographic background as quality defects. These relationships do not merge all overlapping elements into one rigid block.

A photo-background candidate uses the schema's `fit: cover` to crop a canvas background without distortion, then reflows copy in its original horizontal region. Only a non-rotated, ungrouped image covering at least 85% of the source canvas at the bottom of the stack may change to cover. Text content, font family, assets, style runs, opacity, rotation and layer order remain unchanged. Other image fit changes remain rejected. The original asset file is neither rewritten nor replaced.

Rule basis: the supplied Final Chameleon Challenge document explicitly describes adapting “positions, stacking, type size, cropping” as canvas shape changes; the supplied schema lists `contain`, `cover` and `fill`. The supplemental RULES.md forbids regenerating/redrawing/replacing asset pixels. A standard renderer crop of the same original asset follows the brief's adaptation operation. This is a correction to our earlier blanket immutability policy for image fit, which was more restrictive than that brief.

This candidate is deliberately limited to a single photographic background and an unrotated, ungrouped copy region. It is not a complete subject detector or a general document/table reflow solution. Other candidates continue to compete through preservation and quality gates. Cropping may remove peripheral photo content; generated variants still need visual review.

Verification: 42 automated tests pass, including the exact native Crello design, rejection of the old escaped-text layout, background-only fit changes, original-resource preservation and source immutability. Rendered outputs and per-format diagnostics are in output/photo-support/. The score remains an engineering estimate, not an AI or human quality evaluation.
