# Derived model artifacts

`roles.json` is an independently trained 48-tree classifier with portable JSON inference. `layouts.json` contains 638 layouts represented only by aspect ratio, role, and normalized boxes. Both were derived from 1,256 matched public training designs in the first two Crello training shards, joined to DesignAsCode machine-distilled grouping themes. No rival model weights were copied.

The role pilot achieves 62.74% element accuracy on the held-out exact-copy-family partition (924 training families, 231 validation families). This measures agreement with noisy machine-derived labels, not human layout quality. The classifier is off by default; enable learned-role suggestions in the viewer or set `CHAMELEON_ROLE_MODEL=1`. It uses confidence gates and never changes source content or images. Compare its results before enabling it for a submission. Its dataset is smaller and its label mapping differs from the comparator's; reported accuracies are not directly comparable.

The retrieval library uses the classifier's training-family partition and excludes its validation families. Layouts with missing or unsupported source reconstruction are still useful for geometry experiments, but the library is not evidence of a fidelity-verified image dataset. Public sample designs are not used to construct this library.

`data-manifest.json` records dataset revisions, weak-label counts, source-file hash, artifact hashes, and split policy. Raw data is kept under ignored `work/role-data/`, outside the submission image. Dataset references:

- https://huggingface.co/datasets/cyberagent/crello — CDLA-Permissive-2.0; see included `crello-license.txt` and the dataset card's underlying artwork terms.
- https://huggingface.co/datasets/Tony1109/DesignAsCode-training-data — model-distilled themes; consult the pinned dataset card and its MIT declaration.

The normal image ships these small local artifacts, and makes no runtime downloads. Adapted weights must be publicly accessible when the challenge rules require it; local training alone does not satisfy publication. This work did not publish anything externally.

`ranker.json` is deliberately absent until the human preference pipeline yields reviewed weights. Missing weights preserve deterministic scoring. VLM weights are also absent; use an approved local model/projector and `requirements-vlm.txt` to enable that experimental backend.
