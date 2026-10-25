# Research and implementation options for Chameleon

Research checked on October 19, 2026.

The strongest practical approach is a hybrid: infer relationships in the source design, propose several arrangements, fit them with explicit constraints and real text measurements, then render, rank, and repair. No project verified in this review is a ready-made implementation of the Chameleon contract. Several supply useful components; others provide research ideas without a verified reusable release.

This review examined primary papers, official repositories, model cards, release file listings, licenses, and selected implementation files. It did not install or benchmark the research models. Hardware feasibility and hackathon model eligibility remain unverified. Recommendations and experiment sizes below are proposed engineering choices.

**The most relevant research**

| Resource | What it contributes | How we should use it |
|---|---|---|
| [DesignSense, 2026](https://arxiv.org/html/2602.23438v1) | Adobe research on human preferences for alternative arrangements of identical elements across aspect ratios. Its pipeline combines grouping, layout prediction, filtering, diversity selection, and refinement. | Closest conceptual reference for our complete pipeline. Public code, dataset download, and trained judge weights were not verified. |
| [Learning Layouts for Single-Page Graphic Designs, 2014](https://www.dgp.toronto.edu/~donovan/layout/) | Explicitly tackles design retargeting using perceptual importance, alignment, grouping, and an optimization objective. | Basis for a model-free engine. Translate selected design principles into measurable penalties; do not attempt to recreate the entire research system. The project still labels some code/data as forthcoming. |
| [ReLayout, 2026](https://arxiv.org/html/2602.01046v1) | Encodes position and size relationships in a graph to preserve structure during element-level edits. | Borrow its representation. Canvas retargeting needs more freedom than its edit-preservation task: some above/below relationships must become left/right. A public official implementation/checkpoint was not verified. |
| [LayoutPrompter, NeurIPS 2023](https://arxiv.org/abs/2311.06495) | Uses structured serialization, selection of relevant examples, and ranking of multiple layout predictions without task-specific training. | Adapt its candidate-generation workflow to an approved local model. Its original cloud-model results do not establish performance for a smaller local model. |
| [Design-o-meter, WACV 2025](https://arxiv.org/html/2411.14959v1) | Scores rendered designs and refines geometry using evolutionary search with design-specific snapping. | Inspiration for local search after initial composition. Official reusable scorer weights/code were not verified on the project page. |
| [AesthetiQ, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Patnaik_AesthetiQ_Enhancing_Graphic_Layout_Design_via_Aesthetic-Aware_Preference_Alignment_of_CVPR_2025_paper.pdf) | Adobe research on aesthetic preference alignment for multimodal layout prediction. | Supports evaluating rendered aesthetics, rather than only closeness to one reference arrangement. Public implementation/weights were not verified. |

DesignSense is especially relevant to evaluation. Its data includes 10,235 human preference pairs, and its judge is based on InternVL3-8B. The paper reports weaknesses in generic VLM and image-reward judgments. Its data-generation pipeline uses hosted models, so it is not directly reproducible as an offline submission. We should borrow the staged structure and validate any judge against human comparisons, rather than assume a general model understands design quality. These findings do not establish which evaluator Adobe will use in this hackathon. [Paper](https://arxiv.org/html/2602.23438v1)

**Open-source components worth trying**

| Project | Verified availability | Fit for our engine | Main integration issue |
|---|---|---|---|
| [Google OR-Tools](https://github.com/google/or-tools) | Code; Apache-2.0 | Strong option for discrete placement, bounds, alternative size choices, and non-overlap requirements | CP-SAT uses integer variables. Text wrapping and aesthetics require additional modeling. |
| [Kiwi](https://github.com/nucleic/kiwi) | Code; Modified BSD | Lightweight option for margins, alignment, proportions, and preferred positions once an arrangement is selected | Linear constraints do not directly choose between alternative spatial orderings. |
| [LayoutPrompter](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter) | Code; parent repository MIT and archived | Useful example retrieval, serialization, and ranking components | Replace hosted-model calls; adapt category vocabulary and geometry conventions. |
| [RALF](https://github.com/CyberAgentAILab/RALF) | Apache-2.0; code and checkpoint download links; repository archived | Retrieval-based candidate ideas and useful visual/geometric metrics | Primarily content-aware placement around a canvas image, not preservation of all editable elements in an existing design. |
| [LaDeCo / elem2design](https://github.com/microsoft/elem2design) | MIT code; public model adapter and projector files | Most relevant released multimodal composition model to investigate in this shortlist | CUDA-oriented inference, an 8B language-model backbone, a vision encoder, and a custom input/output adapter. |
| [FlexDM](https://github.com/CyberAgentAILab/flex-dm) | Apache-2.0; Crello pretrained checkpoint links; archived | Predict missing geometry while preserving known fields | Legacy TensorFlow/CUDA environment and representation conversion. |
| [LayoutDM](https://github.com/CyberAgentAILab/layout-dm) | Apache-2.0; pretrained starter resources; archived | Conditional geometry generation and a comparative baseline | Common demo checkpoints target UI/document layouts; typography and brand intent are outside the basic box representation. |
| [Layout-Corrector](https://github.com/line/Layout-Corrector) | MIT; code and starter-kit link including pretrained resources | Research option for repairing predictions from discrete layout generators | Not a general-purpose corrector that accepts arbitrary Chameleon JSON. |

For CP-SAT, precompute several text-box options using the renderer: each option has a font size, width, height, and line count. Let the solver choose among those finite options while placing groups. Add non-overlap only for pairs that should be separate. A global non-overlap constraint over every element would incorrectly separate button labels from their backgrounds. Rotated elements require conservative enclosing boxes or a separate geometric validation pass. CP-SAT reports feasibility/optimality statuses; it cannot guarantee a feasible solution for an impossible constraint set. [CP-SAT documentation](https://developers.google.com/optimization/cp/cp_solver), [2D rectangle constraints](https://or-tools.github.io/docs/dotnet/classGoogle_1_1OrTools_1_1Sat_1_1CpModel.html)

Kiwi supports required and weighted preference constraints. It is attractive if we first enumerate a few arrangement structures and solve alignment and spacing within each. We should choose it or OR-Tools based on measured needs, rather than introduce two solvers immediately. [Kiwi constraint documentation](https://kiwisolver.readthedocs.io/en/latest/basis/basic_systems.html), [license](https://github.com/nucleic/kiwi/blob/main/LICENSE)

LaDeCo organizes composition into background, underlay, logo/image, text, and embellishment layers. Its released model uses Llama-3.1-8B and CLIP; the release contains adapter weights and a projector, so base components are also needed. The model card says training filtered designs above 25 elements. Treat this as a generalization concern for Level 3, not proof of a strict inference limit. Freeze source content, fonts, and colors when adapting it. Its MIT release labels do not replace the base model's terms or the competition whitelist. [Repository](https://github.com/microsoft/elem2design), [model card and files](https://huggingface.co/microsoft/elem2design)

**Specific code to examine first**

| File | Reusable capability | Required adaptation |
|---|---|---|
| [LayoutPrompter selection.py](https://github.com/microsoft/LayoutGeneration/blob/main/LayoutPrompter/src/selection.py) | Select examples for type, size, relation, refinement, and content-aware tasks | Retrieve by target ratio, group count, and source structure; never depend on sample element names. |
| [LayoutPrompter ranker.py](https://github.com/microsoft/LayoutGeneration/blob/main/LayoutPrompter/src/ranker.py) | Combine alignment, overlap, and optional reference-layout similarity | Protect constant metric columns from division by zero; add grouping, readability, and brand-preservation checks. |
| [RALF metric.py](https://github.com/CyberAgentAILab/RALF/blob/main/image2layout/train/helpers/metric.py) | Alignment, overlap, validity, underlay effectiveness, and saliency-aware measurements | Map categories and intentional overlaps correctly; avoid importing unrelated heavy model dependencies for simple geometry functions. |
| [LaDeCo layout.py](https://github.com/microsoft/elem2design/blob/main/llava/metrics/layout.py) | Geometry and content-aware evaluation routines | Match Chameleon's coordinate system and categories; use these as development metrics rather than an official scoring replica. |
| [Chameleon wrap.py](https://github.com/jhaaj08/chameleon/blob/main/renderer/converter/wrap.py) | Glyph-width measurement and word wrapping using the renderer's font lookup | Add height, long-word, rotation, and visible-pixel checks around the measured layout. |

The ranker's normalization issue is visible in the inspected source: it divides by each metric's maximum minus minimum without a constant-column guard. A single candidate or identical scores can make that denominator zero. This is a reason to adapt and test small components rather than assume a research repository is production-ready. [Inspected ranker](https://github.com/microsoft/LayoutGeneration/blob/main/LayoutPrompter/src/ranker.py)

**Data and rendering**

[Crello](https://huggingface.co/datasets/cyberagent/crello) is the most relevant public data starting point here because it represents real graphic designs with element metadata. Its card lists CDLA-Permissive-2.0 and separately requires agreement to VistaCreate's asset terms. Preserve the distinction between the data license and underlying artwork rights. Use a development subset first, keeping source templates separated across development and held-out evaluation.

Converting Crello to the hackathon schema is real work: decode category and font indices, denormalize geometry, convert rotations, extract element images, preserve layer order, and handle unsupported or pre-rendered fields. A successful conversion is not automatically an exact editable equivalent. Build and inspect a small representative subset before scaling up. [Crello format documentation](https://github.com/CyberAgentAILab/canvas-vae/blob/main/docs/crello-dataset.md)

[cr-renderer](https://github.com/CyberAgentAILab/cr-renderer) is Apache-2.0 and is already part of the starter pipeline. Keep the official rendering path as the final authority for our previews. A browser preview or alternative font engine can wrap text differently and give misleading quality estimates.

**Attractive projects that should not become our initial dependency**

- [CreatiPoster](https://github.com/graphic-design-ai/creatiposter) explicitly demonstrates resizing, but its README still marks training/inference code, checkpoint, and data releases as incomplete. The inspected repository metadata did not identify a license. Treat it as a paper/demo reference.
- [Graphist](https://github.com/graphic-design-ai/graphist) predicts geometry and layer ordering from supplied elements, but its checkpoint release remains a to-do in the inspected repository. Useful representation idea, unverified runnable model.
- [DesignAsCode](https://github.com/liuziyuan1109/design-as-code) releases code and a local planner, but the complete pipeline calls hosted models and generates/edits images. The README lists at least 24 GB GPU VRAM for GPU inference. Borrow its plan/render/reflect structure; the full pipeline does not meet stage-one restrictions unchanged.
- [DesignAsCode planner](https://huggingface.co/Tony1109/DesignAsCode-planner) is a released Qwen3-8B derivative, labeled Apache-2.0, with approximately 16 GB fp16 weights. Its task is planning new designs from natural-language requests, not resizing an existing element list. It would need task adaptation even if eligible.
- [LayoutNUWA](https://github.com/ProjectNUWA/LayoutNUWA) provides layout-as-code training/inference code, using LLaMA2/CodeLLaMA 7B. A task-finetuned checkpoint was not verified. The supplied training setup is configured for 64 V100 GPUs; that is a training configuration, not an inference requirement. Low priority for a short hackathon.
- [PosterLlama](https://github.com/jaepoong/PosterLlama) does publish code and a [weight file](https://huggingface.co/poong/PosterLlama). It is another possible model experiment, but its background-conditioned box-placement task and legacy CUDA setup require adaptation. The inspected model metadata did not specify a license; do not equate public download access with clear reuse terms.

**Proposed engine**

The following design is a synthesis for this hackathon, not a reproduction of any one paper.

1. Normalize the input and record immutable fields. Preserve every ID once, asset references, text, brand styles, and the required canvas. Keep an untouched source copy for validation.
2. Build groups and relationships. Explicit groups come first. Infer additional associations from containment, proximity, alignment, and source layering. Maintain confidence levels: source overlap alone is not evidence that an overlap is intentional.
3. Separate meaning from geometry. Preserve associations such as a headline belonging to supporting copy, while allowing their orientation to change. Keep decorative layers out of foreground packing constraints.
4. Generate candidate structures. Try modest source-preserving adjustment, vertical flow, horizontal flow, split composition, and compact multi-column arrangements. Select candidates based on the actual source and target, not fixed coordinates for known platform sizes.
5. Fit typography and group internals. Precompute alternatives from real font measurements. Buttons may need flexible padding and text reflow; do not uniformly scale every group as a bitmap. Model text width and height together.
6. Solve geometry. Use an explicit solver or simpler search over candidate structures. Keep validity requirements separate from visual preferences. Prioritize readability and hierarchy when space becomes scarce.
7. Render and inspect computationally. Combine geometry with visible-pixel checks for clipping, local contrast, busy regions behind text, and occlusion. Alpha-aware checks can avoid false collisions caused by transparent image margins.
8. Rank, repair, and return. Retain the best valid incumbent while making small adjustments. A permitted local model can propose or compare layouts, but cannot bypass validation or rewrite immutable fields.

For extreme aspect ratios, preserving the exact original arrangement would defeat the purpose of retargeting. Relationship preservation therefore needs selective relaxation. Similarly, an optimizer cannot make arbitrarily long text comfortably readable on an arbitrarily small canvas while preserving all content. Track such trade-offs explicitly rather than masking them by hiding elements or reducing their opacity.

**Experiments that will decide what to build**

Start with 30–50 varied source designs and five target shapes each, yielding 150–250 tasks. These are proposed initial development sizes, not statistical sufficiency claims. Include groups, transparent artwork, long text, rotations, dense designs, and missing assets. Increase coverage after the pipeline is stable.

| Experiment | What changes | Decision it informs |
|---|---|---|
| Baseline | Proportion-preserving placement with text fitting | Establish the minimum useful quality level |
| Grouping | Explicit groups plus inferred relationships | Does understanding associations reduce visible failures? |
| Search | Several candidate structures plus constraint fitting | Does composition search improve more than extra prompting? |
| Retrieval | Relevant layout exemplars or arrangement priors | Do reusable examples help unseen designs? |
| Local model | Add a permitted model for proposals, with identical repair logic | Does it beat the simpler pipeline at acceptable memory/runtime? |
| Judge | Compare rule-based ranking with model-assisted ranking | Does the model select what humans prefer? |

Report preservation failures, overflow/clipping, unintended overlap, readable-text measures, runtime, peak memory, and blind human pairwise preferences. Keep ties and both-bad judgments. Distributional research metrics such as layout FID should not be the primary selector for a single resized design. Do not claim any development score reproduces Adobe's hidden evaluator.

Split evaluation by original design, not by resized variants, so the same source design does not appear on both sides. Scramble IDs and input order. Include known source defects to prevent the engine from preserving accidental collisions. Compare model additions through ablations using the same tasks and rendering path.

**Recommended order of work**

First, build text measurement, structural validation, grouping, candidate arrangements, and rendering diagnostics. Next, compare a simple search implementation with OR-Tools or Kiwi, choosing the smallest tool that materially improves results. Adapt selected RALF and LayoutPrompter utilities. Only then benchmark LaDeCo or another eligible local model as an optional candidate generator. Model eligibility, total parameter accounting, quantization rules, reference hardware, and external-data permissions need organizer confirmation before committing to that branch of work.

For the finale, use Adobe's [Document Sandbox overview](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/), [add-on samples](https://developer.adobe.com/express/add-ons/docs/guides/learn/samples), and [project generator](https://github.com/adobe/create-ccweb-add-on). Validate the specific read/write operations our workflow needs before promising full round-trip editing. Keep engine JSON independent of the add-on UI, so preview alternatives and user constraints can reuse the same core.

The immediate build recommendation is an offline Python engine using the supplied renderer, relationship-aware candidate search, and precise typography. The highest-priority model experiment is LaDeCo, subject to eligibility and hardware. The highest-priority papers are DesignSense, Learning Layouts, and ReLayout.
