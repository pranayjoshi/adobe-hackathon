---
language:
- en
license: mit
task_categories:
- text-generation
tags:
- graphic-design
- design-generation
- distillation
- layout-planning
pretty_name: DesignAsCode Training Data
size_categories:
- 10K<n<100K
---

# DesignAsCode Training Data

Training data for the [DesignAsCode Semantic Planner](https://huggingface.co/Tony1109/DesignAsCode-planner).

## Overview

| | |
|---|---|
| **Samples** | 19,479 |
| **Format** | JSONL (one JSON object per line) |
| **Size** | ~145 MB |

## Data Source

Each sample corresponds to a real graphic design from the [Crello](https://huggingface.co/datasets/cyberagent/crello) dataset. We distilled structured design semantics from each original design using **GPT-4o** and **GPT-o3**, taking the original design, its individual layers, and design metadata as input.

The distillation produces:

- A natural-language **prompt** describing the design
- A **layout thought** — a detailed reasoning trace about visual layout, element placement, and design rationale
- **Grouping** — a structured grouping of related layers with thematic labels
- **Image generation prompts** — per-layer descriptions for reproducing each image element
- **Text element specifications** — font, size, alignment, content, and styling for each text layer

## Fields

| Field | Description |
|---|---|
| `id` | Unique identifier (matches the original Crello sample ID) |
| `prompt` | Natural-language description of the design (distilled) |
| `layout_thought` | Detailed layout reasoning wrapped in `<layout_thought>...</layout_thought>` |
| `grouping` | Layer grouping with thematic labels as JSON array, wrapped in `<grouping>...</grouping>` |
| `image_generator` | Per-layer image prompts as JSON array, wrapped in `<image_generator>...</image_generator>` |
| `generate_text` | Per-layer text specs as JSON array, wrapped in `<generate_text>...</generate_text>` |

## Intended Use

This dataset is used for **supervised fine-tuning (SFT)** of the DesignAsCode Semantic Planner (Qwen3-8B). During training:

- **Input:** `prompt`
- **Output:** `layout_thought` + `grouping` + `image_generator` + `generate_text`

The trained planner learns to decompose a natural-language design request into a structured, actionable design plan.

## Download

```bash
pip install huggingface_hub
huggingface-cli download Tony1109/DesignAsCode-training-data --repo-type dataset --local-dir training_data
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{liu2026designascode,
  title     = {DesignAsCode: Bridging Structural Editability and 
               Visual Fidelity in Graphic Design Generation},
  author    = {Liu, Ziyuan and Sun, Shizhao and Huang, Danqing 
               and Shi, Yingdong and Zhang, Meisheng and Li, Ji 
               and Yu, Jingsong and Bian, Jiang},
  journal   = {arXiv preprint arXiv:2602.17690},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.17690}
}
```
