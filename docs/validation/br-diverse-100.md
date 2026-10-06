# Full pipeline on the diverse-100 evaluation images (BF16 pair, 2026-10-07)

This report records one run of the current pipeline (BF16 `Qwen/Qwen3.8-27B` and `google/gemma-4-31B-it`, with generator A's server as the image router) on the 100 images of [`validation/diverse_100_20261004_manifest.json`](../../validation/diverse_100_20261004_manifest.json), after the draft-format fix described below. The images are evaluation-only Wikimedia Commons files in ten strata. "Gold" means ratings by GPT-6.1 Sol Ultra, an AI reference rather than human review.

## Question-draft failures of Gemma 4

Before the fix, 43.8% of Gemma's draft calls in four BF16 runs failed schema validation, and 72% of all draft failures were regions with no area or inverted edges. When the region bound was widened to 1000, Gemma wrote 437 of 440 regions above 100: it uses its native 0-1000 box scale, which constrained decoding clamped into `{1, 1, 1, 1}` boxes or whitespace runaways. The draft instruction now defines regions as fractions from 0 to 1 with left < right and top < bottom, and `public_parameters` as an array of name and value objects; retries restate both formats.

| Same 160 Gemma / 80 Qwen draft requests | Gemma admitted | Qwen admitted |
|---|---|---|
| Previous instruction | 88 (55.0%) | 47 (58.8%) |
| Explicit formats (adopted) | 136 (85.0%) | 55 (68.8%) |

The Gemma difference has McNemar p < 0.0001. In the run below, Gemma's draft schema failures fell to 3 of 110 calls (2.7%).

## Run

The run used the standard profile settings with the split three-GPU layout of `configs/split-pilot.yaml`, seed 20261004, English, and four concurrent images per prepared source. The two prepared sources (60 and 40 images) ran at the same time.

| Measure | Result |
|---|---|
| Wall time | 31.7 minutes for 100 images (189 images per hour, 8 images in flight) |
| GPU time for synthesis | 1.58 GPU-hours (three GPUs) |
| Model calls | 1,518; 4.24 million input and 0.19 million output tokens |
| Errors | 0 |
| Quality candidates | 38 conversations, 83 committed turns |
| After selection | 35 (one per image and visual group, no repeated semantic family); export refuses evaluation images |
| Gold-valid conversations | 22 of 38 (58%); 64 of 83 turns had MET question and answer ballots |

Question drafting took 37% of model busy time, the question gate 25% and holistic review 14%.

## Where turns are lost

Of 185 attempted turns, 105 were committed. Of the 80 failed attempts:

- 65 failed holistic answer review (31 NOT_MET, 34 unresolved);
- 7 failed when Qwen's finite-answer parse omitted its declared result (a schema the decoder cannot enforce);
- 3 stopped at the question gate, 2 had no admissible draft, 2 failed answer generation, and 1 hit a repetition stop.

## Diversity

| Aspect | Result |
|---|---|
| Operations | 20 of 72 task IDs, evenly used (normalized entropy 0.93) |
| Families | 9 of 14, but visual description, reference and spatial, and text reading hold 90% of turns |
| Required capabilities | 20 of 50 |
| Question text | 82 of 83 distinct |
| Turns per conversation | 2 turns: 33, 3 turns: 4, 5 turns: 1 |

Turn-count diversity is not yet achieved: later turns usually stop at holistic review or the finite-answer parse, so conversations keep their accepted two-turn prefix. Structured strata such as tables, charts, diagrams and formulas rarely survive.
