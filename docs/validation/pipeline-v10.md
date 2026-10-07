# Verified fixes against the catalog 8.0 baseline (BF16 pair, 2026-10-07)

This report compares two runs of the full pipeline on the same images. Both use BF16 `Qwen/Qwen3.8-27B` and `google/gemma-4-31B-it`, with generator A's server as the image router, the split three-GPU layout, seed 20261004 and English. Each image set runs four images at a time, and the two sets run together.

- **B0** is the baseline at commit 6c8c5d2 (task catalog 8.0).
- **B1** adds eight verified fixes to B0. Routing, drafting and review are otherwise unchanged.

The images are evaluation-only:

- **diverse-269**: the 269 images of [`validation/diverse_269_20261007_manifest.json`](../../validation/diverse_269_20261007_manifest.json).
- **P100**: 100 pinned Open Images V7 validation photos.

"Gold" means ratings by GPT-6.1 Sol Ultra, an AI reference rather than human review. A conversation is gold-valid when every committed turn has a MET question and a MET answer.

**B1 met every acceptance criterion except time per image.** Gold-valid conversations rose by 28% on diverse-269 and by 35% on P100. Time per image on diverse-269 rose to 1.13 times B0, above the 1.1 limit fixed before the run. The rise came from conversations getting further rather than from slower steps, and gold-valid conversations per GPU-hour improved by 15%.

## Why only these fixes

An earlier candidate bundled these fixes with routing changes. It offered structured operations from the first turn and switched once to a lightly verified question when a verifier abstained. That bundle lowered gold-valid conversations on diverse-269 from 47 to 31 and took 1.8 times as long. The fixes below each had separate evidence from replays or gold rationales. B1 measures them without the routing changes.

| Fix | Evidence before the run |
|---|---|
| Answer parsers are bound to one allowed result | Saved invalid finite answers became valid |
| Transcriptions are compared by words and punctuation after unifying dashes, quotes and spacing | Five formatting-only mismatches passed |
| Formula, graph and finite readers no longer fail on form | Malformed outputs |
| Judge reasons are capped at 25 words, and retries name invalid regions | Truncated judge outputs fell from 5 to 0 in a replay |
| Absent objects are asked about only where they would clearly be seen | The gold rejected three undecidable absences |
| Region schemas keep left and top below 1 and right and bottom above 0 | Valid replies on 87 failed region requests rose from 12 to 60 |
| Holistic judges write their reason before their verdict | Split judge pairs fell from 175 to 93 of 211; rejected answers were not accepted more often |
| JSON Lines are split only at line feeds | Model text with U+2028 made a conversation file unreadable |

## Results

| Measure | diverse-269 B0 | diverse-269 B1 | P100 B0 | P100 B1 |
|---|---|---|---|---|
| Quality candidates | 91 | 123 | 41 | 51 |
| Abstained conversations | 103 | 60 | 35 | 20 |
| Gold-valid conversations (precision) | 47 (51.6%) | 60 (48.8%) | 20 (48.8%) | 27 (52.9%) |
| Gold-valid with three or more turns | 6 | 4 | 2 | 3 |
| Families with three or more gold-valid turns | 11 | 11 | 6 | 6 |
| Committed turns, all conversations | 262 | 335 | – | – |
| Seconds per image (first call to last reply) | 28.2 | 31.8 | 35.6 | 37.8 |
| Model calls | 3,857 | 4,289 | 1,472 | 1,624 |
| Errors | 0 | 0 | 0 | 0 |

## Acceptance criteria

| Criterion, fixed before the run | diverse-269 | P100 | Result |
|---|---|---|---|
| 1. Gold-valid conversations at least B0 | 47 → 60 | 20 → 27 | Met |
| 2. Precision at least B0 minus 3 points | 51.6% → 48.8% | 48.8% → 52.9% | Met |
| 3. Time per image at most 1.1 times B0 | 28.2 → 31.8 s (1.13 times) | – | Not met |
| 4. At most 2 more accepted negatives | 41 → 32 of 44 | 23 → 21 of 24 | Met |
| 5. No errors, and no model call on resume | 0 | 0, and 0 calls | Met |

The negatives are the committed B0 turns whose question or answer the gold rated NOT_MET. Each was rerated with `rate-existing` together with its earlier turns. B1 newly accepted one of them, a transcription.

## What changed

- **Fewer judge splits.** Conversations stopped by a judge split fell from 73 to 32 on diverse-269 and from 27 to 13 on P100. More turns therefore reached later steps. Turns committed on diverse-269 rose from 262 to 335, while model calls rose only from 3,857 to 4,289. Per committed turn, calls fell from 14.7 to 12.8, and time per gold-valid conversation fell from 161 to 143 seconds.
- **More verifier abstentions.** Turns that now reached the verifiers ended there more often. Stops with two accepting judges and an UNKNOWN verifier rose from 23 to 33 on diverse-269, mostly in the closed-set check of counting operations. That reader still asks for coverage before content.
- **Undecidable absences.** Absent-object questions still failed the gold in occluded, blurred or cropped areas: 7 of 9 committed `object_presence` turns on P100 were rejected. The visibility rule given to the drafter and the judges did not reach the reference's standard.

## Not verified

- No human reviewed any of these results; every judgment came from the pipeline or the GPT reference.
- No run used the standard layout of four 48 GB GPUs.
- Each fix's individual contribution is unmeasured; only the bundle was measured.
