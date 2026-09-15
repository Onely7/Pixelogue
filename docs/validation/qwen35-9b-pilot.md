# Qwen3.5-9B pilot validation

This report records the one-GPU diagnostic run completed on 15 September 2026. The run used `Qwen/Qwen3.5-9B` as a temporary replacement for both generator and evaluator roles. It did not use the intended standard pair, `Qwen/Qwen3.8-27B` and `google/gemma-4-31B-it`, and it does not measure evaluator diversity or production data quality.

## What was tested

The run processed 50 evaluation-only images: 30 Pixelogue procedural fixtures and 20 fixed images from the Open Images V7 validation split. `Qwen/Qwen3.5-2B` selected instructions. Separate blind calls were made for evaluator roles A and B, but both calls reached the same Qwen3.5-9B checkpoint. All weights used BF16 without quantization on one RTX 6000 Ada GPU.

The test covered ingestion output, instruction choice, question and answer generation, pre-answer gates, claim and rubric evaluation, repair, immutable turn commits, profiling, replay, backup, pool freezing, selection audit, and the pilot export prohibition.

## Conversation results

| Source | Conversations | Quality candidate | Rejected | Abstained | Error | Committed turns |
|---|---:|---:|---:|---:|---:|---:|
| Pixelogue procedural fixtures | 30 | 3 | 19 | 5 | 3 | 28 |
| Open Images V7 | 20 | 1 | 5 | 9 | 5 | 8 |
| Total | 50 | 4 | 24 | 14 | 8 | 36 |

The quality-candidate rate was 8%. This number should not be read as a standard-model acceptance rate. A conversation needs at least two committed turns. Twenty-eight images produced a passing first turn, but only four produced a passing second turn. No rejected or abstained conversation had already accumulated two committed turns, so the requirement to complete later optional turns did not reduce the candidate count in this run.

The 38 semantic non-candidates stopped at the following boundaries. The eight execution errors are listed separately below.

| Terminal boundary | Conversations | Interpretation |
|---|---:|---|
| Question fit | 10 | Every case was a second-turn question that repeated or closely paraphrased an answered request, so `useful_request` was `NOT_MET`. |
| Final answer rating: rejected | 8 | At least one required criterion was clearly not met. |
| Final answer rating: abstained | 7 | The judges could not establish enough evidence or agreement. |
| Requirement extraction | 7 | The two answer-independent requirement inventories did not reconcile. |
| Instruction selection | 3 | No usable new instruction was selected after one committed turn. |
| Question generation | 3 | The generator did not return a usable public question. |

The largest single cause was weak second-turn diversity, not an arbitrary score threshold. Simple square-count fixtures often support only one natural request, and the temporary models frequently generated the same counting question again. Relaxing `useful_request` would admit duplicate turns rather than improve the data.

## Structured-output failures and correction

The initial run had eight terminal execution errors: five duplicate capability arrays, one rubric response missing its required reason, and two incomplete JSON responses at the token limit. These were model-output failures rather than HTTP, GPU-memory, SQLite, or file-write failures.

The prompts were clarified to require unique capability keys, minimal public-requirement inventories, and a short rubric reason. A bounded retry now receives a short correction note without copying the invalid response. Re-running the eight affected images produced four `REJECTED` and four `ABSTAINED` conversations with no terminal `ERROR`. Three invalid model calls were recovered by the bounded retry. This correction improves execution reliability; it does not turn semantically unsuitable dialogues into quality candidates.

## Performance result

The same eight images were processed once serially and once with four independent images in flight.

| Measurement | One worker | Four workers |
|---|---:|---:|
| Wall time | 745 seconds | 286 seconds |
| Completed model calls | 528 | 529 |
| Model calls per minute | 42.52 | 110.98 |

Bounded image concurrency produced a 2.60× speedup and reduced wall time by 61.6%. It preserved input order and did not cause transport or GPU-memory failures. `rubric_item` still accounted for about three quarters of model calls, so any later request-batching work should group only criteria with the same permitted input scope.

## Manual quality review

All four nominal quality candidates were inspected. The procedural examples contained correct answers, but one conversation used a broad “objects” count for only the squares and another question did not realize its selected instruction exactly. The Open Images conversation treated buds and flowers inconsistently in a count and repeated closely related descriptions. The shared 9B evaluators accepted these weaknesses. A later review of all terminal turns also found one uncommitted Open Images question that reproduced the private question-generation instruction instead of asking about the image; its conversation ended as `ABSTAINED` for another gate.

These findings show that the pilot validated execution and failure boundaries, but did not validate production dialogue quality. The quality gates should remain conservative. The next quality measurement must use the intended two model lineages and should report manual-review precision alongside status counts.

The follow-up filtering review added a deterministic private-instruction echo check and an exact repeated-question check before the expensive image-aware question gates. Four completed question-generation calls were normalized exact repeats, so the new early check would avoid their eight question-fit calls. It also corrected rubric applicability: the original run evaluated `H_BINDING` and `H_WITNESS` on all 15 rated later turns solely because their turn index was greater than one, accounting for 60 evaluator calls in the final turn artifacts. Three numeric-only answers were also sent through three prose-only criteria, accounting for another 18 calls. The applicability corrections therefore avoid at least 78 irrelevant evaluator calls for the same final artifacts. The historical counts above have not been recomputed and remain the results of the original run.

## Integrity and data-use checks

The final manifest contained exactly 30 procedural and 20 Open Images records in input order. All 50 records remained evaluation-only. The automated scan found no dataset labels or internal identifiers in public text, but it did not detect the private-instruction echo described above. Replay verified 6,011 stored artifacts, 2,866 model calls, and 36 committed turns. A consistent SQLite and artifact backup passed database integrity and content-hash checks. Normal selection policy excluded every evaluation source, diagnostic selection and independent audit succeeded only under an explicitly evaluation-enabled temporary policy, and pilot export was rejected with `PILOT_EXPORT_FORBIDDEN`.

No record from this run is training data.
