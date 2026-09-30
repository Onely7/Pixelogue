# Measured quality and synthesis performance

These fixed comparisons use repeated development images. An automatic quality candidate is not an independently approved conversation. Repeated image trials are not distinct images, and comparisons using different cohorts do not establish a controlled difference across cohorts.

## Retained settings

Keep the candidate limit at eight, image concurrency at two, the minimum of two accepted turns and the existing quality gates. Validate each candidate strictly after checking whole-response identifiers and duplicates; preserve useful candidates alongside private rejection artifacts. Keep typed object references limited to object identification in the same scope. Do not increase the output allowance for whitespace retries; limited stages can increase it once for genuine truncation. Invalid output remains unaccepted.

Structural table/chart/graph readers and the supported specialist readers receive the controller-bound output Schema in model text as well as the decoder. Answer parsing reads the literal answer; the controller separately decides correctness. Missing values are never filled with passing judgments.

During graph calculations, bind local IDs to unique displayed labels only when their regions correspond. Preserve edge directions, conditions, the question target and completeness. Duplicate labels or ambiguous correspondence abstain. Keep the original extractions and disagreements. Table and graph answer Schemas restrict unused forms to null using the public task alone; they do not fill in correct values or missing evidence.

## Changes not enabled by default

Compact output increased time and retries without increasing candidates. A candidate limit of four produced unstable counts. Concurrency four ran faster but produced no candidates, with fewer calls partly due to early stops. Attribute rechecks succeeded for the dog but added cost without establishing a general improvement; they remain disabled by default.

## Main observations

Each condition was repeated twice on the same images.

| Change | Images per repeat | Baseline candidates | Changed candidates | Other observations |
|---|---:|---|---|---|
| Strict candidate isolation | 20 | 3 / 3 | 6 / 6 | Accepted in both repeats: 0 versus 2 images |
| Typed object references | 8 | 0 / 0 | 0 / 1 | Reference rejections: 10 versus 2 |
| Bounded whitespace retries | 2 | 1 / 0 | 1 / 1 | Mean synthesis phase: 630 versus 380 seconds; one image always abstained |

A twofold improvement normalized by human quality remains unconfirmed. Phase times exclude model loading, so also inspect full allocations including loading, cleanup and reservations. Prices and human-dependent measures remain unmeasured when their inputs are absent.

## Confirmation, calibration and audit

The final runtime was frozen at `0a7c5a5` before confirmation. CPU checks passed 477 tests;
the 42 tests requiring model endpoints were skipped in that CPU run and then passed on
zao01 with the actual standard Qwen/Gemma pair and the Qwen3.5-2B selector.

Nine fixed structural development cases produced two MET, two NOT_MET, three UNKNOWN and
two FAILED results. Five matched their controller-held expected verdict. Table and graph
positive and wrong-answer checks worked; an ambiguous graph abstained. Chart positive
acceptance remains unconfirmed, and chart/table insufficient-evidence responses still had
format failures. The collection commands exited successfully while retaining these failures.

On six untuned evaluation images, the frozen runtime produced one automatic two-turn
quality candidate, two rejections and three abstentions, with no conversation ERROR.
There were eleven attempted turns and four committed turns. The separately confirmed dog
and primate images both abstained; only one turn was committed. Resuming that same trial
added zero calls and preserved call hashes, token budget, commits and public output.
Eligibility of an image for two questions does not establish generated answer quality.

Freeze confirmation images, labels, order, source/model/processor/Schema/catalog/validator identities before evaluation. Evaluate independent chemical and UI cohorts once. Require a one-sided 95% upper bound of at most 5% for false acceptance and a lower bound of at least 80% for positive acceptance. Preserve failed, unknown and unprocessed cases; do not repeat confirmation until it passes.

Check specialist dependencies and renderer isolation on the execution host with doctor and task-status. Procedural positive, wrong-answer and prohibited-syntax checks do not establish general natural-image accuracy. Uncalibrated specialists can use evaluate-specialist but remain unavailable to normal selection.

Question audit ballots hide the answer and are separate from answer ballots. Preserve three independent ratings and adjudication. Until actual ballots arrive, human quality, false acceptance/rejection and human-approved conversations per GPU-hour are unmeasured. Small answer-exposure, history and ablation regressions verify the execution paths; they do not establish research performance differences.

Use explicitly selected idle GPUs and tmux. Save GPU state, endpoint responses and output progress every ten minutes. Start the reservation watcher immediately after every job, regardless of success, and verify the actual allocation. Reservation time belongs in cumulative GPU costs.

## Reproducing the checks

Use `pixelogue run-diagnostics` to count attempted turns, committed turns and completed
conversations separately. Use `pixelogue task-status --junit` to attach actual CPU results;
shared contracts and named task boundaries are distinct evidence. The final CPU inventory
records all 72 shared paths and named boundary tests for 41 tasks. It does not establish
natural-image accuracy across all 72 tasks.

The all-specialist example is `configs/specialist-pilot.yaml`. It explicitly enables all
seven extensions for capability inspection. It uses the temporary shared Qwen3.5-9B endpoint and does not provide evaluator-model diversity; its calibration cannot be transferred to the standard pair. Normal selection still requires dependencies
and an eligible certificate bound to the exact model pair and validator version. See the
[specialist and research guide](../tasks/specialist-and-research.md) for the evaluation,
calibration, exposure, history, ablation and audit commands.

Keep generated comparison, calibration, audit and cost reports under `artifacts/`.
Back up completed runs through the SQLite snapshot API and verify artifact hashes with
`pixelogue replay`. Do not copy an active WAL database file alone.

## Chemical confirmation result

The independent chemical cohort had 30 positive and 60 negative image groups. Eight positives were accepted; no negative was accepted. All 90 results completed with no failed or missing case. Seventy results were UNKNOWN because chemical evidence was incomplete or disagreed. CPU replay from the saved blind readings matched all 90 verdicts. The one-sided 95% positive-acceptance lower bound was 14.02%, and the false-acceptance upper bound was 4.87%. The positive criterion was unmet, so the version-5 chemical validator remains uncalibrated for this model pair and domain.

## UI confirmation result

The independent UI cohort had 50 positive and 100 negative image groups. Thirty-seven positives were accepted; no negative was accepted. All 150 results completed with no failed or missing case. Twelve positives and 24 negatives abstained; one positive was NOT_MET. CPU replay matched all saved verdicts. The one-sided 95% positive-acceptance lower bound was 61.87%, and the false-acceptance upper bound was 2.95%. The positive criterion was unmet. Both measured specialist domains remain ineligible for normal selection; the inventory still has 65 normally selectable core tasks and seven uncalibrated extensions.

## Research regression result

Answer exposure completed twelve trials (two cases, three conditions, two evaluators) without failures or missing trials. Every condition accepted both questions, so plausible-minus-hidden was zero for each model. AIAS remains unmeasured without independently labelled invalid questions.

Two history cases passed both the alternative-history and answer-witness checks; the independent case had no witness.

All 36 ablation trials across 18 cells completed. Both images reached both turns in every cell: depth one committed 36 turns; depth two committed 28, rejected four and abstained four. The before-answer/both-evaluators/verified-history cell committed two turns for each image. These are fixed-question regressions, not automatic candidate-generation results or 36 independent images.

Research expansion stays deferred until actual human ballots and stable main-path quality are available.
