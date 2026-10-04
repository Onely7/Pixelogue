# Comparing synthesis yield and speed

This guide describes the P0–P8 comparison campaign. Experimental options preserve the
minimum of two committed turns, the existing quality gates, the eight-candidate limit,
and the configured model pair. An automatic `QUALITY_CANDIDATE` is not a human-approved
conversation. The planned final audit uses three independent raters. If the user
explicitly chooses a one-rater interim review, record the policy and report those
results as provisional. They do not measure inter-rater agreement or satisfy the
three-rater audit. Missing confirmation ballots remain unmeasured even when development
ballots are complete.

## Fix the comparison before running

Keep immutable source archives and record the commit, prompts, Schemas, catalog,
validators, model and processor revisions, input hashes, visual groups, seed, language,
generator allocation and planned depth. Use the same generator for a conversation's
lifetime. Repetitions are separate runs and separate calls, even with the same seed;
resuming the same trial reuses its saved response.

Serving identity includes the actual vLLM version, structured-output backend, whitespace
control and server manifest hash. Changing a model, input, operation, public history or
contract invalidates reuse. A private blind-judge trial ID distinguishes two calls to
one endpoint without changing their model-visible inputs. A shared endpoint does not
establish evaluator-model diversity.

Compare these evidence conditions first:

| Condition | Evidence wire format | Serving control |
|---|---|---|
| A | `keyed` | Current pinned configuration |
| B | `array` | Current pinned configuration |
| C | `keyed` | Explicit backend plus whitespace control |
| D | `array` | Explicit backend plus whitespace control |

If the backend changes with whitespace control, interpret the combined intervention.
Verify spaces inside strings, Japanese and escaped newlines separately. Array and
compact formats retain visual descriptions, verdicts and regions; missing observations
remain unknown. Compact removes controller-assigned IDs. It is a different contract
from earlier experiments that shortened explanations.

## Compare one change at a time

These defaults remain in place while experimental results and human quality are pending:

| Configuration | Default | Comparison value |
|---|---|---|
| `tasks.evidence_format` | `keyed` | `array` or `compact` |
| `tasks.attribute_recheck_enabled` | `false` | `true`, one bounded recheck |
| `tasks.fact_novelty_enabled` | `false` | `true` |
| `tasks.max_candidate_attempts` | `1` | `2` |
| `tasks.question_operation_guidance` | `baseline` | `object_identification_v1` |
| `tasks.initial_binding_batch_size` | `8` | `2` |
| `runtime.refill_completed_images` | `false` | `true` |
| `runtime.max_concurrent_images` | `2` | `4` |

Use a new run ID for every condition and repetition. Keep full-path control/variant
order balanced. For speed comparisons, use one model session and apportion its loading
and cleanup time explicitly. Record generation-only duration separately. Strict subject
and region containment is a safety contract; a lower automatic yield does not justify
borrowing a neighboring object's attribute or enlarging the evidence region.

Fact identities use explicit subjects, dimensions and public conditions. Broad or
ambiguous requests remain unresolved. The blind novelty gate still runs. A fallback
candidate uses the same committed prefix and a priority fixed before evaluation. Failed
questions, answers and votes stay private; unknown evidence, answer failures and
transport errors do not trigger additional candidates.

The opt-in `object_identification_v1` guidance applies only to question generation
for normal V7 object identification. It requests a visible category at a supported
granularity, keeps the bound target, and distinguishes category identification from
reading a label or explaining a UI function. It never supplies the private category
answer to the generator. Selectors and judges receive the original public contract;
the usual target-disclosure, question and answer gates still run. Treat this as an
experimental option until a full-path comparison supports adoption.

## Run a finite improvement cycle

Freeze 100 diverse development images and a diagnostic subset of approximately 25.
Screen each proposed change on that subset before running a 100-image comparison.
Set a maximum number of hypotheses and corrective variants in advance. Reject a
variant when its proposed mechanism fails; preserve its actual failed calls and
costs without repeating the same expensive failure merely to enlarge the sample.

Select one measured condition, check it against the current version on a second
generation seed, then compare both conditions on the same 100 images. Keep planned
depths fixed across seeds: changing generation seed must not change the conversation
length being compared. Balance execution order within each native host, use one
model session, and preserve the image-to-generator allocation. Report screening,
confirmation and final comparison separately. Previously inspected images remain
development inputs, including the second-seed run.

An independent AI reference may inspect saved question and answer ballots. Freeze
each question depth before releasing the next depth: later public histories contain
earlier answers. Release answer ballots only after all question depths are frozen.
Use full image and public-context hashes for exact reuse, and retain `UNKNOWN` and
missing votes. Label the result as an AI
reference, with its model version; it does not replace human ground truth or establish
specialist calibration. Count a reference-valid completed conversation only when
every committed question and answer satisfies every required reference criterion.

An initial two-candidate batch can omit later opportunities when it finds a valid early
candidate. Measure task exposure and coverage as well as time. Refill preserves input
output order, conversation history order, bounded active work and a bounded waiting
buffer. Keep SQLite operations serialized.

## Recorded development comparison: 2026-10-04

The finite cycle screened changes on 25 diverse development images, checked the
selected condition on a second generation seed, and compared it with the current
condition on the same 100 images. The selected condition combined compact evidence
with an initial binding batch of two candidates. Models, quantization, quality gates
and planned conversation depths were unchanged. Native host order was balanced;
these measurements used the pinned model pair in verified TP1 serving configurations.

| Measure on the matched 100 images | Current | Compact plus initial batch of two |
|---|---:|---:|
| Automatic candidates | 6 | 14 |
| Whole conversations satisfying the AI reference | 4 | 9 |
| Mean image synthesis seconds | 126.53 | 102.21 |
| p95 image synthesis seconds | 217.76 | 201.67 |
| Maximum image synthesis seconds | 327.52 | 376.64 |
| Tasks in reference-valid conversations | 3 | 6 |
| Reference-valid conversations per phase GPU hour | 1.39 | 3.93 |

These counts use the unchanged minimum-two-turn policy, including retained committed
prefixes after a later stop. Three current and five selected reference-valid
conversations finished their planned depth; one and four respectively are retained
prefixes. Only committed public turns enter the reference-valid count.

The mean fell 19.2%. Summed evidence and binding HTTP durations fell 37.3%; this
sum overlaps in time and is not an elapsed-time reduction. Candidate review covered
94 actual question/answer ballots using GPT-6.1 Sol Ultra. The selected candidates
include three reference failures and two unknown conversations; the current candidates
include two failures. Independent human correctness and holdout generalization remain
unmeasured. The target of ten reference-valid conversations was not met.

Defaults remain keyed evidence and an initial batch of eight. The selected condition
is available for explicit comparison. It improves measured yield and mean speed, but
does not establish preserved candidate quality or every operation opportunity. Two
eligible multi-region opportunities were not offered in the selected condition, and
the maximum image duration increased. Separate evidence-format and batching effects
cannot be inferred from this combined condition. Other screened changes remain
disabled by default; the failed large-table prompt change was withdrawn.

Report actual phase intervals for each native host. The first-to-last campaign span
can include other conditions and gaps when execution order is balanced, so it cannot
measure comparative throughput. Phase efficiency excludes shared model loading;
the complete cycle used 12.21 GPU hours including loading, failed and interrupted
tasks, and gaps. Reservation time after job completion remains separate.

## Report all work and missing measurements

Use `pixelogue run-diagnostics` for stage reach, attempted turns, committed turns and
terminal conversations. Use `pixelogue task-status --junit` to distinguish shared CPU
contracts, task-specific boundaries and actual GPU answer cases. A task registration or
passing fixture is not natural-image accuracy.

Keep these quantities separate:

- Actual HTTP attempts, semantic corrections, transport retries and cache accesses.
- Valid and invalid outputs with known input/output usage and missing usage counts.
- Concurrent HTTP duration sums, image elapsed time, phase wall time, model loading,
  cleanup and reservation time.
- Committed turns across all outcomes and committed turns in completed candidates.
- Generation stops, question rejection, answer rejection, uncertainty and execution error.

Invalid output still consumes measured usage. Missing usage is unknown, not zero. Reuse
must not charge the same HTTP response twice or produce duplicate commits. Compare raw
journals, budget totals, public output hashes and commit counts after interruption.
Campaign GPU costs include loading and every reserved device; union overlapping intervals
for the same GPU to avoid double counting.

Text-only answer parsers are attributed using the public operation's view ID and the
exact saved question. Ambiguous question matches retain an unresolved depth. Global
HTTP totals include calls without resolved image attribution; `unattributed_usage`
keeps their measured duration, tokens and missing-usage count separately.

## Separate image distribution from implementation

Open Images validation photographs can exercise visual description, attributes, spatial
relations and some set operations. They do not establish document, table, chart, diagram,
UI or specialist task coverage. Previously inspected images are development inputs.
Freeze a fresh confirmation cohort before tuning and do not adjust from its results.

Report within-source paired differences, task funnels, depth starts and reaches, and
coverage at an equal number of accepted turns. Bootstrap visual groups with repetitions
kept together. Source and image contents change together, so cross-source differences
are descriptive. Query-based categories are not independent operation-opportunity labels.
Small structured-image strata are scope checks, not precise general success rates.

## Audit and portable examples

Keep the audit sampling manifest, method/model mapping, automatic statuses and selection
probabilities outside the material distributed to raters. Sample one conversation per
visual group when possible. Release question packets in depth order after all required
raters' previous ballots are saved; prior public answers in later histories cannot then
change earlier question votes. Release answer packets after question votes are frozen.
Preserve original votes, disagreements, adjudication, unknown and unevaluated items.

Keep development and confirmation pack identities separate. A completed development
review cannot label later confirmation outputs. For an explicitly requested one-rater
review, use the [local audit UI](../tasks/specialist-and-research.md) and preserve the
actual ballots. The resolution CLI still defaults to three raters; use an explicit
`--required-raters 1` only for the recorded provisional policy.

Stops without an answer do not enter completed-conversation false-rejection estimates.
Use actual inclusion probabilities for population estimates; observed cell fractions
alone need not be inclusion probabilities. An empty ballot template is not a human vote.

The final `examples.html` embeds its images and contains actual committed
image/instruction/answer examples for the explicitly named display condition.
The report compares both conditions. All 72 catalog rows stay visible, and tasks
without an observed example remain empty. Exclude rejected terminal text from public
dialogue and mark committed prefixes from stopped conversations as diagnostics.
Keep validation inputs and their visual groups evaluation-only.

See the [task guide](../tasks/README.md), [GPU guide](../models-and-gpu.md),
[recovery guide](../recovery-and-ci.md) and
[previous quality/performance results](quality-and-performance.md).
