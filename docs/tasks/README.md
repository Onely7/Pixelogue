# Task catalog v7

[日本語](README_ja.md) · [72 operation definitions](TASKS.md) · [185-subset research map](finevision_mapping_185.md)

See [Specialist and research guide](specialist-and-research.md) for the optional environment, calibration and experiment commands.

The installed catalog contains **65 core candidates and 7 validator-gated extensions**, organized into 14 families. The research map preserves 185 subsets and their 527 operation phrases. These are design/provenance records, not source annotations, training records, or evidence that any image supports a task. They never enter model requests.

## Catalog membership and runtime admission

`pixelogue compile` emits the complete catalog, strict JSON Schemas, an advisory 24-ID migration map, and `task_admission`. Every `task_admission` entry reports the required validators and concrete reasons that normal-profile generation is blocked.

All **65 core operations** now have normal verification paths. The 31 newly connected operations use blind structured extraction followed by deterministic checks for finite sets, arithmetic, tables, charts, documents, formulas, graphs, scales, marked geometry and finite patterns. Unknown notation, incomplete extraction, ambiguous results or disagreement between extractors causes abstention. `grounded_arithmetic` uses the same exact numeric engine, including its original add/subtract/multiply/divide cases. Public precision and unit rules determine admissible calculations.

Table source extraction uses the configured evidence token allowance. If a response reaches that limit before producing complete JSON, one blind retry can use up to twice the allowance, with an 8,192-token ceiling unless the configured allowance is already higher. A second incomplete response still abstains; no table cells are filled from the candidate answer.

Chart verification also supports bars whose exact values are printed next to the marks but whose numeric axis has no labeled ticks. The extractor marks that axis `unmarked` and cannot use it for pixel-based estimates. Every relevant value must have a directly printed label; an unlabeled mark remains unverified. Calibrated linear and log axes still require at least two ordered labeled ticks.

All 7 specialist validators are implemented within declared first-version ranges and can be named in `tasks.enabled_extensions`. Normal selection still requires an exact model-bound calibration certificate, a working validator environment and image-local eligibility. The all-seven example is `configs/specialist-pilot.yaml`. On a host without a working OS sandbox, both code reconstruction operations report an environment block. Static SVG/limited TikZ and HTML/CSS are rendered only within that sandbox. A UI action is checked as data and never executed.

## Per-image flow

1. Extract bounded scope records in the actual delivered view. Each capability has a local region, evidence ID and `MET`, `NOT_MET` or `UNKNOWN` observation. Missing observations are unknown. Capabilities from different scopes cannot be pooled.
2. Filter operation templates by local evidence and implemented validators. Rotate deterministically among eligible families, then diversify operations. The default candidate budget is 8, independent of the total catalog size.
3. Bind concrete public targets, parameters and eligibility checks to those templates. Every normal guard must be `MET`. Parameters sourced from pixels or history require references to the same scope or committed public messages. Output verdict vocabularies and fixed policies are not bindable desired answers.
4. Reject unsupported parameter combinations, duplicate operation fingerprints and output estimates above the configured budget before calling the selector. Fingerprints include catalog, image, scope, profile and normalized public choices, independently of turn index and question wording.
5. The configured selector sees the image, committed public history and sanitized candidate contracts. Private capability verdicts, source names, annotations and other judges' decisions are excluded.
6. Generate one operation-specific question. Both blind judges check the actual question, public parameters and local eligibility before answer generation. Compound independent requests are rejected rather than assigned the first operation's label; explicit subtask composition is not yet a generation interface.
7. Holistic mode retains two blind whole-turn reviews and the compact expected-operation contract. Both modes also require every applicable operation validator. Each structured operation obtains two answer-blind image extractions before parsing the candidate answer. The controller checks exact set, arithmetic, schema and topology conditions and keeps the evidence private.

A reciprocal object-identification question is rejected before the question judges when the new target label already appeared in a committed identification question and the new question repeats that turn's answer label. The same public-text check runs when saved conversations are re-evaluated. Distinct targets with no such two-way disclosure remain eligible.

`visible_action_relation` requires an action or contact shown in the still image. An overt question about what an object can or could do is stopped before the question judges, including when a vehicle's design merely suggests an action. The blind judges also check less explicit ability claims; a static pose alone does not establish movement.

A second turn is not automatically history-dependent. Requested regrouping can be useful without a new visual fact. The default `normal` profile requires all normal capabilities. Opt-in `limitation` instead requires a local anchor and an observed missing/uncertain relevant capability, followed by independent verification of the specific limitation. `false_premise` conservatively also requires closed scope and independent local contradiction checks. Failure to find something never establishes absence. Alternative profiles do not run the normal transcription/count verifier against an intentionally unanswerable question.

## Configuration and budgets

```yaml
tasks:
  catalog_version: "7.0"
  candidate_limit: 8
  max_scopes: 4
  max_observations_per_scope: 20
  evidence_max_tokens: 4096
  binding_max_tokens: 4096
  answer_max_tokens: 1024
  profiles: [normal]
  enabled_extensions: []
```

Scope and observation limits are operational settings, not a demand to enumerate all 50 capabilities. Focused candidate binding follows the initial scope discovery. A dense page must be bounded publicly, given a reviewed larger budget, or skipped. Public question/answer completions ending at the token limit are rejected even if JSON delimiters could be repaired. A valid schema or two matching judges is not proof of visual truth.
Each candidate exposes separate required check IDs, allowed public parameter names, and required public parameter names. A binding with MET checks but missing a required choice is a contract error with a bounded retry; the controller never fills it in automatically.
Binding input also lists that candidate's local evidence IDs and the IDs required by its capabilities. A reference from another scope is rejected with the invalid and allowed IDs in private retry feedback.
The controller replaces verbose model-generated evidence IDs with short, unique per-image IDs before binding. It saves the original-to-local mapping privately; regions, verdicts and scope membership are preserved.
Visible attribute lookup requires a public `attribute` parameter naming the requested property without its value. Distinct properties on one target receive distinct candidate identities, so a later turn can ask about another visible property while exact repeats remain excluded. Salient, independently visible properties are named separately during evidence discovery.
Discovery checks the main subject's visible attributes even when its actions or background are also described. Candidate admission removes an object-identification request when the same scope's public dialogue already names its target; the private rejection record preserves the reason. Attribute requests hide the source target description from question generation when it could disclose the value.
The subject's attribute evidence belongs in its entity scope so it can support a later request. A question asking for body posture is treated as an attribute request; an action request asks what the subject is doing or how it interacts with another visible object.

The evidence response Schema names only the supplied capability vocabulary and permits each capability key once per scope. Duplicate JSON keys still fail strict decoding; missing observations remain unknown. A repeated question or an object-identification question that names its selected target gets bounded feedback and another model attempt before any question evaluator call. Model-facing operation contracts omit answer-bearing target labels and free-form scope descriptions for object identification and scene categorization. Scene categorization binds its private target to one image-supported category, while the question must state at least two contrastive, nonoverlapping public alternatives. Before answer rating, deterministic checks reject a short object label already in the question or same-scope history, including simple quantity, separator and plural variants; quoted transcription answers; and action answers already stated in the question or same-scope history. These checks preserve the minimum conversation length and the existing visual quality gates.
For action questions on still images, lack of motion blur alone is insufficient evidence that an object is stationary.
The model-facing action contract hides the free-form target and scope description, which may themselves contain the requested posture. Question generation must identify the subject with non-action cues and abstain if that is impossible.
Text-transcription questions use an absolute image region or public scope. Relative locators such as "below the logo" are rejected before judging because the current transcript evidence has no coordinates to verify the relation; rerating applies the same gate.

`tasks.max_candidate_attempts` defaults to `1`. An evaluation configuration may set it to `2`
to try one additional admitted candidate after a local question-generation failure or a
`NOT_MET` question gate. The controller fixes the order before either attempt. Both candidates
see the same committed public prefix; rejected text and votes remain private. An unknown gate,
answer failure, transport error, or exhausted budget stops the turn. A normalized question
already judged in the failed attempt cannot be judged again. Private attempt plans and outcomes
retain stable identities for replay, and the coordinator commits at most one turn.

`tasks.question_operation_guidance` defaults to `baseline`. The experimental
`object_identification_v1` value adds fixed public guidance only to question generation
for normal V7 object identification: ask for a visually supported category, keep the
same target, and distinguish identification from label reading or UI function explanation.
The generator receives no private category answer. Selector and evaluator contracts,
target-disclosure checks, and question and answer gates remain unchanged. Compare actual
completed conversations before adopting this option; an intent-check improvement alone
does not establish a synthesis improvement.

`tasks.fact_novelty_enabled` defaults to `false` while its effect is compared. When enabled,
private request identities combine the view, scope, subject, operation, requested property or
count unit, and public conditions. Formatting alone does not create a new fact. The existing
attribute filter keeps fur color and nose color distinct; an explicit legacy count and the same
current count share a fact without relabeling either artifact. Only committed facts are used.
Broad descriptions, summaries, action requests, arbitrary QA and relations without a declared
dimension remain unresolved, so this filter does not replace the blind novelty gate.

`tasks.initial_binding_batch_size` defaults to `8`. The comparison value `2` binds the first
two scheduled candidates, extending by two only when none remains admissible and new. It never
exceeds the original eight-candidate pool. A malformed response or execution failure stops the
binding stage. Batch membership and admitted IDs are saved privately; task coverage must be
measured because accepting the early batch can omit a later specialized opportunity.

Public operation contracts include the exact view, original scope box and resolved subject box.
Binding preserves the subject's entity box for identification, attributes and actions, even if
the target was an instruction choice. Attribute evidence marked `MET` must lie within the
same subject's entity box; a broad scene box does not authorize borrowing a neighbor's property.
The optional single attribute recheck uses that subject box and retains unknown or invalid
evidence without enlarging it. A generic person category is permitted for identification;
an individual's identity remains excluded. These geometric checks do not establish visual
truth, and action, reference, category granularity and closed-set requirements still need the
blind question and answer checks.

`tasks.evidence_format` accepts `keyed` (default), `array`, or the comparison format `compact`.
The compact format keeps capability, verdict, visual detail and every region, while the
controller assigns the image, view, scope and observation IDs. It supplies no visual verdicts
and fills no missing observations. Unknown extra IDs, duplicate capabilities and invalid
regions remain rejected. The selected format and Schema participate in the run and call
identities; changing them requires a new run.

## Migration and analysis

Use a **new run ID**. The run identity includes configuration, code, prompts, Schemas, catalog, specialist lock, model and processor revisions, seed and the prepared input/rights manifest identity. Old 24-task artifacts retain their original labels; `task_catalog_legacy.yaml` and `legacy_24_migration.json` provide advisory mappings only. In particular, `chart_lookup` requires reclassification from the actual question, not automatic assignment to two new operations. Old stores can still be inspected with `replay` and backed up.

Synthesis and rerating write an `.operations.json` sidecar counting all committed operation IDs and the final committed operation per conversation. Stopped tails are excluded. V7 selection uses the final committed family as its primary label; the semantic fingerprint retains the full task sequence. Export records operation IDs and catalog versions only in provenance, keeping training messages public-only. Existing evaluation-source exclusions and rights checks remain in force.

The selection example uses current family names. Its small quotas illustrate constraints; neither that example nor the initial family rotation is an empirically established sampling policy. Derive production quotas from eligible input and observed coverage; natural photographs cannot guarantee document or diagram coverage.

## Verification boundary

CPU tests cover shared contracts, structured operation fixtures, schema/reference rejection, scope isolation, guarded profiles, parameter lineage, bounded output, family rotation, migration identities and information boundaries. `pixelogue task-status --junit artifacts/cpu-junit.xml` records which shared fixture modules passed for each task; it does not imply per-task natural-image accuracy. Model-bound specialist calibration, judge error rates, cross-language fidelity, model cost and downstream training benefit still require external evaluation. Existing live rubric probes cover only their original scenarios.
