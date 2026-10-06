# Task catalog v7

[日本語](README_ja.md) · [72 operation definitions](TASKS.md) · [185-subset research map](finevision_mapping_185.md)

See [Specialist and research guide](specialist-and-research.md) for the optional environment, calibration and experiment commands.

The installed catalog contains **65 core candidates and 7 validator-gated extensions**, organized into 14 families. The research map preserves 185 subsets and their 527 operation phrases. These are design/provenance records, not source annotations, training records, or evidence that any image supports a task. They never enter model requests.

## Catalog membership and runtime admission

`pixelogue compile` emits the complete catalog, strict JSON Schemas, an advisory 24-ID migration map, and `task_admission`. Every `task_admission` entry reports the required validators, their environments, any certified calibration domains, and the concrete reasons that normal-profile use is blocked.

All **65 core operations** have normal verification paths, and synthesis drafts questions only for these core operations. The 31 operations connected in v7 use blind structured extraction followed by deterministic checks for finite sets, arithmetic, tables, charts, documents, formulas, graphs, scales, marked geometry and finite patterns. Unknown notation, incomplete extraction, ambiguous results or disagreement between extractors causes abstention. `grounded_arithmetic` uses the same exact numeric engine, including its original add/subtract/multiply/divide cases. Public precision and unit rules determine admissible calculations.

Table, table-lookup, chart, graph and specialist source readers use `tasks.source_max_tokens` (default 4096). A chart or graph reading that stops at that limit before producing complete JSON is retried with up to twice the allowance, capped at 8,192 tokens unless the configured allowance is already higher. A table reading keeps its allowance and must return `UNKNOWN` when the complete table does not fit. No table cell is filled from the candidate answer.

Chart verification also supports bars whose exact values are printed next to the marks but whose numeric axis has no labeled ticks. The extractor marks that axis `unmarked` and cannot use it for pixel-based estimates. Every relevant value must have a directly printed label; an unlabeled mark remains unverified. Calibrated linear and log axes still require at least two ordered labeled ticks.

All 7 specialist validators are implemented within declared first-version ranges and can be named in `tasks.enabled_extensions`. `task_admission` reports an enabled extension as available only with an exact model-bound calibration certificate and a working validator environment; the all-seven example is `configs/specialist-pilot.yaml`. Direct drafting never offers an extension, so extensions are exercised through `evaluate-specialist`. On a host without a working OS sandbox, both code reconstruction operations report an environment block. Static SVG/limited TikZ and HTML/CSS are rendered only within that sandbox. A UI action is checked as data and never executed.

## Per-image flow

The [generation guide](../pipeline/generation-and-evaluation.md) describes each step in detail.

1. The router model (`Qwen/Qwen3.5-2B`) profiles the image once and lists the families its visible content can support. Malformed output falls back to broad default families.
2. For each planned turn, the controller routes a primary family with up to four operations and a secondary family with up to two. It prefers families not yet used in the conversation and with the largest deficit against the run-wide family targets. Routes are saved, so a resumed turn receives the same offer.
3. The conversation's generator drafts up to `tasks.draft_count` questions for the offered operations. Each draft carries its public target and parameters, its scope and target regions, and a private fact key. The controller validates parameters against the catalog and applies deterministic public-text checks.
4. Both blind judges gate each admitted question, one call each, before any answer exists: local anchor, operation coherence, usefulness, and an independent operation label.
5. Only the first question that passes is answered. Both judges review the answer as a whole with the expected operation contract; one full-view tie-break and one answer repair are the only follow-ups.
6. Every applicable operation validator runs before the turn is committed. Each structured operation obtains two independent readings and keeps the evidence private.

### Routing rules

The first `tasks.anchor_turns` turns (default 2) offer only lightly verified operations: those whose verification contracts are limited to `dual_visual_review`, `evidence_binding_check` and `transcript_alignment`. With the current catalog, 26 of the 65 core operations qualify. These turns decide whether a conversation reaches its two-turn minimum, and structured extraction verifiers abstain more often.

`screen_ui` operations are offered only for images that the router profiled as `screen`; annotated figures, diagrams and photographs are not screens. When the profile is missing or names no feasible family, routing uses `visual_description`, `text_reading`, `reference_spatial`, `set_logic` and `evidence_verification`.

`tasks.family_targets: uniform` gives every available family the same target share. A mapping of family IDs to positive weights sets other targets; unlisted families get no target share but remain routable. `tasks.task_weights` holds weights in (0, 1] that divide an operation's committed count during operation ordering. Its default of 0.25 for table, chart and document structure reconstruction keeps whole-structure reconstructions reachable but offered less often, because their verifiers need complete grids or series and abstain more often than targeted lookups.

### Draft admission

A draft names exactly one offered operation. Its parameters must use the operation's bindable names, include every required public choice, and choose only permitted values. The controller never fills a missing choice. Visible attribute lookup requires a public `attribute` naming the requested property without its value, and scene categorization requires at least two distinct options in `category_set`. For verbatim text, reading order and code transcription, the whole drafted scope becomes the bound text region.

The fact key, a subject and a dimension, becomes a normalized private request key; articles and `colour`/`color` spelling do not create a new fact. A draft whose request key matches a committed turn of the conversation is rejected, and earlier fact keys are listed for the drafter so it can avoid them. Distinct properties of one subject, such as fur color and nose color, remain separate facts.

Object-identification drafts refer to the subject by location or non-category traits. An answer whose short label already appears in its own question is rejected before review. Re-rating additionally rejects a reciprocal identification question when the new target label already appeared in a committed identification question and the new question repeats that turn's answer label; distinct targets with no such two-way disclosure remain eligible.

`visible_action_relation` requires an action or contact shown in the still image. An overt question about what an object can or could do is rejected at draft admission, including when a vehicle's design merely suggests an action. The blind judges also check less explicit ability claims; a static pose alone does not establish movement, and a body-posture question is an attribute request. Lack of motion blur alone is insufficient evidence that an object is stationary.

Text-transcription questions use an absolute image region or public scope. Relative locators such as “below the logo” are rejected at draft admission because the transcript evidence has no coordinates to verify the relation; re-rating applies the same check. A scene-categorization question must state every option in its category set.

Under `evaluation.question_gate_label_policy: same_contract`, a judge label that names a neighboring operation with identical verification contracts is accepted when the other label names the drafted operation, because the label then changes only provenance. When both judges name the same such neighbor and the draft already carries its required public choices, the turn is relabeled. `strict` requires both labels to name the drafted operation.

A second turn is not automatically history-dependent, and a requested regrouping can be useful without a new visual fact. Direct drafting creates only `normal`-profile operations. The catalog's `limitation` and `false_premise` profile contracts remain valid for saved turns and re-rating, where failure to find something never establishes absence.

## Configuration and budgets

```yaml
tasks:
  catalog_version: "7.0"
  source_max_tokens: 4096
  answer_max_tokens: 1024
  enabled_extensions: []
  draft_count: 2
  draft_max_tokens: 1024
  extra_draft_calls_per_turn: 1
  profile_max_tokens: 384
  anchor_turns: 2
  family_targets: uniform
  task_weights:
    table_structure_reconstruction: 0.25
    chart_data_reconstruction: 0.25
    document_structure_reconstruction: 0.25
```

`tasks.calibration_manifest` may name a specialist calibration manifest; a relative path resolves against the configuration file. An unknown family in `family_targets`, an unknown task in `task_weights`, or an unknown or repeated extension fails validation, as do settings of the retired scoped planner. Public question/answer completions ending at the token limit are rejected even if JSON delimiters could be repaired. A valid schema or two matching judges is not proof of visual truth.

Model-facing operation contracts omit answer-bearing target labels and free-form scope descriptions for object identification, attribute lookup, visible actions and scene categorization. Public operation contracts include the exact view, scope region and target region. A generic person category is permitted for identification; an individual's identity remains excluded. Geometric checks do not establish visual truth, and action, reference, category granularity and closed-set requirements still need the blind question and answer checks.

## Migration and analysis

Use a **new run ID**. The run identity includes configuration, code, prompts, Schemas, catalog, specialist lock, model and processor revisions, seed and the prepared input/rights manifest identity. Old 24-task artifacts retain their original labels; `task_catalog_legacy.yaml` and `legacy_24_migration.json` provide advisory mappings only. In particular, `chart_lookup` requires reclassification from the actual question, not automatic assignment to two new operations. Turns saved by the retired scoped planner keep `origin: scoped`, their evidence references and requirements; they remain readable and can be re-rated. Old stores can still be inspected with `replay` and backed up.

Synthesis and rerating write an `.operations.json` sidecar counting all committed operation IDs and the final committed operation per conversation. Stopped tails are excluded. V7 selection uses the final committed family as its primary label; the semantic fingerprint retains the full task sequence. Export records operation IDs and catalog versions only in provenance, keeping training messages public-only. Existing evaluation-source exclusions and rights checks remain in force.

The selection example uses current family names. Its small quotas illustrate constraints; neither that example nor the default uniform family targets is an empirically established sampling policy. Derive production quotas from eligible input and observed coverage; natural photographs cannot guarantee document or diagram coverage.

## Verification boundary

CPU tests cover shared contracts, structured operation fixtures, schema and reference rejection, draft parameter contracts, routing and ledger rules, question-gate label rules, holistic follow-ups, bounded output, migration identities and information boundaries. `pixelogue task-status --junit artifacts/cpu-junit.xml` records which shared fixture modules passed for each task; it does not imply per-task natural-image accuracy. Model-bound specialist calibration, judge error rates, cross-language fidelity, model cost and downstream training benefit still require external evaluation. Existing live review probes cover only their original scenarios.
