# Task catalog v8

[日本語](README_ja.md) · [76 task definitions](TASKS.md) · [185-subset research map](finevision_mapping_185.md)

See [Specialist and research guide](specialist-and-research.md) for the optional environment, calibration and experiment commands.

The installed catalog contains **69 core tasks and 7 validator-gated extensions** in 17 families. [TASKS.md](TASKS.md) is generated from `src/pixelogue/resources/task_catalog.yaml`. Regenerate it after every catalog edit; a test fails when the committed file differs from the catalog:

```bash
uv run --locked pixelogue compile --tasks-markdown docs/tasks/TASKS.md
```

## How a task is defined

Every task has these fields:

- `label` is a short, verb-first name, such as "Read a value from a chart".
- `definition` says in plain English what the user asks and what the answer gives. When a neighboring task is easy to confuse, a final sentence names it, such as "Use count_comparison instead to compare two counts".
- `parameters` are the choices that the question itself must state. Each one has a kind (`text`, `choice`, `list` or `integer`), a description and a `required` flag. A choice lists its values, and a list may bound its number of items.
- `answer_format` is present when a validator compares a specific answer shape, such as the JSON of field extraction or chart data, or a fixed verdict vocabulary.
- `do_not_infer` names what the answer must not claim, and the task lists its required image capabilities, eligibility checks and verification contracts.
- `example_question` and `related_finevision_subsets` are for people only. They are never sent to a model.

The question drafter receives the definition, `do_not_infer`, the eligibility checks, the parameter contract and the answer format. The judges receive the same definition and answer format in the operation contract. Tests require all model-facing catalog text to be ASCII English without internal jargon, and every definition to state what is asked and what the answer gives.

## Catalog membership and runtime admission

`pixelogue compile` emits the complete catalog, strict JSON Schemas and `task_admission`. Every `task_admission` entry reports whether the task is available, the required validators and their environments, any certified calibration domains, and the concrete reasons that block it.

All **69 core tasks** have verification paths, and synthesis drafts questions only for core tasks. 35 of them use blind structured extraction followed by deterministic checks for finite sets, arithmetic, tables, charts, documents, formulas, diagrams, scales, marked geometry and finite patterns. Five knowledge tasks commit only when two blind readers give the same short answer, and object boxes must overlap two blind box readings. Panel comparison, panel sequences and UI element location use two blind visual contract reviews. The remaining 25 are light tasks, described under the routing rules. Unknown notation, incomplete extraction, ambiguous results or disagreement between extractors causes abstention. `grounded_arithmetic` uses the same exact numeric engine as the other numeric tasks. The precision and unit rules that the question states determine which calculations are admissible.

Table, table-lookup, chart, graph and specialist source readers use `tasks.source_max_tokens` (default 4096). A chart or graph reading that stops at that limit before producing complete JSON is retried with up to twice the allowance, capped at 8,192 tokens unless the configured allowance is already higher. A table reading keeps its allowance and must return `UNKNOWN` when the complete table does not fit. No table cell is filled from the candidate answer.

Chart verification also supports bars whose exact values are printed next to the marks but whose numeric axis has no labeled ticks. The extractor marks that axis `unmarked` and cannot use it for pixel-based estimates. Every relevant value must have a directly printed label; an unlabeled mark remains unverified. Calibrated linear and log axes still require at least two ordered labeled ticks.

All 7 specialist validators are implemented within their first-version ranges and can be named in `tasks.enabled_extensions`. `task_admission` reports an enabled extension as available only with an exact model-bound calibration certificate and a working validator environment; the all-seven example is `configs/specialist-pilot.yaml`. Direct drafting never offers an extension, so extensions are exercised through `evaluate-specialist`. On a host without a working OS sandbox, both code reconstruction tasks report an environment block. Static SVG, limited TikZ and HTML/CSS are rendered only within that sandbox. A UI action is checked as data and never executed.

## Knowledge, specialist and creative tasks

The [FineVision row audit](finevision_mapping_185.md#row-level-audit-2026-10-07) added three families that go beyond reading the image itself:

- `knowledge_recognition` names well-known landmarks, artworks, styles and map regions from world knowledge. People are never identified.
- `domain_reasoning` applies standard textbook knowledge: it explains labeled scientific or technical diagrams, interprets standard notation and solves math problems printed in the image.
- `grounded_creation` writes short creative texts. Mood and narrative may be invented, but every statement about what the image shows must be true.

The five tasks with short knowledge answers (`named_entity_recognition`, `style_recognition`, `map_region_identification`, `notation_interpretation` and `math_word_problem`) use `answer_consensus_check`. Two readers answer the question without seeing the candidate answer, and the turn commits only when both short answers match the candidate's after normalizing case, punctuation, leading articles and number format. Readers who disagree, or an alias such as a translated name, leave the turn uncommitted. Agreement between two models is evidence, not proof, so these tasks also pass both blind judges. `concept_explanation` and `grounded_creative_writing` are light tasks checked by the judges and by evidence binding.

The audit also added three tasks to existing families. `object_box_grounding` answers with normalized boxes, which `box_iou_check` matches one-to-one to two blind box readings at an overlap (IoU) of at least 0.5. `chart_value_arithmetic` computes a result from both chart readings, exactly for printed values and as a range for estimates. `visible_text_translation` translates a delimited piece of visible text. Medical images, multiple-choice answer formats and several images per question were not added; the audit table gives the decision for each subset.

## Per-image flow

The [generation guide](../pipeline/generation-and-evaluation.md) describes each step in detail.

1. The router (generator A's `Qwen/Qwen3.8-27B` server) profiles the image once and lists the families its visible content can support. Malformed output falls back to broad default families.
2. For each planned turn, the controller routes a primary family with up to four tasks and a secondary family with up to two. It prefers families not yet used in the conversation and with the largest deficit against the run-wide family targets. Routes are saved, so a resumed turn receives the same offer.
3. The conversation's generator drafts up to `tasks.draft_count` questions for the offered tasks. Each draft carries its target and parameters, its scope and target regions, and a private fact key. The controller validates parameters against the catalog and applies deterministic public-text checks.
4. Both blind judges gate each admitted question, one call each, before any answer exists: local anchor, task coherence, usefulness, and an independent task label.
5. Only the first question that passes is answered. Both judges review the answer as a whole with the expected operation contract; one full-view tie-break and one answer repair are the only follow-ups.
6. Every applicable validator runs before the turn is committed. Each structured task obtains two independent readings and keeps the evidence private.

### Routing rules

The first `tasks.anchor_turns` turns (default 2) offer only light tasks: those whose verification contracts are limited to `dual_visual_review`, `evidence_binding_check` and `transcript_alignment`. With the current catalog, 25 of the 69 core tasks qualify; [TASKS.md](TASKS.md) marks each task as light or structured. These turns decide whether a conversation reaches its two-turn minimum, and structured extraction verifiers abstain more often.

`screen_ui` tasks are offered only for images that the router profiled as `screen`; annotated figures, diagrams and photographs are not screens. When the profile is missing or names no feasible family, routing uses `visual_description`, `text_reading`, `reference_spatial`, `counting_and_sets` and `evidence_verification`.

`tasks.family_targets: uniform` gives every available family the same target share. A mapping of family IDs to positive weights sets other targets; unlisted families get no target share but remain routable. `tasks.task_weights` holds weights in (0, 1] that divide a task's committed count during task ordering. Its default of 0.25 for table, chart and document structure reconstruction keeps whole-structure reconstructions reachable but offered less often, because their verifiers need complete grids or series and abstain more often than targeted lookups.

### Draft admission

A draft names exactly one offered task. Its parameters must use the task's parameter names or `target`, include every required parameter, and match each parameter's kind: a choice must be one of its values, a list must hold distinct strings within its item bounds, and an integer must be a whole number. The controller never fills a missing choice. Attribute lookup requires an `attribute` naming the requested property without its value, and scene categorization requires two to five distinct options in `category_set`. For text transcription, including several text blocks and code, the whole drafted scope becomes the text region.

The fact key, a subject and a dimension, becomes a normalized private request key; articles and `colour`/`color` spelling do not create a new fact. A draft whose request key matches a committed turn of the conversation is rejected, and earlier fact keys are listed for the drafter so it can avoid them. Distinct properties of one subject, such as fur color and nose color, remain separate facts.

Object-identification drafts refer to the subject by location or non-category traits. An answer whose short label already appears in its own question is rejected before review. Re-rating additionally rejects a reciprocal identification question when the new target label already appeared in a committed identification question and the new question repeats that turn's answer label; distinct targets with no such two-way disclosure remain eligible.

`visible_action` requires an action or contact shown in the still image. An overt question about what an object can or could do is rejected at draft admission, including when a vehicle's design merely suggests an action. The blind judges also check less explicit ability claims; a static pose alone does not establish movement, and a body-posture question is an attribute request. Lack of motion blur alone is insufficient evidence that an object is stationary.

Text-transcription questions use an absolute image region or the drafted scope. Relative locators such as "below the logo" are rejected at draft admission because the transcript evidence has no coordinates to verify the relation; re-rating applies the same check. A scene-categorization question must state every option in its category set.

Under `evaluation.question_gate_label_policy: same_contract`, a judge label that names a neighboring task with identical verification contracts is accepted when the other label names the drafted task, because the label then changes only provenance. When both judges name the same such neighbor and the draft already carries its required parameters, the turn is relabeled. `strict` requires both labels to name the drafted task.

A second turn is not automatically history-dependent, and a requested regrouping can be useful without a new visual fact. A question about what the image cannot show is the `answerability_assessment` task; failure to find something never establishes absence.

## Configuration and budgets

```yaml
tasks:
  catalog_version: "8.0"
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
    table_reconstruction: 0.25
    chart_data_reconstruction: 0.25
    document_structure_reconstruction: 0.25
```

`tasks.calibration_manifest` may name a specialist calibration manifest; a relative path resolves against the configuration file. An unknown family in `family_targets`, an unknown task in `task_weights`, or an unknown or repeated extension fails validation, as do settings of the retired scoped planner. Public question/answer completions ending at the token limit are rejected even if JSON delimiters could be repaired. A valid schema or two matching judges is not proof of visual truth.

Model-facing operation contracts omit answer-bearing target labels and free-form scope descriptions for object identification, attribute lookup, visible actions and scene categorization. Operation contracts include the exact view, scope region and target region. A generic person category is permitted for identification; an individual's identity remains excluded. Geometric checks do not establish visual truth, and action, reference, category granularity and closed-set requirements still need the blind question and answer checks.

### Runtime feasibility

- The expected output must fit the configured response and context budget. Otherwise the question names a smaller region, a reviewed larger budget is configured, or the task is skipped. A truncated answer is never certified as complete.
- Eligibility is judged on the image view that the models actually receive, and coordinates refer to that same view.
- A task whose applicable validator is unavailable is not offered; a catalog entry alone is not a running implementation.
- Pattern rules are limited to the rule types that `pattern_rule` lists. When two reasonable answers fit, the validator abstains.

## Changes from catalog 7.0

Catalog 8.0 rewrites every definition in plain English, makes every required parameter explicit, and adds `answer_format` where a validator needs a specific answer shape. The verification semantics of the remaining tasks are unchanged, except that `diagram_element_lookup` no longer runs `graph_check`; it is now a light task.

| 7.0 task ID | 8.0 task ID |
|---|---|
| `visible_action_relation` | `visible_action` |
| `visual_summary`, `screen_summary` | merged into `grounded_description` |
| `referring_object_resolution` | `described_object_lookup` |
| `referring_expression_generation` | `distinguishing_description` |
| `predicate_selection`, `set_operation` | `select_by_conditions` |
| `attribute_grouping` | `group_by_attribute` |
| `set_cardinality_comparison` | `count_comparison` |
| `quantified_statement_verification` | `quantified_claim_verification` |
| `grounded_hypothetical_update` | `hypothetical_set_update` |
| `text_reading_order`, `code_transcription` | merged into `text_transcription` |
| `label_value_linking` | `label_value_lookup` |
| `text_visual_binding` | `text_object_binding` |
| `cross_region_consistency_check` | `stated_value_consistency` |
| `document_layout_role` | `document_element_role` |
| `table_predicate_selection` | `table_row_selection` |
| `table_structure_reconstruction` | `table_reconstruction` |
| `table_cross_reference` | `table_join` |
| `grounded_aggregation` | `value_aggregation` |
| `graph_connectivity` | `diagram_connectivity` |
| `graph_path_tracing` | `diagram_path_tracing` |
| `diagram_branch_evaluation` | `flowchart_evaluation` |
| `geometric_relation_analysis` | `geometric_relations` |
| `pattern_rule_identification` | `pattern_rule` |
| `rule_based_exception` | `pattern_exception` |
| `ui_element_grounding` | `ui_element_location` |
| `visible_sequence_description` | `panel_sequence_description` |
| `evidence_localization` | merged into `visual_claim_verification` |

Six families were renamed: `set_logic` to `counting_and_sets`, `chart_map_understanding` to `charts_and_maps`, `diagram_graph` to `diagrams`, `pattern_geometry` to `patterns_and_geometry`, `multi_region` to `multi_panel`, and `validated_extensions` to `specialist_notation`. Task status is now `core` or `extension`.

The catalog no longer has task numbers, counts, facets, classification rules, routing precedence, answer profiles (`limitation`, `false_premise`), policy-only parameters, inspiration subset numbers or source URLs. Routing precedence now lives in the definitions, the runtime feasibility rules are in this guide, and source URLs remain in the research map. The unused `readable_code` capability was removed.

## Migration and analysis

Use a **new run ID**. The run identity includes configuration, code, prompts, Schemas, catalog, the task contract version (`scope-operations-v3`), specialist lock, model and processor revisions, seed and the prepared input/rights manifest identity. Records written before catalog 8.0 do not validate under 8.0: every saved instruction carries the removed `profile` field, and versioned ones use 7.0 task IDs. Inspect, replay and re-rate such runs with the code that wrote them, such as commit `6f224e0`.

Synthesis and rerating write an `.operations.json` sidecar counting all committed task IDs and the final committed task per conversation. Stopped tails are excluded. Selection uses the final committed family as its primary label; the semantic fingerprint retains the full task sequence. Export records task IDs and catalog versions only in provenance, keeping training messages public-only. Existing evaluation-source exclusions and rights checks remain in force.

The selection example uses current family names. Its small quotas illustrate constraints; neither that example nor the default uniform family targets is an empirically established sampling policy. Derive production quotas from eligible input and observed coverage; natural photographs cannot guarantee document or diagram coverage.

## Provenance

The catalog began as a design proposal on the Onely7/Pixelogue foundation branch and a survey of the 185 FineVision subsets. The [research map](finevision_mapping_185.md) keeps that survey with its operation phrases and source URLs. A task's `related_finevision_subsets` lists subsets with a similar task; it is a design record, not source supervision, and it never enters model requests.

## Verification boundary

CPU tests cover shared contracts, structured task fixtures, schema and reference rejection, draft parameter contracts, routing and ledger rules, question-gate label rules, holistic follow-ups, bounded output, run identities, information boundaries and the generated task reference. `pixelogue task-status --junit artifacts/cpu-junit.xml` records which shared fixture modules passed for each task; it does not imply per-task natural-image accuracy. Model-bound specialist calibration, judge error rates, cross-language fidelity, model cost and downstream training benefit still require external evaluation. Existing live review probes cover only their original scenarios.
