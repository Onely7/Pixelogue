# Task catalog v7

[日本語](README_ja.md) · [72 operation definitions](TASKS.md) · [185-subset research map](finevision_mapping_185.md)

The installed catalog contains **65 core candidates and 7 validator-gated extensions**, organized into 14 families. The research map preserves 185 subsets and their 527 operation phrases. These are design/provenance records, not source annotations, training records, or evidence that any image supports a task. They never enter model requests.

## Catalog membership and runtime admission

`pixelogue compile` emits the complete catalog, strict JSON Schemas, an advisory 24-ID migration map, and `task_admission`. Every `task_admission` entry reports the required validators and concrete reasons that normal-profile generation is blocked.

Currently **34 normal operations have an available verification path**. The remaining 31 core definitions are retained but blocked when their required validator or operation mode is unavailable. This includes table reconstruction, numeric chart recovery, graph topology, finite-grammar pattern solving, calibrated scales, formal formula structure, general aggregates, quantified propositions, numeric ordering, and unit conversion. Do not replace these checks with generic model agreement. Grounded arithmetic currently supports one exact add/subtract/multiply/divide expression with the existing unit rules; derived percentages, rounding, compound expressions and unsupported unit algebra are not admitted.

All 7 specialized extensions are disabled. Nonempty `tasks.enabled_extensions` fails configuration validation until actual specialized implementations and domain calibration are added. Naming a validator or changing a flag cannot enable an absent implementation. No generated program, renderer, or UI action is executed.

## Per-image flow

1. Extract bounded scope records in the actual delivered view. Each capability has a local region, evidence ID and `MET`, `NOT_MET` or `UNKNOWN` observation. Missing observations are unknown. Capabilities from different scopes cannot be pooled.
2. Filter operation templates by local evidence and implemented validators. Rotate deterministically among eligible families, then diversify operations. The default candidate budget is 8, independent of the total catalog size.
3. Bind concrete public targets, parameters and eligibility checks to those templates. Every normal guard must be `MET`. Parameters sourced from pixels or history require references to the same scope or committed public messages. Output verdict vocabularies and fixed policies are not bindable desired answers.
4. Reject unsupported parameter combinations, duplicate operation fingerprints and output estimates above the configured budget before calling the selector. Fingerprints include catalog, image, scope, profile and normalized public choices, independently of turn index and question wording.
5. The configured selector sees the image, committed public history and sanitized candidate contracts. Private capability verdicts, source names, annotations and other judges' decisions are excluded.
6. Generate one operation-specific question. Both blind judges check the actual question, public parameters and local eligibility before answer generation. Compound independent requests are rejected rather than assigned the first operation's label; explicit subtask composition is not yet a generation interface.
7. Holistic mode retains two blind whole-turn reviews and now includes the compact expected-operation contract. Both modes also require every applicable operation validator. Member/count extraction and primitive arithmetic use independent inventories followed by deterministic checks. Transcription compares exact source/answer text. Evidence, UI and panel checks retain independent answer-to-region bindings.

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

## Migration and analysis

Use a **new run ID**. The run identity includes the canonical catalog content, configuration, and task contract version. Old 24-task artifacts retain their original labels; `task_catalog_legacy.yaml` and `legacy_24_migration.json` provide advisory mappings only. In particular, `chart_lookup` requires reclassification from the actual question, not automatic assignment to two new operations. Old stores can still be inspected with `replay` and backed up.

Synthesis and rerating write an `.operations.json` sidecar counting all committed operation IDs and the final committed operation per conversation. Stopped tails are excluded. V7 selection uses the final committed family as its primary label; the semantic fingerprint retains the full task sequence. Export records operation IDs and catalog versions only in provenance, keeping training messages public-only. Existing evaluation-source exclusions and rights checks remain in force.

The selection example uses current family names. Its small quotas illustrate constraints; neither that example nor the initial family rotation is an empirically established sampling policy. Derive production quotas from eligible input and observed coverage; natural photographs cannot guarantee document or diagram coverage.

## Verification boundary

CPU tests cover schema/reference rejection, scope isolation, guarded profiles, parameter lineage, bounded output, family rotation, required supplementary checks, migration identities and information boundaries. They do not validate visual extraction accuracy, judge false positives, cross-language fidelity, specialist calibration, model cost or downstream training benefit. Run representative images and manual review with the configured servers before producing a corpus. Existing live rubric probes cover only their original scenarios, not all v7 operations.
