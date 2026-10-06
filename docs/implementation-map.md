# Implementation map

This page connects the pipeline promises to the code, saved artifacts, commands, and tests. Read the workflow guides first if terms such as a *quality gate* or *visual group* are new to you.

| Promise | Main implementation | Saved or emitted evidence | Entry point |
|---|---|---|---|
| Versioned 72-operation catalog, runtime admission and required verifiers | `task_catalog.py`, `catalog.py`, `task_registry.py`, `task_runtime.py`, `task_verification.py` | catalog identity, `task_admission`, operation checks, operation counts | `compile`, `synthesize` |
| Strict configuration, exact quotas, and pinned model roles | `config.py`, `planner.py` | compiled config, quota table, model revisions | `compile` |
| Rights checks, canonical images, duplicate groups, and split isolation | `images.py`, `operations.py`, `sscd.py` | image ledger, failures, split map, manifest hash | `ingest` |
| Fixed Open Images V7 validation sample | `open_images.py`, `validation/open_images_v7_manifest.jsonl` | source and rights JSONL plus private download metadata | `prepare` |
| Local structured model calls with stage-specific inputs, output-scaled timeouts and a repetition guard | `prompts.py`, `serving.py` | content-addressed requests, responses, usage, and model lock | `synthesize`, `rate-existing` |
| Bounded image concurrency, overlapping judge calls, and model-call timing | `pipeline.py`, `store.py`, `profiling.py` | ordered conversations and stage timing summary | `synthesize`, `profile` |
| One router profile per image and run-wide family routing before drafting | `routing.py`, `pipeline.py`, `store.py` | router response, `turn_route` rows, `image-profile-abstentions` | `synthesize` |
| Direct question drafts and deterministic admission before any judge call | `drafting.py`, `pipeline.py`, `evaluation.py`, `prompts.py` | `question-drafts`, `draft-rejections`, `draft-abstentions`, `public-text-rejections` | `synthesize` |
| Merged two-judge question gate, holistic review, full-view tie-break, and one answer repair | `gates.py`, `pipeline.py`, `focused_views.py`, `evaluation.py` | `question-gate-decisions`, `rating-decisions`, `answer-attempts`, `focus-views` | `synthesize`, `rate-existing` |
| Operation validators with paired blind readers | `task_verification.py`, `rules.py`, and the verifier modules | `operation-checks` | `synthesize`, `rate-existing` |
| Accepted-prefix retention and resumable turns | `pipeline.py`, `store.py` | `conversation-stops`, `conversation-stop-reasons`, `turn_commit`, `conversation_commit` | `synthesize` |
| Crash-safe local state, replay, backup, and restore | `store.py` | SQLite ledger and hashed artifact tree | `replay`, `backup`, `restore` |
| Frozen-pool CP-SAT selection and independent recount | `selection.py`, `operations.py` | pool hash, selection manifest, audit hash | `freeze-pool`, `select`, `audit` |
| Public-only training output with evaluation-source exclusion | `export.py` | separate training, rating, provenance, and selection files | `export` |
| Procedural positive and single-error probes | `fixtures.py`, `capabilities.py` | split and stratum-specific capability report | `make-fixtures`, `evaluate-capabilities` |

`compile` emits JSON Schema for the public configuration, source, generation, evaluation, selection, and rule contracts, including the image profile, the draft batch, and the question-gate vote. Pydantic rejects unknown fields and coercion. The JSON and YAML readers add duplicate-key and non-finite-number rejection because those checks are outside ordinary field validation.

The implementation tests the controller with scripted clients; those clients are test doubles and cannot create exportable standard data. Actual BF16 model startup and a small natural-image pilot remain operator-run checks because they require enough idle GPUs and locally available pinned model weights. A pilot result is diagnostic and is never treated as a 30,000-dialogue certification.
