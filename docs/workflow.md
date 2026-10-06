# Generation, selection, and outputs

This guide starts after images have been ingested and all required model endpoints have passed `doctor --check-servers`.

For the reasoning behind every gate and annotated artifact examples, use the [pipeline guide](pipeline/README.md).

## 1. Generate a pilot

```sh
uv run --locked pixelogue synthesize \
  --config configs/pilot.yaml \
  --images artifacts/prepared-open-images/images.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-pilot \
  --output artifacts/open-images-pilot/conversations.jsonl
```

`runtime.max_concurrent_images` is the upper bound on independent images in flight. The standard and regular pilot profiles allow four; the one-GPU paired pilot allows two. Use `--workers 1` for a serial diagnostic or `--workers N` up to the configured bound for a measured run. Set the bound explicitly in a separate config when comparing higher concurrency. Pixelogue still processes the turns within one conversation in order and writes final conversation rows in input order.

The coordinator assigns languages and generators with exact batch quotas. A conversation uses one generator throughout. Each image follows this flow:

1. the router (`Qwen/Qwen3.5-2B`) profiles the image once and lists the task families it can support;
2. for each planned turn, the controller routes a primary and a secondary family from the 65 core operations, offering only lightly verified operations in the first `tasks.anchor_turns` turns;
3. the conversation's generator drafts up to `tasks.draft_count` questions, each with its operation, public parameters, regions and a private fact key;
4. deterministic checks reject contract violations, private-prompt echoes, internal references, repeated questions and facts, and operation-specific defects before any judge call;
5. both judges gate the question in one blind call each, including an independent operation label; one extra drafting call may follow when nothing passes;
6. the generator answers only the first question that passes, and both judges review the whole turn; a full-view tie-break and a single answer repair are the only follow-ups;
7. every applicable operation validator runs before the turn is committed.

Holistic review covers language, explicit formats, safety and visual facts together; its rating item ID is `Q_HOLISTIC`. See the [generation guide](pipeline/generation-and-evaluation.md) for each step and the [catalog guide](tasks/README.md) for availability and parameter restrictions.

A turn that ends without a committed answer stops its conversation. With `evaluation.retain_accepted_prefix: true` (default), a conversation that has already committed at least two turns keeps that prefix as a quality candidate; failed tails remain private. Disagreement and insufficient evidence abstain, while operational errors remain errors. Use a new run ID for any configuration change.

Outputs include `conversations.jsonl`, dataset-specific `conversations.summary.json`, and `conversations.operations.json`, which counts committed operations and the final committed primary operation. Evaluation sources remain ineligible for training.

To inspect model-call latency and token use during or after a run, point `profile` at its local database:

```sh
uv run --locked pixelogue profile \
  --database /var/tmp/pixelogue/open-images-pilot/run.sqlite3 \
  --output artifacts/open-images-pilot/inference-profile.json
```

New runs record each HTTP request duration directly. Older databases use the interval between consecutive saved responses as a serial-run estimate, and the report labels that method `completion_gap_estimate`.

The current ledger retains one `model_call` row for each model-lock, stage, and request-hash identity. Identical repeated requests can therefore share one profile row. Use `budget.request_count` or count response artifacts when you need the actual number of HTTP attempts; use `profile` for latency and stage distribution. The [throughput report](validation/qwen35-9b-throughput.md) shows both measurements in one example.

## 2. Re-rate immutable dialogue

```sh
uv run --locked pixelogue rate-existing \
  --config configs/pilot.yaml \
  --conversations artifacts/open-images-pilot/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-rerate \
  --output artifacts/open-images-pilot/rated-conversations.jsonl
```

This sends saved questions and answers through the deterministic public-text checks, the two-judge question gate, holistic review and operation validators again. It does not rewrite or repair public text. The result and its source-separated summary are sidecars.

Exhausted malformed evaluator outputs produce `ABSTAINED`; transport and server failures remain `ERROR`. Re-rating records the stopped turn's unchanged question, answer and private failure reason, then continues with other conversations. The failed turn is never committed or accepted.

## 3. Freeze and select a training pool

Only fully accepted training-purpose conversations become pool candidates.

```sh
uv run --locked pixelogue freeze-pool \
  --conversations artifacts/conversations.jsonl \
  --output artifacts/frozen-pool.json

uv run --locked pixelogue select \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --output artifacts/selection.json

uv run --locked pixelogue audit \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --selection artifacts/selection.json \
  --output artifacts/audit.json
```

CP-SAT enforces the exact target and language quotas, minimum task-family counts, and maximum counts per image, visual group, and semantic family. `INFEASIBLE` means the frozen pool cannot meet the conditions. `UNKNOWN` means the solver did not establish an answer within its limit. The audit recomputes conditions without solver variables and binds its hash to `selection.json` only when all checks pass.

## 4. Export a standard training bundle

Pilot profiles are intentionally rejected here. Use a reviewed standard run whose selected sources have training permission.

```sh
uv run --locked pixelogue export \
  --config configs/standard.yaml \
  --conversations artifacts/conversations.jsonl \
  --selection artifacts/selection.json \
  --destination artifacts/export
```

The output files have different audiences:

| File | Contents |
|---|---|
| `training.jsonl` | Public user and assistant messages; the image appears once in the first user message |
| `ratings.jsonl` | Per-turn rating items and aggregate results |
| `provenance.jsonl` | Source, visual group, generator, operation IDs, and independent student processor lock |
| `selection.json` | Frozen pool identity, selected IDs, solver status, and audit hash |

Candidate IDs, drafts, fact keys, judge reasons, source titles, and operational fields never enter `training.jsonl`. The training-side Qwen3-VL-8B processor lock is independent of every teacher model, including the router, and is recorded in provenance.

### Synthesis progress

`synthesize` uses a tqdm progress bar on stderr by default, at startup, after each image is
saved, and every 10 seconds while waiting. It shows saved/total images, worker count,
QUALITY_CANDIDATE / REJECTED / ABSTAINED / ERROR counts, elapsed time, rate and estimated remaining time. The total
is capped by `data.target_dialogues`; a processed image is not necessarily accepted.
Counts follow input-order output persistence, so later completed workers may not yet
be counted. Waiting updates indicate the controller is waiting, not model-server health.
Interrupted runs are labelled `interrupted`, not `finished`. Use `--quiet` to hide
progress, `2> progress.log` to save it, or `> report.json` for the final stdout JSON.
