# Generation, selection, and outputs

This guide starts after images have been ingested and all required model endpoints have passed `doctor --check-servers`.

For the reasoning behind every gate and annotated artifact examples, use the [detailed pipeline guide](pipeline/README.md).

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

The coordinator assigns languages and generators with exact batch quotas. A conversation uses one generator throughout. The v7 flow is:

1. extract scope-bound capability observations;
2. build eligible templates from the 65 core candidates and any enabled specialist extensions with a working environment and exact model-bound calibration;
3. bind public parameters and eligibility checks within the answer budget;
4. select using the image and exact committed public history;
5. generate a question, reject prompt echoes and repeats, classify its operation without revealing the selected task, then obtain two independent operation/eligibility checks;
6. generate the answer and obtain two blind holistic reviews including the expected operation;
7. run every applicable declared validator before committing the turn.

Holistic review covers language, explicit formats, safety and visual facts together. Detailed mode additionally extracts public requirements before the answer and applies the legacy claim/rubric process with one possible repair. V7 operation validators apply in both modes. See the [catalog guide](tasks/README.md) for availability and parameter restrictions.

`NO_SUITABLE_CANDIDATE` ends the plan without selector fallback. Holistic runs may retain at least two accepted turns when configured; failed tails remain private. Disagreement and insufficient evidence abstain, while operational errors remain errors. Use a new run ID for v7 or any configuration change.

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

This sends saved questions and answers through fresh dual evaluation. It does not rewrite public text. The result and its source-separated summary are sidecars.

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
| `ratings.jsonl` | Per-turn criteria and aggregate results |
| `provenance.jsonl` | Source, visual group, generator, and independent student processor lock |
| `selection.json` | Frozen pool identity, selected IDs, solver status, and audit hash |

Candidate IDs, selector reasons, judge reasons, source titles, and operational fields never enter `training.jsonl`. The training-side Qwen3-VL-8B processor lock is independent of the instruction selector and is recorded in provenance.

### Synthesis progress

`synthesize` uses a tqdm progress bar on stderr by default, at startup, after each image is
saved, and every 10 seconds while waiting. It shows saved/total images, worker count,
QUALITY_CANDIDATE / REJECTED / ABSTAINED / ERROR counts, elapsed time, rate and estimated remaining time. The total
is capped by `data.target_dialogues`; a processed image is not necessarily accepted.
Counts follow input-order output persistence, so later completed workers may not yet
be counted. Waiting updates indicate the controller is waiting, not model-server health.
Interrupted runs are labelled `interrupted`, not `finished`. Use `--quiet` to hide
progress, `2> progress.log` to save it, or `> report.json` for the final stdout JSON.
