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

`runtime.max_concurrent_images` controls how many independent images can be in flight. The checked-in profiles use four so vLLM can apply continuous batching. Use `--workers 1` for a serial diagnostic or `--workers N` for a measured override. Pixelogue still processes the turns within one conversation in order and writes final conversation rows in input order.

The coordinator assigns languages and the two generators with exact batch quotas. One generator is fixed for a complete conversation, including its one permitted answer repair. The sequence for each turn is:

1. observe bounded image capabilities;
2. build compatible candidates from the 24-task catalog;
3. ask the configured selector to choose a candidate using only the image and committed history;
4. generate a concrete question;
5. require question-fit agreement from evaluator roles A and B;
6. have both judges independently inventory active public requirements before seeing an answer;
7. generate an answer only after question fit and requirement inventory agree;
8. extract claims and apply every relevant rubric item with both judges;
9. exactly recompute agreed typed arithmetic and compare exhaustive answer sets without floats;
10. if a clear failure is repairable, ask the same generator once and repeat all evaluation;
11. commit the turn only after every required gate passes.

`NO_SUITABLE_CANDIDATE` ends that plan. It does not call the alternative selector. Disagreement, insufficient evidence, and transport or schema errors remain distinct from a clear failure. A requirement must point to an exact span in a public user message. Inventory disagreement ends the plan before answer generation, so an evaluator cannot weaken a requirement after seeing an answer.

The command writes `conversations.jsonl` and `conversations.summary.json`. The summary groups counts by dataset, so Open Images is visible separately instead of being hidden in one mixed average. Evaluation conversations exercise the complete path but remain ineligible for training.

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
