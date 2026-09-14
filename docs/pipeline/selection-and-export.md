# Select and export dialogue

A conversation can pass every turn-level check and still be the wrong record for the final corpus. It may duplicate another image, over-represent one task, or come from an evaluation-only source. Selection handles those collection-level conditions.

[Previous: generate and evaluate](generation-and-evaluation.md) · [Back to contents](README.md) · [Next: inspect real artifacts](artifact-examples.md)

## 1. Re-rate saved public dialogue when needed

`rate-existing` sends saved questions and answers through fresh evaluation without changing their public text:

```sh
uv run --locked pixelogue rate-existing \
  --config configs/pilot.yaml \
  --conversations artifacts/open-images-pilot-001/conversations.jsonl \
  --artifact-root artifacts/prepared-open-images \
  --run-id open-images-rerate-001 \
  --output artifacts/open-images-pilot-001/rated-conversations.jsonl
```

Use a new run ID. Re-rating is useful after an evaluator change, but it is not permission to rewrite an accepted answer in place. The result and source-separated summary are new sidecar files.

## 2. Freeze the eligible pool

```sh
uv run --locked pixelogue freeze-pool \
  --conversations artifacts/conversations.jsonl \
  --output artifacts/frozen-pool.json
```

Only `QUALITY_CANDIDATE` conversations enter the frozen pool. Each rich conversation is reduced to the fields needed by selection: conversation, language, image and visual-group IDs, task family, semantic family, profile pattern, and source purpose. A hash binds the complete ordered pool.

Freezing first matters. If the input pool changes while solving or auditing, the later decision no longer describes the same candidates.

## 3. Select a balanced subset

```sh
uv run --locked pixelogue select \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --output artifacts/selection.json
```

CP-SAT is a constraint solver: it chooses Boolean “included or not included” values while satisfying integer conditions. Pixelogue can require:

- an exact total count;
- exact language counts;
- minimum counts for task families;
- maximum counts per image, visual group, and semantic family;
- exclusion of evaluation sources.

The solver uses one worker and a fixed seed for reproducibility. `OPTIMAL` or `FEASIBLE` includes a selected ID list. `INFEASIBLE` means the frozen pool cannot satisfy the policy. `UNKNOWN` means the solver did not establish an answer within its time limit. Those last two outcomes require different responses: add or rebalance candidates for `INFEASIBLE`; investigate the limit or solver behavior for `UNKNOWN`.

## 4. Audit without solver variables

```sh
uv run --locked pixelogue audit \
  --pool artifacts/frozen-pool.json \
  --policy configs/selection.example.json \
  --selection artifacts/selection.json \
  --output artifacts/audit.json
```

The auditor reads the selected IDs and counts every condition again with ordinary integer logic. It also checks the pool hash, missing or duplicate IDs, and evaluation-source exclusions. A passing audit hash is bound back into `selection.json`.

`export` rejects an unbound selection. This gives the solver and the final output boundary separate implementations of the critical conditions.

## 5. Export the four-file bundle

The pilot profile cannot export training data. Use a reviewed standard configuration and training-eligible sources:

```sh
uv run --locked pixelogue export \
  --config configs/standard.yaml \
  --conversations artifacts/conversations.jsonl \
  --selection artifacts/selection.json \
  --destination artifacts/export
```

| File | Intended contents |
|---|---|
| `training.jsonl` | Public user and assistant messages. The image appears once, in the first user message. |
| `ratings.jsonl` | Per-turn rubric items and aggregate ratings. |
| `provenance.jsonl` | Source, image, visual group, generation model, and training-side processor lock. |
| `selection.json` | Frozen-pool identity, selected IDs, solver result, and audit binding. |

Candidate IDs, selector reasons, evaluator reasons, source titles, and operational fields never enter `training.jsonl`. An evaluation image is rejected again at export even if an earlier step was misconfigured.

## 6. Verify, back up, and restore

After a run finishes, verify its database and every complete artifact:

```sh
uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id open-images-pilot-001
```

Create a consistent backup with SQLite's backup API:

```sh
uv run --locked pixelogue backup \
  --config configs/pilot.yaml \
  --run-id open-images-pilot-001 \
  --destination /shared/pixelogue-backups/open-images-pilot-001
```

Do not copy a live `.sqlite3`, `-wal`, and `-shm` separately. Restore into a new empty local directory and run `replay` before resuming. The [recovery guide](../recovery-and-ci.md) gives the complete commands and failure rules.

The public dialogue, ratings, provenance, and audited selection must all point to the same immutable inputs.
