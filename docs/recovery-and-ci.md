# Recovery and CI

Pixelogue uses one local SQLite WAL database and immutable content-addressed files per run. The default root is `/var/tmp/pixelogue`. SQLite documents that [WAL does not work over a network filesystem](https://sqlite.org/wal.html), so a known NFS, CIFS, SMB, or SSHFS root is rejected.

## Verify a saved run

```sh
uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id open-images-pilot
```

`replay` runs SQLite integrity checking, hashes every complete artifact again, and verifies that complete model calls have response artifacts. A mismatch is an audit failure; do not copy a missing file from another run under the same hash.

Only one coordinator can open a run at a time. A second process receives `RUN_ALREADY_ACTIVE`. Request and output-token reservations are written before inference. If a process crashes, the reservation remains consumed, preventing a resume from silently exceeding its budget.

## Make a consistent backup

```sh
uv run --locked pixelogue backup \
  --config configs/pilot.yaml \
  --run-id open-images-pilot \
  --destination /shared/pixelogue-backups/open-images-pilot-001
```

The command uses SQLite's backup API while the local database is open, then copies immutable artifacts. Do not copy the live `.sqlite3`, `-wal`, and `-shm` files separately.

Restore into a new empty local directory:

```sh
uv run --locked pixelogue restore \
  --database /shared/pixelogue-backups/open-images-pilot-001/open-images-pilot.sqlite3 \
  --artifacts /shared/pixelogue-backups/open-images-pilot-001/open-images-pilot-artifacts \
  --destination /var/tmp/pixelogue/open-images-pilot-restored
```

Run `replay` against the restored run before continuing. A changed configuration hash is rejected; start a new run ID for an intentional configuration change.

The configuration hash now binds application code, prompts, Schemas, task resources, specialist worker and lock files, model and processor settings, seed and pinned Open Images input. `synthesize` also requires the prepared `manifest.json`; its identity includes the original source and rights manifests. The provided `images.jsonl` must match that prepared manifest exactly. A changed contract or input therefore requires a new run ID. Research experiment plans and completed trials use separate frozen files under `artifacts/` and are not normal conversation commits.

## Local and GitHub checks

The required local sequence is:

```sh
uv sync --locked --extra cpu --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
uv build
uv run --locked pre-commit run --all-files
```

`.github/workflows/quality.yml` uses the same locked environment. ty uses its native GitHub output format, which creates workflow annotations without a custom SARIF converter. gitleaks remains a pre-commit check.

`.github/workflows/codeql.yml` is a separate CPU-only advanced CodeQL job. In GitHub repository settings, open **Code security → Code scanning**, disable CodeQL default setup, and keep the workflow setup. Enabling both creates duplicate analysis configurations and confusing results. The workflow uses CodeQL Action v4 as recommended by the [current CodeQL Action documentation](https://github.com/github/codeql-action).


## Completed holistic conversations

Holistic synthesis records terminal outputs in `conversation_commit`, pointing to immutable
`conversations` artifacts. Restarting the same run returns that recorded output, including a
retained prefix, without trying to extend it again. A crash before the terminal record resumes
from the ordered `turn_commit` prefix. Existing databases acquire the new table on opening;
configuration hashes still prevent mixing detailed and holistic results in one run.

A quality stop after at least two accepted turns may produce `QUALITY_CANDIDATE` for the prefix.
The original stopped conversation is preserved in a private `conversation-stops` artifact.
No rejected turn is exported. Runtime errors remain errors. Set
`evaluation.retain_accepted_prefix: false` to require the planned length. Use a new run ID after
changing settings, including when migrating configurations created before evaluation settings
were introduced. `rate-existing` rejudges stored Q/A without generating missing turns; an empty
or one-turn input cannot become a quality candidate.

Re-rating contains inference failures within the affected conversation and continues with the
remaining inputs. Exhausted malformed-output retries produce `ABSTAINED`; transport and server
failures remain `ERROR`. The unchanged question and answer stay attached to the stopped turn,
and a private failure artifact records its conversation, turn index and reason. A failed turn
cannot be committed or included in an accepted prefix.

## Large synthesis runs

Synthesis flushes each completed conversation to JSONL instead of rewriting the entire file.
The summary is refreshed after the first record, every 100 records, and on exit. Restarting
rebuilds the output from the run's saved conversation/turn commits; use the same immutable input
manifest and configuration.

New request artifacts externalize repeated image data URLs into content-addressed
`request-images` artifacts (`archive_format: image-refs-v1`). The HTTP payload and request hash
remain unchanged. Use `pixelogue.serving.read_request_artifact(store, artifact_hash)` to restore
the exact request envelope for auditing; it also reads legacy inline-image requests. Backups
must include all artifacts, including the referenced images.
