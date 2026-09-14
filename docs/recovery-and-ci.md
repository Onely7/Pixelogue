# Recovery and CI

Pixelogue uses one local SQLite WAL database and immutable content-addressed files per run. The
default root is `/var/tmp/pixelogue`. SQLite documents that [WAL does not work over a network
filesystem](https://sqlite.org/wal.html), so a known NFS, CIFS, SMB, or SSHFS root is rejected.

## Verify a saved run

```sh
uv run --locked pixelogue replay \
  --config configs/pilot.yaml \
  --run-id open-images-pilot
```

`replay` runs SQLite integrity checking, hashes every complete artifact again, and verifies that
complete model calls have response artifacts. A mismatch is an audit failure; do not copy a missing
file from another run under the same hash.

Only one coordinator can open a run at a time. A second process receives `RUN_ALREADY_ACTIVE`.
Request and output-token reservations are written before inference. If a process crashes, the
reservation remains consumed, preventing a resume from silently exceeding its budget.

## Make a consistent backup

```sh
uv run --locked pixelogue backup \
  --config configs/pilot.yaml \
  --run-id open-images-pilot \
  --destination /shared/pixelogue-backups/open-images-pilot-001
```

The command uses SQLite's backup API while the local database is open, then copies immutable
artifacts. Do not copy the live `.sqlite3`, `-wal`, and `-shm` files separately.

Restore into a new empty local directory:

```sh
uv run --locked pixelogue restore \
  --database /shared/pixelogue-backups/open-images-pilot-001/open-images-pilot.sqlite3 \
  --artifacts /shared/pixelogue-backups/open-images-pilot-001/open-images-pilot-artifacts \
  --destination /var/tmp/pixelogue/open-images-pilot-restored
```

Run `replay` against the restored run before continuing. A changed configuration hash is rejected;
start a new run ID for an intentional configuration change.

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

`.github/workflows/quality.yml` uses the same locked environment. ty uses its native GitHub output
format, which creates workflow annotations without a custom SARIF converter. gitleaks remains a
pre-commit check.

`.github/workflows/codeql.yml` is a separate CPU-only advanced CodeQL job. In GitHub repository
settings, open **Code security → Code scanning**, disable CodeQL default setup, and keep the workflow
setup. Enabling both creates duplicate analysis configurations and confusing results. The workflow
uses CodeQL Action v4 as recommended by the [current CodeQL Action
documentation](https://github.com/github/codeql-action).
