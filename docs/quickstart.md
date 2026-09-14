# CPU quickstart

This guide confirms the installation without downloading model weights. Run every command from the
repository root.

## 1. Install the locked environment

Pixelogue requires Python 3.12. uv reads `.python-version` and creates `.venv` automatically.

```sh
uv sync --locked --extra cpu --group dev
uv run --locked pixelogue --help
```

`--locked` stops when `pyproject.toml` and `uv.lock` disagree. This protects reproducibility.

## 2. Compile the pilot plan

```sh
uv run --locked pixelogue compile \
  --config configs/pilot.yaml \
  --output artifacts/compiled-plan.json
```

Expected output is one JSON line containing `output` and `compiled_hash`. The output file contains
resolved settings, quotas, catalogs, and schemas. Compilation proves that settings are consistent;
it does not prove that GPUs or model servers are ready.

## 3. Make small answer-keyed fixtures

```sh
uv run --locked pixelogue make-fixtures \
  --destination data/fixtures \
  --pairs-per-stratum 2
```

This creates positive and single-error negative examples for eight capability strata in separate
development and confirmation splits. A value of 2 is diagnostic. It is not the full capability
qualification set.

## 4. Run CPU checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
```

If `uv sync --locked` reports a stale lock, review the dependency change, run `uv lock`
intentionally, and commit both `pyproject.toml` and `uv.lock`.
