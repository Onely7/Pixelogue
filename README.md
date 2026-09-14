# Pixelogue

Pixelogue is a Python 3.12 pipeline for building image-grounded, multi-turn question-answer
dialogues. It validates image rights, lets a dedicated vision model choose a natural instruction,
generates two to six turns, obtains blind verdicts from two large models, selects a diverse subset,
and exports public training content separately from ratings and provenance.

The repository implements the pipeline and a diagnostic pilot. It does not contain a completed
30,000-dialogue corpus or fine-tuned weights.

## Safeguards built into the workflow

- The active instruction selector is configured explicitly. The other selector is never called as
  an automatic fallback.
- A question must pass both large-model image-fit checks before an answer is generated.
- Both judges must agree on answer-independent public requirements before generation; every active
  requirement is then rated separately.
- Qwen3.8 and Gemma judge independently. Neither receives the other verdict or the generator name.
- Open Images V7 validation images and their visual-copy groups are evaluation-only and cannot be
  exported for training.
- Model requests, responses, revisions, processor revisions, and token use are content-addressed.
- SQLite WAL state stays on a local filesystem; consistent backups can be copied elsewhere.
- BF16 is fixed in configuration and quantization is rejected.

## Start on CPU

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv sync --locked --extra cpu --group dev
uv run --locked pixelogue compile \
  --config configs/pilot.yaml \
  --output artifacts/compiled-plan.json
uv run --locked pixelogue make-fixtures \
  --destination data/fixtures \
  --pairs-per-stratum 2
uv run --locked pytest
```

The first command validates the complete configuration and emits the 24-task catalog, 28 rating
criteria, exact language quotas, and JSON Schemas. Model weights are not needed for these steps.

## Guides

Read the guides in this order:

1. [CPU quickstart](docs/quickstart.md)
2. [Images and Open Images V7](docs/data.md)
3. [Models and GPU checks](docs/models-and-gpu.md)
4. [Generation, selection, and outputs](docs/workflow.md)
5. [Recovery and CI](docs/recovery-and-ci.md)
6. [Implementation map](docs/implementation-map.md)

Japanese documentation begins at [README_ja.md](README_ja.md).

## Model roles

| Role | Default repository |
|---|---|
| Instruction selector | `Qwen/Qwen3.5-2B` |
| Optional selector, selected only by configuration | `Qwen/Qwen3.6-35B-A3B` |
| Generator and judge A | `Qwen/Qwen3.8-27B` |
| Generator and judge B | `google/gemma-4-31B-it` |
| Independent training-side image processor | `Qwen/Qwen3-VL-8B-Instruct` |

All repository and processor revisions are pinned in the example configurations. Re-run `doctor`
after intentionally changing any lock.

## Development checks

```sh
uv sync --locked --extra cpu --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check
uv run --locked pytest
uv build
uv run --locked pre-commit run --all-files
```

Generated images, model weights, active databases, `_references/`, and private `_docs/` records are
not committed.
