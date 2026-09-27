# Pixelogue

Pixelogue is a Python 3.12 pipeline for building image-grounded, multi-turn question-answer dialogues. It validates image rights, lets a dedicated vision model choose a natural instruction, generates two to six turns, obtains two blind evaluator verdicts, selects a diverse subset, and exports public training content separately from ratings and provenance.

The repository implements the pipeline and a diagnostic pilot. It does not contain a completed 30,000-dialogue corpus or fine-tuned weights.

## Safeguards built into the workflow

- The active instruction selector is configured explicitly. The other selector is never called as an automatic fallback.
- The default `evaluation.mode: holistic` generates a question and answer, then obtains one whole-turn review from each judge. Only two MET votes accept a turn; disagreement or uncertainty abstains.
- Holistic review checks image facts, request fulfillment, history, target language, explicit formats, and safety together. V7 first checks the bound question with both judges and also requires the operation’s applicable validators. It skips detailed requirement/claim extraction, per-item grading, and automatic answer repair.
- Empty text, exact repeated questions, and substantial private-prompt echoes fail controller checks. A requested regrouping can be useful without adding a new fact.
- With `evaluation.retain_accepted_prefix: true`, a quality stop retains an accepted prefix of at least two turns. Failed tails stay in private `conversation-stops` artifacts, never training output. Operational errors are not converted to accepted conversations.
- `evaluation.mode: detailed` retains the legacy decomposed evaluator for comparisons and requires completion of the planned conversation. Changing evaluation settings requires a new run ID.
- Evaluator roles A and B receive separate blind calls. Neither receives the other verdict or the generator role. The standard profile uses Qwen3.8-27B-FP8 and Gemma 4 31B (W4A16 compressed-tensors) as distinct model lineages.
- Open Images V7 validation images and their visual-copy groups are evaluation-only and cannot be exported for training.
- Model requests, responses, revisions, processor revisions, and token use are content-addressed.
- SQLite WAL state stays on a local filesystem; consistent backups can be copied elsewhere.
- Model dtype is fixed to BF16; only each generator's checked-in quantization method (FP8 for Qwen3.8-27B, compressed-tensors W4A16 for Gemma 4 31B) is accepted, and configuration rejects any other quantization value.

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

The first command validates the complete configuration and emits the versioned 72-task catalog (65 core candidates and 7 gated extensions), 28 legacy detailed rating criteria, the effective evaluation settings, exact language quotas, and JSON Schemas. Model weights are not needed for these steps.

See the [v7 catalog and runtime admission guide](docs/tasks/README.md) for scoped evidence, validator status, and migration. All 65 core operations have CPU verification paths. The 7 specialist extensions require exact model-bound calibration and their runtime dependencies before normal selection.

## Guides

Read the guides in this order:

1. [CPU quickstart](docs/quickstart.md)
2. [Detailed pipeline guide](docs/pipeline/README.md)
3. [Images and Open Images V7](docs/data.md)
4. [Models and GPU checks](docs/models-and-gpu.md)
5. [Generation, selection, and outputs](docs/workflow.md)
6. [Recovery and CI](docs/recovery-and-ci.md)
7. [Measured Qwen3.5-9B pilot result](docs/validation/qwen35-9b-pilot.md)
8. [Qwen3.5-9B throughput validation](docs/validation/qwen35-9b-throughput.md)
9. [Implementation map](docs/implementation-map.md)

Japanese documentation begins at [README_ja.md](README_ja.md).

## Model roles

| Role | Default repository |
|---|---|
| Instruction selector | `Qwen/Qwen3.5-2B` |
| Optional selector, selected only by configuration | `Qwen/Qwen3.6-35B-A3B` |
| Generator and evaluator A | [`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) |
| Generator and evaluator B | [`google/gemma-4-31B-it-qat-w4a16-ct`](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct) |
| Independent training-side image processor | `Qwen/Qwen3-VL-8B-Instruct` |

`configs/pilot.yaml` is a temporary one-GPU validation profile. It maps both logical generator and evaluator roles to one `Qwen/Qwen3.5-9B` endpoint, so it checks pipeline operation without providing evaluator-model diversity. It does not change the intended models in `configs/standard.yaml`.

All repository and processor revisions are pinned in the example configurations. Re-run `doctor` after intentionally changing any lock.

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

Generated images, model weights, active databases, `_references/`, and private `_docs/` records are not committed.
