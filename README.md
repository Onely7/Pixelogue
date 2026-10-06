# Pixelogue

Pixelogue is a Python 3.12 pipeline for building image-grounded, multi-turn question-answer dialogues. It validates image rights, profiles each image with a small router model, drafts two to six turns that two blind evaluators check before and after each answer, selects a diverse subset, and exports public training content separately from ratings and provenance.

The repository implements the pipeline and a diagnostic pilot. It does not contain a completed 30,000-dialogue corpus or fine-tuned weights.

## Safeguards built into the workflow

- The router (`models.router`) is fixed to `Qwen/Qwen3.5-2B` and only profiles each image for task-family routing. It writes no dialogue, and no other model replaces it after a failure.
- The conversation's generator drafts candidate questions for the operations routed to each turn. Deterministic checks and a merged question gate by both judges run before any answer exists, and only the first question that passes is answered.
- Holistic review checks image facts, request fulfillment, history, target language, explicit formats, and safety together. Only two MET votes accept a turn; disagreement or uncertainty abstains, apart from one full-view tie-break and one answer repair that both judges review again. Every applicable operation validator must also pass.
- Empty text, exact repeated questions, repeated facts, and substantial private-prompt echoes fail controller checks. A requested regrouping can be useful without adding a new fact.
- With `evaluation.retain_accepted_prefix: true`, a quality stop retains an accepted prefix of at least two turns. Failed tails stay in private `conversation-stops` artifacts, never training output. Operational errors are not converted to accepted conversations. Changing evaluation settings requires a new run ID.
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

The first command validates the complete configuration and emits the versioned 72-task catalog (65 core candidates and 7 gated extensions), per-task runtime admission, the effective settings, exact language quotas, and JSON Schemas. Model weights are not needed for these steps.

See the [v7 catalog and runtime admission guide](docs/tasks/README.md) for routing, draft admission, validator status, and migration. All 65 core operations have CPU verification paths, and synthesis drafts questions only for them. The 7 specialist extensions are evaluated with `evaluate-specialist` and require exact model-bound calibration and their runtime dependencies.

## Guides

See [verification contract revision 2](docs/tasks/verification-contract-v2.md) for
public output formats, blind extraction, query binding and small performance comparisons.

Read the guides in this order:

1. [CPU quickstart](docs/quickstart.md)
2. [Pipeline guide](docs/pipeline/README.md)
3. [Images and Open Images V7](docs/data.md)
4. [Models and GPU checks](docs/models-and-gpu.md)
5. [Generation, selection, and outputs](docs/workflow.md)
6. [Recovery and CI](docs/recovery-and-ci.md)
7. [Measured Qwen3.5-9B pilot result](docs/validation/qwen35-9b-pilot.md)
8. [Qwen3.5-9B throughput validation](docs/validation/qwen35-9b-throughput.md)
9. [Measured quality and synthesis performance](docs/validation/quality-and-performance.md)
10. [Implementation map](docs/implementation-map.md)
11. [Open Images synthesis and examples by task](docs/validation/synthesis-campaign.md)

Japanese documentation begins at [README_ja.md](README_ja.md).

[Yield and speed comparison procedure](docs/validation/yield-and-speed.md) documents the controlled comparison and audit procedure. Reports under `docs/validation/` record measurements of earlier pipeline versions; they do not describe the current direct-drafting path.

## Model roles

| Role | Default repository |
|---|---|
| Image router | `Qwen/Qwen3.5-2B` |
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
