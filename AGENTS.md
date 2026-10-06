# Pixelogue work guide

## Project layout

- `src/pixelogue/`: application code. Domain decisions must not depend on Typer or HTTPX.
- `tests/`: public-contract and failure-recovery tests.
- `docs/`: user-facing guides. Keep English and Japanese entry points consistent.
- `configs/`: validated standard, pilot, and selection inputs. Keep pilot and standard semantics distinct.
- `runtime/vllm/`: separate uv lock and pinned local-server configurations. Do not add vLLM to the application environment.
- `validation/`: committed identities and provenance needed to reproduce evaluation inputs. Do not commit downloaded image bytes.
- `_docs/`, `_references/`, `data/`, `artifacts/`, and `models/`: local-only material; do not commit.

## Environment and commands

Run commands from the repository root with Python 3.12.

- Install: `uv sync --locked --extra cpu --group dev`
- Lint: `uv run --locked ruff check .`
- Format check: `uv run --locked ruff format --check .`
- Type check: `uv run --locked ty check`
- Tests: `uv run --locked pytest`
- Build: `uv build`

Before GPU work, inspect all GPUs with `nvidia-smi`. Use only idle devices and set the selected device explicitly. Run `pixelogue doctor` after inspection. Run long GPU jobs and model servers in `tmux`, keep per-run logs, and monitor more than process existence before leaving them unattended. Do not silently change a model's checked-in quantization method (or add quantization to an endpoint that is not on the approved list); dtype stays fixed to bfloat16. Count model loading and every allocated device toward the four GPU-hour pilot cap.

## Change rules

- Keep public dialogue, model evidence, ratings, and operational metadata in separate models.
- Never pass candidate answers, future turns, dataset annotations, or other judges' decisions to the image router, the question drafter, or an evaluator that is not allowed to see them. A holistic tie-break call receives neither the other judge's verdict nor its own prior reason, and an answer repair receives only the objecting judge's reason.
- Treat the pinned Open Images V7 validation samples and their visual groups as evaluation-only. Never export them as training records. Explicit local train imports use the separate `prepare-local-train` path with train metadata and rights checks; do not reclassify validation samples.
- Keep `Qwen/Qwen3.5-2B` as the image router (`models.router`). It returns only an image profile and per-family feasibility; the controller chooses the families and the conversation's generator drafts the questions. A conversation uses one configured generator role for its full lifetime. The standard pair is `Qwen/Qwen3.8-27B-FP8` (FP8 quantization) and `google/gemma-4-31B-it-qat-w4a16-ct` (compressed-tensors W4A16 quantization); each repository's `quantization` value is pinned in `ModelConfig.validate_roles` and cannot be set to anything else. The temporary one-GPU pilot maps both roles to separate blind calls through one unquantized `Qwen/Qwen3.5-9B` endpoint; never present that override as the intended model pair or as evaluator-model diversity.
- Keep the training-side `Qwen/Qwen3-VL-8B-Instruct` processor lock independent of the router and generator models.
- Keep SQLite WAL databases on a local filesystem. Back up complete snapshots to shared storage.
- Reject unknown configuration and malformed model output; do not coerce it into a passing result.
- Reject normalized exact question repeats, substantial private-prompt echoes, and drafts whose private request key repeats a committed turn before spending image-aware evaluator calls. Preserve the rejected text and reason as a private run artifact.
- In research history studies, apply history binding and witness checks only from resolved binding IDs and an explicit witness designation. A later turn index alone is not evidence of either condition.
- Keep image-level synthesis concurrency bounded by `runtime.max_concurrent_images`. Judge pairs and paired blind source readers may run concurrently within one conversation. Preserve input order and conversation-local history order, and keep run-store database operations serialized even while model HTTP requests overlap.
- If a large split and a performance change are both needed, commit the behavior-preserving split first as `refactor:`, verify it, and make the measured performance change separately as `perf:`.
- Use Conventional Commits. Do not include unrelated refactoring in a feature or fix commit.

Python docstrings follow PEP 257 and Google style. Put types in annotations; document the meaning, constraints, and relevant exceptions without repeating types.

## Completion

Run the checks appropriate to the changed area. GPU and network tests may be skipped only when the required external inputs are absent; report them as skipped, never passed. Review the final diff for unused code, local paths, secrets, model weights, downloaded images, and generated artifacts.

Before changing image ingestion, read `docs/data.md`. Before changing model payloads or adapters, read `docs/models-and-gpu.md`. Before changing persistence or CI, read `docs/recovery-and-ci.md`. At completion, report the change, commands actually run, and every remaining unverified external check.
