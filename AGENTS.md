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

Before GPU work, inspect all GPUs with `nvidia-smi`. Use only idle devices and set the selected device explicitly. Run `pixelogue doctor` after inspection. Run long GPU jobs and model servers in `tmux`, keep per-run logs, and monitor more than process existence before leaving them unattended. Do not silently quantize BF16 model configurations. Count model loading and every allocated device toward the four GPU-hour pilot cap.

## Change rules

- Keep public dialogue, model evidence, ratings, and operational metadata in separate models.
- Never pass candidate answers, future turns, dataset annotations, or other judges' decisions to the instruction selector or an evaluator that is not allowed to see them.
- Treat Open Images V7 samples as evaluation-only. Never export them as training records.
- Keep `Qwen/Qwen3.5-2B` as the default selector and instantiate only the selector named by `models.active_selector`; `Qwen/Qwen3.6-35B-A3B` is not a fallback. A conversation uses one configured generator role for its full lifetime. The standard pair is `Qwen/Qwen3.8-27B` and `google/gemma-4-31B-it`. The temporary one-GPU pilot maps both roles to separate blind calls through one `Qwen/Qwen3.5-9B` endpoint; never present that override as the intended model pair or as evaluator-model diversity.
- Keep the training-side `Qwen/Qwen3-VL-8B-Instruct` processor lock independent of every selector.
- Keep SQLite WAL databases on a local filesystem. Back up complete snapshots to shared storage.
- Reject unknown configuration and malformed model output; do not coerce it into a passing result.
- Keep image-level synthesis concurrency bounded by `runtime.max_concurrent_images`. Preserve input order and conversation-local history order, and keep run-store database operations serialized even while model HTTP requests overlap.
- If a large split and a performance change are both needed, commit the behavior-preserving split first as `refactor:`, verify it, and make the measured performance change separately as `perf:`.
- Use Conventional Commits. Do not include unrelated refactoring in a feature or fix commit.

Python docstrings follow PEP 257 and Google style. Put types in annotations; document the meaning, constraints, and relevant exceptions without repeating types.

## Completion

Run the checks appropriate to the changed area. GPU and network tests may be skipped only when the required external inputs are absent; report them as skipped, never passed. Review the final diff for unused code, local paths, secrets, model weights, downloaded images, and generated artifacts.

Before changing image ingestion, read `docs/data.md`. Before changing model payloads or adapters, read `docs/models-and-gpu.md`. Before changing persistence or CI, read `docs/recovery-and-ci.md`. At completion, report the change, commands actually run, and every remaining unverified external check.
