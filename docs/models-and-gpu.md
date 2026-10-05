# Models and GPU checks

Pixelogue separates instruction choice from dialogue generation. The standard profile uses two different generator and evaluator model lineages. A temporary one-GPU pilot substitutes a smaller model to validate the complete execution path.

| Role | Model | Default endpoint |
|---|---|---|
| Instruction selector | `Qwen/Qwen3.5-2B` | `http://127.0.0.1:8000/v1` |
| Optional selector | `Qwen/Qwen3.6-35B-A3B` | `http://127.0.0.1:8001/v1` |
| Generator and evaluator A | `Qwen/Qwen3.8-27B-FP8` | `http://127.0.0.1:8002/v1` |
| Generator and evaluator B | `google/gemma-4-31B-it-qat-w4a16-ct` | `http://127.0.0.1:8003/v1` |

In `configs/standard.yaml`, Qwen3.8-27B-FP8 and Gemma 4 31B generate an equal share of conversations. The assigned model remains fixed for the entire conversation, including its one allowed repair in detailed mode. Both models also judge every turn through separate blind calls. By default (`evaluation.mode: holistic`), each model reviews the whole question/answer against the image and public history once; both must return MET. V7 first verifies the bound question with both judges and runs the operation’s required supplements (including set extraction when applicable). Detailed requirement/claim extraction is omitted. `evaluation.mode: detailed` retains the legacy rubric for controlled comparisons.

`configs/pilot.yaml` is a temporary validation override. It maps both logical roles to a shared `Qwen/Qwen3.5-9B` server on port 8002. The calls remain separate and blind, but the shared checkpoint means this profile tests pipeline wiring rather than evaluator-model diversity.

`configs/paired-one-gpu-pilot.yaml` keeps the exact standard Qwen3.8/Gemma 4 pair and Qwen3.5-2B selector, but schedules all three servers on a single idle 96 GiB-class GPU. Its paired `runtime/vllm/*-onegpu.yaml` files use ports 18102, 18103, and 18100 with memory fractions 0.44, 0.41, and 0.10. Start them sequentially on the same explicitly selected device, then run `doctor --config configs/paired-one-gpu-pilot.yaml --check-servers`. This is a pilot scheduling profile with the standard model identities and quantization, not a change to the standard profile. The three-server startup, image requests, synthesis, and replay were exercised on one RTX PRO 6000 Blackwell GPU; recheck capacity on other hardware.
`configs/paired-one-gpu-diverse.yaml` uses the same endpoints and model locks with a 60-image evaluation target for the pinned Commons category sweep. It keeps evaluation-only inputs separate from training exports.

## 1. Inspect before using a GPU

Run these commands immediately before launch:

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

A GPU is considered idle by `doctor` only when utilization is 0% and used memory is below 1 GiB. Inspect the process table in `nvidia-smi` as well. Never take a device used by another process.

Static `ready: true` means the pinned BF16 model servers can be assigned to the currently idle devices. Generator roles that share the same repository, revision, and endpoint are counted as one server. Static readiness is a capacity check, not a successful inference.
For the pinned FP8 and W4A16 standard checkpoints, `doctor` uses conservative rounded resident-weight estimates and an 8 GiB cache allowance per shard. Recheck these estimates after a checkpoint revision or hardware change.

After launch, `doctor --check-servers` checks the configured served model names. The chosen GPU is no longer idle at that point; a healthy already-running server can satisfy its role without being allocated again.

## 2. Install the separate serving environment

The application and vLLM use different lock files so GPU packages cannot silently change the CPU development environment.

```sh
uv sync --project runtime/vllm --locked
```

The runtime is pinned to vLLM 0.29.0. Server files pin model revisions, dtype (BF16), context length, tensor parallelism, memory utilization, and generation defaults. Both standard generators cap concurrent sequences at 64; vLLM's higher default exceeded the Qwen model's available Mamba cache blocks in a one-GPU startup probe. Quantization is pinned per generator: `generator-a.yaml` sets `quantization: fp8` for Qwen3.8-27B-FP8, and `generator-b.yaml` sets `quantization: compressed-tensors` for the W4A16 Gemma checkpoint. `configs/standard.yaml` mirrors the same pinned values, and `ModelConfig.validate_roles` rejects any other quantization value for these repositories.

## 3. Run long GPU work in tmux

The standard servers use these files:

```text
runtime/vllm/generator-a.yaml        -> Qwen3.8-27B-FP8, port 8002, tensor parallel size 2
runtime/vllm/generator-b.yaml        -> Gemma 4 31B, port 8003, tensor parallel size 2
runtime/vllm/selector-default.yaml   -> Qwen3.5-2B, port 8000, tensor parallel size 1
```

Run `doctor --config configs/standard.yaml` immediately before launch and assign only the idle GPUs it reports. Start each server in its own `tmux` window. For example, after replacing the device IDs with GPUs confirmed to be idle:

```sh
CUDA_VISIBLE_DEVICES=0,1 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a.yaml

CUDA_VISIBLE_DEVICES=2,3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-b.yaml

CUDA_VISIBLE_DEVICES=4 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/selector-default.yaml
```

These device numbers are examples. Required memory depends on the hardware and serving runtime, so the static `doctor` result and an actual startup check are both required.

### Temporary one-GPU pilot

Create separate windows for the generator, selector, pilot, and monitoring commands:

```sh
tmux new-session -d -s pixelogue-qwen35-pilot -n generator
tmux new-window -t pixelogue-qwen35-pilot -n control
tmux new-window -t pixelogue-qwen35-pilot -n selector
tmux new-window -t pixelogue-qwen35-pilot -n pilot
```

Use an explicitly selected idle device. The number below is only an example. This command launches the temporary Qwen3.5-9B substitute, not either standard generator.

Start the larger server first:

```sh
CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-qwen35-9b.yaml
```

Wait for the generator model list, then start the selector on the same device:

```sh
curl -fsS http://127.0.0.1:8002/v1/models

CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/selector-default.yaml
```

The checked-in memory-utilization limits are 0.68 for the 9B server and 0.20 for the 2B selector. They were validated for an RTX 6000 Ada with 48 GiB-class memory. Re-run capacity checks on a different GPU instead of copying those values blindly.

Redirect each window to a unique local log. A pane with `pane_dead=0` is only one health signal; also inspect `/v1/models`, logs, GPU activity, output rows, and model-call status.

### Bounded idle-GPU watcher

`gpu_watch.py` checks every local physical GPU twice with `nvidia-smi`, including the compute-process table. It starts only after `doctor` reports the one-GPU pilot ready. The memory reservation adapts the supplied CUDA allocation script, but omits continuous matrix multiplication. During model startup the holder keeps 25% of GPU memory allocated; the generator starts while that process remains present. After the generator answers `/v1/models`, the watcher releases the holder and acknowledges the handoff before the selector starts. This avoids an unreserved gap. After the pilot, the watcher reacquires an idle GPU with a 90% holder until stopped or the cumulative budget is reached. It never takes a GPU with another compute process, even if Slurm reports its node as idle.

The watcher uses separate ports 18002 and 18000 because the usual selector port may be occupied by another local service. It uses the same pinned, unquantized Qwen3.5-9B and Qwen3.5-2B model revisions. The job runs one image probe, an intentional interruption and resume on two images the user confirmed can support different visual questions, the full 20-image Open Images evaluation, and a 60-category web-image evaluation. It verifies completed-run reuse and replay, served names, `doctor --check-servers`, output row counts, and terminal errors. The two confirmed images do not have independent answer labels. Server and run logs are saved separately. This pilot does not certify the standard model pair or specialist calibration.

```sh
mkdir -p artifacts/gpu-watch
tmux new-session -d -s pixelogue-gpu-watch -c "$PWD" -n monitor \
  'uv run --locked python src/pixelogue/gpu_watch.py watch >> artifacts/gpu-watch/monitor.log 2>&1'
tail -f artifacts/gpu-watch/monitor.log
```

`artifacts/gpu-watch/state.json` records the last scan, active PID, GPU, phase, accumulated GPU seconds, and each completed allocation interval. A run's `phase-events.jsonl` records model loading, each inference batch, and shutdown boundaries. Intervals count an allocated GPU once even while the holder and model overlap. The log also emits a periodic heartbeat. To stop and release a reservation, send `Ctrl-C` to the tmux pane. The watcher stops automatically at its cumulative limit; restarting it preserves the ledger. A live model run appears under `artifacts/gpu-watch/<run-id>/`. GPU 0–5 on this host are outside the visible Slurm GPU partition, so this watcher checks the actual local devices directly.

On hosts with a shared home directory, use `--state-file PATH` with a separate parent directory for each host and boot. The ledger lock and `doctor.json` follow that directory; pass the same option to `wait-release`. This prevents another host's reservation or a PID from an earlier boot from being reused. Keep previous boot directories for accounting. The watcher excludes devices with unavailable numeric measurements while continuing to inspect healthy devices. `doctor` reports unavailable utilization as `null` and never marks that device idle.

The watcher resolves the selected physical device to its `nvidia-smi` UUID and uses that UUID
in `CUDA_VISIBLE_DEVICES` for the holder and pilot. CUDA enumeration can differ from physical
indices when a device is unavailable. Verify the holder PID and memory on the selected UUID;
a live process alone does not establish a reservation. Use inspected UUIDs for manual launches
on such hosts too.

For two reservations on one host, run two reserve-only supervisors with separate ledger directories and the same local `--admission-lock PATH`. The shared lock covers a fresh idle-device check and the holder's startup until memory allocation is confirmed. Each supervisor holds one distinct GPU and accounts for its own allocation; sum their GPU time. Pass the same admission lock to every supervisor on that host.

`tmux` survives an SSH disconnect but does not restart after its own host reboots. For unattended reservations, arrange a host-local boot trigger and periodic restart check, and verify user services survive logout. A restart check must inspect the reservation's GPU process and memory as well as its PID. Disable the restart check before an intentional GPU handoff, then re-enable it after the job, regardless of success or failure.

After every pilot exit, including a failed exit, the watcher records a pending post-job reservation immediately and attempts it on the first idle-device scan. It prefers the GPU used by the pilot and retries pending requests every five seconds when no device is idle. The reservation status and pilot exit code are kept in `state.json`. The cumulative GPU-hour limit still applies; a request that cannot start within the remaining budget is recorded as `budget_exhausted`.

Before an independent tmux job takes over a held device, stop the reservation and run `uv run --locked python src/pixelogue/gpu_watch.py wait-release --gpu-index 4 --timeout-seconds 180`. This waits for two idle device readings, a cleared active reservation, and release of the ledger lock. GPU memory alone can fall before the watcher finishes cleanup; do not start the next model job until this command succeeds.

The current pilot also processes the [60-category evaluation set](data.md#diverse-web-evaluation-sample) after the fixed Open Images comparison. Its separate output, diagnostics, replay, and per-category JSON/CSV/Markdown report appear in the run's `diverse/` directory and `diverse-*` files. This pilot uses the temporary one-GPU model override and has no independent gold labels; its acceptance count does not establish verifier accuracy.

For a completed frozen ABBA comparison, run `python3 validation/compare_paired_runs.py artifacts/plan-followup/EXPERIMENT_DIR`. The script checks the planned image order, identical generator and language allocation, and completed phases, then writes JSON, CSV, and Markdown comparisons with attempted and committed turns, completed conversations, repeated-run overlap, model calls, retries, median and maximum image duration, and measured phase GPU hours. It also reconciles committed task counts against the saved public turns, counts images with two or more committed turns, and reports the stop stage for images with no committed turn. Human-approved conversations per GPU hour and human error rates remain null until independent ballots are resolved. The comparison counts automatic quality candidates as diagnostic outcomes, not gold labels.

A resumed comparison must record each GPU allocation interval and retain interrupted phase attempts. The report includes model loading, interrupted work and teardown, and excludes gaps between model sessions from experiment GPU hours. Reservation time stays in the cumulative ledger. Wall time and the number of model sessions are reported separately; a comparison spanning multiple sessions does not establish performance within a single model startup session.

For an explicitly authorized later validation round, add `--campaign-id recheck-YYYYMMDD --additional-gpu-hours 2` to the `watch` command. The extension is added to the preserved cumulative ledger once per campaign ID. Reusing that ID on restart neither adds budget again nor reruns a completed pilot. The watcher runs one new pilot for a new campaign, then reacquires an idle GPU after the job until its extended budget expires.
Each named extension can now allocate up to 24 GPU-hours for sustained, user-authorized validation. Record all model loading, inference and reservation intervals in the same ledger. A larger allowance does not change model identities, quantization or the idle-device check.

Add `--reserve-only` while diagnosing a failed pilot. This holds an idle GPU without launching the pilot; stop the watcher and restart with the same campaign ID without that flag to run the corrected job. Both phases consume the same cumulative GPU ledger.

After a failed pilot with budget remaining, `--retry-failed-pilot --campaign-id ID` resets only the one-time attempt flag for that campaign. It does not add GPU hours. Keep the original failure logs and use a new run ID for the corrected code.

After a completed diagnostic pilot, `--rerun-completed-pilot --campaign-id ID` starts another run with the current code if that campaign still has at least ten minutes of unspent GPU time. It does not increase the budget. Stop an existing post-job holder cleanly before using this option; the next run receives a new run ID and again restores the reservation after its job.

## 4. Wait for real readiness

vLLM continues with compilation, CUDA graph capture, and multimodal warmup after loading weights. Allocated GPU memory or a live PID does not mean the server is ready.

```sh
curl -fsS http://127.0.0.1:8002/v1/models
curl -fsS http://127.0.0.1:8003/v1/models
curl -fsS http://127.0.0.1:8000/v1/models
uv run --locked pixelogue doctor \
  --config configs/standard.yaml \
  --check-servers
```

Use `configs/pilot.yaml` and omit the port 8003 check when validating the temporary one-GPU profile.

Before a 50-image run, send one image through the same structured-output stages. A server can pass the model-list check while its guided-decoding backend rejects a particular JSON Schema.

If a completion stops at its token limit and lacks a required field, the run records an incomplete completion rather than a generic schema mismatch. Evidence extraction retries with a larger output limit, up to 8192 tokens. The retry must still satisfy the full schema and scope checks; missing observations are never filled in as MET.

Missing JSON braces or brackets are never appended, even when other fields appear complete. The original response must already contain one complete JSON object.

If a length-limited JSON completion ends with at least 512 whitespace characters and remains invalid, the run records `MODEL_WHITESPACE_RUNAWAY`. A bounded retry receives explicit finish guidance but keeps the original output limit. A complete, schema-valid JSON object with trailing whitespace is still accepted. The invalid response remains a private artifact; no missing field or visual fact is inferred.

Qwen receives `enable_thinking=false` in both server defaults and each request. Thinking text and internal control fields must not enter public dialogue.

## 5. Treat the alternative selector separately

`Qwen/Qwen3.6-35B-A3B` is used only when a copied configuration explicitly sets:

```yaml
models:
  active_selector: alternative
```

It is a 35B-total-parameter model for memory planning. Test it in a separate run on port 8001 with enough idle capacity. Pixelogue does not start it alongside the default selector and does not switch to it after a failure.

## 6. Account for the complete GPU allocation

GPU-hours equal elapsed hours multiplied by allocated GPU count. One GPU used for 45 minutes consumes 0.75 GPU-hours; two GPUs for the same time consume 1.5. Include model loading, smoke tests and reservation holders. The initial pilot allocation was four GPU-hours; named extensions record subsequent user-authorized validation in the same ledger. Record start and stop times in a local `_docs/` note.

Pixelogue records requests and token use, but it cannot observe time spent starting an external vLLM process. The operator adds that time. An unavailable server is an unrun check, not a pass.

## 7. Run answer-keyed capability checks

After all required servers are healthy:

```sh
uv run --locked pixelogue evaluate-capabilities \
  --config configs/pilot.yaml \
  --fixtures data/fixtures/fixtures.jsonl \
  --fixture-root data/fixtures \
  --output artifacts/capability-report.json
```

Expected labels remain in the controller and are never sent to evaluator calls. Development and confirmation counts remain separate. A small fixture result checks wiring; it is not a natural-image accuracy claim.

The [detailed pipeline guide](pipeline/README.md) explains how these servers participate in every generation and evaluation stage.

Table, chart and graph readers receive the exact output Schema in model-visible text as well as
the constrained decoder. Answer parsers emit every nullable result field explicitly; `MET`
describes a literal parse, and correctness is computed separately against blind image readings.
Source readers receive no proposed answer, and answer parsers receive no image. Malformed output
remains rejected after bounded retries.


## Evaluation regression checks

After inspecting GPUs and confirming `doctor --check-servers`, the opt-in checks use already-running
standard model servers. They do not launch or allocate another model server. Run in tmux with a log:

```sh
PIXELOGUE_LIVE_RUBRIC=1 uv run --locked pytest -q -s tests/test_rubric_semantics.py
```

For the checked one-GPU standard-pair configuration on ports 18102 and 18103, set `PIXELOGUE_LIVE_CONFIG=configs/paired-one-gpu-pilot.yaml` as well. This changes only the test endpoint addresses and keeps the standard model identities.

Ordinary test runs skip these external checks. The holistic checks include both supported answers
and deliberately false image claims. `rate-existing` can compare holistic review on immutable
saved questions and answers; it does not regenerate previously missing turns. Compare against
human-reviewed defects, not acceptance rate alone, before adopting a large production corpus.


## Bind controlled server settings to a run

Experimental endpoints may set `serving_runtime` with `vllm_version`,
`structured_output_backend`, `disable_any_whitespace` and `server_manifest_sha256`.
The manifest records the actual launch configuration, pinned runtime lock, command and version
check. These fields change both run and response-cache identities. A declaration alone does
not prove the remote server used it. Verify the controlled launch and preserve its logs.
For the pinned vLLM, whitespace suppression requires explicit `xgrammar` or `guidance`;
compare changes in backend and whitespace as a combined condition.

## Compare evidence wire formats

Set `tasks.evidence_format: array` in an experimental configuration to return observations
with a bounded capability enum. The default remains `keyed` until controlled comparisons
justify a change. Both formats use the same descriptions, regions, capability limits and
internal validation. Missing observations remain UNKNOWN. Duplicate capability or evidence
IDs, unrelated views and observations outside the parent region remain rejected.

## Stop runaway JSON generation

The experimental `runtime/vllm/*-whitespace.yaml` generator configurations select
`xgrammar` and `disable_any_whitespace: true`. This constrains whitespace between JSON tokens;
ordinary spaces inside strings remain available. In the pinned vLLM 0.29.0 backend,
this setting must be applied when launching the server. Sending a similarly named
request field alone does not establish that the setting was applied.
Restart existing servers and record their launch manifest before comparing runs.
The standard server configurations retain their existing decoding behavior. A faster
experimental condition must also preserve supported candidate yield and task coverage
before adoption. Use `configs/repetition-detection-pilot.yaml` to evaluate the
separate repetition guard with the pinned standard pair.

For evaluation, an optional engine repetition guard can be configured separately:

```yaml
runtime:
  repetition_detection:
    min_pattern_size: 1
    max_pattern_size: 4
    min_count: 64
    recovery: retry
    stages: [evidence_extraction, candidate_binding]
```

The guard is disabled by default. It stops consecutive token-pattern repetition
without changing sampling probabilities. These thresholds are experimental; a long
legitimate repeated sequence may also stop. Only evidence extraction and candidate
binding may use this option. Public question and answer generation retain their
existing behavior.

`finish_reason: repetition` is recorded as `MODEL_OUTPUT_REPETITION`, preserves the
raw response and incurred usage, and rejects even parseable JSON from the stopped
completion. In the configured extraction stages, `recovery: retry` permits a new
complete response within `structured_output_max_attempts`, with correction feedback
and the original output limit. It never accepts partial output or increases the
budget because of repetition. `recovery: abstain` keeps immediate abstention for
comparisons. A configuration change produces a different run identity. This option
requires an endpoint implementing the pinned vLLM repetition-detection API.

The [vLLM 0.29.0 serving reference](https://docs.vllm.ai/en/v0.29.0/cli/serve/),
[SamplingParams](https://docs.vllm.ai/en/v0.29.0/api/vllm/sampling_params/) and
[structured output guide](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/)
describe the available interfaces. Inspect the installed runtime with
`uv run --project runtime/vllm --locked vllm serve --help=all`; a listed option does
not establish support for every model or backend.

### Bounded whitespace without forcing compact JSON

An opt-in `runtime.json_whitespace_max_chars: 32` adds a private Schema extension
only to evidence extraction and candidate binding. It limits whitespace between
JSON elements while preserving normal newlines and indentation. Spaces inside
text values remain content. No observation, eligibility verdict or public
parameter is synthesized by the controller to repair a stopped response.

This requires the checked-in bridge in the separate vLLM environment:

```bash
runtime/vllm/.venv/bin/python runtime/vllm/whitespace_patch.py
runtime/vllm/.venv/bin/python runtime/vllm/whitespace_patch.py --check
```

Stop model servers before installing or restoring the patch, then restart them.
After `uv sync` in that environment, verify or reapply it. The installer checks
vLLM 0.29.0, XGrammar 0.2.6 and the complete upstream file digest, keeps a verified
backup, and refuses unrelated runtime edits. `--restore` restores those exact
upstream bytes. Record the returned `patched_sha256` in each generator endpoint's
`serving_runtime.xgrammar_whitespace_patch_sha256`, along with the actual server
manifest. Use explicit `structured_output_backend: xgrammar` and
`disable_any_whitespace: false`. The client refuses absent or incompatible
runtime identities. The bound appears in the Schema and therefore the request
and grammar cache identities; use new run IDs after changing it. Baseline requests
without the extension retain upstream compilation behavior. Questions, answers
and specialist source schemas do not receive this extension.

The bridge forwards XGrammar's existing `max_whitespace_cnt`; it does not change
model weights or quantization. See the [pinned XGrammar implementation](https://github.com/mlc-ai/xgrammar/blob/v0.2.6/python/xgrammar/compiler.py)
and [pinned vLLM backend](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/structured_output/backend_xgrammar.py).
The bound changes allowed generation paths, so validate support and end-to-end
quality before adopting it for a new model or image domain.

## Bounded table and verbatim-text verification

Single-cell lookups in simple labeled tables use two answer-blind readings of the complete row/column header domains and only the requested cell. Missing addressing domains, ambiguous selected headers, misaligned or off-scope cells, and disagreements abstain. Selected row and column header anchors must align with the requested cell; unrelated grid coordinates and values are not generated. Merged or hierarchical headers, relevant footnotes, and multi-cell requests retain full-grid verification. Predicate selection, joins and reconstruction always retain complete grids.

Verbatim text, reading-order and code transcription use two answer-blind source readings as literal line arrays. The controller compares the complete answer, permitting only outer quotations or code fences; it never accepts a matching substring that omits or adds text. The blind readers receive the original image so they can detect text continuing outside the bound rectangle. The controller requires the whole requested unit to fit its bound image region. Non-verbatim extractive QA and label/value linking retain their existing span-based checks. These contracts require empirical visual calibration; valid JSON alone is not proof of correct image reading.

For local object identification, attribute and verbatim transcription questions, question-fit and holistic judges receive only pixels inside the target rectangle, falling back to the scope rectangle. Crops carry their source view identity, exact source coordinates and encoded hash. Rounding does not expand the region; empty crops abstain. Question/answer generators retain the full original view. Full-grid table retries keep their original token budget; an incomplete grid may abstain but cannot become a closed certificate.

UI location answers that exactly repeat the public target reference are rejected before visual review, with the text and reason preserved privately. A location description remains eligible.

The [100-image development inspection manifest](../validation/diverse_100_20261004_manifest.json) fixes image identities, strata, host/model allocation and seed. These are previously used evaluation images, not an independent holdout. Natural-image results and the remaining task coverage must be reported separately from shared CPU contract tests.
