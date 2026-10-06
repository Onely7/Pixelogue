# Models and GPU checks

Pixelogue separates image routing from dialogue generation and judging. The standard profile uses two different generator and evaluator model lineages, both unquantized. A temporary one-GPU pilot substitutes a smaller model to validate the complete execution path.

| Role | Model | Default endpoint |
|---|---|---|
| Image router | `Qwen/Qwen3.8-27B` (generator A's server) | `http://127.0.0.1:8002/v1` |
| Generator and evaluator A | `Qwen/Qwen3.8-27B` | `http://127.0.0.1:8002/v1` |
| Generator and evaluator B | `google/gemma-4-31B-it` | `http://127.0.0.1:8003/v1` |

In `configs/standard.yaml`, Qwen3.8-27B and Gemma 4 31B generate an equal share of conversations. The assigned model drafts the questions, writes the answers, and makes the one allowed answer repair for the entire conversation. Both models also judge every turn through separate blind calls: one question-gate call each before the answer, one holistic review each of the whole turn, and both readings of every operation validator. A gate or review passes only with two MET votes. The router (`models.router`) only profiles each image for task-family routing; it receives one image per request and writes no dialogue. It is generator A's own server: `models.router` must repeat `models.generator_a` exactly (the checked-in files use a YAML anchor), so routing needs no extra server or memory.

No model is quantized, and dtype stays BF16. Configuration accepts only the `Qwen/Qwen3.8-27B` and `google/gemma-4-31B-it` pair, or the temporary Qwen3.5-9B pilot pair, and rejects any `quantization` setting as an unknown field. The earlier FP8 and W4A16 checkpoints and their separate Qwen3.5-2B router are no longer accepted for the standard pair.

`configs/pilot.yaml` is a temporary validation override. It maps both logical roles to a shared `Qwen/Qwen3.5-9B` server on port 8002, and a separate `Qwen/Qwen3.5-2B` server on port 8000 routes its images. The calls remain separate and blind, but the shared checkpoint means this profile tests pipeline wiring rather than evaluator-model diversity.

`configs/split-pilot.yaml` keeps the standard models but places them on three GPUs that need not share a host: Gemma 4 31B on two 48 GiB GPUs (`runtime/vllm/generator-b-split.yaml`, port 18703) and Qwen3.8-27B, which also routes images, on one 96 GiB GPU (`runtime/vllm/generator-a-split.yaml`, port 18702, memory fraction 0.88). This layout ran with Gemma on two RTX 6000 Ada GPUs and Qwen on an RTX PRO 6000 Blackwell GPU of another host behind an SSH forward ([section 3](#split-layout-across-two-hosts)); recheck capacity on other hardware. This is a pilot scheduling profile, not a change to the standard profile.
`configs/split-diverse.yaml` uses the same endpoints and model locks with a 60-image evaluation target for the pinned Commons category sweep. It keeps evaluation-only inputs separate from training exports.

## 1. Inspect before using a GPU

Run these commands immediately before launch:

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

A GPU is considered idle by `doctor` only when utilization is 0% and used memory is below 1 GiB. Inspect the process table in `nvidia-smi` as well. Never take a device used by another process.

Static `ready: true` means the pinned BF16 model servers can be assigned to the currently idle devices. Roles that share the same repository, revision, and endpoint, such as the router and generator A, are counted as one server. Static readiness is a capacity check, not a successful inference.
`doctor` estimates resident BF16 weights from the published checkpoint sizes plus 10%: about 57 GiB for Qwen3.8-27B and 64 GiB for Gemma 4 31B, split evenly across tensor-parallel shards. Each model therefore needs two 48 GiB GPUs or one 96 GiB GPU. Recheck these estimates after a checkpoint revision or hardware change.

After launch, `doctor --check-servers` checks the configured served model names. The chosen GPU is no longer idle at that point; a healthy already-running server can satisfy its role without being allocated again.

## 2. Install the separate serving environment

The application and vLLM use different lock files so GPU packages cannot silently change the CPU development environment.

```sh
uv sync --project runtime/vllm --locked
```

The runtime is pinned to vLLM 0.29.0. Server files set dtype (BF16), context length, tensor parallelism, memory utilization, the structured-output backend, and generation defaults. Both generators cap concurrent sequences at 32, the value used in the measured runs; vLLM's higher default exceeded the Qwen model's available Mamba cache blocks in an earlier one-GPU startup probe. Both select `xgrammar` with `disable_any_whitespace: false`. No server file sets a quantization method.

Every generator server file sets `limit-mm-per-prompt` to two images, because a judge can receive the complete image and a crop of the bound region in one request. The pilot router files keep one image.

### Download the pinned weights

The server files read the weights from `models/`, which is local-only and never committed, and serve them under their repository IDs. Start every server from the repository root. Download each pinned revision once to shared storage:

```sh
uv run --project runtime/vllm --locked hf download Qwen/Qwen3.8-27B \
  --revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 --local-dir models/Qwen3.8-27B
uv run --project runtime/vllm --locked hf download google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --local-dir models/gemma-4-31B-it
```

Gemma requires accepting its license on Hugging Face and a token in the environment. Before serving, compare every file against the Hub's SHA-256 listing for the pinned revision and keep that record with the weights; the configurations pin the same revisions for model locks. vLLM detects a network filesystem and prefetches the checkpoint into the page cache. On the measured NFS mount, each model's weights loaded in under 30 seconds, and a server became ready in four to five minutes including compilation.

## 3. Run long GPU work in tmux

The standard servers use these files:

```text
runtime/vllm/generator-a.yaml   -> Qwen3.8-27B, port 8002, tensor parallel size 2 (also the router)
runtime/vllm/generator-b.yaml   -> Gemma 4 31B, port 8003, tensor parallel size 2
```

Run `doctor --config configs/standard.yaml` immediately before launch and assign only the idle GPUs it reports. Start each server in its own `tmux` window from the repository root. For example, after replacing the device IDs with GPUs confirmed to be idle:

```sh
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a.yaml

CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-b.yaml
```

These device numbers are examples. Required memory depends on the hardware and serving runtime, so the static `doctor` result and an actual startup check are both required. On 96 GiB GPUs, one GPU per model is enough when `tensor-parallel-size` is 1, as in the split files.

### Split layout across two hosts

The split files fit hosts with different GPUs. Run Pixelogue on the host with the two 48 GiB GPUs, start Gemma there, start Qwen on the 96 GiB GPU of the other host, and forward that host's port to the same local port. Both servers listen only on 127.0.0.1.

```sh
# Host with the 96 GiB GPU (replace the UUID with an inspected idle device)
CUDA_VISIBLE_DEVICES=GPU-QWEN HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a-split.yaml

# Host with the two 48 GiB GPUs, which also runs Pixelogue
CUDA_VISIBLE_DEVICES=GPU-GEMMA-0,GPU-GEMMA-1 HF_HUB_OFFLINE=1 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-b-split.yaml
ssh -N -o ControlMaster=no -o ControlPath=none -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=15 -L 127.0.0.1:18702:127.0.0.1:18702 QWEN-HOST
uv run --locked pixelogue doctor --config configs/split-pilot.yaml --check-servers
```

Keep the forward in its own `tmux` window and restart it if it exits. Disable SSH connection sharing for the forward as shown: with a shared master connection, the master owns the forwarded port and keeps it after the forwarding client ends. If that happens, remove it with `ssh -O cancel -L 127.0.0.1:18702:127.0.0.1:18702 QWEN-HOST`.

### Temporary one-GPU pilot

Create separate windows for the generator, router, pilot, and monitoring commands:

```sh
tmux new-session -d -s pixelogue-qwen35-pilot -n generator
tmux new-window -t pixelogue-qwen35-pilot -n control
tmux new-window -t pixelogue-qwen35-pilot -n router
tmux new-window -t pixelogue-qwen35-pilot -n pilot
```

Use an explicitly selected idle device. The number below is only an example. This command launches the temporary Qwen3.5-9B substitute, not either standard generator.

Start the larger server first:

```sh
CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-qwen35-9b.yaml
```

Wait for the generator model list, then start the router on the same device:

```sh
curl -fsS http://127.0.0.1:8002/v1/models

CUDA_VISIBLE_DEVICES=3 HF_HOME=/var/tmp/pixelogue-hf \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/router-default.yaml
```

The checked-in memory-utilization limits are 0.68 for the 9B server and 0.20 for the 2B router. They were validated for an RTX 6000 Ada with 48 GiB-class memory. Re-run capacity checks on a different GPU instead of copying those values blindly.

Redirect each window to a unique local log. A pane with `pane_dead=0` is only one health signal; also inspect `/v1/models`, logs, GPU activity, output rows, and model-call status.

### Bounded idle-GPU watcher

`gpu_watch.py` checks every local physical GPU twice with `nvidia-smi`, including the compute-process table. It starts only after `doctor` reports the one-GPU pilot ready. The memory reservation adapts the supplied CUDA allocation script, but omits continuous matrix multiplication. During model startup the holder keeps 25% of GPU memory allocated; the generator starts while that process remains present. After the generator answers `/v1/models`, the watcher releases the holder and acknowledges the handoff before the router starts. This avoids an unreserved gap. After the pilot, the watcher reacquires an idle GPU with a 90% holder until stopped or the cumulative budget is reached. It never takes a GPU with another compute process, even if Slurm reports its node as idle.

The watcher uses separate ports 18002 and 18000 because the usual router port may be occupied by another local service. It uses the same pinned, unquantized Qwen3.5-9B and Qwen3.5-2B model revisions. The job runs one image probe, an intentional interruption and resume on two images the user confirmed can support different visual questions, the full 20-image Open Images evaluation, and a 60-category web-image evaluation. It verifies completed-run reuse and replay, served names, `doctor --check-servers`, output row counts, and terminal errors. The two confirmed images do not have independent answer labels. Server and run logs are saved separately. This pilot does not certify the standard model pair or specialist calibration.

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
Each named extension can now allocate up to 24 GPU-hours for sustained, user-authorized validation. Record all model loading, inference and reservation intervals in the same ledger. A larger allowance does not change model identities or the idle-device check.

Add `--reserve-only` while diagnosing a failed pilot. This holds an idle GPU without launching the pilot; stop the watcher and restart with the same campaign ID without that flag to run the corrected job. Both phases consume the same cumulative GPU ledger.

After a failed pilot with budget remaining, `--retry-failed-pilot --campaign-id ID` resets only the one-time attempt flag for that campaign. It does not add GPU hours. Keep the original failure logs and use a new run ID for the corrected code.

After a completed diagnostic pilot, `--rerun-completed-pilot --campaign-id ID` starts another run with the current code if that campaign still has at least ten minutes of unspent GPU time. It does not increase the budget. Stop an existing post-job holder cleanly before using this option; the next run receives a new run ID and again restores the reservation after its job.

## 4. Wait for real readiness

vLLM continues with compilation, CUDA graph capture, and multimodal warmup after loading weights. Allocated GPU memory or a live PID does not mean the server is ready.

```sh
curl -fsS http://127.0.0.1:8002/v1/models
curl -fsS http://127.0.0.1:8003/v1/models
uv run --locked pixelogue doctor \
  --config configs/standard.yaml \
  --check-servers
```

For the temporary one-GPU pilot, check ports 8002 and 8000 with `configs/pilot.yaml`.

Before a 50-image run, send one image through the same structured-output stages. A server can pass the model-list check while its guided-decoding backend rejects a particular JSON Schema.

If a completion stops at its token limit and lacks a required field, the run records an incomplete completion rather than a generic schema mismatch. Chart and graph source readers retry with up to twice their output limit, capped at 8,192 tokens unless the configured limit is already higher; every other stage, including table readers, keeps its limit. The retry must still satisfy the full schema and scope checks; nothing missing is filled in.

Missing JSON braces or brackets are never appended, even when other fields appear complete. The original response must already contain one complete JSON object.

If a length-limited JSON completion ends with at least 512 whitespace characters and remains invalid, the run records `MODEL_WHITESPACE_RUNAWAY`. A bounded retry receives explicit finish guidance but keeps the original output limit. A complete, schema-valid JSON object with trailing whitespace is still accepted. The invalid response remains a private artifact; no missing field or visual fact is inferred.

Qwen receives `enable_thinking=false` in both server defaults and each request; Gemma receives `reasoning_effort: none`. Thinking text and internal control fields must not enter public dialogue.

### Request timeouts

Each request's read timeout grows with its output budget: `min(900, max(request_timeout_seconds, 60 + max_tokens × timeout_seconds_per_1k_output_tokens / 1000))` seconds, with a connect timeout of at most 30 seconds. With the defaults of 180 and 45, a 1,024-token answer keeps 180 seconds, and a 4,096-token source reading receives about 244 seconds. Setting `runtime.timeout_seconds_per_1k_output_tokens: null` restores the fixed `request_timeout_seconds`, which may be at most 600. A timeout is a transport failure retried within `runtime.transport_max_attempts`; it is never a verdict.

## 5. Account for the complete GPU allocation

GPU-hours equal elapsed hours multiplied by allocated GPU count. One GPU used for 45 minutes consumes 0.75 GPU-hours; two GPUs for the same time consume 1.5. Include model loading, smoke tests and reservation holders. The initial pilot allocation was four GPU-hours; named extensions record subsequent user-authorized validation in the same ledger. Record start and stop times in a local `_docs/` note.

Pixelogue records requests and token use, but it cannot observe time spent starting an external vLLM process. The operator adds that time. An unavailable server is an unrun check, not a pass.

## 6. Run answer-keyed capability checks

After all required servers are healthy:

```sh
uv run --locked pixelogue evaluate-capabilities \
  --config configs/pilot.yaml \
  --fixtures data/fixtures/fixtures.jsonl \
  --fixture-root data/fixtures \
  --output artifacts/capability-report.json
```

Expected labels remain in the controller and are never sent to evaluator calls. Development and confirmation counts remain separate. A small fixture result checks wiring; it is not a natural-image accuracy claim.

The [pipeline guide](pipeline/README.md) explains how these servers participate in every generation and evaluation stage.

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

For the split layout on ports 18702 and 18703, set `PIXELOGUE_LIVE_CONFIG=configs/split-pilot.yaml` as well. This changes only the test endpoint addresses and keeps the standard model identities.

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

## Stop runaway JSON generation

The generator server files keep whitespace between JSON tokens available (`xgrammar`, `disable_any_whitespace: false`). Suppressing it was evaluated earlier and not adopted. To test it again, set `disable_any_whitespace: true` when launching the server: in the pinned vLLM 0.29.0 backend the setting applies only at launch, and a similarly named request field alone does not establish that it was applied. Ordinary spaces inside strings remain available either way. Restart existing servers, record their launch manifest before comparing runs, and adopt a faster condition only if it also preserves supported candidate yield and task coverage.

The engine repetition guard is on by default:

```yaml
runtime:
  repetition_detection:
    min_pattern_size: 1
    max_pattern_size: 4
    min_count: 64
    recovery: retry
    stages: ["*_source", question_draft, image_profile]
```

It stops consecutive token-pattern repetition without changing sampling probabilities.
Stage patterns use shell-style matching and may cover only the blind source readers
(`*_source`), question drafting and the image profile. Answers, repairs, answer parsers
and every judge keep unmodified decoding; a pattern that matches any of those stages
fails validation. The thresholds are experimental, and a long legitimate repeated sequence
may also stop. Set `runtime.repetition_detection: null` to disable the guard.
`configs/repetition-detection-pilot.yaml` spells out the default guard and otherwise
matches `configs/split-pilot.yaml`.

`finish_reason: repetition` is recorded as `MODEL_OUTPUT_REPETITION`, preserves the
raw response and incurred usage, and rejects even parseable JSON from the stopped
completion. With `recovery: retry`, a guarded stage receives a new complete response
within `structured_output_max_attempts`, with correction feedback and the original output
limit. It never accepts partial output or increases the budget because of repetition.
With `recovery: abstain`, the first stop is final: a drafting call yields no drafts, the
image profile falls back to default families, and a stopped source reader ends the image
as abstained. A configuration change produces a different run identity. This option
requires an endpoint implementing the pinned vLLM repetition-detection API.

The [vLLM 0.29.0 serving reference](https://docs.vllm.ai/en/v0.29.0/cli/serve/),
[SamplingParams](https://docs.vllm.ai/en/v0.29.0/api/vllm/sampling_params/) and
[structured output guide](https://docs.vllm.ai/en/v0.29.0/features/structured_outputs/)
describe the available interfaces. Inspect the installed runtime with
`uv run --project runtime/vllm --locked vllm serve --help=all`; a listed option does
not establish support for every model or backend.

## Bounded table and verbatim-text verification

Single-cell lookups in simple labeled tables use two answer-blind readings of the complete row/column header domains and only the requested cell. Missing addressing domains, ambiguous selected headers, misaligned or off-scope cells, and disagreements abstain. Selected row and column header anchors must align with the requested cell; unrelated grid coordinates and values are not generated. Merged or hierarchical headers, relevant footnotes, and multi-cell requests retain full-grid verification. Predicate selection, joins and reconstruction always retain complete grids.

Verbatim text, reading-order and code transcription use two answer-blind source readings as literal line arrays. The controller compares the complete answer, permitting only outer quotations or code fences; it never accepts a matching substring that omits or adds text. The blind readers receive the original image so they can detect text continuing outside the bound rectangle. The controller requires the whole requested unit to fit its bound image region. Non-verbatim extractive QA and label/value linking retain their existing span-based checks. These contracts require empirical visual calibration; valid JSON alone is not proof of correct image reading.

For local object identification, attribute lookup, and verbatim transcription, reading-order and code questions, the question gate and holistic judges receive a crop of the target rectangle, falling back to the scope rectangle. With `evaluation.judge_views: full_and_crop` (default), the complete image comes first and the crop second; `crop` sends the crop alone. A full-view tie-break is available only to judges that saw a crop. Crops carry their source view identity, exact source coordinates and encoded hash. Rounding does not expand the region, and an empty crop can never certify a turn. Question/answer generators retain the full original view. Full-grid table retries keep their original token budget; an incomplete grid may abstain but cannot become a closed certificate.

UI location answers that exactly repeat the public target reference are rejected before visual review, with the text and reason preserved privately. A location description remains eligible.

The [100-image development inspection manifest](../validation/diverse_100_20261004_manifest.json) fixes image identities, strata, host/model allocation and seed. These are previously used evaluation images, not an independent holdout. Natural-image results and the remaining task coverage must be reported separately from shared CPU contract tests.
