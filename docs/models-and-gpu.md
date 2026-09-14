# Models and GPU checks

Pixelogue separates instruction choice from dialogue generation. The standard profile uses two different generator and evaluator model lineages. A temporary one-GPU pilot substitutes a smaller model to validate the complete execution path.

| Role | Model | Default endpoint |
|---|---|---|
| Instruction selector | `Qwen/Qwen3.5-2B` | `http://127.0.0.1:8000/v1` |
| Optional selector | `Qwen/Qwen3.6-35B-A3B` | `http://127.0.0.1:8001/v1` |
| Generator and evaluator A | `Qwen/Qwen3.8-27B` | `http://127.0.0.1:8002/v1` |
| Generator and evaluator B | `google/gemma-4-31B-it` | `http://127.0.0.1:8003/v1` |

In `configs/standard.yaml`, Qwen3.8-27B and Gemma 4 31B generate an equal share of conversations. The assigned model remains fixed for the entire conversation, including its one allowed repair. Both models also judge every conversation through separate blind calls.

`configs/pilot.yaml` is a temporary validation override. It maps both logical roles to a shared `Qwen/Qwen3.5-9B` server on port 8002. The calls remain separate and blind, but the shared checkpoint means this profile tests pipeline wiring rather than evaluator-model diversity.

## 1. Inspect before using a GPU

Run these commands immediately before launch:

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

A GPU is considered idle by `doctor` only when utilization is 0% and used memory is below 1 GiB. Inspect the process table in `nvidia-smi` as well. Never take a device used by another process.

Static `ready: true` means the pinned BF16 model servers can be assigned to the currently idle devices. Generator roles that share the same repository, revision, and endpoint are counted as one server. Static readiness is a capacity check, not a successful inference.

After launch, `doctor --check-servers` checks the configured served model names. The chosen GPU is no longer idle at that point; a healthy already-running server can satisfy its role without being allocated again.

## 2. Install the separate serving environment

The application and vLLM use different lock files so GPU packages cannot silently change the CPU development environment.

```sh
uv sync --project runtime/vllm --locked
```

The runtime is pinned to vLLM 0.29.0. Server files pin model revisions, BF16, context length, tensor parallelism, memory utilization, and generation defaults. Quantization is not enabled.

## 3. Run long GPU work in tmux

The standard servers use these files:

```text
runtime/vllm/generator-a.yaml        -> Qwen3.8-27B, port 8002, tensor parallel size 2
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

Qwen receives `enable_thinking=false` in both server defaults and each request. Thinking text and internal control fields must not enter public dialogue.

## 5. Treat the alternative selector separately

`Qwen/Qwen3.6-35B-A3B` is used only when a copied configuration explicitly sets:

```yaml
models:
  active_selector: alternative
```

It is a 35B-total-parameter model for memory planning. Test it in a separate run on port 8001 with enough idle capacity. Pixelogue does not start it alongside the default selector and does not switch to it after a failure.

## 6. Keep the four GPU-hour pilot limit

GPU-hours equal elapsed hours multiplied by allocated GPU count. One GPU used for 45 minutes consumes 0.75 GPU-hours; two GPUs for the same time consume 1.5. Include model loading and smoke tests. Record start and stop times in a local `_docs/` note and stop before the cumulative 4.0 limit.

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
