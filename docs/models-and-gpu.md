# Models and GPU checks

Pixelogue separates instruction choice from question and answer generation. `Qwen/Qwen3.5-2B` is
the default selector. `Qwen/Qwen3.6-35B-A3B` is an explicit alternative; its 35B total parameter
count must be used for memory planning. Qwen3.8 and Gemma each generate conversations and both judge
every conversation, regardless of which one generated it.

## 1. Inspect before using a GPU

Run these commands immediately before each model launch:

```sh
nvidia-smi
uv run --locked pixelogue doctor --config configs/pilot.yaml
```

A GPU is considered idle by `doctor` only when utilization is 0% and used memory is below 1 GiB.
Inspect the process table in `nvidia-smi` as well. Do not take a device used by another process.
`doctor --check-servers` also calls the three endpoints required by the active configuration.

`ready: true` from static checks means the pinned BF16 weights fit a set of distinct currently idle
devices for every required simultaneous server. Read each `assigned_gpu_indices` value before
launching. This is not a successful model inference. A server check is ready only when its model
listing contains the configured served name.

## 2. Install the separate serving environment

The application and vLLM have different lock files so GPU packages cannot silently change CPU
development dependencies.

```sh
uv sync --project runtime/vllm --locked
```

The runtime is pinned to vLLM 0.29.0. The server files also pin model revisions, BF16, context size,
tensor parallelism, and vLLM sampling defaults.

## 3. Launch only on explicitly selected devices

These are examples. Replace device numbers after checking the current machine.

```sh
CUDA_VISIBLE_DEVICES=3 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/selector-default.yaml

CUDA_VISIBLE_DEVICES=0,3 \
  uv run --project runtime/vllm --locked vllm serve \
  --config runtime/vllm/generator-a.yaml
```

The full online pipeline needs the active selector and both large-model endpoints reachable at the
same time. The example ports are 8000, 8002, and 8003. Use distinct idle devices for simultaneous
servers. If the machine cannot supply them, stop the pilot; do not enable quantization or reuse a
busy GPU. Test the alternative selector separately on port 8001 after changing
`models.active_selector` to `alternative` in a copied pilot configuration.

Qwen requests carry `enable_thinking=false` at both server and request level. Gemma requests carry
`reasoning_effort=none`; any reasoning control text found in public content is rejected. vLLM
documents [server arguments](https://docs.vllm.ai/en/latest/configuration/serve_args/) and
[server-level thinking controls](https://docs.vllm.ai/en/latest/features/reasoning_outputs/).

## 4. Keep the four GPU-hour pilot limit

GPU-hours equal wall-clock hours multiplied by the number of allocated GPUs. Two GPUs used for 30
minutes consume one GPU-hour. Count model loading and capability probes. Record start and stop times
in a local `_docs/` run note. Stop before the cumulative value reaches 4.0.

Pixelogue records requests and token usage, but it cannot observe time spent starting an external
vLLM process. The operator must include that time. An unavailable GPU or server is an unrun check,
not a passing result.

## 5. Run answer-keyed capability checks

After all required servers are healthy:

```sh
uv run --locked pixelogue evaluate-capabilities \
  --config configs/pilot.yaml \
  --fixtures data/fixtures/fixtures.jsonl \
  --fixture-root data/fixtures \
  --output artifacts/capability-report.json
```

Expected labels remain in the controller and are never sent to either judge. Development and
confirmation counts remain separate. A small fixture count checks wiring only.
