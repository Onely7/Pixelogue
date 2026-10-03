# Experimental Jev routing

[日本語](jev-routing_ja.md)

This opt-in path targets evidence extraction and candidate binding. A generator
first proposes short, located observations or concrete public parameters. A Jev
classifier then evaluates each visual claim or eligibility condition separately
as `MET`, `NOT_MET`, or `UNKNOWN`. The controller supplies stable identifiers and
checks the assembled records with the existing validators.

The classifier does not discover coordinates, transcribe text, reconstruct a
table, or produce an answer by itself. Missing claims and incomplete sets do not
become `MET`. Unsupported binding tasks use the bounded existing binding path.
The default selector, generator pair, question gate, answer verification and
evaluation-only data rules continue to apply.

## Enable one stage at a time

The validated task settings are:

```yaml
tasks:
  decision_routing:
    evidence_enabled: false
    binding_enabled: true
    model_repository: autotrust/JEV-27B-VL
    model_revision: f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc
    proposal_max_tokens: 1536
```

Both switches default to `false`. The `synthesize` CLI creates only the enabled
classifier and closes it on success, failure or interruption. For JEV, configure
its native HTTP `base_url`. For Omni, provide absolute `runtime_python` and
`snapshot_path` paths, with the revision as the snapshot directory name, and set
`CUDA_VISIBLE_DEVICES` to one inspected physical GPU UUID before starting.
The job coordinator starts model servers and manages reservations. `rate-existing`
does not start the classifier. Python callers can inject a client into
`SynthesisCoordinator(..., decision_client=client)`.

## JEV-27B-VL: native HTTP service

Start the inspected, pinned native `serve_decide.py` in the separate vLLM
environment, using the snapshot's decision LoRA and head. The endpoint is
`POST /v1/decide`; an ordinary chat endpoint is not equivalent.

```python
from pixelogue.decision_serving import JevHttpDecisionClient

client = JevHttpDecisionClient(
    "http://127.0.0.1:18104/v1",
    "autotrust/JEV-27B-VL",
    "f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc",
)
try:
    # Inject client into the already configured SynthesisCoordinator.
    pass
finally:
    client.close()
```

The adapter fixes three options and disables generative thinking. It verifies
the response model name, native protocol, option order, selected label and full
normalized probability distribution. The revision also needs verification in
the actual server launch manifest: the API reports a name, not a weight hash.
By default only loopback hosts are accepted. For a remote host, prefer an
explicit SSH tunnel terminating on loopback. The programmatic external-host
override is separate from the application's current local-only runtime policy.

## Jev-Omni: separate persistent worker

Keep ML packages in `runtime/vllm`, outside the application environment. The
application starts one persistent local JSONL worker through an explicitly
configured runtime Python. The worker loads the pinned local snapshot once in
BF16, uses the native forward-pass decision head, and serializes requests because
the head's hook capture is shared mutable state. It does not call `.generate()`.

```python
from pathlib import Path
from pixelogue.decision_serving import SubprocessOmniDecisionClient

revision = "5addda86ddee081a68fb067477ea100c221b8917"
client = SubprocessOmniDecisionClient(
    Path("runtime/vllm/.venv/bin/python").resolve(),
    Path("models/decision-classifiers/Jev-Omni", revision).resolve(),
    revision,
    env={"CUDA_VISIBLE_DEVICES": "GPU-REPLACE-WITH-INSPECTED-IDLE-UUID"},
    stderr_path=Path("artifacts/jev-worker/worker.log"),
)
try:
    # Inject client into the already configured SynthesisCoordinator.
    pass
finally:
    client.close()
```

The GPU UUID must be selected before construction. The adapter does not search
for a GPU, reserve one, change quantization, or download a snapshot. The worker
uses `local_files_only=True` and offline settings. A job supervisor remains
responsible for inspecting `nvidia-smi`, running `doctor`, using tmux, restoring
reservations after success or failure, and accounting for model loading and every
allocated GPU. Actual memory requirements need a startup and image smoke test.

The worker can also be run directly from the repository root:

```sh
CUDA_VISIBLE_DEVICES=GPU-REPLACE-WITH-INSPECTED-IDLE-UUID \
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
runtime/vllm/.venv/bin/python -u -m pixelogue.decision_worker \
  --snapshot-path /absolute/path/to/Jev-Omni/5addda86ddee081a68fb067477ea100c221b8917 \
  --revision 5addda86ddee081a68fb067477ea100c221b8917 \
  2>artifacts/jev-worker/worker.log
```

Create the log directory first. Standard input contains one serialized
`DecisionRequest` per line. Standard output is reserved for a `ready` frame,
then `result` or explicit `error` frames; model progress goes to stderr. Worker
failure and invalid output are never converted into a passing label. A dead or
timed-out worker is closed without an implicit restart; resume the saved run with
a new explicitly configured client.

## Private records and measurement

`DecisionRequest` carries exact image bytes' identity, view, normalized region,
public context, contract version and independent trial ID. Reference labels,
candidate answers, future turns and other judges' results must stay outside its
inputs. A local path is a locator, not the image identity. The Omni adapter passes
an immutable copy of verified image bytes to its native predictor.

`DecisionResult` preserves all three scores and optional observed token usage.
Scores are not calibration certificates. Missing usage remains missing. Cached
replay must match the same trial, public history, image/view, contract and pinned
model; different experiment trials require different calls. The application
stores decisions privately and continues the existing public-output validation.

Use approximately 24 image groups for the first cycle, with six connection and
development images and eighteen comparison images. Freeze the split before
reviewing model results. Compare the four fixed models on identical finite
questions against an independently collected GPT-6.1 Sol Ultra reference. Report
three-way agreement, false `MET`, missed `MET`, `UNKNOWN`, failures and opportunity
coverage by image family, along with median/p95 latency. Measure completed two-turn
conversations per GPU hour only after a development condition preserves evidence
correctness. If none qualifies, stop expansion and record the non-adoption reason
and unperformed comparisons. Include short-label Qwen/Gemma baselines so shortening
output is distinguishable from switching to a native classifier. A small pilot
supports diagnosis; it does not establish general accuracy or specialist
calibration.
