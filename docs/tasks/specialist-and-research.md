# Specialist verification and research runs

[日本語](specialist-and-research_ja.md) · [Task catalog](README.md)

## Admission and environments

The 65 core tasks have normal verification paths. Seven extensions have first-version validators. `configs/specialist-pilot.yaml` names all seven, but naming an extension does not admit it. `compile`, `doctor` and `task-status` report each validator's version, environment, calibration domain and selection reason. Normal selection requires a certificate for the **exact active generator/processor pair**, validator version and supported domain. The per-image capability and scope checks and eight-candidate limit still apply.

```bash
uv sync --locked --extra cpu --group dev
uv sync --locked --directory runtime/validators
uv run --locked pixelogue compile --config configs/specialist-pilot.yaml --output artifacts/specialist-compile.json
uv run --locked pixelogue task-status --config configs/specialist-pilot.yaml --output-stem artifacts/task-status
```

Pass `--junit artifacts/junit.xml` after a full CPU test run, and `--conversations artifacts/<run>/conversations.jsonl` for a specific GPU run. The per-task report lists attempted answer turns and committed turns separately, with each turn's status and generator model. Shared CPU fixtures do not establish positive, incorrect-answer and insufficient-evidence boundaries for every task; those task-specific branches remain `not_recorded` until measured. Candidate-only GPU attempts are not counted as answer turns.

The specialist lock contains SymPy, music21, RDKit and Playwright. It is separate from the application and `runtime/vllm/` environments. The registered rules cover explicit rational geometry, single-voice complete score bars, nonstereo atom/bond graphs, resolved two-terminal circuit nets, one visible UI action, static SVG or limited TikZ, and static HTML/CSS. Unsupported notation, incomplete source evidence and disagreement between two blind image readings produce `UNKNOWN`. A passing syntax check alone cannot certify a visual task.
Normal selection checks the version registered for each specialist validator against its certificate. Increment that version when a source prompt or verifier contract changes, so older certificates cannot enable the revised task.
For geometry, the two blind readings must agree on each typed premise and its image region (intersection over union at least 0.1). Differences in prose or premise listing order do not change a proof fact. Repeated or nonlocal premises are rejected.
For a complete single-voice music bar, written note and rest durations must sum exactly to the public meter. A whole note uses base `1`, a half note `2`, a quarter note `4`, and one dot multiplies duration by `3/2`; inconsistent model output is retried, then remains unverified if unresolved.
Chemical source retries identify the first atom or bond box outside the scoped image region. The source still fails if the corrected box does not fit the actual visible molecule.
An aromatic bond outside a closed ring is rejected before graph comparison and receives a specific blind retry. A visually unsupported aromatic chain cannot certify a molecule.

The render worker accepts a small static syntax grammar, disables JavaScript and Service Workers, blocks browser requests, and disables Chromium GPU rendering. It uses `bwrap` when user namespaces work, or a bounded user `systemd` service plus Linux Landlock when they do not. The latter probes IPv4/IPv6 denial and denial of project-file reads and writes before enabling the renderer. It grants read-only access to system libraries, fonts, and `/proc`; one private scratch directory under `/tmp` is readable and writable, while other `/tmp` file contents are denied. The service also limits memory, task count, and runtime. The controller compares the isolated screenshot with the delivered image view under the published RGB and foreground thresholds. If OS isolation or Chromium is unavailable, both render tasks are environment-blocked. Never launch candidate code outside this path. UI actions are checked as declarative data and are never performed.
The blind render source reader reports only visible labels and image coverage. Renderer availability and calibration are controller checks, so an evaluator must not treat missing sandbox details in the image request as missing visual evidence.

For a UI action, both blind readings must copy the public target string as the control ID and agree on enabled state. The actionable regions must overlap with intersection-over-union of at least 0.5 for a click or 0.7 for focus and input. A click accepts different control-kind labels when both readings identify the same enabled target; focus and input require both readings to identify a field. Other controls may differ. The reported point must lie inside both target regions and match the delivered view's pixel conversion; unresolved or conflicting targets abstain.
The public target may describe a visible function instead of quoting its label. Readers may use that description only when the screenshot identifies one control unambiguously; they must not infer a hidden action or treat different actions as synonyms.

## Held-out evaluation and calibration

`evaluate-specialist` is the only route for exercising an uncalibrated extension. Its JSONL input uses `SpecialistEvaluationCase` from the `compile` Schemas, an evaluation-only `ImageArtifact`, a public question and answer, and an independently sourced gold label for confirmation images. The gold label is kept out of every model request. Each evaluator makes a separate blind source extraction; the specialist controller check and evidence hashes are stored privately. Evaluation artifacts never enter the normal training export.

```bash
nvidia-smi
uv run --locked pixelogue doctor --config configs/specialist-pilot.yaml --output artifacts/doctor.json
uv run --locked pixelogue evaluate-specialist --config configs/specialist-pilot.yaml --cases validation/local-specialist-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/specialist-evaluation --run-id specialist-eval-v1
uv run --locked pixelogue calibration-build --results-dir artifacts/specialist-evaluation/results --output artifacts/specialist-calibration.json --report artifacts/specialist-calibration-report.json
```

Use the above model command only after confirming an idle device and explicitly selecting it. Long jobs and servers belong in `tmux` with logs and output progress checks. One available GPU is enough for sequential pilot calls; no four-GPU reservation is required. Count loading and every allocated device toward the cumulative four GPU-hour initial budget. The one-GPU pilot's two calls to the same `Qwen/Qwen3.5-9B` endpoint are blind calls, not model diversity or evidence for the standard Qwen3.8/Gemma pair. Do not silently quantize a BF16 configuration.

`calibration-build` requires the frozen `input.json` beside the results directory and checks every expected case ID. If any expected result file is absent, it withholds all certificates from that input and reports the missing IDs. It computes one-sided 95% Clopper–Pearson bounds from independent confirmation image groups only after the frozen input is complete. Eligibility requires a false-accept upper bound at most 5% and a positive-accept lower bound at least 80%. `UNKNOWN` is not positive acceptance. If any confirmation case fails or lacks its independent gold label, the entire matching task/domain/model/validator group remains pending; completed cases in that group cannot produce a certificate by themselves. Failed calls, invalid output, development cases and groups lacking a positive or negative label remain pending in the report. Keep development and confirmation images separate. After a certificate is eligible, set `tasks.calibration_manifest` in a copied config to its path and re-run `compile`; never edit a certificate by hand. A pilot certificate cannot authorize the standard model pair.

`evaluate-specialist`, `research-history`, and `research-ablation` reuse completed trial records on resume. Failed records remain visible; use `--retry-failed` to attempt them again. Specialist evaluation also saves each model stage independently, so evaluator endpoints can be brought up sequentially across retries with the same input and configuration identity.
For slow specialist graphs on a shared GPU, a copied evaluation config may set `runtime.request_timeout_seconds` up to 600. The default remains 180 seconds; record the copied config with each run and do not treat a timeout as a verifier verdict.
For a specialist source that fails structured-output validation, evaluation makes one additional blind call with the exact schema error and region/netlist constraints. The invalid attempt remains recorded; a second failure stays `FAILED` and never counts as a calibration result.
If a specialist source reaches its token limit before a complete JSON object, it receives the same single bounded blind retry with explicit truncation feedback. Schema and length retries have separate counters. The chemical source alone switches to JSON-object generation on this retry to avoid a measured constrained-decoder whitespace loop; the resulting JSON still must pass the unchanged strict `ChemicalSource` model and two-reader graph check. The original incomplete output is retained, and a second invalid output remains `FAILED`.
The specialist source schema fixes the public calibrated domain, scope ID, and delivered view ID; music also fixes the requested bar range. Source extraction uses `tasks.evidence_max_tokens` so a truncated molecular graph cannot be accepted as complete.

## Research-only commands

`research-exposure` fixes image, question, public history, operation, plausible answer, incorrect answer and trial order before any calls. It evaluates hidden, plausible and incorrect conditions through a separate five-item question assessment. Each condition and repetition makes an independent model request. Complete trial files are reused on resume; failures and absent prices remain explicit. The paired plausible-minus-hidden result and AIAS on independently labelled invalid questions are reported in JSON, CSV and Markdown. Run generator A and B in separate output directories when only one endpoint can be loaded at a time.

```bash
uv run --locked pixelogue research-exposure --cases validation/local-exposure-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-exposure-a --evaluator generator_a --plan-only
uv run --locked pixelogue research-exposure --cases validation/local-exposure-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-exposure-a --evaluator generator_a
uv run --locked pixelogue research-exposure-report --output-dir artifacts/research-exposure-a
```

`audit-pack` samples accepted, rejected and abstained outputs separately, recording the source population and actual sampling rate. `questions.html` omits candidate answers; `answers.html` shows them on a separate sheet. Both hide methods, model names, automatic verdicts and error-injection types. The JSONL templates must be completed by independent raters. `audit-resolve` defaults to three raters, keeps every original vote and preserves unreviewed, unknown and disagreement states until adjudication. Automatic verdicts never become human gold labels.

`audit-cases` builds a frame from saved synthesis output. Committed turns become accepted turn cases; a conversation that ends before its next public question or answer contributes a separate stop case. Stops remain in the sampling denominator, while unavailable question and answer ballots are marked not applicable. Execution errors are counted separately. A committed turn does not imply a completed quality candidate.

```bash
uv run --locked pixelogue audit-cases --conversations artifacts/pilot/conversations.jsonl --artifact-root artifacts/prepared --output artifacts/audit-cases.jsonl
uv run --locked pixelogue audit-pack --cases validation/local-audit-cases.jsonl --output-dir artifacts/audit --rate 0.1
uv run --locked pixelogue audit-resolve --pack artifacts/audit/pack.json --question-votes artifacts/audit/question-votes.jsonl --answer-votes artifacts/audit/answer-votes.jsonl --output artifacts/audit/resolution.json
```

`research-history` binds an exact span in a committed public message. For coreference and public-constraint changes it substitutes a plausible alternative before showing the candidate answer, then checks whether that same answer holds under the original history and fails under the alternative. Independent turns receive no witness. Both stages use separate evaluator calls. A turn index alone never establishes dependency.

`research-ablation` freezes questions and compares 18 cells: question gate before/after answer or omitted; evaluator A, B or both; and verified-only or all generated history. Repair is disabled. Holistic plus task-specific answer review remains active. All generated history, including failed turns, is stored only in a research trace and cannot enter normal export. Its reports include every starting case and reached depth, including stopped conversations. This first version fixes questions to isolate gate timing; it does not measure question-generation changes.
The [two-image development cases](../../validation/confirmed_two_image_ablation.jsonl) freeze two separate visible property or relation questions per image for this pilot. Question admissibility labels remain unset; the 36 planned cells are behavior observations until independent auditing is complete.

The ablation `depth.csv` includes human-label false accept and false reject counts, plus overlapping evaluator errors where both evaluators voted. These fields are unmeasured (JSON `null`, empty CSV cells) without independently supplied human labels; zero means labels were checked and no error was found. The answer-exposure report uses the same distinction. The history study and audit resolution each write JSON, CSV, and Markdown reports.

```bash
uv run --locked pixelogue research-history --cases validation/local-history-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-history
uv run --locked pixelogue research-ablation --cases validation/local-ablation-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-ablation --plan-only
uv run --locked pixelogue research-ablation --cases validation/local-ablation-cases.jsonl --artifact-root artifacts/prepared --output-dir artifacts/research-ablation
uv run --locked pixelogue research-ablation-report --output-dir artifacts/research-ablation
```

These commands require locally prepared cases and model servers; the repository does not contain image bytes or human labels. CPU fixtures verify controller logic and fail-closed behavior, not natural-image extraction accuracy. Costs are `null` without a price schedule. Large corpus generation, multi-student SFT, learning curves and full contamination analysis are later work.
