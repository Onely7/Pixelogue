"""Native Jev transports outside the finite-decision domain contract."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib
import importlib.util
import io
import math
import os
import selectors
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pixelogue.decision import (
    DECISION_LABELS,
    DecisionClient,
    DecisionLabel,
    DecisionModelIdentity,
    DecisionProbability,
    DecisionRequest,
    DecisionResult,
)
from pixelogue.errors import CapabilityError, ExecutionError, ExternalInputError
from pixelogue.serialization import canonical_json, strict_json_object

OMNI_REPOSITORY = "akhilaaa3/Jev-Omni"


class _Usage(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None


class _NativeChoice(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    kind: Literal["choice"]
    options: list[DecisionLabel]
    probabilities: list[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]
    choice_index: Annotated[int, Field(ge=0, lt=3)]
    choice: DecisionLabel
    protocol: Literal["jev27-bare-v1"]
    model: str
    usage: _Usage | None = None
    num_model_requests: Annotated[int, Field(ge=1)]


def _verified_image(request: DecisionRequest) -> bytes:
    """Read only the immutable local view bound to this request."""
    try:
        payload = Path(request.image_path).read_bytes()
    except OSError as error:
        raise ExecutionError("DECISION_IMAGE_UNREADABLE", str(error)) from error
    if hashlib.sha256(payload).hexdigest() != request.image_sha256:
        raise ExecutionError("DECISION_IMAGE_CHANGED", "Decision image bytes changed")
    return payload


def _public_state(request: DecisionRequest) -> str:
    """Describe the supplied view and scope without adding model or gold output."""
    return canonical_json(
        {
            "public_context": request.public_context,
            "view_id": request.view_id,
            "scope_region": request.region.model_dump(),
            "coordinates": "normalized in this delivered image; origin is top left",
            "labels": {
                "MET": "The requested visual condition is established in the stated scope.",
                "NOT_MET": "The requested visual condition is contradicted or absent.",
                "UNKNOWN": "The view is insufficient or ambiguous to decide the condition.",
            },
        }
    ).decode("utf-8")


def _result(
    request: DecisionRequest,
    *,
    repository: str,
    revision: str,
    verdict: DecisionLabel,
    probabilities: tuple[DecisionProbability, ...],
    started: float,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    num_model_requests: int = 1,
    native_protocol: str | None = None,
) -> DecisionResult:
    """Validate native scores without renormalizing or inventing usage."""
    try:
        return DecisionResult(
            request_id=request.request_id,
            request_hash=request.identity(repository, revision),
            model_repository=repository,
            model_revision=revision,
            verdict=verdict,
            probabilities=probabilities,
            elapsed_seconds=time.perf_counter() - started,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            num_model_requests=num_model_requests,
            native_protocol=native_protocol,
        )
    except ValidationError as error:
        raise ExecutionError("DECISION_OUTPUT_INVALID", str(error)) from error


class JevHttpDecisionClient:
    """Call the pinned native JEV choice endpoint with thinking disabled.

    The configured revision must also be verified in the server launch manifest;
    the native endpoint reports a model name, not an attested weight revision.
    Transport failures are raised without an implicit model or API fallback.
    """

    def __init__(
        self,
        base_url: str,
        model_repository: str,
        model_revision: str,
        *,
        timeout_seconds: float = 120.0,
        http_client: httpx.Client | None = None,
        served_model_name: str | None = None,
        allow_external_inference: bool = False,
    ) -> None:
        """Bind a native endpoint and an exact externally verified model identity."""
        parsed = urlparse(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Decision endpoint must be an HTTP URL without credentials or query")
        if not allow_external_inference and parsed.hostname not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ExecutionError(
                "EXTERNAL_INFERENCE_FORBIDDEN",
                f"Decision inference host is not local: {parsed.hostname}",
            )
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Decision timeout must be positive")
        DecisionModelIdentity(repository=model_repository, revision=model_revision)
        base = base_url.rstrip("/")
        self.endpoint = f"{base}/decide" if base.endswith("/v1") else f"{base}/v1/decide"
        self.model_repository = model_repository
        self.model_revision = model_revision
        self.served_model_name = served_model_name or model_repository
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(timeout=timeout_seconds)

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Send one three-label native choice and preserve complete scores.

        Raises:
            ExecutionError: If bytes, transport or native output fail validation.
        """
        started = time.perf_counter()
        payload = _verified_image(request)
        from PIL import Image

        try:
            with Image.open(io.BytesIO(payload)) as image:
                media_type = Image.MIME.get(image.format or "")
            if media_type not in {"image/png", "image/jpeg", "image/webp"}:
                raise ValueError("Unsupported decision image format")
        except (OSError, ValueError) as error:
            raise ExecutionError("DECISION_IMAGE_INVALID", str(error)) from error
        data_uri = f"data:{media_type};base64,{base64.b64encode(payload).decode('ascii')}"
        body = {
            "kind": "choice",
            "state": [_public_state(request), {"image": data_uri}],
            "question": request.question,
            "options": list(DECISION_LABELS),
            "strategy": "single",
            "thinking": "off",
            "return_reasoning": False,
            "debug": False,
        }
        try:
            response = self._http.post(self.endpoint, json=body)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ExecutionError("DECISION_TRANSPORT_FAILED", str(error)) from error
        try:
            raw = strict_json_object(response.content)
            if raw.get("thinking", {}).get("used") is True:
                raise ValueError("Native endpoint used unrequested generative thinking")
            native = _NativeChoice.model_validate(raw)
            if tuple(native.options) != DECISION_LABELS or len(native.probabilities) != 3:
                raise ValueError("Native endpoint changed the requested label set or order")
            if native.model != self.served_model_name:
                raise ValueError("Native endpoint served a different model")
            if native.options[native.choice_index] != native.choice:
                raise ValueError("Native choice and index disagree")
        except (ExternalInputError, ValidationError, ValueError, AttributeError) as error:
            raise ExecutionError("DECISION_OUTPUT_INVALID", str(error)) from error
        usage = native.usage
        return _result(
            request,
            repository=self.model_repository,
            revision=self.model_revision,
            verdict=native.choice,
            probabilities=tuple(
                DecisionProbability(label=label, probability=probability)
                for label, probability in zip(DECISION_LABELS, native.probabilities, strict=True)
            ),
            started=started,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            num_model_requests=native.num_model_requests,
            native_protocol=native.protocol,
        )

    def close(self) -> None:
        """Close an adapter-owned transport while preserving injected clients."""
        if self._owns_client:
            self._http.close()


class OmniPredictor(Protocol):
    """Native forward-only classifier, which owns mutable hook capture state."""

    def predict(
        self, *, state: str, question: str, options: list[str], media: str, modality: str
    ) -> dict[str, Any]:
        """Return the native label, index and complete probability mapping."""
        ...


def _load_local_omni(snapshot_path: Path, device: str) -> OmniPredictor:
    """Assemble the inspected local snapshot without a network download."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as error:
        raise CapabilityError(
            "DECISION_RUNTIME_MISSING", "Use the separate pinned GPU runtime for Jev-Omni"
        ) from error
    if device != "cuda" or not torch.cuda.is_available():
        raise CapabilityError(
            "DECISION_CUDA_REQUIRED", "Jev-Omni requires an explicitly assigned GPU"
        )
    script = snapshot_path / "jev_omni.py"
    spec = importlib.util.spec_from_file_location("pixelogue_pinned_jev_omni", script)
    if spec is None or spec.loader is None:
        raise CapabilityError("DECISION_SNAPSHOT_INVALID", "Cannot load the pinned Omni helper")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    config = transformers.AutoConfig.from_pretrained(str(snapshot_path), local_files_only=True)
    architecture = config.architectures[0]
    if architecture != "Gemma4UnifiedForConditionalGeneration":
        raise CapabilityError("DECISION_SNAPSHOT_INVALID", "Unexpected Omni model architecture")
    model = (
        getattr(transformers, architecture)
        .from_pretrained(str(snapshot_path), dtype=torch.bfloat16, local_files_only=True)
        .to(device)
        .eval()
    )
    decision = strict_json_object((snapshot_path / "decision_config.json").read_bytes())
    head = native._Head256(decision["hidden_size"]).to(device).eval()
    head.load_state_dict(
        torch.load(snapshot_path / "head.pt", map_location=device, weights_only=True)
    )
    _, decoder = native._find_backbone(model)
    processor = transformers.AutoProcessor.from_pretrained(
        str(snapshot_path), local_files_only=True
    )
    return native.JevOmni(model, head, processor, decoder, device)


class LocalOmniDecisionClient:
    """Serialize forward passes through the native Jev-Omni hook-based head.

    Construction allocates no GPU. Call ``load`` before measured requests to
    record startup separately. If omitted, the first call includes loading time.
    No free-form generation or remote snapshot lookup is used.
    """

    def __init__(
        self,
        snapshot_path: str | Path,
        revision: str,
        *,
        device: str = "cuda",
        predictor: OmniPredictor | None = None,
    ) -> None:
        """Bind a pinned local snapshot or an already loaded native worker."""
        self.snapshot_path = Path(snapshot_path)
        DecisionModelIdentity(repository=OMNI_REPOSITORY, revision=revision)
        if self.snapshot_path.name != revision:
            raise ValueError("Omni snapshot directory must be named by its pinned revision")
        self.model_repository = OMNI_REPOSITORY
        self.model_revision = revision
        self.device = device
        self._predictor = predictor
        self._lock = threading.Lock()
        self.model_load_seconds: float | None = None

    def _load_locked(self) -> OmniPredictor:
        if self._predictor is None:
            started = time.perf_counter()
            self._predictor = _load_local_omni(self.snapshot_path, self.device)
            self.model_load_seconds = time.perf_counter() - started
        return self._predictor

    def load(self) -> None:
        """Load once inside the dedicated worker and retain its startup duration."""
        with self._lock:
            self._load_locked()

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Verify one immutable view and evaluate it without concurrent hook access."""
        started = time.perf_counter()
        with self._lock:
            image_bytes = _verified_image(request)
            predictor = self._load_locked()
            try:
                with tempfile.NamedTemporaryFile(suffix=".image") as immutable_view:
                    immutable_view.write(image_bytes)
                    immutable_view.flush()
                    output = predictor.predict(
                        state=_public_state(request),
                        question=request.question,
                        options=list(DECISION_LABELS),
                        media=immutable_view.name,
                        modality="image",
                    )
                probabilities = output["probabilities"]
                if not isinstance(probabilities, dict) or set(probabilities) != set(
                    DECISION_LABELS
                ):
                    raise ValueError("Omni output must contain exactly the requested labels")
                choice = output["prediction"]
                index = output["prediction_index"]
                if type(index) is not int or not 0 <= index < 3 or choice != DECISION_LABELS[index]:
                    raise ValueError("Omni choice and index disagree")
                parsed = tuple(
                    DecisionProbability(label=label, probability=probabilities[label])
                    for label in DECISION_LABELS
                )
                if "confidence" in output and output["confidence"] != probabilities[choice]:
                    raise ValueError("Omni confidence and selected score disagree")
            except (KeyError, TypeError, ValueError, ValidationError) as error:
                raise ExecutionError("DECISION_OUTPUT_INVALID", str(error)) from error
            return _result(
                request,
                repository=self.model_repository,
                revision=self.model_revision,
                verdict=choice,
                probabilities=parsed,
                started=started,
                native_protocol="jev-omni-forward-head-v1",
            )


class ThreadedAsyncDecisionClient:
    """Adapt a synchronous service to an event loop without sharing Omni hooks."""

    def __init__(self, client: DecisionClient) -> None:
        """Keep the synchronous service's existing serialization and ownership."""
        self.client = client

    async def decide(self, request: DecisionRequest) -> DecisionResult:
        """Run one decision in a worker while native adapters enforce their locks."""
        return await asyncio.to_thread(self.client.decide, request)


class SubprocessOmniDecisionClient:
    """Own one long-lived Omni worker in the separately pinned GPU environment.

    The caller chooses an available physical GPU UUID before constructing the
    client. Neither GPU discovery nor automatic worker restart is performed.
    Worker failures preserve the distinction between an invalid decision and a
    broken process; the caller can resume a saved run with a new explicit client.
    """

    def __init__(
        self,
        runtime_python: str | Path,
        snapshot_path: str | Path,
        revision: str,
        *,
        env: Mapping[str, str],
        stderr_path: str | Path,
        startup_timeout_seconds: float = 600.0,
        request_timeout_seconds: float = 120.0,
    ) -> None:
        """Start a serialized local worker with explicit runtime, GPU and logs."""
        executable = Path(runtime_python)
        if (
            not executable.is_absolute()
            or not executable.is_file()
            or not os.access(executable, os.X_OK)
        ):
            raise ValueError("Omni runtime Python must be an absolute executable path")
        gpu = env.get("CUDA_VISIBLE_DEVICES", "")
        if not gpu.startswith("GPU-") or "," in gpu:
            raise ValueError("Omni worker requires one explicitly selected physical GPU UUID")
        if any(
            not math.isfinite(value) or value <= 0
            for value in (startup_timeout_seconds, request_timeout_seconds)
        ):
            raise ValueError("Omni worker timeouts must be finite and positive")
        DecisionModelIdentity(repository=OMNI_REPOSITORY, revision=revision)
        snapshot = Path(snapshot_path)
        if not snapshot.is_absolute() or snapshot.name != revision:
            raise ValueError("Omni worker snapshot must be an absolute pinned revision directory")
        self.model_repository = OMNI_REPOSITORY
        self.model_revision = revision
        self.request_timeout_seconds = request_timeout_seconds
        self.model_load_seconds: float | None = None
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._closed = False
        logs = Path(stderr_path)
        logs.parent.mkdir(parents=True, exist_ok=True)
        self._log = logs.open("ab")
        child_env = {**os.environ, **env}
        source_parent = str(Path(__file__).resolve().parent.parent)
        existing_pythonpath = child_env.get("PYTHONPATH")
        child_env["PYTHONPATH"] = source_parent + (
            os.pathsep + existing_pythonpath if existing_pythonpath else ""
        )
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        try:
            self._process = subprocess.Popen(
                [
                    str(executable),
                    "-u",
                    "-m",
                    "pixelogue.decision_worker",
                    "--snapshot-path",
                    str(snapshot),
                    "--revision",
                    revision,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                env=child_env,
            )
        except OSError:
            self._log.close()
            raise
        self._selector: selectors.BaseSelector | None = None
        try:
            self._selector = selectors.DefaultSelector()
            if self._process.stdout is None:
                raise ExecutionError("DECISION_WORKER_START_FAILED", "Worker stdout is unavailable")
            self._selector.register(self._process.stdout, selectors.EVENT_READ)
            ready = self._read_frame(startup_timeout_seconds)
            self._check_error(ready)
            if (
                ready.get("status") != "ready"
                or ready.get("model_repository") != self.model_repository
                or ready.get("model_revision") != self.model_revision
            ):
                raise ExecutionError(
                    "DECISION_WORKER_PROTOCOL", "Unexpected worker identity or ready frame"
                )
            load_seconds = ready.get("model_load_seconds")
            if load_seconds is not None:
                if (
                    type(load_seconds) not in {int, float}
                    or not math.isfinite(load_seconds)
                    or load_seconds < 0
                ):
                    raise ExecutionError("DECISION_WORKER_PROTOCOL", "Invalid worker load duration")
                self.model_load_seconds = float(load_seconds)
        except BaseException:
            self.close()
            raise

    def _read_frame(self, timeout_seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        while True:
            if b"\n" in self._buffer:
                payload, _, remaining = self._buffer.partition(b"\n")
                self._buffer = bytearray(remaining)
                try:
                    return strict_json_object(bytes(payload))
                except ExternalInputError as error:
                    raise ExecutionError("DECISION_WORKER_PROTOCOL", str(error)) from error
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise ExecutionError("DECISION_WORKER_TIMEOUT", "Omni worker response timed out")
            if self._selector is None:
                raise ExecutionError(
                    "DECISION_WORKER_EXITED", "Omni worker selector is unavailable"
                )
            events = self._selector.select(remaining_seconds)
            if not events:
                continue
            stream = self._process.stdout
            if stream is None:
                raise ExecutionError("DECISION_WORKER_EXITED", "Omni worker stdout closed")
            chunk = os.read(stream.fileno(), 8192)
            if not chunk:
                raise ExecutionError(
                    "DECISION_WORKER_EXITED", "Omni worker exited before a complete response"
                )
            self._buffer.extend(chunk)
            if len(self._buffer) > 1024 * 1024:
                raise ExecutionError(
                    "DECISION_WORKER_PROTOCOL", "Omni worker response exceeded its bound"
                )

    @staticmethod
    def _check_error(frame: dict[str, Any]) -> None:
        if frame.get("status") == "error":
            reason = frame.get("reason")
            message = frame.get("message")
            if not isinstance(reason, str) or not isinstance(message, str):
                raise ExecutionError("DECISION_WORKER_PROTOCOL", "Malformed Omni worker error")
            raise ExecutionError(reason, message)

    def decide(self, request: DecisionRequest) -> DecisionResult:
        """Request one native decision and reject mismatched or failed replies."""
        started = time.perf_counter()
        with self._lock:
            if self._closed or self._process.poll() is not None:
                raise ExecutionError("DECISION_WORKER_EXITED", "Omni worker is closed or exited")
            stream = self._process.stdin
            if stream is None:
                raise ExecutionError("DECISION_WORKER_EXITED", "Omni worker stdin is unavailable")
            try:
                stream.write(request.model_dump_json().encode("utf-8") + b"\n")
                stream.flush()
                frame = self._read_frame(self.request_timeout_seconds)
            except (OSError, ExecutionError) as error:
                self.close()
                if isinstance(error, ExecutionError):
                    raise
                raise ExecutionError("DECISION_WORKER_EXITED", str(error)) from error
            self._check_error(frame)
            try:
                if frame.get("status") != "result" or set(frame) != {"status", "result"}:
                    raise ValueError("Unexpected Omni worker response frame")
                result = DecisionResult.model_validate_json(canonical_json(frame["result"]))
                if (
                    result.model_repository != self.model_repository
                    or result.model_revision != self.model_revision
                ):
                    raise ValueError("Omni worker result changed its model identity")
                result.validate_request(request)
            except (ValueError, ValidationError, TypeError) as error:
                self.close()
                raise ExecutionError("DECISION_WORKER_PROTOCOL", str(error)) from error
            return result.model_copy(update={"elapsed_seconds": time.perf_counter() - started})

    def close(self) -> None:
        """Stop the owned worker and close its logs and pipes; safe to repeat."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()
        if self._selector is not None:
            self._selector.close()
        self._log.close()
