"""OpenAI-compatible local vLLM transport and model adapters."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ValidationError

from pixelogue.config import ModelEndpoint, RuntimeConfig
from pixelogue.contracts import ClaimExtraction, EvidenceInventory, TextPayload
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.prompts import STAGE_INSTRUCTIONS, SYSTEM_PROMPT, validate_stage_payload
from pixelogue.serialization import canonical_hash, canonical_json, strict_json_object
from pixelogue.store import RunStore
from pixelogue.task_evidence import (
    AttributeRecheckReport,
    CandidateBindingsReport,
    ScopedEvidenceInventory,
    ScopedEvidenceReport,
)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
SPECIALIST_SOURCE_STAGES = frozenset(
    {
        "specialist_render_source",
        "specialist_ui_source",
        "specialist_circuit_source",
        "specialist_chemistry_source",
        "specialist_music_source",
        "specialist_geometry_source",
    }
)


def _bind_specialist_source_schema(
    schema: dict[str, Any], stage: str, payload: dict[str, Any]
) -> None:
    """Fix public domain and image IDs before a blind specialist extraction."""
    operation = payload.get("expected_operation")
    views = payload.get("image_views")
    if not isinstance(operation, dict) or not isinstance(views, list) or len(views) != 1:
        raise ExecutionError(
            "MODEL_PAYLOAD_FIELD", "Specialist source needs one bound operation and view"
        )
    view = views[0]
    properties = schema.get("properties")
    if not isinstance(view, dict) or not isinstance(properties, dict):
        raise ExecutionError("MODEL_PAYLOAD_FIELD", "Specialist source view or schema is invalid")
    fixed = {
        "domain": operation.get("calibrated_domain"),
        "scope_id": operation.get("scope_id"),
        "view_id": view.get("view_id"),
    }
    if any(
        not isinstance(value, str) or not value or not isinstance(properties.get(name), dict)
        for name, value in fixed.items()
    ):
        raise ExecutionError("MODEL_PAYLOAD_FIELD", "Specialist source lacks fixed public IDs")
    for name, value in fixed.items():
        properties[name]["const"] = value
    if stage == "specialist_music_source":
        parameters = operation.get("public_parameters")
        if not isinstance(parameters, list):
            raise ExecutionError("MODEL_PAYLOAD_FIELD", "Music source lacks public parameters")
        ranges = [
            item.get("value")
            for item in parameters
            if isinstance(item, dict) and item.get("name") == "bar_range"
        ]
        if len(ranges) != 1 or not isinstance(ranges[0], str) or not ranges[0]:
            raise ExecutionError("MODEL_PAYLOAD_FIELD", "Music source lacks a unique bar range")
        properties["bar_range"]["const"] = ranges[0]


def _model_json_object(payload: str | bytes) -> dict[str, Any]:
    """Translate malformed model JSON into a retryable execution failure.

    Preserve strict parsing for envelopes and generated content, including duplicate
    keys and non-finite values. Input-file parsing retains its external-input errors.
    """
    try:
        return strict_json_object(payload)
    except ExternalInputError as error:
        raise ExecutionError("MODEL_SCHEMA_MISMATCH", f"{error.reason}: {error}") from error


@dataclass(frozen=True)
class ModelImage:
    """Verified image bytes attached to one model request."""

    view_id: str
    path: Path
    encoded_sha256: str
    media_type: str

    def data_uri(self) -> str:
        """Return bytes as a data URI after checking the recorded hash.

        Raises:
            ExecutionError: If the file no longer matches the immutable image view.
        """
        payload = self.path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.encoded_sha256:
            raise ExecutionError("IMAGE_BYTES_MISMATCH", f"Changed image view: {self.view_id}")
        encoded = base64.b64encode(payload).decode("ascii")
        return f"data:{self.media_type};base64,{encoded}"


@dataclass(frozen=True)
class ModelResponse:
    """Validated typed model output and usage."""

    value: BaseModel
    request_hash: str
    response_hash: str
    prompt_tokens: int
    completion_tokens: int


class ModelAdapter:
    """Model-family differences that do not change domain payloads."""

    def __init__(self, repo_id: str) -> None:
        """Select adapter behavior for a configured repository."""
        self.repo_id = repo_id

    def extra_body(self) -> dict[str, Any]:
        """Return explicit non-thinking controls supported by the model family."""
        if self.repo_id.startswith("Qwen/"):
            return {"chat_template_kwargs": {"enable_thinking": False}}
        if self.repo_id.startswith("google/gemma-4"):
            return {"reasoning_effort": "none"}
        return {}

    def clean_content(self, content: str) -> str:
        """Remove only Gemma's documented empty disabled-thinking wrapper."""
        if not self.repo_id.startswith("google/gemma-4"):
            return content
        empty_prefixes = (
            "<|channel|>thought\n<|channel|>",
            "<|channel>thought\n<channel|>",
        )
        for prefix in empty_prefixes:
            if content.startswith(prefix):
                return content[len(prefix) :]
        if "<|channel|>thought" in content or "<|channel>thought" in content:
            raise ExecutionError(
                "MODEL_REASONING_LEAK", "Gemma returned non-empty reasoning content"
            )
        return content


def read_request_artifact(store: RunStore, artifact_hash: str) -> dict[str, Any]:
    """Restore the exact request envelope from legacy or deduplicated audit storage."""
    request = strict_json_object(store.read_artifact(artifact_hash))
    version = request.pop("archive_format", None)
    if version is None:
        return request
    if version != "image-refs-v1":
        raise ExecutionError("REQUEST_ARCHIVE_FORMAT", f"Unsupported archive format: {version}")
    for message in request["request"]["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") == "image_url":
                url = part["image_url"]["url"]
                if url.startswith("pixelogue-image:"):
                    part["image_url"]["url"] = store.read_artifact(
                        url.removeprefix("pixelogue-image:")
                    ).decode("utf-8")
    return request


class VllmClient:
    """Strict local client for image-capable vLLM chat completions."""

    def __init__(
        self,
        endpoint: ModelEndpoint,
        runtime: RuntimeConfig,
        *,
        run_id: str,
        store: RunStore | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize a client without contacting the server."""
        self.endpoint = endpoint
        self.runtime = runtime
        self.run_id = run_id
        self.store = store
        self.adapter = ModelAdapter(endpoint.repo_id)
        self._validate_local_endpoint()
        api_key = os.environ.get(endpoint.api_key_env, "EMPTY")
        self.client = client or httpx.Client(
            base_url=str(endpoint.base_url),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=runtime.request_timeout_seconds,
        )

    def _validate_local_endpoint(self) -> None:
        host = urlparse(str(self.endpoint.base_url)).hostname
        if host not in {"127.0.0.1", "localhost", "::1"}:
            raise ExecutionError(
                "EXTERNAL_INFERENCE_FORBIDDEN", f"Inference host is not local: {host}"
            )

    def invoke(
        self,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        response_model: type[ResponseModel],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        bypass_cache: bool = False,
        retry_feedback: str | None = None,
        json_object_fallback: bool = False,
    ) -> ModelResponse:
        """Send and validate one non-streaming structured-output request.

        Transport failures and 429/5xx responses are retried within the configured attempt limit.
        Schema retries are handled by the caller because they consume a new semantic model response.

        Raises:
            ExecutionError: If the request, transport, completion, or typed output is invalid.
        """
        validate_stage_payload(stage, payload)
        body = self._build_body(
            stage,
            payload,
            images,
            response_model,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            retry_feedback=retry_feedback,
            json_object_fallback=json_object_fallback,
        )
        if bypass_cache:
            body["metadata"] = {"probe_id": canonical_hash(body)}
        model_lock = {
            "repo_id": self.endpoint.repo_id,
            "revision": self.endpoint.revision,
            "processor_revision": self.endpoint.processor_revision,
            "dtype": self.endpoint.dtype,
            "quantization": self.endpoint.quantization,
            "max_model_len": self.endpoint.max_model_len,
            "gpu_memory_utilization": self.endpoint.gpu_memory_utilization,
        }
        request_envelope = {"model_lock": model_lock, "request": body}
        request_hash = canonical_hash(request_envelope)
        model_lock_hash = canonical_hash(model_lock)
        if self.store is not None and not bypass_cache:
            cached = self.store.completed_model_call(model_lock_hash, stage, request_hash)
            if cached is not None:
                cached_response, prompt_tokens, completion_tokens = cached
                typed, response_hash = self._decode_typed_response(
                    cached_response,
                    response_model,
                    max_tokens=max_tokens,
                )
                return ModelResponse(
                    value=typed,
                    request_hash=request_hash,
                    response_hash=response_hash,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
        reserved = False
        raw_content: bytes | None = None
        duration_ms = 0
        if self.store is not None:
            self.store.reserve_model_request(
                max_tokens,
                request_limit=self.runtime.max_total_requests,
                output_token_limit=self.runtime.max_total_output_tokens,
            )
            reserved = True
        try:
            request_started = time.perf_counter()
            raw_response = self._request(body)
            duration_ms = round((time.perf_counter() - request_started) * 1000)
            raw_content = raw_response.content
            typed, response_hash = self._decode_typed_response(
                raw_content,
                response_model,
                max_tokens=max_tokens,
            )
            parsed = _model_json_object(raw_response.content)
            _, usage = self._validate_completion(parsed)
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            if self.store is not None:
                self._record(
                    stage,
                    request_envelope,
                    raw_content,
                    request_hash,
                    response_hash,
                    prompt_tokens,
                    completion_tokens,
                    duration_ms,
                )
                self.store.finalize_model_request(max_tokens, completion_tokens)
                reserved = False
        except BaseException:
            if self.store is not None and reserved:
                self.store.release_model_reservation(max_tokens)
                if raw_content is not None:
                    self._record(
                        stage,
                        request_envelope,
                        raw_content,
                        request_hash,
                        hashlib.sha256(raw_content).hexdigest(),
                        0,
                        0,
                        duration_ms,
                        status="INVALID",
                    )
            raise
        return ModelResponse(
            value=typed,
            request_hash=request_hash,
            response_hash=response_hash,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _decode_typed_response(
        self,
        raw_response: bytes,
        response_model: type[ResponseModel],
        *,
        max_tokens: int,
    ) -> tuple[ResponseModel, str]:
        """Validate one saved or fresh completion and return its typed content."""
        parsed = _model_json_object(raw_response)
        content, usage = self._validate_completion(parsed)
        if response_model is TextPayload and parsed["choices"][0]["finish_reason"] != "stop":
            raise ExecutionError(
                "MODEL_FINISH_REASON", "Public text must finish before its token limit"
            )
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if prompt_tokens < 0 or completion_tokens < 0 or completion_tokens > max_tokens:
            raise ExecutionError("MODEL_USAGE_INVALID", "Token usage is outside request bounds")
        cleaned = self.adapter.clean_content(content)
        _model_json_object(cleaned)
        try:
            typed = response_model.model_validate_json(cleaned)
        except ValidationError as error:
            if parsed["choices"][0]["finish_reason"] == "length":
                raise ExecutionError(
                    "MODEL_FINISH_REASON", "Completion reached its token limit before the schema"
                ) from error
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", str(error)) from error
        return typed, hashlib.sha256(raw_response).hexdigest()

    def _build_body(
        self,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        response_model: type[BaseModel],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        retry_feedback: str | None = None,
        json_object_fallback: bool = False,
    ) -> dict[str, Any]:
        if json_object_fallback and (
            stage != "specialist_chemistry_source" or retry_feedback is None
        ):
            raise ExecutionError(
                "MODEL_OUTPUT_FORMAT",
                "JSON-object fallback is limited to corrected blind chemical extraction",
            )
        declared_views = payload.get("image_views", [])
        view_ids = [item["view_id"] for item in declared_views]
        attached_ids = [image.view_id for image in images]
        if view_ids != attached_ids:
            raise ExecutionError(
                "IMAGE_PART_MISMATCH",
                "Declared image views and attached image parts differ in count or order",
            )
        request_text: dict[str, Any] = {
            "instruction": STAGE_INSTRUCTIONS[stage],
            "input": payload,
        }
        if retry_feedback is not None:
            request_text["retry_feedback"] = retry_feedback
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": canonical_json(request_text).decode(),
            }
        ]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": image.data_uri()}} for image in images
        )
        schema = response_model.model_json_schema()
        if stage in SPECIALIST_SOURCE_STAGES:
            _bind_specialist_source_schema(schema, stage, payload)
        if response_model in (EvidenceInventory, ScopedEvidenceInventory, ScopedEvidenceReport):
            image_id = payload.get("image_id")
            if not isinstance(image_id, str) or not image_id:
                raise ExecutionError("MODEL_PAYLOAD_FIELD", "Evidence requires an image identity")
            schema["properties"]["image_id"]["const"] = image_id
        if response_model is AttributeRecheckReport:
            for field in ("image_id", "scope_id", "view_id"):
                value = payload.get(field)
                if not isinstance(value, str) or not value:
                    raise ExecutionError("MODEL_PAYLOAD_FIELD", f"Attribute recheck lacks {field}")
                schema["properties"][field]["const"] = value
        if response_model is ScopedEvidenceReport:
            vocabulary = payload.get("capability_vocabulary")
            limit = payload.get("max_observations_per_scope")
            max_scopes = payload.get("max_scopes")
            if (
                not isinstance(vocabulary, dict)
                or not vocabulary
                or any(not isinstance(name, str) or not name for name in vocabulary)
                or not isinstance(limit, int)
                or not 0 < limit <= 50
                or not isinstance(max_scopes, int)
                or not 0 < max_scopes <= 8
            ):
                raise ExecutionError(
                    "MODEL_PAYLOAD_FIELD", "Scoped evidence requires bounded capability names"
                )
            scopes_schema = schema["properties"]["scopes"]
            scopes_schema["maxItems"] = max_scopes
            observations_schema = schema["$defs"]["ScopeEvidenceReport"]["properties"][
                "observations"
            ]
            observations_schema["properties"] = {
                name: {"$ref": "#/$defs/CapabilityReport"} for name in sorted(vocabulary)
            }
            observations_schema["additionalProperties"] = False
            observations_schema["maxProperties"] = limit
        if response_model is CandidateBindingsReport:
            candidates = payload.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ExecutionError("MODEL_PAYLOAD_FIELD", "Bindings require candidates")
            candidate_ids = sorted({item["candidate_id"] for item in candidates})
            check_ids = sorted(
                {check for item in candidates for check in item["required_check_ids"]}
            )
            parameter_names = sorted(
                {name for item in candidates for name in item["bindable_parameter_names"]}
                - {"target"}
            )
            definitions = schema["$defs"]
            definitions["CandidateBindingReport"]["properties"]["candidate_id"]["enum"] = (
                candidate_ids
            )
            check_lengths = [len(item["required_check_ids"]) for item in candidates]
            checks_schema = definitions["CandidateBindingReport"]["properties"]["checks"]
            checks_schema["minItems"] = min(check_lengths)
            checks_schema["maxItems"] = max(check_lengths)
            definitions["EligibilityObservation"]["properties"]["check_id"]["enum"] = check_ids
            if parameter_names:
                definitions["PublicParameter"]["properties"]["name"]["enum"] = parameter_names
            else:
                # An empty enum makes the referenced object unsatisfiable in vLLM's
                # grammar compiler, even when the containing array is empty.
                definitions["CandidateBindingReport"]["properties"]["public_parameters"][
                    "maxItems"
                ] = 0
        if response_model is ClaimExtraction:
            tokens = payload.get("answer_tokens")
            if not isinstance(tokens, list) or any(
                not isinstance(token, dict) or token.get("index") != index
                for index, token in enumerate(tokens)
            ):
                raise ExecutionError(
                    "MODEL_PAYLOAD_FIELD", "Claim extraction requires indexed answer_tokens"
                )
            count = len(tokens)
            properties = schema["$defs"]["ClaimSpan"]["properties"]
            properties["start_token"]["maximum"] = max(0, count - 1)
            properties["end_token"]["maximum"] = max(1, count)
            if count == 0:
                schema["properties"]["claims"]["maxItems"] = 0
        body: dict[str, Any] = {
            "model": self.endpoint.model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "n": 1,
            "stream": False,
            "temperature": temperature,
            "top_p": 1.0,
            "seed": seed,
            "max_tokens": max_tokens,
            "response_format": (
                {"type": "json_object"}
                if json_object_fallback
                else {
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": schema,
                    },
                }
            ),
        }
        body.update(self.adapter.extra_body())
        return body

    def _request(self, body: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.runtime.transport_max_attempts):
            try:
                response = self.client.post("chat/completions", json=body)
                if response.status_code == 429 or response.status_code >= 500:
                    if self.store is not None:
                        self.store.write_json_artifact(
                            "transport-errors",
                            {
                                "request_hash": canonical_hash(body),
                                "model_repo": self.endpoint.repo_id,
                                "attempt": attempt + 1,
                                "status_code": response.status_code,
                                "response_body": response.text[:4096],
                                "truncated": len(response.text) > 4096,
                            },
                        )
                    last_error = ExecutionError(
                        "MODEL_TRANSPORT_RETRYABLE",
                        f"Inference server returned {response.status_code}",
                    )
                else:
                    response.raise_for_status()
                    return response
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
                last_error = error
            except httpx.HTTPStatusError as error:
                response_body = error.response.text[:2048]
                if self.store is not None:
                    self.store.write_json_artifact(
                        "transport-errors",
                        {
                            "request_hash": canonical_hash(body),
                            "model_repo": self.endpoint.repo_id,
                            "attempt": attempt + 1,
                            "status_code": error.response.status_code,
                            "response_body": response_body,
                            "truncated": len(error.response.text) > 2048,
                        },
                    )
                raise ExecutionError(
                    "MODEL_REQUEST_REJECTED",
                    f"Inference server returned {error.response.status_code}: {response_body}",
                ) from error
            if attempt + 1 < self.runtime.transport_max_attempts:
                time.sleep(2**attempt)
        raise ExecutionError("MODEL_TRANSPORT_FAILED", str(last_error)) from last_error

    @staticmethod
    def _validate_completion(response: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ExecutionError("MODEL_CHOICE_COUNT", "Completion must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise ExecutionError("MODEL_FINISH_REASON", "Completion did not finish with stop")
        finish_reason = choice.get("finish_reason")
        if finish_reason not in {"stop", "length"}:
            raise ExecutionError("MODEL_FINISH_REASON", "Completion did not finish with stop")
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("tool_calls"):
            raise ExecutionError("MODEL_MESSAGE_INVALID", "Completion contains tools or no message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ExecutionError("MODEL_CONTENT_EMPTY", "Completion content is empty")
        if finish_reason == "length":
            completed = VllmClient._close_json_delimiters(content)
            if completed is None:
                raise ExecutionError("MODEL_FINISH_REASON", "Completion ended before a JSON value")
            content = completed
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return content, usage

    @staticmethod
    def _close_json_delimiters(content: str) -> str | None:
        """Close only missing terminal JSON containers in an otherwise complete object."""
        stripped = content.rstrip()
        stack: list[str] = []
        in_string = False
        escaped = False
        pairs = {"}": "{", "]": "["}
        for character in stripped:
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "{[":
                stack.append(character)
            elif character in "}]":
                if not stack or stack.pop() != pairs[character]:
                    return None
        if in_string or len(stack) > 8:
            return None
        suffix = "".join("}" if opener == "{" else "]" for opener in reversed(stack))
        candidate = stripped + suffix
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return candidate if isinstance(parsed, dict) else None

    def _record(
        self,
        stage: str,
        request: dict[str, Any],
        response: bytes,
        request_hash: str,
        response_hash: str,
        prompt_tokens: int,
        completion_tokens: int,
        duration_ms: int,
        status: Literal["COMPLETE", "INVALID"] = "COMPLETE",
    ) -> None:
        assert self.store is not None
        archived = json.loads(canonical_json(request))
        for message in archived["request"].get("messages", []):
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if part.get("type") == "image_url":
                    url = part["image_url"]["url"]
                    if url.startswith("data:image/"):
                        image_hash = self.store.write_artifact(
                            "request-images", url.encode("utf-8")
                        )
                        part["image_url"]["url"] = f"pixelogue-image:{image_hash}"
                        archived["archive_format"] = "image-refs-v1"
        request_artifact = self.store.write_artifact("requests", canonical_json(archived))
        response_artifact = self.store.write_artifact("responses", response)
        model_lock_hash = canonical_hash(request["model_lock"])
        call_id = canonical_hash({"run": self.run_id, "stage": stage, "request": request_hash})
        with self.store.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO model_call(
                       call_id, stage, model_repo, model_revision, model_lock_hash,
                       request_hash, request_artifact_hash, response_artifact_hash,
                       input_tokens, output_tokens, duration_ms, status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call_id,
                    stage,
                    self.endpoint.repo_id,
                    self.endpoint.revision,
                    model_lock_hash,
                    request_hash,
                    request_artifact,
                    response_artifact,
                    prompt_tokens,
                    completion_tokens,
                    duration_ms,
                    status,
                ),
            )
        if response_artifact != response_hash:
            raise ExecutionError("MODEL_RESPONSE_HASH", "Stored response identity changed")

    def health(self) -> dict[str, Any]:
        """Return the local server's model listing for doctor checks."""
        try:
            response = self.client.get("models")
            response.raise_for_status()
            value = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as error:
            raise ExecutionError("MODEL_HEALTH_FAILED", str(error)) from error
        if not isinstance(value, dict):
            raise ExecutionError("MODEL_HEALTH_INVALID", "Model listing must be an object")
        return value
