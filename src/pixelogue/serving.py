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
from pixelogue.errors import ExecutionError
from pixelogue.prompts import STAGE_INSTRUCTIONS, SYSTEM_PROMPT, validate_stage_payload
from pixelogue.serialization import canonical_hash, canonical_json, strict_json_object
from pixelogue.store import RunStore

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


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
            parsed = strict_json_object(raw_response.content)
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
        parsed = strict_json_object(raw_response)
        content, usage = self._validate_completion(parsed)
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        if prompt_tokens < 0 or completion_tokens < 0 or completion_tokens > max_tokens:
            raise ExecutionError("MODEL_USAGE_INVALID", "Token usage is outside request bounds")
        cleaned = self.adapter.clean_content(content)
        strict_json_object(cleaned)
        try:
            typed = response_model.model_validate_json(cleaned)
        except ValidationError as error:
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
    ) -> dict[str, Any]:
        declared_views = payload.get("image_views", [])
        view_ids = [item["view_id"] for item in declared_views]
        attached_ids = [image.view_id for image in images]
        if view_ids != attached_ids:
            raise ExecutionError(
                "IMAGE_PART_MISMATCH",
                "Declared image views and attached image parts differ in count or order",
            )
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": canonical_json(
                    {"instruction": STAGE_INSTRUCTIONS[stage], "input": payload}
                ).decode(),
            }
        ]
        user_content.extend(
            {"type": "image_url", "image_url": {"url": image.data_uri()}} for image in images
        )
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
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
        }
        body.update(self.adapter.extra_body())
        return body

    def _request(self, body: dict[str, Any]) -> httpx.Response:
        last_error: Exception | None = None
        for attempt in range(self.runtime.transport_max_attempts):
            try:
                response = self.client.post("chat/completions", json=body)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = ExecutionError(
                        "MODEL_TRANSPORT_RETRYABLE",
                        f"Inference server returned {response.status_code}",
                    )
                else:
                    response.raise_for_status()
                    return response
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                last_error = error
            except httpx.HTTPStatusError as error:
                raise ExecutionError("MODEL_REQUEST_REJECTED", str(error)) from error
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
        request_artifact = self.store.write_artifact("requests", canonical_json(request))
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
