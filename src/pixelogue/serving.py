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

from pixelogue.call_usage import measure_usage
from pixelogue.config import ModelEndpoint, RuntimeConfig
from pixelogue.contracts import TextPayload
from pixelogue.drafting import QuestionDraftBatch
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.gates import QuestionGateVote
from pixelogue.prompts import STAGE_INSTRUCTIONS, SYSTEM_PROMPT, validate_stage_payload
from pixelogue.routing import ImageProfile
from pixelogue.serialization import canonical_hash, canonical_json, strict_json_object
from pixelogue.store import RunStore
from pixelogue.task_evidence import ImageRegion

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
STRUCTURAL_OUTPUT_STAGES = frozenset(
    {
        "table_source",
        "table_lookup_source",
        "table_answer",
        "transcript_source",
        "extractive_source",
        "document_source",
        "chart_source",
        "chart_answer",
        "graph_source",
        "graph_answer",
        "geometry_answer",
        "specialist_geometry_source",
        "specialist_circuit_source",
    }
)
STRUCTURAL_ANSWER_FORMS = {
    "chart_answer": {
        "chart_value_lookup": frozenset({"value"}),
        "chart_comparison": frozenset({"relation"}),
        "chart_value_arithmetic": frozenset({"value"}),
        "chart_extremum_ranking": frozenset({"rank_groups"}),
        "chart_trend_summary": frozenset({"trend", "trend_segments"}),
        "chart_series_relation": frozenset({"relation"}),
    },
    "table_answer": {
        "table_cell_lookup": frozenset({"value"}),
        "table_row_selection": frozenset({"rows"}),
        "table_join": frozenset({"pairs"}),
    },
    "graph_answer": {
        "diagram_connectivity": frozenset({"members", "edges"}),
        "diagram_path_tracing": frozenset({"paths"}),
        "diagram_process_description": frozenset({"edges"}),
        "flowchart_evaluation": frozenset({"paths"}),
    },
}


# Answer-only parses fill exactly one result form when MET; these are the forms each operation
# allows, and every result field of the parse.
ANSWER_PARSE_FORMS: dict[str, dict[str, tuple[str, ...]]] = {
    "finite_answer": {
        "spatial_ordering": ("members",),
        "count_comparison": ("relation",),
        "quantified_claim_verification": ("truth",),
        "hypothetical_set_update": ("count", "members"),
    },
    "quantity_answer": {
        "quantity_comparison": ("relation",),
        "stated_value_consistency": ("truth", "value"),
        "grounded_arithmetic": ("value",),
        "value_aggregation": ("value",),
        "unit_conversion": ("value",),
    },
    "scale_answer": {"measurement_reading": ("value", "time")},
    "pattern_answer": {
        "pattern_rule": ("rule",),
        "pattern_completion": ("option_id",),
        "pattern_exception": ("frame_id",),
    },
}
ANSWER_PARSE_FIELDS = {
    "finite_answer": ("members", "count", "truth", "relation"),
    "quantity_answer": ("value", "relation", "truth"),
    "scale_answer": ("value", "time"),
    "pattern_answer": ("rule", "option_id", "frame_id"),
}
FINITE_ANSWER_FORMS = {
    "members": "members",
    "count": "count",
    "truth": "boolean",
    "relation": "relation",
}
# Parses quote the answer and explain briefly; bounded text keeps a runaway completion short.
PARSE_QUOTE_MAX_LENGTH = 600
PARSE_REASON_MAX_LENGTH = 600
# Bounded fields fit well inside this; a parse that needs more is runaway text.
PARSE_MAX_TOKENS = 768


def _bind_answer_parse_schema(
    schema: dict[str, Any], stage: str, payload: dict[str, Any]
) -> dict[str, Any]:
    """Bind an answer-only parse so that MET carries exactly one allowed result.

    Every field must be emitted. A MET branch fills one result form that the public operation
    allows and leaves the others empty; the abstaining branch leaves every result empty. An
    operation without known forms keeps the unbound schema, which the typed model still checks.
    """
    operation = payload.get("expected_operation")
    task_id = operation.get("task_id") if isinstance(operation, dict) else None
    forms = ANSWER_PARSE_FORMS[stage].get(task_id) if isinstance(task_id, str) else None
    if forms is None:
        return schema
    properties = schema["properties"]
    fields = ANSWER_PARSE_FIELDS[stage]

    def filled(name: str) -> dict[str, Any]:
        if name == "members":
            minimum = 1 if task_id == "spatial_ordering" else 0
            return {**properties["members"], "minItems": minimum}
        options = [item for item in properties[name]["anyOf"] if item.get("type") != "null"]
        return options[0]

    def empty(name: str) -> dict[str, Any]:
        if name == "members":
            return {**properties["members"], "maxItems": 0}
        return {"type": "null"}

    reason = {"type": "string", "minLength": 1, "maxLength": PARSE_REASON_MAX_LENGTH}
    base = {"type": "object", "additionalProperties": False, "required": list(properties)}
    branches = []
    for form in forms:
        branch = {**properties, "coverage": {"type": "string", "const": "MET"}, "reason": reason}
        if "answer_quote" in properties:
            branch["answer_quote"] = {
                "type": "string",
                "minLength": 1,
                "maxLength": PARSE_QUOTE_MAX_LENGTH,
            }
        if stage == "finite_answer":
            branch["answer_form"] = {"type": "string", "const": FINITE_ANSWER_FORMS[form]}
        for name in fields:
            branch[name] = filled(name) if name == form else empty(name)
        branches.append({**base, "properties": branch})
    abstain = {
        **properties,
        "coverage": {"type": "string", "enum": ["NOT_MET", "UNKNOWN"]},
        "reason": reason,
    }
    if "answer_quote" in properties:
        abstain["answer_quote"] = {"type": "string", "maxLength": 0}
    if stage == "finite_answer":
        abstain["answer_form"] = {
            "type": "string",
            "enum": sorted({FINITE_ANSWER_FORMS[form] for form in forms}),
        }
    for name in fields:
        abstain[name] = empty(name)
    branches.append({**base, "properties": abstain})
    return {**({"$defs": schema["$defs"]} if "$defs" in schema else {}), "anyOf": branches}


def _bind_text_payload_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Bind public text: either nonempty text, or no text with the reason for abstaining."""
    properties = schema["properties"]
    base = {"type": "object", "additionalProperties": False, "required": list(properties)}
    return {
        "anyOf": [
            {
                **base,
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "reason": properties["reason"],
                },
            },
            {
                **base,
                "properties": {
                    "text": {"type": "null"},
                    "reason": {"type": "string", "minLength": 1},
                },
            },
        ]
    }


def _bind_chart_source_schema(schema: dict[str, Any], payload: dict[str, Any]) -> None:
    """Prevent contradictory axis shapes and absent operands in a complete reading.

    Bind only the public operation. Marks, labels and values still come from
    the independent reader, and an uncertain reading may leave operands empty.
    """
    definitions = schema["$defs"]
    axis = definitions["ChartAxis"]
    axis_fields = axis["properties"]
    axis_base = {key: value for key, value in axis.items() if key != "properties"}
    definitions["ChartAxis"] = {
        "anyOf": [
            {
                **axis_base,
                "properties": {
                    **axis_fields,
                    "scale": {"type": "string", "const": "unmarked"},
                    "ticks": {**axis_fields["ticks"], "maxItems": 0},
                },
            },
            {
                **axis_base,
                "properties": {
                    **axis_fields,
                    "scale": {"type": "string", "enum": ["linear", "log"]},
                    "ticks": {**axis_fields["ticks"], "minItems": 2},
                },
            },
        ]
    }
    query = definitions["ChartQuery"]
    query["required"] = sorted(set(query["required"]) | {"series", "categories"})
    operation = payload.get("expected_operation")
    task_id = operation.get("task_id") if isinstance(operation, dict) else None
    forms = {
        "chart_value_lookup": ("value", 1, 1),
        "chart_comparison": ("compare", 2, 1),
        "chart_value_arithmetic": ("arithmetic", None, None),
        "chart_extremum_ranking": ("rank", 1, None),
        "chart_trend_summary": ("trend", 1, None),
        "chart_series_relation": ("relation", 2, None),
        "chart_data_reconstruction": ("reconstruct", None, None),
    }
    if task_id not in forms:
        return
    verb, series_count, category_count = forms[task_id]
    query["properties"]["operation"]["const"] = verb
    operands: dict[str, Any] = {}
    if verb != "reconstruct":
        for field, count in (("series", series_count), ("categories", category_count)):
            operands[field] = {**query["properties"][field], "minItems": count or 1}
            if count is not None:
                operands[field]["maxItems"] = count
    if verb == "arithmetic":
        operands["operands"] = {**query["properties"]["operands"], "minItems": 2}
        parameters = {
            item["name"]: item["value"]
            for item in (operation or {}).get("public_parameters", [])
            if isinstance(item, dict)
        }
        if isinstance(parameters.get("operator"), str):
            operands["operator"] = {"type": "string", "const": parameters["operator"]}
        query["required"] = sorted(set(query["required"]) | {"operands", "operator"})
    fields = schema["properties"]
    base = {key: value for key, value in schema.items() if key not in {"properties", "$defs"}}
    # XGrammar compiles each union branch independently. Put the entire object
    # in both branches rather than relying on intersection with sibling fields.
    schema["anyOf"] = [
        {
            **base,
            "properties": {
                **fields,
                "coverage": {"const": "MET"},
                "query": {**query, "properties": {**query["properties"], **operands}},
            },
        },
        {
            **base,
            "properties": {
                **fields,
                "coverage": {"enum": ["UNKNOWN", "NOT_MET"]},
                "axis": {"type": "null"},
                "marks": {**fields["marks"], "maxItems": 0},
                "closed": {"const": False},
            },
        },
    ]


def _bind_direct_schema(
    schema: dict[str, Any], response_model: type[BaseModel], payload: dict[str, Any]
) -> None:
    """Restrict direct-planner identifiers to the operations and families offered."""
    if response_model is QuestionDraftBatch:
        offered = [item["task_id"] for item in payload.get("allowed_tasks", [])]
        count = payload.get("draft_count")
        if not offered or not isinstance(count, int) or not 1 <= count <= 4:
            raise ExecutionError("MODEL_PAYLOAD_FIELD", "Drafting needs offered operations")
        schema["$defs"]["QuestionDraft"]["properties"]["task_id"]["enum"] = offered
        schema["properties"]["drafts"]["maxItems"] = count
    elif response_model is QuestionGateVote:
        known = [item["task_id"] for item in payload.get("task_definitions", [])]
        if not known:
            raise ExecutionError("MODEL_PAYLOAD_FIELD", "Question gate needs task definitions")
        branch = next(
            option
            for option in schema["properties"]["realized_task_id"]["anyOf"]
            if option.get("type") == "string"
        )
        branch["enum"] = known
    elif response_model is ImageProfile:
        families = [item["family"] for item in payload.get("family_definitions", [])]
        if not families:
            raise ExecutionError("MODEL_PAYLOAD_FIELD", "Image profile needs family definitions")
        schema["properties"]["supported_families"]["items"]["enum"] = families
        schema["properties"]["supported_families"]["maxItems"] = len(families)


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
    prompt_tokens: int | None
    completion_tokens: int | None


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

    def _timeout(self, max_tokens: int) -> httpx.Timeout:
        """Allow long structured outputs to finish instead of timing out and resending."""
        seconds = float(self.runtime.request_timeout_seconds)
        per_1k = self.runtime.timeout_seconds_per_1k_output_tokens
        if per_1k is not None:
            seconds = min(900.0, max(seconds, 60.0 + max_tokens * per_1k / 1000))
        return httpx.Timeout(seconds, connect=min(seconds, 30.0))

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
        trial_id: str | None = None,
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
            "max_model_len": self.endpoint.max_model_len,
            "gpu_memory_utilization": self.endpoint.gpu_memory_utilization,
            "serving_runtime": (
                self.endpoint.serving_runtime.model_dump(mode="json")
                if self.endpoint.serving_runtime is not None
                else None
            ),
        }
        request_envelope: dict[str, Any] = {"model_lock": model_lock, "request": body}
        if trial_id is not None:
            if not trial_id.strip():
                raise ExecutionError("MODEL_TRIAL_ID", "Trial identity cannot be empty")
            request_envelope["trial_id"] = trial_id
        request_hash = canonical_hash(request_envelope)
        model_lock_hash = canonical_hash(model_lock)
        if self.store is not None and not bypass_cache:
            saved = self.store.saved_model_attempt(model_lock_hash, stage, request_hash)
            if saved is not None:
                raw_content, prompt_tokens, completion_tokens, attempt_id = saved
                self.store.write_json_artifact(
                    "cache-accesses",
                    {
                        "stage": stage,
                        "request_hash": request_hash,
                        "attempt_id": attempt_id,
                        "accessed_at": time.time(),
                    },
                )
                try:
                    typed, response_hash = self._decode_typed_response(
                        raw_content,
                        response_model,
                        max_tokens=max_tokens,
                    )
                except ExecutionError:
                    self.store.finish_model_attempt(attempt_id, "INVALID")
                    raise
                self._record(
                    stage,
                    request_envelope,
                    raw_content,
                    request_hash,
                    response_hash,
                    prompt_tokens,
                    completion_tokens,
                    0,
                )
                self.store.finish_model_attempt(attempt_id, "COMPLETE")
                return ModelResponse(
                    typed, request_hash, response_hash, prompt_tokens, completion_tokens
                )
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
        request_artifact = (
            self._archive_request(request_envelope) if self.store is not None else None
        )
        raw_response, attempt_id, duration_ms = self._request(
            body,
            stage=stage,
            request_hash=request_hash,
            model_lock_hash=model_lock_hash,
            request_artifact=request_artifact,
            max_tokens=max_tokens,
        )
        raw_content = raw_response.content
        usage = measure_usage(raw_content, max_tokens)
        try:
            typed, response_hash = self._decode_typed_response(
                raw_content,
                response_model,
                max_tokens=max_tokens,
            )
        except ExecutionError:
            if self.store is not None:
                self._record(
                    stage,
                    request_envelope,
                    raw_content,
                    request_hash,
                    hashlib.sha256(raw_content).hexdigest(),
                    usage.prompt_tokens,
                    usage.completion_tokens,
                    duration_ms,
                    status="INVALID",
                )
                if attempt_id is not None:
                    self.store.finish_model_attempt(attempt_id, "INVALID")
            raise
        if self.store is not None:
            self._record(
                stage,
                request_envelope,
                raw_content,
                request_hash,
                response_hash,
                usage.prompt_tokens,
                usage.completion_tokens,
                duration_ms,
            )
            if attempt_id is not None:
                self.store.finish_model_attempt(attempt_id, "COMPLETE")
        return ModelResponse(
            value=typed,
            request_hash=request_hash,
            response_hash=response_hash,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
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
        choices = parsed.get("choices")
        choice = (
            choices[0]
            if isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0], dict)
            else {}
        )
        if choice.get("finish_reason") == "repetition":
            raise ExecutionError(
                "MODEL_OUTPUT_REPETITION", "Engine stopped a repeated token pattern"
            )
        message = choice.get("message")
        raw_content = message.get("content") if isinstance(message, dict) else None
        whitespace_runaway = (
            choice.get("finish_reason") == "length"
            and isinstance(raw_content, str)
            and len(raw_content) - len(raw_content.rstrip()) >= 512
        )
        try:
            content, usage = self._validate_completion(parsed)
        except ExecutionError as error:
            if whitespace_runaway and error.reason == "MODEL_FINISH_REASON":
                raise ExecutionError(
                    "MODEL_WHITESPACE_RUNAWAY", "Completion exhausted its limit in whitespace"
                ) from error
            raise
        if response_model is TextPayload and parsed["choices"][0]["finish_reason"] != "stop":
            raise ExecutionError(
                "MODEL_FINISH_REASON", "Public text must finish before its token limit"
            )
        if measure_usage(raw_response, max_tokens).status == "INVALID":
            raise ExecutionError("MODEL_USAGE_INVALID", "Token usage is outside request bounds")
        cleaned = self.adapter.clean_content(content)
        try:
            _model_json_object(cleaned)
        except ExecutionError as error:
            if whitespace_runaway and error.reason == "MODEL_SCHEMA_MISMATCH":
                raise ExecutionError(
                    "MODEL_WHITESPACE_RUNAWAY", "Completion exhausted its limit in whitespace"
                ) from error
            raise
        try:
            typed = response_model.model_validate_json(cleaned)
        except ValidationError as error:
            if parsed["choices"][0]["finish_reason"] == "length":
                if whitespace_runaway:
                    raise ExecutionError(
                        "MODEL_WHITESPACE_RUNAWAY",
                        "Completion exhausted its limit in whitespace",
                    ) from error
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
        _bind_direct_schema(schema, response_model, payload)
        if stage in SPECIALIST_SOURCE_STAGES:
            _bind_specialist_source_schema(schema, stage, payload)
        if stage in ANSWER_PARSE_FORMS:
            schema = _bind_answer_parse_schema(schema, stage, payload)
        if response_model is TextPayload:
            schema = _bind_text_payload_schema(schema)
        if stage in STRUCTURAL_OUTPUT_STAGES:
            if stage in {
                "chart_source",
                "graph_source",
                "table_source",
                "transcript_source",
                "extractive_source",
                "document_source",
            }:
                operation = payload.get("expected_operation")
                region = operation.get("scope_region") if isinstance(operation, dict) else None
                if stage in {"transcript_source", "extractive_source"} and isinstance(
                    operation, dict
                ):
                    region = operation.get("target_region") or region
                if region is not None:
                    try:
                        bound = ImageRegion.model_validate(region)
                    except ValidationError as error:
                        raise ExecutionError(
                            "MODEL_PAYLOAD_FIELD", "Source operation has an invalid public region"
                        ) from error
                    coordinates = schema["$defs"]["ImageRegion"]["properties"]
                    for field, minimum, maximum in (
                        ("left", bound.left, bound.right),
                        ("right", bound.left, bound.right),
                        ("top", bound.top, bound.bottom),
                        ("bottom", bound.top, bound.bottom),
                    ):
                        coordinates[field].update(minimum=minimum, maximum=maximum)
            if stage == "table_lookup_source":
                operation = payload.get("expected_operation")
                if not isinstance(operation, dict):
                    raise ExecutionError("MODEL_PAYLOAD_FIELD", "Lookup requires a bound operation")
                for field in ("scope_id", "view_id"):
                    value = operation.get(field)
                    if not isinstance(value, str) or not value:
                        raise ExecutionError("MODEL_PAYLOAD_FIELD", f"Lookup lacks {field}")
                    schema["properties"][field]["const"] = value
            if stage.endswith("_answer"):
                # Nullable result fields must be emitted, even when the parse abstains.
                # Their omission otherwise permits an acknowledgement-only JSON object.
                schema["required"] = sorted(schema["properties"])
            if stage == "document_source":
                properties = schema["properties"]
                base = {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(properties),
                }
                schema = {
                    "$defs": schema["$defs"],
                    "anyOf": [
                        {
                            **base,
                            "properties": {**properties, "coverage": {"const": "MET"}},
                        },
                        {
                            **base,
                            "properties": {
                                **properties,
                                "coverage": {"enum": ["UNKNOWN", "NOT_MET"]},
                                "closed": {"const": False},
                                **{
                                    name: {**properties[name], "maxItems": 0}
                                    for name in ("fields", "nodes")
                                },
                            },
                        },
                    ],
                }
            elif stage == "chart_source":
                operation = payload.get("expected_operation")
                if (
                    isinstance(operation, dict)
                    and operation.get("task_id") == "chart_extremum_ranking"
                ):
                    parameters = {
                        item["name"]: item["value"]
                        for item in operation.get("public_parameters", [])
                    }
                    for field in ("rank_mode", "rank_order"):
                        if field in parameters:
                            schema["$defs"]["ChartQuery"]["properties"][field]["const"] = (
                                parameters[field]
                            )
                            required = schema["$defs"]["ChartQuery"].setdefault("required", [])
                            if field not in required:
                                required.append(field)
                mark_schema = schema["$defs"]["ChartMark"]
                mark_properties = mark_schema["properties"]
                mark_schema["required"] = sorted(mark_properties)
                mark_properties["visible_label"]["description"] = (
                    "Copy the visible numeric value label for explicit_label precision, "
                    "including the literal lower/upper number. A category name is not "
                    "a numeric value label. Use null only for an interval estimate."
                )
                for endpoint in ("lower", "upper"):
                    mark_properties[endpoint]["description"] = (
                        "Bare finite decimal string. No inequality, unit, label or explanation."
                    )
                    mark_properties[endpoint]["pattern"] = r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
                schema["$defs"]["ChartAxis"]["properties"]["ticks"]["items"]["pattern"] = (
                    r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$"
                )
                _bind_chart_source_schema(schema, payload)
            elif stage in STRUCTURAL_ANSWER_FORMS:
                operation = payload.get("expected_operation")
                task_id = operation.get("task_id") if isinstance(operation, dict) else None
                forms = STRUCTURAL_ANSWER_FORMS[stage]
                allowed = forms.get(task_id)
                if allowed is not None:
                    result_fields = set().union(*forms.values())
                    for field in result_fields - allowed:
                        schema["properties"][field] = {"type": "null", "const": None}
            elif stage == "specialist_circuit_source":
                schema["properties"]["closed"]["description"] = (
                    "The visible component and terminal inventory is complete. "
                    "A fully read passive network with no power source is closed."
                )
            elif stage in {"transcript_source", "extractive_source", "table_lookup_source"}:
                # A nullable region otherwise permits a syntactically valid MET
                # response without a complete source. Bind the two coverage shapes
                # in the decoder, keeping visual truth with the independent reader.
                properties = schema["properties"]
                base = {
                    "type": "object",
                    "additionalProperties": False,
                    "required": sorted(properties),
                }
                if stage in {"transcript_source", "extractive_source"}:
                    complete_fields = {
                        "coverage": {"type": "string", "const": "MET"},
                        "expected_lines": {**properties["expected_lines"], "minItems": 1},
                        "source_region": {"$ref": "#/$defs/ImageRegion"},
                        "requested_unit_complete": {"type": "boolean", "const": True},
                    }
                    uncertain_fields = {
                        "coverage": {"type": "string", "enum": ["UNKNOWN", "NOT_MET"]},
                        "expected_lines": {**properties["expected_lines"], "maxItems": 0},
                        "source_region": {"type": "null"},
                        "requested_unit_complete": {"type": "boolean", "const": False},
                    }
                else:
                    complete_fields = {
                        "coverage": {"type": "string", "const": "MET"},
                        "layout": {"type": "string", "const": "simple_grid"},
                        "closed": {"type": "boolean", "const": True},
                        "table_region": {"$ref": "#/$defs/ImageRegion"},
                        "data_region": {"$ref": "#/$defs/ImageRegion"},
                        "cell": {"$ref": "#/$defs/LookupCell"},
                        "row_anchor": {"$ref": "#/$defs/HeaderAnchor"},
                        "col_anchor": {"$ref": "#/$defs/HeaderAnchor"},
                    }
                    uncertain_fields = {
                        "coverage": {"type": "string", "enum": ["UNKNOWN", "NOT_MET"]},
                        "layout": {"type": "string", "enum": ["requires_full_grid", "unreadable"]},
                        "closed": {"type": "boolean", "const": False},
                        "row_anchor": {"type": "null"},
                        "col_anchor": {"type": "null"},
                        "data_region": {"type": "null"},
                        "cell": {"type": "null"},
                        **{
                            name: {**properties[name], "maxItems": 0}
                            for name in ("row_headers", "col_headers")
                        },
                    }
                complete: dict[str, Any] = {
                    **base,
                    "properties": {**properties, **complete_fields},
                }
                uncertain: dict[str, Any] = {
                    **base,
                    "properties": {**properties, **uncertain_fields},
                }
                if stage in {"transcript_source", "extractive_source"}:
                    # Establish whole-unit readability before choosing a coverage
                    # branch; choosing MET first prevents a later abstention.
                    order = (
                        "reason",
                        "requested_unit_complete",
                        "expected_lines",
                        "source_region",
                        "coverage",
                    )
                    for branch in (complete, uncertain):
                        branch["properties"] = {name: branch["properties"][name] for name in order}
                schema = {"$defs": schema["$defs"], "anyOf": [complete, uncertain]}
            request_text["response_schema"] = schema
            user_content[0]["text"] = canonical_json(request_text).decode()
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
        detection = self.runtime.repetition_detection
        if detection is not None and detection.applies_to(stage):
            body["repetition_detection"] = detection.model_dump(exclude={"stages", "recovery"})
        body.update(self.adapter.extra_body())
        return body

    def _request(
        self,
        body: dict[str, Any],
        *,
        stage: str,
        request_hash: str,
        model_lock_hash: str,
        request_artifact: str | None,
        max_tokens: int,
    ) -> tuple[httpx.Response, str | None, int]:
        last_error: Exception | None = None
        for attempt in range(self.runtime.transport_max_attempts):
            attempt_id = None
            if self.store is not None:
                assert request_artifact is not None
                attempt_id = self.store.begin_model_attempt(
                    stage=stage,
                    model_repo=self.endpoint.repo_id,
                    model_revision=self.endpoint.revision,
                    model_lock_hash=model_lock_hash,
                    request_hash=request_hash,
                    request_artifact_hash=request_artifact,
                    reserved_output_tokens=max_tokens,
                    request_limit=self.runtime.max_total_requests,
                    output_token_limit=self.runtime.max_total_output_tokens,
                    transport_attempt=attempt + 1,
                )
            started = time.perf_counter()
            try:
                response = self.client.post(
                    "chat/completions", json=body, timeout=self._timeout(max_tokens)
                )
                duration_ms = round((time.perf_counter() - started) * 1000)
                if self.store is not None and attempt_id is not None:
                    self.store.save_model_attempt_response(
                        attempt_id,
                        response.content,
                        duration_ms,
                        response.status_code,
                    )
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
                    return response, attempt_id, duration_ms
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
                if self.store is not None and attempt_id is not None:
                    duration_ms = round((time.perf_counter() - started) * 1000)
                    self.store.finish_model_attempt(attempt_id, "TRANSPORT_FAILED", duration_ms)
                    self.store.write_json_artifact(
                        "transport-errors",
                        {
                            "request_hash": request_hash,
                            "attempt_id": attempt_id,
                            "model_repo": self.endpoint.repo_id,
                            "attempt": attempt + 1,
                            "error_type": type(error).__name__,
                            "reason": str(error),
                        },
                    )
                last_error = error
            except httpx.HTTPStatusError as error:
                if self.store is not None and attempt_id is not None:
                    self.store.finish_model_attempt(attempt_id, "HTTP_ERROR")
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
            try:
                _model_json_object(content)
            except ExecutionError as error:
                raise ExecutionError(
                    "MODEL_FINISH_REASON", "Completion ended before a complete JSON object"
                ) from error
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return content, usage

    def _record(
        self,
        stage: str,
        request: dict[str, Any],
        response: bytes,
        request_hash: str,
        response_hash: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        duration_ms: int,
        status: Literal["COMPLETE", "INVALID"] = "COMPLETE",
    ) -> None:
        assert self.store is not None
        request_artifact = self._archive_request(request)
        response_artifact = self.store.write_artifact("responses", response)
        model_lock_hash = canonical_hash(request["model_lock"])
        call_id = canonical_hash({"run": self.run_id, "stage": stage, "request": request_hash})
        usage = measure_usage(response, request["request"]["max_tokens"])
        with self.store.transaction() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO model_call(
                       call_id, stage, model_repo, model_revision, model_lock_hash,
                       request_hash, request_artifact_hash, response_artifact_hash,
                       input_tokens, output_tokens, duration_ms, status, usage_status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call_id,
                    stage,
                    self.endpoint.repo_id,
                    self.endpoint.revision,
                    model_lock_hash,
                    request_hash,
                    request_artifact,
                    response_artifact,
                    prompt_tokens if prompt_tokens is not None else 0,
                    completion_tokens if completion_tokens is not None else 0,
                    duration_ms,
                    status,
                    usage.status,
                ),
            )
        if response_artifact != response_hash:
            raise ExecutionError("MODEL_RESPONSE_HASH", "Stored response identity changed")

    def _archive_request(self, request: dict[str, Any]) -> str:
        """Keep image bytes deduplicated in private request artifacts."""
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
        return self.store.write_artifact("requests", canonical_json(archived))

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
