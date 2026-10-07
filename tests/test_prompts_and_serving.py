import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from pydantic import HttpUrl

from pixelogue.chart_verifiers import ChartAnswer, ChartSource
from pixelogue.config import ModelEndpoint, RuntimeConfig, load_config
from pixelogue.contracts import RubricVerdict, TextPayload
from pixelogue.document_verifiers import DocumentSource
from pixelogue.errors import ExecutionError
from pixelogue.graph_verifiers import GraphAnswer, GraphSource
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.profiling import profile_database
from pixelogue.prompts import STAGE_INSTRUCTIONS, validate_stage_payload
from pixelogue.serialization import canonical_hash
from pixelogue.serving import (
    ModelAdapter,
    ModelImage,
    VllmClient,
    _model_json_object,
    read_request_artifact,
)
from pixelogue.specialist_chemistry import ChemicalSource
from pixelogue.specialist_circuit import CircuitSource
from pixelogue.specialist_geometry import GeometryProblem
from pixelogue.specialist_music import MusicSource
from pixelogue.store import RunStore
from pixelogue.table_lookup import TableLookupSource
from pixelogue.table_verifiers import TableAnswer, TableSource
from pixelogue.task_evidence import (
    TranscriptSource,
)


@pytest.mark.parametrize(
    ("stage", "model"),
    [
        ("table_source", TableSource),
        ("table_lookup_source", TableLookupSource),
        ("transcript_source", TranscriptSource),
        ("extractive_source", TranscriptSource),
        ("table_answer", TableAnswer),
        ("document_source", DocumentSource),
        ("chart_source", ChartSource),
        ("chart_answer", ChartAnswer),
        ("graph_source", GraphSource),
        ("graph_answer", GraphAnswer),
    ],
)
def test_structural_reader_sees_the_exact_decoder_contract_without_mutating_models(
    tmp_path, stage, model
):
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="schema")
    original_schema = model.model_json_schema()
    source = stage.endswith("_source")
    payload = {
        "target_language": "en",
        "question": "Read the requested visible value.",
        "expected_operation": {"scope_id": "table", "view_id": "full:view"},
    }
    images = ()
    if source:
        path = tmp_path / "image.png"
        path.write_bytes(b"test image")
        image = ModelImage(
            view_id="full:view",
            path=path,
            encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            media_type="image/png",
        )
        payload.update(public_history=[], image_views=[{"view_id": image.view_id}])
        images = (image,)
    else:
        payload["candidate_answer"] = "Unambiguous reported value"
    try:
        validate_stage_payload(stage, payload)
        body = client._build_body(
            stage, payload, images, model, max_tokens=2048, temperature=0.0, seed=1
        )
    finally:
        client.client.close()
    text = json.loads(body["messages"][1]["content"][0]["text"])
    decoder_schema = body["response_format"]["json_schema"]["schema"]
    assert text["response_schema"] == decoder_schema
    assert text["input"] == payload
    assert model.model_json_schema() == original_schema
    if source:
        assert "candidate_answer" not in text["input"]
        with pytest.raises(ExecutionError, match="candidate_answer"):
            validate_stage_payload(stage, {**payload, "candidate_answer": "hidden answer"})
        with pytest.raises(ExecutionError, match="Answer-independent"):
            validate_stage_payload(
                stage, {**payload, "expected_operation": {"candidate_answer": "hidden answer"}}
            )
        if stage == "chart_source":
            mark = decoder_schema["$defs"]["ChartMark"]
            assert "visible_label" in mark["required"]
            assert "numeric value label" in mark["properties"]["visible_label"]["description"]
    else:
        assert set(decoder_schema["required"]) == set(decoder_schema["properties"])
        assert [part["type"] for part in body["messages"][1]["content"]] == ["text"]


def test_chart_rank_decoder_is_bound_to_the_public_objective():
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="rank")
    operation = {
        "task_id": "chart_extremum_ranking",
        "public_parameters": [
            {"name": "rank_mode", "value": "max_min"},
            {"name": "rank_order", "value": "descending"},
        ],
    }
    try:
        body = client._build_body(
            "chart_source",
            {"expected_operation": operation},
            (),
            ChartSource,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
        answer = client._build_body(
            "chart_answer",
            {
                "expected_operation": operation,
                "candidate_answer": "The upper-right diamond, approximately 35.6",
            },
            (),
            ChartAnswer,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    schema = body["response_format"]["json_schema"]["schema"]
    query = schema["$defs"]["ChartQuery"]
    assert query["properties"]["rank_mode"]["const"] == "max_min"
    assert query["properties"]["rank_order"]["const"] == "descending"
    assert {"rank_mode", "rank_order"} <= set(query["required"])
    answer_schema = answer["response_format"]["json_schema"]["schema"]
    assert answer_schema["properties"]["value"] == {"type": "null", "const": None}
    assert "rank_groups" in answer_schema["required"]


@pytest.mark.parametrize(
    "task_id,operation,series_count,category_count",
    [
        ("chart_value_lookup", "value", 1, 1),
        ("chart_comparison", "compare", 2, 1),
        ("chart_extremum_ranking", "rank", 1, None),
        ("chart_trend_summary", "trend", 1, None),
        ("chart_series_relation", "relation", 2, None),
        ("chart_data_reconstruction", "reconstruct", None, None),
    ],
)
def test_chart_decoder_requires_operands_only_for_a_complete_reading(
    task_id, operation, series_count, category_count
):
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="chart")
    try:
        body = client._build_body(
            "chart_source",
            {"expected_operation": {"task_id": task_id}},
            (),
            ChartSource,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    schema = body["response_format"]["json_schema"]["schema"]
    query = schema["$defs"]["ChartQuery"]
    assert {"series", "categories"} <= set(query["required"])
    assert query["properties"]["operation"]["const"] == operation
    complete, uncertain = schema["anyOf"]
    assert complete["properties"]["coverage"] == {"const": "MET"}
    assert uncertain["properties"]["coverage"] == {"enum": ["UNKNOWN", "NOT_MET"]}
    assert uncertain["properties"]["axis"] == {"type": "null"}
    assert uncertain["properties"]["marks"]["maxItems"] == 0
    assert set(complete["properties"]) == set(uncertain["properties"]) == set(schema["properties"])
    if operation != "reconstruct":
        for field, count in (("series", series_count), ("categories", category_count)):
            operand = complete["properties"]["query"]["properties"][field]
            assert operand["minItems"] == (count or 1)
            assert operand.get("maxItems") == count
    unmarked, calibrated = schema["$defs"]["ChartAxis"]["anyOf"]
    assert (
        set(unmarked["properties"]) == set(calibrated["properties"]) == {"scale", "unit", "ticks"}
    )
    assert unmarked["properties"]["scale"] == {"type": "string", "const": "unmarked"}
    assert unmarked["properties"]["ticks"]["maxItems"] == 0
    assert calibrated["properties"]["scale"] == {"type": "string", "enum": ["linear", "log"]}
    assert calibrated["properties"]["ticks"]["minItems"] == 2


@pytest.mark.parametrize(
    "stage,model",
    [
        ("chart_source", ChartSource),
        ("graph_source", GraphSource),
        ("table_source", TableSource),
        ("document_source", DocumentSource),
        ("transcript_source", TranscriptSource),
        ("extractive_source", TranscriptSource),
    ],
)
def test_source_decoder_limits_coordinates_to_the_declared_public_region(stage, model):
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="bounds")
    scope = {"left": 0.1, "top": 0.2, "right": 0.9, "bottom": 0.8}
    target = {"left": 0.3, "top": 0.4, "right": 0.7, "bottom": 0.6}
    try:
        body = client._build_body(
            stage,
            {"expected_operation": {"scope_region": scope, "target_region": target}},
            (),
            model,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    fields = body["response_format"]["json_schema"]["schema"]["$defs"]["ImageRegion"]["properties"]
    bound = target if stage in {"transcript_source", "extractive_source"} else scope
    for field, lower, upper in (
        ("left", "left", "right"),
        ("right", "left", "right"),
        ("top", "top", "bottom"),
        ("bottom", "top", "bottom"),
    ):
        assert fields[field]["minimum"] == bound[lower]
        assert fields[field]["maximum"] == bound[upper]


def test_incomplete_document_decoder_cannot_certify_partial_source_facts():
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="doc")
    try:
        body = client._build_body(
            "document_source",
            {"expected_operation": {"task_id": "text_field_extraction"}},
            (),
            DocumentSource,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    schema = body["response_format"]["json_schema"]["schema"]
    complete, uncertain = schema["anyOf"]
    assert set(complete["properties"]) == set(uncertain["properties"])
    assert set(complete["required"]) == set(complete["properties"])
    assert complete["properties"]["coverage"] == {"const": "MET"}
    assert uncertain["properties"]["coverage"] == {"enum": ["UNKNOWN", "NOT_MET"]}
    assert uncertain["properties"]["closed"] == {"const": False}
    assert uncertain["properties"]["fields"]["maxItems"] == 0
    assert uncertain["properties"]["nodes"]["maxItems"] == 0


@pytest.mark.parametrize(
    ("stage", "model", "task_id", "allowed"),
    [
        ("table_answer", TableAnswer, "table_cell_lookup", {"value"}),
        ("table_answer", TableAnswer, "table_row_selection", {"rows"}),
        ("table_answer", TableAnswer, "table_join", {"pairs"}),
        ("table_answer", TableAnswer, "unknown_operation", None),
        ("graph_answer", GraphAnswer, "diagram_connectivity", {"members", "edges"}),
        ("graph_answer", GraphAnswer, "diagram_path_tracing", {"paths"}),
        ("graph_answer", GraphAnswer, "diagram_process_description", {"edges"}),
        ("graph_answer", GraphAnswer, "flowchart_evaluation", {"paths"}),
        ("graph_answer", GraphAnswer, "unknown_operation", None),
    ],
)
def test_parser_schema_limits_forms_using_only_the_public_operation(stage, model, task_id, allowed):
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="forms")
    original = model.model_json_schema()
    try:
        body = client._build_body(
            stage,
            {"expected_operation": {"task_id": task_id}, "candidate_answer": "Wrong value"},
            (),
            model,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    schema = body["response_format"]["json_schema"]["schema"]
    text = json.loads(body["messages"][1]["content"][0]["text"])
    assert text["response_schema"] == schema
    assert schema["properties"]["coverage"] == original["properties"]["coverage"]
    fields = ("value", "rows", "pairs") if model is TableAnswer else ("members", "edges", "paths")
    for field in fields:
        if allowed is None or field in allowed:
            assert schema["properties"][field] == original["properties"][field]
        else:
            assert schema["properties"][field] == {"type": "null", "const": None}
    assert model.model_json_schema() == original


@pytest.mark.parametrize(
    ("stage", "model"),
    [
        ("specialist_geometry_source", GeometryProblem),
        ("specialist_circuit_source", CircuitSource),
    ],
)
def test_specialist_structural_reader_sees_the_bound_schema_and_no_answer(tmp_path, stage, model):
    path = tmp_path / "image.png"
    path.write_bytes(b"test image")
    image = ModelImage(
        view_id="full:view",
        path=path,
        encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_type="image/png",
    )
    payload = {
        "image_views": [{"view_id": image.view_id}],
        "expected_operation": {"calibrated_domain": "public-domain", "scope_id": "public-scope"},
    }
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="schema")
    try:
        body = client._build_body(
            stage, payload, (image,), model, max_tokens=4096, temperature=0.0, seed=1
        )
    finally:
        client.client.close()
    text = json.loads(body["messages"][1]["content"][0]["text"])
    schema = body["response_format"]["json_schema"]["schema"]
    assert text["response_schema"] == schema
    assert text["input"] == payload
    assert schema["properties"]["domain"]["const"] == "public-domain"
    assert "candidate_answer" not in text["input"]
    assert "const" not in model.model_json_schema()["properties"]["domain"]
    if stage == "specialist_circuit_source":
        assert "inventory" in schema["properties"]["closed"]["description"]


def test_router_and_drafter_information_boundary() -> None:
    profile = {"image_views": [], "family_definitions": []}
    validate_stage_payload("image_profile", profile)
    with pytest.raises(ExecutionError) as caught:
        validate_stage_payload("image_profile", {**profile, "gold_answer": "secret"})
    assert caught.value.reason == "MODEL_INFORMATION_LEAK"
    draft = {
        "target_language": "en",
        "turn_index": 1,
        "public_history": [],
        "allowed_tasks": [],
        "preferred_task_ids": [],
        "family_plan": {},
        "excluded_fact_keys": [],
        "draft_count": 2,
        "image_views": [],
    }
    validate_stage_payload("question_draft", draft)
    with pytest.raises(ExecutionError) as caught:
        validate_stage_payload("question_draft", {**draft, "candidate_answer": "white"})
    assert caught.value.reason == "MODEL_INFORMATION_LEAK"
    with pytest.raises(ExecutionError):
        validate_stage_payload("holistic_review", {"other_judge_verdict": "MET"})


def test_text_payload_derives_status_from_public_text() -> None:
    assert TextPayload(text="Visible answer.", reason="Internal note.").status == "OK"
    assert TextPayload(text=None, reason="The request is unsupported.").status == "UNSUPPORTED"
    with pytest.raises(ValueError):
        TextPayload(text=None)


def test_model_adapters_disable_thinking_and_reject_reasoning_leak() -> None:
    assert ModelAdapter("Qwen/Qwen3.5-2B").extra_body() == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    gemma = ModelAdapter("google/gemma-4-31B-it")
    assert gemma.extra_body() == {"reasoning_effort": "none"}
    assert gemma.clean_content('<|channel|>thought\n<|channel|>{"verdict":"MET"}') == (
        '{"verdict":"MET"}'
    )
    with pytest.raises(ExecutionError) as caught:
        gemma.clean_content("prefix <|channel|>thought hidden")
    assert caught.value.reason == "MODEL_REASONING_LEAK"


def test_external_inference_endpoint_is_rejected() -> None:
    endpoint = ModelEndpoint(
        repo_id="Qwen/Qwen3.5-2B",
        base_url=HttpUrl("https://example.com/v1"),
    )
    with pytest.raises(ExecutionError) as caught:
        VllmClient(endpoint, RuntimeConfig(), run_id="test")
    assert caught.value.reason == "EXTERNAL_INFERENCE_FORBIDDEN"


def test_image_part_declaration_must_match() -> None:
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="x")
    with pytest.raises(ExecutionError) as caught:
        client._build_body(
            "rubric_item",
            {
                "target_language": "en",
                "public_history": [],
                "question": "q",
                "candidate_answer": "a",
                "image_views": [{"view_id": "missing", "encoded_sha256": "0" * 64}],
                "criterion": {},
            },
            (),
            RubricVerdict,
            max_tokens=10,
            temperature=0,
            seed=1,
        )
    assert caught.value.reason == "IMAGE_PART_MISMATCH"


def _rubric_payload() -> dict[str, object]:
    return {
        "target_language": "en",
        "public_history": [],
        "question": "q",
        "candidate_answer": "a",
        "image_views": [],
        "criterion": {},
    }


def test_identical_complete_model_call_replays_saved_response(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"verdict":"MET","reason":"Supported."}'},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="http://127.0.0.1:8000/v1/", transport=transport)
    with RunStore(tmp_path, "cache", require_local_wal=False) as store:
        store.initialize_run("cache", "a" * 64, "pilot")
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
            RuntimeConfig(),
            run_id="cache",
            store=store,
            client=http_client,
        )
        first = client.invoke(
            "rubric_item",
            _rubric_payload(),
            (),
            RubricVerdict,
            max_tokens=32,
            temperature=0.0,
            seed=1,
        )
        second = client.invoke(
            "rubric_item",
            _rubric_payload(),
            (),
            RubricVerdict,
            max_tokens=32,
            temperature=0.0,
            seed=1,
        )
        budget = store.connection.execute(
            "SELECT request_count, output_tokens FROM budget WHERE singleton=1"
        ).fetchone()

    assert first == second
    assert calls == 1
    assert tuple(budget) == (1, 3)


def test_history_model_and_prompt_changes_do_not_reuse_saved_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"verdict":"MET","reason":"Supported."}'},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
            request=request,
        )

    http_client = httpx.Client(
        base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(handler)
    )
    with RunStore(tmp_path, "changed", require_local_wal=False) as store:
        store.initialize_run("changed", "a" * 64, "pilot")
        first_model = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B", revision="a" * 40),
            RuntimeConfig(),
            run_id="changed",
            store=store,
            client=http_client,
        )
        history = _rubric_payload()
        changed_history = {
            **history,
            "public_history": [{"message_id": "m1", "role": "user", "content": "Earlier?"}],
        }
        for payload in (history, history, changed_history):
            first_model.invoke(
                "rubric_item",
                payload,
                (),
                RubricVerdict,
                max_tokens=32,
                temperature=0.0,
                seed=1,
            )
        second_model = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B", revision="b" * 40),
            RuntimeConfig(),
            run_id="changed",
            store=store,
            client=http_client,
        )
        second_model.invoke(
            "rubric_item",
            history,
            (),
            RubricVerdict,
            max_tokens=32,
            temperature=0.0,
            seed=1,
        )
        monkeypatch.setitem(
            STAGE_INSTRUCTIONS, "rubric_item", STAGE_INSTRUCTIONS["rubric_item"] + " Updated."
        )
        first_model.invoke(
            "rubric_item",
            history,
            (),
            RubricVerdict,
            max_tokens=32,
            temperature=0.0,
            seed=1,
        )
    assert calls == 4


@pytest.mark.parametrize(
    "content",
    [
        "{}",
        '{"verdict":"MET","reason":"unescaped\nnewline"}',
        '{"verdict":"MET","verdict":"NOT_MET","reason":"duplicate"}',
        '{"verdict":"MET","reason":NaN}',
    ],
)
def test_invalid_model_output_releases_output_reservation(tmp_path: Path, content: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="http://127.0.0.1:8000/v1/", transport=transport)
    with RunStore(tmp_path, "failure", require_local_wal=False) as store:
        store.initialize_run("failure", "a" * 64, "pilot")
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
            RuntimeConfig(),
            run_id="failure",
            store=store,
            client=http_client,
        )
        with pytest.raises(ExecutionError) as caught:
            client.invoke(
                "rubric_item",
                _rubric_payload(),
                (),
                RubricVerdict,
                max_tokens=32,
                temperature=0.0,
                seed=1,
            )
        assert caught.value.reason == "MODEL_SCHEMA_MISMATCH"
        budget = store.connection.execute(
            "SELECT request_count, reserved_output_tokens FROM budget WHERE singleton=1"
        ).fetchone()
        failed_call = store.connection.execute(
            "SELECT status, response_artifact_hash FROM model_call"
        ).fetchone()

    assert tuple(budget) == (1, 0)
    assert failed_call["status"] == "INVALID"
    assert failed_call["response_artifact_hash"] is not None


def test_concurrent_model_calls_are_recorded_with_duration(tmp_path: Path) -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"verdict":"MET","reason":"Supported."}'},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
            request=request,
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(base_url="http://127.0.0.1:8000/v1/", transport=transport)
    with RunStore(tmp_path, "concurrent", require_local_wal=False) as store:
        store.initialize_run("concurrent", "a" * 64, "pilot")
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
            RuntimeConfig(),
            run_id="concurrent",
            store=store,
            client=http_client,
        )
        payloads = [{**_rubric_payload(), "question": f"question-{index}"} for index in range(4)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = tuple(
                executor.map(
                    lambda payload: client.invoke(
                        "rubric_item",
                        payload,
                        (),
                        RubricVerdict,
                        max_tokens=32,
                        temperature=0.0,
                        seed=1,
                    ),
                    payloads,
                )
            )
        rows = store.connection.execute(
            "SELECT duration_ms FROM model_call ORDER BY call_id"
        ).fetchall()
        budget = store.connection.execute(
            "SELECT request_count, output_tokens, reserved_output_tokens FROM budget WHERE singleton=1"
        ).fetchone()
        profile = profile_database(store.database_path)

    assert len(responses) == 4
    assert peak == 4
    assert len(rows) == 4 and all(row["duration_ms"] >= 10 for row in rows)
    assert tuple(budget) == (4, 12, 0)
    assert profile.measurement == "recorded_duration"
    assert profile.completed_calls == 4
    assert profile.stages[0].stage == "rubric_item"


@pytest.mark.parametrize(
    "content",
    [
        '{"verdict":"MET","reason":"Supported."',
        '{"coverage":"MET","marks":[{"value":"5"}',
        '{"verdict":"MET","reason":',
    ],
)
def test_length_completion_never_repairs_missing_json_containers(content) -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": content},
            }
        ],
        "usage": {"completion_tokens": 20},
    }
    with pytest.raises(ExecutionError) as caught:
        VllmClient._validate_completion(response)
    assert caught.value.reason == "MODEL_FINISH_REASON"


def test_length_completion_missing_required_field_is_incomplete_not_schema_error() -> None:
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="x")
    response = {
        "choices": [{"finish_reason": "length", "message": {"content": '{"verdict":"MET"'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 20},
    }
    try:
        with pytest.raises(ExecutionError) as caught:
            client._decode_typed_response(
                json.dumps(response).encode(), RubricVerdict, max_tokens=20
            )
        assert caught.value.reason == "MODEL_FINISH_REASON"
        response["choices"][0]["finish_reason"] = "stop"
        with pytest.raises(ExecutionError) as caught:
            client._decode_typed_response(
                json.dumps(response).encode(), RubricVerdict, max_tokens=20
            )
        assert caught.value.reason == "MODEL_SCHEMA_MISMATCH"
    finally:
        client.client.close()


def test_length_completion_with_runaway_whitespace_is_distinct_from_valid_json() -> None:
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="x")
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": '{"verdict":"MET"}' + " " * 512},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1000},
    }
    try:
        with pytest.raises(ExecutionError) as caught:
            client._decode_typed_response(
                json.dumps(response).encode(), RubricVerdict, max_tokens=1000
            )
        assert caught.value.reason == "MODEL_WHITESPACE_RUNAWAY"

        response["choices"][0]["message"]["content"] = (
            '{"verdict":"MET","reason":"Supported."}' + " " * 512
        )
        result, _ = client._decode_typed_response(
            json.dumps(response).encode(), RubricVerdict, max_tokens=1000
        )
        assert result.verdict == "MET"
    finally:
        client.client.close()


@pytest.mark.parametrize("recovers", [True, False])
def test_malformed_json_uses_bounded_structured_retries(tmp_path, recovers):
    config = load_config(Path("configs/pilot.yaml"))
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        content = (
            '{"verdict":"MET","reason":"Supported."}'
            if recovers and calls > 1
            else '{"verdict":"MET","reason":"invalid\nnewline"}'
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with RunStore(tmp_path, "json-retry", require_local_wal=False) as store:
        store.initialize_run("json-retry", config.config_hash, config.profile)
        with httpx.Client(
            base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(handler)
        ) as http:
            client = VllmClient(
                config.models.generator_a,
                config.runtime,
                run_id="json-retry",
                store=store,
                client=http,
            )
            coordinator = SynthesisCoordinator(config, "json-retry", store, client, client, client)
            if recovers:
                result = coordinator._invoke(
                    client,
                    "rubric_item",
                    _rubric_payload(),
                    (),
                    RubricVerdict,
                    max_tokens=32,
                    temperature=0.0,
                    seed=1,
                )
                assert result.verdict == "MET"
            else:
                with pytest.raises(ExecutionError) as caught:
                    coordinator._invoke(
                        client,
                        "rubric_item",
                        _rubric_payload(),
                        (),
                        RubricVerdict,
                        max_tokens=32,
                        temperature=0.0,
                        seed=1,
                    )
                assert caught.value.reason == "MODEL_SCHEMA_MISMATCH"
        assert calls == config.runtime.structured_output_max_attempts == 2
        assert (
            store.connection.execute("SELECT reserved_output_tokens FROM budget").fetchone()[0] == 0
        )
        assert store.connection.execute(
            "SELECT COUNT(*) FROM model_call WHERE status='INVALID'"
        ).fetchone()[0] == (1 if recovers else 2)


@pytest.mark.parametrize("recovers", [True, False])
def test_server_disconnect_retries_and_keeps_unmeasured_cost_reserved(
    tmp_path, monkeypatch, recovers
):
    calls = 0
    delays = []
    monkeypatch.setattr("pixelogue.serving.time.sleep", delays.append)

    def handler(request):
        nonlocal calls
        calls += 1
        if not recovers or calls < 3:
            raise httpx.RemoteProtocolError(
                "Server disconnected without sending a response.", request=request
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"verdict":"MET","reason":"Supported."}'},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    with RunStore(tmp_path, "disconnect", require_local_wal=False) as store:
        store.initialize_run("disconnect", "a" * 64, "pilot")
        with httpx.Client(
            base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(handler)
        ) as http:
            client = VllmClient(
                ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
                RuntimeConfig(),
                run_id="disconnect",
                store=store,
                client=http,
            )
            if recovers:
                response = client.invoke(
                    "rubric_item",
                    _rubric_payload(),
                    (),
                    RubricVerdict,
                    max_tokens=32,
                    temperature=0.0,
                    seed=1,
                )
                assert isinstance(response.value, RubricVerdict)
                assert response.value.verdict == "MET"
            else:
                with pytest.raises(ExecutionError) as caught:
                    client.invoke(
                        "rubric_item",
                        _rubric_payload(),
                        (),
                        RubricVerdict,
                        max_tokens=32,
                        temperature=0.0,
                        seed=1,
                    )
                assert caught.value.reason == "MODEL_TRANSPORT_FAILED"
                assert isinstance(caught.value.__cause__, httpx.RemoteProtocolError)
        assert calls == 3
        assert delays == [1, 2]
        budget = store.connection.execute("SELECT * FROM budget").fetchone()
        assert budget["request_count"] == 3
        assert budget["reserved_output_tokens"] == (64 if recovers else 96)
        assert budget["output_tokens"] == (1 if recovers else 0)
        unknown = store.connection.execute(
            "SELECT output_tokens, usage_status FROM model_call_attempt WHERE status='TRANSPORT_FAILED'"
        ).fetchall()
        assert len(unknown) == (2 if recovers else 3)
        assert all(
            row["output_tokens"] is None and row["usage_status"] == "MISSING" for row in unknown
        )


def test_persistent_disconnect_is_confined_to_images(tmp_path, monkeypatch, image_artifact):
    image, root = image_artifact
    monkeypatch.setattr("pixelogue.serving.time.sleep", lambda delay: None)
    config = load_config(Path("configs/pilot.yaml"))
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    with RunStore(tmp_path / "runs", "disconnect", require_local_wal=False) as store:
        store.initialize_run("disconnect", config.config_hash, config.profile)
        with httpx.Client(
            base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(handler)
        ) as http:
            client = VllmClient(
                config.models.generator_a,
                config.runtime,
                run_id="disconnect",
                store=store,
                client=http,
            )
            coordinator = SynthesisCoordinator(config, "disconnect", store, client, client, client)
            jobs = [
                SynthesisJob(
                    image=image.model_copy(update={"image_id": f"{i:064x}"}),
                    target_language="en",
                    generator_role="generator_a",
                )
                for i in range(2)
            ]
            results = list(coordinator.synthesize_batch(jobs, root, max_workers=1))
        assert [result.status for result in results] == ["ERROR", "ERROR"]
        assert calls == 6


def test_http_500_records_bounded_response_body_without_headers(tmp_path, monkeypatch):
    monkeypatch.setattr("pixelogue.serving.time.sleep", lambda delay: None)
    with RunStore(tmp_path, "http500", require_local_wal=False) as store:
        with httpx.Client(
            base_url="http://127.0.0.1:8000/v1/",
            transport=httpx.MockTransport(
                lambda request: httpx.Response(500, text="backend error: " + "x" * 5000)
            ),
        ) as http:
            client = VllmClient(
                ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
                RuntimeConfig(),
                run_id="http500",
                store=store,
                client=http,
            )
            with pytest.raises(ExecutionError, match="500"):
                client.invoke(
                    "rubric_item",
                    _rubric_payload(),
                    (),
                    RubricVerdict,
                    max_tokens=32,
                    temperature=0.0,
                    seed=1,
                )
        artifacts = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind='transport-errors'"
        ).fetchall()
        assert len(artifacts) == 3
        for (artifact_hash,) in artifacts:
            value = json.loads(store.read_artifact(artifact_hash))
            assert value["status_code"] == 500 and value["truncated"]
            assert value["response_body"].startswith("backend error:")
            assert len(value["response_body"]) == 4096
            assert "headers" not in value


def test_request_images_are_deduplicated_and_restore_exact_envelope(tmp_path):
    with RunStore(tmp_path, "dedup", require_local_wal=False) as store:
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="dedup", store=store
        )
        uri = "data:image/png;base64," + "A" * 100000
        envelopes = []
        try:
            for number in range(2):
                envelope = {
                    "model_lock": {"repo": "test"},
                    "request": {
                        "max_tokens": 32,
                        "messages": [
                            {"role": "system", "content": "fixed"},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": str(number)},
                                    {"type": "image_url", "image_url": {"url": uri}},
                                ],
                            },
                        ],
                    },
                }
                envelopes.append(envelope)
                request_hash = canonical_hash(envelope)
                response = b'{"answer":"ok"}'
                client._record(
                    "holistic_review",
                    envelope,
                    response,
                    request_hash,
                    hashlib.sha256(response).hexdigest(),
                    1,
                    1,
                    1,
                )
                assert envelope["request"]["messages"][1]["content"][1]["image_url"]["url"] == uri
            assert (
                store.connection.execute(
                    "SELECT count(*) FROM artifact WHERE kind='request-images'"
                ).fetchone()[0]
                == 1
            )
            rows = store.connection.execute(
                "SELECT request_artifact_hash, request_hash FROM model_call ORDER BY rowid"
            ).fetchall()
            for row, envelope in zip(rows, envelopes, strict=True):
                assert len(store.read_artifact(row[0])) < 1000
                restored = read_request_artifact(store, row[0])
                assert restored == envelope
                assert canonical_hash(restored) == row[1]
            legacy = store.write_json_artifact("requests", envelopes[0])
            assert read_request_artifact(store, legacy) == envelopes[0]
            assert store.verify()["model_calls"] == 2
        finally:
            client.client.close()


def test_specialist_source_schema_fixes_public_ids_and_music_bar_range(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"test image")
    image = ModelImage(
        view_id="full:view",
        path=path,
        encoded_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        media_type="image/png",
    )
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="fixed")
    try:
        for stage, model in (
            ("specialist_chemistry_source", ChemicalSource),
            ("specialist_music_source", MusicSource),
        ):
            payload = {
                "image_views": [{"view_id": image.view_id}],
                "expected_operation": {
                    "calibrated_domain": "bounded-domain",
                    "scope_id": "whole-image",
                    "public_parameters": [{"name": "bar_range", "value": "1-1"}],
                },
            }
            body = client._build_body(
                stage, payload, (image,), model, max_tokens=4096, temperature=0.0, seed=1
            )
            properties = body["response_format"]["json_schema"]["schema"]["properties"]
            assert {
                name: properties[name]["const"] for name in ("domain", "scope_id", "view_id")
            } == {
                "domain": "bounded-domain",
                "scope_id": "whole-image",
                "view_id": image.view_id,
            }
            if model is MusicSource:
                assert properties["bar_range"]["const"] == "1-1"
            assert "const" not in model.model_json_schema()["properties"]["domain"]
        with pytest.raises(ExecutionError) as error:
            client._build_body(
                "specialist_chemistry_source",
                {"image_views": [{"view_id": image.view_id}]},
                (image,),
                ChemicalSource,
                max_tokens=4096,
                temperature=0.0,
                seed=1,
            )
        assert error.value.reason == "MODEL_PAYLOAD_FIELD"
        fallback = client._build_body(
            "specialist_chemistry_source",
            payload,
            (image,),
            ChemicalSource,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
            retry_feedback="The previous chemical graph was truncated; output the full source graph.",
            json_object_fallback=True,
        )
        assert fallback["response_format"] == {"type": "json_object"}
        assert "candidate_answer" not in fallback["messages"][1]["content"][0]["text"]
        for stage, feedback in (
            ("specialist_chemistry_source", None),
            ("specialist_music_source", "correct output"),
        ):
            with pytest.raises(ExecutionError) as fallback_error:
                client._build_body(
                    stage,
                    payload,
                    (image,),
                    ChemicalSource,
                    max_tokens=4096,
                    temperature=0.0,
                    seed=1,
                    retry_feedback=feedback,
                    json_object_fallback=True,
                )
            assert fallback_error.value.reason == "MODEL_OUTPUT_FORMAT"
    finally:
        client.client.close()


def test_chemical_json_object_fallback_cannot_accept_an_answer_instead_of_a_graph():
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="x")
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"CCO"}'}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 5},
    }
    try:
        with pytest.raises(ExecutionError) as error:
            client._decode_typed_response(
                json.dumps(response).encode(), ChemicalSource, max_tokens=1024
            )
        assert error.value.reason == "MODEL_SCHEMA_MISMATCH"
    finally:
        client.client.close()


def test_bad_request_keeps_server_schema_reason():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            400,
            json={"error": {"message": "propertyNames is not supported"}},
            request=request,
        )
    )
    with httpx.Client(base_url="http://127.0.0.1:8000/v1/", transport=transport) as http:
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
            RuntimeConfig(),
            run_id="diagnostic",
            client=http,
        )
        with pytest.raises(ExecutionError) as caught:
            client.invoke(
                "rubric_item",
                _rubric_payload(),
                (),
                RubricVerdict,
                max_tokens=32,
                temperature=0.0,
                seed=1,
            )
    assert caught.value.reason == "MODEL_REQUEST_REJECTED"
    assert "propertyNames is not supported" in str(caught.value)


def test_repeated_capability_key_is_not_silently_overwritten():
    with pytest.raises(ExecutionError) as caught:
        _model_json_object(
            '{"observations":{"visible_entity":{"verdict":"MET"},'
            '"visible_entity":{"verdict":"UNKNOWN"}}}'
        )
    assert caught.value.reason == "MODEL_SCHEMA_MISMATCH"
