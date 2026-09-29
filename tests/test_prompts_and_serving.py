import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest
from pydantic import HttpUrl

from pixelogue.config import ModelEndpoint, RuntimeConfig, load_config
from pixelogue.contracts import ClaimExtraction, EvidenceInventory, RubricVerdict, TextPayload
from pixelogue.errors import ExecutionError
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
from pixelogue.specialist_music import MusicSource
from pixelogue.store import RunStore
from pixelogue.task_evidence import CandidateBindingsReport, ScopedEvidenceReport


def test_instruction_selector_information_boundary() -> None:
    allowed = {
        "target_language": "en",
        "public_history": [],
        "candidates": [],
        "image_views": [],
    }
    validate_stage_payload("instruction_selection", allowed)
    with pytest.raises(ExecutionError) as caught:
        validate_stage_payload("instruction_selection", {**allowed, "gold_answer": "secret"})
    assert caught.value.reason == "MODEL_INFORMATION_LEAK"
    with pytest.raises(ExecutionError):
        validate_stage_payload("rubric_item", {"other_judge_verdict": "MET"})


def test_text_payload_derives_status_from_public_text() -> None:
    assert TextPayload(text="Visible answer.", reason="Internal note.").status == "OK"
    assert TextPayload(text=None, reason="The request is unsupported.").status == "UNSUPPORTED"
    with pytest.raises(ValueError):
        TextPayload(text=None)


def test_model_adapters_disable_thinking_and_reject_reasoning_leak() -> None:
    assert ModelAdapter("Qwen/Qwen3.5-2B").extra_body() == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    gemma = ModelAdapter("google/gemma-4-31B-it-qat-w4a16-ct")
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


def test_candidate_binding_schema_restricts_supplied_names() -> None:
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="x")
    try:
        body = client._build_body(
            "candidate_binding",
            {
                "image_views": [],
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "required_check_ids": ["scope_resolved", "count_unit_defined"],
                        "bindable_parameter_names": ["target", "count_unit"],
                    }
                ],
            },
            (),
            CandidateBindingsReport,
            max_tokens=100,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    definitions = body["response_format"]["json_schema"]["schema"]["$defs"]
    assert definitions["CandidateBindingReport"]["properties"]["candidate_id"]["enum"] == [
        "candidate-1"
    ]
    assert definitions["EligibilityObservation"]["properties"]["check_id"]["enum"] == [
        "count_unit_defined",
        "scope_resolved",
    ]
    assert definitions["CandidateBindingReport"]["properties"]["checks"]["minItems"] == 2
    assert definitions["CandidateBindingReport"]["properties"]["checks"]["maxItems"] == 2
    assert definitions["PublicParameter"]["properties"]["name"]["enum"] == ["count_unit"]


def test_candidate_binding_schema_allows_no_optional_parameters() -> None:
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="x")
    try:
        body = client._build_body(
            "candidate_binding",
            {
                "image_views": [],
                "candidates": [
                    {
                        "candidate_id": "candidate-1",
                        "required_check_ids": ["scope_resolved"],
                        "bindable_parameter_names": ["target"],
                    }
                ],
            },
            (),
            CandidateBindingsReport,
            max_tokens=100,
            temperature=0.0,
            seed=1,
        )
    finally:
        client.client.close()
    definitions = body["response_format"]["json_schema"]["schema"]["$defs"]
    assert "enum" not in definitions["PublicParameter"]["properties"]["name"]
    assert definitions["CandidateBindingReport"]["properties"]["public_parameters"]["maxItems"] == 0


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


def test_length_completion_repairs_only_missing_json_containers() -> None:
    response = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": '{"verdict":"MET","reason":"Supported."'},
            }
        ],
        "usage": {"completion_tokens": 20},
    }
    content, _ = VllmClient._validate_completion(response)
    assert content == '{"verdict":"MET","reason":"Supported."}'

    response["choices"][0]["message"]["content"] = '{"verdict":"MET","reason":'
    with pytest.raises(ExecutionError) as caught:
        VllmClient._validate_completion(response)
    assert caught.value.reason == "MODEL_FINISH_REASON"


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
def test_server_disconnect_retries_and_releases_budget(tmp_path, monkeypatch, recovers):
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
        assert (
            store.connection.execute("SELECT reserved_output_tokens FROM budget").fetchone()[0] == 0
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


def test_coverage_requires_its_inventory_and_restricts_it_to_coverage() -> None:
    payload = {"criterion": {"template_id": "C_COVERAGE"}}
    with pytest.raises(ExecutionError, match="requires candidate_claim_inventory"):
        validate_stage_payload("rubric_item", payload)
    validate_stage_payload("rubric_item", {**payload, "candidate_claim_inventory": []})
    with pytest.raises(ExecutionError, match="restricted to C_COVERAGE"):
        validate_stage_payload(
            "rubric_item", {"criterion": {"template_id": "R_CORE"}, "candidate_claim_inventory": []}
        )


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
                client._request({"model": "test", "messages": []})
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


def test_claim_schema_bounds_follow_each_answer_without_mutating_contract():
    with httpx.Client(base_url="http://127.0.0.1:8000/v1/") as http:
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="bounds", client=http
        )
        for text in ("The cap says TITANS.", "2", ""):
            tokens = SynthesisCoordinator._answer_tokens(text)
            body = client._build_body(
                "claim_inventory",
                {"candidate_answer": text, "answer_tokens": tokens},
                (),
                ClaimExtraction,
                max_tokens=2048,
                temperature=0.0,
                seed=1,
            )
            schema = body["response_format"]["json_schema"]["schema"]
            bounds = schema["$defs"]["ClaimSpan"]["properties"]
            assert bounds["start_token"]["minimum"] == 0
            assert bounds["start_token"]["maximum"] == max(0, len(tokens) - 1)
            assert bounds["end_token"]["maximum"] == max(1, len(tokens))
            if not tokens:
                assert schema["properties"]["claims"]["maxItems"] == 0
        assert (
            "maximum"
            not in ClaimExtraction.model_json_schema()["$defs"]["ClaimSpan"]["properties"][
                "end_token"
            ]
        )
        with pytest.raises(ExecutionError, match="indexed answer_tokens"):
            client._build_body(
                "claim_inventory",
                {"answer_tokens": [{"index": 10}]},
                (),
                ClaimExtraction,
                max_tokens=2048,
                temperature=0.0,
                seed=1,
            )


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
                        "messages": [
                            {"role": "system", "content": "fixed"},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": str(number)},
                                    {"type": "image_url", "image_url": {"url": uri}},
                                ],
                            },
                        ]
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


def test_evidence_schema_fixes_only_the_requested_image_identity():
    with httpx.Client(base_url="http://127.0.0.1:8000/v1/") as http:
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
            RuntimeConfig(),
            run_id="evidence-identity",
            client=http,
        )
        for image_id in ("a" * 64, "b" * 64):
            body = client._build_body(
                "evidence_extraction",
                {"image_id": image_id},
                (),
                EvidenceInventory,
                max_tokens=1024,
                temperature=0.0,
                seed=1,
            )
            schema = body["response_format"]["json_schema"]["schema"]
            assert schema["properties"]["image_id"]["const"] == image_id
            assert "const" not in schema["properties"]["capabilities"]
        assert "const" not in EvidenceInventory.model_json_schema()["properties"]["image_id"]


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


def test_model_facing_evidence_and_binding_schemas_enforce_shape():
    evidence_schema = ScopedEvidenceReport.model_json_schema()
    scope_schema = evidence_schema["$defs"]["ScopeEvidenceReport"]
    assert scope_schema["properties"]["observations"]["type"] == "object"
    assert "propertyNames" not in scope_schema["properties"]["observations"]
    binding_schema = CandidateBindingsReport.model_json_schema()["$defs"]["CandidateBindingReport"]
    assert "target" in binding_schema["required"]
    assert "public_parameters" in binding_schema["required"]


def test_scoped_evidence_guidance_names_each_capability_once():
    with httpx.Client(base_url="http://127.0.0.1:8000/v1/") as http:
        client = VllmClient(
            ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
            RuntimeConfig(),
            run_id="evidence-capabilities",
            client=http,
        )
        body = client._build_body(
            "evidence_extraction",
            {
                "image_id": "a" * 64,
                "image_views": [],
                "capability_vocabulary": {
                    "visible_entity": "Visible entity",
                    "readable_text": "Text",
                },
                "max_scopes": 2,
                "max_observations_per_scope": 3,
            },
            (),
            ScopedEvidenceReport,
            max_tokens=1024,
            temperature=0.0,
            seed=1,
        )
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["scopes"]["maxItems"] == 2
    observations = schema["$defs"]["ScopeEvidenceReport"]["properties"]["observations"]
    assert list(observations["properties"]) == ["readable_text", "visible_entity"]
    assert observations["additionalProperties"] is False
    assert observations["maxProperties"] == 3
    assert (
        "properties"
        not in ScopedEvidenceReport.model_json_schema()["$defs"]["ScopeEvidenceReport"][
            "properties"
        ]["observations"]
    )


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
            client._request({"model": "test"})
    assert caught.value.reason == "MODEL_REQUEST_REJECTED"
    assert "propertyNames is not supported" in str(caught.value)


def test_repeated_capability_key_is_not_silently_overwritten():
    with pytest.raises(ExecutionError) as caught:
        _model_json_object(
            '{"observations":{"visible_entity":{"verdict":"MET"},'
            '"visible_entity":{"verdict":"UNKNOWN"}}}'
        )
    assert caught.value.reason == "MODEL_SCHEMA_MISMATCH"
