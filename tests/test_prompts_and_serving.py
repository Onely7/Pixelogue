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
from pixelogue.contracts import ClaimExtraction, RubricVerdict, TextPayload
from pixelogue.errors import ExecutionError
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.profiling import profile_database
from pixelogue.prompts import validate_stage_payload
from pixelogue.serialization import canonical_hash
from pixelogue.serving import ModelAdapter, VllmClient, read_request_artifact
from pixelogue.store import RunStore


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
