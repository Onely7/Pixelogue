from pathlib import Path

import httpx
import pytest
from pydantic import HttpUrl

from pixelogue.config import ModelEndpoint, RuntimeConfig
from pixelogue.contracts import RubricVerdict
from pixelogue.errors import ExecutionError
from pixelogue.prompts import validate_stage_payload
from pixelogue.serving import ModelAdapter, VllmClient
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


def test_invalid_model_output_releases_output_reservation(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
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
        with pytest.raises(ExecutionError, match="validation errors"):
            client.invoke(
                "rubric_item",
                _rubric_payload(),
                (),
                RubricVerdict,
                max_tokens=32,
                temperature=0.0,
                seed=1,
            )
        budget = store.connection.execute(
            "SELECT request_count, reserved_output_tokens FROM budget WHERE singleton=1"
        ).fetchone()

    assert tuple(budget) == (1, 0)
