"""Reject repetitive completions while permitting bounded extraction recovery."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from pixelogue.config import ModelEndpoint, RepetitionDetectionConfig, RuntimeConfig, load_config
from pixelogue.contracts import RubricVerdict
from pixelogue.errors import ExecutionError
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.routing import ImageProfile
from pixelogue.serving import VllmClient
from pixelogue.store import RunStore


@pytest.mark.parametrize(
    "settings",
    [
        {"min_pattern_size": 5, "max_pattern_size": 4},
        {"min_count": 1},
        {"max_pattern_size": 17},
        {"stages": []},
        {"stages": ["answer_generation"]},
        {"stages": ["question_draft", "question_draft"]},
        {"stages": ["evidence_extraction"]},
        {"unknown": True},
    ],
)
def test_repetition_configuration_rejects_unbounded_or_unintended_stages(settings):
    with pytest.raises(ValidationError):
        RepetitionDetectionConfig.model_validate_json(json.dumps(settings))


@pytest.mark.parametrize(
    "stage,expected",
    [
        ("quantity_source", True),
        ("question_draft", True),
        ("image_profile", True),
        ("answer_generation", False),
        ("question_gate", False),
        ("holistic_review", False),
    ],
)
def test_engine_guard_is_only_sent_to_private_structured_stages(stage, expected):
    client = VllmClient(
        ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
        RuntimeConfig(repetition_detection=RepetitionDetectionConfig()),
        run_id="guard",
    )
    try:
        body = client._build_body(
            stage, {}, (), RubricVerdict, max_tokens=512, temperature=0.0, seed=1
        )
    finally:
        client.client.close()
    assert ("repetition_detection" in body) is expected
    if expected:
        assert body["repetition_detection"] == {
            "min_pattern_size": 1,
            "max_pattern_size": 4,
            "min_count": 64,
        }


def test_guard_is_on_by_default_and_disabling_changes_configuration_identity():
    original = load_config(Path("configs/pilot.yaml"))
    assert original.runtime.repetition_detection == RepetitionDetectionConfig()
    disabled = original.model_copy(
        update={"runtime": original.runtime.model_copy(update={"repetition_detection": None})}
    )
    assert disabled.config_hash != original.config_hash
    bodies = []
    for runtime in (original.runtime, disabled.runtime):
        client = VllmClient(original.models.generator_a, runtime, run_id="default")
        try:
            bodies.append(
                client._build_body(
                    "question_draft", {}, (), RubricVerdict, max_tokens=512, temperature=0.0, seed=1
                )
            )
        finally:
            client.client.close()
    assert "repetition_detection" in bodies[0]
    assert "repetition_detection" not in bodies[1]


@pytest.mark.parametrize("content", ['{"verdict":"MET","reason":"Valid JSON"}', "{"])
def test_repetition_stop_never_accepts_even_parseable_json(content):
    client = VllmClient(ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"), RuntimeConfig(), run_id="decode")
    raw = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "repetition",
                    "stop_reason": "repetition_detected",
                    "message": {"content": content},
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 64},
        }
    ).encode()
    try:
        with pytest.raises(ExecutionError) as caught:
            client._decode_typed_response(raw, RubricVerdict, max_tokens=512)
    finally:
        client.client.close()
    assert caught.value.reason == "MODEL_OUTPUT_REPETITION"


def test_repetition_is_costed_preserved_and_not_retried(tmp_path):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "runtime": config.runtime.model_copy(
                update={"repetition_detection": RepetitionDetectionConfig()}
            )
        }
    )
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "repetition",
                        "stop_reason": "repetition_detected",
                        "message": {"content": "{" + " " * 64},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 64},
            },
        )

    with RunStore(tmp_path, "terminal", require_local_wal=False) as store:
        store.initialize_run("terminal", config.config_hash, config.profile)
        with httpx.Client(
            base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(respond)
        ) as http:
            client = VllmClient(
                config.models.generator_a,
                config.runtime,
                run_id="terminal",
                store=store,
                client=http,
            )
            coordinator = SynthesisCoordinator(config, "terminal", store, client, client, client)
            payload = {
                "target_language": "en",
                "public_history": [],
                "question": "q",
                "candidate_answer": "a",
                "image_views": [],
                "criterion": {},
            }
            with pytest.raises(ExecutionError) as caught:
                coordinator._invoke(
                    client,
                    "rubric_item",
                    payload,
                    (),
                    RubricVerdict,
                    max_tokens=512,
                    temperature=0.0,
                    seed=1,
                )
        budget = store.connection.execute(
            "SELECT request_count, output_tokens FROM budget WHERE singleton=1"
        ).fetchone()
        attempt = store.connection.execute("SELECT status FROM model_call_attempt").fetchone()
        assert tuple(budget) == (1, 64)
        assert attempt["status"] == "INVALID"
        assert store.connection.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] > 0
    assert caught.value.reason == "MODEL_OUTPUT_REPETITION"
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 512


@pytest.mark.parametrize(
    "recovery,attempt_limit,finishes,expected_calls,expected_success",
    [
        ("retry", 2, ["repetition", "stop"], 2, True),
        ("retry", 2, ["repetition", "repetition"], 2, False),
        ("retry", 1, ["repetition", "stop"], 1, False),
        ("abstain", 2, ["repetition", "stop"], 1, False),
    ],
)
def test_extraction_recovers_within_existing_attempt_limit_and_resumes_without_http(
    tmp_path, recovery, attempt_limit, finishes, expected_calls, expected_success
):
    original = load_config(Path("configs/pilot.yaml"))
    config = original.model_copy(
        update={
            "runtime": original.runtime.model_copy(
                update={
                    "repetition_detection": RepetitionDetectionConfig(recovery=recovery),
                    "structured_output_max_attempts": attempt_limit,
                }
            )
        }
    )
    calls = []
    report = {
        "image_kind": "photo",
        "readable_text": "none",
        "supported_families": [],
        "reason": "No family is clearly supported",
    }

    def respond(request):
        calls.append(json.loads(request.content))
        finish = finishes[len(calls) - 1]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": finish,
                        "message": {
                            "content": json.dumps(report) if finish == "stop" else "{" + " " * 64
                        },
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 64},
            },
        )

    payload = {
        "image_views": [],
        "family_definitions": [{"family": "visual_description", "operations": []}],
    }
    with RunStore(tmp_path, "recover", require_local_wal=False) as store:
        store.initialize_run("recover", config.config_hash, config.profile)
        with httpx.Client(
            base_url="http://127.0.0.1:8000/v1/", transport=httpx.MockTransport(respond)
        ) as http:
            client = VllmClient(
                config.models.generator_a,
                config.runtime,
                run_id="recover",
                store=store,
                client=http,
            )
            coordinator = SynthesisCoordinator(config, "recover", store, client, client, client)
            for _ in range(2):
                if expected_success:
                    result = coordinator._invoke(
                        client,
                        "image_profile",
                        payload,
                        (),
                        ImageProfile,
                        max_tokens=512,
                        temperature=0.0,
                        seed=1,
                    )
                    assert result.model_dump(mode="json") == report
                else:
                    with pytest.raises(ExecutionError, match="repeated token pattern"):
                        coordinator._invoke(
                            client,
                            "image_profile",
                            payload,
                            (),
                            ImageProfile,
                            max_tokens=512,
                            temperature=0.0,
                            seed=1,
                        )
        budget = store.connection.execute(
            "SELECT request_count, output_tokens FROM budget WHERE singleton=1"
        ).fetchone()
        assert tuple(budget) == (expected_calls, expected_calls * 64)
        statuses = [
            row[0]
            for row in store.connection.execute(
                "SELECT status FROM model_call_attempt ORDER BY rowid"
            )
        ]
        assert statuses == (
            ["INVALID", "COMPLETE"] if expected_success else ["INVALID"] * expected_calls
        )
    assert len(calls) == expected_calls
    assert all(call["max_tokens"] == 512 for call in calls)
    assert calls[0]["seed"] == 1
    if expected_calls == 2:
        retry = json.loads(calls[1]["messages"][1]["content"][0]["text"])
        assert calls[1]["seed"] == 100_001
        assert "use UNKNOWN" in retry["retry_feedback"]
        assert "original token limit" in retry["retry_feedback"]
