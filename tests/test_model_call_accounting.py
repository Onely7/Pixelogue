"""Usage and replay contracts for actual inference attempts, including interrupted calls."""

import json
from pathlib import Path

import httpx
import pytest

from pixelogue.config import ModelEndpoint, RuntimeConfig, ServingRuntimeIdentity, load_config
from pixelogue.contracts import RubricVerdict
from pixelogue.errors import ExecutionError
from pixelogue.profiling import profile_database
from pixelogue.serving import VllmClient
from pixelogue.store import RunStore


def completion(content: str, usage: dict | None) -> dict:
    result = {
        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
    }
    if usage is not None:
        result["usage"] = usage
    return result


def invoke(client: VllmClient, trial_id: str = "trial-1"):
    return client.invoke(
        "rubric_item",
        {
            "target_language": "en",
            "public_history": [],
            "question": "q",
            "candidate_answer": "a",
            "image_views": [],
            "criterion": {},
        },
        (),
        RubricVerdict,
        max_tokens=32,
        temperature=0.0,
        seed=1,
        trial_id=trial_id,
    )


def client_for(store: RunStore, http: httpx.Client) -> VllmClient:
    return VllmClient(
        ModelEndpoint(repo_id="Qwen/Qwen3.5-2B"),
        RuntimeConfig(),
        run_id=store.run_dir.name,
        store=store,
        client=http,
    )


def test_invalid_content_is_billed_and_same_trial_replays_failure(tmp_path: Path) -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json=completion("not JSON", {"prompt_tokens": 12, "completion_tokens": 9})
        )

    with RunStore(tmp_path, "invalid", require_local_wal=False) as store:
        with httpx.Client(
            base_url="http://localhost/v1/", transport=httpx.MockTransport(handler)
        ) as http:
            client = client_for(store, http)
            for _ in range(2):
                with pytest.raises(ExecutionError, match="JSON"):
                    invoke(client)
        budget = store.connection.execute("SELECT * FROM budget").fetchone()
        row = store.connection.execute("SELECT * FROM model_call_attempt").fetchone()
        assert calls == budget["request_count"] == 1
        assert budget["output_tokens"] == row["output_tokens"] == 9
        assert budget["reserved_output_tokens"] == 0
        assert row["status"] == "INVALID"
        assert row["usage_status"] == "MEASURED"
        assert row["input_tokens"] == 12
        profile = profile_database(store.database_path)
        assert profile.completed_calls == 0
        assert profile.actual_attempts == 1 and profile.cache_accesses == 1
        assert profile.stages[0].output_tokens == 9
        assert profile.status_counts == {"INVALID": 1}


def test_missing_usage_remains_unknown_in_success_and_capacity(tmp_path: Path) -> None:
    response = completion('{"verdict":"MET","reason":"Visible."}', None)
    with RunStore(tmp_path, "missing", require_local_wal=False) as store:
        with httpx.Client(
            base_url="http://localhost/v1/",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        ) as http:
            result = invoke(client_for(store, http))
        assert result.value.verdict == "MET"
        assert result.prompt_tokens is None and result.completion_tokens is None
        row = store.connection.execute("SELECT * FROM model_call_attempt").fetchone()
        assert row["output_tokens"] is None and row["usage_status"] == "MISSING"
        assert (
            store.connection.execute("SELECT reserved_output_tokens FROM budget").fetchone()[0]
            == 32
        )
        profile = profile_database(store.database_path)
        assert profile.completed_calls == 1
        assert profile.stages[0].output_tokens is None
        assert profile.stages[0].known_output_tokens == 0
        assert profile.stages[0].missing_output_usage_calls == 1


@pytest.mark.parametrize("count", [True, "9", -1, 33])
def test_invalid_usage_is_rejected_and_not_reported_as_measured(tmp_path, count) -> None:
    response = completion(
        '{"verdict":"MET","reason":"Visible."}', {"prompt_tokens": 12, "completion_tokens": count}
    )
    with RunStore(tmp_path, "bad-usage", require_local_wal=False) as store:
        with httpx.Client(
            base_url="http://localhost/v1/",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
        ) as http:
            with pytest.raises(ExecutionError) as caught:
                invoke(client_for(store, http))
            assert caught.value.reason == "MODEL_USAGE_INVALID"
        row = store.connection.execute("SELECT * FROM model_call_attempt").fetchone()
        assert row["output_tokens"] is None and row["usage_status"] == "INVALID"
        assert (
            store.connection.execute("SELECT reserved_output_tokens FROM budget").fetchone()[0]
            == 32
        )


@pytest.mark.parametrize("crash_point", ["before_accounting", "before_typed_commit"])
def test_saved_response_recovers_once_without_another_http_call(
    tmp_path, monkeypatch, crash_point
) -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=completion(
                '{"verdict":"MET","reason":"Visible."}',
                {"prompt_tokens": 12, "completion_tokens": 9},
            ),
        )

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("simulated interruption after response saving")

    with httpx.Client(
        base_url="http://localhost/v1/", transport=httpx.MockTransport(handler)
    ) as http:
        with RunStore(tmp_path, "recover", require_local_wal=False) as store:
            client = client_for(store, http)
            with monkeypatch.context() as patch:
                if crash_point == "before_accounting":
                    patch.setattr(store, "_apply_attempt_journal", interrupt)
                else:
                    patch.setattr(client, "_record", interrupt)
                with pytest.raises(KeyboardInterrupt):
                    invoke(client)
        for _ in range(2):
            with RunStore(tmp_path, "recover", require_local_wal=False) as recovered:
                result = invoke(client_for(recovered, http))
                assert result.value.verdict == "MET"
                budget = recovered.connection.execute("SELECT * FROM budget").fetchone()
                assert budget["request_count"] == 1
                assert budget["output_tokens"] == 9 and budget["reserved_output_tokens"] == 0
                assert (
                    recovered.connection.execute(
                        "SELECT COUNT(*) FROM model_call_attempt"
                    ).fetchone()[0]
                    == 1
                )
                assert recovered.verify()["model_calls"] == 1
    assert calls == 1


def test_independent_trials_call_independently_and_replays_do_not(tmp_path: Path) -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        assert "trial_id" not in json.loads(request.content)
        return httpx.Response(
            200,
            json=completion(
                '{"verdict":"MET","reason":"Visible."}',
                {"prompt_tokens": 12, "completion_tokens": 9},
            ),
        )

    with RunStore(tmp_path, "replicates", require_local_wal=False) as store:
        with httpx.Client(
            base_url="http://localhost/v1/", transport=httpx.MockTransport(handler)
        ) as http:
            client = client_for(store, http)
            for trial in ["trial-1", "trial-2", "trial-1", "trial-2"]:
                invoke(client, trial)
        assert calls == 2
        assert store.connection.execute("SELECT output_tokens FROM budget").fetchone()[0] == 18
        assert (
            store.connection.execute("SELECT COUNT(*) FROM model_call_attempt").fetchone()[0] == 2
        )
        assert (
            store.connection.execute(
                "SELECT COUNT(*) FROM artifact WHERE kind='cache-accesses'"
            ).fetchone()[0]
            == 2
        )


def test_server_configuration_changes_run_and_call_identity(tmp_path: Path) -> None:
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json=completion(
                '{"verdict":"MET","reason":"Visible."}',
                {"prompt_tokens": 12, "completion_tokens": 9},
            ),
        )

    config = load_config(Path("configs/pilot.yaml"))
    identities = [
        ServingRuntimeIdentity.model_validate(
            {
                "vllm_version": "0.29.0",
                "structured_output_backend": backend,
                "disable_any_whitespace": whitespace,
                "server_manifest_sha256": digest * 64,
            }
        )
        for backend, whitespace, digest in [("auto", False, "a"), ("xgrammar", True, "b")]
    ]
    config_hashes = []
    with RunStore(tmp_path, "server-identity", require_local_wal=False) as store:
        with httpx.Client(
            base_url="http://localhost/v1/", transport=httpx.MockTransport(handler)
        ) as http:
            for identity in identities:
                endpoint = config.models.generator_a.model_copy(
                    update={"serving_runtime": identity}
                )
                models = config.models.model_copy(update={"generator_a": endpoint})
                config_hashes.append(config.model_copy(update={"models": models}).config_hash)
                client = VllmClient(
                    endpoint, config.runtime, run_id="server-identity", store=store, client=http
                )
                invoke(client)
                invoke(client)
    assert calls == 2
    assert config_hashes[0] != config_hashes[1]
