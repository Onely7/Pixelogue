"""Explicit decision deployments and CLI-owned worker lifecycle."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError
from typer.testing import CliRunner

from pixelogue.cli import app
from pixelogue.config import DecisionRoutingConfig, load_config
from pixelogue.contracts import ConversationArtifact
from pixelogue.decision_factory import create_decision_client, decision_client_context
from pixelogue.decision_serving import JevHttpDecisionClient
from pixelogue.errors import ConfigurationError, ExecutionError
from pixelogue.io import write_json, write_jsonl
from pixelogue.operations import PreparedDataset
from pixelogue.serialization import canonical_hash

OMNI_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
JEV_REVISION = "f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc"


def configured(routing: DecisionRoutingConfig):
    config = load_config(Path("configs/pilot.yaml"))
    return config.model_copy(
        update={"tasks": config.tasks.model_copy(update={"decision_routing": routing})}
    )


def jev_config(**deployment):
    return configured(
        DecisionRoutingConfig(
            evidence_enabled=True,
            model_repository="autotrust/JEV-27B-VL",
            model_revision=JEV_REVISION,
            **deployment,
        )
    )


def omni_config(**deployment):
    return configured(
        DecisionRoutingConfig(
            binding_enabled=True,
            model_repository="akhilaaa3/Jev-Omni",
            model_revision=OMNI_REVISION,
            **deployment,
        )
    )


def test_disabled_factory_does_not_start_any_native_client(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Disabled classifier was constructed")

    monkeypatch.setattr("pixelogue.decision_serving.JevHttpDecisionClient", forbidden)
    monkeypatch.setattr("pixelogue.decision_serving.SubprocessOmniDecisionClient", forbidden)
    assert create_decision_client(load_config(Path("configs/standard.yaml")), tmp_path) is None


def test_jev_native_factory_requires_endpoint_and_refuses_worker_paths(tmp_path):
    with pytest.raises(ConfigurationError) as error:
        create_decision_client(jev_config(), tmp_path)
    assert error.value.reason == "DECISION_ENDPOINT_REQUIRED"
    with pytest.raises(ConfigurationError) as error:
        create_decision_client(
            jev_config(base_url=HttpUrl("http://localhost:18304"), snapshot_path=tmp_path), tmp_path
        )
    assert error.value.reason == "DECISION_DEPLOYMENT_MISMATCH"


def test_native_http_factory_preserves_exact_identity_and_refuses_remote_hosts(tmp_path):
    client = create_decision_client(
        jev_config(base_url=HttpUrl("http://127.0.0.1:18304/v1")), tmp_path
    )
    assert isinstance(client, JevHttpDecisionClient)
    assert client.model_repository == "autotrust/JEV-27B-VL"
    assert client.model_revision == JEV_REVISION
    client.close()
    with pytest.raises(ExecutionError) as error:
        create_decision_client(jev_config(base_url=HttpUrl("https://example.org")), tmp_path)
    assert error.value.reason == "EXTERNAL_INFERENCE_FORBIDDEN"


def test_omni_factory_requires_runtime_snapshot_and_explicit_physical_gpu(tmp_path, monkeypatch):
    with pytest.raises(ConfigurationError) as error:
        create_decision_client(omni_config(), tmp_path)
    assert error.value.reason == "DECISION_WORKER_INPUTS_REQUIRED"
    snapshot = tmp_path / OMNI_REVISION
    snapshot.mkdir()
    config = omni_config(runtime_python=Path(sys.executable), snapshot_path=snapshot)
    for gpu in (None, "0", "GPU-one,GPU-two"):
        if gpu is None:
            monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        else:
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", gpu)
        with pytest.raises(ConfigurationError) as error:
            create_decision_client(config, tmp_path)
        assert error.value.reason == "DECISION_GPU_REQUIRED"


def test_omni_factory_rejects_unpinned_snapshot_and_relative_runtime(tmp_path):
    with pytest.raises(ConfigurationError) as error:
        create_decision_client(
            omni_config(runtime_python=Path("python"), snapshot_path=tmp_path), tmp_path
        )
    assert error.value.reason == "DECISION_RUNTIME_INVALID"
    with pytest.raises(ConfigurationError) as error:
        create_decision_client(
            omni_config(runtime_python=Path(sys.executable), snapshot_path=tmp_path), tmp_path
        )
    assert error.value.reason == "DECISION_SNAPSHOT_INVALID"


def test_omni_factory_passes_gpu_and_uses_separate_runtime_worker(tmp_path, monkeypatch):
    received = {}
    snapshot = tmp_path / OMNI_REVISION
    snapshot.mkdir()

    class Worker:
        def __init__(self, runtime_python, snapshot_path, revision, **kwargs):
            received.update(
                runtime_python=runtime_python,
                snapshot_path=snapshot_path,
                revision=revision,
                **kwargs,
            )

    monkeypatch.setattr("pixelogue.decision_serving.SubprocessOmniDecisionClient", Worker)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-inspected-device")
    config = omni_config(runtime_python=Path(sys.executable), snapshot_path=snapshot)
    create_decision_client(config, tmp_path / "run")
    assert received["runtime_python"] == Path(sys.executable)
    assert received["snapshot_path"] == snapshot
    assert received["revision"] == OMNI_REVISION
    assert received["env"] == {"CUDA_VISIBLE_DEVICES": "GPU-inspected-device"}
    assert received["stderr_path"] == tmp_path / "run" / "decision-worker.log"


def test_unknown_native_deployment_configuration_is_rejected():
    with pytest.raises(ValidationError):
        DecisionRoutingConfig.model_validate({"quantization": "fp8"})


@pytest.mark.parametrize("fail", [False, True])
def test_owned_client_is_closed_after_success_or_failure(tmp_path, monkeypatch, fail):
    closed = []

    class Client:
        def close(self):
            closed.append(True)

    client = Client()
    monkeypatch.setattr("pixelogue.decision_factory.create_decision_client", lambda *args: client)
    try:
        with decision_client_context(jev_config(), tmp_path) as value:
            assert value is client
            if fail:
                raise RuntimeError("failed synthesis")
    except RuntimeError:
        assert fail
    assert closed == [True]


@pytest.mark.parametrize("fail", [False, True])
def test_synthesize_cli_injects_explicit_client_and_closes_it(
    tmp_path, monkeypatch, image_artifact, fail
):
    image, root = image_artifact
    images = tmp_path / "images.jsonl"
    write_jsonl(images, [image])
    write_json(
        root / "manifest.json",
        PreparedDataset(
            images=(image,),
            failures=(),
            splits={},
            manifest_hash=canonical_hash({"image": image.image_id}),
        ),
    )
    config = jev_config()
    config = config.model_copy(
        update={"storage": config.storage.model_copy(update={"run_root": tmp_path / "runs"})}
    )
    monkeypatch.setattr("pixelogue.cli.load_config", lambda path: config)
    monkeypatch.setattr("pixelogue.cli._clients", lambda *args: (None, None, None))
    closed = []

    class Client:
        def close(self):
            closed.append(True)

    client = Client()

    def create_client(actual_config, run_dir):
        assert actual_config == config
        # Run identity is checked before allocating a classifier worker.
        with sqlite3.connect(run_dir / "run.sqlite3") as connection:
            assert connection.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 1
        return client

    monkeypatch.setattr("pixelogue.decision_factory.create_decision_client", create_client)

    class Coordinator:
        def __init__(self, actual_config, run_id, store, selector, gen_a, gen_b, decision_client):
            assert actual_config.tasks.decision_routing.evidence_enabled
            assert decision_client is client

        def synthesize_batch(self, jobs, artifact_root, **kwargs):
            if fail:
                raise RuntimeError("failed generation")
            for job in jobs:
                yield ConversationArtifact(
                    conversation_id="conversation",
                    image=job.image,
                    target_language="en",
                    generation_model="test",
                    turns=(),
                    status="REJECTED",
                )

    monkeypatch.setattr("pixelogue.cli.SynthesisCoordinator", Coordinator)
    result = CliRunner().invoke(
        app,
        [
            "synthesize",
            "--images",
            str(images),
            "--artifact-root",
            str(root),
            "--run-id",
            "native",
            "--output",
            str(tmp_path / "output.jsonl"),
            "--quiet",
        ],
    )
    assert (result.exit_code != 0) is fail
    assert closed == [True]


def test_rate_existing_does_not_construct_classifier_and_preserves_rating_configuration(
    tmp_path, monkeypatch, image_artifact
):
    image, root = image_artifact
    config = jev_config()
    config = config.model_copy(
        update={"storage": config.storage.model_copy(update={"run_root": tmp_path / "runs"})}
    )
    monkeypatch.setattr("pixelogue.cli.load_config", lambda path: config)
    monkeypatch.setattr("pixelogue.cli._clients", lambda *args: (None, None, None))

    def forbidden(*args):
        raise AssertionError("Rerating started a classifier")

    monkeypatch.setattr("pixelogue.decision_factory.create_decision_client", forbidden)
    original = ConversationArtifact(
        conversation_id="saved",
        image=image,
        target_language="en",
        generation_model="test",
        turns=(),
        status="REJECTED",
    )
    records = tmp_path / "conversations.jsonl"
    write_jsonl(records, [original])
    received = []

    class Coordinator:
        def __init__(self, rating_config, *args):
            assert not rating_config.tasks.decision_routing.evidence_enabled
            assert not rating_config.tasks.decision_routing.binding_enabled
            assert rating_config.models == config.models
            assert rating_config.evaluation == config.evaluation
            received.append(rating_config)

        def rate_existing(self, item, artifact_root):
            assert item == original
            assert artifact_root == root
            return item

    monkeypatch.setattr("pixelogue.cli.SynthesisCoordinator", Coordinator)
    result = CliRunner().invoke(
        app,
        [
            "rate-existing",
            "--conversations",
            str(records),
            "--artifact-root",
            str(root),
            "--run-id",
            "rated",
            "--output",
            str(tmp_path / "rated.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(received) == 1
