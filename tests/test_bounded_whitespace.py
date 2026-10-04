"""Protect the runtime patch identity and the extraction-only schema extension."""

import hashlib
import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelogue.config import ModelEndpoint, RuntimeConfig, ServingRuntimeIdentity
from pixelogue.contracts import RubricVerdict
from pixelogue.errors import ExecutionError
from pixelogue.serialization import canonical_hash
from pixelogue.serving import VllmClient


@pytest.mark.parametrize("stage", ["evidence_extraction", "candidate_binding", "rubric_item"])
def test_whitespace_bound_is_limited_to_extraction_schema_and_changes_request_identity(stage):
    endpoint = ModelEndpoint(
        repo_id="Qwen/Qwen3.5-2B",
        serving_runtime=ServingRuntimeIdentity(
            vllm_version="0.29.0",
            structured_output_backend="xgrammar",
            disable_any_whitespace=False,
            server_manifest_sha256="a" * 64,
            xgrammar_whitespace_patch_sha256="97e96f276ad6ad10d536f7c88e294a8115cce11bd219763c0a708dabf913020e",
        ),
    )
    client = VllmClient(endpoint, RuntimeConfig(json_whitespace_max_chars=32), run_id="bounded")
    original = VllmClient(endpoint, RuntimeConfig(), run_id="original")
    try:
        body = client._build_body(
            stage, {}, (), RubricVerdict, max_tokens=512, temperature=0.0, seed=1
        )
        baseline = original._build_body(
            stage, {}, (), RubricVerdict, max_tokens=512, temperature=0.0, seed=1
        )
    finally:
        client.client.close()
        original.client.close()
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema.get("x-pixelogue-max-whitespace-chars") == (
        None if stage == "rubric_item" else 32
    )
    assert (canonical_hash(body) != canonical_hash(baseline)) is (stage != "rubric_item")


@pytest.mark.parametrize(
    "runtime_identity",
    [
        None,
        {
            "vllm_version": "0.29.0",
            "structured_output_backend": "xgrammar",
            "disable_any_whitespace": False,
            "server_manifest_sha256": "a" * 64,
        },
        {
            "vllm_version": "0.29.0",
            "structured_output_backend": "xgrammar",
            "disable_any_whitespace": True,
            "server_manifest_sha256": "a" * 64,
            "xgrammar_whitespace_patch_sha256": "b" * 64,
        },
    ],
)
def test_bound_refuses_an_unverified_or_incompatible_endpoint(runtime_identity):
    identity = ServingRuntimeIdentity.model_validate(runtime_identity) if runtime_identity else None
    client = VllmClient(
        ModelEndpoint(repo_id="Qwen/Qwen3.5-2B", serving_runtime=identity),
        RuntimeConfig(json_whitespace_max_chars=32),
        run_id="unverified",
    )
    try:
        with pytest.raises(ExecutionError) as caught:
            client._build_body(
                "evidence_extraction",
                {},
                (),
                RubricVerdict,
                max_tokens=512,
                temperature=0.0,
                seed=1,
            )
    finally:
        client.client.close()
    assert caught.value.reason == "MODEL_RUNTIME_PATCH_REQUIRED"


@pytest.mark.parametrize("bound", [0, 257, True, "32"])
def test_whitespace_limit_rejects_invalid_configuration(bound):
    with pytest.raises(ValidationError):
        RuntimeConfig(json_whitespace_max_chars=bound)


def test_pinned_patch_is_idempotent_restorable_and_refuses_foreign_changes(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "pixelogue_test_whitespace_patch", Path("runtime/vllm/whitespace_patch.py")
    )
    assert spec is not None and spec.loader is not None
    patch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patch)
    upstream = ("# Synthetic upstream fixture\n" + patch.ORIGINAL).encode()
    monkeypatch.setattr(patch, "UPSTREAM_SHA256", hashlib.sha256(upstream).hexdigest())
    target = tmp_path / "backend.py"
    target.write_bytes(upstream)
    with pytest.raises(ValueError, match="not installed"):
        patch.install(target, check_only=True)
    result = patch.install(target)
    assert patch.install(target, check_only=True) == result
    assert patch.install(target) == result
    assert "max_whitespace_cnt=whitespace_bound" in target.read_text()
    target.write_text(target.read_text() + "\n# Foreign change\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        patch.install(target)
    target.write_bytes(patch.patch_content(upstream))
    restored = patch.install(target, restore=True)
    assert restored["restored"]
    assert target.read_bytes() == upstream
