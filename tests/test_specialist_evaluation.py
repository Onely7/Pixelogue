"""Specialist calibration retains unprocessed cases and certifies independent images only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pixelogue.config import load_config
from pixelogue.contracts import GateVerdict, SourcePurpose, TextPayload
from pixelogue.errors import ExecutionError
from pixelogue.specialist_evaluation import (
    SpecialistEvaluationCase,
    SpecialistEvaluationResult,
    _schema_retry_feedback,
    build_calibration_manifest,
    run_specialist_evaluation,
)


def test_specialist_retry_uses_stage_specific_feedback() -> None:
    error = ExecutionError("MODEL_SCHEMA_MISMATCH", "Circuit netlist terminals differ")
    circuit = _schema_retry_feedback("specialist_circuit_source", error)
    music = _schema_retry_feedback("specialist_music_source", error)
    geometry = _schema_retry_feedback("specialist_geometry_answer", error)
    assert "junction names" in circuit
    assert "scope_region must contain every event" in music
    assert "answer_quote and reported" in geometry


def _result(
    index: int, *, positive: bool, verdict: str = "MET", status: str = "COMPLETE"
) -> SpecialistEvaluationResult:
    return SpecialistEvaluationResult.model_validate(
        {
            "trial_id": f"{index:064x}",
            "case_id": f"case-{index}",
            "status": status,
            "verdict": verdict if status == "COMPLETE" else None,
            "validator_name": "music_notation_validator",
            "validator_version": "1",
            "model_locks": ("a" * 64, "b" * 64),
            "response_artifact_hashes": ("c" * 64, "d" * 64) if status == "COMPLETE" else None,
            "error": None if status == "COMPLETE" else "transport failed",
            "gold_accept": positive,
            "split": "confirmation",
            "image_group_id": f"image-{index}",
            "domain": "single-voice",
            "task_id": "music_notation_reading",
        }
    )


def test_complete_confirmation_group_computes_model_bound_certificate() -> None:
    results = (
        *(_result(index, positive=True) for index in range(20)),
        *(_result(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
    )
    manifest, report = build_calibration_manifest(results)
    assert report["eligible"] == 1
    assert manifest.certificates[0].positive_images == 20
    assert manifest.certificates[0].negative_images == 60


def test_failed_calls_are_pending_and_never_counted_as_zero_cost_or_success() -> None:
    manifest, report = build_calibration_manifest(
        (
            _result(0, positive=True),
            _result(1, positive=False, status="FAILED"),
        )
    )
    assert manifest.certificates == ()
    assert report["certificates"] == 0
    assert {item["reason"] for item in report["pending"]} == {
        "transport failed",
        "missing_positive_or_negative",
    }


def test_failed_second_evaluator_resumes_from_first_saved_stage(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    from pixelogue import specialist_evaluation

    image, root = image_artifact
    image = image.model_copy(update={"purpose": SourcePurpose.EVALUATION})
    case = SpecialistEvaluationCase(
        case_id="case",
        image=image,
        task_id="music_notation_reading",
        domain="single-voice",
        scope_id="score",
        view_id=image.full_view.view_id,
        visible_scope="whole score",
        public_parameters=(),
        target_language="en",
        public_history=(),
        question="Read this bar",
        candidate_answer="C4",
        split="development",
    )
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"run_root": tmp_path / "runs", "require_local_wal": False}
            )
        }
    )
    calls = [0, 0]
    second_online = False

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.client = self

        def close(self):
            pass

        def invoke(self, stage, body, images, model, **kwargs):
            judge = int(kwargs["bypass_cache"])
            calls[judge] += 1
            if judge and not second_online:
                raise ExecutionError("MODEL_OFFLINE", "Second evaluator is offline")
            return SimpleNamespace(
                value=TextPayload(text="source"),
                request_hash="a" * 64,
                response_hash="b" * 64,
                prompt_tokens=1,
                completion_tokens=1,
            )

    def fake_verify(instruction, payload, invoke, image_path):
        invoke("specialist_music_source", {"question": payload["question"]}, TextPayload, 0)
        invoke("specialist_music_source", {"question": payload["question"]}, TextPayload, 1)
        return (
            SimpleNamespace(
                name="music_notation_validator",
                verdict=GateVerdict.MET,
                evidence=({"judge": 0}, {"judge": 1}),
            ),
        )

    monkeypatch.setattr(specialist_evaluation, "VllmClient", FakeClient)
    monkeypatch.setattr(specialist_evaluation, "verify_operation", fake_verify)
    output = tmp_path / "evaluation"
    first = run_specialist_evaluation((case,), config, root, output, "study")
    assert first["failed"] == 1 and calls == [1, 1]
    second_online = True
    second = run_specialist_evaluation((case,), config, root, output, "study", retry_failed=True)
    assert second["completed"] == 1 and second["stage_reused"] == 1
    assert calls == [1, 2]


@pytest.mark.parametrize(
    ("failure_reason", "counter"),
    [
        ("MODEL_SCHEMA_MISMATCH", "schema_retries"),
        ("MODEL_FINISH_REASON", "finish_retries"),
    ],
)
def test_invalid_specialist_reading_gets_one_bounded_retry(
    tmp_path: Path, image_artifact, monkeypatch, failure_reason: str, counter: str
) -> None:
    from pixelogue import specialist_evaluation

    image, root = image_artifact
    image = image.model_copy(update={"purpose": SourcePurpose.EVALUATION})
    case = SpecialistEvaluationCase(
        case_id="retry-case",
        image=image,
        task_id="music_notation_reading",
        domain="single-voice",
        scope_id="score",
        view_id=image.full_view.view_id,
        visible_scope="whole score",
        public_parameters=(),
        target_language="en",
        public_history=(),
        question="Read this bar",
        candidate_answer="C4",
        split="development",
    )
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "storage": config.storage.model_copy(
                update={"run_root": tmp_path / "runs", "require_local_wal": False}
            )
        }
    )
    feedback: list[str | None] = []
    output_limits: list[int] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.client = self

        def close(self):
            pass

        def invoke(self, stage, body, images, model, **kwargs):
            feedback.append(kwargs.get("retry_feedback"))
            output_limits.append(kwargs["max_tokens"])
            if len(feedback) == 1:
                raise ExecutionError(failure_reason, "scope_region needs positive extent")
            return SimpleNamespace(
                value=TextPayload(text="source"),
                request_hash="a" * 64,
                response_hash="b" * 64,
                prompt_tokens=1,
                completion_tokens=1,
            )

    def fake_verify(instruction, payload, invoke, image_path):
        invoke("specialist_music_source", {"question": payload["question"]}, TextPayload, 0)
        invoke("specialist_music_source", {"question": payload["question"]}, TextPayload, 1)
        return (
            SimpleNamespace(
                name="music_notation_validator",
                verdict=GateVerdict.MET,
                evidence=({"judge": 0}, {"judge": 1}),
            ),
        )

    monkeypatch.setattr(specialist_evaluation, "VllmClient", FakeClient)
    monkeypatch.setattr(specialist_evaluation, "verify_operation", fake_verify)
    stats = run_specialist_evaluation((case,), config, root, tmp_path / "evaluation", "retry")
    assert stats["completed"] == 1 and stats[counter] == 1
    assert feedback[0] is None
    if failure_reason == "MODEL_FINISH_REASON":
        assert "token limit" in (feedback[1] or "")
    else:
        assert "positive extent" in (feedback[1] or "")
    assert len(feedback) == 3
    assert output_limits == [config.tasks.evidence_max_tokens] * 3
