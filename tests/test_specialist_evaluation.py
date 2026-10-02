"""Specialist calibration retains unprocessed cases and certifies independent images only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pixelogue.calibration import CalibrationManifest
from pixelogue.config import StrictModel, load_config
from pixelogue.contracts import GateVerdict, SourcePurpose, TextPayload
from pixelogue.errors import ExecutionError
from pixelogue.specialist_evaluation import (
    SpecialistEvaluationCase,
    SpecialistEvaluationResult,
    _schema_retry_feedback,
    build_calibration_manifest,
    run_specialist_evaluation,
)
from pixelogue.task_registry import ValidatorRegistration


def test_specialist_retry_uses_stage_specific_feedback() -> None:
    error = ExecutionError("MODEL_SCHEMA_MISMATCH", "Circuit netlist terminals differ")
    circuit = _schema_retry_feedback("specialist_circuit_source", error)
    assert "component/terminal inventory" in circuit
    assert "without a power source" in circuit
    ui = _schema_retry_feedback("specialist_ui_source", error)
    music = _schema_retry_feedback("specialist_music_source", error)
    chemistry = _schema_retry_feedback("specialist_chemistry_source", error)
    geometry = _schema_retry_feedback("specialist_geometry_answer", error)
    assert "junction names" in circuit
    assert "controls=[]" in ui
    assert "scope_region must contain every event" in music
    assert "measures=[]" in music
    assert "complete measure's written durations" in music
    assert "For a rest event, set staff_step, accidental, and tie to null" in music
    assert "never a coordinate array" in chemistry
    assert "specific atoms[] or bonds[] box outside scope" in chemistry
    assert "explicitly drawn closed ring" in chemistry
    assert "answer_quote and reported" in geometry


def test_missing_specialist_environment_fails_before_spending_model_calls(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    from pixelogue import specialist_evaluation

    image, root = image_artifact
    case = SpecialistEvaluationCase(
        case_id="missing-chemistry",
        image=image.model_copy(update={"purpose": SourcePurpose.EVALUATION}),
        task_id="chemical_structure_reading",
        domain="explicit-atoms",
        scope_id="molecule",
        view_id=image.full_view.view_id,
        visible_scope="whole molecule",
        public_parameters=(),
        target_language="en",
        public_history=(),
        question="Read the visible molecule",
        candidate_answer='{"smiles":"CO"}',
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

    class NeverCalledClient:
        def __init__(self, *args, **kwargs):
            self.client = self

        def invoke(self, *args, **kwargs):
            raise AssertionError("Missing computation environment must stop before model calls")

        def close(self):
            pass

    monkeypatch.setattr(specialist_evaluation, "VllmClient", NeverCalledClient)
    monkeypatch.setattr(
        ValidatorRegistration,
        "environment_error",
        lambda _: "specialist lock environment is not installed",
    )
    output = tmp_path / "evaluation"
    stats = run_specialist_evaluation((case,), config, root, output, "missing-environment")
    assert stats["failed"] == 1 and stats["completed"] == 0
    (result_file,) = (output / "results").glob("*.json")
    result = SpecialistEvaluationResult.model_validate_json(result_file.read_text())
    assert result.status == "FAILED" and result.verdict is None
    assert result.response_artifact_hashes is None
    assert result.error is not None and "not installed" in result.error


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


def _manifest(
    results: tuple[SpecialistEvaluationResult, ...],
) -> tuple[CalibrationManifest, dict[str, Any]]:
    return build_calibration_manifest(results, tuple(result.case_id for result in results))


def test_complete_confirmation_group_computes_model_bound_certificate() -> None:
    results = (
        *(_result(index, positive=True) for index in range(20)),
        *(_result(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
    )
    manifest, report = _manifest(results)
    assert report["eligible"] == 1
    assert manifest.certificates[0].positive_images == 20
    assert manifest.certificates[0].negative_images == 60


def test_failed_calls_hold_the_whole_confirmation_group_pending() -> None:
    manifest, report = _manifest(
        (
            _result(0, positive=True),
            _result(1, positive=False, status="FAILED"),
        )
    )
    assert manifest.certificates == ()
    assert report["certificates"] == 0
    assert {item["reason"] for item in report["pending"]} == {
        "transport failed",
        "incomplete_confirmation_group",
    }
    assert report["incomplete_confirmation_groups"] == 1


def test_failed_confirmation_cannot_inflate_eligible_certificate() -> None:
    results = (
        *(_result(index, positive=True) for index in range(20)),
        *(_result(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
        _result(80, positive=True, status="FAILED"),
    )
    manifest, report = _manifest(results)
    assert not manifest.certificates
    assert report["eligible"] == 0
    assert report["incomplete_confirmation_groups"] == 1
    assert len(report["pending"]) == len(results)


def test_unlabeled_confirmation_cannot_be_dropped_from_certificate() -> None:
    results = (
        *(_result(index, positive=True) for index in range(20)),
        *(_result(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
        _result(80, positive=True).model_copy(update={"gold_accept": None}),
    )
    manifest, report = _manifest(results)
    assert not manifest.certificates
    assert report["eligible"] == 0
    assert {item["reason"] for item in report["pending"]} == {
        "missing_gold_label",
        "incomplete_confirmation_group",
    }


def test_missing_result_blocks_an_otherwise_eligible_confirmation_group() -> None:
    results = (
        *(_result(index, positive=True) for index in range(20)),
        *(_result(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
    )
    expected = (*[result.case_id for result in results], "case-80")
    manifest, report = build_calibration_manifest(results, expected)
    assert not manifest.certificates
    assert report["expected_cases"] == 81
    assert report["missing_results"] == 1
    assert {item["reason"] for item in report["pending"]} == {
        "missing_result",
        "incomplete_input_manifest",
    }
    with pytest.raises(ValueError, match="outside frozen input"):
        build_calibration_manifest(results, expected[:-2])


def test_failed_second_evaluator_resumes_from_first_saved_stage(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    from pixelogue import specialist_evaluation

    monkeypatch.setattr(ValidatorRegistration, "environment_error", lambda _: None)

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

    class TupleSource(StrictModel):
        values: tuple[str, ...]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.client = self

        def close(self):
            pass

        def invoke(self, stage, body, images, model, **kwargs):
            judge = int(kwargs["trial_id"].rsplit(":", 1)[1])
            calls[judge] += 1
            if judge and not second_online:
                raise ExecutionError("MODEL_OFFLINE", "Second evaluator is offline")
            return SimpleNamespace(
                value=TupleSource(values=("note",)),
                request_hash="a" * 64,
                response_hash="b" * 64,
                prompt_tokens=1,
                completion_tokens=1,
            )

    def fake_verify(instruction, payload, invoke, image_path):
        first = invoke("specialist_music_source", {"question": payload["question"]}, TupleSource, 0)
        second = invoke(
            "specialist_music_source", {"question": payload["question"]}, TupleSource, 1
        )
        assert first.values == second.values == ("note",)
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
    ("failure_reason", "counter", "task_id", "stage", "uses_json_fallback"),
    [
        (
            "MODEL_SCHEMA_MISMATCH",
            "schema_retries",
            "music_notation_reading",
            "specialist_music_source",
            False,
        ),
        (
            "MODEL_FINISH_REASON",
            "finish_retries",
            "music_notation_reading",
            "specialist_music_source",
            False,
        ),
        (
            "MODEL_FINISH_REASON",
            "finish_retries",
            "chemical_structure_reading",
            "specialist_chemistry_source",
            True,
        ),
        (
            "MODEL_WHITESPACE_RUNAWAY",
            "finish_retries",
            "music_notation_reading",
            "specialist_music_source",
            False,
        ),
        (
            "MODEL_WHITESPACE_RUNAWAY",
            "finish_retries",
            "chemical_structure_reading",
            "specialist_chemistry_source",
            True,
        ),
    ],
)
def test_invalid_specialist_reading_gets_one_bounded_retry(
    tmp_path: Path,
    image_artifact,
    monkeypatch,
    failure_reason: str,
    counter: str,
    task_id: str,
    stage: str,
    uses_json_fallback: bool,
) -> None:
    from pixelogue import specialist_evaluation

    monkeypatch.setattr(ValidatorRegistration, "environment_error", lambda _: None)

    image, root = image_artifact
    image = image.model_copy(update={"purpose": SourcePurpose.EVALUATION})
    case = SpecialistEvaluationCase(
        case_id="retry-case",
        image=image,
        task_id=task_id,
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
    fallback_modes: list[bool] = []

    class FakeClient:
        def __init__(self, *args, **kwargs):
            self.client = self

        def close(self):
            pass

        def invoke(self, stage, body, images, model, **kwargs):
            feedback.append(kwargs.get("retry_feedback"))
            output_limits.append(kwargs["max_tokens"])
            fallback_modes.append(kwargs.get("json_object_fallback", False))
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
        invoke(stage, {"question": payload["question"]}, TextPayload, 0)
        invoke(stage, {"question": payload["question"]}, TextPayload, 1)
        return (
            SimpleNamespace(
                name=(
                    "chemical_graph_validator"
                    if task_id == "chemical_structure_reading"
                    else "music_notation_validator"
                ),
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
    elif failure_reason == "MODEL_WHITESPACE_RUNAWAY":
        assert "blank lines" in (feedback[1] or "")
    else:
        assert "positive extent" in (feedback[1] or "")
    assert len(feedback) == 3
    assert output_limits == [config.tasks.evidence_max_tokens] * 3
    assert fallback_modes == [False, uses_json_fallback, False]
