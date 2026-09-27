"""Specialist calibration retains unprocessed cases and certifies independent images only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pixelogue.config import load_config
from pixelogue.contracts import GateVerdict, SourcePurpose, TextPayload
from pixelogue.errors import ExecutionError
from pixelogue.specialist_evaluation import (
    SpecialistEvaluationCase,
    SpecialistEvaluationResult,
    build_calibration_manifest,
    run_specialist_evaluation,
)


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
