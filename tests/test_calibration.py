"""Contract checks for independent, model-bound specialist calibration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelogue.calibration import (
    CalibrationManifest,
    CalibrationObservation,
    calibrate_observations,
    lower_positive_accept_bound,
    model_calibration_lock,
    upper_false_accept_bound,
)
from pixelogue.catalog import task_catalog
from pixelogue.config import load_config
from pixelogue.io import write_json
from pixelogue.task_registry import registration
from pixelogue.task_runtime import certified_domains, unavailable_reasons


def _observation(index: int, *, positive: bool, verdict: str = "MET") -> CalibrationObservation:
    validator = registration("music_notation_validator")
    assert validator is not None
    return CalibrationObservation.model_validate(
        {
            "case_id": f"case-{index}",
            "image_group_id": f"image-{index}",
            "task_id": "music_notation_reading",
            "domain": "printed_monophonic_v1",
            "split": "confirmation",
            "gold_accept": positive,
            "verdict": verdict,
            "model_locks": ("a" * 64, "b" * 64),
            "validator_version": validator.version,
            "response_artifact_hashes": ("c" * 64, "d" * 64),
        }
    )


def test_calibration_requires_enough_independent_positive_and_negative_evidence() -> None:
    positives = [_observation(index, positive=True) for index in range(20)]
    negatives = [_observation(index + 20, positive=False, verdict="NOT_MET") for index in range(60)]
    certificate = calibrate_observations((*positives, *negatives))
    assert certificate.eligible
    assert certificate.false_accept_upper_95 < 0.05
    assert certificate.positive_accept_lower_95 > 0.80
    assert not calibrate_observations((*positives, *negatives[:10])).eligible


def test_unknown_positive_lowers_acceptance_and_false_acceptance_blocks() -> None:
    positives = [_observation(index, positive=True) for index in range(20)]
    negatives = [_observation(index + 20, positive=False, verdict="NOT_MET") for index in range(60)]
    assert not calibrate_observations(
        (*positives[:-1], _observation(19, positive=True, verdict="UNKNOWN"), *negatives)
    ).eligible
    assert not calibrate_observations(
        (*positives, _observation(20, positive=False), *negatives[1:])
    ).eligible


def test_calibration_rejects_repeated_image_groups_and_development_data() -> None:
    positive = _observation(0, positive=True)
    duplicate = _observation(1, positive=False, verdict="NOT_MET").model_copy(
        update={"image_group_id": positive.image_group_id}
    )
    with pytest.raises(ValueError, match="image groups"):
        calibrate_observations((positive, duplicate))
    with pytest.raises(ValueError, match="Confirmation"):
        calibrate_observations(
            (positive, _observation(1, positive=False).model_copy(update={"split": "development"}))
        )


def test_calibration_manifest_recomputes_conclusion() -> None:
    records = (
        *(_observation(index, positive=True) for index in range(20)),
        *(_observation(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
    )
    certificate = calibrate_observations(records)
    CalibrationManifest(observations=records, certificates=(certificate,))
    with pytest.raises(ValidationError, match="differ"):
        CalibrationManifest(
            observations=records,
            certificates=(certificate.model_copy(update={"eligible": False}),),
        )


def test_exact_one_sided_bounds_handle_extremes() -> None:
    assert upper_false_accept_bound(0, 60) == pytest.approx(1 - 0.05 ** (1 / 60))
    assert lower_positive_accept_bound(20, 20) == pytest.approx(0.05 ** (1 / 20))
    assert upper_false_accept_bound(60, 60) == 1.0
    assert lower_positive_accept_bound(0, 20) == 0.0


def test_certificate_is_bound_to_active_models_and_manifest_content(tmp_path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    locks = (
        model_calibration_lock(config.models.generator_a),
        model_calibration_lock(config.models.generator_b),
    )
    records = (
        *(_observation(index, positive=True) for index in range(20)),
        *(_observation(index + 20, positive=False, verdict="NOT_MET") for index in range(60)),
    )
    records = tuple(item.model_copy(update={"model_locks": locks}) for item in records)
    manifest = CalibrationManifest(
        observations=records,
        certificates=(calibrate_observations(records),),
    )
    path = tmp_path / "calibration.json"
    write_json(path, manifest)
    settings = config.tasks.model_copy(
        update={
            "enabled_extensions": ("music_notation_reading",),
            "calibration_manifest": path,
        }
    )
    task = next(task for task in task_catalog().tasks if task.id == "music_notation_reading")
    assert certified_domains(task, settings, config.models) == ("printed_monophonic_v1",)
    assert (
        certified_domains(task, settings, load_config(Path("configs/standard.yaml")).models) == ()
    )
    assert unavailable_reasons(task, settings, config.models) == ()
    original_hash = config.model_copy(update={"tasks": settings}).config_hash
    changed_records = records[:-1] + (records[-1].model_copy(update={"verdict": "UNKNOWN"}),)
    write_json(
        path,
        CalibrationManifest(
            observations=changed_records,
            certificates=(calibrate_observations(changed_records),),
        ),
    )
    assert config.model_copy(update={"tasks": settings}).config_hash != original_hash


def test_specialist_selection_uses_its_registered_validator_version(tmp_path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    locks = (
        model_calibration_lock(config.models.generator_a),
        model_calibration_lock(config.models.generator_b),
    )
    current = registration("ui_action_validator")
    assert current is not None
    records = tuple(
        _observation(
            index, positive=index < 20, verdict="MET" if index < 20 else "NOT_MET"
        ).model_copy(
            update={
                "task_id": "ui_action_specification",
                "domain": "static-text-click",
                "model_locks": locks,
                "validator_version": current.version,
            }
        )
        for index in range(80)
    )
    path = tmp_path / "ui-calibration.json"
    write_json(
        path,
        CalibrationManifest(
            observations=records,
            certificates=(calibrate_observations(records),),
        ),
    )
    settings = config.tasks.model_copy(
        update={"enabled_extensions": ("ui_action_specification",), "calibration_manifest": path}
    )
    task = next(task for task in task_catalog().tasks if task.id == "ui_action_specification")
    assert certified_domains(task, settings, config.models) == ("static-text-click",)
    assert unavailable_reasons(task, settings, config.models) == ()
    outdated = tuple(item.model_copy(update={"validator_version": "4"}) for item in records)
    write_json(
        path,
        CalibrationManifest(
            observations=outdated,
            certificates=(calibrate_observations(outdated),),
        ),
    )
    assert certified_domains(task, settings, config.models) == ()
    assert "no calibration for the active models and validator" in unavailable_reasons(
        task, settings, config.models
    )
