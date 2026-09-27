"""Answer presentation trials freeze inputs and preserve paired, missing outcomes."""

from __future__ import annotations

from pathlib import Path

import pytest

from pixelogue.config import load_config
from pixelogue.contracts import ImageArtifact, ImageView, SourcePurpose
from pixelogue.errors import ExternalInputError
from pixelogue.io import write_json
from pixelogue.research_exposure import (
    ExposureCase,
    ExposureResult,
    FiveQuestionRating,
    build_exposure_plan,
    exposure_report,
    freeze_exposure_plan,
)


def _case() -> ExposureCase:
    sha = "a" * 64
    image = ImageArtifact(
        image_id="image",
        source_id="source",
        purpose=SourcePurpose.TRAINING,
        raw_sha256=sha,
        canonical_pixel_sha256=sha,
        source_group_ids=("group",),
        visual_group_id="group",
        full_view=ImageView(
            view_id="view",
            relative_path="view.png",
            encoded_sha256=sha,
            pixel_sha256=sha,
            width=100,
            height=100,
            media_type="image/png",
        ),
    )
    return ExposureCase(
        case_id="case-1",
        image=image,
        target_language="en",
        public_history=(),
        selected_instruction={"task_id": "count"},
        question="How many circles?",
        plausible_answer="Three.",
        incorrect_answer="Seventeen.",
        gold_question_admissible=False,
        gold_label_provenance="independent human consensus",
    )


def _rating(verdict: str) -> FiveQuestionRating:
    return FiveQuestionRating.model_validate(
        {
            "grounding": verdict,
            "operation_match": "MET",
            "answerability": "MET",
            "nonredundancy": "MET",
            "naturalness": "MET",
            "reason": "Independent judgment",
        }
    )


def test_frozen_trials_have_distinct_calls_and_paired_missing_report(tmp_path: Path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    plan = build_exposure_plan((_case(),), config, "generator_a", 1, 17)
    assert len(plan.trials) == len({trial.trial_id for trial in plan.trials}) == 3
    freeze_exposure_plan(tmp_path / "plan.json", plan)
    freeze_exposure_plan(tmp_path / "plan.json", plan)
    changed = build_exposure_plan((_case(),), config, "generator_a", 1, 18)
    with pytest.raises(ExternalInputError):
        freeze_exposure_plan(tmp_path / "plan.json", changed)
    for trial in plan.trials:
        if trial.condition == "incorrect":
            continue
        write_json(
            tmp_path / "results" / f"{trial.trial_id}.json",
            ExposureResult(
                trial_id=trial.trial_id,
                status="COMPLETE",
                rating=_rating("MET" if trial.condition == "plausible" else "NOT_MET"),
                error=None,
                request_hash="b" * 64,
                response_hash="c" * 64,
                prompt_tokens=12,
                completion_tokens=6,
                duration_ms=10,
                cost_usd=None,
            ),
        )
    report = exposure_report(plan, tmp_path)
    assert report["paired_plausible_minus_hidden"] == {"pairs": 1, "mean": 1.0}
    assert report["aias_invalid_questions"] == {"pairs": 1, "mean": 1.0}
    assert report["conditions"]["incorrect"]["missing"] == 1
    assert report["cost_usd"] is None
