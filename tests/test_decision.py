from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from pixelogue.decision import DecisionProbability, DecisionRequest, DecisionResult
from pixelogue.task_evidence import ImageRegion

REVISION = "a" * 40


def decision_request(tmp_path: Path) -> DecisionRequest:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image_path)
    return DecisionRequest(
        request_id="scope-condition-1",
        trial_id="independent-trial-1",
        image_id="image-1",
        image_path=str(image_path),
        image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        view_id="full-view",
        region=ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0),
        question="Is the requested surface color visible in the stated scope?",
        public_context=("The target is the visible object in the center.",),
    )


def decision_result(request: DecisionRequest, **changes: object) -> DecisionResult:
    fields = {
        "request_id": request.request_id,
        "request_hash": request.identity("model/test", REVISION),
        "model_repository": "model/test",
        "model_revision": REVISION,
        "verdict": "MET",
        "probabilities": (
            DecisionProbability(label="MET", probability=0.7),
            DecisionProbability(label="NOT_MET", probability=0.2),
            DecisionProbability(label="UNKNOWN", probability=0.1),
        ),
        "elapsed_seconds": 0.25,
        **changes,
    }
    return DecisionResult.model_validate(fields)


def test_probability_mapping_cannot_mutate_stored_result(tmp_path: Path) -> None:
    result = decision_result(decision_request(tmp_path))
    result.probability_map["MET"] = 0.0
    assert result.confidence == 0.7
    assert result.prompt_tokens is None
    assert result.completion_tokens is None


@pytest.mark.parametrize(
    ("labels", "probabilities", "verdict"),
    [
        (("MET", "MET", "UNKNOWN"), (0.7, 0.2, 0.1), "MET"),
        (("MET", "NOT_MET", "UNKNOWN"), (0.7, 0.1, 0.1), "MET"),
        (("MET", "NOT_MET", "UNKNOWN"), (0.7, 0.2, 0.1), "NOT_MET"),
        (("MET", "NOT_MET", "UNKNOWN"), (float("nan"), 0.2, 0.1), "MET"),
        (("MET", "NOT_MET", "UNKNOWN"), (float("inf"), 0.2, 0.1), "MET"),
        (("MET", "NOT_MET", "UNKNOWN"), (-0.1, 0.2, 0.9), "UNKNOWN"),
    ],
    ids=["duplicate-label", "not-normalized", "wrong-verdict", "nan", "inf", "negative"],
)
def test_invalid_decision_scores_are_not_coerced(
    tmp_path: Path, labels: tuple[str, ...], probabilities: tuple[float, ...], verdict: str
) -> None:
    with pytest.raises(ValidationError):
        decision_result(
            decision_request(tmp_path),
            verdict=verdict,
            probabilities=tuple(
                {"label": label, "probability": probability}
                for label, probability in zip(labels, probabilities, strict=True)
            ),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"trial_id": "another-independent-trial"},
        {"public_context": ("The earlier public turn changed the condition.",)},
        {"view_id": "another-view"},
        {"region": ImageRegion(left=0.2, top=0.2, right=0.8, bottom=0.8)},
        {"contract_version": "finite-decision-v2"},
    ],
    ids=["trial", "history", "view", "region", "contract"],
)
def test_result_cannot_replay_for_changed_inputs(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    request = decision_request(tmp_path)
    result = decision_result(request)
    with pytest.raises(ValueError, match="does not belong"):
        result.validate_request(request.model_copy(update=changes))


def test_moving_same_image_does_not_invalidate_identity(tmp_path: Path) -> None:
    request = decision_request(tmp_path)
    assert request.identity("model/test", REVISION) == request.model_copy(
        update={"image_path": str(tmp_path / "moved.png")}
    ).identity("model/test", REVISION)
    assert request.identity("model/test", REVISION) != request.identity("model/test", "b" * 40)


@pytest.mark.parametrize(
    "forbidden", ["candidate_answer", "gold_label", "other_judge", "future_turns"]
)
def test_reference_and_forbidden_context_fields_are_rejected(
    tmp_path: Path, forbidden: str
) -> None:
    raw = decision_request(tmp_path).model_dump()
    raw[forbidden] = "MET"
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(raw)
