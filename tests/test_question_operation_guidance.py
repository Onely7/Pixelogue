from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from pixelogue.catalog import task_catalog
from pixelogue.config import ModelEndpoint, load_config
from pixelogue.contracts import ImageArtifact, InstructionCandidate, build_history_snapshot
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.prompts import validate_stage_payload
from pixelogue.serialization import canonical_hash
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_evidence import ImageRegion, PublicParameter
from pixelogue.task_runtime import (
    operation_contract,
    question_operation_contract,
    selector_candidate,
)


def candidate(
    task_id: str = "object_identification",
    *,
    label: str = "PRIVATE_CATEGORY_SENTINEL",
    profile: Literal["normal", "limitation", "false_premise"] = "normal",
) -> InstructionCandidate:
    task = next(task for task in task_catalog().tasks if task.id == task_id)
    return InstructionCandidate(
        candidate_id="candidate-1",
        task_id=task.id,
        family=task.family,
        profile=profile,
        visible_scope=f"A visible {label} on the left.",
        instruction_summary=f"Name the {label}.",
        required_capabilities=task.required_capabilities,
        catalog_version="7.0",
        scope_id="scope-1",
        view_id="view-1",
        scope_region=ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0),
        target_region=ImageRegion(left=0.1, top=0.2, right=0.4, bottom=0.8),
        public_parameters=(
            PublicParameter(
                name="target",
                value=label,
                origin="image",
                evidence_refs=("private-observation",),
            ),
        ),
        evidence_refs=("private-observation",),
        verification_contracts=(
            task.verification_contracts if profile == "normal" else ("dual_visual_review",)
        ),
    )


@pytest.mark.parametrize("task_id", ["object_identification", "attribute_lookup", "entity_count"])
def test_baseline_question_contract_preserves_the_existing_public_contract(task_id: str) -> None:
    selected = candidate(task_id)

    assert question_operation_contract(selected) == operation_contract(selected)
    assert question_operation_contract(selected, guidance="baseline") == operation_contract(
        selected
    )


def test_identification_guidance_exposes_operation_without_private_category_or_evidence() -> None:
    selected = candidate()

    public = question_operation_contract(selected, guidance="object_identification_v1")

    serialized = json.dumps(public)
    assert "PRIVATE_CATEGORY_SENTINEL" not in serialized
    assert "private-observation" not in serialized
    assert all(parameter["name"] != "target" for parameter in public["public_parameters"])
    assert selected.target_region is not None
    assert public["target_region"] == selected.target_region.model_dump()
    guidance = public["question_contract"]
    assert "visible category" in guidance["requested_operation"]
    assert "finest category" in guidance["requested_granularity"]
    assert "Generic person categories are permitted" in guidance["requested_granularity"]
    assert "non-category traits" in guidance["target_locator"]
    assert "text=null" in guidance["unsupported_question"]


def test_identification_guidance_is_independent_of_the_private_answer_binding() -> None:
    first = candidate(label="PRIVATE_ANSWER_A")
    second = candidate(label="PRIVATE_ANSWER_B")

    first_public = question_operation_contract(first, guidance="object_identification_v1")
    second_public = question_operation_contract(second, guidance="object_identification_v1")

    assert first_public == second_public
    assert canonical_hash(first_public) == canonical_hash(second_public)


def test_question_guidance_does_not_change_selector_or_judge_operation_contracts() -> None:
    selected = candidate()
    judge_contract = operation_contract(selected)
    selector_contract = selector_candidate(selected)

    generated_question_contract = question_operation_contract(
        selected, guidance="object_identification_v1"
    )

    assert operation_contract(selected) == judge_contract
    assert selector_candidate(selected) == selector_contract
    assert "question_contract" not in judge_contract
    assert "question_contract" not in selector_contract
    assert generated_question_contract != judge_contract
    assert canonical_hash(generated_question_contract) != canonical_hash(judge_contract)


@pytest.mark.parametrize(
    ("task_id", "profile"),
    [
        ("attribute_lookup", "normal"),
        ("entity_count", "normal"),
        ("object_identification", "limitation"),
        ("object_identification", "false_premise"),
    ],
)
def test_identification_experiment_preserves_other_operations_and_profiles(
    task_id: str, profile: Literal["normal", "limitation", "false_premise"]
) -> None:
    selected = candidate(task_id, profile=profile)

    assert question_operation_contract(
        selected, guidance="object_identification_v1"
    ) == operation_contract(selected)


def test_identification_experiment_preserves_legacy_operation_contracts() -> None:
    selected = InstructionCandidate(
        candidate_id="legacy-1",
        task_id="object_identification",
        family="visual_description",
        visible_scope="the left object",
        instruction_summary="Identify the object",
        required_capabilities=("visible_entity",),
    )

    assert question_operation_contract(
        selected, guidance="object_identification_v1"
    ) == operation_contract(selected)


def test_unknown_guidance_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported question operation guidance"):
        question_operation_contract(candidate(), guidance="unknown")  # ty: ignore[invalid-argument-type]


def test_coordinator_routes_identification_guidance_only_to_question_generation(
    tmp_path: Path, image_artifact: tuple[ImageArtifact, Path]
) -> None:
    class RecordingClient:
        def __init__(self, endpoint: ModelEndpoint) -> None:
            self.endpoint = endpoint
            self.calls: list[tuple[str, dict[str, Any]]] = []

        def invoke(
            self,
            stage: str,
            payload: dict[str, Any],
            images: tuple[ModelImage, ...],
            response_model: type[BaseModel],
            **kwargs: Any,
        ) -> ModelResponse:
            validate_stage_payload(stage, payload)
            self.calls.append((stage, json.loads(json.dumps(payload))))
            responses = {
                "instruction_selection": {
                    "candidate_id": "candidate-1",
                    "reason": "The bound operation is suitable.",
                },
                "question_generation": {"text": "What kind of object appears on the left?"},
                "question_intent": {
                    "task_id": "object_identification",
                    "reason": "The question asks for a visible object category.",
                },
                "question_fit": {
                    "local_anchor": "MET",
                    "operation_coherent": "MET",
                    "useful_request": "MET",
                    "reason": "The bound object is the question's target.",
                },
                "answer_generation": {"text": "A rectangle."},
                "holistic_review": {"verdict": "MET", "reason": "The turn is supported."},
            }
            return ModelResponse(
                value=response_model.model_validate_json(json.dumps(responses[stage])),
                request_hash=canonical_hash({"stage": stage, "payload": payload}),
                response_hash=canonical_hash(responses[stage]),
                prompt_tokens=1,
                completion_tokens=1,
            )

    image, root = image_artifact
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={"question_operation_guidance": "object_identification_v1"}
            )
        }
    )
    selected = candidate().model_copy(update={"view_id": image.full_view.view_id})
    snapshot = build_history_snapshot("guidance-conversation", 1, ())
    model_image = ModelImage(
        view_id=image.full_view.view_id,
        path=root / image.full_view.relative_path,
        encoded_sha256=image.full_view.encoded_sha256,
        media_type=image.full_view.media_type,
    )
    image_views = [SynthesisCoordinator._view_metadata(image)]
    selector = RecordingClient(config.models.active_selector_endpoint)
    generator_a = RecordingClient(config.models.generator_a)
    generator_b = RecordingClient(config.models.generator_b)
    store = RunStore(tmp_path / "runs", "question-guidance")
    try:
        store.initialize_run("question-guidance", config.config_hash, config.profile)
        coordinator = SynthesisCoordinator(
            config, "question-guidance", store, selector, generator_a, generator_b
        )
        chosen = coordinator._select_instruction(
            (selected,), snapshot, "en", image_views, model_image, 1
        )
        assert chosen is not None
        result = coordinator._attempt_turn(
            chosen, snapshot, generator_a, "en", model_image, image_views, ()
        )
    finally:
        store.close()

    assert result.status == "COMMITTED"
    assert result.turn is not None
    assert result.turn.instruction == selected
    calls = selector.calls + generator_a.calls + generator_b.calls
    stages = [stage for stage, _ in calls]
    assert stages.count("instruction_selection") == 1
    assert stages.count("question_generation") == 1
    assert stages.count("answer_generation") == 1
    assert stages.count("question_intent") == 2
    assert stages.count("question_fit") == 2
    assert stages.count("holistic_review") == 2
    public_contract = operation_contract(selected)
    guided_contract = question_operation_contract(selected, guidance="object_identification_v1")
    for stage, payload in calls:
        serialized = json.dumps(payload)
        assert "PRIVATE_CATEGORY_SENTINEL" not in serialized
        assert "private-observation" not in serialized
        if stage == "question_generation":
            assert payload["selected_instruction"] == guided_contract
            assert payload["selected_instruction"]["question_contract"]["version"] == (
                "object_identification_v1"
            )
        else:
            assert "question_contract" not in serialized
        if stage == "question_fit":
            assert payload["selected_instruction"] == public_contract
        if "expected_operation" in payload:
            assert payload["expected_operation"] == public_contract
