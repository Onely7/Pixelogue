from __future__ import annotations

from itertools import combinations

import pytest
from pydantic import ValidationError

from pixelogue.contracts import (
    ConversationArtifact,
    InstructionCandidate,
    PublicMessage,
    SelectionCandidate,
    SourcePurpose,
    TrainingContent,
    TurnArtifact,
    TurnRating,
)
from pixelogue.errors import AuditError
from pixelogue.export import training_record
from pixelogue.selection import (
    SelectionPolicy,
    audit_selection,
    bind_audit,
    freeze_pool,
    select_candidates,
)
from pixelogue.serialization import canonical_hash


def _candidate(index: int, language: str) -> SelectionCandidate:
    return SelectionCandidate(
        conversation_id=f"conversation-{index}",
        language=language,
        image_id=f"image-{index}",
        visual_group_id=f"group-{index}",
        task_family="observation_attribute" if index % 2 == 0 else "visible_count",
        semantic_family=f"semantic-{index}",
        pattern="normal/normal",
        purpose=SourcePurpose.TRAINING,
    )


def test_cp_sat_selection_matches_independent_audit() -> None:
    candidates = [_candidate(0, "en"), _candidate(1, "en"), _candidate(2, "ja")]
    policy = SelectionPolicy(
        target_count=2,
        language_quotas={"en": 1, "ja": 1},
        family_minimums={"observation_attribute": 1},
    )
    manifest = select_candidates(candidates, policy)
    report = audit_selection(candidates, manifest, policy)

    assert manifest.solver_status == "OPTIMAL"
    assert report.status == "MET"
    bound = bind_audit(manifest, report)
    assert bound.audit_report_hash == report.audit_hash
    assert bound.has_valid_audit_binding()
    assert not bound.model_copy(update={"selected_ids": ()}).has_valid_audit_binding()


def test_infeasible_and_tampered_selection_are_distinct() -> None:
    candidates = [_candidate(0, "en")]
    policy = SelectionPolicy(target_count=2, language_quotas={"en": 2})
    manifest = select_candidates(candidates, policy)
    assert manifest.solver_status == "INFEASIBLE"

    tampered = manifest.model_copy(
        update={"selected_ids": ("outside",), "solver_status": "FEASIBLE"}
    )
    report = audit_selection(candidates, tampered, policy)
    assert report.status == "NOT_MET"
    assert "SELECTED_ID_OUTSIDE_POOL" in report.violations
    with pytest.raises(AuditError):
        bind_audit(tampered, report)


def test_cp_sat_feasibility_matches_small_exhaustive_search() -> None:
    candidates = [
        _candidate(0, "en"),
        _candidate(1, "en"),
        _candidate(2, "ja"),
        _candidate(3, "ja"),
    ]
    policy = SelectionPolicy(
        target_count=2,
        language_quotas={"en": 1, "ja": 1},
        family_minimums={"observation_attribute": 1},
    )
    _, pool_hash = freeze_pool(candidates)
    exhaustive_feasible = False
    for subset in combinations(candidates, policy.target_count):
        selected_ids = tuple(item.conversation_id for item in subset)
        solver_trial = select_candidates(candidates, policy)
        trial = solver_trial.model_copy(
            update={"pool_hash": pool_hash, "selected_ids": selected_ids}
        )
        if audit_selection(candidates, trial, policy).status == "MET":
            exhaustive_feasible = True
            break
    solved = select_candidates(candidates, policy)
    assert (solved.solver_status in {"OPTIMAL", "FEASIBLE"}) == exhaustive_feasible


def _conversation(image_artifact, purpose: SourcePurpose) -> ConversationArtifact:
    image, _ = image_artifact
    image = image.model_copy(update={"purpose": purpose})
    instruction = InstructionCandidate(
        candidate_id="candidate",
        task_id="object_identification",
        family="observation_attribute",
        visible_scope="image",
        instruction_summary="Identify a visible object.",
        required_capabilities=("visible_entity",),
    )
    turns = []
    transcript = []
    for index in (1, 2):
        question = PublicMessage(
            message_id=f"q-{index}", turn_index=index, role="user", content="Question?"
        )
        answer = PublicMessage(
            message_id=f"a-{index}", turn_index=index, role="assistant", content="Answer."
        )
        turns.append(
            TurnArtifact(
                turn_index=index,
                instruction=instruction,
                question=question,
                answer=answer,
                history_hash=canonical_hash(transcript),
                generation_model="Qwen/Qwen3.8-27B",
                selector_model="Qwen/Qwen3.5-2B",
                rating=TurnRating(items=(), aggregate="PASS"),
                status="COMMITTED",
            )
        )
        transcript.extend((question.model_dump(mode="json"), answer.model_dump(mode="json")))
    return ConversationArtifact(
        conversation_id="conversation",
        image=image,
        target_language="en",
        generation_model="Qwen/Qwen3.8-27B",
        turns=tuple(turns),
        status="QUALITY_CANDIDATE",
    )


def test_training_export_places_image_once(image_artifact) -> None:
    record = training_record(_conversation(image_artifact, SourcePurpose.TRAINING))
    image_parts = [
        part for message in record.messages for part in message.content if part.type == "image"
    ]
    assert len(image_parts) == 1
    assert record.messages[0].role == "user"
    view = image_artifact[0].full_view
    assert (image_parts[0].image, image_parts[0].width, image_parts[0].height) == (
        view.relative_path,
        256,
        192,
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"type": "image", "image": "images/a.png", "width": 640},
        {"type": "image", "image": "images/a.png", "height": 480},
        {"type": "image", "image": "images/a.png", "width": 0, "height": 480},
        {"type": "text", "text": "A question?", "width": 640, "height": 480},
    ],
)
def test_training_parts_require_pixel_size_only_on_images(fields: dict) -> None:
    with pytest.raises(ValidationError):
        TrainingContent.model_validate(fields)


def test_evaluation_image_can_never_be_training_output(image_artifact) -> None:
    with pytest.raises(AuditError) as caught:
        training_record(_conversation(image_artifact, SourcePurpose.EVALUATION))
    assert caught.value.reason == "EVALUATION_IMAGE_EXPORT"


def test_v7_selection_uses_final_family_and_counts_all_committed_operations(image_artifact):
    from pixelogue.catalog import task_catalog
    from pixelogue.operations import candidate_from_conversation, summarize_operations

    conversation = _conversation(image_artifact, SourcePurpose.TRAINING)
    turns = []
    for turn, task_id in zip(
        conversation.turns, ("object_identification", "text_transcription"), strict=True
    ):
        task = next(task for task in task_catalog().tasks if task.id == task_id)
        instruction = InstructionCandidate(
            candidate_id=task_id,
            task_id=task.id,
            family=task.family,
            visible_scope="image",
            instruction_summary=task.definition_en,
            required_capabilities=task.required_capabilities,
            catalog_version="7.0",
            scope_id="image",
            view_id="view",
            evidence_refs=("evidence",),
            verification_contracts=task.verification_contracts,
        )
        turns.append(turn.model_copy(update={"instruction": instruction}))
    conversation = conversation.model_copy(update={"turns": tuple(turns)})
    assert candidate_from_conversation(conversation).task_family == "text_reading"
    counts = summarize_operations((conversation,))
    assert counts["committed_turns"] == {
        "7.0:object_identification": 1,
        "7.0:text_transcription": 1,
    }
    assert counts["primary_conversations"] == {"7.0:text_transcription": 1}
    public = training_record(conversation).model_dump_json()
    assert "catalog_version" not in public and "verification_contracts" not in public
    stopped = conversation.model_copy(
        update={"turns": (turns[0], turns[1].model_copy(update={"status": "REJECTED"}))}
    )
    assert summarize_operations((stopped,))["primary_conversations"] == {
        "7.0:object_identification": 1
    }
