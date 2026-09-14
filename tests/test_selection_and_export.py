from __future__ import annotations

from itertools import combinations

import pytest

from pixelogue.contracts import (
    ConversationArtifact,
    InstructionCandidate,
    PublicMessage,
    SelectionCandidate,
    SourcePurpose,
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


def test_evaluation_image_can_never_be_training_output(image_artifact) -> None:
    with pytest.raises(AuditError) as caught:
        training_record(_conversation(image_artifact, SourcePurpose.EVALUATION))
    assert caught.value.reason == "EVALUATION_IMAGE_EXPORT"
