from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pixelogue.config import ModelEndpoint, load_config
from pixelogue.contracts import (
    AtomicClaim,
    ClaimInventory,
    EvidenceInventory,
    GateVerdict,
    InstructionSelection,
    QuestionFit,
    RubricVerdict,
    TextPayload,
)
from pixelogue.ledger import RequirementInventory, RequirementSpec
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore


class ScriptedClient:
    def __init__(
        self,
        endpoint: ModelEndpoint,
        *,
        reject_selection: bool = False,
        fail_first_rating: bool = False,
        requirement_text: str | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.reject_selection = reject_selection
        self.fail_first_rating = fail_first_rating
        self.requirement_text = requirement_text
        self.rating_failed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def invoke[OutputModel: BaseModel](
        self,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        response_model: type[OutputModel],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        bypass_cache: bool = False,
    ) -> ModelResponse:
        del images, max_tokens, temperature, seed, bypass_cache
        self.calls.append((stage, payload))
        value: BaseModel
        if response_model is EvidenceInventory:
            value = EvidenceInventory(
                image_id=payload["image_id"],
                capabilities=("visible_entity",),
                visible_scopes=("the blue region",),
                scope_limited=False,
                reason="A visible entity is available.",
            )
        elif response_model is InstructionSelection:
            value = InstructionSelection(
                status="NO_SUITABLE_CANDIDATE" if self.reject_selection else "SELECTED",
                candidate_id=None
                if self.reject_selection
                else payload["candidates"][0]["candidate_id"],
                reason="Scripted selection result.",
            )
        elif response_model is TextPayload:
            value = TextPayload(
                status="OK",
                text="What color is the visible region?"
                if stage == "question_generation"
                else "Blue.",
            )
        elif response_model is QuestionFit:
            value = QuestionFit(
                local_anchor=GateVerdict.MET,
                operation_coherent=GateVerdict.MET,
                useful_request=GateVerdict.MET,
                reason="The question is visibly grounded.",
            )
        elif response_model is RequirementInventory:
            text = self.requirement_text
            start = payload["question"].find(text) if text is not None else 0
            value = RequirementInventory(
                requirements=()
                if text is None
                else (
                    RequirementSpec(
                        kind="content",
                        text=text,
                        lifetime="current_turn",
                        source_message_id=payload["question_message_id"],
                        start=start,
                        end=start + len(text),
                    ),
                ),
                coverage="MET",
                reason="The question has no separate explicit constraints.",
            )
        elif response_model is ClaimInventory:
            answer = payload["candidate_answer"]
            value = ClaimInventory(
                claims=(
                    AtomicClaim(
                        claim_id="claim",
                        text=answer,
                        source_message_id=payload["candidate_answer_message_id"],
                        start=0,
                        end=len(answer),
                    ),
                ),
                coverage=GateVerdict.MET,
                reason="The one factual claim is covered.",
            )
        elif response_model is RubricVerdict:
            if self.fail_first_rating and not self.rating_failed:
                self.rating_failed = True
                value = RubricVerdict(verdict="NOT_MET", reason="Repair this answer.")
            else:
                value = RubricVerdict(verdict="MET", reason="The criterion is satisfied.")
        else:
            raise AssertionError(f"Unexpected response model: {response_model}")
        return ModelResponse(
            value=value,
            request_hash="1" * 64,
            response_hash="2" * 64,
            prompt_tokens=1,
            completion_tokens=1,
        )


def _coordinator(
    tmp_path: Path,
    reject_selection: bool = False,
    fail_first_rating: bool = False,
    requirement_text: str | None = None,
    disagree_on_requirements: bool = False,
):
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "test", require_local_wal=False)
    store.initialize_run("test", config.config_hash, config.profile)
    selector = ScriptedClient(
        config.models.active_selector_endpoint,
        reject_selection=reject_selection,
    )
    generator_a = ScriptedClient(
        config.models.generator_a,
        fail_first_rating=fail_first_rating,
        requirement_text=requirement_text,
    )
    generator_b = ScriptedClient(
        config.models.generator_b,
        fail_first_rating=fail_first_rating,
        requirement_text="visible region" if disagree_on_requirements else requirement_text,
    )
    coordinator = SynthesisCoordinator(
        config,
        "test",
        store,
        selector,
        generator_a,
        generator_b,
    )
    return coordinator, store, selector, generator_a, generator_b


def test_selection_rejection_stops_before_question_or_answer(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, generator_b = _coordinator(
        tmp_path, reject_selection=True
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    stages = [stage for client in (generator_a, generator_b) for stage, _ in client.calls]
    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert "answer_generation" not in stages
    assert "question_generation" not in stages
    assert [stage for stage, _ in selector.calls] == ["instruction_selection"]


def test_successful_generation_uses_full_history_and_dual_blind_judges(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, generator_b = _coordinator(tmp_path)
    try:
        conversation = coordinator.synthesize_image(image, root)
        verification = store.verify()
    finally:
        store.close()

    assert conversation.status == "QUALITY_CANDIDATE"
    assert 2 <= len(conversation.turns) <= 6
    assert all(turn.status == "COMMITTED" for turn in conversation.turns)
    assert {turn.generation_model for turn in conversation.turns} == {conversation.generation_model}
    assert verification["turn_commits"] == len(conversation.turns)
    for turn in conversation.turns:
        prior = 2 * (turn.turn_index - 1)
        selector_payload = selector.calls[turn.turn_index - 1][1]
        assert len(selector_payload["public_history"]) == prior
        assert "candidate_answer" not in selector_payload
    for judge in (generator_a, generator_b):
        assert any(stage == "question_fit" for stage, _ in judge.calls)
        assert any(stage == "rubric_item" for stage, _ in judge.calls)
        assert all("other_judge" not in payload for _, payload in judge.calls)


def test_answer_repair_uses_generator_and_repeats_all_evaluation(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path, fail_first_rating=True)
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    generation_client = next(
        client
        for client in (generator_a, generator_b)
        if any(stage == "answer_generation" for stage, _ in client.calls)
    )
    other_client = generator_b if generation_client is generator_a else generator_a
    assert conversation.status == "QUALITY_CANDIDATE"
    assert sum(stage == "answer_repair" for stage, _ in generation_client.calls) == 1
    assert all(stage != "answer_repair" for stage, _ in other_client.calls)
    assert sum(stage == "claim_inventory" for stage, _ in generator_a.calls) >= 2
    assert sum(stage == "claim_inventory" for stage, _ in generator_b.calls) >= 2


def test_completed_conversation_resumes_without_model_calls(tmp_path: Path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, generator_b = _coordinator(tmp_path)
    try:
        first = coordinator.synthesize_image(image, root)
        selector.calls.clear()
        generator_a.calls.clear()
        generator_b.calls.clear()
        resumed = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert resumed == first
    assert not selector.calls and not generator_a.calls and not generator_b.calls


def test_public_requirement_is_fixed_before_answer_and_rated_separately(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        requirement_text="color",
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert conversation.status == "QUALITY_CANDIDATE"
    assert all(turn.requirements for turn in conversation.turns)
    generation_client = next(
        client
        for client in (generator_a, generator_b)
        if any(stage == "answer_generation" for stage, _ in client.calls)
    )
    generation_stages = [stage for stage, _ in generation_client.calls]
    assert generation_stages.index("requirement_extraction") < generation_stages.index(
        "answer_generation"
    )
    for client in (generator_a, generator_b):
        requirement_payloads = [
            payload
            for stage, payload in client.calls
            if stage == "rubric_item" and "target_requirement" in payload
        ]
        assert requirement_payloads


def test_requirement_disagreement_abstains_before_answer(tmp_path: Path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        requirement_text="color",
        disagree_on_requirements=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert conversation.status == "ABSTAINED"
    assert not conversation.turns
    assert all(
        stage != "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )
