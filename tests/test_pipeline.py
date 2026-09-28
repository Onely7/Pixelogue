from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from pixelogue.catalog import load_rubric_catalog
from pixelogue.config import EvaluationConfig, ModelEndpoint, load_config
from pixelogue.contracts import (
    AtomicClaim,
    ClaimExtraction,
    ClaimSpan,
    EvidenceInventory,
    GateVerdict,
    InstructionCandidate,
    InstructionSelection,
    PublicMessage,
    QuestionFit,
    QuestionIntent,
    RubricContext,
    RubricVerdict,
    TextPayload,
)
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import (
    applicable_rubric_items,
    has_natural_language_content,
    repeated_answered_request,
)
from pixelogue.export import training_record
from pixelogue.ledger import RequirementInventory, RequirementSpec
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.prompts import STAGE_INSTRUCTIONS
from pixelogue.rules import CountGroup, SetCheck, SetInventory, verify_set_inventories
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_evidence import (
    CandidateBinding,
    CandidateBindingReport,
    CandidateBindingsReport,
    CapabilityReport,
    EligibilityObservation,
    ImageRegion,
    PublicParameter,
    ScopedEvidenceReport,
    ScopeEvidenceReport,
    TargetReport,
)


class ScriptedClient:
    def __init__(
        self,
        endpoint: ModelEndpoint,
        *,
        reject_selection: bool = False,
        fail_first_rating: bool = False,
        requirement_text: str | None = None,
        concurrency_probe: ConcurrencyProbe | None = None,
        fail_first_schema: bool = False,
        repeat_question: bool = False,
        echo_question_prompt: bool = False,
        echo_answer_prompt: bool = False,
        unknown_capability: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.reject_selection = reject_selection
        self.fail_first_rating = fail_first_rating
        self.requirement_text = requirement_text
        self.concurrency_probe = concurrency_probe
        self.fail_first_schema = fail_first_schema
        self.repeat_question = repeat_question
        self.echo_question_prompt = echo_question_prompt
        self.echo_answer_prompt = echo_answer_prompt
        self.unknown_capability = unknown_capability
        self.rating_failed = False
        self.schema_failed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.retry_feedback: list[str | None] = []
        self.intent_by_question: dict[tuple[str, str], str] = {}

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
        retry_feedback: str | None = None,
    ) -> ModelResponse:
        del images, max_tokens, temperature, seed, bypass_cache
        self.retry_feedback.append(retry_feedback)
        if self.fail_first_schema and not self.schema_failed:
            self.schema_failed = True
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", "missing required field")
        if self.concurrency_probe is not None:
            self.concurrency_probe.enter()
            time.sleep(0.002)
            self.concurrency_probe.exit()
        self.calls.append((stage, payload))
        value: BaseModel
        if response_model is ScopedEvidenceReport:
            region = ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0)
            value = ScopedEvidenceReport(
                image_id=payload["image_id"],
                reason="A visible entity is available.",
                scopes=(
                    ScopeEvidenceReport(
                        scope_id="blue",
                        view_id=payload["image_views"][0]["view_id"],
                        public_description="the blue region",
                        region=region,
                        observations={
                            "unsupported_capability"
                            if self.unknown_capability
                            else "visible_entity": CapabilityReport(
                                evidence_id="entity",
                                verdict="MET",
                                region=region,
                                detail="A blue object is visible.",
                            ),
                        },
                    ),
                ),
            )
        elif response_model is CandidateBindingsReport:
            value = CandidateBindingsReport(
                bindings=tuple(
                    CandidateBindingReport(
                        candidate_id=candidate["candidate_id"],
                        target=TargetReport(
                            value=f"the blue region, aspect {len(payload['public_history'])}",
                            origin="instruction",
                        ),
                        public_parameters=(),
                        checks=tuple(
                            EligibilityObservation(check_id=key, verdict="MET", reason="Visible")
                            for key in candidate["eligibility_checks"]
                        ),
                        evidence_refs=("entity",),
                        estimated_answer_tokens=100,
                    )
                    for candidate in payload["candidates"]
                )
            )
        elif response_model is EvidenceInventory:
            value = EvidenceInventory(
                image_id=payload["image_id"],
                capabilities=("unsupported_capability",)
                if self.unknown_capability
                else ("visible_entity",),
                visible_scopes=("the blue region",),
                scope_limited=False,
                reason="A visible entity is available.",
            )
        elif response_model is InstructionSelection:
            value = InstructionSelection(
                candidate_id=None
                if self.reject_selection
                else payload["candidates"][0]["candidate_id"],
                reason="Scripted selection result.",
            )
        elif response_model is TextPayload:
            if stage == "question_generation" and self.echo_question_prompt:
                text = STAGE_INSTRUCTIONS["question_generation"]
            elif stage == "question_generation" and self.repeat_question:
                text = (
                    "What color is the visible region?"
                    if payload["turn_index"] == 1
                    else "  WHAT COLOR IS THE VISIBLE REGION？  "
                )
            elif stage == "question_generation":
                text = f"What color is visible region {payload['turn_index']}?"
            elif stage == "answer_generation" and self.echo_answer_prompt:
                text = STAGE_INSTRUCTIONS["answer_generation"]
            else:
                text = "Blue."
            value = TextPayload(
                text=text,
            )
            if stage == "question_generation" and text is not None:
                self.intent_by_question[(payload["image_views"][0]["view_id"], text)] = payload[
                    "selected_instruction"
                ]["task_id"]
        elif response_model is QuestionIntent:
            value = QuestionIntent(
                task_id=self.intent_by_question.get(
                    (payload["image_views"][0]["view_id"], payload["question"])
                ),
                reason="The public request matches the observed task.",
            )
        elif response_model is QuestionFit:
            value = QuestionFit(
                local_anchor="MET",
                operation_coherent="MET",
                useful_request="MET",
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
                extraction_complete=True,
                reason="The question has no separate explicit constraints.",
            )
        elif response_model is ClaimExtraction:
            answer = payload["candidate_answer"]
            value = ClaimExtraction(
                claims=(
                    ClaimSpan(
                        start_token=0,
                        end_token=len(SynthesisCoordinator._answer_tokens(answer)),
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


def test_structured_retry_receives_bounded_correction_feedback(tmp_path: Path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a, fail_first_schema=True)
    coordinator = SynthesisCoordinator(config, "retry", store, client, client, client)
    try:
        result = coordinator._invoke(
            client,
            "rubric_item",
            {},
            (),
            RubricVerdict,
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.verdict == "MET"
    assert client.retry_feedback == [
        None,
        (
            "The previous response failed schema validation. Return one complete JSON object "
            "with every required field, unique array items, and a short non-empty reason."
            " Required top-level fields: verdict, reason."
        ),
    ]


def test_semantic_contract_failure_retries_without_accepting_invalid_result(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "semantic-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    coordinator = SynthesisCoordinator(config, "semantic-retry", store, client, client, client)
    attempts = 0

    def validate(_result: RubricVerdict) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ExecutionError("CANDIDATE_PARAMETER_UNKNOWN", "Unknown public parameter")

    try:
        result = coordinator._invoke(
            client,
            "rubric_item",
            {},
            (),
            RubricVerdict,
            max_tokens=256,
            temperature=0.0,
            seed=1,
            post_validate=validate,
        )
    finally:
        store.close()

    assert result.verdict == "MET"
    assert attempts == 2
    assert client.retry_feedback[0] is None
    assert "parameter_contract" in (client.retry_feedback[1] or "")


def test_candidate_binding_requires_exact_target_parameter() -> None:
    with pytest.raises(ValueError, match="requires a target parameter"):
        CandidateBinding(
            candidate_id="candidate",
            public_parameters=(
                PublicParameter(name="target_binding", value="the object", origin="instruction"),
            ),
            checks=(),
            evidence_refs=("obs_1",),
            estimated_answer_tokens=10,
        )


def test_exhausted_candidate_binding_checks_abstain_without_accepting(
    tmp_path: Path, image_artifact, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    original = coordinator._invoke

    def invoke(client, stage, payload, images, model, **kwargs):
        if stage == "candidate_binding":
            raise ExecutionError(
                "CANDIDATE_CHECKS_MISMATCH", "Missing or unknown eligibility check"
            )
        return original(client, stage, payload, images, model, **kwargs)

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        conversation = coordinator.synthesize_image(image, root)
        private_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'binding-abstentions'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "ABSTAINED"
    assert not conversation.turns
    assert private_count == 1


def test_repeated_invalid_model_json_abstains_but_transport_failure_remains_error(
    tmp_path: Path, image_artifact, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path, evaluation_mode="holistic")
    original = coordinator._invoke
    reason = "MODEL_SCHEMA_MISMATCH"

    def invoke(client, stage, payload, images, model, **kwargs):
        if stage == "evidence_extraction":
            raise ExecutionError(reason, "DUPLICATE_JSON_KEY: readable_text")
        return original(client, stage, payload, images, model, **kwargs)

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        job = SynthesisJob(image=image, target_language="en", generator_role="generator_a")
        abstained = coordinator._synthesize_job(job, root)
        resumed = coordinator.synthesize_image(image, root, generator_role="generator_a")
        reason = "MODEL_HTTP_STATUS"
        failed = coordinator._synthesize_job(
            SynthesisJob(
                image=image.model_copy(update={"image_id": "f" * 64}),
                target_language="en",
                generator_role="generator_a",
            ),
            root,
        )
        abstention_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'model-output-abstentions'"
        ).fetchone()[0]
    finally:
        store.close()

    assert abstained.status == "ABSTAINED"
    assert not abstained.turns
    assert resumed == abstained
    assert failed.status == "ERROR"
    assert abstention_count == 1


class ConcurrencyProbe:
    """Track overlapping scripted model calls across test clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        """Record the start of one model call."""
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def exit(self) -> None:
        """Record the completion of one model call."""
        with self._lock:
            self.active -= 1


def _coordinator(
    tmp_path: Path,
    reject_selection: bool = False,
    fail_first_rating: bool = False,
    requirement_text: str | None = None,
    disagree_on_requirements: bool = False,
    concurrency_probe: ConcurrencyProbe | None = None,
    repeat_question: bool = False,
    echo_question_prompt: bool = False,
    echo_answer_prompt: bool = False,
    unknown_capability: bool = False,
    evaluation_mode: str = "detailed",
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={"evaluation": EvaluationConfig.model_validate({"mode": evaluation_mode})}
    )
    store = RunStore(tmp_path / "runs", "test", require_local_wal=False)
    store.initialize_run("test", config.config_hash, config.profile)
    selector = ScriptedClient(
        config.models.active_selector_endpoint,
        reject_selection=reject_selection,
        concurrency_probe=concurrency_probe,
    )
    generator_a = ScriptedClient(
        config.models.generator_a,
        fail_first_rating=fail_first_rating,
        requirement_text=requirement_text,
        concurrency_probe=concurrency_probe,
        repeat_question=repeat_question,
        echo_question_prompt=echo_question_prompt,
        echo_answer_prompt=echo_answer_prompt,
        unknown_capability=unknown_capability,
    )
    generator_b = ScriptedClient(
        config.models.generator_b,
        fail_first_rating=fail_first_rating,
        requirement_text="visible region" if disagree_on_requirements else requirement_text,
        concurrency_probe=concurrency_probe,
        repeat_question=repeat_question,
        echo_question_prompt=echo_question_prompt,
        echo_answer_prompt=echo_answer_prompt,
        unknown_capability=unknown_capability,
    )
    generator_b.intent_by_question = generator_a.intent_by_question
    coordinator = SynthesisCoordinator(
        config,
        "test",
        store,
        selector,
        generator_a,
        generator_b,
    )
    return coordinator, store, selector, generator_a, generator_b


def test_batch_synthesis_overlaps_images_and_preserves_input_order(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    probe = ConcurrencyProbe()
    coordinator, store, _, _, _ = _coordinator(tmp_path, concurrency_probe=probe)
    images = tuple(image.model_copy(update={"image_id": f"{index:064x}"}) for index in range(1, 5))
    jobs = tuple(
        SynthesisJob(
            image=item,
            target_language="en",
            generator_role="generator_a" if index % 2 else "generator_b",
        )
        for index, item in enumerate(images)
    )
    try:
        conversations = tuple(coordinator.synthesize_batch(jobs, root, max_workers=3))
        verification = store.verify()
    finally:
        store.close()

    assert tuple(item.image.image_id for item in conversations) == tuple(
        item.image_id for item in images
    )
    assert probe.peak >= 2
    assert verification["turn_commits"] == sum(
        len(item.turns) for item in conversations if item.status == "QUALITY_CANDIDATE"
    )


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


def test_normalized_repeated_question_stops_before_second_fit(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        repeat_question=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert len(conversation.turns) == 1
    assert rejection_count == 1
    assert sum(stage == "question_fit" for stage, _ in generator_a.calls) == 1
    assert sum(stage == "question_fit" for stage, _ in generator_b.calls) == 1


def test_repeated_answered_request_catches_observed_sheep_paraphrase() -> None:
    first_question = "What are the colors and shapes of the sheep visible in the foreground?"
    second_question = "What colors and shapes are visible in the sheep located in the foreground?"
    answer = (
        "The sheep in the foreground are primarily white and brown. Their shapes are "
        "quadrupedal with woolly bodies, curved horns, and four legs."
    )
    history = (
        PublicMessage(message_id="q1", turn_index=1, role="user", content=first_question),
        PublicMessage(message_id="a1", turn_index=1, role="assistant", content=answer),
    )
    assert repeated_answered_request(second_question, answer, history)
    assert not repeated_answered_request(second_question, "A distinct supported answer.", history)
    assert not repeated_answered_request(second_question, "white", history)


def test_repeated_substantial_answer_stops_before_second_review(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
    repeated_answer = "The visible region is blue and contains the same resolved object."
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, **kwargs):
            if stage == "answer_generation":
                return ModelResponse(
                    value=TextPayload(text=repeated_answer),
                    request_hash="request",
                    response_hash="response",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return _original(stage, payload, *args, **kwargs)

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        result = coordinator.synthesize_image(image, root)
        rejected = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()
    assert result.status == "REJECTED"
    assert len(result.turns) == 1
    assert rejected == 1
    assert (
        sum(
            stage == "holistic_review"
            for client in (generator_a, generator_b)
            for stage, _ in client.calls
        )
        == 2
    )


def test_blind_question_intent_rejects_wrong_operation_before_answer(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, **kwargs):
            if stage == "question_intent":
                assert "selected_instruction" not in payload
                assert "candidate_answer" not in payload
                assert len(payload["task_definitions"]) == 72
                return ModelResponse(
                    value=QuestionIntent(task_id="spatial_relation", reason="Wrong operation."),
                    request_hash="request",
                    response_hash="response",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return _original(stage, payload, *args, **kwargs)

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        result = coordinator.synthesize_image(image, root)
    finally:
        store.close()
    assert result.status == "REJECTED"
    assert all(
        stage != "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_private_prompt_echo_is_rejected_before_question_fit(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        echo_question_prompt=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert rejection_count == 1
    assert all(
        stage != "question_fit"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_private_prompt_echo_is_rejected_before_answer_rating(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        echo_answer_prompt=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert rejection_count == 1
    assert any(
        stage == "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )
    assert all(
        stage != "claim_inventory"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_unknown_evidence_capability_is_not_silently_ignored(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path, unknown_capability=True)
    try:
        with pytest.raises(ExecutionError) as caught:
            coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert caught.value.reason == "EVIDENCE_CAPABILITY_UNKNOWN"


def test_later_turn_does_not_imply_binding_or_strong_dependency() -> None:
    context = RubricContext(
        turn_index=2,
        profile="normal",
        has_natural_language_answer=True,
    )
    templates = {item["template_id"] for item in applicable_rubric_items(context)}
    assert {"H_CONSISTENCY", "H_TURN_PROGRESS"} <= templates
    assert "H_BINDING" not in templates
    assert "H_WITNESS" not in templates

    dependent = context.model_copy(
        update={"history_binding_ids": ("binding-1",), "requires_witness_check": True}
    )
    dependent_templates = {item["template_id"] for item in applicable_rubric_items(dependent)}
    assert {"H_BINDING", "H_WITNESS"} <= dependent_templates


def test_numeric_only_answer_skips_natural_language_criteria() -> None:
    assert not has_natural_language_content("42")
    assert has_natural_language_content("42 kg")
    context = RubricContext(
        turn_index=1,
        profile="normal",
        has_natural_language_answer=False,
    )
    templates = {item["template_id"] for item in applicable_rubric_items(context)}
    assert "F_A_CLEAR" not in templates
    assert "F_REDUNDANCY" not in templates
    assert "L_A_TARGET" not in templates


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


def test_claim_offsets_are_corrected_only_for_a_unique_quoted_span() -> None:
    answer = PublicMessage(
        message_id="a1",
        turn_index=1,
        role="assistant",
        content="There are five squares.",
    )
    claim = AtomicClaim(
        text="five squares",
        source_message_id="a1",
        start=0,
        end=99,
    )
    normalized = SynthesisCoordinator._normalize_claim(claim, answer)
    assert normalized is not None
    assert (normalized.start, normalized.end) == (10, 22)

    repeated = answer.model_copy(update={"content": "five squares and five squares"})
    assert SynthesisCoordinator._normalize_claim(claim, repeated) is None


def test_incomplete_extraction_abstains_without_generating_answer(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    original = coordinator._invoke

    def invoke(client, stage, payload, images, model, **kwargs):
        if model is RequirementInventory:
            return RequirementInventory(
                requirements=(),
                extraction_complete=False,
                reason="Unable to extract all explicit requirements.",
            )
        return original(client, stage, payload, images, model, **kwargs)

    monkeypatch.setattr(coordinator, "_invoke", invoke)
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


@pytest.mark.parametrize("always_wrong", [False, True])
def test_evidence_identity_is_regenerated_or_fails_closed(
    tmp_path: Path, always_wrong: bool
) -> None:
    class WrongImageClient(ScriptedClient):
        attempts = 0

        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            response = super().invoke(*args, **kwargs)
            self.attempts += 1
            if always_wrong or self.attempts == 1:
                return ModelResponse(
                    value=response.value.model_copy(update={"image_id": "wrong-image"}),
                    request_hash=response.request_hash,
                    response_hash=response.response_hash,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return response

    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path / "runs", "identity", require_local_wal=False) as store:
        client = WrongImageClient(config.models.generator_a)
        co = SynthesisCoordinator(config, "identity", store, client, client, client)

        def invoke() -> EvidenceInventory:
            return co._invoke(
                client,
                "evidence_extraction",
                {"image_id": "expected-image"},
                (),
                EvidenceInventory,
                max_tokens=1024,
                temperature=0.0,
                seed=1,
            )

        if always_wrong:
            with pytest.raises(ExecutionError, match="another image"):
                invoke()
            assert client.attempts == config.runtime.structured_output_max_attempts
        else:
            assert invoke().image_id == "expected-image"
            assert client.attempts == 2
        assert "expected-image" in (client.retry_feedback[1] or "")
        assert "regenerate the complete evidence" in (client.retry_feedback[1] or "")


def test_coverage_receives_verified_inventory_and_ids_stay_controller_owned(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, requirement_text="color")
    try:
        conversation = co.synthesize_image(image, root)
        assert conversation.status == "QUALITY_CANDIDATE"
        for client in (a, b):
            for stage, payload in client.calls:
                if stage == "requirement_extraction":
                    assert payload["question_message_id"].startswith("m")
                    assert len(payload["question_message_id"]) < 5
                if stage == "claim_inventory":
                    assert "candidate_answer_message_id" not in payload
                if stage == "rubric_item" and payload["criterion"]["template_id"] == "C_COVERAGE":
                    assert "image_views" not in payload
                    assert payload["candidate_claim_inventory"]
                    for claim in payload["candidate_claim_inventory"]:
                        assert claim["source_message_id"].startswith(conversation.conversation_id)
                        assert (
                            payload["candidate_answer"][claim["start"] : claim["end"]]
                            == claim["text"]
                        )
        for turn in conversation.turns:
            assert all(r.source_message_id == turn.question.message_id for r in turn.requirements)
    finally:
        store.close()


@pytest.mark.parametrize("always_invalid", [False, True])
def test_invalid_claim_boundaries_are_retried_without_guessing(
    tmp_path: Path, always_invalid: bool
) -> None:
    class InvalidQuoteClient(ScriptedClient):
        attempts = 0

        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            response = super().invoke(*args, **kwargs)
            self.attempts += 1
            if always_invalid or self.attempts == 1:
                value = ClaimExtraction(
                    claims=(ClaimSpan(start_token=0, end_token=9999),),
                    coverage=GateVerdict.MET,
                    reason="Complete",
                )
                return ModelResponse(
                    value=value,
                    request_hash="1" * 64,
                    response_hash="2" * 64,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return response

    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path / "runs", "quote", require_local_wal=False) as store:
        client = InvalidQuoteClient(config.models.generator_a)
        co = SynthesisCoordinator(config, "quote", store, client, client, client)

        def invoke():
            return co._invoke(
                client,
                "claim_inventory",
                {"candidate_answer": "Blue."},
                (),
                ClaimExtraction,
                max_tokens=1024,
                temperature=0.0,
                seed=1,
            )

        if always_invalid:
            with pytest.raises(ExecutionError, match="token boundaries"):
                invoke()
        else:
            assert (
                co._bind_claim_span(
                    invoke().claims[0],
                    PublicMessage(message_id="a", turn_index=1, role="assistant", content="Blue."),
                ).text
                == "Blue."
            )
        assert client.attempts == 2


@pytest.mark.parametrize(
    "text", ['The cap says "TITANS SWIMWEAR".', "看板には「停止」と書いてあります。", "2"]
)
def test_token_boundaries_reconstruct_original_quotes_and_unicode(text: str) -> None:
    answer = PublicMessage(message_id="actual-answer", turn_index=1, role="assistant", content=text)
    tokens = SynthesisCoordinator._answer_tokens(text)
    claim = SynthesisCoordinator._bind_claim_span(
        ClaimSpan(start_token=0, end_token=len(tokens)), answer
    )
    assert claim.text == text
    assert claim.source_message_id == answer.message_id
    claim.validate_span(answer)


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("How many signs are there, and can you group them by color?", "white"),
        ("How many lights are attached to each of the concentric rings?", "innermost ring"),
    ],
)
def test_visible_categories_need_not_be_literal_question_substrings(
    tmp_path: Path, question: str, category: str
):
    inventory = SetInventory(
        coverage="MET",
        mode="count",
        counts=(CountGroup(scope=category, expected=2, reported=3),),
        expected_members=(),
        reported_members=(),
        empty_scope_is_explicit=False,
        reason="Visible category",
    )

    class CategoryClient(ScriptedClient):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                value=inventory,
                request_hash="1" * 64,
                response_hash="2" * 64,
                prompt_tokens=1,
                completion_tokens=1,
            )

    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path / "runs", "categories", require_local_wal=False) as store:
        client = CategoryClient(config.models.generator_a)
        co = SynthesisCoordinator(config, "categories", store, client, client, client)
        result = co._invoke(
            client,
            "set_inventory",
            {"question": question},
            (),
            SetInventory,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
        assert result.counts[0].scope == category
        check = verify_set_inventories((result, result))
        assert check is not None and check.verdict == "NOT_MET"
        other = result.model_copy(
            update={"counts": (CountGroup(scope=category, expected=3, reported=3),)}
        )
        assert verify_set_inventories((result, other)) is None


@pytest.mark.parametrize(
    ("votes", "set_verdict", "expected", "aggregate"),
    [
        (("NOT_MET", "NOT_MET"), None, "NOT_MET", "FAIL"),
        (("MET", "MET"), None, "UNKNOWN", "ABSTAIN"),
        (("MET", "NOT_MET"), None, "UNKNOWN", "ABSTAIN"),
        (("UNKNOWN", "UNKNOWN"), None, "UNKNOWN", "ABSTAIN"),
        (("NOT_MET", "NOT_MET"), "MET", "NOT_MET", "FAIL"),
        (("MET", "MET"), "MET", "MET", "PASS"),
        (("MET", "NOT_MET"), "MET", "UNKNOWN", "ABSTAIN"),
        (("UNKNOWN", "UNKNOWN"), "MET", "UNKNOWN", "ABSTAIN"),
        (("NOT_MET", "NOT_MET"), "NOT_MET", "NOT_MET", "FAIL"),
        (("MET", "MET"), "NOT_MET", "NOT_MET", "FAIL"),
        (("MET", "NOT_MET"), "NOT_MET", "NOT_MET", "FAIL"),
        (("UNKNOWN", "UNKNOWN"), "NOT_MET", "NOT_MET", "FAIL"),
    ],
)
def test_set_rating_preserves_failure_without_rescuing_uncertainty(
    tmp_path, image_artifact, monkeypatch, votes, set_verdict, expected, aggregate
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    template = next(
        t for t in load_rubric_catalog()["items"] if t["template_id"] == "C_SET_COMPLETE"
    )
    monkeypatch.setattr("pixelogue.pipeline.applicable_rubric_items", lambda context: [template])
    check = (
        None
        if set_verdict is None
        else SetCheck(verdict=set_verdict, missing=(), unexpected=(), duplicates=())
    )
    monkeypatch.setattr(coordinator, "_check_complete_set", lambda *args: check)
    remaining_votes = iter(votes)

    def invoke(client, stage, *args, **kwargs):
        if stage == "claim_inventory":
            return ClaimExtraction(
                claims=(ClaimSpan(start_token=0, end_token=1),),
                coverage=GateVerdict.MET,
                reason="The numeric answer is covered.",
            )
        assert stage == "rubric_item"
        return RubricVerdict(verdict=next(remaining_votes), reason="Independent semantic vote.")

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        rating = coordinator._rate_turn(
            "test-conversation",
            "1" * 64,
            (),
            PublicMessage(message_id="q", turn_index=1, role="user", content="How many cords?"),
            PublicMessage(message_id="a", turn_index=1, role="assistant", content="2"),
            InstructionCandidate(
                candidate_id="count",
                task_id="count",
                family="visible_count",
                visible_scope="cords",
                instruction_summary="Count cords.",
                required_capabilities=(),
            ),
            "en",
            [],
            ModelImage(
                view_id=image.full_view.view_id,
                path=root / image.full_view.relative_path,
                encoded_sha256=image.full_view.encoded_sha256,
                media_type=image.full_view.media_type,
            ),
            1,
            (),
        )
        assert rating.items[0].verdict == expected
        assert rating.aggregate == aggregate
    finally:
        store.close()


def test_v7_holistic_checks_bound_question_and_two_blind_whole_turn_reviews(
    tmp_path, image_artifact
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    try:
        conversation = co.synthesize_image(image, root)
        assert conversation.status == "QUALITY_CANDIDATE"
        for client in (a, b):
            stages = [stage for stage, _ in client.calls]
            assert stages.count("holistic_review") == len(conversation.turns)
            assert stages.count("question_fit") == len(conversation.turns)
            assert not set(stages) & {
                "requirement_extraction",
                "claim_inventory",
                "set_inventory",
                "rubric_item",
                "answer_repair",
            }
            for stage, payload in client.calls:
                if stage == "holistic_review":
                    assert set(payload) == {
                        "target_language",
                        "public_history",
                        "question",
                        "candidate_answer",
                        "expected_operation",
                        "image_views",
                    }
        assert all(not turn.requirements for turn in conversation.turns)
        assert all(turn.rating.items[0].template_id == "Q_HOLISTIC" for turn in conversation.turns)
        calls = len(a.calls) + len(b.calls)
        assert co.synthesize_image(image, root) == conversation
        assert len(a.calls) + len(b.calls) == calls
    finally:
        store.close()


@pytest.mark.parametrize(
    ("votes", "expected"),
    [
        (("NOT_MET", "NOT_MET"), "REJECTED"),
        (("MET", "NOT_MET"), "ABSTAINED"),
        (("UNKNOWN", "MET"), "ABSTAINED"),
    ],
)
@pytest.mark.parametrize("passing_turns", [0, 1, 2])
def test_holistic_stop_retains_only_two_or_more_accepted_turns(
    tmp_path, image_artifact, monkeypatch, votes, expected, passing_turns
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    for client, vote in zip((a, b), votes, strict=True):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, _vote=vote, **kwargs):
            response = _original(stage, payload, *args, **kwargs)
            if stage == "holistic_review" and len(payload["public_history"]) >= 2 * passing_turns:
                return ModelResponse(
                    value=RubricVerdict(verdict=_vote, reason="Observed defect or uncertainty."),
                    request_hash=response.request_hash,
                    response_hash=response.response_hash,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return response

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        conversation = co.synthesize_image(image, root)
        if passing_turns >= 2:
            assert conversation.status == "QUALITY_CANDIDATE"
            assert len(conversation.turns) == passing_turns
            assert all(t.status == "COMMITTED" for t in conversation.turns)
            rows = store.connection.execute(
                "SELECT artifact_hash FROM artifact WHERE kind='conversation-stops'"
            ).fetchall()
            assert len(rows) == 1

            stopped = json.loads(store.read_artifact(rows[0][0]))
            assert stopped["conversation"]["status"] == expected
            assert len(stopped["conversation"]["turns"]) == passing_turns + 1
        else:
            assert conversation.status == expected
        assert co.synthesize_image(image, root) == conversation
        assert not any(stage == "answer_repair" for client in (a, b) for stage, _ in client.calls)
    finally:
        store.close()


def test_holistic_prefix_retention_can_be_disabled(tmp_path, image_artifact, monkeypatch):
    image, root = image_artifact
    co, store, _, _, _ = _coordinator(tmp_path, evaluation_mode="holistic")
    co.config = co.config.model_copy(
        update={"evaluation": EvaluationConfig(mode="holistic", retain_accepted_prefix=False)}
    )
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 3)
    original = co._select_instruction
    monkeypatch.setattr(
        co, "_select_instruction", lambda *args: None if args[-1] == 3 else original(*args)
    )
    try:
        conversation = co.synthesize_image(image, root)
        assert len(conversation.turns) == 2
        assert conversation.status == "REJECTED"
    finally:
        store.close()


def test_holistic_rerating_rejects_empty_conversation(tmp_path, image_artifact):
    image, root = image_artifact
    co, store, _, _, _ = _coordinator(tmp_path, reject_selection=True, evaluation_mode="holistic")
    try:
        conversation = co.synthesize_image(image, root)
        assert not conversation.turns
        assert co.rate_existing(conversation, root).status == "REJECTED"
    finally:
        store.close()


def test_holistic_prefix_survives_reopening_and_exports_only_committed_turns(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    original = co._select_instruction
    monkeypatch.setattr(
        co, "_select_instruction", lambda *args: None if args[-1] == 3 else original(*args)
    )
    accepted = co.synthesize_image(image, root)
    assert accepted.status == "QUALITY_CANDIDATE" and len(accepted.turns) == 2
    store.close()
    with RunStore(tmp_path / "runs", "test", require_local_wal=False) as reopened:
        reopened.initialize_run("test", co.config.config_hash, co.config.profile)
        resumed = SynthesisCoordinator(co.config, "test", reopened, co.selector, a, b)
        calls = len(a.calls) + len(b.calls)
        assert resumed.synthesize_image(image, root) == accepted
        assert len(a.calls) + len(b.calls) == calls
        changed = image.model_copy(update={"source_id": "different-source"})
        with pytest.raises(ExecutionError, match="conflicts with the current input"):
            resumed.synthesize_image(changed, root)
        record = training_record(accepted)
        assert len(record.messages) == 4
        assert record.messages[-1].content[0].text == accepted.turns[-1].answer.content


def test_holistic_rerating_uses_new_votes_and_preserves_failed_tail_privately(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    try:
        original = co.synthesize_image(image, root)
        for client in (a, b):
            invoke_original = client.invoke

            def invoke(stage, payload, *args, _original=invoke_original, **kwargs):
                result = _original(stage, payload, *args, **kwargs)
                if stage == "holistic_review" and len(payload["public_history"]) == 4:
                    return ModelResponse(
                        value=RubricVerdict(verdict="NOT_MET", reason="Wrong visible fact."),
                        request_hash=result.request_hash,
                        response_hash=result.response_hash,
                        prompt_tokens=1,
                        completion_tokens=1,
                    )
                return result

            monkeypatch.setattr(client, "invoke", invoke)
        rated = co.rate_existing(original, root)
        assert rated.status == "QUALITY_CANDIDATE"
        assert len(rated.turns) == 2
        assert rated.public_messages == original.public_messages[:4]
        assert all(t.rating.items[0].template_id == "Q_HOLISTIC" for t in rated.turns)
    finally:
        store.close()


def test_v7_holistic_wrong_operation_stops_before_answer_generation(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    monkeypatch.setattr(co, "_question_fit", lambda *args: GateVerdict.NOT_MET)
    try:
        result = co.synthesize_image(image, root)
        assert result.status == "REJECTED"
        assert not any(
            stage == "answer_generation" for client in (a, b) for stage, _ in client.calls
        )
    finally:
        store.close()
