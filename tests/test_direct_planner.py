from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from pixelogue.catalog import task_catalog
from pixelogue.config import EvaluationConfig, ModelEndpoint, TaskRuntimeConfig, load_config
from pixelogue.contracts import GateVerdict, RubricVerdict, TextPayload
from pixelogue.drafting import (
    DraftParameter,
    FactKey,
    QuestionDraft,
    QuestionDraftBatch,
    draft_to_candidate,
    request_key,
)
from pixelogue.errors import ExecutionError
from pixelogue.gates import QuestionGateVote, holistic_decision, question_gate_decision
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.routing import FamilyLedger, ImageProfile, choose_route
from pixelogue.serving import ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_catalog import TaskDefinition
from pixelogue.task_evidence import ImageRegion

FULL = ImageRegion(left=0, top=0, right=1, bottom=1)
LEFT = ImageRegion(left=0.1, top=0.2, right=0.4, bottom=0.8)


def draft(task_id: str = "object_identification", **changes: Any) -> QuestionDraft:
    values: dict[str, Any] = {
        "task_id": task_id,
        "question": "What is the blue object on the left side of the image?",
        "target": "blue object on the left",
        "public_parameters": (),
        "scope_region": FULL,
        "target_region": LEFT,
        "fact_key": FactKey(subject="blue object on the left", dimension="category"),
    }
    values.update(changes)
    return QuestionDraft(**values)


def convert(value: QuestionDraft, allowed: tuple[str, ...] = ("object_identification",), **kw: Any):
    return draft_to_candidate(
        value,
        image_id="a" * 64,
        view_id="full:test",
        turn_index=1,
        draft_index=0,
        allowed_task_ids=allowed,
        **kw,
    )


def test_draft_becomes_direct_candidate_without_evidence() -> None:
    candidate = convert(draft())
    assert candidate.origin == "direct"
    assert candidate.evidence_refs == ()
    assert candidate.scope_region == FULL and candidate.target_region == LEFT
    assert candidate.request_key == "blue object on left|category"
    assert all(parameter.origin == "instruction" for parameter in candidate.public_parameters)


def test_draft_rejects_unoffered_task_and_invalid_choices() -> None:
    with pytest.raises(ExecutionError, match="was not offered"):
        convert(draft(), allowed=("attribute_lookup",))
    with pytest.raises(ExecutionError) as missing:
        convert(draft("attribute_lookup"), allowed=("attribute_lookup",))
    assert missing.value.reason == "DRAFT_PARAMETER_MISSING"
    bad = draft(
        "scene_categorization",
        target="whole image",
        public_parameters=(DraftParameter(name="category_set", value=("indoor",)),),
    )
    with pytest.raises(ExecutionError) as scene:
        convert(bad, allowed=("scene_categorization",))
    assert scene.value.reason == "DRAFT_PARAMETER_VALUE"
    unknown = draft(public_parameters=(DraftParameter(name="answer", value="cat"),))
    with pytest.raises(ExecutionError) as extra:
        convert(unknown)
    assert extra.value.reason == "DRAFT_PARAMETER_UNKNOWN"


def test_transcription_binds_the_whole_drafted_scope() -> None:
    value = draft(
        "text_transcription",
        question="Transcribe the sign at the top of the image.",
        scope_region=ImageRegion(left=0.2, top=0.0, right=0.8, bottom=0.3),
        target_region=ImageRegion(left=0.3, top=0.05, right=0.7, bottom=0.2),
    )
    candidate = convert(value, allowed=("text_transcription",))
    assert candidate.target_region == candidate.scope_region


def test_request_key_ignores_spelling_and_articles() -> None:
    first = request_key(FactKey(subject="The dog", dimension="fur colour"))
    second = request_key(FactKey(subject="dog", dimension="Fur Color"))
    assert first == second == "dog|fur color"


def test_target_must_stay_inside_scope() -> None:
    with pytest.raises(ValueError):
        draft(scope_region=LEFT, target_region=FULL)


def _tasks(*families: str) -> dict[str, TaskDefinition]:
    return {
        task.id: task
        for task in task_catalog().tasks
        if task.family in families and task.status == "core"
    }


def _profile(*families: str) -> ImageProfile:
    return ImageProfile(
        image_kind="photo", readable_text="none", supported_families=families, reason="test"
    )


def _route(profile, ledger, used=frozenset(), tasks=None):
    return choose_route(
        profile,
        ledger,
        tasks=tasks or _tasks("visual_description", "counting_and_sets", "reference_spatial"),
        family_targets="uniform",
        task_weights={},
        used_families=used,
        used_task_ids=frozenset(),
        seed=1,
        image_id="img",
        turn_index=1,
    )


def test_route_prefers_the_least_covered_feasible_family() -> None:
    ledger = FamilyLedger(["object_identification"] * 5 + ["entity_count"] * 2)
    route = _route(_profile("visual_description", "counting_and_sets", "reference_spatial"), ledger)
    assert route is not None
    assert route.primary_family == "reference_spatial"
    assert route.secondary_family == "counting_and_sets"
    assert all(task in _tasks("reference_spatial") for task in route.primary_task_ids)


def test_route_avoids_families_already_used_in_the_conversation() -> None:
    route = _route(
        _profile("visual_description", "counting_and_sets"),
        FamilyLedger(),
        used=frozenset({"counting_and_sets"}),
    )
    assert route is not None and route.primary_family == "visual_description"


def test_route_is_deterministic_and_falls_back_without_profile() -> None:
    first = _route(_profile("visual_description", "counting_and_sets"), FamilyLedger())
    second = _route(_profile("visual_description", "counting_and_sets"), FamilyLedger())
    assert first == second
    fallback = _route(None, FamilyLedger())
    assert fallback is not None and fallback.basis == "fallback"
    assert fallback.primary_family in {
        "visual_description",
        "reference_spatial",
        "counting_and_sets",
    }


def test_ledger_counts_concurrent_commits() -> None:
    ledger = FamilyLedger()
    threads = [
        threading.Thread(target=lambda: [ledger.record("entity_count") for _ in range(50)])
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    families, tasks = ledger.snapshot()
    assert tasks["entity_count"] == 400 and families["counting_and_sets"] == 400


def vote(
    label: str | None, *, fit: Literal["MET", "NOT_MET", "UNKNOWN"] = "MET"
) -> QuestionGateVote:
    return QuestionGateVote(
        realized_task_id=label,
        local_anchor=fit,
        operation_coherent="MET",
        useful_request="MET",
        reason="r",
    )


CONTRACTS = {task.id: task.verification_contracts for task in task_catalog().tasks}


@pytest.mark.parametrize(
    ("labels", "policy", "verdict", "rule", "task"),
    [
        (
            ("object_identification", "object_identification"),
            "strict",
            "MET",
            "exact",
            "object_identification",
        ),
        (
            ("object_identification", "attribute_lookup"),
            "strict",
            "UNKNOWN",
            "label_unresolved",
            "object_identification",
        ),
        (
            ("object_identification", "attribute_lookup"),
            "same_contract",
            "MET",
            "same_contract",
            "object_identification",
        ),
        (
            ("attribute_lookup", "attribute_lookup"),
            "strict",
            "NOT_MET",
            "label_mismatch",
            "object_identification",
        ),
        (
            ("scene_categorization", "scene_categorization"),
            "same_contract",
            "MET",
            "relabel",
            "scene_categorization",
        ),
        (
            ("object_identification", None),
            "same_contract",
            "UNKNOWN",
            "label_unresolved",
            "object_identification",
        ),
        (
            ("entity_count", "entity_count"),
            "same_contract",
            "NOT_MET",
            "label_mismatch",
            "object_identification",
        ),
    ],
)
def test_question_gate_label_policy(labels, policy, verdict, rule, task) -> None:
    decision = question_gate_decision(
        [vote(labels[0]), vote(labels[1])],
        "object_identification",
        policy=policy,
        contracts=CONTRACTS,
        relabel_allowed=lambda task_id: task_id == "scene_categorization",
    )
    assert (decision.verdict.value, decision.label_rule, decision.task_id) == (verdict, rule, task)


def test_question_gate_fit_failure_dominates_label_agreement() -> None:
    decision = question_gate_decision(
        [
            vote("object_identification", fit="NOT_MET"),
            vote("object_identification", fit="NOT_MET"),
        ],
        "object_identification",
        policy="same_contract",
        contracts=CONTRACTS,
        relabel_allowed=lambda task_id: False,
    )
    assert decision.verdict is GateVerdict.NOT_MET


def test_holistic_decision_names_one_follow_up() -> None:
    met, not_met, unknown = GateVerdict.MET, GateVerdict.NOT_MET, GateVerdict.UNKNOWN
    repair = holistic_decision(
        [met, not_met], ["ok", "wrong color"], tiebreak_available=True, repair_available=True
    )
    assert (repair.action, repair.judge_index, repair.objection) == ("repair", 1, "wrong color")
    tie = holistic_decision(
        [unknown, met], ["unclear", "ok"], tiebreak_available=True, repair_available=True
    )
    assert (tie.action, tie.judge_index) == ("tiebreak", 0)
    none = holistic_decision(
        [met, not_met], ["ok", "no"], tiebreak_available=True, repair_available=False
    )
    assert none.action == "none" and none.verdict is GateVerdict.UNKNOWN
    assert (
        holistic_decision(
            [met, met], ["", ""], tiebreak_available=True, repair_available=True
        ).verdict
        is GateVerdict.MET
    )


class DirectClient:
    """Scripted endpoint for the direct planner's stages."""

    def __init__(self, endpoint: ModelEndpoint, script: dict[str, Any]) -> None:
        self.endpoint = endpoint
        self.script = script
        self.calls: list[tuple[str, dict[str, Any], int]] = []
        self.lock = threading.Lock()

    def invoke(
        self,
        stage,
        payload,
        images,
        response_model,
        *,
        max_tokens,
        temperature,
        seed,
        bypass_cache=False,
        retry_feedback=None,
        trial_id=None,
    ) -> ModelResponse:
        del max_tokens, temperature, seed, bypass_cache, retry_feedback
        with self.lock:
            self.calls.append((stage, payload, len(images)))
        handler = self.script[stage]
        value = handler(payload, trial_id) if callable(handler) else handler
        assert isinstance(value, BaseModel) and isinstance(value, response_model)
        return ModelResponse(value, "r" * 64, "s" * 64, 1, 1)


def profile_value(payload: dict[str, Any], _trial: str | None) -> ImageProfile:
    assert any(row["family"] == "visual_description" for row in payload["family_definitions"])
    return ImageProfile(
        image_kind="photo",
        readable_text="none",
        supported_families=("visual_description",),
        reason="photo",
    )


def drafts_value(payload: dict[str, Any], _trial: str | None) -> QuestionDraftBatch:
    turn = payload["turn_index"]
    offered = payload["preferred_task_ids"]
    task = "object_identification" if "object_identification" in offered else offered[0]
    rows = []
    for index in range(payload["draft_count"]):
        parameters = []
        if task == "attribute_lookup":
            parameters.append(DraftParameter(name="attribute", value="color"))
        if task == "scene_categorization":
            parameters.append(DraftParameter(name="category_set", value=("indoor", "outdoor")))
        rows.append(
            draft(
                task,
                question=f"Question {turn}-{index}: indoor or outdoor about object {turn}-{index}?",
                target=f"object {turn}-{index}",
                public_parameters=tuple(parameters),
                fact_key=FactKey(subject=f"object {turn}-{index}", dimension="category"),
            )
        )
    return QuestionDraftBatch(drafts=tuple(rows))


def make_coordinator(
    tmp_path: Path, script_a: dict, script_b: dict, router: dict | None = None, **evaluation: Any
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "evaluation": EvaluationConfig.model_validate(
                {"question_gate_label_policy": "same_contract", **evaluation}
            ),
        }
    )
    store = RunStore(tmp_path / "runs", "direct", require_local_wal=False)
    store.initialize_run("direct", config.config_hash, config.profile)
    selector = DirectClient(config.models.router, router or {"image_profile": profile_value})
    a = DirectClient(config.models.generator_a, script_a)
    b = DirectClient(config.models.generator_b, script_b)
    return SynthesisCoordinator(config, "direct", store, selector, a, b), store, selector, a, b


def gate_ok(payload: dict[str, Any], _trial: str | None) -> QuestionGateVote:
    task = payload["selected_instruction"]["task_id"]
    return vote(task)


def judge_script(**overrides: Any) -> dict[str, Any]:
    script = {
        "question_gate": gate_ok,
        "holistic_review": RubricVerdict(verdict="MET", reason="supported"),
        "question_draft": drafts_value,
        "answer_generation": TextPayload(text="a red ball"),
        "answer_repair": TextPayload(text="a blue ball"),
    }
    script.update(overrides)
    return script


def stages(client: DirectClient) -> list[str]:
    return [stage for stage, _, _ in client.calls]


def test_direct_planner_commits_turns_without_scoped_stages(tmp_path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, selector, a, b = make_coordinator(tmp_path, judge_script(), judge_script())
    try:
        conversation = coordinator.synthesize_image(image, root, generator_role="generator_a")
    finally:
        store.close()
    assert conversation.status == "QUALITY_CANDIDATE"
    assert all(turn.instruction.origin == "direct" for turn in conversation.turns)
    assert stages(selector) == ["image_profile"]
    used = set(stages(a)) | set(stages(b))
    assert not used & {
        "evidence_extraction",
        "candidate_binding",
        "instruction_selection",
        "question_intent",
        "question_fit",
        "question_generation",
    }
    for client in (a, b):
        for stage, payload, _ in client.calls:
            if stage in {"question_draft", "question_gate"}:
                assert "candidate_answer" not in payload
            assert "other_judge" not in payload and "votes" not in payload


def test_direct_planner_tries_later_drafts_after_a_gate_rejection(tmp_path, image_artifact) -> None:
    image, root = image_artifact
    seen: list[str] = []

    def picky(payload: dict[str, Any], _trial: str | None) -> QuestionGateVote:
        seen.append(payload["question"])
        task = payload["selected_instruction"]["task_id"]
        return (
            vote(task, fit="NOT_MET")
            if payload["question"].startswith("Question 1-0")
            else vote(task)
        )

    coordinator, store, _, a, b = make_coordinator(
        tmp_path, judge_script(question_gate=picky), judge_script(question_gate=picky)
    )
    try:
        conversation = coordinator.synthesize_image(image, root, generator_role="generator_a")
    finally:
        store.close()
    assert conversation.turns[0].question.content.startswith("Question 1-1")
    assert stages(a).count("question_draft") == len(conversation.turns)
    first_turn_answers = [s for s in stages(a) if s == "answer_generation"]
    assert len(first_turn_answers) == len(conversation.turns)


def test_direct_planner_drafts_once_more_then_stops(tmp_path, image_artifact) -> None:
    image, root = image_artifact

    def reject(payload: dict[str, Any], _trial: str | None) -> QuestionGateVote:
        return vote(payload["selected_instruction"]["task_id"], fit="NOT_MET")

    coordinator, store, _, a, _ = make_coordinator(
        tmp_path, judge_script(question_gate=reject), judge_script(question_gate=reject)
    )
    try:
        conversation = coordinator.synthesize_image(image, root, generator_role="generator_a")
    finally:
        store.close()
    assert conversation.status == "REJECTED" and conversation.turns == ()
    assert stages(a).count("question_draft") == 2
    assert "answer_generation" not in stages(a)


def test_split_review_repairs_once_and_rejudges(tmp_path, image_artifact) -> None:
    image, root = image_artifact
    reviews: list[str] = []

    def strict_review(payload: dict[str, Any], _trial: str | None) -> RubricVerdict:
        reviews.append(payload["candidate_answer"])
        if payload["candidate_answer"] == "a red ball":
            return RubricVerdict(verdict="NOT_MET", reason="The ball is blue, not red.")
        return RubricVerdict(verdict="MET", reason="supported")

    coordinator, store, _, a, b = make_coordinator(
        tmp_path,
        judge_script(),
        judge_script(holistic_review=strict_review),
        repair_once=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root, generator_role="generator_a")
    finally:
        store.close()
    assert conversation.status == "QUALITY_CANDIDATE"
    assert all(turn.answer.content == "a blue ball" for turn in conversation.turns)
    repairs = [payload for stage, payload, _ in a.calls if stage == "answer_repair"]
    assert len(repairs) == len(conversation.turns)
    assert all("The ball is blue" in repairs[0]["failed_criteria"][0] for _ in repairs)


def test_unknown_review_gets_one_full_view_tiebreak(tmp_path, image_artifact) -> None:
    image, root = image_artifact

    def unsure(payload: dict[str, Any], trial: str | None) -> RubricVerdict:
        if trial and trial.endswith(":tiebreak"):
            return RubricVerdict(verdict="MET", reason="clear in the complete image")
        return RubricVerdict(verdict="UNKNOWN", reason="crop lacks context")

    coordinator, store, _, a, b = make_coordinator(
        tmp_path,
        judge_script(),
        judge_script(holistic_review=unsure),
        judge_views="full_and_crop",
        holistic_tiebreak="full_view",
    )
    try:
        conversation = coordinator.synthesize_image(image, root, generator_role="generator_a")
    finally:
        store.close()
    # Local operations get the crop plus a full-view tie-break; whole-image operations have no
    # alternate view, so a later UNKNOWN there abstains.
    assert conversation.turns[0].status == "COMMITTED"
    tiebreaks = [
        images
        for stage, payload, images in b.calls
        if stage == "holistic_review" and len(payload["image_views"]) == 1
    ]
    first_reviews = [
        images
        for stage, payload, images in b.calls
        if stage == "holistic_review" and len(payload["image_views"]) == 2
    ]
    assert tiebreaks and all(count == 1 for count in tiebreaks)
    assert first_reviews and all(count == 2 for count in first_reviews)


def test_completed_direct_conversation_resumes_without_calls(tmp_path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, selector, a, b = make_coordinator(tmp_path, judge_script(), judge_script())
    try:
        first = coordinator.synthesize_image(image, root, generator_role="generator_a")
        before = len(a.calls) + len(b.calls) + len(selector.calls)
        again = coordinator.synthesize_image(image, root, generator_role="generator_a")
        after = len(a.calls) + len(b.calls) + len(selector.calls)
    finally:
        store.close()
    assert again == first and before == after


@pytest.mark.parametrize(
    "retired",
    [
        {"planner": "scoped"},
        {"evidence_format": "compact"},
        {"candidate_limit": 4},
        {"profiles": ["normal", "limitation"]},
    ],
)
def test_retired_scoped_settings_are_rejected(retired: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        TaskRuntimeConfig.model_validate(retired)


def test_image_views_match_attached_images(tmp_path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, _, a, b = make_coordinator(
        tmp_path, judge_script(), judge_script(), judge_views="full_and_crop"
    )
    try:
        coordinator.synthesize_image(image, root, generator_role="generator_a")
    finally:
        store.close()
    for stage, payload, count in a.calls + b.calls:
        assert len(payload["image_views"]) == count, stage


def test_anchor_turns_offer_only_lightly_verified_operations() -> None:
    tasks = _tasks("visual_description", "counting_and_sets", "table_understanding")
    ledger = FamilyLedger(["object_identification"] * 9)
    profile = _profile("visual_description", "counting_and_sets", "table_understanding")
    light = choose_route(
        profile,
        ledger,
        tasks=tasks,
        family_targets="uniform",
        task_weights={},
        used_families=frozenset(),
        used_task_ids=frozenset(),
        seed=1,
        image_id="img",
        turn_index=1,
        light_only=True,
    )
    assert light is not None and light.primary_family == "visual_description"
    assert all(
        tasks[task].verification_contracts == ("dual_visual_review",) for task in light.task_ids
    )
    later = choose_route(
        profile,
        ledger,
        tasks=tasks,
        family_targets="uniform",
        task_weights={},
        used_families=frozenset(),
        used_task_ids=frozenset(),
        seed=1,
        image_id="img",
        turn_index=3,
    )
    assert later is not None and later.primary_family in {
        "counting_and_sets",
        "table_understanding",
    }


def test_screen_operations_need_a_screen_image() -> None:
    tasks = _tasks("screen_ui", "visual_description")
    figure = ImageProfile(
        image_kind="diagram",
        readable_text="some",
        supported_families=("screen_ui", "visual_description"),
        reason="r",
    )
    route = choose_route(
        figure,
        FamilyLedger(["object_identification"] * 3),
        tasks=tasks,
        family_targets="uniform",
        task_weights={},
        used_families=frozenset(),
        used_task_ids=frozenset(),
        seed=1,
        image_id="img",
        turn_index=1,
    )
    assert route is not None
    assert "screen_ui" not in {route.primary_family, route.secondary_family}


def test_rerating_uses_the_merged_question_gate_without_relabeling(
    tmp_path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, a, b = make_coordinator(tmp_path, judge_script(), judge_script())
    neighbor = {"object_identification": "described_object_lookup"}

    def neighbor_label(payload: dict[str, Any], _trial: str | None) -> QuestionGateVote:
        task = payload["selected_instruction"]["task_id"]
        return vote(neighbor.get(task, task))

    try:
        conversation = coordinator.synthesize_image(image, root, generator_role="generator_a")
        marks = len(a.calls), len(b.calls)
        rated = coordinator.rate_existing(conversation, root)
        rerating = {stage for stage, _, _ in a.calls[marks[0] :] + b.calls[marks[1] :]}
        for client in (a, b):
            client.script["question_gate"] = neighbor_label
        relabeled = coordinator.rate_existing(conversation, root)
    finally:
        store.close()
    assert conversation.turns[0].instruction.task_id == "object_identification"
    assert rated.status == "QUALITY_CANDIDATE" and len(rated.turns) == len(conversation.turns)
    assert "question_gate" in rerating
    assert not rerating & {"question_intent", "question_fit"}
    # A saved turn has no draft, so a same-contract neighbor label cannot replace its operation.
    assert relabeled.turns[0].status == "REJECTED"
    assert relabeled.turns[0].instruction == conversation.turns[0].instruction
