"""Presence and false-premise answers commit only when two blind readers agree with them."""

from __future__ import annotations

from collections import Counter

import pytest

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.drafting import (
    DraftParameter,
    FactKey,
    QuestionDraft,
    draft_task_contract,
    plan_mismatch,
    presence_draft_plan,
)
from pixelogue.premise_verifiers import (
    PREMISE_TASKS,
    PremiseAnswer,
    PremiseSource,
    verify_premise,
)
from pixelogue.prompts import STAGE_INSTRUCTIONS
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_verification import verify_operation

ABSENT = (
    "The image shows a person holding a baseball bat on a field. There is no refrigerator "
    "visible in the image. So the answer is no."
)
PRESENT = "The kitchen has a sink and a stove. Yes, there is a refrigerator on the left wall."


def _reader(status: str, coverage: str = "MET") -> PremiseSource:
    return PremiseSource.model_validate(
        {
            "visible_objects": ("a person", "a baseball bat", "grass"),
            "premise": "a refrigerator",
            "status": status,
            "coverage": coverage,
            "reason": "read blind",
        }
    )


def _parse(quote: str, status: str, coverage: str = "MET") -> PremiseAnswer:
    return PremiseAnswer.model_validate(
        {"answer_quote": quote, "status": status, "coverage": coverage, "reason": "parsed"}
    )


def test_presence_answers_must_match_both_blind_readers() -> None:
    absent = _parse("There is no refrigerator visible in the image", "absent")
    readers_absent = (_reader("absent"), _reader("absent"))
    assert verify_premise("object_presence", readers_absent, (absent, absent), ABSENT) is (
        GateVerdict.MET
    )
    readers_present = (_reader("present"), _reader("present"))
    assert verify_premise("object_presence", readers_present, (absent, absent), ABSENT) is (
        GateVerdict.NOT_MET
    )
    present = _parse("Yes, there is a refrigerator on the left wall", "present")
    assert verify_premise("object_presence", readers_present, (present, present), PRESENT) is (
        GateVerdict.MET
    )


@pytest.mark.parametrize(
    ("readers", "parses"),
    [
        # The readers disagree.
        (("absent", "present"), ("absent", "absent")),
        # One reader is unsure.
        (("absent", "UNKNOWN"), ("absent", "absent")),
        # The parses disagree.
        (("absent", "absent"), ("absent", "present")),
        # One parse found no conclusion.
        (("absent", "absent"), ("absent", "UNKNOWN")),
    ],
)
def test_disagreement_or_abstention_leaves_presence_undecided(readers, parses) -> None:
    def reader(value: str) -> PremiseSource:
        return _reader("absent", "UNKNOWN") if value == "UNKNOWN" else _reader(value)

    def parse(value: str) -> PremiseAnswer:
        quote = "There is no refrigerator visible in the image"
        return _parse("", "absent", "UNKNOWN") if value == "UNKNOWN" else _parse(quote, value)

    sources = (reader(readers[0]), reader(readers[1]))
    answers = (parse(parses[0]), parse(parses[1]))
    assert verify_premise("object_presence", sources, answers, ABSENT) is GateVerdict.UNKNOWN


def test_a_quote_that_is_not_in_the_answer_is_not_a_parse() -> None:
    invented = _parse("There is no fridge", "absent")
    readers = (_reader("absent"), _reader("absent"))
    assert verify_premise("object_presence", readers, (invented, invented), ABSENT) is (
        GateVerdict.UNKNOWN
    )


def test_a_false_premise_must_be_absent_for_both_readers() -> None:
    answer = "The woman holds a phone, but there is no purse, so its color cannot be determined."
    parse = _parse("there is no purse", "absent")
    absent = (_reader("absent"), _reader("absent"))
    assert verify_premise("false_premise_question", absent, (parse, parse), answer) is (
        GateVerdict.MET
    )
    # The assumed purse is actually there: the question is not a false-premise question.
    present = (_reader("present"), _reader("present"))
    described = _parse("The purse is red", "present")
    assert (
        verify_premise(
            "false_premise_question", present, (described, described), "The purse is red."
        )
        is GateVerdict.UNKNOWN
    )
    # An answer that invents the missing purse fails when both readers see no purse.
    assert (
        verify_premise(
            "false_premise_question", absent, (described, described), "The purse is red."
        )
        is GateVerdict.NOT_MET
    )


def test_unknown_operations_have_no_presence_verifier() -> None:
    parse = _parse("no", "absent")
    with pytest.raises(ValueError):
        verify_premise("entity_count", (_reader("absent"), _reader("absent")), (parse, parse), "no")


def test_presence_readers_never_see_the_candidate_answer() -> None:
    task = next(item for item in task_catalog().tasks if item.id == "object_presence")
    assert set(task.verification_contracts) == {"dual_visual_review", "premise_check"}
    assert task.id in PREMISE_TASKS
    candidate = InstructionCandidate(
        candidate_id="presence",
        task_id=task.id,
        family=task.family,
        visible_scope="the whole image",
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version="8.0",
        scope_id="canvas",
        view_id="view",
        evidence_refs=("scope-evidence",),
        verification_contracts=task.verification_contracts,
    )
    stages: list[tuple[str, int]] = []

    def invoke(stage, payload, model, judge):
        stages.append((stage, judge))
        if stage == "premise_source":
            assert "candidate_answer" not in payload
            return _reader("absent")
        assert stage == "premise_answer"
        assert "image_views" not in payload
        return _parse("There is no refrigerator visible in the image", "absent")

    result = verify_operation(
        candidate,
        {"question": "Is there a refrigerator in the image?", "candidate_answer": ABSENT},
        invoke,
    )
    assert stages == [
        ("premise_source", 0),
        ("premise_source", 1),
        ("premise_answer", 0),
        ("premise_answer", 1),
    ]
    assert [(check.name, check.verdict) for check in result] == [("premise_check", GateVerdict.MET)]


def test_presence_draft_plans_are_seeded_and_balanced() -> None:
    def plan(task_id: str, conversation: int) -> dict[str, str] | None:
        return presence_draft_plan(
            task_id, seed=7, conversation_id=f"c{conversation}", turn_index=1, call_index=0
        )

    assert plan("object_presence", 3) == plan("object_presence", 3)
    assert plan("entity_count", 3) is None
    presence = [plan("object_presence", index) for index in range(400)]
    answers = Counter(item["answer"] for item in presence if item)
    assert 160 <= answers["yes"] <= 240
    kinds = Counter(item["absent_kind"] for item in presence if item and item["answer"] == "no")
    assert set(kinds) == {"co_occurring", "common", "unrelated"}
    assert kinds["co_occurring"] > kinds["common"] and kinds["co_occurring"] > kinds["unrelated"]
    details = Counter(
        item["asked_detail"]
        for index in range(400)
        if (item := plan("false_premise_question", index))
    )
    assert set(details) == {"count", "attribute", "location", "action", "kind"}


def test_absent_objects_are_asked_about_where_they_would_be_seen() -> None:
    """The reference rejected 3 of 3 B0 absences asked about blurred or covered areas."""
    plans = [
        presence_draft_plan(
            "object_presence", seed=7, conversation_id=f"c{index}", turn_index=1, call_index=0
        )
        for index in range(40)
    ]
    absent = [plan for plan in plans if plan and plan["answer"] == "no"]
    assert absent
    assert all("would clearly be seen if it were there" in plan["instruction"] for plan in absent)
    gate, review = (
        " ".join(STAGE_INSTRUCTIONS[stage].split())
        for stage in ("question_gate", "holistic_review")
    )
    assert "blur, darkness, cropping or something in front could hide that object" in gate
    assert "due to blur, darkness, cropping" in review


def test_a_false_premise_draft_must_follow_its_planned_detail() -> None:
    draft = QuestionDraft(
        task_id="false_premise_question",
        question="How many horses are next to the car?",
        target="the car",
        public_parameters=(DraftParameter(name="asked_detail", value="count"),),
        scope_region=ImageRegion(left=0, top=0, right=1, bottom=1),
        target_region=None,
        fact_key=FactKey(subject="horses", dimension="count"),
    )
    assert not plan_mismatch(draft, {"asked_detail": "count", "instruction": "..."})
    assert plan_mismatch(draft, {"asked_detail": "attribute", "instruction": "..."})
    assert not plan_mismatch(draft, {"answer": "no", "instruction": "..."})
    assert not plan_mismatch(draft, None)


def test_the_drafter_never_sees_a_presence_answer_form() -> None:
    tasks = {task.id: task for task in task_catalog().tasks}
    for task_id in ("object_presence", "false_premise_question"):
        assert tasks[task_id].answer_format is not None
        assert draft_task_contract(tasks[task_id])["answer_format"] is None
    assert draft_task_contract(tasks["text_field_extraction"])["answer_format"] is not None
