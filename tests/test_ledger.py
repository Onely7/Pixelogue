import pytest
from pydantic import ValidationError

from pixelogue.errors import ExecutionError
from pixelogue.ledger import (
    Requirement,
    RequirementEvent,
    RequirementInventory,
    RequirementLedger,
    RequirementSpec,
    reconcile_inventories,
)


class _Message:
    def __init__(self, message_id: str, role: str, content: str) -> None:
        self.message_id = message_id
        self.role = role
        self.content = content


def test_requirement_lifetimes_and_unknown_transitions() -> None:
    persistent = Requirement(
        requirement_id="persistent",
        template_id="format",
        kind="format",
        description="Answer briefly.",
        lifetime="persistent",
        introduced_turn=1,
        source_message_id="q1",
        start=0,
        end=15,
    )
    local = Requirement(
        requirement_id="local",
        template_id="format",
        kind="format",
        description="Use one word.",
        lifetime="current_turn",
        introduced_turn=1,
        source_message_id="q1",
        start=0,
        end=13,
    )
    ledger = RequirementLedger().apply(
        1,
        (
            RequirementEvent(action="introduce", requirement=persistent),
            RequirementEvent(action="introduce", requirement=local),
        ),
    )
    next_turn = ledger.apply(
        2,
        (RequirementEvent(action="retain", previous_requirement_id="persistent"),),
    )
    assert {item.requirement_id for item in next_turn.active} == {"persistent"}
    with pytest.raises(ExecutionError) as caught:
        next_turn.apply(
            3,
            (RequirementEvent(action="retain", previous_requirement_id="local"),),
        )
    assert caught.value.reason == "UNKNOWN_REQUIREMENT"


def test_requirement_inventory_requires_blind_agreement_and_public_span() -> None:
    spec = RequirementSpec(
        kind="format",
        text="briefly",
        lifetime="current_turn",
        source_message_id="q1",
        start=7,
        end=14,
    )
    inventory = RequirementInventory(
        requirements=(spec,), extraction_complete=True, reason="Complete."
    )
    message = _Message("q1", "user", "Answer briefly")
    requirements = reconcile_inventories((inventory, inventory), (message,), 1)
    assert requirements is not None and requirements[0].description == "briefly"

    disagreement = RequirementInventory(requirements=(), extraction_complete=True, reason="None.")
    assert reconcile_inventories((inventory, disagreement), (message,), 1) is None
    assistant = _Message("q1", "assistant", "Answer briefly")
    assert reconcile_inventories((inventory, inventory), (assistant,), 1) is None


def test_requirement_inventory_corrects_only_a_unique_quoted_span() -> None:
    miscounted = RequirementSpec(
        kind="content",
        text="How many squares?",
        lifetime="current_turn",
        source_message_id="q1",
        start=0,
        end=40,
    )
    inventory = RequirementInventory(
        requirements=(miscounted,),
        extraction_complete=True,
        reason="Complete.",
    )
    message = _Message("q1", "user", "How many squares?")
    requirements = reconcile_inventories((inventory, inventory), (message,), 1)
    assert requirements is not None
    assert (requirements[0].start, requirements[0].end) == (0, 16)

    ambiguous = _Message("q1", "user", "How many squares? How many squares?")
    assert reconcile_inventories((inventory, inventory), (ambiguous,), 1) is None


def test_requirement_extraction_completeness_is_strict_and_answer_independent():
    question = _Message("q1", "user", "How many dogs are on the left?")
    inventory = RequirementInventory(
        requirements=(
            RequirementSpec(
                kind="content",
                text="How many dogs are on the left",
                lifetime="current_turn",
                source_message_id="q1",
                start=0,
                end=29,
            ),
        ),
        extraction_complete=True,
        reason="The explicit question has been fully extracted.",
    )
    assert reconcile_inventories((inventory, inventory), (question,), 1) is not None
    incomplete = inventory.model_copy(update={"extraction_complete": False})
    assert reconcile_inventories((inventory, incomplete), (question,), 1) is None
    legacy = inventory.model_dump()
    legacy.pop("extraction_complete")
    legacy["coverage"] = "MET"
    with pytest.raises(ValidationError):
        RequirementInventory.model_validate(legacy)
    for value in ("true", 1, "MET"):
        with pytest.raises(ValidationError):
            RequirementInventory.model_validate(
                {**inventory.model_dump(), "extraction_complete": value}
            )


@pytest.mark.parametrize(
    ("question", "left", "right", "agree"),
    [
        ("Can you group the clothes?", "Can you group the clothes?", "group the clothes", True),
        ("Please count the cups.", "Please count the cups.", "count the cups", True),
        (
            "Based only on what is visible, count the cups.",
            "Based only on what is visible, count the cups",
            "count the cups",
            False,
        ),
        ("Do not count the cups.", "Do not count the cups", "count the cups", False),
    ],
)
def test_only_courtesy_and_boundary_punctuation_are_normalized(question, left, right, agree):
    message = _Message("q1", "user", question)
    inventories = [
        RequirementInventory(
            requirements=(
                RequirementSpec(
                    kind="content",
                    text=text,
                    lifetime="current_turn",
                    source_message_id="q1",
                    start=question.index(text),
                    end=question.index(text) + len(text),
                ),
            ),
            extraction_complete=True,
            reason="Complete",
        )
        for text in (left, right)
    ]
    result = reconcile_inventories(inventories, (message,), 1)
    assert (result is not None) == agree
    if result:
        assert question[result[0].start : result[0].end] == result[0].description
