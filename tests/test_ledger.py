import pytest

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
    inventory = RequirementInventory(requirements=(spec,), coverage="MET", reason="Complete.")
    message = _Message("q1", "user", "Answer briefly")
    requirements = reconcile_inventories((inventory, inventory), (message,), 1)
    assert requirements is not None and requirements[0].description == "briefly"

    disagreement = RequirementInventory(requirements=(), coverage="MET", reason="None.")
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
        coverage="MET",
        reason="Complete.",
    )
    message = _Message("q1", "user", "How many squares?")
    requirements = reconcile_inventories((inventory, inventory), (message,), 1)
    assert requirements is not None
    assert (requirements[0].start, requirements[0].end) == (0, 17)

    ambiguous = _Message("q1", "user", "How many squares? How many squares?")
    assert reconcile_inventories((inventory, inventory), (ambiguous,), 1) is None
