"""Immutable public-requirement contracts and ledger transitions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.errors import ExecutionError
from pixelogue.serialization import canonical_hash

RequirementKind = Literal["content", "format", "language", "scope", "style"]
RequirementLifetime = Literal["current_turn", "persistent"]


class RequirementSpec(StrictModel):
    """One public constraint extracted without looking at an answer."""

    kind: RequirementKind
    text: str = Field(min_length=1)
    lifetime: RequirementLifetime
    source_message_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class RequirementInventory(StrictModel):
    """A judge's complete inventory of active public constraints."""

    requirements: tuple[RequirementSpec, ...]
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: str = Field(min_length=1, max_length=240)


class Requirement(StrictModel):
    """One active public constraint introduced by a user turn."""

    requirement_id: str
    template_id: str
    kind: RequirementKind
    description: str
    lifetime: RequirementLifetime
    introduced_turn: int = Field(ge=1)
    source_message_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    def validate_source(self, messages: Sequence[object]) -> None:
        """Require an exact span in a public user message.

        Raises:
            ExecutionError: If the source message or exact text span is invalid.
        """
        for message in messages:
            if getattr(message, "message_id", None) != self.source_message_id:
                continue
            if getattr(message, "role", None) != "user":
                break
            content = getattr(message, "content", "")
            if self.end <= len(content) and content[self.start : self.end] == self.description:
                return
            break
        raise ExecutionError(
            "REQUIREMENT_SOURCE_MISMATCH",
            "A requirement must point to an exact span in a public user message",
        )


def reconcile_inventories(
    inventories: Sequence[RequirementInventory],
    messages: Sequence[object],
    turn_index: int,
) -> tuple[Requirement, ...] | None:
    """Return controller-owned requirements when two blind inventories agree.

    Reasons are intentionally excluded from agreement. Any incomplete inventory,
    disagreement, duplicate, or invalid public source span returns ``None``.
    """
    if len(inventories) != 2 or any(item.coverage != "MET" for item in inventories):
        return None
    normalized = [
        tuple(
            sorted(
                (spec.model_dump(mode="json") for spec in inventory.requirements),
                key=lambda value: (
                    value["source_message_id"],
                    value["start"],
                    value["end"],
                    value["kind"],
                    value["lifetime"],
                ),
            )
        )
        for inventory in inventories
    ]
    if normalized[0] != normalized[1]:
        return None
    requirements: list[Requirement] = []
    seen: set[str] = set()
    for value in normalized[0]:
        requirement_id = canonical_hash(value)
        if requirement_id in seen:
            return None
        seen.add(requirement_id)
        requirement = Requirement(
            requirement_id=requirement_id,
            template_id="R_REQUIREMENT",
            kind=value["kind"],
            description=value["text"],
            lifetime=value["lifetime"],
            introduced_turn=turn_index,
            source_message_id=value["source_message_id"],
            start=value["start"],
            end=value["end"],
        )
        try:
            requirement.validate_source(messages)
        except ExecutionError:
            return None
        requirements.append(requirement)
    return tuple(requirements)


class RequirementEvent(StrictModel):
    """One explicit change to the requirement ledger."""

    action: Literal["introduce", "retain", "override", "revoke"]
    requirement: Requirement | None = None
    previous_requirement_id: str | None = None

    @model_validator(mode="after")
    def validate_action(self) -> RequirementEvent:
        """Require fields appropriate to each action."""
        if self.action == "introduce" and self.requirement is None:
            raise ValueError("introduce requires a requirement")
        if self.action in {"retain", "revoke"} and self.previous_requirement_id is None:
            raise ValueError(f"{self.action} requires previous_requirement_id")
        if self.action == "override" and (
            self.previous_requirement_id is None or self.requirement is None
        ):
            raise ValueError("override requires old and new requirements")
        return self


class RequirementLedger(StrictModel):
    """Active constraints after one public user turn."""

    turn_index: int = Field(default=0, ge=0)
    active: tuple[Requirement, ...] = ()

    def apply(self, turn_index: int, events: tuple[RequirementEvent, ...]) -> RequirementLedger:
        """Apply explicit events and expire local constraints.

        Raises:
            ExecutionError: If an event duplicates or refers to an inactive requirement.
        """
        active = {
            item.requirement_id: item
            for item in self.active
            if item.lifetime == "persistent" or item.introduced_turn == turn_index
        }
        for event in events:
            previous_id = event.previous_requirement_id
            if event.action == "introduce":
                assert event.requirement is not None
                if event.requirement.requirement_id in active:
                    raise ExecutionError("DUPLICATE_REQUIREMENT", "Requirement already exists")
                active[event.requirement.requirement_id] = event.requirement
            elif event.action == "retain":
                if previous_id not in active:
                    raise ExecutionError(
                        "UNKNOWN_REQUIREMENT", "Cannot retain an inactive requirement"
                    )
            elif event.action == "revoke":
                if previous_id not in active:
                    raise ExecutionError(
                        "UNKNOWN_REQUIREMENT", "Cannot revoke an inactive requirement"
                    )
                del active[previous_id]
            else:
                if previous_id not in active:
                    raise ExecutionError(
                        "UNKNOWN_REQUIREMENT", "Cannot override an inactive requirement"
                    )
                assert event.requirement is not None
                del active[previous_id]
                if event.requirement.requirement_id in active:
                    raise ExecutionError("DUPLICATE_REQUIREMENT", "Replacement ID already exists")
                active[event.requirement.requirement_id] = event.requirement
        return RequirementLedger(turn_index=turn_index, active=tuple(active.values()))
