"""Public-requirement records that keep turns saved by the retired ledger readable."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.errors import ExecutionError

RequirementKind = Literal["content", "format", "language", "scope", "style"]
RequirementLifetime = Literal["current_turn", "persistent"]


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
