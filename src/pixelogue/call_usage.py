"""Measure reported usage independently of the validity of generated content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pixelogue.errors import ExternalInputError
from pixelogue.serialization import strict_json_object


@dataclass(frozen=True)
class TokenUsage:
    """Known token counts, retaining absent or untrustworthy measurements as unknown."""

    prompt_tokens: int | None
    completion_tokens: int | None
    status: Literal["MEASURED", "PARTIAL", "MISSING", "INVALID"]


def measure_usage(response: bytes, max_tokens: int) -> TokenUsage:
    """Read usage without requiring valid choices or schema-compliant model content."""
    try:
        envelope = strict_json_object(response)
    except ExternalInputError:
        return TokenUsage(None, None, "INVALID")
    usage = envelope.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage(None, None, "MISSING")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if any(
        value is not None and (type(value) is not int or value < 0)
        for value in (prompt, completion)
    ):
        return TokenUsage(None, None, "INVALID")
    if completion is not None and completion > max_tokens:
        return TokenUsage(None, None, "INVALID")
    status: Literal["MEASURED", "PARTIAL", "MISSING", "INVALID"] = (
        "MEASURED"
        if prompt is not None and completion is not None
        else "PARTIAL"
        if prompt is not None or completion is not None
        else "MISSING"
    )
    return TokenUsage(prompt, completion, status)
