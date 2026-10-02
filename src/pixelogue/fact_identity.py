"""Conservative controller identities for explicitly specified visual requests."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pixelogue.contracts import InstructionCandidate
from pixelogue.serialization import canonical_hash


@dataclass(frozen=True)
class RequestedFactKey:
    """An operation and its unchanged subject, dimension and public conditions.

    This is private routing metadata, not a claim that a visual answer is correct. Missing
    choices leave the identity unresolved rather than treating a broad task as one fact.
    """

    view_id: str
    scope_id: str
    subject: str
    operation: str
    dimension: str
    conditions_hash: str


def _normalize(value: str) -> str:
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold().replace("_", " "))
    while words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return " ".join("color" if word == "colour" else word for word in words)


def requested_fact_key(candidate: InstructionCandidate) -> RequestedFactKey | None:
    """Identify only requests whose subject and operation choices are explicit.

    Legacy and current labels can share an operation. Description, summary, action and
    arbitrary QA are intentionally unresolved because their task label does not specify
    exactly which fact will be requested.
    """
    if candidate.profile != "normal" or not candidate.view_id or not candidate.scope_id:
        return None
    parameters = {parameter.name: parameter for parameter in candidate.public_parameters}
    target = parameters.get("target")
    if target is None or not isinstance(target.value, str):
        return None
    operation: str
    dimension: str
    used = {"target", "format", "detail_level"}
    if candidate.task_id == "attribute_lookup":
        property_choice = parameters.get("attribute")
        if (
            property_choice is None
            or property_choice.origin != "instruction"
            or not isinstance(property_choice.value, str)
        ):
            return None
        operation, dimension = "attribute", _normalize(property_choice.value)
        used.add("attribute")
    elif candidate.task_id in {"entity_count", "visible_count"}:
        unit = parameters.get("count_unit")
        if unit is None or not isinstance(unit.value, str):
            return None
        operation, dimension = "cardinality", _normalize(unit.value)
        used.add("count_unit")
    elif candidate.task_id == "text_transcription":
        operation, dimension = "transcription", "literal text"
    elif candidate.task_id == "object_identification":
        operation, dimension = "category", "visible category"
    elif candidate.task_id in {"spatial_relation", "relation_lookup"}:
        relation, frame = parameters.get("relation"), parameters.get("frame")
        if (
            relation is None
            or not isinstance(relation.value, str)
            or frame is None
            or not isinstance(frame.value, str)
        ):
            return None
        operation, dimension = "spatial relation", _normalize(relation.value)
        used.add("relation")
    else:
        return None
    if not dimension or not (subject := _normalize(target.value)):
        return None
    return RequestedFactKey(
        view_id=candidate.view_id,
        scope_id=candidate.scope_id,
        subject=subject,
        operation=operation,
        dimension=dimension,
        conditions_hash=canonical_hash(
            {
                "version": 1,
                "target_region": (
                    candidate.target_region.model_dump(mode="json")
                    if candidate.target_region is not None
                    else None
                ),
                "choices": {
                    name: parameter.value
                    for name, parameter in sorted(parameters.items())
                    if name not in used
                },
            }
        ),
    )
