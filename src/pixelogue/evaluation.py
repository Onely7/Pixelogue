"""Question-fit consensus and rubric aggregation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pixelogue.catalog import load_rubric_catalog
from pixelogue.contracts import GateVerdict, QuestionFit, RubricItem, TurnRating
from pixelogue.serialization import canonical_hash


def consensus(verdicts: Sequence[GateVerdict]) -> GateVerdict:
    """Require two matching semantic votes without majority fallback."""
    if len(verdicts) != 2:
        return GateVerdict.ERROR
    if GateVerdict.ERROR in verdicts:
        return GateVerdict.ERROR
    if verdicts[0] is verdicts[1] and verdicts[0] in {
        GateVerdict.MET,
        GateVerdict.NOT_MET,
    }:
        return verdicts[0]
    return GateVerdict.UNKNOWN


def question_fit_consensus(votes: Sequence[QuestionFit]) -> GateVerdict:
    """Reduce two complete question-fit votes."""
    return consensus([vote.aggregate for vote in votes])


def applicable_rubric_items(context: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Instantiate catalog predicates using controller-owned context fields."""
    catalog = load_rubric_catalog()
    applicable: list[dict[str, Any]] = []
    for item in catalog["items"]:
        predicate = item["applies_when"]
        if predicate == "always":
            applicable.append(item)
        elif predicate == "natural_language_answer" and context.get("has_natural_language_answer"):
            applicable.append(item)
        elif predicate == "answer_has_non_exempt_natural_language" and context.get(
            "has_natural_language_answer"
        ):
            applicable.append(item)
        elif predicate == "has_history" and context.get("turn_index", 1) > 1:
            applicable.append(item)
        elif predicate == "has_history_binding" and context.get("has_history_binding"):
            applicable.append(item)
        elif predicate == "has_computation" and context.get("has_computation"):
            applicable.append(item)
        elif predicate == "limitation_profile" and context.get("profile") == "limitation":
            applicable.append(item)
        elif predicate == "false_premise_profile" and context.get("profile") == "false_premise":
            applicable.append(item)
        elif predicate == "designated_strong_dependency_turn" and context.get("strong_dependency"):
            applicable.append(item)
        elif predicate == "exhaustive_request" and context.get("exhaustive_request"):
            applicable.append(item)
        elif predicate == "later_turn" and context.get("turn_index", 1) > 1:
            applicable.append(item)
        elif predicate in {"public_format_constraint", "each_public_requirement"} and context.get(
            "format_requirements" if predicate == "public_format_constraint" else "requirements"
        ):
            applicable.append(item)
        elif predicate == "each_factual_claim" and context.get("claims"):
            applicable.append(item)
    return applicable


def aggregate_rating(items: Sequence[RubricItem]) -> TurnRating:
    """Aggregate gate criteria while retaining annotations and every axis."""
    catalog = load_rubric_catalog()
    uses = {item["template_id"]: item["default_use"] for item in catalog["items"]}
    gates = [item.verdict for item in items if uses[item.template_id] == "gate"]
    if GateVerdict.ERROR in gates:
        aggregate = "ERROR"
    elif GateVerdict.NOT_MET in gates:
        aggregate = "FAIL"
    elif any(verdict in {GateVerdict.UNKNOWN, GateVerdict.NOT_EVALUATED} for verdict in gates):
        aggregate = "ABSTAIN"
    else:
        aggregate = "PASS"
    return TurnRating(items=tuple(items), aggregate=aggregate)


def item_id(
    conversation_id: str,
    turn_index: int,
    attempt: int,
    template_id: str,
    subject_id: str,
    context_hash: str,
) -> str:
    """Build a stable identity for one rubric instance."""
    return canonical_hash(
        {
            "conversation_id": conversation_id,
            "turn_index": turn_index,
            "attempt": attempt,
            "template_id": template_id,
            "subject_id": subject_id,
            "context_hash": context_hash,
        }
    )
