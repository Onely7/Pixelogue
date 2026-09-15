"""Question-fit consensus and rubric aggregation."""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

from pixelogue.catalog import load_rubric_catalog
from pixelogue.contracts import (
    GateVerdict,
    PublicMessage,
    QuestionFit,
    RubricContext,
    RubricItem,
    TurnRating,
)
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


def repeated_public_question(question: str, history: Sequence[PublicMessage]) -> bool:
    """Return whether a question repeats an earlier user message after surface normalization."""
    normalized = _normalize_public_question(question)
    return any(
        message.role == "user" and _normalize_public_question(message.content) == normalized
        for message in history
    )


def _normalize_public_question(question: str) -> str:
    """Normalize harmless Unicode, case, whitespace, and terminal punctuation differences."""
    normalized = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(normalized.split()).rstrip(".!?。！？ ")


def has_natural_language_content(text: str) -> bool:
    """Return whether text contains a Unicode letter that needs language evaluation."""
    return any(character.isalpha() for character in text)


def applicable_rubric_items(context: RubricContext) -> list[dict[str, Any]]:
    """Instantiate catalog predicates using controller-owned context fields."""
    catalog = load_rubric_catalog()
    applicable: list[dict[str, Any]] = []
    for item in catalog["items"]:
        predicate = item["applies_when"]
        if predicate == "always":
            applicable.append(item)
        elif predicate == "natural_language_answer" and context.has_natural_language_answer:
            applicable.append(item)
        elif predicate == "answer_has_non_exempt_natural_language" and (
            context.has_natural_language_answer
        ):
            applicable.append(item)
        elif predicate == "has_history" and context.turn_index > 1:
            applicable.append(item)
        elif predicate == "has_history_binding" and context.history_binding_ids:
            applicable.append(item)
        elif predicate == "has_computation" and context.computation_ids:
            applicable.append(item)
        elif predicate == "limitation_profile" and context.profile == "limitation":
            applicable.append(item)
        elif predicate == "false_premise_profile" and context.profile == "false_premise":
            applicable.append(item)
        elif predicate == "designated_strong_dependency_turn" and (context.requires_witness_check):
            applicable.append(item)
        elif predicate == "exhaustive_request" and context.exhaustive_scope_ids:
            applicable.append(item)
        elif predicate == "later_turn" and context.turn_index > 1:
            applicable.append(item)
        elif predicate == "public_format_constraint" and context.format_requirements:
            applicable.append(item)
        elif predicate == "each_public_requirement" and context.requirements:
            applicable.append(item)
        elif predicate == "each_factual_claim" and context.claims:
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
