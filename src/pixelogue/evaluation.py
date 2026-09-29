"""Question-fit consensus and rubric aggregation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any

from pixelogue.catalog import load_rubric_catalog, task_catalog
from pixelogue.contracts import (
    GateVerdict,
    PublicMessage,
    QuestionFit,
    RubricContext,
    RubricItem,
    TurnRating,
)
from pixelogue.serialization import canonical_hash

GENERIC_IDENTIFICATION_REFERENTS = frozenset(
    {
        "animal",
        "bird",
        "car",
        "creature",
        "dog",
        "equipment",
        "flower",
        "food",
        "fruit",
        "insect",
        "item",
        "object",
        "person",
        "plant",
        "structure",
        "subject",
        "thing",
        "vehicle",
    }
)


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


def repeated_answered_request(question: str, answer: str, history: Sequence[PublicMessage]) -> bool:
    """Catch a paraphrased request that reproduces the same substantial answer."""
    normalized_answer = _normalize_public_question(answer)
    if len(normalized_answer) < 32:
        return False
    normalized_question = _normalize_public_question(question)
    questions = {
        message.turn_index: _normalize_public_question(message.content)
        for message in history
        if message.role == "user"
    }
    return any(
        _normalize_public_question(message.content) == normalized_answer
        and (earlier := questions.get(message.turn_index)) is not None
        and SequenceMatcher(None, earlier, normalized_question).ratio() >= 0.75
        for message in history
        if message.role == "assistant"
    )


def identification_label_in_question(question: str, label: str) -> bool:
    """Find a complete object label already disclosed in an identification question."""
    label_words = _public_words(label)
    if (
        not label_words
        or not any(character.isalpha() for character in label)
        or " ".join(label_words) in GENERIC_IDENTIFICATION_REFERENTS
    ):
        return False
    return _contains_public_phrase(question, label_words)


def identification_answer_in_question(question: str, answer: str) -> bool:
    """Catch short literal identification answers that the question already supplies."""
    words = _public_words(answer)
    if words[:2] in (["it", "is"], ["this", "is"], ["that", "is"]):
        words = words[2:]
    if words and words[0] in {"a", "an", "the"}:
        words = words[1:]
    return bool(0 < len(words) <= 6 and _contains_public_phrase(question, words))


def identification_answer_in_history(answer: str, history: Sequence[PublicMessage]) -> bool:
    """Reject a short identification label already present in public dialogue."""
    words = normalized_identification_words(answer)
    return bool(
        0 < len(words) <= 6
        and len("".join(words)) >= 3
        and any(_contains_identification_phrase(message.content, words) for message in history)
    )


_QUANTITY_PREFIXES = frozenset(
    {
        "a",
        "an",
        "the",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    }
)


def _singular_identification_word(word: str) -> str:
    if word in {"series", "species"}:
        return word
    if len(word) > 4 and word.endswith("ies"):
        return f"{word[:-3]}y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def normalized_identification_words(text: str) -> list[str]:
    """Normalize a short object label for duplicate checks, not visual truth."""
    words = _public_words(text)
    while words and (words[0] in _QUANTITY_PREFIXES or words[0].isdigit()):
        words = words[1:]
    if words[:2] in (["pair", "of"], ["group", "of"], ["set", "of"]):
        words = words[2:]
    if words:
        words[-1] = _singular_identification_word(words[-1])
    return words


def identification_target_named_in_evidence(label: str, details: Sequence[str]) -> bool:
    """Check that cited observations name an identification target's head noun.

    This lexical guard checks citation consistency, not whether the image is correct.
    """
    words = normalized_identification_words(label)
    return bool(
        words and any(_contains_identification_phrase(detail, words[-1:]) for detail in details)
    )


def _contains_identification_phrase(text: str, words: list[str]) -> bool:
    if not words:
        return False
    observed = _public_words(text)
    return any(
        observed[start : start + len(words) - 1] == words[:-1]
        and _singular_identification_word(observed[start + len(words) - 1]) == words[-1]
        for start in range(len(observed) - len(words) + 1)
    )


def transcription_answer_in_question(question: str, answer: str) -> bool:
    """Reject a transcription already quoted in its own question."""
    words = _public_words(answer)
    return bool(words and len("".join(words)) >= 4 and _contains_public_phrase(question, words))


_UNVERIFIED_TEXT_RELATION = re.compile(
    r"\b(?:above|below|beneath|underneath|next\s+to|beside|"
    r"(?:to\s+the\s+)?(?:left|right)\s+of)\b",
    re.IGNORECASE,
)


def unverified_transcription_relation(question: str) -> bool:
    """Flag spatial text locators unsupported by the transcript evidence contract."""
    return _UNVERIFIED_TEXT_RELATION.search(question) is not None


def scene_options_in_question(question: str, options: tuple[str, ...]) -> bool:
    """Require every declared scene choice to be visible in the public request."""
    return bool(
        len(options) >= 2
        and all(
            (words := _public_words(option)) and _contains_public_phrase(question, words)
            for option in options
        )
    )


_ACTION_FORMS = {
    form: action
    for action, forms in {
        "sit": ("sit", "sits", "sitting", "seated"),
        "stand": ("stand", "stands", "standing"),
        "lie": ("lie", "lies", "lying"),
        "walk": ("walk", "walks", "walking"),
        "run": ("run", "runs", "running"),
        "drink": ("drink", "drinks", "drinking"),
        "eat": ("eat", "eats", "eating"),
        "draw": ("draw", "draws", "drawing"),
        "rest": ("rest", "rests", "resting"),
        "paint": ("paint", "paints", "painting"),
        "write": ("write", "writes", "writing"),
        "read": ("read", "reads", "reading"),
        "drive": ("drive", "drives", "driving"),
        "ride": ("ride", "rides", "riding"),
        "hold": ("hold", "holds", "holding"),
        "grip": ("grip", "grips", "gripping"),
        "look": ("look", "looks", "looking"),
        "jump": ("jump", "jumps", "jumping"),
    }.items()
    for form in forms
}


def action_answer_already_public(
    question: str, answer: str, prior_scope_texts: Sequence[str]
) -> bool:
    """Catch a leading action supplied by the question or same-scope history."""
    action = next(
        (_ACTION_FORMS[word] for word in _public_words(answer) if word in _ACTION_FORMS),
        None,
    )
    return bool(
        action is not None
        and any(
            action in {_ACTION_FORMS[word] for word in _public_words(text) if word in _ACTION_FORMS}
            for text in (question, *prior_scope_texts)
        )
    )


def _contains_public_phrase(question: str, phrase_words: list[str]) -> bool:
    question_words = _public_words(question)
    for start, word in enumerate(question_words):
        if word != phrase_words[0]:
            continue
        position = start
        for next_word in phrase_words[1:]:
            position = next(
                (
                    index
                    for index in range(position + 1, min(position + 4, len(question_words)))
                    if question_words[index] == next_word
                ),
                len(question_words),
            )
            if position == len(question_words):
                break
        else:
            return True
    return False


def _public_words(text: str) -> list[str]:
    """Use Unicode word boundaries and normalize common label separators."""
    return re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold().replace("_", " "))


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
    uses["Q_HOLISTIC"] = "gate"
    uses.update({name: "gate" for name in task_catalog().verification_contracts})
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
