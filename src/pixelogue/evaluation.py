"""Question-fit consensus and rubric aggregation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher

from pixelogue.catalog import task_catalog
from pixelogue.contracts import (
    GateVerdict,
    PublicMessage,
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


def reciprocal_identification_disclosure(
    question: str,
    target: str,
    prior_pairs: Sequence[tuple[PublicMessage, PublicMessage]],
) -> bool:
    """Catch a reciprocal naming question whose answer was already public.

    Both directions must be literal short labels. This keeps distinct targets with the
    same category eligible when the prior question did not disclose the new answer.
    """
    return any(
        identification_answer_in_question(prior_question.content, target)
        and identification_answer_in_question(question, prior_answer.content)
        for prior_question, prior_answer in prior_pairs
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


_ACTION_AFFORDANCE = re.compile(
    r"\b(?:supported\s+to\s+(?:perform|do)|capable\s+of|able\s+to|"
    r"(?:what|which)\s+(?:(?:action|activity)\s+)?(?:can|could)\s+"
    r"(?:the|this|that|it|he|she|they)\b[^?.!]{0,80}\b(?:do|perform))\b",
    re.IGNORECASE,
)


def action_affordance_question(question: str) -> bool:
    """Catch an overt ability request where the task requires an observed action."""
    return _ACTION_AFFORDANCE.search(question) is not None


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


def question_fingerprint(question: str) -> str:
    """Identify exact normalized repeats without exposing the rejected question text."""
    return canonical_hash(_normalize_public_question(question))


def aggregate_rating(items: Sequence[RubricItem]) -> TurnRating:
    """Aggregate the holistic review and operation checks, each of which is a gate.

    Raises:
        KeyError: If an item names neither the holistic review nor a verification contract.
    """
    gate_ids = {"Q_HOLISTIC", *task_catalog().verification_contracts}
    unknown = sorted({item.template_id for item in items} - gate_ids)
    if unknown:
        raise KeyError(f"Unknown rating criteria: {unknown}")
    gates = [item.verdict for item in items]
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
