"""Short knowledge answers checked by agreement with two blind, independent readers.

World and specialist knowledge cannot be recomputed from pixels, so these operations commit only
when two readers who never see the candidate answer give the same short answer as the candidate.
Normalization removes case, punctuation, leading articles and number formatting; aliases and
other spelling differences are not resolved and therefore leave the result undecided.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Literal

from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict

CONSENSUS_TASKS = frozenset(
    {
        "named_entity_recognition",
        "style_recognition",
        "map_region_identification",
        "notation_interpretation",
        "math_word_problem",
    }
)
_ARTICLES = frozenset({"a", "an", "the"})
_NUMBER = re.compile(r"([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)(.*)")


class ConsensusSource(StrictModel):
    """One reader's answer-blind short answer and the visible cues it rests on.

    Fields are ordered so that a reader states its cues and answer before deciding coverage.
    """

    visible_evidence: str = Field(max_length=400)
    short_answer: str = Field(max_length=200)
    coverage: Literal["MET", "UNKNOWN"]
    reason: str = Field(min_length=1)


class ConsensusAnswer(StrictModel):
    """An image-free parse of the candidate's final short answer.

    The quote and short answer come before coverage, so the parse is written before the parser
    decides whether it is complete.
    """

    answer_quote: str = Field(max_length=600)
    short_answer: str = Field(max_length=200)
    coverage: Literal["MET", "UNKNOWN"]
    reason: str = Field(min_length=1)


def normalize_short_answer(text: str) -> str:
    """Return a comparison key that ignores case, punctuation, articles and number format.

    A leading number becomes an exact rational value followed by its unit words, so ``1,200 m``
    and ``1200.0 m`` compare equal while ``1200 km`` does not.
    """
    folded = unicodedata.normalize("NFKC", text).casefold().strip().rstrip(".")
    match = _NUMBER.fullmatch(folded)
    if match is not None:
        try:
            value = Fraction(Decimal(match[1].replace(",", "")))
        except InvalidOperation:
            value = None
        if value is not None:
            unit = " ".join(re.findall(r"[^\W_]+|%", match[2]))
            return f"number {value} {unit}".strip()
    words = re.findall(r"[^\W_]+", folded)
    while words and words[0] in _ARTICLES:
        words = words[1:]
    return " ".join(words)


def verify_consensus(
    sources: tuple[ConsensusSource, ConsensusSource],
    answers: tuple[ConsensusAnswer, ConsensusAnswer],
    candidate_answer: str,
) -> GateVerdict:
    """Compare the candidate's short answer with two blind readers.

    Returns:
        MET when all three agree, NOT_MET when the readers agree with each other but not with
        the candidate, and UNKNOWN when any reader abstains, the readers disagree, or the
        candidate's short answer cannot be parsed from its text.
    """
    folded_candidate = unicodedata.normalize("NFKC", candidate_answer).casefold()
    for answer in answers:
        if (
            answer.coverage != "MET"
            or not answer.short_answer.strip()
            or not answer.answer_quote
            or answer.answer_quote not in candidate_answer
            or unicodedata.normalize("NFKC", answer.short_answer).casefold().strip()
            not in folded_candidate
        ):
            return GateVerdict.UNKNOWN
    parsed = {normalize_short_answer(answer.short_answer) for answer in answers}
    if len(parsed) != 1 or "" in parsed:
        return GateVerdict.UNKNOWN
    if any(source.coverage != "MET" or not source.short_answer.strip() for source in sources):
        return GateVerdict.UNKNOWN
    readers = {normalize_short_answer(source.short_answer) for source in sources}
    if len(readers) != 1 or "" in readers:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if parsed == readers else GateVerdict.NOT_MET
