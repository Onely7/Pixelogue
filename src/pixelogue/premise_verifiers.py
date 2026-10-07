"""Presence and false-premise answers checked against two blind presence readings.

An object-presence question asks whether something is in the image; a false-premise question
assumes something the image does not show. Two readers who never see the candidate answer decide
whether the asked or assumed object is present, and two image-free parses read the candidate's
conclusion. A turn commits only when both readers agree with each other and with that conclusion.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict

PREMISE_TASKS = frozenset({"object_presence", "false_premise_question"})
# A false-premise question is valid only when the assumed object or detail is absent.
REQUIRED_STATUS = {"false_premise_question": "absent"}
PresenceStatus = Literal["present", "absent"]


class PremiseSource(StrictModel):
    """One reader's answer-blind decision on whether the asked or assumed object is present.

    The reader lists what it sees and names the object before deciding, and decides coverage last.
    """

    visible_objects: str = Field(max_length=400)
    premise: str = Field(max_length=200)
    status: PresenceStatus
    coverage: Literal["MET", "UNKNOWN"]
    reason: str = Field(min_length=1)


class PremiseAnswer(StrictModel):
    """An image-free parse of whether the candidate answer says the object is present."""

    answer_quote: str = Field(max_length=600)
    status: PresenceStatus
    coverage: Literal["MET", "UNKNOWN"]
    reason: str = Field(min_length=1)


def verify_premise(
    task_id: str,
    sources: tuple[PremiseSource, PremiseSource],
    answers: tuple[PremiseAnswer, PremiseAnswer],
    candidate_answer: str,
) -> GateVerdict:
    """Compare the candidate's presence conclusion with two blind readers.

    Returns:
        MET when both readers and both parses give the same status, NOT_MET when the readers
        agree with each other but not with the candidate, and UNKNOWN when a reader or parse
        abstains, they disagree, a quote is not in the answer, or a false-premise question's
        assumed object turns out to be present.

    Raises:
        ValueError: If the task has no presence verifier.
    """
    if task_id not in PREMISE_TASKS:
        raise ValueError("No presence verifier for this task")
    if any(
        answer.coverage != "MET"
        or not answer.answer_quote.strip()
        or answer.answer_quote not in candidate_answer
        for answer in answers
    ):
        return GateVerdict.UNKNOWN
    if answers[0].status != answers[1].status:
        return GateVerdict.UNKNOWN
    if any(source.coverage != "MET" for source in sources):
        return GateVerdict.UNKNOWN
    if sources[0].status != sources[1].status:
        return GateVerdict.UNKNOWN
    observed = sources[0].status
    if REQUIRED_STATUS.get(task_id, observed) != observed:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if answers[0].status == observed else GateVerdict.NOT_MET
