"""Pre-answer question gates and holistic decisions for the direct planner.

Two blind judges vote independently. Neither sees the other's vote, and no rule averages
or majority-votes a disagreement. The only follow-ups are one fresh view for an UNKNOWN
judge and one answer repair after a single concrete NOT_MET objection.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.evaluation import consensus

Verdict = Literal["MET", "NOT_MET", "UNKNOWN"]


class QuestionGateVote(StrictModel):
    """One judge's pre-answer question assessment and independent operation label.

    The label comes last: emitted first, it was chosen before any reading of the question and
    then rationalized, which also flipped the coherence verdict.
    """

    local_anchor: Verdict
    operation_coherent: Verdict
    useful_request: Verdict
    reason: Annotated[str, Field(min_length=1, max_length=240)]
    realized_task_id: Annotated[str, Field(min_length=1, max_length=64)] | None

    @property
    def fit(self) -> GateVerdict:
        """Reduce the three question checks without averaging."""
        values = {
            GateVerdict(self.local_anchor),
            GateVerdict(self.operation_coherent),
            GateVerdict(self.useful_request),
        }
        if GateVerdict.NOT_MET in values:
            return GateVerdict.NOT_MET
        if values == {GateVerdict.MET}:
            return GateVerdict.MET
        return GateVerdict.UNKNOWN


@dataclass(frozen=True)
class GateDecision:
    """Combined label and fit outcome for one drafted question."""

    verdict: GateVerdict
    task_id: str
    label_rule: Literal["exact", "same_contract", "relabel", "label_mismatch", "label_unresolved"]
    fit: GateVerdict


def _combine(first: GateVerdict, second: GateVerdict) -> GateVerdict:
    if GateVerdict.ERROR in (first, second):
        return GateVerdict.ERROR
    if GateVerdict.NOT_MET in (first, second):
        return GateVerdict.NOT_MET
    if first is GateVerdict.MET and second is GateVerdict.MET:
        return GateVerdict.MET
    return GateVerdict.UNKNOWN


def question_gate_decision(
    votes: Sequence[QuestionGateVote],
    selected_task_id: str,
    *,
    policy: Literal["strict", "same_contract"],
    contracts: Mapping[str, tuple[str, ...]],
    relabel_allowed: Callable[[str], bool],
) -> GateDecision:
    """Decide whether a question realizes its operation and may receive an answer.

    ``strict`` requires both labels to equal the drafted operation. ``same_contract`` also
    accepts a neighboring label with identical verification contracts, because the label
    then only changes provenance. When both judges agree on such a neighbor, the turn is
    relabeled if the draft already carries that operation's required public choices.
    """
    if len(votes) != 2:
        return GateDecision(
            GateVerdict.ERROR, selected_task_id, "label_unresolved", GateVerdict.ERROR
        )
    fit = consensus([vote.fit for vote in votes])
    labels = [vote.realized_task_id for vote in votes]
    expected = contracts.get(selected_task_id)

    def equivalent(label: str | None) -> bool:
        return label is not None and expected is not None and contracts.get(label) == expected

    task_id = selected_task_id
    if labels == [selected_task_id, selected_task_id]:
        label_verdict, rule = GateVerdict.MET, "exact"
    elif (
        policy == "same_contract"
        and selected_task_id in labels
        and all(equivalent(label) for label in labels)
    ):
        label_verdict, rule = GateVerdict.MET, "same_contract"
    elif labels[0] is not None and labels[0] == labels[1]:
        neighbor = labels[0]
        if policy == "same_contract" and equivalent(neighbor) and relabel_allowed(neighbor):
            label_verdict, rule, task_id = GateVerdict.MET, "relabel", neighbor
        else:
            label_verdict, rule = GateVerdict.NOT_MET, "label_mismatch"
    else:
        label_verdict, rule = GateVerdict.UNKNOWN, "label_unresolved"
    return GateDecision(_combine(label_verdict, fit), task_id, rule, fit)


@dataclass(frozen=True)
class HolisticDecision:
    """Outcome of two whole-turn votes, with at most one permitted follow-up."""

    verdict: GateVerdict
    action: Literal["none", "tiebreak", "repair"]
    judge_index: int | None = None
    objection: str | None = None


def holistic_decision(
    verdicts: Sequence[GateVerdict],
    reasons: Sequence[str],
    *,
    tiebreak_available: bool,
    repair_available: bool,
) -> HolisticDecision:
    """Reduce two blind holistic votes and name the one allowed follow-up, if any."""
    verdict = consensus(verdicts)
    if verdict is not GateVerdict.UNKNOWN or len(verdicts) != 2:
        return HolisticDecision(verdict, "none")
    pair = list(verdicts)
    if repair_available and sorted(item.value for item in pair) == ["MET", "NOT_MET"]:
        index = pair.index(GateVerdict.NOT_MET)
        return HolisticDecision(GateVerdict.UNKNOWN, "repair", index, reasons[index])
    if tiebreak_available and sorted(item.value for item in pair) == ["MET", "UNKNOWN"]:
        return HolisticDecision(GateVerdict.UNKNOWN, "tiebreak", pair.index(GateVerdict.UNKNOWN))
    return HolisticDecision(GateVerdict.UNKNOWN, "none")
