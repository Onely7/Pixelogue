"""Controller checks for closed, image-bound finite-set operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion

FINITE_TASKS = frozenset(
    {
        "spatial_ordering",
        "set_cardinality_comparison",
        "quantified_statement_verification",
        "grounded_hypothetical_update",
    }
)


class FiniteMember(StrictModel):
    """One individually visible member of a closed scope."""

    member_id: str = Field(min_length=1)
    groups: tuple[str, ...] = ()
    predicate_met: bool | None = None
    order_index: int | None = None
    region: ImageRegion


class FiniteQuery(StrictModel):
    """Answer-independent interpretation of the public question."""

    answer_form: Literal["members", "count", "boolean", "relation"]
    left_group: str | None = None
    right_group: str | None = None
    quantifier: Literal["all", "none", "some", "exactly", "at_least", "at_most"] | None = None
    threshold: Annotated[int, Field(ge=0)] | None = None
    update: Literal["add", "remove", "relabel"] | None = None
    update_ids: tuple[str, ...] = ()
    update_group: str | None = None


class FiniteSource(StrictModel):
    """Independent complete visual inventory, with no candidate answer."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    scope_id: str
    view_id: str = Field(min_length=1)
    scope_region: ImageRegion
    members: Annotated[tuple[FiniteMember, ...], Field(max_length=256)]
    query: FiniteQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_members(self) -> FiniteSource:
        """Prevent ambiguous member identities and visual closure claims."""
        names = [member.member_id for member in self.members]
        if len(names) != len(set(names)):
            raise ValueError("Finite member IDs must be unique")
        if self.coverage != "MET" and (self.closed or self.members):
            raise ValueError("Incomplete visual extraction cannot certify members or closure")
        for member in self.members:
            region = member.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Finite member is outside the declared scope")
        return self


class FiniteAnswer(StrictModel):
    """Typed parsing of public answer text without the image or source inventory."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_form: Literal["members", "count", "boolean", "relation"]
    members: tuple[str, ...] = ()
    count: Annotated[int, Field(ge=0)] | None = None
    truth: bool | None = None
    relation: Literal["less", "equal", "greater"] | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> FiniteAnswer:
        """Require exactly the declared result kind."""
        populated = {
            "count": self.count is not None,
            "boolean": self.truth is not None,
            "relation": self.relation is not None,
        }
        if self.coverage != "MET" and (self.members or any(populated.values())):
            raise ValueError("Incomplete answer extraction cannot present a result")
        if self.coverage == "MET":
            if self.answer_form == "members":
                if any(populated.values()):
                    raise ValueError("Member enumeration cannot include a scalar result")
            elif self.members or sum(populated.values()) != 1 or not populated[self.answer_form]:
                raise ValueError("Complete answer extraction needs its declared result")
        if self.coverage == "MET" and len(self.members) != len(set(self.members)):
            raise ValueError("Duplicate answer members are not a valid enumeration")
        return self


def _canonical_source(source: FiniteSource) -> dict[str, object]:
    """Compare semantic extraction while tolerating slightly different region coordinates."""
    return {
        "task_id": source.task_id,
        "scope_id": source.scope_id,
        "view_id": source.view_id,
        "scope_region": source.scope_region.model_dump(mode="json"),
        "query": source.query.model_dump(mode="json"),
        "members": sorted(
            (
                member.member_id,
                tuple(sorted(member.groups)),
                member.predicate_met,
                member.order_index,
            )
            for member in source.members
        ),
    }


def _expected(source: FiniteSource) -> tuple[str, object] | None:
    """Compute the public operation on the certified finite inventory."""
    task = source.task_id
    query = source.query
    members = source.members
    if task == "spatial_ordering":
        indices = [member.order_index for member in members]
        if any(index is None for index in indices) or set(indices) != set(range(len(members))):
            return None
        return "members", tuple(
            member.member_id for member in sorted(members, key=lambda item: item.order_index or 0)
        )
    if task == "set_cardinality_comparison":
        if not query.left_group or not query.right_group:
            return None
        left = sum(query.left_group in member.groups for member in members)
        right = sum(query.right_group in member.groups for member in members)
        relation = "less" if left < right else "greater" if left > right else "equal"
        return "relation", relation
    if task == "quantified_statement_verification":
        if query.quantifier is None or any(member.predicate_met is None for member in members):
            return None
        matches = sum(member.predicate_met is True for member in members)
        total = len(members)
        if query.quantifier == "all":
            truth = matches == total
        elif query.quantifier == "none":
            truth = matches == 0
        elif query.quantifier == "some":
            truth = matches > 0
        elif query.threshold is None:
            return None
        elif query.quantifier == "exactly":
            truth = matches == query.threshold
        elif query.quantifier == "at_least":
            truth = matches >= query.threshold
        else:
            truth = matches <= query.threshold
        return "boolean", truth
    if task == "grounded_hypothetical_update":
        if query.update is None or not query.update_ids:
            return None
        current = {member.member_id for member in members}
        updates = set(query.update_ids)
        if query.update == "add" and current & updates:
            return None
        if query.update in {"remove", "relabel"} and not updates <= current:
            return None
        if query.update == "relabel" and (not query.update_group or not query.left_group):
            return None
        groups = {member.member_id: set(member.groups) for member in members}
        if query.update == "add":
            for member_id in updates:
                groups[member_id] = {query.update_group} if query.update_group else set()
        elif query.update == "remove":
            for member_id in updates:
                del groups[member_id]
        else:
            if query.update_group is None:
                return None
            for member_id in updates:
                groups[member_id] = {query.update_group}
        result = {
            member_id
            for member_id, memberships in groups.items()
            if query.left_group is None or query.left_group in memberships
        }
        if query.answer_form == "count":
            return "count", len(result)
        if query.answer_form == "members":
            return "members", tuple(sorted(result))
    return None


def verify_finite(
    task_id: str,
    sources: tuple[FiniteSource, FiniteSource],
    answers: tuple[FiniteAnswer, FiniteAnswer],
    public_parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
) -> GateVerdict:
    """Reject disagreement and compare an independently parsed answer to a closed source."""
    if task_id not in FINITE_TASKS:
        raise ValueError("No finite verifier for this task")
    if any(
        source.coverage != "MET"
        or not source.closed
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN
    if _canonical_source(sources[0]) != _canonical_source(sources[1]):
        return GateVerdict.UNKNOWN
    if any(answer.coverage != "MET" for answer in answers):
        return GateVerdict.UNKNOWN
    parsed = [answer.model_dump(mode="json", exclude={"reason"}) for answer in answers]
    if parsed[0] != parsed[1]:
        return GateVerdict.UNKNOWN
    query = sources[0].query
    if query.quantifier is not None and public_parameters.get("quantifier") != query.quantifier:
        return GateVerdict.UNKNOWN
    if query.update is not None and public_parameters.get("update") != query.update:
        return GateVerdict.UNKNOWN
    expected = _expected(sources[0])
    if (
        expected is None
        or query.answer_form != expected[0]
        or answers[0].answer_form != expected[0]
    ):
        return GateVerdict.UNKNOWN
    form, value = expected
    observed: object
    if form == "members":
        observed = answers[0].members
        if task_id != "spatial_ordering":
            observed = tuple(sorted(observed))
    elif form == "count":
        observed = answers[0].count
    elif form == "boolean":
        observed = answers[0].truth
    else:
        observed = answers[0].relation
    return GateVerdict.MET if observed == value else GateVerdict.NOT_MET
