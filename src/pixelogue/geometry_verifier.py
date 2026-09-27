"""Geometry relations grounded in explicit diagram marks and visible outlines."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion

GeometryPredicate = Literal[
    "shape_class",
    "symmetry",
    "composition",
    "equal",
    "parallel",
    "perpendicular",
]


class GeometryFact(StrictModel):
    """One relation with visible evidence and its exact support strength."""

    predicate: GeometryPredicate
    subjects: Annotated[tuple[str, ...], Field(min_length=1, max_length=4)]
    value: str | bool
    support: Literal["explicit_mark", "printed_constraint", "visible_outline", "appearance_only"]
    region: ImageRegion
    visible_detail: str = Field(min_length=1)


class GeometryQuery(StrictModel):
    """Public relation and requested answer form."""

    predicate: GeometryPredicate
    subjects: tuple[str, ...] = ()
    answer_form: Literal["value", "boolean", "relations"]


class GeometrySource(StrictModel):
    """Answer-blind closed inventory of explicitly marked geometric facts."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    facts: Annotated[tuple[GeometryFact, ...], Field(max_length=128)]
    query: GeometryQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> GeometrySource:
        """Reject duplicate relation claims and off-scope diagram evidence."""
        identities = [(fact.predicate, fact.subjects) for fact in self.facts]
        if len(identities) != len(set(identities)):
            raise ValueError("Conflicting duplicate geometric relation")
        if self.coverage != "MET" and (self.closed or self.facts):
            raise ValueError("Incomplete geometry extraction cannot certify facts")
        for fact in self.facts:
            region = fact.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Geometry mark outside requested scope")
        return self


class GeometryAnswer(StrictModel):
    """Answer-only value, truth claim or list of marked relations."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    value: str | None = None
    truth: bool | None = None
    relations: tuple[tuple[str, ...], ...] | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> GeometryAnswer:
        """Require one complete quoted answer form."""
        populated = sum(item is not None for item in (self.value, self.truth, self.relations))
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete geometry answer requires one quoted form")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete geometry answer cannot supply a result")
        return self


def _canonical(source: GeometrySource) -> dict[str, object]:
    result = source.model_dump(mode="json", exclude={"reason"})
    for fact in result["facts"]:
        fact.pop("region", None)
        fact.pop("visible_detail", None)
    result["facts"].sort(key=lambda item: (item["predicate"], item["subjects"]))
    return result


def verify_geometry(
    sources: tuple[GeometrySource, GeometrySource],
    answers: tuple[GeometryAnswer, GeometryAnswer],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> GateVerdict:
    """Compare literal marked relations; approximate shape appearance never proves equality."""
    if any(
        source.coverage != "MET"
        or not source.closed
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ) or _canonical(sources[0]) != _canonical(sources[1]):
        return GateVerdict.UNKNOWN
    if any(
        answer.coverage != "MET" or answer.answer_quote not in candidate_answer
        for answer in answers
    ):
        return GateVerdict.UNKNOWN
    if answers[0].model_dump(mode="json", exclude={"reason"}) != answers[1].model_dump(
        mode="json", exclude={"reason"}
    ):
        return GateVerdict.UNKNOWN
    source, answer = sources[0], answers[0]
    query = source.query
    matching = [fact for fact in source.facts if fact.predicate == query.predicate]
    if query.predicate in {"equal", "parallel", "perpendicular", "symmetry"}:
        if any(fact.support not in {"explicit_mark", "printed_constraint"} for fact in matching):
            return GateVerdict.UNKNOWN
    elif any(fact.support == "appearance_only" for fact in matching):
        return GateVerdict.UNKNOWN
    expected: object | None
    observed: object | None
    if query.answer_form == "relations":
        if query.subjects:
            return GateVerdict.UNKNOWN
        expected = tuple(sorted(fact.subjects for fact in matching if fact.value is True))
        observed = tuple(sorted(answer.relations)) if answer.relations is not None else None
        if answer.relations is not None and any(
            subject not in answer.answer_quote
            for relation in answer.relations
            for subject in relation
        ):
            return GateVerdict.UNKNOWN
    else:
        fact = next((fact for fact in matching if fact.subjects == query.subjects), None)
        if fact is None:
            return GateVerdict.UNKNOWN
        if query.answer_form == "boolean" and isinstance(fact.value, bool):
            expected, observed = fact.value, answer.truth
        elif query.answer_form == "value" and isinstance(fact.value, str):
            expected, observed = fact.value, answer.value
            if answer.value is not None and answer.value not in answer.answer_quote:
                return GateVerdict.UNKNOWN
        else:
            return GateVerdict.UNKNOWN
    if observed is None:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if observed == expected else GateVerdict.NOT_MET
