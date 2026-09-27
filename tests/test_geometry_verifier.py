"""Explicit geometry marks versus unreliable visual appearance."""

from __future__ import annotations

from pixelogue.contracts import GateVerdict
from pixelogue.geometry_verifier import (
    GeometryAnswer,
    GeometryFact,
    GeometryQuery,
    GeometrySource,
    verify_geometry,
)
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _source(facts: tuple[GeometryFact, ...], query: GeometryQuery) -> GeometrySource:
    return GeometrySource(
        coverage="MET",
        closed=True,
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        facts=facts,
        query=query,
        reason="All explicit relation marks visible",
    )


def _answer(quote: str, **value: object) -> GeometryAnswer:
    return GeometryAnswer.model_validate(
        {"coverage": "MET", "answer_quote": quote, "reason": "literal", **value}
    )


def _check(source: GeometrySource, answer: GeometryAnswer) -> GateVerdict:
    return verify_geometry(
        (source, source), (answer, answer), source.scope_id, source.view_id, answer.answer_quote
    )


def test_marked_equal_sides_are_enumerated_exactly() -> None:
    fact = GeometryFact(
        predicate="equal",
        subjects=("AB", "CD"),
        value=True,
        support="explicit_mark",
        region=REGION,
        visible_detail="matching tick marks",
    )
    source = _source((fact,), GeometryQuery(predicate="equal", answer_form="relations"))
    assert _check(source, _answer("AB and CD", relations=(("AB", "CD"),))) is GateVerdict.MET
    assert _check(source, _answer("AB", relations=(("AB",),))) is GateVerdict.NOT_MET


def test_approximate_parallelism_cannot_prove_exact_relation() -> None:
    appearance = GeometryFact(
        predicate="parallel",
        subjects=("AB", "CD"),
        value=True,
        support="appearance_only",
        region=REGION,
        visible_detail="looks nearly parallel",
    )
    source = _source(
        (appearance,),
        GeometryQuery(predicate="parallel", subjects=("AB", "CD"), answer_form="boolean"),
    )
    assert _check(source, _answer("yes", truth=True)) is GateVerdict.UNKNOWN
    marked = source.model_copy(
        update={"facts": (appearance.model_copy(update={"support": "explicit_mark"}),)}
    )
    assert _check(marked, _answer("yes", truth=True)) is GateVerdict.MET


def test_visible_shape_class_is_allowed_but_unmarked_symmetry_abstains() -> None:
    outline = GeometryFact(
        predicate="shape_class",
        subjects=("S",),
        value="triangle",
        support="visible_outline",
        region=REGION,
        visible_detail="three sides",
    )
    source = _source(
        (outline,), GeometryQuery(predicate="shape_class", subjects=("S",), answer_form="value")
    )
    assert _check(source, _answer("triangle", value="triangle")) is GateVerdict.MET
    symmetry = GeometryFact(
        predicate="symmetry",
        subjects=("S",),
        value=True,
        support="visible_outline",
        region=REGION,
        visible_detail="roughly mirrored",
    )
    source = _source(
        (symmetry,), GeometryQuery(predicate="symmetry", subjects=("S",), answer_form="boolean")
    )
    assert _check(source, _answer("yes", truth=True)) is GateVerdict.UNKNOWN


def test_source_disagreement_and_other_view_abstain() -> None:
    fact = GeometryFact(
        predicate="perpendicular",
        subjects=("AB", "BC"),
        value=True,
        support="explicit_mark",
        region=REGION,
        visible_detail="right-angle square",
    )
    source = _source(
        (fact,),
        GeometryQuery(predicate="perpendicular", subjects=("AB", "BC"), answer_form="boolean"),
    )
    answer = _answer("yes", truth=True)
    different = source.model_copy(update={"facts": (fact.model_copy(update={"value": False}),)})
    assert (
        verify_geometry(
            (source, different),
            (answer, answer),
            source.scope_id,
            source.view_id,
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_geometry(
            (source, source), (answer, answer), source.scope_id, "other", answer.answer_quote
        )
        is GateVerdict.UNKNOWN
    )
