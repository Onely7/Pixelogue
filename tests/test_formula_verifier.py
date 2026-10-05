"""Two-dimensional formula syntax rather than algebraic equivalence."""

from __future__ import annotations

from typing import Literal

import pytest

from pixelogue.contracts import GateVerdict
from pixelogue.formula_verifier import FormulaSource, parse_formula, verify_formula
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _source(formula: str, notation: Literal["latex", "unicode_math"] = "latex") -> FormulaSource:
    return FormulaSource.model_validate(
        {
            "coverage": "MET",
            "scope_id": "scope",
            "view_id": "view",
            "scope_region": REGION,
            "formula_region": REGION,
            "notation": notation,
            "root": parse_formula(formula, notation),
            "reason": "All marks are legible",
        }
    )


def _check(source: FormulaSource, answer: str) -> GateVerdict:
    return verify_formula(
        (source, source), source.notation, source.scope_id, source.view_id, answer
    )


def test_stacked_fraction_is_not_slash_or_algebraic_equivalence() -> None:
    source = _source(r"\frac{a+b}{2}")
    assert _check(source, r"\frac{a+b}{2}") is GateVerdict.MET
    assert _check(source, "(a+b)/2") is GateVerdict.NOT_MET
    assert _check(_source("a+b"), "b+a") is GateVerdict.NOT_MET


def test_scripts_grouping_and_greek_symbols_are_structural() -> None:
    source = _source(r"\alpha_{i}^{2}+x")
    assert _check(source, r"\alpha^{2}_{i}+x") is GateVerdict.MET
    assert _check(source, r"\alpha_{i}+x") is GateVerdict.NOT_MET
    assert _check(_source("a+(b)"), "a+b") is GateVerdict.NOT_MET


def test_unicode_notation_is_supported_without_equation_solving() -> None:
    source = _source("α²+β", "unicode_math")
    assert _check(source, "α²+β") is GateVerdict.MET
    assert _check(source, "β+α²") is GateVerdict.NOT_MET


def test_unknown_notation_and_visual_disagreement_abstain() -> None:
    source = _source(r"\sqrt{x}")
    assert _check(source, r"\unknown{x}") is GateVerdict.UNKNOWN
    assert _check(_source("arrow"), r"\leftarrow") is GateVerdict.UNKNOWN
    other = _source(r"\sqrt{y}")
    assert (
        verify_formula((source, other), "latex", "scope", "view", r"\sqrt{x}")
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_formula((source, source), "latex", "scope", "other", r"\sqrt{x}")
        is GateVerdict.UNKNOWN
    )


def test_malformed_root_and_nonlocal_region_are_rejected() -> None:
    with pytest.raises(ValueError):
        parse_formula(r"\frac{a}", "latex")
    source = _source("a+b")
    with pytest.raises(ValueError):
        FormulaSource.model_validate_json(
            source.model_copy(
                update={
                    "formula_region": ImageRegion(left=0.9, top=0.9, right=1, bottom=1),
                    "scope_region": ImageRegion(left=0, top=0, right=0.8, bottom=0.8),
                }
            ).model_dump_json()
        )


@pytest.mark.parametrize(
    "opening,closing", [(r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")]
)
def test_display_containers_preserve_the_literal_formula(opening: str, closing: str) -> None:
    source = _source(r"\frac{a+b}{2}")
    assert _check(source, opening + r"\frac{a+b}{2}" + closing) is GateVerdict.MET
    assert _check(source, opening + "(a+b)/2" + closing) is GateVerdict.NOT_MET
    assert _check(source, opening + r"\frac{a+b}{2}") is GateVerdict.UNKNOWN
    assert _check(source, opening + r"\frac{a+b}{2}" + closing + " extra") is GateVerdict.UNKNOWN
