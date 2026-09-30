"""Isolated symbolic proofs from explicit visual premises."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_geometry import (
    GeometryNumericAnswer,
    GeometryPremise,
    GeometryProblem,
    verify_geometry_problem,
)
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _premise(
    rule: str, variables: tuple[str, ...], constants: tuple[str, ...] = ()
) -> GeometryPremise:
    text = f"{rule} " + " ".join(constants)
    return GeometryPremise.model_validate(
        {
            "rule": rule,
            "variables": variables,
            "constants": constants,
            "evidence_text": text,
            "region": REGION,
        }
    )


def _problem() -> GeometryProblem:
    return GeometryProblem(
        coverage="MET",
        domain="right-triangle-rational",
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        premises=(
            _premise("given", ("a",), ("3",)),
            _premise("given", ("b",), ("4",)),
            _premise("pythagorean", ("a", "b", "c")),
        ),
        target="c",
        target_domain="length",
        reason="Printed lengths and right-angle mark",
    )


@pytest.mark.parametrize(
    ("rule", "variables", "constants"),
    [
        ("given", ("angle_A",), ()),
        ("right_angle", ("angle_D",), ("90",)),
        ("triangle_angle_sum", ("angle_A", "angle_B"), ()),
        ("parallel_equal_angle", ("angle_A",), ()),
        ("similar_ratio", ("a", "b"), ("2",)),
        ("pythagorean", ("a", "b"), ()),
    ],
)
def test_incomplete_registered_premise_cannot_claim_valid_geometry(
    rule: str, variables: tuple[str, ...], constants: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError, match="requires"):
        _premise(rule, variables, constants)


def test_geometry_variables_and_target_must_reach_the_symbolic_worker() -> None:
    with pytest.raises(ValidationError, match="pattern"):
        _premise("given", ("angle A",), ("30",))
    with pytest.raises(ValidationError, match="distinct"):
        _premise("triangle_angle_sum", ("A", "A", "B"))
    source = _problem().model_dump()
    source["target"] = "unbound"
    with pytest.raises(ValidationError, match="bind the target"):
        GeometryProblem.model_validate(source)


def _answer(text: str) -> GeometryNumericAnswer:
    return GeometryNumericAnswer(coverage="MET", answer_quote=text, reported=text, reason="literal")


def test_numeric_answer_requires_literal_fields_in_schema() -> None:
    schema = GeometryNumericAnswer.model_json_schema()
    assert {"coverage", "answer_quote", "reported", "reason"} <= set(schema["required"])
    with pytest.raises(ValidationError, match="answer_quote"):
        GeometryNumericAnswer.model_validate({"coverage": "MET", "reason": "looks plausible"})


@pytest.mark.parametrize(
    ("reported", "expected"), [("60", GateVerdict.MET), ("70", GateVerdict.NOT_MET)]
)
def test_triangle_angle_consensus_preserves_commutativity_and_literal_givens(reported, expected):
    source = _problem().model_copy(
        update={
            "target": "B",
            "target_domain": "angle",
            "premises": (
                _premise("given", ("A",), ("30",)),
                _premise("right_angle", ("D",)),
                _premise("triangle_angle_sum", ("A", "B", "D")),
            ),
        }
    )
    reordered = source.model_copy(
        update={
            "premises": (
                *source.premises[:2],
                _premise("triangle_angle_sum", ("D", "A", "B")),
            )
        }
    )
    answer = _answer(reported)
    assert (
        verify_geometry_problem(
            (source, reordered),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            reported,
        )[0]
        is expected
    )
    changed = reordered.model_copy(
        update={
            "premises": (
                _premise("given", ("A",), ("40",)),
                *reordered.premises[1:],
            )
        }
    )
    assert (
        verify_geometry_problem(
            (source, changed),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            reported,
        )[0]
        is GateVerdict.UNKNOWN
    )
    with_extra = reordered.model_copy(
        update={"premises": (*reordered.premises, _premise("given", ("B",), ("60",)))}
    )
    assert (
        verify_geometry_problem(
            (source, with_extra),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            reported,
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_pythagorean_leg_order_is_exchangeable_but_hypotenuse_is_not():
    source = _problem()
    answer = _answer("5")

    def other(variables):
        return source.model_copy(
            update={"premises": (*source.premises[:2], _premise("pythagorean", variables))}
        )

    assert (
        verify_geometry_problem(
            (source, other(("b", "a", "c"))),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            "5",
        )[0]
        is GateVerdict.MET
    )
    assert (
        verify_geometry_problem(
            (source, other(("a", "c", "b"))),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            "5",
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_equivalent_premise_permutations_cannot_hide_duplicate_facts():
    source = _problem().model_dump()
    source["premises"] = (
        *source["premises"],
        _premise("pythagorean", ("b", "a", "c")).model_dump(),
    )
    with pytest.raises(ValidationError, match="Repeated geometry premise"):
        GeometryProblem.model_validate(source)


@pytest.mark.parametrize(
    ("rule", "constants", "reported", "swapped_verdict"),
    [
        ("parallel_equal_angle", (), "30", GateVerdict.MET),
        ("similar_ratio", ("2", "3"), "45", GateVerdict.UNKNOWN),
    ],
)
def test_equality_is_symmetric_but_ratio_argument_order_is_preserved(
    rule, constants, reported, swapped_verdict
):
    source = _problem().model_copy(
        update={
            "target": "B",
            "target_domain": "angle",
            "premises": (_premise("given", ("A",), ("30",)), _premise(rule, ("A", "B"), constants)),
        }
    )
    swapped = source.model_copy(
        update={"premises": (source.premises[0], _premise(rule, ("B", "A"), constants))}
    )
    answer = _answer(reported)
    assert (
        verify_geometry_problem(
            (source, swapped),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            reported,
        )[0]
        is swapped_verdict
    )


def test_right_triangle_rule_derives_unique_positive_length() -> None:
    source = _problem()
    answer = _answer("5")
    verdict, evidence = verify_geometry_problem(
        (source, source), (answer, answer), source.domain, source.scope_id, source.view_id, "5"
    )
    assert verdict is GateVerdict.MET
    assert evidence["expected"] == "5"
    wrong, _ = verify_geometry_problem(
        (source, source),
        (_answer("6"), _answer("6")),
        source.domain,
        source.scope_id,
        source.view_id,
        "6",
    )
    assert wrong is GateVerdict.NOT_MET


def test_geometry_agreement_compares_facts_independent_of_prose_and_order() -> None:
    source = _problem()
    reordered = source.model_copy(
        update={
            "reason": "Independent visual explanation",
            "premises": tuple(
                item.model_copy(update={"evidence_text": f"Visible mark {item.constants}"})
                for item in reversed(source.premises)
            ),
        }
    )
    answer = _answer("5")
    verdict, evidence = verify_geometry_problem(
        (source, reordered), (answer, answer), source.domain, source.scope_id, source.view_id, "5"
    )
    assert verdict is GateVerdict.MET
    assert evidence["expected"] == "5"
    changed = reordered.model_copy(
        update={
            "premises": (
                reordered.premises[0],
                reordered.premises[1],
                _premise("given", ("a",), ("5",)),
            )
        }
    )
    assert (
        verify_geometry_problem(
            (source, changed),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            "5",
        )[0]
        is GateVerdict.UNKNOWN
    )
    displaced = reordered.model_copy(
        update={
            "premises": (
                reordered.premises[0].model_copy(
                    update={"region": ImageRegion(left=0.8, top=0.8, right=0.9, bottom=0.9)}
                ),
                *reordered.premises[1:],
            )
        }
    )
    assert (
        verify_geometry_problem(
            (source, displaced),
            (answer, answer),
            source.domain,
            source.scope_id,
            source.view_id,
            "5",
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_duplicate_geometry_premises_are_rejected() -> None:
    source = _problem()
    with pytest.raises(ValidationError, match="Repeated geometry premise"):
        GeometryProblem.model_validate(
            source.model_dump() | {"premises": (*source.premises, source.premises[0])}
        )


def test_missing_premises_and_model_domain_mismatch_abstain() -> None:
    source = _problem()
    incomplete = source.model_copy(update={"premises": source.premises[-1:]})
    answer = _answer("5")
    verdict, _ = verify_geometry_problem(
        (incomplete, incomplete),
        (answer, answer),
        source.domain,
        source.scope_id,
        source.view_id,
        "5",
    )
    assert verdict is GateVerdict.UNKNOWN
    verdict, _ = verify_geometry_problem(
        (source, source),
        (answer, answer),
        "other-model-domain",
        source.scope_id,
        source.view_id,
        "5",
    )
    assert verdict is GateVerdict.UNKNOWN
