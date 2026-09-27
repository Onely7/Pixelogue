"""Isolated symbolic proofs from explicit visual premises."""

from __future__ import annotations

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


def _answer(text: str) -> GeometryNumericAnswer:
    return GeometryNumericAnswer(coverage="MET", answer_quote=text, reported=text, reason="literal")


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
