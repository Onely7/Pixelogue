"""Exact quantitative comparisons from independent visual evidence."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.quantitative_verifiers import (
    CONVERSION_VERSION,
    QuantityAnswer,
    QuantityOperand,
    QuantitySource,
    verify_quantitative,
)
from pixelogue.rules import NumericValue
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_verification import verify_operation

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _operand(key: str, number: str, unit: str | None, weight: str | None = None) -> QuantityOperand:
    return QuantityOperand(
        operand_id=key,
        value=NumericValue(value=number, unit=unit),
        printed_text=f"{key}: {number} {unit or ''} {weight or ''}",
        region=REGION,
        weight=weight,
    )


def _source(
    task_id: str,
    operation: str,
    operands: tuple[QuantityOperand, ...],
    **options: object,
) -> QuantitySource:
    return QuantitySource.model_validate(
        {
            "task_id": task_id,
            "coverage": "MET",
            "closed": True,
            "scope_id": "scope-a",
            "view_id": "view-a",
            "scope_region": REGION,
            "operands": operands,
            "query": {
                "operation": operation,
                "operand_ids": tuple(item.operand_id for item in operands),
                **options,
            },
            "reason": "Every requested value is visible in this scope",
        }
    )


def _answer(number: str, unit: str | None = None) -> QuantityAnswer:
    quote = f"{number} {unit}" if unit else number
    return QuantityAnswer(
        coverage="MET",
        answer_quote=quote,
        value=NumericValue(value=number, unit=unit),
        reason="Parsed literal answer",
    )


def _check(source: QuantitySource, answer: QuantityAnswer, **parameters: object) -> GateVerdict:
    return verify_quantitative(
        source.task_id,
        (source, source),
        (answer, answer),
        parameters,
        source.scope_id,
        source.view_id,
        answer.answer_quote,
    )


def test_primitive_arithmetic_keeps_exact_unit_contract() -> None:
    source = _source(
        "grounded_arithmetic", "add", (_operand("a", "2.5", "kg"), _operand("b", "1.25", "kg"))
    )
    assert _check(source, _answer("3.75", "kg"), operators="add") is GateVerdict.MET
    assert _check(source, _answer("3.74", "kg"), operators="add") is GateVerdict.NOT_MET
    assert _check(source, _answer("3.75", "g"), operators="add") is GateVerdict.NOT_MET
    assert _check(source, _answer("3.75", "kg"), operators="divide") is GateVerdict.UNKNOWN


def test_comparison_and_cross_region_consistency_require_comparable_units() -> None:
    values = (_operand("a", "2", "kg"), _operand("b", "3", "kg"))
    comparison = _source("quantity_comparison", "compare", values)
    answer = QuantityAnswer(coverage="MET", answer_quote="less", relation="less", reason="literal")
    assert _check(comparison, answer) is GateVerdict.MET
    equality = _source("stated_value_consistency", "equal", values)
    boolean = QuantityAnswer(coverage="MET", answer_quote="false", truth=False, reason="literal")
    assert _check(equality, boolean) is GateVerdict.MET
    incomparable = comparison.model_copy(update={"operands": (values[0], _operand("b", "3", "m"))})
    assert _check(incomparable, answer) is GateVerdict.UNKNOWN


def test_quantity_comparison_and_consistency_reject_wrong_relation_and_unreadable_source() -> None:
    values = (_operand("a", "2", "kg"), _operand("b", "3", "kg"))
    comparison = _source("quantity_comparison", "compare", values)
    wrong_relation = QuantityAnswer(
        coverage="MET", answer_quote="greater", relation="greater", reason="literal"
    )
    assert _check(comparison, wrong_relation) is GateVerdict.NOT_MET
    assert (
        _check(comparison.model_copy(update={"coverage": "UNKNOWN"}), wrong_relation)
        is GateVerdict.UNKNOWN
    )
    consistency = _source("stated_value_consistency", "equal", values)
    wrong_truth = QuantityAnswer(coverage="MET", answer_quote="true", truth=True, reason="literal")
    assert _check(consistency, wrong_truth) is GateVerdict.NOT_MET
    assert (
        _check(consistency.model_copy(update={"coverage": "UNKNOWN"}), wrong_truth)
        is GateVerdict.UNKNOWN
    )


def test_closed_aggregation_and_half_up_rounding() -> None:
    values = (_operand("a", "1", "m"), _operand("b", "2", "m"), _operand("c", "2", "m"))
    source = _source("value_aggregation", "mean", values, decimal_places=2, rounding="half_up")
    assert _check(source, _answer("1.67", "m"), operator="mean") is GateVerdict.MET
    assert _check(source, _answer("1.66", "m"), operator="mean") is GateVerdict.NOT_MET
    assert (
        _check(source.model_copy(update={"closed": False}), _answer("1.67", "m"))
        is GateVerdict.UNKNOWN
    )


def test_weighted_mean_and_versioned_conversion() -> None:
    weighted = _source(
        "value_aggregation",
        "weighted_mean",
        (_operand("a", "2", "m", "1"), _operand("b", "4", "m", "3")),
    )
    assert _check(weighted, _answer("3.5", "m")) is GateVerdict.MET
    conversion = _source(
        "unit_conversion",
        "convert",
        (_operand("a", "2.5", "m"),),
        target_unit="cm",
        conversion_version=CONVERSION_VERSION,
    )
    assert _check(conversion, _answer("250", "cm")) is GateVerdict.MET
    assert _check(conversion, _answer("2.5", "cm")) is GateVerdict.NOT_MET
    unsupported = conversion.model_copy(
        update={"query": conversion.query.model_copy(update={"target_unit": "USD"})}
    )
    assert _check(unsupported, _answer("250", "USD")) is GateVerdict.UNKNOWN


def test_disagreement_and_wrong_view_abstain() -> None:
    source = _source(
        "grounded_arithmetic", "multiply", (_operand("a", "2", None), _operand("b", "3", None))
    )
    answer = _answer("6")
    other = source.model_copy(
        update={"operands": (_operand("a", "2", None), _operand("b", "4", None))}
    )
    assert (
        verify_quantitative(
            source.task_id,
            (source, other),
            (answer, answer),
            {},
            source.scope_id,
            source.view_id,
            "6",
        )
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_quantitative(
            source.task_id, (source, source), (answer, answer), {}, source.scope_id, "other", "6"
        )
        is GateVerdict.UNKNOWN
    )


def test_pipeline_never_shows_answer_to_quantity_source() -> None:
    task = next(item for item in task_catalog().tasks if item.id == "quantity_comparison")
    instruction = InstructionCandidate(
        candidate_id="candidate",
        task_id=task.id,
        family=task.family,
        visible_scope="two labels",
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version="8.0",
        scope_id="scope-a",
        view_id="view-a",
        evidence_refs=("e1",),
        verification_contracts=task.verification_contracts,
    )
    source = _source(
        "quantity_comparison", "compare", (_operand("a", "2", "kg"), _operand("b", "3", "kg"))
    )
    answer = QuantityAnswer(coverage="MET", answer_quote="less", relation="less", reason="literal")
    seen: list[str] = []

    def invoke(
        stage: str, payload: dict[str, Any], model: type[BaseModel], judge: int
    ) -> BaseModel:
        seen.append(stage)
        if stage == "quantity_source":
            assert "candidate_answer" not in payload
            assert model is QuantitySource
            return source
        assert stage == "quantity_answer"
        assert "image_views" not in payload
        assert model is QuantityAnswer
        return answer

    checks = verify_operation(
        instruction,
        {
            "target_language": "en",
            "public_history": [],
            "question": "Compare a and b",
            "candidate_answer": "a is less",
            "image_views": [{"view_id": "view-a"}],
        },
        invoke,
    )
    assert seen == ["quantity_source", "quantity_source", "quantity_answer", "quantity_answer"]
    assert checks[0].verdict is GateVerdict.MET
