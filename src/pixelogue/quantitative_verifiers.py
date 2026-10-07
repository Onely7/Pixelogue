"""Exact, image-bound quantitative operations over blind source extractions."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from statistics import median
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError
from pixelogue.rules import NumericValue, exact_calculate, parse_numeric_lexeme
from pixelogue.task_evidence import ImageRegion

QUANTITATIVE_TASKS = frozenset(
    {
        "quantity_comparison",
        "stated_value_consistency",
        "grounded_arithmetic",
        "value_aggregation",
        "unit_conversion",
    }
)
ArithmeticOperation = Literal[
    "compare",
    "equal",
    "add",
    "subtract",
    "multiply",
    "divide",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "weighted_mean",
    "convert",
]

# Exact, immutable physical definitions. This list deliberately has no currencies or
# context-dependent units. A changed definition requires a validator version change.
CONVERSION_VERSION = "si-simple-1"
CONVERSIONS: dict[str, tuple[str, Fraction]] = {
    "mm": ("length", Fraction(1, 1000)),
    "cm": ("length", Fraction(1, 100)),
    "m": ("length", Fraction(1)),
    "km": ("length", Fraction(1000)),
    "mg": ("mass", Fraction(1, 1000000)),
    "g": ("mass", Fraction(1, 1000)),
    "kg": ("mass", Fraction(1)),
    "mL": ("volume", Fraction(1, 1000)),
    "L": ("volume", Fraction(1)),
    "s": ("duration", Fraction(1)),
    "min": ("duration", Fraction(60)),
    "h": ("duration", Fraction(3600)),
}


class QuantityOperand(StrictModel):
    """One observed number and its bound image location."""

    operand_id: str = Field(min_length=1)
    value: NumericValue
    printed_text: str = Field(min_length=1)
    region: ImageRegion
    weight: str | None = None

    @model_validator(mode="after")
    def check_text(self) -> QuantityOperand:
        """Require the numerical spelling to occur in the observed text."""
        if self.value.value not in self.printed_text:
            raise ValueError("Operand value must appear in its printed evidence")
        if self.weight is not None and self.weight not in self.printed_text:
            raise ValueError("Weight must appear in its printed evidence")
        return self


class QuantityQuery(StrictModel):
    """Public operation interpreted without seeing a proposed answer."""

    operation: ArithmeticOperation
    operand_ids: tuple[str, ...]
    target_unit: str | None = None
    conversion_version: str | None = None
    decimal_places: Annotated[int, Field(ge=0, le=12)] | None = None
    rounding: Literal["exact", "half_up"] = "exact"
    comparable_basis: bool = True


class QuantitySource(StrictModel):
    """Answer-blind visual extraction of a complete requested numeric expression."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    operands: Annotated[tuple[QuantityOperand, ...], Field(max_length=64)]
    query: QuantityQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> QuantitySource:
        """Reject partial or nonlocal number inventories."""
        ids = [item.operand_id for item in self.operands]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate operand IDs")
        if self.coverage != "MET" and (self.closed or self.operands):
            raise ValueError("Incomplete extraction cannot certify operands")
        for item in self.operands:
            region = item.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Operand outside declared image scope")
        return self


class QuantityAnswer(StrictModel):
    """Parse an answer from text alone, with its exact source substring."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    value: NumericValue | None = None
    relation: Literal["less", "equal", "greater"] | None = None
    truth: bool | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> QuantityAnswer:
        """Do not accept partial results or multiple conflicting answer forms."""
        populated = sum(item is not None for item in (self.value, self.relation, self.truth))
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete answer requires one quoted result")
        if self.value is not None and self.value.value not in self.answer_quote:
            raise ValueError("Parsed number is absent from its answer quote")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete answer cannot supply a result")
        return self


def _number(value: NumericValue) -> Fraction:
    return parse_numeric_lexeme(
        value.value,
        decimal_separator=value.decimal_separator,
        group_separator=value.group_separator,
    )


def _round_half_up(value: Fraction, digits: int) -> Fraction:
    scale = 10**digits
    scaled = value * scale
    absolute = abs(scaled)
    quotient, remainder = divmod(absolute.numerator, absolute.denominator)
    rounded = quotient + (2 * remainder >= absolute.denominator)
    return Fraction((1 if scaled >= 0 else -1) * rounded, scale)


def _canonical_source(source: QuantitySource) -> dict[str, object]:
    """Compare extracted values and rules while tolerating small region differences."""
    result = source.model_dump(mode="json", exclude={"reason"})
    for operand in result["operands"]:
        operand.pop("region", None)
    return result


def _expected(source: QuantitySource) -> tuple[str, Fraction | str | bool, str | None] | None:
    query = source.query
    by_id = {item.operand_id: item for item in source.operands}
    if not query.operand_ids or len(query.operand_ids) != len(set(query.operand_ids)):
        return None
    if any(key not in by_id for key in query.operand_ids) or not query.comparable_basis:
        return None
    if source.task_id == "value_aggregation" and not source.closed:
        return None
    selected = [by_id[key] for key in query.operand_ids]
    values = [_number(item.value) for item in selected]
    units = [item.value.unit for item in selected]
    op = query.operation
    if op == "compare" and source.task_id == "quantity_comparison":
        if len(values) != 2 or units[0] != units[1]:
            return None
        return (
            "relation",
            "less" if values[0] < values[1] else "greater" if values[0] > values[1] else "equal",
            None,
        )
    if op == "equal" and source.task_id == "stated_value_consistency":
        if len(values) != 2 or units[0] != units[1]:
            return None
        return "truth", values[0] == values[1], None
    if op in {"add", "subtract", "multiply", "divide"} and source.task_id in {
        "grounded_arithmetic",
        "stated_value_consistency",
    }:
        value, unit = exact_calculate(op, values, units)
        return "value", value, unit
    if (
        op in {"sum", "mean", "median", "min", "max", "weighted_mean"}
        and source.task_id == "value_aggregation"
    ):
        if len(set(units)) != 1:
            return None
        if op == "sum":
            value = sum(values, Fraction(0))
        elif op == "mean":
            value = sum(values, Fraction(0)) / len(values)
        elif op == "median":
            value = Fraction(median(values))
        elif op == "min":
            value = min(values)
        elif op == "max":
            value = max(values)
        else:
            if any(item.weight is None for item in selected):
                return None
            weights = [parse_numeric_lexeme(item.weight or "") for item in selected]
            if any(weight < 0 for weight in weights) or sum(weights) == 0:
                return None
            value = sum((v * w for v, w in zip(values, weights, strict=True)), Fraction(0)) / sum(
                weights
            )
        return "value", value, units[0]
    if op == "convert" and source.task_id == "unit_conversion":
        if len(selected) != 1 or query.conversion_version != CONVERSION_VERSION:
            return None
        source_unit = units[0]
        target_unit = query.target_unit
        if source_unit not in CONVERSIONS or target_unit not in CONVERSIONS:
            return None
        source_dimension, source_factor = CONVERSIONS[source_unit]
        target_dimension, target_factor = CONVERSIONS[target_unit]
        if source_dimension != target_dimension:
            return None
        return "value", values[0] * source_factor / target_factor, target_unit
    return None


def verify_quantitative(
    task_id: str,
    sources: tuple[QuantitySource, QuantitySource],
    answers: tuple[QuantityAnswer, QuantityAnswer],
    public_parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> GateVerdict:
    """Calculate from agreed blind source data and compare a separately parsed answer."""
    if task_id not in QUANTITATIVE_TASKS:
        raise ValueError("No quantitative verifier for task")
    if any(
        source.coverage != "MET"
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN
    if _canonical_source(sources[0]) != _canonical_source(sources[1]):
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
    query = sources[0].query
    declared = public_parameters.get("operator", public_parameters.get("operators"))
    if declared is not None and declared != query.operation:
        return GateVerdict.UNKNOWN
    if query.decimal_places is not None and public_parameters.get("precision") not in {
        None,
        query.decimal_places,
    }:
        return GateVerdict.UNKNOWN
    try:
        expected = _expected(sources[0])
    except (ExecutionError, ValueError, ZeroDivisionError):
        return GateVerdict.UNKNOWN
    if expected is None:
        return GateVerdict.UNKNOWN
    form, value, unit = expected
    answer = answers[0]
    if form == "truth":
        observed: object = answer.truth
    elif form == "relation":
        observed = answer.relation
    else:
        if answer.value is None or answer.value.unit != unit:
            return GateVerdict.NOT_MET if answer.value is not None else GateVerdict.UNKNOWN
        try:
            observed = _number(answer.value)
        except ExecutionError:
            return GateVerdict.UNKNOWN
        if query.rounding == "half_up" and query.decimal_places is not None:
            assert isinstance(value, Fraction)
            value = _round_half_up(value, query.decimal_places)
        elif query.rounding != "exact":
            return GateVerdict.UNKNOWN
    if observed is None:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if observed == value else GateVerdict.NOT_MET
