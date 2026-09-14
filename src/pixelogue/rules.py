"""Deterministic set and arithmetic checks used by quality gates."""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from typing import Any, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.errors import ExecutionError


class SetCheck(StrictModel):
    """Exact comparison of an observed answer set with a closed expected set."""

    verdict: Literal["MET", "NOT_MET"]
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    duplicates: tuple[str, ...]


class SetInventory(StrictModel):
    """One judge's member bindings for an exhaustive question and answer."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    expected_members: tuple[str, ...] = ()
    reported_members: tuple[str, ...] = ()
    empty_scope_is_explicit: bool = False
    reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_coverage(self) -> SetInventory:
        """Forbid partial bindings when extraction is incomplete."""
        has_expression = bool(self.expected_members or self.reported_members)
        if self.coverage != "MET" and (has_expression or self.empty_scope_is_explicit):
            raise ValueError("Incomplete extraction must not supply partial set bindings")
        return self


class NumericValue(StrictModel):
    """One model-extracted number with an explicit punctuation and unit policy."""

    value: str = Field(min_length=1)
    decimal_separator: Literal[".", ","] | None = "."
    group_separator: Literal[".", ",", " "] | None = None
    unit: str | None = None


class ComputationInventory(StrictModel):
    """One judge's typed extraction of the requested operation and reported result."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    operation: Literal["add", "subtract", "multiply", "divide"] | None = None
    operands: tuple[NumericValue, ...] = ()
    reported_result: NumericValue | None = None
    reason: str = Field(min_length=1, max_length=240)

    @model_validator(mode="after")
    def validate_coverage(self) -> ComputationInventory:
        """Require a complete typed expression only for complete extraction."""
        complete = (
            self.operation is not None and bool(self.operands) and self.reported_result is not None
        )
        if (self.coverage == "MET") != complete:
            raise ValueError("MET requires an operation, operands, and reported result")
        if self.coverage != "MET" and (
            self.operation is not None or self.operands or self.reported_result is not None
        ):
            raise ValueError("Incomplete extraction must not supply a partial expression")
        return self


class ComputationCheck(StrictModel):
    """Controller verdict and canonical rule inputs for one arithmetic expression."""

    verdict: Literal["MET", "NOT_MET", "UNKNOWN"]
    typed_rule_inputs: dict[str, Any]


def compare_complete_set(
    expected: Sequence[str],
    observed: Sequence[str],
    *,
    empty_scope_is_explicit: bool = False,
) -> SetCheck:
    """Compare set members without hiding omissions or duplicate answers.

    Args:
        expected: Controller-fixed members of a verified closed scope.
        observed: Members bound to the candidate answer.
        empty_scope_is_explicit: Whether the public task explicitly defines an empty scope.

    Raises:
        ExecutionError: If the expected scope is ambiguous or internally duplicated.
    """
    if not expected and not empty_scope_is_explicit:
        raise ExecutionError(
            "EMPTY_SCOPE_NOT_EXPLICIT",
            "An empty exhaustive scope needs an explicit public definition",
        )
    if any(not member for member in (*expected, *observed)):
        raise ExecutionError("EMPTY_SET_MEMBER", "Set member identifiers cannot be empty")
    expected_counts = Counter(expected)
    if any(count != 1 for count in expected_counts.values()):
        raise ExecutionError(
            "DUPLICATE_EXPECTED_MEMBER",
            "A verified closed scope cannot contain duplicate member identifiers",
        )
    observed_counts = Counter(observed)
    missing = tuple(sorted(expected_counts.keys() - observed_counts.keys()))
    unexpected = tuple(sorted(observed_counts.keys() - expected_counts.keys()))
    duplicates = tuple(sorted(key for key, count in observed_counts.items() if count > 1))
    verdict = "MET" if not missing and not unexpected and not duplicates else "NOT_MET"
    return SetCheck(
        verdict=verdict,
        missing=missing,
        unexpected=unexpected,
        duplicates=duplicates,
    )


def verify_set_inventories(inventories: Sequence[SetInventory]) -> SetCheck | None:
    """Compare a closed set only after two blind member inventories agree."""
    if len(inventories) != 2 or any(item.coverage != "MET" for item in inventories):
        return None
    bindings = [
        inventory.model_dump(mode="json", exclude={"reason", "coverage"})
        for inventory in inventories
    ]
    if bindings[0] != bindings[1]:
        return None
    inventory = inventories[0]
    try:
        return compare_complete_set(
            inventory.expected_members,
            inventory.reported_members,
            empty_scope_is_explicit=inventory.empty_scope_is_explicit,
        )
    except ExecutionError:
        return None


def parse_numeric_lexeme(
    text: str,
    *,
    decimal_separator: Literal[".", ","] | None = ".",
    group_separator: Literal[".", ",", " "] | None = None,
) -> Fraction:
    """Parse a finite decimal with an explicit punctuation policy.

    Units and currency symbols must be supplied separately. Unicode decimal digits and
    the Unicode minus sign are accepted, while exponent notation and guessed separators
    are rejected.

    Raises:
        ExecutionError: If the numeric spelling or separator policy is ambiguous.
    """
    if decimal_separator == group_separator:
        raise ExecutionError("AMBIGUOUS_SEPARATORS", "Decimal and group separators must differ")
    normalized_characters: list[str] = []
    for character in text.strip():
        try:
            normalized_characters.append(str(unicodedata.decimal(character)))
        except ValueError:
            normalized_characters.append(character)
    normalized = "".join(normalized_characters).replace("\u2212", "-")
    sign = 1
    if normalized[:1] in {"+", "-"}:
        if normalized[0] == "-":
            sign = -1
        normalized = normalized[1:]
    parts = normalized.split(decimal_separator) if decimal_separator is not None else [normalized]
    if len(parts) > 2 or any(not part for part in parts):
        raise ExecutionError("INVALID_NUMERIC_LEXEME", "The decimal spelling is invalid")
    whole = parts[0]
    if group_separator is not None and group_separator in whole:
        groups = whole.split(group_separator)
        if (
            not 1 <= len(groups[0]) <= 3
            or any(len(group) != 3 for group in groups[1:])
            or not all(group.isascii() and group.isdigit() for group in groups)
        ):
            raise ExecutionError("INVALID_DIGIT_GROUPING", "Digit groups are malformed")
        whole = "".join(groups)
    fraction = parts[1] if len(parts) == 2 else ""
    if (
        not whole.isascii()
        or not whole.isdigit()
        or (fraction and (not fraction.isascii() or not fraction.isdigit()))
    ):
        raise ExecutionError(
            "NONNUMERIC_OR_UNDECLARED_PUNCTUATION",
            "The value contains an undeclared separator or nonnumeric text",
        )
    numerator = int(whole + fraction)
    return sign * Fraction(numerator, 10 ** len(fraction))


def exact_calculate(
    operation: Literal["add", "subtract", "multiply", "divide"],
    values: Sequence[Fraction],
    units: Sequence[str | None],
) -> tuple[Fraction, str | None]:
    """Evaluate an allowlisted operation without binary floating-point rounding.

    Raises:
        ExecutionError: If arity, units, division, or operand types violate the contract.
    """
    if len(values) != len(units) or not values:
        raise ExecutionError("OPERAND_UNIT_MISMATCH", "Every operand needs one unit entry")
    if any(type(value) is not Fraction for value in values):
        raise ExecutionError(
            "EXACT_FRACTION_OPERANDS_REQUIRED",
            "Arithmetic operands must use exact Fraction values",
        )
    if any(unit is not None and not unit for unit in units):
        raise ExecutionError("INVALID_UNIT", "Unit strings cannot be empty")
    if operation == "add":
        if len(values) < 2:
            raise ExecutionError("ADD_REQUIRES_TWO_OR_MORE", "Addition needs at least two operands")
        if len(set(units)) != 1:
            raise ExecutionError("INCOMPATIBLE_UNITS", "Addition units must match exactly")
        return sum(values, Fraction(0)), units[0]
    if operation == "subtract":
        if len(values) != 2:
            raise ExecutionError("SUBTRACT_REQUIRES_TWO", "Subtraction needs two ordered operands")
        if units[0] != units[1]:
            raise ExecutionError("INCOMPATIBLE_UNITS", "Subtraction units must match exactly")
        return values[0] - values[1], units[0]
    if operation == "multiply":
        if len(values) != 2 or sum(unit is not None for unit in units) > 1:
            raise ExecutionError(
                "UNSUPPORTED_COMPOUND_UNIT",
                "Multiplication needs two operands and at least one dimensionless operand",
            )
        return values[0] * values[1], next((unit for unit in units if unit is not None), None)
    if len(values) != 2 or values[1] == 0:
        raise ExecutionError(
            "INVALID_DIVISION", "Division needs two operands and a nonzero divisor"
        )
    if units[1] is None:
        return values[0] / values[1], units[0]
    if units[0] == units[1]:
        return values[0] / values[1], None
    raise ExecutionError(
        "UNSUPPORTED_COMPOUND_UNIT",
        "Division supports a dimensionless denominator or equal units only",
    )


def verify_computation_inventories(
    inventories: Sequence[ComputationInventory],
) -> ComputationCheck:
    """Check exact arithmetic only after two blind typed extractions agree."""
    empty = ComputationCheck(verdict="UNKNOWN", typed_rule_inputs={})
    if len(inventories) != 2 or any(item.coverage != "MET" for item in inventories):
        return empty
    expressions = [
        inventory.model_dump(mode="json", exclude={"reason", "coverage"})
        for inventory in inventories
    ]
    if expressions[0] != expressions[1]:
        return empty
    inventory = inventories[0]
    assert inventory.operation is not None
    assert inventory.reported_result is not None
    try:
        operand_values = tuple(
            parse_numeric_lexeme(
                operand.value,
                decimal_separator=operand.decimal_separator,
                group_separator=operand.group_separator,
            )
            for operand in inventory.operands
        )
        expected_value, expected_unit = exact_calculate(
            inventory.operation,
            operand_values,
            tuple(operand.unit for operand in inventory.operands),
        )
        reported_value = parse_numeric_lexeme(
            inventory.reported_result.value,
            decimal_separator=inventory.reported_result.decimal_separator,
            group_separator=inventory.reported_result.group_separator,
        )
    except ExecutionError:
        return ComputationCheck(verdict="NOT_MET", typed_rule_inputs=expressions[0])
    verdict = (
        "MET"
        if reported_value == expected_value and inventory.reported_result.unit == expected_unit
        else "NOT_MET"
    )
    return ComputationCheck(verdict=verdict, typed_rule_inputs=expressions[0])
