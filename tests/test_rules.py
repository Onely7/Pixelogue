from fractions import Fraction

import pytest

from pixelogue.errors import ExecutionError
from pixelogue.rules import (
    ComputationInventory,
    NumericValue,
    SetInventory,
    compare_complete_set,
    exact_calculate,
    parse_numeric_lexeme,
    verify_computation_inventories,
    verify_set_inventories,
)


def test_complete_set_reports_omissions_duplicates_and_extras() -> None:
    check = compare_complete_set(("a", "b", "c"), ("a", "a", "d"))
    assert check.verdict == "NOT_MET"
    assert check.missing == ("b", "c")
    assert check.unexpected == ("d",)
    assert check.duplicates == ("a",)


def test_complete_set_rejects_an_unexplained_empty_scope() -> None:
    with pytest.raises(ExecutionError, match="explicit public definition"):
        compare_complete_set((), ())
    assert compare_complete_set((), (), empty_scope_is_explicit=True).verdict == "MET"


def test_dual_set_inventory_must_agree_before_comparison() -> None:
    inventory = SetInventory(
        coverage="MET",
        expected_members=("red-left", "blue-right"),
        reported_members=("red-left",),
        reason="The closed scope is readable.",
    )
    check = verify_set_inventories((inventory, inventory))
    assert check is not None and check.missing == ("blue-right",)
    other = inventory.model_copy(update={"expected_members": ("red-left",)})
    assert verify_set_inventories((inventory, other)) is None


def test_numeric_parser_uses_explicit_separators_and_exact_values() -> None:
    assert parse_numeric_lexeme(
        "１ 234,50", decimal_separator=",", group_separator=" "
    ) == Fraction(2469, 2)
    with pytest.raises(ExecutionError, match="undeclared separator"):
        parse_numeric_lexeme("1,234.50")


@pytest.mark.parametrize(
    ("operation", "values", "units", "expected"),
    [
        ("add", (Fraction(300), Fraction(250)), ("JPY", "JPY"), (Fraction(550), "JPY")),
        ("subtract", (Fraction(2), Fraction(5)), (None, None), (Fraction(-3), None)),
        ("multiply", (Fraction(3), Fraction(2)), ("JPY", None), (Fraction(6), "JPY")),
        ("divide", (Fraction(4), Fraction(2)), ("m", "m"), (Fraction(2), None)),
    ],
)
def test_exact_calculation_preserves_units(operation, values, units, expected) -> None:
    assert exact_calculate(operation, values, units) == expected


def test_exact_calculation_rejects_unit_conflicts_and_single_operand_addition() -> None:
    with pytest.raises(ExecutionError, match="units must match"):
        exact_calculate("add", (Fraction(1), Fraction(2)), ("m", "s"))
    with pytest.raises(ExecutionError, match="at least two"):
        exact_calculate("add", (Fraction(1),), ("m",))


def test_dual_computation_inventory_checks_reported_value_exactly() -> None:
    inventory = ComputationInventory(
        coverage="MET",
        operation="add",
        operands=(NumericValue(value="1.2", unit="kg"), NumericValue(value="2.3", unit="kg")),
        reported_result=NumericValue(value="3.5", unit="kg"),
        reason="The expression is complete.",
    )
    assert verify_computation_inventories((inventory, inventory)).verdict == "MET"
    wrong = inventory.model_copy(update={"reported_result": NumericValue(value="3.6", unit="kg")})
    assert verify_computation_inventories((wrong, wrong)).verdict == "NOT_MET"
    assert verify_computation_inventories((inventory, wrong)).verdict == "UNKNOWN"
