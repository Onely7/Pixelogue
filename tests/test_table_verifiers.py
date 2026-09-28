"""Contract cases for closed tables, merged cells and output schemas."""

from __future__ import annotations

import json

import pytest

from pixelogue.contracts import GateVerdict
from pixelogue.table_verifiers import (
    TableAnswer,
    TableCell,
    TableGrid,
    TableQuery,
    TableSource,
    verify_table,
)
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _grid(table_id: str = "t") -> TableGrid:
    return TableGrid(
        table_id=table_id,
        rows=3,
        cols=3,
        cells=(
            TableCell(row=0, col=0, rowspan=2, text="Item", kind="header", region=REGION),
            TableCell(row=0, col=1, colspan=2, text="2024", kind="header", region=REGION),
            TableCell(row=1, col=1, text="Sales", kind="header", region=REGION),
            TableCell(row=1, col=2, text="Count", kind="header", region=REGION),
            TableCell(row=2, col=0, text="A", kind="data", region=REGION),
            TableCell(row=2, col=1, text="20", kind="data", region=REGION),
            TableCell(row=2, col=2, text="", kind="data", region=REGION),
        ),
        data_rows=(2,),
        closed=True,
    )


def _source(
    task_id: str, query: TableQuery, tables: tuple[TableGrid, ...] | None = None
) -> TableSource:
    return TableSource(
        task_id=task_id,
        coverage="MET",
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        tables=tables or (_grid(),),
        query=query,
        reason="Complete visible table",
    )


def _check(
    source: TableSource, answer: TableAnswer | None, text: str, **parameters: object
) -> tuple[GateVerdict, GateVerdict | None]:
    return verify_table(
        source.task_id,
        (source, source),
        (answer, answer) if answer else None,
        parameters,
        source.scope_id,
        source.view_id,
        text,
    )


def test_grid_requires_explicit_blanks_and_rejects_overlapping_merges() -> None:
    grid = _grid()
    assert grid.at(0, 0) == grid.at(1, 0)
    blank = grid.at(2, 2)
    assert blank is not None and blank.text == ""
    with pytest.raises(ValueError):
        TableGrid.model_validate_json(
            grid.model_copy(update={"cells": grid.cells[:-1]}).model_dump_json()
        )
    with pytest.raises(ValueError):
        TableGrid.model_validate_json(
            grid.model_copy(update={"cells": (*grid.cells, grid.cells[-1])}).model_dump_json()
        )
    with pytest.raises(ValueError):
        TableGrid.model_validate_json(grid.model_copy(update={"data_rows": ()}).model_dump_json())
    with pytest.raises(ValueError):
        TableGrid.model_validate_json(
            grid.model_copy(update={"data_rows": (1, 2)}).model_dump_json()
        )


def test_lookup_checks_cell_position_and_blank_value() -> None:
    source = _source(
        "table_cell_lookup", TableQuery(operation="lookup", table_ids=("t",), row=2, col=1)
    )
    answer = TableAnswer(coverage="MET", answer_quote="20", value="20", reason="literal")
    assert _check(source, answer, "20")[0] is GateVerdict.MET
    wrong = answer.model_copy(update={"answer_quote": "21", "value": "21"})
    assert _check(source, wrong, "21")[0] is GateVerdict.NOT_MET


def test_selection_checks_closed_rows_and_numeric_predicate() -> None:
    source = _source(
        "table_predicate_selection",
        TableQuery(
            operation="select",
            table_ids=("t",),
            predicate_col=1,
            predicate="gt",
            predicate_value="10",
            output_col=0,
        ),
    )
    answer = TableAnswer(coverage="MET", answer_quote="A", rows=("A",), reason="literal")
    assert _check(source, answer, "A")[0] is GateVerdict.MET
    assert (
        _check(
            source.model_copy(update={"tables": (_grid().model_copy(update={"closed": False}),)}),
            answer,
            "A",
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_join_requires_unique_keys_and_complete_visible_tables() -> None:
    left = TableGrid(
        table_id="left",
        rows=2,
        cols=2,
        data_rows=(1,),
        closed=True,
        cells=(
            TableCell(row=0, col=0, text="ID", kind="header", region=REGION),
            TableCell(row=0, col=1, text="Item", kind="header", region=REGION),
            TableCell(row=1, col=0, text="p1", kind="data", region=REGION),
            TableCell(row=1, col=1, text="Apple", kind="data", region=REGION),
        ),
    )
    right = left.model_copy(
        update={
            "table_id": "right",
            "cells": (
                TableCell(row=0, col=0, text="ID", kind="header", region=REGION),
                TableCell(row=0, col=1, text="Category", kind="header", region=REGION),
                TableCell(row=1, col=0, text="p1", kind="data", region=REGION),
                TableCell(row=1, col=1, text="Fruit", kind="data", region=REGION),
            ),
        }
    )
    source = _source(
        "table_cross_reference",
        TableQuery(
            operation="join",
            table_ids=("left", "right"),
            left_key_col=0,
            right_key_col=0,
            output_col=1,
            right_output_col=1,
        ),
        (left, right),
    )
    answer = TableAnswer(
        coverage="MET", answer_quote="Apple: Fruit", pairs=(("Apple", "Fruit"),), reason="literal"
    )
    assert _check(source, answer, "Apple: Fruit")[0] is GateVerdict.MET


def test_reconstruction_separates_schema_from_content_and_preserves_spans() -> None:
    source = _source(
        "table_structure_reconstruction",
        TableQuery(operation="reconstruct", table_ids=("t",), format="structured_json"),
    )
    grid = _grid().model_dump(mode="json")
    for cell in grid["cells"]:
        cell.pop("region")
    correct = json.dumps(grid)
    assert _check(source, None, correct, format="structured_json") == (
        GateVerdict.MET,
        GateVerdict.MET,
    )
    changed = json.loads(correct)
    changed["cells"][-1]["text"] = "0"
    assert _check(source, None, json.dumps(changed), format="structured_json") == (
        GateVerdict.NOT_MET,
        GateVerdict.MET,
    )
    assert _check(source, None, "{bad", format="structured_json") == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )
    duplicated = correct.replace('"text": "20"', '"text": "21", "text": "20"')
    assert _check(source, None, duplicated, format="structured_json") == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )


def test_html_and_simple_markdown_are_parsed_without_executing_markup() -> None:
    source = _source(
        "table_structure_reconstruction",
        TableQuery(operation="reconstruct", table_ids=("t",), format="html_table"),
    )
    html = "<table><tr><th rowspan='2'>Item</th><th colspan='2'>2024</th></tr><tr><th>Sales</th><th>Count</th></tr><tr><td>A</td><td>20</td><td></td></tr></table>"
    assert _check(source, None, html, format="html_table") == (GateVerdict.MET, GateVerdict.MET)
    assert _check(
        source, None, html.replace("<td>A</td>", "<script>A</script>"), format="html_table"
    ) == (GateVerdict.UNKNOWN, GateVerdict.NOT_MET)
    simple = TableGrid(
        table_id="t",
        rows=2,
        cols=2,
        data_rows=(1,),
        closed=True,
        cells=(
            TableCell(row=0, col=0, text="A", kind="header", region=REGION),
            TableCell(row=0, col=1, text="B", kind="header", region=REGION),
            TableCell(row=1, col=0, text="x", kind="data", region=REGION),
            TableCell(row=1, col=1, text="", kind="data", region=REGION),
        ),
    )
    markdown_source = _source(
        "table_structure_reconstruction",
        TableQuery(operation="reconstruct", table_ids=("t",), format="markdown_simple_only"),
        (simple,),
    )
    assert _check(
        markdown_source, None, "| A | B |\n| --- | --- |\n| x |  |", format="markdown_simple_only"
    ) == (GateVerdict.MET, GateVerdict.MET)
