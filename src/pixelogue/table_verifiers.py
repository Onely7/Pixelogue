"""Closed table-grid reconstruction and controller-owned lookup, filter and join."""

from __future__ import annotations

from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.rules import parse_numeric_lexeme
from pixelogue.serialization import strict_json_object
from pixelogue.task_evidence import ImageRegion

TABLE_TASKS = frozenset(
    {
        "table_cell_lookup",
        "table_predicate_selection",
        "table_structure_reconstruction",
        "table_cross_reference",
    }
)


class TableCell(StrictModel):
    """One cell, including its explicit blank and merged-cell extent."""

    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]
    rowspan: Annotated[int, Field(ge=1)] = 1
    colspan: Annotated[int, Field(ge=1)] = 1
    text: str
    kind: Literal["header", "data"]
    region: ImageRegion | None = None


class TableGrid(StrictModel):
    """A fully covered rectangular table with unique cell ownership."""

    table_id: str = Field(min_length=1)
    rows: Annotated[int, Field(ge=1, le=64)]
    cols: Annotated[int, Field(ge=1, le=32)]
    cells: Annotated[tuple[TableCell, ...], Field(max_length=512)]
    data_rows: tuple[int, ...]
    closed: bool

    @model_validator(mode="after")
    def check_grid(self) -> TableGrid:
        """Reject holes, overlapping spans and a partial row domain."""
        if self.rows * self.cols > 512:
            raise ValueError("Table grid exceeds supported size")
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if cell.row + cell.rowspan > self.rows or cell.col + cell.colspan > self.cols:
                raise ValueError("Cell span exceeds table bounds")
            for row in range(cell.row, cell.row + cell.rowspan):
                for col in range(cell.col, cell.col + cell.colspan):
                    if (row, col) in occupied:
                        raise ValueError("Overlapping table cells")
                    occupied.add((row, col))
        if self.closed and len(occupied) != self.rows * self.cols:
            raise ValueError("Closed table has missing cells; blanks must be explicit")
        if len(set(self.data_rows)) != len(self.data_rows) or any(
            row < 0 or row >= self.rows for row in self.data_rows
        ):
            raise ValueError("Invalid data row indices")
        return self

    def at(self, row: int, col: int) -> TableCell | None:
        """Return the cell whose span owns one grid location."""
        return next(
            (
                cell
                for cell in self.cells
                if cell.row <= row < cell.row + cell.rowspan
                and cell.col <= col < cell.col + cell.colspan
            ),
            None,
        )


class TableQuery(StrictModel):
    """One public table operation interpreted without candidate answer text."""

    operation: Literal["lookup", "select", "reconstruct", "join"]
    table_ids: tuple[str, ...]
    row: int | None = None
    col: int | None = None
    predicate_col: int | None = None
    predicate: Literal["eq", "lt", "gt", "le", "ge"] | None = None
    predicate_value: str | None = None
    output_col: int | None = None
    sort_col: int | None = None
    descending: bool = False
    left_key_col: int | None = None
    right_key_col: int | None = None
    right_output_col: int | None = None
    format: Literal["structured_json", "html_table", "markdown_simple_only"] | None = None


class TableSource(StrictModel):
    """Blind extraction of all table cells needed by the public operation."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    tables: Annotated[tuple[TableGrid, ...], Field(max_length=4)]
    query: TableQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> TableSource:
        """Require local, uniquely identified complete tables for MET coverage."""
        ids = [table.table_id for table in self.tables]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate table IDs")
        if self.coverage != "MET" and self.tables:
            raise ValueError("Incomplete extraction cannot provide partial tables")
        for table in self.tables:
            if not table.closed:
                continue
            for cell in table.cells:
                if cell.region is None:
                    raise ValueError("Certified cell has no image region")
                region = cell.region
                scope = self.scope_region
                if not (
                    scope.left <= region.left < region.right <= scope.right
                    and scope.top <= region.top < region.bottom <= scope.bottom
                ):
                    raise ValueError("Cell is outside its image scope")
        return self


class TableAnswer(StrictModel):
    """Answer-only extraction of lookup text, row labels or matched pairs."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    value: str | None = None
    rows: tuple[str, ...] | None = None
    pairs: tuple[tuple[str, str], ...] | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> TableAnswer:
        """Keep uncertain or compound parses from becoming successful checks."""
        populated = sum(item is not None for item in (self.value, self.rows, self.pairs))
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete table answer needs one quoted result")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete table answer cannot present values")
        return self


def _canonical_source(source: TableSource) -> dict[str, object]:
    result = source.model_dump(mode="json", exclude={"reason"})
    for table in result["tables"]:
        for cell in table["cells"]:
            cell.pop("region", None)
        table["cells"].sort(key=lambda item: (item["row"], item["col"]))
    result["tables"].sort(key=lambda item: item["table_id"])
    return result


def _select_rows(table: TableGrid, query: TableQuery) -> tuple[str, ...] | None:
    if (
        query.predicate_col is None
        or query.predicate is None
        or query.predicate_value is None
        or query.output_col is None
    ):
        return None
    if not 0 <= query.predicate_col < table.cols or not 0 <= query.output_col < table.cols:
        return None
    selected: list[tuple[int, str, str]] = []
    for row in table.data_rows:
        operand = table.at(row, query.predicate_col)
        output = table.at(row, query.output_col)
        if operand is None or output is None or not operand.text or not output.text:
            return None
        if query.predicate == "eq":
            matched = operand.text == query.predicate_value
        else:
            try:
                lhs = parse_numeric_lexeme(operand.text)
                rhs = parse_numeric_lexeme(query.predicate_value)
            except ExecutionError:
                return None
            matched = {
                "lt": lhs < rhs,
                "gt": lhs > rhs,
                "le": lhs <= rhs,
                "ge": lhs >= rhs,
            }[query.predicate]
        if matched:
            sort_cell = table.at(row, query.sort_col) if query.sort_col is not None else None
            selected.append((row, output.text, sort_cell.text if sort_cell else ""))
    if query.sort_col is not None:
        if not 0 <= query.sort_col < table.cols:
            return None
        try:
            selected.sort(key=lambda item: parse_numeric_lexeme(item[2]), reverse=query.descending)
        except ExecutionError:
            selected.sort(key=lambda item: item[2], reverse=query.descending)
    return tuple(item[1] for item in selected)


def _join_rows(
    left: TableGrid, right: TableGrid, query: TableQuery
) -> tuple[tuple[str, str], ...] | None:
    if any(
        item is None
        for item in (
            query.left_key_col,
            query.right_key_col,
            query.output_col,
            query.right_output_col,
        )
    ):
        return None
    assert query.left_key_col is not None
    assert query.right_key_col is not None
    assert query.output_col is not None
    assert query.right_output_col is not None
    if (
        query.left_key_col >= left.cols
        or query.output_col >= left.cols
        or query.right_key_col >= right.cols
        or query.right_output_col >= right.cols
        or min(query.left_key_col, query.output_col, query.right_key_col, query.right_output_col)
        < 0
    ):
        return None
    right_by_key: dict[str, str] = {}
    for row in right.data_rows:
        key = right.at(row, query.right_key_col)
        value = right.at(row, query.right_output_col)
        if key is None or value is None or not key.text or key.text in right_by_key:
            return None
        right_by_key[key.text] = value.text
    result: list[tuple[str, str]] = []
    left_keys: set[str] = set()
    for row in left.data_rows:
        key = left.at(row, query.left_key_col)
        value = left.at(row, query.output_col)
        if key is None or value is None or not key.text or key.text in left_keys:
            return None
        left_keys.add(key.text)
        if key.text in right_by_key:
            result.append((value.text, right_by_key[key.text]))
    if not result:
        return None
    return tuple(result)


def _cells_signature(cells: tuple[TableCell, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (cell.row, cell.col, cell.rowspan, cell.colspan, cell.text, cell.kind) for cell in cells
        )
    )


def _parse_json_table(answer: str) -> TableGrid | None:
    try:
        decoded = strict_json_object(answer)
        if set(decoded) != {
            "table_id",
            "rows",
            "cols",
            "cells",
            "data_rows",
            "closed",
        }:
            return None
        for cell in decoded["cells"]:
            if not isinstance(cell, dict) or set(cell) != {
                "row",
                "col",
                "rowspan",
                "colspan",
                "text",
                "kind",
            }:
                return None
        return TableGrid.model_validate_json(answer)
    except (ExternalInputError, ValueError, TypeError, KeyError):
        return None


class _RestrictedTableParser(HTMLParser):
    """Read only static table tags; reject scripts, attributes and nested markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.invalid = False
        self.table_open = False
        self.complete = False
        self.row = -1
        self.col = 0
        self.row_open = False
        self.cell_kind: Literal["header", "data"] | None = None
        self.cell_span = (1, 1)
        self.cell_text = ""
        self.cells: list[TableCell] = []
        self.occupied: set[tuple[int, int]] = set()
        self.data_rows: set[int] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and not attrs and not self.table_open and not self.complete:
            self.table_open = True
        elif (
            tag in {"thead", "tbody", "tfoot"}
            and not attrs
            and self.table_open
            and not self.row_open
        ):
            return
        elif tag == "tr" and not attrs and self.table_open and not self.row_open:
            self.row += 1
            self.col = 0
            self.row_open = True
        elif tag in {"th", "td"} and self.row_open and self.cell_kind is None:
            attribute_map = dict(attrs)
            if len(attribute_map) != len(attrs) or set(attribute_map) - {"rowspan", "colspan"}:
                self.invalid = True
                return
            try:
                rowspan = int(attribute_map.get("rowspan") or "1")
                colspan = int(attribute_map.get("colspan") or "1")
            except ValueError:
                self.invalid = True
                return
            if not 1 <= rowspan <= 64 or not 1 <= colspan <= 32:
                self.invalid = True
                return
            self.cell_kind = "header" if tag == "th" else "data"
            self.cell_span = (rowspan, colspan)
            self.cell_text = ""
        else:
            self.invalid = True

    def handle_data(self, data: str) -> None:
        if self.cell_kind is not None:
            self.cell_text += data
        elif data.strip():
            self.invalid = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.cell_kind == ("header" if tag == "th" else "data"):
            while (self.row, self.col) in self.occupied:
                self.col += 1
            rowspan, colspan = self.cell_span
            for row in range(self.row, self.row + rowspan):
                for col in range(self.col, self.col + colspan):
                    if (row, col) in self.occupied:
                        self.invalid = True
                    self.occupied.add((row, col))
            self.cells.append(
                TableCell(
                    row=self.row,
                    col=self.col,
                    rowspan=rowspan,
                    colspan=colspan,
                    text=self.cell_text.strip(),
                    kind=self.cell_kind,
                )
            )
            if self.cell_kind == "data":
                self.data_rows.add(self.row)
            self.col += colspan
            self.cell_kind = None
        elif tag == "tr" and self.row_open and self.cell_kind is None:
            self.row_open = False
        elif tag in {"thead", "tbody", "tfoot"} and self.table_open and not self.row_open:
            return
        elif tag == "table" and self.table_open and not self.row_open:
            self.table_open = False
            self.complete = True
        else:
            self.invalid = True


def _parse_html_table(answer: str, table_id: str) -> TableGrid | None:
    parser = _RestrictedTableParser()
    parser.feed(answer)
    parser.close()
    if parser.invalid or not parser.complete or not parser.cells or parser.cell_kind is not None:
        return None
    try:
        return TableGrid(
            table_id=table_id,
            rows=max(row for row, _ in parser.occupied) + 1,
            cols=max(col for _, col in parser.occupied) + 1,
            cells=tuple(parser.cells),
            data_rows=tuple(sorted(parser.data_rows)),
            closed=True,
        )
    except ValueError:
        return None


def _parse_markdown_table(answer: str, table_id: str) -> TableGrid | None:
    lines = [line.strip() for line in answer.strip().splitlines() if line.strip()]
    if len(lines) < 2 or any(not line.startswith("|") or not line.endswith("|") for line in lines):
        return None
    rows = [tuple(part.strip() for part in line[1:-1].split("|")) for line in lines]
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        return None
    if any(not item or set(item.replace(":", "")) != {"-"} for item in rows[1]):
        return None
    content = (rows[0], *rows[2:])
    try:
        return TableGrid(
            table_id=table_id,
            rows=len(content),
            cols=width,
            cells=tuple(
                TableCell(row=row, col=col, text=value, kind="header" if row == 0 else "data")
                for row, values in enumerate(content)
                for col, value in enumerate(values)
            ),
            data_rows=tuple(range(1, len(content))),
            closed=True,
        )
    except ValueError:
        return None


def verify_table(
    task_id: str,
    sources: tuple[TableSource, TableSource],
    answers: tuple[TableAnswer, TableAnswer] | None,
    public_parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, GateVerdict | None]:
    """Return content verdict and, for reconstruction, a separate schema verdict."""
    if task_id not in TABLE_TASKS:
        raise ValueError("No table verifier for this task")
    unknown = (
        GateVerdict.UNKNOWN,
        GateVerdict.UNKNOWN if task_id == "table_structure_reconstruction" else None,
    )
    if any(
        source.coverage != "MET"
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
        or any(not table.closed for table in source.tables)
        for source in sources
    ) or _canonical_source(sources[0]) != _canonical_source(sources[1]):
        return unknown
    query = sources[0].query
    by_id = {table.table_id: table for table in sources[0].tables}
    if len(query.table_ids) != (2 if query.operation == "join" else 1) or any(
        table_id not in by_id for table_id in query.table_ids
    ):
        return unknown
    table = by_id[query.table_ids[0]]
    if task_id == "table_structure_reconstruction":
        if query.operation != "reconstruct" or query.format is None:
            return unknown
        if public_parameters.get("format") != query.format:
            return unknown
        parsed = {
            "structured_json": lambda: _parse_json_table(candidate_answer),
            "html_table": lambda: _parse_html_table(candidate_answer, table.table_id),
            "markdown_simple_only": lambda: _parse_markdown_table(candidate_answer, table.table_id),
        }[query.format]()
        if parsed is None:
            return GateVerdict.UNKNOWN, GateVerdict.NOT_MET
        schema = GateVerdict.MET
        content = (
            GateVerdict.MET
            if parsed.rows == table.rows
            and parsed.cols == table.cols
            and parsed.data_rows == table.data_rows
            and parsed.closed == table.closed
            and _cells_signature(parsed.cells) == _cells_signature(table.cells)
            else GateVerdict.NOT_MET
        )
        return content, schema
    if answers is None or any(
        answer.coverage != "MET" or answer.answer_quote not in candidate_answer
        for answer in answers
    ):
        return unknown
    if answers[0].model_dump(mode="json", exclude={"reason"}) != answers[1].model_dump(
        mode="json", exclude={"reason"}
    ):
        return unknown
    answer = answers[0]
    if answer.value is not None and answer.value not in answer.answer_quote:
        return unknown
    if answer.rows is not None and any(row not in answer.answer_quote for row in answer.rows):
        return unknown
    if answer.pairs is not None and any(
        value not in answer.answer_quote for pair in answer.pairs for value in pair
    ):
        return unknown
    if task_id == "table_cell_lookup" and query.operation == "lookup":
        if query.row is None or query.col is None or query.row not in table.data_rows:
            return unknown
        cell = table.at(query.row, query.col)
        expected: object = cell.text if cell else None
        observed: object = answer.value
    elif task_id == "table_predicate_selection" and query.operation == "select":
        expected = _select_rows(table, query)
        observed = answer.rows
    elif task_id == "table_cross_reference" and query.operation == "join":
        expected = _join_rows(table, by_id[query.table_ids[1]], query)
        observed = answer.pairs
    else:
        return unknown
    if expected is None or observed is None:
        return unknown
    return (GateVerdict.MET if observed == expected else GateVerdict.NOT_MET), None
