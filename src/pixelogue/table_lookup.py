"""Blind, closed addressing witnesses for single-cell lookups in simple tables."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.table_verifiers import TableAnswer
from pixelogue.task_evidence import ImageRegion


class LookupCell(StrictModel):
    """The requested data cell, indexed into complete row and column domains."""

    row: Annotated[int, Field(ge=0)]
    col: Annotated[int, Field(ge=0)]
    text: str
    region: ImageRegion


class HeaderAnchor(StrictModel):
    """One selected literal header and its visible text location."""

    text: str
    region: ImageRegion


def _inside(inner: ImageRegion, outer: ImageRegion) -> bool:
    return (
        outer.left <= inner.left < inner.right <= outer.right
        and outer.top <= inner.top < inner.bottom <= outer.bottom
    )


class TableLookupSource(StrictModel):
    """Complete headers and selected anchors, without unrelated data values.

    Blank header labels remain explicit. Grouping headers with spans, relevant
    footnotes or multi-cell requests use the full-grid verifier instead.
    """

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    layout: Literal["simple_grid", "requires_full_grid", "unreadable"]
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    table_region: ImageRegion | None = None
    data_region: ImageRegion | None = None
    row_headers: Annotated[tuple[str, ...], Field(max_length=64)]
    col_headers: Annotated[tuple[str, ...], Field(max_length=32)]
    row_anchor: HeaderAnchor | None = None
    col_anchor: HeaderAnchor | None = None
    closed: bool
    cell: LookupCell | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_witness(self) -> TableLookupSource:
        """Reject missing, overlapping, off-scope or misaligned certificates."""
        if self.coverage != "MET":
            if (
                self.row_headers
                or self.col_headers
                or self.row_anchor
                or self.col_anchor
                or self.cell
                or self.closed
            ):
                raise ValueError("Incomplete lookup cannot contain a partial certificate")
            return self
        if (
            self.layout != "simple_grid"
            or not self.closed
            or self.table_region is None
            or self.data_region is None
        ):
            raise ValueError("Complete lookup requires a closed simple table and data region")
        if not _inside(self.table_region, self.scope_region) or not _inside(
            self.data_region, self.table_region
        ):
            raise ValueError("Lookup table/data region is outside its bound scope")
        if not self.row_headers or not self.col_headers or self.cell is None:
            raise ValueError("Lookup needs every header and exactly one cell")
        cell = self.cell
        if cell.row >= len(self.row_headers) or cell.col >= len(self.col_headers):
            raise ValueError("Lookup cell index is outside the closed header domains")
        row, col = self.row_anchor, self.col_anchor
        if row is None or col is None:
            raise ValueError("Lookup needs the selected row and column header anchors")
        if row.text != self.row_headers[cell.row] or col.text != self.col_headers[cell.col]:
            raise ValueError("Selected anchors do not match their indexed literal headers")
        if (
            not _inside(row.region, self.table_region)
            or not _inside(col.region, self.table_region)
            or not _inside(cell.region, self.data_region)
        ):
            raise ValueError("Lookup cell or header is outside its table/data region")
        if not (
            row.region.top < cell.region.bottom
            and cell.region.top < row.region.bottom
            and col.region.left < cell.region.right
            and cell.region.left < col.region.right
        ):
            raise ValueError("Lookup cell does not align with its selected header anchors")
        return self


def _signature(source: TableLookupSource) -> tuple[object, ...]:
    """Compare literal addressing and values without approximate box equality."""
    assert source.cell is not None
    return (
        source.scope_id,
        source.view_id,
        source.row_headers,
        source.col_headers,
        source.cell.row,
        source.cell.col,
        source.cell.text,
    )


def verify_table_lookup(
    sources: tuple[TableLookupSource, TableLookupSource],
    answers: tuple[TableAnswer, TableAnswer],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
    scope_region: ImageRegion | None = None,
) -> GateVerdict:
    """Compare two blind closed witnesses and two literal answer parses."""
    for source in sources:
        if (
            source.coverage != "MET"
            or source.scope_id != scope_id
            or source.view_id != view_id
            or source.cell is None
            or (
                scope_region is not None
                and (source.table_region is None or not _inside(source.table_region, scope_region))
            )
        ):
            return GateVerdict.UNKNOWN
        for headers, index in (
            (source.row_headers, source.cell.row),
            (source.col_headers, source.cell.col),
        ):
            labels = [header.casefold().strip() for header in headers]
            if not labels[index] or labels.count(labels[index]) != 1:
                return GateVerdict.UNKNOWN
    if _signature(sources[0]) != _signature(sources[1]):
        return GateVerdict.UNKNOWN
    if any(
        answer.coverage != "MET"
        or answer.value is None
        or answer.rows is not None
        or answer.pairs is not None
        or answer.answer_quote not in candidate_answer
        or answer.value not in answer.answer_quote
        for answer in answers
    ):
        return GateVerdict.UNKNOWN
    if answers[0].value != answers[1].value or answers[0].answer_quote != answers[1].answer_quote:
        return GateVerdict.UNKNOWN
    assert sources[0].cell is not None
    return GateVerdict.MET if answers[0].value == sources[0].cell.text else GateVerdict.NOT_MET
