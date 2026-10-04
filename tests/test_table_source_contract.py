"""Complete table sources and empty abstentions keep the same blind gates."""

from __future__ import annotations

import json
from typing import Literal

import pytest

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.table_verifiers import TableCell, TableGrid, TableQuery, TableSource, verify_table
from pixelogue.task_evidence import ImageRegion, PublicParameter
from pixelogue.task_verification import verify_operation

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _grid(*, closed: bool = True, complete: bool = True) -> TableGrid:
    cells = (
        TableCell(row=0, col=0, text="Item", kind="header", region=REGION),
        TableCell(row=1, col=0, text="A", kind="data", region=REGION),
    )
    return TableGrid(
        table_id="table_0",
        rows=2,
        cols=1,
        cells=cells if complete else cells[:1],
        data_rows=(1,) if complete else (),
        closed=closed,
    )


def _source(
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"], tables: tuple[TableGrid, ...]
) -> TableSource:
    return TableSource(
        task_id="table_structure_reconstruction",
        coverage=coverage,
        scope_id="table",
        view_id="view",
        scope_region=REGION,
        tables=tables,
        query=TableQuery(
            operation="reconstruct",
            table_ids=("table_0",),
            format="structured_json",
        ),
        reason="Complete grid" if coverage == "MET" else "Complete grid cannot be supplied",
    )


@pytest.mark.parametrize("coverage", ["UNKNOWN", "NOT_MET"])
def test_incomplete_coverage_cannot_carry_even_a_readable_table(coverage: str) -> None:
    value = _source("MET", (_grid(),)).model_dump(mode="json")
    value["coverage"] = coverage

    with pytest.raises(ValueError, match="cannot provide partial tables"):
        TableSource.model_validate_json(json.dumps(value))


def test_a_closed_partial_grid_cannot_claim_complete_coverage() -> None:
    with pytest.raises(ValueError, match="missing cells"):
        _source("MET", (_grid(complete=False),))


def test_open_partial_met_extraction_cannot_certify_a_matching_reconstruction() -> None:
    partial = _source("MET", (_grid(closed=False, complete=False),))
    verdict = verify_table(
        partial.task_id,
        (partial, partial),
        None,
        {"format": "structured_json"},
        partial.scope_id,
        partial.view_id,
        _grid().model_dump_json(exclude={"cells": {"__all__": {"region"}}}),
    )

    assert verdict == (GateVerdict.UNKNOWN, GateVerdict.UNKNOWN)


@pytest.mark.parametrize("coverage", ["UNKNOWN", "NOT_MET"])
def test_empty_source_abstentions_still_use_two_blind_readers(coverage: str) -> None:
    task = next(
        task for task in task_catalog().tasks if task.id == "table_structure_reconstruction"
    )
    instruction = InstructionCandidate(
        candidate_id="reconstruction",
        task_id=task.id,
        family=task.family,
        visible_scope="the entire table",
        instruction_summary=task.definition_en,
        required_capabilities=task.required_capabilities,
        catalog_version="7.0",
        scope_id="table",
        view_id="view",
        evidence_refs=("table-evidence",),
        public_parameters=(
            PublicParameter(name="format", value="structured_json", origin="instruction"),
        ),
        verification_contracts=task.verification_contracts,
    )
    calls: list[int] = []

    def invoke(stage, payload, model, judge):
        assert stage == "table_source" and model is TableSource
        assert "candidate_answer" not in payload
        assert "other_judge" not in payload
        calls.append(judge)
        value = _source("UNKNOWN", ()).model_dump(mode="json")
        value["coverage"] = coverage
        return TableSource.model_validate_json(json.dumps(value))

    checks = verify_operation(
        instruction,
        {
            "question": "Reconstruct the complete table as structured JSON.",
            "candidate_answer": _grid().model_dump_json(exclude={"cells": {"__all__": {"region"}}}),
            "image_views": [],
        },
        invoke,
    )

    assert calls == [0, 1]
    assert {check.name for check in checks} == {"table_structure_check", "schema_check"}
    assert all(check.verdict is GateVerdict.UNKNOWN for check in checks)
