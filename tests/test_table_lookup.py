"""Single-cell witnesses must preserve closed addressing and blind extraction."""

from __future__ import annotations

import json

import pytest

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.table_lookup import TableLookupSource, verify_table_lookup
from pixelogue.table_verifiers import TableAnswer, TableSource
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_verification import verify_operation


def witness() -> dict:
    return {
        "coverage": "MET",
        "layout": "simple_grid",
        "scope_id": "table",
        "view_id": "view",
        "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        "table_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        "data_region": {"left": 0.2, "top": 0.2, "right": 1, "bottom": 1},
        "row_headers": ["A", "B"],
        "col_headers": ["Count", "Mass (kg)"],
        "row_anchor": {
            "text": "A",
            "region": {"left": 0, "top": 0.3, "right": 0.18, "bottom": 0.4},
        },
        "col_anchor": {
            "text": "Count",
            "region": {"left": 0.3, "top": 0, "right": 0.4, "bottom": 0.18},
        },
        "closed": True,
        "cell": {
            "row": 0,
            "col": 0,
            "text": "20",
            "region": {"left": 0.2, "top": 0.2, "right": 0.6, "bottom": 0.6},
        },
        "reason": "Every addressing header is visible and the one requested cell is readable",
    }


def answer(text: str = "20") -> TableAnswer:
    return TableAnswer(coverage="MET", answer_quote=text, value=text, reason="Literal answer")


def check(first: dict, second: dict | None = None, text: str = "20") -> GateVerdict:
    return verify_table_lookup(
        (
            TableLookupSource.model_validate_json(json.dumps(first)),
            TableLookupSource.model_validate_json(json.dumps(second or first)),
        ),
        (answer(text), answer(text)),
        "table",
        "view",
        text,
    )


def test_one_cell_lookup_needs_no_unrelated_data_values_but_checks_wrong_answers() -> None:
    assert check(witness()) is GateVerdict.MET
    assert check(witness(), text="21") is GateVerdict.NOT_MET
    blank = witness()
    blank["cell"]["text"] = ""
    assert check(blank) is GateVerdict.NOT_MET


@pytest.mark.parametrize("corruption", ["missing", "overlap", "off_scope", "wrong_row", "index"])
def test_missing_or_inconsistent_addressing_is_not_a_certificate(corruption: str) -> None:
    value = witness()
    if corruption == "missing":
        value["row_headers"].clear()
    elif corruption == "overlap":
        value["row_anchor"]["region"].update(top=0.7, bottom=0.8)
    elif corruption == "off_scope":
        value["scope_region"]["right"] = 0.9
    elif corruption == "wrong_row":
        value["cell"]["row"] = 1
    else:
        value["cell"]["col"] = 2
    with pytest.raises(ValueError):
        TableLookupSource.model_validate_json(json.dumps(value))


def test_readers_must_agree_on_unselected_header_domains_too() -> None:
    shortened = witness()
    shortened["row_headers"].pop()
    assert check(witness(), shortened) is GateVerdict.UNKNOWN


def test_a_reader_cannot_expand_the_controller_bound_table_scope() -> None:
    reading = TableLookupSource.model_validate_json(json.dumps(witness()))
    assert (
        verify_table_lookup(
            (reading, reading),
            (answer(), answer()),
            "table",
            "view",
            "20",
            scope_region=ImageRegion(left=0, top=0, right=0.8, bottom=1),
        )
        is GateVerdict.UNKNOWN
    )


def test_ambiguous_headers_unit_disagreement_and_different_views_abstain() -> None:
    value = witness()
    value["row_headers"][1] = "A"
    assert check(value) is GateVerdict.UNKNOWN
    second = witness()
    second["col_headers"][1] = "Mass (g)"
    assert check(witness(), second) is GateVerdict.UNKNOWN
    second = witness()
    second["view_id"] = "other"
    assert check(witness(), second) is GateVerdict.UNKNOWN


def test_unreadable_axes_cannot_present_partial_supported_values() -> None:
    value = witness()
    value["coverage"] = "UNKNOWN"
    with pytest.raises(ValueError):
        TableLookupSource.model_validate_json(json.dumps(value))


@pytest.mark.parametrize("full_grid", [False, True])
def test_routing_keeps_sources_blind_and_hierarchical_lookup_on_full_grid(full_grid: bool) -> None:
    task = next(task for task in task_catalog().tasks if task.id == "table_cell_lookup")
    instruction = InstructionCandidate(
        candidate_id="lookup",
        task_id=task.id,
        family=task.family,
        visible_scope="the table",
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version="8.0",
        scope_id="table",
        view_id="view",
        evidence_refs=("table-evidence",),
        verification_contracts=task.verification_contracts,
    )
    calls = []

    def invoke(stage, payload, model, judge):
        calls.append((stage, judge))
        if stage.endswith("source"):
            assert "candidate_answer" not in payload
        else:
            assert "image_views" not in payload and "public_history" not in payload
        if model is TableLookupSource:
            value = witness()
            if full_grid:
                value.update(
                    coverage="UNKNOWN",
                    layout="requires_full_grid",
                    closed=False,
                    row_headers=[],
                    col_headers=[],
                    row_anchor=None,
                    col_anchor=None,
                    data_region=None,
                    cell=None,
                )
            return TableLookupSource.model_validate_json(json.dumps(value))
        if model is TableSource:
            return TableSource.model_validate(
                {
                    "task_id": "table_cell_lookup",
                    "coverage": "UNKNOWN",
                    "scope_id": "table",
                    "view_id": "view",
                    "scope_region": witness()["scope_region"],
                    "tables": (),
                    "query": {"operation": "lookup", "table_ids": ()},
                    "reason": "Unreadable merge",
                }
            )
        assert model is TableAnswer
        return answer()

    checks = verify_operation(
        instruction,
        {"question": "What is the count for A?", "candidate_answer": "20", "image_views": []},
        invoke,
    )
    assert calls[:2] == [("table_lookup_source", 0), ("table_lookup_source", 1)]
    assert ("table_source", 0) in calls if full_grid else ("table_source", 0) not in calls
    assert checks[0].verdict is (GateVerdict.UNKNOWN if full_grid else GateVerdict.MET)
