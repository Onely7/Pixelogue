"""Document field schema and visible reading-tree verification."""

from __future__ import annotations

import json

import pytest

from pixelogue.contracts import GateVerdict
from pixelogue.document_verifiers import (
    DocumentNode,
    DocumentQuery,
    DocumentSource,
    FieldBinding,
    verify_document,
)
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _fields() -> DocumentSource:
    return DocumentSource(
        task_id="text_field_extraction",
        coverage="MET",
        closed=True,
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        fields=(
            FieldBinding(field="invoice", label="Invoice #", value="INV-42", region=REGION),
            FieldBinding(field="date", label=None, value=None, region=None),
        ),
        query=DocumentQuery(
            operation="fields", format="structured_json", requested_fields=("invoice", "date")
        ),
        reason="Invoice label visible; date absent in closed scope",
    )


def _nodes() -> DocumentSource:
    return DocumentSource(
        task_id="document_structure_reconstruction",
        coverage="MET",
        closed=True,
        scope_id="scope",
        view_id="view",
        scope_region=REGION,
        nodes=(
            DocumentNode(
                node_id="h", kind="heading", text="Notice", parent_id=None, order=0, region=REGION
            ),
            DocumentNode(
                node_id="p",
                kind="paragraph",
                text="Open today.",
                parent_id="h",
                order=1,
                region=REGION,
            ),
        ),
        query=DocumentQuery(operation="structure", format="structured_json"),
        reason="Both visible blocks and reading order resolved",
    )


def _check(
    source: DocumentSource, text: str, **parameters: object
) -> tuple[GateVerdict, GateVerdict]:
    return verify_document(
        source.task_id, (source, source), parameters, source.scope_id, source.view_id, text
    )


def test_requested_field_values_missing_status_and_schema() -> None:
    source = _fields()
    correct = json.dumps({"fields": {"invoice": "INV-42", "date": None}})
    assert _check(source, correct, format="structured_json", fields=("invoice", "date")) == (
        GateVerdict.MET,
        GateVerdict.MET,
    )
    invented = json.dumps({"fields": {"invoice": "INV-42", "date": "2026-01-01"}})
    assert _check(source, invented, format="structured_json", fields=("invoice", "date")) == (
        GateVerdict.NOT_MET,
        GateVerdict.MET,
    )
    omitted = json.dumps({"fields": {"invoice": "INV-42"}})
    assert _check(source, omitted, format="structured_json", fields=("invoice", "date")) == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )
    duplicated = '{"fields":{"invoice":"wrong","invoice":"INV-42","date":null}}'
    assert _check(source, duplicated, format="structured_json", fields=("invoice", "date")) == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )


def test_fields_must_be_publicly_requested_and_two_sources_agree() -> None:
    source = _fields()
    answer = json.dumps({"fields": {"invoice": "INV-42", "date": None}})
    assert (
        _check(source, answer, format="structured_json", fields=("date", "invoice"))[0]
        is GateVerdict.UNKNOWN
    )
    different = source.model_copy(
        update={
            "fields": (source.fields[0].model_copy(update={"value": "INV-43"}), source.fields[1])
        }
    )
    assert (
        verify_document(
            source.task_id,
            (source, different),
            {"format": "structured_json", "fields": ("invoice", "date")},
            source.scope_id,
            source.view_id,
            answer,
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_document_tree_preserves_content_hierarchy_and_order() -> None:
    source = _nodes()
    correct = {
        "nodes": [
            {"node_id": "h", "kind": "heading", "text": "Notice", "parent_id": None, "order": 0},
            {
                "node_id": "p",
                "kind": "paragraph",
                "text": "Open today.",
                "parent_id": "h",
                "order": 1,
            },
        ]
    }
    assert _check(source, json.dumps(correct), format="structured_json") == (
        GateVerdict.MET,
        GateVerdict.MET,
    )
    duplicated = json.dumps(correct).replace(
        '"text": "Notice"', '"text": "Wrong", "text": "Notice"'
    )
    assert _check(source, duplicated, format="structured_json") == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )
    correct["nodes"][1]["text"] = "Open tomorrow."
    assert _check(source, json.dumps(correct), format="structured_json") == (
        GateVerdict.NOT_MET,
        GateVerdict.MET,
    )
    correct["nodes"][1]["region"] = "not allowed"
    assert _check(source, json.dumps(correct), format="structured_json") == (
        GateVerdict.UNKNOWN,
        GateVerdict.NOT_MET,
    )


def test_invalid_parent_and_off_scope_regions_cannot_certify_tree() -> None:
    source = _nodes()
    with pytest.raises(ValueError):
        DocumentSource.model_validate_json(
            source.model_copy(
                update={
                    "nodes": (
                        source.nodes[0],
                        source.nodes[1].model_copy(update={"parent_id": "missing"}),
                    )
                }
            ).model_dump_json()
        )
    outside = ImageRegion(left=0.9, top=0.9, right=1, bottom=1)
    scope = ImageRegion(left=0, top=0, right=0.8, bottom=0.8)
    with pytest.raises(ValueError):
        DocumentSource.model_validate_json(
            source.model_copy(
                update={
                    "scope_region": scope,
                    "nodes": (
                        source.nodes[0],
                        source.nodes[1].model_copy(update={"region": outside}),
                    ),
                }
            ).model_dump_json()
        )
