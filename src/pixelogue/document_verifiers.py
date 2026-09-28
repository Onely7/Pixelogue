"""Field-value and document-tree checks against blind visual extraction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExternalInputError
from pixelogue.serialization import strict_json_object
from pixelogue.task_evidence import ImageRegion

DOCUMENT_TASKS = frozenset({"text_field_extraction", "document_structure_reconstruction"})
DocumentKind = Literal[
    "heading",
    "paragraph",
    "list",
    "list_item",
    "table",
    "table_row",
    "table_cell",
    "formula",
    "caption",
    "footnote",
]


class FieldBinding(StrictModel):
    """One requested field with a visually bound value or explicit absence."""

    field: str = Field(min_length=1)
    label: str | None = None
    value: str | None = None
    region: ImageRegion | None = None

    @model_validator(mode="after")
    def check_presence(self) -> FieldBinding:
        """Require evidence for present values; absent fields cannot carry text."""
        if (self.value is None) != (self.region is None):
            raise ValueError("Present field values require an image region")
        return self


class DocumentNode(StrictModel):
    """One visible document element with an explicit parent and reading position."""

    node_id: str = Field(min_length=1)
    kind: DocumentKind
    text: str
    parent_id: str | None = None
    order: Annotated[int, Field(ge=0)]
    region: ImageRegion | None = None


class DocumentQuery(StrictModel):
    """Public output schema and requested field names."""

    operation: Literal["fields", "structure"]
    format: Literal["structured_json"]
    requested_fields: tuple[str, ...] = ()


class DocumentSource(StrictModel):
    """Complete answer-independent extraction for one document operation."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    fields: Annotated[tuple[FieldBinding, ...], Field(max_length=64)] = ()
    nodes: Annotated[tuple[DocumentNode, ...], Field(max_length=256)] = ()
    query: DocumentQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> DocumentSource:
        """Reject partial inventories, duplicate fields and invalid reading hierarchies."""
        if self.coverage != "MET" and (self.closed or self.fields or self.nodes):
            raise ValueError("Incomplete document extraction cannot certify source facts")
        if self.task_id == "text_field_extraction":
            if self.nodes or self.query.operation != "fields":
                raise ValueError("Field operation cannot carry document nodes")
            if len(set(self.query.requested_fields)) != len(self.query.requested_fields):
                raise ValueError("Repeated requested field")
            if self.coverage == "MET" and (
                len(self.fields) != len(self.query.requested_fields)
                or {item.field for item in self.fields} != set(self.query.requested_fields)
            ):
                raise ValueError("Every requested field needs one present or missing binding")
        elif self.task_id == "document_structure_reconstruction":
            if self.fields or self.query.operation != "structure" or self.query.requested_fields:
                raise ValueError("Structure operation cannot carry field bindings")
            ids = [node.node_id for node in self.nodes]
            orders = [node.order for node in self.nodes]
            if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
                raise ValueError("Document node IDs and reading positions must be unique")
            if self.coverage == "MET" and not self.nodes:
                raise ValueError("Visible document structure needs at least one node")
            by_id = {node.node_id: node for node in self.nodes}
            for node in self.nodes:
                if node.parent_id is not None and (
                    node.parent_id not in by_id or by_id[node.parent_id].order >= node.order
                ):
                    raise ValueError("Document parent must precede child")
        else:
            raise ValueError("Unsupported document task")
        regions = [item.region for item in self.fields if item.region is not None] + [
            item.region for item in self.nodes if item.region is not None
        ]
        for region in regions:
            if not (
                self.scope_region.left <= region.left < region.right <= self.scope_region.right
                and self.scope_region.top <= region.top < region.bottom <= self.scope_region.bottom
            ):
                raise ValueError("Document evidence lies outside its scope")
        if (
            self.coverage == "MET"
            and self.task_id == "document_structure_reconstruction"
            and any(node.region is None for node in self.nodes)
        ):
            raise ValueError("Certified document nodes need image regions")
        return self


class PublicDocumentNode(StrictModel):
    """Public document node without private image coordinates."""

    node_id: str
    kind: DocumentKind
    text: str
    parent_id: str | None
    order: int


def _canonical(source: DocumentSource) -> dict[str, object]:
    body = source.model_dump(mode="json", exclude={"reason"})
    for field in body["fields"]:
        field.pop("region", None)
    for node in body["nodes"]:
        node.pop("region", None)
    body["fields"].sort(key=lambda item: item["field"])
    body["nodes"].sort(key=lambda item: item["order"])
    return body


def _parse_fields(
    candidate_answer: str, requested_fields: tuple[str, ...]
) -> dict[str, str | None] | None:
    try:
        decoded = strict_json_object(candidate_answer)
    except ExternalInputError:
        return None
    if set(decoded) != {"fields"}:
        return None
    fields = decoded["fields"]
    if not isinstance(fields, dict) or set(fields) != set(requested_fields):
        return None
    if any(value is not None and not isinstance(value, str) for value in fields.values()):
        return None
    return fields


def _parse_nodes(candidate_answer: str) -> tuple[PublicDocumentNode, ...] | None:
    try:
        decoded = strict_json_object(candidate_answer)
        if set(decoded) != {"nodes"}:
            return None
        if not isinstance(decoded["nodes"], list) or any(
            not isinstance(item, dict)
            or set(item) != {"node_id", "kind", "text", "parent_id", "order"}
            for item in decoded["nodes"]
        ):
            return None
        return tuple(PublicDocumentNode.model_validate(item) for item in decoded["nodes"])
    except (ExternalInputError, ValueError, TypeError):
        return None


def verify_document(
    task_id: str,
    sources: tuple[DocumentSource, DocumentSource],
    public_parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, GateVerdict]:
    """Return separate content and public-schema verdicts."""
    if task_id not in DOCUMENT_TASKS:
        raise ValueError("No document verifier for task")
    if any(
        source.coverage != "MET"
        or not source.closed
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ) or _canonical(sources[0]) != _canonical(sources[1]):
        return GateVerdict.UNKNOWN, GateVerdict.UNKNOWN
    source = sources[0]
    if public_parameters.get("format") != source.query.format:
        return GateVerdict.UNKNOWN, GateVerdict.UNKNOWN
    if task_id == "text_field_extraction":
        if public_parameters.get("fields") != source.query.requested_fields:
            return GateVerdict.UNKNOWN, GateVerdict.UNKNOWN
        parsed = _parse_fields(candidate_answer, source.query.requested_fields)
        if parsed is None:
            return GateVerdict.UNKNOWN, GateVerdict.NOT_MET
        expected = {field.field: field.value for field in source.fields}
        return (GateVerdict.MET if parsed == expected else GateVerdict.NOT_MET), GateVerdict.MET
    parsed_nodes = _parse_nodes(candidate_answer)
    if parsed_nodes is None:
        return GateVerdict.UNKNOWN, GateVerdict.NOT_MET
    expected_nodes = tuple(
        (node.node_id, node.kind, node.text, node.parent_id, node.order)
        for node in sorted(source.nodes, key=lambda item: item.order)
    )
    actual_nodes = tuple(
        (node.node_id, node.kind, node.text, node.parent_id, node.order)
        for node in sorted(parsed_nodes, key=lambda item: item.order)
    )
    return (
        GateVerdict.MET if actual_nodes == expected_nodes else GateVerdict.NOT_MET
    ), GateVerdict.MET
