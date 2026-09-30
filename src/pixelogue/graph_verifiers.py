"""Image-bound directed graph operations and public flowchart simulation."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError
from pixelogue.rules import parse_numeric_lexeme
from pixelogue.task_evidence import ImageRegion

GRAPH_TASKS = frozenset(
    {
        "diagram_element_lookup",
        "graph_connectivity",
        "graph_path_tracing",
        "diagram_process_description",
        "diagram_branch_evaluation",
    }
)


class GraphNode(StrictModel):
    """One visibly labeled or otherwise uniquely identified diagram element."""

    node_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    region: ImageRegion


class BranchCondition(StrictModel):
    """A registered comparison of the public numeric input to a printed threshold."""

    operator: Literal["lt", "le", "eq", "ge", "gt"]
    threshold: str
    printed_text: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_threshold(self) -> BranchCondition:
        """Require the threshold's exact spelling in visible branch text."""
        if self.threshold not in self.printed_text:
            raise ValueError("Branch threshold lacks printed evidence")
        try:
            parse_numeric_lexeme(self.threshold)
        except ExecutionError as exc:
            raise ValueError("Invalid branch threshold") from exc
        return self


class GraphEdge(StrictModel):
    """One resolved edge including direction, visible junctions and branch rule."""

    source: str
    target: str
    directed: bool
    condition: BranchCondition | None = None
    region: ImageRegion


class GraphQuery(StrictModel):
    """Public graph objective and optional hypothetical input."""

    operation: Literal["element", "neighbors", "edges", "paths", "process", "branch"]
    start: str | None = None
    end: str | None = None
    node_id: str | None = None
    input_value: str | None = None
    neighbor_direction: Literal["incoming", "outgoing", "both"] = "outgoing"


class GraphSource(StrictModel):
    """Blind extraction of nodes, edges and explicit crossing conventions."""

    task_id: str
    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    junctions_resolved: bool
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    nodes: Annotated[tuple[GraphNode, ...], Field(max_length=64)]
    edges: Annotated[tuple[GraphEdge, ...], Field(max_length=128)]
    query: GraphQuery
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_graph(self) -> GraphSource:
        """Reject dangling links, duplicate nodes and nonlocal visual evidence."""
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Repeated graph node ID")
        known = set(ids)
        signatures: set[tuple[str, str, bool, str | None]] = set()
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError("Edge has an unresolved endpoint")
            endpoints = (
                (edge.source, edge.target)
                if edge.directed
                else tuple(sorted((edge.source, edge.target)))
            )
            left, right = endpoints
            signature = (
                left,
                right,
                edge.directed,
                edge.condition.model_dump_json() if edge.condition else None,
            )
            if signature in signatures:
                raise ValueError("Repeated graph edge")
            signatures.add(signature)
        if self.coverage != "MET" and (self.closed or self.nodes or self.edges):
            raise ValueError("Incomplete graph cannot certify nodes or edges")
        if self.coverage == "MET" and not self.nodes:
            raise ValueError("Certified diagram must contain a visible node")
        for region in [node.region for node in self.nodes] + [edge.region for edge in self.edges]:
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Graph evidence lies outside image scope")
        return self


class GraphAnswer(StrictModel):
    """Answer-only parse of a label, neighbor set, edge set or explicit paths."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    answer_quote: str = ""
    label: str | None = None
    members: tuple[str, ...] | None = None
    edges: tuple[tuple[str, str], ...] | None = None
    paths: tuple[tuple[str, ...], ...] | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_shape(self) -> GraphAnswer:
        """Forbid partial and compound answer parses."""
        populated = sum(
            item is not None for item in (self.label, self.members, self.edges, self.paths)
        )
        if self.coverage == "MET" and (populated != 1 or not self.answer_quote):
            raise ValueError("Complete graph answer needs one quoted form")
        if self.coverage != "MET" and (populated or self.answer_quote):
            raise ValueError("Incomplete graph answer cannot supply a result")
        return self


def _canonical(source: GraphSource) -> dict[str, object]:
    result = source.model_dump(mode="json", exclude={"reason"})
    for node in result["nodes"]:
        node.pop("region", None)
    for edge in result["edges"]:
        edge.pop("region", None)
    result["nodes"].sort(key=lambda item: item["node_id"])
    result["edges"].sort(key=lambda item: (item["source"], item["target"], str(item["condition"])))
    return result


def _label_bound_source(source: GraphSource) -> GraphSource | None:
    """Resolve local references to unique displayed labels without changing evidence."""
    labels = {node.node_id: node.label for node in source.nodes}
    if len(set(labels.values())) != len(labels):
        return None
    query = source.query.model_dump()
    for field in ("start", "end", "node_id"):
        reference = query[field]
        if reference is not None:
            if reference not in labels:
                return None
            query[field] = labels[reference]
    return source.model_copy(
        update={
            "nodes": tuple(
                node.model_copy(update={"node_id": node.label}) for node in source.nodes
            ),
            "edges": tuple(
                edge.model_copy(
                    update={"source": labels[edge.source], "target": labels[edge.target]}
                )
                for edge in source.edges
            ),
            "query": GraphQuery.model_validate(query),
        }
    )


def _region_overlap(left: ImageRegion, right: ImageRegion) -> float:
    """Return intersection over union for two positive image rectangles."""
    intersection = max(0, min(left.right, right.right) - max(left.left, right.left)) * max(
        0, min(left.bottom, right.bottom) - max(left.top, right.top)
    )
    left_area = (left.right - left.left) * (left.bottom - left.top)
    right_area = (right.right - right.left) * (right.bottom - right.top)
    return intersection / (left_area + right_area - intersection)


def _regions_agree(left: GraphSource, right: GraphSource) -> bool:
    """Bind agreed labels and edges to overlapping regions in the same view."""
    right_nodes = {node.node_id: node for node in right.nodes}
    if any(
        _region_overlap(node.region, right_nodes[node.node_id].region) < 0.1 for node in left.nodes
    ):
        return False
    left_edges = sorted(
        left.edges, key=lambda edge: (edge.source, edge.target, str(edge.condition))
    )
    right_edges = sorted(
        right.edges, key=lambda edge: (edge.source, edge.target, str(edge.condition))
    )
    return all(
        _region_overlap(a.region, b.region) >= 0.1
        for a, b in zip(left_edges, right_edges, strict=True)
    )


def _neighbors(source: GraphSource, node_id: str, direction: str) -> tuple[str, ...]:
    result: set[str] = set()
    for edge in source.edges:
        if edge.source == node_id and direction in {"outgoing", "both"}:
            result.add(edge.target)
        if edge.target == node_id and direction in {"incoming", "both"}:
            result.add(edge.source)
        if not edge.directed and edge.target == node_id and direction == "outgoing":
            result.add(edge.source)
        if not edge.directed and edge.source == node_id and direction == "incoming":
            result.add(edge.target)
    return tuple(sorted(result))


def _edge_pairs(source: GraphSource) -> tuple[tuple[str, str], ...]:
    pairs = {(edge.source, edge.target) for edge in source.edges}
    pairs |= {(edge.target, edge.source) for edge in source.edges if not edge.directed}
    return tuple(sorted(pairs))


def _all_paths(source: GraphSource, start: str, end: str) -> tuple[tuple[str, ...], ...] | None:
    if (
        start == end
        or start not in {node.node_id for node in source.nodes}
        or end not in {node.node_id for node in source.nodes}
    ):
        return None
    adjacency: dict[str, list[str]] = {node.node_id: [] for node in source.nodes}
    for edge in source.edges:
        if edge.condition is not None:
            return None
        adjacency[edge.source].append(edge.target)
        if not edge.directed:
            adjacency[edge.target].append(edge.source)
    routes: list[tuple[str, ...]] = []

    def visit(path: tuple[str, ...]) -> None:
        if len(routes) > 128:
            return
        current = path[-1]
        if current == end:
            routes.append(path)
            return
        for neighbor in adjacency[current]:
            if neighbor not in path:
                visit((*path, neighbor))

    visit((start,))
    if not routes or len(routes) > 128:
        return None
    return tuple(sorted(routes))


def _matches(condition: BranchCondition, value: Fraction) -> bool:
    threshold = parse_numeric_lexeme(condition.threshold)
    return {
        "lt": value < threshold,
        "le": value <= threshold,
        "eq": value == threshold,
        "ge": value >= threshold,
        "gt": value > threshold,
    }[condition.operator]


def _branch_path(source: GraphSource, start: str, input_value: str) -> tuple[str, ...] | None:
    try:
        value = parse_numeric_lexeme(input_value)
    except ExecutionError:
        return None
    known = {node.node_id for node in source.nodes}
    if start not in known:
        return None
    path = [start]
    while True:
        outgoing = [edge for edge in source.edges if edge.source == path[-1]]
        if not outgoing:
            return tuple(path)
        if any(not edge.directed for edge in outgoing):
            return None
        chosen = [
            edge for edge in outgoing if edge.condition is None or _matches(edge.condition, value)
        ]
        if len(chosen) != 1 or chosen[0].target in path:
            return None
        path.append(chosen[0].target)
        if len(path) > len(known):
            return None


def verify_graph(
    task_id: str,
    sources: tuple[GraphSource, GraphSource],
    answers: tuple[GraphAnswer, GraphAnswer],
    public_parameters: Mapping[str, object],
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> GateVerdict:
    """Run a declared graph operation only on two agreed, complete visual graphs."""
    if task_id not in GRAPH_TASKS:
        raise ValueError("No graph verifier for task")
    if any(
        source.coverage != "MET"
        or not source.closed
        or not source.junctions_resolved
        or source.task_id != task_id
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN
    bound = tuple(_label_bound_source(source) for source in sources)
    if bound[0] is None or bound[1] is None:
        return GateVerdict.UNKNOWN
    sources = (bound[0], bound[1])
    if _canonical(sources[0]) != _canonical(sources[1]) or not _regions_agree(*sources):
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
    source = sources[0]
    answer = answers[0]
    if answer.label is not None and answer.label not in answer.answer_quote:
        return GateVerdict.UNKNOWN
    if answer.members is not None and any(
        item not in answer.answer_quote for item in answer.members
    ):
        return GateVerdict.UNKNOWN
    if answer.paths is not None and any(
        item not in answer.answer_quote for path in answer.paths for item in path
    ):
        return GateVerdict.UNKNOWN
    if answer.edges is not None and any(
        item not in answer.answer_quote for edge in answer.edges for item in edge
    ):
        return GateVerdict.UNKNOWN
    query = source.query
    expected: object | None = None
    observed: object | None = None
    if task_id == "diagram_element_lookup" and query.operation == "element":
        node = next((item for item in source.nodes if item.node_id == query.node_id), None)
        expected, observed = (node.label if node else None), answer.label
    elif task_id == "graph_connectivity" and query.operation == "neighbors":
        if query.node_id not in {node.node_id for node in source.nodes}:
            return GateVerdict.UNKNOWN
        expected, observed = (
            _neighbors(source, query.node_id, query.neighbor_direction),
            tuple(sorted(answer.members)) if answer.members is not None else None,
        )
    elif task_id == "graph_connectivity" and query.operation == "edges":
        expected, observed = (
            _edge_pairs(source),
            tuple(sorted(answer.edges)) if answer.edges is not None else None,
        )
    elif task_id == "graph_path_tracing" and query.operation == "paths":
        if query.start is None or query.end is None:
            return GateVerdict.UNKNOWN
        expected = _all_paths(source, query.start, query.end)
        observed = tuple(sorted(answer.paths)) if answer.paths is not None else None
    elif task_id == "diagram_process_description" and query.operation == "process":
        expected, observed = (
            _edge_pairs(source),
            tuple(sorted(answer.edges)) if answer.edges is not None else None,
        )
    elif task_id == "diagram_branch_evaluation" and query.operation == "branch":
        if query.start is None or query.input_value is None:
            return GateVerdict.UNKNOWN
        public_input = public_parameters.get("input_values")
        if isinstance(public_input, tuple) and len(public_input) == 1:
            public_input = public_input[0]
        if str(public_input) != query.input_value:
            return GateVerdict.UNKNOWN
        route = _branch_path(source, query.start, query.input_value)
        expected, observed = (route,) if route else None, answer.paths
    if expected is None or observed is None:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if observed == expected else GateVerdict.NOT_MET
