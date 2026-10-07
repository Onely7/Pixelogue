"""Diagram edges, routes, crossings and public branch inputs."""

from __future__ import annotations

import json

import pytest

from pixelogue.contracts import GateVerdict
from pixelogue.graph_verifiers import (
    BranchCondition,
    GraphAnswer,
    GraphEdge,
    GraphNode,
    GraphQuery,
    GraphSource,
    verify_graph,
)
from pixelogue.task_evidence import ImageRegion

REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


def _node(key: str) -> GraphNode:
    return GraphNode(node_id=key, label=key, kind="box", region=REGION)


def _edge(a: str, b: str, condition: BranchCondition | None = None) -> GraphEdge:
    return GraphEdge(source=a, target=b, directed=True, condition=condition, region=REGION)


def _source(task: str, operation: str, **query: object) -> GraphSource:
    return GraphSource.model_validate(
        {
            "task_id": task,
            "coverage": "MET",
            "closed": True,
            "junctions_resolved": True,
            "scope_id": "scope",
            "view_id": "view",
            "scope_region": REGION,
            "nodes": (_node("A"), _node("B"), _node("C")),
            "edges": (_edge("A", "B"), _edge("B", "C"), _edge("A", "C")),
            "query": {"operation": operation, **query},
            "reason": "All arrows and endpoints visible",
        }
    )


def _answer(quote: str, **result: object) -> GraphAnswer:
    return GraphAnswer.model_validate(
        {"coverage": "MET", "answer_quote": quote, "reason": "literal", **result}
    )


def _check(source: GraphSource, answer: GraphAnswer, **parameters: object) -> GateVerdict:
    return verify_graph(
        source.task_id,
        (source, source),
        (answer, answer),
        parameters,
        source.scope_id,
        source.view_id,
        answer.answer_quote,
    )


def test_directed_neighbors_follow_drawn_edges() -> None:
    outgoing = _source("diagram_connectivity", "neighbors", node_id="A")
    assert _check(outgoing, _answer("B and C", members=("B", "C"))) is GateVerdict.MET
    incoming = outgoing.model_copy(
        update={
            "query": GraphQuery(operation="neighbors", node_id="C", neighbor_direction="incoming")
        }
    )
    assert _check(incoming, _answer("A and B", members=("A", "B"))) is GateVerdict.MET


def test_neighbors_compare_topology_without_requiring_an_unused_node_subtype() -> None:
    source = _source("diagram_connectivity", "neighbors", node_id="A")
    other = source.model_copy(
        update={
            "nodes": tuple(node.model_copy(update={"kind": "station"}) for node in source.nodes)
        }
    )
    answer = _answer("B and C", members=("B", "C"))
    assert (
        verify_graph(
            source.task_id,
            (source, other),
            (answer, answer),
            {},
            "scope",
            "view",
            answer.answer_quote,
        )
        is GateVerdict.MET
    )
    missing = other.model_copy(update={"edges": other.edges[:-1]})
    assert (
        verify_graph(
            source.task_id,
            (source, missing),
            (answer, answer),
            {},
            "scope",
            "view",
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )


def test_bilingual_neighbor_names_require_explicit_unique_visible_labels() -> None:
    source = _source("diagram_connectivity", "neighbors", node_id="A")
    answer = _answer("Alpha (B), Beta (C)", members=("Alpha (B)", "Beta (C)"))
    assert _check(source, answer) is GateVerdict.MET
    wrong = _answer("Alpha (B), Gamma (D)", members=("Alpha (B)", "Gamma (D)"))
    assert _check(source, wrong) is GateVerdict.NOT_MET
    duplicate = _answer("Alpha (B), Beta (B)", members=("Alpha (B)", "Beta (B)"))
    assert _check(source, duplicate) is GateVerdict.NOT_MET


def test_all_paths_include_every_visible_alternative() -> None:
    source = _source("diagram_path_tracing", "paths", start="A", end="C")
    correct = _answer("A B C; A C", paths=(("A", "B", "C"), ("A", "C")))
    assert _check(source, correct) is GateVerdict.MET
    missing = _answer("A C", paths=(("A", "C"),))
    assert _check(source, missing) is GateVerdict.NOT_MET


def test_process_description_checks_all_explicit_edges() -> None:
    source = _source("diagram_process_description", "process")
    answer = _answer("A to B, B to C, A to C", edges=(("A", "B"), ("B", "C"), ("A", "C")))
    assert _check(source, answer) is GateVerdict.MET
    incomplete = _answer("A to B, B to C", edges=(("A", "B"), ("B", "C")))
    assert _check(source, incomplete) is GateVerdict.NOT_MET


def test_public_numeric_branch_uses_printed_conditions() -> None:
    high = BranchCondition(operator="ge", threshold="5", printed_text="input >= 5")
    low = BranchCondition(operator="lt", threshold="5", printed_text="input < 5")
    base = _source("flowchart_evaluation", "branch", start="A", input_value="8")
    source = base.model_copy(update={"edges": (_edge("A", "B", high), _edge("A", "C", low))})
    answer = _answer("A to B", paths=(("A", "B"),))
    assert _check(source, answer, input_values="8") is GateVerdict.MET
    assert _check(source, answer, input_values="7") is GateVerdict.UNKNOWN
    overlapping = source.model_copy(
        update={"edges": (_edge("A", "B", high), _edge("A", "C", high))}
    )
    assert _check(overlapping, answer, input_values="8") is GateVerdict.UNKNOWN


def test_unresolved_crossing_and_source_disagreement_abstain() -> None:
    source = _source("diagram_connectivity", "neighbors", node_id="A")
    answer = _answer("B and C", members=("B", "C"))
    crossing = source.model_copy(update={"junctions_resolved": False})
    assert _check(crossing, answer) is GateVerdict.UNKNOWN
    other = source.model_copy(update={"edges": source.edges[:-1]})
    assert (
        verify_graph(
            source.task_id,
            (source, other),
            (answer, answer),
            {},
            source.scope_id,
            source.view_id,
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )


def _renamed(source: GraphSource) -> GraphSource:
    data = source.model_dump()
    names = {node.node_id: "node_" + node.node_id for node in source.nodes}
    for node in data["nodes"]:
        node["node_id"] = names[node["node_id"]]
    for edge in data["edges"]:
        edge["source"], edge["target"] = names[edge["source"]], names[edge["target"]]
    for field in ("start", "end", "node_id"):
        if data["query"][field] is not None:
            data["query"][field] = names[data["query"][field]]
    return GraphSource.model_validate(data)


@pytest.mark.parametrize("rename_both", [False, True])
def test_unique_visible_labels_bind_local_ids_without_changing_saved_evidence(rename_both) -> None:
    original = _source("diagram_connectivity", "neighbors", node_id="A")
    other = _renamed(original)
    source = other if rename_both else original
    before = other.model_dump_json()
    for answer, expected in (
        (_answer("B and C", members=("B", "C")), GateVerdict.MET),
        (_answer("B", members=("B",)), GateVerdict.NOT_MET),
    ):
        assert (
            verify_graph(
                source.task_id,
                (source, other),
                (answer, answer),
                {},
                "scope",
                "view",
                answer.answer_quote,
            )
            is expected
        )
    assert other.model_dump_json() == before


@pytest.mark.parametrize(
    "mismatch",
    [
        "duplicate_label",
        "region",
        "edge_region",
        "direction",
        "query",
        "missing_reference",
        "missing_edge",
    ],
)
def test_label_binding_preserves_ambiguity_and_structural_disagreement(mismatch) -> None:
    source = _source("diagram_connectivity", "neighbors", node_id="A")
    data = _renamed(source).model_dump()
    if mismatch == "duplicate_label":
        data["nodes"][1]["label"] = "A"
    elif mismatch == "region":
        data["nodes"][0]["region"] = {"left": 0.99, "top": 0.99, "right": 1, "bottom": 1}
    elif mismatch == "edge_region":
        data["edges"][0]["region"] = {"left": 0.99, "top": 0.99, "right": 1, "bottom": 1}
    elif mismatch == "direction":
        data["edges"][0]["directed"] = False
    elif mismatch == "query":
        data["query"]["node_id"] = "node_B"
    elif mismatch == "missing_reference":
        data["query"]["node_id"] = "absent"
    else:
        data["edges"] = data["edges"][:-1]
    other = GraphSource.model_validate(data)
    answer = _answer("B and C", members=("B", "C"))
    assert (
        verify_graph(
            source.task_id,
            (source, other),
            (answer, answer),
            {},
            "scope",
            "view",
            answer.answer_quote,
        )
        is GateVerdict.UNKNOWN
    )


def test_a_repeated_edge_is_read_as_the_same_link() -> None:
    source = _source("diagram_connectivity", "neighbors", node_id="A")
    data = source.model_dump(mode="json")
    data["edges"].append(data["edges"][0])
    repeated = GraphSource.model_validate_json(json.dumps(data))
    assert len(repeated.edges) == 4
    answer = _answer("B and C", members=("B", "C"))
    assert (
        verify_graph(
            source.task_id,
            (source, repeated),
            (answer, answer),
            {},
            "scope",
            "view",
            answer.answer_quote,
        )
        is GateVerdict.MET
    )
