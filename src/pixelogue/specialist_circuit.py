"""Closed circuit netlist extraction and local topology verification."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExternalInputError
from pixelogue.serialization import strict_json_object
from pixelogue.task_evidence import ImageRegion


class CircuitComponent(StrictModel):
    """Registered visible two-terminal component."""

    component_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
    kind: Literal["resistor", "capacitor", "inductor", "battery", "switch", "diode", "lamp"]
    region: ImageRegion


class CircuitNet(StrictModel):
    """One resolved electrical junction, including crossover decisions."""

    terminals: Annotated[tuple[str, ...], Field(min_length=1, max_length=64)]
    region: ImageRegion


class CircuitSource(StrictModel):
    """Complete blind netlist from the scoped drawing."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    closed: bool
    junctions_resolved: bool
    domain: str = Field(min_length=1)
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    notation: Literal["two_terminal_netlist"]
    components: Annotated[tuple[CircuitComponent, ...], Field(max_length=64)]
    nets: Annotated[tuple[CircuitNet, ...], Field(max_length=128)]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_netlist(self) -> CircuitSource:
        """Require every component terminal exactly once in a local, closed net."""
        if self.coverage != "MET" and (self.closed or self.components or self.nets):
            raise ValueError("Incomplete circuit cannot certify a netlist")
        if self.coverage != "MET":
            return self
        ids = [component.component_id for component in self.components]
        if not self.closed or not self.junctions_resolved or not ids or len(ids) != len(set(ids)):
            raise ValueError("Circuit is open or component IDs are repeated")
        expected = {f"{component_id}:{pin}" for component_id in ids for pin in ("a", "b")}
        observed = [terminal for net in self.nets for terminal in net.terminals]
        missing = sorted(expected - set(observed))
        duplicate = sorted(terminal for terminal, count in Counter(observed).items() if count > 1)
        unexpected = sorted(set(observed) - expected)
        if missing or duplicate or unexpected:
            raise ValueError(
                "Circuit netlist terminals differ: "
                f"missing={missing}, duplicate={duplicate}, unexpected={unexpected}"
            )
        for item in (*self.components, *self.nets):
            region = item.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Circuit evidence lies outside scope")
        return self


class CircuitAnswer(StrictModel):
    """Public topology result with exact declared output shape."""

    pairs: tuple[tuple[str, str], ...] | None = None
    components: dict[str, str] | None = None
    nets: tuple[tuple[str, ...], ...] | None = None


def _topology(source: CircuitSource) -> tuple[dict[str, str], tuple[frozenset[str], ...]]:
    return (
        {item.component_id: item.kind for item in source.components},
        tuple(
            sorted(
                (frozenset(net.terminals) for net in source.nets),
                key=lambda net: tuple(sorted(net)),
            )
        ),
    )


def _pairs(source: CircuitSource, relation: str) -> tuple[tuple[str, str], ...]:
    net_by_terminal = {
        terminal: index for index, net in enumerate(source.nets) for terminal in net.terminals
    }
    result = []
    for first, second in combinations(sorted(item.component_id for item in source.components), 2):
        a = {net_by_terminal[f"{first}:{pin}"] for pin in ("a", "b")}
        b = {net_by_terminal[f"{second}:{pin}"] for pin in ("a", "b")}
        if relation == "parallel_pairs" and len(a) == len(b) == 2 and a == b:
            result.append((first, second))
        elif relation == "series_pairs" and len(a) == len(b) == 2 and len(a & b) == 1:
            shared = next(iter(a & b))
            if len(source.nets[shared].terminals) == 2:
                result.append((first, second))
    return tuple(result)


def verify_circuit(
    sources: tuple[CircuitSource, CircuitSource],
    domain: str,
    notation: object,
    operation: object,
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, dict[str, object]]:
    """Compare a public netlist or relationship against two blind readings."""
    if any(
        source.coverage != "MET"
        or not source.closed
        or not source.junctions_resolved
        or source.domain != domain
        or source.notation != notation
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN, {"reason": "Circuit topology or notation is unresolved"}
    if _topology(sources[0]) != _topology(sources[1]):
        return GateVerdict.UNKNOWN, {"reason": "Independent circuit extractions disagree"}
    try:
        raw = strict_json_object(candidate_answer)
        answer = CircuitAnswer.model_validate_json(candidate_answer)
    except (ExternalInputError, ValueError, TypeError):
        return GateVerdict.UNKNOWN, {"reason": "Malformed circuit output"}
    if operation in {"parallel_pairs", "series_pairs"}:
        if set(raw) != {"pairs"} or answer.pairs is None:
            return GateVerdict.UNKNOWN, {"reason": "Wrong circuit output schema"}
        expected = _pairs(sources[0], str(operation))
        observed = tuple(sorted(tuple(sorted(pair)) for pair in answer.pairs))
        verdict = GateVerdict.MET if observed == expected else GateVerdict.NOT_MET
        return verdict, {"expected_pairs": expected, "operation": operation}
    if operation == "netlist":
        if set(raw) != {"components", "nets"} or answer.components is None or answer.nets is None:
            return GateVerdict.UNKNOWN, {"reason": "Wrong netlist output schema"}
        expected_components, expected_nets = _topology(sources[0])
        observed_nets = tuple(frozenset(net) for net in answer.nets)
        verdict = (
            GateVerdict.MET
            if answer.components == expected_components
            and set(observed_nets) == set(expected_nets)
            and len(observed_nets) == len(expected_nets)
            else GateVerdict.NOT_MET
        )
        return verdict, {
            "expected_components": expected_components,
            "expected_nets": [sorted(net) for net in expected_nets],
        }
    return GateVerdict.UNKNOWN, {"reason": "Unsupported circuit operation"}
