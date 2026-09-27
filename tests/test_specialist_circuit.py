"""Circuit verification distinguishes junction connectivity from drawn proximity."""

from __future__ import annotations

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_circuit import CircuitSource, verify_circuit

REGION = {"left": 0, "top": 0, "right": 1, "bottom": 1}


def _source() -> CircuitSource:
    return CircuitSource.model_validate(
        {
            "coverage": "MET",
            "closed": True,
            "junctions_resolved": True,
            "domain": "simple-two-terminal",
            "scope_id": "s",
            "view_id": "v",
            "scope_region": REGION,
            "notation": "two_terminal_netlist",
            "components": (
                {"component_id": "R1", "kind": "resistor", "region": REGION},
                {"component_id": "R2", "kind": "resistor", "region": REGION},
                {"component_id": "R3", "kind": "resistor", "region": REGION},
            ),
            "nets": (
                {"terminals": ("R1:a", "R2:a"), "region": REGION},
                {"terminals": ("R1:b", "R2:b", "R3:a"), "region": REGION},
                {"terminals": ("R3:b",), "region": REGION},
            ),
            "reason": "All junctions and crossings resolved",
        }
    )


def test_parallel_pair_requires_both_shared_nets() -> None:
    source = _source()
    args = ((source, source), source.domain, "two_terminal_netlist", "parallel_pairs", "s", "v")
    verdict, evidence = verify_circuit(*args, '{"pairs":[["R1","R2"]]}')
    assert verdict is GateVerdict.MET
    assert evidence["expected_pairs"] == (("R1", "R2"),)
    assert verify_circuit(*args, '{"pairs":[["R1","R3"]]}')[0] is GateVerdict.NOT_MET


def test_ambiguous_crossing_abstains() -> None:
    source = _source()
    unresolved = source.model_copy(update={"junctions_resolved": False})
    verdict, _ = verify_circuit(
        (source, unresolved),
        source.domain,
        "two_terminal_netlist",
        "parallel_pairs",
        "s",
        "v",
        '{"pairs":[]}',
    )
    assert verdict is GateVerdict.UNKNOWN
