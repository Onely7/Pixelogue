"""Circuit verification distinguishes junction connectivity from drawn proximity."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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
    assert verify_circuit(*args, '{"pairs":[],"pairs":[["R1","R2"]]}')[0] is GateVerdict.UNKNOWN


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


def test_netlist_must_preserve_declared_component_terminal_names() -> None:
    source = _source()
    args = ((source, source), source.domain, "two_terminal_netlist", "netlist", "s", "v")
    answer = {
        "components": {component.component_id: component.kind for component in source.components},
        "nets": [list(net.terminals) for net in source.nets],
    }
    assert verify_circuit(*args, json.dumps(answer))[0] is GateVerdict.MET
    answer["nets"][0] = ["R1:a", "R2:b"]
    answer["nets"][1] = ["R1:b", "R2:a", "R3:a"]
    assert verify_circuit(*args, json.dumps(answer))[0] is GateVerdict.NOT_MET


def test_circuit_error_identifies_missing_duplicate_and_unexpected_terminals() -> None:
    source = _source().model_dump(mode="python")
    source["nets"][0]["terminals"] = ("R1:a", "R1:a", "R9:b")
    with pytest.raises(ValidationError) as error:
        CircuitSource.model_validate(source)
    message = str(error.value)
    assert "missing=['R2:a']" in message
    assert "duplicate=['R1:a']" in message
    assert "unexpected=['R9:b']" in message


def test_inconsistent_complete_flag_is_rejected_without_repairing_the_source() -> None:
    source = _source().model_dump()
    source["closed"] = False
    with pytest.raises(ValidationError, match="Circuit is open"):
        CircuitSource.model_validate(source)
    assert source["closed"] is False
