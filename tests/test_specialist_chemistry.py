"""Chemical candidates require a complete agreed visual graph and RDKit valence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_chemistry import (
    PUBCHEM_SMALL_DOMAIN,
    ChemicalSource,
    chemical_domain_conditions,
    verify_chemistry,
)

REGION = {"left": 0, "top": 0, "right": 1, "bottom": 1}


def _source() -> ChemicalSource:
    return ChemicalSource.model_validate(
        {
            "coverage": "MET",
            "domain": "small-nonstereo",
            "scope_id": "s",
            "view_id": "v",
            "scope_region": REGION,
            "notation": "nonstereo_smiles",
            "atoms": (
                {"atom_id": "c1", "element": "C", "region": REGION},
                {"atom_id": "c2", "element": "C", "region": REGION},
                {"atom_id": "o", "element": "O", "region": REGION},
            ),
            "bonds": (
                {"a": "c1", "b": "c2", "order": "single", "region": REGION},
                {"a": "c2", "b": "o", "order": "single", "region": REGION},
            ),
            "reason": "Complete visible structure",
        }
    )


def test_rdkit_matches_graph_and_rejects_wrong_connectivity() -> None:
    source = _source()
    args = ((source, source), source.domain, "nonstereo_smiles", "s", "v")
    verdict, evidence = verify_chemistry(*args, '{"smiles":"CCO"}')
    assert verdict is GateVerdict.MET
    assert evidence["expected_smiles"] == "CCO"
    assert verify_chemistry(*args, '{"smiles":"COC"}')[0] is GateVerdict.NOT_MET
    assert verify_chemistry(*args, '{"smiles":"COC","smiles":"CCO"}')[0] is GateVerdict.UNKNOWN


def test_unresolved_stereochemistry_and_disputed_graph_abstain() -> None:
    source = _source()
    args = ((source, source), source.domain, "nonstereo_smiles", "s", "v")
    assert verify_chemistry(*args, '{"smiles":"C[C@H]O"}')[0] is GateVerdict.UNKNOWN
    disputed = source.model_copy(update={"bonds": source.bonds[:1]})
    assert (
        verify_chemistry(
            (source, disputed), source.domain, "nonstereo_smiles", "s", "v", '{"smiles":"CCO"}'
        )[0]
        is GateVerdict.UNKNOWN
    )


def test_out_of_scope_chemical_box_identifies_the_offending_region() -> None:
    source = _source().model_dump()
    source["scope_region"] = {"left": 0.2, "top": 0.2, "right": 0.8, "bottom": 0.8}
    with pytest.raises(ValidationError, match=r"atoms\[0\].*exceeds scope"):
        ChemicalSource.model_validate(source)


def test_inverted_diagonal_and_zero_width_bond_boxes_remain_invalid() -> None:
    for region in (
        {"left": 0.3, "top": 0.7, "right": 0.6, "bottom": 0.4},
        {"left": 0.4, "top": 0.3, "right": 0.4, "bottom": 0.6},
    ):
        source = _source().model_dump()
        source["bonds"][0]["region"] = region.copy()
        with pytest.raises(ValidationError, match="positive extent"):
            ChemicalSource.model_validate(source)
        assert source["bonds"][0]["region"] == region


@pytest.mark.parametrize("smiles", ["[13CH3]CO", "[CH2]CO"])
def test_unrepresented_isotope_and_radical_smiles_abstain(smiles: str) -> None:
    import json

    source = _source()
    args = ((source, source), source.domain, "nonstereo_smiles", "s", "v")
    verdict, evidence = verify_chemistry(*args, json.dumps({"smiles": smiles}))
    assert verdict is GateVerdict.UNKNOWN
    assert evidence["worker_reasons"] == [
        "Isotope or radical SMILES are unsupported",
        "Isotope or radical SMILES are unsupported",
    ]


def test_pubchem_domain_accepts_supported_graph_and_keeps_wrong_connectivity_rejected() -> None:
    source = _source().model_copy(update={"domain": PUBCHEM_SMALL_DOMAIN})
    args = ((source, source), source.domain, "nonstereo_smiles", "s", "v")
    assert verify_chemistry(*args, '{"smiles":"CCO"}')[0] is GateVerdict.MET
    assert verify_chemistry(*args, '{"smiles":"COC"}')[0] is GateVerdict.NOT_MET


@pytest.mark.parametrize(
    "violation", ["small", "large", "element", "charge", "cycle", "disconnected"]
)
def test_pubchem_domain_rejects_complete_graphs_outside_the_calibrated_range(
    violation: str, monkeypatch
) -> None:
    from pixelogue import specialist_chemistry

    graph = _source().model_dump()
    graph["domain"] = PUBCHEM_SMALL_DOMAIN
    if violation == "small":
        graph["atoms"] = graph["atoms"][:2]
        graph["bonds"] = graph["bonds"][:1]
    elif violation == "large":
        graph["atoms"] = tuple(
            {"atom_id": f"c{i}", "element": "C", "region": REGION} for i in range(6)
        )
        graph["bonds"] = tuple(
            {"a": f"c{i}", "b": f"c{i + 1}", "order": "single", "region": REGION} for i in range(5)
        )
    elif violation == "element":
        graph["atoms"][-1]["element"] = "S"
    elif violation == "charge":
        graph["atoms"][-1]["charge"] = 1
    else:
        graph["bonds"] += ({"a": "o", "b": "c1", "order": "single", "region": REGION},)
        if violation == "disconnected":
            # A triangle and an isolated atom still have n-1 edges, but are not a tree.
            graph["atoms"] += ({"atom_id": "n", "element": "N", "region": REGION},)
    source = ChemicalSource.model_validate(graph)

    def never_compare(*args, **kwargs):
        raise AssertionError("A graph outside calibration cannot reach SMILES comparison")

    monkeypatch.setattr(specialist_chemistry, "call_worker", never_compare)
    args = ((source, source), source.domain, "nonstereo_smiles", "s", "v")
    verdict, evidence = verify_chemistry(*args, '{"smiles":"CCO"}')
    assert verdict is GateVerdict.UNKNOWN
    assert evidence["domain_conditions"] == chemical_domain_conditions(PUBCHEM_SMALL_DOMAIN)


def test_calibrated_chemical_conditions_are_public_without_an_answer() -> None:
    from pixelogue.catalog import task_catalog
    from pixelogue.contracts import InstructionCandidate
    from pixelogue.task_runtime import operation_contract

    task = next(task for task in task_catalog().tasks if task.id == "chemical_structure_reading")
    instruction = InstructionCandidate(
        candidate_id="public-conditions",
        task_id=task.id,
        family=task.family,
        profile="normal",
        visible_scope="complete molecular drawing",
        instruction_summary=task.definition_en,
        required_capabilities=task.required_capabilities,
        catalog_version="7.0",
        scope_id="s",
        view_id="v",
        evidence_refs=("image-evidence",),
        verification_contracts=task.verification_contracts,
        calibrated_domain=PUBCHEM_SMALL_DOMAIN,
    )
    contract = operation_contract(instruction)
    conditions = chemical_domain_conditions(PUBCHEM_SMALL_DOMAIN)
    assert conditions in contract["eligibility_checks"]["calibrated_domain_supported"]
    assert "candidate_answer" not in contract


def test_aromatic_bond_requires_a_closed_ring() -> None:
    source = _source().model_dump()
    source["bonds"][0]["order"] = "aromatic"
    with pytest.raises(ValidationError, match="needs a visible closed ring"):
        ChemicalSource.model_validate(source)

    source["atoms"] = tuple(
        {"atom_id": f"c{index}", "element": "C", "aromatic": True, "region": REGION}
        for index in range(6)
    )
    source["bonds"] = tuple(
        {"a": f"c{index}", "b": f"c{(index + 1) % 6}", "order": "aromatic", "region": REGION}
        for index in range(6)
    )
    assert ChemicalSource.model_validate(source).coverage == "MET"


def test_graph_equivalence_ignores_atom_ids_and_explicit_hydrogens() -> None:
    source = _source()
    graph = source.model_dump(mode="json")
    renamed = dict(graph)
    renamed["atoms"] = tuple(
        {**atom, "atom_id": {"c1": "left", "c2": "middle", "o": "right"}[atom["atom_id"]]}
        for atom in reversed(graph["atoms"])
    )
    renamed["bonds"] = tuple(
        {
            **bond,
            "a": {"c1": "left", "c2": "middle", "o": "right"}[bond["a"]],
            "b": {"c1": "left", "c2": "middle", "o": "right"}[bond["b"]],
        }
        for bond in reversed(graph["bonds"])
    )
    equivalent = ChemicalSource.model_validate(renamed)
    args = ((source, equivalent), source.domain, "nonstereo_smiles", "s", "v")
    assert verify_chemistry(*args, '{"smiles":"CCO"}')[0] is GateVerdict.MET
    assert verify_chemistry(*args, '{"smiles":"COC"}')[0] is GateVerdict.NOT_MET

    explicit = source.model_dump(mode="json")
    explicit["atoms"].append({"atom_id": "h", "element": "H", "region": REGION})
    explicit["bonds"].append({"a": "o", "b": "h", "order": "single", "region": REGION})
    explicit["atoms"] = tuple(explicit["atoms"])
    explicit["bonds"] = tuple(explicit["bonds"])
    with_h = ChemicalSource.model_validate(explicit)
    assert (
        verify_chemistry(
            (source, with_h), source.domain, "nonstereo_smiles", "s", "v", '{"smiles":"CCO"}'
        )[0]
        is GateVerdict.MET
    )
