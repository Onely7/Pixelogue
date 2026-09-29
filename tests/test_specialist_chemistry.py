"""Chemical candidates require a complete agreed visual graph and RDKit valence."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_chemistry import ChemicalSource, verify_chemistry

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
