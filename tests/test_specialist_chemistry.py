"""Chemical candidates require a complete agreed visual graph and RDKit valence."""

from __future__ import annotations

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
