"""Blind atom-bond extraction and isolated RDKit structure comparison."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.serialization import strict_json_object
from pixelogue.specialist_env import call_worker
from pixelogue.task_evidence import ImageRegion


class ChemicalAtom(StrictModel):
    """One visible or explicitly conventional skeletal atom."""

    atom_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,15}$")
    element: Literal["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"]
    charge: Annotated[int, Field(ge=-3, le=3)] = 0
    aromatic: bool = False
    region: ImageRegion


class ChemicalBond(StrictModel):
    """One visible bond with explicit order and endpoints."""

    a: str
    b: str
    order: Literal["single", "double", "triple", "aromatic"]
    region: ImageRegion


class ChemicalSource(StrictModel):
    """Complete supported molecular graph from a blind image reading."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    domain: str = Field(min_length=1)
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    notation: Literal["nonstereo_smiles"]
    atoms: Annotated[tuple[ChemicalAtom, ...], Field(max_length=128)]
    bonds: Annotated[tuple[ChemicalBond, ...], Field(max_length=256)]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_graph(self) -> ChemicalSource:
        """Require a local complete graph with all bond endpoints present."""
        if self.coverage != "MET" and (self.atoms or self.bonds):
            raise ValueError("Incomplete chemical reading cannot certify graph")
        if self.coverage == "MET" and not self.atoms:
            raise ValueError("Complete chemical graph needs atoms")
        ids = {atom.atom_id for atom in self.atoms}
        if len(ids) != len(self.atoms):
            raise ValueError("Repeated atom IDs")
        for item in (*self.atoms, *self.bonds):
            region = item.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Chemical evidence lies outside scope")
        if any(bond.a not in ids or bond.b not in ids or bond.a == bond.b for bond in self.bonds):
            raise ValueError("Invalid chemical bond endpoints")
        return self


class ChemicalAnswer(StrictModel):
    """Declared public nonstereochemical SMILES result."""

    smiles: str = Field(min_length=1, max_length=512)


def verify_chemistry(
    sources: tuple[ChemicalSource, ChemicalSource],
    domain: str,
    notation: object,
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, dict[str, object]]:
    """Match a candidate to two agreed graphs with RDKit atom and valence checks."""
    if any(
        source.coverage != "MET"
        or source.domain != domain
        or source.notation != notation
        or source.scope_id != scope_id
        or source.view_id != view_id
        for source in sources
    ):
        return GateVerdict.UNKNOWN, {"reason": "Incomplete or mismatched chemical evidence"}
    canonical = [
        source.model_dump(mode="json", exclude={"reason", "scope_region"}) for source in sources
    ]
    for item in canonical:
        for atom in item["atoms"]:
            atom.pop("region", None)
        for bond in item["bonds"]:
            bond.pop("region", None)
    if canonical[0] != canonical[1]:
        return GateVerdict.UNKNOWN, {"reason": "Independent molecular graphs disagree"}
    try:
        raw = strict_json_object(candidate_answer)
        if set(raw) != {"smiles"}:
            raise ValueError("Unsupported chemical output schema")
        answer = ChemicalAnswer.model_validate_json(candidate_answer)
    except (ExternalInputError, ValueError, TypeError):
        return GateVerdict.UNKNOWN, {"reason": "Malformed chemical output"}
    worker_source = sources[0].model_dump(
        mode="json",
        exclude={"reason", "scope_id", "view_id", "scope_region", "domain", "coverage", "notation"},
    )
    for atom in worker_source["atoms"]:
        atom.pop("region", None)
    for bond in worker_source["bonds"]:
        bond.pop("region", None)
    try:
        response = call_worker(
            "chemistry",
            "rdkit",
            {
                "operation": "chemistry",
                "source": worker_source,
                "reported_smiles": answer.smiles,
            },
            timeout_seconds=20,
        )
    except ExecutionError as exc:
        return GateVerdict.UNKNOWN, {"reason": str(exc)}
    return GateVerdict(response["verdict"]), response
