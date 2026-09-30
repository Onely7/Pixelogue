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

PUBCHEM_SMALL_DOMAIN = "pubchem-2d-acyclic-cnohalogen-3-5-v1"


def chemical_domain_conditions(domain: str) -> str | None:
    """Return public conditions for the frozen small-molecule calibration cohort."""
    if domain != PUBCHEM_SMALL_DOMAIN:
        return None
    return (
        "One connected, acyclic, nonaromatic, neutral molecule with 3 to 5 heavy atoms. "
        "Heavy atoms may only be C, N, O, F, Cl, or Br; explicitly drawn H is allowed. "
        "Isotope and radical symbols are unsupported."
    )


def _within_pubchem_domain(source: ChemicalSource) -> bool:
    """Check computable cohort limits on a complete validated molecular graph."""
    heavy_atoms = sum(atom.element != "H" for atom in source.atoms)
    if not 3 <= heavy_atoms <= 5 or any(
        atom.element not in {"C", "N", "O", "F", "Cl", "Br", "H"}
        or atom.charge != 0
        or atom.aromatic
        for atom in source.atoms
    ):
        return False
    if len(source.bonds) != len(source.atoms) - 1 or any(
        bond.order == "aromatic" for bond in source.bonds
    ):
        return False
    neighbors: dict[str, set[str]] = {atom.atom_id: set() for atom in source.atoms}
    for bond in source.bonds:
        neighbors[bond.a].add(bond.b)
        neighbors[bond.b].add(bond.a)
    reached: set[str] = set()
    pending = [source.atoms[0].atom_id]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(neighbors[current] - reached)
    return len(reached) == len(source.atoms)


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
        for kind, items in (("atoms", self.atoms), ("bonds", self.bonds)):
            for index, item in enumerate(items):
                region = item.region
                scope = self.scope_region
                if not (
                    scope.left <= region.left < region.right <= scope.right
                    and scope.top <= region.top < region.bottom <= scope.bottom
                ):
                    raise ValueError(
                        f"Chemical {kind}[{index}] region "
                        f"[{region.left:g}, {region.top:g}, {region.right:g}, {region.bottom:g}] "
                        f"exceeds scope [{scope.left:g}, {scope.top:g}, "
                        f"{scope.right:g}, {scope.bottom:g}]"
                    )
        if any(bond.a not in ids or bond.b not in ids or bond.a == bond.b for bond in self.bonds):
            raise ValueError("Invalid chemical bond endpoints")
        neighbors: dict[str, set[str]] = {atom_id: set() for atom_id in ids}
        for bond in self.bonds:
            if bond.b in neighbors[bond.a]:
                raise ValueError("Repeated chemical bond")
            neighbors[bond.a].add(bond.b)
            neighbors[bond.b].add(bond.a)
        for bond in self.bonds:
            if bond.order != "aromatic":
                continue
            reached = {bond.a}
            pending = [bond.a]
            while pending:
                current = pending.pop()
                for adjacent in neighbors[current]:
                    if {current, adjacent} == {bond.a, bond.b} or adjacent in reached:
                        continue
                    reached.add(adjacent)
                    pending.append(adjacent)
            if bond.b not in reached:
                raise ValueError(f"Aromatic bond {bond.a}-{bond.b} needs a visible closed ring")
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
    if (conditions := chemical_domain_conditions(domain)) is not None and any(
        not _within_pubchem_domain(source) for source in sources
    ):
        return GateVerdict.UNKNOWN, {
            "reason": "Chemical source is outside its declared calibration domain",
            "domain_conditions": conditions,
        }
    try:
        raw = strict_json_object(candidate_answer)
        if set(raw) != {"smiles"}:
            raise ValueError("Unsupported chemical output schema")
        answer = ChemicalAnswer.model_validate_json(candidate_answer)
    except (ExternalInputError, ValueError, TypeError):
        return GateVerdict.UNKNOWN, {"reason": "Malformed chemical output"}
    try:
        responses = []
        for source in sources:
            worker_source = source.model_dump(
                mode="json",
                exclude={
                    "reason",
                    "scope_id",
                    "view_id",
                    "scope_region",
                    "domain",
                    "coverage",
                    "notation",
                },
            )
            for atom in worker_source["atoms"]:
                atom.pop("region", None)
            for bond in worker_source["bonds"]:
                bond.pop("region", None)
            responses.append(
                call_worker(
                    "chemistry",
                    "rdkit",
                    {
                        "operation": "chemistry",
                        "source": worker_source,
                        "reported_smiles": answer.smiles,
                    },
                    timeout_seconds=20,
                )
            )
    except ExecutionError as exc:
        return GateVerdict.UNKNOWN, {"reason": str(exc)}
    if any(response.get("expected_smiles") is None for response in responses):
        return GateVerdict.UNKNOWN, {
            "reason": "Molecular graph or SMILES could not be resolved",
            "worker_reasons": [response.get("reason") for response in responses],
        }
    if responses[0]["expected_smiles"] != responses[1]["expected_smiles"]:
        return GateVerdict.UNKNOWN, {"reason": "Independent molecular graphs disagree"}
    return GateVerdict(responses[0]["verdict"]), responses[0]
