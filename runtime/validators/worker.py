"""Pinned specialist computation worker; one bounded JSON request per process."""

from __future__ import annotations

import json
import re
import resource
import sys
from typing import Any


def _limit_resources() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (12, 12))
    resource.setrlimit(resource.RLIMIT_AS, (2_500_000_000, 2_500_000_000))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))


def _rational(value: object) -> Any:
    import sympy as sp

    if (
        not isinstance(value, str)
        or len(value) > 32
        or not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?(?:/[0-9]+)?", value)
    ):
        raise ValueError("Unsupported numeric value")
    return sp.Rational(value)


def _geometry(request: dict[str, Any]) -> dict[str, Any]:
    import sympy as sp

    premises = request["premises"]
    if not isinstance(premises, list) or not 1 <= len(premises) <= 32:
        raise ValueError("Invalid premise count")
    names = {name for premise in premises for name in premise["variables"]}
    target = request["target"]
    if (
        target not in names
        or len(names) > 16
        or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", name) for name in names)
    ):
        raise ValueError("Invalid geometry variable")
    symbols = {name: sp.Symbol(name, real=True) for name in names}
    equations: list[Any] = []
    trace: list[str] = []
    for premise in premises:
        rule = premise["rule"]
        variables = [symbols[name] for name in premise["variables"]]
        constants = [_rational(item) for item in premise["constants"]]
        if rule == "given" and len(variables) == len(constants) == 1:
            equation = variables[0] - constants[0]
        elif rule == "right_angle" and len(variables) == 1 and not constants:
            equation = variables[0] - 90
        elif rule == "triangle_angle_sum" and len(variables) == 3 and not constants:
            equation = sum(variables) - 180
        elif rule == "parallel_equal_angle" and len(variables) == 2 and not constants:
            equation = variables[0] - variables[1]
        elif (
            rule == "similar_ratio"
            and len(variables) == 2
            and len(constants) == 2
            and constants[1] != 0
        ):
            equation = constants[1] * variables[0] - constants[0] * variables[1]
        elif rule == "pythagorean" and len(variables) == 3 and not constants:
            equation = variables[0] ** 2 + variables[1] ** 2 - variables[2] ** 2
        else:
            raise ValueError("Unsupported geometry rule or arity")
        equations.append(equation)
        trace.append(f"{rule}: {equation}=0")
    solutions = sp.solve(equations, list(symbols.values()), dict=True)
    domain = request["target_domain"]
    target_values: set[Any] = set()
    for solution in solutions:
        value = solution.get(symbols[target])
        if value is None or value.free_symbols or value.is_real is not True:
            continue
        if domain == "length" and value <= 0:
            continue
        if domain == "angle" and not 0 <= value <= 180:
            continue
        target_values.add(sp.simplify(value))
    if len(target_values) != 1:
        return {
            "verdict": "UNKNOWN",
            "reason": "Target has no unique in-domain solution",
            "trace": trace,
        }
    expected = next(iter(target_values))
    if expected.is_Rational is not True:
        return {
            "verdict": "UNKNOWN",
            "reason": "Nonrational output is outside first-version contract",
            "trace": trace,
        }
    observed = _rational(request["reported"])
    return {
        "verdict": "MET" if observed == expected else "NOT_MET",
        "expected": str(expected),
        "trace": trace,
    }


def _music(request: dict[str, Any]) -> dict[str, Any]:
    """Decode a bounded single-voice score with music21."""
    from fractions import Fraction

    from music21 import duration, key, note, pitch, tie

    source = request["source"]
    clef = source["clef"]
    if clef not in {"treble", "bass"}:
        raise ValueError("Unsupported clef")
    base_letter, base_octave = ("E", 4) if clef == "treble" else ("G", 2)
    scale = "CDEFGAB"
    base_index = scale.index(base_letter) + 7 * base_octave
    signature = key.KeySignature(source["key_sharps"])
    altered = {item.step: int(item.accidental.alter) for item in signature.alteredPitches}
    top, bottom = source["meter_top"], source["meter_bottom"]
    if top < 1 or top > 12 or bottom not in {2, 4, 8}:
        raise ValueError("Unsupported meter")
    bar_duration = Fraction(4 * top, bottom)
    expected: list[dict[str, Any]] = []
    measures = source["measures"]
    if not 1 <= len(measures) <= 8:
        raise ValueError("Unsupported score length")
    for measure in measures:
        accidental_memory: dict[tuple[str, int], int] = {}
        total = Fraction(0)
        for event in measure["events"]:
            base = event["base"]
            dots = event["dots"]
            if base not in {1, 2, 4, 8, 16} or dots not in {0, 1}:
                raise ValueError("Unsupported duration")
            written = duration.Duration(
                quarterLength=float(Fraction(4, base) * (Fraction(3, 2) if dots else 1))
            )
            length = Fraction(str(written.quarterLength))
            total += length
            if event["kind"] == "rest":
                if (
                    event["staff_step"] is not None
                    or event["accidental"] is not None
                    or event["tie"] is not None
                ):
                    raise ValueError("Rest cannot carry pitch or tie")
                item = note.Rest()
                item.duration = written
                pitch_name = None
            elif event["kind"] == "note":
                staff_step = event["staff_step"]
                if not isinstance(staff_step, int) or not -8 <= staff_step <= 16:
                    raise ValueError("Unsupported staff position")
                index = base_index + staff_step
                letter, octave = scale[index % 7], index // 7
                explicit = event["accidental"]
                if explicit is not None:
                    alteration = {"flat": -1, "natural": 0, "sharp": 1}[explicit]
                    accidental_memory[(letter, octave)] = alteration
                else:
                    alteration = accidental_memory.get((letter, octave), altered.get(letter, 0))
                accidental_text = "#" if alteration == 1 else "-" if alteration == -1 else ""
                item = note.Note(pitch.Pitch(f"{letter}{accidental_text}{octave}"))
                item.duration = written
                if event["tie"] is not None:
                    item.tie = tie.Tie(event["tie"])
                pitch_name = item.pitch.nameWithOctave
            else:
                raise ValueError("Unsupported score event")
            expected.append({"pitch": pitch_name, "duration": str(length), "tie": event["tie"]})
        if total != bar_duration:
            return {"verdict": "UNKNOWN", "reason": "Visible measure does not fill declared meter"}
    reported = request["reported_events"]
    if not isinstance(reported, list) or len(reported) != len(expected):
        return {"verdict": "NOT_MET", "expected_events": expected}
    return {"verdict": "MET" if reported == expected else "NOT_MET", "expected_events": expected}


def _chemistry(request: dict[str, Any]) -> dict[str, Any]:
    """Sanitize a visual molecular graph and compare canonical, nonstereo SMILES."""
    from rdkit import Chem

    source = request["source"]
    atoms, bonds = source["atoms"], source["bonds"]
    if not 1 <= len(atoms) <= 128 or len(bonds) > 256:
        raise ValueError("Unsupported chemical graph size")
    allowed = {"C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "H"}
    molecule = Chem.RWMol()
    indices: dict[str, int] = {}
    for atom in atoms:
        element = atom["element"]
        if element not in allowed or atom["atom_id"] in indices:
            raise ValueError("Unsupported or repeated atom")
        rd_atom = Chem.Atom(element)
        rd_atom.SetFormalCharge(atom["charge"])
        rd_atom.SetIsAromatic(atom["aromatic"])
        indices[atom["atom_id"]] = molecule.AddAtom(rd_atom)
    bond_types = {
        "single": Chem.BondType.SINGLE,
        "double": Chem.BondType.DOUBLE,
        "triple": Chem.BondType.TRIPLE,
        "aromatic": Chem.BondType.AROMATIC,
    }
    for bond in bonds:
        if bond["a"] not in indices or bond["b"] not in indices:
            raise ValueError("Bond has missing endpoint")
        molecule.AddBond(indices[bond["a"]], indices[bond["b"]], bond_types[bond["order"]])
    expected_mol = molecule.GetMol()
    Chem.SanitizeMol(expected_mol)
    expected = Chem.MolToSmiles(expected_mol, isomericSmiles=False, canonical=True)
    reported = request["reported_smiles"]
    if (
        not isinstance(reported, str)
        or not 1 <= len(reported) <= 512
        or any(mark in reported for mark in ("@", "/", "\\", ">", "."))
    ):
        raise ValueError("Unsupported SMILES or stereochemical notation")
    observed_mol = Chem.MolFromSmiles(reported)
    if observed_mol is None:
        return {"verdict": "UNKNOWN", "reason": "Candidate SMILES is invalid"}
    observed = Chem.MolToSmiles(observed_mol, isomericSmiles=False, canonical=True)
    return {"verdict": "MET" if observed == expected else "NOT_MET", "expected_smiles": expected}


def main() -> None:
    """Validate one bounded request and emit a strict verdict envelope."""
    _limit_resources()
    try:
        raw = sys.stdin.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Request exceeds worker input limit")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise ValueError("Worker request must be an object")
        operation = request.get("operation")
        if operation == "geometry":
            response = _geometry(request)
        elif operation == "music":
            response = _music(request)
        elif operation == "chemistry":
            response = _chemistry(request)
        else:
            response = {"verdict": "UNKNOWN", "reason": "Unsupported worker operation"}
    except (KeyError, TypeError, ValueError) as exc:
        response = {"verdict": "UNKNOWN", "reason": str(exc)[:240]}
    sys.stdout.write(json.dumps(response, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
