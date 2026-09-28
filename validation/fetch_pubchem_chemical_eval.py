#!/usr/bin/env python3
"""Restore pinned, evaluation-only PubChem 2D chemical depictions and labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/pubchem_2d_eval_manifest.jsonl"
HEADERS = {"User-Agent": "Pixelogue evaluation-only research test/0.1"}
ALLOWED_ELEMENTS = {"C", "N", "O", "F", "Cl", "Br"}


def _sha256(value: bytes) -> str:
    """Return the complete SHA-256 content identity."""
    return hashlib.sha256(value).hexdigest()


def _request(url: str) -> bytes:
    """Fetch one bounded official response, retrying transient transport errors."""
    for attempt in range(4):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=HEADERS), timeout=60
            ) as response:
                body = response.read(20_000_001)
                if response.status != 200 or len(body) > 20_000_000:
                    raise ValueError(f"Invalid PubChem response: {response.status}")
                return body
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def _canonical_smiles(raw: str) -> str:
    """Check the frozen supported notation before deriving a private label."""
    if any(char in raw for char in ("@", "/", "\\", ".")):
        raise ValueError("PubChem structure has unsupported stereo or components")
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None or not 3 <= molecule.GetNumHeavyAtoms() <= 5:
        raise ValueError("PubChem structure has unsupported size")
    if molecule.GetRingInfo().NumRings() or any(
        atom.GetSymbol() not in ALLOWED_ELEMENTS
        or atom.GetFormalCharge()
        or atom.GetIsotope()
        or atom.GetIsAromatic()
        or atom.GetNumRadicalElectrons()
        for atom in molecule.GetAtoms()
    ):
        raise ValueError("PubChem structure is outside the frozen chemical domain")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _rows() -> list[dict[str, Any]]:
    """Load the committed identities and reject edited or duplicate split records."""
    rows = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
    if len(rows) != 89 or len({row["cid"] for row in rows}) != 89:
        raise ValueError("Expected 89 distinct PubChem CIDs")
    if len({row["source_group_id"] for row in rows}) != 89:
        raise ValueError("Molecular structures must be independent image groups")
    if [row["split"] for row in rows] != ["development"] * 10 + ["confirmation"] * 79:
        raise ValueError("Development and confirmation splits changed")
    for row in rows:
        cid = row["cid"]
        if not isinstance(cid, int) or cid <= 0:
            raise ValueError("Invalid CID")
        if row["source_id"] != f"pubchem-cid:{cid}":
            raise ValueError("PubChem source ID changed")
        if row["source_page"] != f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}":
            raise ValueError("PubChem source page changed")
        if row["image_url"] != (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=800x800"
        ):
            raise ValueError("PubChem image specification changed")
    return rows


def _property_labels(rows: list[dict[str, Any]]) -> dict[int, str]:
    """Verify current official SMILES against committed label hashes."""
    properties: dict[int, str] = {}
    for offset in range(0, len(rows), 100):
        cids = ",".join(str(row["cid"]) for row in rows[offset : offset + 100])
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + cids
            + "/property/SMILES/JSON"
        )
        payload = json.loads(_request(url))
        for item in payload["PropertyTable"]["Properties"]:
            properties[int(item["CID"])] = item["SMILES"]
        time.sleep(0.35)
    if set(properties) != {row["cid"] for row in rows}:
        raise ValueError("PubChem property results differ from pinned CIDs")
    labels = {}
    for row in rows:
        canonical = _canonical_smiles(properties[row["cid"]])
        if _sha256(canonical.encode()) != row["canonical_smiles_sha256"]:
            raise ValueError(f"Pinned PubChem structure changed: CID {row['cid']}")
        if row["source_group_id"] != "pubchem-structure:" + _sha256(canonical.encode()):
            raise ValueError("PubChem structure group changed")
        labels[row["cid"]] = canonical
    return labels


def main() -> None:
    """Verify pixels and labels, then restore local-only ingestion manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=ROOT / "data/pubchem-2d-simple-eval")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.error")
    rows = _rows()
    labels = _property_labels(rows)
    images: dict[int, bytes] = {}
    for index, row in enumerate(rows):
        raw = _request(row["image_url"])
        if _sha256(raw) != row["raw_sha256"]:
            raise ValueError(f"Pinned PubChem image changed: CID {row['cid']}")
        with Image.open(io.BytesIO(raw)) as image:
            if (
                image.format != "PNG"
                or image.size != (row["width"], row["height"])
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(f"Pinned PubChem image geometry changed: CID {row['cid']}")
        images[row["cid"]] = raw
        time.sleep(0.35)
        if index % 10 == 9:
            print(f"verified {index + 1}/89", flush=True)
    output = args.destination
    (output / "images").mkdir(parents=True, exist_ok=True)
    sources, rights, private_labels = [], [], []
    for row in rows:
        cid = row["cid"]
        image_path = output / "images" / f"{cid}.png"
        if image_path.exists() and _sha256(image_path.read_bytes()) != row["raw_sha256"]:
            raise ValueError(f"Existing local PubChem image differs: CID {cid}")
        image_path.write_bytes(images[cid])
        sources.append(
            {
                "source_id": row["source_id"],
                "image_path": f"images/{cid}.png",
                "source_group_ids": [row["source_group_id"]],
                "rights_record_id": row["source_id"],
                "purpose": "evaluation",
                "dataset": "PubChem 2D structure depiction",
                "dataset_split": row["split"],
                "dataset_image_id": str(cid),
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": row["source_id"],
                "license_uri": row["rights_basis"],
                "attribution": f"PubChem CID {cid} 2D structure depiction",
                "processing_allowed": True,
                "qa_redistribution_allowed": False,
                "image_redistribution_allowed": False,
                "training_allowed": False,
                "valid_from": "2026-09-01T00:00:00Z",
                "valid_until": None,
            }
        )
        private_labels.append(
            {
                "source_id": row["source_id"],
                "cid": cid,
                "canonical_smiles": labels[cid],
                "raw_sha256": row["raw_sha256"],
                "source_group_id": row["source_group_id"],
                "split": row["split"],
                "label_provenance": row["source_page"] + "#section=2D-Structure",
            }
        )
    for name, records in (
        ("sources.jsonl", sources),
        ("rights.jsonl", rights),
        ("private_labels.jsonl", private_labels),
    ):
        (output / name).write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
        )
    print(json.dumps({"restored": len(rows), "split": {"development": 10, "confirmation": 79}}))


if __name__ == "__main__":
    main()
