#!/usr/bin/env python3
"""Freeze a disjoint PubChem 2D confirmation cohort before model evaluation."""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

from fetch_pubchem_chemical_eval import _canonical_smiles, _request
from PIL import Image
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data/pubchem-2d-v2-eval"
MANIFEST = ROOT / "validation/pubchem_2d_eval_v2_manifest.jsonl"
SELECTION = ROOT / "validation/pubchem_2d_eval_v2_selection.json"
EARLIER = ROOT / "validation/pubchem_2d_eval_manifest.jsonl"
SEED = "pixelogue-pubchem-2d-independent-confirmation-20260930"
QUERY = "3:5[hac] AND 1:10000[cid]"
TOTAL = 90
POSITIVE = 30
RIGHTS_URL = "https://pubchem.ncbi.nlm.nih.gov/citations.html"


def _digest(value: bytes) -> str:
    """Return a stable content identity."""
    return hashlib.sha256(value).hexdigest()


def _write_frozen(path: Path, value: bytes) -> None:
    """Write once, or reject changes to a previously frozen identity."""
    if path.exists():
        if path.read_bytes() != value:
            raise ValueError(f"Frozen PubChem identity changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _selection_pool() -> tuple[list[tuple[str, str, int]], str, int]:
    """Select unused structures from the same documented bounded PubChem query."""
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        + urllib.parse.urlencode(
            {"db": "pccompound", "term": QUERY, "retmax": "1000", "retmode": "json"}
        )
    )
    search_bytes = _request(search_url)
    result = json.loads(search_bytes)["esearchresult"]
    ids = result["idlist"]
    if len(ids) != int(result["count"]) or len(ids) > 1000:
        raise ValueError("PubChem search result was truncated")
    properties: list[dict[str, Any]] = []
    for start in range(0, len(ids), 100):
        url = (
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
            + ",".join(ids[start : start + 100])
            + "/property/SMILES/JSON"
        )
        properties.extend(json.loads(_request(url))["PropertyTable"]["Properties"])
        time.sleep(0.35)
    if {int(item["CID"]) for item in properties} != {int(cid) for cid in ids}:
        raise ValueError("PubChem properties do not cover the frozen search result")
    existing_groups = {
        row["source_group_id"] for row in map(json.loads, EARLIER.read_text().splitlines())
    }
    candidates: dict[str, int] = {}
    for entry in properties:
        try:
            canonical = _canonical_smiles(entry["SMILES"])
        except ValueError:
            continue
        group = "pubchem-structure:" + _digest(canonical.encode())
        if group in existing_groups:
            continue
        cid = int(entry["CID"])
        if canonical not in candidates or cid < candidates[canonical]:
            candidates[canonical] = cid
    ranked = sorted(
        (_digest(f"{SEED}:{canonical}".encode()), canonical, cid)
        for canonical, cid in candidates.items()
    )
    if len(ranked) < TOTAL:
        raise ValueError(f"Only {len(ranked)} independent supported structures remain")
    return ranked[:TOTAL], _digest(search_bytes), len(ranked)


def main() -> None:
    """Pin independent image hashes, labels and source records before GPU review."""
    RDLogger.DisableLog("rdApp.error")
    selected, search_hash, pool_size = _selection_pool()
    manifest: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    rights: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for index, (_, canonical, cid) in enumerate(selected):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size=800x800"
        raw = _request(url)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.size != (800, 800):
                raise ValueError(f"Unexpected PubChem depiction: CID {cid}")
        source_id = f"pubchem-cid:{cid}"
        group = "pubchem-structure:" + _digest(canonical.encode())
        raw_hash = _digest(raw)
        filename = f"images/{cid}.png"
        _write_frozen(DESTINATION / filename, raw)
        manifest.append(
            {
                "canonical_smiles_sha256": _digest(canonical.encode()),
                "cid": cid,
                "height": 800,
                "image_url": url,
                "raw_sha256": raw_hash,
                "rights_basis": RIGHTS_URL,
                "source_group_id": group,
                "source_id": source_id,
                "source_page": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
                "split": "confirmation",
                "width": 800,
            }
        )
        sources.append(
            {
                "source_id": source_id,
                "image_path": filename,
                "source_group_ids": [group],
                "rights_record_id": source_id,
                "purpose": "evaluation",
                "dataset": "PubChem 2D structure depiction",
                "dataset_split": "confirmation-v2",
                "dataset_image_id": str(cid),
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": source_id,
                "license_uri": RIGHTS_URL,
                "attribution": f"PubChem CID {cid} 2D structure depiction",
                "processing_allowed": True,
                "qa_redistribution_allowed": False,
                "image_redistribution_allowed": False,
                "training_allowed": False,
                "valid_from": "2026-09-01T00:00:00Z",
                "valid_until": None,
            }
        )
        labels.append(
            {
                "source_id": source_id,
                "cid": cid,
                "canonical_smiles": canonical,
                "raw_sha256": raw_hash,
                "source_group_id": group,
                "split": "confirmation",
                "label_provenance": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=2D-Structure",
            }
        )
        time.sleep(0.35)
        if (index + 1) % 10 == 0:
            print(f"verified {index + 1}/{TOTAL} depictions", flush=True)
    for name, records in (
        ("sources.jsonl", sources),
        ("rights.jsonl", rights),
        ("private_labels.jsonl", labels),
    ):
        _write_frozen(
            DESTINATION / name,
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode(),
        )
    _write_frozen(
        MANIFEST,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest).encode(),
    )
    _write_frozen(
        SELECTION,
        (
            json.dumps(
                {
                    "seed": SEED,
                    "query": QUERY,
                    "search_sha256": search_hash,
                    "prior_manifest_sha256": _digest(EARLIER.read_bytes()),
                    "eligible_unused_structures": pool_size,
                    "selected": TOTAL,
                    "confirmation_positive": POSITIVE,
                    "confirmation_negative": TOTAL - POSITIVE,
                    "selection": "SHA256(seed:canonical_smiles), first 90 after excluding v1 structures",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    print(json.dumps({"selected": TOTAL, "eligible_unused_structures": pool_size}))


if __name__ == "__main__":
    main()
