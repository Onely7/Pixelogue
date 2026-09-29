#!/usr/bin/env python3
"""Restore pinned ScreenSpot screenshots and private click target annotations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/screenspot_ui_eval_manifest.jsonl"
DESTINATION = ROOT / "data/screenspot-ui-eval"
API = "https://datasets-server.huggingface.co/rows?dataset=bevaya/ScreenSpot&config=default&split=test"
HEADERS = {"User-Agent": "Pixelogue evaluation-only research test/0.1"}


def _hash(value: bytes) -> str:
    """Return the SHA-256 identity of supplied bytes."""
    return hashlib.sha256(value).hexdigest()


def _request(url: str, limit: int = 20_000_000) -> bytes:
    """Fetch one bounded benchmark response."""
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=HEADERS), timeout=90
    ) as response:
        data = response.read(limit + 1)
        if response.status != 200 or len(data) > limit:
            raise ValueError("ScreenSpot response is unavailable or oversized")
        return data


def _rows() -> dict[int, dict[str, Any]]:
    """Read the public dataset records without exposing labels to model inputs."""
    rows: dict[int, dict[str, Any]] = {}
    for offset in range(0, 1272, 100):
        payload = json.loads(_request(f"{API}&offset={offset}&length=100", 8_000_000))
        if payload["num_rows_total"] != 1272:
            raise ValueError("ScreenSpot row count changed")
        for item in payload["rows"]:
            rows[item["row_idx"]] = item["row"]
    if len(rows) != 1272:
        raise ValueError("ScreenSpot rows are incomplete")
    return rows


def _annotation(row: dict[str, Any]) -> tuple[str, tuple[float, float, float, float]]:
    """Validate the frozen public instruction and independent target box."""
    instruction = row["instruction"]
    box = tuple(row["bbox"])
    if (
        row["data_type"] != "text"
        or not isinstance(instruction, str)
        or not instruction.strip()
        or len(box) != 4
        or not all(isinstance(value, (int, float)) for value in box)
        or not 0 <= box[0] < box[2] <= 1
        or not 0 <= box[1] < box[3] <= 1
    ):
        raise ValueError("ScreenSpot text target is outside the click domain")
    return instruction, box


def _image(row: dict[str, Any]) -> tuple[bytes, int, int, str, str, str]:
    """Fetch and validate one static screenshot and its visual identity."""
    raw = _request(row["image"]["src"])
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError("ScreenSpot image format is unsupported")
        width, height = image.size
        if min(width, height) < 128 or width * height > 40_000_000:
            raise ValueError("ScreenSpot image size is outside ingestion limits")
        pixel_hash = _hash(image.convert("RGB").tobytes())
        sampled = list(image.convert("L").resize((9, 8)).getdata())
        dhash = sum(
            (sampled[y * 9 + x] > sampled[y * 9 + x + 1]) << (y * 8 + x)
            for y in range(8)
            for x in range(8)
        )
        image_format = image.format
    if (width, height) != (row["image"]["width"], row["image"]["height"]):
        raise ValueError("ScreenSpot raster dimensions differ from metadata")
    return raw, width, height, pixel_hash, image_format, f"{dhash:016x}"


def _record(index: int, row: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    """Build one immutable provenance record from source pixels and annotation."""
    instruction, box = _annotation(row)
    raw, width, height, pixel_hash, image_format, dhash = _image(row)
    name = row["file_name"]
    if Path(name).name != name or Path(name).suffix.lower() not in {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    }:
        raise ValueError("ScreenSpot filename is not a safe image basename")
    return (
        {
            "row_idx": index,
            "file_name": name,
            "raw_sha256": _hash(raw),
            "pixel_sha256": pixel_hash,
            "dhash64": dhash,
            "annotation_sha256": _hash(
                json.dumps([instruction, box], ensure_ascii=False, separators=(",", ":")).encode()
            ),
            "width": width,
            "height": height,
            "image_format": image_format,
            "source_group_id": "screenspot-image:" + pixel_hash,
        },
        raw,
    )


def main() -> None:
    """Freeze a disjoint sample once, or restore and verify its committed identities."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    rows = _rows()
    if args.freeze:
        if MANIFEST.exists():
            raise ValueError("Refusing to replace a pinned ScreenSpot manifest")
        eligible = [index for index, row in rows.items() if row["data_type"] == "text"]
        eligible.sort(key=lambda index: _hash(f"screenspot-ui-v1:{index}".encode()))
        selected: list[tuple[dict[str, Any], bytes]] = []
        pixels: set[str] = set()
        hashes: list[int] = []
        for start in range(0, len(eligible), 16):
            with ThreadPoolExecutor(max_workers=8) as pool:
                downloaded = pool.map(
                    lambda index: _record(index, rows[index]), eligible[start : start + 16]
                )
                for record, raw in downloaded:
                    candidate_hash = int(record["dhash64"], 16)
                    if record["pixel_sha256"] in pixels or any(
                        (candidate_hash ^ previous).bit_count() <= 5 for previous in hashes
                    ):
                        continue
                    pixels.add(record["pixel_sha256"])
                    hashes.append(candidate_hash)
                    selected.append((record, raw))
            print(f"verified {len(selected)} distinct ScreenSpot screenshots", flush=True)
            if len(selected) >= 89:
                selected = selected[:89]
                break
        if len(selected) != 89:
            raise ValueError("ScreenSpot has fewer than 89 distinct supported screenshots")
        manifest = []
        for position, (record, _) in enumerate(selected):
            record["split"] = "development" if position < 10 else "confirmation"
            manifest.append(record)
        MANIFEST.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest))
        downloaded_by_index = {item["row_idx"]: raw for item, raw in selected}
    else:
        manifest = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
        if (
            len(manifest) != 89
            or [item["split"] for item in manifest] != ["development"] * 10 + ["confirmation"] * 79
        ):
            raise ValueError("ScreenSpot split or sample size changed")
        if len({item["pixel_sha256"] for item in manifest}) != 89:
            raise ValueError("ScreenSpot image groups are not independent")
        hashes = [int(item["dhash64"], 16) for item in manifest]
        if any(
            (value ^ other).bit_count() <= 5
            for position, value in enumerate(hashes)
            for other in hashes[position + 1 :]
        ):
            raise ValueError("ScreenSpot near-duplicate screenshots share a calibration sample")
        downloaded_by_index = {}
        for item in manifest:
            current, raw = _record(item["row_idx"], rows[item["row_idx"]])
            current["split"] = item["split"]
            if current != item:
                raise ValueError(f"Pinned ScreenSpot row changed: {item['row_idx']}")
            downloaded_by_index[item["row_idx"]] = raw
    output = args.destination
    (output / "images").mkdir(parents=True, exist_ok=True)
    sources, rights, private_labels = [], [], []
    for item in manifest:
        index = item["row_idx"]
        instruction, box = _annotation(rows[index])
        extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}[item["image_format"]]
        path = output / "images" / f"{index}{extension}"
        if path.exists() and _hash(path.read_bytes()) != item["raw_sha256"]:
            raise ValueError(f"Existing ScreenSpot image changed: row {index}")
        path.write_bytes(downloaded_by_index[index])
        source_id = f"screenspot:{index}"
        sources.append(
            {
                "source_id": source_id,
                "image_path": f"images/{index}{extension}",
                "source_group_ids": [item["source_group_id"]],
                "rights_record_id": source_id,
                "purpose": "evaluation",
                "dataset": "ScreenSpot",
                "dataset_split": item["split"],
                "dataset_image_id": str(index),
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": source_id,
                "license_uri": "https://www.apache.org/licenses/LICENSE-2.0",
                "attribution": f"ScreenSpot test screenshot, row {index}",
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
                "source_id": source_id,
                "instruction": instruction,
                "bbox": box,
                "width": item["width"],
                "height": item["height"],
                "split": item["split"],
                "label_provenance": f"https://huggingface.co/datasets/bevaya/ScreenSpot/viewer/default/test?row={index}",
            }
        )
    for name, records in (
        ("sources.jsonl", sources),
        ("rights.jsonl", rights),
        ("private_labels.jsonl", private_labels),
    ):
        (output / name).write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in records)
        )
    print(json.dumps({"restored": len(manifest), "development": 10, "confirmation": 79}))


if __name__ == "__main__":
    main()
