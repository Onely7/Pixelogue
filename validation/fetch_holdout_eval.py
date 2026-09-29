#!/usr/bin/env python3
"""Restore six pinned, evaluation-only Commons images without storing bytes in Git."""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
from pathlib import Path

from PIL import Image

if __package__:
    from .fetch_diverse_eval import PUBLIC_DOMAIN, _page, _request
else:
    from fetch_diverse_eval import PUBLIC_DOMAIN, _page, _request

ROOT = Path(__file__).resolve().parents[1]
PINNED = ROOT / "validation/holdout_web_eval_manifest.jsonl"
DEVELOPMENT = ROOT / "validation/diverse_web_eval_manifest.jsonl"
OUTPUT = ROOT / "data/holdout-web-eval"
FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


def _plain(value: object) -> str:
    """Remove Commons metadata markup before comparing pinned attribution."""
    return html.unescape(re.sub(r"<[^>]+>", "", str(value))).strip()


def _verified_bytes(row: dict[str, object]) -> tuple[bytes, str]:
    """Require the same Commons page, licence, thumbnail bytes and dimensions."""
    page = _page(str(row["title"]))
    if page["pageid"] != row["commons_page_id"]:
        raise RuntimeError(f"Commons page changed: {row['title']}")
    info = page["imageinfo"][0]
    metadata = {
        key: _plain(value.get("value", "")) for key, value in info.get("extmetadata", {}).items()
    }
    licence = metadata.get("LicenseUrl")
    if not licence and metadata.get("LicenseShortName") == "Public domain":
        licence = PUBLIC_DOMAIN
    if licence and licence.startswith("http://creativecommons.org/"):
        licence = "https://" + licence.removeprefix("http://")
    if licence != row["license_uri"]:
        raise RuntimeError(f"Commons licence changed: {row['title']}")
    content = _request(str(row["thumbnail_url"]))
    if hashlib.sha256(content).hexdigest() != row["raw_sha256"]:
        raise RuntimeError(f"Commons thumbnail changed: {row['title']}")
    with Image.open(io.BytesIO(content)) as image:
        if (
            image.format not in FORMATS
            or image.size != (row["width"], row["height"])
            or getattr(image, "n_frames", 1) != 1
            or min(image.size) < 128
        ):
            raise RuntimeError(f"Commons image format changed: {row['title']}")
        extension = FORMATS[image.format]
    return content, extension


def main() -> None:
    """Verify identities, restore raster bytes and write ingest-only manifests."""
    pinned = [json.loads(line) for line in PINNED.read_text().splitlines()]
    development = {
        json.loads(line)["commons_page_id"] for line in DEVELOPMENT.read_text().splitlines()
    }
    ids = [row["commons_page_id"] for row in pinned]
    if len(pinned) != 6 or len(ids) != len(set(ids)) or development.intersection(ids):
        raise RuntimeError("Holdout groups overlap the development set or are incomplete")
    sources: list[dict[str, object]] = []
    rights: list[dict[str, object]] = []
    for row in pinned:
        content, extension = _verified_bytes(row)
        number = row["category_number"]
        path = Path("images") / f"{number:02d}.{extension}"
        destination = OUTPUT / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            not destination.is_file()
            or hashlib.sha256(destination.read_bytes()).hexdigest() != (row["raw_sha256"])
        ):
            destination.write_bytes(content)
        source_id = f"commons-holdout:{row['commons_page_id']}"
        sources.append(
            {
                "source_id": source_id,
                "image_path": path.as_posix(),
                "source_group_ids": [f"commons:{row['commons_page_id']}"],
                "rights_record_id": source_id,
                "purpose": "evaluation",
                "dataset": "Wikimedia Commons holdout evaluation",
                "dataset_split": "web-holdout-2026-09",
                "dataset_image_id": str(row["commons_page_id"]),
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": source_id,
                "license_uri": row["license_uri"],
                "attribution": row["artist"],
                "processing_allowed": True,
                "qa_redistribution_allowed": False,
                "image_redistribution_allowed": False,
                "training_allowed": False,
                "valid_from": "2026-09-01T00:00:00Z",
                "valid_until": None,
            }
        )
        print(f"{number}: {row['category']} {source_id}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name, records in (("sources.jsonl", sources), ("rights.jsonl", rights)):
        (OUTPUT / name).write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        )


if __name__ == "__main__":
    main()
