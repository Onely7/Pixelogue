#!/usr/bin/env python3
"""Restore three pinned, evaluation-only specialist diagrams from Commons."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

from fetch_diverse_eval import _request
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/specialist_web_eval_manifest.jsonl"


def main() -> None:
    """Download pinned rasters and write source and rights manifests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, object]]] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        kind = row["category"]
        group = "specialist-web-eval" if kind == "music" else "specialist-extra-eval"
        output = args.output_root / group
        raw = _request(row["thumbnail_url"])
        if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
            raise ValueError(f"Pinned Commons raster changed: {row['title']}")
        with Image.open(io.BytesIO(raw)) as image:
            if (
                image.format != "PNG"
                or image.size != (row["width"], row["height"])
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(f"Pinned image geometry changed: {row['title']}")
        (output / "images").mkdir(parents=True, exist_ok=True)
        (output / "images" / f"{kind}.png").write_bytes(raw)
        source_id = f"commons-specialist:{row['commons_page_id']}"
        source = {
            "source_id": source_id,
            "image_path": f"images/{kind}.png",
            "source_group_ids": [f"commons:{row['commons_page_id']}"],
            "rights_record_id": source_id,
            "purpose": "evaluation",
            "dataset": "Wikimedia Commons specialist evaluation",
            "dataset_split": "web-eval-2026-09",
            "dataset_image_id": str(row["commons_page_id"]),
            "rotation_degrees": 0,
        }
        rights = {
            "rights_record_id": source_id,
            "license_uri": row["license_uri"],
            "attribution": row.get("artist") or row["title"],
            "processing_allowed": True,
            "qa_redistribution_allowed": False,
            "image_redistribution_allowed": False,
            "training_allowed": False,
            "valid_from": "2026-09-01T00:00:00Z",
            "valid_until": None,
        }
        grouped.setdefault(group, []).append({"source": source, "rights": rights})
    for group, entries in grouped.items():
        output = args.output_root / group
        for filename, key in (("sources.jsonl", "source"), ("rights.jsonl", "rights")):
            (output / filename).write_text(
                "".join(json.dumps(entry[key], sort_keys=True) + "\n" for entry in entries),
                encoding="utf-8",
            )
    print(json.dumps({group: len(entries) for group, entries in grouped.items()}))


if __name__ == "__main__":
    main()
