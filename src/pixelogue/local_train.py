"""Prepare bounded, reproducible training inputs from local CVDF images."""

from __future__ import annotations

import csv
import hashlib
import heapq
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from pydantic import HttpUrl

from pixelogue.contracts import ImageArtifact, RightsRecord, SourcePurpose, SourceRecord
from pixelogue.errors import ExternalInputError, ShortfallError
from pixelogue.io import read_jsonl, write_json, write_jsonl
from pixelogue.open_images import OpenImagesPinnedRecord
from pixelogue.operations import prepare_sources
from pixelogue.serialization import canonical_hash

_ID = re.compile(r"[0-9a-f]{16}")
_LICENSES = {f"https://creativecommons.org/licenses/by/{v}/" for v in ("2.0", "2.5", "3.0", "4.0")}
_COLUMNS = {"ImageID", "Subset", "License", "Author", "OriginalLandingURL", "Rotation"}
_ROTATIONS = {str(value) + suffix: value for value in (0, 90, 180, 270) for suffix in ("", ".0")}


def _hashed_lines(path: Path, update: Callable[[bytes], None]) -> Iterator[str]:
    """Stream UTF-8 CSV lines while recording the exact input bytes."""
    with path.open("rb") as stream:
        for line in stream:
            update(line)
            yield line.decode("utf-8")


def prepare_local_train(
    metadata: Path,
    image_root: Path,
    destination: Path,
    *,
    count: int,
    seed: int,
    validation_manifest: Path,
    evaluation_images: Sequence[Path] = (),
    progress: Callable[[str, int, int | None], None] | None = None,
) -> dict[str, object]:
    """Sample local train JPEGs, validate them, and write manifests and canonical views.

    Only explicit CC BY metadata with attribution and known rotation is eligible.
    The count bounds metadata candidates, not successful dialogues or decoded images.
    Missing/corrupt images are never downloaded or silently replaced after decoding.
    Reference manifests exclude exact evaluation copies; near copies require a separate
    joint SSCD check. Existing output directories are refused to protect reproducibility.
    An optional progress callback receives stage, processed count, and total (if known).

    Raises:
        ExternalInputError: If metadata is malformed or output already exists.
        ShortfallError: If fewer eligible local candidates than requested are available.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if destination.exists():
        raise ExternalInputError("DESTINATION_EXISTS", str(destination))
    if progress is not None:
        progress("load_references", 0, 1)
    root = image_root.resolve(strict=True)
    pinned = read_jsonl(validation_manifest, OpenImagesPinnedRecord)
    excluded_ids = {item.image_id for item in pinned}
    excluded_raw = {item.raw_sha256 for item in pinned}
    excluded_pixels: set[str] = set()
    references = []
    for path in evaluation_images:
        images = read_jsonl(path, ImageArtifact)
        if any(image.purpose is not SourcePurpose.EVALUATION for image in images):
            raise ExternalInputError("REFERENCE_NOT_EVALUATION", str(path))
        excluded_ids.update(image.source_id.rsplit(":", 1)[-1] for image in images)
        excluded_raw.update(image.raw_sha256 for image in images)
        excluded_pixels.update(image.canonical_pixel_sha256 for image in images)
        references.append(
            {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    if progress is not None:
        progress("load_references", 1, 1)
        progress("scan_metadata", 0, None)
    digest = hashlib.sha256()
    skipped: Counter[str] = Counter()
    heap: list[tuple[int, str, dict[str, str]]] = []
    selected_ids: set[str] = set()
    rows = 0
    reader = csv.DictReader(_hashed_lines(metadata, digest.update))
    if not reader.fieldnames or not _COLUMNS <= set(reader.fieldnames):
        raise ExternalInputError("METADATA_COLUMNS", "Missing required Open Images columns")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ExternalInputError("METADATA_COLUMNS", "Duplicate metadata columns")
    for row in reader:
        rows += 1
        if progress is not None and rows % 10000 == 0:
            progress("scan_metadata", rows, None)
        if None in row or any(value is None for value in row.values()):
            raise ExternalInputError("METADATA_ROW", f"Malformed CSV row {reader.line_num}")
        identifier = row["ImageID"]
        if not _ID.fullmatch(identifier):
            skipped["invalid_id"] += 1
            continue
        if identifier in selected_ids:
            continue
        if row["Subset"] != "train":
            skipped["not_train"] += 1
            continue
        if identifier in excluded_ids:
            skipped["evaluation_id"] += 1
            continue
        if row["License"] not in _LICENSES:
            skipped["unsupported_license"] += 1
            continue
        if not row["Author"].strip() or not row["OriginalLandingURL"].startswith("https://"):
            skipped["missing_attribution"] += 1
            continue
        if row["Rotation"] not in _ROTATIONS:
            skipped["unknown_rotation"] += 1
            continue
        rank = int.from_bytes(hashlib.sha256(f"{seed}:{identifier}".encode()).digest(), "big")
        # Only probe the filesystem when a row could enter the bounded sample.
        if len(heap) == count and rank >= -heap[0][0]:
            continue
        path = root / f"{identifier}.jpg"
        if path.is_symlink() or not path.is_file():
            skipped["missing_or_symlink_when_probed"] += 1
            continue
        entry = (-rank, identifier, row)
        if len(heap) == count:
            removed = heapq.heapreplace(heap, entry)
            selected_ids.remove(removed[1])
        else:
            heapq.heappush(heap, entry)
        selected_ids.add(identifier)
    if progress is not None:
        progress("scan_metadata", rows, rows)
    if len(heap) < count:
        raise ShortfallError(
            "LOCAL_TRAIN_SHORTFALL", f"Requested {count}, found {len(heap)} candidates"
        )
    # A second streaming pass detects duplicates of sampled IDs even if an earlier
    # occurrence was ineligible or had already left the heap. Memory stays bounded.
    verification_digest = hashlib.sha256()
    occurrences: Counter[str] = Counter()
    if progress is not None:
        progress("verify_metadata", 0, rows)
    verified_rows = 0
    for row in csv.DictReader(_hashed_lines(metadata, verification_digest.update)):
        verified_rows += 1
        if progress is not None and verified_rows % 10000 == 0:
            progress("verify_metadata", verified_rows, rows)
        if row["ImageID"] in selected_ids:
            occurrences[row["ImageID"]] += 1
    if verification_digest.digest() != digest.digest():
        raise ExternalInputError("METADATA_CHANGED", "Metadata changed during preparation")
    if any(value != 1 for value in occurrences.values()):
        raise ExternalInputError("DUPLICATE_SOURCE_ID", "Sampled IDs must have unique metadata")
    if progress is not None:
        progress("verify_metadata", verified_rows, rows)
        progress("build_records", 0, count)
    selected = [entry[2] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]
    sources: list[SourceRecord] = []
    rights: list[RightsRecord] = []
    timestamp = datetime.now(UTC)
    for row in selected:
        identifier = row["ImageID"]
        source_id = f"open-images-v7:{identifier}"
        sources.append(
            SourceRecord(
                source_id=source_id,
                image_path=f"{identifier}.jpg",
                rights_record_id=source_id,
                source_group_ids=(source_id,),
                purpose=SourcePurpose.TRAINING,
                dataset="Open Images V7",
                dataset_split="train",
                dataset_image_id=identifier,
                rotation_degrees=cast(Literal[0, 90, 180, 270], _ROTATIONS[row["Rotation"]]),
            )
        )
        rights.append(
            RightsRecord(
                rights_record_id=source_id,
                license_uri=HttpUrl(row["License"]),
                attribution=f"{row['Author']} | {row['OriginalLandingURL']}",
                processing_allowed=True,
                training_allowed=True,
                qa_redistribution_allowed=True,
                image_redistribution_allowed=False,
                valid_from=timestamp,
            )
        )
    if progress is not None:
        progress("build_records", count, count)
    destination.mkdir(parents=True)
    write_jsonl(destination / "private-metadata.jsonl", selected)
    write_json(destination / "selected-ids.json", [row["ImageID"] for row in selected])
    # All calls are CPU-only; no model payload contains the private metadata below.
    prepared = prepare_sources(
        sources, rights, root, destination, seed=seed, at=timestamp, progress=progress
    )
    if progress is not None:
        progress("write_outputs", 0, 1)
    accepted = []
    failures = [item.model_dump(mode="json") for item in prepared.failures]
    for image in prepared.images:
        if image.raw_sha256 in excluded_raw or image.canonical_pixel_sha256 in excluded_pixels:
            failures.append(
                {
                    "source_id": image.source_id,
                    "reason": "EVALUATION_EXACT_COPY",
                    "message": "Matches a supplied evaluation reference",
                }
            )
        else:
            accepted.append(image)
    accepted_ids = {image.source_id for image in accepted}
    write_jsonl(destination / "sources.jsonl", [s for s in sources if s.source_id in accepted_ids])
    write_jsonl(
        destination / "rights.jsonl", [r for r in rights if r.rights_record_id in accepted_ids]
    )
    write_jsonl(destination / "images.jsonl", accepted)
    write_jsonl(destination / "failures.jsonl", failures)
    report: dict[str, object] = {
        "metadata_sha256": digest.hexdigest(),
        "metadata_rows": rows,
        "metadata_path": str(metadata),
        "image_root": str(root),
        "seed": seed,
        "requested_candidates": count,
        "accepted": len(accepted),
        "rejected": len(failures),
        "skipped_metadata": dict(skipped),
        "evaluation_references": references,
        "validation_manifest_sha256": hashlib.sha256(validation_manifest.read_bytes()).hexdigest(),
        "near_duplicate_check": "NOT_PERFORMED",
        "rights_policy": "cc-by-metadata-v1; retain attribution; no image redistribution",
        "prepared_at": timestamp.isoformat(),
        "images_hash": canonical_hash([image.model_dump(mode="json") for image in accepted]),
        "splits": {image.image_id: prepared.splits[image.image_id] for image in accepted},
    }
    write_json(destination / "manifest.json", report)
    if progress is not None:
        progress("write_outputs", 1, 1)
    return report
