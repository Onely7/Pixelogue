"""Deterministic Open Images V7 validation-sample acquisition."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx
from pydantic import Field

from pixelogue.config import StrictModel
from pixelogue.contracts import RightsRecord, SourcePurpose, SourceRecord
from pixelogue.errors import ExternalInputError, ShortfallError

ALLOWED_LICENSE_PREFIX = "https://creativecommons.org/licenses/by/"


class OpenImagesPinnedRecord(StrictModel):
    """Committed identity and provenance for one fixed validation image."""

    image_id: str
    split: Literal["validation"] = "validation"
    license_uri: str
    attribution: str
    landing_url: str
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class OpenImagesRow:
    """Private source metadata retained outside model-visible inputs."""

    image_id: str
    original_url: str
    landing_url: str
    license_uri: str
    author: str
    title: str
    thumbnail_url: str
    rotation: Literal[0, 90, 180, 270] | None

    @classmethod
    def from_csv(cls, row: dict[str, str]) -> OpenImagesRow | None:
        """Build an eligible validation record from an official metadata row."""
        if row.get("Subset") != "validation":
            return None
        license_uri = row.get("License", "")
        if not license_uri.startswith(ALLOWED_LICENSE_PREFIX):
            return None
        original_url = row.get("OriginalURL", "")
        thumbnail_url = row.get("Thumbnail300KURL", "")
        if not original_url and not thumbnail_url:
            return None
        rotation_raw = row.get("Rotation", "")
        rotation: Literal[0, 90, 180, 270] | None = None
        if rotation_raw == "0":
            rotation = 0
        elif rotation_raw == "90":
            rotation = 90
        elif rotation_raw == "180":
            rotation = 180
        elif rotation_raw == "270":
            rotation = 270
        return cls(
            image_id=row["ImageID"],
            original_url=original_url,
            landing_url=row.get("OriginalLandingURL", ""),
            license_uri=license_uri,
            author=row.get("Author", ""),
            title=row.get("Title", ""),
            thumbnail_url=thumbnail_url,
            rotation=rotation,
        )


def deterministic_sample(
    rows: Iterable[OpenImagesRow],
    count: int,
    seed: int,
) -> list[OpenImagesRow]:
    """Choose rows with the smallest stable hashes without depending on CSV order."""
    heap: list[tuple[int, str]] = []
    indexed: dict[str, OpenImagesRow] = {}
    for row in rows:
        rank = int.from_bytes(hashlib.sha256(f"{seed}:{row.image_id}".encode()).digest(), "big")
        entry = (-rank, row.image_id)
        indexed[row.image_id] = row
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [indexed[entry[1]] for entry in sorted(heap, key=lambda item: (-item[0], item[1]))]


class OpenImagesDownloader:
    """Download a bounded, reproducible evaluation-only sample."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        """Initialize with an optional testable HTTP client."""
        self._client = client or httpx.Client(follow_redirects=True, timeout=60)

    def fetch_metadata(self, url: str) -> list[OpenImagesRow]:
        """Read eligible rows from the official validation metadata CSV."""
        try:
            response = self._client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ExternalInputError("OPEN_IMAGES_METADATA_FAILED", str(error)) from error
        reader = csv.DictReader(response.text.splitlines())
        rows: list[OpenImagesRow] = []
        for raw in reader:
            if row := OpenImagesRow.from_csv(raw):
                rows.append(row)
        if not rows:
            raise ExternalInputError(
                "OPEN_IMAGES_METADATA_EMPTY", "No eligible validation images found"
            )
        return rows

    def download_sample(
        self,
        metadata_url: str,
        destination: Path,
        *,
        count: int = 20,
        seed: int = 20260915,
        pinned: tuple[OpenImagesPinnedRecord, ...] = (),
    ) -> tuple[list[SourceRecord], list[RightsRecord]]:
        """Download exactly `count` evaluation-only images and provenance records.

        More rows than requested are selected deterministically so unavailable image URLs can be
        skipped without random replacement.

        Raises:
            ExternalInputError: If fewer than the requested number can be downloaded.
        """
        metadata = self.fetch_metadata(metadata_url)
        pinned_by_id = {record.image_id: record for record in pinned}
        if pinned:
            if len(pinned_by_id) != count or len(pinned_by_id) != len(pinned):
                raise ExternalInputError(
                    "OPEN_IMAGES_PIN_COUNT",
                    "Pinned Open Images records must be unique and match the requested count",
                )
            metadata_by_id = {row.image_id: row for row in metadata}
            missing = [
                record.image_id for record in pinned if record.image_id not in metadata_by_id
            ]
            if missing:
                raise ExternalInputError("OPEN_IMAGES_PIN_MISSING", ", ".join(missing))
            candidates = [metadata_by_id[record.image_id] for record in pinned]
        else:
            candidates = deterministic_sample(metadata, count * 20, seed)
        image_dir = destination / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        sources: list[SourceRecord] = []
        rights_records: list[RightsRecord] = []
        private_rows: list[dict[str, object]] = []
        failures: list[dict[str, object]] = []
        for row in candidates:
            if len(sources) == count:
                break
            pinned_record = pinned_by_id.get(row.image_id)
            if pinned_record is not None and (
                row.license_uri != pinned_record.license_uri
                or row.landing_url != pinned_record.landing_url
                or (row.author or row.landing_url) != pinned_record.attribution
            ):
                failures.append(
                    {
                        "image_id": row.image_id,
                        "reason": "OPEN_IMAGES_PIN_PROVENANCE_MISMATCH",
                        "message": "Official metadata differs from the committed provenance",
                    }
                )
                continue
            downloaded = self._download_row(row)
            if isinstance(downloaded, ExternalInputError):
                failures.append(
                    {
                        "image_id": row.image_id,
                        "reason": downloaded.reason,
                        "message": str(downloaded),
                    }
                )
                continue
            image_bytes, media_type, url = downloaded
            downloaded_hash = hashlib.sha256(image_bytes).hexdigest()
            if pinned_record is not None and downloaded_hash != pinned_record.raw_sha256:
                failures.append(
                    {
                        "image_id": row.image_id,
                        "reason": "OPEN_IMAGES_PIN_HASH_MISMATCH",
                        "message": f"expected {pinned_record.raw_sha256}; received {downloaded_hash}",
                    }
                )
                continue
            suffix = _image_suffix(url, media_type)
            relative_path = Path("images") / f"{row.image_id}{suffix}"
            _atomic_write(image_dir / relative_path.name, image_bytes)
            rights_id = f"open-images-v7:{row.image_id}"
            sources.append(
                SourceRecord(
                    source_id=rights_id,
                    image_path=relative_path.as_posix(),
                    source_group_ids=(rights_id,),
                    rights_record_id=rights_id,
                    purpose=SourcePurpose.EVALUATION,
                    dataset="Open Images V7",
                    dataset_split="validation",
                    dataset_image_id=row.image_id,
                    rotation_degrees=row.rotation or 0,
                )
            )
            rights_records.append(
                RightsRecord.model_validate_json(
                    json.dumps(
                        {
                            "rights_record_id": rights_id,
                            "license_uri": row.license_uri,
                            "attribution": row.author or row.landing_url,
                            "processing_allowed": True,
                            "qa_redistribution_allowed": False,
                            "image_redistribution_allowed": False,
                            "training_allowed": False,
                            "valid_from": "2021-12-01T00:00:00Z",
                            "valid_until": None,
                        }
                    )
                )
            )
            private_rows.append(
                {
                    "image_id": row.image_id,
                    "original_url": row.original_url,
                    "landing_url": row.landing_url,
                    "license": row.license_uri,
                    "author": row.author,
                    "title": row.title,
                    "rotation": row.rotation,
                    "download_url": url,
                    "sha256": downloaded_hash,
                }
            )
        _write_jsonl(destination / "open_images_private_metadata.jsonl", private_rows)
        _write_jsonl(destination / "open_images_failures.jsonl", failures)
        if len(sources) != count:
            raise ShortfallError(
                "OPEN_IMAGES_SHORTFALL",
                f"Downloaded {len(sources)} of {count} requested validation images",
            )
        return sources, rights_records

    def _download_row(self, row: OpenImagesRow) -> tuple[bytes, str, str] | ExternalInputError:
        urls = (
            f"https://open-images-dataset.s3.amazonaws.com/validation/{row.image_id}.jpg",
            row.thumbnail_url,
            row.original_url,
        )
        errors: list[str] = []
        for url in dict.fromkeys(candidate for candidate in urls if candidate):
            try:
                image_bytes, media_type = self._download_image(url)
                return image_bytes, media_type, url
            except ExternalInputError as error:
                errors.append(f"{url}: {error}")
        return ExternalInputError("OPEN_IMAGES_DOWNLOAD_FAILED", " | ".join(errors))

    def _download_image(self, url: str) -> tuple[bytes, str]:
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").split(";", 1)[0]
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 20 * 1024 * 1024:
                        raise ExternalInputError("OPEN_IMAGES_TOO_LARGE", "Image exceeds 20 MiB")
                    chunks.append(chunk)
        except httpx.HTTPError as error:
            raise ExternalInputError("OPEN_IMAGES_DOWNLOAD_FAILED", str(error)) from error
        return b"".join(chunks), content_type


def _image_suffix(url: str, media_type: str) -> str:
    suffixes = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    if media_type in suffixes:
        return suffixes[media_type]
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return suffix
    raise ExternalInputError(
        "OPEN_IMAGES_CONTENT_TYPE", "Downloaded resource is not a supported image"
    )


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write(path, payload.encode())
