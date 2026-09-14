"""Image validation, canonicalization, and visual grouping."""

from __future__ import annotations

import hashlib
import io
import math
import os
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from pixelogue.contracts import (
    ImageArtifact,
    ImageView,
    RightsRecord,
    SourceRecord,
)
from pixelogue.errors import ExternalInputError
from pixelogue.serialization import canonical_hash

MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_DECODED_PIXELS = 40_000_000
MIN_SHORT_EDGE = 128
MAX_INFERENCE_EDGE = 2048
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


def _safe_image_path(image_root: Path, relative_path: str) -> Path:
    root = image_root.resolve(strict=True)
    candidate = (root / relative_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ExternalInputError(
            "IMAGE_PATH_ESCAPE", f"Image leaves allowed root: {relative_path}"
        ) from error
    if not candidate.is_file():
        raise ExternalInputError("IMAGE_NOT_FILE", f"Image is not a regular file: {relative_path}")
    return candidate


def _rgb_image(image: Image.Image, rotation_degrees: int = 0) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    if rotation_degrees:
        oriented = oriented.rotate(rotation_degrees, expand=True)
    icc = oriented.info.get("icc_profile")
    if icc:
        try:
            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            target_profile = ImageCms.createProfile("sRGB")
            converted = ImageCms.profileToProfile(oriented, source_profile, target_profile)
            if converted is None:
                raise ExternalInputError(
                    "ICC_CONVERSION_FAILED", "ICC conversion returned no image"
                )
            oriented = converted
        except (OSError, ValueError) as error:
            raise ExternalInputError("ICC_CONVERSION_FAILED", str(error)) from error
    if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return oriented.convert("RGB")


def canonical_pixel_hash(image: Image.Image) -> str:
    """Hash an RGB8 image using the documented Pixelogue byte contract.

    Raises:
        ValueError: If the supplied image is not RGB8.
    """
    if image.mode != "RGB":
        raise ValueError("canonical pixel hashing requires an RGB image")
    payload = b"RGB8-v1\0" + struct.pack(">QQ", image.width, image.height) + image.tobytes()
    return hashlib.sha256(payload).hexdigest()


def canonicalize_image(
    source: SourceRecord,
    rights: RightsRecord,
    image_root: Path,
    artifact_root: Path,
    *,
    at: datetime | None = None,
) -> ImageArtifact:
    """Validate a source and write its deterministic inference view.

    Args:
        source: Source record whose path is relative to `image_root`.
        rights: Rights record referenced by the source.
        image_root: Only directory from which image bytes may be read.
        artifact_root: Directory where canonical views are stored.
        at: Time at which the supplied rights are evaluated.

    Raises:
        ExternalInputError: If rights, paths, image format, or dimensions are invalid.
    """
    if source.rights_record_id != rights.rights_record_id:
        raise ExternalInputError(
            "RIGHTS_REFERENCE_MISMATCH", "Source references another rights record"
        )
    now = at or datetime.now(UTC)
    if not rights.permits(source.purpose, now):
        raise ExternalInputError("RIGHTS_NOT_PERMITTED", f"Rights do not permit {source.purpose}")
    path = _safe_image_path(image_root, source.image_path)
    raw = path.read_bytes()
    if len(raw) > MAX_INPUT_BYTES:
        raise ExternalInputError("IMAGE_TOO_LARGE", "Image exceeds the 20 MiB input limit")
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        with Image.open(io.BytesIO(raw)) as decoded:
            if decoded.format not in ALLOWED_FORMATS:
                raise ExternalInputError("IMAGE_FORMAT_UNSUPPORTED", "Use JPEG, PNG, or WebP")
            if getattr(decoded, "n_frames", 1) != 1:
                raise ExternalInputError(
                    "ANIMATED_IMAGE_UNSUPPORTED", "Animated images are not supported"
                )
            if decoded.width * decoded.height > MAX_DECODED_PIXELS:
                raise ExternalInputError(
                    "IMAGE_PIXEL_LIMIT", "Image exceeds 40 million decoded pixels"
                )
            decoded.load()
            rgb = _rgb_image(decoded, source.rotation_degrees)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ExternalInputError("IMAGE_DECODE_FAILED", str(error)) from error
    if min(rgb.size) < MIN_SHORT_EDGE:
        raise ExternalInputError(
            "IMAGE_EDGE_TOO_SHORT", "The shortest image edge must be at least 128 px"
        )

    pixel_sha256 = canonical_pixel_hash(rgb)
    view = rgb.copy()
    if max(view.size) > MAX_INFERENCE_EDGE:
        view.thumbnail((MAX_INFERENCE_EDGE, MAX_INFERENCE_EDGE), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    view.save(encoded, "PNG", optimize=False)
    encoded_bytes = encoded.getvalue()
    encoded_sha256 = hashlib.sha256(encoded_bytes).hexdigest()
    relative_path = Path("images") / encoded_sha256[:2] / f"{encoded_sha256}.png"
    destination = artifact_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(destination, encoded_bytes)

    group_ids = tuple(sorted(set(source.source_group_ids)))
    image_id = canonical_hash({"source_id": source.source_id, "pixel": pixel_sha256})
    return ImageArtifact(
        image_id=image_id,
        source_id=source.source_id,
        purpose=source.purpose,
        dataset=source.dataset,
        dataset_split=source.dataset_split,
        raw_sha256=raw_sha256,
        canonical_pixel_sha256=pixel_sha256,
        source_group_ids=group_ids,
        visual_group_id=pixel_sha256,
        full_view=ImageView(
            view_id=f"full:{pixel_sha256}",
            relative_path=relative_path.as_posix(),
            encoded_sha256=encoded_sha256,
            pixel_sha256=canonical_pixel_hash(view),
            width=view.width,
            height=view.height,
            media_type="image/png",
        ),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except FileExistsError:
        if not path.exists() or path.read_bytes() != data:
            raise
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class _DisjointSet:
    parent: dict[str, str]

    @classmethod
    def for_ids(cls, values: Sequence[str]) -> _DisjointSet:
        return cls(parent={value: value for value in values})

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            first, second = sorted((left_root, right_root))
            self.parent[second] = first


def group_visual_sources(
    images: Sequence[ImageArtifact],
    embeddings: Mapping[str, Sequence[float]],
    *,
    threshold: float = 0.95,
) -> dict[str, str]:
    """Group exact, declared, and SSCD-near sources using cosine similarity.

    `embeddings` must contain normalized or normalizable SSCD vectors keyed by image ID.
    Missing vectors are allowed and leave that image grouped only by exact or declared links.

    Returns:
        Mapping from image ID to a stable visual-group ID.

    Raises:
        ValueError: If an embedding is empty, non-finite, or has a mismatched dimension.
    """
    ids = [image.image_id for image in images]
    groups = _DisjointSet.for_ids(ids)
    by_pixel: dict[str, str] = {}
    by_source_group: dict[str, str] = {}
    for image in images:
        if representative := by_pixel.get(image.canonical_pixel_sha256):
            groups.union(image.image_id, representative)
        else:
            by_pixel[image.canonical_pixel_sha256] = image.image_id
        for source_group in image.source_group_ids:
            if representative := by_source_group.get(source_group):
                groups.union(image.image_id, representative)
            else:
                by_source_group[source_group] = image.image_id

    normalized: dict[str, tuple[float, ...]] = {}
    dimension: int | None = None
    for image_id, values in embeddings.items():
        if image_id not in groups.parent:
            continue
        vector = tuple(float(value) for value in values)
        if not vector or any(not math.isfinite(value) for value in vector):
            raise ValueError(f"Invalid SSCD embedding for {image_id}")
        if dimension is None:
            dimension = len(vector)
        if len(vector) != dimension:
            raise ValueError("SSCD embeddings must have one dimension")
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            raise ValueError(f"Zero SSCD embedding for {image_id}")
        normalized[image_id] = tuple(value / norm for value in vector)

    embedded_ids = sorted(normalized)
    for index, left in enumerate(embedded_ids):
        for right in embedded_ids[index + 1 :]:
            cosine = sum(a * b for a, b in zip(normalized[left], normalized[right], strict=True))
            if cosine >= threshold:
                groups.union(left, right)

    members: dict[str, list[str]] = {}
    for image_id in ids:
        members.setdefault(groups.find(image_id), []).append(image_id)
    output: dict[str, str] = {}
    for component in members.values():
        group_id = canonical_hash(sorted(component))
        for image_id in component:
            output[image_id] = group_id
    return output


def assign_split(visual_group_id: str, seed: int) -> str:
    """Assign a stable 90/5/5 train, development, or test bucket."""
    digest = hashlib.sha256(f"{seed}:{visual_group_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10_000
    if bucket < 9000:
        return "train"
    if bucket < 9500:
        return "development"
    return "test"
