from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from PIL import Image

from pixelogue.contracts import RightsRecord, SourcePurpose, SourceRecord
from pixelogue.errors import ExternalInputError, ShortfallError
from pixelogue.images import canonicalize_image, group_visual_sources
from pixelogue.open_images import OpenImagesDownloader, OpenImagesPinnedRecord
from pixelogue.operations import prepare_sources


def _rights(identifier: str, *, training: bool = True) -> RightsRecord:
    return RightsRecord.model_validate(
        {
            "rights_record_id": identifier,
            "license_uri": "https://creativecommons.org/licenses/by/4.0/",
            "attribution": "Test Author",
            "processing_allowed": True,
            "qa_redistribution_allowed": training,
            "image_redistribution_allowed": False,
            "training_allowed": training,
            "valid_from": datetime(2020, 1, 1, tzinfo=UTC),
        }
    )


def _png(size: tuple[int, int] = (256, 192)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "red").save(buffer, "PNG")
    return buffer.getvalue()


def test_canonicalization_applies_rotation_and_rights(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / "image.png").write_bytes(_png((256, 128)))
    source = SourceRecord(
        source_id="one",
        image_path="image.png",
        rights_record_id="rights",
        purpose=SourcePurpose.TRAINING,
        rotation_degrees=90,
    )

    artifact = canonicalize_image(source, _rights("rights"), source_root, tmp_path / "out")

    assert (artifact.full_view.width, artifact.full_view.height) == (128, 256)
    assert (tmp_path / "out" / artifact.full_view.relative_path).is_file()


def test_path_escape_and_training_rights_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.png").write_bytes(_png())
    source = SourceRecord(
        source_id="one",
        image_path="../outside.png",
        rights_record_id="rights",
        purpose=SourcePurpose.TRAINING,
    )
    with pytest.raises(ExternalInputError) as caught:
        canonicalize_image(source, _rights("rights"), root, tmp_path / "out")
    assert caught.value.reason == "IMAGE_PATH_ESCAPE"

    local = source.model_copy(update={"image_path": "image.png"})
    (root / "image.png").write_bytes(_png())
    with pytest.raises(ExternalInputError) as caught:
        canonicalize_image(local, _rights("rights", training=False), root, tmp_path / "out")
    assert caught.value.reason == "RIGHTS_NOT_PERMITTED"


def test_visual_grouping_combines_declared_and_near_duplicates(image_artifact) -> None:
    first, _ = image_artifact
    second = first.model_copy(
        update={
            "image_id": hashlib.sha256(b"second").hexdigest(),
            "source_id": "source-2",
            "source_group_ids": (),
            "canonical_pixel_sha256": hashlib.sha256(b"different").hexdigest(),
        }
    )
    groups = group_visual_sources(
        (first, second),
        {first.image_id: (1.0, 0.0), second.image_id: (0.99, 0.01)},
    )
    assert groups[first.image_id] == groups[second.image_id]


def test_open_images_download_keeps_private_metadata_out_of_source(tmp_path: Path) -> None:
    fieldnames = [
        "ImageID",
        "Subset",
        "OriginalURL",
        "OriginalLandingURL",
        "License",
        "Author",
        "Title",
        "Thumbnail300KURL",
        "Rotation",
    ]
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=fieldnames)
    writer.writeheader()
    for index in range(3):
        writer.writerow(
            {
                "ImageID": f"id-{index}",
                "Subset": "validation",
                "OriginalURL": f"https://images.example/{index}.png",
                "OriginalLandingURL": f"https://pages.example/{index}",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "Author": "Private Author",
                "Title": "Private title",
                "Thumbnail300KURL": f"https://thumbs.example/{index}.png",
                "Rotation": "90",
            }
        )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "metadata.example":
            return httpx.Response(200, text=text.getvalue())
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sources, rights = OpenImagesDownloader(client).download_sample(
        "https://metadata.example/validation.csv", tmp_path, count=2, seed=9
    )

    assert len(sources) == len(rights) == 2
    assert all(source.purpose is SourcePurpose.EVALUATION for source in sources)
    assert all(source.dataset_split == "validation" for source in sources)
    assert all(source.rotation_degrees == 90 for source in sources)
    assert "Private title" not in "\n".join(source.model_dump_json() for source in sources)
    assert all(not record.training_allowed for record in rights)
    private = (tmp_path / "open_images_private_metadata.jsonl").read_text()
    assert "Private title" in private and "Private Author" in private


def test_open_images_pinned_hash_is_enforced(tmp_path: Path) -> None:
    csv_payload = (
        "ImageID,Subset,OriginalURL,OriginalLandingURL,License,Author,Title,"
        "Thumbnail300KURL,Rotation\n"
        "abc,validation,https://images.example/a.png,https://pages.example/a,"
        "https://creativecommons.org/licenses/by/2.0/,Author,Title,"
        "https://thumbs.example/a.png,0\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "metadata.example":
            return httpx.Response(200, text=csv_payload)
        return httpx.Response(200, content=_png(), headers={"content-type": "image/png"})

    pinned = OpenImagesPinnedRecord(
        image_id="abc",
        license_uri="https://creativecommons.org/licenses/by/2.0/",
        attribution="Author",
        landing_url="https://pages.example/a",
        raw_sha256="0" * 64,
    )
    with pytest.raises(ShortfallError) as caught:
        OpenImagesDownloader(httpx.Client(transport=httpx.MockTransport(handler))).download_sample(
            "https://metadata.example/validation.csv",
            tmp_path,
            count=1,
            pinned=(pinned,),
        )
    assert caught.value.reason == "OPEN_IMAGES_SHORTFALL"
    assert "OPEN_IMAGES_PIN_HASH_MISMATCH" in (tmp_path / "open_images_failures.jsonl").read_text()


def test_evaluation_purpose_spreads_to_exact_visual_group(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "training.png").write_bytes(_png())
    (root / "evaluation.png").write_bytes(_png())
    sources = (
        SourceRecord(
            source_id="training",
            image_path="training.png",
            rights_record_id="training-rights",
            purpose=SourcePurpose.TRAINING,
        ),
        SourceRecord(
            source_id="evaluation",
            image_path="evaluation.png",
            rights_record_id="evaluation-rights",
            purpose=SourcePurpose.EVALUATION,
        ),
    )
    report = prepare_sources(
        sources,
        (_rights("training-rights"), _rights("evaluation-rights", training=False)),
        root,
        tmp_path / "prepared",
        seed=1,
    )
    assert len({image.visual_group_id for image in report.images}) == 1
    assert {image.purpose for image in report.images} == {SourcePurpose.EVALUATION}
