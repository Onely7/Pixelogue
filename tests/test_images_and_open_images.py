from __future__ import annotations

import csv
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier, Lock, get_ident

import httpx
import pytest
from PIL import Image, ImageCms

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


def test_invalid_icc_transform_is_an_image_failure(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    Image.new("RGB", (192, 128), "red").save(root / "bad.jpg", icc_profile=profile)
    Image.new("RGB", (192, 128), "blue").save(root / "good.jpg")
    sources = [
        SourceRecord(
            source_id=name,
            image_path=f"{name}.jpg",
            rights_record_id="rights",
            purpose=SourcePurpose.TRAINING,
        )
        for name in ("bad", "good")
    ]
    result = prepare_sources(sources, [_rights("rights")], root, tmp_path / "out", seed=42)
    assert len(result.images) == 1
    assert result.images[0].source_id == "good"
    assert len(result.failures) == 1
    assert result.failures[0].reason == "ICC_CONVERSION_FAILED"


def test_parallel_ingestion_overlaps_and_matches_serial_with_duplicate_pixels(
    tmp_path, monkeypatch
):
    root = tmp_path / "source"
    root.mkdir()
    for index in range(7):
        (root / f"{index}.png").write_bytes(_png())
    (root / "7.png").write_bytes(_png((64, 64)))
    sources = [
        SourceRecord(
            source_id=str(index),
            image_path=f"{index}.png",
            rights_record_id="rights",
            purpose=SourcePurpose.TRAINING,
        )
        for index in reversed(range(8))
    ]
    serial = prepare_sources(sources, [_rights("rights")], root, tmp_path / "serial", seed=42)
    barrier = Barrier(3)
    lock = Lock()
    active = 0
    peak = 0
    completed = []
    main_thread = get_ident()

    def canonicalize(*args, **kwargs):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            if args[0].source_id in {"0", "1", "2"}:
                barrier.wait(timeout=5)
            return canonicalize_image(*args, **kwargs)
        finally:
            with lock:
                active -= 1

    def progress(stage, processed, total):
        assert get_ident() == main_thread
        if stage == "images":
            completed.append(processed)

    monkeypatch.setattr("pixelogue.operations.canonicalize_image", canonicalize)
    parallel = prepare_sources(
        sources,
        [_rights("rights")],
        root,
        tmp_path / "parallel",
        seed=42,
        workers=3,
        progress=progress,
    )
    assert 1 < peak <= 3
    assert completed == list(range(9))
    assert parallel == serial
    for image in parallel.images:
        path = tmp_path / "parallel" / image.full_view.relative_path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == image.full_view.encoded_sha256
    assert not list((tmp_path / "parallel").rglob(".*.png.*"))


def test_canonical_png_does_not_retain_private_exif_or_generated_icc(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    exif = Image.Exif()
    exif[270] = "Private image description"
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (192, 128), "red").save(root / "input.png", icc_profile=profile, exif=exif)
    source = SourceRecord(
        source_id="image",
        image_path="input.png",
        rights_record_id="rights",
        purpose=SourcePurpose.TRAINING,
    )
    artifact = canonicalize_image(source, _rights("rights"), root, tmp_path / "out")
    with Image.open(tmp_path / "out" / artifact.full_view.relative_path) as image:
        assert "icc_profile" not in image.info
        assert "exif" not in image.info
        assert image.getpixel((0, 0)) == (255, 0, 0)
