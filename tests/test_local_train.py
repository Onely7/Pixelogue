from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

from pixelogue.cli import _local_train_progress, app
from pixelogue.contracts import ImageArtifact, RightsRecord, SourceRecord
from pixelogue.errors import ExternalInputError, ShortfallError
from pixelogue.io import read_jsonl, write_jsonl
from pixelogue.local_train import prepare_local_train
from pixelogue.operations import prepare_sources


def _inputs(tmp_path: Path, count: int = 4):
    root = tmp_path / "jpeg"
    root.mkdir()
    rows = []
    for index in range(count):
        identifier = f"{index:016x}"
        Image.new("RGB", (192, 128), (index * 30, 20, 10)).save(root / f"{identifier}.jpg")
        rows.append(
            {
                "ImageID": identifier,
                "Subset": "train",
                "License": "https://creativecommons.org/licenses/by/2.0/",
                "Author": "Example author",
                "OriginalLandingURL": "https://example.org/photo",
                "Rotation": "90.0",
                "Title": "Private title, not a model input",
            }
        )
    metadata = tmp_path / "metadata.csv"
    _csv(metadata, rows)
    pinned = tmp_path / "pinned.jsonl"
    pinned.write_text("")
    return root, metadata, pinned, rows


def _csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_sample_is_order_independent_and_ingestable(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    report = prepare_local_train(
        metadata, root, first, count=2, seed=42, validation_manifest=pinned
    )
    _csv(metadata, list(reversed(rows)))
    prepare_local_train(metadata, root, second, count=2, seed=42, validation_manifest=pinned)
    assert (first / "selected-ids.json").read_bytes() == (second / "selected-ids.json").read_bytes()
    assert report["accepted"] == 2
    sources = read_jsonl(first / "sources.jsonl", SourceRecord)
    rights = read_jsonl(first / "rights.jsonl", RightsRecord)
    assert all(source.rotation_degrees == 90 for source in sources)
    assert "Private title" not in (first / "sources.jsonl").read_text()
    assert "Private title" in (first / "private-metadata.jsonl").read_text()
    assert all(
        right.training_allowed and not right.image_redistribution_allowed for right in rights
    )
    prepared = prepare_sources(sources, rights, root, tmp_path / "reingested", seed=42)
    assert len(prepared.images) == 2
    assert all(
        (image.full_view.width, image.full_view.height) == (128, 192) for image in prepared.images
    )


def test_metadata_filters_and_missing_files(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path, 8)
    rows[0]["Subset"] = "validation"
    rows[1]["License"] = "https://creativecommons.org/licenses/by-nc/2.0/"
    rows[2]["Rotation"] = ""
    rows[3]["Author"] = ""
    (root / f"{rows[4]['ImageID']}.jpg").unlink()
    (root / f"{rows[5]['ImageID']}.jpg").unlink()
    (root / f"{rows[5]['ImageID']}.jpg").symlink_to(root / f"{rows[6]['ImageID']}.jpg")
    _csv(metadata, rows)
    output = tmp_path / "out"
    prepare_local_train(metadata, root, output, count=2, seed=42, validation_manifest=pinned)
    assert {s.dataset_image_id for s in read_jsonl(output / "sources.jsonl", SourceRecord)} == {
        rows[6]["ImageID"],
        rows[7]["ImageID"],
    }
    with pytest.raises(ShortfallError):
        prepare_local_train(
            metadata, root, tmp_path / "short", count=3, seed=42, validation_manifest=pinned
        )
    assert not (tmp_path / "short").exists()


def test_exact_evaluation_copy_is_excluded_even_with_another_id(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path, 1)
    baseline = tmp_path / "baseline"
    prepare_local_train(metadata, root, baseline, count=1, seed=42, validation_manifest=pinned)
    image = read_jsonl(baseline / "images.jsonl", ImageArtifact)[0]
    reference = image.model_dump(mode="json")
    reference.update(purpose="evaluation", source_id="other-evaluation-id", raw_sha256="f" * 64)
    evaluation = tmp_path / "evaluation.jsonl"
    write_jsonl(evaluation, [reference])
    output = tmp_path / "excluded"
    report = prepare_local_train(
        metadata,
        root,
        output,
        count=1,
        seed=42,
        validation_manifest=pinned,
        evaluation_images=[evaluation],
    )
    assert report["accepted"] == 0
    assert "EVALUATION_EXACT_COPY" in (output / "failures.jsonl").read_text()
    assert (output / "sources.jsonl").read_text() == ""


def test_duplicate_selected_id_and_existing_destination_are_rejected(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path, 1)
    _csv(metadata, rows * 2)
    with pytest.raises(ExternalInputError, match="unique metadata"):
        prepare_local_train(
            metadata, root, tmp_path / "out", count=1, seed=42, validation_manifest=pinned
        )
    with pytest.raises(ExternalInputError) as caught:
        prepare_local_train(metadata, root, root, count=1, seed=42, validation_manifest=pinned)
    assert caught.value.reason == "DESTINATION_EXISTS"


def test_corrupt_image_is_reported_without_replacement(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path, 1)
    (root / f"{rows[0]['ImageID']}.jpg").write_bytes(b"not a jpeg")
    result = CliRunner().invoke(
        app,
        [
            "prepare-local-train",
            "--metadata",
            str(metadata),
            "--image-root",
            str(root),
            "--destination",
            str(tmp_path / "out"),
            "--count",
            "1",
            "--validation-manifest",
            str(pinned),
        ],
    )
    assert result.exit_code == 0, result.output
    assert '"accepted": 0' in result.output
    assert "IMAGE_DECODE_FAILED" in (tmp_path / "out" / "failures.jsonl").read_text()


def test_pinned_validation_id_cannot_be_reclassified_as_train(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path, 1)
    write_jsonl(
        pinned,
        [
            {
                "image_id": rows[0]["ImageID"],
                "split": "validation",
                "license_uri": rows[0]["License"],
                "attribution": "Author",
                "landing_url": rows[0]["OriginalLandingURL"],
                "raw_sha256": "a" * 64,
            }
        ],
    )
    with pytest.raises(ShortfallError):
        prepare_local_train(
            metadata, root, tmp_path / "out", count=1, seed=42, validation_manifest=pinned
        )


def test_duplicate_id_with_ineligible_first_occurrence_is_rejected(tmp_path):
    root, metadata, pinned, rows = _inputs(tmp_path, 1)
    _csv(metadata, [dict(rows[0], Subset="validation"), rows[0]])
    with pytest.raises(ExternalInputError, match="unique metadata"):
        prepare_local_train(
            metadata, root, tmp_path / "out", count=1, seed=42, validation_manifest=pinned
        )


def test_training_reference_is_not_treated_as_evaluation(tmp_path):
    root, metadata, pinned, _ = _inputs(tmp_path, 1)
    baseline = tmp_path / "baseline"
    prepare_local_train(metadata, root, baseline, count=1, seed=42, validation_manifest=pinned)
    with pytest.raises(ExternalInputError) as caught:
        prepare_local_train(
            metadata,
            root,
            tmp_path / "out",
            count=1,
            seed=42,
            validation_manifest=pinned,
            evaluation_images=[baseline / "images.jsonl"],
        )
    assert caught.value.reason == "REFERENCE_NOT_EVALUATION"


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_progress_preserves_json_stdout_and_counts_rejections(tmp_path, quiet):
    root, metadata, pinned, rows = _inputs(tmp_path, 2)
    (root / f"{rows[0]['ImageID']}.jpg").write_bytes(b"invalid image")
    arguments = [
        "prepare-local-train",
        "--metadata",
        str(metadata),
        "--image-root",
        str(root),
        "--destination",
        str(tmp_path / "out"),
        "--count",
        "2",
        "--validation-manifest",
        str(pinned),
    ]
    if quiet:
        arguments.append("--quiet")
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["accepted"] == 1
    assert report["rejected"] == 1
    if quiet:
        assert result.stderr == ""
    else:
        assert "scan_metadata: 0" in result.stderr
        assert "verify_metadata: 2/2" in result.stderr
        assert "images: 2/2" in result.stderr
        assert "write_outputs: 1/1" in result.stderr
        assert "elapsed=" in result.stderr


def test_progress_throttles_updates_but_always_reports_stage_end(monkeypatch, capsys):
    clock = iter([0.0, 0.0, 1.0, 2.1, 2.2, 2.3])
    monkeypatch.setattr("pixelogue.cli.time.monotonic", lambda: next(clock))
    report = _local_train_progress()
    report("scan_metadata", 0, None)
    report("scan_metadata", 10000, None)
    report("scan_metadata", 20000, None)
    report("scan_metadata", 20001, 20001)
    report("verify_metadata", 0, 20001)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "10,000" not in captured.err
    assert "20,000" in captured.err
    assert "20,001/20,001" in captured.err
    assert "verify_metadata: 0/20,001" in captured.err
