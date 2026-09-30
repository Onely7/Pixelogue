"""Campaign reports preserve adoption boundaries and honest coverage denominators."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pixelogue.catalog import load_task_catalog
from pixelogue.synthesis_campaign_report import build_report, write_reports


def conversation(status: str, turn_statuses: tuple[str, ...], source: str = "sample") -> dict:
    """Supply only the public dialogue fields consumed by the report."""
    return {
        "conversation_id": source,
        "image": {"source_id": source},
        "status": status,
        "generation_model": "generator",
        "target_language": "en",
        "turns": [
            {
                "turn_index": index,
                "status": turn_status,
                "instruction": {"task_id": task},
                "question": {"content": f"Question {index}"},
                "answer": {"content": f"Answer {index}"},
            }
            for index, (turn_status, task) in enumerate(
                zip(turn_statuses, ("object_identification", "attribute_lookup"), strict=True),
                start=1,
            )
        ],
    }


def test_rejected_prefix_is_not_counted_as_candidate_data() -> None:
    rejected = conversation("REJECTED", ("COMMITTED", "REJECTED"))
    report = build_report([{"source_id": "sample"}], [rejected], load_task_catalog()["tasks"])
    assert report["complete"]
    assert report["committed_turns_all_outcomes"] == 1
    assert report["candidate_turns"] == 0
    assert report["automatic_candidate_rate"] == 0
    assert report["tasks"][0]["example_status"] == "diagnostic_prefix"
    assert report["tasks"][1]["examples"] == []
    assert report["human_approved_conversations"] is None


def test_partial_run_keeps_unprocessed_images_and_all_72_tasks_visible() -> None:
    report = build_report([{"source_id": "sample"}], [], load_task_catalog()["tasks"])
    assert not report["complete"]
    assert report["unprocessed_images"] == 1
    assert report["automatic_candidate_rate"] is None
    assert report["catalog_tasks"] == len(report["tasks"]) == 72
    assert report["normal_catalog_tasks"] == 65
    assert all(task["examples"] == [] for task in report["tasks"])


@pytest.mark.parametrize("statuses", [("COMMITTED", "REJECTED"), ("COMMITTED", "ABSTAINED")])
def test_candidate_cannot_include_a_noncommitted_turn(statuses: tuple[str, ...]) -> None:
    invalid = conversation("QUALITY_CANDIDATE", statuses)
    with pytest.raises(ValueError, match="two to six committed"):
        build_report([{"source_id": "sample"}], [invalid], load_task_catalog()["tasks"])


def test_duplicate_or_unprepared_sources_are_rejected() -> None:
    candidate = conversation("QUALITY_CANDIDATE", ("COMMITTED", "COMMITTED"))
    with pytest.raises(ValueError, match="must be unique"):
        build_report(
            [{"source_id": "sample"}], [candidate, candidate], load_task_catalog()["tasks"]
        )
    with pytest.raises(ValueError, match="unprepared"):
        build_report([{"source_id": "other"}], [candidate], load_task_catalog()["tasks"])


def test_candidate_task_counts_and_exact_duplicates_use_only_candidate_turns() -> None:
    first = conversation("QUALITY_CANDIDATE", ("COMMITTED", "COMMITTED"))
    second = conversation("QUALITY_CANDIDATE", ("COMMITTED", "COMMITTED"), "second")
    second["turns"][1]["question"]["content"] = "  QUESTION 1  "
    report = build_report(
        [{"source_id": "sample"}, {"source_id": "second"}],
        [first, second],
        load_task_catalog()["tasks"],
    )
    assert report["candidate_task_diversity"]["counts"] == {
        "object_identification": 2,
        "attribute_lookup": 2,
    }
    assert report["candidate_task_diversity"]["top_task_share"] == 0.5
    assert report["exact_question_repeats_within_candidates"] == 1
    assert report["normalized_unique_candidate_questions"] == 2


def test_gallery_escapes_generated_markup_and_preserves_public_history(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    Image.new("RGB", (128, 128), "white").save(prepared / "image.png")
    candidate = conversation("QUALITY_CANDIDATE", ("COMMITTED", "COMMITTED"))
    candidate["image"]["full_view"] = {"relative_path": "image.png", "encoded_sha256": "pixels"}
    candidate["turns"][0]["question"]["content"] = '<script>alert("question")</script>'
    report = build_report([{"source_id": "sample"}], [candidate], load_task_catalog()["tasks"])
    destination = tmp_path / "report"
    write_reports(report, prepared, destination)
    page = (destination / "examples.html").read_text()
    assert '<script>alert("question")</script>' not in page
    assert "&lt;script&gt;alert" in page
    assert report["tasks"][1]["examples"][0]["public_history"] == [
        {"role": "user", "content": '<script>alert("question")</script>'},
        {"role": "assistant", "content": "Answer 1"},
    ]
    assert (destination / "thumbnails/pixels.jpg").is_file()
    assert "no_observed_example" in (destination / "task-coverage.csv").read_text()


def test_gallery_rejects_an_image_outside_prepared_root(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    Image.new("RGB", (128, 128), "white").save(tmp_path / "outside.png")
    candidate = conversation("QUALITY_CANDIDATE", ("COMMITTED", "COMMITTED"))
    candidate["image"]["full_view"] = {
        "relative_path": "../outside.png",
        "encoded_sha256": "pixels",
    }
    report = build_report([{"source_id": "sample"}], [candidate], load_task_catalog()["tasks"])
    with pytest.raises(ValueError, match="leaves"):
        write_reports(report, prepared, tmp_path / "report")
