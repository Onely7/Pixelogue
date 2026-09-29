"""Diverse evaluation reports keep unattempted categories visible."""

from __future__ import annotations

import pytest

from pixelogue.diverse_eval_report import build_report


def test_report_includes_unattempted_category_and_does_not_infer_gold() -> None:
    manifest = [
        {
            "category_number": index,
            "category": category,
            "commons_page_id": index,
            "file_page_url": f"https://commons.wikimedia.org/wiki/File:{index}.png",
        }
        for index, category in ((1, "写真"), (2, "表"))
    ]
    diagnostics = {
        "run_id": "diverse-test",
        "rows": [
            {
                "source_id": "commons-eval:1",
                "status": "QUALITY_CANDIDATE",
                "committed_turns": 2,
                "stop_stage": "holistic_review",
                "stop_category": "QUALITY_CANDIDATE",
                "stop_reason": None,
                "model_calls": 21,
                "retry_calls": 1,
                "invalid_calls": 0,
                "duration_ms": 1000,
                "input_tokens": 100,
                "output_tokens": 20,
            }
        ],
        "model_calls": 21,
        "retry_calls": 1,
        "invalid_calls": 0,
        "stage_reach_images": {"evidence_extraction": 1},
    }
    conversations = [
        {
            "image": {"source_id": "commons-eval:1"},
            "turns": [
                {"status": "COMMITTED", "instruction": {"task_id": "attribute_lookup"}},
                {"status": "COMMITTED", "instruction": {"task_id": "object_identification"}},
            ],
        }
    ]
    summary, rows = build_report(diagnostics, manifest, conversations)
    assert summary["attempted"] == 1
    assert summary["status_counts"] == {"NOT_RUN": 1, "QUALITY_CANDIDATE": 1}
    assert summary["gold_labels"] is False
    assert summary["human_audit_complete"] is False
    assert summary["committed_task_counts"] == {
        "attribute_lookup": 1,
        "object_identification": 1,
    }
    assert [row["status"] for row in rows] == ["QUALITY_CANDIDATE", "NOT_RUN"]
    assert rows[0]["task_ids"] == "attribute_lookup,object_identification"
    assert rows[1]["model_calls"] is None


def test_report_excludes_stopped_turns_from_committed_tasks() -> None:
    """A failed terminal turn must not appear as a committed task."""
    manifest = [
        {
            "category_number": 1,
            "category": "写真",
            "commons_page_id": 1,
            "file_page_url": "https://commons.wikimedia.org/wiki/File:1.png",
        }
    ]
    diagnostics = {
        "run_id": "stopped-test",
        "rows": [
            {
                "source_id": "commons-eval:1",
                "status": "REJECTED",
                "committed_turns": 1,
                "stop_stage": "answer_verification",
                "stop_category": "QUALITY_REJECTION",
                "stop_reason": "RATING_FAIL",
                "model_calls": 12,
                "retry_calls": 0,
                "invalid_calls": 0,
                "duration_ms": 1000,
                "input_tokens": 100,
                "output_tokens": 20,
            }
        ],
        "model_calls": 12,
        "retry_calls": 0,
        "invalid_calls": 0,
        "stage_reach_images": {"evidence_extraction": 1},
    }
    conversations = [
        {
            "image": {"source_id": "commons-eval:1"},
            "turns": [
                {"status": "COMMITTED", "instruction": {"task_id": "object_identification"}},
                {"status": "REJECTED", "instruction": {"task_id": "attribute_lookup"}},
            ],
        }
    ]
    summary, rows = build_report(diagnostics, manifest, conversations)
    assert summary["committed_task_counts"] == {"object_identification": 1}
    assert rows[0]["task_ids"] == "object_identification"

    diagnostics["rows"][0]["committed_turns"] = 2
    with pytest.raises(ValueError, match="Committed turns differ"):
        build_report(diagnostics, manifest, conversations)


def test_report_rejects_sources_outside_pinned_manifest() -> None:
    diagnostics = {
        "rows": [{"source_id": "commons-eval:2"}],
    }
    manifest = [{"commons_page_id": 1}]
    with pytest.raises(ValueError, match="Unpinned source"):
        build_report(diagnostics, manifest)
