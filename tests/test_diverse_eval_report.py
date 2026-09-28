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
                {"instruction": {"task_id": "attribute_lookup"}},
                {"instruction": {"task_id": "object_identification"}},
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


def test_report_rejects_sources_outside_pinned_manifest() -> None:
    diagnostics = {
        "rows": [{"source_id": "commons-eval:2"}],
    }
    manifest = [{"commons_page_id": 1}]
    with pytest.raises(ValueError, match="Unpinned source"):
        build_report(diagnostics, manifest)
