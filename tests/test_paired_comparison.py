"""Paired comparisons must expose repeat variation and preserve missing human labels."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

summarize = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "validation/compare_paired_runs.py")
)["summarize"]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_abba_comparison_counts_paired_changes_without_inventing_gold(tmp_path: Path) -> None:
    _write(
        tmp_path / "experiment-plan.json",
        {
            "phase_order": ["baseline-1", "variant-1", "variant-2", "baseline-2"],
            "images": [{"source_id": "a"}, {"source_id": "b"}],
            "baseline_commit": "old",
            "variant_commit": "new",
            "config_sha256": "config",
            "prepared_manifest_sha256": "images",
            "seed": 42,
            "workers": 2,
        },
    )
    _write(
        tmp_path / "progress.json",
        {
            "success": True,
            "started_at": 0,
            "ended_at": 180,
            "phases": {
                name: {"status": "COMPLETED", "started_at": index * 40, "ended_at": index * 40 + 40}
                for index, name in enumerate(("baseline-1", "variant-1", "variant-2", "baseline-2"))
            },
        },
    )
    outcomes = {
        "baseline-1": ("QUALITY_CANDIDATE", "REJECTED"),
        "variant-1": ("REJECTED", "QUALITY_CANDIDATE"),
        "variant-2": ("QUALITY_CANDIDATE", "REJECTED"),
        "baseline-2": ("QUALITY_CANDIDATE", "REJECTED"),
    }
    for name, statuses in outcomes.items():
        _write(
            tmp_path / name / "diagnostics.json",
            {
                "conversations": 2,
                "attempted_turns": 4,
                "committed_turns": 2,
                "completed_conversations": 1,
                "model_calls": 10,
                "retry_calls": 1,
                "contract_failure_attempts": 1,
                "binding_rejections": 1,
                "token_usage_missing_calls": 0,
                "rows": [
                    {
                        "source_id": source,
                        "status": status,
                        "attempted_turns": 2,
                        "committed_turns": 1,
                        "output_tokens": 50,
                    }
                    for source, status in zip(("a", "b"), statuses, strict=True)
                ],
            },
        )
    report = summarize(tmp_path)
    assert report["total_allocated_gpu_hours"] == 0.05
    assert report["repeat_overlap"]["baseline"] == {
        "both": 1,
        "left_only": 0,
        "right_only": 0,
        "neither": 1,
    }
    assert report["repeat_overlap"]["variant"]["both"] == 0
    assert report["paired_overlap"]["first"]["left_only"] == 1
    assert report["phases"]["baseline-1"]["human_approved_per_gpu_hour"] is None
    assert report["human_error_rates"] is None

    progress = json.loads((tmp_path / "progress.json").read_text())
    progress["success"] = False
    _write(tmp_path / "progress.json", progress)
    with pytest.raises(ValueError, match="not completed"):
        summarize(tmp_path)
