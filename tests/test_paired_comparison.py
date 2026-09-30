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


@pytest.fixture
def paired_experiment(tmp_path: Path) -> Path:
    """Save a complete two-image comparison with distinct automatic outcomes."""
    _write(
        tmp_path / "experiment-plan.json",
        {
            "phase_order": ["baseline-1", "variant-1", "variant-2", "baseline-2"],
            "images": [
                {"source_id": "a", "generator_role": "generator_a"},
                {"source_id": "b", "generator_role": "generator_b"},
            ],
            "generator_models": {"generator_a": "generator-a", "generator_b": "generator-b"},
            "baseline_commit": "old",
            "variant_commit": "new",
            "config_sha256": "config",
            "variant_config_sha256": "variant-config",
            "intervention": "tasks.candidate_limit: 8 versus 4",
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
        (tmp_path / name).mkdir(parents=True, exist_ok=True)
        (tmp_path / name / "conversations.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "image": {"source_id": source},
                        "generation_model": "generator-a" if source == "a" else "generator-b",
                        "target_language": "en",
                        "turns": (
                            [
                                {
                                    "instruction": {"task_id": "object_identification"},
                                    "status": "COMMITTED",
                                },
                                {
                                    "instruction": {"task_id": "color_attribute"},
                                    "status": "COMMITTED",
                                },
                            ]
                            if status == "QUALITY_CANDIDATE"
                            else []
                        ),
                    }
                )
                + "\n"
                for source, status in zip(("a", "b"), statuses, strict=True)
            )
        )
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
                        "committed_turns": 2 if status == "QUALITY_CANDIDATE" else 0,
                        "output_tokens": 50,
                        "duration_ms": 1500 if source == "a" else 2500,
                        "stop_stage": None if status == "QUALITY_CANDIDATE" else "question_gate",
                    }
                    for source, status in zip(("a", "b"), statuses, strict=True)
                ],
            },
        )
    return tmp_path


def test_abba_comparison_counts_paired_changes_without_inventing_gold(
    paired_experiment: Path,
) -> None:
    tmp_path = paired_experiment
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
    assert report["phases"]["baseline-1"]["two_or_more_committed_images"] == 1
    assert report["phases"]["baseline-1"]["committed_task_counts"] == {
        "color_attribute": 1,
        "object_identification": 1,
    }
    assert report["phases"]["baseline-1"]["median_image_duration_seconds"] == 2.0
    assert report["phases"]["baseline-1"]["max_image_duration_seconds"] == 2.5
    assert report["phases"]["baseline-1"]["stop_stage_counts"] == {
        "completed": 1,
        "question_gate": 1,
    }
    assert report["phases"]["baseline-1"]["zero_committed_stop_stage_counts"] == {
        "question_gate": 1
    }
    assert report["human_error_rates"] is None
    assert report["generator_and_language_assignments_identical"]
    assert report["plan"]["variant_config_sha256"] == "variant-config"
    assert report["plan"]["generator_models"]["generator_a"] == "generator-a"

    incomplete = tmp_path / "baseline-1" / "diagnostics.json"
    incomplete_report = json.loads(incomplete.read_text())
    del incomplete_report["rows"][0]["duration_ms"]
    _write(incomplete, incomplete_report)
    with pytest.raises(ValueError, match="Image duration missing"):
        summarize(tmp_path)
    incomplete_report["rows"][0]["duration_ms"] = 1500
    _write(incomplete, incomplete_report)

    changed = tmp_path / "variant-2" / "conversations.jsonl"
    original = changed.read_text()
    changed.write_text(original.replace("generator-b", "generator-a"))
    with pytest.raises(ValueError, match="allocation changed"):
        summarize(tmp_path)
    changed.write_text(original)

    plan = json.loads((tmp_path / "experiment-plan.json").read_text())
    plan["generator_models"]["generator_a"] = "different-model"
    _write(tmp_path / "experiment-plan.json", plan)
    with pytest.raises(ValueError, match="Planned generator allocation changed"):
        summarize(tmp_path)
    plan["generator_models"]["generator_a"] = "generator-a"
    _write(tmp_path / "experiment-plan.json", plan)

    progress = json.loads((tmp_path / "progress.json").read_text())
    progress["success"] = False
    _write(tmp_path / "progress.json", progress)
    with pytest.raises(ValueError, match="not completed"):
        summarize(tmp_path)


def test_resumed_comparison_counts_allocated_sessions_and_interrupted_work(
    paired_experiment: Path,
) -> None:
    path = paired_experiment / "progress.json"
    progress = json.loads(path.read_text())
    progress.update(
        ended_at=1020,
        prior_attempts=[{"ended_at": 150, "error": "operator interruption"}],
        allocation_intervals=[
            {"gpu": 4, "started_at": 0, "ended_at": 150, "gpu_seconds": 150},
            {"gpu": 4, "started_at": 900, "ended_at": 1020, "gpu_seconds": 120},
        ],
    )
    progress["phases"]["baseline-2"] = {
        "status": "COMPLETED",
        "started_at": 960,
        "ended_at": 1000,
        "prior_attempts": [{"status": "INTERRUPTED", "started_at": 120, "ended_at": 140}],
    }
    _write(path, progress)
    report = summarize(paired_experiment)
    assert report["total_allocated_gpu_hours"] == 270 / 3600
    assert report["wall_elapsed_hours"] == 1020 / 3600
    assert report["model_server_sessions"] == 2
    assert report["phases"]["baseline-2"]["phase_gpu_hours"] == 60 / 3600
    assert report["phases"]["baseline-2"]["prior_attempt_count"] == 1
    assert "Multiple sessions" in report["timing_interpretation"]

    del progress["allocation_intervals"]
    _write(path, progress)
    with pytest.raises(ValueError, match="require recorded GPU allocation"):
        summarize(paired_experiment)


@pytest.mark.parametrize("failure", ["overlap", "missing", "nonfinite", "disagree"])
def test_comparison_rejects_invalid_allocation_records(
    paired_experiment: Path, failure: str
) -> None:
    path = paired_experiment / "progress.json"
    progress = json.loads(path.read_text())
    intervals: list[dict[str, int | float]] = [
        {"gpu": 4, "started_at": 0, "ended_at": 180, "gpu_seconds": 180}
    ]
    if failure == "overlap":
        intervals.append({"gpu": 4, "started_at": 170, "ended_at": 180, "gpu_seconds": 10})
    elif failure == "missing":
        intervals[0]["ended_at"] = 150
        intervals[0]["gpu_seconds"] = 150
    elif failure == "nonfinite":
        intervals[0]["ended_at"] = float("inf")
    else:
        intervals[0]["gpu_seconds"] = 0
    progress["allocation_intervals"] = intervals
    _write(path, progress)
    with pytest.raises(ValueError):
        summarize(paired_experiment)
