"""Summarize a frozen ABBA synthesis experiment without using automatic labels as gold."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

PHASES = ("baseline-1", "variant-1", "variant-2", "baseline-2")


def _read(path: Path) -> dict[str, Any]:
    """Load one generated experiment record."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _interval_seconds(interval: dict[str, Any]) -> float:
    """Require a finite, positive recorded execution interval."""
    start, end = interval.get("started_at"), interval.get("ended_at")
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or not math.isfinite(start)
        or not math.isfinite(end)
        or end <= start
    ):
        raise ValueError("Invalid recorded execution interval")
    return end - start


def _recorded_seconds(intervals: list[dict[str, Any]]) -> float:
    """Sum ordered execution intervals while excluding pauses between them."""
    total = 0.0
    previous_end: float | None = None
    for interval in intervals:
        total += _interval_seconds(interval)
        if previous_end is not None and interval["started_at"] < previous_end:
            raise ValueError("Recorded execution intervals overlap or are out of order")
        previous_end = interval["ended_at"]
    return total


def _allocation_intervals(progress: dict[str, Any]) -> list[dict[str, Any]]:
    """Read measured single-GPU sessions, retaining the legacy continuous-run format."""
    intervals = progress.get("allocation_intervals")
    if intervals is None:
        if progress.get("prior_attempts"):
            raise ValueError("Resumed experiments require recorded GPU allocation intervals")
        return [progress]
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("GPU allocation intervals must be a nonempty list")
    _recorded_seconds(intervals)
    for interval in intervals:
        if not isinstance(interval.get("gpu"), int) or isinstance(interval["gpu"], bool):
            raise ValueError("Each allocation interval must identify one GPU")
        seconds = interval.get("gpu_seconds")
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or not math.isclose(seconds, _interval_seconds(interval), abs_tol=1e-6)
        ):
            raise ValueError("GPU seconds disagree with the single-GPU allocation interval")
    return intervals


def summarize(experiment_dir: Path) -> dict[str, Any]:
    """Compare paired outcomes and variability using recorded GPU allocation intervals."""
    plan = _read(experiment_dir / "experiment-plan.json")
    progress = _read(experiment_dir / "progress.json")
    if plan.get("phase_order") != list(PHASES) or not progress.get("success"):
        raise ValueError("The ABBA experiment has not completed as planned")
    wall_seconds = _interval_seconds(progress)
    allocations = _allocation_intervals(progress)
    allocated_seconds = _recorded_seconds(allocations)
    planned_ids = [image["source_id"] for image in plan["images"]]
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError("Planned source IDs must be unique")

    phases: dict[str, Any] = {}
    accepted: dict[str, set[str]] = {}
    first_assignments: list[tuple[str, str]] | None = None
    for name in PHASES:
        timing = progress["phases"].get(name)
        if timing is None or timing.get("status") != "COMPLETED":
            raise ValueError(f"Incomplete phase: {name}")
        attempts = [*timing.get("prior_attempts", []), timing]
        phase_seconds = _recorded_seconds(attempts)
        if any(
            not any(
                allocation["started_at"] <= attempt["started_at"]
                and attempt["ended_at"] <= allocation["ended_at"]
                for allocation in allocations
            )
            for attempt in attempts
        ):
            raise ValueError(f"Phase lies outside its GPU allocation intervals: {name}")
        report = _read(experiment_dir / name / "diagnostics.json")
        rows = report["rows"]
        if len(rows) != len(planned_ids) or [row["source_id"] for row in rows] != planned_ids:
            raise ValueError(f"Image order or count changed in {name}")
        conversations = [
            json.loads(line)
            for line in (experiment_dir / name / "conversations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if [item["image"]["source_id"] for item in conversations] != planned_ids:
            raise ValueError(f"Public conversation order changed in {name}")
        assignments = [
            (item["generation_model"], item["target_language"]) for item in conversations
        ]
        if first_assignments is None:
            first_assignments = assignments
        elif assignments != first_assignments:
            raise ValueError(f"Generator or language allocation changed in {name}")
        if any(
            planned.get("language", actual[1]) != actual[1]
            for planned, actual in zip(plan["images"], assignments, strict=True)
        ):
            raise ValueError(f"Planned language allocation changed in {name}")
        if "generator_models" in plan and any(
            plan["generator_models"][planned["generator_role"]] != actual[0]
            for planned, actual in zip(plan["images"], assignments, strict=True)
        ):
            raise ValueError(f"Planned generator allocation changed in {name}")
        if report["conversations"] != len(rows):
            raise ValueError(f"Diagnostic conversation count changed in {name}")
        accepted[name] = {row["source_id"] for row in rows if row["status"] == "QUALITY_CANDIDATE"}
        attempted = sum(row["attempted_turns"] for row in rows)
        committed = sum(row["committed_turns"] for row in rows)
        if attempted != report["attempted_turns"] or committed != report["committed_turns"]:
            raise ValueError(f"Turn counts disagree in {name}")
        if report["completed_conversations"] != len(accepted[name]):
            raise ValueError(f"Completed conversation count disagrees in {name}")
        committed_tasks = Counter(
            turn["instruction"]["task_id"]
            for conversation in conversations
            for turn in conversation.get("turns", [])
            if turn["status"] == "COMMITTED"
        )
        if sum(committed_tasks.values()) != committed:
            raise ValueError(f"Public committed turns disagree in {name}")
        image_durations = [row["duration_ms"] / 1000 for row in rows if "duration_ms" in row]
        if len(image_durations) != len(rows):
            raise ValueError(f"Image duration missing in {name}")
        phases[name] = {
            "images": len(rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "attempted_turns": attempted,
            "committed_turns": committed,
            "committed_task_counts": dict(sorted(committed_tasks.items())),
            "completed_conversations": len(accepted[name]),
            "zero_committed_images": sum(row["committed_turns"] == 0 for row in rows),
            "two_or_more_committed_images": sum(row["committed_turns"] >= 2 for row in rows),
            "stop_stage_counts": dict(
                Counter(row["stop_stage"] or "completed" for row in rows if "stop_stage" in row)
            ),
            "zero_committed_stop_stage_counts": dict(
                Counter(
                    row["stop_stage"] or "unknown"
                    for row in rows
                    if row["committed_turns"] == 0 and "stop_stage" in row
                )
            ),
            "model_calls": report["model_calls"],
            "retry_calls": report["retry_calls"],
            "contract_failure_attempts": report["contract_failure_attempts"],
            "candidate_binding_rejections": report["binding_rejections"],
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "missing_token_usage_calls": report["token_usage_missing_calls"],
            "median_image_duration_seconds": median(image_durations),
            "max_image_duration_seconds": max(image_durations),
            "phase_gpu_hours": phase_seconds / 3600,
            "prior_attempt_count": len(attempts) - 1,
            "quality_source_ids": sorted(accepted[name]),
            "human_approved_completed_conversations": None,
            "human_approved_per_gpu_hour": None,
            "human_labeled": 0,
        }

    def overlap(left: str, right: str) -> dict[str, int]:
        return {
            "both": len(accepted[left] & accepted[right]),
            "left_only": len(accepted[left] - accepted[right]),
            "right_only": len(accepted[right] - accepted[left]),
            "neither": len(planned_ids) - len(accepted[left] | accepted[right]),
        }

    plan_fields = (
        "baseline_commit",
        "variant_commit",
        "prepared_manifest_sha256",
        "seed",
        "workers",
    )
    config_fields = ("config_sha256", "baseline_config_sha256", "variant_config_sha256")
    if not any(key in plan for key in config_fields[:2]):
        raise ValueError("Frozen plan is missing its baseline configuration hash")
    return {
        "plan": {
            **{key: plan[key] for key in plan_fields},
            **{
                key: plan[key]
                for key in (*config_fields, "intervention", "generator_models")
                if key in plan
            },
        },
        "images": len(planned_ids),
        "generator_and_language_assignments_identical": True,
        "total_allocated_gpu_hours": allocated_seconds / 3600,
        "wall_elapsed_hours": wall_seconds / 3600,
        "model_server_sessions": len(allocations),
        "timing_interpretation": (
            "Includes recorded loading, inference, interrupted attempts and teardown; "
            "excludes gaps between allocated sessions. Multiple sessions do not establish "
            "a comparison within one model startup session."
            if len(allocations) > 1
            else "One continuous GPU allocation, including model loading and teardown."
        ),
        "phases": phases,
        "repeat_overlap": {
            "baseline": overlap("baseline-1", "baseline-2"),
            "variant": overlap("variant-1", "variant-2"),
        },
        "paired_overlap": {
            "first": overlap("baseline-1", "variant-1"),
            "second": overlap("baseline-2", "variant-2"),
        },
        "human_error_rates": None,
        "interpretation": "Automatic outcomes only; independent human quality labels are pending.",
    }


def main() -> None:
    """Write JSON, CSV, and Markdown beside the private phase diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    result = summarize(args.experiment_dir)
    stem = args.experiment_dir / "comparison"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        columns = (
            "phase",
            "images",
            "attempted_turns",
            "committed_turns",
            "completed_conversations",
            "zero_committed_images",
            "two_or_more_committed_images",
            "model_calls",
            "retry_calls",
            "contract_failure_attempts",
            "candidate_binding_rejections",
            "output_tokens",
            "median_image_duration_seconds",
            "max_image_duration_seconds",
            "phase_gpu_hours",
            "human_approved_per_gpu_hour",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, phase in result["phases"].items():
            writer.writerow({"phase": name, **{key: phase[key] for key in columns[1:]}})
    lines = [
        "# Paired synthesis comparison",
        "",
        f"Images per phase: {result['images']}; total allocated GPU hours: "
        f"{result['total_allocated_gpu_hours']:.3f}.",
        f"Model-server sessions: {result['model_server_sessions']}; wall elapsed hours: "
        f"{result['wall_elapsed_hours']:.3f}.",
        result["timing_interpretation"],
        "",
        "| Phase | Attempted turns | Committed turns | Completed conversations | "
        "Zero-commit images | 2+ committed images | Model calls | Retries | Binding rejections | Median image seconds | Phase GPU hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {name} | {phase['attempted_turns']} | {phase['committed_turns']} | "
            f"{phase['completed_conversations']} | {phase['zero_committed_images']} | "
            f"{phase['two_or_more_committed_images']} | "
            f"{phase['model_calls']} | {phase['retry_calls']} | "
            f"{phase['candidate_binding_rejections']} | "
            f"{phase['median_image_duration_seconds']} | "
            f"{phase['phase_gpu_hours']:.3f} |"
            for name, phase in result["phases"].items()
        ),
        "",
        "| Comparison | Both | Left only | Right only | Neither |",
        "|---|---:|---:|---:|---:|",
        *(
            f"| {group} {name} | {counts['both']} | {counts['left_only']} | "
            f"{counts['right_only']} | {counts['neither']} |"
            for group in ("repeat_overlap", "paired_overlap")
            for name, counts in result[group].items()
        ),
        "",
        "Zero-commit stop stages by phase:",
        "",
        *(
            f"- {name}: {', '.join(f'{stage}={count}' for stage, count in sorted(phase['zero_committed_stop_stage_counts'].items())) or 'none'}"
            for name, phase in result["phases"].items()
        ),
        "",
        "Committed task counts by phase (automatic decisions):",
        "",
        *(
            f"- {name}: {', '.join(f'{task}={count}' for task, count in phase['committed_task_counts'].items()) or 'none'}"
            for name, phase in result["phases"].items()
        ),
        "",
        "Human-approved conversations per GPU hour and human error rates are unmeasured "
        "until independent ballots are resolved. These are automatic quality-candidate outcomes.",
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(stem)


if __name__ == "__main__":
    main()
