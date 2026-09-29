"""Compare a balanced one-GPU image-concurrency experiment without human gold labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

WORKERS = (2, 1, 4, 4, 1, 2)
PHASES = tuple(f"workers-{count}-{index}" for index, count in enumerate(WORKERS, start=1))


def _read(path: Path) -> dict[str, Any]:
    """Read one experiment record as a JSON object."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def summarize(experiment_dir: Path) -> dict[str, Any]:
    """Validate the frozen six-phase experiment and aggregate descriptive outcomes."""
    plan = _read(experiment_dir / "experiment-plan.json")
    progress = _read(experiment_dir / "progress.json")
    if (
        plan.get("phase_order") != list(PHASES)
        or plan.get("workers") != list(WORKERS)
        or not progress.get("success")
    ):
        raise ValueError("The balanced concurrency experiment has not completed as planned")
    planned_ids = [item["source_id"] for item in plan["images"]]
    if len(planned_ids) != len(set(planned_ids)):
        raise ValueError("Planned image IDs must be unique")
    for count in set(WORKERS):
        path = experiment_dir / "configs" / f"workers-{count}.yaml"
        if hashlib.sha256(path.read_bytes()).hexdigest() != plan["config_sha256"][str(count)]:
            raise ValueError(f"Worker {count} config changed after the experiment")

    phases: dict[str, dict[str, Any]] = {}
    accepted: dict[str, set[str]] = {}
    for name, count in zip(PHASES, WORKERS, strict=True):
        timing = progress["phases"].get(name)
        if timing is None or timing.get("status") != "COMPLETED":
            raise ValueError(f"Incomplete phase: {name}")
        report = _read(experiment_dir / name / "diagnostics.json")
        rows = report["rows"]
        if len(rows) != len(planned_ids) or [item["source_id"] for item in rows] != planned_ids:
            raise ValueError(f"Image order changed in {name}")
        conversations = [
            json.loads(line)
            for line in (experiment_dir / name / "conversations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if [item["image"]["source_id"] for item in conversations] != planned_ids:
            raise ValueError(f"Public conversation order changed in {name}")
        expected = [
            (
                plan["model_config"][image["generator_role"]]["repo_id"],
                image["language"],
            )
            for image in plan["images"]
        ]
        actual = [(item["generation_model"], item["target_language"]) for item in conversations]
        if actual != expected:
            raise ValueError(f"Generator or language assignment changed in {name}")
        accepted[name] = {
            item["source_id"] for item in rows if item["status"] == "QUALITY_CANDIDATE"
        }
        attempted = sum(item["attempted_turns"] for item in rows)
        committed = sum(item["committed_turns"] for item in rows)
        if (
            report["conversations"] != len(rows)
            or report["attempted_turns"] != attempted
            or report["committed_turns"] != committed
            or report["completed_conversations"] != len(accepted[name])
        ):
            raise ValueError(f"Diagnostic counts disagree in {name}")
        duration = timing["ended_at"] - timing["started_at"]
        if duration <= 0:
            raise ValueError(f"Invalid phase duration: {name}")
        phases[name] = {
            "workers": count,
            "images": len(rows),
            "status_counts": dict(Counter(item["status"] for item in rows)),
            "attempted_turns": attempted,
            "committed_turns": committed,
            "completed_conversations": len(accepted[name]),
            "zero_committed_images": sum(item["committed_turns"] == 0 for item in rows),
            "model_calls": report["model_calls"],
            "retry_calls": report["retry_calls"],
            "output_tokens": sum(item["output_tokens"] for item in rows),
            "duration_seconds": duration,
            "peak_gpu_memory_mib": timing["peak_gpu_memory_mib"],
            "quality_source_ids": sorted(accepted[name]),
            "human_approved_completed_conversations": None,
        }

    by_workers: dict[str, dict[str, Any]] = {}
    for count in (1, 2, 4):
        names = [name for name, workers in zip(PHASES, WORKERS, strict=True) if workers == count]
        durations = [phases[name]["duration_seconds"] for name in names]
        completed = [phases[name]["completed_conversations"] for name in names]
        repeated = accepted[names[0]] & accepted[names[1]]
        by_workers[str(count)] = {
            "phases": names,
            "median_duration_seconds": median(durations),
            "completed_conversations": completed,
            "repeat_quality_overlap": len(repeated),
            "auto_quality_per_synthesis_gpu_hour": sum(completed) / (sum(durations) / 3600),
            "human_approved_per_allocated_gpu_hour": None,
        }
    return {
        "source_commit": plan["source_commit"],
        "images_per_phase": len(planned_ids),
        "total_allocated_gpu_hours": (progress["ended_at"] - progress["started_at"]) / 3600,
        "phases": phases,
        "by_workers": by_workers,
        "human_quality_and_error_rates": None,
        "interpretation": "Descriptive automatic outcomes only; independent human ballots are pending.",
    }


def main() -> None:
    """Write JSON, CSV, and Markdown descriptive reports beside private diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    report = summarize(args.experiment_dir)
    stem = args.experiment_dir / "comparison"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        columns = (
            "phase",
            "workers",
            "images",
            "attempted_turns",
            "committed_turns",
            "completed_conversations",
            "zero_committed_images",
            "model_calls",
            "retry_calls",
            "output_tokens",
            "duration_seconds",
            "peak_gpu_memory_mib",
        )
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, phase in report["phases"].items():
            writer.writerow({"phase": name, **{key: phase[key] for key in columns[1:]}})
    lines = [
        "# Image concurrency comparison",
        "",
        f"Eight fixed images per phase; total GPU allocation: {report['total_allocated_gpu_hours']:.3f} hours.",
        "",
        "| Phase | Workers | Seconds | Auto quality candidates | Committed turns | Calls | Peak GPU MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {name} | {phase['workers']} | {phase['duration_seconds']:.1f} | "
            f"{phase['completed_conversations']} | {phase['committed_turns']} | "
            f"{phase['model_calls']} | {phase['peak_gpu_memory_mib']} |"
            for name, phase in report["phases"].items()
        ),
        "",
        "Human-approved conversations per allocated GPU hour and error rates remain unmeasured.",
        "Each condition has two repeats; automatic candidate counts are descriptive only.",
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines))
    print(stem)


if __name__ == "__main__":
    main()
