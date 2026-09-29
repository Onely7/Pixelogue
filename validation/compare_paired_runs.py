"""Summarize a frozen ABBA synthesis experiment without using automatic labels as gold."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PHASES = ("baseline-1", "variant-1", "variant-2", "baseline-2")


def _read(path: Path) -> dict[str, Any]:
    """Load one generated experiment record."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def summarize(experiment_dir: Path) -> dict[str, Any]:
    """Compare paired outcomes and repeat variability from one completed model session."""
    plan = _read(experiment_dir / "experiment-plan.json")
    progress = _read(experiment_dir / "progress.json")
    if plan.get("phase_order") != list(PHASES) or not progress.get("success"):
        raise ValueError("The ABBA experiment has not completed as planned")
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
        phases[name] = {
            "images": len(rows),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "attempted_turns": attempted,
            "committed_turns": committed,
            "completed_conversations": len(accepted[name]),
            "zero_committed_images": sum(row["committed_turns"] == 0 for row in rows),
            "model_calls": report["model_calls"],
            "retry_calls": report["retry_calls"],
            "contract_failure_attempts": report["contract_failure_attempts"],
            "candidate_binding_rejections": report["binding_rejections"],
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "missing_token_usage_calls": report["token_usage_missing_calls"],
            "phase_gpu_hours": (timing["ended_at"] - timing["started_at"]) / 3600,
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
        "config_sha256",
        "prepared_manifest_sha256",
        "seed",
        "workers",
    )
    return {
        "plan": {
            **{key: plan[key] for key in plan_fields},
            **{
                key: plan[key]
                for key in ("intervention", "variant_config_sha256", "generator_models")
                if key in plan
            },
        },
        "images": len(planned_ids),
        "generator_and_language_assignments_identical": True,
        "total_allocated_gpu_hours": (progress["ended_at"] - progress["started_at"]) / 3600,
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
            "model_calls",
            "retry_calls",
            "contract_failure_attempts",
            "candidate_binding_rejections",
            "output_tokens",
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
        "",
        "| Phase | Attempted turns | Committed turns | Completed conversations | "
        "Zero-commit images | Model calls | Retries | Binding rejections | Phase GPU hours |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {name} | {phase['attempted_turns']} | {phase['committed_turns']} | "
            f"{phase['completed_conversations']} | {phase['zero_committed_images']} | "
            f"{phase['model_calls']} | {phase['retry_calls']} | "
            f"{phase['candidate_binding_rejections']} | {phase['phase_gpu_hours']:.3f} |"
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
        "Human-approved conversations per GPU hour and human error rates are unmeasured "
        "until independent ballots are resolved. These are automatic quality-candidate outcomes.",
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(stem)


if __name__ == "__main__":
    main()
