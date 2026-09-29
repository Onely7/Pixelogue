#!/usr/bin/env python3
"""Join private diverse-evaluation categories to post-run diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = (
    "category_number",
    "category",
    "source_id",
    "status",
    "committed_turns",
    "task_ids",
    "stop_stage",
    "stop_category",
    "stop_reason",
    "model_calls",
    "retry_calls",
    "invalid_calls",
    "duration_ms",
    "input_tokens",
    "output_tokens",
    "cost_usd",
    "token_usage_missing_calls",
    "file_page_url",
)


def build_report(
    diagnostics: dict,
    manifest: list[dict],
    conversations: list[dict] | None = None,
    *,
    source_prefix: str = "commons-eval",
) -> tuple[dict, list[dict]]:
    """Produce a complete category ledger from one run's diagnostic rows."""
    if source_prefix not in {"commons-eval", "commons-holdout"}:
        raise ValueError(f"Unsupported Commons source prefix: {source_prefix}")
    by_source = {f"{source_prefix}:{item['commons_page_id']}": item for item in manifest}
    if len(by_source) != len(manifest):
        raise ValueError("The pinned Commons manifest contains duplicate page IDs")
    observed: dict[str, dict] = {}
    for row in diagnostics["rows"]:
        source_id = row["source_id"]
        if source_id not in by_source:
            raise ValueError(f"Unpinned source in diagnostics: {source_id}")
        if source_id in observed:
            raise ValueError(f"Duplicate source in diagnostics: {source_id}")
        observed[source_id] = row
    turns_by_source = {
        conversation["image"]["source_id"]: tuple(
            turn["instruction"]["task_id"]
            for turn in conversation["turns"]
            if turn["status"] == "COMMITTED"
        )
        for conversation in conversations or ()
    }
    if conversations is not None:
        if len(turns_by_source) != len(conversations):
            raise ValueError("Conversation output contains duplicate sources")
        if set(turns_by_source) != set(observed):
            raise ValueError("Conversation output differs from diagnostic sources")
        for source_id, task_ids in turns_by_source.items():
            if len(task_ids) != observed[source_id]["committed_turns"]:
                raise ValueError(f"Committed turns differ from diagnostics: {source_id}")
    rows = []
    for source_id, item in sorted(by_source.items(), key=lambda pair: pair[1]["category_number"]):
        outcome = observed.get(source_id)
        rows.append(
            {
                "category_number": item["category_number"],
                "category": item["category"],
                "source_id": source_id,
                "status": outcome["status"] if outcome else "NOT_RUN",
                "committed_turns": outcome["committed_turns"] if outcome else None,
                "task_ids": ",".join(turns_by_source.get(source_id, ())),
                "stop_stage": outcome["stop_stage"] if outcome else None,
                "stop_category": outcome["stop_category"] if outcome else None,
                "stop_reason": outcome["stop_reason"] if outcome else None,
                "model_calls": outcome["model_calls"] if outcome else None,
                "retry_calls": outcome["retry_calls"] if outcome else None,
                "invalid_calls": outcome["invalid_calls"] if outcome else None,
                "duration_ms": outcome["duration_ms"] if outcome else None,
                "input_tokens": outcome["input_tokens"] if outcome else None,
                "output_tokens": outcome["output_tokens"] if outcome else None,
                "cost_usd": outcome.get("cost_usd") if outcome else None,
                "token_usage_missing_calls": (
                    outcome.get("token_usage_missing_calls") if outcome else None
                ),
                "file_page_url": item["file_page_url"],
            }
        )
    summary = {
        "run_id": diagnostics["run_id"],
        "categories": len(rows),
        "attempted": len(observed),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "committed_task_counts": dict(
            sorted(
                Counter(
                    task_id for task_ids in turns_by_source.values() for task_id in task_ids
                ).items()
            )
        ),
        "stop_category_counts": dict(
            sorted(
                Counter(str(row["stop_category"]) for row in rows if row["stop_category"]).items()
            )
        ),
        "candidate_turns": sum(
            int(row["committed_turns"] or 0) for row in rows if row["status"] == "QUALITY_CANDIDATE"
        ),
        "model_calls": diagnostics["model_calls"],
        "retry_calls": diagnostics["retry_calls"],
        "invalid_calls": diagnostics["invalid_calls"],
        "stage_reach_images": diagnostics["stage_reach_images"],
        "cost_usd": diagnostics.get("cost_usd"),
        "token_usage_missing_calls": diagnostics.get("token_usage_missing_calls"),
        "gold_labels": False,
        "human_audit_complete": False,
    }
    return summary, rows


def main() -> None:
    """Write JSON, CSV, and Markdown reports without exposing labels to models."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--conversations", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-stem", required=True, type=Path)
    parser.add_argument(
        "--source-prefix", choices=("commons-eval", "commons-holdout"), default="commons-eval"
    )
    args = parser.parse_args()
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    conversations = [
        json.loads(line) for line in args.conversations.read_text(encoding="utf-8").splitlines()
    ]
    manifest = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    summary, rows = build_report(
        diagnostics, manifest, conversations, source_prefix=args.source_prefix
    )
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    args.output_stem.with_suffix(".json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with args.output_stem.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        f"# Diverse evaluation: {summary['run_id']}",
        "",
        f"Attempted: {summary['attempted']}/{summary['categories']}; "
        f"status counts: {summary['status_counts']}.",
        "",
        "These are automatic pipeline outcomes. No independent gold labels or human audit are attached.",
        "",
        "| # | Category | Status | Turns | Task IDs | Stop stage | Model calls | Source |",
        "|---:|---|---|---:|---|---|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['category_number']} | {row['category']} | {row['status']} | "
            f"{row['committed_turns'] if row['committed_turns'] is not None else ''} | "
            f"{row['task_ids']} | "
            f"{row['stop_stage'] or ''} | "
            f"{row['model_calls'] if row['model_calls'] is not None else ''} | "
            f"[Commons]({row['file_page_url']}) |"
        )
    args.output_stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
