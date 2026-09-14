"""Read-only inference timing summaries for completed or active runs."""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from pixelogue.errors import ExternalInputError


class StageTiming(BaseModel):
    """Aggregate timing and token use for one model and pipeline stage."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage: str
    model_repo: str
    calls: int
    input_tokens: int
    output_tokens: int
    total_duration_ms: int
    mean_duration_ms: float


class InferenceProfile(BaseModel):
    """Run-level model-call profile with its measurement method."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    measurement: Literal["recorded_duration", "completion_gap_estimate"]
    completed_calls: int
    wall_duration_ms: int
    calls_per_minute: float
    stages: tuple[StageTiming, ...]


def profile_database(database: Path) -> InferenceProfile:
    """Summarize model calls without taking the run's writer lock.

    New runs store HTTP request durations directly. Older ledgers fall back to the time between
    consecutive response-artifact commits, which is meaningful only as a serial-run estimate.

    Raises:
        ExternalInputError: If the database is unreadable or has no completed calls.
    """
    resolved = database.resolve()
    if not resolved.is_file():
        raise ExternalInputError("PROFILE_DATABASE_MISSING", str(resolved))
    try:
        connection = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(model_call)").fetchall()
        }
        duration_expression = "mc.duration_ms" if "duration_ms" in columns else "0"
        rows = connection.execute(
            f"""SELECT mc.stage, mc.model_repo, mc.input_tokens, mc.output_tokens,
                       {duration_expression} AS duration_ms, artifact.created_at,
                       artifact.rowid AS completion_order
                FROM model_call AS mc
                JOIN artifact ON artifact.artifact_hash = mc.response_artifact_hash
                WHERE mc.status = 'COMPLETE'
                ORDER BY artifact.rowid"""
        ).fetchall()
    except sqlite3.Error as error:
        raise ExternalInputError("PROFILE_DATABASE_UNREADABLE", str(error)) from error
    finally:
        if "connection" in locals():
            connection.close()
    if not rows:
        raise ExternalInputError("PROFILE_CALLS_MISSING", "No complete model calls were recorded")

    recorded = any(row["duration_ms"] > 0 for row in rows)
    measurement: Literal["recorded_duration", "completion_gap_estimate"] = (
        "recorded_duration" if recorded else "completion_gap_estimate"
    )
    aggregate: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    previous_completion: datetime | None = None
    for row in rows:
        completed = datetime.fromisoformat(row["created_at"])
        duration_ms = int(row["duration_ms"])
        if not recorded:
            duration_ms = (
                0
                if previous_completion is None
                else max(0, round((completed - previous_completion).total_seconds() * 1000))
            )
        previous_completion = completed
        values = aggregate[(row["stage"], row["model_repo"])]
        values[0] += 1
        values[1] += int(row["input_tokens"])
        values[2] += int(row["output_tokens"])
        values[3] += duration_ms

    first = datetime.fromisoformat(rows[0]["created_at"])
    last = datetime.fromisoformat(rows[-1]["created_at"])
    wall_duration_ms = max(0, round((last - first).total_seconds() * 1000))
    elapsed_minutes = wall_duration_ms / 60_000
    stages = tuple(
        StageTiming(
            stage=stage,
            model_repo=model_repo,
            calls=values[0],
            input_tokens=values[1],
            output_tokens=values[2],
            total_duration_ms=values[3],
            mean_duration_ms=round(values[3] / values[0], 2),
        )
        for (stage, model_repo), values in sorted(
            aggregate.items(),
            key=lambda item: (-item[1][3], item[0]),
        )
    )
    return InferenceProfile(
        measurement=measurement,
        completed_calls=len(rows),
        wall_duration_ms=wall_duration_ms,
        calls_per_minute=round(len(rows) / elapsed_minutes, 2) if elapsed_minutes else 0.0,
        stages=stages,
    )
