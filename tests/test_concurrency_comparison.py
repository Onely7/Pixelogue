"""Balanced worker comparisons must reject drift and leave human quality unmeasured."""

from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import pytest

summarize = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "validation/compare_concurrency_runs.py")
)["summarize"]


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_concurrency_comparison_checks_frozen_inputs_and_missing_gold(tmp_path: Path) -> None:
    workers = (2, 1, 4, 4, 1, 2)
    names = [f"workers-{count}-{index}" for index, count in enumerate(workers, start=1)]
    hashes = {}
    for count in (1, 2, 4):
        config = tmp_path / "configs" / f"workers-{count}.yaml"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(f"workers: {count}\n")
        hashes[str(count)] = hashlib.sha256(config.read_bytes()).hexdigest()
    _write(
        tmp_path / "experiment-plan.json",
        {
            "phase_order": names,
            "workers": list(workers),
            "images": [{"source_id": "image-a", "generator_role": "generator_a", "language": "en"}],
            "config_sha256": hashes,
            "model_config": {"generator_a": {"repo_id": "model-a"}},
            "source_commit": "fixed",
        },
    )
    _write(
        tmp_path / "progress.json",
        {
            "success": True,
            "started_at": 0,
            "ended_at": 70,
            "phases": {
                name: {
                    "status": "COMPLETED",
                    "started_at": index * 10,
                    "ended_at": index * 10 + 10,
                    "peak_gpu_memory_mib": 10 + count,
                }
                for index, (name, count) in enumerate(zip(names, workers, strict=True))
            },
        },
    )
    for index, name in enumerate(names):
        (tmp_path / name).mkdir()
        (tmp_path / name / "conversations.jsonl").write_text(
            json.dumps(
                {
                    "image": {"source_id": "image-a"},
                    "generation_model": "model-a",
                    "target_language": "en",
                }
            )
            + "\n"
        )
        accepted = index in (0, 1, 5)
        _write(
            tmp_path / name / "diagnostics.json",
            {
                "conversations": 1,
                "attempted_turns": 2,
                "committed_turns": 1,
                "completed_conversations": int(accepted),
                "model_calls": 5,
                "retry_calls": 0,
                "rows": [
                    {
                        "source_id": "image-a",
                        "status": "QUALITY_CANDIDATE" if accepted else "REJECTED",
                        "attempted_turns": 2,
                        "committed_turns": 1,
                        "output_tokens": 20,
                    }
                ],
            },
        )
    report = summarize(tmp_path)
    assert report["by_workers"]["2"]["repeat_quality_overlap"] == 1
    assert report["by_workers"]["4"]["completed_conversations"] == [0, 0]
    assert report["human_quality_and_error_rates"] is None
    assert report["by_workers"]["2"]["human_approved_per_allocated_gpu_hour"] is None

    changed = tmp_path / names[0] / "conversations.jsonl"
    changed.write_text(changed.read_text().replace("model-a", "model-b"))
    with pytest.raises(ValueError, match="assignment changed"):
        summarize(tmp_path)
