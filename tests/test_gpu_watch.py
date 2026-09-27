"""GPU reservation refuses ambiguous devices and accounts for active allocations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixelogue import gpu_watch, pilot_smoke


def test_idle_device_needs_low_memory_zero_utilization_and_no_compute_process() -> None:
    rows: list[dict[str, int | str]] = [
        {"index": 0, "used_mib": 16, "utilization": 0, "compute_process": 0},
        {"index": 1, "used_mib": 100, "utilization": 100, "compute_process": 1},
        {"index": 2, "used_mib": 1500, "utilization": 0, "compute_process": 0},
        {"index": 3, "used_mib": 500, "utilization": 0, "compute_process": 1},
    ]
    assert gpu_watch.idle_indices(rows) == {0}


def test_active_reservation_consumes_budget_and_stale_pid_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(gpu_watch, "STATE", state_path)
    monkeypatch.setattr(gpu_watch.time, "time", lambda: 1000.0)
    state_path.write_text(
        json.dumps(
            {
                "used_gpu_seconds": 100.0,
                "active": {"phase": "reservation", "gpu": 0, "pid": 99999999, "started_at": 900.0},
                "pilot_attempted": False,
            }
        ),
        encoding="utf-8",
    )
    state = gpu_watch._load()
    assert state["used_gpu_seconds"] == 200.0
    assert state["active"] is None
    assert gpu_watch._remaining(state) == gpu_watch.MAX_GPU_SECONDS - 200


def test_pilot_summary_requires_all_rows_without_execution_errors(tmp_path: Path) -> None:
    output = tmp_path / "pilot"
    output.mkdir()
    path = output / "conversations.jsonl"
    path.write_text('{"status":"REJECTED"}\n{"status":"ABSTAINED"}\n', encoding="utf-8")
    pilot_smoke._check_results(output, 2)
    assert json.loads((output / "smoke-summary.json").read_text())["actual"] == 2
    path.write_text('{"status":"ERROR"}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing rows or terminal execution errors"):
        pilot_smoke._check_results(output, 2)


def test_pilot_detects_lost_budget_supervisor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIXELOGUE_WATCH_PID", "12345")
    monkeypatch.setattr(pilot_smoke.os, "getppid", lambda: 12345)
    assert pilot_smoke._parent_alive()
    monkeypatch.setattr(pilot_smoke.os, "getppid", lambda: 1)
    assert not pilot_smoke._parent_alive()
