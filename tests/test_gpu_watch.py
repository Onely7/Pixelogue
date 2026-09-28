"""GPU reservation refuses ambiguous devices and accounts for active allocations."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

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
    assert state["allocation_intervals"][0]["phase"] == "reservation"
    assert state["allocation_intervals"][0]["gpu_seconds"] == 100.0
    assert gpu_watch._remaining(state) == gpu_watch.MAX_GPU_SECONDS - 200


def test_stale_pilot_queues_post_job_reservation_on_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(gpu_watch, "STATE", state_path)
    monkeypatch.setattr(gpu_watch.time, "time", lambda: 1000.0)
    state_path.write_text(
        json.dumps(
            {
                "used_gpu_seconds": 100.0,
                "active": {"phase": "pilot", "gpu": 4, "pid": 99999999, "started_at": 900.0},
                "pilot_attempted": False,
            }
        ),
        encoding="utf-8",
    )
    state = gpu_watch._load()
    assert state["used_gpu_seconds"] == 200.0
    assert state["active"] is None
    assert state["pilot_attempted"] is True
    assert state["pilot_exit_code"] == 1
    assert state["post_job_reservation"]["status"] == "pending"
    assert gpu_watch._post_job_candidates({4}, set(), state) == [4]


def test_named_validation_campaign_extends_budget_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_watch, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(gpu_watch.time, "time", lambda: 1000.0)
    state = {"used_gpu_seconds": 14340.0, "active": None, "pilot_attempted": True}
    gpu_watch._authorize_campaign(state, "recheck-20260928", 2.0)
    assert state["authorized_gpu_seconds"] == 21600.0
    assert gpu_watch._remaining(state) == 7260.0
    assert state["pilot_attempted"] is False
    state["pilot_attempted"] = True
    gpu_watch._authorize_campaign(state, "recheck-20260928", 2.0)
    assert state["authorized_gpu_seconds"] == 21600.0
    assert state["pilot_attempted"] is True
    with pytest.raises(RuntimeError, match="different GPU budget"):
        gpu_watch._authorize_campaign(state, "recheck-20260928", 3.0)


def test_failed_pilot_can_retry_only_within_existing_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_watch, "STATE", tmp_path / "state.json")
    state = {
        "used_gpu_seconds": 100.0,
        "authorized_gpu_seconds": 1000.0,
        "active": None,
        "pilot_attempted": True,
        "pilot_exit_code": 1,
        "campaigns": [{"id": "pilot", "additional_gpu_seconds": 600}],
    }
    with pytest.raises(RuntimeError, match="failed pilot"):
        gpu_watch._retry_failed_pilot(state, "unknown")
    gpu_watch._retry_failed_pilot(state, "pilot")
    assert state["pilot_attempted"] is False
    assert state["prior_pilot_exit_code"] == 1
    assert state["authorized_gpu_seconds"] == 1000.0
    with pytest.raises(RuntimeError, match="failed pilot"):
        gpu_watch._retry_failed_pilot(state, "pilot")


def test_completed_pilot_can_rerun_without_extending_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_watch, "STATE", tmp_path / "state.json")
    state = {
        "used_gpu_seconds": 100.0,
        "authorized_gpu_seconds": 1000.0,
        "active": None,
        "pilot_attempted": True,
        "pilot_exit_code": 0,
        "campaigns": [{"id": "pilot", "additional_gpu_seconds": 600}],
    }
    with pytest.raises(RuntimeError, match="completed pilot"):
        gpu_watch._rerun_completed_pilot(state, "unknown")
    gpu_watch._rerun_completed_pilot(state, "pilot")
    assert state["pilot_attempted"] is False
    assert state["pilot_exit_code"] is None
    assert state["authorized_gpu_seconds"] == 1000.0
    with pytest.raises(RuntimeError, match="completed pilot"):
        gpu_watch._rerun_completed_pilot(state, "pilot")


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


def test_generator_handoff_keeps_reservation_until_acknowledged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PIXELOGUE_WATCH_PID", "12345")
    monkeypatch.setattr(pilot_smoke, "_parent_alive", lambda: True)
    monkeypatch.setattr(pilot_smoke, "stop", False)

    def acknowledge(_seconds: float) -> None:
        assert (tmp_path / "handoff-request").read_text() == "generator-ready\n"
        (tmp_path / "handoff-complete").write_text("released\n")

    monkeypatch.setattr(pilot_smoke.time, "sleep", acknowledge)
    pilot_smoke._request_holder_handoff(tmp_path)


def test_diverse_pilot_uses_its_own_prepared_manifest(tmp_path: Path) -> None:
    command = pilot_smoke._synthesis_command(
        "diverse-run",
        tmp_path / "conversations.jsonl",
        config="configs/gpu-watch-diverse.yaml",
        prepared=tmp_path / "prepared-diverse",
    )
    assert command[command.index("--config") + 1] == "configs/gpu-watch-diverse.yaml"
    assert command[command.index("--images") + 1] == str(tmp_path / "prepared-diverse/images.jsonl")
    assert command[command.index("--artifact-root") + 1] == str(tmp_path / "prepared-diverse")


@pytest.mark.parametrize("exit_code", [0, 1])
def test_pilot_queues_reservation_immediately_after_either_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exit_code: int
) -> None:
    monkeypatch.setattr(gpu_watch, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(gpu_watch.time, "time", lambda: 1234.0)
    state: dict[str, Any] = {
        "used_gpu_seconds": 0.0,
        "active": None,
        "pilot_attempted": False,
    }

    def finish(_gpu: int, saved: dict, _holder: object) -> None:
        saved["pilot_attempted"] = True
        saved["pilot_exit_code"] = exit_code

    monkeypatch.setattr(gpu_watch, "_pilot", finish)
    gpu_watch._run_pilot_and_queue(4, state, cast(subprocess.Popen[str], SimpleNamespace(pid=123)))
    request = state["post_job_reservation"]
    assert request == {
        "gpu": 4,
        "requested_at": 1234.0,
        "pilot_exit_code": exit_code,
        "status": "pending",
    }
    assert gpu_watch._post_job_candidates({4}, set(), state) == [4]
    assert json.loads((tmp_path / "state.json").read_text())["post_job_reservation"] == request


def test_supervisor_failure_still_queues_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gpu_watch, "STATE", tmp_path / "state.json")
    state: dict[str, Any] = {
        "used_gpu_seconds": 0.0,
        "active": None,
        "pilot_attempted": False,
    }

    def fail(_gpu: int, _saved: dict, _holder: object) -> None:
        raise RuntimeError("pilot failed before inference")

    monkeypatch.setattr(gpu_watch, "_pilot", fail)
    gpu_watch._run_pilot_and_queue(2, state, cast(subprocess.Popen[str], SimpleNamespace(pid=123)))
    assert state["pilot_exit_code"] == 1
    assert state["post_job_reservation"]["status"] == "pending"
    assert gpu_watch._post_job_candidates({2, 4}, set(), state) == [2, 4]
