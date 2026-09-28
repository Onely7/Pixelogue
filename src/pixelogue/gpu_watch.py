#!/usr/bin/env python3
"""Watch local GPUs and run one bounded pilot before restoring a memory reservation."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STATE = ROOT / "artifacts/gpu-watch/state.json"
PYTHON_WITH_TORCH = ROOT / "runtime/vllm/.venv/bin/python"
MAX_GPU_SECONDS = 4 * 3600
stop = False


def _signal_stop(_signum: int, _frame: object) -> None:
    global stop
    stop = True


def _save(state: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True, indent=2), encoding="utf-8")
    temporary.replace(STATE)


def _load() -> dict[str, Any]:
    if not STATE.exists():
        return {"used_gpu_seconds": 0.0, "active": None, "pilot_attempted": False}
    state = json.loads(STATE.read_text(encoding="utf-8"))
    active = state.get("active")
    if active is not None:
        pid = int(active["pid"])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            state["used_gpu_seconds"] += max(0, time.time() - float(active["started_at"]))
            state["active"] = None
            _save(state)
        else:
            raise RuntimeError(f"An earlier reservation or pilot is still active: PID {pid}")
    return state


def _remaining(state: dict[str, Any]) -> float:
    used = float(state["used_gpu_seconds"])
    active = state.get("active")
    if active is not None:
        used += max(0, time.time() - float(active["started_at"]))
    return float(state.get("authorized_gpu_seconds", MAX_GPU_SECONDS)) - used


def _authorize_campaign(state: dict[str, Any], campaign_id: str, additional_hours: float) -> None:
    """Extend the preserved GPU ledger once for an explicitly named validation round."""
    additional_seconds = round(additional_hours * 3600)
    campaigns = state.setdefault("campaigns", [])
    previous = next((item for item in campaigns if item["id"] == campaign_id), None)
    if previous is not None:
        if previous["additional_gpu_seconds"] != additional_seconds:
            raise RuntimeError("Campaign ID already has a different GPU budget")
        return
    if state.get("active") is not None:
        raise RuntimeError("Cannot extend the budget during an active allocation")
    state["authorized_gpu_seconds"] = (
        float(state.get("authorized_gpu_seconds", MAX_GPU_SECONDS)) + additional_seconds
    )
    state["pilot_attempted"] = False
    state["pilot_exit_code"] = None
    campaigns.append(
        {"id": campaign_id, "additional_gpu_seconds": additional_seconds, "started_at": time.time()}
    )
    _save(state)
    print(f"authorized campaign {campaign_id}: +{additional_hours:.3f} GPU-hours", flush=True)


def _retry_failed_pilot(state: dict[str, Any], campaign_id: str | None) -> None:
    """Permit another diagnostic run within the existing campaign budget."""
    if (
        campaign_id is None
        or not any(item["id"] == campaign_id for item in state.get("campaigns", []))
        or state.get("active") is not None
        or not state.get("pilot_attempted")
        or state.get("pilot_exit_code") in (None, 0)
        or _remaining(state) <= 600
    ):
        raise RuntimeError("Retry requires a failed pilot and remaining campaign budget")
    state["pilot_attempted"] = False
    state["prior_pilot_exit_code"] = state["pilot_exit_code"]
    state["pilot_exit_code"] = None
    _save(state)
    print(f"retrying failed pilot under campaign {campaign_id}", flush=True)


def _begin(state: dict[str, Any], phase: str, gpu: int, pid: int) -> None:
    state["active"] = {
        "phase": phase,
        "gpu": gpu,
        "pid": pid,
        "started_at": time.time(),
    }
    _save(state)


def _end(state: dict[str, Any]) -> None:
    active = state["active"]
    if active is None:
        return
    state["used_gpu_seconds"] += max(0, time.time() - float(active["started_at"]))
    state["active"] = None
    _save(state)


def _metrics() -> list[dict[str, int | str]]:
    """Require a successful physical-device and process query before admission."""
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    rows = subprocess.check_output(command, text=True, timeout=10).strip().splitlines()
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    occupied = {line.split(",", 1)[0].strip() for line in processes.splitlines() if "," in line}
    result: list[dict[str, int | str]] = []
    for row in rows:
        values = [value.strip() for value in row.split(",")]
        if len(values) != 5:
            raise ValueError(f"Unexpected nvidia-smi row: {row}")
        index, uuid, used, total, utilization = values
        result.append(
            {
                "index": int(index),
                "uuid": uuid,
                "used_mib": int(used),
                "total_mib": int(total),
                "utilization": int(utilization),
                "compute_process": int(uuid in occupied),
            }
        )
    return result


def idle_indices(rows: list[dict[str, int | str]]) -> set[int]:
    """Match the doctor's zero-utilization, sub-GiB definition and process table."""
    return {
        int(row["index"])
        for row in rows
        if int(row["used_mib"]) < 1024
        and int(row["utilization"]) == 0
        and not int(row["compute_process"])
    }


def _wait_for_handoff(gpu: int, seconds: int = 30) -> bool:
    """Allow the released CUDA context's utilization sample to settle."""
    deadline = time.monotonic() + seconds
    while not stop and time.monotonic() < deadline:
        row = next((item for item in _metrics() if int(item["index"]) == gpu), None)
        if row is None or int(row["used_mib"]) >= 1024 or int(row["compute_process"]):
            return False
        if int(row["utilization"]) == 0:
            return True
        time.sleep(1)
    return False


def _doctor_ready() -> bool:
    output = ROOT / "artifacts/gpu-watch/doctor.json"
    result = subprocess.run(
        [
            "uv",
            "run",
            "--locked",
            "pixelogue",
            "doctor",
            "--config",
            "configs/gpu-watch-pilot.yaml",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    if result.returncode != 0:
        print(f"doctor not ready: {result.returncode}", flush=True)
        return False
    return json.loads(output.read_text(encoding="utf-8"))["ready"] is True


def _holder(gpu: int, state: dict[str, Any]) -> subprocess.Popen[str] | None:
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        [
            str(PYTHON_WITH_TORCH),
            str(Path(__file__)),
            "hold",
            "--memory-fraction",
            "0.90",
            "--parent-pid",
            str(os.getpid()),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _begin(state, "reservation", gpu, process.pid)
    assert process.stdout is not None
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and process.poll() is None and _remaining(state) > 600:
        ready, _, _ = select.select([process.stdout], [], [], 1)
        if ready:
            line = process.stdout.readline().strip()
            if line:
                print(f"holder gpu={gpu}: {line}", flush=True)
            if line.startswith("READY "):
                return process
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    _end(state)
    return None


def _release_holder(process: subprocess.Popen[str], state: dict[str, Any]) -> None:
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    if process.stdout is not None:
        process.stdout.close()
    _end(state)


def _pilot(gpu: int, state: dict[str, Any]) -> None:
    run_id = f"gpu-watch-pilot-{int(time.time())}"
    output = ROOT / "artifacts/gpu-watch" / run_id
    output.mkdir(parents=True, exist_ok=False)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONUNBUFFERED": "1",
        "PIXELOGUE_WATCH_PID": str(os.getpid()),
    }
    with (output / "pilot.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "src/pixelogue/pilot_smoke.py"),
                "--run-id",
                run_id,
                "--output-dir",
                str(output),
            ],
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _begin(state, "pilot", gpu, process.pid)
        print(f"pilot started: GPU {gpu}, PID {process.pid}, {output}", flush=True)
        try:
            while process.poll() is None and _remaining(state) > 60 and not stop:
                time.sleep(5)
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=35)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            print(f"pilot ended: exit={process.returncode}, {output}", flush=True)
            pids_path = output / "server-pids.jsonl"
            if pids_path.exists():
                for line in pids_path.read_text(encoding="utf-8").splitlines():
                    pid = int(json.loads(line)["pid"])
                    command = Path(f"/proc/{pid}/cmdline")
                    if command.exists() and b"vllm" in command.read_bytes():
                        print(f"stopping orphaned model server group {pid}", flush=True)
                        os.killpg(pid, signal.SIGTERM)
            state["pilot_attempted"] = True
            state["pilot_exit_code"] = process.returncode
        finally:
            _end(state)
            _save(state)


def _hold_visible_gpu(memory_fraction: float, parent_pid: int) -> None:
    """Adapt the attached memory holder without its continuous GEMM load."""
    torch = importlib.import_module("torch")

    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") or not os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ):
        raise RuntimeError("Exactly one explicit CUDA_VISIBLE_DEVICES value is required")
    torch.cuda.set_device(0)
    free, total = torch.cuda.mem_get_info(0)
    if total - free >= 1024 * 1024 * 1024:
        raise RuntimeError("GPU became occupied before the reservation")
    allocations = []
    target = int(total * memory_fraction)
    while torch.cuda.memory_allocated(0) < target:
        remaining = target - torch.cuda.memory_allocated(0)
        chunk = min(256 * 1024 * 1024, remaining)
        if chunk < 4 * 1024 * 1024:
            break
        allocations.append(torch.empty(chunk, dtype=torch.uint8, device="cuda:0"))
    torch.cuda.synchronize()
    own = torch.cuda.memory_allocated(0)
    if own < total * 0.85:
        raise RuntimeError("Reservation did not acquire sufficient GPU memory")
    print(f"READY allocated_mib={own // (1024 * 1024)}", flush=True)
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    while not stop and os.getppid() == parent_pid:
        time.sleep(1)


def _watch(
    poll_seconds: int,
    campaign_id: str | None,
    additional_hours: float,
    reserve_only: bool,
    retry_failed_pilot: bool,
) -> None:
    if not PYTHON_WITH_TORCH.is_file():
        raise RuntimeError("Install the separate runtime/vllm lock first")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with (STATE.parent / "watch.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = _load()
        if campaign_id is not None and additional_hours > 0:
            _authorize_campaign(state, campaign_id, additional_hours)
        if retry_failed_pilot:
            _retry_failed_pilot(state, campaign_id)
        last: set[int] = set()
        last_heartbeat = time.monotonic()
        print(
            f"watching local GPUs; remaining GPU-hours={_remaining(state) / 3600:.3f}", flush=True
        )
        while not stop and _remaining(state) > 600:
            try:
                current = idle_indices(_metrics())
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                print(f"GPU inspection failed: {exc}", flush=True)
                current = set()
            state["last_scan_at"] = time.time()
            state["last_idle_indices"] = sorted(current)
            _save(state)
            if time.monotonic() - last_heartbeat >= 60:
                print(
                    f"watch heartbeat: idle={sorted(current)}, "
                    f"remaining_gpu_hours={_remaining(state) / 3600:.3f}",
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            eligible = sorted(current & last)
            last = current
            if not eligible:
                time.sleep(poll_seconds)
                continue
            gpu = eligible[0]
            if not _doctor_ready():
                time.sleep(poll_seconds)
                continue
            holder = _holder(gpu, state)
            if holder is None:
                last = set()
                time.sleep(poll_seconds)
                continue
            if not state["pilot_attempted"] and not reserve_only:
                _release_holder(holder, state)
                if not _wait_for_handoff(gpu):
                    print("GPU handoff lost; returning to watch", flush=True)
                    last = set()
                    continue
                _pilot(gpu, state)
                last = set()
                continue
            purpose = "diagnostic" if reserve_only else "post-job"
            print(
                f"{purpose} GPU {gpu} reserved; holding until remaining budget expires", flush=True
            )
            try:
                while holder.poll() is None and _remaining(state) > 60 and not stop:
                    time.sleep(5)
            finally:
                _release_holder(holder, state)
            last = set()
        print(f"watch stopped; used GPU-hours={state['used_gpu_seconds'] / 3600:.3f}", flush=True)


def main() -> None:
    """Select the bounded local watcher or its single-device holder mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("watch", "hold"))
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--memory-fraction", type=float, default=0.90)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--campaign-id")
    parser.add_argument("--additional-gpu-hours", type=float, default=0.0)
    parser.add_argument("--reserve-only", action="store_true")
    parser.add_argument("--retry-failed-pilot", action="store_true")
    args = parser.parse_args()
    if not 5 <= args.poll_seconds <= 300 or not 0.85 <= args.memory_fraction <= 0.92:
        parser.error("Invalid polling or reservation fraction")
    if not 0 <= args.additional_gpu_hours <= 4 or (
        args.additional_gpu_hours and not args.campaign_id
    ):
        parser.error("Budget extension must be at most four GPU-hours with a campaign ID")
    if args.retry_failed_pilot and (args.reserve_only or args.mode != "watch"):
        parser.error("Pilot retry requires watch mode without --reserve-only")
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    if args.mode == "hold":
        if args.parent_pid is None or args.parent_pid <= 1:
            parser.error("Holder requires its supervising parent PID")
        _hold_visible_gpu(args.memory_fraction, args.parent_pid)
    else:
        _watch(
            args.poll_seconds,
            args.campaign_id,
            args.additional_gpu_hours,
            args.reserve_only,
            args.retry_failed_pilot,
        )


if __name__ == "__main__":
    main()
