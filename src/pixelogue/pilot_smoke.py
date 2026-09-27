#!/usr/bin/env python3
"""Run the pinned one-GPU pilot with supervised servers and integrity checks."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/gpu-watch-pilot.yaml"
SERVERS: list[tuple[subprocess.Popen[str], TextIO]] = []
stop = False


def _parent_alive() -> bool:
    """Stop model servers if the GPU-hour supervisor disappears."""
    expected = os.environ.get("PIXELOGUE_WATCH_PID")
    return expected is None or os.getppid() == int(expected)


def _signal_stop(_signum: int, _frame: object) -> None:
    global stop
    stop = True


def _stop_servers() -> None:
    for process, _log in reversed(SERVERS):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
    for process, log in reversed(SERVERS):
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.close()


def _port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def _start_server(name: str, configuration: str, output_dir: Path) -> subprocess.Popen[str]:
    path = output_dir / f"{name}.log"
    log = path.open("w", encoding="utf-8")
    environment = {
        **os.environ,
        "HF_HOME": "/var/tmp/pixelogue-hf",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    process = subprocess.Popen(
        [
            "uv",
            "run",
            "--project",
            "runtime/vllm",
            "--locked",
            "vllm",
            "serve",
            "--config",
            configuration,
        ],
        cwd=ROOT,
        env=environment,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    SERVERS.append((process, log))
    with (output_dir / "server-pids.jsonl").open("a", encoding="utf-8") as pids:
        pids.write(json.dumps({"pid": process.pid, "name": name}) + "\n")
    print(f"{name} launched: PID {process.pid}, {path}", flush=True)
    return process


def _wait_model(process: subprocess.Popen[str], port: int, expected: str, seconds: int) -> None:
    deadline = time.monotonic() + seconds
    last_error = "no HTTP response"
    while time.monotonic() < deadline and not stop and _parent_alive():
        if process.poll() is not None:
            raise RuntimeError(
                f"Server exited before readiness: {expected}, code={process.returncode}"
            )
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as response:
                body = json.load(response)
            names = {item["id"] for item in body["data"]}
            if expected in names:
                print(f"model ready: {expected}", flush=True)
                return
            last_error = f"served names: {sorted(names)}"
        except (OSError, ValueError, KeyError, urllib.error.URLError) as exc:
            last_error = str(exc)
        time.sleep(5)
    raise RuntimeError(f"Model readiness timed out: {expected}; {last_error}")


def _run_logged(
    command: list[str], log_path: Path, timeout: int, progress_path: Path | None = None
) -> None:
    print(f"running: {' '.join(command)}; log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        last_report = started
        while (
            process.poll() is None
            and not stop
            and _parent_alive()
            and time.monotonic() - started < timeout
        ):
            time.sleep(5)
            if time.monotonic() - last_report >= 30:
                progress = progress_path or log_path.parent / "conversations.jsonl"
                lines = sum(1 for _ in progress.open(encoding="utf-8")) if progress.exists() else 0
                alive = [item.poll() is None for item, _ in SERVERS]
                print(f"progress rows={lines}, servers_alive={alive}", flush=True)
                if not all(alive):
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait()
                    raise RuntimeError("Model server exited during the pilot")
                last_report = time.monotonic()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError("Pilot command stopped or timed out")
        if process.returncode != 0:
            raise RuntimeError(f"Pilot command failed: code={process.returncode}, log={log_path}")


def _check_results(output_dir: Path, expected: int) -> None:
    rows = [
        json.loads(line)
        for line in (output_dir / "conversations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    counts: dict[str, int] = {}
    for row in rows:
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    report = {"expected": expected, "actual": len(rows), "status_counts": counts}
    (output_dir / "smoke-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"pilot summary: {report}", flush=True)
    if len(rows) != expected or counts.get("ERROR", 0):
        raise RuntimeError("Pilot has missing rows or terminal execution errors")


def main() -> None:
    """Start pinned servers, exercise evaluation-only inputs, then release the GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise RuntimeError("Exactly one physical GPU must be selected explicitly")
    if not _port_is_free(18002) or not _port_is_free(18000):
        raise RuntimeError("Pilot ports 18002 and 18000 are already occupied")
    if not Path("/var/tmp/pixelogue-hf/hub/models--Qwen--Qwen3.5-9B").exists():
        raise RuntimeError("The pinned local model cache is missing")
    prepared = ROOT / "artifacts/prepared-open-images"
    expected = sum(1 for _ in (prepared / "images.jsonl").open(encoding="utf-8"))
    if not expected or not (prepared / "manifest.json").exists():
        raise RuntimeError("Prepared evaluation-only images are missing")
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    output_dir: Path = args.output_dir
    try:
        generator = _start_server("generator", "runtime/vllm/generator-gpu-watch.yaml", output_dir)
        _wait_model(generator, 18002, "Qwen/Qwen3.5-9B", 1800)
        selector = _start_server("selector", "runtime/vllm/selector-gpu-watch.yaml", output_dir)
        _wait_model(selector, 18000, "Qwen/Qwen3.5-2B", 900)
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "doctor",
                "--config",
                CONFIG,
                "--check-servers",
                "--output",
                str(output_dir / "doctor.json"),
            ],
            output_dir / "doctor.log",
            90,
        )
        first_image = output_dir / "first-image" / "conversations.jsonl"
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "synthesize",
                "--config",
                "configs/gpu-watch-smoke.yaml",
                "--images",
                str(prepared / "images.jsonl"),
                "--artifact-root",
                str(prepared),
                "--run-id",
                f"{args.run_id}-first-image",
                "--workers",
                "1",
                "--output",
                str(first_image),
            ],
            output_dir / "first-image.log",
            1800,
            first_image,
        )
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "replay",
                "--config",
                "configs/gpu-watch-smoke.yaml",
                "--run-id",
                f"{args.run_id}-first-image",
            ],
            output_dir / "first-image-replay.log",
            300,
        )
        _check_results(output_dir / "first-image", 1)
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "synthesize",
                "--config",
                CONFIG,
                "--images",
                str(prepared / "images.jsonl"),
                "--artifact-root",
                str(prepared),
                "--run-id",
                args.run_id,
                "--workers",
                "2",
                "--output",
                str(output_dir / "conversations.jsonl"),
            ],
            output_dir / "synthesize.log",
            5400,
            output_dir / "conversations.jsonl",
        )
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "replay",
                "--config",
                CONFIG,
                "--run-id",
                args.run_id,
            ],
            output_dir / "replay.log",
            300,
        )
        _check_results(output_dir, expected)
    finally:
        _stop_servers()


if __name__ == "__main__":
    main()
