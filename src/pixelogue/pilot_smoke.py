#!/usr/bin/env python3
"""Run the pinned one-GPU pilot with supervised servers and integrity checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "configs/gpu-watch-pilot.yaml"
TARGETED_SOURCE_IDS = (
    "open-images-v7:1d21ad091d7d5898",
    "open-images-v7:23a34e17e221bc47",
)
PRESCREEN_SOURCE_IDS = (
    "open-images-v7:f3b66b38c1b8e6f6",
    "open-images-v7:1507692d028aaffd",
)
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


def _request_holder_handoff(output_dir: Path) -> None:
    """Keep the reservation present until the generator owns the GPU."""
    if os.environ.get("PIXELOGUE_WATCH_PID") is None:
        return
    (output_dir / "handoff-request").write_text("generator-ready\n", encoding="utf-8")
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and not stop and _parent_alive():
        if (output_dir / "handoff-complete").exists():
            print("reservation released after generator readiness", flush=True)
            return
        time.sleep(1)
    raise RuntimeError("GPU reservation handoff was not acknowledged")


def _phase(output_dir: Path, name: str) -> None:
    """Append a wall-clock boundary for model loading, inference, and shutdown."""
    with (output_dir / "phase-events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": name, "at": time.time()}) + "\n")


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


def _synthesis_command(
    run_id: str,
    output: Path,
    *,
    config: str = CONFIG,
    workers: int = 2,
    source_ids: tuple[str, ...] = (),
    prepared: Path | None = None,
) -> list[str]:
    """Build one exact synthesis invocation over the rights-checked manifest."""
    prepared = prepared or ROOT / "artifacts/prepared-open-images"
    command = [
        "uv",
        "run",
        "--locked",
        "pixelogue",
        "synthesize",
        "--config",
        config,
        "--images",
        str(prepared / "images.jsonl"),
        "--artifact-root",
        str(prepared),
        "--run-id",
        run_id,
        "--workers",
        str(workers),
        "--output",
        str(output),
    ]
    for source_id in source_ids:
        command.extend(("--source-id", source_id))
    return command


def _model_call_count(run_id: str) -> int:
    """Count persisted calls from a completed or interrupted local run."""
    path = Path("/var/tmp/pixelogue") / run_id / "run.sqlite3"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM model_call").fetchone()[0])


def _run_interrupted(
    command: list[str], log_path: Path, progress_path: Path, *, timeout: int
) -> None:
    """Stop after the first persisted image to exercise WAL and CLI resume."""
    print(f"running interruption probe: {' '.join(command)}; log={log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while (
            process.poll() is None and not stop and _parent_alive() and time.monotonic() < deadline
        ):
            time.sleep(1)
            rows = (
                sum(1 for _ in progress_path.open(encoding="utf-8"))
                if progress_path.exists()
                else 0
            )
            if rows >= 1:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                if rows != 1:
                    raise RuntimeError("Interruption probe missed the one-image boundary")
                print("interruption probe stopped after one persisted image", flush=True)
                return
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=20)
        raise RuntimeError("Interruption probe ended before saving one image")


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
    diverse_prepared = ROOT / "artifacts/prepared-diverse-web-eval"
    if not (diverse_prepared / "images.jsonl").exists():
        raise RuntimeError("The diverse evaluation set must be prepared before GPU work")
    diverse_count = sum(1 for _ in (diverse_prepared / "images.jsonl").open(encoding="utf-8"))
    if diverse_count != 60 or not (diverse_prepared / "manifest.json").exists():
        raise RuntimeError("The 60-image diverse evaluation set is incomplete")
    signal.signal(signal.SIGTERM, _signal_stop)
    signal.signal(signal.SIGINT, _signal_stop)
    output_dir: Path = args.output_dir
    try:
        _phase(output_dir, "generator_loading")
        generator = _start_server("generator", "runtime/vllm/generator-gpu-watch.yaml", output_dir)
        _wait_model(generator, 18002, "Qwen/Qwen3.5-9B", 1800)
        _phase(output_dir, "generator_ready")
        _request_holder_handoff(output_dir)
        _phase(output_dir, "selector_loading")
        selector = _start_server("selector", "runtime/vllm/selector-gpu-watch.yaml", output_dir)
        _wait_model(selector, 18000, "Qwen/Qwen3.5-2B", 900)
        _phase(output_dir, "selector_ready")
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
        _phase(output_dir, "first_image_inference")
        _run_logged(
            _synthesis_command(
                f"{args.run_id}-first-image",
                first_image,
                config="configs/gpu-watch-smoke.yaml",
                workers=1,
            ),
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
        _phase(output_dir, "first_image_complete")
        targeted_output = output_dir / "targeted" / "conversations.jsonl"
        targeted_run_id = f"{args.run_id}-targeted"
        (output_dir / "targeted-selection.json").write_text(
            json.dumps(
                {
                    "source_ids": TARGETED_SOURCE_IDS,
                    "purpose": "multi-turn functional probe",
                    "human_verified": False,
                    "screening": "assistant visual triage; human confirmation pending",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        targeted_command = _synthesis_command(
            targeted_run_id,
            targeted_output,
            workers=1,
            source_ids=TARGETED_SOURCE_IDS,
        )
        _phase(output_dir, "targeted_inference")
        _run_interrupted(
            targeted_command,
            output_dir / "targeted-interrupted.log",
            targeted_output,
            timeout=1800,
        )
        _check_results(output_dir / "targeted", 1)
        diagnostic_command = [
            "uv",
            "run",
            "--locked",
            "pixelogue",
            "run-diagnostics",
            "--config",
            CONFIG,
            "--run-id",
            targeted_run_id,
            "--conversations",
            str(targeted_output),
        ]
        _run_logged(
            [*diagnostic_command, "--output-stem", str(output_dir / "targeted-before")],
            output_dir / "targeted-before.log",
            90,
        )
        before = json.loads((output_dir / "targeted-before.json").read_text(encoding="utf-8"))
        first_source = before["rows"][0]["source_id"]
        first_calls = before["rows"][0]["model_calls"]
        _run_logged(targeted_command, output_dir / "targeted-resumed.log", 1800, targeted_output)
        _check_results(output_dir / "targeted", 2)
        _run_logged(
            [*diagnostic_command, "--output-stem", str(output_dir / "targeted-after")],
            output_dir / "targeted-after.log",
            90,
        )
        after = json.loads((output_dir / "targeted-after.json").read_text(encoding="utf-8"))
        resumed_first = next(row for row in after["rows"] if row["source_id"] == first_source)
        if resumed_first["model_calls"] != first_calls:
            raise RuntimeError("Resuming the interrupted run repeated first-image model calls")
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
                targeted_run_id,
            ],
            output_dir / "targeted-replay.log",
            300,
        )
        _phase(output_dir, "targeted_complete")
        prescreen_output = output_dir / "prescreen" / "conversations.jsonl"
        prescreen_run_id = f"{args.run_id}-prescreen"
        (output_dir / "prescreen-selection.json").write_text(
            json.dumps(
                {
                    "source_ids": PRESCREEN_SOURCE_IDS,
                    "purpose": "human-confirmed two-question route probe",
                    "human_verified_for_multi_turn": True,
                    "human_gold_answers_supplied": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        _phase(output_dir, "prescreen_inference")
        _run_logged(
            _synthesis_command(
                prescreen_run_id,
                prescreen_output,
                workers=1,
                source_ids=PRESCREEN_SOURCE_IDS,
            ),
            output_dir / "prescreen.log",
            1800,
            prescreen_output,
        )
        _check_results(output_dir / "prescreen", len(PRESCREEN_SOURCE_IDS))
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "run-diagnostics",
                "--config",
                CONFIG,
                "--run-id",
                prescreen_run_id,
                "--conversations",
                str(prescreen_output),
                "--output-stem",
                str(output_dir / "prescreen-diagnostics"),
            ],
            output_dir / "prescreen-diagnostics.log",
            90,
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
                prescreen_run_id,
            ],
            output_dir / "prescreen-replay.log",
            300,
        )
        _phase(output_dir, "prescreen_complete")
        full_output = output_dir / "conversations.jsonl"
        full_command = _synthesis_command(args.run_id, full_output)
        _phase(output_dir, "full_inference")
        _run_logged(
            full_command,
            output_dir / "synthesize.log",
            5400,
            full_output,
        )
        _check_results(output_dir, expected)
        _phase(output_dir, "full_complete")
        diverse_output = output_dir / "diverse" / "conversations.jsonl"
        diverse_run_id = f"{args.run_id}-diverse"
        _phase(output_dir, "diverse_inference")
        _run_logged(
            _synthesis_command(
                diverse_run_id,
                diverse_output,
                config="configs/gpu-watch-diverse.yaml",
                prepared=diverse_prepared,
            ),
            output_dir / "diverse.log",
            5400,
            diverse_output,
        )
        _check_results(output_dir / "diverse", diverse_count)
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "run-diagnostics",
                "--config",
                "configs/gpu-watch-diverse.yaml",
                "--run-id",
                diverse_run_id,
                "--conversations",
                str(diverse_output),
                "--output-stem",
                str(output_dir / "diverse-diagnostics"),
            ],
            output_dir / "diverse-diagnostics.log",
            90,
        )
        _run_logged(
            [
                sys.executable,
                "-m",
                "pixelogue.diverse_eval_report",
                "--diagnostics",
                str(output_dir / "diverse-diagnostics.json"),
                "--manifest",
                str(ROOT / "validation/diverse_web_eval_manifest.jsonl"),
                "--output-stem",
                str(output_dir / "diverse-category-report"),
            ],
            output_dir / "diverse-category-report.log",
            90,
        )
        _run_logged(
            [
                "uv",
                "run",
                "--locked",
                "pixelogue",
                "replay",
                "--config",
                "configs/gpu-watch-diverse.yaml",
                "--run-id",
                diverse_run_id,
            ],
            output_dir / "diverse-replay.log",
            300,
        )
        _phase(output_dir, "diverse_complete")
        model_calls_before_resume = _model_call_count(args.run_id)
        output_hash_before_resume = hashlib.sha256(full_output.read_bytes()).hexdigest()
        _run_logged(
            full_command,
            output_dir / "resume.log",
            300,
            full_output,
        )
        if _model_call_count(args.run_id) != model_calls_before_resume:
            raise RuntimeError("Resuming completed outcomes created extra model calls")
        if hashlib.sha256(full_output.read_bytes()).hexdigest() != output_hash_before_resume:
            raise RuntimeError("Resuming changed the committed conversation output")
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
        _phase(output_dir, "server_shutdown")
        _stop_servers()
        _phase(output_dir, "ended")


if __name__ == "__main__":
    main()
