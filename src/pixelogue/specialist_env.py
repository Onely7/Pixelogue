"""Separate pinned worker environment and bounded process protocol."""

from __future__ import annotations

import json
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pixelogue.errors import ExecutionError

SpecialistEnvironment = Literal["symbolic", "notation", "chemistry", "renderer"]


def worker_root() -> Path:
    """Find the source checkout's specialist lock and worker."""
    return Path(__file__).resolve().parents[2] / "runtime" / "validators"


@lru_cache(maxsize=8)
def environment_error(environment: SpecialistEnvironment, dependency: str | None) -> str | None:
    """Check actual pinned environment and OS renderer isolation capability."""
    root = worker_root()
    python = root / ".venv" / "bin" / "python"
    if not python.is_file() or not (root / "uv.lock").is_file():
        return "specialist lock environment is not installed"
    if not (root / "worker.py").is_file():
        return "specialist worker is missing"
    if dependency is not None:
        probe = subprocess.run(
            [str(python), "-c", f"import {dependency}"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if probe.returncode != 0:
            return f"specialist dependency unavailable: {dependency}"
    if environment == "renderer":
        bwrap = shutil.which("bwrap")
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if bwrap is None or chrome is None:
            return "isolated renderer requires bwrap and Chromium"
        probe = subprocess.run(
            [
                bwrap,
                "--unshare-net",
                "--ro-bind",
                "/usr",
                "/usr",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--chdir",
                "/tmp",
                "/usr/bin/true",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if probe.returncode != 0:
            return "OS network and filesystem isolation is unavailable"
    return None


def call_worker(
    environment: SpecialistEnvironment,
    dependency: str | None,
    request: dict[str, Any],
    *,
    timeout_seconds: float = 10,
) -> dict[str, Any]:
    """Invoke one bounded JSON operation outside the application environment.

    Raises:
        ExecutionError: If the worker, its lock or response is unavailable or invalid.
    """
    if error := environment_error(environment, dependency):
        raise ExecutionError("VALIDATOR_ENV_MISSING", error)
    root = worker_root()
    python = root / ".venv" / "bin" / "python"
    try:
        result = subprocess.run(
            [str(python), str(root / "worker.py")],
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=root,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError("VALIDATOR_TIMEOUT", "Specialist worker timed out") from exc
    if result.returncode != 0 or len(result.stdout) > 1_000_000:
        raise ExecutionError(
            "VALIDATOR_WORKER_FAILED", "Specialist worker failed or exceeded output limit"
        )
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionError(
            "VALIDATOR_OUTPUT_INVALID", "Specialist worker returned invalid JSON"
        ) from exc
    if not isinstance(response, dict) or response.get("verdict") not in {
        "MET",
        "NOT_MET",
        "UNKNOWN",
    }:
        raise ExecutionError(
            "VALIDATOR_OUTPUT_INVALID", "Specialist worker returned no valid verdict"
        )
    return response
