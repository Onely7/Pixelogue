"""Separate pinned worker environment and bounded process protocol."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pixelogue.errors import ExecutionError

SpecialistEnvironment = Literal["symbolic", "notation", "chemistry", "renderer"]
RendererBackend = Literal["bwrap", "landlock"]
_RENDER_SMOKE: dict[str, Any] = {
    "format": "svg",
    "code": '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"></svg>',
    "width": 8,
    "height": 8,
}


def worker_root() -> Path:
    """Find the source checkout's specialist lock and worker."""
    return Path(__file__).resolve().parents[2] / "runtime" / "validators"


def _bwrap_available() -> bool:
    """Check that mount and network namespaces actually work for this user."""
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return False
    try:
        probe = subprocess.run(
            [
                bwrap,
                "--unshare-all",
                "--die-with-parent",
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
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if probe.returncode != 0:
        return False
    try:
        rendered = _call_bwrap_renderer(_RENDER_SMOKE, 10)
        return rendered.returncode == 0 and json.loads(rendered.stdout).get("verdict") == "MET"
    except (OSError, ExecutionError, ValueError):
        return False


def _landlock_command(
    scratch: Path,
    action: Literal["probe", "render"],
    seconds: int,
    probe_file: Path | None = None,
) -> list[str]:
    """Start the restricted launcher in a bounded user service."""
    systemd_run = shutil.which("systemd-run")
    if systemd_run is None:
        raise FileNotFoundError("systemd-run is unavailable")
    root = worker_root()
    command = [
        systemd_run,
        "--user",
        "--wait",
        "--pipe",
        "--collect",
        "--quiet",
        "-p",
        "RestrictAddressFamilies=AF_UNIX AF_NETLINK",
        "-p",
        "NoNewPrivileges=yes",
        "-p",
        "MemoryMax=3G",
        "-p",
        "TasksMax=256",
        "-p",
        f"RuntimeMaxSec={seconds}s",
        str(root / ".venv/bin/python"),
        str(root / "landlock_launcher.py"),
        str(scratch),
        action,
    ]
    if probe_file is not None:
        command.append(str(probe_file))
    return command


def _landlock_available() -> bool:
    """Probe OS restrictions and one actual Chromium render before enabling."""
    if not (worker_root() / "landlock_launcher.py").is_file():
        return False
    try:
        with (
            tempfile.TemporaryDirectory(prefix="pixelogue-render-", dir="/tmp") as directory,
            tempfile.NamedTemporaryFile(prefix="pixelogue-render-probe-", dir="/tmp") as probe_file,
        ):
            probe_file.write(b"private")
            probe_file.flush()
            probe = subprocess.run(
                _landlock_command(Path(directory), "probe", 10, Path(probe_file.name)),
                capture_output=True,
                text=True,
                timeout=12,
                check=False,
            )
            if probe.returncode != 0:
                return False
            checks = json.loads(probe.stdout)
        if checks != {
            "ipv4_denied": True,
            "ipv6_denied": True,
            "workspace_read_denied": True,
            "workspace_write_denied": True,
            "tmp_sibling_read_denied": True,
            "scratch_writable": True,
        }:
            return False
        with tempfile.TemporaryDirectory(prefix="pixelogue-render-", dir="/tmp") as directory:
            rendered = subprocess.run(
                _landlock_command(Path(directory), "render", 12),
                input=json.dumps(_RENDER_SMOKE),
                capture_output=True,
                text=True,
                timeout=14,
                check=False,
            )
        return rendered.returncode == 0 and json.loads(rendered.stdout).get("verdict") == "MET"
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return False


@lru_cache(maxsize=1)
def _renderer_backend() -> RendererBackend | None:
    """Select only a backend whose OS isolation has actually passed a probe."""
    if _bwrap_available():
        return "bwrap"
    if _landlock_available():
        return "landlock"
    return None


@lru_cache(maxsize=8)
def environment_error(environment: SpecialistEnvironment, dependency: str | None) -> str | None:
    """Check actual pinned environment and OS renderer isolation capability."""
    root = worker_root()
    python = root / ".venv" / "bin" / "python"
    if not python.is_file() or not (root / "uv.lock").is_file():
        return "specialist lock environment is not installed"
    if not (root / "worker.py").is_file():
        return "specialist worker is missing"
    if environment == "renderer" and not (root / "render_worker.py").is_file():
        return "isolated renderer worker is missing"
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
        chrome = shutil.which("google-chrome") or shutil.which("chromium")
        if chrome is None:
            return "isolated renderer requires Chromium"
        if _renderer_backend() is None:
            return "OS network and filesystem isolation is unavailable"
    return None


def call_renderer(request: dict[str, Any], *, timeout_seconds: float = 25) -> dict[str, Any]:
    """Render approved static markup inside a probed OS isolation backend.

    Raises:
        ExecutionError: If OS isolation, rendering or the bounded result fails.
    """
    if error := environment_error("renderer", "playwright"):
        raise ExecutionError("VALIDATOR_ENV_MISSING", error)
    backend = _renderer_backend()
    assert backend is not None
    if backend == "landlock":
        result, attempts = _call_landlock_renderer(request, timeout_seconds)
    else:
        result = _call_bwrap_renderer(request, timeout_seconds)
        attempts = 1
    if result.returncode != 0 or len(result.stdout) > 5_000_000:
        raise ExecutionError("RENDER_FAILED", "Isolated renderer failed or exceeded output limit")
    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ExecutionError("RENDER_OUTPUT_INVALID", "Renderer returned invalid JSON") from exc
    if not isinstance(response, dict) or response.get("verdict") not in {"MET", "UNKNOWN"}:
        raise ExecutionError("RENDER_OUTPUT_INVALID", "Renderer returned invalid verdict")
    response["isolation_attempts"] = attempts
    return response


def _call_landlock_renderer(
    request: dict[str, Any], timeout_seconds: float
) -> tuple[subprocess.CompletedProcess[str], int]:
    """Retry one transient service failure with fresh private scratch space."""
    for attempt in (1, 2):
        with tempfile.TemporaryDirectory(prefix="pixelogue-render-", dir="/tmp") as directory:
            try:
                result = subprocess.run(
                    _landlock_command(Path(directory), "render", round(timeout_seconds) + 2),
                    input=json.dumps(request, ensure_ascii=False),
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds + 4,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                if attempt == 2:
                    raise ExecutionError("RENDER_TIMEOUT", "Isolated renderer timed out") from exc
            else:
                if result.returncode == 0:
                    return result, attempt
                if attempt == 2:
                    return result, attempt
        time.sleep(2)
    raise AssertionError("Renderer retry loop ended without a result")


def _call_bwrap_renderer(
    request: dict[str, Any], timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Run the original namespace backend when user namespaces are available."""
    root = worker_root()
    bwrap = shutil.which("bwrap")
    assert bwrap is not None
    command = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        str(root),
        str(root),
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for directory in ("/lib", "/lib64", "/etc", "/opt/google/chrome"):
        if Path(directory).is_dir():
            command.extend(("--ro-bind", directory, directory))
    interpreter = (root / ".venv/bin/python").resolve(strict=True)
    command.extend(("--ro-bind", str(interpreter.parents[2]), str(interpreter.parents[2])))
    command.extend(
        ("--chdir", "/tmp", str(root / ".venv/bin/python"), str(root / "render_worker.py"))
    )
    try:
        result = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            cwd=root,
        )
    except subprocess.TimeoutExpired as exc:
        raise ExecutionError("RENDER_TIMEOUT", "Isolated renderer timed out") from exc
    return result


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
