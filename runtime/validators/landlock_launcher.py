"""Constrain one renderer with Landlock inside a bounded user service."""

from __future__ import annotations

import ctypes
import json
import os
import socket
import stat
import sys
from collections.abc import Callable
from pathlib import Path


class _Ruleset(ctypes.Structure):
    """Filesystem rights handled by one Landlock ruleset."""

    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    """Rights granted beneath an already opened directory or file."""

    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_EXECUTE = 1 << 0
_WRITE_FILE = 1 << 1
_READ_FILE = 1 << 2
_READ_DIR = 1 << 3
_READ = _EXECUTE | _READ_FILE | _READ_DIR
_WRITE = sum(1 << bit for bit in (1, 4, 5, 7, 8, 9, 10, 12, 13, 14))
_ALL = _READ | _WRITE
_PR_SET_NO_NEW_PRIVS = 38
_LANDLOCK_RULE_PATH_BENEATH = 1
_SYSCALL_CREATE_RULESET = 444
_SYSCALL_ADD_RULE = 445
_SYSCALL_RESTRICT_SELF = 446


def _syscall(number: int, *args: object) -> int:
    """Call a pinned x86-64 Linux Landlock syscall or propagate its errno."""
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.syscall(number, *args)
    if result < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return int(result)


def _grant(ruleset_fd: int, path: Path, access: int) -> None:
    """Grant access beneath one existing resolved path."""
    descriptor = os.open(path, os.O_PATH | os.O_CLOEXEC)
    try:
        rule = _PathRule(access, descriptor)
        _syscall(
            _SYSCALL_ADD_RULE,
            ruleset_fd,
            _LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(rule),
            0,
        )
    finally:
        os.close(descriptor)


def _grant_if_present(ruleset_fd: int, path: Path, access: int) -> None:
    """Grant optional host paths without broadening access on another host."""
    if path.exists():
        _grant(ruleset_fd, path, access)


def _apply_landlock(scratch: Path, worker_root: Path, home: Path) -> None:
    """Allow system reads and one private writable tree; deny workspace reads."""
    if os.uname().machine != "x86_64" or _syscall(_SYSCALL_CREATE_RULESET, 0, 0, 1) < 4:
        raise RuntimeError("Landlock ABI 4 on x86-64 is required")
    interpreter = (worker_root / ".venv/bin/python").resolve(strict=True)
    interpreter_root = interpreter.parents[1]
    ruleset = _Ruleset(_ALL)
    ruleset_fd = _syscall(_SYSCALL_CREATE_RULESET, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    try:
        _grant(ruleset_fd, Path("/"), _EXECUTE | _READ_DIR)
        for path in (
            home,
            home / ".cache",
            Path("/run"),
            Path("/var"),
            Path("/var/cache"),
            Path("/dev"),
        ):
            _grant_if_present(ruleset_fd, path, _READ_DIR)
        if not any(
            interpreter_root.is_relative_to(system_root)
            for system_root in (Path("/usr"), Path("/opt"))
        ):
            for parent in interpreter_root.parents:
                if parent != Path("/"):
                    _grant_if_present(ruleset_fd, parent, _READ_DIR)
            _grant(ruleset_fd, interpreter_root, _READ)
        for path in (
            Path("/usr"),
            Path("/lib"),
            Path("/lib64"),
            Path("/etc"),
            Path("/proc"),
            Path("/sys"),
            Path("/opt"),
            Path("/var/cache/fontconfig"),
            home / ".cache/fontconfig",
            worker_root,
        ):
            _grant_if_present(ruleset_fd, path, _READ)
        for name in ("null", "zero", "random", "urandom"):
            _grant_if_present(ruleset_fd, Path("/dev") / name, _READ_FILE | _WRITE_FILE)
        _grant(ruleset_fd, scratch, _ALL)
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0):
            code = ctypes.get_errno()
            raise OSError(code, os.strerror(code))
        _syscall(_SYSCALL_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)


def _probe(worker_root: Path, scratch: Path, forbidden_read: Path) -> None:
    """Fail unless service networking and private filesystem reads are denied."""
    project_file = worker_root.parents[1] / "pyproject.toml"

    def denied(action: Callable[[], object]) -> bool:
        try:
            action()
        except OSError:
            return True
        return False

    checks = {
        "ipv4_denied": denied(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM)),
        "ipv6_denied": denied(lambda: socket.socket(socket.AF_INET6, socket.SOCK_STREAM)),
        "workspace_read_denied": denied(project_file.read_bytes),
        "workspace_write_denied": denied(lambda: os.open(project_file, os.O_WRONLY)),
        "tmp_sibling_read_denied": denied(forbidden_read.read_bytes),
        "scratch_writable": (scratch / "probe.txt").write_text("ok") == 2,
    }
    print(json.dumps(checks), flush=True)
    if not all(checks.values()):
        raise RuntimeError("Renderer isolation probe failed")


def main() -> None:
    """Restrict this process before probing or replacing it with the worker."""
    if (
        len(sys.argv) not in {3, 4}
        or sys.argv[2] not in {"probe", "render"}
        or (sys.argv[2] == "probe") != (len(sys.argv) == 4)
    ):
        raise SystemExit("usage: landlock_launcher.py SCRATCH probe PROBE_FILE|render")
    scratch = Path(sys.argv[1]).resolve(strict=True)
    info = scratch.stat()
    if (
        scratch.parent != Path("/tmp")
        or not scratch.name.startswith("pixelogue-render-")
        or not scratch.is_dir()
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) != 0o700
    ):
        raise ValueError("Renderer scratch directory must be private")
    worker_root = Path(__file__).resolve().parent
    forbidden_read: Path | None = None
    if sys.argv[2] == "probe":
        forbidden_read = Path(sys.argv[3]).resolve(strict=True)
        forbidden_info = forbidden_read.stat()
        if (
            forbidden_read.parent != Path("/tmp")
            or not forbidden_read.name.startswith("pixelogue-render-probe-")
            or not forbidden_read.is_file()
            or forbidden_info.st_uid != os.getuid()
            or stat.S_IMODE(forbidden_info.st_mode) != 0o600
        ):
            raise ValueError("Renderer probe file must be private and outside scratch")
    real_home = Path.home()
    interpreter = str(worker_root / ".venv/bin/python")
    os.environ.clear()
    os.environ.update(
        {
            "HOME": str(scratch),
            "TMPDIR": str(scratch),
            "XDG_CACHE_HOME": str(scratch),
            "XDG_CONFIG_HOME": str(scratch),
            "XDG_DATA_HOME": str(scratch),
            "LANG": "C.UTF-8",
            "PATH": "/usr/bin:/bin",
            "PIXELOGUE_RENDER_CGROUP": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    os.chdir(scratch)
    _apply_landlock(scratch, worker_root, real_home)
    if sys.argv[2] == "probe":
        assert forbidden_read is not None
        _probe(worker_root, scratch, forbidden_read)
        return
    os.execv(interpreter, [interpreter, str(worker_root / "render_worker.py")])


if __name__ == "__main__":
    main()
