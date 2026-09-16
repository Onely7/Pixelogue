"""Console progress for saved synthesis results, without accessing inference state."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Lock, Thread
from typing import TextIO

_STATUSES = ("QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR")


@contextmanager
def synthesis_progress(
    total: int,
    workers: int,
    stream: TextIO,
    *,
    enabled: bool = True,
    interval: float = 10.0,
) -> Iterator[Callable[[str], None]]:
    """Report persisted image counts and periodic waiting updates on a supplied stream.

    The yielded callback accepts a conversation status only after its output is saved.
    The reporter never reads the database or model clients. Its thread is stopped on
    both normal and exceptional exits. Waiting updates do not imply model health.
    """
    if not enabled:
        yield lambda status: None
        return
    if interval <= 0:
        raise ValueError("interval must be positive")
    started = time.monotonic()
    counts: Counter[str] = Counter()
    lock = Lock()
    stopped = Event()

    def emit(state: str) -> None:
        details = " ".join(f"{status}={counts[status]}" for status in _STATUSES)
        print(
            f"[synthesize] {state} saved={sum(counts.values()):,}/{total:,} "
            f"workers={workers} {details} elapsed={time.monotonic() - started:.1f}s",
            file=stream,
            flush=True,
        )

    def record(status: str) -> None:
        with lock:
            counts[status] += 1
            emit("running")

    def heartbeat() -> None:
        while not stopped.wait(interval):
            with lock:
                emit("waiting")

    emit("started")
    thread = Thread(target=heartbeat, name="pixelogue-progress", daemon=True)
    thread.start()
    state = "interrupted"
    try:
        yield record
        state = "finished" if sum(counts.values()) == total else "incomplete"
    finally:
        stopped.set()
        thread.join()
        with lock:
            emit(state)
