"""Console progress for saved synthesis results, without accessing inference state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event, Lock, Thread
from typing import TextIO

from tqdm import tqdm

_BAR_FORMAT = "{desc}: {n_fmt}/{total_fmt} [{bar}] elapsed={elapsed} remaining={remaining} {rate_fmt}{postfix}"

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
    bar = tqdm(
        total=total,
        desc="synthesize",
        unit="image",
        file=stream,
        mininterval=0.5,
        dynamic_ncols=True,
        bar_format=_BAR_FORMAT,
    )
    counts: Counter[str] = Counter()
    lock = Lock()
    stopped = Event()

    def emit(state: str) -> None:
        bar.set_description_str(f"synthesize {state}", refresh=False)
        bar.set_postfix({"workers": workers, **{s: counts[s] for s in _STATUSES}}, refresh=False)
        bar.refresh()

    def record(status: str) -> None:
        with lock:
            counts[status] += 1
            bar.update(1)
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
            bar.close()


@contextmanager
def preparation_progress(
    stream: TextIO, *, enabled: bool = True
) -> Iterator[Callable[[str, int, int | None], None] | None]:
    """Display one tqdm bar per preparation stage, closing it even on failure."""
    if not enabled:
        yield None
        return
    bar = None
    current_stage = ""

    def report(stage: str, processed: int, total: int | None) -> None:
        nonlocal bar, current_stage
        if bar is None or stage != current_stage:
            if bar is not None:
                bar.close()
            current_stage = stage
            bar = tqdm(
                total=total,
                desc=f"prepare-local-train {stage}",
                file=stream,
                unit="row" if "metadata" in stage else "item",
                mininterval=0.5,
                dynamic_ncols=True,
                bar_format=_BAR_FORMAT,
            )
        bar.total = total
        bar.update(processed - bar.n)
        if processed == total:
            bar.refresh()

    try:
        yield report
    except BaseException:
        if bar is not None:
            bar.set_description_str(f"prepare-local-train {current_stage} interrupted")
        raise
    finally:
        if bar is not None:
            bar.close()
