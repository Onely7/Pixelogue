from __future__ import annotations

import threading
import time
from pathlib import Path

from test_pipeline import _coordinator


class Overlap:
    """Record the largest number of overlapping judge calls."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def wrap(self, original):
        def call(*args, **kwargs):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.01)
            try:
                return original(*args, **kwargs)
            finally:
                with self.lock:
                    self.active -= 1

        return call


def test_blind_judge_pairs_overlap_and_keep_judge_order(tmp_path: Path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    overlap = Overlap()
    generator_a.invoke = overlap.wrap(generator_a.invoke)
    generator_b.invoke = overlap.wrap(generator_b.invoke)
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()
    assert conversation.status == "QUALITY_CANDIDATE"
    assert overlap.peak == 2
    judged = [stage for stage, _ in generator_b.calls if stage == "holistic_review"]
    assert len(judged) == len(conversation.turns)
