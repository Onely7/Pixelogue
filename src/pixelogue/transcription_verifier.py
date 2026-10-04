"""Exact complete-answer comparison against answer-blind visible transcriptions."""

from __future__ import annotations

import re

from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion, TranscriptSource


def transcript_content(answer: str) -> str:
    """Remove only a surrounding quotation or one complete code fence.

    No matching substring, explanatory sentence or extra trailing content is
    discarded. Interior punctuation, indentation and line breaks remain exact.
    """
    fence = re.fullmatch(r"```[A-Za-z0-9_+-]*\n([\s\S]*)\n```", answer)
    if fence is not None:
        return fence.group(1)
    for opening, closing in (('"', '"'), ("“", "”")):
        if len(answer) >= 2 and answer.startswith(opening) and answer.endswith(closing):
            return answer[1:-1]
    return answer


def verify_transcription(
    sources: tuple[TranscriptSource, TranscriptSource],
    candidate_answer: str,
    region: ImageRegion | None,
) -> GateVerdict:
    """Require agreement, full-unit containment and equality of the whole answer."""
    for source in sources:
        if source.coverage != "MET" or not source.requested_unit_complete:
            return GateVerdict.UNKNOWN
        source_region = source.source_region
        if source_region is None:
            return GateVerdict.UNKNOWN
        if region is not None and not (
            region.left <= source_region.left < source_region.right <= region.right
            and region.top <= source_region.top < source_region.bottom <= region.bottom
        ):
            return GateVerdict.UNKNOWN
    if sources[0].expected_text != sources[1].expected_text:
        return GateVerdict.UNKNOWN
    if candidate_answer == sources[0].expected_text:
        return GateVerdict.MET
    content = transcript_content(candidate_answer)
    if not content:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if content == sources[0].expected_text else GateVerdict.NOT_MET
