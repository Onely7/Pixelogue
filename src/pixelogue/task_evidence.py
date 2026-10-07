"""Image regions, public operation choices, and blind source-reader contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel

Nonempty = Annotated[str, Field(min_length=1, max_length=512)]
ObservationVerdict = Literal["MET", "NOT_MET", "UNKNOWN"]


class ImageRegion(StrictModel):
    """Normalized location in the exact delivered image view."""

    left: Annotated[float, Field(ge=0, le=1)]
    top: Annotated[float, Field(ge=0, le=1)]
    right: Annotated[float, Field(ge=0, le=1)]
    bottom: Annotated[float, Field(ge=0, le=1)]

    @model_validator(mode="after")
    def validate_extent(self) -> ImageRegion:
        """Reject empty and inverted regions."""
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("Evidence region must have positive extent")
        return self


def region_iou(left: ImageRegion, right: ImageRegion) -> float:
    """Return the intersection over union of two regions in the same view."""
    width = max(0.0, min(left.right, right.right) - max(left.left, right.left))
    height = max(0.0, min(left.bottom, right.bottom) - max(left.top, right.top))
    intersection = width * height
    left_area = (left.right - left.left) * (left.bottom - left.top)
    right_area = (right.right - right.left) * (right.bottom - right.top)
    return intersection / (left_area + right_area - intersection)


class PublicParameter(StrictModel):
    """A public operation choice or explicitly sourced factual parameter."""

    name: Nonempty
    value: Nonempty | int | bool | tuple[Nonempty, ...]
    origin: Literal["instruction", "image", "history"]
    evidence_refs: tuple[Nonempty, ...] = ()

    @model_validator(mode="after")
    def validate_source(self) -> PublicParameter:
        """Require source references for image facts and committed history."""
        if self.origin == "instruction" and self.evidence_refs:
            raise ValueError("Instruction choices cannot carry factual source references")
        if self.origin != "instruction" and not self.evidence_refs:
            raise ValueError("Factual parameters require explicit evidence references")
        return self


class TranscriptInventory(StrictModel):
    """Literal source and answer spans for non-verbatim extractive operations."""

    coverage: ObservationVerdict
    expected_text: str
    answer_text: str
    reason: Nonempty


class TranscriptSource(StrictModel):
    """Blind transcription of the entire requested visible unit."""

    coverage: ObservationVerdict
    expected_lines: Annotated[tuple[str, ...], Field(max_length=256)]
    source_region: ImageRegion | None = None
    requested_unit_complete: bool
    reason: Nonempty

    @property
    def expected_text(self) -> str:
        """Join literal source lines without interpreting generated separators."""
        return "\n".join(self.expected_lines)

    @model_validator(mode="after")
    def check_coverage(self) -> TranscriptSource:
        """Keep incomplete text or off-image guesses from becoming certificates."""
        if self.coverage == "MET":
            if (
                not any(self.expected_lines)
                or self.source_region is None
                or not self.requested_unit_complete
            ):
                raise ValueError("Complete transcription needs text, its region and the whole unit")
        elif self.expected_lines or self.source_region is not None or self.requested_unit_complete:
            raise ValueError("Incomplete transcription cannot provide partial source text")
        return self


class AnswerEvidence(StrictModel):
    """An exact answer fragment independently bound to a delivered image region."""

    answer_quote: Nonempty
    region: ImageRegion
    visible_evidence: Nonempty


class VisualContractReview(StrictModel):
    """Material semantic review with explicit evidence for the answer's essential parts."""

    verdict: ObservationVerdict
    coverage: ObservationVerdict
    bindings: Annotated[tuple[AnswerEvidence, ...], Field(max_length=32)]
    reason: Nonempty
