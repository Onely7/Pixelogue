"""Single-voice staff evidence and isolated music21 notation checks."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.errors import ExecutionError, ExternalInputError
from pixelogue.serialization import strict_json_object
from pixelogue.specialist_env import call_worker
from pixelogue.task_evidence import ImageRegion


class ScoreEvent(StrictModel):
    """One visible note or rest with staff position and written duration."""

    kind: Literal["note", "rest"]
    staff_step: Annotated[int, Field(ge=-8, le=16)] | None = None
    base: Literal[1, 2, 4, 8, 16]
    dots: Literal[0, 1] = 0
    accidental: Literal["flat", "natural", "sharp"] | None = None
    tie: Literal["start", "continue", "stop"] | None = None
    region: ImageRegion

    @model_validator(mode="after")
    def check_kind(self) -> ScoreEvent:
        """Rests have no pitch context; notes must locate a staff step."""
        if self.kind == "rest" and (
            self.staff_step is not None or self.accidental is not None or self.tie is not None
        ):
            raise ValueError("Rest cannot carry pitch or tie")
        if self.kind == "note" and self.staff_step is None:
            raise ValueError("Note needs a resolved staff position")
        return self


class ScoreMeasure(StrictModel):
    """One complete public measure in reading order."""

    number: Annotated[int, Field(ge=1)]
    events: Annotated[tuple[ScoreEvent, ...], Field(min_length=1, max_length=64)]


class MusicSource(StrictModel):
    """Blind score extraction with complete clef, key, meter and bar range."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    domain: str = Field(min_length=1)
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    clef: Literal["treble", "bass"]
    key_sharps: Annotated[int, Field(ge=-7, le=7)]
    meter_top: Annotated[int, Field(ge=1, le=12)]
    meter_bottom: Literal[2, 4, 8]
    bar_range: str = Field(pattern=r"^[1-9][0-9]*-[1-9][0-9]*$")
    measures: Annotated[tuple[ScoreMeasure, ...], Field(max_length=8)]
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_measures(self) -> MusicSource:
        """Require consecutive complete bars and local event evidence."""
        if self.coverage != "MET" and self.measures:
            raise ValueError("Incomplete score extraction cannot certify events")
        first, last = (int(item) for item in self.bar_range.split("-"))
        if self.coverage == "MET" and (
            last < first
            or last - first + 1 != len(self.measures)
            or [item.number for item in self.measures] != list(range(first, last + 1))
        ):
            raise ValueError("Score measures do not match public bar range")
        for event in (event for measure in self.measures for event in measure.events):
            region = event.region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Score event lies outside requested scope")
        return self


class MusicResultEvent(StrictModel):
    """Public note or rest transcription with explicit duration and tie."""

    pitch: str | None
    duration: str
    tie: Literal["start", "continue", "stop"] | None


class MusicAnswer(StrictModel):
    """Strict public score event schema."""

    events: tuple[MusicResultEvent, ...]


def verify_music(
    sources: tuple[MusicSource, MusicSource],
    domain: str,
    bar_range: object,
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> tuple[GateVerdict, dict[str, object]]:
    """Use music21 only after two blind notation extractions agree."""
    if any(
        source.coverage != "MET"
        or source.domain != domain
        or source.scope_id != scope_id
        or source.view_id != view_id
        or source.bar_range != bar_range
        for source in sources
    ):
        return GateVerdict.UNKNOWN, {"reason": "Score evidence or public bar range is incomplete"}
    canonical = [
        source.model_dump(mode="json", exclude={"reason", "scope_region"}) for source in sources
    ]
    for item in canonical:
        for measure in item["measures"]:
            for event in measure["events"]:
                event.pop("region", None)
    if canonical[0] != canonical[1]:
        return GateVerdict.UNKNOWN, {"reason": "Independent score extractions disagree"}
    try:
        raw = strict_json_object(candidate_answer)
        if set(raw) != {"events"}:
            raise ValueError("Unsupported score output schema")
        parsed = MusicAnswer.model_validate_json(candidate_answer)
    except (ExternalInputError, ValueError, TypeError):
        return GateVerdict.UNKNOWN, {"reason": "Malformed public score output"}
    worker_source = sources[0].model_dump(
        mode="json", exclude={"reason", "scope_id", "view_id", "scope_region", "domain", "coverage"}
    )
    for measure in worker_source["measures"]:
        for event in measure["events"]:
            event.pop("region", None)
    try:
        response = call_worker(
            "notation",
            "music21",
            {
                "operation": "music",
                "source": worker_source,
                "reported_events": [item.model_dump(mode="json") for item in parsed.events],
            },
            timeout_seconds=20,
        )
    except ExecutionError as exc:
        return GateVerdict.UNKNOWN, {"reason": str(exc)}
    return GateVerdict(response["verdict"]), response
