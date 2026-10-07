"""Purpose-specific domain contracts for Pixelogue artifacts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, HttpUrl, model_validator

from pixelogue.config import StrictModel
from pixelogue.errors import ExecutionError
from pixelogue.ledger import Requirement
from pixelogue.serialization import canonical_hash
from pixelogue.task_evidence import ImageRegion, PublicParameter

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
# Judges are asked for at most 25 words; the cap leaves room so a long reason ends naturally
# instead of being cut mid-sentence, after which a model may emit whitespace until its limit.
REASON_MAX_LENGTH = 320


class SourcePurpose(StrEnum):
    """Allowed downstream purpose for a source image."""

    TRAINING = "training"
    EVALUATION = "evaluation"


class GateVerdict(StrEnum):
    """Possible semantic evaluation outcomes."""

    MET = "MET"
    NOT_MET = "NOT_MET"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    NOT_EVALUATED = "NOT_EVALUATED"
    ERROR = "ERROR"


class SourceRecord(StrictModel):
    """A local source image and its non-model provenance."""

    source_id: str
    image_path: str
    source_group_ids: tuple[str, ...] = ()
    rights_record_id: str
    purpose: SourcePurpose
    dataset: str | None = None
    dataset_split: str | None = None
    dataset_image_id: str | None = None
    rotation_degrees: Literal[0, 90, 180, 270] = 0


class RightsRecord(StrictModel):
    """Machine-checkable permissions supplied by a data operator."""

    rights_record_id: str
    license_uri: HttpUrl
    attribution: str
    processing_allowed: bool
    qa_redistribution_allowed: bool
    image_redistribution_allowed: bool
    training_allowed: bool
    valid_from: datetime
    valid_until: datetime | None = None

    @model_validator(mode="after")
    def validate_window(self) -> RightsRecord:
        """Require a non-empty validity window."""
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be later than valid_from")
        return self

    def permits(self, purpose: SourcePurpose, at: datetime) -> bool:
        """Return whether this record permits processing for a purpose."""
        in_window = self.valid_from <= at and (self.valid_until is None or at < self.valid_until)
        if not in_window or not self.processing_allowed:
            return False
        if purpose is SourcePurpose.TRAINING:
            return self.training_allowed and self.qa_redistribution_allowed
        return True


class ImageView(StrictModel):
    """One immutable raster that may be sent to a model."""

    view_id: str
    relative_path: str
    encoded_sha256: Sha256
    pixel_sha256: Sha256
    width: Annotated[int, Field(ge=1)]
    height: Annotated[int, Field(ge=1)]
    media_type: Literal["image/png", "image/jpeg", "image/webp"]


class ImageArtifact(StrictModel):
    """Canonical image identity and inference view."""

    image_id: str
    source_id: str
    purpose: SourcePurpose
    dataset: str | None = None
    dataset_split: str | None = None
    raw_sha256: Sha256
    canonical_pixel_sha256: Sha256
    source_group_ids: tuple[str, ...]
    visual_group_id: str
    full_view: ImageView


class PublicMessage(StrictModel):
    """An immutable user-visible dialogue message."""

    message_id: str
    turn_index: Annotated[int, Field(ge=1)]
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1)]


class HistorySnapshot(StrictModel):
    """Exact public prefix visible before a turn."""

    conversation_id: str
    branch_id: str = "main"
    turn_index: Annotated[int, Field(ge=1)]
    public_history: tuple[PublicMessage, ...]
    history_hash: Sha256


class InstructionCandidate(StrictModel):
    """One drafted operation with its public parameters and verification contracts."""

    candidate_id: str
    task_id: str
    family: str
    visible_scope: str
    instruction_summary: str
    required_capabilities: tuple[str, ...]
    catalog_version: Literal["8.0"] | None = None
    scope_id: str | None = None
    view_id: str | None = None
    scope_region: ImageRegion | None = None
    target_region: ImageRegion | None = None
    public_parameters: tuple[PublicParameter, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    verification_contracts: tuple[str, ...] = ()
    calibrated_domain: str | None = None
    origin: Literal["scoped", "direct"] = "scoped"
    request_key: str | None = None

    @model_validator(mode="after")
    def validate_operation(self) -> InstructionCandidate:
        """Validate new contracts while preserving immutable legacy labels."""
        if self.scope_region is not None and self.target_region is not None:
            parent, target = self.scope_region, self.target_region
            if not (
                parent.left <= target.left < target.right <= parent.right
                and parent.top <= target.top < target.bottom <= parent.bottom
            ):
                raise ValueError("Target region must remain inside the bound scope")
        if self.catalog_version is not None:
            from pixelogue.catalog import task_catalog

            catalog = task_catalog()
            task = next((task for task in catalog.tasks if task.id == self.task_id), None)
            if task is None or task.family != self.family:
                raise ValueError("Unknown operation or incorrect family")
            if self.required_capabilities != task.required_capabilities:
                raise ValueError("Catalog candidates cannot replace capability requirements")
            if len({parameter.name for parameter in self.public_parameters}) != len(
                self.public_parameters
            ):
                raise ValueError("Public parameter names must be unique")
            if not self.scope_id or not self.view_id:
                raise ValueError("Catalog candidates need scope and view bindings")
            if self.origin == "scoped" and not self.evidence_refs:
                raise ValueError("Scoped catalog candidates need evidence bindings")
            if self.origin == "direct" and (self.evidence_refs or self.scope_region is None):
                raise ValueError("Direct drafts carry a public scope region and no evidence IDs")
            if self.verification_contracts != task.verification_contracts:
                raise ValueError("Catalog candidates cannot omit or replace required verifiers")
            if task.status == "extension" and not self.calibrated_domain:
                raise ValueError("Specialist candidates need a certified domain")
        return self


class TextPayload(StrictModel):
    """Question or answer text returned by a model."""

    text: str | None
    reason: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> TextPayload:
        """Require usable public text or an internal reason for abstaining."""
        if self.text is not None and not self.text.strip():
            raise ValueError("text must contain non-whitespace content")
        if self.text is None and not self.reason:
            raise ValueError("missing text requires a reason")
        return self

    @property
    def status(self) -> Literal["OK", "UNSUPPORTED"]:
        """Derive the status so it cannot contradict the public text."""
        return "OK" if self.text is not None else "UNSUPPORTED"


class RubricVerdict(StrictModel):
    """One model's output for a single applicable rubric criterion."""

    verdict: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: Annotated[str, Field(min_length=1, max_length=REASON_MAX_LENGTH)]


class RubricItem(StrictModel):
    """One instantiated evaluation criterion."""

    item_id: str
    template_id: str
    axis: str
    verdict: GateVerdict
    reason: Annotated[str, Field(min_length=1, max_length=REASON_MAX_LENGTH)]
    actor: str
    history_hash: Sha256


class TurnRating(StrictModel):
    """Complete rating record for one answer attempt."""

    items: tuple[RubricItem, ...]
    aggregate: Literal["PASS", "FAIL", "ABSTAIN", "ERROR"]


class TurnArtifact(StrictModel):
    """A completed public turn plus its private verification references."""

    turn_index: Annotated[int, Field(ge=1)]
    instruction: InstructionCandidate
    question: PublicMessage
    answer: PublicMessage
    history_hash: Sha256
    generation_model: str
    # The image router; the field keeps its earlier name so saved turns stay readable.
    selector_model: str
    requirements: tuple[Requirement, ...] = ()
    rating: TurnRating
    status: Literal["COMMITTED", "REJECTED", "ABSTAINED", "ERROR"]


class ConversationArtifact(StrictModel):
    """An immutable complete or rejected conversation."""

    conversation_id: str
    image: ImageArtifact
    target_language: Literal["en", "ja", "zh-Hans"]
    generation_model: str
    turns: tuple[TurnArtifact, ...]
    status: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"]

    @model_validator(mode="after")
    def validate_conversation(self) -> ConversationArtifact:
        """Require canonical turn order, exact history hashes, and terminal state consistency."""
        transcript: list[PublicMessage] = []
        terminal_seen = False
        for expected_index, turn in enumerate(self.turns, start=1):
            if turn.turn_index != expected_index:
                raise ValueError("turn indices must be contiguous and start at one")
            if turn.question.turn_index != expected_index or turn.question.role != "user":
                raise ValueError("turn question has an invalid role or index")
            if turn.answer.turn_index != expected_index or turn.answer.role != "assistant":
                raise ValueError("turn answer has an invalid role or index")
            if turn.generation_model != self.generation_model:
                raise ValueError("a conversation cannot change generation model")
            if turn.history_hash != canonical_hash(
                [message.model_dump(mode="json") for message in transcript]
            ):
                raise ValueError("turn history hash does not match the exact public prefix")
            for requirement in turn.requirements:
                try:
                    requirement.validate_source((*transcript, turn.question))
                except ExecutionError as error:
                    raise ValueError("turn requirement has an invalid public source") from error
            if terminal_seen:
                raise ValueError("no turn may follow a non-committed turn")
            if turn.status == "COMMITTED":
                transcript.extend((turn.question, turn.answer))
            else:
                terminal_seen = True
        if self.status == "QUALITY_CANDIDATE":
            if not 2 <= len(self.turns) <= 6 or terminal_seen:
                raise ValueError("quality candidates require two to six committed turns")
        return self

    @property
    def public_messages(self) -> tuple[PublicMessage, ...]:
        """Return committed messages in canonical public order."""
        messages: list[PublicMessage] = []
        for turn in self.turns:
            if turn.status == "COMMITTED":
                messages.extend((turn.question, turn.answer))
        return tuple(messages)


class SelectionCandidate(StrictModel):
    """Verified metadata used by the integer selector and independent auditor."""

    conversation_id: str
    language: str
    image_id: str
    visual_group_id: str
    task_family: str
    semantic_family: str
    purpose: SourcePurpose
    status: Literal["QUALITY_CANDIDATE"] = "QUALITY_CANDIDATE"


class SelectionManifest(StrictModel):
    """Immutable selected identifiers and the pool they came from."""

    pool_hash: Sha256
    selected_ids: tuple[str, ...]
    solver_status: Literal["FEASIBLE", "OPTIMAL", "INFEASIBLE", "UNKNOWN"]
    audit_report_hash: Sha256 | None = None
    audit_hash: Sha256 | None = None

    @model_validator(mode="after")
    def validate_audit_hash(self) -> SelectionManifest:
        """Require a self-consistent binding between selection and audit report."""
        if (self.audit_hash is None) != (self.audit_report_hash is None):
            raise ValueError("audit_hash and audit_report_hash must be set together")
        if self.audit_hash is not None and not self.has_valid_audit_binding():
            raise ValueError("audit binding does not match this selection")
        return self

    def has_valid_audit_binding(self) -> bool:
        """Return whether the attached audit digest binds this exact manifest."""
        if self.audit_hash is None or self.audit_report_hash is None:
            return False
        selection = self.model_dump(
            mode="json",
            exclude={"audit_hash", "audit_report_hash"},
        )
        return self.audit_hash == canonical_hash(
            {"selection": selection, "audit_report_hash": self.audit_report_hash}
        )


class TrainingContent(StrictModel):
    """One text or image part in a training message.

    An image part always records the pixel width and height of the exported file, so any
    later training run can apply its own model's resizing rule.
    """

    type: Literal["text", "image"]
    text: str | None = None
    image: str | None = None
    width: Annotated[int, Field(ge=1)] | None = None
    height: Annotated[int, Field(ge=1)] | None = None

    @model_validator(mode="after")
    def validate_content(self) -> TrainingContent:
        """Require exactly the values that match the content kind."""
        if self.type == "text" and (
            self.text is None or self.image is not None or self.width or self.height
        ):
            raise ValueError("text content requires only text")
        if self.type == "image" and (
            self.image is None or self.text is not None or self.width is None or self.height is None
        ):
            raise ValueError("image content requires the image path, width and height")
        return self


class TrainingMessage(StrictModel):
    """A public role and its model-ready content parts."""

    role: Literal["user", "assistant"]
    content: tuple[TrainingContent, ...]


class TrainingRecord(StrictModel):
    """A multi-turn record containing only public content."""

    record_id: str
    messages: tuple[TrainingMessage, ...]


def build_history_snapshot(
    conversation_id: str,
    turn_index: int,
    transcript: tuple[PublicMessage, ...],
) -> HistorySnapshot:
    """Build the exact prefix before a requested turn.

    Raises:
        ValueError: If transcript roles, indices, or prefix length are inconsistent.
    """
    expected_length = 2 * (turn_index - 1)
    if len(transcript) != expected_length:
        raise ValueError("history must contain exactly two messages per prior turn")
    for position, message in enumerate(transcript):
        expected_turn = position // 2 + 1
        expected_role = "user" if position % 2 == 0 else "assistant"
        if message.turn_index != expected_turn or message.role != expected_role:
            raise ValueError("history is not a canonical user/assistant transcript prefix")
    serialized = [message.model_dump(mode="json") for message in transcript]
    return HistorySnapshot(
        conversation_id=conversation_id,
        turn_index=turn_index,
        public_history=transcript,
        history_hash=canonical_hash(serialized),
    )
