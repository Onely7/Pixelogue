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

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


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
    """A grounded task option offered to an instruction selector."""

    candidate_id: str
    task_id: str
    family: str
    profile: Literal["normal", "limitation", "false_premise"] = "normal"
    visible_scope: str
    instruction_summary: str
    required_capabilities: tuple[str, ...]


class EvidenceInventory(StrictModel):
    """Bounded model observation used only to construct candidate instructions."""

    image_id: str
    capabilities: Annotated[tuple[str, ...], Field(max_length=20)]
    visible_scopes: Annotated[tuple[str, ...], Field(max_length=8)]
    scope_limited: bool
    reason: Annotated[str, Field(min_length=1, max_length=240)]

    @model_validator(mode="after")
    def validate_capabilities(self) -> EvidenceInventory:
        """Reject repeated capability keys even outside guided decoding."""
        if len(self.capabilities) != len(set(self.capabilities)):
            raise ValueError("capabilities must be unique")
        return self


class InstructionSelection(StrictModel):
    """Validated output from one explicitly configured selector."""

    candidate_id: str | None
    reason: Annotated[str, Field(min_length=1, max_length=240)]

    @property
    def status(self) -> Literal["SELECTED", "NO_SUITABLE_CANDIDATE"]:
        """Derive the decision so structured output cannot contradict its candidate ID."""
        return "SELECTED" if self.candidate_id is not None else "NO_SUITABLE_CANDIDATE"


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


class QuestionFit(StrictModel):
    """One judge's pre-answer question assessment."""

    local_anchor: Literal["MET", "NOT_MET", "UNKNOWN"]
    operation_coherent: Literal["MET", "NOT_MET", "UNKNOWN"]
    useful_request: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: Annotated[str, Field(min_length=1, max_length=240)]

    @property
    def aggregate(self) -> GateVerdict:
        """Reduce required question-fit checks without averaging."""
        values = {
            GateVerdict(self.local_anchor),
            GateVerdict(self.operation_coherent),
            GateVerdict(self.useful_request),
        }
        if GateVerdict.NOT_MET in values:
            return GateVerdict.NOT_MET
        if values == {GateVerdict.MET}:
            return GateVerdict.MET
        return GateVerdict.UNKNOWN


class AtomicClaim(StrictModel):
    """One factual assertion extracted from an answer."""

    text: Annotated[str, Field(min_length=1)]
    source_message_id: str
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(gt=0)]

    @property
    def claim_id(self) -> str:
        """Return a controller-owned identity derived from the exact public span."""
        return canonical_hash(self.model_dump(mode="json"))

    def validate_span(self, message: PublicMessage) -> None:
        """Validate that the claim span points to unchanged public text.

        Raises:
            ValueError: If the message or exact code-point span does not match.
        """
        if message.message_id != self.source_message_id:
            raise ValueError("claim source message does not exist")
        if self.end > len(message.content) or message.content[self.start : self.end] != self.text:
            raise ValueError("claim span does not match source text")


class ClaimInventory(StrictModel):
    """One judge's complete decomposition of an answer."""

    claims: Annotated[tuple[AtomicClaim, ...], Field(max_length=32)]
    coverage: GateVerdict
    reason: Annotated[str, Field(min_length=1, max_length=240)]


class RubricContext(StrictModel):
    """Controller-owned facts that determine which rubric items apply."""

    turn_index: Annotated[int, Field(ge=1, le=6)]
    profile: Literal["normal", "limitation", "false_premise"]
    has_natural_language_answer: bool
    requirements: tuple[Requirement, ...] = ()
    claims: tuple[AtomicClaim, ...] = ()
    computation_ids: tuple[str, ...] = ()
    history_binding_ids: tuple[str, ...] = ()
    exhaustive_scope_ids: tuple[str, ...] = ()
    requires_witness_check: bool = False

    @property
    def format_requirements(self) -> tuple[Requirement, ...]:
        """Return active requirements that impose a public output format."""
        return tuple(
            requirement for requirement in self.requirements if requirement.kind == "format"
        )

    @model_validator(mode="after")
    def validate_applicability_evidence(self) -> RubricContext:
        """Require unique evidence IDs and keep history evidence off the first turn."""
        id_groups = (
            self.computation_ids,
            self.history_binding_ids,
            self.exhaustive_scope_ids,
        )
        if any(not item for identifiers in id_groups for item in identifiers):
            raise ValueError("rubric applicability IDs must be non-empty")
        if any(len(identifiers) != len(set(identifiers)) for identifiers in id_groups):
            raise ValueError("rubric applicability IDs must be unique")
        if self.turn_index == 1 and (self.history_binding_ids or self.requires_witness_check):
            raise ValueError("first turn cannot contain history dependency evidence")
        return self


class RubricVerdict(StrictModel):
    """One model's output for a single applicable rubric criterion."""

    verdict: Literal["MET", "NOT_MET", "UNKNOWN"]
    reason: Annotated[str, Field(min_length=1, max_length=240)]


class RubricItem(StrictModel):
    """One instantiated evaluation criterion."""

    item_id: str
    template_id: str
    axis: str
    verdict: GateVerdict
    reason: Annotated[str, Field(min_length=1, max_length=240)]
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
    pattern: str
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
    """One text or image part in a training message."""

    type: Literal["text", "image"]
    text: str | None = None
    image: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> TrainingContent:
        """Require exactly one value matching the content kind."""
        if self.type == "text" and (self.text is None or self.image is not None):
            raise ValueError("text content requires only text")
        if self.type == "image" and (self.image is None or self.text is not None):
            raise ValueError("image content requires only image")
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
