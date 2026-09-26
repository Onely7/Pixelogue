"""Scope-bound observations and answer-independent public operation bindings."""

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


class CapabilityObservation(StrictModel):
    """A local capability claim with its own evidence identity and location."""

    evidence_id: Nonempty
    capability: Nonempty
    verdict: ObservationVerdict
    region: ImageRegion
    detail: Nonempty


class ScopeEvidence(StrictModel):
    """Observations that must never be pooled across unrelated scopes."""

    scope_id: Nonempty
    view_id: Nonempty
    public_description: Nonempty
    region: ImageRegion
    observations: Annotated[tuple[CapabilityObservation, ...], Field(max_length=50)]

    @model_validator(mode="after")
    def validate_observations(self) -> ScopeEvidence:
        """Require unique observations located inside their bound scope."""
        capabilities = [item.capability for item in self.observations]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Capabilities must be unique within a scope")
        for item in self.observations:
            if not (
                self.region.left <= item.region.left < item.region.right <= self.region.right
                and self.region.top <= item.region.top < item.region.bottom <= self.region.bottom
            ):
                raise ValueError("Evidence location is outside its scope")
        return self


class ScopedEvidenceInventory(StrictModel):
    """Bounded routing observations; missing observations mean UNKNOWN."""

    image_id: Nonempty
    scopes: Annotated[tuple[ScopeEvidence, ...], Field(max_length=8)]
    reason: Nonempty

    @model_validator(mode="after")
    def validate_ids(self) -> ScopedEvidenceInventory:
        """Prevent ambiguous scope and evidence references."""
        scopes = [scope.scope_id for scope in self.scopes]
        refs = [item.evidence_id for scope in self.scopes for item in scope.observations]
        if len(scopes) != len(set(scopes)) or len(refs) != len(set(refs)):
            raise ValueError("Scope and evidence IDs must be unique")
        return self


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


class EligibilityObservation(StrictModel):
    """One admission check; no answer or acceptance target is included."""

    check_id: Nonempty
    verdict: ObservationVerdict
    reason: Nonempty


class CandidateBinding(StrictModel):
    """Focused operation proposal made before selection or answer generation."""

    candidate_id: Nonempty
    public_parameters: Annotated[tuple[PublicParameter, ...], Field(max_length=20)]
    checks: Annotated[tuple[EligibilityObservation, ...], Field(max_length=16)]
    evidence_refs: Annotated[tuple[Nonempty, ...], Field(min_length=1, max_length=50)]
    estimated_answer_tokens: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def validate_unique_names(self) -> CandidateBinding:
        """Reject contradictory duplicate parameters or eligibility checks."""
        for names in (
            [parameter.name for parameter in self.public_parameters],
            [check.check_id for check in self.checks],
            list(self.evidence_refs),
        ):
            if len(names) != len(set(names)):
                raise ValueError("Binding names and references must be unique")
        return self


class CandidateBindings(StrictModel):
    """A bounded response for controller-provided candidate templates only."""

    bindings: Annotated[tuple[CandidateBinding, ...], Field(max_length=32)]


class TranscriptInventory(StrictModel):
    """Independent transcription aligned with the public answer's exact text."""

    coverage: ObservationVerdict
    expected_text: str
    answer_text: str
    reason: Nonempty


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
