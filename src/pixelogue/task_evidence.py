"""Scope-bound observations and answer-independent public operation bindings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from pixelogue.config import StrictModel
from pixelogue.errors import ExecutionError

Nonempty = Annotated[str, Field(min_length=1, max_length=512)]
ObjectLabel = Annotated[str, Field(min_length=1, max_length=80)]
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
    object_label: ObjectLabel | None = None
    region: ImageRegion
    observations: Annotated[tuple[CapabilityObservation, ...], Field(max_length=50)]

    @model_validator(mode="after")
    def validate_observations(self) -> ScopeEvidence:
        """Require unique observations located inside their bound scope."""
        capabilities = [item.capability for item in self.observations]
        if len(capabilities) != len(set(capabilities)):
            raise ValueError("Capabilities must be unique within a scope")
        if self.object_label is not None and not any(
            item.capability == "visible_entity" and item.verdict == "MET"
            for item in self.observations
        ):
            raise ValueError("An object label requires a visible entity observation")
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


def alias_evidence_ids(
    inventory: ScopedEvidenceInventory,
) -> tuple[ScopedEvidenceInventory, dict[str, str]]:
    """Assign compact controller IDs while preserving every scope and observation."""
    aliases: dict[str, str] = {}
    scopes: list[ScopeEvidence] = []
    for scope in inventory.scopes:
        observations: list[CapabilityObservation] = []
        for observation in scope.observations:
            alias = f"e{len(aliases) + 1}"
            aliases[observation.evidence_id] = alias
            observations.append(observation.model_copy(update={"evidence_id": alias}))
        scopes.append(scope.model_copy(update={"observations": tuple(observations)}))
    return inventory.model_copy(update={"scopes": tuple(scopes)}), aliases


class CapabilityReport(StrictModel):
    """One model observation keyed by its capability in the wire response."""

    evidence_id: Nonempty
    verdict: ObservationVerdict
    region: ImageRegion
    detail: Nonempty


class ScopeEvidenceReport(StrictModel):
    """Wire scope whose capability keys cannot repeat in valid JSON."""

    scope_id: Nonempty
    view_id: Nonempty
    public_description: Nonempty
    object_label: ObjectLabel | None = None
    region: ImageRegion
    observations: Annotated[dict[str, CapabilityReport], Field(max_length=50)]


class ScopedEvidenceReport(StrictModel):
    """Model-facing routing response with capabilities as object keys."""

    image_id: Nonempty
    scopes: Annotated[tuple[ScopeEvidenceReport, ...], Field(max_length=8)]
    reason: Nonempty

    def to_inventory(self) -> ScopedEvidenceInventory:
        """Validate the existing internal scope contract after wire decoding."""
        try:
            return ScopedEvidenceInventory(
                image_id=self.image_id,
                reason=self.reason,
                scopes=tuple(
                    ScopeEvidence(
                        scope_id=scope.scope_id,
                        view_id=scope.view_id,
                        public_description=scope.public_description,
                        object_label=scope.object_label,
                        region=scope.region,
                        observations=tuple(
                            CapabilityObservation(
                                capability=capability,
                                evidence_id=report.evidence_id,
                                verdict=report.verdict,
                                region=report.region,
                                detail=report.detail,
                            )
                            for capability, report in scope.observations.items()
                        ),
                    )
                    for scope in self.scopes
                ),
            )
        except ValidationError as error:
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", str(error)) from error


class AttributeRecheckReport(StrictModel):
    """One answer-blind visual property check for an existing image scope."""

    image_id: Nonempty
    scope_id: Nonempty
    view_id: Nonempty
    verdict: Literal["MET", "UNKNOWN"]
    region: ImageRegion
    detail: Nonempty


class ArrayScopeEvidenceReport(StrictModel):
    """Wire scope with capability names carried by observations in any order."""

    scope_id: Nonempty
    view_id: Nonempty
    public_description: Nonempty
    object_label: ObjectLabel | None = None
    region: ImageRegion
    observations: Annotated[tuple[CapabilityObservation, ...], Field(max_length=50)]


class ArrayScopedEvidenceReport(StrictModel):
    """Alternative routing contract preserving the existing internal evidence rules."""

    image_id: Nonempty
    scopes: Annotated[tuple[ArrayScopeEvidenceReport, ...], Field(max_length=8)]
    reason: Nonempty

    def to_inventory(self) -> ScopedEvidenceInventory:
        """Reject duplicates and invalid regions without depending on observation order."""
        try:
            return ScopedEvidenceInventory.model_validate(self.model_dump())
        except ValidationError as error:
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", str(error)) from error


class CompactCapabilityObservation(StrictModel):
    """Visual content without a controller-owned observation identity."""

    capability: Nonempty
    verdict: ObservationVerdict
    region: ImageRegion
    detail: Nonempty


class CompactScopeEvidenceReport(StrictModel):
    """Scope content without copying fixed image, view or scope identities."""

    public_description: Nonempty
    object_label: ObjectLabel | None = None
    region: ImageRegion
    observations: Annotated[tuple[CompactCapabilityObservation, ...], Field(max_length=50)]


class CompactScopedEvidenceReport(StrictModel):
    """Comparison format retaining visual detail and every observation region."""

    scopes: Annotated[tuple[CompactScopeEvidenceReport, ...], Field(max_length=8)]
    reason: Nonempty

    def to_inventory(self, *, image_id: str, view_id: str) -> ScopedEvidenceInventory:
        """Assign deterministic local identities, then apply the full internal contract."""
        try:
            return ScopedEvidenceInventory(
                image_id=image_id,
                reason=self.reason,
                scopes=tuple(
                    ScopeEvidence(
                        scope_id=f"scope_{index}",
                        view_id=view_id,
                        public_description=scope.public_description,
                        object_label=scope.object_label,
                        region=scope.region,
                        observations=tuple(
                            CapabilityObservation(
                                evidence_id=f"obs_{index}_{observation_index}",
                                **observation.model_dump(),
                            )
                            for observation_index, observation in enumerate(scope.observations, 1)
                        ),
                    )
                    for index, scope in enumerate(self.scopes, 1)
                ),
            )
        except ValidationError as error:
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", str(error)) from error


def bind_evidence_identity(
    report: ScopedEvidenceReport | ArrayScopedEvidenceReport | CompactScopedEvidenceReport,
    image_id: str,
    view_id: str,
) -> ScopedEvidenceInventory:
    """Bind fixed identities only for the compact format; reject mismatches in other formats."""
    inventory = (
        report.to_inventory(image_id=image_id, view_id=view_id)
        if isinstance(report, CompactScopedEvidenceReport)
        else report.to_inventory()
    )
    if inventory.image_id != image_id:
        raise ExecutionError("EVIDENCE_IMAGE_MISMATCH", "Evidence refers to another image")
    return inventory


def unsupported_object_label_feedback(
    report: ScopedEvidenceReport | ArrayScopedEvidenceReport | CompactScopedEvidenceReport,
) -> str:
    """Identify invalid labeled scopes without copying model-generated scope text."""
    invalid = [
        index
        for index, scope in enumerate(report.scopes, start=1)
        if scope.object_label is not None
        and not any(
            capability == "visible_entity" and observation.verdict == "MET"
            for capability, observation in (
                scope.observations.items()
                if isinstance(scope.observations, dict)
                else ((item.capability, item) for item in scope.observations)
            )
        )
    ]
    if not invalid:
        return ""
    listed = ", ".join(str(index) for index in invalid[:3])
    return (
        f" Scope number(s) {listed} set object_label without a MET visible_entity in the"
        " same scope. Add that observation only if the entity is directly visible; otherwise"
        " set object_label=null. Keep a subject's visible attributes in its entity scope."
    )


def out_of_scope_region_feedback(
    report: ScopedEvidenceReport | ArrayScopedEvidenceReport | CompactScopedEvidenceReport,
) -> str:
    """Describe invalid nested boxes without echoing model-generated text."""
    mismatches: list[str] = []
    for scope_index, scope in enumerate(report.scopes, start=1):
        parent = scope.region
        observations = (
            scope.observations.values()
            if isinstance(scope.observations, dict)
            else scope.observations
        )
        for observation_index, observation in enumerate(observations, start=1):
            region = observation.region
            if (
                parent.left <= region.left < region.right <= parent.right
                and parent.top <= region.top < region.bottom <= parent.bottom
            ):
                continue
            mismatches.append(
                f"scope {scope_index} observation {observation_index}:"
                f" observation box [{region.left:g}, {region.top:g},"
                f" {region.right:g}, {region.bottom:g}] exceeds parent box"
                f" [{parent.left:g}, {parent.top:g}, {parent.right:g}, {parent.bottom:g}]"
            )
    if not mismatches:
        return ""
    return (
        " Invalid boxes in the previous response: "
        + "; ".join(mismatches[:3])
        + ". Redraw each observation inside its actual parent region."
        " If evidence spans the full view, define a full-view scope."
    )


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
        """Require a target and reject duplicate parameters or checks."""
        if not any(parameter.name == "target" for parameter in self.public_parameters):
            raise ValueError("Every binding requires a target parameter")
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


class TargetReport(StrictModel):
    """Required target value without a model-generated parameter name."""

    value: Nonempty | int | bool | tuple[Nonempty, ...]
    origin: Literal["instruction", "image", "history"]
    evidence_refs: tuple[Nonempty, ...] = ()


class CandidateBindingReport(StrictModel):
    """Wire binding with target enforced by a required JSON property."""

    candidate_id: Nonempty
    target: TargetReport
    public_parameters: Annotated[tuple[PublicParameter, ...], Field(max_length=19)]
    checks: Annotated[tuple[EligibilityObservation, ...], Field(max_length=16)]
    evidence_refs: Annotated[tuple[Nonempty, ...], Field(min_length=1, max_length=50)]
    estimated_answer_tokens: Annotated[int, Field(ge=1)]


@dataclass(frozen=True)
class BindingParseRejection:
    """One rejected wire binding whose siblings may still be useful."""

    candidate_id: str
    reason: str
    message: str


class CandidateBindingsReport(StrictModel):
    """Model-facing candidate response convertible to the internal contract."""

    bindings: Annotated[tuple[CandidateBindingReport, ...], Field(max_length=32)]

    @staticmethod
    def _convert(binding: CandidateBindingReport) -> CandidateBinding:
        """Apply the internal source and uniqueness checks to one wire binding."""
        return CandidateBinding(
            candidate_id=binding.candidate_id,
            public_parameters=(
                PublicParameter(name="target", **binding.target.model_dump()),
                *binding.public_parameters,
            ),
            checks=binding.checks,
            evidence_refs=binding.evidence_refs,
            estimated_answer_tokens=binding.estimated_answer_tokens,
        )

    def to_bindings(self) -> CandidateBindings:
        """Validate target sourcing and reject duplicate parameter names."""
        try:
            return CandidateBindings(
                bindings=tuple(self._convert(binding) for binding in self.bindings)
            )
        except ValidationError as error:
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", str(error)) from error

    def partition_bindings(
        self, allowed_ids: frozenset[str]
    ) -> tuple[CandidateBindings, tuple[BindingParseRejection, ...]]:
        """Keep valid siblings after checking all candidate identities together."""
        ids = [binding.candidate_id for binding in self.bindings]
        if len(ids) != len(set(ids)) or not set(ids) <= allowed_ids:
            raise ExecutionError("CANDIDATE_BINDING_ID", "Unknown or repeated candidate binding")
        accepted: list[CandidateBinding] = []
        rejected: list[BindingParseRejection] = []
        for binding in self.bindings:
            target = binding.target
            if (target.origin == "instruction" and target.evidence_refs) or (
                target.origin != "instruction" and not target.evidence_refs
            ):
                rejected.append(
                    BindingParseRejection(
                        candidate_id=binding.candidate_id,
                        reason="CANDIDATE_PARAMETER_SOURCE",
                        message=f"Candidate {binding.candidate_id}: target origin and evidence references conflict",
                    )
                )
                continue
            try:
                accepted.append(self._convert(binding))
            except ValidationError as error:
                message = str(error.errors()[0]["msg"])
                rejected.append(
                    BindingParseRejection(
                        candidate_id=binding.candidate_id,
                        reason="CANDIDATE_PARAMETER_VALUE",
                        message=f"Candidate {binding.candidate_id}: {message[:500]}",
                    )
                )
        return CandidateBindings(bindings=tuple(accepted)), tuple(rejected)


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
