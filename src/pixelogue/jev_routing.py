"""Bounded visual proposals and finite decisions for experimental task routing."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

from pydantic import Field, ValidationError, model_validator

from pixelogue.catalog import task_catalog
from pixelogue.config import DecisionRoutingConfig, RuntimeConfig, StrictModel, TaskRuntimeConfig
from pixelogue.contracts import InstructionCandidate, PublicMessage
from pixelogue.decision import DecisionClient, DecisionRequest, DecisionResult
from pixelogue.errors import ExecutionError
from pixelogue.serialization import canonical_hash, canonical_json, strict_json_object
from pixelogue.serving import ModelImage
from pixelogue.store import RunStore
from pixelogue.task_evidence import (
    CandidateBinding,
    CandidateBindings,
    CapabilityObservation,
    EligibilityObservation,
    ImageRegion,
    ObservationVerdict,
    PublicParameter,
    ScopedEvidenceInventory,
    ScopeEvidence,
    TargetReport,
)
from pixelogue.task_runtime import bindable_parameter_names, required_parameter_names

ShortText = Annotated[str, Field(min_length=1, max_length=160)]
TargetText = Annotated[str, Field(min_length=1, max_length=80)]
ROUTING_CONTRACT = "visual-proposal-decision-v2"
SUPPORTED_BINDING_TASKS = frozenset(
    {
        "object_identification",
        "attribute_lookup",
        "entity_count",
        "text_transcription",
        "label_value_linking",
        "document_structure_reconstruction",
        "table_cell_lookup",
        "table_predicate_selection",
        "table_structure_reconstruction",
        "chart_value_lookup",
        "chart_comparison",
        "chart_trend_summary",
        "diagram_element_lookup",
        "graph_connectivity",
        "graph_path_tracing",
        "ui_element_grounding",
        "ui_state_reading",
        "screen_summary",
    }
)
CONTROLLER_TARGET_TASKS = frozenset(
    {"object_identification", "text_transcription", "screen_summary"}
)


class ObservationProposal(StrictModel):
    """An actual visual claim without a generated verdict or identifier."""

    capability: TargetText
    region: ImageRegion
    detail: ShortText


class EvidenceScopeProposal(StrictModel):
    """Located visual content; the controller supplies scope and view identities."""

    public_description: ShortText
    object_label: TargetText | None = None
    region: ImageRegion
    observations: Annotated[tuple[ObservationProposal, ...], Field(max_length=8)]

    @model_validator(mode="after")
    def validate_local_claims(self) -> EvidenceScopeProposal:
        """Reject repeated capabilities and observations outside the proposed scope."""
        names = [item.capability for item in self.observations]
        if len(names) != len(set(names)):
            raise ValueError("Capability proposals must be unique within a scope")
        if self.object_label is not None and "visible_entity" not in names:
            raise ValueError("A proposed object label requires a visible_entity claim")
        for observation in self.observations:
            if not region_contains(self.region, observation.region):
                raise ValueError("Observation proposal lies outside its scope")
        return self


class EvidenceProposals(StrictModel):
    """Sparse image extraction that delegates each affirmative claim to a classifier."""

    scopes: Annotated[tuple[EvidenceScopeProposal, ...], Field(max_length=8)]


class BindingProposal(StrictModel):
    """Concrete public choices without repeating admission checks or estimates."""

    candidate_id: TargetText
    target: TargetReport
    public_parameters: Annotated[tuple[PublicParameter, ...], Field(max_length=19)] = ()


class BindingProposals(StrictModel):
    """A bounded proposal set for the supplied candidate identifiers only."""

    bindings: Annotated[tuple[BindingProposal, ...], Field(max_length=8)]


@dataclass(frozen=True)
class RoutingDecision:
    """One independent three-way visual question for a local claim or condition."""

    query_id: str
    question: str
    region: ImageRegion
    public_context: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProposalRejection:
    """A candidate-local proposal failure retained alongside usable siblings."""

    candidate_id: str
    reason: str
    message: str


def region_contains(parent: ImageRegion, child: ImageRegion) -> bool:
    """Check coordinates without expanding either supplied image region."""
    return (
        parent.left <= child.left < child.right <= parent.right
        and parent.top <= child.top < child.bottom <= parent.bottom
    )


def evidence_decisions(
    proposals: EvidenceProposals, settings: TaskRuntimeConfig
) -> tuple[RoutingDecision, ...]:
    """Build independent capability questions, rejecting invalid proposals first."""
    catalog = task_catalog()
    if len(proposals.scopes) > settings.max_scopes:
        raise ExecutionError("EVIDENCE_SCOPE_LIMIT", "Too many proposed scopes")
    queries: list[RoutingDecision] = []
    for index, scope in enumerate(proposals.scopes, 1):
        if len(scope.observations) > settings.max_observations_per_scope:
            raise ExecutionError("EVIDENCE_OBSERVATION_LIMIT", "Too many proposed observations")
        for observation_index, observation in enumerate(scope.observations, 1):
            if observation.capability not in catalog.capabilities:
                raise ExecutionError("EVIDENCE_CAPABILITY_UNKNOWN", observation.capability)
            label_context = (
                f"Proposed category: {scope.object_label}. "
                if observation.capability == "visible_entity" and scope.object_label is not None
                else ""
            )
            queries.append(
                RoutingDecision(
                    query_id=f"scope_{index}/observation_{observation_index}",
                    question=(
                        "Inspect the image within the specified region. Is this visual claim "
                        "true, and does it support the stated capability? "
                        f"Claim: {observation.detail}. {label_context}"
                        f"Capability: {observation.capability}: "
                        f"{catalog.capabilities[observation.capability]}. "
                        "MET requires both a true claim and sufficient visible support. "
                        "NOT_MET means a clearly false claim or clearly absent support. "
                        "UNKNOWN means unreadable, ambiguous, incomplete, or unsupported inference."
                    ),
                    region=observation.region,
                )
            )
    return tuple(queries)


def assemble_evidence(
    proposals: EvidenceProposals,
    verdicts: Mapping[str, ObservationVerdict],
    *,
    image_id: str,
    view_id: str,
    settings: TaskRuntimeConfig,
) -> ScopedEvidenceInventory:
    """Keep finite verdicts and original visual content; omitted decisions are errors."""
    expected = {query.query_id for query in evidence_decisions(proposals, settings)}
    if set(verdicts) != expected:
        raise ExecutionError("DECISION_RESULTS_MISMATCH", "Missing or extra evidence decisions")
    scopes: list[ScopeEvidence] = []
    for index, scope in enumerate(proposals.scopes, 1):
        observations = tuple(
            CapabilityObservation(
                evidence_id=f"obs_{index}_{observation_index}",
                capability=observation.capability,
                verdict=verdicts[f"scope_{index}/observation_{observation_index}"],
                region=observation.region,
                detail=observation.detail,
            )
            for observation_index, observation in enumerate(scope.observations, 1)
        )
        verified_label = (
            scope.object_label
            if any(
                item.capability == "visible_entity" and item.verdict == "MET"
                for item in observations
            )
            else None
        )
        scopes.append(
            ScopeEvidence(
                scope_id=f"scope_{index}",
                view_id=view_id,
                public_description=scope.public_description,
                object_label=verified_label,
                region=scope.region,
                observations=observations,
            )
        )
    return ScopedEvidenceInventory(
        image_id=image_id,
        scopes=tuple(scopes),
        reason="Located generator proposals checked by independent finite visual decisions",
    )


def validate_evidence_proposals(proposals: EvidenceProposals, settings: TaskRuntimeConfig) -> None:
    """Apply bounded vocabulary checks without manufacturing any visual verdict."""
    evidence_decisions(proposals, settings)


def controller_binding_proposals(
    candidates: tuple[InstructionCandidate, ...], inventory: ScopedEvidenceInventory
) -> tuple[BindingProposals, tuple[InstructionCandidate, ...]]:
    """Bind target-only operations from existing typed anchors; defer open choices."""
    scopes = {scope.scope_id: scope for scope in inventory.scopes}
    tasks = {task.id: task for task in task_catalog().tasks}
    proposals: list[BindingProposal] = []
    unresolved: list[InstructionCandidate] = []
    for candidate in candidates:
        task = tasks[candidate.task_id]
        if candidate.scope_id is None or candidate.scope_id not in scopes:
            raise ExecutionError("CANDIDATE_SCOPE_MISSING", "Candidate scope is absent")
        scope = scopes[candidate.scope_id]
        if (
            candidate.profile != "normal"
            or task.id not in CONTROLLER_TARGET_TASKS
            or required_parameter_names(task)
        ):
            unresolved.append(candidate)
            continue
        refs = tuple(
            item.evidence_id
            for item in scope.observations
            if item.capability in candidate.required_capabilities and item.verdict == "MET"
        )
        if not refs:
            unresolved.append(candidate)
            continue
        if task.id == "object_identification":
            entity = next(
                (
                    item
                    for item in scope.observations
                    if item.capability == "visible_entity"
                    and item.verdict == "MET"
                    and scope.object_label is not None
                ),
                None,
            )
            if entity is None:
                unresolved.append(candidate)
                continue
            target = TargetReport(
                value=f"ref:{entity.evidence_id}",
                origin="image",
                evidence_refs=(entity.evidence_id,),
            )
        else:
            target = TargetReport(
                value=scope.public_description, origin="image", evidence_refs=refs
            )
        proposals.append(BindingProposal(candidate_id=candidate.candidate_id, target=target))
    return BindingProposals(bindings=tuple(proposals)), tuple(unresolved)


def validate_binding_proposals(
    proposals: BindingProposals,
    candidates: tuple[InstructionCandidate, ...],
    inventory: ScopedEvidenceInventory,
    history: tuple[PublicMessage, ...],
) -> None:
    """Reject unknown identifiers, parameter names and nonlocal references before judging."""
    templates = {candidate.candidate_id: candidate for candidate in candidates}
    scopes = {scope.scope_id: scope for scope in inventory.scopes}
    tasks = {task.id: task for task in task_catalog().tasks}
    seen: set[str] = set()
    history_ids = {message.message_id for message in history}
    for proposal in proposals.bindings:
        if proposal.candidate_id not in templates or proposal.candidate_id in seen:
            raise ExecutionError("CANDIDATE_BINDING_ID", "Unknown or repeated proposal")
        seen.add(proposal.candidate_id)
        candidate = templates[proposal.candidate_id]
        task = tasks[candidate.task_id]
        if candidate.scope_id is None or candidate.scope_id not in scopes:
            raise ExecutionError("CANDIDATE_SCOPE_MISSING", "Candidate scope is absent")
        scope = scopes[candidate.scope_id]
        if candidate.view_id != scope.view_id:
            raise ExecutionError("EVIDENCE_VIEW_MISMATCH", "Candidate uses another image view")
        local_refs = {item.evidence_id for item in scope.observations}
        try:
            target = PublicParameter(name="target", **proposal.target.model_dump())
        except ValidationError as error:
            raise ExecutionError(
                "CANDIDATE_PARAMETER_SOURCE",
                "Target origin and factual source references are inconsistent",
            ) from error
        parameters = (target, *proposal.public_parameters)
        names = [parameter.name for parameter in parameters]
        if len(names) != len(set(names)) or not set(names) <= set(bindable_parameter_names(task)):
            raise ExecutionError("CANDIDATE_PARAMETER_UNKNOWN", "Unknown or repeated parameter")
        if missing := set(required_parameter_names(task, candidate.profile)) - set(names):
            raise ExecutionError(
                "CANDIDATE_PARAMETER_MISSING", f"Missing parameters: {sorted(missing)}"
            )
        for parameter in parameters:
            permitted = local_refs if parameter.origin == "image" else history_ids
            if parameter.origin != "instruction" and not set(parameter.evidence_refs) <= permitted:
                raise ExecutionError("CANDIDATE_PARAMETER_SOURCE", "Unknown or nonlocal reference")
            choices = task.parameters.get(parameter.name)
            if isinstance(choices, tuple) and parameter.value not in choices:
                raise ExecutionError("CANDIDATE_PARAMETER_VALUE", "Unsupported public choice")


def partition_binding_proposals(
    proposals: BindingProposals,
    candidates: tuple[InstructionCandidate, ...],
    inventory: ScopedEvidenceInventory,
    history: tuple[PublicMessage, ...],
) -> tuple[BindingProposals, tuple[ProposalRejection, ...]]:
    """Keep structurally valid siblings while rejecting global identifier ambiguity."""
    allowed = {candidate.candidate_id for candidate in candidates}
    identifiers = [proposal.candidate_id for proposal in proposals.bindings]
    if len(identifiers) != len(set(identifiers)) or not set(identifiers) <= allowed:
        raise ExecutionError("CANDIDATE_BINDING_ID", "Unknown or repeated proposal")
    usable: list[BindingProposal] = []
    rejected: list[ProposalRejection] = []
    local_errors = {
        "CANDIDATE_PARAMETER_UNKNOWN",
        "CANDIDATE_PARAMETER_MISSING",
        "CANDIDATE_PARAMETER_SOURCE",
        "CANDIDATE_PARAMETER_VALUE",
    }
    for proposal in proposals.bindings:
        try:
            validate_binding_proposals(
                BindingProposals(bindings=(proposal,)), candidates, inventory, history
            )
        except ExecutionError as error:
            if error.reason not in local_errors:
                raise
            rejected.append(ProposalRejection(proposal.candidate_id, error.reason, str(error)))
        else:
            usable.append(proposal)
    return BindingProposals(bindings=tuple(usable)), tuple(rejected)


def validate_usable_binding_proposals(
    proposals: BindingProposals,
    candidates: tuple[InstructionCandidate, ...],
    inventory: ScopedEvidenceInventory,
    history: tuple[PublicMessage, ...],
) -> None:
    """Retry a wholly invalid response while allowing useful independent candidates."""
    usable, rejected = partition_binding_proposals(proposals, candidates, inventory, history)
    if not usable.bindings and rejected:
        raise ExecutionError(rejected[0].reason, rejected[0].message)


def binding_decisions(
    proposals: BindingProposals,
    candidates: tuple[InstructionCandidate, ...],
    inventory: ScopedEvidenceInventory,
    history: tuple[PublicMessage, ...],
) -> tuple[RoutingDecision, ...]:
    """Ask every required condition separately without exposing another judge's verdict."""
    validate_binding_proposals(proposals, candidates, inventory, history)
    templates = {candidate.candidate_id: candidate for candidate in candidates}
    scopes = {scope.scope_id: scope for scope in inventory.scopes}
    catalog = task_catalog()
    tasks = {task.id: task for task in catalog.tasks}
    public_context = tuple(
        "Committed public message: "
        + canonical_json(message.model_dump(mode="json")).decode("utf-8")
        for message in history
    )
    queries: list[RoutingDecision] = []
    for proposal in proposals.bindings:
        candidate = templates[proposal.candidate_id]
        task = tasks[candidate.task_id]
        if candidate.scope_id is None:
            raise ExecutionError("CANDIDATE_SCOPE_MISSING", "Candidate scope is absent")
        scope = scopes[candidate.scope_id]
        required_capabilities = set(candidate.required_capabilities)
        referenced_image_ids = set(
            proposal.target.evidence_refs if proposal.target.origin == "image" else ()
        )
        for parameter in proposal.public_parameters:
            if parameter.origin == "image":
                referenced_image_ids.update(parameter.evidence_refs)
        claim_context = tuple(
            "Unverified local visual claim: "
            + canonical_json(
                {
                    "evidence_id": item.evidence_id,
                    "capability": item.capability,
                    "region": item.region.model_dump(mode="json"),
                    "detail": item.detail,
                }
            ).decode("utf-8")
            for item in scope.observations
            if item.capability in required_capabilities or item.evidence_id in referenced_image_ids
        )
        binding_context = (
            "Public operation target: "
            + canonical_json(
                {
                    "target": proposal.target.model_dump(mode="json"),
                    "scope_id": scope.scope_id,
                    "view_id": scope.view_id,
                    "scope_description": scope.public_description,
                    "proposed_object_label": scope.object_label,
                }
            ).decode("utf-8"),
            *(
                "Public operation parameter: "
                + canonical_json(parameter.model_dump(mode="json")).decode("utf-8")
                for parameter in proposal.public_parameters
            ),
        )
        for check_id in task.eligibility_checks:
            queries.append(
                RoutingDecision(
                    query_id=f"{proposal.candidate_id}/{check_id}",
                    question=(
                        f"Inspect the specified image region for this operation: {task.definition_en} "
                        "Use the exact public target, parameters, committed history and unverified "
                        "local visual claims in the context. Check those claims against the image. "
                        f"Is this condition satisfied: {catalog.eligibility_checks[check_id]}? "
                        "MET requires direct visual support and the stated public conditions. "
                        "NOT_MET means the condition is clearly violated. UNKNOWN means ambiguity, "
                        "missing evidence, unreadable content, or incomplete scope. Do not infer a "
                        "condition merely because the proposal is well formed."
                    ),
                    region=scope.region,
                    public_context=(*public_context, *binding_context, *claim_context),
                )
            )
    return tuple(queries)


def assemble_bindings(
    proposals: BindingProposals,
    verdicts: Mapping[str, ObservationVerdict],
    candidates: tuple[InstructionCandidate, ...],
    inventory: ScopedEvidenceInventory,
    history: tuple[PublicMessage, ...],
    *,
    answer_max_tokens: int,
) -> CandidateBindings:
    """Construct every catalog check explicitly; missing decisions never become MET."""
    queries = binding_decisions(proposals, candidates, inventory, history)
    if set(verdicts) != {query.query_id for query in queries}:
        raise ExecutionError("DECISION_RESULTS_MISMATCH", "Missing or extra binding decisions")
    templates = {candidate.candidate_id: candidate for candidate in candidates}
    scopes = {scope.scope_id: scope for scope in inventory.scopes}
    tasks = {task.id: task for task in task_catalog().tasks}
    bindings: list[CandidateBinding] = []
    for proposal in proposals.bindings:
        candidate = templates[proposal.candidate_id]
        if candidate.scope_id is None:
            raise ExecutionError("CANDIDATE_SCOPE_MISSING", "Candidate scope is absent")
        scope = scopes[candidate.scope_id]
        refs = {
            item.evidence_id
            for item in scope.observations
            if item.capability in candidate.required_capabilities and item.verdict == "MET"
        }
        refs.update(proposal.target.evidence_refs if proposal.target.origin == "image" else ())
        if not refs:
            raise ExecutionError("CANDIDATE_EVIDENCE_SCOPE", "No local evidence for proposal")
        bindings.append(
            CandidateBinding(
                candidate_id=proposal.candidate_id,
                public_parameters=(
                    PublicParameter(name="target", **proposal.target.model_dump()),
                    *proposal.public_parameters,
                ),
                checks=tuple(
                    EligibilityObservation(
                        check_id=check_id,
                        verdict=verdicts[f"{proposal.candidate_id}/{check_id}"],
                        reason=f"Finite visual decision for {check_id}; original proposal retained privately",
                    )
                    for check_id in tasks[candidate.task_id].eligibility_checks
                ),
                evidence_refs=tuple(sorted(refs)),
                estimated_answer_tokens=answer_max_tokens,
            )
        )
    return CandidateBindings(bindings=tuple(bindings))


class StoredDecisionRouter:
    """Journal finite calls and reuse the exact same trial after interruption."""

    def __init__(
        self,
        client: DecisionClient,
        settings: DecisionRoutingConfig,
        runtime: RuntimeConfig,
        store: RunStore,
    ) -> None:
        """Keep accounting separate from native inference and domain assembly."""
        self.client, self.settings, self.runtime, self.store = client, settings, runtime, store

    def decide(
        self,
        query: RoutingDecision,
        *,
        image_id: str,
        image: ModelImage,
        trial_id: str,
    ) -> ObservationVerdict:
        """Preserve request identity, actual cost and a private response artifact."""
        repository, revision = self.settings.model_repository, self.settings.model_revision
        if repository is None or revision is None:
            raise ExecutionError("DECISION_MODEL_UNCONFIGURED", "Classifier identity is required")
        if hashlib.sha256(image.path.read_bytes()).hexdigest() != image.encoded_sha256:
            raise ExecutionError("IMAGE_BYTES_MISMATCH", "Decision image bytes changed")
        try:
            request = DecisionRequest(
                request_id=canonical_hash({"trial": trial_id, "query": query.query_id}),
                trial_id=trial_id,
                image_id=image_id,
                image_path=str(image.path),
                image_sha256=image.encoded_sha256,
                view_id=image.view_id,
                region=query.region,
                question=query.question,
                public_context=query.public_context,
                contract_version=ROUTING_CONTRACT,
            )
        except ValidationError as error:
            raise ExecutionError("DECISION_REQUEST_INVALID", str(error)) from error
        request_hash = request.identity(repository, revision)
        model_lock_hash = canonical_hash(
            {"repository": repository, "revision": revision, "contract": ROUTING_CONTRACT}
        )
        stage = "finite_routing_decision"
        saved = self.store.saved_model_attempt(model_lock_hash, stage, request_hash)
        if saved is not None:
            envelope = strict_json_object(saved[0])
            result = DecisionResult.model_validate_json(canonical_json(envelope["decision"]))
            self._validate_result(request, request_hash, result, repository, revision)
            self.store.write_json_artifact(
                "finite-decision-cache",
                {
                    "request_hash": request_hash,
                    "trial_id": trial_id,
                    "cache_used": True,
                },
            )
            return result.verdict
        request_artifact = self.store.write_json_artifact(
            "finite-decision-requests", request.model_dump(mode="json")
        )
        attempt = self.store.begin_model_attempt(
            stage=stage,
            model_repo=repository,
            model_revision=revision,
            model_lock_hash=model_lock_hash,
            request_hash=request_hash,
            request_artifact_hash=request_artifact,
            # Native JEV can perform a second one-token log-probability readout.
            # This is capacity; the result's observed usage/request count stays unchanged.
            reserved_output_tokens=2 if repository == "autotrust/JEV-27B-VL" else 1,
            request_limit=self.runtime.max_total_requests,
            output_token_limit=self.runtime.max_total_output_tokens,
            transport_attempt=1,
        )
        started = time.perf_counter()
        try:
            result = self.client.decide(request)
        except Exception as error:
            self.store.finish_model_attempt(
                attempt, "FAILED", round((time.perf_counter() - started) * 1000)
            )
            self.store.write_json_artifact(
                "finite-decision-errors",
                {
                    "request_hash": request_hash,
                    "error_type": type(error).__name__,
                    "message": str(error),
                },
            )
            raise
        envelope = {"decision": result.model_dump(mode="json")}
        if result.prompt_tokens is not None or result.completion_tokens is not None:
            envelope["usage"] = {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            }
        self.store.save_model_attempt_response(
            attempt,
            canonical_json(envelope),
            round((time.perf_counter() - started) * 1000),
            200,
        )
        try:
            self._validate_result(request, request_hash, result, repository, revision)
        except ExecutionError:
            self.store.finish_model_attempt(attempt, "INVALID")
            raise
        self.store.finish_model_attempt(attempt, "COMPLETE")
        return result.verdict

    @staticmethod
    def _validate_result(
        request: DecisionRequest,
        request_hash: str,
        result: DecisionResult,
        repository: str,
        revision: str,
    ) -> None:
        if (
            result.request_id != request.request_id
            or result.request_hash != request_hash
            or result.model_repository != repository
            or result.model_revision != revision
        ):
            raise ExecutionError(
                "DECISION_RESULT_IDENTITY", "Classifier response identity mismatch"
            )
