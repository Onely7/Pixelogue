"""Turn-by-turn synthesis coordinator with explicit information barriers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pixelogue.catalog import load_task_catalog, task_catalog
from pixelogue.config import ModelEndpoint, PixelogueConfig
from pixelogue.contracts import (
    AtomicClaim,
    ClaimExtraction,
    ClaimInventory,
    ClaimSpan,
    ConversationArtifact,
    EvidenceInventory,
    GateVerdict,
    HistorySnapshot,
    ImageArtifact,
    InstructionCandidate,
    InstructionSelection,
    PublicMessage,
    QuestionFit,
    QuestionIntent,
    RubricContext,
    RubricItem,
    RubricVerdict,
    TextPayload,
    TurnArtifact,
    TurnRating,
    build_history_snapshot,
)
from pixelogue.decision import DecisionClient
from pixelogue.drafting import (
    QuestionDraft,
    QuestionDraftBatch,
    draft_task_contract,
    draft_to_candidate,
)
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import (
    action_affordance_question,
    action_answer_already_public,
    aggregate_rating,
    applicable_rubric_items,
    consensus,
    has_natural_language_content,
    identification_answer_in_history,
    identification_answer_in_question,
    identification_label_in_question,
    item_id,
    question_fingerprint,
    question_fit_consensus,
    reciprocal_identification_disclosure,
    repeated_answered_request,
    repeated_public_question,
    scene_options_in_question,
    transcription_answer_in_question,
    unverified_transcription_relation,
)
from pixelogue.fact_identity import requested_fact_key
from pixelogue.focused_views import focused_view
from pixelogue.gates import (
    GateDecision,
    QuestionGateVote,
    holistic_decision,
    question_gate_decision,
)
from pixelogue.jev_routing import (
    SUPPORTED_BINDING_TASKS,
    BindingProposals,
    EvidenceProposals,
    StoredDecisionRouter,
    assemble_bindings,
    assemble_evidence,
    binding_decisions,
    controller_binding_proposals,
    evidence_decisions,
    partition_binding_proposals,
    validate_evidence_proposals,
    validate_usable_binding_proposals,
)
from pixelogue.ledger import (
    Requirement,
    RequirementInventory,
    _normalize_spec,
    reconcile_inventories,
)
from pixelogue.planner import allocated_role, instruction_candidates, planned_turn_count
from pixelogue.prompts import is_private_prompt_echo
from pixelogue.routing import FamilyLedger, ImageProfile, Route, choose_route, family_definitions
from pixelogue.rules import (
    ComputationCheck,
    ComputationInventory,
    SetCheck,
    SetInventory,
    verify_computation_inventories,
    verify_set_inventories,
)
from pixelogue.serialization import canonical_hash, canonical_json
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_evidence import (
    ArrayScopedEvidenceReport,
    AttributeRecheckReport,
    CandidateBindingReport,
    CandidateBindingsReport,
    CapabilityObservation,
    CompactScopedEvidenceReport,
    ImageRegion,
    ScopedEvidenceInventory,
    ScopedEvidenceReport,
    ScopeEvidence,
    alias_evidence_ids,
    bind_evidence_identity,
    out_of_scope_region_feedback,
    unsupported_object_label_feedback,
)
from pixelogue.task_runtime import (
    attribute_fact_key,
    bind_candidates_individually,
    binding_candidate,
    operation_contract,
    question_operation_contract,
    selector_candidate,
    unavailable_reasons,
    validate_evidence,
)
from pixelogue.task_verification import verify_operation

OutputModel = TypeVar("OutputModel", bound=BaseModel)
type PublicTextRejectionReason = Literal[
    "PRIVATE_PROMPT_ECHO",
    "INTERNAL_REFERENCE_IN_QUESTION",
    "REPEATED_PUBLIC_QUESTION",
    "REPEATED_ANSWERED_REQUEST",
    "IDENTIFICATION_TARGET_IN_QUESTION",
    "IDENTIFICATION_ANSWER_IN_QUESTION",
    "IDENTIFICATION_ANSWER_ALREADY_PUBLIC",
    "UI_LOCATION_REPEATS_TARGET",
    "RECIPROCAL_IDENTIFICATION_ALREADY_PUBLIC",
    "TRANSCRIPTION_ANSWER_IN_QUESTION",
    "TEXT_RELATION_UNVERIFIED",
    "ACTION_ALREADY_PUBLIC",
    "ACTION_AFFORDANCE_NOT_VISIBLE",
    "CATEGORY_OPTIONS_NOT_PUBLIC",
    "REPEATED_REJECTED_QUESTION",
    "REQUEST_KEY_ALREADY_COMMITTED",
]
# Malformed or incomplete model output that may abstain locally instead of ending the image.
MODEL_OUTPUT_FAILURES = frozenset(
    {
        "MODEL_CONTENT_EMPTY",
        "MODEL_FINISH_REASON",
        "MODEL_WHITESPACE_RUNAWAY",
        "MODEL_OUTPUT_REPETITION",
        "MODEL_SCHEMA_MISMATCH",
    }
)
_INTERNAL_QUESTION_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:scope|view|candidate|evidence|obs)_[A-Za-z0-9_]+"
    r"|selected (?:image )?(?:region|scope|scene)"
    r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DIRECT_IDENTIFICATION_COLOR = re.compile(
    r"^\s*(?:what|which)\s+is\s+(?:this|that)\s+"
    r"(?:red|orange|yellow|green|blue|purple|pink|brown|black|white|gray|grey|"
    r"golden|silver|beige|tan)\s+[\w-]+\b",
    re.IGNORECASE,
)


def internal_reference_in_question(text: str) -> bool:
    """Detect controller references that cannot identify a visible subject."""
    return _INTERNAL_QUESTION_REFERENCE.search(text) is not None


MODEL_OUTPUT_ABSTENTIONS = frozenset(
    {
        "MODEL_CONTENT_EMPTY",
        "MODEL_FINISH_REASON",
        "MODEL_WHITESPACE_RUNAWAY",
        "MODEL_OUTPUT_REPETITION",
        "MODEL_SCHEMA_MISMATCH",
        "EVIDENCE_IMAGE_MISMATCH",
        "EVIDENCE_CAPABILITY_UNKNOWN",
        "EVIDENCE_SCOPE_LIMIT",
        "EVIDENCE_VIEW_MISMATCH",
        "EVIDENCE_OBSERVATION_LIMIT",
        "EVIDENCE_ATTRIBUTE_SCOPE",
        "EXTRACTION_SOURCE_INVALID",
    }
)

_GENERATED_BINDING_OUTPUT_FAILURES = frozenset(
    {
        "MODEL_CONTENT_EMPTY",
        "MODEL_FINISH_REASON",
        "MODEL_WHITESPACE_RUNAWAY",
        "MODEL_OUTPUT_REPETITION",
        "MODEL_SCHEMA_MISMATCH",
        "CANDIDATE_BINDING_ID",
        "CANDIDATE_EVIDENCE_SCOPE",
        "CANDIDATE_CHECKS_MISMATCH",
        "CANDIDATE_PARAMETER_UNKNOWN",
        "CANDIDATE_PARAMETER_MISSING",
        "CANDIDATE_PARAMETER_SOURCE",
        "CANDIDATE_PARAMETER_VALUE",
    }
)


@dataclass(frozen=True)
class SynthesisJob:
    """One deterministically scheduled image synthesis job."""

    image: ImageArtifact
    target_language: Literal["en", "ja", "zh-Hans"]
    generator_role: Literal["generator_a", "generator_b"]


class TerminalState(NamedTuple):
    """Where and why a conversation's turn loop stopped."""

    status: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"]
    stage: str
    reason: str
    turn_index: int


@dataclass(frozen=True)
class TurnAttemptResult:
    """Private outcome before a turn is appended or committed by its coordinator."""

    turn: TurnArtifact | None
    status: Literal["COMMITTED", "REJECTED", "ABSTAINED", "ERROR"]
    stage: str
    reason: str
    question_fingerprint: str | None = None


class InferenceClient(Protocol):
    """Interface implemented by vLLM and scripted contract-test clients."""

    endpoint: ModelEndpoint

    def invoke(
        self,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        response_model: type[OutputModel],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        bypass_cache: bool = False,
        retry_feedback: str | None = None,
        trial_id: str | None = None,
    ) -> ModelResponse:
        """Return one schema-validated response."""


class SynthesisCoordinator:
    """Coordinate one image conversation without allowing partial commits."""

    def __init__(
        self,
        config: PixelogueConfig,
        run_id: str,
        store: RunStore,
        selector: InferenceClient,
        generator_a: InferenceClient,
        generator_b: InferenceClient,
        *,
        decision_client: DecisionClient | None = None,
    ) -> None:
        """Bind an immutable run configuration and its explicit model clients."""
        self.config = config
        self.run_id = run_id
        self.store = store
        self.selector = selector
        self.generators = {"generator_a": generator_a, "generator_b": generator_b}
        routing = config.tasks.decision_routing
        if (routing.evidence_enabled or routing.binding_enabled) and decision_client is None:
            raise ExecutionError(
                "DECISION_CLIENT_REQUIRED",
                "Enabled decision routing needs its explicit classifier client",
            )
        self.decision_router = (
            StoredDecisionRouter(decision_client, routing, config.runtime, store)
            if decision_client is not None
            else None
        )
        # Direct drafting offers only normal-profile core operations with working validators.
        self._direct_tasks = {
            task.id: task
            for task in task_catalog().tasks
            if config.tasks.planner == "direct"
            and task.status == "core_candidate"
            and not unavailable_reasons(task, config.tasks, config.models)
        }
        self.family_ledger = FamilyLedger(
            store.committed_task_ids() if config.tasks.planner == "direct" else ()
        )

    def synthesize_batch(
        self,
        jobs: Iterable[SynthesisJob],
        artifact_root: Path,
        *,
        max_workers: int,
    ) -> Iterator[ConversationArtifact]:
        """Synthesize independent images concurrently and yield input order.

        The default submits at most ``max_workers`` jobs. Optional completion-based refill keeps
        at most twice that many submitted or buffered results, with at most ``max_workers``
        active jobs. Conversation-local history remains sequential in either mode.

        Raises:
            ValueError: If ``max_workers`` is outside the configured image concurrency range.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if max_workers > self.config.runtime.max_concurrent_images:
            raise ValueError("max_workers exceeds runtime.max_concurrent_images")
        iterator = iter(jobs)
        if max_workers == 1:
            for job in iterator:
                yield self._synthesize_job(job, artifact_root)
            return
        if self.config.runtime.refill_completed_images:
            yield from self._synthesize_with_refill(iterator, artifact_root, max_workers)
            return

        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="pixelogue-image",
        ) as executor:
            pending: deque[Future[ConversationArtifact]] = deque()
            for _ in range(max_workers):
                try:
                    job = next(iterator)
                except StopIteration:
                    break
                pending.append(executor.submit(self._synthesize_job, job, artifact_root))

            while pending:
                yield pending.popleft().result()
                try:
                    job = next(iterator)
                except StopIteration:
                    continue
                pending.append(executor.submit(self._synthesize_job, job, artifact_root))

    def _synthesize_with_refill(
        self,
        jobs: Iterator[SynthesisJob],
        artifact_root: Path,
        max_workers: int,
    ) -> Iterator[ConversationArtifact]:
        """Refill finished slots within a bounded window, then return input order."""
        window = 2 * max_workers
        next_submit = next_return = 0
        exhausted = False
        ready: dict[int, Future[ConversationArtifact]] = {}
        with ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pixelogue-image"
        ) as executor:
            pending: dict[Future[ConversationArtifact], int] = {}
            while pending or ready or not exhausted:
                while (
                    not exhausted
                    and len(pending) < max_workers
                    and len(pending) + len(ready) < window
                ):
                    try:
                        job = next(jobs)
                    except StopIteration:
                        exhausted = True
                        break
                    pending[executor.submit(self._synthesize_job, job, artifact_root)] = next_submit
                    next_submit += 1
                if pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in completed:
                        ready[pending.pop(future)] = future
                while next_return in ready:
                    yield ready.pop(next_return).result()
                    next_return += 1

    def _synthesize_job(
        self,
        job: SynthesisJob,
        artifact_root: Path,
    ) -> ConversationArtifact:
        """Run one scheduled job and preserve image-scoped execution failures."""
        try:
            return self.synthesize_image(
                job.image,
                artifact_root,
                target_language=job.target_language,
                generator_role=job.generator_role,
            )
        except ExecutionError as error:
            return self.record_failed_image(
                job.image,
                target_language=job.target_language,
                generator_role=job.generator_role,
                error=error,
            )

    def synthesize_image(
        self,
        image: ImageArtifact,
        artifact_root: Path,
        *,
        target_language: Literal["en", "ja", "zh-Hans"] = "en",
        generator_role: Literal["generator_a", "generator_b"] | None = None,
    ) -> ConversationArtifact:
        """Synthesize and verify one complete conversation.

        Evaluation-only images can exercise the pilot path but remain ineligible for training export.
        Holistic review may retain an accepted prefix of at least two turns after a quality stop.
        """
        conversation_id = canonical_hash(
            {"run": self.run_id, "image": image.image_id, "language": target_language}
        )
        assigned_role = generator_role or allocated_role(
            conversation_id,
            {
                str(role): weight
                for role, weight in self.config.models.generation_allocation.items()
            },
        )
        generator = self.generators[assigned_role]
        model_image = self._model_image(image, artifact_root)
        image_views = [self._view_metadata(image)]
        if self.config.evaluation.mode == "holistic":
            with self.store.transaction() as connection:
                saved = connection.execute(
                    "SELECT artifact_hash FROM conversation_commit WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
            if saved is not None:
                completed = ConversationArtifact.model_validate_json(
                    self.store.read_artifact(saved[0])
                )
                if (
                    completed.image != image
                    or completed.generation_model != generator.endpoint.repo_id
                ):
                    raise ExecutionError(
                        "RESUME_TURN_MISMATCH",
                        "Saved conversation conflicts with the current input",
                    )
                return completed
        count = planned_turn_count(image.image_id, self.config.seed)
        turns = [
            TurnArtifact.model_validate_json(self.store.read_artifact(artifact_hash))
            for artifact_hash in self.store.committed_artifact_hashes(conversation_id)
        ]
        if len(turns) > count or any(
            turn.status != "COMMITTED" or turn.generation_model != generator.endpoint.repo_id
            for turn in turns
        ):
            raise ExecutionError(
                "RESUME_TURN_MISMATCH",
                "Saved turn prefix conflicts with the current generation plan",
            )
        transcript = tuple(message for turn in turns for message in (turn.question, turn.answer))
        if len(turns) == count:
            conversation = ConversationArtifact(
                conversation_id=conversation_id,
                image=image,
                target_language=target_language,
                generation_model=generator.endpoint.repo_id,
                turns=tuple(turns),
                status="QUALITY_CANDIDATE",
            )
            return self._finish_conversation(conversation, persist=True)
        run_turns = (
            self._direct_turns if self.config.tasks.planner == "direct" else self._scoped_turns
        )
        terminal_status, terminal_stage, terminal_reason, terminal_turn_index = run_turns(
            conversation_id,
            image,
            generator,
            model_image,
            image_views,
            target_language,
            count,
            turns,
            transcript,
        )

        if len(turns) != count or any(turn.status != "COMMITTED" for turn in turns):
            if terminal_status == "QUALITY_CANDIDATE":
                terminal_status = "REJECTED"
        if terminal_status != "QUALITY_CANDIDATE":
            self.store.write_json_artifact(
                "conversation-stop-reasons",
                {
                    "conversation_id": conversation_id,
                    "turn_index": terminal_turn_index,
                    "status": terminal_status,
                    "stage": terminal_stage,
                    "reason": terminal_reason,
                },
            )
        conversation = ConversationArtifact(
            conversation_id=conversation_id,
            image=image,
            target_language=target_language,
            generation_model=generator.endpoint.repo_id,
            turns=tuple(turns),
            status=terminal_status,
        )
        return self._finish_conversation(conversation, persist=True)

    def _scoped_turns(
        self,
        conversation_id: str,
        image: ImageArtifact,
        generator: InferenceClient,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        target_language: Literal["en", "ja", "zh-Hans"],
        count: int,
        turns: list[TurnArtifact],
        transcript: tuple[PublicMessage, ...],
    ) -> TerminalState:
        """Extract scoped evidence once, then bind, select and attempt each planned turn.

        Committed turns are appended to ``turns`` in place.
        """
        if self.config.tasks.decision_routing.evidence_enabled:
            inventory = self._extract_decision_evidence(generator, image, model_image, image_views)
        else:
            inventory = self._extract_generator_evidence(generator, image, model_image, image_views)
        inventory = self._maybe_recheck_attribute(inventory, generator, model_image, image_views)
        inventory, evidence_aliases = alias_evidence_ids(inventory)
        if inventory.image_id != image.image_id:
            raise ExecutionError("EVIDENCE_IMAGE_MISMATCH", "Evidence refers to another image")
        self.store.write_json_artifact(
            "evidence-id-aliases",
            {"image_id": image.image_id, "original_to_local": evidence_aliases},
        )

        terminal_status: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"] = (
            "QUALITY_CANDIDATE"
        )
        terminal_stage = "conversation_length"
        terminal_reason = "REQUESTED_TURN_COUNT_INCOMPLETE"
        terminal_turn_index = min(count, len(turns) + 1)
        for turn_index in range(len(turns) + 1, count + 1):
            terminal_turn_index = turn_index
            snapshot = build_history_snapshot(conversation_id, turn_index, transcript)
            candidates = instruction_candidates(
                inventory,
                seed=self.config.seed,
                turn_index=turn_index,
                limit=self.config.tasks.candidate_limit,
                settings=self.config.tasks,
                models=self.config.models,
                used_task_ids=frozenset(turn.instruction.task_id for turn in turns),
            )
            if candidates:
                try:
                    candidates = self._bind_candidate_batches(
                        candidates,
                        inventory,
                        snapshot,
                        generator,
                        target_language,
                        model_image,
                        image_views,
                        tuple(turns),
                    )
                except ExecutionError as error:
                    if error.reason not in {
                        "MODEL_CONTENT_EMPTY",
                        "MODEL_FINISH_REASON",
                        "MODEL_WHITESPACE_RUNAWAY",
                        "MODEL_OUTPUT_REPETITION",
                        "MODEL_SCHEMA_MISMATCH",
                        "CANDIDATE_BINDING_ID",
                        "CANDIDATE_EVIDENCE_SCOPE",
                        "CANDIDATE_CHECKS_MISMATCH",
                        "CANDIDATE_PARAMETER_UNKNOWN",
                        "CANDIDATE_PARAMETER_MISSING",
                        "CANDIDATE_PARAMETER_SOURCE",
                        "CANDIDATE_PARAMETER_VALUE",
                    }:
                        raise
                    self.store.write_json_artifact(
                        "binding-abstentions",
                        {
                            "conversation_id": conversation_id,
                            "turn_index": turn_index,
                            "reason": error.reason,
                            "message": str(error),
                        },
                    )
                    terminal_status = "ABSTAINED"
                    terminal_stage = "candidate_binding"
                    terminal_reason = error.reason
                    break
            if not candidates:
                terminal_status = "REJECTED"
                terminal_stage = "candidate_admission"
                terminal_reason = "NO_ADMISSIBLE_CANDIDATE"
                break
            selected = self._select_instruction(
                candidates,
                snapshot,
                target_language,
                image_views,
                model_image,
                turn_index,
            )
            if selected is None:
                terminal_status = "REJECTED"
                terminal_stage = "instruction_selection"
                terminal_reason = "NO_SUPPORTED_NEW_INSTRUCTION"
                break
            priority = (selected,) + tuple(
                candidate
                for candidate in candidates
                if candidate.candidate_id != selected.candidate_id
            )[: self.config.tasks.max_candidate_attempts - 1]
            outcome = self._attempt_candidates(
                priority,
                snapshot,
                generator,
                target_language,
                model_image,
                image_views,
                tuple(turns),
            )
            turn = outcome.turn
            if turn is None:
                assert outcome.status != "COMMITTED"
                terminal_status = outcome.status
                terminal_stage = outcome.stage
                terminal_reason = outcome.reason
                break
            turn_status = outcome.status
            turns.append(turn)
            if turn_status != "COMMITTED":
                terminal_status = turn_status
                terminal_stage = "answer_verification"
                terminal_reason = outcome.reason
                break
            transcript = (*transcript, turn.question, turn.answer)
            artifact_hash = self.store.write_json_artifact("turns", turn.model_dump(mode="json"))
            with self.store.transaction() as connection:
                connection.execute(
                    """INSERT INTO turn_commit(
                           conversation_id, branch_id, turn_index, attempt, artifact_hash
                       ) VALUES (?, 'main', ?, 0, ?)""",
                    (conversation_id, turn_index, artifact_hash),
                )
        return TerminalState(terminal_status, terminal_stage, terminal_reason, terminal_turn_index)

    def _direct_turns(
        self,
        conversation_id: str,
        image: ImageArtifact,
        generator: InferenceClient,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        target_language: Literal["en", "ja", "zh-Hans"],
        count: int,
        turns: list[TurnArtifact],
        transcript: tuple[PublicMessage, ...],
    ) -> TerminalState:
        """Profile the image once, then draft, gate, answer and review each planned turn.

        Committed turns are appended to ``turns`` in place and counted in the family ledger.
        """
        profile = self._profile_image(image, model_image, image_views)
        terminal_status: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"] = (
            "QUALITY_CANDIDATE"
        )
        terminal_stage = "conversation_length"
        terminal_reason = "REQUESTED_TURN_COUNT_INCOMPLETE"
        terminal_turn_index = min(count, len(turns) + 1)
        for turn_index in range(len(turns) + 1, count + 1):
            terminal_turn_index = turn_index
            snapshot = build_history_snapshot(conversation_id, turn_index, transcript)
            outcome = self._direct_turn(
                image,
                profile,
                snapshot,
                generator,
                target_language,
                model_image,
                image_views,
                tuple(turns),
            )
            turn = outcome.turn
            if turn is None:
                assert outcome.status != "COMMITTED"
                terminal_status = outcome.status
                terminal_stage = outcome.stage
                terminal_reason = outcome.reason
                break
            turns.append(turn)
            if outcome.status != "COMMITTED":
                terminal_status = outcome.status
                terminal_stage = "answer_verification"
                terminal_reason = outcome.reason
                break
            transcript = (*transcript, turn.question, turn.answer)
            self._commit_turn(conversation_id, turn_index, turn)
            self.family_ledger.record(turn.instruction.task_id)
        return TerminalState(terminal_status, terminal_stage, terminal_reason, terminal_turn_index)

    def _commit_turn(self, conversation_id: str, turn_index: int, turn: TurnArtifact) -> None:
        artifact_hash = self.store.write_json_artifact("turns", turn.model_dump(mode="json"))
        with self.store.transaction() as connection:
            connection.execute(
                """INSERT INTO turn_commit(
                       conversation_id, branch_id, turn_index, attempt, artifact_hash
                   ) VALUES (?, 'main', ?, 0, ?)""",
                (conversation_id, turn_index, artifact_hash),
            )

    def _profile_image(
        self,
        image: ImageArtifact,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> ImageProfile | None:
        """Ask the router model for a coarse profile; malformed output falls back to defaults."""
        try:
            return self._invoke(
                self.selector,
                "image_profile",
                {
                    "image_views": image_views,
                    "family_definitions": family_definitions(self._direct_tasks),
                },
                (model_image,),
                ImageProfile,
                max_tokens=self.config.tasks.profile_max_tokens,
                temperature=0.0,
                seed=self.config.seed,
            )
        except ExecutionError as error:
            if error.reason not in MODEL_OUTPUT_FAILURES:
                raise
            self.store.write_json_artifact(
                "image-profile-abstentions",
                {"image_id": image.image_id, "reason": error.reason, "message": str(error)},
            )
            return None

    def _turn_route(
        self,
        image: ImageArtifact,
        profile: ImageProfile | None,
        snapshot: HistorySnapshot,
        previous_turns: tuple[TurnArtifact, ...],
        call_index: int,
        excluded_families: frozenset[str] = frozenset(),
    ) -> Route | None:
        """Choose or restore the families offered to one drafting call."""
        saved = self.store.saved_turn_route(
            snapshot.conversation_id, snapshot.turn_index, snapshot.history_hash, call_index
        )
        if saved is None:
            committed = [turn for turn in previous_turns if turn.status == "COMMITTED"]
            route = choose_route(
                profile,
                self.family_ledger,
                tasks=self._direct_tasks,
                family_targets=self.config.tasks.family_targets,
                task_weights=self.config.tasks.task_weights,
                used_families=frozenset(turn.instruction.family for turn in committed)
                | excluded_families,
                used_task_ids=frozenset(turn.instruction.task_id for turn in committed),
                seed=self.config.seed,
                image_id=image.image_id,
                turn_index=snapshot.turn_index,
            )
            saved = self.store.save_turn_route(
                snapshot.conversation_id,
                snapshot.turn_index,
                snapshot.history_hash,
                call_index,
                canonical_json(route.to_json() if route is not None else None).decode(),
            )
        value = json.loads(saved)
        return Route.from_json(value) if value is not None else None

    def _direct_turn(
        self,
        image: ImageArtifact,
        profile: ImageProfile | None,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
    ) -> TurnAttemptResult:
        """Gate drafts in order and answer the first admitted question.

        Rejected drafts stay private and never reach a later drafting call. At most one
        question per turn is answered, so a failed answer is never replaced by an easier one.
        """
        stop = TurnAttemptResult(None, "REJECTED", "question_draft", "NO_DRAFT_ADMITTED")
        rejected: set[str] = set()
        offered: frozenset[str] = frozenset()
        for call_index in range(1 + self.config.tasks.extra_draft_calls_per_turn):
            route = self._turn_route(image, profile, snapshot, previous_turns, call_index, offered)
            if route is None:
                return TurnAttemptResult(None, "REJECTED", "family_routing", "NO_AVAILABLE_FAMILY")
            offered |= {route.primary_family} | (
                {route.secondary_family} if route.secondary_family else set()
            )
            drafts = self._draft_questions(
                route,
                snapshot,
                generator,
                target_language,
                model_image,
                image_views,
                previous_turns,
                call_index,
            )
            for draft_index, draft in enumerate(drafts):
                admitted = self._admit_draft(
                    draft,
                    image,
                    snapshot,
                    route,
                    previous_turns,
                    call_index,
                    draft_index,
                    frozenset(rejected),
                )
                if admitted is None:
                    continue
                candidate, question = admitted
                decision, candidate = self._direct_question_gate(
                    candidate,
                    draft,
                    image,
                    snapshot,
                    question,
                    target_language,
                    model_image,
                    image_views,
                )
                if decision.verdict is GateVerdict.ERROR:
                    raise ExecutionError("QUESTION_GATE_ERROR", "Question gate execution failed")
                if decision.verdict is not GateVerdict.MET:
                    rejected.add(question_fingerprint(question.content))
                    stop = TurnAttemptResult(
                        None,
                        "ABSTAINED" if decision.verdict is GateVerdict.UNKNOWN else "REJECTED",
                        "question_gate",
                        f"QUESTION_GATE_{decision.verdict.value}",
                        question_fingerprint(question.content),
                    )
                    continue
                answered = self._generate_answer(
                    candidate,
                    snapshot,
                    generator,
                    question,
                    (),
                    target_language,
                    model_image,
                    image_views,
                    previous_turns,
                )
                if isinstance(answered, TurnAttemptResult):
                    return answered
                return self._review_answer(
                    candidate,
                    snapshot,
                    generator,
                    question,
                    answered,
                    (),
                    target_language,
                    model_image,
                    image_views,
                    previous_turns,
                )
        return stop

    def _draft_questions(
        self,
        route: Route,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
        call_index: int,
    ) -> tuple[QuestionDraft, ...]:
        """Request ordered drafts for the routed operations; malformed batches yield none."""
        excluded = [
            {"subject": subject, "dimension": dimension}
            for turn in previous_turns
            if turn.status == "COMMITTED" and turn.instruction.request_key
            for subject, _, dimension in [turn.instruction.request_key.partition("|")]
        ]
        offered = frozenset(route.task_ids)

        def validate_batch(batch: QuestionDraftBatch) -> None:
            reasons = []
            for index, draft in enumerate(batch.drafts):
                try:
                    draft_to_candidate(
                        draft,
                        image_id="validation",
                        view_id=model_image.view_id,
                        turn_index=snapshot.turn_index,
                        draft_index=index,
                        allowed_task_ids=offered,
                    )
                except ExecutionError as error:
                    reasons.append(error.reason)
                else:
                    return
            if reasons:
                raise ExecutionError("DRAFT_CONTRACT_INVALID", ", ".join(sorted(set(reasons))))

        try:
            batch = self._invoke(
                generator,
                "question_draft",
                {
                    "target_language": target_language,
                    "turn_index": snapshot.turn_index,
                    "public_history": self._history(snapshot.public_history),
                    "allowed_tasks": [
                        draft_task_contract(self._direct_tasks[task_id])
                        for task_id in route.task_ids
                    ],
                    "preferred_task_ids": list(route.task_ids),
                    "family_plan": {
                        "primary": route.primary_family,
                        "secondary": route.secondary_family,
                    },
                    "excluded_fact_keys": excluded,
                    "draft_count": self.config.tasks.draft_count,
                    "image_views": image_views,
                },
                (model_image,),
                QuestionDraftBatch,
                max_tokens=self.config.tasks.draft_max_tokens,
                temperature=0.7,
                seed=self.config.seed + snapshot.turn_index + 1_000 * call_index,
                post_validate=validate_batch,
            )
        except ExecutionError as error:
            if error.reason not in MODEL_OUTPUT_FAILURES | {"DRAFT_CONTRACT_INVALID"}:
                raise
            self.store.write_json_artifact(
                "draft-abstentions",
                {
                    "conversation_id": snapshot.conversation_id,
                    "turn_index": snapshot.turn_index,
                    "call_index": call_index,
                    "reason": error.reason,
                    "message": str(error),
                },
            )
            return ()
        self.store.write_json_artifact(
            "question-drafts",
            {
                "conversation_id": snapshot.conversation_id,
                "turn_index": snapshot.turn_index,
                "call_index": call_index,
                "route": route.to_json(),
                "batch": batch.model_dump(mode="json"),
            },
        )
        return batch.drafts

    def _admit_draft(
        self,
        draft: QuestionDraft,
        image: ImageArtifact,
        snapshot: HistorySnapshot,
        route: Route,
        previous_turns: tuple[TurnArtifact, ...],
        call_index: int,
        draft_index: int,
        rejected: frozenset[str],
    ) -> tuple[InstructionCandidate, PublicMessage] | None:
        """Convert one draft and apply the deterministic public-text checks before judging."""
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index
        try:
            candidate = draft_to_candidate(
                draft,
                image_id=image.image_id,
                view_id=image.full_view.view_id,
                turn_index=turn_index,
                draft_index=call_index * 10 + draft_index,
                allowed_task_ids=frozenset(route.task_ids),
            )
        except ExecutionError as error:
            self.store.write_json_artifact(
                "draft-rejections",
                {
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "call_index": call_index,
                    "draft_index": draft_index,
                    "reason": error.reason,
                    "message": str(error),
                },
            )
            return None
        text = draft.question
        committed_keys = {
            turn.instruction.request_key
            for turn in previous_turns
            if turn.status == "COMMITTED" and turn.instruction.request_key
        }
        reason: PublicTextRejectionReason | None = None
        if question_fingerprint(text) in rejected:
            reason = "REPEATED_REJECTED_QUESTION"
        elif is_private_prompt_echo(text):
            reason = "PRIVATE_PROMPT_ECHO"
        elif internal_reference_in_question(text):
            reason = "INTERNAL_REFERENCE_IN_QUESTION"
        elif repeated_public_question(text, snapshot.public_history):
            reason = "REPEATED_PUBLIC_QUESTION"
        elif candidate.request_key in committed_keys:
            reason = "REQUEST_KEY_ALREADY_COMMITTED"
        elif candidate.task_id == "visible_action_relation" and action_affordance_question(text):
            reason = "ACTION_AFFORDANCE_NOT_VISIBLE"
        elif candidate.task_id == "text_transcription" and unverified_transcription_relation(text):
            reason = "TEXT_RELATION_UNVERIFIED"
        elif candidate.task_id == "scene_categorization":
            options = next(
                (p.value for p in candidate.public_parameters if p.name == "category_set"), None
            )
            if not isinstance(options, tuple) or not scene_options_in_question(text, options):
                reason = "CATEGORY_OPTIONS_NOT_PUBLIC"
        if reason is not None:
            self._record_public_text_rejection(
                conversation_id, turn_index, field="question", reason=reason, content=text
            )
            return None
        question = PublicMessage(
            message_id=f"{conversation_id}:q:{turn_index}",
            turn_index=turn_index,
            role="user",
            content=text,
        )
        return candidate, question

    def _direct_question_gate(
        self,
        candidate: InstructionCandidate,
        draft: QuestionDraft,
        image: ImageArtifact,
        snapshot: HistorySnapshot,
        question: PublicMessage,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> tuple[GateDecision, InstructionCandidate]:
        """Run both blind question judges once and apply the configured label policy."""
        judged = self._judge_inputs(candidate, image_views, model_image)
        if judged is None:
            decision = GateDecision(
                GateVerdict.UNKNOWN, candidate.task_id, "label_unresolved", GateVerdict.UNKNOWN
            )
            votes: list[QuestionGateVote] = []
        else:
            images, views = judged
            catalog = task_catalog()
            votes = self._invoke_judges(
                "question_gate",
                {
                    "target_language": target_language,
                    "public_history": self._history(snapshot.public_history),
                    "selected_instruction": operation_contract(candidate),
                    "task_definitions": [
                        {"task_id": task.id, "definition": task.definition_en}
                        for task in catalog.tasks
                    ],
                    "question": question.content,
                    "image_views": views,
                },
                images,
                QuestionGateVote,
                max_tokens=320,
                seed=self.config.seed + snapshot.turn_index,
            )

            def relabel_allowed(task_id: str) -> bool:
                try:
                    draft_to_candidate(
                        draft,
                        image_id=image.image_id,
                        view_id=image.full_view.view_id,
                        turn_index=snapshot.turn_index,
                        draft_index=0,
                        allowed_task_ids=frozenset(self._direct_tasks),
                        task_id=task_id,
                    )
                except ExecutionError:
                    return False
                return True

            decision = question_gate_decision(
                votes,
                candidate.task_id,
                policy=self.config.evaluation.question_gate_label_policy,
                contracts={task.id: task.verification_contracts for task in catalog.tasks},
                relabel_allowed=relabel_allowed,
            )
        if decision.verdict is GateVerdict.MET and decision.task_id != candidate.task_id:
            relabeled = draft_to_candidate(
                draft,
                image_id=image.image_id,
                view_id=image.full_view.view_id,
                turn_index=snapshot.turn_index,
                draft_index=int(str(candidate.scope_id).rsplit(":", 1)[-1]),
                allowed_task_ids=frozenset(self._direct_tasks),
                task_id=decision.task_id,
            )
            candidate = relabeled
        self.store.write_json_artifact(
            "question-gate-decisions",
            {
                "question_message_id": question.message_id,
                "candidate_id": candidate.candidate_id,
                "drafted_task_id": draft.task_id,
                "votes": [vote.model_dump(mode="json") for vote in votes],
                "label_rule": decision.label_rule,
                "fit": decision.fit.value,
                "decided_task_id": decision.task_id,
                "verdict": decision.verdict.value,
            },
        )
        return decision, candidate

    def _judge_inputs(
        self,
        instruction: InstructionCandidate,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
    ) -> tuple[tuple[ModelImage, ...], list[dict[str, str]]] | None:
        """Choose the pixels local judges receive under the configured view policy."""
        focused = self._local_verification_view(instruction, image_views, model_image)
        if self.config.evaluation.judge_views == "crop":
            return None if focused is None else ((focused[0],), focused[1])
        if focused is None or focused[0] is model_image:
            return (model_image,), image_views
        return (model_image, focused[0]), [*image_views, *focused[1]]

    def _extract_generator_evidence(
        self,
        generator: InferenceClient,
        image: ImageArtifact,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> ScopedEvidenceInventory:
        """Keep the existing extraction and validation when decision routing is disabled."""
        evidence_report = self._invoke(
            generator,
            "evidence_extraction",
            {
                "image_id": image.image_id,
                "capability_vocabulary": self._capability_vocabulary(),
                "max_scopes": self.config.tasks.max_scopes,
                "max_observations_per_scope": self.config.tasks.max_observations_per_scope,
                "image_views": image_views,
            },
            (model_image,),
            {
                "keyed": ScopedEvidenceReport,
                "array": ArrayScopedEvidenceReport,
                "compact": CompactScopedEvidenceReport,
            }[self.config.tasks.evidence_format],
            max_tokens=self.config.tasks.evidence_max_tokens,
            temperature=0.0,
            seed=self.config.seed,
            post_validate=lambda result: validate_evidence(
                bind_evidence_identity(result, image.image_id, model_image.view_id),
                model_image.view_id,
                self.config.tasks,
            ),
        )
        return bind_evidence_identity(evidence_report, image.image_id, model_image.view_id)

    def _extract_decision_evidence(
        self,
        generator: InferenceClient,
        image: ImageArtifact,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> ScopedEvidenceInventory:
        """Generate only visual content and independently check each local capability claim."""
        if self.decision_router is None:
            raise ExecutionError("DECISION_CLIENT_REQUIRED", "Classifier client is absent")
        proposals = self._invoke(
            generator,
            "evidence_proposal",
            {
                "capability_vocabulary": self._capability_vocabulary(),
                "max_scopes": self.config.tasks.max_scopes,
                "max_observations_per_scope": min(8, self.config.tasks.max_observations_per_scope),
                "image_views": image_views,
            },
            (model_image,),
            EvidenceProposals,
            max_tokens=self.config.tasks.decision_routing.proposal_max_tokens,
            temperature=0.0,
            seed=self.config.seed,
            post_validate=lambda report: validate_evidence_proposals(report, self.config.tasks),
        )
        self.store.write_json_artifact(
            "routing-evidence-proposals",
            {
                "image_id": image.image_id,
                "proposals": proposals.model_dump(mode="json"),
            },
        )
        verdicts = {
            query.query_id: self.decision_router.decide(
                query,
                image_id=image.image_id,
                image=model_image,
                trial_id=f"{self.run_id}/{image.image_id}/evidence",
            )
            for query in evidence_decisions(proposals, self.config.tasks)
        }
        inventory = assemble_evidence(
            proposals,
            verdicts,
            image_id=image.image_id,
            view_id=model_image.view_id,
            settings=self.config.tasks,
        )
        validate_evidence(inventory, model_image.view_id, self.config.tasks)
        self.store.write_json_artifact(
            "routing-evidence-inventory", inventory.model_dump(mode="json")
        )
        return inventory

    def _bind_candidate_batches(
        self,
        candidates: tuple[InstructionCandidate, ...],
        inventory: ScopedEvidenceInventory,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
    ) -> tuple[InstructionCandidate, ...]:
        """Bind a fixed prefix and extend only when no admissible new fact remains."""
        size = self.config.tasks.initial_binding_batch_size
        used = frozenset(turn.instruction.candidate_id for turn in previous_turns)
        for offset in range(0, len(candidates), size):
            batch = candidates[offset : offset + size]
            admitted = self._bind_candidate_batch(
                batch,
                inventory,
                snapshot,
                generator,
                target_language,
                model_image,
                image_views,
                used,
            )
            admitted = self._drop_answered_candidates(
                admitted, previous_turns, snapshot.conversation_id, snapshot.turn_index
            )
            self.store.write_json_artifact(
                "candidate-binding-batches",
                {
                    "conversation_id": snapshot.conversation_id,
                    "turn_index": snapshot.turn_index,
                    "offset": offset,
                    "candidate_ids": [candidate.candidate_id for candidate in batch],
                    "admitted_ids": [candidate.candidate_id for candidate in admitted],
                },
            )
            if admitted:
                return admitted
        return ()

    def _bind_candidate_batch(
        self,
        candidates: tuple[InstructionCandidate, ...],
        inventory: ScopedEvidenceInventory,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        used_fingerprints: frozenset[str],
    ) -> tuple[InstructionCandidate, ...]:
        """Bind one fixed set without changing its local validation and retry contract."""
        if self.config.tasks.decision_routing.binding_enabled:
            supported = tuple(
                candidate
                for candidate in candidates
                if candidate.task_id in SUPPORTED_BINDING_TASKS and candidate.profile == "normal"
            )
            fallback = tuple(candidate for candidate in candidates if candidate not in supported)
            admitted = (
                self._bind_decision_candidates(
                    supported,
                    inventory,
                    snapshot,
                    generator,
                    target_language,
                    model_image,
                    image_views,
                    used_fingerprints,
                )
                if supported
                else ()
            )
            if fallback:
                self.store.write_json_artifact(
                    "routing-binding-fallback",
                    {
                        "conversation_id": snapshot.conversation_id,
                        "turn_index": snapshot.turn_index,
                        "candidate_ids": [candidate.candidate_id for candidate in fallback],
                        "reason": "unsupported_task_or_profile",
                        "path": "existing_candidate_binding",
                    },
                )
                try:
                    admitted += self._bind_generator_candidates(
                        fallback,
                        inventory,
                        snapshot,
                        generator,
                        target_language,
                        model_image,
                        image_views,
                        used_fingerprints,
                    )
                except ExecutionError as error:
                    return self._retain_independent_bindings(error, admitted, fallback, snapshot)
            return admitted
        return self._bind_generator_candidates(
            candidates,
            inventory,
            snapshot,
            generator,
            target_language,
            model_image,
            image_views,
            used_fingerprints,
        )

    def _bind_decision_candidates(
        self,
        candidates: tuple[InstructionCandidate, ...],
        inventory: ScopedEvidenceInventory,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        used_fingerprints: frozenset[str],
    ) -> tuple[InstructionCandidate, ...]:
        """Assemble known anchors and short open choices, then classify every required condition."""
        if self.decision_router is None:
            raise ExecutionError("DECISION_CLIENT_REQUIRED", "Classifier client is absent")
        proposals, unresolved = controller_binding_proposals(candidates, inventory)
        admitted = (
            self._judge_decision_proposals(
                proposals,
                candidates,
                inventory,
                snapshot,
                model_image,
                used_fingerprints,
            )
            if proposals.bindings
            else ()
        )
        if not unresolved:
            return admitted
        try:
            generated = self._invoke(
                generator,
                "candidate_proposal",
                {
                    "target_language": target_language,
                    "public_history": self._history(snapshot.public_history),
                    "candidates": [
                        binding_candidate(candidate, inventory) for candidate in unresolved
                    ],
                    "scope_evidence": inventory.model_dump(mode="json", exclude_none=True),
                    "image_views": image_views,
                },
                (model_image,),
                BindingProposals,
                max_tokens=self.config.tasks.decision_routing.proposal_max_tokens,
                temperature=0.0,
                seed=self.config.seed + snapshot.turn_index,
                post_validate=lambda report: validate_usable_binding_proposals(
                    report,
                    unresolved,
                    inventory,
                    snapshot.public_history,
                ),
            )
        except ExecutionError as error:
            return self._retain_independent_bindings(error, admitted, unresolved, snapshot)
        return admitted + self._judge_decision_proposals(
            generated,
            unresolved,
            inventory,
            snapshot,
            model_image,
            used_fingerprints,
        )

    def _retain_independent_bindings(
        self,
        error: ExecutionError,
        admitted: tuple[InstructionCandidate, ...],
        failed_candidates: tuple[InstructionCandidate, ...],
        snapshot: HistorySnapshot,
    ) -> tuple[InstructionCandidate, ...]:
        """Discard a malformed generator response while keeping separate verified candidates."""
        if error.reason not in _GENERATED_BINDING_OUTPUT_FAILURES or not admitted:
            raise error
        self.store.write_json_artifact(
            "routing-binding-branch-failures",
            {
                "conversation_id": snapshot.conversation_id,
                "turn_index": snapshot.turn_index,
                "failed_candidate_ids": [candidate.candidate_id for candidate in failed_candidates],
                "retained_candidate_ids": [candidate.candidate_id for candidate in admitted],
                "reason": error.reason,
                "message": str(error),
                "entire_generated_response_rejected": True,
            },
        )
        return admitted

    def _judge_decision_proposals(
        self,
        proposals: BindingProposals,
        candidates: tuple[InstructionCandidate, ...],
        inventory: ScopedEvidenceInventory,
        snapshot: HistorySnapshot,
        model_image: ModelImage,
        used_fingerprints: frozenset[str],
    ) -> tuple[InstructionCandidate, ...]:
        """Check and validate one independent controller or generator proposal set."""
        if self.decision_router is None:
            raise ExecutionError("DECISION_CLIENT_REQUIRED", "Classifier client is absent")
        self.store.write_json_artifact(
            "routing-binding-proposals",
            {
                "conversation_id": snapshot.conversation_id,
                "turn_index": snapshot.turn_index,
                "proposals": proposals.model_dump(mode="json"),
            },
        )
        proposals, proposal_rejections = partition_binding_proposals(
            proposals,
            candidates,
            inventory,
            snapshot.public_history,
        )
        if proposal_rejections:
            self.store.write_json_artifact(
                "routing-proposal-rejections",
                {
                    "conversation_id": snapshot.conversation_id,
                    "turn_index": snapshot.turn_index,
                    "rejections": [asdict(rejection) for rejection in proposal_rejections],
                },
            )
        verdicts = {
            query.query_id: self.decision_router.decide(
                query,
                image_id=inventory.image_id,
                image=model_image,
                trial_id=f"{self.run_id}/{inventory.image_id}/binding/{snapshot.turn_index}/{snapshot.history_hash}",
            )
            for query in binding_decisions(
                proposals, candidates, inventory, snapshot.public_history
            )
        }
        bindings = assemble_bindings(
            proposals,
            verdicts,
            candidates,
            inventory,
            snapshot.public_history,
            answer_max_tokens=self.config.tasks.answer_max_tokens,
        )
        self.store.write_json_artifact("candidate-bindings", bindings.model_dump(mode="json"))
        result = bind_candidates_individually(
            candidates,
            bindings,
            inventory,
            snapshot.public_history,
            self.config.tasks,
            used_fingerprints,
        )
        if result.rejected:
            self.store.write_json_artifact(
                "candidate-binding-rejections",
                {
                    "conversation_id": snapshot.conversation_id,
                    "turn_index": snapshot.turn_index,
                    "rejections": [asdict(rejection) for rejection in result.rejected],
                },
            )
        return result.admitted

    def _bind_generator_candidates(
        self,
        candidates: tuple[InstructionCandidate, ...],
        inventory: ScopedEvidenceInventory,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        used_fingerprints: frozenset[str],
    ) -> tuple[InstructionCandidate, ...]:
        """Run the original bounded binding path without changing its local validators."""
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index

        def validate_bindings(
            result: CandidateBindingsReport,
            templates: tuple[InstructionCandidate, ...] = candidates,
            public_history: tuple[PublicMessage, ...] = snapshot.public_history,
        ) -> None:
            parsed, parse_rejections = result.partition_bindings(
                frozenset(template.candidate_id for template in templates)
            )
            result_batch = bind_candidates_individually(
                templates,
                parsed,
                inventory,
                public_history,
                self.config.tasks,
                used_fingerprints,
            )
            if not result_batch.admitted and (parse_rejections or result_batch.rejected):
                first = (*parse_rejections, *result_batch.rejected)[0]
                raise ExecutionError(first.reason, first.message)

        bindings_report = self._invoke(
            generator,
            "candidate_binding",
            {
                "target_language": target_language,
                "public_history": self._history(snapshot.public_history),
                "candidates": [binding_candidate(candidate, inventory) for candidate in candidates],
                "scope_evidence": inventory.model_dump(mode="json", exclude_none=True),
                "answer_max_tokens": self.config.tasks.answer_max_tokens,
                "image_views": image_views,
            },
            (model_image,),
            CandidateBindingsReport,
            max_tokens=self.config.tasks.binding_max_tokens,
            temperature=0.0,
            seed=self.config.seed + turn_index,
            post_validate=validate_bindings,
        )
        bindings, parse_rejections = bindings_report.partition_bindings(
            frozenset(candidate.candidate_id for candidate in candidates)
        )
        self.store.write_json_artifact("candidate-bindings", bindings.model_dump(mode="json"))
        binding_result = bind_candidates_individually(
            candidates,
            bindings,
            inventory,
            snapshot.public_history,
            self.config.tasks,
            used_fingerprints,
        )
        rejected_bindings = (*parse_rejections, *binding_result.rejected)
        if rejected_bindings:
            self.store.write_json_artifact(
                "candidate-binding-rejections",
                {
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "rejections": [
                        {
                            "candidate_id": item.candidate_id,
                            "reason": item.reason,
                            "message": item.message,
                        }
                        for item in rejected_bindings
                    ],
                },
            )
        return binding_result.admitted

    def _attempt_candidates(
        self,
        priority: tuple[InstructionCandidate, ...],
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
    ) -> TurnAttemptResult:
        """Try a fixed candidate order after local question failures only.

        The failed questions and decisions remain private. Every candidate sees exactly the
        same committed prefix, and transport failures or unknown evidence stop the attempt.
        """
        if not priority or len(priority) > self.config.tasks.max_candidate_attempts:
            raise ValueError("Candidate attempts exceed the configured bound")
        plan = tuple(
            {
                "attempt_id": canonical_hash(
                    {
                        "run": self.run_id,
                        "conversation": snapshot.conversation_id,
                        "turn": snapshot.turn_index,
                        "history": snapshot.history_hash,
                        "candidate": candidate.candidate_id,
                        "rank": rank,
                    }
                ),
                "candidate_id": candidate.candidate_id,
                "rank": rank,
            }
            for rank, candidate in enumerate(priority)
        )
        self.store.write_json_artifact(
            "candidate-attempt-plans",
            {
                "conversation_id": snapshot.conversation_id,
                "turn_index": snapshot.turn_index,
                "history_hash": snapshot.history_hash,
                "attempts": plan,
            },
        )
        rejected_questions: set[str] = set()
        for candidate, attempt in zip(priority, plan, strict=True):
            try:
                outcome = self._attempt_turn(
                    candidate,
                    snapshot,
                    generator,
                    target_language,
                    model_image,
                    image_views,
                    previous_turns,
                    rejected_question_fingerprints=frozenset(rejected_questions),
                )
            except ExecutionError as error:
                self.store.write_json_artifact(
                    "candidate-attempt-outcomes",
                    {**attempt, "status": "EXECUTION_FAILURE", "reason": error.reason},
                )
                raise
            self.store.write_json_artifact(
                "candidate-attempt-outcomes",
                {
                    **attempt,
                    "status": outcome.status,
                    "stage": outcome.stage,
                    "reason": outcome.reason,
                    "question_fingerprint": outcome.question_fingerprint,
                },
            )
            if not (
                outcome.turn is None
                and outcome.status == "REJECTED"
                and (
                    outcome.stage == "question_generation"
                    or (
                        outcome.stage == "question_gate"
                        and outcome.reason == "QUESTION_GATE_NOT_MET"
                    )
                )
            ):
                return outcome
            if outcome.question_fingerprint is not None:
                rejected_questions.add(outcome.question_fingerprint)
        return outcome

    def _attempt_turn(
        self,
        selected: InstructionCandidate,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
        *,
        rejected_question_fingerprints: frozenset[str] = frozenset(),
    ) -> TurnAttemptResult:
        """Generate and verify one selected operation against its unchanged public prefix."""
        drafted = self._generate_question(
            selected,
            snapshot,
            generator,
            target_language,
            model_image,
            image_views,
            previous_turns,
            rejected_question_fingerprints=rejected_question_fingerprints,
        )
        if isinstance(drafted, TurnAttemptResult):
            return drafted
        question = drafted
        turn_index = snapshot.turn_index
        requirements: tuple[Requirement, ...] = ()
        if self.config.evaluation.mode == "detailed" or selected.catalog_version is not None:
            stopped = self._gate_question(
                selected, snapshot, question, target_language, model_image, image_views
            )
            if stopped is not None:
                return stopped
        if self.config.evaluation.mode == "detailed":
            requirement_status, requirements = self._extract_requirements(
                snapshot.public_history,
                question,
                target_language,
                turn_index,
            )
            if requirement_status is not GateVerdict.MET:
                terminal_status = (
                    "REJECTED" if requirement_status is GateVerdict.NOT_MET else "ABSTAINED"
                )
                terminal_stage = "requirement_extraction"
                terminal_reason = f"REQUIREMENT_{requirement_status.value}"
                return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        answered = self._generate_answer(
            selected,
            snapshot,
            generator,
            question,
            requirements,
            target_language,
            model_image,
            image_views,
            previous_turns,
        )
        if isinstance(answered, TurnAttemptResult):
            return answered
        return self._review_answer(
            selected,
            snapshot,
            generator,
            question,
            answered,
            requirements,
            target_language,
            model_image,
            image_views,
            previous_turns,
        )

    def _generate_question(
        self,
        selected: InstructionCandidate,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
        *,
        rejected_question_fingerprints: frozenset[str] = frozenset(),
    ) -> PublicMessage | TurnAttemptResult:
        """Write one public question and apply the deterministic pre-judge checks."""
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index
        turns = previous_turns
        terminal_status: Literal["REJECTED", "ABSTAINED", "ERROR"]
        target_value = next(
            (
                parameter.value
                for parameter in selected.public_parameters
                if parameter.name == "target"
            ),
            None,
        )

        def validate_question_draft(
            result: TextPayload,
            *,
            history: tuple[PublicMessage, ...] = snapshot.public_history,
            selected_instruction: InstructionCandidate = selected,
            target: str | int | bool | tuple[str, ...] | None = target_value,
            previous_turns: tuple[TurnArtifact, ...] = tuple(turns),
            current_turn: int = turn_index,
        ) -> None:
            if result.text is None:
                return
            if question_fingerprint(result.text) in rejected_question_fingerprints:
                reason = "REPEATED_REJECTED_QUESTION"
            elif internal_reference_in_question(result.text):
                reason = "INTERNAL_REFERENCE_IN_QUESTION"
            elif repeated_public_question(result.text, history):
                reason = "REPEATED_PUBLIC_QUESTION"
            elif (
                selected_instruction.task_id == "object_identification"
                and isinstance(target, str)
                and identification_label_in_question(result.text, target)
            ):
                reason = "IDENTIFICATION_TARGET_IN_QUESTION"
            elif (
                selected_instruction.task_id == "object_identification"
                and isinstance(target, str)
                and reciprocal_identification_disclosure(
                    result.text,
                    target,
                    tuple(
                        (turn.question, turn.answer)
                        for turn in previous_turns
                        if turn.status == "COMMITTED"
                        and turn.instruction.task_id == "object_identification"
                    ),
                )
            ):
                reason = "RECIPROCAL_IDENTIFICATION_ALREADY_PUBLIC"
            elif selected_instruction.task_id == "visible_action_relation" and (
                action_affordance_question(result.text)
            ):
                reason = "ACTION_AFFORDANCE_NOT_VISIBLE"
            elif selected_instruction.task_id == "text_transcription" and (
                unverified_transcription_relation(result.text)
            ):
                reason = "TEXT_RELATION_UNVERIFIED"
            elif selected_instruction.task_id == "scene_categorization" and not (
                isinstance(
                    options := next(
                        (
                            parameter.value
                            for parameter in selected_instruction.public_parameters
                            if parameter.name == "category_set"
                        ),
                        None,
                    ),
                    tuple,
                )
                and scene_options_in_question(result.text, options)
            ):
                reason = "CATEGORY_OPTIONS_NOT_PUBLIC"
            else:
                return
            self._record_public_text_rejection(
                conversation_id,
                current_turn,
                field="question",
                reason=reason,
                content=result.text,
            )
            raise ExecutionError(reason, "Question draft discloses or repeats its requested fact")

        try:
            question_payload = self._invoke(
                generator,
                "question_generation",
                {
                    "target_language": target_language,
                    "turn_index": turn_index,
                    "public_history": self._history(snapshot.public_history),
                    "selected_instruction": question_operation_contract(
                        selected, guidance=self.config.tasks.question_operation_guidance
                    ),
                    "image_views": image_views,
                },
                (model_image,),
                TextPayload,
                max_tokens=256,
                temperature=0.7,
                seed=self.config.seed + turn_index,
                post_validate=validate_question_draft,
            )
        except ExecutionError as error:
            if error.reason not in {
                "INTERNAL_REFERENCE_IN_QUESTION",
                "REPEATED_PUBLIC_QUESTION",
                "IDENTIFICATION_TARGET_IN_QUESTION",
                "RECIPROCAL_IDENTIFICATION_ALREADY_PUBLIC",
                "ACTION_AFFORDANCE_NOT_VISIBLE",
                "TEXT_RELATION_UNVERIFIED",
                "CATEGORY_OPTIONS_NOT_PUBLIC",
                "REPEATED_REJECTED_QUESTION",
            }:
                raise
            terminal_status = "REJECTED"
            terminal_stage = "question_generation"
            terminal_reason = error.reason
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        if question_payload.status != "OK":
            terminal_status = "REJECTED"
            terminal_stage = "question_generation"
            terminal_reason = "QUESTION_NOT_GENERATED"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        assert question_payload.text is not None
        question = PublicMessage(
            message_id=f"{conversation_id}:q:{turn_index}",
            turn_index=turn_index,
            role="user",
            content=question_payload.text,
        )
        if is_private_prompt_echo(question.content):
            self._record_public_text_rejection(
                conversation_id,
                turn_index,
                field="question",
                reason="PRIVATE_PROMPT_ECHO",
                content=question.content,
            )
            terminal_status = "REJECTED"
            terminal_stage = "question_generation"
            terminal_reason = "PRIVATE_PROMPT_ECHO"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        if internal_reference_in_question(question.content):
            self._record_public_text_rejection(
                conversation_id,
                turn_index,
                field="question",
                reason="INTERNAL_REFERENCE_IN_QUESTION",
                content=question.content,
            )
            terminal_status = "REJECTED"
            terminal_stage = "question_generation"
            terminal_reason = "INTERNAL_REFERENCE_IN_QUESTION"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        if repeated_public_question(question.content, snapshot.public_history):
            self._record_public_text_rejection(
                conversation_id,
                turn_index,
                field="question",
                reason="REPEATED_PUBLIC_QUESTION",
                content=question.content,
            )
            terminal_status = "REJECTED"
            terminal_stage = "question_generation"
            terminal_reason = "REPEATED_PUBLIC_QUESTION"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        return question

    def _gate_question(
        self,
        selected: InstructionCandidate,
        snapshot: HistorySnapshot,
        question: PublicMessage,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> TurnAttemptResult | None:
        """Run the blind operation classification and question-fit gates before answering."""
        turn_index = snapshot.turn_index
        fit = self._question_intent(
            snapshot.public_history,
            question,
            selected,
            target_language,
            image_views,
            model_image,
            turn_index,
        )
        if fit is GateVerdict.MET:
            fit = self._question_fit(
                snapshot.public_history,
                question,
                selected,
                target_language,
                image_views,
                model_image,
                turn_index,
            )
        if fit is not GateVerdict.MET:
            terminal_status = "ABSTAINED" if fit is GateVerdict.UNKNOWN else "REJECTED"
            terminal_stage = "question_gate"
            terminal_reason = f"QUESTION_GATE_{fit.value}"
            return TurnAttemptResult(
                None,
                terminal_status,
                terminal_stage,
                terminal_reason,
                question_fingerprint(question.content),
            )
        return None

    def _generate_answer(
        self,
        selected: InstructionCandidate,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        question: PublicMessage,
        requirements: tuple[Requirement, ...],
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...],
    ) -> PublicMessage | TurnAttemptResult:
        """Generate the public answer and stop on deterministic disclosure checks."""
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index
        turns = previous_turns
        answer_payload = self._invoke(
            generator,
            "answer_generation",
            {
                "target_language": target_language,
                "public_history": self._history(snapshot.public_history),
                "question": question.content,
                "expected_operation": operation_contract(selected),
                "active_requirements": [
                    requirement.model_dump(mode="json") for requirement in requirements
                ],
                "image_views": image_views,
            },
            (model_image,),
            TextPayload,
            max_tokens=self.config.tasks.answer_max_tokens,
            temperature=0.0,
            seed=self.config.seed + turn_index,
        )
        if answer_payload.status != "OK":
            terminal_status = "REJECTED"
            terminal_stage = "answer_generation"
            terminal_reason = "ANSWER_NOT_GENERATED"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        assert answer_payload.text is not None
        answer = PublicMessage(
            message_id=f"{conversation_id}:a:{turn_index}",
            turn_index=turn_index,
            role="assistant",
            content=answer_payload.text,
        )
        if is_private_prompt_echo(answer.content):
            self._record_public_text_rejection(
                conversation_id,
                turn_index,
                field="answer",
                reason="PRIVATE_PROMPT_ECHO",
                content=answer.content,
            )
            terminal_status = "REJECTED"
            terminal_stage = "answer_generation"
            terminal_reason = "PRIVATE_PROMPT_ECHO"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        if repeated_answered_request(question.content, answer.content, snapshot.public_history):
            self._record_public_text_rejection(
                conversation_id,
                turn_index,
                field="answer",
                reason="REPEATED_ANSWERED_REQUEST",
                content=answer.content,
            )
            terminal_status = "REJECTED"
            terminal_stage = "answer_generation"
            terminal_reason = "REPEATED_ANSWERED_REQUEST"
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        disclosure = self._answer_disclosure_reason(
            selected, question.content, answer.content, turns
        )
        if disclosure is not None:
            self._record_public_text_rejection(
                conversation_id,
                turn_index,
                field="answer",
                reason=disclosure,
                content=answer.content,
            )
            terminal_status = "REJECTED"
            terminal_stage = "answer_generation"
            terminal_reason = disclosure
            return TurnAttemptResult(None, terminal_status, terminal_stage, terminal_reason)
        return answer

    def _review_answer(
        self,
        selected: InstructionCandidate,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        question: PublicMessage,
        answer: PublicMessage,
        requirements: tuple[Requirement, ...],
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...] = (),
    ) -> TurnAttemptResult:
        """Rate one answered turn, repairing once when configured, and build its artifact."""
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index
        repair = self.config.evaluation.mode == "holistic" and self.config.evaluation.repair_once
        objections: list[str] | None = [] if repair else None
        rating = self._rate_turn(
            conversation_id,
            snapshot.history_hash,
            snapshot.public_history,
            question,
            answer,
            selected,
            target_language,
            image_views,
            model_image,
            turn_index,
            requirements,
            objections=objections,
        )
        if objections and rating.aggregate == "ABSTAIN":
            repaired = self._repair_holistic_answer(
                generator,
                snapshot,
                question,
                answer,
                rating,
                objections[0],
                selected,
                target_language,
                image_views,
                model_image,
                previous_turns,
            )
            if repaired is not None:
                answer = repaired
                rating = self._rate_turn(
                    conversation_id,
                    snapshot.history_hash,
                    snapshot.public_history,
                    question,
                    answer,
                    selected,
                    target_language,
                    image_views,
                    model_image,
                    turn_index,
                    requirements,
                )
        if rating.aggregate == "FAIL" and self.config.evaluation.mode == "detailed":
            self.store.write_json_artifact(
                "answer-attempts",
                {
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "attempt": 0,
                    "answer": answer.model_dump(mode="json"),
                    "rating": rating.model_dump(mode="json"),
                },
            )
            repaired = self._repair_answer(
                generator,
                snapshot.public_history,
                question,
                answer,
                rating,
                target_language,
                image_views,
                model_image,
                turn_index,
                requirements,
            )
            if repaired is not None:
                answer = repaired
                rating = self._rate_turn(
                    conversation_id,
                    snapshot.history_hash,
                    snapshot.public_history,
                    question,
                    answer,
                    selected,
                    target_language,
                    image_views,
                    model_image,
                    turn_index,
                    requirements,
                )
        turn_status: Literal["COMMITTED", "REJECTED", "ABSTAINED", "ERROR"]
        if rating.aggregate == "PASS":
            turn_status = "COMMITTED"
        elif rating.aggregate == "FAIL":
            turn_status = "REJECTED"
        elif rating.aggregate == "ABSTAIN":
            turn_status = "ABSTAINED"
        else:
            turn_status = "ERROR"
        turn = TurnArtifact(
            turn_index=turn_index,
            instruction=selected,
            question=question,
            answer=answer,
            history_hash=snapshot.history_hash,
            generation_model=generator.endpoint.repo_id,
            selector_model=self.selector.endpoint.repo_id,
            requirements=requirements,
            rating=rating,
            status=turn_status,
        )
        return TurnAttemptResult(
            turn, turn_status, "answer_verification", f"RATING_{rating.aggregate}"
        )

    def rate_existing(
        self,
        conversation: ConversationArtifact,
        artifact_root: Path,
    ) -> ConversationArtifact:
        """Re-evaluate immutable text, preserving conversation-local inference failures."""
        model_image = self._model_image(conversation.image, artifact_root)
        image_views = [self._view_metadata(conversation.image)]
        transcript: tuple[PublicMessage, ...] = ()
        rated_turns: list[TurnArtifact] = []
        terminal: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"] = (
            "QUALITY_CANDIDATE"
        )
        for turn in conversation.turns:
            snapshot = build_history_snapshot(
                conversation.conversation_id,
                turn.turn_index,
                transcript,
            )
            repeated = repeated_answered_request(
                turn.question.content, turn.answer.content, snapshot.public_history
            )
            if repeated:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="answer",
                    reason="REPEATED_ANSWERED_REQUEST",
                    content=turn.answer.content,
                )
            disclosure = self._answer_disclosure_reason(
                turn.instruction,
                turn.question.content,
                turn.answer.content,
                rated_turns,
            )
            if disclosure is not None:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="answer",
                    reason=disclosure,
                    content=turn.answer.content,
                )
            options = next(
                (
                    parameter.value
                    for parameter in turn.instruction.public_parameters
                    if parameter.name == "category_set"
                ),
                None,
            )
            missing_scene_options = turn.instruction.task_id == "scene_categorization" and not (
                isinstance(options, tuple)
                and scene_options_in_question(turn.question.content, options)
            )
            if missing_scene_options:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="question",
                    reason="CATEGORY_OPTIONS_NOT_PUBLIC",
                    content=turn.question.content,
                )
            internal_reference = internal_reference_in_question(turn.question.content)
            if internal_reference:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="question",
                    reason="INTERNAL_REFERENCE_IN_QUESTION",
                    content=turn.question.content,
                )
            unverified_text_relation = (
                turn.instruction.task_id == "text_transcription"
                and unverified_transcription_relation(turn.question.content)
            )
            if unverified_text_relation:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="question",
                    reason="TEXT_RELATION_UNVERIFIED",
                    content=turn.question.content,
                )
            unsupported_action_affordance = (
                turn.instruction.task_id == "visible_action_relation"
                and action_affordance_question(turn.question.content)
            )
            if unsupported_action_affordance:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="question",
                    reason="ACTION_AFFORDANCE_NOT_VISIBLE",
                    content=turn.question.content,
                )
            target = next(
                (
                    parameter.value
                    for parameter in turn.instruction.public_parameters
                    if parameter.name == "target"
                ),
                None,
            )
            reciprocal_identification = (
                turn.instruction.task_id == "object_identification"
                and isinstance(target, str)
                and reciprocal_identification_disclosure(
                    turn.question.content,
                    target,
                    tuple(
                        (prior.question, prior.answer)
                        for prior in rated_turns
                        if prior.status == "COMMITTED"
                        and prior.instruction.task_id == "object_identification"
                    ),
                )
            )
            if reciprocal_identification:
                self._record_public_text_rejection(
                    conversation.conversation_id,
                    turn.turn_index,
                    field="question",
                    reason="RECIPROCAL_IDENTIFICATION_ALREADY_PUBLIC",
                    content=turn.question.content,
                )
            fit = (
                GateVerdict.NOT_MET
                if repeated
                or disclosure
                or missing_scene_options
                or unverified_text_relation
                or unsupported_action_affordance
                or reciprocal_identification
                or internal_reference
                else GateVerdict.MET
            )
            try:
                rating = self._rate_existing_fit(
                    fit, conversation, turn, snapshot, image_views, model_image
                )
            except ExecutionError as error:
                abstained = error.reason in MODEL_OUTPUT_ABSTENTIONS
                self.store.write_json_artifact(
                    "model-output-abstentions" if abstained else "errors",
                    {
                        "conversation_id": conversation.conversation_id,
                        "image_id": conversation.image.image_id,
                        "turn_index": turn.turn_index,
                        "reason": error.reason,
                        "message": str(error),
                        "operation": "rate-existing",
                    },
                )
                rating = TurnRating(items=(), aggregate="ABSTAIN" if abstained else "ERROR")
            status: Literal["COMMITTED", "REJECTED", "ABSTAINED", "ERROR"]
            if rating.aggregate == "PASS":
                status = "COMMITTED"
                transcript = (*transcript, turn.question, turn.answer)
            elif rating.aggregate == "FAIL":
                status = "REJECTED"
                terminal = "REJECTED"
            elif rating.aggregate == "ABSTAIN":
                status = "ABSTAINED"
                terminal = "ABSTAINED"
            else:
                status = "ERROR"
                terminal = "ERROR"
            rated_turns.append(
                turn.model_copy(
                    update={
                        "history_hash": snapshot.history_hash,
                        "rating": rating,
                        "status": status,
                    }
                )
            )
            if status != "COMMITTED":
                break
        if terminal == "QUALITY_CANDIDATE" and len(rated_turns) < self.config.data.min_turns:
            terminal = "REJECTED"
        if (
            terminal == "QUALITY_CANDIDATE"
            and self.config.evaluation.mode == "detailed"
            and len(rated_turns) != len(conversation.turns)
        ):
            terminal = "REJECTED"
        rated = conversation.model_copy(update={"turns": tuple(rated_turns), "status": terminal})
        return self._finish_conversation(rated, persist=False)

    def _rate_existing_fit(
        self,
        fit: GateVerdict,
        conversation: ConversationArtifact,
        turn: TurnArtifact,
        snapshot: HistorySnapshot,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
    ) -> TurnRating:
        """Apply question and answer gates to one stored turn without changing its text."""
        if fit is GateVerdict.MET and (
            self.config.evaluation.mode == "detailed"
            or turn.instruction.catalog_version is not None
        ):
            fit = self._question_intent(
                snapshot.public_history,
                turn.question,
                turn.instruction,
                conversation.target_language,
                image_views,
                model_image,
                turn.turn_index,
            )
            if fit is GateVerdict.MET:
                fit = self._question_fit(
                    snapshot.public_history,
                    turn.question,
                    turn.instruction,
                    conversation.target_language,
                    image_views,
                    model_image,
                    turn.turn_index,
                )
        if fit is GateVerdict.MET:
            return self._rate_turn(
                conversation.conversation_id,
                snapshot.history_hash,
                snapshot.public_history,
                turn.question,
                turn.answer,
                turn.instruction,
                conversation.target_language,
                image_views,
                model_image,
                turn.turn_index,
                turn.requirements,
            )
        return TurnRating(items=(), aggregate="ABSTAIN" if fit is GateVerdict.UNKNOWN else "FAIL")

    def _repair_answer(
        self,
        generator: InferenceClient,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        rating: TurnRating,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
        requirements: Sequence[Requirement],
    ) -> PublicMessage | None:
        failed = tuple(
            item.template_id for item in rating.items if item.verdict is GateVerdict.NOT_MET
        )
        result = self._invoke(
            generator,
            "answer_repair",
            {
                "target_language": language,
                "public_history": self._history(history),
                "question": question.content,
                "candidate_answer": answer.content,
                "failed_criteria": failed,
                "active_requirements": [
                    requirement.model_dump(mode="json") for requirement in requirements
                ],
                "image_views": image_views,
            },
            (model_image,),
            TextPayload,
            max_tokens=self.config.tasks.answer_max_tokens,
            temperature=0.0,
            seed=self.config.seed + turn_index + 10_000,
        )
        if result.status != "OK" or result.text is None:
            return None
        return answer.model_copy(update={"content": result.text})

    def _repair_holistic_answer(
        self,
        generator: InferenceClient,
        snapshot: HistorySnapshot,
        question: PublicMessage,
        answer: PublicMessage,
        rating: TurnRating,
        objection: str,
        selected: InstructionCandidate,
        target_language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        previous_turns: tuple[TurnArtifact, ...],
    ) -> PublicMessage | None:
        """Replace an answer once after one judge's concrete objection.

        The generator sees the objection text but never the other judge's vote. The repaired
        answer must pass the same deterministic public-text checks before both judges review it
        again from scratch; otherwise the original abstention stands.
        """
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index
        self.store.write_json_artifact(
            "answer-attempts",
            {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "attempt": 0,
                "answer": answer.model_dump(mode="json"),
                "rating": rating.model_dump(mode="json"),
                "objection": objection,
            },
        )
        result = self._invoke(
            generator,
            "answer_repair",
            {
                "target_language": target_language,
                "public_history": self._history(snapshot.public_history),
                "question": question.content,
                "candidate_answer": answer.content,
                "failed_criteria": ["holistic_review: " + objection],
                "active_requirements": [],
                "image_views": image_views,
            },
            (model_image,),
            TextPayload,
            max_tokens=self.config.tasks.answer_max_tokens,
            temperature=0.0,
            seed=self.config.seed + turn_index + 10_000,
        )
        if result.status != "OK" or result.text is None:
            return None
        repaired = answer.model_copy(update={"content": result.text})
        reason: PublicTextRejectionReason | None = None
        if is_private_prompt_echo(repaired.content):
            reason = "PRIVATE_PROMPT_ECHO"
        elif repeated_answered_request(question.content, repaired.content, snapshot.public_history):
            reason = "REPEATED_ANSWERED_REQUEST"
        else:
            reason = self._answer_disclosure_reason(
                selected, question.content, repaired.content, previous_turns
            )
        if reason is not None:
            self._record_public_text_rejection(
                conversation_id, turn_index, field="answer", reason=reason, content=repaired.content
            )
            return None
        return repaired

    def _extract_requirements(
        self,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        language: str,
        turn_index: int,
    ) -> tuple[GateVerdict, tuple[Requirement, ...]]:
        messages = (*history, question)
        references = {f"m{index}": message.message_id for index, message in enumerate(messages)}
        local_messages = tuple(
            message.model_copy(update={"message_id": reference})
            for reference, message in zip(references, messages, strict=True)
        )
        inventories = [
            self._invoke(
                client,
                "requirement_extraction",
                {
                    "target_language": language,
                    "public_history": self._history(local_messages[:-1]),
                    "question": question.content,
                    "question_message_id": local_messages[-1].message_id,
                },
                (),
                RequirementInventory,
                max_tokens=1024,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                trial_id=f"blind-judge:{judge_index}",
            )
            for judge_index, client in enumerate(self.generators.values())
        ]
        inventories = [
            inventory.model_copy(
                update={
                    "requirements": tuple(
                        spec.model_copy(
                            update={"source_message_id": references[spec.source_message_id]}
                        )
                        for spec in inventory.requirements
                    )
                }
            )
            for inventory in inventories
        ]
        if not all(item.extraction_complete for item in inventories):
            return GateVerdict.UNKNOWN, ()
        requirements = reconcile_inventories(
            inventories,
            (*history, question),
            turn_index,
        )
        if requirements is None:
            return GateVerdict.UNKNOWN, ()
        return GateVerdict.MET, requirements

    def _select_instruction(
        self,
        candidates: tuple[InstructionCandidate, ...],
        history: Any,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
    ) -> InstructionCandidate | None:
        result = self._invoke(
            self.selector,
            "instruction_selection",
            {
                "target_language": language,
                "public_history": self._history(history.public_history),
                "candidates": [selector_candidate(candidate) for candidate in candidates],
                "image_views": image_views,
            },
            (model_image,),
            InstructionSelection,
            max_tokens=256,
            temperature=0.0,
            seed=self.config.seed + turn_index,
        )
        if result.status == "NO_SUITABLE_CANDIDATE":
            return None
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        if result.candidate_id not in by_id:
            return None
        return by_id[result.candidate_id]

    def record_failed_image(
        self,
        image: ImageArtifact,
        *,
        target_language: Literal["en", "ja", "zh-Hans"],
        generator_role: Literal["generator_a", "generator_b"],
        error: ExecutionError,
    ) -> ConversationArtifact:
        """Persist a failed image, abstaining when model output exhausted retries."""
        conversation_id = canonical_hash(
            {"run": self.run_id, "image": image.image_id, "language": target_language}
        )
        generator = self.generators[generator_role]
        turns = tuple(
            TurnArtifact.model_validate_json(self.store.read_artifact(artifact_hash))
            for artifact_hash in self.store.committed_artifact_hashes(conversation_id)
        )
        abstained = error.reason in MODEL_OUTPUT_ABSTENTIONS
        conversation = ConversationArtifact(
            conversation_id=conversation_id,
            image=image,
            target_language=target_language,
            generation_model=generator.endpoint.repo_id,
            turns=turns,
            status="ABSTAINED" if abstained else "ERROR",
        )
        self.store.write_json_artifact(
            "model-output-abstentions" if abstained else "errors",
            {
                "conversation_id": conversation_id,
                "image_id": image.image_id,
                "reason": error.reason,
                "message": str(error),
            },
        )
        if abstained:
            return self._finish_conversation(conversation, persist=True)
        self.store.write_json_artifact("conversations", conversation.model_dump(mode="json"))
        return conversation

    def _question_intent(
        self,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        instruction: InstructionCandidate,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
    ) -> GateVerdict:
        """Classify the public operation without showing its selected task or answer."""
        catalog = task_catalog()
        public_payload = {
            "target_language": language,
            "public_history": self._history(history),
            "question": question.content,
            "task_definitions": [
                {"task_id": task.id, "definition": task.definition_en} for task in catalog.tasks
            ],
            "image_views": image_views,
        }
        votes = self._invoke_judges(
            "question_intent",
            public_payload,
            (model_image,),
            QuestionIntent,
            max_tokens=128,
            seed=self.config.seed + turn_index,
        )
        known = {task.id for task in catalog.tasks}
        task_ids = [vote.task_id for vote in votes]
        if any(task_id not in known for task_id in task_ids):
            verdict = GateVerdict.UNKNOWN
        elif task_ids == [instruction.task_id, instruction.task_id]:
            verdict = GateVerdict.MET
        elif task_ids[0] == task_ids[1]:
            verdict = GateVerdict.NOT_MET
        else:
            verdict = GateVerdict.UNKNOWN
        self.store.write_json_artifact(
            "question-intent-decisions",
            {
                "question_message_id": question.message_id,
                "expected_task_id": instruction.task_id,
                "votes": [vote.model_dump(mode="json") for vote in votes],
                "verdict": verdict.value,
            },
        )
        return verdict

    def _question_fit(
        self,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        instruction: InstructionCandidate,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
    ) -> GateVerdict:
        focused = self._local_verification_view(instruction, image_views, model_image)
        if focused is None:
            return GateVerdict.UNKNOWN
        model_image, image_views = focused
        votes = self._invoke_judges(
            "question_fit",
            {
                "target_language": language,
                "public_history": self._history(history),
                "selected_instruction": operation_contract(instruction),
                "question": question.content,
                "image_views": image_views,
            },
            (model_image,),
            QuestionFit,
            max_tokens=256,
            seed=self.config.seed + turn_index,
        )
        return question_fit_consensus(votes)

    def _local_verification_view(
        self,
        instruction: InstructionCandidate,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
    ) -> tuple[ModelImage, list[dict[str, str]]] | None:
        """Deliver only bound pixels to local question-fit and holistic judges."""
        region = instruction.target_region or instruction.scope_region
        if (
            instruction.task_id
            not in {
                "object_identification",
                "attribute_lookup",
                "text_transcription",
                "text_reading_order",
                "code_transcription",
            }
            or region is None
            or region == ImageRegion(left=0, top=0, right=1, bottom=1)
        ):
            return model_image, image_views
        encoded = model_image.path.read_bytes()
        if hashlib.sha256(encoded).hexdigest() != model_image.encoded_sha256:
            raise ExecutionError("EVIDENCE_VIEW_MISMATCH", "Source pixels changed before cropping")
        view = focused_view(encoded, region)
        if view is None:
            self.store.write_json_artifact(
                "focus-view-abstentions",
                {
                    "candidate_id": instruction.candidate_id,
                    "region": region.model_dump(mode="json"),
                },
            )
            return None
        digest = self.store.write_artifact("focus-images", view.encoded)
        identifier = "focus:" + canonical_hash({"source": model_image.view_id, "pixels": digest})
        metadata = {
            "view_id": identifier,
            "encoded_sha256": digest,
            "width": str(view.width),
            "height": str(view.height),
            "source_view_id": model_image.view_id,
            **{
                "source_" + name: str(value)
                for name, value in view.source_region.model_dump().items()
            },
        }
        self.store.write_json_artifact("focus-views", metadata)
        return (
            ModelImage(
                view_id=identifier,
                path=self.store.artifact_dir / "focus-images" / digest[:2] / digest,
                encoded_sha256=digest,
                media_type="image/png",
            ),
            [metadata],
        )

    def _rate_turn(
        self,
        conversation_id: str,
        history_hash: str,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        instruction: InstructionCandidate,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
        requirements: Sequence[Requirement],
        *,
        objections: list[str] | None = None,
    ) -> TurnRating:
        """Require the configured base review and every applicable operation check."""
        if instruction.catalog_version is not None and instruction.view_id != model_image.view_id:
            raise ExecutionError(
                "EVIDENCE_VIEW_MISMATCH", "Operation refers to a different image view"
            )
        rating = self._rate_base_turn(
            conversation_id,
            history_hash,
            history,
            question,
            answer,
            instruction,
            language,
            image_views,
            model_image,
            turn_index,
            requirements,
            objections=objections,
        )
        if instruction.catalog_version is None or rating.aggregate != "PASS":
            return rating
        payload = {
            "target_language": language,
            "public_history": self._history(history),
            "question": question.content,
            "candidate_answer": answer.content,
            "image_views": image_views,
        }

        focused = self._local_verification_view(instruction, image_views, model_image)
        if focused is None:
            item = RubricItem(
                item_id=item_id(
                    conversation_id,
                    turn_index,
                    0,
                    "dual_visual_review",
                    "empty",
                    canonical_hash(payload),
                ),
                template_id="dual_visual_review",
                axis="factual_correctness",
                verdict=GateVerdict.UNKNOWN,
                reason="Bound subject region contains no complete pixel.",
                actor="controller",
                history_hash=history_hash,
            )
            return aggregate_rating((*rating.items, item))

        def invoke(
            stage: str, body: dict[str, Any], model: type[BaseModel], judge: int
        ) -> BaseModel:
            client = tuple(self.generators.values())[judge]
            # Full context lets blind transcription readers detect a requested unit
            # extending beyond its bound rectangle; the controller checks containment.
            max_tokens = (
                self.config.tasks.evidence_max_tokens
                if stage in {"table_source", "table_lookup_source", "chart_source", "graph_source"}
                or (stage.startswith("specialist_") and stage.endswith("_source"))
                else 2048
            )
            return self._invoke(
                client,
                stage,
                body,
                ()
                if stage
                in {
                    "finite_answer",
                    "quantity_answer",
                    "table_answer",
                    "chart_answer",
                    "graph_answer",
                    "scale_answer",
                    "pattern_answer",
                    "geometry_answer",
                    "specialist_geometry_answer",
                }
                else (model_image,),
                model,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                trial_id=f"blind-judge:{judge}",
            )

        items = list(rating.items)
        for check in verify_operation(
            instruction,
            payload,
            invoke,
            model_image.path,
        ):
            self.store.write_json_artifact(
                "operation-checks",
                {
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "contract": check.name,
                    "candidate_id": instruction.candidate_id,
                    "question_hash": canonical_hash(question.content),
                    "answer_hash": canonical_hash(answer.content),
                    "verdict": check.verdict.value,
                    "independent_evidence": check.evidence,
                },
            )
            items.append(
                RubricItem(
                    item_id=item_id(
                        conversation_id,
                        turn_index,
                        0,
                        check.name,
                        instruction.candidate_id,
                        canonical_hash(payload),
                    ),
                    template_id=check.name,
                    axis="factual_correctness",
                    verdict=check.verdict,
                    reason=check.reason,
                    actor="controller",
                    history_hash=history_hash,
                )
            )
        return aggregate_rating(items)

    def _rate_base_turn(
        self,
        conversation_id: str,
        history_hash: str,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        instruction: InstructionCandidate,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
        requirements: Sequence[Requirement],
        *,
        objections: list[str] | None = None,
    ) -> TurnRating:
        if self.config.evaluation.mode == "holistic":
            return self._rate_holistic(
                conversation_id,
                history_hash,
                history,
                question,
                answer,
                language,
                image_views,
                model_image,
                turn_index,
                instruction,
                objections=objections,
            )
        inventories = self._invoke_judges(
            "claim_inventory",
            {
                "target_language": language,
                "public_history": self._history(history),
                "question": question.content,
                "candidate_answer": answer.content,
                "answer_tokens": self._answer_tokens(answer.content),
            },
            (),
            ClaimExtraction,
            max_tokens=2048,
            seed=self.config.seed + turn_index,
        )
        inventories = [
            ClaimInventory(
                claims=tuple(self._bind_claim_span(claim, answer) for claim in inventory.claims),
                coverage=inventory.coverage,
                reason=inventory.reason,
            )
            for inventory in inventories
        ]
        coverage = consensus([inventory.coverage for inventory in inventories])
        claims = self._claim_union(inventories, answer)
        if claims is None:
            coverage = GateVerdict.UNKNOWN
            claims = ()
        computation = (
            self._check_computation(
                history,
                question,
                answer,
                language,
                image_views,
                model_image,
                turn_index,
            )
            if instruction.catalog_version is None and instruction.family == "grounded_calculation"
            else None
        )
        set_check = (
            self._check_complete_set(
                history,
                question,
                answer,
                language,
                image_views,
                model_image,
                turn_index,
            )
            if instruction.catalog_version is None and instruction.family == "visible_count"
            else None
        )
        context = RubricContext(
            turn_index=turn_index,
            profile=instruction.profile,
            has_natural_language_answer=has_natural_language_content(answer.content),
            requirements=tuple(requirements),
            claims=claims,
            computation_ids=(instruction.candidate_id,) if computation is not None else (),
            history_binding_ids=tuple(
                requirement.requirement_id
                for requirement in requirements
                if requirement.source_message_id != question.message_id
            ),
            exhaustive_scope_ids=(instruction.candidate_id,)
            if instruction.catalog_version is None and instruction.family == "visible_count"
            else (),
        )
        rubric_items: list[RubricItem] = []
        for template in applicable_rubric_items(context):
            predicate = template["applies_when"]
            subjects: Sequence[AtomicClaim | Requirement | None]
            if predicate == "each_factual_claim":
                subjects = claims
            elif predicate == "each_public_requirement":
                subjects = requirements
            elif predicate == "public_format_constraint":
                subjects = context.format_requirements
            elif predicate == "has_history_binding":
                subjects = tuple(
                    requirement
                    for requirement in requirements
                    if requirement.requirement_id in context.history_binding_ids
                )
            else:
                subjects = (None,)
            for subject in subjects:
                votes: list[RubricVerdict] = []
                for judge_index, client in enumerate(self.generators.values()):
                    payload, images = self._rubric_payload(
                        template,
                        history,
                        question,
                        answer,
                        language,
                        image_views,
                        model_image,
                        subject,
                        computation.typed_rule_inputs if computation is not None else {},
                    )
                    if template["template_id"] == "C_COVERAGE":
                        payload["candidate_claim_inventory"] = [
                            claim.model_dump(mode="json") for claim in claims
                        ]
                    votes.append(
                        self._invoke(
                            client,
                            "rubric_item",
                            payload,
                            images,
                            RubricVerdict,
                            max_tokens=256,
                            temperature=0.0,
                            seed=self.config.seed + turn_index,
                            trial_id=f"blind-judge:{judge_index}",
                        )
                    )
                verdict = consensus([GateVerdict(vote.verdict) for vote in votes])
                if template["template_id"] == "C_COVERAGE":
                    verdict = consensus([verdict, coverage])
                elif template["template_id"] == "C_COMPUTATION" and computation is not None:
                    if computation.verdict != "MET":
                        verdict = GateVerdict(computation.verdict)
                elif template["template_id"] == "C_SET_COMPLETE":
                    if set_check is None:
                        # Missing corroboration blocks a pass, not an agreed failure.
                        if verdict == GateVerdict.MET:
                            verdict = GateVerdict.UNKNOWN
                    elif set_check.verdict == "NOT_MET":
                        verdict = GateVerdict.NOT_MET
                self.store.write_json_artifact(
                    "rating-decisions",
                    {
                        "conversation_id": conversation_id,
                        "turn_index": turn_index,
                        "template_id": template["template_id"],
                        "subject": subject.model_dump(mode="json") if subject is not None else None,
                        "votes": [vote.model_dump(mode="json") for vote in votes],
                        "claim_coverage": coverage.value
                        if template["template_id"] == "C_COVERAGE"
                        else None,
                        "set_check": set_check.model_dump(mode="json")
                        if template["template_id"] == "C_SET_COMPLETE" and set_check is not None
                        else None,
                        "controller_reason": "SET_INVENTORY_UNRESOLVED"
                        if template["template_id"] == "C_SET_COMPLETE" and set_check is None
                        else None,
                        "verdict": verdict.value,
                    },
                )
                if isinstance(subject, AtomicClaim):
                    subject_id = subject.claim_id
                elif isinstance(subject, Requirement):
                    subject_id = subject.requirement_id
                else:
                    subject_id = "singleton"
                context_hash = canonical_hash(payload)
                rubric_items.append(
                    RubricItem(
                        item_id=item_id(
                            conversation_id,
                            turn_index,
                            0,
                            template["template_id"],
                            subject_id,
                            context_hash,
                        ),
                        template_id=template["template_id"],
                        axis=template["axis"],
                        verdict=verdict,
                        reason=" | ".join(
                            f"{name}:{vote.reason}"
                            for name, vote in zip(self.generators, votes, strict=True)
                        )[:240],
                        actor="dual-consensus",
                        history_hash=history_hash,
                    )
                )
        return aggregate_rating(rubric_items)

    def _rate_holistic(
        self,
        conversation_id: str,
        history_hash: str,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
        instruction: InstructionCandidate,
        *,
        objections: list[str] | None = None,
    ) -> TurnRating:
        """Make two blind whole-turn judgments without intermediate semantic extraction.

        With a configured tie-break, a judge that answered UNKNOWN beside a MET vote is asked
        once more with the complete image alone. When ``objections`` is supplied, the single
        concrete objection of a MET/NOT_MET split is appended for one answer repair.
        """
        invalid = (
            not question.content.strip()
            or not answer.content.strip()
            or is_private_prompt_echo(question.content)
            or internal_reference_in_question(question.content)
            or is_private_prompt_echo(answer.content)
            or repeated_public_question(question.content, history)
        )
        full_image, full_views = model_image, image_views
        focused = None if invalid else self._judge_inputs(instruction, image_views, model_image)
        images: tuple[ModelImage, ...] = (model_image,)
        if focused is not None:
            images, image_views = focused
        payload = {
            "target_language": language,
            "public_history": self._history(history),
            "question": question.content,
            "candidate_answer": answer.content,
            "image_views": image_views,
        }
        if instruction.catalog_version is not None:
            payload["expected_operation"] = operation_contract(instruction)
        votes = (
            []
            if invalid or focused is None
            else self._invoke_judges(
                "holistic_review",
                payload,
                images,
                RubricVerdict,
                max_tokens=384,
                seed=self.config.seed + turn_index,
            )
        )
        tiebreak: RubricVerdict | None = None
        objection: str | None = None
        if invalid:
            verdict = GateVerdict.NOT_MET
        elif focused is None:
            verdict = GateVerdict.UNKNOWN
        else:
            verdicts = [GateVerdict(vote.verdict) for vote in votes]
            decision = holistic_decision(
                verdicts,
                [vote.reason for vote in votes],
                tiebreak_available=self.config.evaluation.holistic_tiebreak == "full_view"
                and images != (full_image,),
                repair_available=objections is not None,
            )
            verdict = decision.verdict
            if decision.action == "tiebreak" and decision.judge_index is not None:
                index = decision.judge_index
                tiebreak = self._invoke(
                    tuple(self.generators.values())[index],
                    "holistic_review",
                    {**payload, "image_views": full_views},
                    (full_image,),
                    RubricVerdict,
                    max_tokens=384,
                    temperature=0.0,
                    seed=self.config.seed + turn_index,
                    trial_id=f"blind-judge:{index}:tiebreak",
                )
                verdicts[index] = GateVerdict(tiebreak.verdict)
                verdict = consensus(verdicts)
            elif decision.action == "repair" and objections is not None and decision.objection:
                objection = decision.objection
                objections.append(objection)
        reason = (
            "Empty, repeated, or private-prompt public text."
            if invalid
            else "Bound subject region contains no complete pixel."
            if focused is None
            else " | ".join(
                f"{role}:{vote.reason}" for role, vote in zip(self.generators, votes, strict=True)
            )[:240]
        )
        self.store.write_json_artifact(
            "rating-decisions",
            {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "template_id": "Q_HOLISTIC",
                "subject": None,
                "votes": [vote.model_dump(mode="json") for vote in votes],
                "tiebreak_vote": tiebreak.model_dump(mode="json") if tiebreak else None,
                "repair_objection": objection,
                "controller_reason": "INVALID_PUBLIC_TEXT"
                if invalid
                else "EMPTY_FOCUS_VIEW"
                if focused is None
                else None,
                "verdict": verdict.value,
            },
        )
        item = RubricItem(
            item_id=item_id(
                conversation_id, turn_index, 0, "Q_HOLISTIC", "singleton", canonical_hash(payload)
            ),
            template_id="Q_HOLISTIC",
            axis="factual_correctness",
            verdict=verdict,
            reason=reason,
            actor="controller" if invalid or focused is None else "dual-consensus",
            history_hash=history_hash,
        )
        return aggregate_rating((item,))

    def _finish_conversation(
        self,
        conversation: ConversationArtifact,
        *,
        persist: bool,
    ) -> ConversationArtifact:
        """Preserve stopped attempts privately and certify only the accepted public prefix."""
        if self.config.evaluation.mode == "holistic":
            prefix = tuple(turn for turn in conversation.turns if turn.status == "COMMITTED")
            if (
                self.config.evaluation.retain_accepted_prefix
                and conversation.status in {"REJECTED", "ABSTAINED"}
                and len(prefix) >= self.config.data.min_turns
            ):
                self.store.write_json_artifact(
                    "conversation-stops",
                    {
                        "conversation": conversation.model_dump(mode="json"),
                        "retained_turns": len(prefix),
                    },
                )
                conversation = ConversationArtifact(
                    conversation_id=conversation.conversation_id,
                    image=conversation.image,
                    target_language=conversation.target_language,
                    generation_model=conversation.generation_model,
                    turns=prefix,
                    status="QUALITY_CANDIDATE",
                )
        artifact_hash = self.store.write_json_artifact(
            "conversations", conversation.model_dump(mode="json")
        )
        if persist and self.config.evaluation.mode == "holistic":
            with self.store.transaction() as connection:
                connection.execute(
                    "INSERT INTO conversation_commit(conversation_id, artifact_hash) VALUES (?, ?)",
                    (conversation.conversation_id, artifact_hash),
                )
        return conversation

    def _drop_answered_candidates(
        self,
        candidates: Sequence[InstructionCandidate],
        prior_turns: Sequence[TurnArtifact],
        conversation_id: str,
        turn_index: int,
    ) -> tuple[InstructionCandidate, ...]:
        """Omit already answered object-name and explicit attribute requests."""
        retained: list[InstructionCandidate] = []
        answered_attributes = {
            key
            for turn in prior_turns
            if turn.status == "COMMITTED"
            if (key := attribute_fact_key(turn.instruction)) is not None
        }
        answered_facts = {
            key
            for turn in prior_turns
            if turn.status == "COMMITTED"
            if (key := requested_fact_key(turn.instruction)) is not None
        }
        if self.config.tasks.fact_novelty_enabled:
            self.store.write_json_artifact(
                "candidate-fact-identities",
                {
                    "conversation_id": conversation_id,
                    "turn_index": turn_index,
                    "candidates": [
                        {
                            "candidate_id": candidate.candidate_id,
                            "fact_key": asdict(key)
                            if (key := requested_fact_key(candidate))
                            else None,
                        }
                        for candidate in candidates
                    ],
                },
            )
        for candidate in candidates:
            if (fact_key := attribute_fact_key(candidate)) is not None and (
                fact_key in answered_attributes
                or self._direct_identification_stated_color(fact_key, prior_turns)
            ):
                self.store.write_json_artifact(
                    "candidate-admission-rejections",
                    {
                        "conversation_id": conversation_id,
                        "turn_index": turn_index,
                        "candidate_id": candidate.candidate_id,
                        "reason": "ATTRIBUTE_FACT_ALREADY_PUBLIC",
                    },
                )
                continue
            if self.config.tasks.fact_novelty_enabled and (
                (requested := requested_fact_key(candidate)) is not None
                and requested in answered_facts
            ):
                self.store.write_json_artifact(
                    "candidate-admission-rejections",
                    {
                        "conversation_id": conversation_id,
                        "turn_index": turn_index,
                        "candidate_id": candidate.candidate_id,
                        "reason": "REQUESTED_FACT_ALREADY_PUBLIC",
                        "fact_key": asdict(requested),
                    },
                )
                continue
            target = next(
                (item.value for item in candidate.public_parameters if item.name == "target"),
                None,
            )
            same_scope_messages = tuple(
                message
                for turn in prior_turns
                if turn.status == "COMMITTED"
                if turn.instruction.scope_id == candidate.scope_id
                for message in (turn.question, turn.answer)
            )
            if (
                candidate.task_id == "object_identification"
                and isinstance(target, str)
                and identification_answer_in_history(target, same_scope_messages)
            ):
                self.store.write_json_artifact(
                    "candidate-admission-rejections",
                    {
                        "conversation_id": conversation_id,
                        "turn_index": turn_index,
                        "candidate_id": candidate.candidate_id,
                        "reason": "IDENTIFICATION_ANSWER_ALREADY_PUBLIC",
                    },
                )
                continue
            retained.append(candidate)
        return tuple(retained)

    @staticmethod
    def _direct_identification_stated_color(
        fact_key: tuple[str, str, str, str], prior_turns: Sequence[TurnArtifact]
    ) -> bool:
        """Recognize a color already stated in a direct, committed naming question.

        This intentionally handles only a narrow public pattern. A color mentioned for a nearby
        object, or a part-specific request such as nose color, must remain eligible.
        """
        view_id, scope_id, target, attribute = fact_key
        if attribute != "color":
            return False

        def normalized_name(value: str) -> str:
            words = re.findall(r"\w+", value.casefold())
            while words and words[0] in {"a", "an", "the"}:
                words.pop(0)
            return " ".join(words)

        return any(
            turn.status == "COMMITTED"
            and turn.instruction.task_id == "object_identification"
            and turn.instruction.scope_id == scope_id
            and (turn.instruction.view_id or "") == view_id
            and normalized_name(turn.answer.content) == target
            and _DIRECT_IDENTIFICATION_COLOR.search(turn.question.content) is not None
            for turn in prior_turns
        )

    @staticmethod
    def _answer_disclosure_reason(
        instruction: InstructionCandidate,
        question: str,
        answer: str,
        prior_turns: Sequence[TurnArtifact],
    ) -> PublicTextRejectionReason | None:
        """Find answer text or a leading action already disclosed to the reader."""
        same_scope_messages = tuple(
            message
            for turn in prior_turns
            if turn.instruction.scope_id == instruction.scope_id
            for message in (turn.question, turn.answer)
        )
        if instruction.task_id == "object_identification":
            if identification_answer_in_question(question, answer):
                return "IDENTIFICATION_ANSWER_IN_QUESTION"
            if identification_answer_in_history(answer, same_scope_messages):
                return "IDENTIFICATION_ANSWER_ALREADY_PUBLIC"
        elif instruction.task_id == "ui_element_grounding":
            target = next(
                (item.value for item in instruction.public_parameters if item.name == "target"),
                None,
            )
            if isinstance(target, str) and (
                " ".join(answer.casefold().split()).rstrip(".")
                == " ".join(target.casefold().split()).rstrip(".")
            ):
                return "UI_LOCATION_REPEATS_TARGET"
        if instruction.task_id == "text_transcription" and transcription_answer_in_question(
            question, answer
        ):
            return "TRANSCRIPTION_ANSWER_IN_QUESTION"
        if instruction.task_id == "visible_action_relation":
            same_scope_texts = tuple(message.content for message in same_scope_messages)
            if action_answer_already_public(question, answer, same_scope_texts):
                return "ACTION_ALREADY_PUBLIC"
        return None

    def _record_public_text_rejection(
        self,
        conversation_id: str,
        turn_index: int,
        *,
        field: Literal["question", "answer"],
        reason: PublicTextRejectionReason,
        content: str,
    ) -> None:
        """Store a deterministic public-text rejection for replay and diagnosis."""
        self.store.write_json_artifact(
            "public-text-rejections",
            {
                "conversation_id": conversation_id,
                "turn_index": turn_index,
                "field": field,
                "reason": reason,
                "content": content,
            },
        )

    def _check_complete_set(
        self,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
    ) -> SetCheck | None:
        inventories = [
            self._invoke(
                client,
                "set_inventory",
                {
                    "target_language": language,
                    "public_history": self._history(history),
                    "question": question.content,
                    "candidate_answer": answer.content,
                    "image_views": image_views,
                },
                (model_image,),
                SetInventory,
                max_tokens=2048,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                trial_id=f"blind-judge:{judge_index}",
            )
            for judge_index, client in enumerate(self.generators.values())
        ]
        return verify_set_inventories(inventories)

    def _check_computation(
        self,
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        turn_index: int,
    ) -> ComputationCheck:
        inventories = [
            self._invoke(
                client,
                "computation_inventory",
                {
                    "target_language": language,
                    "public_history": self._history(history),
                    "question": question.content,
                    "candidate_answer": answer.content,
                    "image_views": image_views,
                },
                (model_image,),
                ComputationInventory,
                max_tokens=1024,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                trial_id=f"blind-judge:{judge_index}",
            )
            for judge_index, client in enumerate(self.generators.values())
        ]
        return verify_computation_inventories(inventories)

    @staticmethod
    def _answer_tokens(text: str) -> list[dict[str, Any]]:
        """Expose stable token boundaries without requiring copied text or character counting."""
        return [
            {"index": index, "text": match.group(), "start": match.start(), "end": match.end()}
            for index, match in enumerate(re.finditer(r"\w+|[^\w\s]", text))
        ]

    @staticmethod
    def _bind_claim_span(span: ClaimSpan, answer: PublicMessage) -> AtomicClaim:
        """Bind validated token boundaries to exact source bytes and a controller-owned ID."""
        tokens = SynthesisCoordinator._answer_tokens(answer.content)
        if not 0 <= span.start_token < span.end_token <= len(tokens):
            raise ExecutionError("EXTRACTION_SOURCE_INVALID", "Claim token boundaries are invalid")
        start = tokens[span.start_token]["start"]
        end = tokens[span.end_token - 1]["end"]
        return AtomicClaim(
            text=answer.content[start:end],
            start=start,
            end=end,
            source_message_id=answer.message_id,
        )

    @staticmethod
    def _claim_union(
        inventories: Sequence[ClaimInventory],
        answer: PublicMessage,
    ) -> tuple[AtomicClaim, ...] | None:
        claims: dict[tuple[int, int, str], AtomicClaim] = {}
        for inventory in inventories:
            for claim in inventory.claims:
                normalized = SynthesisCoordinator._normalize_claim(claim, answer)
                if normalized is None:
                    return None
                claims[(normalized.start, normalized.end, normalized.text)] = normalized
        return tuple(claims[key] for key in sorted(claims))

    @staticmethod
    def _normalize_claim(claim: AtomicClaim, answer: PublicMessage) -> AtomicClaim | None:
        """Correct an invalid offset only when the quoted answer span is unique."""
        if claim.source_message_id != answer.message_id:
            return None
        if (
            claim.end <= len(answer.content)
            and answer.content[claim.start : claim.end] == claim.text
        ):
            return claim
        starts = [
            index
            for index in range(len(answer.content))
            if answer.content.startswith(claim.text, index)
        ]
        if len(starts) != 1:
            return None
        start = starts[0]
        return claim.model_copy(update={"start": start, "end": start + len(claim.text)})

    @staticmethod
    def _rubric_payload(
        template: Mapping[str, Any],
        history: Sequence[PublicMessage],
        question: PublicMessage,
        answer: PublicMessage,
        language: str,
        image_views: list[dict[str, str]],
        model_image: ModelImage,
        subject: AtomicClaim | Requirement | None,
        rule_inputs: Mapping[str, Any],
    ) -> tuple[dict[str, Any], tuple[ModelImage, ...]]:
        scope = template["input_scope"]
        payload: dict[str, Any] = {
            "target_language": language,
            "public_history": SynthesisCoordinator._history(history),
            "question": question.content,
            "criterion": {
                "template_id": template["template_id"],
                "question": template["question"],
                "met_anchor": template["met_anchor"],
                "not_met_anchor": template["not_met_anchor"],
            },
        }
        if scope not in {"TEXT_HQ", "VISION_IHQ"}:
            payload["candidate_answer"] = answer.content
        images: tuple[ModelImage, ...] = ()
        if scope.startswith("VISION") or scope.startswith("CLAIM"):
            payload["image_views"] = image_views
            images = (model_image,)
        if scope == "CLAIM_IHQA" and isinstance(subject, AtomicClaim):
            payload["target_claim"] = subject.model_dump(mode="json")
        if isinstance(subject, Requirement):
            payload["target_requirement"] = subject.model_dump(mode="json")
        if scope == "RULE":
            payload["typed_rule_inputs"] = (
                {"requirement": subject.model_dump(mode="json")}
                if isinstance(subject, Requirement)
                else dict(rule_inputs)
            )
        return payload, images

    def _invoke_judges(
        self,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        model: type[OutputModel],
        *,
        max_tokens: int,
        seed: int,
        trial_suffix: str = "",
    ) -> list[OutputModel]:
        """Ask both blind judges for one stage with separate trial identities.

        Votes keep the configured judge order. Neither judge receives the other's response.
        """
        return [
            self._invoke(
                client,
                stage,
                payload,
                images,
                model,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=seed,
                trial_id=f"blind-judge:{index}{trial_suffix}",
            )
            for index, client in enumerate(self.generators.values())
        ]

    def _invoke(
        self,
        client: InferenceClient,
        stage: str,
        payload: dict[str, Any],
        images: tuple[ModelImage, ...],
        model: type[OutputModel],
        *,
        post_validate: Callable[[OutputModel], None] | None = None,
        **kwargs: Any,
    ) -> OutputModel:
        retryable = {
            "MODEL_CONTENT_EMPTY",
            "MODEL_FINISH_REASON",
            "MODEL_WHITESPACE_RUNAWAY",
            "MODEL_SCHEMA_MISMATCH",
            "EVIDENCE_IMAGE_MISMATCH",
            "EXTRACTION_SOURCE_INVALID",
            "EVIDENCE_CAPABILITY_UNKNOWN",
            "EVIDENCE_SCOPE_LIMIT",
            "EVIDENCE_VIEW_MISMATCH",
            "EVIDENCE_OBSERVATION_LIMIT",
            "EVIDENCE_ATTRIBUTE_SCOPE",
            "CANDIDATE_BINDING_ID",
            "CANDIDATE_EVIDENCE_SCOPE",
            "CANDIDATE_CHECKS_MISMATCH",
            "CANDIDATE_PARAMETER_UNKNOWN",
            "CANDIDATE_PARAMETER_MISSING",
            "CANDIDATE_PARAMETER_SOURCE",
            "CANDIDATE_PARAMETER_VALUE",
            "REPEATED_PUBLIC_QUESTION",
            "INTERNAL_REFERENCE_IN_QUESTION",
            "IDENTIFICATION_TARGET_IN_QUESTION",
            "TEXT_RELATION_UNVERIFIED",
            "DRAFT_CONTRACT_INVALID",
        }
        retry_feedback: str | None = None
        detection = self.config.runtime.repetition_detection
        if detection is not None and stage in detection.stages and detection.recovery == "retry":
            retryable.add("MODEL_OUTPUT_REPETITION")
        prior_error: str | None = None
        for attempt in range(self.config.runtime.structured_output_max_attempts):
            call_kwargs = dict(kwargs)
            if attempt:
                call_kwargs["seed"] = int(call_kwargs["seed"]) + 100_000 * attempt
                call_kwargs["retry_feedback"] = retry_feedback
                if (
                    stage in {"evidence_extraction", "chart_source", "graph_source"}
                    and prior_error == "MODEL_FINISH_REASON"
                ):
                    original_tokens = int(kwargs["max_tokens"])
                    call_kwargs["max_tokens"] = max(original_tokens, min(original_tokens * 2, 8192))
            response: ModelResponse | None = None
            try:
                response = client.invoke(stage, payload, images, model, **call_kwargs)
                if (
                    isinstance(
                        response.value,
                        (
                            EvidenceInventory,
                            ScopedEvidenceInventory,
                            ScopedEvidenceReport,
                            ArrayScopedEvidenceReport,
                        ),
                    )
                    and response.value.image_id != payload["image_id"]
                ):
                    raise ExecutionError(
                        "EVIDENCE_IMAGE_MISMATCH", "Evidence refers to another image"
                    )
                if isinstance(response.value, ClaimExtraction):
                    answer = PublicMessage(
                        message_id="answer",
                        turn_index=1,
                        role="assistant",
                        content=payload["candidate_answer"],
                    )
                    for claim in response.value.claims:
                        self._bind_claim_span(claim, answer)
                if isinstance(response.value, RequirementInventory):
                    messages = [
                        PublicMessage.model_validate(value) for value in payload["public_history"]
                    ]
                    messages.append(
                        PublicMessage(
                            message_id=payload["question_message_id"],
                            turn_index=1,
                            role="user",
                            content=payload["question"],
                        )
                    )
                    if any(
                        _normalize_spec(spec, messages) is None
                        for spec in response.value.requirements
                    ):
                        raise ExecutionError(
                            "EXTRACTION_SOURCE_INVALID", "Requirement quote or reference is invalid"
                        )
                if not isinstance(response.value, model):
                    raise ExecutionError(
                        "MODEL_TYPE_MISMATCH", f"{stage} returned another contract"
                    )
                if post_validate is not None:
                    post_validate(response.value)
            except ExecutionError as error:
                can_retry = (
                    error.reason in retryable
                    and attempt + 1 < self.config.runtime.structured_output_max_attempts
                )
                if can_retry:
                    retry_feedback = self._structured_retry_feedback(error.reason)
                    if stage == "table_source" and error.reason == "MODEL_FINISH_REASON":
                        retry_feedback += (
                            " Keep the original output budget. If the complete table cannot fit,"
                            " return coverage=UNKNOWN, tables=[] and the public query; never"
                            " declare a partial grid closed."
                        )
                    if error.reason == "MODEL_SCHEMA_MISMATCH":
                        required = model.model_json_schema().get("required", [])
                        retry_feedback += " Required top-level fields: " + ", ".join(required) + "."
                        if stage == "evidence_proposal":
                            retry_feedback = (
                                "The previous evidence proposal violated its contract. Return only "
                                "the required top-level scopes field, without verdicts, IDs or reason. "
                                "Each capability name appears at most once per scope: combine its "
                                "visible facts into one short detail. Set object_label=null unless "
                                "the same scope has a visible_entity claim naming that subject. "
                                "Use positive normalized boxes. Every observation box must be inside "
                                "its parent scope; reuse the scope box only when the claim really "
                                "covers that scope. Omit unsupported claims, without inventing geometry."
                            )
                        if stage == "evidence_extraction":
                            retry_feedback += (
                                (
                                    " observations must be an array with a capability name on each observation."
                                    if model
                                    in {ArrayScopedEvidenceReport, CompactScopedEvidenceReport}
                                    else " observations must be an object keyed by each capability name."
                                )
                                + " Combine visible instances in one observation per capability and use"
                                " only capability_vocabulary. Every scope and observation region"
                                " must satisfy 0 <= left < right <= 1 and"
                                " 0 <= top < bottom <= 1. Keep each observation inside its scope;"
                                " never use a point or an all-1 box."
                            )
                            if response is not None and isinstance(
                                response.value,
                                (
                                    ScopedEvidenceReport,
                                    ArrayScopedEvidenceReport,
                                    CompactScopedEvidenceReport,
                                ),
                            ):
                                retry_feedback += out_of_scope_region_feedback(response.value)
                                retry_feedback += unsupported_object_label_feedback(response.value)
                        elif stage == "candidate_binding":
                            retry_feedback += (
                                " Every binding needs the separate target object; do not repeat"
                                " target in public_parameters. Include only operation-relevant"
                                " parameters, use each name once, and"
                                " match origin with evidence_refs."
                            )
                            retry_feedback += (
                                " Required fields in each binding: "
                                + ", ".join(CandidateBindingReport.model_json_schema()["required"])
                                + ". estimated_answer_tokens must be a positive integer."
                            )
                            if "Factual parameters require explicit evidence references" in str(
                                error
                            ):
                                retry_feedback += (
                                    " A public instruction choice has origin=instruction and no"
                                    " evidence_refs. An image or history fact needs matching"
                                    " evidence_refs. Omit names outside this candidate's"
                                    " bindable_parameter_names."
                                )
                        elif stage == "table_lookup_source":
                            retry_feedback += (
                                " Extract every row/column label as a string, preserving blank"
                                " labels. Include row_anchor and col_anchor for the selected"
                                " indexed headers. Their tight text boxes must overlap the"
                                " cell vertically and horizontally, respectively. Do not"
                                " invent extra fields or coordinates for every row. Use UNKNOWN"
                                " with empty arrays, null anchors and no cell when unclear."
                            )
                        elif stage == "table_source":
                            retry_feedback += (
                                " Every tables[] entry needs table_id, rows, cols, cells,"
                                " data_rows and closed. Include the separate query object with"
                                " operation and table_ids, even for UNKNOWN coverage. Each cell"
                                " region needs 0 <= left < right <= 1 and"
                                " 0 <= top < bottom <= 1 inside scope_region. For UNKNOWN use"
                                " tables=[]; never invent missing cells or set closed=true for"
                                " an incomplete grid."
                            )
                        elif stage == "formula_source":
                            retry_feedback += (
                                " Repair the FormulaNode tree arity: symbol and number have zero"
                                " children and a nonempty value; group, negative and sqrt have"
                                " one child; add, subtract, multiply, slash, equal and fraction"
                                " have two; script has three. Operator nodes use value=null."
                                " If the visual parse is uncertain, return coverage=UNKNOWN,"
                                " root=null and formula_region=null rather than guessing."
                            )
                        elif stage == "chart_source":
                            retry_feedback += (
                                " Exact printed values require numeric lower=upper and"
                                " visible_label containing that literal number. If a requested"
                                " value is unreadable, use coverage=UNKNOWN, axis=null, marks=[],"
                                " closed=false, and still include the public query. Never use"
                                " N/A or unknown as a numeric endpoint. All mark boxes must be"
                                " positive-extent rectangles inside scope_region."
                            )
                            retry_feedback += self._chart_region_retry_feedback(error)
                    elif error.reason == "EVIDENCE_IMAGE_MISMATCH":
                        retry_feedback = (
                            "The previous response referred to another image. Re-examine only the "
                            "attached image and regenerate the complete evidence object. "
                            f"Copy the input image_id exactly: {payload['image_id']}."
                        )
                    if error.reason.startswith("CANDIDATE_"):
                        retry_feedback += (
                            f" Correct this exact contract mismatch: {str(error)[:600]}."
                        )
                    if error.reason == "MODEL_SCHEMA_MISMATCH" and str(error).startswith(
                        "DUPLICATE_JSON_KEY"
                    ):
                        retry_feedback += (
                            f" Duplicate JSON key reported: {str(error)[:180]}."
                            " In observations, each capability is one key per scope."
                        )
                if error.reason in retryable:
                    history = payload.get("public_history")
                    self.store.write_json_artifact(
                        "structured-output-failures",
                        {
                            "stage": stage,
                            "view_ids": [image.view_id for image in images],
                            "question_message_id": payload.get("question_message_id"),
                            "turn_index": (
                                0
                                if stage == "evidence_extraction"
                                else len(history) // 2 + 1
                                if isinstance(history, list)
                                else payload.get("turn_index", 1)
                            ),
                            "attempt": attempt + 1,
                            "reason": error.reason,
                            "message": str(error)[:1000],
                            "response_hash": response.response_hash if response else None,
                            "parsed_output": (
                                response.value.model_dump(mode="json") if response else None
                            ),
                            "next_retry_feedback": retry_feedback if can_retry else None,
                        },
                    )
                if can_retry:
                    prior_error = error.reason
                    continue
                raise
            return response.value
        raise AssertionError("structured output attempt loop did not return")

    @staticmethod
    def _chart_region_retry_feedback(error: ExecutionError) -> str:
        """Name invalid chart regions without copying model values or repairing geometry."""
        cause = error.__cause__
        if not isinstance(cause, ValidationError):
            return ""
        indices = sorted(
            {
                loc[1]
                for item in cause.errors(
                    include_input=False, include_context=False, include_url=False
                )
                if len(loc := item["loc"]) >= 3
                and loc[0] == "marks"
                and isinstance(loc[1], int)
                and loc[2] == "region"
            }
        )[:8]
        if not indices:
            return ""
        paths = ", ".join(f"marks[{index}].region" for index in indices)
        return (
            f" Invalid region fields (zero-based indices): {paths}."
            " Re-read those visible marks in the attached image. Keep every rectangle"
            " positive in width and height and inside the public scope. Do not invent or"
            " automatically expand a box; if it cannot be grounded, return UNKNOWN."
        )

    @staticmethod
    def _structured_retry_feedback(reason: str) -> str:
        """Return bounded correction guidance without copying an invalid model response."""
        if reason == "EXTRACTION_SOURCE_INVALID":
            return (
                "An extracted quote or source reference was invalid. Copy exact substrings "
                "from the supplied source without changing quotes, punctuation or characters. "
                "Use only the supplied local message references. Never paraphrase a quoted span. "
                "For claims select existing answer_tokens indices, with exclusive end_token. "
                "For counts use categories grounded in the requested grouping and visible evidence."
            )
        if reason == "MODEL_SCHEMA_MISMATCH":
            return (
                "The previous response failed schema validation. Return one complete JSON object "
                "with every required field, unique array items, and a short non-empty reason."
            )
        if reason == "EVIDENCE_ATTRIBUTE_SCOPE":
            return (
                "The attribute region lies outside its same-scope visible_entity region. "
                "Keep the subject's entity box fixed. Report a property only inside that box; "
                "if it belongs to another subject or cannot be resolved, report UNKNOWN."
            )
        if reason.startswith("EVIDENCE_"):
            return (
                "Re-examine this image view. Use only names in capability_vocabulary,"
                " one observation per capability per scope, and the exact supplied view_id."
            )
        if reason.startswith("CANDIDATE_"):
            return (
                "Rebind only the supplied candidate IDs. Use only each candidate's"
                " parameter_contract, exact eligibility_checks and local evidence IDs."
                " Include all MET required_capabilities in binding evidence_refs."
                " A parameter with origin=image uses only local evidence IDs; origin=history"
                " uses only exact message_ids from public_history, never obs_* IDs."
            )
        if reason == "MODEL_FINISH_REASON":
            return (
                "The previous response was incomplete. Return a concise, complete JSON object "
                "within the token limit and finish immediately."
            )
        if reason == "MODEL_WHITESPACE_RUNAWAY":
            return (
                "The previous JSON response ended with excessive whitespace. Finish all required"
                " JSON fields, close the object, and stop immediately. Do not emit blank lines."
            )
        if reason == "MODEL_OUTPUT_REPETITION":
            return (
                "The engine stopped a repeated token pattern in the previous response. "
                "Re-examine the supplied input and return a complete concise JSON object "
                "within the original token limit. Use short reasons, no blank lines, and "
                "stop after closing the object. Never omit required judgments or guess "
                "missing evidence; use UNKNOWN when the image does not establish a fact."
            )
        if reason == "DRAFT_CONTRACT_INVALID":
            return (
                "Every draft violated its operation contract. Use only task_id values from"
                " allowed_tasks, include target and every required_parameter_names entry with a"
                " permitted parameter_contract value, use only bindable_parameter_names, and keep"
                " target_region inside scope_region."
            )
        if reason == "REPEATED_PUBLIC_QUESTION":
            return (
                "The draft repeats an answered question. Keep the selected task and ask for a"
                " visibly supported new target, attribute, or public condition."
            )
        if reason == "INTERNAL_REFERENCE_IN_QUESTION":
            return (
                "The draft contains an unresolved controller reference such as scope_0"
                " or selected region. Refer to the target by its visible location or traits;"
                " never copy scope_id, view_id, candidate_id, or evidence IDs into public text."
            )
        if reason == "IDENTIFICATION_TARGET_IN_QUESTION":
            return (
                "The draft identifies the object before asking its identity. Refer to its"
                " location or non-category visible traits without naming the target label."
            )
        if reason == "TEXT_RELATION_UNVERIFIED":
            return (
                "The transcript verifier has no coordinates for a relative locator."
                " Bound the text by a visible absolute region or exact public scope;"
                " do not claim that it is above, below, beside, or left/right of another object."
            )
        return "The previous response was empty. Return one concise, complete JSON object."

    @staticmethod
    def _history(history: Sequence[PublicMessage]) -> list[dict[str, Any]]:
        return [message.model_dump(mode="json") for message in history]

    @staticmethod
    def _view_metadata(image: ImageArtifact) -> dict[str, str]:
        return {
            "view_id": image.full_view.view_id,
            "encoded_sha256": image.full_view.encoded_sha256,
            "width": str(image.full_view.width),
            "height": str(image.full_view.height),
        }

    @staticmethod
    def _model_image(image: ImageArtifact, root: Path) -> ModelImage:
        return ModelImage(
            view_id=image.full_view.view_id,
            path=root / image.full_view.relative_path,
            encoded_sha256=image.full_view.encoded_sha256,
            media_type=image.full_view.media_type,
        )

    @staticmethod
    def _capability_vocabulary() -> dict[str, str]:
        catalog = load_task_catalog()
        return dict(sorted(catalog["capabilities"].items()))

    def _maybe_recheck_attribute(
        self,
        inventory: ScopedEvidenceInventory,
        generator: InferenceClient,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> ScopedEvidenceInventory:
        """Optionally ask about one missing visible property before any answer exists."""
        if not self.config.tasks.attribute_recheck_enabled:
            return inventory
        scope = next(
            (
                item
                for item in inventory.scopes
                if len(item.observations) < self.config.tasks.max_observations_per_scope
                and not any(obs.capability == "visible_attribute" for obs in item.observations)
                and all(
                    any(
                        obs.capability == capability and obs.verdict == "MET"
                        for obs in item.observations
                    )
                    for capability in ("visible_entity", "visible_interaction")
                )
            ),
            None,
        )
        if scope is None:
            return inventory
        subject_region = next(
            obs.region
            for obs in scope.observations
            if obs.capability == "visible_entity" and obs.verdict == "MET"
        )
        try:
            report = self._invoke(
                generator,
                "attribute_recheck",
                {
                    "image_id": inventory.image_id,
                    "scope_id": scope.scope_id,
                    "view_id": scope.view_id,
                    "scope_region": subject_region.model_dump(mode="json"),
                    "scope_description": scope.public_description,
                    "image_views": image_views,
                },
                (model_image,),
                AttributeRecheckReport,
                max_tokens=512,
                temperature=0.0,
                seed=self.config.seed + 17,
            )
            region = report.region
            parent = subject_region
            if (
                report.image_id != inventory.image_id
                or report.scope_id != scope.scope_id
                or report.view_id != scope.view_id
                or not (
                    parent.left <= region.left < region.right <= parent.right
                    and parent.top <= region.top < region.bottom <= parent.bottom
                )
            ):
                raise ExecutionError(
                    "EVIDENCE_RECHECK_SCOPE", "Attribute recheck refers outside its source scope"
                )
            used = {obs.evidence_id for item in inventory.scopes for obs in item.observations}
            index = 1
            while (evidence_id := f"attribute-recheck-{index}") in used:
                index += 1
            updated = ScopeEvidence(
                scope_id=scope.scope_id,
                view_id=scope.view_id,
                public_description=scope.public_description,
                object_label=scope.object_label,
                region=scope.region,
                observations=(
                    *scope.observations,
                    CapabilityObservation(
                        evidence_id=evidence_id,
                        capability="visible_attribute",
                        verdict=report.verdict,
                        region=region,
                        detail=report.detail,
                    ),
                ),
            )
            enriched = ScopedEvidenceInventory(
                image_id=inventory.image_id,
                scopes=tuple(
                    updated if item.scope_id == scope.scope_id else item
                    for item in inventory.scopes
                ),
                reason=inventory.reason,
            )
            validate_evidence(enriched, model_image.view_id, self.config.tasks)
            self.store.write_json_artifact(
                "evidence-attribute-rechecks",
                {
                    "image_id": inventory.image_id,
                    "scope_id": scope.scope_id,
                    "verdict": report.verdict,
                    "evidence_id": evidence_id,
                },
            )
            return enriched
        except ExecutionError as error:
            self.store.write_json_artifact(
                "evidence-attribute-rechecks",
                {
                    "image_id": inventory.image_id,
                    "scope_id": scope.scope_id,
                    "reason": error.reason,
                    "message": str(error)[:600],
                },
            )
            if error.reason not in {
                "MODEL_CONTENT_EMPTY",
                "MODEL_FINISH_REASON",
                "MODEL_WHITESPACE_RUNAWAY",
                "MODEL_OUTPUT_REPETITION",
                "MODEL_SCHEMA_MISMATCH",
                "EVIDENCE_RECHECK_SCOPE",
                "EVIDENCE_ATTRIBUTE_SCOPE",
            }:
                raise
            return inventory
