"""Turn-by-turn synthesis coordinator with explicit information barriers."""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, NamedTuple, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from pixelogue.catalog import task_catalog
from pixelogue.config import ModelEndpoint, PixelogueConfig
from pixelogue.contracts import (
    ConversationArtifact,
    GateVerdict,
    HistorySnapshot,
    ImageArtifact,
    InstructionCandidate,
    PublicMessage,
    RubricItem,
    RubricVerdict,
    TextPayload,
    TurnArtifact,
    TurnRating,
    build_history_snapshot,
)
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
    consensus,
    identification_answer_in_history,
    identification_answer_in_question,
    item_id,
    question_fingerprint,
    reciprocal_identification_disclosure,
    repeated_answered_request,
    repeated_public_question,
    scene_options_in_question,
    transcription_answer_in_question,
    unverified_transcription_relation,
)
from pixelogue.focused_views import focused_view
from pixelogue.gates import (
    GateDecision,
    QuestionGateVote,
    holistic_decision,
    question_gate_decision,
)
from pixelogue.planner import allocated_role, planned_turn_count
from pixelogue.prompts import is_private_prompt_echo
from pixelogue.routing import FamilyLedger, ImageProfile, Route, choose_route, family_definitions
from pixelogue.serialization import canonical_hash, canonical_json
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_runtime import operation_contract, unavailable_reasons
from pixelogue.task_verification import verify_operation

OutputModel = TypeVar("OutputModel", bound=BaseModel)
type PublicTextRejectionReason = Literal[
    "PRIVATE_PROMPT_ECHO",
    "INTERNAL_REFERENCE_IN_QUESTION",
    "REPEATED_PUBLIC_QUESTION",
    "REPEATED_ANSWERED_REQUEST",
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
# Gemma tends to write 0-1000 boxes or a parameter mapping; under constrained decoding those
# collapse to {1, 1, 1, 1} regions or whitespace runaways, so a retry restates both formats.
DRAFT_FORMAT_FEEDBACK = (
    " Write public_parameters as a JSON array of name and value objects ([] when none is"
    " required). Give every scope_region and target_region as fractions from 0 to 1 of the"
    " image width and height with left < right and top < bottom, never pixels or a 0-1000"
    " scale; target_region stays inside scope_region or is null."
)
_INTERNAL_QUESTION_REFERENCE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"(?:scope|view|candidate|evidence|obs)_[A-Za-z0-9_]+"
    r"|selected (?:image )?(?:region|scope|scene)"
    r")(?![A-Za-z0-9])",
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
        "EVIDENCE_VIEW_MISMATCH",
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
        router: InferenceClient,
        generator_a: InferenceClient,
        generator_b: InferenceClient,
    ) -> None:
        """Bind an immutable run configuration and its explicit model clients."""
        self.config = config
        self.run_id = run_id
        self.store = store
        self.router = router
        self.generators = {"generator_a": generator_a, "generator_b": generator_b}
        # Judge pairs and paired blind readers overlap; store writes stay serialized.
        self._judge_pool = ThreadPoolExecutor(
            max_workers=2 * config.runtime.max_concurrent_images,
            thread_name_prefix="pixelogue-judge",
        )
        # Drafting offers only core operations with working validators.
        self._draft_tasks = {
            task.id: task
            for task in task_catalog().tasks
            if task.status == "core" and not unavailable_reasons(task, config.tasks, config.models)
        }
        self.family_ledger = FamilyLedger(store.committed_task_ids())

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
        with self.store.transaction() as connection:
            saved = connection.execute(
                "SELECT artifact_hash FROM conversation_commit WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if saved is not None:
            completed = ConversationArtifact.model_validate_json(self.store.read_artifact(saved[0]))
            if completed.image != image or completed.generation_model != generator.endpoint.repo_id:
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
        terminal_status, terminal_stage, terminal_reason, terminal_turn_index = self._run_turns(
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

    def _run_turns(
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
            outcome = self._attempt_turn(
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
                self.router,
                "image_profile",
                {
                    "image_views": image_views,
                    "family_definitions": family_definitions(self._draft_tasks),
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
                tasks=self._draft_tasks,
                family_targets=self.config.tasks.family_targets,
                task_weights=self.config.tasks.task_weights,
                used_families=frozenset(turn.instruction.family for turn in committed)
                | excluded_families,
                used_task_ids=frozenset(turn.instruction.task_id for turn in committed),
                seed=self.config.seed,
                image_id=image.image_id,
                turn_index=snapshot.turn_index,
                light_only=snapshot.turn_index <= self.config.tasks.anchor_turns,
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

    def _attempt_turn(
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
                decision, candidate = self._question_gate(
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
                        draft_task_contract(self._draft_tasks[task_id])
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
        elif candidate.task_id == "visible_action" and action_affordance_question(text):
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

    def _question_gate(
        self,
        candidate: InstructionCandidate,
        draft: QuestionDraft | None,
        image: ImageArtifact,
        snapshot: HistorySnapshot,
        question: PublicMessage,
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
    ) -> tuple[GateDecision, InstructionCandidate]:
        """Run both blind question judges once and apply the configured label policy.

        Without a draft, as when re-rating a saved turn, a neighboring label is never adopted.
        """
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
                        {"task_id": task.id, "definition": task.definition}
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
                if draft is None:
                    return False
                try:
                    draft_to_candidate(
                        draft,
                        image_id=image.image_id,
                        view_id=image.full_view.view_id,
                        turn_index=snapshot.turn_index,
                        draft_index=0,
                        allowed_task_ids=frozenset(self._draft_tasks),
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
            assert draft is not None
            relabeled = draft_to_candidate(
                draft,
                image_id=image.image_id,
                view_id=image.full_view.view_id,
                turn_index=snapshot.turn_index,
                draft_index=int(str(candidate.scope_id).rsplit(":", 1)[-1]),
                allowed_task_ids=frozenset(self._draft_tasks),
                task_id=decision.task_id,
            )
            candidate = relabeled
        self.store.write_json_artifact(
            "question-gate-decisions",
            {
                "question_message_id": question.message_id,
                "candidate_id": candidate.candidate_id,
                "drafted_task_id": draft.task_id if draft is not None else None,
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

    def _generate_answer(
        self,
        selected: InstructionCandidate,
        snapshot: HistorySnapshot,
        generator: InferenceClient,
        question: PublicMessage,
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
                # Always empty since the requirement ledger was retired; kept so that answer
                # requests stay byte-identical to the measured configuration.
                "active_requirements": [],
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
        target_language: str,
        model_image: ModelImage,
        image_views: list[dict[str, str]],
        previous_turns: tuple[TurnArtifact, ...] = (),
    ) -> TurnAttemptResult:
        """Rate one answered turn, repairing once when configured, and build its artifact."""
        conversation_id = snapshot.conversation_id
        turn_index = snapshot.turn_index
        objections: list[str] | None = [] if self.config.evaluation.repair_once else None
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
            selector_model=self.router.endpoint.repo_id,
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
                turn.instruction.task_id == "visible_action"
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
        if fit is GateVerdict.MET and turn.instruction.catalog_version is not None:
            decision, _ = self._question_gate(
                turn.instruction,
                None,
                conversation.image,
                snapshot,
                turn.question,
                conversation.target_language,
                model_image,
                image_views,
            )
            if decision.verdict is GateVerdict.ERROR:
                raise ExecutionError("QUESTION_GATE_ERROR", "Question gate execution failed")
            fit = decision.verdict
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
            )
        return TurnRating(items=(), aggregate="ABSTAIN" if fit is GateVerdict.UNKNOWN else "FAIL")

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
        *,
        objections: list[str] | None = None,
    ) -> TurnRating:
        """Require the holistic review and every applicable operation check."""
        if instruction.catalog_version is not None and instruction.view_id != model_image.view_id:
            raise ExecutionError(
                "EVIDENCE_VIEW_MISMATCH", "Operation refers to a different image view"
            )
        rating = self._rate_holistic(
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

        def read(stage: str, body: dict[str, Any], model: type[BaseModel], judge: int) -> BaseModel:
            client = tuple(self.generators.values())[judge]
            # Full context lets blind transcription readers detect a requested unit
            # extending beyond its bound rectangle; the controller checks containment.
            max_tokens = (
                self.config.tasks.source_max_tokens
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
                    "consensus_answer",
                }
                else (model_image,),
                model,
                max_tokens=max_tokens,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                trial_id=f"blind-judge:{judge}",
            )

        prefetched: dict[tuple[str, str, str], Future[BaseModel]] = {}

        def invoke(
            stage: str, body: dict[str, Any], model: type[BaseModel], judge: int
        ) -> BaseModel:
            # Validators ask judge 0 and then judge 1 for the same blind input; start the
            # second judge's identical request immediately so the two readings overlap.
            key = (stage, canonical_hash(body), model.__name__)
            if judge == 0:
                prefetched[key] = self._judge_pool.submit(read, stage, body, model, 1)
                return read(stage, body, model, 0)
            future = prefetched.pop(key, None) if judge == 1 else None
            return future.result() if future is not None else read(stage, body, model, judge)

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
        if persist:
            with self.store.transaction() as connection:
                connection.execute(
                    "INSERT INTO conversation_commit(conversation_id, artifact_hash) VALUES (?, ?)",
                    (conversation.conversation_id, artifact_hash),
                )
        return conversation

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
        elif instruction.task_id == "ui_element_location":
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
        if instruction.task_id == "visible_action":
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
        """Ask both blind judges for one stage concurrently with separate trial identities.

        Votes keep the configured judge order. Neither judge receives the other's response.
        """

        def ask(index: int, client: InferenceClient) -> OutputModel:
            return self._invoke(
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

        futures = [
            self._judge_pool.submit(ask, index, client)
            for index, client in enumerate(self.generators.values())
        ]
        return [future.result() for future in futures]

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
            "DRAFT_CONTRACT_INVALID",
        }
        retry_feedback: str | None = None
        detection = self.config.runtime.repetition_detection
        if detection is not None and detection.applies_to(stage) and detection.recovery == "retry":
            retryable.add("MODEL_OUTPUT_REPETITION")
        prior_error: str | None = None
        for attempt in range(self.config.runtime.structured_output_max_attempts):
            call_kwargs = dict(kwargs)
            if attempt:
                call_kwargs["seed"] = int(call_kwargs["seed"]) + 100_000 * attempt
                call_kwargs["retry_feedback"] = retry_feedback
                if (
                    stage in {"chart_source", "graph_source"}
                    and prior_error == "MODEL_FINISH_REASON"
                ):
                    original_tokens = int(kwargs["max_tokens"])
                    call_kwargs["max_tokens"] = max(original_tokens, min(original_tokens * 2, 8192))
            response: ModelResponse | None = None
            try:
                response = client.invoke(stage, payload, images, model, **call_kwargs)
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
                        if stage == "table_lookup_source":
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
                    if stage == "question_draft" and error.reason in {
                        "MODEL_SCHEMA_MISMATCH",
                        "MODEL_OUTPUT_REPETITION",
                    }:
                        retry_feedback += DRAFT_FORMAT_FEEDBACK
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
                                len(history) // 2 + 1
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
        if reason == "MODEL_SCHEMA_MISMATCH":
            return (
                "The previous response failed schema validation. Return one complete JSON object "
                "with every required field, unique array items, and a short non-empty reason."
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
