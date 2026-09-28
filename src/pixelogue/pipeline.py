"""Turn-by-turn synthesis coordinator with explicit information barriers."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

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
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import (
    action_answer_already_public,
    aggregate_rating,
    applicable_rubric_items,
    consensus,
    has_natural_language_content,
    identification_answer_in_history,
    identification_answer_in_question,
    identification_label_in_question,
    item_id,
    question_fit_consensus,
    repeated_answered_request,
    repeated_public_question,
    scene_options_in_question,
    transcription_answer_in_question,
    unverified_transcription_relation,
)
from pixelogue.ledger import (
    Requirement,
    RequirementInventory,
    _normalize_spec,
    reconcile_inventories,
)
from pixelogue.planner import allocated_role, instruction_candidates, planned_turn_count
from pixelogue.prompts import is_private_prompt_echo
from pixelogue.rules import (
    ComputationCheck,
    ComputationInventory,
    SetCheck,
    SetInventory,
    verify_computation_inventories,
    verify_set_inventories,
)
from pixelogue.serialization import canonical_hash
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_evidence import (
    CandidateBindingsReport,
    ScopedEvidenceInventory,
    ScopedEvidenceReport,
)
from pixelogue.task_runtime import (
    bind_candidates,
    operation_contract,
    selector_candidate,
    validate_evidence,
)
from pixelogue.task_verification import verify_operation

OutputModel = TypeVar("OutputModel", bound=BaseModel)
type PublicTextRejectionReason = Literal[
    "PRIVATE_PROMPT_ECHO",
    "REPEATED_PUBLIC_QUESTION",
    "REPEATED_ANSWERED_REQUEST",
    "IDENTIFICATION_TARGET_IN_QUESTION",
    "IDENTIFICATION_ANSWER_IN_QUESTION",
    "IDENTIFICATION_ANSWER_ALREADY_PUBLIC",
    "TRANSCRIPTION_ANSWER_IN_QUESTION",
    "TEXT_RELATION_UNVERIFIED",
    "ACTION_ALREADY_PUBLIC",
    "CATEGORY_OPTIONS_NOT_PUBLIC",
]
MODEL_OUTPUT_ABSTENTIONS = frozenset(
    {
        "MODEL_CONTENT_EMPTY",
        "MODEL_FINISH_REASON",
        "MODEL_SCHEMA_MISMATCH",
        "EVIDENCE_IMAGE_MISMATCH",
        "EVIDENCE_CAPABILITY_UNKNOWN",
        "EVIDENCE_SCOPE_LIMIT",
        "EVIDENCE_VIEW_MISMATCH",
        "EVIDENCE_OBSERVATION_LIMIT",
        "EXTRACTION_SOURCE_INVALID",
    }
)


@dataclass(frozen=True)
class SynthesisJob:
    """One deterministically scheduled image synthesis job."""

    image: ImageArtifact
    target_language: Literal["en", "ja", "zh-Hans"]
    generator_role: Literal["generator_a", "generator_b"]


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
    ) -> None:
        """Bind an immutable run configuration and its explicit model clients."""
        self.config = config
        self.run_id = run_id
        self.store = store
        self.selector = selector
        self.generators = {"generator_a": generator_a, "generator_b": generator_b}

    def synthesize_batch(
        self,
        jobs: Iterable[SynthesisJob],
        artifact_root: Path,
        *,
        max_workers: int,
    ) -> Iterator[ConversationArtifact]:
        """Synthesize independent images concurrently and yield input order.

        At most ``max_workers`` jobs are submitted at once. Conversation-local history remains
        sequential, while independent model requests can be continuously batched by the server.

        Raises:
            ValueError: If ``max_workers`` is not positive.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        iterator = iter(jobs)
        if max_workers == 1:
            for job in iterator:
                yield self._synthesize_job(job, artifact_root)
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
        capability_vocabulary = self._capability_vocabulary()
        evidence_report = self._invoke(
            generator,
            "evidence_extraction",
            {
                "image_id": image.image_id,
                "capability_vocabulary": capability_vocabulary,
                "max_scopes": self.config.tasks.max_scopes,
                "max_observations_per_scope": self.config.tasks.max_observations_per_scope,
                "image_views": image_views,
            },
            (model_image,),
            ScopedEvidenceReport,
            max_tokens=self.config.tasks.evidence_max_tokens,
            temperature=0.0,
            seed=self.config.seed,
            post_validate=lambda result: validate_evidence(
                result.to_inventory(), model_image.view_id, self.config.tasks
            ),
        )
        inventory = evidence_report.to_inventory()
        if inventory.image_id != image.image_id:
            raise ExecutionError("EVIDENCE_IMAGE_MISMATCH", "Evidence refers to another image")

        terminal_status: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"] = (
            "QUALITY_CANDIDATE"
        )
        terminal_stage = "conversation_length"
        terminal_reason = "REQUESTED_TURN_COUNT_INCOMPLETE"
        for turn_index in range(len(turns) + 1, count + 1):
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

                def validate_bindings(
                    result: CandidateBindingsReport,
                    templates: tuple[InstructionCandidate, ...] = candidates,
                    public_history: tuple[PublicMessage, ...] = snapshot.public_history,
                ) -> None:
                    bind_candidates(
                        templates,
                        result.to_bindings(),
                        inventory,
                        public_history,
                        self.config.tasks,
                        frozenset(turn.instruction.candidate_id for turn in turns),
                    )

                try:
                    bindings_report = self._invoke(
                        generator,
                        "candidate_binding",
                        {
                            "target_language": target_language,
                            "public_history": self._history(snapshot.public_history),
                            "candidates": [
                                selector_candidate(candidate) for candidate in candidates
                            ],
                            "scope_evidence": inventory.model_dump(mode="json"),
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
                except ExecutionError as error:
                    if error.reason not in {
                        "MODEL_CONTENT_EMPTY",
                        "MODEL_FINISH_REASON",
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
                bindings = bindings_report.to_bindings()
                self.store.write_json_artifact(
                    "candidate-bindings", bindings.model_dump(mode="json")
                )
                candidates = bind_candidates(
                    candidates,
                    bindings,
                    inventory,
                    snapshot.public_history,
                    self.config.tasks,
                    frozenset(turn.instruction.candidate_id for turn in turns),
                )
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
                current_turn: int = turn_index,
            ) -> None:
                if result.text is None:
                    return
                if repeated_public_question(result.text, history):
                    reason = "REPEATED_PUBLIC_QUESTION"
                elif (
                    selected_instruction.task_id == "object_identification"
                    and isinstance(target, str)
                    and identification_label_in_question(result.text, target)
                ):
                    reason = "IDENTIFICATION_TARGET_IN_QUESTION"
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
                raise ExecutionError(
                    reason, "Question draft discloses or repeats its requested fact"
                )

            try:
                question_payload = self._invoke(
                    generator,
                    "question_generation",
                    {
                        "target_language": target_language,
                        "turn_index": turn_index,
                        "public_history": self._history(snapshot.public_history),
                        "selected_instruction": operation_contract(selected),
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
                    "REPEATED_PUBLIC_QUESTION",
                    "IDENTIFICATION_TARGET_IN_QUESTION",
                    "TEXT_RELATION_UNVERIFIED",
                    "CATEGORY_OPTIONS_NOT_PUBLIC",
                }:
                    raise
                terminal_status = "REJECTED"
                terminal_stage = "question_generation"
                terminal_reason = error.reason
                break
            if question_payload.status != "OK":
                terminal_status = "REJECTED"
                terminal_stage = "question_generation"
                terminal_reason = "QUESTION_NOT_GENERATED"
                break
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
                break
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
                break
            requirements: tuple[Requirement, ...] = ()
            if self.config.evaluation.mode == "detailed" or selected.catalog_version is not None:
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
                    break
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
                    break
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
                break
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
                break
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
                break
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
                break
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
            turns.append(turn)
            if turn_status != "COMMITTED":
                terminal_status = turn_status
                terminal_stage = "answer_verification"
                terminal_reason = f"RATING_{rating.aggregate}"
                break
            transcript = (*transcript, question, answer)
            artifact_hash = self.store.write_json_artifact("turns", turn.model_dump(mode="json"))
            with self.store.transaction() as connection:
                connection.execute(
                    """INSERT INTO turn_commit(
                           conversation_id, branch_id, turn_index, attempt, artifact_hash
                       ) VALUES (?, 'main', ?, 0, ?)""",
                    (conversation_id, turn_index, artifact_hash),
                )

        if len(turns) != count or any(turn.status != "COMMITTED" for turn in turns):
            if terminal_status == "QUALITY_CANDIDATE":
                terminal_status = "REJECTED"
        if terminal_status != "QUALITY_CANDIDATE":
            self.store.write_json_artifact(
                "conversation-stop-reasons",
                {
                    "conversation_id": conversation_id,
                    "turn_index": min(count, len(turns) + 1),
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

    def rate_existing(
        self,
        conversation: ConversationArtifact,
        artifact_root: Path,
    ) -> ConversationArtifact:
        """Re-evaluate immutable questions and answers with both configured judges."""
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
            fit = (
                GateVerdict.NOT_MET
                if repeated or disclosure or missing_scene_options or unverified_text_relation
                else GateVerdict.MET
            )
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
                rating = self._rate_turn(
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
            else:
                rating = TurnRating(
                    items=(),
                    aggregate="ABSTAIN" if fit is GateVerdict.UNKNOWN else "FAIL",
                )
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
        if self.config.evaluation.mode == "detailed" and len(rated_turns) != len(
            conversation.turns
        ):
            terminal = "REJECTED"
        rated = conversation.model_copy(update={"turns": tuple(rated_turns), "status": terminal})
        return self._finish_conversation(rated, persist=False)

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
                bypass_cache=judge_index > 0,
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
        votes = [
            self._invoke(
                client,
                "question_intent",
                public_payload,
                (model_image,),
                QuestionIntent,
                max_tokens=128,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                bypass_cache=judge_index > 0,
            )
            for judge_index, client in enumerate(self.generators.values())
        ]
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
        votes = [
            self._invoke(
                client,
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
                temperature=0.0,
                seed=self.config.seed + turn_index,
                bypass_cache=judge_index > 0,
            )
            for judge_index, client in enumerate(self.generators.values())
        ]
        return question_fit_consensus(votes)

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

        def invoke(
            stage: str, body: dict[str, Any], model: type[BaseModel], judge: int
        ) -> BaseModel:
            client = tuple(self.generators.values())[judge]
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
                max_tokens=2048,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                bypass_cache=judge > 0,
            )

        items = list(rating.items)
        for check in verify_operation(instruction, payload, invoke, model_image.path):
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
            )
        inventories = [
            self._invoke(
                client,
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
                temperature=0.0,
                seed=self.config.seed + turn_index,
                bypass_cache=judge_index > 0,
            )
            for judge_index, client in enumerate(self.generators.values())
        ]
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
                            bypass_cache=judge_index > 0,
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
    ) -> TurnRating:
        """Make two blind whole-turn judgments without intermediate semantic extraction."""
        invalid = (
            not question.content.strip()
            or not answer.content.strip()
            or is_private_prompt_echo(question.content)
            or is_private_prompt_echo(answer.content)
            or repeated_public_question(question.content, history)
        )
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
            if invalid
            else [
                self._invoke(
                    client,
                    "holistic_review",
                    payload,
                    (model_image,),
                    RubricVerdict,
                    max_tokens=384,
                    temperature=0.0,
                    seed=self.config.seed + turn_index,
                    bypass_cache=index > 0,
                )
                for index, client in enumerate(self.generators.values())
            ]
        )
        verdict = (
            GateVerdict.NOT_MET
            if invalid
            else consensus([GateVerdict(vote.verdict) for vote in votes])
        )
        reason = (
            "Empty, repeated, or private-prompt public text."
            if invalid
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
                "controller_reason": "INVALID_PUBLIC_TEXT" if invalid else None,
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
            actor="controller" if invalid else "dual-consensus",
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
                bypass_cache=judge_index > 0,
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
                bypass_cache=judge_index > 0,
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
            "MODEL_SCHEMA_MISMATCH",
            "EVIDENCE_IMAGE_MISMATCH",
            "EXTRACTION_SOURCE_INVALID",
            "EVIDENCE_CAPABILITY_UNKNOWN",
            "EVIDENCE_SCOPE_LIMIT",
            "EVIDENCE_VIEW_MISMATCH",
            "EVIDENCE_OBSERVATION_LIMIT",
            "CANDIDATE_BINDING_ID",
            "CANDIDATE_EVIDENCE_SCOPE",
            "CANDIDATE_CHECKS_MISMATCH",
            "CANDIDATE_PARAMETER_UNKNOWN",
            "CANDIDATE_PARAMETER_MISSING",
            "CANDIDATE_PARAMETER_SOURCE",
            "CANDIDATE_PARAMETER_VALUE",
            "REPEATED_PUBLIC_QUESTION",
            "IDENTIFICATION_TARGET_IN_QUESTION",
            "TEXT_RELATION_UNVERIFIED",
        }
        retry_feedback: str | None = None
        for attempt in range(self.config.runtime.structured_output_max_attempts):
            call_kwargs = dict(kwargs)
            if attempt:
                call_kwargs["seed"] = int(call_kwargs["seed"]) + 100_000 * attempt
                call_kwargs["bypass_cache"] = True
                call_kwargs["retry_feedback"] = retry_feedback
            response: ModelResponse | None = None
            try:
                response = client.invoke(stage, payload, images, model, **call_kwargs)
                if (
                    isinstance(
                        response.value,
                        (EvidenceInventory, ScopedEvidenceInventory, ScopedEvidenceReport),
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
                    if error.reason == "MODEL_SCHEMA_MISMATCH":
                        required = model.model_json_schema().get("required", [])
                        retry_feedback += " Required top-level fields: " + ", ".join(required) + "."
                        if stage == "evidence_extraction":
                            retry_feedback += (
                                " observations must be an object keyed by each capability name."
                                " Combine visible instances in that key's single value and use"
                                " only capability_vocabulary."
                            )
                        elif stage == "candidate_binding":
                            retry_feedback += (
                                " Every binding needs the separate target object; do not repeat"
                                " target in public_parameters. Include only operation-relevant"
                                " parameters, use each name once, and"
                                " match origin with evidence_refs."
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
                    continue
                raise
            return response.value
        raise AssertionError("structured output attempt loop did not return")

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
        if reason == "REPEATED_PUBLIC_QUESTION":
            return (
                "The draft repeats an answered question. Keep the selected task and ask for a"
                " visibly supported new target, attribute, or public condition."
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
