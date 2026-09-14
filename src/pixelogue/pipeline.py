"""Turn-by-turn synthesis coordinator with explicit information barriers."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel

from pixelogue.catalog import load_task_catalog
from pixelogue.config import ModelEndpoint, PixelogueConfig
from pixelogue.contracts import (
    AtomicClaim,
    ClaimInventory,
    ConversationArtifact,
    EvidenceInventory,
    GateVerdict,
    ImageArtifact,
    InstructionCandidate,
    InstructionSelection,
    PublicMessage,
    QuestionFit,
    RubricItem,
    RubricVerdict,
    TextPayload,
    TurnArtifact,
    TurnRating,
    build_history_snapshot,
)
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import (
    aggregate_rating,
    applicable_rubric_items,
    consensus,
    item_id,
    question_fit_consensus,
)
from pixelogue.ledger import Requirement, RequirementInventory, reconcile_inventories
from pixelogue.planner import allocated_role, instruction_candidates, planned_turn_count
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

OutputModel = TypeVar("OutputModel", bound=BaseModel)


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
        A rejected turn ends the conversation and no prefix is certified as a complete conversation.
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
            self.store.write_json_artifact("conversations", conversation.model_dump(mode="json"))
            return conversation
        inventory = self._invoke(
            generator,
            "evidence_extraction",
            {
                "image_id": image.image_id,
                "capability_vocabulary": self._capability_vocabulary(),
                "image_views": image_views,
            },
            (model_image,),
            EvidenceInventory,
            max_tokens=1024,
            temperature=0.0,
            seed=self.config.seed,
        )
        if inventory.image_id != image.image_id:
            raise ExecutionError("EVIDENCE_IMAGE_MISMATCH", "Evidence refers to another image")

        terminal_status: Literal["QUALITY_CANDIDATE", "REJECTED", "ABSTAINED", "ERROR"] = (
            "QUALITY_CANDIDATE"
        )
        for turn_index in range(len(turns) + 1, count + 1):
            snapshot = build_history_snapshot(conversation_id, turn_index, transcript)
            candidates = instruction_candidates(
                inventory,
                seed=self.config.seed,
                turn_index=turn_index,
            )
            if not candidates:
                terminal_status = "REJECTED"
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
                break
            question_payload = self._invoke(
                generator,
                "question_generation",
                {
                    "target_language": target_language,
                    "turn_index": turn_index,
                    "public_history": self._history(snapshot.public_history),
                    "selected_instruction": selected.model_dump(mode="json"),
                    "image_views": image_views,
                },
                (model_image,),
                TextPayload,
                max_tokens=256,
                temperature=0.7,
                seed=self.config.seed + turn_index,
            )
            if question_payload.status != "OK":
                terminal_status = "REJECTED"
                break
            assert question_payload.text is not None
            question = PublicMessage(
                message_id=f"{conversation_id}:q:{turn_index}",
                turn_index=turn_index,
                role="user",
                content=question_payload.text,
            )
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
                break
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
                break
            answer_payload = self._invoke(
                generator,
                "answer_generation",
                {
                    "target_language": target_language,
                    "public_history": self._history(snapshot.public_history),
                    "question": question.content,
                    "active_requirements": [
                        requirement.model_dump(mode="json") for requirement in requirements
                    ],
                    "image_views": image_views,
                },
                (model_image,),
                TextPayload,
                max_tokens=1024,
                temperature=0.0,
                seed=self.config.seed + turn_index,
            )
            if answer_payload.status != "OK":
                terminal_status = "REJECTED"
                break
            assert answer_payload.text is not None
            answer = PublicMessage(
                message_id=f"{conversation_id}:a:{turn_index}",
                turn_index=turn_index,
                role="assistant",
                content=answer_payload.text,
            )
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
            if rating.aggregate == "FAIL":
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
        conversation = ConversationArtifact(
            conversation_id=conversation_id,
            image=image,
            target_language=target_language,
            generation_model=generator.endpoint.repo_id,
            turns=tuple(turns),
            status=terminal_status,
        )
        self.store.write_json_artifact("conversations", conversation.model_dump(mode="json"))
        return conversation

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
        if len(rated_turns) != len(conversation.turns):
            terminal = "REJECTED"
        return conversation.model_copy(update={"turns": tuple(rated_turns), "status": terminal})

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
            max_tokens=1024,
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
        inventories = [
            self._invoke(
                client,
                "requirement_extraction",
                {
                    "target_language": language,
                    "public_history": self._history(history),
                    "question": question.content,
                    "question_message_id": question.message_id,
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
        coverage = consensus([GateVerdict(item.coverage) for item in inventories])
        if coverage is not GateVerdict.MET:
            return coverage, ()
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
                "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
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
        """Persist an image-scoped inference failure without certifying a partial turn."""
        conversation_id = canonical_hash(
            {"run": self.run_id, "image": image.image_id, "language": target_language}
        )
        generator = self.generators[generator_role]
        turns = tuple(
            TurnArtifact.model_validate_json(self.store.read_artifact(artifact_hash))
            for artifact_hash in self.store.committed_artifact_hashes(conversation_id)
        )
        conversation = ConversationArtifact(
            conversation_id=conversation_id,
            image=image,
            target_language=target_language,
            generation_model=generator.endpoint.repo_id,
            turns=turns,
            status="ERROR",
        )
        self.store.write_json_artifact(
            "errors",
            {
                "conversation_id": conversation_id,
                "image_id": image.image_id,
                "reason": error.reason,
                "message": str(error),
            },
        )
        self.store.write_json_artifact("conversations", conversation.model_dump(mode="json"))
        return conversation

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
                    "selected_instruction": instruction.model_dump(mode="json"),
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
        inventories = [
            self._invoke(
                client,
                "claim_inventory",
                {
                    "target_language": language,
                    "public_history": self._history(history),
                    "question": question.content,
                    "candidate_answer": answer.content,
                    "candidate_answer_message_id": answer.message_id,
                    "image_views": image_views,
                },
                (model_image,),
                ClaimInventory,
                max_tokens=1024,
                temperature=0.0,
                seed=self.config.seed + turn_index,
                bypass_cache=judge_index > 0,
            )
            for judge_index, client in enumerate(self.generators.values())
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
            if instruction.family == "grounded_calculation"
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
            if instruction.family == "visible_count"
            else None
        )
        context = {
            "turn_index": turn_index,
            "has_natural_language_answer": True,
            "has_history_binding": turn_index > 1,
            "has_computation": instruction.family == "grounded_calculation",
            "profile": instruction.profile,
            "strong_dependency": turn_index > 1,
            "requirements": requirements,
            "format_requirements": tuple(
                requirement for requirement in requirements if requirement.kind == "format"
            ),
            "exhaustive_request": instruction.family == "visible_count",
            "claims": claims,
        }
        rubric_items: list[RubricItem] = []
        for template in applicable_rubric_items(context):
            predicate = template["applies_when"]
            subjects: Sequence[AtomicClaim | Requirement | None]
            if predicate == "each_factual_claim":
                subjects = claims
            elif predicate == "each_public_requirement":
                subjects = requirements
            elif predicate == "public_format_constraint":
                subjects = context["format_requirements"]
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
                        verdict = GateVerdict.UNKNOWN
                    elif set_check.verdict == "NOT_MET":
                        verdict = GateVerdict.NOT_MET
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
        **kwargs: Any,
    ) -> OutputModel:
        retryable = {
            "MODEL_CONTENT_EMPTY",
            "MODEL_FINISH_REASON",
            "MODEL_SCHEMA_MISMATCH",
        }
        for attempt in range(self.config.runtime.structured_output_max_attempts):
            call_kwargs = dict(kwargs)
            if attempt:
                call_kwargs["seed"] = int(call_kwargs["seed"]) + 100_000 * attempt
                call_kwargs["bypass_cache"] = True
            try:
                response = client.invoke(stage, payload, images, model, **call_kwargs)
            except ExecutionError as error:
                if (
                    error.reason in retryable
                    and attempt + 1 < self.config.runtime.structured_output_max_attempts
                ):
                    continue
                raise
            if not isinstance(response.value, model):
                raise ExecutionError("MODEL_TYPE_MISMATCH", f"{stage} returned another contract")
            return response.value
        raise AssertionError("structured output attempt loop did not return")

    @staticmethod
    def _history(history: Sequence[PublicMessage]) -> list[dict[str, Any]]:
        return [message.model_dump(mode="json") for message in history]

    @staticmethod
    def _view_metadata(image: ImageArtifact) -> dict[str, str]:
        return {
            "view_id": image.full_view.view_id,
            "encoded_sha256": image.full_view.encoded_sha256,
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
