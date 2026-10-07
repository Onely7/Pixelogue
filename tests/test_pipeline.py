from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ValidationError

from pixelogue.chart_verifiers import ChartAnswer, ChartQuery, ChartSource
from pixelogue.config import EvaluationConfig, ModelEndpoint, load_config
from pixelogue.contracts import (
    GateVerdict,
    InstructionCandidate,
    PublicMessage,
    RubricVerdict,
    TextPayload,
    TurnRating,
    build_history_snapshot,
)
from pixelogue.drafting import DraftParameter, FactKey, QuestionDraft, QuestionDraftBatch
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import (
    action_affordance_question,
    action_answer_already_public,
    identification_answer_in_history,
    identification_answer_in_question,
    reciprocal_identification_disclosure,
    repeated_answered_request,
    scene_options_in_question,
    transcription_answer_in_question,
    unverified_transcription_relation,
)
from pixelogue.export import training_record
from pixelogue.finite_verifiers import FiniteSource
from pixelogue.formula_verifier import FormulaSource
from pixelogue.gates import GateDecision, QuestionGateVote
from pixelogue.graph_verifiers import GraphSource
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.prompts import STAGE_INSTRUCTIONS
from pixelogue.routing import ImageProfile
from pixelogue.rules import CountGroup, SetInventory, verify_set_inventories
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.table_lookup import TableLookupSource
from pixelogue.table_verifiers import TableSource
from pixelogue.task_evidence import ImageRegion, PublicParameter, TranscriptSource

FULL_REGION = ImageRegion(left=0, top=0, right=1, bottom=1)


class ScriptedClient:
    def __init__(
        self,
        endpoint: ModelEndpoint,
        *,
        reject_selection: bool = False,
        fail_first_rating: bool = False,
        concurrency_probe: ConcurrencyProbe | None = None,
        fail_first_schema: bool = False,
        repeat_question: bool = False,
        echo_question_prompt: bool = False,
        internal_reference_question: Literal["none", "first", "always", "selected_region"] = "none",
        echo_answer_prompt: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.reject_selection = reject_selection
        self.fail_first_rating = fail_first_rating
        self.concurrency_probe = concurrency_probe
        self.fail_first_schema = fail_first_schema
        self.repeat_question = repeat_question
        self.echo_question_prompt = echo_question_prompt
        self.internal_reference_question = internal_reference_question
        self.echo_answer_prompt = echo_answer_prompt
        self.rating_failed = False
        self.schema_failed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.retry_feedback: list[str | None] = []

    def _question(self, turn_index: int, draft_index: int) -> str:
        if self.echo_question_prompt:
            return STAGE_INSTRUCTIONS["question_draft"][:400]
        if self.internal_reference_question in {"always", "selected_region"} or (
            self.internal_reference_question == "first" and draft_index == 0
        ):
            return (
                "What color is the land in the selected region?"
                if self.internal_reference_question == "selected_region"
                else "What is the object at scope_0?"
            )
        if self.repeat_question:
            return (
                "What color is the visible region?"
                if turn_index == 1
                else "  WHAT COLOR IS THE VISIBLE REGION？  "
            )
        if draft_index:
            return f"What else is visible in region {turn_index}-{draft_index}?"
        return f"What color is visible region {turn_index}?"

    def _drafts(self, payload: dict[str, Any]) -> QuestionDraftBatch:
        if self.reject_selection:
            return QuestionDraftBatch(drafts=(), reason="No offered operation is supported.")
        turn = payload["turn_index"]
        offered = payload["preferred_task_ids"]
        preferred = ("object_identification", "attribute_lookup", "grounded_description")
        task = next((item for item in preferred if item in offered), offered[0])
        contract = next(row for row in payload["allowed_tasks"] if row["task_id"] == task)
        known: dict[str, Any] = {
            "attribute": "color",
            "category_set": ("indoor", "outdoor"),
            "detail_level": "brief",
            "frame": "viewer",
        }
        parameters = tuple(
            DraftParameter(name=name, value=known.get(name, "the blue region"))
            for name in contract["required_parameter_names"]
            if name != "target"
        )
        return QuestionDraftBatch(
            drafts=tuple(
                QuestionDraft(
                    task_id=task,
                    question=self._question(turn, index)
                    + (" Is it indoor or outdoor?" if task == "scene_categorization" else ""),
                    target=f"the blue region, aspect {turn}-{index}",
                    public_parameters=parameters,
                    scope_region=FULL_REGION,
                    target_region=FULL_REGION,
                    fact_key=FactKey(subject=f"blue region {turn}-{index}", dimension="color"),
                )
                for index in range(payload["draft_count"])
            )
        )

    def invoke[OutputModel: BaseModel](
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
        del images, max_tokens, temperature, seed, bypass_cache, trial_id
        self.retry_feedback.append(retry_feedback)
        if self.fail_first_schema and not self.schema_failed:
            self.schema_failed = True
            raise ExecutionError("MODEL_SCHEMA_MISMATCH", "missing required field")
        if self.concurrency_probe is not None:
            self.concurrency_probe.enter()
            time.sleep(0.002)
            self.concurrency_probe.exit()
        self.calls.append((stage, payload))
        value: BaseModel
        if response_model is ImageProfile:
            value = ImageProfile(
                image_kind="photo",
                readable_text="none",
                supported_families=("visual_description",),
                reason="A visible object is available.",
            )
        elif response_model is QuestionDraftBatch:
            value = self._drafts(payload)
        elif response_model is QuestionGateVote:
            value = QuestionGateVote(
                local_anchor="MET",
                operation_coherent="MET",
                useful_request="MET",
                reason="The question is visibly grounded.",
                realized_task_id=payload["selected_instruction"]["task_id"],
            )
        elif response_model in {ChartSource, GraphSource}:
            common = {
                "task_id": "chart_value_lookup"
                if response_model is ChartSource
                else "diagram_connectivity",
                "coverage": "UNKNOWN",
                "scope_id": "whole",
                "view_id": "view",
                "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                "closed": False,
                "reason": "The requested structure is unreadable.",
            }
            structure = (
                {
                    "axis": None,
                    "legend_complete": False,
                    "encoding": "bar",
                    "marks": (),
                    "query": {"operation": "value"},
                }
                if response_model is ChartSource
                else {
                    "junctions_resolved": False,
                    "nodes": (),
                    "edges": (),
                    "query": {"operation": "neighbors"},
                }
            )
            value = response_model.model_validate({**common, **structure})
        elif response_model is TableLookupSource:
            value = TableLookupSource(
                coverage="UNKNOWN",
                layout="unreadable",
                scope_id="whole",
                view_id="view",
                scope_region=ImageRegion(left=0, top=0, right=1, bottom=1),
                row_headers=(),
                col_headers=(),
                closed=False,
                reason="The table is not readable.",
            )
        elif response_model is TableSource:
            value = TableSource.model_validate(
                {
                    "task_id": "table_cell_lookup",
                    "coverage": "UNKNOWN",
                    "scope_id": "whole",
                    "view_id": "view",
                    "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "tables": (),
                    "query": {"operation": "lookup", "table_ids": ()},
                    "reason": "The table is not readable.",
                }
            )
        elif response_model is FormulaSource:
            value = FormulaSource.model_validate(
                {
                    "coverage": "UNKNOWN",
                    "scope_id": "whole",
                    "view_id": "view",
                    "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "notation": "latex",
                    "root": None,
                    "formula_region": None,
                    "reason": "The formula is not readable.",
                }
            )
        elif response_model is FiniteSource:
            value = FiniteSource.model_validate(
                {
                    "task_id": "count_comparison",
                    "coverage": "UNKNOWN",
                    "closed": False,
                    "scope_id": "whole",
                    "view_id": "view",
                    "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
                    "members": (),
                    "query": {"answer_form": "relation"},
                    "reason": "The objects are not countable.",
                }
            )
        elif response_model is TextPayload:
            if stage == "answer_generation" and self.echo_answer_prompt:
                text = STAGE_INSTRUCTIONS["answer_generation"]
            elif stage == "answer_generation":
                text = f"Blue aspect {len(payload['public_history'])}."
            else:
                text = "Blue."
            value = TextPayload(text=text)
        elif response_model is RubricVerdict:
            if self.fail_first_rating and not self.rating_failed:
                self.rating_failed = True
                value = RubricVerdict(verdict="NOT_MET", reason="Repair this answer.")
            else:
                value = RubricVerdict(verdict="MET", reason="The criterion is satisfied.")
        else:
            raise AssertionError(f"Unexpected response model: {response_model}")
        return ModelResponse(
            value=value,
            request_hash="1" * 64,
            response_hash="2" * 64,
            prompt_tokens=1,
            completion_tokens=1,
        )


def test_structured_retry_receives_bounded_correction_feedback(tmp_path: Path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a, fail_first_schema=True)
    coordinator = SynthesisCoordinator(config, "retry", store, client, client, client)
    try:
        result = coordinator._invoke(
            client,
            "rubric_item",
            {},
            (),
            RubricVerdict,
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.verdict == "MET"
    assert client.retry_feedback == [
        None,
        (
            "The previous response failed schema validation. Return one complete JSON object "
            "with every required field, unique array items, and a short non-empty reason."
            " Required top-level fields: verdict, reason."
        ),
    ]


@pytest.mark.parametrize(
    ("stage", "model", "expected"),
    [
        (
            "table_lookup_source",
            TableLookupSource,
            ("row_anchor", "col_anchor", "overlap", "null anchors"),
        ),
        ("table_source", TableSource, ("data_rows", "query object", "left < right")),
        ("formula_source", FormulaSource, ("zero children", "script has three", "root=null")),
        ("chart_source", ChartSource, ("lower=upper", "marks=[]", "N/A")),
    ],
)
def test_source_schema_retry_names_nested_contracts(
    tmp_path: Path,
    stage: str,
    model: type[TableLookupSource] | type[TableSource] | type[FormulaSource] | type[ChartSource],
    expected: tuple[str, ...],
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", stage, require_local_wal=False)
    client = ScriptedClient(config.models.generator_a, fail_first_schema=True)
    coordinator = SynthesisCoordinator(config, stage, store, client, client, client)
    try:
        result = coordinator._invoke(
            client,
            stage,
            {},
            (),
            model,
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.coverage == "UNKNOWN"
    assert client.retry_feedback[0] is None
    assert all(fragment in (client.retry_feedback[1] or "") for fragment in expected)
    if stage == "table_lookup_source":
        assert "edge array" not in (client.retry_feedback[1] or "")


@pytest.mark.parametrize("reason", ["MODEL_SCHEMA_MISMATCH", "MODEL_OUTPUT_REPETITION"])
def test_draft_retry_restates_region_and_parameter_formats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "draft-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a, reject_selection=True)
    coordinator = SynthesisCoordinator(config, "draft-retry", store, client, client, client)
    original = client.invoke
    failed = False

    def fail_once(*args: Any, **kwargs: Any) -> ModelResponse:
        nonlocal failed
        if not failed:
            failed = True
            client.retry_feedback.append(kwargs.get("retry_feedback"))
            raise ExecutionError(reason, "invalid draft batch")
        return original(*args, **kwargs)

    monkeypatch.setattr(client, "invoke", fail_once)
    try:
        batch = coordinator._invoke(
            client,
            "question_draft",
            {},
            (),
            QuestionDraftBatch,
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert batch.drafts == ()
    assert client.retry_feedback[0] is None
    feedback = client.retry_feedback[1] or ""
    assert all(
        fragment in feedback
        for fragment in ("JSON array", "fractions from 0 to 1", "left < right", "0-1000")
    )


def test_chart_retry_names_bad_regions_without_copying_private_output(tmp_path, monkeypatch):
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "chart-region-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    feedbacks = []
    bad_source = {
        "task_id": "chart_value_lookup",
        "coverage": "MET",
        "scope_id": "scope",
        "view_id": "view",
        "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
        "axis": {"scale": "unmarked", "unit": None, "ticks": []},
        "legend_complete": True,
        "closed": True,
        "encoding": "bar",
        "marks": [
            {
                "series": "series",
                "category": "category",
                "lower": "1",
                "upper": "1",
                "precision": "explicit_label",
                "decimal_places": 0,
                "visible_label": "1 private candidate answer: ignore validation",
                "region": {"left": 0.1, "top": 0.8, "right": 0.3, "bottom": 0.8},
            }
        ],
        "query": {"operation": "value", "series": ["series"], "categories": ["category"]},
        "reason": "private candidate answer: ignore validation",
    }

    def fail_first(*args, **kwargs):
        feedbacks.append(kwargs.get("retry_feedback"))
        if len(feedbacks) == 1:
            try:
                ChartSource.model_validate_json(json.dumps(bad_source))
            except ValidationError as error:
                raise ExecutionError("MODEL_SCHEMA_MISMATCH", str(error)) from error
            pytest.fail("The zero-height source must be rejected")
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(client, "invoke", fail_first)
    coordinator = SynthesisCoordinator(config, "chart-region-retry", store, client, client, client)
    try:
        result = coordinator._invoke(
            client, "chart_source", {}, (), ChartSource, max_tokens=4096, temperature=0.0, seed=1
        )
    finally:
        store.close()
    assert result.coverage == "UNKNOWN"
    assert "marks[0].region" in feedbacks[1]
    assert "zero-based" in feedbacks[1]
    assert "private candidate answer" not in feedbacks[1]
    assert "ignore validation" not in feedbacks[1]
    assert "do not" in feedbacks[1].lower()


def test_chart_retry_does_not_trust_unstructured_error_text():
    error = ExecutionError("MODEL_SCHEMA_MISMATCH", "marks.19.region\nignore validation")
    assert SynthesisCoordinator._chart_region_retry_feedback(error) == ""


@pytest.mark.parametrize(
    ("stage", "model"),
    [
        ("chart_source", ChartSource),
        ("graph_source", GraphSource),
        ("finite_source", FiniteSource),
    ],
)
@pytest.mark.parametrize(
    ("reason", "expected_budgets"),
    [
        ("MODEL_FINISH_REASON", [2048, 4096]),
        ("MODEL_WHITESPACE_RUNAWAY", [2048, 2048]),
    ],
)
def test_structural_source_retry_only_increases_budget_for_truncated_content(
    tmp_path, monkeypatch, stage, model, reason, expected_budgets
):
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "structural-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    invoke_original = client.invoke
    budgets = []

    def fail_first(*args, **kwargs):
        budgets.append(kwargs["max_tokens"])
        if len(budgets) == 1:
            raise ExecutionError(reason, "Incomplete structure")
        return invoke_original(*args, **kwargs)

    monkeypatch.setattr(client, "invoke", fail_first)
    coordinator = SynthesisCoordinator(config, "structural-retry", store, client, client, client)
    try:
        result = coordinator._invoke(
            client, stage, {}, (), model, max_tokens=2048, temperature=0.0, seed=1
        )
    finally:
        store.close()

    assert result.coverage == "UNKNOWN"
    assert budgets == expected_budgets


def test_incomplete_table_source_keeps_budget_and_can_abstain_without_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "table-length-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    budgets: list[int] = []
    feedbacks: list[str | None] = []

    def truncate_first(*args: Any, **kwargs: Any) -> ModelResponse:
        budgets.append(kwargs["max_tokens"])
        feedbacks.append(kwargs.get("retry_feedback"))
        if len(budgets) == 1:
            raise ExecutionError("MODEL_FINISH_REASON", "Completion reached its token limit")
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(client, "invoke", truncate_first)
    coordinator = SynthesisCoordinator(config, "table-length-retry", store, client, client, client)
    try:
        result = coordinator._invoke(
            client,
            "table_source",
            {},
            (),
            TableSource,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.coverage == "UNKNOWN"
    assert budgets == [4096, 4096]
    assert "complete table cannot fit" in (feedbacks[1] or "")


def test_whitespace_runaway_retries_source_without_doubling_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "source-whitespace-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    budgets: list[int] = []
    feedback: list[str | None] = []

    def whitespace_first(*args: Any, **kwargs: Any) -> ModelResponse:
        budgets.append(kwargs["max_tokens"])
        feedback.append(kwargs.get("retry_feedback"))
        if len(budgets) == 1:
            raise ExecutionError("MODEL_WHITESPACE_RUNAWAY", "Completion exhausted in whitespace")
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(client, "invoke", whitespace_first)
    coordinator = SynthesisCoordinator(
        config, "source-whitespace-retry", store, client, client, client
    )
    try:
        result = coordinator._invoke(
            client,
            "chart_source",
            {},
            (),
            ChartSource,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.coverage == "UNKNOWN"
    assert budgets == [4096, 4096]
    assert "Do not emit blank lines" in (feedback[1] or "")


def test_semantic_contract_failure_retries_without_accepting_invalid_result(
    tmp_path: Path,
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "semantic-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    coordinator = SynthesisCoordinator(config, "semantic-retry", store, client, client, client)
    attempts = 0

    def validate(_result: RubricVerdict) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ExecutionError("DRAFT_CONTRACT_INVALID", "DRAFT_PARAMETER_UNKNOWN")

    try:
        result = coordinator._invoke(
            client,
            "holistic_review",
            {},
            (),
            RubricVerdict,
            max_tokens=256,
            temperature=0.0,
            seed=1,
            post_validate=validate,
        )
        failure_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'structured-output-failures'"
        ).fetchone()[0]
        failure = json.loads(store.read_artifact(failure_hash))
    finally:
        store.close()

    assert result.verdict == "MET"
    assert attempts == 2
    assert client.retry_feedback[0] is None
    assert "parameter_contract" in (client.retry_feedback[1] or "")
    assert failure["reason"] == "DRAFT_CONTRACT_INVALID"
    assert failure["message"] == "DRAFT_PARAMETER_UNKNOWN"
    assert failure["parsed_output"] == {"verdict": "MET", "reason": "The criterion is satisfied."}
    assert "parameter_contract" in failure["next_retry_feedback"]


def test_answer_stop_records_the_failed_turn_instead_of_the_next_turn(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    monkeypatch.setattr(
        coordinator, "_rate_turn", lambda *args, **kwargs: TurnRating(items=(), aggregate="ABSTAIN")
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        stop_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'conversation-stop-reasons'"
        ).fetchone()[0]
        stop = json.loads(store.read_artifact(stop_hash))
    finally:
        store.close()
    assert conversation.status == "ABSTAINED"
    assert len(conversation.turns) == 1
    assert conversation.turns[0].status != "COMMITTED"
    assert stop["turn_index"] == 1


@pytest.mark.parametrize("reason", ["MODEL_SCHEMA_MISMATCH", "MODEL_OUTPUT_REPETITION"])
def test_repeated_invalid_model_json_abstains_but_transport_failure_remains_error(
    tmp_path: Path, image_artifact, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    original = coordinator._invoke

    def invoke(client, stage, payload, images, model, **kwargs):
        if stage == "question_gate":
            raise ExecutionError(reason, "Invalid model output")
        return original(client, stage, payload, images, model, **kwargs)

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        job = SynthesisJob(image=image, target_language="en", generator_role="generator_a")
        abstained = coordinator._synthesize_job(job, root)
        resumed = coordinator.synthesize_image(image, root, generator_role="generator_a")
        reason = "MODEL_HTTP_STATUS"
        failed = coordinator._synthesize_job(
            SynthesisJob(
                image=image.model_copy(update={"image_id": "f" * 64}),
                target_language="en",
                generator_role="generator_a",
            ),
            root,
        )
        abstention_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'model-output-abstentions'"
        ).fetchone()[0]
    finally:
        store.close()

    assert abstained.status == "ABSTAINED"
    assert not abstained.turns
    assert resumed == abstained
    assert failed.status == "ERROR"
    assert abstention_count == 1


@pytest.mark.parametrize("stage", ["_question_gate", "_rate_turn"])
@pytest.mark.parametrize(
    ("reason", "expected", "artifact_kind"),
    [
        ("MODEL_SCHEMA_MISMATCH", "ABSTAINED", "model-output-abstentions"),
        ("MODEL_HTTP_STATUS", "ERROR", "errors"),
    ],
)
def test_rerating_keeps_failures_with_their_turn_and_continues_other_conversations(
    tmp_path, image_artifact, monkeypatch, stage, reason, expected, artifact_kind
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 2)
    try:
        original = coordinator.synthesize_image(image, root)
        assert len(original.turns) == 2
        monkeypatch.setattr(coordinator, "_question_gate", _gate_met)
        monkeypatch.setattr(
            coordinator, "_rate_turn", lambda *args: TurnRating(items=(), aggregate="PASS")
        )
        evaluate = getattr(coordinator, stage)
        calls = 0

        def fail_once(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ExecutionError(reason, "Rejected evaluator output")
            return evaluate(*args)

        monkeypatch.setattr(coordinator, stage, fail_once)
        failed = coordinator.rate_existing(original, root)
        assert calls == 1
        sibling = coordinator.rate_existing(
            original.model_copy(update={"conversation_id": "other-conversation"}), root
        )
        row = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = ? ORDER BY rowid DESC LIMIT 1",
            (artifact_kind,),
        ).fetchone()
        failure = json.loads(store.read_artifact(row[0]))
    finally:
        store.close()

    assert failed.status == expected
    assert len(failed.turns) == 1
    assert failed.turns[0].status == expected
    assert failed.turns[0].question == original.turns[0].question
    assert failed.turns[0].answer == original.turns[0].answer
    assert sibling.status == "QUALITY_CANDIDATE"
    assert len(sibling.turns) == 2
    assert failure["conversation_id"] == original.conversation_id
    assert failure["turn_index"] == 1
    assert failure["reason"] == reason
    assert failure["operation"] == "rate-existing"


def _gate_met(candidate, *args):
    return GateDecision(GateVerdict.MET, candidate.task_id, "exact", GateVerdict.MET), candidate


class ConcurrencyProbe:
    """Track overlapping scripted model calls across test clients."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def enter(self) -> None:
        """Record the start of one model call."""
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def exit(self) -> None:
        """Record the completion of one model call."""
        with self._lock:
            self.active -= 1


def _coordinator(
    tmp_path: Path,
    reject_selection: bool = False,
    fail_first_rating: bool = False,
    concurrency_probe: ConcurrencyProbe | None = None,
    repeat_question: bool = False,
    echo_question_prompt: bool = False,
    internal_reference_question: Literal["none", "first", "always", "selected_region"] = "none",
    echo_answer_prompt: bool = False,
):
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "test", require_local_wal=False)
    store.initialize_run("test", config.config_hash, config.profile)
    selector = ScriptedClient(config.models.router, concurrency_probe=concurrency_probe)
    generators = [
        ScriptedClient(
            endpoint,
            reject_selection=reject_selection,
            fail_first_rating=fail_first_rating,
            concurrency_probe=concurrency_probe,
            repeat_question=repeat_question,
            echo_question_prompt=echo_question_prompt,
            internal_reference_question=internal_reference_question,
            echo_answer_prompt=echo_answer_prompt,
        )
        for endpoint in (config.models.generator_a, config.models.generator_b)
    ]
    coordinator = SynthesisCoordinator(config, "test", store, selector, *generators)
    return coordinator, store, selector, generators[0], generators[1]


def test_batch_synthesis_overlaps_images_and_preserves_input_order(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    probe = ConcurrencyProbe()
    coordinator, store, _, _, _ = _coordinator(tmp_path, concurrency_probe=probe)
    images = tuple(image.model_copy(update={"image_id": f"{index:064x}"}) for index in range(1, 5))
    jobs = tuple(
        SynthesisJob(
            image=item,
            target_language="en",
            generator_role="generator_a" if index % 2 else "generator_b",
        )
        for index, item in enumerate(images)
    )
    try:
        conversations = tuple(coordinator.synthesize_batch(jobs, root, max_workers=3))
        verification = store.verify()
    finally:
        store.close()

    assert tuple(item.image.image_id for item in conversations) == tuple(
        item.image_id for item in images
    )
    assert probe.peak >= 2
    assert verification["turn_commits"] == sum(
        len(item.turns) for item in conversations if item.status == "QUALITY_CANDIDATE"
    )


def test_batch_synthesis_rejects_workers_above_configured_limit(
    tmp_path: Path, image_artifact
) -> None:
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        with pytest.raises(ValueError, match="runtime.max_concurrent_images"):
            tuple(coordinator.synthesize_batch((), image_artifact[1], max_workers=5))
    finally:
        store.close()


def test_normalized_repeated_question_stops_before_second_gate(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        repeat_question=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        reasons = {
            json.loads(store.read_artifact(row[0]))["reason"]
            for row in store.connection.execute(
                "SELECT artifact_hash FROM artifact WHERE kind = 'public-text-rejections'"
            )
        }
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert len(conversation.turns) == 1
    assert reasons and reasons == {"REPEATED_PUBLIC_QUESTION"}
    assert sum(stage == "question_gate" for stage, _ in generator_a.calls) == 1
    assert sum(stage == "question_gate" for stage, _ in generator_b.calls) == 1


def test_identification_question_must_not_contain_its_answer():
    assert identification_answer_in_question("What object is this red strawberry?", "A strawberry.")
    assert identification_answer_in_question(
        "What is this silver vintage race car?", "silver race car"
    )
    assert not identification_answer_in_question("What insect is visible on the fabric?", "ladybug")
    assert identification_answer_in_question("What kind of animal is visible?", "animal")


def test_public_text_checks_catch_observed_multiturn_leaks() -> None:
    history = (
        PublicMessage(
            message_id="q1",
            turn_index=1,
            role="user",
            content="What action is the man performing with the striped knit hat?",
        ),
        PublicMessage(
            message_id="a1",
            turn_index=1,
            role="assistant",
            content="The man is looking left.",
        ),
    )
    assert identification_answer_in_history("knit hat", history)
    assert not identification_answer_in_history("sedan", history)
    assert identification_answer_in_history(
        "Three QR codes",
        (PublicMessage(message_id="a0", turn_index=1, role="assistant", content="QR code"),),
    )
    assert transcription_answer_in_question(
        'What does the text "PSEUDO COLOR 0243_2" read?', "PSEUDO COLOR\n0243_2"
    )
    assert not transcription_answer_in_question("What does the label read?", "PSEUDO COLOR")
    assert unverified_transcription_relation("What text is directly below the Roland logo?")
    assert unverified_transcription_relation("Read the word to the right of the logo.")
    assert not unverified_transcription_relation("What text is in the upper left corner?")
    assert action_answer_already_public(
        "What action is the gibbon displaying while seated?",
        "The gibbon is sitting upright on the railing.",
        (),
    )
    assert action_answer_already_public(
        "What action is the hand performing?",
        "The hand is drawing a sketch.",
        ("What object is being used to draw the sketch?",),
    )
    assert action_answer_already_public(
        "What is the beetle doing while resting on the fabric?",
        "The beetle is resting on the fabric.",
        (),
    )
    assert not action_answer_already_public(
        "What is the man doing near the wall?", "The man is painting.", ("The man is standing.",)
    )
    assert scene_options_in_question(
        "Is this a residential street or a rural road?", ("residential street", "rural road")
    )
    assert not scene_options_in_question(
        "What type of scene is this?", ("residential street", "rural road")
    )


@pytest.mark.parametrize("region_field", ["target_region", "scope_region"])
def test_local_question_judges_receive_only_a_hashed_focus_view(
    tmp_path, image_artifact, monkeypatch, region_field
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    coordinator.config = coordinator.config.model_copy(
        update={
            "evaluation": coordinator.config.evaluation.model_copy(update={"judge_views": "crop"})
        }
    )
    try:
        instruction = InstructionCandidate(
            candidate_id="local",
            task_id="object_identification",
            family="visual_description",
            visible_scope="left half",
            instruction_summary="Identify the local subject",
            required_capabilities=("visible_entity",),
            target_region=ImageRegion(left=0, top=0, right=0.5, bottom=1)
            if region_field == "target_region"
            else None,
            scope_region=ImageRegion(left=0, top=0, right=0.5, bottom=1)
            if region_field == "scope_region"
            else None,
        )
        original = coordinator._model_image(image, root)
        calls = []

        def invoke(client, stage, payload, images, model, **kwargs):
            assert stage == "question_gate"
            assert len(images) == 1 and images[0].view_id.startswith("focus:")
            assert images[0].encoded_sha256 != original.encoded_sha256
            view = payload["image_views"][0]
            assert view["source_view_id"] == original.view_id
            assert float(view["source_right"]) == 0.5
            assert "candidate_answer" not in payload
            calls.append(images[0].encoded_sha256)
            return QuestionGateVote(
                local_anchor="MET",
                operation_coherent="MET",
                useful_request="MET",
                reason="Visible inside the delivered crop",
                realized_task_id="object_identification",
            )

        monkeypatch.setattr(coordinator, "_invoke", invoke)
        decision, _ = coordinator._question_gate(
            instruction,
            None,
            image,
            build_history_snapshot("conversation", 1, ()),
            PublicMessage(
                message_id="q", turn_index=1, role="user", content="What object is on the left?"
            ),
            "en",
            original,
            [coordinator._view_metadata(image)],
        )
        assert decision.verdict is GateVerdict.MET
        assert len(calls) == 2 and calls[0] == calls[1]
    finally:
        store.close()


def test_reciprocal_identification_disclosure_requires_both_public_directions() -> None:
    prior = (
        PublicMessage(
            message_id="q1",
            turn_index=1,
            role="user",
            content="What is the object mounted on the side of the aircraft?",
        ),
        PublicMessage(message_id="a1", turn_index=1, role="assistant", content="missile"),
    )
    assert reciprocal_identification_disclosure(
        "What is the object that the missile is attached to?", "aircraft", (prior,)
    )
    assert not reciprocal_identification_disclosure(
        "What is the object to the left of the tree?", "aircraft", (prior,)
    )
    assert not reciprocal_identification_disclosure(
        "What is the object that the missile is attached to?", "helicopter", (prior,)
    )


def test_visible_action_affordance_is_not_an_observed_action() -> None:
    assert action_affordance_question(
        "Based on its wheels touching the ground, what action is the car supported to perform?"
    )
    assert action_affordance_question("What can the car do?")
    assert action_affordance_question("What is the athlete capable of doing?")
    assert not action_affordance_question("What is the athlete doing with the ball?")
    assert not action_affordance_question("What can you see the athlete do with the ball?")


def test_rerating_rejects_reciprocal_identification_without_second_judge_call(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        original = coordinator.synthesize_image(image, root)
        first = original.turns[0]
        first = first.model_copy(
            update={
                "instruction": first.instruction.model_copy(
                    update={"task_id": "object_identification", "scope_id": "scope-missile"}
                ),
                "question": first.question.model_copy(
                    update={"content": "What is mounted on the side of the aircraft?"}
                ),
                "answer": first.answer.model_copy(update={"content": "missile"}),
            }
        )
        second = first.model_copy(
            update={
                "turn_index": 2,
                "instruction": first.instruction.model_copy(
                    update={
                        "scope_id": "scope-aircraft",
                        "public_parameters": (
                            PublicParameter(
                                name="target",
                                value="aircraft",
                                origin="image",
                                evidence_refs=("e1",),
                            ),
                        ),
                    }
                ),
                "question": first.question.model_copy(
                    update={
                        "turn_index": 2,
                        "content": "What is the object that the missile is attached to?",
                    }
                ),
                "answer": first.answer.model_copy(update={"turn_index": 2, "content": "aircraft"}),
            }
        )
        saved = original.model_copy(update={"turns": (first, second), "status": "REJECTED"})
        monkeypatch.setattr(coordinator, "_question_gate", _gate_met)
        rated_calls = []

        def rate(*args):
            rated_calls.append(args)
            return TurnRating(items=(), aggregate="PASS")

        monkeypatch.setattr(coordinator, "_rate_turn", rate)
        rerated = coordinator.rate_existing(saved, root)
        rejection_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'public-text-rejections' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        rejection = json.loads(store.read_artifact(rejection_hash))
    finally:
        store.close()
    assert [turn.status for turn in rerated.turns] == ["COMMITTED", "REJECTED"]
    assert len(rated_calls) == 1
    assert rejection["reason"] == "RECIPROCAL_IDENTIFICATION_ALREADY_PUBLIC"


def test_rerating_rejects_action_affordance_without_judge_calls(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    try:
        original = coordinator.synthesize_image(image, root)
        first = original.turns[0]
        affordance = first.model_copy(
            update={
                "instruction": first.instruction.model_copy(update={"task_id": "visible_action"}),
                "question": first.question.model_copy(
                    update={"content": "What action is the car supported to perform?"}
                ),
                "answer": first.answer.model_copy(update={"content": "driving"}),
            }
        )
        saved = original.model_copy(update={"turns": (affordance,), "status": "REJECTED"})
        calls_before = sum(len(client.calls) for client in (generator_a, generator_b))
        rerated = coordinator.rate_existing(saved, root)
        calls_after = sum(len(client.calls) for client in (generator_a, generator_b))
        rejection_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'public-text-rejections' "
            "ORDER BY rowid DESC LIMIT 1"
        ).fetchone()[0]
        rejection = json.loads(store.read_artifact(rejection_hash))
    finally:
        store.close()
    assert rerated.turns[0].status == "REJECTED"
    assert calls_after == calls_before
    assert rejection["reason"] == "ACTION_AFFORDANCE_NOT_VISIBLE"


def test_identification_answer_echo_stops_before_answer_rating(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _client=client, _original=original, **kwargs):
            response = _original(stage, payload, *args, **kwargs)
            if stage == "question_draft":
                value = response.value.model_copy(
                    update={
                        "drafts": tuple(
                            draft.model_copy(update={"question": "What is this gibbon?"})
                            for draft in response.value.drafts
                        )
                    }
                )
            elif stage == "answer_generation":
                value = TextPayload(text="gibbon")
            else:
                return response
            return ModelResponse(
                value=value,
                request_hash=response.request_hash,
                response_hash=response.response_hash,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        result = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()
    assert result.status == "REJECTED"
    assert not result.turns
    assert rejection_count == 1
    assert not any(
        stage == "holistic_review"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_rerating_rejects_saved_identification_answer_echo_without_model_calls(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    try:
        original = coordinator.synthesize_image(image, root)
        assert original.status == "QUALITY_CANDIDATE"
        first = original.turns[0]
        assert first.instruction.task_id == "object_identification"
        leaked = first.model_copy(
            update={
                "question": first.question.model_copy(update={"content": "What is this gibbon?"}),
                "answer": first.answer.model_copy(update={"content": "gibbon"}),
            }
        )
        saved = original.model_copy(update={"turns": (leaked,), "status": "REJECTED"})
        calls_before = sum(len(client.calls) for client in (generator_a, generator_b))
        rerated = coordinator.rate_existing(saved, root)
        calls_after = sum(len(client.calls) for client in (generator_a, generator_b))
    finally:
        store.close()
    assert rerated.status == "REJECTED"
    assert rerated.turns[0].status == "REJECTED"
    assert calls_after == calls_before


def test_rerating_rejects_internal_question_reference_without_model_calls(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    try:
        original = coordinator.synthesize_image(image, root)
        assert original.status == "QUALITY_CANDIDATE"
        first = original.turns[0]
        leaked = first.model_copy(
            update={
                "question": first.question.model_copy(
                    update={"content": "What is the object at scope_0?"}
                )
            }
        )
        saved = original.model_copy(update={"turns": (leaked,), "status": "REJECTED"})
        calls_before = sum(len(client.calls) for client in (generator_a, generator_b))
        rerated = coordinator.rate_existing(saved, root)
        calls_after = sum(len(client.calls) for client in (generator_a, generator_b))
        rejection_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'public-text-rejections' LIMIT 1"
        ).fetchone()[0]
        rejection = json.loads(store.read_artifact(rejection_hash))
    finally:
        store.close()
    assert rerated.status == "REJECTED"
    assert rerated.turns[0].status == "REJECTED"
    assert calls_after == calls_before
    assert rejection["reason"] == "INTERNAL_REFERENCE_IN_QUESTION"


def test_rerating_rejects_unverified_text_locator_without_model_calls(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    try:
        original = coordinator.synthesize_image(image, root)
        assert original.status == "QUALITY_CANDIDATE"
        first = original.turns[0]
        unsupported = first.model_copy(
            update={
                "instruction": first.instruction.model_copy(
                    update={"task_id": "text_transcription"}
                ),
                "question": first.question.model_copy(
                    update={"content": "What text is directly below the Roland logo?"}
                ),
            }
        )
        saved = original.model_copy(update={"turns": (unsupported,), "status": "REJECTED"})
        calls_before = sum(len(client.calls) for client in (generator_a, generator_b))
        rerated = coordinator.rate_existing(saved, root)
        calls_after = sum(len(client.calls) for client in (generator_a, generator_b))
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()
    assert rerated.status == "REJECTED"
    assert rerated.turns[0].status == "REJECTED"
    assert calls_after == calls_before
    assert rejection_count == 1


def test_repeated_answered_request_catches_observed_sheep_paraphrase() -> None:
    first_question = "What are the colors and shapes of the sheep visible in the foreground?"
    second_question = "What colors and shapes are visible in the sheep located in the foreground?"
    answer = (
        "The sheep in the foreground are primarily white and brown. Their shapes are "
        "quadrupedal with woolly bodies, curved horns, and four legs."
    )
    history = (
        PublicMessage(message_id="q1", turn_index=1, role="user", content=first_question),
        PublicMessage(message_id="a1", turn_index=1, role="assistant", content=answer),
    )
    assert repeated_answered_request(second_question, answer, history)
    assert not repeated_answered_request(second_question, "A distinct supported answer.", history)
    assert not repeated_answered_request(second_question, "white", history)


def test_repeated_substantial_answer_stops_before_second_review(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    repeated_answer = "The visible region is blue and contains the same resolved object."
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, **kwargs):
            if stage == "answer_generation":
                return ModelResponse(
                    value=TextPayload(text=repeated_answer),
                    request_hash="request",
                    response_hash="response",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return _original(stage, payload, *args, **kwargs)

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        result = coordinator.synthesize_image(image, root)
        rejected = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()
    assert result.status == "REJECTED"
    assert len(result.turns) == 1
    assert rejected == 1
    assert (
        sum(
            stage == "holistic_review"
            for client in (generator_a, generator_b)
            for stage, _ in client.calls
        )
        == 2
    )


def test_private_prompt_echo_is_rejected_before_question_gate(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        echo_question_prompt=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        reasons = [
            json.loads(store.read_artifact(row[0]))["reason"]
            for row in store.connection.execute(
                "SELECT artifact_hash FROM artifact WHERE kind = 'public-text-rejections'"
            )
        ]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert reasons and set(reasons) == {"PRIVATE_PROMPT_ECHO"}
    assert all(
        stage != "question_gate"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_internal_reference_draft_is_skipped_before_question_gate(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, internal_reference_question="first"
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'public-text-rejections' LIMIT 1"
        ).fetchone()[0]
        rejection = json.loads(store.read_artifact(rejection_hash))
    finally:
        store.close()

    assert conversation.turns
    assert all("scope_0" not in turn.question.content for turn in conversation.turns)
    assert rejection["reason"] == "INTERNAL_REFERENCE_IN_QUESTION"
    assert rejection["content"] == "What is the object at scope_0?"
    gated = [
        payload["question"]
        for client in (generator_a, generator_b)
        for stage, payload in client.calls
        if stage == "question_gate"
    ]
    assert gated and all("scope_0" not in question for question in gated)


@pytest.mark.parametrize("reference_mode", ["always", "selected_region"])
def test_persistent_internal_reference_stops_before_question_gate(
    tmp_path: Path, image_artifact, reference_mode: Literal["always", "selected_region"]
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, internal_reference_question=reference_mode
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert rejection_count >= 1
    assert all(
        stage != "question_gate"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_private_prompt_echo_is_rejected_before_answer_rating(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        echo_answer_prompt=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert rejection_count == 1
    assert any(
        stage == "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )
    assert all(
        stage != "holistic_review"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_successful_generation_uses_full_history_and_dual_blind_judges(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, generator_b = _coordinator(tmp_path)
    try:
        conversation = coordinator.synthesize_image(image, root)
        verification = store.verify()
    finally:
        store.close()

    assert conversation.status == "QUALITY_CANDIDATE"
    assert 2 <= len(conversation.turns) <= 6
    assert all(turn.status == "COMMITTED" for turn in conversation.turns)
    assert {turn.generation_model for turn in conversation.turns} == {conversation.generation_model}
    assert verification["turn_commits"] == len(conversation.turns)
    assert [stage for stage, _ in selector.calls] == ["image_profile"]
    assert "public_history" not in selector.calls[0][1]
    drafts = [
        payload
        for client in (generator_a, generator_b)
        for stage, payload in client.calls
        if stage == "question_draft"
    ]
    for turn in conversation.turns:
        payload = next(item for item in drafts if item["turn_index"] == turn.turn_index)
        assert len(payload["public_history"]) == 2 * (turn.turn_index - 1)
        assert "candidate_answer" not in payload
    for judge in (generator_a, generator_b):
        assert any(stage == "question_gate" for stage, _ in judge.calls)
        assert any(stage == "holistic_review" for stage, _ in judge.calls)
        assert all("other_judge" not in payload for _, payload in judge.calls)


def test_completed_conversation_resumes_without_model_calls(tmp_path: Path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, generator_b = _coordinator(tmp_path)
    try:
        first = coordinator.synthesize_image(image, root)
        selector.calls.clear()
        generator_a.calls.clear()
        generator_b.calls.clear()
        resumed = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert resumed == first
    assert not selector.calls and not generator_a.calls and not generator_b.calls


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("How many signs are there, and can you group them by color?", "white"),
        ("How many lights are attached to each of the concentric rings?", "innermost ring"),
    ],
)
def test_visible_categories_need_not_be_literal_question_substrings(
    tmp_path: Path, question: str, category: str
):
    inventory = SetInventory(
        coverage="MET",
        mode="count",
        counts=(CountGroup(scope=category, expected=2, reported=3),),
        expected_members=(),
        reported_members=(),
        empty_scope_is_explicit=False,
        reason="Visible category",
    )

    class CategoryClient(ScriptedClient):
        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            return ModelResponse(
                value=inventory,
                request_hash="1" * 64,
                response_hash="2" * 64,
                prompt_tokens=1,
                completion_tokens=1,
            )

    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path / "runs", "categories", require_local_wal=False) as store:
        client = CategoryClient(config.models.generator_a)
        co = SynthesisCoordinator(config, "categories", store, client, client, client)
        result = co._invoke(
            client,
            "set_inventory",
            {"question": question},
            (),
            SetInventory,
            max_tokens=2048,
            temperature=0.0,
            seed=1,
        )
        assert result.counts[0].scope == category
        check = verify_set_inventories((result, result))
        assert check is not None and check.verdict == "NOT_MET"
        other = result.model_copy(
            update={"counts": (CountGroup(scope=category, expected=3, reported=3),)}
        )
        assert verify_set_inventories((result, other)) is None


def test_v7_holistic_checks_bound_question_and_two_blind_whole_turn_reviews(
    tmp_path, image_artifact
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path)
    try:
        conversation = co.synthesize_image(image, root)
        assert conversation.status == "QUALITY_CANDIDATE"
        for client in (a, b):
            stages = [stage for stage, _ in client.calls]
            assert stages.count("holistic_review") == len(conversation.turns)
            assert stages.count("question_gate") == len(conversation.turns)
            assert not set(stages) & {
                "requirement_extraction",
                "claim_inventory",
                "set_inventory",
                "rubric_item",
                "answer_repair",
            }
            for stage, payload in client.calls:
                if stage == "holistic_review":
                    assert set(payload) == {
                        "target_language",
                        "public_history",
                        "question",
                        "candidate_answer",
                        "expected_operation",
                        "image_views",
                    }
        assert all(not turn.requirements for turn in conversation.turns)
        assert all(turn.rating.items[0].template_id == "Q_HOLISTIC" for turn in conversation.turns)
        calls = len(a.calls) + len(b.calls)
        assert co.synthesize_image(image, root) == conversation
        assert len(a.calls) + len(b.calls) == calls
    finally:
        store.close()


@pytest.mark.parametrize(
    ("votes", "expected"),
    [
        (("NOT_MET", "NOT_MET"), "REJECTED"),
        (("MET", "NOT_MET"), "ABSTAINED"),
        (("UNKNOWN", "MET"), "ABSTAINED"),
    ],
)
@pytest.mark.parametrize("passing_turns", [0, 1, 2])
def test_holistic_stop_retains_only_two_or_more_accepted_turns(
    tmp_path, image_artifact, monkeypatch, votes, expected, passing_turns
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path)
    # Retention is independent of the single repair, which a split vote would otherwise trigger.
    co.config = co.config.model_copy(
        update={"evaluation": co.config.evaluation.model_copy(update={"repair_once": False})}
    )
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    for client, vote in zip((a, b), votes, strict=True):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, _vote=vote, **kwargs):
            response = _original(stage, payload, *args, **kwargs)
            if stage == "holistic_review" and len(payload["public_history"]) >= 2 * passing_turns:
                return ModelResponse(
                    value=RubricVerdict(verdict=_vote, reason="Observed defect or uncertainty."),
                    request_hash=response.request_hash,
                    response_hash=response.response_hash,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return response

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        conversation = co.synthesize_image(image, root)
        if passing_turns >= 2:
            assert conversation.status == "QUALITY_CANDIDATE"
            assert len(conversation.turns) == passing_turns
            assert all(t.status == "COMMITTED" for t in conversation.turns)
            rows = store.connection.execute(
                "SELECT artifact_hash FROM artifact WHERE kind='conversation-stops'"
            ).fetchall()
            assert len(rows) == 1

            stopped = json.loads(store.read_artifact(rows[0][0]))
            assert stopped["conversation"]["status"] == expected
            assert len(stopped["conversation"]["turns"]) == passing_turns + 1
        else:
            assert conversation.status == expected
        assert co.synthesize_image(image, root) == conversation
        assert not any(stage == "answer_repair" for client in (a, b) for stage, _ in client.calls)
    finally:
        store.close()


def test_holistic_prefix_retention_can_be_disabled(tmp_path, image_artifact, monkeypatch):
    image, root = image_artifact
    co, store, _, _, _ = _coordinator(tmp_path)
    co.config = co.config.model_copy(
        update={"evaluation": EvaluationConfig(retain_accepted_prefix=False)}
    )
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 3)
    original = co._draft_questions
    monkeypatch.setattr(
        co,
        "_draft_questions",
        lambda route, snapshot, *args: (
            () if snapshot.turn_index == 3 else original(route, snapshot, *args)
        ),
    )
    try:
        conversation = co.synthesize_image(image, root)
        assert len(conversation.turns) == 2
        assert conversation.status == "REJECTED"
    finally:
        store.close()


def test_holistic_rerating_rejects_empty_conversation(tmp_path, image_artifact):
    image, root = image_artifact
    co, store, _, _, _ = _coordinator(tmp_path, reject_selection=True)
    try:
        conversation = co.synthesize_image(image, root)
        assert not conversation.turns
        assert co.rate_existing(conversation, root).status == "REJECTED"
    finally:
        store.close()


def test_holistic_prefix_survives_reopening_and_exports_only_committed_turns(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path)
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    original = co._draft_questions
    monkeypatch.setattr(
        co,
        "_draft_questions",
        lambda route, snapshot, *args: (
            () if snapshot.turn_index == 3 else original(route, snapshot, *args)
        ),
    )
    accepted = co.synthesize_image(image, root)
    assert accepted.status == "QUALITY_CANDIDATE" and len(accepted.turns) == 2
    store.close()
    with RunStore(tmp_path / "runs", "test", require_local_wal=False) as reopened:
        reopened.initialize_run("test", co.config.config_hash, co.config.profile)
        resumed = SynthesisCoordinator(co.config, "test", reopened, co.router, a, b)
        calls = len(a.calls) + len(b.calls)
        assert resumed.synthesize_image(image, root) == accepted
        assert len(a.calls) + len(b.calls) == calls
        changed = image.model_copy(update={"source_id": "different-source"})
        with pytest.raises(ExecutionError, match="conflicts with the current input"):
            resumed.synthesize_image(changed, root)
        record = training_record(accepted)
        assert len(record.messages) == 4
        assert record.messages[-1].content[0].text == accepted.turns[-1].answer.content


def test_holistic_rerating_uses_new_votes_and_preserves_failed_tail_privately(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path)
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    try:
        original = co.synthesize_image(image, root)
        for client in (a, b):
            invoke_original = client.invoke

            def invoke(stage, payload, *args, _original=invoke_original, **kwargs):
                result = _original(stage, payload, *args, **kwargs)
                if stage == "holistic_review" and len(payload["public_history"]) == 4:
                    return ModelResponse(
                        value=RubricVerdict(verdict="NOT_MET", reason="Wrong visible fact."),
                        request_hash=result.request_hash,
                        response_hash=result.response_hash,
                        prompt_tokens=1,
                        completion_tokens=1,
                    )
                return result

            monkeypatch.setattr(client, "invoke", invoke)
        rated = co.rate_existing(original, root)
        assert rated.status == "QUALITY_CANDIDATE"
        assert len(rated.turns) == 2
        assert rated.public_messages == original.public_messages[:4]
        assert all(t.rating.items[0].template_id == "Q_HOLISTIC" for t in rated.turns)
    finally:
        store.close()


def test_v7_holistic_wrong_operation_stops_before_answer_generation(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path)
    monkeypatch.setattr(
        co,
        "_question_gate",
        lambda candidate, *args: (
            GateDecision(GateVerdict.NOT_MET, candidate.task_id, "label_mismatch", GateVerdict.MET),
            candidate,
        ),
    )
    try:
        result = co.synthesize_image(image, root)
        assert result.status == "REJECTED"
        assert not any(
            stage == "answer_generation" for client in (a, b) for stage, _ in client.calls
        )
    finally:
        store.close()


def test_completed_slots_refill_but_slow_head_bounds_buffer_and_output_order(
    tmp_path: Path, image_artifact, monkeypatch: pytest.MonkeyPatch
):
    from concurrent.futures import ThreadPoolExecutor

    from pixelogue.contracts import ConversationArtifact
    from pixelogue.serialization import canonical_hash

    coordinator, store, _, _, _ = _coordinator(tmp_path)
    coordinator.config = coordinator.config.model_copy(
        update={
            "runtime": coordinator.config.runtime.model_copy(
                update={"max_concurrent_images": 2, "refill_completed_images": True}
            )
        }
    )
    image, root = image_artifact
    release_head, fourth_finished = threading.Event(), threading.Event()
    probe = ConcurrencyProbe()
    consumed = []

    def jobs():
        for index in range(8):
            consumed.append(index)
            yield SynthesisJob(
                image=image.model_copy(update={"image_id": f"{index:064x}"}),
                target_language="en",
                generator_role="generator_a",
            )

    def synthesize(job, artifact_root):
        index = int(job.image.image_id, 16)
        probe.enter()
        try:
            if index == 0:
                assert release_head.wait(5), "Consumer did not release the held first image"
            if index == 3:
                fourth_finished.set()
            return ConversationArtifact(
                conversation_id=canonical_hash(index),
                image=job.image,
                target_language="en",
                generation_model=coordinator.config.models.generator_a.repo_id,
                turns=(),
                status="ERROR",
            )
        finally:
            probe.exit()

    monkeypatch.setattr(coordinator, "_synthesize_job", synthesize)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: list(coordinator.synthesize_batch(jobs(), root, max_workers=2))
            )
            try:
                assert fourth_finished.wait(3), (
                    "Finished slots were not refilled behind the held head"
                )
                assert consumed == [0, 1, 2, 3]
                assert not future.done()
            finally:
                release_head.set()
            results = future.result(timeout=5)
        assert [int(result.image.image_id, 16) for result in results] == list(range(8))
        assert probe.peak == 2 and probe.active == 0
    finally:
        release_head.set()
        store.close()


def test_refill_keeps_saved_conversation_replay_and_store_serialization(
    tmp_path: Path, image_artifact
):
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    coordinator.config = coordinator.config.model_copy(
        update={
            "runtime": coordinator.config.runtime.model_copy(
                update={"max_concurrent_images": 2, "refill_completed_images": True}
            )
        }
    )
    image, root = image_artifact
    jobs = [
        SynthesisJob(
            image=image.model_copy(update={"image_id": f"{index + 1:064x}"}),
            target_language="en",
            generator_role="generator_a",
        )
        for index in range(4)
    ]
    try:
        first = list(coordinator.synthesize_batch(jobs, root, max_workers=2))
        calls = len(generator_a.calls) + len(generator_b.calls)
        resumed = list(coordinator.synthesize_batch(jobs, root, max_workers=2))
        assert resumed == first
        assert len(generator_a.calls) + len(generator_b.calls) == calls
        assert [row.image.image_id for row in first] == [job.image.image_id for job in jobs]
        assert store.verify()["artifacts"] > 0
    finally:
        store.close()


def test_empty_local_pixels_abstain_even_after_a_base_review_pass(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        instruction = InstructionCandidate(
            candidate_id="subpixel",
            task_id="object_identification",
            family="visual_description",
            visible_scope="empty pixel region",
            instruction_summary="Identify a local entity",
            required_capabilities=("visible_entity",),
            catalog_version="8.0",
            scope_id="tiny",
            evidence_refs=("visible-local-evidence",),
            verification_contracts=("dual_visual_review",),
            view_id=image.full_view.view_id,
            target_region=ImageRegion(left=0, top=0, right=0.00001, bottom=0.00001),
        )
        monkeypatch.setattr(
            coordinator,
            "_rate_holistic",
            lambda *args, **kwargs: TurnRating(items=(), aggregate="PASS"),
        )
        monkeypatch.setattr(
            coordinator,
            "_invoke",
            lambda *args, **kwargs: pytest.fail("Empty pixels called a model"),
        )
        rating = coordinator._rate_turn(
            "conversation",
            "a" * 64,
            (),
            PublicMessage(message_id="q", turn_index=1, role="user", content="What is here?"),
            PublicMessage(message_id="a", turn_index=1, role="assistant", content="object"),
            instruction,
            "en",
            [coordinator._view_metadata(image)],
            coordinator._model_image(image, root),
            1,
        )
        assert rating.aggregate == "ABSTAIN"
        assert rating.items[-1].verdict is GateVerdict.UNKNOWN
    finally:
        store.close()


def test_blind_transcription_sees_full_context_and_rejects_an_incomplete_bound(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        original = coordinator._model_image(image, root)
        instruction = InstructionCandidate(
            candidate_id="title",
            task_id="text_transcription",
            family="text_reading",
            visible_scope="top half",
            instruction_summary="Read the complete title",
            required_capabilities=("readable_text",),
            catalog_version="8.0",
            scope_id="title",
            evidence_refs=("title-evidence",),
            verification_contracts=("dual_visual_review", "transcript_alignment"),
            view_id=original.view_id,
            target_region=ImageRegion(left=0, top=0, right=1, bottom=0.5),
        )
        monkeypatch.setattr(
            coordinator,
            "_rate_holistic",
            lambda *args, **kwargs: TurnRating(items=(), aggregate="PASS"),
        )
        calls = []

        def invoke(client, stage, payload, images, model, **kwargs):
            assert stage == "transcript_source"
            assert "candidate_answer" not in payload
            assert images == (original,)
            assert payload["image_views"][0]["view_id"] == original.view_id
            calls.append(stage)
            return TranscriptSource(
                coverage="MET",
                expected_lines=("Complete title", "second line"),
                source_region=ImageRegion(left=0.1, top=0.2, right=0.9, bottom=0.7),
                requested_unit_complete=True,
                reason="The original image shows the second line below the bound.",
            )

        monkeypatch.setattr(coordinator, "_invoke", invoke)
        rating = coordinator._rate_turn(
            "conversation",
            "a" * 64,
            (),
            PublicMessage(message_id="q", turn_index=1, role="user", content="Read the title."),
            PublicMessage(
                message_id="a",
                turn_index=1,
                role="assistant",
                content="Complete title\nsecond line",
            ),
            instruction,
            "en",
            [coordinator._view_metadata(image)],
            original,
            1,
        )
        assert calls == ["transcript_source", "transcript_source"]
        assert rating.aggregate == "ABSTAIN"
        assert rating.items[-1].verdict is GateVerdict.UNKNOWN
    finally:
        store.close()


def test_chart_inventory_uses_the_configured_source_budget_without_accepting_unknown(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        original = coordinator._model_image(image, root)
        instruction = InstructionCandidate(
            candidate_id="ranking",
            task_id="chart_extremum_ranking",
            family="charts_and_maps",
            visible_scope="complete bar chart",
            instruction_summary="Identify the highest and lowest bars",
            required_capabilities=("readable_chart", "complete_series"),
            catalog_version="8.0",
            scope_id="chart",
            evidence_refs=("chart-evidence",),
            verification_contracts=(
                "dual_visual_review",
                "chart_encoding_check",
                "closed_set_check",
            ),
            view_id=original.view_id,
            scope_region=ImageRegion(left=0, top=0, right=1, bottom=1),
            public_parameters=(
                PublicParameter(name="rank_mode", value="max_min", origin="instruction"),
                PublicParameter(name="rank_order", value="descending", origin="instruction"),
            ),
        )
        monkeypatch.setattr(
            coordinator,
            "_rate_holistic",
            lambda *args, **kwargs: TurnRating(items=(), aggregate="PASS"),
        )
        calls = []

        def invoke(client, stage, payload, images, model, **kwargs):
            calls.append(stage)
            if stage == "chart_source":
                assert "candidate_answer" not in payload
                assert images == (original,)
                assert kwargs["max_tokens"] == coordinator.config.tasks.source_max_tokens
                assert instruction.scope_region is not None
                return ChartSource(
                    task_id=instruction.task_id,
                    coverage="UNKNOWN",
                    scope_id="chart",
                    view_id=original.view_id,
                    scope_region=instruction.scope_region,
                    axis=None,
                    legend_complete=False,
                    closed=False,
                    encoding="bar",
                    marks=(),
                    query=ChartQuery(operation="rank", rank_mode="max_min"),
                    reason="Unclear labels.",
                )
            assert stage == "chart_answer" and not images
            return ChartAnswer(coverage="UNKNOWN", reason="No unambiguous category.")

        monkeypatch.setattr(coordinator, "_invoke", invoke)
        rating = coordinator._rate_turn(
            "conversation",
            "a" * 64,
            (),
            PublicMessage(
                message_id="q",
                turn_index=1,
                role="user",
                content="Which bars are highest and lowest?",
            ),
            PublicMessage(message_id="a", turn_index=1, role="assistant", content="A and B."),
            instruction,
            "en",
            [coordinator._view_metadata(image)],
            original,
            1,
        )
        assert calls == ["chart_source", "chart_source", "chart_answer", "chart_answer"]
        assert rating.aggregate == "ABSTAIN"
        assert rating.items[-1].verdict is GateVerdict.UNKNOWN
    finally:
        store.close()


def test_rerating_rejects_a_ui_name_instead_of_a_location_without_model_calls(
    tmp_path, image_artifact
):
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    try:
        original = coordinator.synthesize_image(image, root)
        first = original.turns[0]
        instruction = InstructionCandidate(
            candidate_id="menu",
            task_id="ui_element_location",
            family="screen_ui",
            visible_scope="visible desktop",
            instruction_summary="Locate the named menu",
            required_capabilities=("ui_controls",),
            catalog_version="8.0",
            scope_id="screen",
            evidence_refs=("menu-evidence",),
            verification_contracts=("dual_visual_review", "ui_grounding_check"),
            view_id=image.full_view.view_id,
            public_parameters=(
                PublicParameter(name="target", value="the 'Aktionen' menu", origin="instruction"),
            ),
        )
        changed = first.model_copy(
            update={
                "instruction": instruction,
                "question": first.question.model_copy(
                    update={"content": "Where is the 'Aktionen' menu located?"}
                ),
                "answer": first.answer.model_copy(update={"content": "the 'Aktionen' menu"}),
            }
        )
        calls_before = sum(len(client.calls) for client in (generator_a, generator_b))
        rerated = coordinator.rate_existing(original.model_copy(update={"turns": (changed,)}), root)
        calls_after = sum(len(client.calls) for client in (generator_a, generator_b))
        assert rerated.turns[0].status == "REJECTED" and calls_after == calls_before
        assert (
            coordinator._answer_disclosure_reason(
                instruction, changed.question.content, "In the upper-left menu bar.", ()
            )
            is None
        )
    finally:
        store.close()
