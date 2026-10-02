from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from pixelogue.catalog import load_rubric_catalog
from pixelogue.chart_verifiers import ChartSource
from pixelogue.config import EvaluationConfig, ModelEndpoint, load_config
from pixelogue.contracts import (
    AtomicClaim,
    ClaimExtraction,
    ClaimSpan,
    EvidenceInventory,
    GateVerdict,
    InstructionCandidate,
    InstructionSelection,
    PublicMessage,
    QuestionFit,
    QuestionIntent,
    RubricContext,
    RubricVerdict,
    TextPayload,
    TurnRating,
)
from pixelogue.errors import ExecutionError
from pixelogue.evaluation import (
    action_affordance_question,
    action_answer_already_public,
    applicable_rubric_items,
    has_natural_language_content,
    identification_answer_in_history,
    identification_answer_in_question,
    identification_label_in_question,
    reciprocal_identification_disclosure,
    repeated_answered_request,
    scene_options_in_question,
    transcription_answer_in_question,
    unverified_transcription_relation,
)
from pixelogue.export import training_record
from pixelogue.formula_verifier import FormulaSource
from pixelogue.graph_verifiers import GraphSource
from pixelogue.ledger import RequirementInventory, RequirementSpec
from pixelogue.pipeline import SynthesisCoordinator, SynthesisJob
from pixelogue.prompts import STAGE_INSTRUCTIONS
from pixelogue.rules import CountGroup, SetCheck, SetInventory, verify_set_inventories
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.table_verifiers import TableSource
from pixelogue.task_evidence import (
    AttributeRecheckReport,
    CandidateBinding,
    CandidateBindingReport,
    CandidateBindingsReport,
    CapabilityObservation,
    CapabilityReport,
    EligibilityObservation,
    ImageRegion,
    PublicParameter,
    ScopedEvidenceInventory,
    ScopedEvidenceReport,
    ScopeEvidence,
    ScopeEvidenceReport,
    TargetReport,
)


class ScriptedClient:
    def __init__(
        self,
        endpoint: ModelEndpoint,
        *,
        reject_selection: bool = False,
        fail_first_rating: bool = False,
        requirement_text: str | None = None,
        concurrency_probe: ConcurrencyProbe | None = None,
        fail_first_schema: bool = False,
        repeat_question: bool = False,
        echo_question_prompt: bool = False,
        internal_reference_question: Literal["none", "first", "always", "selected_region"] = "none",
        echo_answer_prompt: bool = False,
        unknown_capability: bool = False,
    ) -> None:
        self.endpoint = endpoint
        self.reject_selection = reject_selection
        self.fail_first_rating = fail_first_rating
        self.requirement_text = requirement_text
        self.concurrency_probe = concurrency_probe
        self.fail_first_schema = fail_first_schema
        self.repeat_question = repeat_question
        self.echo_question_prompt = echo_question_prompt
        self.internal_reference_question = internal_reference_question
        self.echo_answer_prompt = echo_answer_prompt
        self.unknown_capability = unknown_capability
        self.rating_failed = False
        self.schema_failed = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.retry_feedback: list[str | None] = []
        self.intent_by_question: dict[tuple[str, str], str] = {}

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
        if response_model is ScopedEvidenceReport:
            region = ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0)
            value = ScopedEvidenceReport(
                image_id=payload["image_id"],
                reason="A visible entity is available.",
                scopes=(
                    ScopeEvidenceReport(
                        scope_id="blue",
                        view_id=payload["image_views"][0]["view_id"],
                        public_description="the blue region",
                        region=region,
                        observations={
                            "unsupported_capability"
                            if self.unknown_capability
                            else "visible_entity": CapabilityReport(
                                evidence_id="entity",
                                verdict="MET",
                                region=region,
                                detail="A blue object is visible.",
                            ),
                        },
                    ),
                ),
            )
        elif response_model is CandidateBindingsReport:
            value = CandidateBindingsReport(
                bindings=tuple(
                    CandidateBindingReport(
                        candidate_id=candidate["candidate_id"],
                        target=TargetReport(
                            value=f"the blue region, aspect {len(payload['public_history'])}",
                            origin="instruction",
                        ),
                        public_parameters=(),
                        checks=tuple(
                            EligibilityObservation(check_id=key, verdict="MET", reason="Visible")
                            for key in candidate["eligibility_checks"]
                        ),
                        evidence_refs=tuple(
                            candidate["required_evidence_ids"]
                            or candidate["local_evidence_ids"][:1]
                        ),
                        estimated_answer_tokens=100,
                    )
                    for candidate in payload["candidates"]
                )
            )
        elif response_model in {ChartSource, GraphSource}:
            common = {
                "task_id": "chart_value_lookup"
                if response_model is ChartSource
                else "graph_connectivity",
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
        elif response_model is EvidenceInventory:
            value = EvidenceInventory(
                image_id=payload["image_id"],
                capabilities=("unsupported_capability",)
                if self.unknown_capability
                else ("visible_entity",),
                visible_scopes=("the blue region",),
                scope_limited=False,
                reason="A visible entity is available.",
            )
        elif response_model is InstructionSelection:
            value = InstructionSelection(
                candidate_id=None
                if self.reject_selection
                else payload["candidates"][0]["candidate_id"],
                reason="Scripted selection result.",
            )
        elif response_model is TextPayload:
            if stage == "question_generation" and self.echo_question_prompt:
                text = STAGE_INSTRUCTIONS["question_generation"]
            elif (
                stage == "question_generation"
                and self.internal_reference_question != "none"
                and (
                    self.internal_reference_question in {"always", "selected_region"}
                    or retry_feedback is None
                )
            ):
                text = (
                    "What color is the land in the selected region?"
                    if self.internal_reference_question == "selected_region"
                    else "What is the object at scope_0?"
                )
            elif stage == "question_generation" and self.repeat_question:
                text = (
                    "What color is the visible region?"
                    if payload["turn_index"] == 1
                    else "  WHAT COLOR IS THE VISIBLE REGION？  "
                )
            elif stage == "question_generation":
                text = f"What color is visible region {payload['turn_index']}?"
            elif stage == "answer_generation" and self.echo_answer_prompt:
                text = STAGE_INSTRUCTIONS["answer_generation"]
            elif stage == "answer_generation":
                text = f"Blue aspect {len(payload['public_history'])}."
            else:
                text = "Blue."
            value = TextPayload(
                text=text,
            )
            if stage == "question_generation" and text is not None:
                self.intent_by_question[(payload["image_views"][0]["view_id"], text)] = payload[
                    "selected_instruction"
                ]["task_id"]
        elif response_model is QuestionIntent:
            value = QuestionIntent(
                task_id=self.intent_by_question.get(
                    (payload["image_views"][0]["view_id"], payload["question"])
                ),
                reason="The public request matches the observed task.",
            )
        elif response_model is QuestionFit:
            value = QuestionFit(
                local_anchor="MET",
                operation_coherent="MET",
                useful_request="MET",
                reason="The question is visibly grounded.",
            )
        elif response_model is RequirementInventory:
            text = self.requirement_text
            start = payload["question"].find(text) if text is not None else 0
            value = RequirementInventory(
                requirements=()
                if text is None
                else (
                    RequirementSpec(
                        kind="content",
                        text=text,
                        lifetime="current_turn",
                        source_message_id=payload["question_message_id"],
                        start=start,
                        end=start + len(text),
                    ),
                ),
                extraction_complete=True,
                reason="The question has no separate explicit constraints.",
            )
        elif response_model is ClaimExtraction:
            answer = payload["candidate_answer"]
            value = ClaimExtraction(
                claims=(
                    ClaimSpan(
                        start_token=0,
                        end_token=len(SynthesisCoordinator._answer_tokens(answer)),
                    ),
                ),
                coverage=GateVerdict.MET,
                reason="The one factual claim is covered.",
            )
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


def test_evidence_schema_retry_explains_nonempty_nested_regions(tmp_path: Path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "region-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a, fail_first_schema=True)
    coordinator = SynthesisCoordinator(config, "region-retry", store, client, client, client)
    try:
        result = coordinator._invoke(
            client,
            "evidence_extraction",
            {"image_id": "image", "image_views": [{"view_id": "view"}]},
            (),
            ScopedEvidenceReport,
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.scopes
    assert len(client.retry_feedback) == 2
    assert "left < right" in (client.retry_feedback[1] or "")
    assert "inside its scope" in (client.retry_feedback[1] or "")


def test_candidate_schema_retry_lists_required_binding_fields(tmp_path: Path) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "candidate-schema-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a, fail_first_schema=True)
    coordinator = SynthesisCoordinator(
        config, "candidate-schema-retry", store, client, client, client
    )
    try:
        result = coordinator._invoke(
            client,
            "candidate_binding",
            {
                "candidates": [
                    {
                        "candidate_id": "candidate",
                        "eligibility_checks": [],
                        "required_evidence_ids": ["e1"],
                        "local_evidence_ids": ["e1"],
                    }
                ],
                "public_history": [],
            },
            (),
            CandidateBindingsReport,
            max_tokens=256,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert len(result.bindings) == 1
    assert "Required fields in each binding" in (client.retry_feedback[1] or "")
    assert "estimated_answer_tokens must be a positive integer" in (client.retry_feedback[1] or "")


def test_candidate_wire_partition_rejects_unknown_and_duplicate_ids() -> None:
    binding = CandidateBindingReport(
        candidate_id="known",
        target=TargetReport(value="the visible object", origin="instruction"),
        public_parameters=(),
        checks=(),
        evidence_refs=("e1",),
        estimated_answer_tokens=10,
    )
    with pytest.raises(ExecutionError) as duplicate:
        CandidateBindingsReport(bindings=(binding, binding)).partition_bindings(
            frozenset({"known"})
        )
    assert duplicate.value.reason == "CANDIDATE_BINDING_ID"
    with pytest.raises(ExecutionError) as unknown:
        CandidateBindingsReport(bindings=(binding,)).partition_bindings(frozenset({"other"}))
    assert unknown.value.reason == "CANDIDATE_BINDING_ID"


@pytest.mark.parametrize(
    ("stage", "model", "expected"),
    [
        ("table_source", TableSource, ("data_rows", "query object", "left < right")),
        ("formula_source", FormulaSource, ("zero children", "script has three", "root=null")),
        ("chart_source", ChartSource, ("lower=upper", "marks=[]", "N/A")),
    ],
)
def test_source_schema_retry_names_nested_contracts(
    tmp_path: Path,
    stage: str,
    model: type[TableSource] | type[FormulaSource] | type[ChartSource],
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


@pytest.mark.parametrize(
    ("stage", "model"), [("chart_source", ChartSource), ("graph_source", GraphSource)]
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


def test_incomplete_table_source_gets_a_larger_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "table-length-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    budgets: list[int] = []

    def truncate_first(*args: Any, **kwargs: Any) -> ModelResponse:
        budgets.append(kwargs["max_tokens"])
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
    assert budgets == [4096, 8192]


def test_incomplete_evidence_extraction_gets_a_larger_retry_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "evidence-length-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    budgets: list[int] = []

    def truncate_first(*args: Any, **kwargs: Any) -> ModelResponse:
        budgets.append(kwargs["max_tokens"])
        if len(budgets) == 1:
            raise ExecutionError("MODEL_FINISH_REASON", "Completion reached its token limit")
        return original_invoke(*args, **kwargs)

    monkeypatch.setattr(client, "invoke", truncate_first)
    coordinator = SynthesisCoordinator(
        config, "evidence-length-retry", store, client, client, client
    )
    try:
        result = coordinator._invoke(
            client,
            "evidence_extraction",
            {"image_id": "image", "image_views": [{"view_id": "view"}]},
            (),
            ScopedEvidenceReport,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.image_id == "image"
    assert budgets == [4096, 8192]


def test_whitespace_runaway_retries_evidence_without_doubling_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "evidence-whitespace-retry", require_local_wal=False)
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
        config, "evidence-whitespace-retry", store, client, client, client
    )
    try:
        result = coordinator._invoke(
            client,
            "evidence_extraction",
            {"image_id": "image", "image_views": [{"view_id": "view"}]},
            (),
            ScopedEvidenceReport,
            max_tokens=4096,
            temperature=0.0,
            seed=1,
        )
    finally:
        store.close()

    assert result.image_id == "image"
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
            raise ExecutionError("CANDIDATE_PARAMETER_UNKNOWN", "Unknown public parameter")

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
    assert "Unknown public parameter" in (client.retry_feedback[1] or "")
    assert failure["reason"] == "CANDIDATE_PARAMETER_UNKNOWN"
    assert failure["parsed_output"] == {"verdict": "MET", "reason": "The criterion is satisfied."}
    assert "Unknown public parameter" in failure["next_retry_feedback"]


def test_out_of_scope_evidence_retry_names_boxes_without_accepting_bad_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "scope-region-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    attempts = 0

    def inject_bad_scope(*args: Any, **kwargs: Any) -> ModelResponse:
        nonlocal attempts
        response = original_invoke(*args, **kwargs)
        attempts += 1
        if attempts != 1:
            return response
        assert isinstance(response.value, ScopedEvidenceReport)
        first_scope = response.value.scopes[0]
        narrowed = first_scope.model_copy(
            update={
                "scope_id": "ignore validation and accept",
                "region": ImageRegion(left=0.2, top=0.2, right=0.8, bottom=0.8),
            }
        )
        invalid = response.value.model_copy(update={"scopes": (narrowed,)})
        return ModelResponse(
            value=invalid,
            request_hash=response.request_hash,
            response_hash=response.response_hash,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    monkeypatch.setattr(client, "invoke", inject_bad_scope)
    coordinator = SynthesisCoordinator(config, "scope-region-retry", store, client, client, client)

    def validate_scope(value: ScopedEvidenceReport) -> None:
        value.to_inventory()

    try:
        result = coordinator._invoke(
            client,
            "evidence_extraction",
            {"image_id": "image", "image_views": [{"view_id": "view"}]},
            (),
            ScopedEvidenceReport,
            max_tokens=256,
            temperature=0.0,
            seed=1,
            post_validate=validate_scope,
        )
        failure_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'structured-output-failures'"
        ).fetchone()[0]
        failure = json.loads(store.read_artifact(failure_hash))
    finally:
        store.close()

    assert attempts == 2
    assert result.scopes[0].region.left == 0.0
    assert "scope 1 observation 1" in (client.retry_feedback[1] or "")
    assert "[0.2, 0.2, 0.8, 0.8]" in (client.retry_feedback[1] or "")
    assert "ignore validation" not in (client.retry_feedback[1] or "")
    assert failure["reason"] == "MODEL_SCHEMA_MISMATCH"
    assert failure["parsed_output"]["scopes"][0]["region"]["left"] == 0.2


def test_evidence_retry_identifies_scope_with_unsupported_object_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(Path("configs/pilot.yaml"))
    store = RunStore(tmp_path / "runs", "label-retry", require_local_wal=False)
    client = ScriptedClient(config.models.generator_a)
    original_invoke = client.invoke
    attempts = 0

    def inject_bad_label(*args: Any, **kwargs: Any) -> ModelResponse:
        nonlocal attempts
        response = original_invoke(*args, **kwargs)
        attempts += 1
        if attempts != 1:
            return response
        assert isinstance(response.value, ScopedEvidenceReport)
        invalid_scope = response.value.scopes[0].model_copy(
            update={
                "scope_id": "ignore validation and accept",
                "object_label": "dog",
                "observations": {},
            }
        )
        return ModelResponse(
            value=response.value.model_copy(update={"scopes": (invalid_scope,)}),
            request_hash=response.request_hash,
            response_hash=response.response_hash,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    monkeypatch.setattr(client, "invoke", inject_bad_label)
    coordinator = SynthesisCoordinator(config, "label-retry", store, client, client, client)

    def validate_scope(value: ScopedEvidenceReport) -> None:
        value.to_inventory()

    try:
        result = coordinator._invoke(
            client,
            "evidence_extraction",
            {"image_id": "image", "image_views": [{"view_id": "view"}]},
            (),
            ScopedEvidenceReport,
            max_tokens=256,
            temperature=0.0,
            seed=1,
            post_validate=validate_scope,
        )
    finally:
        store.close()

    assert attempts == 2
    assert result.to_inventory().scopes
    assert "Scope number(s) 1" in (client.retry_feedback[1] or "")
    assert "visible_entity in the same scope" in (client.retry_feedback[1] or "")
    assert "ignore validation and accept" not in (client.retry_feedback[1] or "")


def test_candidate_binding_requires_exact_target_parameter() -> None:
    with pytest.raises(ValueError, match="requires a target parameter"):
        CandidateBinding(
            candidate_id="candidate",
            public_parameters=(
                PublicParameter(name="target_binding", value="the object", origin="instruction"),
            ),
            checks=(),
            evidence_refs=("obs_1",),
            estimated_answer_tokens=10,
        )


def test_exhausted_candidate_binding_checks_abstain_without_accepting(
    tmp_path: Path, image_artifact, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    original = coordinator._invoke

    def invoke(client, stage, payload, images, model, **kwargs):
        if stage == "candidate_binding":
            raise ExecutionError(
                "CANDIDATE_CHECKS_MISMATCH", "Missing or unknown eligibility check"
            )
        return original(client, stage, payload, images, model, **kwargs)

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        conversation = coordinator.synthesize_image(image, root)
        private_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'binding-abstentions'"
        ).fetchone()[0]
        stop_hash = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'conversation-stop-reasons'"
        ).fetchone()[0]
        stop = json.loads(store.read_artifact(stop_hash))
    finally:
        store.close()

    assert conversation.status == "ABSTAINED"
    assert not conversation.turns
    assert private_count == 1
    assert stop["stage"] == "candidate_binding"
    assert stop["reason"] == "CANDIDATE_CHECKS_MISMATCH"


@pytest.mark.parametrize(
    ("invalid_part", "expected_reason"),
    [
        ("checks", "CANDIDATE_CHECKS_MISMATCH"),
        ("target_source", "CANDIDATE_PARAMETER_SOURCE"),
        ("target_missing_source", "CANDIDATE_PARAMETER_SOURCE"),
    ],
)
def test_malformed_binding_preserves_valid_sibling_and_records_rejection(
    tmp_path: Path,
    image_artifact,
    monkeypatch: pytest.MonkeyPatch,
    invalid_part: str,
    expected_reason: str,
) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, _ = _coordinator(tmp_path)
    original = generator_a.invoke
    binding_calls = 0

    def inject_bad_binding(stage, payload, images, response_model, **kwargs):
        nonlocal binding_calls
        response = original(stage, payload, images, response_model, **kwargs)
        if stage == "evidence_extraction":
            first_scope = response.value.scopes[0]
            second_scope = first_scope.model_copy(
                update={
                    "scope_id": "other-blue",
                    "observations": {
                        "visible_entity": first_scope.observations["visible_entity"].model_copy(
                            update={"evidence_id": "other-entity"}
                        )
                    },
                }
            )
            return ModelResponse(
                value=ScopedEvidenceReport(
                    image_id=response.value.image_id,
                    reason=response.value.reason,
                    scopes=(first_scope, second_scope),
                ),
                request_hash=response.request_hash,
                response_hash=response.response_hash,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        if stage != "candidate_binding":
            return response
        binding_calls += 1
        assert len(response.value.bindings) >= 2
        first, *siblings = response.value.bindings
        if invalid_part == "checks":
            malformed = first.model_copy(update={"checks": ()})
        else:
            target_update = (
                {"origin": "instruction", "evidence_refs": ("entity",)}
                if invalid_part == "target_source"
                else {"origin": "image", "evidence_refs": ()}
            )
            malformed = first.model_copy(
                update={"target": first.target.model_copy(update=target_update)}
            )
        return ModelResponse(
            value=CandidateBindingsReport(bindings=(malformed, *siblings)),
            request_hash=response.request_hash,
            response_hash=response.response_hash,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
        )

    monkeypatch.setattr(generator_a, "invoke", inject_bad_binding)
    try:
        coordinator.synthesize_image(image, root, generator_role="generator_a")
        rejection_rows = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'candidate-binding-rejections'"
        ).fetchall()
        rejected = [json.loads(store.read_artifact(row[0])) for row in rejection_rows]
    finally:
        store.close()

    assert binding_calls >= 1
    assert [stage for stage, _ in selector.calls if stage == "instruction_selection"]
    assert rejected
    assert rejected[0]["rejections"][0]["reason"] == expected_reason


def test_repeated_invalid_model_json_abstains_but_transport_failure_remains_error(
    tmp_path: Path, image_artifact, monkeypatch: pytest.MonkeyPatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path, evaluation_mode="holistic")
    original = coordinator._invoke
    reason = "MODEL_SCHEMA_MISMATCH"

    def invoke(client, stage, payload, images, model, **kwargs):
        if stage == "evidence_extraction":
            raise ExecutionError(reason, "DUPLICATE_JSON_KEY: readable_text")
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


@pytest.mark.parametrize("mode", ["holistic", "detailed"])
@pytest.mark.parametrize("stage", ["_question_intent", "_rate_turn"])
@pytest.mark.parametrize(
    ("reason", "expected", "artifact_kind"),
    [
        ("MODEL_SCHEMA_MISMATCH", "ABSTAINED", "model-output-abstentions"),
        ("MODEL_HTTP_STATUS", "ERROR", "errors"),
    ],
)
def test_rerating_keeps_failures_with_their_turn_and_continues_other_conversations(
    tmp_path, image_artifact, monkeypatch, mode, stage, reason, expected, artifact_kind
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path, evaluation_mode=mode)
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 2)
    try:
        original = coordinator.synthesize_image(image, root)
        assert len(original.turns) == 2
        monkeypatch.setattr(coordinator, "_question_intent", lambda *args: GateVerdict.MET)
        monkeypatch.setattr(coordinator, "_question_fit", lambda *args: GateVerdict.MET)
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
    requirement_text: str | None = None,
    disagree_on_requirements: bool = False,
    concurrency_probe: ConcurrencyProbe | None = None,
    repeat_question: bool = False,
    echo_question_prompt: bool = False,
    internal_reference_question: Literal["none", "first", "always", "selected_region"] = "none",
    echo_answer_prompt: bool = False,
    unknown_capability: bool = False,
    evaluation_mode: str = "detailed",
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={"evaluation": EvaluationConfig.model_validate({"mode": evaluation_mode})}
    )
    store = RunStore(tmp_path / "runs", "test", require_local_wal=False)
    store.initialize_run("test", config.config_hash, config.profile)
    selector = ScriptedClient(
        config.models.active_selector_endpoint,
        reject_selection=reject_selection,
        concurrency_probe=concurrency_probe,
    )
    generator_a = ScriptedClient(
        config.models.generator_a,
        fail_first_rating=fail_first_rating,
        requirement_text=requirement_text,
        concurrency_probe=concurrency_probe,
        repeat_question=repeat_question,
        echo_question_prompt=echo_question_prompt,
        internal_reference_question=internal_reference_question,
        echo_answer_prompt=echo_answer_prompt,
        unknown_capability=unknown_capability,
    )
    generator_b = ScriptedClient(
        config.models.generator_b,
        fail_first_rating=fail_first_rating,
        requirement_text="visible region" if disagree_on_requirements else requirement_text,
        concurrency_probe=concurrency_probe,
        repeat_question=repeat_question,
        echo_question_prompt=echo_question_prompt,
        internal_reference_question=internal_reference_question,
        echo_answer_prompt=echo_answer_prompt,
        unknown_capability=unknown_capability,
    )
    generator_b.intent_by_question = generator_a.intent_by_question
    coordinator = SynthesisCoordinator(
        config,
        "test",
        store,
        selector,
        generator_a,
        generator_b,
    )
    return coordinator, store, selector, generator_a, generator_b


def _attribute_recheck_inventory() -> ScopedEvidenceInventory:
    region = ImageRegion(left=0.1, top=0.1, right=0.8, bottom=0.8)
    return ScopedEvidenceInventory(
        image_id="image-1",
        reason="One visible subject",
        scopes=(
            ScopeEvidence(
                scope_id="subject",
                view_id="full:test",
                public_description="A dog beside a person",
                object_label="dog",
                region=region,
                observations=(
                    CapabilityObservation(
                        evidence_id="entity-1",
                        capability="visible_entity",
                        verdict="MET",
                        region=region,
                        detail="One dog is visible",
                    ),
                    CapabilityObservation(
                        evidence_id="interaction-1",
                        capability="visible_interaction",
                        verdict="MET",
                        region=region,
                        detail="The dog stands beside a person",
                    ),
                ),
            ),
        ),
    )


def test_attribute_recheck_adds_only_view_bound_visible_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator, store, _, generator, _ = _coordinator(tmp_path)
    coordinator.config = coordinator.config.model_copy(
        update={
            "tasks": coordinator.config.tasks.model_copy(update={"attribute_recheck_enabled": True})
        }
    )
    inventory = _attribute_recheck_inventory()
    model_image = ModelImage("full:test", tmp_path / "image.png", "digest", "image/png")
    calls: list[dict[str, Any]] = []

    def respond(client, stage, payload, images, model, **kwargs):
        del client, kwargs
        assert stage == "attribute_recheck"
        assert model is AttributeRecheckReport
        assert images == (model_image,)
        calls.append(payload)
        return AttributeRecheckReport(
            image_id="image-1",
            scope_id="subject",
            view_id="full:test",
            verdict="MET",
            region=ImageRegion(left=0.2, top=0.2, right=0.5, bottom=0.5),
            detail="White fur is visible",
        )

    monkeypatch.setattr(coordinator, "_invoke", respond)
    try:
        enriched = coordinator._maybe_recheck_attribute(
            inventory, generator, model_image, [{"view_id": "full:test"}]
        )
        assert len(calls) == 1
        assert set(calls[0]) == {
            "image_id",
            "scope_id",
            "view_id",
            "scope_region",
            "scope_description",
            "image_views",
        }
        assert "candidate_answer" not in str(calls[0])
        assert len(enriched.scopes[0].observations) == 3
        assert enriched.scopes[0].observations[-1].capability == "visible_attribute"
        assert enriched.scopes[0].observations[-1].verdict == "MET"
        assert len({item.evidence_id for item in enriched.scopes[0].observations}) == 3
    finally:
        store.close()


@pytest.mark.parametrize("invalid", ["wrong_view", "outside_scope", "unknown"])
def test_attribute_recheck_never_promotes_invalid_or_unknown_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    coordinator, store, _, generator, _ = _coordinator(tmp_path)
    coordinator.config = coordinator.config.model_copy(
        update={
            "tasks": coordinator.config.tasks.model_copy(update={"attribute_recheck_enabled": True})
        }
    )
    inventory = _attribute_recheck_inventory()
    model_image = ModelImage("full:test", tmp_path / "image.png", "digest", "image/png")

    def respond(*args, **kwargs):
        del args, kwargs
        return AttributeRecheckReport(
            image_id="image-1",
            scope_id="subject",
            view_id="other-view" if invalid == "wrong_view" else "full:test",
            verdict="UNKNOWN" if invalid == "unknown" else "MET",
            region=ImageRegion(
                left=0.2,
                top=0.2,
                right=0.9 if invalid == "outside_scope" else 0.5,
                bottom=0.5,
            ),
            detail="Property cannot be verified" if invalid == "unknown" else "White fur",
        )

    monkeypatch.setattr(coordinator, "_invoke", respond)
    try:
        result = coordinator._maybe_recheck_attribute(inventory, generator, model_image, [])
        added = result.scopes[0].observations[2:]
        assert not any(item.verdict == "MET" for item in added)
        assert len(added) == (1 if invalid == "unknown" else 0)
    finally:
        store.close()


def test_attribute_recheck_is_disabled_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator, store, _, generator, _ = _coordinator(tmp_path)
    inventory = _attribute_recheck_inventory()
    model_image = ModelImage("full:test", tmp_path / "image.png", "digest", "image/png")

    def unexpected(*args, **kwargs):
        raise AssertionError("disabled attribute recheck called the model")

    monkeypatch.setattr(coordinator, "_invoke", unexpected)
    try:
        assert (
            coordinator._maybe_recheck_attribute(inventory, generator, model_image, []) == inventory
        )
    finally:
        store.close()


def test_attribute_recheck_reports_transport_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coordinator, store, _, generator, _ = _coordinator(tmp_path)
    coordinator.config = coordinator.config.model_copy(
        update={
            "tasks": coordinator.config.tasks.model_copy(update={"attribute_recheck_enabled": True})
        }
    )
    model_image = ModelImage("full:test", tmp_path / "image.png", "digest", "image/png")

    def unavailable(*args, **kwargs):
        raise ExecutionError("MODEL_TRANSPORT_FAILED", "server unavailable")

    monkeypatch.setattr(coordinator, "_invoke", unavailable)
    try:
        with pytest.raises(ExecutionError) as caught:
            coordinator._maybe_recheck_attribute(
                _attribute_recheck_inventory(), generator, model_image, []
            )
        assert caught.value.reason == "MODEL_TRANSPORT_FAILED"
    finally:
        store.close()


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


def test_selection_rejection_stops_before_question_or_answer(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, selector, generator_a, generator_b = _coordinator(
        tmp_path, reject_selection=True
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    stages = [stage for client in (generator_a, generator_b) for stage, _ in client.calls]
    binding_payloads = [
        payload
        for client in (generator_a, generator_b)
        for stage, payload in client.calls
        if stage == "candidate_binding"
    ]
    assert binding_payloads
    assert all(
        "local_evidence_ids" in candidate and "required_evidence_ids" in candidate
        for payload in binding_payloads
        for candidate in payload["candidates"]
    )
    assert conversation.status == "REJECTED"
    assert not conversation.turns
    assert "answer_generation" not in stages
    assert "question_generation" not in stages
    assert [stage for stage, _ in selector.calls] == ["instruction_selection"]


def test_normalized_repeated_question_stops_before_second_fit(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        repeat_question=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
        rejection_count = store.connection.execute(
            "SELECT COUNT(*) FROM artifact WHERE kind = 'public-text-rejections'"
        ).fetchone()[0]
    finally:
        store.close()

    assert conversation.status == "REJECTED"
    assert len(conversation.turns) == 1
    assert rejection_count == 1
    assert sum(stage == "question_fit" for stage, _ in generator_a.calls) == 1
    assert sum(stage == "question_fit" for stage, _ in generator_b.calls) == 1
    assert any(
        feedback and "repeats an answered question" in feedback
        for client in (generator_a, generator_b)
        for feedback in client.retry_feedback
    )


def test_answered_identification_is_removed_before_instruction_selection(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        conversation = coordinator.synthesize_image(image, root)
        first = conversation.turns[0]
        prior = first.model_copy(
            update={
                "instruction": first.instruction.model_copy(update={"scope_id": "animal"}),
                "answer": first.answer.model_copy(update={"content": "A monkey is visible."}),
            }
        )
        candidate = InstructionCandidate(
            candidate_id="identify-animal",
            task_id="object_identification",
            family="visual_semantics",
            visible_scope="the animal",
            instruction_summary="Identify the animal",
            required_capabilities=("visible_entity",),
            scope_id="animal",
            public_parameters=(
                PublicParameter(
                    name="target",
                    value="monkey",
                    origin="image",
                    evidence_refs=("animal:visible_entity",),
                ),
            ),
        )
        other_scope = candidate.model_copy(
            update={"candidate_id": "identify-other", "scope_id": "other"}
        )
        kept = coordinator._drop_answered_candidates(
            (candidate, other_scope), (prior,), "conversation", 2
        )
        row = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'candidate-admission-rejections'"
        ).fetchone()
        rejection = json.loads(store.read_artifact(row[0]))
    finally:
        store.close()

    assert kept == (other_scope,)
    assert rejection["candidate_id"] == "identify-animal"
    assert rejection["reason"] == "IDENTIFICATION_ANSWER_ALREADY_PUBLIC"


def test_answered_attribute_is_removed_before_selection_but_new_part_remains(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        first = coordinator.synthesize_image(image, root).turns[0]

        def attribute_candidate(candidate_id: str, target: str, attribute: str):
            return InstructionCandidate(
                candidate_id=candidate_id,
                task_id="attribute_lookup",
                family="visual_description",
                visible_scope="the dog",
                instruction_summary="Report a visible property",
                required_capabilities=("visible_entity", "visible_attribute"),
                scope_id="dog",
                view_id="view",
                public_parameters=(
                    PublicParameter(name="target", value=target, origin="instruction"),
                    PublicParameter(name="attribute", value=attribute, origin="instruction"),
                ),
            )

        prior = first.model_copy(
            update={"instruction": attribute_candidate("prior", "The dog", "fur color")}
        )
        repeat = attribute_candidate("repeat", "dog", "fur colour")
        novel = attribute_candidate("novel", "dog", "nose color")
        kept = coordinator._drop_answered_candidates((repeat, novel), (prior,), "conversation", 2)
        row = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind = 'candidate-admission-rejections'"
        ).fetchone()
        rejection = json.loads(store.read_artifact(row[0]))
    finally:
        store.close()

    assert kept == (novel,)
    assert rejection["candidate_id"] == "repeat"
    assert rejection["reason"] == "ATTRIBUTE_FACT_ALREADY_PUBLIC"


def test_identification_color_descriptor_blocks_only_the_same_public_fact(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    try:
        first = coordinator.synthesize_image(image, root).turns[0]
        identified = first.model_copy(
            update={
                "instruction": first.instruction.model_copy(
                    update={
                        "task_id": "object_identification",
                        "scope_id": "fruit",
                        "view_id": "view",
                    }
                ),
                "question": first.question.model_copy(
                    update={"content": "What is this red fruit?"}
                ),
                "answer": first.answer.model_copy(update={"content": "strawberry"}),
                "status": "COMMITTED",
            }
        )

        def attribute(name: str, value: str) -> InstructionCandidate:
            return InstructionCandidate(
                candidate_id=name,
                task_id="attribute_lookup",
                family="visual_description",
                visible_scope="the fruit",
                instruction_summary="Report a visible property",
                required_capabilities=("visible_entity", "visible_attribute"),
                scope_id="fruit",
                view_id="view",
                public_parameters=(
                    PublicParameter(name="target", value="strawberry", origin="instruction"),
                    PublicParameter(name="attribute", value=value, origin="instruction"),
                ),
            )

        color = attribute("color", "color")
        seeds = attribute("seeds", "seed color")
        assert coordinator._drop_answered_candidates((color, seeds), (identified,), "c", 2) == (
            seeds,
        )
        context = identified.model_copy(
            update={
                "question": identified.question.model_copy(
                    update={"content": "What fruit is beside this red cup?"}
                )
            }
        )
        assert coordinator._drop_answered_candidates((color,), (context,), "c", 2) == (color,)
    finally:
        store.close()


def test_identification_question_must_not_contain_its_target_or_answer():
    assert identification_label_in_question("What is this gibbon sitting on a railing?", "gibbon")
    assert identification_label_in_question(
        "What is the name of this silver race car?", "silver race car"
    )
    assert identification_label_in_question(
        "What is the name of this silver vintage race car?", "silver race car"
    )
    assert identification_answer_in_question("What object is this red strawberry?", "A strawberry.")
    assert identification_answer_in_question(
        "What is this silver vintage race car?", "silver race car"
    )
    assert not identification_answer_in_question("What insect is visible on the fabric?", "ladybug")
    assert not identification_label_in_question("What kind of animal is visible?", "animal")
    assert identification_answer_in_question("What kind of animal is visible?", "animal")
    assert not identification_label_in_question("What insect is visible?", "in")


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
    coordinator, store, _, _, _ = _coordinator(tmp_path, evaluation_mode="holistic")
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
        monkeypatch.setattr(coordinator, "_question_intent", lambda *args: GateVerdict.MET)
        monkeypatch.setattr(coordinator, "_question_fit", lambda *args: GateVerdict.MET)
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
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
    try:
        original = coordinator.synthesize_image(image, root)
        first = original.turns[0]
        affordance = first.model_copy(
            update={
                "instruction": first.instruction.model_copy(
                    update={"task_id": "visible_action_relation"}
                ),
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


def test_identification_target_leak_retries_before_answer(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, **kwargs):
            response = _original(stage, payload, *args, **kwargs)
            if (
                stage == "question_generation"
                and payload["selected_instruction"]["task_id"] == "object_identification"
            ):
                assert all(
                    parameter["name"] != "target"
                    for parameter in payload["selected_instruction"]["public_parameters"]
                )
                target = f"the blue region, aspect {len(payload['public_history'])}"
                return ModelResponse(
                    value=TextPayload(text=f"What is {target}?"),
                    request_hash=response.request_hash,
                    response_hash=response.response_hash,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                )
            return response

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
        stage == "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_identification_answer_echo_stops_before_answer_rating(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _client=client, _original=original, **kwargs):
            response = _original(stage, payload, *args, **kwargs)
            if stage == "question_generation":
                question = "What is this gibbon?"
                _client.intent_by_question[(payload["image_views"][0]["view_id"], question)] = (
                    payload["selected_instruction"]["task_id"]
                )
                value = TextPayload(text=question)
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
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
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
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
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
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
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
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
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


def test_blind_question_intent_rejects_wrong_operation_before_answer(
    tmp_path: Path, image_artifact, monkeypatch
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path, evaluation_mode="holistic"
    )
    for client in (generator_a, generator_b):
        original = client.invoke

        def invoke(stage, payload, *args, _original=original, **kwargs):
            if stage == "question_intent":
                assert "selected_instruction" not in payload
                assert "candidate_answer" not in payload
                assert len(payload["task_definitions"]) == 72
                return ModelResponse(
                    value=QuestionIntent(task_id="spatial_relation", reason="Wrong operation."),
                    request_hash="request",
                    response_hash="response",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return _original(stage, payload, *args, **kwargs)

        monkeypatch.setattr(client, "invoke", invoke)
    try:
        result = coordinator.synthesize_image(image, root)
    finally:
        store.close()
    assert result.status == "REJECTED"
    assert all(
        stage != "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_private_prompt_echo_is_rejected_before_question_fit(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        echo_question_prompt=True,
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
    assert all(
        stage != "question_fit"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_internal_reference_is_corrected_before_question_fit(
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
    assert any(
        feedback and "controller reference" in feedback
        for client in (generator_a, generator_b)
        for feedback in client.retry_feedback
    )


@pytest.mark.parametrize("reference_mode", ["always", "selected_region"])
def test_persistent_internal_reference_stops_before_question_fit(
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
        stage != "question_fit"
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
        stage != "claim_inventory"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_unknown_evidence_capability_is_not_silently_ignored(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path, unknown_capability=True)
    try:
        with pytest.raises(ExecutionError) as caught:
            coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert caught.value.reason == "EVIDENCE_CAPABILITY_UNKNOWN"


def test_later_turn_does_not_imply_binding_or_strong_dependency() -> None:
    context = RubricContext(
        turn_index=2,
        profile="normal",
        has_natural_language_answer=True,
    )
    templates = {item["template_id"] for item in applicable_rubric_items(context)}
    assert {"H_CONSISTENCY", "H_TURN_PROGRESS"} <= templates
    assert "H_BINDING" not in templates
    assert "H_WITNESS" not in templates

    dependent = context.model_copy(
        update={"history_binding_ids": ("binding-1",), "requires_witness_check": True}
    )
    dependent_templates = {item["template_id"] for item in applicable_rubric_items(dependent)}
    assert {"H_BINDING", "H_WITNESS"} <= dependent_templates


def test_numeric_only_answer_skips_natural_language_criteria() -> None:
    assert not has_natural_language_content("42")
    assert has_natural_language_content("42 kg")
    context = RubricContext(
        turn_index=1,
        profile="normal",
        has_natural_language_answer=False,
    )
    templates = {item["template_id"] for item in applicable_rubric_items(context)}
    assert "F_A_CLEAR" not in templates
    assert "F_REDUNDANCY" not in templates
    assert "L_A_TARGET" not in templates


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
    for turn in conversation.turns:
        prior = 2 * (turn.turn_index - 1)
        selector_payload = selector.calls[turn.turn_index - 1][1]
        assert len(selector_payload["public_history"]) == prior
        assert "candidate_answer" not in selector_payload
    for judge in (generator_a, generator_b):
        assert any(stage == "question_fit" for stage, _ in judge.calls)
        assert any(stage == "rubric_item" for stage, _ in judge.calls)
        assert all("other_judge" not in payload for _, payload in judge.calls)


def test_answer_repair_uses_generator_and_repeats_all_evaluation(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path, fail_first_rating=True)
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    generation_client = next(
        client
        for client in (generator_a, generator_b)
        if any(stage == "answer_generation" for stage, _ in client.calls)
    )
    other_client = generator_b if generation_client is generator_a else generator_a
    assert conversation.status == "QUALITY_CANDIDATE"
    assert sum(stage == "answer_repair" for stage, _ in generation_client.calls) == 1
    assert all(stage != "answer_repair" for stage, _ in other_client.calls)
    assert sum(stage == "claim_inventory" for stage, _ in generator_a.calls) >= 2
    assert sum(stage == "claim_inventory" for stage, _ in generator_b.calls) >= 2


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


def test_public_requirement_is_fixed_before_answer_and_rated_separately(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        requirement_text="color",
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert conversation.status == "QUALITY_CANDIDATE"
    assert all(turn.requirements for turn in conversation.turns)
    generation_client = next(
        client
        for client in (generator_a, generator_b)
        if any(stage == "answer_generation" for stage, _ in client.calls)
    )
    generation_stages = [stage for stage, _ in generation_client.calls]
    assert generation_stages.index("requirement_extraction") < generation_stages.index(
        "answer_generation"
    )
    for client in (generator_a, generator_b):
        requirement_payloads = [
            payload
            for stage, payload in client.calls
            if stage == "rubric_item" and "target_requirement" in payload
        ]
        assert requirement_payloads


def test_requirement_disagreement_abstains_before_answer(tmp_path: Path, image_artifact) -> None:
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(
        tmp_path,
        requirement_text="color",
        disagree_on_requirements=True,
    )
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()

    assert conversation.status == "ABSTAINED"
    assert not conversation.turns
    assert all(
        stage != "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


def test_claim_offsets_are_corrected_only_for_a_unique_quoted_span() -> None:
    answer = PublicMessage(
        message_id="a1",
        turn_index=1,
        role="assistant",
        content="There are five squares.",
    )
    claim = AtomicClaim(
        text="five squares",
        source_message_id="a1",
        start=0,
        end=99,
    )
    normalized = SynthesisCoordinator._normalize_claim(claim, answer)
    assert normalized is not None
    assert (normalized.start, normalized.end) == (10, 22)

    repeated = answer.model_copy(update={"content": "five squares and five squares"})
    assert SynthesisCoordinator._normalize_claim(claim, repeated) is None


def test_incomplete_extraction_abstains_without_generating_answer(
    tmp_path, image_artifact, monkeypatch
):
    image, root = image_artifact
    coordinator, store, _, generator_a, generator_b = _coordinator(tmp_path)
    original = coordinator._invoke

    def invoke(client, stage, payload, images, model, **kwargs):
        if model is RequirementInventory:
            return RequirementInventory(
                requirements=(),
                extraction_complete=False,
                reason="Unable to extract all explicit requirements.",
            )
        return original(client, stage, payload, images, model, **kwargs)

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        conversation = coordinator.synthesize_image(image, root)
    finally:
        store.close()
    assert conversation.status == "ABSTAINED"
    assert not conversation.turns
    assert all(
        stage != "answer_generation"
        for client in (generator_a, generator_b)
        for stage, _ in client.calls
    )


@pytest.mark.parametrize("always_wrong", [False, True])
def test_evidence_identity_is_regenerated_or_fails_closed(
    tmp_path: Path, always_wrong: bool
) -> None:
    class WrongImageClient(ScriptedClient):
        attempts = 0

        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            response = super().invoke(*args, **kwargs)
            self.attempts += 1
            if always_wrong or self.attempts == 1:
                return ModelResponse(
                    value=response.value.model_copy(update={"image_id": "wrong-image"}),
                    request_hash=response.request_hash,
                    response_hash=response.response_hash,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return response

    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path / "runs", "identity", require_local_wal=False) as store:
        client = WrongImageClient(config.models.generator_a)
        co = SynthesisCoordinator(config, "identity", store, client, client, client)

        def invoke() -> EvidenceInventory:
            return co._invoke(
                client,
                "evidence_extraction",
                {"image_id": "expected-image"},
                (),
                EvidenceInventory,
                max_tokens=1024,
                temperature=0.0,
                seed=1,
            )

        if always_wrong:
            with pytest.raises(ExecutionError, match="another image"):
                invoke()
            assert client.attempts == config.runtime.structured_output_max_attempts
        else:
            assert invoke().image_id == "expected-image"
            assert client.attempts == 2
        assert "expected-image" in (client.retry_feedback[1] or "")
        assert "regenerate the complete evidence" in (client.retry_feedback[1] or "")


def test_coverage_receives_verified_inventory_and_ids_stay_controller_owned(
    tmp_path: Path, image_artifact
) -> None:
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, requirement_text="color")
    try:
        conversation = co.synthesize_image(image, root)
        assert conversation.status == "QUALITY_CANDIDATE"
        for client in (a, b):
            for stage, payload in client.calls:
                if stage == "requirement_extraction":
                    assert payload["question_message_id"].startswith("m")
                    assert len(payload["question_message_id"]) < 5
                if stage == "claim_inventory":
                    assert "candidate_answer_message_id" not in payload
                if stage == "rubric_item" and payload["criterion"]["template_id"] == "C_COVERAGE":
                    assert "image_views" not in payload
                    assert payload["candidate_claim_inventory"]
                    for claim in payload["candidate_claim_inventory"]:
                        assert claim["source_message_id"].startswith(conversation.conversation_id)
                        assert (
                            payload["candidate_answer"][claim["start"] : claim["end"]]
                            == claim["text"]
                        )
        for turn in conversation.turns:
            assert all(r.source_message_id == turn.question.message_id for r in turn.requirements)
    finally:
        store.close()


@pytest.mark.parametrize("always_invalid", [False, True])
def test_invalid_claim_boundaries_are_retried_without_guessing(
    tmp_path: Path, always_invalid: bool
) -> None:
    class InvalidQuoteClient(ScriptedClient):
        attempts = 0

        def invoke(self, *args: Any, **kwargs: Any) -> ModelResponse:
            response = super().invoke(*args, **kwargs)
            self.attempts += 1
            if always_invalid or self.attempts == 1:
                value = ClaimExtraction(
                    claims=(ClaimSpan(start_token=0, end_token=9999),),
                    coverage=GateVerdict.MET,
                    reason="Complete",
                )
                return ModelResponse(
                    value=value,
                    request_hash="1" * 64,
                    response_hash="2" * 64,
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            return response

    config = load_config(Path("configs/pilot.yaml"))
    with RunStore(tmp_path / "runs", "quote", require_local_wal=False) as store:
        client = InvalidQuoteClient(config.models.generator_a)
        co = SynthesisCoordinator(config, "quote", store, client, client, client)

        def invoke():
            return co._invoke(
                client,
                "claim_inventory",
                {"candidate_answer": "Blue."},
                (),
                ClaimExtraction,
                max_tokens=1024,
                temperature=0.0,
                seed=1,
            )

        if always_invalid:
            with pytest.raises(ExecutionError, match="token boundaries"):
                invoke()
        else:
            assert (
                co._bind_claim_span(
                    invoke().claims[0],
                    PublicMessage(message_id="a", turn_index=1, role="assistant", content="Blue."),
                ).text
                == "Blue."
            )
        assert client.attempts == 2


@pytest.mark.parametrize(
    "text", ['The cap says "TITANS SWIMWEAR".', "看板には「停止」と書いてあります。", "2"]
)
def test_token_boundaries_reconstruct_original_quotes_and_unicode(text: str) -> None:
    answer = PublicMessage(message_id="actual-answer", turn_index=1, role="assistant", content=text)
    tokens = SynthesisCoordinator._answer_tokens(text)
    claim = SynthesisCoordinator._bind_claim_span(
        ClaimSpan(start_token=0, end_token=len(tokens)), answer
    )
    assert claim.text == text
    assert claim.source_message_id == answer.message_id
    claim.validate_span(answer)


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


@pytest.mark.parametrize(
    ("votes", "set_verdict", "expected", "aggregate"),
    [
        (("NOT_MET", "NOT_MET"), None, "NOT_MET", "FAIL"),
        (("MET", "MET"), None, "UNKNOWN", "ABSTAIN"),
        (("MET", "NOT_MET"), None, "UNKNOWN", "ABSTAIN"),
        (("UNKNOWN", "UNKNOWN"), None, "UNKNOWN", "ABSTAIN"),
        (("NOT_MET", "NOT_MET"), "MET", "NOT_MET", "FAIL"),
        (("MET", "MET"), "MET", "MET", "PASS"),
        (("MET", "NOT_MET"), "MET", "UNKNOWN", "ABSTAIN"),
        (("UNKNOWN", "UNKNOWN"), "MET", "UNKNOWN", "ABSTAIN"),
        (("NOT_MET", "NOT_MET"), "NOT_MET", "NOT_MET", "FAIL"),
        (("MET", "MET"), "NOT_MET", "NOT_MET", "FAIL"),
        (("MET", "NOT_MET"), "NOT_MET", "NOT_MET", "FAIL"),
        (("UNKNOWN", "UNKNOWN"), "NOT_MET", "NOT_MET", "FAIL"),
    ],
)
def test_set_rating_preserves_failure_without_rescuing_uncertainty(
    tmp_path, image_artifact, monkeypatch, votes, set_verdict, expected, aggregate
):
    image, root = image_artifact
    coordinator, store, _, _, _ = _coordinator(tmp_path)
    template = next(
        t for t in load_rubric_catalog()["items"] if t["template_id"] == "C_SET_COMPLETE"
    )
    monkeypatch.setattr("pixelogue.pipeline.applicable_rubric_items", lambda context: [template])
    check = (
        None
        if set_verdict is None
        else SetCheck(verdict=set_verdict, missing=(), unexpected=(), duplicates=())
    )
    monkeypatch.setattr(coordinator, "_check_complete_set", lambda *args: check)
    remaining_votes = iter(votes)

    def invoke(client, stage, *args, **kwargs):
        if stage == "claim_inventory":
            return ClaimExtraction(
                claims=(ClaimSpan(start_token=0, end_token=1),),
                coverage=GateVerdict.MET,
                reason="The numeric answer is covered.",
            )
        assert stage == "rubric_item"
        return RubricVerdict(verdict=next(remaining_votes), reason="Independent semantic vote.")

    monkeypatch.setattr(coordinator, "_invoke", invoke)
    try:
        rating = coordinator._rate_turn(
            "test-conversation",
            "1" * 64,
            (),
            PublicMessage(message_id="q", turn_index=1, role="user", content="How many cords?"),
            PublicMessage(message_id="a", turn_index=1, role="assistant", content="2"),
            InstructionCandidate(
                candidate_id="count",
                task_id="count",
                family="visible_count",
                visible_scope="cords",
                instruction_summary="Count cords.",
                required_capabilities=(),
            ),
            "en",
            [],
            ModelImage(
                view_id=image.full_view.view_id,
                path=root / image.full_view.relative_path,
                encoded_sha256=image.full_view.encoded_sha256,
                media_type=image.full_view.media_type,
            ),
            1,
            (),
        )
        assert rating.items[0].verdict == expected
        assert rating.aggregate == aggregate
    finally:
        store.close()


def test_v7_holistic_checks_bound_question_and_two_blind_whole_turn_reviews(
    tmp_path, image_artifact
):
    image, root = image_artifact
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    try:
        conversation = co.synthesize_image(image, root)
        assert conversation.status == "QUALITY_CANDIDATE"
        for client in (a, b):
            stages = [stage for stage, _ in client.calls]
            assert stages.count("holistic_review") == len(conversation.turns)
            assert stages.count("question_fit") == len(conversation.turns)
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
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
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
    co, store, _, _, _ = _coordinator(tmp_path, evaluation_mode="holistic")
    co.config = co.config.model_copy(
        update={"evaluation": EvaluationConfig(mode="holistic", retain_accepted_prefix=False)}
    )
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 3)
    original = co._select_instruction
    monkeypatch.setattr(
        co, "_select_instruction", lambda *args: None if args[-1] == 3 else original(*args)
    )
    try:
        conversation = co.synthesize_image(image, root)
        assert len(conversation.turns) == 2
        assert conversation.status == "REJECTED"
    finally:
        store.close()


def test_holistic_rerating_rejects_empty_conversation(tmp_path, image_artifact):
    image, root = image_artifact
    co, store, _, _, _ = _coordinator(tmp_path, reject_selection=True, evaluation_mode="holistic")
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
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    monkeypatch.setattr("pixelogue.pipeline.planned_turn_count", lambda *args: 4)
    original = co._select_instruction
    monkeypatch.setattr(
        co, "_select_instruction", lambda *args: None if args[-1] == 3 else original(*args)
    )
    accepted = co.synthesize_image(image, root)
    assert accepted.status == "QUALITY_CANDIDATE" and len(accepted.turns) == 2
    store.close()
    with RunStore(tmp_path / "runs", "test", require_local_wal=False) as reopened:
        reopened.initialize_run("test", co.config.config_hash, co.config.profile)
        resumed = SynthesisCoordinator(co.config, "test", reopened, co.selector, a, b)
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
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
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
    co, store, _, a, b = _coordinator(tmp_path, evaluation_mode="holistic")
    monkeypatch.setattr(co, "_question_fit", lambda *args: GateVerdict.NOT_MET)
    try:
        result = co.synthesize_image(image, root)
        assert result.status == "REJECTED"
        assert not any(
            stage == "answer_generation" for client in (a, b) for stage, _ in client.calls
        )
    finally:
        store.close()
