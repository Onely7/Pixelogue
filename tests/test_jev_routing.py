"""Public routing contracts, strict visual admission and exact-trial recovery."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from pixelogue.catalog import task_catalog
from pixelogue.config import DecisionRoutingConfig, RuntimeConfig, TaskRuntimeConfig, load_config
from pixelogue.contracts import InstructionCandidate, PublicMessage, build_history_snapshot
from pixelogue.decision import DecisionLabel, DecisionProbability, DecisionRequest, DecisionResult
from pixelogue.errors import ExecutionError
from pixelogue.fact_identity import requested_fact_key
from pixelogue.jev_routing import (
    BindingProposal,
    BindingProposals,
    EvidenceProposals,
    EvidenceScopeProposal,
    ObservationProposal,
    RoutingDecision,
    StoredDecisionRouter,
    assemble_bindings,
    assemble_evidence,
    binding_decisions,
    controller_binding_proposals,
    evidence_decisions,
    partition_binding_proposals,
    validate_binding_proposals,
)
from pixelogue.pipeline import SynthesisCoordinator
from pixelogue.prompts import validate_stage_payload
from pixelogue.serving import ModelImage, ModelResponse
from pixelogue.store import RunStore
from pixelogue.task_evidence import (
    CapabilityObservation,
    ImageRegion,
    PublicParameter,
    TargetReport,
)
from pixelogue.task_runtime import bind_candidates_individually, validate_evidence

OMNI_REVISION = "5addda86ddee081a68fb067477ea100c221b8917"
REGION = ImageRegion(left=0.1, top=0.1, right=0.9, bottom=0.9)


def proposals() -> EvidenceProposals:
    return EvidenceProposals(
        scopes=(
            EvidenceScopeProposal(
                public_description="the visible dog",
                object_label="dog",
                region=REGION,
                observations=(
                    ObservationProposal(
                        capability="visible_entity", region=REGION, detail="A dog is visible"
                    ),
                    ObservationProposal(
                        capability="visible_attribute",
                        region=REGION,
                        detail="Fur color and nose color are visible",
                    ),
                ),
            ),
        )
    )


def inventory():
    return assemble_evidence(
        proposals(),
        {"scope_1/observation_1": "MET", "scope_1/observation_2": "MET"},
        image_id="image",
        view_id="full:test",
        settings=TaskRuntimeConfig(),
    )


def candidate(task_id: str, *, candidate_id: str = "candidate") -> InstructionCandidate:
    task = next(task for task in task_catalog().tasks if task.id == task_id)
    return InstructionCandidate(
        candidate_id=candidate_id,
        task_id=task_id,
        family=task.family,
        visible_scope="the visible dog",
        instruction_summary=task.definition_en,
        required_capabilities=task.required_capabilities,
        catalog_version="7.0",
        scope_id="scope_1",
        view_id="full:test",
        scope_region=REGION,
        evidence_refs=("obs_1_1", "obs_1_2"),
        verification_contracts=task.verification_contracts,
    )


def attribute_proposal(name: str = "nose color") -> BindingProposals:
    return BindingProposals(
        bindings=(
            BindingProposal(
                candidate_id="candidate",
                target=TargetReport(
                    value="the visible dog", origin="image", evidence_refs=("obs_1_1",)
                ),
                public_parameters=(
                    PublicParameter(name="attribute", value=name, origin="instruction"),
                ),
            ),
        )
    )


class FiniteClient:
    def __init__(self, verdict: DecisionLabel = "MET"):
        self.verdict, self.requests = verdict, []

    def decide(self, request: DecisionRequest) -> DecisionResult:
        self.requests.append(request)
        return DecisionResult(
            request_id=request.request_id,
            request_hash=request.identity("akhilaaa3/Jev-Omni", OMNI_REVISION),
            model_repository="akhilaaa3/Jev-Omni",
            model_revision=OMNI_REVISION,
            verdict=self.verdict,
            probabilities=tuple(
                DecisionProbability(
                    label=label,
                    probability=0.8 if label == self.verdict else 0.1,
                )
                for label in ("MET", "NOT_MET", "UNKNOWN")
            ),
            elapsed_seconds=0.01,
        )


class ProposalClient:
    def __init__(self, endpoint):
        self.endpoint, self.stages = endpoint, []

    def invoke(self, stage, payload, images, response_model, **kwargs):
        self.stages.append(stage)
        validate_stage_payload(stage, payload)
        if stage != "evidence_proposal":
            raise AssertionError(f"Unexpected generator stage {stage}")
        return ModelResponse(
            value=proposals(),
            request_hash="r",
            response_hash="s",
            prompt_tokens=10,
            completion_tokens=20,
        )


def routing_config(**flags):
    return DecisionRoutingConfig(
        model_repository="akhilaaa3/Jev-Omni", model_revision=OMNI_REVISION, **flags
    )


def model_image(image_artifact) -> ModelImage:
    image, root = image_artifact
    return ModelImage(
        view_id=image.full_view.view_id,
        path=root / image.full_view.relative_path,
        encoded_sha256=image.full_view.encoded_sha256,
        media_type="image/png",
    )


def test_defaults_leave_both_decision_routes_disabled():
    config = load_config(Path("configs/standard.yaml"))
    assert config.models.active_selector == "default"
    assert not config.tasks.decision_routing.evidence_enabled
    assert not config.tasks.decision_routing.binding_enabled


def test_enabled_route_requires_explicit_model_and_client(tmp_path):
    with pytest.raises(ValidationError, match="explicit repository and revision"):
        DecisionRoutingConfig(evidence_enabled=True)
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={
                    "decision_routing": routing_config(evidence_enabled=True),
                }
            )
        }
    )
    client = ProposalClient(config.models.generator_a)
    with RunStore(tmp_path, "missing") as store, pytest.raises(ExecutionError) as error:
        SynthesisCoordinator(config, "missing", store, client, client, client)
    assert error.value.reason == "DECISION_CLIENT_REQUIRED"


def test_proposal_rejects_duplicate_or_outside_claims():
    scope = proposals().scopes[0]
    with pytest.raises(ValidationError, match="unique"):
        EvidenceScopeProposal.model_validate(
            {**scope.model_dump(), "observations": (scope.observations[0],) * 2}
        )
    with pytest.raises(ValidationError, match="outside"):
        EvidenceScopeProposal.model_validate(
            {
                **scope.model_dump(),
                "observations": (
                    scope.observations[0].model_copy(
                        update={"region": ImageRegion(left=0.0, top=0.0, right=1.0, bottom=1.0)}
                    ),
                ),
            }
        )


def test_unknown_capability_is_rejected_before_classifier():
    proposal = EvidenceProposals(
        scopes=(
            proposals()
            .scopes[0]
            .model_copy(
                update={
                    "object_label": None,
                    "observations": (
                        ObservationProposal(capability="invented", region=REGION, detail="visible"),
                    ),
                }
            ),
        )
    )
    with pytest.raises(ExecutionError) as error:
        evidence_decisions(proposal, TaskRuntimeConfig())
    assert error.value.reason == "EVIDENCE_CAPABILITY_UNKNOWN"


def test_unknown_evidence_stays_unknown_and_unverified_label_is_removed():
    result = assemble_evidence(
        proposals(),
        {"scope_1/observation_1": "UNKNOWN", "scope_1/observation_2": "NOT_MET"},
        image_id="image",
        view_id="full:test",
        settings=TaskRuntimeConfig(),
    )
    assert [item.verdict for item in result.scopes[0].observations] == ["UNKNOWN", "NOT_MET"]
    assert result.scopes[0].object_label is None
    assert result.scopes[0].observations[0].detail == "A dog is visible"
    validate_evidence(result, "full:test", TaskRuntimeConfig())


def test_missing_decision_is_rejected_not_filled_with_met():
    with pytest.raises(ExecutionError) as error:
        assemble_evidence(
            proposals(),
            {"scope_1/observation_1": "MET"},
            image_id="image",
            view_id="full:test",
            settings=TaskRuntimeConfig(),
        )
    assert error.value.reason == "DECISION_RESULTS_MISMATCH"


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"candidate_id": "unknown"}, "CANDIDATE_BINDING_ID"),
        (
            {
                "public_parameters": (
                    PublicParameter(name="bogus", value="nose", origin="instruction"),
                )
            },
            "CANDIDATE_PARAMETER_UNKNOWN",
        ),
        ({"public_parameters": ()}, "CANDIDATE_PARAMETER_MISSING"),
        (
            {"target": TargetReport(value="dog", origin="image", evidence_refs=("other_scope",))},
            "CANDIDATE_PARAMETER_SOURCE",
        ),
        (
            {"target": TargetReport(value="dog", origin="history", evidence_refs=("future",))},
            "CANDIDATE_PARAMETER_SOURCE",
        ),
        (
            {"target": TargetReport(value="dog", origin="instruction", evidence_refs=("obs_1_1",))},
            "CANDIDATE_PARAMETER_SOURCE",
        ),
        (
            {"target": TargetReport(value="dog", origin="image")},
            "CANDIDATE_PARAMETER_SOURCE",
        ),
        (
            {"target": TargetReport(value="dog", origin="history")},
            "CANDIDATE_PARAMETER_SOURCE",
        ),
    ],
)
def test_binding_proposal_rejects_unknown_names_and_references(change, reason):
    proposal = BindingProposals(
        bindings=(attribute_proposal().bindings[0].model_copy(update=change),)
    )
    with pytest.raises(ExecutionError) as error:
        validate_binding_proposals(proposal, (candidate("attribute_lookup"),), inventory(), ())
    assert error.value.reason == reason


def test_duplicate_bindings_and_wrong_view_are_rejected():
    proposal = attribute_proposal()
    with pytest.raises(ExecutionError) as error:
        validate_binding_proposals(
            BindingProposals(bindings=proposal.bindings * 2),
            (candidate("attribute_lookup"),),
            inventory(),
            (),
        )
    assert error.value.reason == "CANDIDATE_BINDING_ID"
    with pytest.raises(ExecutionError) as error:
        binding_decisions(
            proposal,
            (candidate("attribute_lookup").model_copy(update={"view_id": "crop:other"}),),
            inventory(),
            (),
        )
    assert error.value.reason == "EVIDENCE_VIEW_MISMATCH"


def test_bad_parameter_proposal_keeps_good_sibling_and_records_reason():
    good = attribute_proposal().bindings[0]
    bad = good.model_copy(update={"candidate_id": "bad", "public_parameters": ()})
    usable, rejected = partition_binding_proposals(
        BindingProposals(bindings=(bad, good)),
        (candidate("attribute_lookup", candidate_id="bad"), candidate("attribute_lookup")),
        inventory(),
        (),
    )
    assert usable.bindings == (good,)
    assert rejected[0].candidate_id == "bad"
    assert rejected[0].reason == "CANDIDATE_PARAMETER_MISSING"


@pytest.mark.parametrize(
    "target",
    [
        TargetReport(value="dog", origin="instruction", evidence_refs=("obs_1_1",)),
        TargetReport(value="dog", origin="image"),
        TargetReport(value="dog", origin="history"),
    ],
)
def test_inconsistent_target_source_rejects_only_affected_proposal(target):
    good = attribute_proposal().bindings[0]
    bad = good.model_copy(update={"candidate_id": "bad", "target": target})
    usable, rejected = partition_binding_proposals(
        BindingProposals(bindings=(bad, good)),
        (candidate("attribute_lookup", candidate_id="bad"), candidate("attribute_lookup")),
        inventory(),
        (),
    )
    assert usable.bindings == (good,)
    assert [(item.candidate_id, item.reason) for item in rejected] == [
        ("bad", "CANDIDATE_PARAMETER_SOURCE")
    ]
    assert bad.target == target


@pytest.mark.parametrize("bad_id", ["unknown", "candidate"])
def test_global_proposal_identifier_error_rejects_response_before_local_target_error(bad_id):
    good = attribute_proposal().bindings[0]
    bad = good.model_copy(
        update={
            "candidate_id": bad_id,
            "target": TargetReport(value="dog", origin="instruction", evidence_refs=("obs_1_1",)),
        }
    )
    with pytest.raises(ExecutionError) as error:
        partition_binding_proposals(
            BindingProposals(bindings=(bad, good)),
            (candidate("attribute_lookup"),),
            inventory(),
            (),
        )
    assert error.value.reason == "CANDIDATE_BINDING_ID"


def test_every_visual_check_is_independent_and_unknown_blocks_admission():
    template, proposal = candidate("attribute_lookup"), attribute_proposal()
    queries = binding_decisions(proposal, (template,), inventory(), ())
    assert len(queries) == 2
    verdicts: dict[str, DecisionLabel] = {query.query_id: "MET" for query in queries}
    verdicts[queries[-1].query_id] = "UNKNOWN"
    bindings = assemble_bindings(
        proposal, verdicts, (template,), inventory(), (), answer_max_tokens=1024
    )
    result = bind_candidates_individually(
        (template,), bindings, inventory(), (), TaskRuntimeConfig()
    )
    assert not result.admitted
    assert {check.check_id for check in bindings.bindings[0].checks} == {
        "scope_resolved",
        "attribute_visible",
    }


def test_controller_typed_object_reference_preserves_strict_existing_validator():
    template = candidate("object_identification")
    proposal, unresolved = controller_binding_proposals((template,), inventory())
    assert not unresolved
    assert proposal.bindings[0].target.value == "ref:obs_1_1"
    queries = binding_decisions(proposal, (template,), inventory(), ())
    bindings = assemble_bindings(
        proposal,
        {query.query_id: "MET" for query in queries},
        (template,),
        inventory(),
        (),
        answer_max_tokens=1024,
    )
    result = bind_candidates_individually(
        (template,), bindings, inventory(), (), TaskRuntimeConfig()
    )
    assert len(result.admitted) == 1
    assert result.admitted[0].public_parameters[0].value == "dog"


def test_distinct_properties_keep_distinct_fact_identity():
    keys = []
    template = candidate("attribute_lookup")
    for property_name in ("fur color", "nose color", "nose colour"):
        proposal = attribute_proposal(property_name)
        queries = binding_decisions(proposal, (template,), inventory(), ())
        bindings = assemble_bindings(
            proposal,
            {query.query_id: "MET" for query in queries},
            (template,),
            inventory(),
            (),
            answer_max_tokens=1024,
        )
        admitted = bind_candidates_individually(
            (template,), bindings, inventory(), (), TaskRuntimeConfig()
        ).admitted
        keys.append(requested_fact_key(admitted[0]))
    assert keys[0] != keys[1]
    assert keys[1] == keys[2]


def test_finite_binding_judge_sees_public_history_and_claims_without_other_verdicts():
    history = (
        PublicMessage(message_id="q1", turn_index=1, role="user", content="What is the fur color?"),
        PublicMessage(message_id="a1", turn_index=1, role="assistant", content="Brown."),
    )
    queries = binding_decisions(
        attribute_proposal(), (candidate("attribute_lookup"),), inventory(), history
    )
    assert [
        json.loads(item.removeprefix("Committed public message: "))
        for item in queries[0].public_context[:2]
    ] == [message.model_dump(mode="json") for message in history]
    assert "verdict=" not in queries[0].question
    assert "gold" not in queries[0].question


@pytest.mark.parametrize("stage", ["evidence_proposal", "candidate_proposal"])
def test_short_proposal_stage_rejects_gold_and_candidate_answer(stage):
    with pytest.raises(ExecutionError) as error:
        validate_stage_payload(stage, {"candidate_answer": "secret"})
    assert error.value.reason == "MODEL_INFORMATION_LEAK"
    with pytest.raises(ExecutionError):
        validate_stage_payload(stage, {"gold_label": "MET"})


def test_same_trial_reuses_saved_decision_after_store_reopen(tmp_path, image_artifact):
    client = FiniteClient("UNKNOWN")
    image = model_image(image_artifact)
    query = RoutingDecision("claim1", "Is the dog visible?", REGION)
    with RunStore(tmp_path, "recover") as store:
        router = StoredDecisionRouter(
            client, routing_config(evidence_enabled=True), RuntimeConfig(), store
        )
        assert router.decide(query, image_id="image", image=image, trial_id="trial-1") == "UNKNOWN"
        usage = store.connection.execute("SELECT usage_status FROM model_call_attempt").fetchone()[
            0
        ]
        assert usage == "MISSING"
    with RunStore(tmp_path, "recover") as store:
        router = StoredDecisionRouter(
            client, routing_config(evidence_enabled=True), RuntimeConfig(), store
        )
        assert router.decide(query, image_id="image", image=image, trial_id="trial-1") == "UNKNOWN"
        assert len(client.requests) == 1
        router.decide(query, image_id="image", image=image, trial_id="trial-2")
        router.decide(
            RoutingDecision("claim1", query.question, REGION, ("user: different history",)),
            image_id="image",
            image=image,
            trial_id="trial-1",
        )
        assert len(client.requests) == 3


def test_saved_decision_rejects_changed_image_bytes(tmp_path, image_artifact):
    image = model_image(image_artifact)
    with RunStore(tmp_path, "changed") as store:
        router = StoredDecisionRouter(
            FiniteClient(), routing_config(evidence_enabled=True), RuntimeConfig(), store
        )
        query = RoutingDecision("claim1", "Visible?", REGION)
        router.decide(query, image_id="image", image=image, trial_id="trial")
        image.path.write_bytes(b"changed")
        with pytest.raises(ExecutionError) as error:
            router.decide(query, image_id="image", image=image, trial_id="trial")
        assert error.value.reason == "IMAGE_BYTES_MISMATCH"


def test_wrong_model_response_is_journaled_as_invalid_and_cannot_be_reused_as_success(
    tmp_path, image_artifact
):
    class WrongIdentityClient(FiniteClient):
        def decide(self, request):
            return super().decide(request).model_copy(update={"model_revision": "a" * 40})

    client = WrongIdentityClient()
    query = RoutingDecision("claim1", "Visible?", REGION)
    with RunStore(tmp_path, "invalid") as store:
        router = StoredDecisionRouter(
            client, routing_config(evidence_enabled=True), RuntimeConfig(), store
        )
        for _ in range(2):
            with pytest.raises(ExecutionError) as error:
                router.decide(
                    query, image_id="image", image=model_image(image_artifact), trial_id="trial"
                )
            assert error.value.reason == "DECISION_RESULT_IDENTITY"
        assert len(client.requests) == 1
        row = store.connection.execute(
            "SELECT status, response_artifact_hash FROM model_call_attempt"
        ).fetchone()
        assert row["status"] == "INVALID"
        assert row["response_artifact_hash"] is not None


def test_failed_decision_can_resume_without_treating_failure_as_unknown_or_met(
    tmp_path, image_artifact
):
    class InterruptedClient(FiniteClient):
        def __init__(self):
            super().__init__()
            self.failed = False

        def decide(self, request):
            if not self.failed:
                self.failed = True
                raise ExecutionError("DECISION_TRANSPORT_FAILED", "interrupted")
            return super().decide(request)

    client = InterruptedClient()
    query = RoutingDecision("claim1", "Visible?", REGION)
    with RunStore(tmp_path, "failed") as store:
        router = StoredDecisionRouter(
            client, routing_config(evidence_enabled=True), RuntimeConfig(), store
        )
        with pytest.raises(ExecutionError):
            router.decide(
                query, image_id="image", image=model_image(image_artifact), trial_id="trial"
            )
    with RunStore(tmp_path, "failed") as store:
        router = StoredDecisionRouter(
            client, routing_config(evidence_enabled=True), RuntimeConfig(), store
        )
        assert (
            router.decide(
                query, image_id="image", image=model_image(image_artifact), trial_id="trial"
            )
            == "MET"
        )
        assert [
            row[0]
            for row in store.connection.execute(
                "SELECT status FROM model_call_attempt ORDER BY started_at"
            )
        ] == ["FAILED", "COMPLETE"]


def test_pipeline_short_extraction_has_no_generator_verdicts_and_reuses_native_calls(
    tmp_path, image_artifact
):
    image, _ = image_artifact
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={
                    "decision_routing": routing_config(evidence_enabled=True),
                }
            )
        }
    )
    generator, finite = ProposalClient(config.models.generator_a), FiniteClient()
    with RunStore(tmp_path, "pipeline") as store:
        coordinator = SynthesisCoordinator(
            config, "pipeline", store, generator, generator, generator, decision_client=finite
        )
        result = coordinator._extract_decision_evidence(
            generator, image, model_image(image_artifact), []
        )
        assert len(finite.requests) == 2
        assert all(observation.verdict == "MET" for observation in result.scopes[0].observations)
        coordinator._extract_decision_evidence(generator, image, model_image(image_artifact), [])
        assert len(finite.requests) == 2
        assert generator.stages == ["evidence_proposal", "evidence_proposal"]


def test_binding_request_keeps_large_legacy_claims_and_exact_references_outside_question(
    tmp_path, image_artifact
):
    source = inventory()
    scope = source.scopes[0]
    capabilities = [
        *scope.observations,
        *(
            CapabilityObservation(
                evidence_id=f"extra_{index}",
                capability=capability,
                verdict="MET",
                region=REGION,
                detail="x" * 512,
            )
            for index, capability in enumerate(
                (
                    name
                    for name in task_catalog().capabilities
                    if name not in {"visible_entity", "visible_attribute"}
                ),
                1,
            )
        ),
    ][:20]
    source = source.model_copy(
        update={"scopes": (scope.model_copy(update={"observations": tuple(capabilities)}),)}
    )
    template = candidate("attribute_lookup")
    proposal = attribute_proposal()
    target = proposal.bindings[0].target.model_copy(
        update={"evidence_refs": tuple(item.evidence_id for item in capabilities)}
    )
    proposal = BindingProposals(
        bindings=(proposal.bindings[0].model_copy(update={"target": target}),)
    )
    history = (
        PublicMessage(
            message_id="public-quote",
            turn_index=1,
            role="user",
            content='Keep this exact: "quoted"\nline',
        ),
    )
    queries = binding_decisions(proposal, (template,), source, history)
    for query in queries:
        assert len(query.question) <= 4096
        claim_context = [
            json.loads(item.removeprefix("Unverified local visual claim: "))
            for item in query.public_context
            if item.startswith("Unverified local visual claim: ")
        ]
        assert {item["evidence_id"] for item in claim_context} == {
            item.evidence_id for item in capabilities
        }
        assert all(
            item["detail"] == "x" * 512
            for item in claim_context
            if item["evidence_id"].startswith("extra_")
        )
        assert json.loads(
            query.public_context[0].removeprefix("Committed public message: ")
        ) == history[0].model_dump(mode="json")
    client = FiniteClient()
    with RunStore(tmp_path, "large-claims") as store:
        router = StoredDecisionRouter(
            client, routing_config(binding_enabled=True), RuntimeConfig(), store
        )
        assert (
            router.decide(
                queries[0], image_id="image", image=model_image(image_artifact), trial_id="large"
            )
            == "MET"
        )


@pytest.mark.parametrize(
    "query",
    [
        RoutingDecision("oversized", "x" * 4097, REGION),
        RoutingDecision("oversized", "Visible?", REGION, ("x" * 16385,)),
    ],
)
def test_invalid_finite_request_is_execution_error_before_model_call(
    tmp_path, image_artifact, query
):
    client = FiniteClient()
    with RunStore(tmp_path, "bad-request") as store:
        router = StoredDecisionRouter(
            client, routing_config(binding_enabled=True), RuntimeConfig(), store
        )
        with pytest.raises(ExecutionError) as error:
            router.decide(
                query, image_id="image", image=model_image(image_artifact), trial_id="invalid"
            )
        assert error.value.reason == "DECISION_REQUEST_INVALID"
        assert not client.requests
        assert (
            store.connection.execute("SELECT COUNT(*) FROM model_call_attempt").fetchone()[0] == 0
        )


def test_native_jev_second_readout_usage_is_measured_and_preserves_model_request_count(
    tmp_path, image_artifact
):
    revision = "f34b598d4ef4bcefd337bee8d8e7ddd3b7733ccc"

    class TwoReadoutClient(FiniteClient):
        def decide(self, request):
            result = super().decide(request)
            return result.model_copy(
                update={
                    "model_repository": "autotrust/JEV-27B-VL",
                    "model_revision": revision,
                    "request_hash": request.identity("autotrust/JEV-27B-VL", revision),
                    "prompt_tokens": 100,
                    "completion_tokens": 2,
                    "num_model_requests": 2,
                }
            )

    settings = DecisionRoutingConfig(
        binding_enabled=True, model_repository="autotrust/JEV-27B-VL", model_revision=revision
    )
    with RunStore(tmp_path, "two-readouts") as store:
        router = StoredDecisionRouter(TwoReadoutClient(), settings, RuntimeConfig(), store)
        router.decide(
            RoutingDecision("claim", "Visible?", REGION),
            image_id="image",
            image=model_image(image_artifact),
            trial_id="two",
        )
        row = store.connection.execute("SELECT * FROM model_call_attempt").fetchone()
        assert row["usage_status"] == "MEASURED"
        assert row["output_tokens"] == 2
        assert row["reserved_output_tokens"] == 2
        envelope = json.loads(store.read_artifact(row["response_artifact_hash"]))
        assert envelope["decision"]["num_model_requests"] == 2
        assert (
            store.connection.execute("SELECT reserved_output_tokens FROM budget").fetchone()[0] == 0
        )


def test_bad_generated_target_preserves_controller_and_valid_generated_sibling(
    tmp_path, image_artifact
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={"decision_routing": routing_config(binding_enabled=True)}
            )
        }
    )
    finite = FiniteClient()
    good = attribute_proposal().bindings[0].model_copy(update={"candidate_id": "open"})
    bad = good.model_copy(
        update={
            "candidate_id": "bad",
            "target": TargetReport(value="dog", origin="instruction", evidence_refs=("obs_1_1",)),
        }
    )

    class MixedGenerator(ProposalClient):
        def invoke(self, stage, payload, images, response_model, **kwargs):
            self.stages.append(stage)
            assert stage == "candidate_proposal"
            assert len(finite.requests) == 2
            return ModelResponse(
                value=BindingProposals(bindings=(bad, good)),
                request_hash="r",
                response_hash="s",
                prompt_tokens=10,
                completion_tokens=20,
            )

    generator = MixedGenerator(config.models.generator_a)
    with RunStore(tmp_path, "mixed-targets") as store:
        coordinator = SynthesisCoordinator(
            config, "mixed-targets", store, generator, generator, generator, decision_client=finite
        )
        admitted = coordinator._bind_candidate_batch(
            (
                candidate("object_identification", candidate_id="anchored"),
                candidate("attribute_lookup", candidate_id="bad"),
                candidate("attribute_lookup", candidate_id="open"),
            ),
            inventory(),
            build_history_snapshot("conversation", 1, ()),
            generator,
            "en",
            model_image(image_artifact),
            [],
            frozenset(),
        )
        assert [item.task_id for item in admitted] == ["object_identification", "attribute_lookup"]
        assert len(finite.requests) == 4
        assert generator.stages == ["candidate_proposal"]
        artifact = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind='routing-proposal-rejections'"
        ).fetchone()
        record = json.loads(store.read_artifact(artifact[0]))
        assert [(item["candidate_id"], item["reason"]) for item in record["rejections"]] == [
            ("bad", "CANDIDATE_PARAMETER_SOURCE")
        ]
        assert bad.target.origin == "instruction"
        assert bad.target.evidence_refs == ("obs_1_1",)


@pytest.mark.parametrize("failure", ["MODEL_SCHEMA_MISMATCH", "CANDIDATE_BINDING_ID"])
def test_finite_verified_controller_candidate_survives_independent_bad_proposal_branch(
    tmp_path, image_artifact, failure
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={"decision_routing": routing_config(binding_enabled=True)}
            )
        }
    )
    finite = FiniteClient()

    class InvalidGenerator(ProposalClient):
        def invoke(self, stage, payload, images, response_model, **kwargs):
            assert (
                len(finite.requests) == 2
            )  # Controller checks happen before open-choice generation.
            self.stages.append(stage)
            if failure == "MODEL_SCHEMA_MISMATCH":
                raise ExecutionError(failure, "malformed independent branch")
            wrong = attribute_proposal().bindings[0].model_copy(update={"candidate_id": "unknown"})
            # The whole affected response is rejected, even though one sibling has a valid ID.
            valid = attribute_proposal().bindings[0].model_copy(update={"candidate_id": "open"})
            return ModelResponse(
                value=BindingProposals(bindings=(wrong, valid)),
                request_hash="r",
                response_hash="s",
                prompt_tokens=10,
                completion_tokens=20,
            )

    generator = InvalidGenerator(config.models.generator_a)
    snapshot = build_history_snapshot("conversation", 1, ())
    with RunStore(tmp_path, "independent") as store:
        coordinator = SynthesisCoordinator(
            config, "independent", store, generator, generator, generator, decision_client=finite
        )
        templates = (
            candidate("object_identification", candidate_id="anchored"),
            candidate("attribute_lookup", candidate_id="open"),
        )
        admitted = coordinator._bind_candidate_batch(
            templates,
            inventory(),
            snapshot,
            generator,
            "en",
            model_image(image_artifact),
            [],
            frozenset(),
        )
        assert len(admitted) == 1
        assert admitted[0].task_id == "object_identification"
        assert len(finite.requests) == 2
        artifact = store.connection.execute(
            "SELECT artifact_hash FROM artifact WHERE kind='routing-binding-branch-failures'"
        ).fetchone()
        failure_record = json.loads(store.read_artifact(artifact[0]))
        assert failure_record["reason"] == failure
        assert failure_record["entire_generated_response_rejected"] is True
        assert failure_record["failed_candidate_ids"] == ["open"]


@pytest.mark.parametrize(
    "failure",
    [
        "MODEL_SCHEMA_MISMATCH",
        "CANDIDATE_BINDING_ID",
        "MODEL_TRANSPORT_RETRYABLE",
        "DECISION_WORKER_FAILED",
        "REQUEST_BUDGET_EXHAUSTED",
    ],
)
def test_no_valid_controller_or_operational_failure_is_propagated(
    tmp_path, image_artifact, failure
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={"decision_routing": routing_config(binding_enabled=True)}
            )
        }
    )

    class InvalidGenerator(ProposalClient):
        def invoke(self, *args, **kwargs):
            raise ExecutionError(failure, "failure must propagate")

    generator = InvalidGenerator(config.models.generator_a)
    with RunStore(tmp_path, "propagate") as store:
        coordinator = SynthesisCoordinator(
            config,
            "propagate",
            store,
            generator,
            generator,
            generator,
            decision_client=FiniteClient(),
        )
        with pytest.raises(ExecutionError) as error:
            coordinator._bind_candidate_batch(
                (candidate("attribute_lookup"),),
                inventory(),
                build_history_snapshot("c", 1, ()),
                generator,
                "en",
                model_image(image_artifact),
                [],
                frozenset(),
            )
        assert error.value.reason == failure


@pytest.mark.parametrize(
    "failure", ["MODEL_TRANSPORT_RETRYABLE", "DECISION_WORKER_FAILED", "REQUEST_BUDGET_EXHAUSTED"]
)
def test_operational_failure_never_becomes_success_even_with_controller_sibling(
    tmp_path, image_artifact, failure
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={"decision_routing": routing_config(binding_enabled=True)}
            )
        }
    )

    class InvalidGenerator(ProposalClient):
        def invoke(self, *args, **kwargs):
            raise ExecutionError(failure, "failure must propagate")

    generator = InvalidGenerator(config.models.generator_a)
    finite = FiniteClient()
    with RunStore(tmp_path, "operational") as store:
        coordinator = SynthesisCoordinator(
            config, "operational", store, generator, generator, generator, decision_client=finite
        )
        with pytest.raises(ExecutionError) as error:
            coordinator._bind_candidate_batch(
                (
                    candidate("object_identification", candidate_id="anchored"),
                    candidate("attribute_lookup", candidate_id="open"),
                ),
                inventory(),
                build_history_snapshot("c", 1, ()),
                generator,
                "en",
                model_image(image_artifact),
                [],
                frozenset(),
            )
        assert len(finite.requests) == 2
        assert error.value.reason == failure


@pytest.mark.parametrize("failure", ["MODEL_SCHEMA_MISMATCH", "CANDIDATE_BINDING_ID"])
def test_verified_supported_candidate_survives_malformed_separate_fallback(
    tmp_path, image_artifact, failure
):
    config = load_config(Path("configs/pilot.yaml"))
    config = config.model_copy(
        update={
            "tasks": config.tasks.model_copy(
                update={"decision_routing": routing_config(binding_enabled=True)}
            )
        }
    )

    class InvalidGenerator(ProposalClient):
        def invoke(self, stage, *args, **kwargs):
            assert stage == "candidate_binding"
            raise ExecutionError(failure, "malformed fallback")

    generator = InvalidGenerator(config.models.generator_a)
    with RunStore(tmp_path, "fallback") as store:
        coordinator = SynthesisCoordinator(
            config,
            "fallback",
            store,
            generator,
            generator,
            generator,
            decision_client=FiniteClient(),
        )
        admitted = coordinator._bind_candidate_batch(
            (
                candidate("object_identification", candidate_id="anchored"),
                candidate("grounded_description", candidate_id="fallback"),
            ),
            inventory(),
            build_history_snapshot("c", 1, ()),
            generator,
            "en",
            model_image(image_artifact),
            [],
            frozenset(),
        )
        assert len(admitted) == 1
        assert admitted[0].task_id == "object_identification"
