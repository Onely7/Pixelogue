"""Behavioral cases for scope-closed finite operation checks."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.finite_verifiers import (
    FiniteAnswer,
    FiniteMember,
    FiniteQuery,
    FiniteSource,
    verify_finite,
)
from pixelogue.task_evidence import ImageRegion
from pixelogue.task_verification import verify_operation


def _source(task_id: str, query: FiniteQuery) -> FiniteSource:
    region = ImageRegion(left=0, top=0, right=1, bottom=1)
    members = (
        FiniteMember(
            member_id="a", groups=("red",), predicate_met=True, order_index=1, region=region
        ),
        FiniteMember(
            member_id="b", groups=("blue",), predicate_met=False, order_index=0, region=region
        ),
        FiniteMember(
            member_id="c", groups=("red",), predicate_met=True, order_index=2, region=region
        ),
    )
    return FiniteSource(
        task_id=task_id,
        coverage="MET",
        closed=True,
        scope_id="scope-a",
        view_id="view-a",
        scope_region=region,
        members=members,
        query=query,
        reason="All three objects visible in the requested scope",
    )


def _answer(form: str, value: object) -> FiniteAnswer:
    return FiniteAnswer.model_validate(
        {
            "coverage": "MET",
            "answer_form": form,
            form if form != "boolean" else "truth": value,
            "reason": "Result copied from the answer text",
        }
    )


def _check(source: FiniteSource, answer: FiniteAnswer, **parameters: object) -> GateVerdict:
    return verify_finite(
        source.task_id,
        (source, source),
        (answer, answer),
        parameters,
        source.scope_id,
        source.view_id,
    )


def test_order_requires_all_visible_members_in_requested_sequence() -> None:
    source = _source("spatial_ordering", FiniteQuery(answer_form="members"))
    assert _check(source, _answer("members", ("b", "a", "c"))) is GateVerdict.MET
    assert _check(source, _answer("members", ("a", "b", "c"))) is GateVerdict.NOT_MET
    assert _check(source, _answer("members", ("b", "a"))) is GateVerdict.NOT_MET


def test_cardinality_comparison_uses_closed_group_counts() -> None:
    source = _source(
        "set_cardinality_comparison",
        FiniteQuery(answer_form="relation", left_group="red", right_group="blue"),
    )
    assert _check(source, _answer("relation", "greater")) is GateVerdict.MET
    assert _check(source, _answer("relation", "equal")) is GateVerdict.NOT_MET


def test_quantifier_checks_public_choice_and_threshold() -> None:
    source = _source(
        "quantified_statement_verification",
        FiniteQuery(answer_form="boolean", quantifier="exactly", threshold=2),
    )
    assert _check(source, _answer("boolean", True), quantifier="exactly") is GateVerdict.MET
    assert _check(source, _answer("boolean", False), quantifier="exactly") is GateVerdict.NOT_MET
    assert _check(source, _answer("boolean", True), quantifier="some") is GateVerdict.UNKNOWN


def test_hypothetical_add_remove_and_relabel_follow_public_update() -> None:
    add = _source(
        "grounded_hypothetical_update",
        FiniteQuery(answer_form="count", update="add", update_ids=("d",), update_group="red"),
    )
    assert _check(add, _answer("count", 4), update="add") is GateVerdict.MET
    remove = add.model_copy(
        update={"query": FiniteQuery(answer_form="count", update="remove", update_ids=("b",))}
    )
    assert _check(remove, _answer("count", 2), update="remove") is GateVerdict.MET
    relabel = add.model_copy(
        update={
            "query": FiniteQuery(
                answer_form="count",
                left_group="red",
                update="relabel",
                update_ids=("b",),
                update_group="red",
            )
        }
    )
    assert _check(relabel, _answer("count", 3), update="relabel") is GateVerdict.MET


def test_hypothetical_removal_can_yield_a_closed_empty_member_set() -> None:
    source = _source(
        "grounded_hypothetical_update",
        FiniteQuery(answer_form="members", update="remove", update_ids=("a", "b", "c")),
    )
    empty = _answer("members", ())
    assert _check(source, empty, update="remove") is GateVerdict.MET
    assert _check(source, _answer("members", ("a",)), update="remove") is GateVerdict.NOT_MET
    with pytest.raises(ValueError, match="Member enumeration cannot include a scalar result"):
        FiniteAnswer(
            coverage="MET", answer_form="members", members=(), count=0, reason="conflicting"
        )


def test_each_finite_operation_rejects_wrong_results_and_unclosed_sources() -> None:
    cases = (
        (
            _source("spatial_ordering", FiniteQuery(answer_form="members")),
            _answer("members", ("b", "a", "c")),
            _answer("members", ("a", "b", "c")),
            {},
        ),
        (
            _source(
                "set_cardinality_comparison",
                FiniteQuery(answer_form="relation", left_group="red", right_group="blue"),
            ),
            _answer("relation", "greater"),
            _answer("relation", "equal"),
            {},
        ),
        (
            _source(
                "quantified_statement_verification",
                FiniteQuery(answer_form="boolean", quantifier="exactly", threshold=2),
            ),
            _answer("boolean", True),
            _answer("boolean", False),
            {"quantifier": "exactly"},
        ),
        (
            _source(
                "grounded_hypothetical_update",
                FiniteQuery(answer_form="count", update="remove", update_ids=("b",)),
            ),
            _answer("count", 2),
            _answer("count", 3),
            {"update": "remove"},
        ),
    )
    for source, correct, wrong, parameters in cases:
        assert _check(source, correct, **parameters) is GateVerdict.MET
        assert _check(source, wrong, **parameters) is GateVerdict.NOT_MET
        assert (
            _check(source.model_copy(update={"closed": False}), correct, **parameters)
            is GateVerdict.UNKNOWN
        )


def test_uncertain_source_disagreement_and_nonlocal_scope_never_pass() -> None:
    source = _source("spatial_ordering", FiniteQuery(answer_form="members"))
    answer = _answer("members", ("b", "a", "c"))
    assert (
        verify_finite(
            source.task_id,
            (source, source.model_copy(update={"closed": False})),
            (answer, answer),
            {},
            source.scope_id,
            source.view_id,
        )
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_finite(
            source.task_id,
            (source, source),
            (answer, answer),
            {},
            "another-scope",
            source.view_id,
        )
        is GateVerdict.UNKNOWN
    )


def test_operation_extraction_keeps_answer_out_of_visual_source_calls() -> None:
    task = next(item for item in task_catalog().tasks if item.id == "spatial_ordering")
    instruction = InstructionCandidate(
        candidate_id="candidate",
        task_id=task.id,
        family=task.family,
        visible_scope="three boxes",
        instruction_summary=task.definition_en,
        required_capabilities=task.required_capabilities,
        catalog_version="7.0",
        scope_id="scope-a",
        view_id="view-a",
        evidence_refs=("e1",),
        verification_contracts=task.verification_contracts,
    )
    source = _source("spatial_ordering", FiniteQuery(answer_form="members"))
    answer = _answer("members", ("b", "a", "c"))
    seen: list[str] = []

    def invoke(
        stage: str, payload: dict[str, Any], model: type[BaseModel], judge: int
    ) -> BaseModel:
        seen.append(stage)
        if stage == "finite_source":
            assert "candidate_answer" not in payload
            assert model is FiniteSource
            return source
        assert stage == "finite_answer"
        assert "image_views" not in payload
        assert model is FiniteAnswer
        return answer

    checks = verify_operation(
        instruction,
        {
            "target_language": "en",
            "public_history": [],
            "question": "List the boxes from left to right.",
            "candidate_answer": "b, a, c",
            "image_views": [{"view_id": "view-a", "encoded_sha256": "0" * 64}],
        },
        invoke,
    )
    assert seen == ["finite_source", "finite_source", "finite_answer", "finite_answer"]
    assert len(checks) == 1
    assert checks[0].verdict is GateVerdict.MET
