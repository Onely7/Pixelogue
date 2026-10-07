"""Knowledge answers commit only when two blind readers give the same short answer."""

from __future__ import annotations

from pixelogue.catalog import task_catalog
from pixelogue.contracts import GateVerdict, InstructionCandidate
from pixelogue.knowledge_verifiers import (
    CONSENSUS_TASKS,
    ConsensusAnswer,
    ConsensusSource,
    normalize_short_answer,
    verify_consensus,
)
from pixelogue.task_verification import verify_operation

TEXT = "This is the Golden Gate Bridge, recognizable by its red towers and suspension cables."


def _reader(answer: str, coverage: str = "MET") -> ConsensusSource:
    return ConsensusSource.model_validate(
        {
            "coverage": coverage,
            "short_answer": answer,
            "visible_evidence": "red towers, suspension cables",
            "reason": "read blind",
        }
    )


def _parse(quote: str, answer: str, coverage: str = "MET") -> ConsensusAnswer:
    return ConsensusAnswer.model_validate(
        {"coverage": coverage, "answer_quote": quote, "short_answer": answer, "reason": "parsed"}
    )


def test_consensus_needs_both_blind_readers_to_match_the_answer() -> None:
    parsed = _parse("This is the Golden Gate Bridge", "the Golden Gate Bridge")
    agreeing = (_reader("Golden Gate Bridge"), _reader("golden gate bridge."))
    assert verify_consensus(agreeing, (parsed, parsed), TEXT) is GateVerdict.MET
    other = (_reader("Bay Bridge"), _reader("the Bay Bridge"))
    assert verify_consensus(other, (parsed, parsed), TEXT) is GateVerdict.NOT_MET
    split = (_reader("Golden Gate Bridge"), _reader("Bay Bridge"))
    assert verify_consensus(split, (parsed, parsed), TEXT) is GateVerdict.UNKNOWN
    abstaining = (_reader("Golden Gate Bridge"), _reader("", "UNKNOWN"))
    assert verify_consensus(abstaining, (parsed, parsed), TEXT) is GateVerdict.UNKNOWN
    invented_quote = _parse("This is the Bay Bridge", "Bay Bridge")
    invented = (invented_quote, invented_quote)
    assert verify_consensus(agreeing, invented, TEXT) is GateVerdict.UNKNOWN
    uneven_parse = (parsed, _parse("This is the Golden Gate Bridge", "Golden Gate"))
    assert verify_consensus(agreeing, uneven_parse, TEXT) is GateVerdict.UNKNOWN


def test_numbers_compare_by_value_and_unit() -> None:
    assert normalize_short_answer("1,200 m") == normalize_short_answer("1200.0 m")
    assert normalize_short_answer("1200 km") != normalize_short_answer("1200 m")
    assert normalize_short_answer("25%") == normalize_short_answer("25 %")
    assert normalize_short_answer("D major") == normalize_short_answer("d Major.")
    assert normalize_short_answer("The Nile") == normalize_short_answer("Nile")
    text = "Divide the total rise by five years.\nAnswer: 3.16 mm/year"
    parsed = _parse("Answer: 3.16 mm/year", "3.16 mm/year")
    readers = (_reader("3.160 mm/year"), _reader("3.16 mm / year"))
    assert verify_consensus(readers, (parsed, parsed), text) is GateVerdict.MET
    assert (
        verify_consensus((_reader("3.2 mm/year"), _reader("3.2 mm/year")), (parsed, parsed), text)
        is GateVerdict.NOT_MET
    )


def test_consensus_readers_never_see_the_candidate_answer() -> None:
    task = next(item for item in task_catalog().tasks if item.id == "named_entity_recognition")
    assert set(task.verification_contracts) == {"dual_visual_review", "answer_consensus_check"}
    assert task.id in CONSENSUS_TASKS
    candidate = InstructionCandidate(
        candidate_id="landmark",
        task_id=task.id,
        family=task.family,
        visible_scope="the bridge",
        instruction_summary=task.definition,
        required_capabilities=task.required_capabilities,
        catalog_version="8.0",
        scope_id="canvas",
        view_id="view",
        evidence_refs=("scope-evidence",),
        verification_contracts=task.verification_contracts,
    )
    stages: list[tuple[str, int]] = []

    def invoke(stage, payload, model, judge):
        stages.append((stage, judge))
        if stage == "consensus_source":
            assert "candidate_answer" not in payload
            return _reader("Golden Gate Bridge")
        assert stage == "consensus_answer"
        assert "image_views" not in payload
        return _parse("This is the Golden Gate Bridge", "Golden Gate Bridge")

    result = verify_operation(
        candidate, {"question": "Which bridge is this?", "candidate_answer": TEXT}, invoke
    )
    assert stages == [
        ("consensus_source", 0),
        ("consensus_source", 1),
        ("consensus_answer", 0),
        ("consensus_answer", 1),
    ]
    assert [(check.name, check.verdict) for check in result] == [
        ("answer_consensus_check", GateVerdict.MET)
    ]
