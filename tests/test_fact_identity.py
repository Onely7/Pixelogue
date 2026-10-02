"""Explicit request identity boundaries without semantic guesses from broad task names."""

from pixelogue.contracts import InstructionCandidate
from pixelogue.fact_identity import requested_fact_key
from pixelogue.task_evidence import PublicParameter


def request(task="attribute_lookup", target="The dog", **choices):
    return InstructionCandidate(
        candidate_id="candidate",
        task_id=task,
        family="visual_description",
        visible_scope="the subject",
        instruction_summary="Request a visible fact",
        required_capabilities=("visible_entity",),
        scope_id="subject",
        view_id="view",
        public_parameters=(PublicParameter(name="target", value=target, origin="instruction"),)
        + tuple(
            PublicParameter(name=name, value=value, origin="instruction")
            for name, value in choices.items()
        ),
    )


def test_same_attribute_and_new_part_have_distinct_identities():
    assert requested_fact_key(request(attribute="fur color")) == requested_fact_key(
        request(target="dog", attribute="fur colour")
    )
    assert requested_fact_key(request(attribute="fur color")) != requested_fact_key(
        request(attribute="nose color")
    )
    assert requested_fact_key(request(attribute="fur color")) != requested_fact_key(
        request(target="cat", attribute="fur color")
    )
    assert requested_fact_key(request(attribute="color")) != requested_fact_key(
        request(attribute="fur color")
    )


def test_different_task_labels_do_not_make_the_same_explicit_count_novel():
    current = request("entity_count", count_unit="dogs")
    old = request("visible_count", count_unit="dogs")
    assert requested_fact_key(current) is not None
    assert requested_fact_key(current) == requested_fact_key(old)
    assert requested_fact_key(request("visible_count")) is None
    assert requested_fact_key(request("set_cardinality_comparison", count_unit="dogs")) is None


def test_scope_view_predicate_and_reference_frame_are_substantive():
    original = request("entity_count", count_unit="dogs", predicate="in the foreground")
    assert requested_fact_key(original) != requested_fact_key(
        request("entity_count", count_unit="dogs", predicate="in the background")
    )
    assert requested_fact_key(original) != requested_fact_key(
        original.model_copy(update={"view_id": "crop"})
    )
    assert requested_fact_key(original) != requested_fact_key(
        original.model_copy(update={"scope_id": "other"})
    )
    left = request("spatial_relation", relation="left-right", frame="image")
    assert requested_fact_key(left) == requested_fact_key(
        request("relation_lookup", relation="left-right", frame="image")
    )
    assert requested_fact_key(left) != requested_fact_key(
        request("spatial_relation", relation="left-right", frame="subject")
    )
    assert requested_fact_key(request("spatial_relation", frame="image")) is None


def test_formatting_does_not_create_a_new_fact_and_broad_requests_stay_unresolved():
    assert requested_fact_key(request(attribute="color", format="text")) == requested_fact_key(
        request(attribute="color", format="json")
    )
    for task in (
        "grounded_description",
        "visual_summary",
        "visible_action_relation",
        "attribute_comparison",
        "answerability_assessment",
    ):
        assert requested_fact_key(request(task)) is None
    assert (
        requested_fact_key(request(attribute="color").model_copy(update={"profile": "limitation"}))
        is None
    )
    assert (
        requested_fact_key(request(attribute="color").model_copy(update={"view_id": None})) is None
    )
