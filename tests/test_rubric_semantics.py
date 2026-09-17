"""Opt-in live checks for semantic compliance, independent of image factual truth."""

import json
import os
from pathlib import Path

import pytest

from pixelogue.catalog import load_rubric_catalog
from pixelogue.config import load_config
from pixelogue.contracts import RubricVerdict
from pixelogue.serving import ModelImage, VllmClient

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("PIXELOGUE_LIVE_RUBRIC") != "1",
        reason="Requires the two configured local model servers; opt in after GPU/doctor checks.",
    ),
]


@pytest.fixture(scope="module", params=["generator_a", "generator_b"])
def judge(request):
    config = load_config(Path("configs/standard.yaml"))
    client = VllmClient(
        getattr(config.models, request.param), config.runtime, run_id="rubric-semantics-regression"
    )
    yield client, config.seed + 1
    client.client.close()


@pytest.mark.parametrize("template_id", ["R_CORE", "R_REQUIREMENT"])
@pytest.mark.parametrize(
    ("question", "answer", "expected"),
    [
        ("Please transcribe the main word written on the red octagonal sign.", "STOP", "MET"),
        ("Is there any other text on the banner besides the welcome message?", "No", "MET"),
        (
            "Which is closer to the camera: the upper log or the thinner log near the bottom?",
            "the thinner log near the bottom",
            "MET",
        ),
        (
            "Can you group the butterfly and the red flower by their color?",
            "The butterfly is black and white, while the flower is red.",
            "MET",
        ),
        (
            "Can you group the items the person is wearing by their color?",
            "The person is wearing a dark-colored jacket and hat, and light-colored pants.",
            "MET",
        ),
        ("看板の文字を書き写してください。", "止まれ", "MET"),
        ("How many cars are there?", "The cars are red.", "NOT_MET"),
        (
            "Group the jacket, hat and pants by their color.",
            "A jacket, a hat and pants.",
            "NOT_MET",
        ),
    ],
)
def test_live_short_answers_and_semantic_grouping(judge, template_id, question, answer, expected):
    client, seed = judge
    template = next(t for t in load_rubric_catalog()["items"] if t["template_id"] == template_id)
    payload = {
        "target_language": "ja" if question.startswith("看板") else "en",
        "public_history": [],
        "question": question,
        "candidate_answer": answer,
        "criterion": {
            key: template[key]
            for key in ("template_id", "question", "met_anchor", "not_met_anchor")
        },
    }
    if template_id == "R_REQUIREMENT":
        payload["target_requirement"] = {
            "kind": "content",
            "description": question,
            "source_message_id": "q",
            "start": 0,
            "end": len(question),
            "lifetime": "current_turn",
        }
    result = client.invoke(
        "rubric_item",
        payload,
        (),
        RubricVerdict,
        max_tokens=256,
        temperature=0.0,
        seed=seed,
        bypass_cache=True,
    ).value
    record = {
        "model": client.endpoint.repo_id,
        "criterion": template_id,
        "question": question,
        "answer": answer,
        "expected": expected,
        **result.model_dump(mode="json"),
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    assert result.verdict == expected, record


def test_live_explicit_format_constraint_is_not_waived(judge):
    client, seed = judge
    template = next(
        t for t in load_rubric_catalog()["items"] if t["template_id"] == "R_REQUIREMENT"
    )
    question = "Group the jacket, hat and pants by color. Return only a JSON object."
    payload = {
        "target_language": "en",
        "public_history": [],
        "question": question,
        "candidate_answer": "The jacket and hat are dark; the pants are light.",
        "criterion": {
            key: template[key]
            for key in ("template_id", "question", "met_anchor", "not_met_anchor")
        },
        "target_requirement": {
            "kind": "format",
            "description": "Return only a JSON object.",
            "source_message_id": "q",
            "start": question.index("Return"),
            "end": len(question),
            "lifetime": "current_turn",
        },
    }
    result = client.invoke(
        "rubric_item",
        payload,
        (),
        RubricVerdict,
        max_tokens=256,
        temperature=0.0,
        seed=seed,
        bypass_cache=True,
    ).value
    print(json.dumps({"model": client.endpoint.repo_id, **result.model_dump(mode="json")}))
    assert result.verdict == "NOT_MET", result


@pytest.mark.parametrize("restructure", [True, False])
def test_live_requested_regrouping_counts_as_progress(judge, image_artifact, restructure):
    client, seed = judge
    image, root = image_artifact
    template = next(
        t for t in load_rubric_catalog()["items"] if t["template_id"] == "H_TURN_PROGRESS"
    )
    prior_question = "What color is the visible region?"
    payload = {
        "target_language": "en",
        "public_history": [
            {"role": "user", "content": prior_question},
            {"role": "assistant", "content": "The visible region is blue."},
        ],
        "question": "Group the visible region by its color." if restructure else prior_question,
        "candidate_answer": "The blue group contains the visible region."
        if restructure
        else "The visible region is blue.",
        "image_views": [
            {"view_id": image.full_view.view_id, "encoded_sha256": image.full_view.encoded_sha256}
        ],
        "criterion": {
            key: template[key]
            for key in ("template_id", "question", "met_anchor", "not_met_anchor")
        },
    }
    model_image = ModelImage(
        view_id=image.full_view.view_id,
        path=root / image.full_view.relative_path,
        encoded_sha256=image.full_view.encoded_sha256,
        media_type=image.full_view.media_type,
    )
    result = client.invoke(
        "rubric_item",
        payload,
        (model_image,),
        RubricVerdict,
        max_tokens=256,
        temperature=0.0,
        seed=seed,
        bypass_cache=True,
    ).value
    print(
        json.dumps(
            {
                "model": client.endpoint.repo_id,
                "restructure": restructure,
                **result.model_dump(mode="json"),
            }
        )
    )
    assert result.verdict == ("MET" if restructure else "NOT_MET"), result
