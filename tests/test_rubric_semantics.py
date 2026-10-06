"""Opt-in live check that each local judge reviews visual truth as a whole."""

import json
import os
from pathlib import Path

import pytest

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
    config = load_config(Path(os.environ.get("PIXELOGUE_LIVE_CONFIG", "configs/standard.yaml")))
    client = VllmClient(
        getattr(config.models, request.param), config.runtime, run_id="rubric-semantics-regression"
    )
    yield client, config.seed + 1
    client.client.close()


@pytest.mark.parametrize(("answer", "expected"), [("Blue", "MET"), ("Red", "NOT_MET")])
def test_live_holistic_review_checks_visual_truth(judge, image_artifact, answer, expected):
    client, seed = judge
    image, root = image_artifact
    model_image = ModelImage(
        view_id=image.full_view.view_id,
        path=root / image.full_view.relative_path,
        encoded_sha256=image.full_view.encoded_sha256,
        media_type=image.full_view.media_type,
    )
    result = client.invoke(
        "holistic_review",
        {
            "target_language": "en",
            "public_history": [],
            "question": "What color fills the image?",
            "candidate_answer": answer,
            "image_views": [
                {"view_id": model_image.view_id, "encoded_sha256": model_image.encoded_sha256}
            ],
        },
        (model_image,),
        RubricVerdict,
        max_tokens=384,
        temperature=0.0,
        seed=seed,
        bypass_cache=True,
    ).value
    print(
        json.dumps(
            {"model": client.endpoint.repo_id, "answer": answer, **result.model_dump(mode="json")}
        )
    )
    assert result.verdict == expected, result
