"""The pinned diverse-269 evaluation manifest agrees with its configuration and sources."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from pixelogue.config import load_config
from pixelogue.planner import exact_schedule, planned_turn_count

MANIFEST = Path("validation/diverse_269_20261007_manifest.json")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def test_schedule_is_the_one_synthesize_assigns_under_the_pinned_config() -> None:
    manifest = _read(MANIFEST)
    config = load_config(Path(manifest["config"]))
    images = manifest["images"]
    assert config.seed == manifest["seed"]
    assert config.data.target_dialogues == len(images) == 269
    source_ids = [image["source_id"] for image in images]
    assert source_ids == sorted(source_ids)
    roles = exact_schedule(
        len(images),
        {str(role): weight for role, weight in config.models.generation_allocation.items()},
        seed=config.seed,
        namespace="generators",
    )
    assert [image["generator_role"] for image in images] == list(roles)
    for image in images:
        assert image["planned_turns"] == planned_turn_count(image["image_id"], config.seed)


def test_set_keeps_the_earlier_100_and_one_distinct_image_per_finevision_subset() -> None:
    manifest = _read(MANIFEST)
    images = manifest["images"]
    assert Counter(image["cohort"] for image in images) == {
        "commons_web60": 60,
        "commons_qg40": 40,
        "finevision": 169,
    }
    assert len({image["visual_group_id"] for image in images}) == len(images)
    assert {image["purpose"] for image in images} == {"evaluation"}
    assert manifest["training_export_allowed"] is False
    by_source = {image["source_id"]: image for image in images}
    for image in _read(Path("validation/diverse_100_20261004_manifest.json"))["images"]:
        assert by_source[image["source_id"]]["image_id"] == image["image_id"]
    rows = [
        json.loads(line)
        for line in Path("validation/finevision_185_eval_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with_images = {row["subset"]: row for row in rows if not row["text_only"]}
    finevision = [image for image in images if image["cohort"] == "finevision"]
    assert {image["finevision_subset"] for image in finevision} == with_images.keys()
    for image in finevision:
        row = with_images[image["finevision_subset"]]
        assert image["raw_sha256"] == row["evaluation_image"]["sha256"]
        assert image["stratum"].startswith("finevision_")
