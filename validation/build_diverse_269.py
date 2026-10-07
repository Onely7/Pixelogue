"""Assemble the 269-image diverse evaluation set and pin its manifest.

The set joins the 100 images of ``validation/diverse_100_20261004_manifest.json`` (60 Commons
category images and 40 stratified Commons images) with the 169 FineVision subset images of
``validation/finevision_185_eval_manifest.jsonl``. Every image is evaluation-only.

Run from the repository root after the three source roots have been fetched::

    uv run --locked python validation/build_diverse_269.py assemble
    uv run --locked pixelogue ingest --config configs/split-diverse-269.yaml \
        --sources data/diverse-269-eval/sources.jsonl \
        --rights data/diverse-269-eval/rights.jsonl \
        --image-root data/diverse-269-eval \
        --artifact-root artifacts/prepared-diverse-269-eval
    uv run --locked python validation/build_diverse_269.py manifest

``assemble`` copies the image bytes under one root and rewrites only ``image_path``, so every
source ID, and therefore every image ID, stays the same. ``manifest`` reads the prepared root
and records each image's identity together with the generator role and planned turn count
that ``pixelogue synthesize`` assigns under ``configs/split-diverse-269.yaml``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pixelogue.config import load_config
from pixelogue.contracts import ImageArtifact, RightsRecord, SourcePurpose, SourceRecord
from pixelogue.io import read_json, read_jsonl, write_jsonl
from pixelogue.operations import PreparedDataset
from pixelogue.planner import exact_schedule, planned_turn_count

DIVERSE_100 = Path("validation/diverse_100_20261004_manifest.json")
FINEVISION = Path("validation/finevision_185_eval_manifest.jsonl")
MANIFEST = Path("validation/diverse_269_20261007_manifest.json")
CONFIG = Path("configs/split-diverse-269.yaml")
DESTINATION = Path("data/diverse-269-eval")
PREPARED = Path("artifacts/prepared-diverse-269-eval")
# Cohort name, source ID prefix, existing data root and directory under DESTINATION.
COHORTS = (
    ("commons_web60", "commons-eval:", Path("data/diverse-web-eval"), "commons-web60"),
    ("commons_qg40", "commons-qg50:", Path("data/qg-fast-50-20261003"), "commons-qg40"),
    ("finevision", "finevision-eval:", Path("data/finevision-185-eval"), "finevision"),
)
EXPECTED = {"commons_web60": 60, "commons_qg40": 40, "finevision": 169}


def sha256(path: Path) -> str:
    """Return the SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_hashes() -> dict[str, str]:
    """Map every selected source ID to the byte hash pinned by its committed manifest."""
    pinned = {
        image["source_id"]: image["raw_sha256"]
        for image in json.loads(DIVERSE_100.read_text(encoding="utf-8"))["images"]
    }
    for line in FINEVISION.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if not row["text_only"]:
            pinned[f"finevision-eval:{row['number']:03d}"] = row["evaluation_image"]["sha256"]
    return pinned


def copy(source: Path, target: Path) -> None:
    """Copy one image, refusing to replace different bytes.

    A copy rather than a hard link keeps a later refetch of a source root from changing the
    assembled bytes in place.
    """
    if target.exists():
        if sha256(target) != sha256(source):
            raise SystemExit(f"{target} exists with different bytes")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def assemble() -> None:
    """Collect the 269 images and their source and rights records under one root."""
    pinned = expected_hashes()
    sources: list[SourceRecord] = []
    rights: list[RightsRecord] = []
    counts: Counter[str] = Counter()
    for cohort, prefix, root, directory in COHORTS:
        rights_by_id = {
            record.rights_record_id: record
            for record in read_jsonl(root / "rights.jsonl", RightsRecord)
        }
        for source in read_jsonl(root / "sources.jsonl", SourceRecord):
            if source.source_id not in pinned:
                continue
            if not source.source_id.startswith(prefix):
                raise SystemExit(f"{source.source_id} is outside cohort {cohort}")
            original = root / source.image_path
            if sha256(original) != pinned[source.source_id]:
                raise SystemExit(f"{original} does not match its pinned SHA-256")
            if source.purpose is not SourcePurpose.EVALUATION:
                raise SystemExit(f"{source.source_id} is not evaluation-only")
            image_path = f"{directory}/{source.image_path}"
            copy(original, DESTINATION / image_path)
            sources.append(source.model_copy(update={"image_path": image_path}))
            rights.append(rights_by_id[source.rights_record_id])
            counts[cohort] += 1
    if dict(counts) != EXPECTED:
        raise SystemExit(f"Unexpected cohort sizes: {dict(counts)}")
    if {source.source_id for source in sources} != pinned.keys():
        raise SystemExit("Some pinned images were not found in their source roots")
    write_jsonl(DESTINATION / "sources.jsonl", sorted(sources, key=lambda item: item.source_id))
    write_jsonl(
        DESTINATION / "rights.jsonl", sorted(rights, key=lambda item: item.rights_record_id)
    )
    print(json.dumps({"sources": len(sources), "cohorts": dict(counts)}))


def stratum_slug(category: str) -> str:
    """Name a FineVision category stratum, such as ``finevision_chart_table``."""
    return "finevision_" + re.sub(r"[^a-z]+", "_", category.lower()).strip("_")


def manifest() -> None:
    """Pin the prepared 269 images, their strata and the synthesis schedule."""
    config = load_config(CONFIG)
    images = read_jsonl(PREPARED / "images.jsonl", ImageArtifact)
    prepared = read_json(PREPARED / "manifest.json", PreparedDataset)
    if tuple(images) != prepared.images or prepared.failures:
        raise SystemExit("The prepared root is incomplete or changed after ingest")
    if len(images) != 269 or config.data.target_dialogues != len(images):
        raise SystemExit(f"Expected 269 prepared images, found {len(images)}")
    earlier = {
        image["source_id"]: image
        for image in json.loads(DIVERSE_100.read_text(encoding="utf-8"))["images"]
    }
    finevision = {
        f"finevision-eval:{row['number']:03d}": row
        for row in map(json.loads, FINEVISION.read_text(encoding="utf-8").splitlines())
    }
    if len({image.visual_group_id for image in images}) != len(images):
        raise SystemExit("Two prepared images share a visual group")
    by_source = {image.source_id: image for image in images}
    for source_id, image in earlier.items():
        if by_source[source_id].image_id != image["image_id"]:
            raise SystemExit(f"{source_id} changed its image ID")
    roles = exact_schedule(
        len(images),
        {str(role): weight for role, weight in config.models.generation_allocation.items()},
        seed=config.seed,
        namespace="generators",
    )
    entries: list[dict[str, Any]] = []
    for number, (image, role) in enumerate(zip(images, roles, strict=True), start=1):
        row = finevision.get(image.source_id)
        cohort = next(name for name, prefix, *_ in COHORTS if image.source_id.startswith(prefix))
        stratum = stratum_slug(row["category"]) if row else earlier[image.source_id]["stratum"]
        entries.append(
            {
                "number": number,
                "image_id": image.image_id,
                "visual_group_id": image.visual_group_id,
                "source_id": image.source_id,
                "cohort": cohort,
                "stratum": stratum,
                "finevision_subset": row["subset"] if row else None,
                "finevision_category": row["category"] if row else None,
                "dataset": image.dataset,
                "dataset_split": image.dataset_split,
                "raw_sha256": image.raw_sha256,
                "canonical_pixel_sha256": image.canonical_pixel_sha256,
                "encoded_sha256": image.full_view.encoded_sha256,
                "width": image.full_view.width,
                "height": image.full_view.height,
                "generator_role": role,
                "planned_turns": planned_turn_count(image.image_id, config.seed),
                "purpose": image.purpose.value,
            }
        )
    document = {
        "version": "1.0",
        "name": "diverse-269-20261007",
        "purpose": "evaluation",
        "split": "previously used development images plus one FineVision row per image subset;"
        " no independent holdout",
        "config": str(CONFIG),
        "seed": config.seed,
        "source_manifests": [str(DIVERSE_100), str(FINEVISION)],
        "prepared_manifest_hash": prepared.manifest_hash,
        "schedule": {
            "order": "prepared images in source_id order, as pixelogue synthesize reads them",
            "generator_role": "planner.exact_schedule over models.generation_allocation, "
            "namespace generators",
            "planned_turns": "planner.planned_turn_count(image_id, seed)",
            "languages": {language.tag: language.weight for language in config.data.languages},
        },
        "cohorts": dict(Counter(entry["cohort"] for entry in entries)),
        "strata": dict(Counter(entry["stratum"] for entry in entries)),
        "generator_roles": dict(sorted(Counter(roles).items())),
        "planned_turns": {
            str(turns): count
            for turns, count in sorted(Counter(e["planned_turns"] for e in entries).items())
        },
        "training_export_allowed": False,
        "images": entries,
    }
    MANIFEST.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", "utf-8")
    print(
        json.dumps(
            {
                "images": len(entries),
                "cohorts": document["cohorts"],
                "strata": len(document["strata"]),
                "generator_roles": document["generator_roles"],
                "planned_turns": document["planned_turns"],
            }
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run one build step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=("assemble", "manifest"))
    arguments = parser.parse_args(argv)
    if arguments.step == "assemble":
        assemble()
    else:
        manifest()
    return 0


if __name__ == "__main__":
    sys.exit(main())
