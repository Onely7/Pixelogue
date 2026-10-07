#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "huggingface-hub>=0.34",
#     "pillow>=11",
#     "pyarrow>=18",
# ]
# ///
"""Fetch one private evaluation row from every FineVision subset.

For each of the 185 subsets (configs) of ``HuggingFaceM4/FineVision`` at a pinned commit, the
script reads the first row group of the first parquet shard, which keeps the original image
bytes. It selects the lowest row whose first image already satisfies the Pixelogue ingest
limits. When no row in that group qualifies, the first row's image is minimally derived
(lossless PNG conversion, white padding to the minimum edge, or downscaling) and every step is
recorded. Text-only subsets keep their first row without an image. Two subsets whose first row
repeats another subset's picture skip that row, and the run fails if any two selected images
still decode to the same pixels.

Row texts and ratings are dataset annotations. They stay in the local output directory for
task analysis and never become model inputs or committed files. The committed manifest holds
identities only: subset, row, image hashes, sizes and derivation steps.

Run from the repository root:

    uv run --locked --script validation/fetch_finevision_eval.py
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
DATASET = "HuggingFaceM4/FineVision"
REVISION = "3c380a731a3429c1d04693d6ec16d7e683def84c"
DATASET_URL = "https://huggingface.co/datasets/HuggingFaceM4/FineVision"
MAPPING_DOC = ROOT / "docs/tasks/finevision_mapping_185.md"
MANIFEST = ROOT / "validation/finevision_185_eval_manifest.jsonl"
DESTINATION = ROOT / "data/finevision-185-eval"
SHARD = re.compile(r"train-(\d{5})-of-(\d{5})\.parquet$")
MAPPING_ROW = re.compile(r"^\| (\d+) \| `([^`]+)` \|")
# Mirrors src/pixelogue/images.py; `pixelogue ingest` re-checks every limit.
MAX_INPUT_BYTES = 20 * 1024 * 1024
MAX_DECODED_PIXELS = 40_000_000
MIN_SHORT_EDGE = 128
EXTENSIONS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
VALID_FROM = "2026-10-07T00:00:00Z"
# Row 0 of these subsets shows the same picture as row 0 of clevr and ureader_kg_processed, so
# the next eligible row keeps every evaluation image distinct.
REPEATED_ROWS: dict[str, set[int]] = {"clevr_math": {0}, "ureader_qa_processed": {0}}
DATASET_SPLIT = "finevision-185-eval-2026-10"
RATING_COLUMNS = (
    "relevance_ratings",
    "image_correspondence_ratings",
    "visual_dependency_ratings",
    "formatting_ratings",
)
# Category grouping published with FineVision (blog table); every subset appears once.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "Captioning & Knowledge": (
        "coco_colors",
        "densefusion_1m",
        "face_emotion",
        "google_landmarks",
        "image_textualization(filtered)",
        "laion_gpt4v",
        "localized_narratives",
        "sharegpt4o",
        "sharegpt4v(coco)",
        "sharegpt4v(knowledge)",
        "sharegpt4v(llava)",
        "sharegpt4v(sam)",
        "textcaps",
    ),
    "Chart & Table": (
        "chart2text",
        "chartqa",
        "CoSyn_400k_chart",
        "CoSyn_400k_table",
        "dvqa",
        "figureqa",
        "figureqa(mathv360k)",
        "finqa",
        "hitab",
        "lrv_chart",
        "mmc_instruct",
        "multihiertt",
        "plotqa",
        "robut_sqa",
        "robut_wikisql",
        "robut_wtq",
        "SynthChartNet",
        "tabmwp",
        "tabmwp(mathv360k)",
        "tat_dqa",
        "tat_qa",
        "Unichart",
        "vistext",
        "vqaonbd",
    ),
    "General VQA": (
        "alfworldgpt",
        "allava_laion",
        "allava_vflan",
        "cambrian(filtered)_processed",
        "chinesememe",
        "cocoqa",
        "CoSyn_400k_graphic",
        "datik",
        "datikz",
        "drivelm",
        "hateful_memes",
        "iconqa",
        "iconqa(mathv360k)",
        "idk",
        "indoor_qa",
        "LLaVA_Instruct_150K",
        "llavar_gpt4_20k",
        "lnqa",
        "lrv_normal(filtered)",
        "lvis_instruct4v",
        "mimic_cgd",
        "mmevol",
        "mmra",
        "nlvr2",
        "sketchyvqa",
        "spark",
        "spatialsense",
        "spot_the_diff",
        "vision_flan(filtered)",
        "visual7w",
        "vizwiz(mathv360k)",
        "vqav2",
        "vsr",
        "websight",
        "wildvision",
        "yesbut",
    ),
    "Grounding & Counting": (
        "aguvis-stage-1",
        "groundui",
        "objects365_qa",
        "oodvqa",
        "tallyqa",
    ),
    "Mathematics": (
        "clevr",
        "clevr_math",
        "clevr_math(mathv360k)",
        "CoSyn_400k_math",
        "geo170k(align)",
        "geo170k(qa)",
        "geo3k",
        "geometry3k(mathv360k)",
        "geomverse",
        "geoqa+(mathv360k)",
        "geos(mathv360k)",
        "intergps",
        "mavis_math_metagen",
        "mavis_math_rule_geo",
        "raven",
        "super_clevr(mathv360k)",
        "unigeo(mathv360k)",
    ),
    "Naive OCR": (
        "art",
        "captcha",
        "chrome_writting",
        "cocotext",
        "ctw",
        "funsd",
        "hme100k",
        "hw_squad",
        "iam",
        "iiit5k",
        "imgur5k",
        "k12_printing",
        "latex_handwritten",
        "latexformulas",
        "maptext",
        "mathwriting-google",
        "memotion",
        "orand_car_a",
        "rendered_text",
        "sroie",
        "svrd",
        "SynthCodeNet",
        "synthdog",
        "SynthFormulaNet",
        "tal_ocr_eng",
        "wordart",
        "olmOCR-mix-0225-documents",
        "olmOCR-mix-0225-books",
    ),
    "OCR QA": (
        "a_okvqa",
        "aokvqa",
        "arxivqa",
        "bentham",
        "blockdiagramcomputerized",
        "blockdiagramhandwritten",
        "CoSyn_400k_diagram",
        "CoSyn_400k_document",
        "CoSyn_400k_music",
        "CoSyn_400k_nutrition",
        "diagram_image_to_text",
        "DoclingMatix",
        "docvqa",
        "est_vqa",
        "handwriting_forms",
        "infographic_vqa",
        "infographic_vqa_llava_format",
        "infographic(gpt4v)",
        "invoices_receipts",
        "mapqa",
        "mapqa(mathv360k)",
        "mmsoc_memotion",
        "ocrvqa",
        "pdfvqa",
        "screen2words",
        "screenqa",
        "slidevqa",
        "st_vqa",
        "sujet_finance",
        "textocr(gpt4v)",
        "textvqa",
        "ureader_cap",
        "ureader_ie",
        "ureader_kg_processed",
        "ureader_qa_processed",
        "visualmrc",
    ),
    "Science": (
        "ai2d_merged",
        "CoSyn_400k_chemical",
        "CoSyn_400k_circuit",
        "pathvqa",
        "pmc_vqa(mathv360k)",
        "scienceqa",
        "scienceqa(nona_context)",
        "tqa",
        "visualwebinstruct(filtered)",
        "vqarad",
    ),
    "Text-only": (
        "text_code_feedback",
        "text_codefeedback_filtered_instruction",
        "text_infinitymath",
        "text_mathinstruct",
        "text_mathqa",
        "text_mathstepdpo10k",
        "text_numinamath_cot",
        "text_openhermes_2_5",
        "text_openorca",
        "text_orcamath",
        "text_pythoncode25k",
        "text_pythoncodealpaca",
        "text_ruozhiba",
        "text_theoremqa",
        "text_wizardlm_evol",
        "text_OpenMathInstruct-2",
    ),
}


def sha256(value: bytes) -> str:
    """Return the SHA-256 identity of bytes."""
    return hashlib.sha256(value).hexdigest()


def slug(name: str) -> str:
    """Return a filesystem-safe lowercase form of a subset name."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def subsets() -> list[tuple[int, str, str]]:
    """Return (number, subset, category) in research-map order.

    Raises:
        SystemExit: If the research map and the category grouping disagree.
    """
    category_of = {name: category for category, names in CATEGORIES.items() for name in names}
    # The main table comes before the first section heading; later sections reuse its numbers.
    table = MAPPING_DOC.read_text(encoding="utf-8").split("\n## ", 1)[0]
    rows = [
        (int(match.group(1)), match.group(2))
        for line in table.splitlines()
        if (match := MAPPING_ROW.match(line))
    ]
    numbers = [number for number, _ in rows]
    names = [name for _, name in rows]
    if numbers != list(range(1, 186)) or len(set(names)) != 185:
        raise SystemExit("The research map must list subsets 1-185 exactly once")
    if set(names) != set(category_of) or len(category_of) != 185:
        raise SystemExit("The research map and the category grouping list different subsets")
    return [(number, name, category_of[name]) for number, name in rows]


def ingest_problems(raw: bytes) -> tuple[dict[str, Any] | None, list[str]]:
    """Describe an image and list the ingest limits it fails.

    The checks follow `pixelogue.images.canonicalize_image`: size, format, frame count, pixel
    count, EXIF orientation, ICC conversion to sRGB and the minimum short edge.
    """
    problems: list[str] = []
    if len(raw) > MAX_INPUT_BYTES:
        problems.append("bytes")
    try:
        with Image.open(io.BytesIO(raw)) as decoded:
            facts = {
                "format": decoded.format,
                "width": decoded.width,
                "height": decoded.height,
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
            if decoded.format not in EXTENSIONS:
                problems.append(f"format:{decoded.format}")
            if getattr(decoded, "n_frames", 1) != 1:
                problems.append("frames")
            if decoded.width * decoded.height > MAX_DECODED_PIXELS:
                problems.append("pixels")
                return facts, problems
            decoded.load()
            oriented = ImageOps.exif_transpose(decoded)
            icc = oriented.info.get("icc_profile")
            if icc:
                try:
                    converted = ImageCms.profileToProfile(
                        oriented,
                        ImageCms.ImageCmsProfile(io.BytesIO(icc)),
                        ImageCms.createProfile("sRGB"),
                    )
                    if converted is None:
                        problems.append("icc")
                except (OSError, ValueError, ImageCms.PyCMSError):
                    problems.append("icc")
            if min(oriented.size) < MIN_SHORT_EDGE:
                problems.append(f"edge:{min(oriented.size)}")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, SyntaxError):
        return None, ["decode"]
    return facts, problems


def derive(raw: bytes) -> tuple[bytes, list[str]]:
    """Minimally transform an ineligible image so the ingest limits hold.

    The first frame is kept, an unusable ICC profile is dropped, the image is downscaled to the
    pixel limit if needed, a short edge is padded with white to the minimum, and the result is
    stored losslessly as PNG.

    Raises:
        ValueError: If the bytes cannot be decoded at all.
    """
    steps: list[str] = []
    with Image.open(io.BytesIO(raw)) as decoded:
        decoded.seek(0)
        image = ImageOps.exif_transpose(decoded.copy())
        if decoded.format not in EXTENSIONS:
            steps.append(f"converted_from:{decoded.format}")
        if getattr(decoded, "n_frames", 1) != 1:
            steps.append("first_frame_only")
    if image.info.get("icc_profile"):
        try:
            converted = ImageCms.profileToProfile(
                image,
                ImageCms.ImageCmsProfile(io.BytesIO(image.info["icc_profile"])),
                ImageCms.createProfile("sRGB"),
            )
            if converted is None:
                raise ValueError("ICC conversion returned no image")
            image = converted
        except (OSError, ValueError, ImageCms.PyCMSError):
            steps.append("icc_profile_dropped")
        image.info.pop("icc_profile", None)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    image = image.convert("RGB")
    if image.width * image.height > MAX_DECODED_PIXELS:
        scale = (MAX_DECODED_PIXELS / (image.width * image.height)) ** 0.5
        size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
        steps.append(f"downscaled_to:{size[0]}x{size[1]}")
    if min(image.size) < MIN_SHORT_EDGE:
        width, height = max(image.width, MIN_SHORT_EDGE), max(image.height, MIN_SHORT_EDGE)
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
        steps.append(f"padded_from:{image.width}x{image.height}")
        image = canvas
    encoded = io.BytesIO()
    image.save(encoded, "PNG", optimize=False)
    steps.append("stored_as_png")
    return encoded.getvalue(), steps


def first_shard(fs: HfFileSystem, subset: str) -> tuple[str, int]:
    """Return the first train shard path of a subset and the shard count.

    Raises:
        RuntimeError: If the subset directory has no train shard.
    """
    base = f"datasets/{DATASET}@{REVISION}/{subset}"
    shards = sorted(
        (match.group(0), int(match.group(2)))
        for path in fs.ls(base, detail=False)
        if (match := SHARD.search(str(path)))
    )
    if not shards:
        raise RuntimeError(f"No train shard for {subset}")
    return f"{base}/{shards[0][0]}", shards[0][1]


def read_first_row_group(fs: HfFileSystem, path: str) -> tuple[list[dict[str, Any]], int]:
    """Read every column of the first row group as Python rows, with the group's row count."""
    with fs.open(path, "rb", block_size=64 * 1024 * 1024) as handle:
        parquet = pq.ParquetFile(handle)
        names = parquet.schema_arrow.names
        columns = [name for name in ("images", "texts", "source", *RATING_COLUMNS) if name in names]
        table = parquet.read_row_group(0, columns=columns)
    return table.to_pylist(), table.num_rows


def image_bytes(row: dict[str, Any]) -> list[bytes]:
    """Return the encoded bytes of every image in a row, in order."""
    return [item["bytes"] for item in row.get("images") or [] if item and item.get("bytes")]


def select_row(
    rows: list[dict[str, Any]], excluded: set[int]
) -> tuple[int, bytes | None, list[str], list[dict[str, Any]]]:
    """Pick the lowest eligible row, or derive the first usable image when none qualifies.

    Returns the row index, the evaluation image bytes (None for text-only rows), the derivation
    steps and the reasons earlier rows were skipped.
    """
    skipped: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index in excluded:
            skipped.append({"row": index, "problems": ["excluded_by_operator"]})
            continue
        images = image_bytes(row)
        if not images:
            skipped.append({"row": index, "problems": ["no_image"]})
            continue
        _, problems = ingest_problems(images[0])
        if not problems:
            return index, images[0], [], skipped
        skipped.append({"row": index, "problems": problems})
    for index, row in enumerate(rows):
        images = image_bytes(row)
        if index in excluded or not images:
            continue
        try:
            derived, steps = derive(images[0])
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            continue
        if not ingest_problems(derived)[1]:
            return index, derived, steps, [item for item in skipped if item["row"] < index]
    first_text = next((index for index in range(len(rows)) if index not in excluded), 0)
    return first_text, None, [], skipped[:first_text]


def fetch_subset(
    fs: HfFileSystem,
    number: int,
    subset: str,
    category: str,
    destination: Path,
    excluded: set[int],
) -> dict[str, Any]:
    """Fetch, select and store one subset's row; return its manifest record."""
    started = time.time()
    shard, shard_count = first_shard(fs, subset)
    rows, group_rows = read_first_row_group(fs, shard)
    row_index, chosen, derivation, skipped = select_row(rows, excluded)
    row = rows[row_index]
    images = image_bytes(row)
    stem = f"{number:03d}-{slug(subset)}"
    image_records = []
    for position, raw in enumerate(images):
        facts, problems = ingest_problems(raw)
        extension = EXTENSIONS.get((facts or {}).get("format") or "", "bin")
        target = destination / "row-images" / f"{stem}-{position}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        image_records.append(
            {
                "index": position,
                "path": target.relative_to(destination).as_posix(),
                **(facts or {"sha256": sha256(raw), "bytes": len(raw)}),
                "ingest_problems": problems,
            }
        )
    selected: dict[str, Any] | None = None
    if chosen is not None:
        facts, problems = ingest_problems(chosen)
        if problems or facts is None:
            raise RuntimeError(f"{subset}: selected image still fails {problems}")
        extension = EXTENSIONS[facts["format"]]
        target = destination / "images" / f"{stem}.{extension}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(chosen)
        selected = {
            "path": target.relative_to(destination).as_posix(),
            "source_image_index": 0,
            "derivation": derivation,
            **facts,
        }
    texts = [
        {"user": turn.get("user"), "assistant": turn.get("assistant")}
        for turn in row.get("texts") or []
    ]
    private = {
        "number": number,
        "subset": subset,
        "category": category,
        "revision": REVISION,
        "shard": shard.split("@", 1)[1].split("/", 1)[1],
        "shard_count": shard_count,
        "row_group": 0,
        "row_group_rows": group_rows,
        "row_index": row_index,
        "source": row.get("source"),
        "texts": texts,
        "ratings": {name: row[name] for name in RATING_COLUMNS if name in row},
        "images": image_records,
        "selected_image": selected,
        "skipped_rows": skipped,
        "excluded_rows": sorted(excluded),
        "text_only": not images,
    }
    rows_dir = destination / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    (rows_dir / f"{stem}.json").write_text(
        json.dumps(private, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "number": number,
                "subset": subset,
                "row": row_index,
                "images": len(images),
                "turns": len(texts),
                "selected": selected["path"] if selected else None,
                "derivation": derivation,
                "skipped": len(skipped),
                "seconds": round(time.time() - started, 1),
            }
        ),
        flush=True,
    )
    return manifest_record(private)


def manifest_record(private: dict[str, Any]) -> dict[str, Any]:
    """Project a private row record onto the committed identity fields."""
    selected = private["selected_image"]
    return {
        "number": private["number"],
        "subset": private["subset"],
        "category": private["category"],
        "revision": private["revision"],
        "shard": private["shard"],
        "row_index": private["row_index"],
        "row_turns": len(private["texts"]),
        "row_images": [
            {key: image.get(key) for key in ("format", "width", "height", "bytes", "sha256")}
            for image in private["images"]
        ],
        "text_only": private["text_only"],
        "skipped_rows": len(private["skipped_rows"]),
        "excluded_rows": private["excluded_rows"],
        "evaluation_image": None
        if selected is None
        else {
            "file": selected["path"],
            "format": selected["format"],
            "width": selected["width"],
            "height": selected["height"],
            "bytes": selected["bytes"],
            "sha256": selected["sha256"],
            "derivation": selected["derivation"],
        },
    }


def write_records(destination: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write evaluation-only source and rights records for every selected image."""
    sources, rights = [], []
    for record in records:
        image = record["evaluation_image"]
        if image is None:
            continue
        source_id = f"finevision-eval:{record['number']:03d}"
        sources.append(
            {
                "source_id": source_id,
                "image_path": image["file"],
                "source_group_ids": [f"finevision:{record['subset']}:{record['row_index']}"],
                "rights_record_id": source_id,
                "purpose": "evaluation",
                "dataset": f"{DATASET} {record['subset']}",
                "dataset_split": DATASET_SPLIT,
                "dataset_image_id": f"{record['shard']}#row={record['row_index']}&image=0",
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": source_id,
                "license_uri": DATASET_URL,
                "attribution": (
                    f"{DATASET} subset {record['subset']}, {record['shard']} row "
                    f"{record['row_index']} (revision {REVISION[:7]}); the original subset "
                    "licence applies"
                ),
                "processing_allowed": True,
                "qa_redistribution_allowed": False,
                "image_redistribution_allowed": False,
                "training_allowed": False,
                "valid_from": VALID_FROM,
                "valid_until": None,
            }
        )
    for name, rows in (("sources.jsonl", sources), ("rights.jsonl", rights)):
        (destination / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    return len(sources)


def saved_record(destination: Path, number: int, subset: str) -> dict[str, Any] | None:
    """Return a completed subset's manifest record when its saved files are intact."""
    path = destination / "rows" / f"{number:03d}-{slug(subset)}.json"
    if not path.exists():
        return None
    private = json.loads(path.read_text(encoding="utf-8"))
    files = [(item["path"], item["sha256"]) for item in private["images"]]
    if private["selected_image"] is not None:
        files.append((private["selected_image"]["path"], private["selected_image"]["sha256"]))
    for relative, digest in files:
        file = destination / relative
        if not file.exists() or sha256(file.read_bytes()) != digest:
            return None
    return manifest_record(private)


def repeated_images(destination: Path, records: Iterable[dict[str, Any]]) -> list[list[str]]:
    """Group subsets whose evaluation images decode to identical pixels."""
    by_pixels: dict[str, list[str]] = {}
    for record in records:
        if record["evaluation_image"] is None:
            continue
        with Image.open(destination / record["evaluation_image"]["file"]) as image:
            pixels = f"{image.size}".encode() + image.convert("RGBA").tobytes()
        by_pixels.setdefault(sha256(pixels), []).append(record["subset"])
    return [sorted(subsets) for subsets in by_pixels.values() if len(subsets) > 1]


def parse_exclusions(values: list[str]) -> dict[str, set[int]]:
    """Parse repeated SUBSET:ROW options into per-subset row sets."""
    excluded: dict[str, set[int]] = {}
    for value in values:
        subset, _, row = value.rpartition(":")
        if not subset or not row.isdigit():
            raise SystemExit(f"--exclude-row expects SUBSET:ROW, got {value!r}")
        excluded.setdefault(subset, set()).add(int(row))
    return excluded


def main() -> int:
    """Fetch every subset (or the named ones) and write records and the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DESTINATION)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--only", action="append", default=[], help="fetch only this subset")
    parser.add_argument("--exclude-row", action="append", default=[], metavar="SUBSET:ROW")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    plan = subsets()
    names = {name for _, name, _ in plan}
    unknown = sorted(set(args.only) - names)
    if unknown:
        raise SystemExit(f"Unknown subsets: {unknown}")
    excluded = parse_exclusions(args.exclude_row)
    if set(excluded) - names:
        raise SystemExit(f"Unknown subsets in --exclude-row: {sorted(set(excluded) - names)}")
    for subset, rows in REPEATED_ROWS.items():
        excluded.setdefault(subset, set()).update(rows)
    fs = HfFileSystem()
    remote = {
        str(path).rsplit("/", 1)[-1]
        for path in fs.ls(f"datasets/{DATASET}@{REVISION}", detail=False)
    }
    if not names <= remote:
        raise SystemExit(f"Subsets missing at the pinned revision: {sorted(names - remote)}")
    args.output.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict[str, Any]] = {}
    todo = []
    for number, subset, category in plan:
        if args.only and subset not in args.only:
            continue
        saved = None if subset in excluded else saved_record(args.output, number, subset)
        if saved is not None and saved["excluded_rows"] == sorted(excluded.get(subset, set())):
            records[number] = saved
        else:
            todo.append((number, subset, category))
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                fetch_with_retry, fs, number, subset, category, args.output, excluded
            ): subset
            for number, subset, category in todo
        }
        for future in as_completed(futures):
            subset = futures[future]
            try:
                record = future.result()
                records[record["number"]] = record
            except Exception as error:  # noqa: BLE001 - reported per subset, then the run fails
                failures[subset] = f"{type(error).__name__}: {error}"[:400]
                print(json.dumps({"subset": subset, "error": failures[subset]}), flush=True)
    if repeated := repeated_images(args.output, records.values()):
        failures["repeated_images"] = json.dumps(repeated)
    summary = {
        "revision": REVISION,
        "subsets": len(records),
        "evaluation_images": sum(r["evaluation_image"] is not None for r in records.values()),
        "text_only": sum(r["text_only"] for r in records.values()),
        "derived": sorted(
            r["subset"]
            for r in records.values()
            if r["evaluation_image"] and r["evaluation_image"]["derivation"]
        ),
        "failures": failures,
    }
    (args.output / "fetch-summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary), flush=True)
    if failures or len(records) != 185:
        return 1
    ordered = [records[number] for number in sorted(records)]
    count = write_records(args.output, ordered)
    args.manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in ordered),
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(args.manifest), "sources": count}), flush=True)
    return 0


def fetch_with_retry(
    fs: HfFileSystem,
    number: int,
    subset: str,
    category: str,
    destination: Path,
    excluded: dict[str, set[int]],
) -> dict[str, Any]:
    """Fetch one subset with bounded retries for transient network failures."""
    for attempt in range(1, 4):
        try:
            return fetch_subset(
                fs, number, subset, category, destination, excluded.get(subset, set())
            )
        except (OSError, TimeoutError, RuntimeError) as error:
            if attempt == 3:
                raise
            print(
                json.dumps({"subset": subset, "retry": attempt, "error": str(error)[:200]}),
                file=sys.stderr,
                flush=True,
            )
            time.sleep(20 * attempt)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
