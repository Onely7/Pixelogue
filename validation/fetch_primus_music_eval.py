#!/usr/bin/env python3
"""Restore pinned PrIMuS score images and controller-only first-bar labels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "validation/primus_music_eval_manifest.jsonl"
DESTINATION = ROOT / "data/primus-music-eval"
ARCHIVE = DESTINATION / "primus.tgz"
ARCHIVE_URL = "https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz"
ARCHIVE_SHA256 = "fbc8dd5a20fa12d8b9c26b549f3bff1ce7c034ec740bb40ba50fc14d238f486f"
NS = {"m": "http://www.music-encoding.org/ns/mei"}
SUPPORTED = {"note", "rest"}


def _sha256(raw: bytes) -> str:
    """Return the full byte content identity."""
    return hashlib.sha256(raw).hexdigest()


def _archive() -> Path:
    """Restore and verify the official source archive outside Git."""
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        temporary = ARCHIVE.with_suffix(".partial")
        with (
            urllib.request.urlopen(ARCHIVE_URL, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(ARCHIVE)
    digest = hashlib.sha256()
    with ARCHIVE.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Official PrIMuS archive hash changed")
    return ARCHIVE


def _first_bar(raw: bytes) -> dict[str, Any] | None:
    """Accept only complete, simple G/F-clef bars with no accidental context."""
    try:
        root = ET.fromstring(raw)
        score = root.find(".//m:scoreDef", NS)
        staff = root.find(".//m:staffDef", NS)
        measure = root.find(".//m:measure", NS)
        if score is None or staff is None or measure is None:
            return None
        clef_mark = (staff.get("clef.shape"), staff.get("clef.line"))
        if clef_mark not in {("G", "2"), ("F", "4")}:
            return None
        clef = "treble" if clef_mark == ("G", "2") else "bass"
        base_index = 30 if clef == "treble" else 18
        if score.get("key.sig") not in {None, "0"}:
            return None
        top, bottom = int(score.get("meter.count", "0")), int(score.get("meter.unit", "0"))
        if not 1 <= top <= 12 or bottom not in {2, 4, 8}:
            return None
        if any(
            measure.find(f".//m:{tag}", NS) is not None
            for tag in ("chord", "tuplet", "mRest", "clef", "keySig", "meterSig", "tie")
        ):
            return None
        symbols = [item for item in measure.iter() if item.tag.rsplit("}", 1)[-1] in SUPPORTED]
        if not 1 <= len(symbols) <= 16:
            return None
        answer_events: list[dict[str, Any]] = []
        source_events: list[dict[str, Any]] = []
        total = Fraction()
        for item in symbols:
            if any(item.get(name) is not None for name in ("grace", "accid", "accid.ges", "tie")):
                return None
            base, dots = int(item.get("dur", "0")), int(item.get("dots", "0"))
            if base not in {1, 2, 4, 8, 16} or dots not in {0, 1}:
                return None
            length = Fraction(4, base) * (Fraction(3, 2) if dots else 1)
            total += length
            if item.tag.endswith("note"):
                letter, octave = item.get("pname"), item.get("oct")
                if (
                    letter not in {"a", "b", "c", "d", "e", "f", "g"}
                    or octave is None
                    or not octave.isdecimal()
                ):
                    return None
                pitch = letter.upper() + octave
                staff_step = int(octave) * 7 + "cdefgab".index(letter) - base_index
                if not -8 <= staff_step <= 16:
                    return None
                kind = "note"
            else:
                pitch = None
                staff_step = None
                kind = "rest"
            answer_events.append({"pitch": pitch, "duration": str(length), "tie": None})
            source_events.append(
                {
                    "kind": kind,
                    "staff_step": staff_step,
                    "base": base,
                    "dots": dots,
                    "accidental": None,
                    "tie": None,
                }
            )
        if total != Fraction(4 * top, bottom):
            return None
        if not any(item["kind"] == "note" for item in source_events):
            return None
        return {
            "clef": clef,
            "key_sharps": 0,
            "meter_top": top,
            "meter_bottom": bottom,
            "answer_events": answer_events,
            "source_events": source_events,
        }
    except (ET.ParseError, ValueError, TypeError):
        return None


def _member_stem(name: str) -> tuple[str, str] | None:
    """Reject AppleDouble and unrelated archive entries without extracting paths."""
    path = Path(name)
    if path.name.startswith("._") or path.suffix not in {".mei", ".png"}:
        return None
    if path.parent.name != path.stem or not name.startswith("./package_"):
        return None
    return path.stem, path.suffix


def _process_image(raw: bytes) -> tuple[bytes, int, int]:
    """Enlarge a thin score strip by exactly two for the model-visible view."""
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
            raise ValueError("PrIMuS score is not one PNG")
        resized = image.convert("RGB").resize(
            (image.width * 2, image.height * 2), Image.Resampling.BICUBIC
        )
        width, height = resized.size
        if min(width, height) < 128 or width * height > 4_000_000:
            raise ValueError("PrIMuS score remains outside image limits")
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
    return buffer.getvalue(), width, height


def _collect(
    archive: Path, selected: set[str] | None
) -> list[tuple[dict[str, Any], bytes, dict[str, Any]]]:
    """Stream score pairs without unpacking the full 87k-item corpus."""
    pending: dict[str, dict[str, bytes]] = {}
    found: list[tuple[dict[str, Any], bytes, dict[str, Any]]] = []
    works: set[str] = set()
    with tarfile.open(archive, mode="r|gz") as source:
        for member in source:
            identity = _member_stem(member.name)
            if identity is None or not member.isfile():
                continue
            stem, suffix = identity
            if selected is not None and stem not in selected:
                continue
            if member.size > 20_000_000:
                continue
            stream = source.extractfile(member)
            if stream is None:
                continue
            pair = pending.setdefault(stem, {})
            pair[suffix] = stream.read()
            if len(pair) != 2:
                continue
            pending.pop(stem)
            label = _first_bar(pair[".mei"])
            if label is None:
                if selected is not None:
                    raise ValueError(f"Pinned PrIMuS notation changed: {stem}")
                continue
            try:
                visible, width, height = _process_image(pair[".png"])
            except ValueError:
                if selected is not None:
                    raise
                continue
            work = stem.split("-", 1)[0]
            if selected is None and work in works:
                continue
            works.add(work)
            record = {
                "sample_id": stem,
                "work_id": work,
                "raw_sha256": _sha256(pair[".png"]),
                "mei_sha256": _sha256(pair[".mei"]),
                "label_sha256": _sha256(json.dumps(label, sort_keys=True).encode()),
                "processed_sha256": _sha256(visible),
                "width": width,
                "height": height,
                "source_group_id": "primus-work:" + work,
            }
            found.append((record, visible, label))
            if len(found) == 89:
                break
    if len(found) != 89:
        raise ValueError(f"Expected 89 distinct supported PrIMuS scores, found {len(found)}")
    return found


def main() -> None:
    """Freeze or verify a disjoint development and confirmation sample."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    args = parser.parse_args()
    archive = _archive()
    if args.freeze:
        if MANIFEST.exists():
            raise ValueError("Refusing to replace a pinned PrIMuS manifest")
        records = _collect(archive, None)
        records.sort(key=lambda item: _sha256(f"primus-music-v1:{item[0]['sample_id']}".encode()))
        manifest = []
        for index, (record, _, _) in enumerate(records):
            record["split"] = "development" if index < 10 else "confirmation"
            manifest.append(record)
        MANIFEST.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in manifest))
    else:
        manifest = [json.loads(line) for line in MANIFEST.read_text().splitlines()]
        if (
            len(manifest) != 89
            or [item["split"] for item in manifest] != ["development"] * 10 + ["confirmation"] * 79
        ):
            raise ValueError("PrIMuS split changed")
        if len({item["work_id"] for item in manifest}) != 89:
            raise ValueError("PrIMuS score groups are not independent")
        records = _collect(archive, {item["sample_id"] for item in manifest})
        current = {record["sample_id"]: (record, image, label) for record, image, label in records}
        records = []
        for item in manifest:
            record, image, label = current[item["sample_id"]]
            record["split"] = item["split"]
            if record != item:
                raise ValueError(f"Pinned PrIMuS score changed: {item['sample_id']}")
            records.append((record, image, label))
    output = args.destination
    (output / "images").mkdir(parents=True, exist_ok=True)
    sources, rights, private_labels = [], [], []
    for record, visible, label in records:
        sample = record["sample_id"]
        path = output / "images" / f"{sample}.png"
        if path.exists() and _sha256(path.read_bytes()) != record["processed_sha256"]:
            raise ValueError(f"Existing PrIMuS image changed: {sample}")
        path.write_bytes(visible)
        source_id = "primus:" + sample
        sources.append(
            {
                "source_id": source_id,
                "image_path": f"images/{sample}.png",
                "source_group_ids": [record["source_group_id"]],
                "rights_record_id": source_id,
                "purpose": "evaluation",
                "dataset": "PrIMuS printed music incipits",
                "dataset_split": record["split"],
                "dataset_image_id": sample,
                "rotation_degrees": 0,
            }
        )
        rights.append(
            {
                "rights_record_id": source_id,
                "license_uri": "https://grfia.dlsi.ua.es/primus/",
                "attribution": f"PrIMuS score {sample}; evaluation-only under operator testing authorization",
                "processing_allowed": True,
                "qa_redistribution_allowed": False,
                "image_redistribution_allowed": False,
                "training_allowed": False,
                "valid_from": "2026-09-01T00:00:00Z",
                "valid_until": None,
            }
        )
        private_labels.append(
            {
                "source_id": source_id,
                "label": label,
                "split": record["split"],
                "label_provenance": ARCHIVE_URL + "#" + sample + ".mei",
            }
        )
    for name, rows in (
        ("sources.jsonl", sources),
        ("rights.jsonl", rights),
        ("private_labels.jsonl", private_labels),
    ):
        (output / name).write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows)
        )
    print(json.dumps({"restored": len(records), "development": 10, "confirmation": 79}))


if __name__ == "__main__":
    main()
