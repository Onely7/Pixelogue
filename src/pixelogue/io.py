"""Typed JSON and JSON Lines file boundaries."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ValidationError

from pixelogue.errors import ExternalInputError
from pixelogue.serialization import strict_json_object


def read_json[Model: BaseModel](path: Path, model: type[Model]) -> Model:
    """Read one strict JSON object into a typed contract."""
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ExternalInputError("FILE_UNREADABLE", str(error)) from error
    strict_json_object(payload)
    try:
        return model.model_validate_json(payload)
    except ValidationError as error:
        raise ExternalInputError("CONTRACT_MISMATCH", str(error)) from error


def read_jsonl[Model: BaseModel](path: Path, model: type[Model]) -> list[Model]:
    """Read independent strict JSON objects from a JSON Lines file."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExternalInputError("FILE_UNREADABLE", str(error)) from error
    output: list[Model] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        strict_json_object(line)
        try:
            output.append(model.model_validate_json(line))
        except ValidationError as error:
            raise ExternalInputError(
                "CONTRACT_MISMATCH",
                f"{path}:{number}: {error}",
            ) from error
    return output


def write_json(path: Path, value: object) -> None:
    """Atomically write readable JSON."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode()
    _atomic_write(path, payload + b"\n")


def write_jsonl(path: Path, values: Iterable[object]) -> None:
    """Atomically write one compact JSON object per line."""
    rows: list[str] = []
    for value in values:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        rows.append(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    _atomic_write(path, ("\n".join(rows) + ("\n" if rows else "")).encode())


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
