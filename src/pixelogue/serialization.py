"""Strict serialization and stable content identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from yaml import resolver

from pixelogue.errors import ConfigurationError, ExternalInputError


def canonical_json(value: Any) -> bytes:
    """Serialize a value to the project's canonical JSON representation.

    Raises:
        ValueError: If the value contains a non-finite number or is not serializable.
    """
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_hash(value: Any) -> str:
    """Return the SHA-256 identity of canonical JSON data."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExternalInputError("DUPLICATE_JSON_KEY", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ExternalInputError("NON_FINITE_JSON_NUMBER", f"Unsupported JSON number: {value}")


def strict_json_object(payload: str | bytes) -> dict[str, Any]:
    """Parse one strict JSON object.

    Duplicate keys, non-finite values, trailing data, and non-object roots are rejected.

    Raises:
        ExternalInputError: If the payload is not a strict JSON object.
    """
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except ExternalInputError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExternalInputError("INVALID_JSON", str(error)) from error
    if not isinstance(value, dict):
        raise ExternalInputError("JSON_ROOT_NOT_OBJECT", "JSON root must be an object")
    _assert_finite(value)
    return value


def _assert_finite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExternalInputError("NON_FINITE_JSON_NUMBER", "JSON numbers must be finite")
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite(child)


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError("NON_STRING_YAML_KEY", "Configuration keys must be strings")
        if key in result:
            raise ConfigurationError("DUPLICATE_YAML_KEY", f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML mapping while rejecting duplicate keys."""
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except OSError as error:
        raise ExternalInputError("CONFIG_UNREADABLE", str(error)) from error
    except yaml.YAMLError as error:
        raise ConfigurationError("INVALID_YAML", str(error)) from error
    if not isinstance(value, dict):
        raise ConfigurationError("CONFIG_ROOT_NOT_MAPPING", "Configuration root must be a mapping")
    return value
