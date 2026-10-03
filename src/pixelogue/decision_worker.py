"""Serialized Jev-Omni JSONL worker for the separate pinned GPU runtime."""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

from pixelogue.decision import DecisionClient, DecisionRequest
from pixelogue.decision_serving import OMNI_REPOSITORY, LocalOmniDecisionClient
from pixelogue.errors import ExternalInputError, PixelogueError
from pixelogue.serialization import canonical_json, strict_json_object

LOGGER = logging.getLogger(__name__)


def _emit(output: TextIO, frame: dict[str, object]) -> None:
    output.write(canonical_json(frame).decode("utf-8") + "\n")
    output.flush()


def serve_decisions(client: DecisionClient, source: TextIO, output: TextIO) -> int:
    """Serve requests serially, retaining explicit errors without passing output.

    This loop is also usable with an already loaded native worker. Bad requests
    and typed inference errors do not become verdicts. Unexpected native runtime
    failures terminate the process after recording an error frame.
    """
    for line in source:
        request_id: str | None = None
        try:
            raw = strict_json_object(line)
            request_id = raw.get("request_id") if isinstance(raw.get("request_id"), str) else None
            request = DecisionRequest.model_validate_json(line)
        except (ExternalInputError, ValidationError) as error:
            _emit(
                output,
                {
                    "status": "error",
                    "request_id": request_id,
                    "reason": "DECISION_REQUEST_INVALID",
                    "message": str(error),
                },
            )
            continue
        try:
            with contextlib.redirect_stdout(sys.stderr):
                result = client.decide(request)
            result.validate_request(request)
        except PixelogueError as error:
            _emit(
                output,
                {
                    "status": "error",
                    "request_id": request_id,
                    "reason": error.reason,
                    "message": str(error),
                },
            )
            continue
        except Exception as error:
            LOGGER.exception("Omni native inference failed")
            _emit(
                output,
                {
                    "status": "error",
                    "request_id": request_id,
                    "reason": "DECISION_WORKER_INFERENCE_FAILED",
                    "message": str(error),
                },
            )
            return 7
        _emit(output, {"status": "result", "result": result.model_dump(mode="json")})
    return 0


def main(argv: list[str] | None = None) -> int:
    """Load a pinned local Omni snapshot and keep stdout reserved for JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    arguments = parser.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    output = sys.stdout
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.startswith("GPU-") or "," in gpu:
        _emit(
            output,
            {
                "status": "error",
                "reason": "DECISION_GPU_NOT_SELECTED",
                "message": "Set one explicitly selected physical GPU UUID before starting the worker",
            },
        )
        return 2
    try:
        with contextlib.redirect_stdout(sys.stderr):
            client = LocalOmniDecisionClient(arguments.snapshot_path, arguments.revision)
            client.load()
    except Exception as error:
        LOGGER.exception("Omni worker loading failed")
        _emit(
            output,
            {"status": "error", "reason": "DECISION_WORKER_LOAD_FAILED", "message": str(error)},
        )
        return 7
    _emit(
        output,
        {
            "status": "ready",
            "model_repository": OMNI_REPOSITORY,
            "model_revision": arguments.revision,
            "model_load_seconds": client.model_load_seconds,
        },
    )
    return serve_decisions(client, sys.stdin, output)


if __name__ == "__main__":
    raise SystemExit(main())
