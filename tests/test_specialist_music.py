"""Notation transcription must be grounded in two complete visible score readings."""

from __future__ import annotations

import json

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_music import MusicSource, verify_music


def _score() -> MusicSource:
    return MusicSource.model_validate(
        {
            "coverage": "MET",
            "domain": "single-voice-treble",
            "scope_id": "s",
            "view_id": "v",
            "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
            "clef": "treble",
            "key_sharps": 0,
            "meter_top": 4,
            "meter_bottom": 4,
            "bar_range": "1-1",
            "reason": "All symbols visible",
            "measures": (
                {
                    "number": 1,
                    "events": (
                        {
                            "kind": "note",
                            "staff_step": 0,
                            "base": 4,
                            "region": {"left": 0, "top": 0, "right": 0.2, "bottom": 0.2},
                        },
                        {
                            "kind": "note",
                            "staff_step": 1,
                            "base": 4,
                            "region": {"left": 0.2, "top": 0, "right": 0.4, "bottom": 0.2},
                        },
                        {
                            "kind": "rest",
                            "base": 2,
                            "region": {"left": 0.4, "top": 0, "right": 0.6, "bottom": 0.2},
                        },
                    ),
                },
            ),
        }
    )


def test_music21_score_accepts_exact_pitches_durations_and_rests() -> None:
    score = _score()
    answer = json.dumps(
        {
            "events": [
                {"pitch": "E4", "duration": "1", "tie": None},
                {"pitch": "F4", "duration": "1", "tie": None},
                {"pitch": None, "duration": "2", "tie": None},
            ]
        }
    )
    verdict, evidence = verify_music((score, score), score.domain, "1-1", "s", "v", answer)
    assert verdict is GateVerdict.MET
    assert evidence["verdict"] == "MET"
    incorrect = answer.replace("F4", "G4")
    assert (
        verify_music((score, score), score.domain, "1-1", "s", "v", incorrect)[0]
        is GateVerdict.NOT_MET
    )
    duplicated = answer.replace('"pitch": "E4"', '"pitch": "G4", "pitch": "E4"')
    assert (
        verify_music((score, score), score.domain, "1-1", "s", "v", duplicated)[0]
        is GateVerdict.UNKNOWN
    )


def test_music21_score_abstains_on_incomplete_or_disputed_reading() -> None:
    score = _score()
    other = score.model_copy(update={"key_sharps": 1})
    answer = '{"events":[]}'
    assert (
        verify_music((score, other), score.domain, "1-1", "s", "v", answer)[0]
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_music((score, score), score.domain, "2-2", "s", "v", answer)[0]
        is GateVerdict.UNKNOWN
    )


def test_key_signature_changes_pitch_without_changing_staff_position() -> None:
    score = _score().model_copy(update={"key_sharps": 1})
    answer = json.dumps(
        {
            "events": [
                {"pitch": "E4", "duration": "1", "tie": None},
                {"pitch": "F#4", "duration": "1", "tie": None},
                {"pitch": None, "duration": "2", "tie": None},
            ]
        }
    )
    assert verify_music((score, score), score.domain, "1-1", "s", "v", answer)[0] is GateVerdict.MET
