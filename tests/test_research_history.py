"""Strong history witnesses must point to a committed exact quote and change it."""

from __future__ import annotations

import pytest

from pixelogue.contracts import PublicMessage
from pixelogue.research_history import HistoryBinding, resolve_alternative_history


def _history() -> tuple[PublicMessage, ...]:
    return (
        PublicMessage(message_id="q1", turn_index=1, role="user", content="Count the red circles."),
        PublicMessage(message_id="a1", turn_index=1, role="assistant", content="There are three."),
    )


def test_binding_replaces_only_exact_prior_user_span() -> None:
    history = _history()
    binding = HistoryBinding(
        relation="public_constraint_change",
        source_message_id="q1",
        start=10,
        end=13,
        source_quote="red",
        alternative_quote="blue",
    )
    alternative = resolve_alternative_history(history, binding)
    assert alternative[0].content == "Count the blue circles."
    assert alternative[1] == history[1]
    with pytest.raises(ValueError):
        resolve_alternative_history(history, binding.model_copy(update={"source_quote": "blue"}))


def test_independent_and_paraphrase_do_not_create_strong_witness() -> None:
    history = _history()
    assert resolve_alternative_history(history, HistoryBinding(relation="independent")) == history
    binding = HistoryBinding(
        relation="coreference",
        source_message_id="q1",
        start=10,
        end=13,
        source_quote="red",
        alternative_quote="Red",
    )
    with pytest.raises(ValueError):
        resolve_alternative_history(history, binding)
