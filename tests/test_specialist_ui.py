"""UI target coordinates must bind to the delivered view and public action."""

from __future__ import annotations

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_ui import UIActionSource, verify_ui_action


def _source() -> UIActionSource:
    return UIActionSource.model_validate(
        {
            "coverage": "MET",
            "domain": "static-screen",
            "scope_id": "s",
            "view_id": "view-1",
            "scope_region": {"left": 0, "top": 0, "right": 1, "bottom": 1},
            "controls": (
                {
                    "control_id": "search",
                    "kind": "field",
                    "enabled": True,
                    "region": {"left": 0.2, "top": 0.1, "right": 0.5, "bottom": 0.2},
                },
            ),
            "reason": "Visible search field",
        }
    )


def test_click_point_checks_both_target_bounds_and_view_pixel_transform() -> None:
    source = _source()
    args = (
        (source, source),
        source.domain,
        {"target": "search", "action": "focus"},
        "s",
        "view-1",
        [{"view_id": "view-1", "width": "1000", "height": "500"}],
    )
    answer = (
        '{"action":"focus","target_id":"search","view_id":"view-1",'
        '"x":0.3,"y":0.15,"pixel_x":300,"pixel_y":75,"text":null}'
    )
    assert verify_ui_action(*args, answer)[0] is GateVerdict.MET
    assert (
        verify_ui_action(*args, answer.replace('"pixel_x":300', '"pixel_x":301'))[0]
        is GateVerdict.NOT_MET
    )
    assert verify_ui_action(*args, answer.replace('"x":0.3', '"x":0.7'))[0] is GateVerdict.NOT_MET


def test_input_needs_public_text_and_exact_view() -> None:
    source = _source()
    answer = (
        '{"action":"input","target_id":"search","view_id":"view-1",'
        '"x":0.3,"y":0.15,"pixel_x":300,"pixel_y":75,"text":"hello"}'
    )
    views = [{"view_id": "view-1", "width": "1000", "height": "500"}]
    assert (
        verify_ui_action(
            (source, source),
            source.domain,
            {"target": "search", "action": "input"},
            "s",
            "view-1",
            views,
            answer,
        )[0]
        is GateVerdict.UNKNOWN
    )
    assert (
        verify_ui_action(
            (source, source),
            source.domain,
            {"target": "search", "action": "input", "input_text": "hello"},
            "s",
            "wrong-view",
            views,
            answer,
        )[0]
        is GateVerdict.UNKNOWN
    )
