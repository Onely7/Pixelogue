"""Static code parsing rejects executable inputs before any browser is launched."""

from __future__ import annotations

import base64
import io
import json
import runpy
import subprocess
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from pixelogue.contracts import GateVerdict
from pixelogue.specialist_env import call_renderer, environment_error, worker_root
from pixelogue.specialist_render import RenderSource, _pixel_metrics, verify_render
from pixelogue.task_evidence import ImageRegion


def _parse(format_name: str, code: object) -> dict[str, object]:
    result = subprocess.run(
        [str(worker_root() / ".venv/bin/python"), str(worker_root() / "render_worker.py")],
        input=json.dumps({"format": format_name, "code": code, "width": 400, "height": 300}),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    return json.loads(result.stdout)


def test_svg_and_html_reject_executable_syntax() -> None:
    assert (
        _parse("svg", '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')[
            "verdict"
        ]
        == "UNKNOWN"
    )
    assert (
        _parse(
            "svg",
            '<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.com/x"/></svg>',
        )["verdict"]
        == "UNKNOWN"
    )
    assert (
        _parse("html_css", {"html": "<button onclick='alert(1)'>Go</button>", "css": ""})["verdict"]
        == "UNKNOWN"
    )
    assert (
        _parse(
            "html_css",
            {"html": "<div>Go</div>", "css": "div{background:url(https://example.com/x)}"},
        )["verdict"]
        == "UNKNOWN"
    )


def test_tikz_subset_never_executes_tex_commands() -> None:
    assert _parse("tikz_subset", r"\input{/etc/passwd}")["verdict"] == "UNKNOWN"


def test_allowed_static_grammars_parse_without_running_tex_or_browser() -> None:
    validate = runpy.run_path(str(worker_root() / "render_worker.py"))["validate_markup"]
    svg, labels = validate(
        "svg",
        '<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">A</text></svg>',
        100,
        100,
    )
    assert labels == ("A",) and "<text" in svg
    tikz, labels = validate(
        "tikz_subset", r"\draw (0.1,0.2) -- (0.8,0.2); \node at (0.5,0.5) {B};", 100, 100
    )
    assert labels == ("B",) and "<line" in tikz
    html, labels = validate(
        "html_css", {"html": "<button>Go</button>", "css": "button{color:#000}"}, 100, 100
    )
    assert labels == ("Go",) and "<button" in html


def test_isolated_renderer_draws_and_checks_static_svg_when_host_supports_it(
    tmp_path: Path,
) -> None:
    """Exercise the real OS isolation and screenshot path on a capable host."""
    if error := environment_error("renderer", "playwright"):
        pytest.skip(f"isolated renderer unavailable: {error}")
    code = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60">'
        '<rect x="10" y="10" width="40" height="30" fill="black"/></svg>'
    )
    result = call_renderer({"format": "svg", "code": code, "width": 80, "height": 60})
    assert result["verdict"] == "MET"
    assert result["isolation_attempts"] in {1, 2}
    image_bytes = base64.b64decode(result["png_base64"], validate=True)
    with Image.open(io.BytesIO(image_bytes)) as image:
        assert image.size == (80, 60)
    reference = tmp_path / "reference.png"
    reference.write_bytes(image_bytes)
    source = RenderSource(
        coverage="MET",
        domain="test-svg",
        scope_id="whole",
        view_id="test-view",
        scope_region=ImageRegion(left=0, top=0, right=1, bottom=1),
        visible_text=(),
        reason="The rectangle and background are complete",
    )
    args = (
        (source, source),
        "test-svg",
        "svg",
        "whole",
        "test-view",
        [{"view_id": "test-view", "width": "80", "height": "60"}],
        reference,
    )
    assert verify_render(*args, json.dumps({"format": "svg", "code": code}))[0] is GateVerdict.MET
    blank = '<svg xmlns="http://www.w3.org/2000/svg" width="80" height="60"></svg>'
    assert (
        verify_render(*args, json.dumps({"format": "svg", "code": blank}))[0] is GateVerdict.NOT_MET
    )


def test_sparse_blank_render_does_not_pass_foreground_comparison(
    tmp_path: Path, monkeypatch
) -> None:
    reference = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(reference).line((10, 50, 90, 50), fill="black", width=2)
    path = tmp_path / "reference.png"
    reference.save(path)
    blank = Image.new("RGB", (100, 100), "white")
    source = RenderSource(
        coverage="MET",
        domain="simple-svg",
        scope_id="scope",
        view_id="view",
        scope_region=ImageRegion(left=0, top=0, right=1, bottom=1),
        visible_text=(),
        reason="Complete diagram",
    )
    mean, overlap = _pixel_metrics(reference, blank, source.scope_region)
    assert mean < 0.12 and overlap < 0.80

    buffer = io.BytesIO()
    reference.save(buffer, format="PNG")
    monkeypatch.setattr(
        "pixelogue.specialist_render.call_renderer",
        lambda request: {
            "verdict": "MET",
            "texts": [],
            "png_base64": base64.b64encode(buffer.getvalue()).decode(),
        },
    )
    args = (
        (source, source),
        source.domain,
        "svg",
        "scope",
        "view",
        [{"view_id": "view", "width": "100", "height": "100"}],
        path,
    )
    candidate = json.dumps(
        {"format": "svg", "code": '<svg xmlns="http://www.w3.org/2000/svg"></svg>'}
    )
    assert verify_render(*args, candidate)[0] is GateVerdict.MET
    duplicated = candidate.replace('"format": "svg"', '"format": "html_css", "format": "svg"')
    assert verify_render(*args, duplicated)[0] is GateVerdict.UNKNOWN
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")
    monkeypatch.setattr(
        "pixelogue.specialist_render.call_renderer",
        lambda request: {
            "verdict": "MET",
            "texts": [],
            "png_base64": base64.b64encode(buffer.getvalue()).decode(),
        },
    )
    assert verify_render(*args, candidate)[0] is GateVerdict.NOT_MET
