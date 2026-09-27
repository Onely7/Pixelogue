"""Static code parsing rejects executable inputs before any browser is launched."""

from __future__ import annotations

import json
import subprocess

from pixelogue.specialist_env import worker_root


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
