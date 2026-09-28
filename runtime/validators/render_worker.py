"""Parse bounded static markup and render it inside the caller's OS sandbox."""

from __future__ import annotations

import base64
import html
import json
import os
import re
import resource
import sys
from html.parser import HTMLParser
from xml.etree import ElementTree

SVG_TAGS = {"svg", "g", "rect", "circle", "ellipse", "line", "polyline", "polygon", "path", "text"}
SVG_ATTRIBUTES = {
    "xmlns",
    "width",
    "height",
    "viewBox",
    "x",
    "y",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "r",
    "rx",
    "ry",
    "points",
    "d",
    "fill",
    "stroke",
    "stroke-width",
    "font-size",
    "text-anchor",
    "transform",
    "opacity",
    "stroke-linecap",
    "stroke-linejoin",
}
HTML_TAGS = {
    "html",
    "body",
    "main",
    "section",
    "header",
    "footer",
    "div",
    "span",
    "p",
    "button",
    "input",
    "label",
    "h1",
    "h2",
    "h3",
    "ul",
    "li",
}
HTML_ATTRIBUTES = {"class", "id", "type", "placeholder", "value", "disabled"}
CSS_PROPERTIES = {
    "position",
    "top",
    "left",
    "right",
    "bottom",
    "width",
    "height",
    "display",
    "align-items",
    "justify-content",
    "gap",
    "padding",
    "margin",
    "color",
    "background",
    "background-color",
    "border",
    "border-radius",
    "font-family",
    "font-size",
    "font-weight",
    "text-align",
    "line-height",
    "box-sizing",
    "overflow",
    "flex-direction",
    "grid-template-columns",
}


def _svg(code: str) -> tuple[str, tuple[str, ...]]:
    """Accept a small closed vector grammar with no script, reference or stylesheet."""
    if "<!" in code or len(code) > 65_536:
        raise ValueError("SVG contains a declaration or exceeds limit")
    root = ElementTree.fromstring(code)
    elements = list(root.iter())
    if len(elements) > 128 or elements[0].tag not in {"svg", "{http://www.w3.org/2000/svg}svg"}:
        raise ValueError("SVG root or element count is invalid")
    texts = []
    for element in elements:
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in SVG_TAGS or any(attr not in SVG_ATTRIBUTES for attr in element.attrib):
            raise ValueError("Unsupported SVG element or attribute")
        if any(
            "url(" in value.lower() or "javascript:" in value.lower() or len(value) > 4096
            for value in element.attrib.values()
        ):
            raise ValueError("SVG external reference or oversized attribute")
        if tag == "text":
            value = "".join(element.itertext()).strip()
            if not value or len(value) > 256:
                raise ValueError("Invalid SVG label")
            texts.append(value)
        elif (element.text or "").strip():
            raise ValueError("Unexpected text in SVG geometry")
    return code, tuple(texts)


class _StaticHTML(HTMLParser):
    """Parse only static visible elements and text with no executable attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.count = 0
        self.stack: list[str] = []
        self.texts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.count += 1
        if (
            tag not in HTML_TAGS
            or self.count > 128
            or any(
                key not in HTML_ATTRIBUTES or value is None or len(value) > 256
                for key, value in attrs
            )
        ):
            raise ValueError("Unsupported HTML element or attribute")
        if tag == "input":
            if dict(attrs).get("type", "text") != "text":
                raise ValueError("Only static text input is supported")
        else:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack.pop() != tag:
            raise ValueError("Unbalanced HTML tags")

    def handle_data(self, data: str) -> None:
        value = data.strip()
        if value:
            if not self.stack or len(value) > 256:
                raise ValueError("Unbound or oversized HTML text")
            self.texts.append(value)

    def handle_decl(self, decl: str) -> None:
        if decl.lower() != "doctype html":
            raise ValueError("Unsupported HTML declaration")

    def unknown_decl(self, data: str) -> None:
        raise ValueError("Unsupported HTML declaration")


def _html_css(code: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    """Allow static HTML and a bounded declaration-only CSS grammar."""
    if not isinstance(code, dict) or set(code) != {"html", "css"}:
        raise ValueError("HTML/CSS code must contain only html and css")
    markup, css = code["html"], code["css"]
    if not isinstance(markup, str) or not isinstance(css, str) or len(markup) + len(css) > 65_536:
        raise ValueError("HTML/CSS size is invalid")
    if any(marker in markup.lower() for marker in ("<script", "<style", "<!--", "<![")):
        raise ValueError("Executable or hidden markup")
    parser = _StaticHTML()
    parser.feed(markup)
    parser.close()
    if parser.stack:
        raise ValueError("Unbalanced HTML")
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    if len(rules) > 128 or re.sub(r"[^{}]+\{[^{}]*\}", "", css).strip():
        raise ValueError("Unsupported CSS grammar")
    for selector, declarations in rules:
        if not re.fullmatch(r"\s*[.#]?[A-Za-z][A-Za-z0-9_-]{0,31}\s*", selector):
            raise ValueError("Unsupported CSS selector")
        for declaration in declarations.split(";"):
            if not declaration.strip():
                continue
            if ":" not in declaration:
                raise ValueError("Invalid CSS declaration")
            prop, value = declaration.split(":", 1)
            prop, value = prop.strip(), value.strip()
            if prop not in CSS_PROPERTIES or not re.fullmatch(
                r"[A-Za-z0-9#.,() %_+\-/]{1,128}", value
            ):
                raise ValueError("Unsupported CSS property or value")
            if any(word in value.lower() for word in ("url", "expression", "javascript", "import")):
                raise ValueError("Executable or external CSS")
    return (
        f"<!doctype html><html><head><style>{css}</style></head><body>{markup}</body></html>",
        tuple(parser.texts),
    )


def _tikz(code: str, width: int, height: int) -> tuple[str, tuple[str, ...]]:
    """Translate a finite line, arrow, rectangle and label grammar to SVG."""
    if len(code) > 16_384:
        raise ValueError("TikZ subset exceeds limit")
    number = r"(?:0(?:\.\d{1,4})?|1(?:\.0{1,4})?)"
    point = rf"\(\s*({number})\s*,\s*({number})\s*\)"
    line = re.compile(rf"\\draw(\[->\])?\s*{point}\s*(--|rectangle)\s*{point}\s*;")
    label = re.compile(rf"\\node\s+at\s*{point}\s*\{{([^{{}}\\]{{1,128}})\}}\s*;")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    ]
    texts = []
    remaining = code.strip()
    count = 0
    while remaining:
        count += 1
        if count > 128:
            raise ValueError("Too many TikZ subset commands")
        match = line.match(remaining)
        if match:
            arrow, x1, y1, relation, x2, y2 = match.groups()
            left, top, right, bottom = (
                float(x1) * width,
                float(y1) * height,
                float(x2) * width,
                float(y2) * height,
            )
            if relation == "rectangle" and arrow is None:
                parts.append(
                    f'<rect x="{min(left, right)}" y="{min(top, bottom)}" width="{abs(right - left)}" height="{abs(bottom - top)}" fill="none" stroke="black"/>'
                )
            elif relation == "--":
                parts.append(
                    f'<line x1="{left}" y1="{top}" x2="{right}" y2="{bottom}" stroke="black"/>'
                )
                if arrow:
                    parts.append(f'<circle cx="{right}" cy="{bottom}" r="3" fill="black"/>')
            else:
                raise ValueError("Unsupported TikZ relation")
        else:
            match = label.match(remaining)
            if match is None:
                raise ValueError("Unsupported TikZ subset syntax")
            x, y, value = match.groups()
            texts.append(value)
            parts.append(
                f'<text x="{float(x) * width}" y="{float(y) * height}" fill="black">{html.escape(value)}</text>'
            )
        remaining = remaining[match.end() :].strip()
    parts.append("</svg>")
    return "".join(parts), tuple(texts)


def validate_markup(
    format_name: str, code: str | dict[str, str], width: int, height: int
) -> tuple[str, tuple[str, ...]]:
    """Return isolated static page body and its leaf labels."""
    if format_name == "svg":
        if not isinstance(code, str):
            raise ValueError("SVG code must be a string")
        return _svg(code)
    if format_name == "tikz_subset":
        if not isinstance(code, str):
            raise ValueError("TikZ code must be a string")
        return _tikz(code, width, height)
    if format_name == "html_css":
        if not isinstance(code, dict):
            raise ValueError("HTML/CSS code must be an object")
        return _html_css(code)
    raise ValueError("Unsupported rendering format")


def main() -> None:
    """Render one bounded request without network, script or persistent browser state."""
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    if os.environ.get("PIXELOGUE_RENDER_CGROUP") != "1":
        resource.setrlimit(resource.RLIMIT_AS, (3_000_000_000, 3_000_000_000))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
    if os.environ.get("PIXELOGUE_RENDER_CGROUP") != "1":
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    try:
        request = json.loads(sys.stdin.read(100_001))
        width, height = request["width"], request["height"]
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or not 1 <= width <= 2048
            or not 1 <= height <= 2048
        ):
            raise ValueError("Unsupported renderer viewport")
        markup, texts = validate_markup(request["format"], request["code"], width, height)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                executable_path="/usr/bin/google-chrome",
                args=["--disable-gpu"],
            )
            context = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
                java_script_enabled=False,
                service_workers="block",
            )
            page = context.new_page()
            page.route("**/*", lambda route: route.abort())
            page.set_content(markup, wait_until="load", timeout=5000)
            screenshot = page.screenshot(full_page=False, timeout=5000)
            context.close()
            browser.close()
        result = {
            "verdict": "MET",
            "texts": texts,
            "png_base64": base64.b64encode(screenshot).decode("ascii"),
        }
    except (KeyError, TypeError, ValueError) as exc:
        result = {"verdict": "UNKNOWN", "reason": str(exc)[:240]}
    sys.stdout.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
