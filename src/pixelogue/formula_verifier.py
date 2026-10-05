"""Literal formula-structure comparison for a bounded mathematical grammar."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field, model_validator

from pixelogue.config import StrictModel
from pixelogue.contracts import GateVerdict
from pixelogue.task_evidence import ImageRegion

SYMBOLS = {
    "\\alpha": "α",
    "\\beta": "β",
    "\\gamma": "γ",
    "\\theta": "θ",
    "\\pi": "π",
    "\\infty": "∞",
    "\\sum": "∑",
}
SUPERSCRIPTS = str.maketrans({"²": "^2", "³": "^3", "⁰": "^0", "¹": "^1", "⁴": "^4", "⁵": "^5"})
SUBSCRIPTS = str.maketrans({"₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4", "₅": "_5"})
TOKEN_RE = re.compile(r"\\[A-Za-z]+|[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[αβγθπ∞∑√=+\-*/×·^_{}()]|\S")


class FormulaNode(StrictModel):
    """A symbol, explicit grouping, operator, script, radical or stacked fraction."""

    kind: Literal[
        "symbol",
        "number",
        "group",
        "add",
        "subtract",
        "multiply",
        "slash",
        "equal",
        "negative",
        "fraction",
        "sqrt",
        "script",
    ]
    value: str | None = None
    children: tuple[FormulaNode, ...] = ()

    @model_validator(mode="after")
    def check_shape(self) -> FormulaNode:
        """Forbid inconsistent recursive syntax trees."""
        arity = {
            "symbol": 0,
            "number": 0,
            "group": 1,
            "negative": 1,
            "sqrt": 1,
            "add": 2,
            "subtract": 2,
            "multiply": 2,
            "slash": 2,
            "equal": 2,
            "fraction": 2,
            "script": 3,
        }[self.kind]
        if len(self.children) != arity:
            raise ValueError("Formula node has incorrect arity")
        if self.kind in {"symbol", "number"} and not self.value:
            raise ValueError("Leaf formula node needs a value")
        if self.kind not in {"symbol", "number"} and self.value is not None:
            raise ValueError("Operator node cannot contain a leaf value")
        return self


class FormulaSource(StrictModel):
    """Answer-independent two-dimensional formula recovered from one image view."""

    coverage: Literal["MET", "NOT_MET", "UNKNOWN"]
    scope_id: str
    view_id: str
    scope_region: ImageRegion
    formula_region: ImageRegion | None = None
    notation: Literal["latex", "unicode_math"]
    root: FormulaNode | None = None
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def check_source(self) -> FormulaSource:
        """Require complete and local evidence for an affirmative formula parse."""
        if (self.coverage == "MET") != (self.root is not None and self.formula_region is not None):
            raise ValueError("MET requires a rooted formula and its image region")
        if self.formula_region is not None:
            region = self.formula_region
            scope = self.scope_region
            if not (
                scope.left <= region.left < region.right <= scope.right
                and scope.top <= region.top < region.bottom <= scope.bottom
            ):
                raise ValueError("Formula lies outside requested image scope")
        return self


def _leaf(kind: Literal["symbol", "number"], value: str) -> FormulaNode:
    return FormulaNode(kind=kind, value=value)


def _node(kind: str, *children: FormulaNode) -> FormulaNode:
    return FormulaNode.model_validate({"kind": kind, "children": tuple(children)})


class _Parser:
    def __init__(self, text: str, notation: str) -> None:
        if notation == "unicode_math":
            text = text.translate(SUPERSCRIPTS).translate(SUBSCRIPTS)
        if notation == "latex":
            text = re.sub(r"\\(?:left|right)(?=[()])", "", text)
        self.tokens = TOKEN_RE.findall(text)
        if "".join(self.tokens).replace(" ", "") != re.sub(r"\s+", "", text):
            raise ValueError("Unsupported formula token")
        self.index = 0

    def peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise ValueError("Unexpected end of formula")
        self.index += 1
        return token

    def expect(self, expected: str) -> None:
        if self.take() != expected:
            raise ValueError("Formula delimiter mismatch")

    def grouped(self, opener: str, closer: str) -> FormulaNode:
        self.expect(opener)
        child = self.expression(0)
        self.expect(closer)
        return child if opener == "{" else _node("group", child)

    def atom(self) -> FormulaNode:
        token = self.peek()
        if token is None:
            raise ValueError("Missing formula atom")
        if token == "-":
            self.take()
            return _node("negative", self.expression(30))
        if token == "(":
            return self.grouped("(", ")")
        if token == "{":
            return self.grouped("{", "}")
        if token == "\\frac":
            self.take()
            return _node("fraction", self.grouped("{", "}"), self.grouped("{", "}"))
        if token in {"\\sqrt", "√"}:
            self.take()
            opener = "{" if self.peek() == "{" else "("
            return _node("sqrt", self.grouped(opener, "}" if opener == "{" else ")"))
        self.take()
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
            return _leaf("number", token)
        if token in SYMBOLS:
            return _leaf("symbol", SYMBOLS[token])
        if re.fullmatch(r"[A-Za-z]+|[αβγθπ∞∑]", token):
            return _leaf("symbol", token)
        raise ValueError("Unsupported formula atom")

    def expression(self, min_binding: int) -> FormulaNode:
        left = self.atom()
        subscript: FormulaNode | None = None
        superscript: FormulaNode | None = None
        while self.peek() in {"^", "_"}:
            script = self.take()
            value = self.grouped("{", "}") if self.peek() == "{" else self.atom()
            if script == "^":
                if superscript is not None:
                    raise ValueError("Repeated superscript")
                superscript = value
            else:
                if subscript is not None:
                    raise ValueError("Repeated subscript")
                subscript = value
        if subscript is not None or superscript is not None:
            left = _node(
                "script",
                left,
                subscript or _leaf("symbol", "∅"),
                superscript or _leaf("symbol", "∅"),
            )
        while True:
            token = self.peek()
            binding = {
                "=": 1,
                "+": 10,
                "-": 10,
                "*": 20,
                "·": 20,
                "×": 20,
                "\\cdot": 20,
                "\\times": 20,
                "/": 20,
            }.get(token or "")
            implicit = token is not None and (
                token in {"(", "{", "\\frac", "\\sqrt", "√"}
                or bool(
                    re.fullmatch(
                        r"[A-Za-z]+|[0-9]+(?:\.[0-9]+)?|[αβγθπ∞∑]|\\(?:alpha|beta|gamma|theta|pi|infty|sum)",
                        token,
                    )
                )
            )
            if binding is None and implicit:
                binding = 20
            if binding is None or binding < min_binding:
                break
            if not implicit:
                assert token is not None
                self.take()
            right = self.expression(binding + 1)
            kind = (
                {"=": "equal", "+": "add", "-": "subtract", "/": "slash"}.get(
                    token or "", "multiply"
                )
                if not implicit
                else "multiply"
            )
            left = _node(kind, left, right)
        return left


def parse_formula(text: str, notation: Literal["latex", "unicode_math"]) -> FormulaNode:
    """Parse only the registered literal grammar; never simplify equivalent forms."""
    text = text.strip()
    if notation == "latex":
        # Display delimiters describe the container, not the formula's syntax tree.
        for opening, closing in ((r"\[", r"\]"), (r"\(", r"\)"), ("$$", "$$"), ("$", "$")):
            if text.startswith(opening) and text.endswith(closing):
                text = text[len(opening) : -len(closing)]
                break
    parser = _Parser(text, notation)
    if not parser.tokens:
        raise ValueError("Empty formula")
    root = parser.expression(0)
    if parser.peek() is not None:
        raise ValueError("Trailing or unmatched formula token")
    return root


def verify_formula(
    sources: tuple[FormulaSource, FormulaSource],
    notation: str | None,
    scope_id: str,
    view_id: str,
    candidate_answer: str,
) -> GateVerdict:
    """Compare the actual submitted syntax tree with two blind visual structures."""
    if any(
        source.coverage != "MET"
        or source.scope_id != scope_id
        or source.view_id != view_id
        or source.notation != notation
        for source in sources
    ):
        return GateVerdict.UNKNOWN
    if sources[0].root != sources[1].root:
        return GateVerdict.UNKNOWN
    try:
        if notation not in {"latex", "unicode_math"}:
            return GateVerdict.UNKNOWN
        parsed = parse_formula(candidate_answer, notation)
    except ValueError:
        return GateVerdict.UNKNOWN
    return GateVerdict.MET if parsed == sources[0].root else GateVerdict.NOT_MET
