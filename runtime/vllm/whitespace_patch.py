"""Apply or verify the bounded whitespace bridge in the pinned server environment.

Run with runtime/vllm's Python after uv sync and before starting model servers.
The patch accepts a schema extension only; ordinary requests keep upstream behavior.
"""

import argparse
import hashlib
import json
import sysconfig
from importlib.metadata import version
from pathlib import Path

UPSTREAM_SHA256 = "d5452b55e7dba4bb06e47d22ee9ac7cb2c478f65a303cc17b826601b812e0c22"
SCHEMA_KEY = "x-pixelogue-max-whitespace-chars"
ORIGINAL = """            ctx = self.compiler.compile_json_schema(
                grammar_spec, any_whitespace=not self.disable_any_whitespace
            )"""
REPLACEMENT = """            # Pixelogue: retain ordinary JSON formatting while bounding whitespace.
            whitespace_bound = json.loads(grammar_spec).get(
                "x-pixelogue-max-whitespace-chars"
            )
            if whitespace_bound is not None and (
                type(whitespace_bound) is not int or not 1 <= whitespace_bound <= 256
            ):
                raise ValueError("Invalid Pixelogue JSON whitespace bound")
            ctx = self.compiler.compile_json_schema(
                grammar_spec,
                any_whitespace=not self.disable_any_whitespace,
                max_whitespace_cnt=whitespace_bound,
            )"""


def sha(content: bytes) -> str:
    """Return a digest for the complete runtime source file."""
    return hashlib.sha256(content).hexdigest()


def patch_content(original: bytes) -> bytes:
    """Reject unsupported upstream files before producing the versioned patch."""
    if sha(original) != UPSTREAM_SHA256:
        raise ValueError("Unsupported XGrammar backend source; refusing to patch")
    text = original.decode()
    if text.count(ORIGINAL) != 1:
        raise ValueError("Expected exactly one upstream schema compilation site")
    return text.replace(ORIGINAL, REPLACEMENT).encode()


def install(target: Path, *, check_only: bool = False, restore: bool = False) -> dict:
    """Verify exact file identities and preserve the upstream bytes for restoration."""
    content = target.read_bytes()
    backup = target.with_suffix(".pixelogue-upstream")
    if sha(content) == UPSTREAM_SHA256:
        original = content
    elif backup.exists() and sha(backup.read_bytes()) == UPSTREAM_SHA256:
        original = backup.read_bytes()
    else:
        raise ValueError("Neither upstream nor a verified patch backup was found")
    patched = patch_content(original)
    if content not in (original, patched):
        raise ValueError("Runtime source differs from the supported patch; refusing to overwrite")
    desired = original if restore else patched
    if check_only and content != desired:
        raise ValueError("Required runtime source is not installed")
    if not check_only and content != desired:
        if not backup.exists():
            backup.write_bytes(original)
        elif backup.read_bytes() != original:
            raise ValueError("Existing backup differs; refusing to overwrite")
        temporary = target.with_suffix(".pixelogue-tmp")
        temporary.write_bytes(desired)
        temporary.replace(target)
    return {
        "patch": "pixelogue-xgrammar-whitespace-v1",
        "schema_key": SCHEMA_KEY,
        "upstream_sha256": sha(original),
        "patched_sha256": sha(patched),
        "installed_sha256": sha(desired),
        "restored": restore,
    }


def main() -> None:
    """Operate only on the pinned vLLM and XGrammar versions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if version("vllm") != "0.29.0" or version("xgrammar") != "0.2.6":
        raise ValueError("This patch requires vLLM 0.29.0 and XGrammar 0.2.6")
    target = Path(sysconfig.get_path("purelib")) / "vllm/v1/structured_output/backend_xgrammar.py"
    print(json.dumps(install(target, check_only=args.check, restore=args.restore)))


if __name__ == "__main__":
    main()
