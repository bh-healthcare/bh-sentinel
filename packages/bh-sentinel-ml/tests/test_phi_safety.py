"""PHI-safety umbrella check across bh-sentinel-ml.

No module in the ml package may:
- log input text at any level
- include raw input text in exception messages
- include raw input text in candidate basis_description fields
  (matched_context_hint is allowed -- it's a bounded, short excerpt
  that clinicians need to locate the signal)

These tests are defensive. They do not prove absence, but they catch
the most common patterns that would silently introduce a PHI leak.
"""

from __future__ import annotations

import re
from pathlib import Path

ML_SRC = Path(__file__).resolve().parents[1] / "src" / "bh_sentinel" / "ml"


def _ml_source_files() -> list[Path]:
    return sorted(p for p in ML_SRC.rglob("*.py") if "_default_config" not in str(p))


def test_no_logger_debug_or_info_on_input_text() -> None:
    """The package ships no logger.info / logger.debug calls that take
    a variable named like input text. Catches the easy-to-write bug
    where a developer adds `logger.info(text)` for debugging.
    """
    danger = re.compile(r"log(ger)?\.(info|debug|warning|error|exception)\s*\([^)]*\btext\b")
    for src in _ml_source_files():
        content = src.read_text()
        matches = danger.findall(content)
        assert not matches, (
            f"{src} appears to log a variable named 'text'. "
            "Review for PHI exposure before committing."
        )


def test_no_print_statements_in_library_code() -> None:
    """Library code must not contain bare print() calls. The CLI module
    is exempt -- it's intentionally console-facing."""
    for src in _ml_source_files():
        if "/cli/" in str(src):
            continue
        content = src.read_text()
        # Allow print() inside test-only stubs; at the library level it
        # shouldn't appear.
        assert not re.search(r"^\s*print\(", content, re.MULTILINE), (
            f"{src} contains a bare print() call"
        )


def test_error_messages_do_not_interpolate_input_text_patterns() -> None:
    """Scan for f-string errors that interpolate names commonly used
    for input text. Not an exhaustive check but catches the easy
    mistakes."""
    # Looks for f"{text}..." or f"... {input_text}..."
    patterns = [
        re.compile(r'raise \w+\([^)]*f"[^"]*\{text[^}]*\}[^"]*"'),
        re.compile(r'raise \w+\([^)]*f"[^"]*\{input_text[^}]*\}[^"]*"'),
        re.compile(r'raise \w+\([^)]*f"[^"]*\{premise[^}]*\}[^"]*"'),
        re.compile(r'raise \w+\([^)]*f"[^"]*\{hypothesis[^}]*\}[^"]*"'),
        re.compile(r'raise \w+\([^)]*f"[^"]*\{sentence\.text[^}]*\}[^"]*"'),
    ]
    for src in _ml_source_files():
        content = src.read_text()
        for pattern in patterns:
            assert not pattern.search(content), (
                f"{src} may interpolate input text into an exception (pattern: {pattern.pattern!r})"
            )
