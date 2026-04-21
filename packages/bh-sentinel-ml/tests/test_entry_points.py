"""Packaging regression: the bh-sentinel-ml console entry point must
be declared in pyproject.toml so `pip install bh-sentinel-ml` creates
the `bh-sentinel-ml` command on $PATH.

Caught once in v0.2.0-rc when the [project.scripts] table was missing;
this test keeps it from regressing silently.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _load_pyproject() -> dict:
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_pyproject_declares_console_script() -> None:
    data = _load_pyproject()
    scripts = data.get("project", {}).get("scripts", {})
    assert "bh-sentinel-ml" in scripts, (
        "pyproject.toml [project.scripts] must declare a "
        "`bh-sentinel-ml` entry point so pip install creates the CLI "
        "command on PATH."
    )


def test_console_script_targets_cli_main() -> None:
    """The entry point must resolve to bh_sentinel.ml.cli.__main__:main
    -- the module and function that actually exist and have tests."""
    data = _load_pyproject()
    target = data["project"]["scripts"]["bh-sentinel-ml"]
    assert target == "bh_sentinel.ml.cli.__main__:main", (
        f"bh-sentinel-ml entry point must be 'bh_sentinel.ml.cli.__main__:main', got {target!r}"
    )


def test_cli_main_is_importable() -> None:
    """The module:function pair referenced in the entry point must actually
    resolve. Catches typos that the static pyproject check would miss."""
    from bh_sentinel.ml.cli.__main__ import main

    assert callable(main)
