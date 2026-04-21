"""Tests for the `bh-sentinel-ml evaluate` CLI subcommand."""

from __future__ import annotations

from pathlib import Path

from bh_sentinel.ml.cli.__main__ import main


def _write_fixtures(path: Path) -> None:
    """Write a minimal YAML fixture file compatible with the evaluate CLI."""
    path.write_text(
        "fixtures:\n"
        "  - id: ex1\n"
        "    text: I want to end my life.\n"
        "    expect_flags: [SH-002]\n"
        "  - id: ex2\n"
        "    text: I have a lot of hope for tomorrow.\n"
        "    expect_flags: []\n"
    )


def _run_cli(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as e:
        return int(e.code) if e.code is not None else 0


def test_evaluate_runs_against_fixtures(tmp_path: Path, capsys) -> None:
    fixtures = tmp_path / "fixtures.yaml"
    _write_fixtures(fixtures)
    # Default mode: L1-only evaluation (no --enable-transformer).
    rc = _run_cli(["evaluate", "--fixtures", str(fixtures)])
    assert rc in (0, 1)  # 0 if all pass, 1 if some fail -- both are OK
    captured = capsys.readouterr()
    out = captured.out + captured.err
    # Human-readable summary markers.
    assert "ex1" in out
    assert "ex2" in out


def test_evaluate_missing_fixtures_file_errors(tmp_path: Path) -> None:
    rc = _run_cli(["evaluate", "--fixtures", str(tmp_path / "nope.yaml")])
    assert rc != 0
