"""Tests for the bh-sentinel-core version guard in bh_sentinel.ml.__init__.

The authoritative enforcement of ``bh-sentinel-core>=0.1.1`` is pip at install
time (see ``pyproject.toml``). These tests cover the belt-and-suspenders
runtime check that catches ``--no-deps`` / vendored / editable-monorepo cases.
"""

from __future__ import annotations

import pytest

from bh_sentinel.ml import (
    _MIN_CORE_VERSION,
    _check_core_compatibility,
    _parse_release,
)


class TestParseRelease:
    """Parser must handle normal PEP 440 releases and bail out on exotic ones."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0.1.1", (0, 1, 1)),
            ("0.1.0", (0, 1, 0)),
            ("1.2.3", (1, 2, 3)),
            ("10.20.30", (10, 20, 30)),
            ("0.1.1.dev0", (0, 1, 1)),
            ("0.1.1.post1", (0, 1, 1)),
            ("0.1.1a0", (0, 1, 1)),
            ("0.1.1b2", (0, 1, 1)),
            ("0.1.1rc1", (0, 1, 1)),
            ("0.1.1+local.build", (0, 1, 1)),
            ("0.1.1-dev", (0, 1, 1)),
        ],
    )
    def test_parses_release_segment(self, raw: str, expected: tuple[int, int, int]) -> None:
        assert _parse_release(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "0.1",
            "not-a-version",
            "0.1.x",
        ],
    )
    def test_returns_none_for_unparseable(self, raw: str) -> None:
        assert _parse_release(raw) is None

    def test_ignores_trailing_segments_beyond_major_minor_patch(self) -> None:
        """PEP 440 allows epochs/4-segment versions; we only compare leading X.Y.Z."""
        assert _parse_release("0.1.1.4") == (0, 1, 1)
        assert _parse_release("1.2.3.4.5") == (1, 2, 3)


class TestCheckCoreCompatibility:
    """The runtime guard must raise only on an older-than-min installed core."""

    def test_passes_with_installed_core(self) -> None:
        _check_core_compatibility()

    def test_raises_when_core_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from importlib.metadata import PackageNotFoundError

        import bh_sentinel.ml as ml

        def _fake_version(name: str) -> str:
            raise PackageNotFoundError(name)

        monkeypatch.setattr(ml, "_pkg_version", _fake_version)

        with pytest.raises(ImportError, match="bh-sentinel-core is not installed"):
            ml._check_core_compatibility()

    def test_raises_when_core_too_old(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import bh_sentinel.ml as ml

        monkeypatch.setattr(ml, "_pkg_version", lambda _name: "0.1.0")

        with pytest.raises(ImportError) as exc_info:
            ml._check_core_compatibility()

        message = str(exc_info.value)
        assert "requires bh-sentinel-core>=0.1.1" in message
        assert "bh-sentinel-core==0.1.0" in message
        assert "pip install -U bh-sentinel-core" in message

    def test_accepts_exact_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import bh_sentinel.ml as ml

        monkeypatch.setattr(ml, "_pkg_version", lambda _name: "0.1.1")

        _check_core_compatibility()

    def test_accepts_newer_core(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import bh_sentinel.ml as ml

        monkeypatch.setattr(ml, "_pkg_version", lambda _name: "0.2.5")

        _check_core_compatibility()

    def test_unparseable_version_does_not_block_import(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exotic version strings fall back to pip's install-time check, never block."""
        import bh_sentinel.ml as ml

        monkeypatch.setattr(ml, "_pkg_version", lambda _name: "not-a-real-version")

        _check_core_compatibility()


class TestMinCoreVersionConstant:
    """Keep the constant synchronized with pyproject.toml and the README table."""

    def test_matches_documented_minimum(self) -> None:
        assert _MIN_CORE_VERSION == (0, 1, 1)
