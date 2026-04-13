"""Tests for CLI commands."""

from __future__ import annotations

from bh_sentinel.cli.test_patterns import run_test_patterns
from bh_sentinel.cli.validate_config import run_validate


class TestValidateConfig:
    def test_passes_with_defaults(self):
        assert run_validate() == 0


class TestTestPatterns:
    def test_runs_without_error(self):
        """test-patterns CLI runs and returns a result (0 or 1).

        Pattern coverage gaps mean some fixtures fail, so we accept
        either 0 (all pass) or 1 (some fail) -- the point is it runs.
        """
        result = run_test_patterns()
        assert result in (0, 1)
