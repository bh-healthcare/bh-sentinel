"""Tests for FlagTaxonomy -- written before implementation (TDD)."""

from __future__ import annotations

import pytest

from bh_sentinel.core._config import default_flag_taxonomy_path
from bh_sentinel.core.taxonomy import FlagTaxonomy


@pytest.fixture
def taxonomy() -> FlagTaxonomy:
    return FlagTaxonomy(default_flag_taxonomy_path())


class TestLoading:
    def test_loads_default(self, taxonomy):
        assert taxonomy is not None

    def test_version_is_1_0_0(self, taxonomy):
        assert taxonomy.version == "1.0.0"

    def test_flag_count_is_40(self, taxonomy):
        assert len(taxonomy.all_flag_ids()) == 40

    def test_domain_count_is_6(self, taxonomy):
        assert len(taxonomy.all_domains()) == 6


class TestLookup:
    def test_get_flag_sh001(self, taxonomy):
        flag = taxonomy.get_flag("SH-001")
        assert flag is not None
        assert flag["flag_id"] == "SH-001"
        assert flag["name"] == "Passive death wish"

    def test_get_flag_returns_none_for_unknown(self, taxonomy):
        assert taxonomy.get_flag("UNKNOWN-999") is None

    def test_get_domain_for_flag(self, taxonomy):
        assert taxonomy.get_domain_for_flag("SH-001") == "self_harm"
        assert taxonomy.get_domain_for_flag("HO-001") == "harm_to_others"
        assert taxonomy.get_domain_for_flag("PF-001") == "protective_factors"

    def test_get_flags_by_domain_self_harm_has_8(self, taxonomy):
        flags = taxonomy.get_flags_by_domain("self_harm")
        assert len(flags) == 8

    def test_get_flags_by_severity_critical(self, taxonomy):
        flags = taxonomy.get_flags_by_severity("CRITICAL")
        assert len(flags) > 0
        for f in flags:
            assert f["default_severity"] == "CRITICAL"


class TestVersionCheck:
    def test_satisfies_1_0_x(self, taxonomy):
        assert taxonomy.satisfies_version("1.0.x") is True

    def test_rejects_2_0_x(self, taxonomy):
        assert taxonomy.satisfies_version("2.0.x") is False

    def test_rejects_1_1_x(self, taxonomy):
        assert taxonomy.satisfies_version("1.1.x") is False


class TestIntegrity:
    def test_all_flag_ids_unique(self, taxonomy):
        ids = taxonomy.all_flag_ids()
        assert len(ids) == len(set(ids))

    def test_all_domains_have_flags(self, taxonomy):
        for domain in taxonomy.all_domains():
            assert len(taxonomy.get_flags_by_domain(domain)) > 0

    def test_every_flag_has_required_fields(self, taxonomy):
        required = {"flag_id", "name", "description", "default_severity"}
        for flag_id in taxonomy.all_flag_ids():
            flag = taxonomy.get_flag(flag_id)
            assert flag is not None
            for field in required:
                assert field in flag, f"{flag_id} missing {field}"
