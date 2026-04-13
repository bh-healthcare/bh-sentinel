"""Flag taxonomy definitions and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class FlagTaxonomy:
    """Versioned clinical safety flag taxonomy (40 flags across 6 domains).

    Loads from config/flag_taxonomy.json and provides lookup by flag ID,
    domain, and severity level.
    """

    def __init__(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)

        self._version: str = data["taxonomy_version"]
        self._flag_by_id: dict[str, dict[str, Any]] = {}
        self._domain_for_flag: dict[str, str] = {}
        self._flags_by_domain: dict[str, list[dict[str, Any]]] = {}
        self._flags_by_severity: dict[str, list[dict[str, Any]]] = {}
        self._domains: list[str] = []

        for domain in data["domains"]:
            domain_id = domain["id"]
            self._domains.append(domain_id)
            self._flags_by_domain[domain_id] = []

            for flag in domain["flags"]:
                flag_id = flag["flag_id"]
                self._flag_by_id[flag_id] = flag
                self._domain_for_flag[flag_id] = domain_id
                self._flags_by_domain[domain_id].append(flag)

                severity = flag["default_severity"]
                if severity not in self._flags_by_severity:
                    self._flags_by_severity[severity] = []
                self._flags_by_severity[severity].append(flag)

    @property
    def version(self) -> str:
        return self._version

    def get_flag(self, flag_id: str) -> dict[str, Any] | None:
        return self._flag_by_id.get(flag_id)

    def get_domain_for_flag(self, flag_id: str) -> str | None:
        return self._domain_for_flag.get(flag_id)

    def get_flags_by_domain(self, domain_id: str) -> list[dict[str, Any]]:
        return self._flags_by_domain.get(domain_id, [])

    def get_flags_by_severity(self, severity: str) -> list[dict[str, Any]]:
        return self._flags_by_severity.get(severity, [])

    def all_flag_ids(self) -> list[str]:
        return list(self._flag_by_id.keys())

    def all_domains(self) -> list[str]:
        return list(self._domains)

    def satisfies_version(self, requirement: str) -> bool:
        """Check if taxonomy version satisfies a requirement like '1.0.x'."""
        req_parts = requirement.split(".")
        ver_parts = self._version.split(".")
        for req, ver in zip(req_parts, ver_parts, strict=False):
            if req == "x":
                continue
            if req != ver:
                return False
        return True
