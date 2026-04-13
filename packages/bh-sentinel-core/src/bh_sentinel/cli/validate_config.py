"""7-point config validation for bh-sentinel."""

from __future__ import annotations

import json
import re

import yaml

from bh_sentinel.core._config import (
    default_emotion_lexicon_path,
    default_flag_taxonomy_path,
    default_patterns_path,
    default_rules_path,
)


def run_validate() -> int:
    """Run all 7 validation checks. Returns 0 on success, 1 on failure."""
    errors: list[str] = []

    tax_path = default_flag_taxonomy_path()
    pat_path = default_patterns_path()
    rul_path = default_rules_path()
    lex_path = default_emotion_lexicon_path()

    with open(tax_path) as f:
        taxonomy = json.load(f)
    with open(pat_path) as f:
        patterns = yaml.safe_load(f)
    with open(rul_path) as f:
        rules = json.load(f)
    with open(lex_path) as f:
        lexicon = json.load(f)

    # Build taxonomy flag set.
    tax_flags: set[str] = set()
    for domain in taxonomy["domains"]:
        for flag in domain["flags"]:
            tax_flags.add(flag["flag_id"])

    # Build pattern flag set.
    pat_flags: set[str] = set()
    for domain_id, flags in patterns.items():
        if domain_id.startswith("_") or not isinstance(flags, dict):
            continue
        for flag_id in flags:
            pat_flags.add(flag_id)

    # 1. Taxonomy-pattern coverage.
    missing = tax_flags - pat_flags
    if missing:
        errors.append(f"1. Flags in taxonomy without patterns: {sorted(missing)}")

    # 2. Rule flag references.
    rule_flags = _extract_rule_flags(rules)
    unknown = rule_flags - tax_flags
    if unknown:
        errors.append(f"2. Rule references to unknown flags: {sorted(unknown)}")

    # 3. Version compatibility.
    tax_version = taxonomy["taxonomy_version"]
    pat_req = patterns.get("_meta", {}).get("requires_taxonomy_version", "")
    if pat_req and not _version_satisfies(tax_version, pat_req):
        errors.append(f"3. Patterns require {pat_req} but taxonomy is {tax_version}")

    # 4. Negation phrase validity.
    for domain_id, flags in patterns.items():
        if domain_id.startswith("_") or not isinstance(flags, dict):
            continue
        for flag_id, flag_data in flags.items():
            if not isinstance(flag_data, dict):
                continue
            for phrase in flag_data.get("negation_phrases", []):
                try:
                    re.compile(phrase)
                except re.error as e:
                    errors.append(f"4. Invalid negation regex in {flag_id}: {phrase} ({e})")

    # 5. Confidence range.
    for domain_id, flags in patterns.items():
        if domain_id.startswith("_") or not isinstance(flags, dict):
            continue
        for flag_id, flag_data in flags.items():
            if not isinstance(flag_data, dict):
                continue
            conf = flag_data.get("confidence", 0.85)
            if not (0.0 <= conf <= 1.0):
                errors.append(f"5. Confidence out of range in {flag_id}: {conf}")

    # 6. Emotion category coverage.
    lex_cats = set(lexicon.get("categories", []))
    for cat in _extract_emotion_categories(rules):
        if cat not in lex_cats:
            errors.append(f"6. Rule references unknown emotion category: {cat}")

    # 7. Duplicate flag IDs.
    all_ids: list[str] = []
    for domain in taxonomy["domains"]:
        for flag in domain["flags"]:
            all_ids.append(flag["flag_id"])
    dupes = [fid for fid in all_ids if all_ids.count(fid) > 1]
    if dupes:
        errors.append(f"7. Duplicate flag IDs: {sorted(set(dupes))}")

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"All 7 checks passed. {len(tax_flags)} flags, {len(pat_flags)} pattern groups.")
    return 0


def _extract_rule_flags(rules: dict) -> set[str]:
    """Extract all flag_id references from rules."""
    flags: set[str] = set()
    for category in ["escalation_rules", "de_escalation_rules", "compound_rules", "action_rules"]:
        for rule in rules.get(category, []):
            _scan_condition_for_flags(rule.get("condition", {}), flags)
            action = rule.get("action", {})
            if "escalate_flag" in action:
                flags.add(action["escalate_flag"])
            for fid in action.get("escalate_flags", []):
                flags.add(fid)
    return flags


def _scan_condition_for_flags(condition: dict, flags: set[str]) -> None:
    """Recursively scan a condition for flag references."""
    if "flag_present" in condition:
        flags.add(condition["flag_present"])
    if "any_flag_present" in condition:
        flags.update(condition["any_flag_present"])
    for child in condition.get("all_of", []):
        _scan_condition_for_flags(child, flags)
    for child in condition.get("any_of", []):
        _scan_condition_for_flags(child, flags)


def _extract_emotion_categories(rules: dict) -> set[str]:
    """Extract all emotion category references from rules."""
    cats: set[str] = set()
    for category in ["escalation_rules", "compound_rules"]:
        for rule in rules.get(category, []):
            _scan_condition_for_emotions(rule.get("condition", {}), cats)
    return cats


def _scan_condition_for_emotions(condition: dict, cats: set[str]) -> None:
    if "emotion_above" in condition:
        cats.add(condition["emotion_above"]["category"])
    for child in condition.get("all_of", []):
        _scan_condition_for_emotions(child, cats)
    for child in condition.get("any_of", []):
        _scan_condition_for_emotions(child, cats)


def _version_satisfies(version: str, requirement: str) -> bool:
    """Check if version satisfies requirement like '1.0.x'."""
    req_parts = requirement.split(".")
    ver_parts = version.split(".")
    for req, ver in zip(req_parts, ver_parts, strict=False):
        if req == "x":
            continue
        if req != ver:
            return False
    return True
