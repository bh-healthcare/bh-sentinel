"""Assertions about the release workflow YAML files.

These tests parse the workflow YAML at the repo level and verify the
per-package release discipline laid out in docs/release-process.md:

- bh-sentinel-core is published by a workflow triggered only on core-v* tags
- bh-sentinel-ml is published by a workflow triggered only on ml-v* tags
- The legacy ambiguous "v*" trigger is gone
- Each workflow verifies the tag version matches the pyproject.toml version
  before calling `python -m build`
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import yaml
except ImportError:
    pytest.skip("pyyaml not installed", allow_module_level=True)


REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
PUBLISH_CORE = WORKFLOWS_DIR / "publish-core.yml"
PUBLISH_ML = WORKFLOWS_DIR / "publish-ml.yml"
LEGACY_PUBLISH = WORKFLOWS_DIR / "publish.yml"


def _load(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def test_legacy_publish_yml_is_removed() -> None:
    """The ambiguous `v*` trigger must not exist anymore."""
    assert not LEGACY_PUBLISH.exists(), (
        "publish.yml must be removed -- it triggered on v* which is ambiguous "
        "across packages. Use publish-core.yml and publish-ml.yml instead."
    )


def test_publish_core_workflow_exists() -> None:
    assert PUBLISH_CORE.exists(), "publish-core.yml must exist at .github/workflows/"


def test_publish_ml_workflow_exists() -> None:
    assert PUBLISH_ML.exists(), "publish-ml.yml must exist at .github/workflows/"


def test_publish_core_triggers_on_core_v_tags_only() -> None:
    data = _load(PUBLISH_CORE)
    on = data.get(True) or data.get("on")
    tags = on["push"]["tags"]
    assert tags == ["core-v*"], f"publish-core.yml must trigger ONLY on core-v* tags, got {tags}"


def test_publish_ml_triggers_on_ml_v_tags_only() -> None:
    data = _load(PUBLISH_ML)
    on = data.get(True) or data.get("on")
    tags = on["push"]["tags"]
    assert tags == ["ml-v*"], f"publish-ml.yml must trigger ONLY on ml-v* tags, got {tags}"


def test_publish_core_builds_core_package() -> None:
    """The build step must target packages/bh-sentinel-core/."""
    data = _load(PUBLISH_CORE)
    steps_text = yaml.safe_dump(data)
    assert "packages/bh-sentinel-core/" in steps_text
    assert "packages/bh-sentinel-ml/" not in steps_text, (
        "publish-core.yml must NOT build the ml package"
    )


def test_publish_ml_builds_ml_package() -> None:
    data = _load(PUBLISH_ML)
    steps_text = yaml.safe_dump(data)
    assert "packages/bh-sentinel-ml/" in steps_text
    assert "packages/bh-sentinel-core/" not in steps_text, (
        "publish-ml.yml must NOT build the core package"
    )


def test_publish_core_has_verify_version_step() -> None:
    """Must verify the git tag matches the pyproject.toml version before build."""
    data = _load(PUBLISH_CORE)
    steps_text = yaml.safe_dump(data)
    assert "core-v" in steps_text, "Must strip core-v prefix from tag"
    assert "pyproject.toml" in steps_text
    assert "version" in steps_text.lower()


def test_publish_ml_has_verify_version_step() -> None:
    data = _load(PUBLISH_ML)
    steps_text = yaml.safe_dump(data)
    assert "ml-v" in steps_text, "Must strip ml-v prefix from tag"
    assert "pyproject.toml" in steps_text
    assert "version" in steps_text.lower()


def _uses_trusted_publishing(workflow_path: Path) -> bool:
    """A workflow uses Trusted Publishing if any job declares both
    `environment: pypi` and `id-token: write` permissions."""
    data = _load(workflow_path)
    for job in data.get("jobs", {}).values():
        env = job.get("environment")
        env_name = env if isinstance(env, str) else (env or {}).get("name")
        perms = job.get("permissions") or {}
        if env_name == "pypi" and perms.get("id-token") == "write":
            return True
    return False


def test_publish_core_uses_trusted_publishing() -> None:
    """Environment: pypi + id-token: write indicates Trusted Publishing."""
    assert _uses_trusted_publishing(PUBLISH_CORE)


def test_publish_ml_uses_trusted_publishing() -> None:
    assert _uses_trusted_publishing(PUBLISH_ML)
