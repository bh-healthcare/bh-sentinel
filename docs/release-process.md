# Release Process

This repository publishes two independently-versioned PyPI packages from the same monorepo: `bh-sentinel-core` and `bh-sentinel-ml`. Each package has its own release workflow, its own git tag prefix, and its own PyPI Trusted Publisher.

## Tag Scheme

| Package | Tag prefix | Example | Workflow |
|---|---|---|---|
| `bh-sentinel-core` | `core-v` | `core-v0.1.1` | [`.github/workflows/publish-core.yml`](../.github/workflows/publish-core.yml) |
| `bh-sentinel-ml` | `ml-v` | `ml-v0.2.0` | [`.github/workflows/publish-ml.yml`](../.github/workflows/publish-ml.yml) |

Bare `v*` tags (for example the historical `v0.1.0`) are **deprecated**. They exist for audit history but must not be used for any new release.

## One-Time PyPI Setup

Both PyPI Trusted Publishers must be configured before any `core-v*` or `ml-v*` tag is pushed. Without them, the publish workflow will run, the build will succeed, and the upload step will fail with `Trusted publisher mismatch`.

### Update the existing `bh-sentinel-core` publisher

When the workflow file was renamed from `publish.yml` to `publish-core.yml`, the OIDC identity tuple (`{owner, repo, workflow_filename, environment}`) changed. The existing publisher must be updated to match.

PyPI does not have an "edit" button on Trusted Publishers -- only Remove and Re-add. Do both in one browser session; the gap is harmless for already-published artifacts (they are immutable, and `pip install` continues to work), but it does block new releases for the duration.

1. Go to pypi.org -> Your projects -> `bh-sentinel-core` -> Manage -> Publishing
2. Remove the existing entry that references `Workflow: publish.yml`
3. Immediately add a new entry with:
   - Owner: `bh-healthcare`
   - Repository name: `bh-sentinel`
   - Workflow name: `publish-core.yml`
   - Environment name: `pypi`

### Pre-register the `bh-sentinel-ml` pending publisher

The PyPI project `bh-sentinel-ml` does not exist yet. PyPI's "pending publisher" flow lets you pre-register the identity tuple; the first successful publish creates the project and promotes the pending publisher to a real one.

1. Go to pypi.org -> Your account -> Publishing -> "Add a new pending publisher"
2. Fill in:
   - PyPI Project Name: `bh-sentinel-ml`
   - Owner: `bh-healthcare`
   - Repository name: `bh-sentinel`
   - Workflow name: `publish-ml.yml`
   - Environment name: `pypi`

Pending publishers expire after 180 days if unused.

## Pre-Tag Checklist (per package)

Run this checklist before pushing any release tag. CI enforces most of it, but the checklist is the authoritative spec.

- [ ] `pyproject.toml` version bumped to the target version
- [ ] `src/<package>/__init__.py` `__version__` matches
- [ ] `CHANGELOG.md` has a new dated section for this version with a comparison link at the bottom
- [ ] `tests/test_public_api.py` version assertion matches
- [ ] Any README version references match
- [ ] `ruff check` and `ruff format --check` clean across both packages
- [ ] Full core test suite green (`pytest packages/bh-sentinel-core/tests/`)
- [ ] Full ml test suite green (`pytest packages/bh-sentinel-ml/tests/`)
- [ ] Banned-string grep returns zero hits: `rg -i 'overstory|overstory health|overstory-bh|overstory-datalake|org_overstory'`
- [ ] `python -m build packages/<package>/` succeeds locally
- [ ] Clean-venv smoke install of the locally-built wheel works end-to-end

## Release Commands

Releases are triggered solely by pushing an annotated tag. Do all work on `main` first and ensure the above checklist passes before tagging.

### Release order matters for dependent packages

`bh-sentinel-ml` declares `bh-sentinel-core>=0.1.1` (or higher, matching whatever version introduced the L2 pipeline hooks) in its `pyproject.toml`. Always publish core first, wait for PyPI to index it, then publish ml. If you tag ml first, `pip install bh-sentinel-ml` can briefly resolve an older core and fail at import.

### Release `bh-sentinel-core`

```bash
git checkout main
git pull

# Replace VERSION with the actual version, e.g. 0.1.1
git tag -a core-vVERSION -m "bh-sentinel-core VERSION: <one-line summary>"
git push origin core-vVERSION
```

Then watch the Actions tab for the `Publish bh-sentinel-core to PyPI` run. On success, confirm the new version at <https://pypi.org/project/bh-sentinel-core/> and wait 1-2 minutes for the CDN to propagate.

### Release `bh-sentinel-ml`

```bash
git checkout main
git pull

# Replace VERSION with the actual version, e.g. 0.2.0
git tag -a ml-vVERSION -m "bh-sentinel-ml VERSION: <one-line summary>"
git push origin ml-vVERSION
```

Watch the Actions tab for the `Publish bh-sentinel-ml to PyPI` run. Confirm at <https://pypi.org/project/bh-sentinel-ml/>.

## CI Version Verification

Both workflows refuse to build if the pushed tag and the in-repo `pyproject.toml` disagree. For example, pushing `core-v0.1.2` while `packages/bh-sentinel-core/pyproject.toml` still says `version = "0.1.1"` fails the verify step with:

```
::error::Tag core-v0.1.2 implies version 0.1.2 but pyproject.toml has 0.1.1
```

Nothing is uploaded. Fix the version mismatch, delete the tag (`git tag -d core-v0.1.2 && git push origin :refs/tags/core-v0.1.2`), and re-tag after bumping.

## Rollback and Failure Recovery

- **PyPI never allows re-publishing the same version.** If a release ships broken, yank the version on PyPI (which hides it from `pip install` by default but preserves the artifact for anyone who pinned it) and release a patch (e.g., `core-v0.1.2`).
- **Never delete a published release's tag.** It breaks release notes, the PyPI "Project links" section, and the audit trail. Tag deletion is only appropriate when the release workflow failed and nothing was actually published.
- **Partial-failure recovery** (build + PyPI succeeded, GitHub Release step failed): re-run the `release` job only; artifact upload is idempotent.
- **Full-failure recovery** (build or PyPI publish failed): fix the underlying issue, delete the tag locally and remotely (`git tag -d <tag> && git push origin :refs/tags/<tag>`), and re-tag after the fix is on `main`.

## Historical Tags

| Tag | Date | Package | Notes |
|---|---|---|---|
| `v0.1.0` | 2026-04-12 | `bh-sentinel-core` | Created under the legacy `v*` scheme, before the per-package split. Equivalent to `core-v0.1.0` in spirit. |

## Related Documents

- [`CHANGELOG.md`](../CHANGELOG.md) -- Keep-a-Changelog formatted release notes
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) -- contribution workflow and standards
- [`docs/architecture.md`](architecture.md) -- package boundaries and layer design
