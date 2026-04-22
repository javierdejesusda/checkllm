# Release Process

checkllm releases are fully automated. Pushing a tag of the form
`vMAJOR.MINOR.PATCH` runs the entire release pipeline: verification,
full test matrix, build, Sigstore signing, SLSA level-3 provenance,
PyPI trusted-publisher upload with attestations, GitHub Release
creation, and a follow-up PR that bumps `main` to the next
development version.

## Prerequisites (one-time)

- Configure a PyPI **trusted publisher** pointing at this repository and
  the `release.yml` workflow, environment `pypi`.
- Protect the `pypi` environment so only maintainers can approve deploys.
- Ensure the GitHub repository allows Actions to create pull requests
  (Settings -> Actions -> General -> Workflow permissions).

## Cutting a release

1. **Bump the version**

   ```bash
   python scripts/bump_version.py patch        # or minor / major
   # or pin an exact value:
   python scripts/bump_version.py --set 5.1.0
   ```

   This updates `pyproject.toml` and `src/checkllm/__init__.py` in lock
   step. Verify with `python scripts/sync_version.py`.

2. **Update the changelog**

   ```bash
   python scripts/generate_changelog.py --dry-run   # preview
   python scripts/generate_changelog.py             # write
   ```

   The generator groups commits by conventional-commit prefix. Edit
   the resulting section if you need to reword entries, then commit.

3. **Open a release PR**

   Title it `release: vX.Y.Z`. Merge when CI is green and the diff is
   limited to version + CHANGELOG updates.

4. **Tag and push**

   ```bash
   git checkout main
   git pull
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. **Watch the workflow**

   The `Release` workflow will:

   - Verify the tag matches `pyproject.toml`.
   - Run the test matrix across Python 3.10-3.13 on Linux, macOS, and Windows.
   - Build the sdist and wheel with Hatchling.
   - Sign the artifacts with Sigstore and generate SLSA provenance.
   - Upload to PyPI using the trusted publisher (no API token needed).
   - Create a GitHub Release with auto-generated notes and attach the
     signed distributions plus the provenance bundle.
   - Open a follow-up PR bumping the in-repo version to
     `X.Y.(Z+1)-dev`.

## Hotfixes

For a same-day fix, repeat the flow with a patch bump and a new tag.
Never rewrite or force-push a released tag; cut a new patch instead.

## Draft release notes

`release-drafter.yml` maintains a running draft of the next release's
notes, grouped by PR label (Features, Bug Fixes, Maintenance, Docs,
Dependencies). Apply the right label when merging PRs so the draft is
accurate when you come to ship.
