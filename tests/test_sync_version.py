"""Tests for scripts/sync_version.py."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sync_version  # noqa: E402


@pytest.fixture()
def tmp_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sync_version, "ROOT", tmp_path)
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "pkg"\nversion = "1.2.3"\n')
    src_pkg = tmp_path / "src" / "pkg"
    src_pkg.mkdir(parents=True)
    init_file = src_pkg / "__init__.py"
    init_file.write_text('__version__ = "1.2.3"\n')
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("## v1.2.3 (2026-01-01)\n\nSome notes.\n")
    monkeypatch.setattr(sync_version, "INIT_FILE", init_file)
    monkeypatch.setattr(sync_version, "CHANGELOG_FILE", changelog)
    return tmp_path


def test_get_pyproject_version(tmp_project: Path) -> None:
    assert sync_version.get_pyproject_version() == "1.2.3"


def test_get_init_version(tmp_project: Path) -> None:
    assert sync_version.get_init_version() == "1.2.3"


def test_set_init_version(tmp_project: Path) -> None:
    sync_version.set_init_version("2.0.0")
    assert sync_version.get_init_version() == "2.0.0"


def test_changelog_has_version_true(tmp_project: Path) -> None:
    assert sync_version.changelog_has_version("1.2.3") is True


def test_changelog_has_version_false(tmp_project: Path) -> None:
    assert sync_version.changelog_has_version("9.9.9") is False


def test_main_all_in_sync(tmp_project: Path) -> None:
    assert sync_version.main() == 0


def test_main_init_out_of_sync(tmp_project: Path) -> None:
    sync_version.set_init_version("0.0.0")
    assert sync_version.main() == 1
