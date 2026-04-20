"""Verify version consistency across pyproject.toml, __init__.py, and CHANGELOG.md."""

import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).parent.parent
INIT_FILE = ROOT / "src" / "checkllm" / "__init__.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"


def get_pyproject_version() -> str:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def get_init_version() -> str:
    text = INIT_FILE.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise ValueError(f"No __version__ found in {INIT_FILE}")
    return match.group(1)


def set_init_version(version: str) -> None:
    text = INIT_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'^(__version__\s*=\s*["\'])[^"\']+(["\'])',
        rf"\g<1>{version}\2",
        text,
        flags=re.MULTILINE,
    )
    INIT_FILE.write_text(updated, encoding="utf-8")


def changelog_has_version(version: str) -> bool:
    text = CHANGELOG_FILE.read_text(encoding="utf-8")
    return bool(re.search(rf"^## v{re.escape(version)}\b", text, re.MULTILINE))


def main() -> int:
    pyproject_ver = get_pyproject_version()
    init_ver = get_init_version()
    changelog_ok = changelog_has_version(pyproject_ver)

    ok = True
    if init_ver != pyproject_ver:
        print(
            f"ERROR: __init__.py version {init_ver!r} != pyproject.toml {pyproject_ver!r}",
            file=sys.stderr,
        )
        ok = False
    if not changelog_ok:
        print(
            f"ERROR: CHANGELOG.md missing entry for v{pyproject_ver}",
            file=sys.stderr,
        )
        ok = False

    if ok:
        print(f"OK: all version references agree on {pyproject_ver}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
