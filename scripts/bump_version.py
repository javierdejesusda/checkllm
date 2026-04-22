"""Bump or set the checkllm version across all version-bearing files.

Usage:
    python scripts/bump_version.py major
    python scripts/bump_version.py minor
    python scripts/bump_version.py patch
    python scripts/bump_version.py --set 5.0.2-dev

The script is idempotent: running it twice with the same ``--set`` value
produces no further changes. It updates ``pyproject.toml`` and
``src/checkllm/__init__.py`` (and ``src/checkllm/_version.py`` if present).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_FILE = ROOT / "src" / "checkllm" / "__init__.py"
VERSION_FILE = ROOT / "src" / "checkllm" / "_version.py"

VERSION_RE = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?P<suffix>[-+\.][0-9A-Za-z\.\-+]*)?$"
)


def read_pyproject_version() -> str:
    """Return the current version recorded in ``pyproject.toml``."""
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def parse_version(version: str) -> tuple[int, int, int, str]:
    """Split a semver-ish string into ``(major, minor, patch, suffix)``."""
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError(f"Unrecognised version string: {version!r}")
    return (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
        match["suffix"] or "",
    )


def bump(current: str, part: str) -> str:
    """Return the next version for ``part`` in ``{major, minor, patch}``.

    Args:
        current: Existing version string.
        part: Segment to increment.

    Returns:
        The new version string with pre-release suffixes stripped.
    """
    major, minor, patch, _suffix = parse_version(current)
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    elif part == "patch":
        patch += 1
    else:
        raise ValueError(f"Unknown bump part: {part}")
    return f"{major}.{minor}.{patch}"


def _replace(path: Path, pattern: str, replacement: str) -> bool:
    """Perform a regex substitution on ``path``; return True if it changed."""
    if not path.exists():
        return False
    original = path.read_text(encoding="utf-8")
    updated = re.sub(pattern, replacement, original, count=1, flags=re.MULTILINE)
    if original == updated:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def write_version(new_version: str) -> list[Path]:
    """Write ``new_version`` into every version-bearing file.

    Returns:
        The list of files that were modified.
    """
    changed: list[Path] = []

    pyproject_pattern = r'^(version\s*=\s*")[^"]+(")'
    if _replace(PYPROJECT, pyproject_pattern, rf"\g<1>{new_version}\g<2>"):
        changed.append(PYPROJECT)

    init_pattern = r'^(__version__\s*=\s*")[^"]+(")'
    if _replace(INIT_FILE, init_pattern, rf"\g<1>{new_version}\g<2>"):
        changed.append(INIT_FILE)

    if VERSION_FILE.exists():
        version_pattern = r'^(__version__\s*=\s*")[^"]+(")'
        if _replace(VERSION_FILE, version_pattern, rf"\g<1>{new_version}\g<2>"):
            changed.append(VERSION_FILE)

    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "part",
        nargs="?",
        choices=("major", "minor", "patch"),
        help="Which semantic version component to bump.",
    )
    group.add_argument(
        "--set",
        dest="explicit",
        metavar="VERSION",
        help="Set the version to an explicit value instead of bumping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the new version without writing any files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    current = read_pyproject_version()

    if args.explicit is not None:
        new_version = args.explicit
    else:
        new_version = bump(current, args.part)

    if new_version == current:
        print(f"Version already at {current}; nothing to do.")
        return 0

    if args.dry_run:
        print(f"Would bump {current} -> {new_version}")
        return 0

    changed = write_version(new_version)
    if not changed:
        print(f"No files updated (already at {new_version}).")
        return 0

    print(f"Bumped {current} -> {new_version} in:")
    for path in changed:
        print(f"  - {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
