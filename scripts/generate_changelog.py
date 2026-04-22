"""Generate or update CHANGELOG.md from conventional commits since the last tag.

Usage:
    python scripts/generate_changelog.py              # write under new heading
    python scripts/generate_changelog.py --dry-run    # print only
    python scripts/generate_changelog.py --version X  # override detected version
    python scripts/generate_changelog.py --from TAG   # override starting ref

The script groups commits by conventional-commit prefix (``feat``, ``fix``,
``docs``, ``chore``, ``build``, ``ci``, ``refactor``, ``test``) and writes a
new section under ``# Changelog``. It is safe to run repeatedly: an existing
heading for the same version is replaced in place rather than duplicated.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]

ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"

CATEGORY_ORDER: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Features", ("feat",)),
    ("Bug Fixes", ("fix",)),
    ("Documentation", ("docs",)),
    ("Build", ("build",)),
    ("CI", ("ci",)),
    ("Refactor", ("refactor",)),
    ("Tests", ("test",)),
    ("Maintenance", ("chore",)),
)

COMMIT_RE = re.compile(
    r"^(?P<type>[a-zA-Z]+)(?:\((?P<scope>[^)]+)\))?(?P<bang>!?):\s*(?P<subject>.+)$"
)


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def read_project_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def last_tag() -> str | None:
    try:
        return run(["git", "describe", "--tags", "--abbrev=0"])
    except subprocess.CalledProcessError:
        return None


def collect_commits(since: str | None) -> list[str]:
    log_range = f"{since}..HEAD" if since else "HEAD"
    out = run(["git", "log", "--pretty=format:%s", log_range])
    return [line for line in out.splitlines() if line.strip()]


def group_commits(subjects: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {title: [] for title, _ in CATEGORY_ORDER}
    groups["Other"] = []
    for subject in subjects:
        match = COMMIT_RE.match(subject)
        if not match:
            groups["Other"].append(subject)
            continue
        ctype = match["type"].lower()
        body = match["subject"].strip()
        placed = False
        for title, prefixes in CATEGORY_ORDER:
            if ctype in prefixes:
                groups[title].append(body)
                placed = True
                break
        if not placed:
            groups["Other"].append(subject)
    return groups


def render_section(version: str, groups: dict[str, list[str]]) -> str:
    today = date.today().isoformat()
    lines = [f"## v{version} ({today})", ""]
    has_any = False
    for title, _ in CATEGORY_ORDER:
        entries = groups.get(title) or []
        if not entries:
            continue
        has_any = True
        lines.append(f"### {title}")
        for entry in entries:
            lines.append(f"- {entry}")
        lines.append("")
    other = groups.get("Other") or []
    if other:
        has_any = True
        lines.append("### Other")
        for entry in other:
            lines.append(f"- {entry}")
        lines.append("")
    if not has_any:
        lines.append("_No notable changes._")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def splice_section(existing: str, version: str, section: str) -> str:
    """Insert or replace the section for ``version`` in ``existing`` text."""
    heading_re = re.compile(rf"^## v{re.escape(version)}\b.*?$", re.MULTILINE)
    match = heading_re.search(existing)
    if match:
        start = match.start()
        next_heading = re.search(r"^## v", existing[match.end() :], re.MULTILINE)
        if next_heading:
            end = match.end() + next_heading.start()
        else:
            end = len(existing)
        return existing[:start] + section + "\n" + existing[end:]

    header_re = re.compile(r"^# Changelog\s*\n+", re.MULTILINE)
    header_match = header_re.search(existing)
    if header_match:
        insert_at = header_match.end()
        return existing[:insert_at] + section + "\n" + existing[insert_at:]
    return "# Changelog\n\n" + section + "\n" + existing


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", help="Override the version heading.")
    parser.add_argument(
        "--from",
        dest="from_ref",
        help="Git ref to start from (default: most recent tag).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generated section instead of writing CHANGELOG.md.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    version = args.version or read_project_version()
    since = args.from_ref if args.from_ref is not None else last_tag()

    subjects = collect_commits(since)
    groups = group_commits(subjects)
    section = render_section(version, groups)

    if args.dry_run:
        sys.stdout.write(section)
        return 0

    existing = CHANGELOG.read_text(encoding="utf-8") if CHANGELOG.exists() else ""
    updated = splice_section(existing, version, section)
    if updated == existing:
        print("CHANGELOG.md already up to date.")
        return 0
    CHANGELOG.write_text(updated, encoding="utf-8")
    print(f"Updated CHANGELOG.md with v{version} ({len(subjects)} commits).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
