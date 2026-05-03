"""Structural lint for the CheckLLM paper.

Performs three checks:

1. ``\\begin{...}`` and ``\\end{...}`` blocks balance.
2. Every ``\\citep{...}`` / ``\\citet{...}`` key is defined in
   ``bibliography.bib``.
3. Every ``\\ref{...}`` / ``\\autoref{...}`` target is defined as a
   ``\\label{...}`` somewhere in the source.

Exits with status 1 (and prints the offending fragments) if any check
fails. This script is fast enough to run on every commit.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
TEX_PATH = PAPER_DIR / "checkllm.tex"
BIB_PATH = PAPER_DIR / "bibliography.bib"
GENERATED_PATH = PAPER_DIR / "figures" / "generated_tables.tex"

_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_RE = re.compile(r"\\end\{([^}]+)\}")
_CITE_RE = re.compile(r"\\cite[pt]?\{([^}]+)\}")
_REF_RE = re.compile(r"\\(?:auto)?ref\{([^}]+)\}")
_LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
_BIBKEY_RE = re.compile(r"^\s*@\w+\{\s*([^,\s]+)\s*,", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    """Remove TeX comments ``% ...`` for lint purposes.

    A ``%`` introduces a comment unless escaped as ``\\%``.
    """
    out: list[str] = []
    for line in src.splitlines():
        cleaned: list[str] = []
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "\\" and i + 1 < len(line):
                cleaned.append(line[i : i + 2])
                i += 2
                continue
            if ch == "%":
                break
            cleaned.append(ch)
            i += 1
        out.append("".join(cleaned))
    return "\n".join(out)


def _check_begin_end(src: str) -> list[str]:
    """Return list of error strings for unbalanced begin/end blocks."""
    errs: list[str] = []
    stack: list[str] = []
    tokens = re.findall(r"\\(begin|end)\{([^}]+)\}", src)
    for kind, name in tokens:
        if kind == "begin":
            stack.append(name)
        else:
            if not stack:
                errs.append(f"\\end{{{name}}} with no matching \\begin")
            elif stack[-1] != name:
                errs.append(f"\\end{{{name}}} does not close \\begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
    if stack:
        errs.append(f"unclosed \\begin blocks: {stack}")
    return errs


def _check_citations(src: str, bib_keys: set[str]) -> list[str]:
    """Return list of error strings for unknown citation keys."""
    errs: list[str] = []
    for match in _CITE_RE.finditer(src):
        keys = [k.strip() for k in match.group(1).split(",")]
        for key in keys:
            if key and key not in bib_keys:
                errs.append(f"unknown citation key: {key}")
    return errs


def _check_refs(src: str) -> list[str]:
    """Return list of error strings for refs without matching labels."""
    labels = set(_LABEL_RE.findall(src))
    errs: list[str] = []
    for match in _REF_RE.finditer(src):
        target = match.group(1).strip()
        if target not in labels:
            errs.append(f"undefined ref/autoref target: {target}")
    return errs


def main() -> int:
    """Run all checks and return a process exit code."""
    if not TEX_PATH.exists():
        print(f"[lint] missing source: {TEX_PATH}", file=sys.stderr)
        return 1
    if not BIB_PATH.exists():
        print(f"[lint] missing bibliography: {BIB_PATH}", file=sys.stderr)
        return 1

    raw = _read(TEX_PATH)
    if GENERATED_PATH.exists():
        raw = raw + "\n" + _read(GENERATED_PATH)
    src = _strip_comments(raw)

    bib_keys = set(_BIBKEY_RE.findall(_read(BIB_PATH)))

    errs: list[str] = []
    errs += _check_begin_end(src)
    errs += _check_citations(src, bib_keys)
    errs += _check_refs(src)

    if errs:
        print("[lint] FAIL", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("[lint] OK")
    print(f"  bibliography keys: {len(bib_keys)}")
    print("  begin/end blocks balanced")
    print("  citations + refs resolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
