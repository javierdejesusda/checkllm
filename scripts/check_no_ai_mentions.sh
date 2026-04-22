#!/usr/bin/env bash
# Fail the commit if any staged file references AI/Claude/Anthropic.
#
# Enforces the project rule in CLAUDE.md: commit messages and committed
# files must not mention Claude, Anthropic, AI assistants, or
# Co-Authored-By trailers attributing work to an AI tool.

set -euo pipefail

if [ "$#" -eq 0 ]; then
  exit 0
fi

pattern='(Co-Authored-By:[[:space:]]*Claude|Generated (with|by) Claude|🤖 Generated with|Made with Claude|written by Claude|Claude Code|Anthropic[[:space:]]+Claude|assisted by (Claude|Anthropic|AI assistant)|AI assistant\b)'
status=0

for file in "$@"; do
  if [ ! -f "$file" ]; then
    continue
  fi
  if grep -HInE "$pattern" "$file" >/dev/null 2>&1; then
    echo "error: disallowed AI/Claude/Anthropic reference found in: $file" >&2
    grep -HInE "$pattern" "$file" >&2 || true
    status=1
  fi
done

exit "$status"
