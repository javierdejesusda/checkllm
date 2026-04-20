#!/usr/bin/env bash
# Regenerate requirements.lock from pyproject.toml.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
cd "$ROOT"
pip-compile \
  --no-header \
  --strip-extras \
  --output-file requirements.lock \
  pyproject.toml
echo "requirements.lock updated."
