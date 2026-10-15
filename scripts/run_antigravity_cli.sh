#!/usr/bin/env bash
# Official Antigravity CLI installed in the ignored repository runtime directory.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGY_RUNTIME="$REPO_ROOT/.venv/antigravity_cli"
umask 077
mkdir -p "$AGY_RUNTIME/session"
cd "$AGY_RUNTIME/session"
exec "$AGY_RUNTIME/agy" "$@"
