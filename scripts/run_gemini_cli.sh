#!/usr/bin/env bash
# Repository-local Gemini CLI. Account state stays in the ignored .venv tree.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GEMINI_RUNTIME="$REPO_ROOT/.venv/gemini_cli"
export PATH="$GEMINI_RUNTIME/node_modules/.bin:$PATH"
export GEMINI_CLI_HOME="$GEMINI_RUNTIME/account"
umask 077
mkdir -p "$GEMINI_CLI_HOME" "$GEMINI_RUNTIME/session"
if [[ ! -x "$GEMINI_RUNTIME/node_modules/.bin/gemini" ]]; then
  echo 'Gemini CLI is not installed in .venv/gemini_cli.' >&2
  exit 1
fi
cd "$GEMINI_RUNTIME/session"
if [[ "${1:-}" == login || "${1:-}" == login-only ]]; then
  GEMINI_LOGIN_MODE="$1"
  shift
  # Manual browser/code flow works from SSH without a local callback tunnel.
  export NO_BROWSER=true
  export GEMINI_DEFAULT_AUTH_TYPE=oauth-personal
  unset GEMINI_API_KEY GOOGLE_API_KEY GOOGLE_GENAI_USE_VERTEXAI
  # A previously selected API-key login bypasses the provider picker. Explicitly
  # select Google OAuth for this subscription-login command; preserve other settings.
  node - "$GEMINI_CLI_HOME/.gemini/settings.json" <<'NODE'
const fs = require('node:fs');
const path = require('node:path');
const file = process.argv[2];
const settings = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : {};
settings.security ??= {};
settings.security.auth ??= {};
settings.security.auth.selectedType = 'oauth-personal';
// Keep the login dialog on Google OAuth, including after an authentication error.
// Switching providers clears cached OAuth credentials in Gemini CLI.
settings.security.auth.enforcedType = 'oauth-personal';
fs.mkdirSync(path.dirname(file), { recursive: true, mode: 0o700 });
fs.writeFileSync(file, JSON.stringify(settings, null, 2) + '\n', { mode: 0o600 });
NODE
  if [[ "$GEMINI_LOGIN_MODE" == login-only ]]; then
    exec node "$REPO_ROOT/scripts/gemini_login_only.mjs" "$@"
  fi
fi
exec "$GEMINI_RUNTIME/node_modules/.bin/gemini" "$@"
