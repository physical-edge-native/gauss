#!/usr/bin/env bash
# Checks run by CI on Linux, macOS and Windows (see .github/workflows/ci.yml).
# gauss has no code yet, so these check only the repository's shared conventions.
# Add checks for code here when code arrives.
set -euo pipefail
cd "$(dirname "$0")/.."

failures=0
fail() {
  echo "error: $*" >&2
  failures=$((failures + 1))
}

# Byte-wise fixed-string search, so the result does not depend on the locale.
has_line() {
  LC_ALL=C grep -qF -- "$2" "$1"
}

for f in LICENSE-APACHE LICENSE-MIT; do
  [ -f "$f" ] || fail "$f is missing (pena-meta contracts.md: MIT OR Apache-2.0)"
done

# The copyright line fixed in pena-meta contracts.md.
copyright_line="Copyright (c) 2026 The Physical-Edge Native contributors"
if [ -f LICENSE-MIT ] && ! LC_ALL=C grep -qxF -- "$copyright_line" <(tr -d '\r' < LICENSE-MIT); then
  fail "LICENSE-MIT does not have the line '$copyright_line' (pena-meta contracts.md)"
fi

if [ ! -f README.md ]; then
  fail "README.md is missing"
elif ! has_line README.md "Status:"; then
  fail "README.md has no 'Status:' line"
fi

if [ ! -f CLAUDE.md ]; then
  fail "CLAUDE.md is missing"
elif ! has_line CLAUDE.md "## リポジトリをまたぐ決まり"; then
  fail "CLAUDE.md has no section 'リポジトリをまたぐ決まり' (pena-meta cross-repo-rules.md)"
fi

if [ "$failures" -ne 0 ]; then
  echo "ci-check: $failures check(s) failed" >&2
  exit 1
fi
echo "ci-check: all checks passed"
