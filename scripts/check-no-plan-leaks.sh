#!/usr/bin/env bash
#
# check-no-plan-leaks.sh — Plan-structure leak gate.
#
# Scans shipped artefacts (production source, book, CHANGELOG) for
# plan-structure tokens that must not appear in user-facing
# content per CLAUDE.md hard rule:
#
#   "No plan-structure references in user-facing artifacts"
#
# Forbidden patterns:
#   [Ee]pic[ -][0-9]+   — "Epic 06", "epic-03", "Epic-12", etc.
#   [Tt]icket[ -][0-9]+ — "ticket-001", "Ticket 42", etc.
#   T0[0-9][0-9]     — "T002", "T007", "T015", etc.
#   \bsprint\b       — sprint planning vocabulary
#
# Scope: production source under crates/*/src/, book/, CHANGELOG.md,
#   and README.md.
#
# Excluded: plans/ (gitignored), .github/, target/, .git/, Cargo.lock,
#   and the script itself.
#
# cfg(test) tail-block exclusion (borrowed from check-infra-genericity.sh):
#   For each .rs file under crates/*/src/, scanning stops at the first line
#   matching `#[cfg(test)]`; all subsequent lines until end-of-file are
#   considered test scope and skipped. Plan refs in test names and test-only
#   comments are out of the user-facing-artifact scope, so the gate truncates
#   each .rs file at the test-module boundary before applying PATTERN. This
#   mirrors the awk pre-filter mechanism in check-infra-genericity.sh.
#
#   Known limitation (same as check-infra-genericity.sh): the exclusion
#   assumes the test module is a tail block. Files with mid-file test modules
#   followed by production code would incorrectly skip that trailing code. In
#   practice, cobre files follow the tail-block convention.
#
#   Non-.rs targets (book/, CHANGELOG.md, README.md) have no cfg(test) concept
#   and stay on the plain whole-file grep path.
#
# Exit codes:
#   0 — No leaks found.
#   1 — Leaks found (details printed to stdout).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

readonly PATTERN='[Ee]pic[ -][0-9]+|[Tt]icket[ -][0-9]+|T0[0-9][0-9]|\bsprint\b'

# .rs source directories: scanned per-file with the cfg(test) tail-block
# exclusion (see header).
readonly SCAN_DIRS=(
    "${REPO_ROOT}/crates/cobre-core/src"
    "${REPO_ROOT}/crates/cobre-io/src"
    "${REPO_ROOT}/crates/cobre-solver/src"
    "${REPO_ROOT}/crates/cobre-comm/src"
    "${REPO_ROOT}/crates/cobre-stochastic/src"
    "${REPO_ROOT}/crates/cobre-sddp/src"
    "${REPO_ROOT}/crates/cobre-cli/src"
    "${REPO_ROOT}/crates/cobre-python/src"
    "${REPO_ROOT}/crates/cobre-mcp/src"
    "${REPO_ROOT}/crates/cobre-tui/src"
)

# Non-.rs targets: scanned whole-file with plain grep (no cfg(test) concept).
readonly SCAN_FILES=(
    "${REPO_ROOT}/book"
    "${REPO_ROOT}/CHANGELOG.md"
    "${REPO_ROOT}/README.md"
)

violations=""

# Stage 1 (awk): structural pre-filter — stop emitting lines once a bare
#   `#[cfg(test)]` line is encountered (tail test-module exclusion). For each
#   production line, emit "FILE:LINENO:CONTENT" so grep output retains location.
# Stage 2 (grep -E): apply PATTERN. awk performs no regex matching here; it only
#   handles line truncation + prefix.
for dir in "${SCAN_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' file; do
        matches=$(awk -v f="$file" '
            /^#\[cfg\(test\)\]/ { exit }
            { printf "%s:%d:%s\n", f, NR, $0 }
        ' "$file" | grep -E "$PATTERN") || true
        if [[ -n "$matches" ]]; then
            violations+="${matches}"$'\n'
        fi
    done < <(find "$dir" -name "*.rs" -print0)
done

# Non-.rs targets: plain whole-file recursive grep.
file_violations=$(grep -rnE "$PATTERN" "${SCAN_FILES[@]}" 2>/dev/null \
    || true)
if [[ -n "$file_violations" ]]; then
    violations+="${file_violations}"$'\n'
fi

# Strip the trailing newline accumulated above.
violations="${violations%$'\n'}"

if [[ -n "$violations" ]]; then
    echo "FAIL: plan-structure leaks found in shipped artefacts."
    echo ""
    echo "$violations"
    echo ""
    echo "Per CLAUDE.md, plan-structure references must not appear"
    echo "in shipped source, the book, or the CHANGELOG. Rewrite"
    echo "in behavioural terms or move to plans/ (gitignored)."
    exit 1
fi

echo "OK: no plan-structure leaks found in shipped artefacts."
exit 0
