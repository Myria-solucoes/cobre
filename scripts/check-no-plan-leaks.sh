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
#   \bF[0-9]?-[0-9]{2,}\b — workstream feature IDs: "F2-002", "F3-006",
#                          "F1-007", etc. (mandatory >=2 trailing digits so
#                          single-digit tokens do not collide).
#   \bW[0-9]+ (reset|rebake|workstream|phase)\b — context-anchored workstream
#                          phase language: "W2 reset", "W3 rebake", etc. Bare
#                          "W2" / "W3" without the anchored phrase is allowed.
#
# Scope: production source under crates/*/src/ (including the umbrella
#   crates/cobre/src and the reserved stub crates), test/bench source under
#   crates/*/tests/ and crates/*/benches/, book/, CHANGELOG.md, and README.md.
#   Vendored *.min.js files are excluded from every pass.
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
readonly REPO_ROOT

# shellcheck source=scripts/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

readonly PATTERN='[Ee]pic[ -][0-9]+|[Tt]icket[ -][0-9]+|T0[0-9][0-9]|\bsprint\b|\bF[0-9]?-[0-9]{2,}\b|\bW[0-9]+ (reset|rebake|workstream|phase)\b'

# .rs source directories: scanned per-file with the cfg(test) tail-block
# exclusion (see header). The umbrella crates/cobre/src and the reserved stub
# crates are scanned too. The crates/*/tests and crates/*/benches dirs are
# glob-expanded below and appended; the cfg(test) awk filter is a harmless
# no-op there (test/bench files carry no #[cfg(test)] tail boundary).
SCAN_DIRS=(
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
    "${REPO_ROOT}/crates/cobre/src"
    "${REPO_ROOT}/crates/cobre-flow/src"
    "${REPO_ROOT}/crates/cobre-uc/src"
    "${REPO_ROOT}/crates/cobre-emt/src"
)

# Append the test/bench source dirs. nullglob keeps the loop a no-op when a
# glob matches nothing (e.g. a crate without a benches/ dir), so unexpanded
# literal globs never enter SCAN_DIRS. The [[ -d ]] guard in the scan loop is
# the second line of defence.
shopt -s nullglob
for dir in "${REPO_ROOT}"/crates/*/tests "${REPO_ROOT}"/crates/*/benches; do
    SCAN_DIRS+=("$dir")
done
shopt -u nullglob

readonly SCAN_DIRS

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
        matches=$(cs_emit_production_lines "$file" | grep -E "$PATTERN") || true
        if [[ -n "$matches" ]]; then
            violations+="${matches}"$'\n'
        fi
    done < <(find "$dir" -name "*.rs" -print0)
done

# Non-.rs targets: plain whole-file recursive grep. Vendored *.min.js files
# (e.g. under book/) are excluded — the widened PATTERN could otherwise match
# minified JS tokens.
file_violations=$(grep -rnE "$PATTERN" \
    --exclude='*.min.js' "${SCAN_FILES[@]}" 2>/dev/null \
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
    echo ""
    echo "For workstream feature IDs (F2-002, F3-006, ...) and anchored"
    echo "workstream phase language (W3 rebake, W2 reset, ...), strip the"
    echo "rot, keep the invariant: replace the plan ID with the behavioural"
    echo "phrase it tags and leave the surrounding invariant intact — never"
    echo "delete the comment line."
    exit 1
fi

echo "OK: no plan-structure leaks found in shipped artefacts."
exit 0
