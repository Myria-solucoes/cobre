#!/usr/bin/env bash
#
# check-comment-line-refs.sh — Drift-prone line-reference advisory (ADVISORY).
#
# A shipped comment that pins a `file.rs:NNN` line reference is drift-prone: the
# line moves, the reference rots. The durability contract (see
# .claude/rules/comments.md §4 "Durability" and N3) says reference by stable
# SYMBOL, never by `file.rs:NNN`. Some pinned line refs are acceptable
# navigation hints, so this gate is ADVISORY — it reports candidates so authors
# prefer the stable-symbol form, but NEVER fails the build.
#
# What it flags, over production `//`/`///`/`//!` comment lines:
#   A `file.rs:NNN` token: a `[a-z_]+\.rs:[0-9]+` basename followed by a line
#   number, INCLUDING the drift-prone range/chain extensions:
#     * en-dash range form   — `file.rs:10–20`  (U+2013 between bounds)
#     * comma-chained form   — `file.rs:10,15`  (`:N,M` line list)
#   Only the token span is reported (grep -oP style), not the whole comment.
#
# Scope: production source under crates/*/src/ (including stub crates). The
#   cfg(test) tail-block awk pre-filter (borrowed from check-comment-refs.sh /
#   check-no-plan-leaks.sh) drops everything from the first `#[cfg(test)]` line
#   onward, so test-scope comments are out of scope. Production src is currently
#   clean (the line refs were swept to stable symbols), so a clean tree prints
#   no hits.
#
#   Known limitation (same as the sibling gates): the exclusion assumes the test
#   module is a tail block. Files with a mid-file test module followed by
#   production code would incorrectly skip that trailing code. In practice cobre
#   files follow the tail-block convention.
#
# Reporting: each hit is printed as `FILE:LINE: <matched token span>` under an
#   `ADVISORY:` banner, followed by a footer.
#
# Exit code: ALWAYS 0 (advisory — never fails the build).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT

# shellcheck source=scripts/ci/lib/comment_scan.sh
source "${REPO_ROOT}/scripts/ci/lib/comment_scan.sh"
command -v cs_emit_production_lines >/dev/null \
    || { echo "FATAL: scripts/ci/lib/comment_scan.sh did not load its helpers." >&2; exit 2; }

# Production .rs source directories, scanned per-file with the cfg(test)
# tail-block exclusion. Mirrors check-comment-refs.sh SCAN_DIRS.
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

# Token pattern (grep -oP extracts the matched span only).
#   [a-z_]+\.rs            — a lowercase-snake basename ending in `.rs`
#   :[0-9]+                — the first line number
#   (\x{2013}[0-9]+ | ,[0-9]+)*  — optional drift-prone tail:
#     * en-dash range bound(s)  `\x{2013}` is U+2013 EN DASH
#     * comma-chained line(s)
# -P is required for the `\x{2013}` (multibyte UTF-8) en-dash class.
readonly TOKEN_PATTERN='[a-z_]+\.rs:[0-9]+((\x{2013}[0-9]+)|(,[0-9]+))*'

# Emit production comment lines as `FILE:LINENO:CONTENT`, truncated at the
# cfg(test) tail block, restricted to lines carrying a `//` comment marker
# (covers `//`, `///`, and `//!`).
emit_comment_lines() {
    cs_emit_comment_lines "$1"
}

# From a pre-filtered `FILE:LINE:CONTENT` stream, re-extract each matched TOKEN
# while preserving the `FILE:LINE:` location prefix. Emits `FILE:LINE: TOKEN`.
extract_tokens() {
    local line file_part rest line_part content token
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        file_part="${line%%:*}"
        rest="${line#*:}"
        line_part="${rest%%:*}"
        content="${rest#*:}"
        while IFS= read -r token; do
            [[ -n "$token" ]] || continue
            printf '%s:%s: %s\n' "$file_part" "$line_part" "$token"
        done < <(printf '%s\n' "$content" | grep -oP "$TOKEN_PATTERN" || true)
    done
}

# Build a single combined comment stream across every production .rs file, in
# `FILE:LINE:CONTENT` form with the cfg(test) tail block already truncated.
comment_stream=""
for dir in "${SCAN_DIRS[@]}"; do
    [[ -d "$dir" ]] || continue
    while IFS= read -r -d '' file; do
        file_stream="$(emit_comment_lines "$file")"
        [[ -n "$file_stream" ]] || continue
        comment_stream+="${file_stream}"$'\n'
    done < <(find "$dir" -name "*.rs" -print0)
done
comment_stream="${comment_stream%$'\n'}"

# Pre-filter to candidate lines (cheap), then extract the matched token spans.
# The stream is `FILE:LINENO:CONTENT`, and the `FILE` part itself ends in
# `<basename>.rs` followed by `:LINENO` — which matches TOKEN_PATTERN. Anchor the
# filter PAST the `FILE:LINENO:` prefix (`^[^:]+:[0-9]+:`) so only a token in the
# comment CONTENT qualifies; otherwise every comment line is a false candidate and
# the per-line extraction below runs tens of thousands of times.
hits=""
candidate_lines="$(printf '%s\n' "$comment_stream" \
    | grep -P "^[^:]+:[0-9]+:.*${TOKEN_PATTERN}" || true)"
if [[ -n "$candidate_lines" ]]; then
    hits="$(printf '%s\n' "$candidate_lines" | extract_tokens || true)"
fi
hits="${hits%$'\n'}"

if [[ -n "$hits" ]]; then
    echo "ADVISORY: drift-prone line references in shipped comments"
    echo ""
    echo "$hits"
    echo ""
    echo "Strip the rot, keep the invariant: prefer the stable symbol form —"
    echo "reference Type::method (or a named test / intra-doc link), not"
    echo "file.rs:NNN. A pinned line number moves and the reference rots."
    echo "Advisory only — this does not fail the build."
    exit 0
fi

echo "OK: no drift-prone line references found in shipped comments."
exit 0
