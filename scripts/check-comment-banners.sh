#!/usr/bin/env bash
#
# check-comment-banners.sh — Box-drawing banner-glyph advisory (ADVISORY).
#
# Box-drawing banner glyphs (U+2500..U+257F) used to fence sections inside a
# comment are a Narrator-voice tell (see .claude/rules/comments.md, the banned
# Narrator voice and N5). There are many legitimate section-divider uses across
# the tree, so this gate is ADVISORY — it reports the count and the candidate
# spans so authors prefer plain section headers, but NEVER fails the build.
#
# What it flags, over production `//`/`///`/`//!` comment lines:
#   Any box-drawing glyph in the Unicode range U+2500..U+257F. Matching uses
#   grep -P with the `\x{2500}-\x{257F}` UTF-8-aware class — a byte class would
#   mis-handle these multibyte glyphs. Only the token span is reported.
#
# Scope: production source under crates/*/src/ (including stub crates), with two
#   pre-filter exclusions applied in the awk stage:
#     1. cfg(test) tail block — everything from the first `#[cfg(test)]` line
#        onward is dropped (borrowed from the sibling gates).
#     2. `extern "C"` blocks — from an `extern "C"` opener line to its matching
#        column-0 closing `}` (inclusive), all lines are suppressed. Dividers
#        that mirror a C header inside an `extern "C"` block are an allowed N5
#        carve-out, so they are out of scope here.
#
#   Known limitation (same as the sibling gates): the cfg(test) exclusion
#   assumes the test module is a tail block. The `extern "C"` skip assumes the
#   block's closing brace is the first column-0 `}` after the opener (true for
#   the cobre FFI blocks, whose items are indented).
#
# Reporting: each hit is printed as `FILE:LINE: <matched glyph span>` under an
#   `ADVISORY:` banner, followed by a non-zero hit count and a footer.
#
# Exit code: ALWAYS 0 (advisory — never fails the build).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly REPO_ROOT

# Production .rs source directories, scanned per-file with the cfg(test)
# tail-block + extern "C" block exclusions. Mirrors check-comment-refs.sh.
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

# Box-drawing glyph range U+2500..U+257F. `\x{...}` requires grep -P (PCRE);
# the class is UTF-8-aware so it matches the whole multibyte glyph, not a byte.
# The `+` collapses a contiguous divider run (`────────`) into one matched span
# so a banner reports as a single span, not one entry per glyph.
readonly GLYPH_PATTERN='[\x{2500}-\x{257F}]+'

# Emit production comment lines as `FILE:LINENO:CONTENT`, restricted to lines
# carrying a `//` comment marker (covers `//`, `///`, `//!`), with two
# structural pre-filters applied as the file is walked:
#   * exit once a bare `#[cfg(test)]` line is reached (tail test-module skip).
#   * suppress every line from an `extern "C"` opener through its matching
#     column-0 closing `}` (inclusive) via the in_extern_c state flag.
emit_comment_lines() {
    local file="$1"
    awk -v f="$file" '
        /^[[:space:]]*#\[cfg\((all\()?test[,)]/ { exit }
        # Enter extern "C" block: suppress the opener and set state.
        /extern[[:space:]]+"C"/ { in_extern_c = 1; next }
        # Inside extern "C": suppress; the first column-0 `}` closes the block.
        in_extern_c == 1 {
            if ($0 ~ /^\}/) { in_extern_c = 0 }
            next
        }
        /\/\// { printf "%s:%d:%s\n", f, NR, $0 }
    ' "$file"
}

# From a pre-filtered `FILE:LINE:CONTENT` stream, emit ONE entry per flagged
# comment line — the divider line is the unit of finding — preserving the
# `FILE:LINE:` location prefix and showing the first matched glyph-run span as
# the representative token. Emits `FILE:LINE: SPAN`. One-per-line keeps the hit
# count at divider granularity (a banner is one finding, not one-per-glyph).
extract_tokens() {
    local line file_part rest line_part content token
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        file_part="${line%%:*}"
        rest="${line#*:}"
        line_part="${rest%%:*}"
        content="${rest#*:}"
        token="$(printf '%s\n' "$content" | grep -oP "$GLYPH_PATTERN" | head -n 1 || true)"
        [[ -n "$token" ]] || continue
        printf '%s:%s: %s\n' "$file_part" "$line_part" "$token"
    done
}

# Build a single combined comment stream across every production .rs file, in
# `FILE:LINE:CONTENT` form with the cfg(test) tail block and extern "C" blocks
# already removed.
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

# Pre-filter to candidate lines (cheap), then extract the matched glyph spans.
hits=""
candidate_lines="$(printf '%s\n' "$comment_stream" \
    | grep -P "$GLYPH_PATTERN" || true)"
if [[ -n "$candidate_lines" ]]; then
    hits="$(printf '%s\n' "$candidate_lines" | extract_tokens || true)"
fi
hits="${hits%$'\n'}"

if [[ -n "$hits" ]]; then
    hit_count="$(printf '%s\n' "$hits" | grep -c '' || true)"
    echo "ADVISORY: box-drawing banner glyphs in shipped comments (${hit_count} hits)"
    echo ""
    echo "$hits"
    echo ""
    echo "Banner glyphs are a Narrator-voice tell. Prefer a plain section header"
    echo "(a short capitalised label line) over a box-drawing divider. Advisory"
    echo "only — this does not fail the build."
    exit 0
fi

echo "OK: no box-drawing banner glyphs found in shipped comments."
exit 0
