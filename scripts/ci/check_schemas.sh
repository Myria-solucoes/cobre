#!/usr/bin/env bash
# Verify that book/src/schemas/ is in sync with what `cobre schema export` produces.
# Exits non-zero on any drift, printing the diff for the failing files.
#
# Usage:
#   scripts/check_schemas.sh        — assumes ./target/release/cobre is built
#   scripts/check_schemas.sh --build  — builds the release binary first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

BIN="$REPO_ROOT/target/release/cobre"
if [[ $BUILD -eq 1 || ! -x "$BIN" ]]; then
  cargo build --release --bin cobre
fi

TMP_DIR="$(mktemp -d -t cobre-schemas-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

"$BIN" schema export --output-dir "$TMP_DIR" > /dev/null

if diff -ruN "$REPO_ROOT/book/src/schemas/" "$TMP_DIR/" > "$TMP_DIR/drift.diff"; then
  echo "Schemas in book/src/schemas/ match \`cobre schema export\` output. ✓"
  exit 0
fi

echo "ERROR: book/src/schemas/ drifts from \`cobre schema export\` output."
echo "Run \`scripts/check_schemas.sh --build\` locally, then regenerate with:"
echo "  cargo build --release --bin cobre"
echo "  ./target/release/cobre schema export --output-dir book/src/schemas"
echo ""
echo "--- Drift diff ---"
cat "$TMP_DIR/drift.diff"
exit 1
