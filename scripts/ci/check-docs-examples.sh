#!/usr/bin/env bash
# Lock structural invariants of a fresh `cobre init` -> `run` -> `report`
# against the live binary, so the CLI's on-disk/stdout output shape cannot
# silently drift.
#
# Formerly framed as "book cannot drift from the CLI" (each invariant below
# was a claim made in book/src/); the book was retired (mdBook decommission)
# but these constants were always hardcoded here, not re-read from book/ at
# run time, so the gate's value is independent of the book's existence — it is
# now a plain CLI output-shape regression gate. What it asserts (bump the
# constant here when the CLI's output shape intentionally changes, or the
# gate fails):
#   1. `cobre init --template 1dtoy` materializes exactly EXPECTED_INPUT_FILES
#      regular files. Single source of truth below.
#   2. training/metadata.json EXISTS and carries every TRAINING_METADATA_KEYS
#      top-level key. Routed here per the file each key actually lives in:
#      warm_start_* are NOT here.
#   3. policy/metadata.json EXISTS and carries warm_start_counts + warm_start_cuts.
#   4. `cobre report` stdout JSON has EXACTLY the REPORT_KEYS_SORTED key set.
#
# This gate is scoped to structural invariants the assert_cmd integration
# suite (init.rs / cli_run.rs / cli_report.rs / cli_e2e_*) does not already
# pin exactly (exact input-file count, exact top-level key SETS rather than
# individual key/value spot-checks); it deliberately does not re-cover
# command-execution behavior.
#
# Usage:
#   scripts/ci/check-docs-examples.sh           — assumes ./target/release/cobre is built
#   scripts/ci/check-docs-examples.sh --build   — builds the release binary first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Single source of truth for the expected init file count ───────────────────
# A future template addition must update this constant, or this gate fails.
readonly EXPECTED_INPUT_FILES=11
readonly TEMPLATE="1dtoy"

# Expected top-level keys of training/metadata.json. Per
# the routing decision, warm_start_* belong to policy/metadata.json, NOT here.
readonly TRAINING_METADATA_KEYS=(
  cobre_version
  hostname
  solver
  solver_version
  started_at
  completed_at
  duration_seconds
  status
  configuration
  problem_dimensions
  iterations
  convergence
  row_pool
  bounds
  solve_stats
  distribution
)

# Expected keys that policy/metadata.json must carry.
readonly POLICY_METADATA_KEYS=(
  warm_start_counts
  warm_start_cuts
)

# Expected ReportOutput top-level key set. Sorted; the `cobre report` stdout
# must match this set EXACTLY (no missing, no extra).
readonly REPORT_KEYS_SORTED='bounds,cost,output_directory,simulation,status,training'

BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build) BUILD=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

command -v jq >/dev/null 2>&1 || { echo "ERROR: jq is required but not found on PATH." >&2; exit 2; }

BIN="$REPO_ROOT/target/release/cobre"
if [[ $BUILD -eq 1 || ! -x "$BIN" ]]; then
  cargo build --release --bin cobre
fi

# Fresh mktemp tree: never reuse a pre-existing local output dir, so a stale
# artifact cannot mask a regression with a spurious pass.
TMP_DIR="$(mktemp -d -t cobre-docs-examples-XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

CASE_DIR="$TMP_DIR/case"
OUT_DIR="$TMP_DIR/output"

fail() {
  echo "ERROR: structural invariant drifted — $1" >&2
  exit 1
}

# ── Invariant 1: init materializes exactly the expected file count ───────────
"$BIN" init --template "$TEMPLATE" "$CASE_DIR" >/dev/null
actual_files="$(find "$CASE_DIR" -type f | wc -l | tr -d '[:space:]')"
if [[ "$actual_files" -ne "$EXPECTED_INPUT_FILES" ]]; then
  fail "\`cobre init --template $TEMPLATE\` wrote $actual_files input files, expected $EXPECTED_INPUT_FILES. Update EXPECTED_INPUT_FILES in this script if the template intentionally changed."
fi
echo "init file count: $actual_files == $EXPECTED_INPUT_FILES (expected) ✓"

# Drive a fresh run so the metadata/report assertions read live output.
"$BIN" run "$CASE_DIR" --output "$OUT_DIR" --quiet --color never >/dev/null

# ── Invariant 2: training/metadata.json exists + expected top-level keys ─────
training_meta="$OUT_DIR/training/metadata.json"
[[ -f "$training_meta" ]] || fail "training/metadata.json was not written."
for key in "${TRAINING_METADATA_KEYS[@]}"; do
  jq -e "has(\"$key\")" "$training_meta" >/dev/null \
    || fail "training/metadata.json is missing expected top-level key \`$key\`."
done
echo "training/metadata.json: ${#TRAINING_METADATA_KEYS[@]} expected top-level keys present (incl. row_pool) ✓"

# ── Invariant 3: policy/metadata.json exists + warm_start_* keys ──────────────
policy_meta="$OUT_DIR/policy/metadata.json"
[[ -f "$policy_meta" ]] || fail "policy/metadata.json was not written."
for key in "${POLICY_METADATA_KEYS[@]}"; do
  jq -e "has(\"$key\")" "$policy_meta" >/dev/null \
    || fail "policy/metadata.json is missing expected key \`$key\`."
done
echo "policy/metadata.json: expected keys present (${POLICY_METADATA_KEYS[*]}) ✓"

# ── Invariant 4: report stdout has EXACTLY the expected ReportOutput keys ─────
report_keys="$("$BIN" report "$OUT_DIR" --color never | jq -r 'keys | sort | join(",")')"
if [[ "$report_keys" != "$REPORT_KEYS_SORTED" ]]; then
  fail "\`cobre report\` top-level keys are {$report_keys}, expected {$REPORT_KEYS_SORTED}."
fi
echo "cobre report key set: {$report_keys} == expected ✓"

echo "All init/run/report structural invariants hold. ✓"
