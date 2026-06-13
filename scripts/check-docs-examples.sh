#!/usr/bin/env bash
# Lock the DOCUMENTED structural invariants of a fresh `cobre init` -> `run` ->
# `report` against the live binary, so the book cannot silently drift from what
# the CLI actually produces.
#
# What it asserts (each is a claim made in book/src/, named in the failure
# message that fires when it drifts):
#   1. `cobre init --template 1dtoy` materializes exactly EXPECTED_INPUT_FILES
#      regular files (book/src/tutorial/quickstart.md: "Cobre writes N input
#      files"; book/src/tutorial/anatomy-of-a-case.md). Single source of truth
#      below — bump it here AND in the book together when a template file is
#      added, or this gate fails.
#   2. training/metadata.json EXISTS and carries every documented top-level key
#      (book/src/reference/output-format.md, "training/metadata.json"). Routed
#      here per the file each key actually lives in: warm_start_* are NOT here.
#   3. policy/metadata.json EXISTS and carries warm_start_counts + warm_start_cuts
#      (book/src/reference/output-format.md, "policy/metadata.json").
#   4. `cobre report` stdout JSON has EXACTLY the documented ReportOutput key set
#      (book/src/guide/cli-reference.md + tutorial/understanding-results.md).
#
# This gate is scoped to the DOCUMENTED structural invariants the assert_cmd
# integration suite (init.rs / cli_run.rs / cli_report.rs / cli_e2e_*) does not
# already pin; it deliberately does not re-cover command-execution behavior.
#
# Usage:
#   scripts/check-docs-examples.sh           — assumes ./target/release/cobre is built
#   scripts/check-docs-examples.sh --build   — builds the release binary first
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Single source of truth for the documented init file count ─────────────────
# The book states this verbatim ("Cobre writes 11 input files"). Keep them in
# lockstep: a future template addition must update BOTH the book prose and this
# constant, and this gate fails until it does.
readonly EXPECTED_INPUT_FILES=11
readonly TEMPLATE="1dtoy"

# Documented top-level keys of training/metadata.json (output-format.md). Per
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

# Documented keys that policy/metadata.json must carry (output-format.md).
readonly POLICY_METADATA_KEYS=(
  warm_start_counts
  warm_start_cuts
)

# Documented ReportOutput top-level key set (cli-reference.md). Sorted; the
# `cobre report` stdout must match this set EXACTLY (no missing, no extra).
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
  echo "ERROR: documented invariant drifted — $1" >&2
  exit 1
}

# ── Invariant 1: init materializes exactly the documented file count ──────────
"$BIN" init --template "$TEMPLATE" "$CASE_DIR" >/dev/null
actual_files="$(find "$CASE_DIR" -type f | wc -l | tr -d '[:space:]')"
if [[ "$actual_files" -ne "$EXPECTED_INPUT_FILES" ]]; then
  fail "\`cobre init --template $TEMPLATE\` wrote $actual_files input files, but the book documents $EXPECTED_INPUT_FILES (book/src/tutorial/quickstart.md). Update both the book prose and EXPECTED_INPUT_FILES in this script."
fi
echo "init file count: $actual_files == $EXPECTED_INPUT_FILES (documented) ✓"

# Drive a fresh run so the metadata/report assertions read live output.
"$BIN" run "$CASE_DIR" --output "$OUT_DIR" --quiet --color never >/dev/null

# ── Invariant 2: training/metadata.json exists + documented top-level keys ────
training_meta="$OUT_DIR/training/metadata.json"
[[ -f "$training_meta" ]] || fail "training/metadata.json was not written (book/src/reference/output-format.md documents it)."
for key in "${TRAINING_METADATA_KEYS[@]}"; do
  jq -e "has(\"$key\")" "$training_meta" >/dev/null \
    || fail "training/metadata.json is missing documented top-level key \`$key\` (book/src/reference/output-format.md, training/metadata.json table)."
done
echo "training/metadata.json: ${#TRAINING_METADATA_KEYS[@]} documented top-level keys present (incl. row_pool) ✓"

# ── Invariant 3: policy/metadata.json exists + warm_start_* keys ──────────────
policy_meta="$OUT_DIR/policy/metadata.json"
[[ -f "$policy_meta" ]] || fail "policy/metadata.json was not written (book/src/reference/output-format.md documents it)."
for key in "${POLICY_METADATA_KEYS[@]}"; do
  jq -e "has(\"$key\")" "$policy_meta" >/dev/null \
    || fail "policy/metadata.json is missing documented key \`$key\` (book/src/reference/output-format.md, policy/metadata.json table)."
done
echo "policy/metadata.json: documented keys present (${POLICY_METADATA_KEYS[*]}) ✓"

# ── Invariant 4: report stdout has EXACTLY the documented ReportOutput keys ───
report_keys="$("$BIN" report "$OUT_DIR" --color never | jq -r 'keys | sort | join(",")')"
if [[ "$report_keys" != "$REPORT_KEYS_SORTED" ]]; then
  fail "\`cobre report\` top-level keys are {$report_keys}, but the book documents {$REPORT_KEYS_SORTED} (book/src/guide/cli-reference.md, cobre report)."
fi
echo "cobre report key set: {$report_keys} == documented ✓"

echo "All documented init/run/report structural invariants hold. ✓"
