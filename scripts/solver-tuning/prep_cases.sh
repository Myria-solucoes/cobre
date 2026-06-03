#!/usr/bin/env bash
#
# Prepare per-cut-selection-method case copies for the tuning sweep.
#
# Each grid cell runs a full `cobre run <case>`; cobre always reads
# <case>/config.json (no --config flag), so config-varying knobs (cut selection)
# require separate case directories. Warm-start and solver params ride on
# COBRE_TUNE_* env vars and need no case copy.
#
# Usage:
#   prep_cases.sh <BASE_CASE_DIR> <CASES_OUT_DIR> [method ...]
# Default methods: none level1 dominated
#
# Existing method dirs are left untouched (remove manually to rebuild) — this
# script never deletes.

set -euo pipefail

readonly HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE="${1:?usage: prep_cases.sh <BASE_CASE_DIR> <CASES_OUT_DIR> [method ...]}"
readonly OUT="${2:?usage: prep_cases.sh <BASE_CASE_DIR> <CASES_OUT_DIR> [method ...]}"
shift 2 || true
methods=("$@")
[[ ${#methods[@]} -eq 0 ]] && methods=(none level1 dominated)

command -v python3 >/dev/null 2>&1 || { echo "python3 is required" >&2; exit 1; }
[[ -f "$BASE/config.json" ]] || { echo "no config.json in base case: $BASE" >&2; exit 1; }

mkdir -p "$OUT"
for method in "${methods[@]}"; do
    dst="$OUT/$method"
    if [[ -e "$dst" ]]; then
        echo "[skip] $dst already exists (remove it to rebuild)"
        continue
    fi
    echo "[copy] $BASE -> $dst"
    cp -r "$BASE" "$dst"
    python3 "$HERE/patch_config.py" "$dst/config.json" --method "$method"
done

echo "Done. Case copies under: $OUT"
