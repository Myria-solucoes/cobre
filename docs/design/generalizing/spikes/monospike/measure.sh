#!/usr/bin/env bash
# D10 spike measurement protocol. Run from monospike/.
set -u
cd "$(dirname "$0")/work"
RESULTS=../results.txt
: > "$RESULTS"

for v in base a b; do
  echo "=== variant $v ===" | tee -a "$RESULTS"
  cd "$v"
  # generated source volume (model + engines)
  SRC_LINES=$(cat spike-model/src/lib.rs spike-direct/src/lib.rs spike-sddp/src/lib.rs 2>/dev/null | wc -l)
  echo "model+engine source lines: $SRC_LINES" | tee -a "../$RESULTS"

  # cold release build
  cargo clean -q
  T0=$(date +%s.%N)
  cargo build --release -q 2>/dev/null
  T1=$(date +%s.%N)
  echo "cold release build: $(echo "$T1 - $T0" | bc)s" | tee -a "../$RESULTS"

  # binary size (profile already strips symbols)
  BIN=target/release/spike-cli
  echo "binary size: $(stat -c%s $BIN) bytes" | tee -a "../$RESULTS"

  # incremental: touch model, rebuild
  touch spike-model/src/lib.rs
  T0=$(date +%s.%N)
  cargo build --release -q 2>/dev/null
  T1=$(date +%s.%N)
  echo "incremental (touch model): $(echo "$T1 - $T0" | bc)s" | tee -a "../$RESULTS"

  # total LLVM IR lines across workspace (fresh, single codegen pass)
  cargo clean -q
  RUSTFLAGS="--emit=llvm-ir" cargo build --release -q 2>/dev/null
  IR=$(cat $(find target/release/deps -name '*.ll') | wc -l)
  echo "total LLVM IR lines: $IR" | tee -a "../$RESULTS"

  # llvm-lines on the leaf binary (captures monomorphizations codegenned there)
  for crate in spike-cli spike-direct spike-sddp; do
    if [ -d "$crate" ]; then
      LL=$(cargo llvm-lines --release -p $crate 2>/dev/null | head -2 | tail -1 | awk '{print $1}')
      echo "llvm-lines $crate total: $LL" | tee -a "../$RESULTS"
    fi
  done
  cd ..
  echo | tee -a "$RESULTS"
done
echo "done"
