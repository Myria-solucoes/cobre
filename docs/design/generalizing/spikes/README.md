# Verification spikes (2026-07-16)

Evidence harnesses behind `../feasibility-verification-2026-07.md`. Both are
session artifacts, kept so the report's measurements are regenerable — not
wired into the build or CI.

## monospike/ — D10 monomorphization spike

`gen_spike.py <outdir>` generates three synthetic Cargo workspaces (base /
value-template / type-level matrix) mirroring the proposed cobre-model
dispatch design; `measure.sh` builds each and records cold/incremental build
time, binary size, and total LLVM IR lines. Results snapshot in
`SPIKE_REPORT.md` + `results.txt`.

## mipdet/ — D3 HiGHS MIP determinism probe

`highs_det.c` links the vendored HiGHS static library (build it with cmake
from `crates/cobre-solver/vendor/HiGHS`) and solves seeded symmetric UC-family
MIPs repeatedly in fresh instances, comparing objective bit patterns, an
FNV-1a hash of the solution vector, and node counts.

```
gcc -O2 -o highs_det highs_det.c -I<highs-install>/include/highs \
    -L<highs-install>/lib -lhighs -lstdc++ -lm -lpthread -lz
./highs_det <runs> <threads> [stress] [node_cap] [case]
```

Result snapshots: `run_t1.log` (threads=1, 2,466-node tree, bit-identical
across runs), `run_final.log` (cross-process replication + threads=8 +
column-permutation probe). This harness is the prototype for the V.4 gate's
extension of the determinism reference suite to MIP.
