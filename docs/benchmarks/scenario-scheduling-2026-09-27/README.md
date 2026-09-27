# Whole-point backward scheduling — 2026-09-27

Baseline: `v0.16.0-myria.3`, commit
`64e40cb82f0d4e543b2192856413e0c402783345`. Candidate changes only
`by_scenario` task ownership: free workers claim another complete point, with
exclusive scenario bases and coefficient slots. Child resets, opening chains,
point population, risk aggregation and final cut order remain unchanged.
Initial scheduler measurements used `66bfb2fa`; build experiments use
`d74ed075`, which explicitly destructures the same borrowed buffers for Clippy
and renames one test variable. The paired
[refactor check](scheduler-refactor-parity.json) confirmed exact numerical parity.
The final candidate, `3031c188`, also uses an allocation-free sort for the unique
trial indices; the final confirmation below measures that source.

## Compact comparison

Both binaries were built with Rust 1.94.1, GCC in Debian Bookworm, vendored
HiGHS 1.13.1, and `cargo build --locked --release -p cobre-cli -j12`.
The host is an Intel Xeon E5-2680 v4 with 28 physical cores, SMT disabled,
two NUMA nodes and approximately 121 GiB RAM. Runs used eight workers,
`--comm-backend local --cpu-bind none`, with no concurrent builds or benchmarks.

Three matched epochs used a seeded shuffled arm order. The case is
`examples/4ree` with the adjacent `compact-config.json` and
`compact-stages.json`: 40 iterations, 96 forward trajectories, ten openings,
CVaR alpha 0.15/lambda 0.4, LML1 every iteration, and 16 common out-of-sample
simulation scenarios. All other example files are unchanged.

| Metric | Baseline | Dynamic claims |
|---|---:|---:|
| Median training time | 21.039 s | 20.494 s |
| Training range | 20.848–21.067 s | 20.448–20.704 s |
| Median process wall time | 21.274 s | 20.677 s |
| Median backward wall time | 16.791 s | 16.224 s |
| Median backward imbalance estimate | 0.802 s | 0.300 s |
| Median peak RSS | 124,988 KiB | 126,444 KiB |
| LP solves | 468,880 | 468,880 |
| LP solves requiring retries | 318 | 318 |
| Terminal LP failures | 0 | 0 |

The training median decreased 2.59% on this host/case. This is a modest,
case-specific result; it is not evidence of the same gain on a different CPU,
deck, scheduler or worker count. More workers than whole-point chains cannot
increase chain concurrency. The `by_node` path is unchanged.
The imbalance estimate sums the per-stage difference between the slowest
worker's measured solver/setup time and the worker average; it is not a CPU
utilization counter or the sum of idle time across all workers.

All six runs have identical cut and basis bytes, convergence values, simulation
results, training solver work totals and simulation solver counters. Numerical
comparison excludes manifest timestamps and explicitly named timing columns.
The initial comparator accidentally included four simulation solver timing
columns; it was corrected, regression-tested and rerun on the same outputs.
No simulation values or solver counters were excluded by that correction.

Raw commands, input/binary hashes, per-run timings and numerical signatures are
in [compact-results.json](compact-results.json). These local binaries are not
published release binaries. Timing repetitions use the same seed; they are not
independent policy-quality observations and do not certify convergence.

## Final scheduler confirmation

Three additional matched epochs compared `.3` with the final candidate on the
same compact `by_scenario` case. Baseline median: 20.885 s (20.675–21.085 s).
Candidate median: 20.596 s (20.547–20.754 s), a 1.38% reduction. Paired reductions
were 0.62%, 2.32% and 0.63%. All six numerical signatures match.
[scheduler-confirmation.json](scheduler-confirmation.json) records the source
commits, binary hashes and complete results.

The two compact series support a modest, case-specific scheduling benefit,
not a broad training-speed claim. The scheduling change is retained for Myria
`.4`; native CLI/wheel publication validation remains a release gate.

## Reproduction

Build the baseline and candidate separately, then prepare the compact case:

```bash
cp -a examples/4ree /tmp/cobre-scenario-case
cp docs/benchmarks/scenario-scheduling-2026-09-27/compact-config.json /tmp/cobre-scenario-case/config.json
cp docs/benchmarks/scenario-scheduling-2026-09-27/compact-stages.json /tmp/cobre-scenario-case/stages.json
```

Create `arms.json` with absolute paths to the two binaries:

```json
[
  {"name":"baseline", "binary":"/path/to/baseline", "case":"/tmp/cobre-scenario-case", "threads":8},
  {"name":"dynamic", "binary":"/path/to/candidate", "case":"/tmp/cobre-scenario-case", "threads":8}
]
```

With PyArrow installed, use a fresh output directory:

```bash
python scripts/benchmarks/compare_execution.py arms.json --output /tmp/cobre-scenario-results --repeats 3
python scripts/benchmarks/test_compare_execution.py
```

## Correctness checks

The candidate passed 2,462 SDDP unit tests in the normal test profile,
41 `mpi_wire` integration tests in release, 21 CLI/checkpoint tests and the
CLI affinity parity test. The new integration test compares every cut,
active mask, bounds and solve counts with 1/3/8 workers, repeated scheduling,
sparse progressive selection, and dynamic/frozen cut pools.

An initial release-profile unit run had 14 existing `should_panic` tests fail
because they rely on `debug_assert!`. The full unit suite was rerun in its
normal profile; no assertion or test was weakened. The native release workflow
is configured to run the new scheduling integration test on x86_64 and aarch64.

## Real-deck follow-up

One paired full-deck trial used the converted `teko-152` case (111 stages,
157 hydros), two iterations and 16 trajectories with eight workers. Training
was 725.141 s baseline and 718.685 s candidate (0.89% lower); backward time
was 646.525 s and 640.215 s, with imbalance estimates of 146.537 s and 137.786 s.
Both executed 73,992 LPs, with 372 retries, zero terminal failures and exactly
equal numerical signatures. See [real-pilot-results.json](real-pilot-results.json);
the complete private-case output remains at the local path recorded there.

The single-pair timing difference is too small to establish a real-deck speedup.
This trial validates numerical parity at the larger scale; two initial iterations
do not characterize performance near convergence. It used the initial candidate,
before the explicit-borrow and allocation-free-sort refinements measured in the
final compact confirmation. Original cases are untouched.

## Affinity comparison

The baseline binary also ran three matched epochs of the same compact case
with `by_node`, block size 10 and eight workers. The only changed command option
was `--cpu-bind`; [affinity-results.json](affinity-results.json) retains each run
and [affinity-config.json](affinity-config.json) is the case configuration overlay.

| Binding | Median training | Range |
|---|---:|---:|
| `none` | 20.319 s | 20.077–20.405 s |
| `core` | 20.282 s | 20.155–20.314 s |
| `numa` | 20.043 s | 19.901–20.082 s |

All nine numerical signatures match. The 1.36% median difference for `numa`
is small and specific to this two-socket host with one model running. It does
not establish a fleet-wide default; concurrent runs need disjoint CPU sets
before a fixed placement policy can avoid colliding on the same cores.
No runner default was changed.

## Build experiments

Three matched epochs used the compact `by_node` case above, with eight workers
and `--cpu-bind none`. The source and toolchain were identical; only the build
setting differed. [Build provenance](build-provenance.json) records the binary
hashes and [build-results.json](build-results.json) records the runs.

| Build | Median training | Range |
|---|---:|---:|
| Existing `release` | 20.577 s | 20.170–20.897 s |
| HiGHS C++ AVX2 | 20.611 s | 20.459–20.863 s |
| Existing `dist` / ThinLTO | 20.535 s | 20.419–20.648 s |

All nine numerical signatures match. The overlapping ranges show no convincing
improvement in this case. A paired real `by_node` comparison (two iterations,
eight trajectories, eight workers) also retained exact numerical parity but
took 387.066 s with AVX2 versus 381.837 s with the existing build, 1.37% longer.
See [real-build-results.json](real-build-results.json). Neither build experiment
was retained; defaults remain unchanged.

The AVX2 experiment used GCC 12.2.0 and temporary existing compiler settings,
not a new runtime configuration:

```bash
CXXFLAGS='-mavx2 -msse4.2 -ffp-contract=off' CARGO_TARGET_DIR=/tmp/cobre-avx2-target \
  cargo build --locked --release -p cobre-cli -j12
cargo build --locked --profile dist -p cobre-cli -j12
```

The existing Rust x86_64 Linux target already requires AVX2. Floating-point
contraction was disabled for the C++ experiment to avoid introducing fused
multiply-add rounding. This did not bypass the numerical comparison gate.
No timing run overlapped a build or another benchmark started by this task;
the host remains a desktop environment with background services.
