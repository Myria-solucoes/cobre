# Full-case profiling and progressive population — in progress

The converted round-1687 input was recovered from its archived evidence and copied
to `/home/jackson/tmp/cobre-optimization-next/real-1687` on the manager. It has
112 stages, 157 hydros, 158 thermals, five buses and six lines. Validation reported
zero errors and 494 warnings. The input CVaR (alpha 0.15, lambda 0.4) was retained;
historical outputs from before the CVaR correction are not the reference policy.

## Calibration, not a speed comparison

The opt.2 comparison binary completed two iterations with eight trajectories and
eight workers: 428.124 seconds training, 37,352 LP solves, no failed solves,
114 retries. Forward took 38.0 seconds and backward 387.9 seconds. The CLI attributed
about 30% of backward phase time to worker imbalance. These are diagnostic timings:
a 20-second perf collection and a separate four-CPU build overlapped this run.
No baseline comparison or policy evaluation was performed in this calibration.
The adjacent `v3-real-calibration-{runs,summary}.json` files record provenance and
counters. Zero backward load/bound-patch times do not mean zero work: the per-opening
statistics snapshots in `by_scenario.rs` start after loading and patching.

## Function profile

A separate `cargo build --profile profiling --bin cobre` build of the pre-ramp
source supplied function symbols. The isolated profiler container used Debian
`linux-perf`, `CAP_PERFMON`, and an unconfined seccomp profile; the host's global
`perf_event_paranoid` was not changed. The process ran the same 112-stage input
with two trajectories, two workers and two iterations, without simulation.
Sampling used `cpu-clock:u`, 49 Hz and DWARF call stacks. It recorded 26,189 samples
with zero reported lost samples. A subsequent compile overlapped this diagnostic
run; do not use its elapsed time as a speed benchmark.

Largest self CPU shares:

| Function | Share |
|---|---:|
| `solveHyper` | 19.74% |
| `HighsSparseMatrix::priceByRowWithSwitch` | 7.53% |
| `HighsSparseMatrix::priceByRowDenseResult` | 6.86% |
| `HEkkDualRow::choosePossible` | 4.88% |
| `HEkkDualRow::chooseMakepack` | 4.32% |
| `HVectorBase<double>::reIndex` | 4.27% |
| `HEkkDualRHS::chooseNormal` | 3.86% |

The profile supports prioritizing fewer LP solves, fewer simplex pivots through
basis reuse, and better work distribution. It does not establish a benefit from
any particular new basis strategy. Manager artifacts are `real-symbols.perf.data`
and `real-symbols.perf.txt` under the scratch directory above.

## Progressive trajectories

The working source adds optional `training.forward_schedule`. Population grows
geometrically, with a fixed maximum capacity and a full-population refinement
iteration. Its configuration crosses the CLI broadcast carrier and common study
setup, so CLI and Python share the same training implementation.

An initial numerical test exposed a basis-window mismatch: trajectory partitions
used the active population but basis partitions used maximum capacity. Both forward
and backward now partition the active prefix. The CVaR refinement test passes with
one and four workers, checks actual forward LP counts, and checks identical final
lower bounds across worker counts for the progressive mode. Rank-distribution and
stale-state exclusion unit tests pass. The CLI broadcast round-trip and checkpoint
resume before refinement also pass locally. The shape matrix also passes for
both schedulers, DCS on/off, uniform/nonuniform cut-state projections and one/four
workers, under CVaR. The complete `mpi_wire` integration binary passed all 42 tests with
`slow-tests,test-support` enabled; these include rank stubs, not real MPI
processes. Real multi-process MPI execution remains unverified.

Local macOS tests: 1,767 I/O unit tests passed; 2,343 core unit tests passed, one was
ignored and one FPHA-plane-generation assertion failed on a tiny negative volume
coefficient. The unchanged baseline at `10e07898` reproduces exactly the same
failure (`gamma_v = -7.538531311133364e-19` on plane 11) on this macOS toolchain;
this is not a clean full-suite result. Schema regeneration passed. Clippy completed with the same 14 existing warnings
(three I/O, ten core, one CLI), without a new warning from the progressive option. Real MPI execution, runtime release/integration,
full-case comparative benchmarks, and cross-point basis reuse remain pending.

The published experimental runtime remains opt.2. No stable installation, service,
merge or deploy was changed. Manager SSH requested fresh Tailscale authorization
after profiling; local implementation and verification continued while those
connections remained pending.

## Local paired performance and policy checks

Native macOS ARM64, four threads, same Rust 1.94 toolchain and release profile;
reference source `10e07898` (the previous optimized implementation, not the original
stable release). Fixture `examples/4ree`, 12 stages, CVaR alpha 0.15/lambda 0.4,
40 iterations, maximum 96 forward trajectories, LML1 selection every iteration,
256 common out-of-sample evaluation paths. The progressive population starts at
12, doubles every eight iterations and forces full population by iteration 27
(the geometric schedule reaches 96 at iteration 25). JSON files alongside this
report contain binary hashes, commands, LP counts and paired cost intervals.

| Experiment | Reference training | Progressive training | Reduction | Mean cost difference |
|---|---:|---:|---:|---:|
| Main, median of three repetitions | 16.895 s | 8.954 s | 47.0% | -0.0825% |
| Seed 7, one repetition | 19.077 s | 11.165 s | 41.5% | -0.0012% |
| Seed 91, one repetition | 18.316 s | 10.950 s | 40.2% | -0.0233% |

The main experiment reduces LP solves from 468,880 to 269,776. Its paired 95%
interval for mean cost difference is [-0.1711%, +0.0062%]; mean deficit changes
by -0.4625%. Terminal storage falls from 2,796.19 to 2,621.42 hm3: -6.25%
relative to the almost-empty terminal reservoirs, or -0.05994 percentage points
of the fixture's 291,579.1 hm3 total usable capacity. Both representations matter.
A more conservative 24-initial/full-by-17 schedule also kept mean cost close
(-0.0710%) but did not eliminate the terminal-storage difference (-5.41%). Its
single timing measurement is insufficient for a robust comparative speed claim.

Combining progressive population with selected backward points, deduplication
and audit reduced training by 49.0% for seed 7 and 41.1% for seed 91, each one
repetition. Paired mean-cost differences were -0.1103% and -0.0035%; the latter's
95% interval includes +0.0397%. These are additional experiments, not gains to
add to earlier percentages. They support proceeding to larger-case validation;
mean cost checks alone do not certify preservation of nested CVaR policy quality.
No full-case speedup, operational non-inferiority or released feature is claimed.

## Cross-point basis prototype

The working source adds optional `by_node.point_block_size`, default one. Groups
use a deterministic nearest-neighbor ordering of the full state normalized by
coordinate ranges, then a serpentine point/opening traversal. A group and each
successor child begin an independent solver chain. Within that chain frozen
solves retain basis/factorization and DCS retains its fully checked resident set.
Original point/opening indices still own the arena and canonical risk aggregation.
The initial implementation rejects multiple MPI ranks and enumerated training.

The expanded local `mpi_wire` suite passes all 43 tests, including a new DCS fan
comparison against the one-point reference and worker invariance. The progressive
shape matrix now covers both independent and three-point chains, with a partial
last group, DCS on/off and nonuniform state dimensions. Postcard carries the new
parameter; its round-trip test passes. Schema regeneration succeeds and Clippy
reports the same 14 existing warnings, no additional warning.

A separate three-repeat 4REE/CVaR comparison completed with fixed 96 forwards,
40 iterations, four workers, five-opening blocks, 256 common out-of-sample paths:
reference independent-point chains versus candidate four-point chains. Median training fell from 25.292 to 23.185 seconds (8.33%), but ranges overlap:
reference 22.162–25.437 s, candidate 22.134–23.247 s. Backward pivots fell from
972,809 to 641,457 (-34.1%); backward solve time barely changed in the first
repetition. Both arms solved 468,880 LPs and generated 42,240 cuts. Mean evaluation
cost changed -0.0817%, with paired 95% interval [-0.1614%, -0.0020%]; terminal
storage changed from 2,686.04 to 2,638.09 hm3. These results support further
experiments, not a robust end-to-end speedup claim against the faster by-scenario
scheduler. Binary SHA-256 values:

- Reference: `26bc734436794ccd780831eb4f628d363553482a31cd60a80b0761c79d378671`.
- Candidate: `7e2421302be6949d39c8a782db0b2b90fe8bed3fe760b8722f3259059345e524`.

The reference includes progressive-population support but the schedule is absent
in both arms, isolating cross-point reuse. No stable runtime has been changed.

### Comparison against by-scenario

A second three-repeat comparison used the by-scenario reference versus two-point
chains with ten-opening blocks. Both arms kept 96 forward points and 40 iterations;
all other settings and common evaluation paths were unchanged. Training medians
were 23.127 → 21.384 seconds (-7.54%). Each paired repetition improved, although
this remains a small local-machine sample. Backward pivots fell 946,471 → 738,413
(-22.0%), and first-repeat cumulative backward solve time 66.023 → 59.675 seconds.
Both arms solved 468,880 LPs and generated 42,240 cuts.

Mean simulated cost changed -0.1903%, paired 95% interval [-0.2688%, -0.1118%].
Mean deficit changed 6,688,324.58 → 6,609,650.99 MWh and terminal storage
2,796.19 → 2,624.61 hm3. Lower mean cost and lower storage do not certify nested
risk non-inferiority. Keep the setting experimental and validate larger cases and
multiple training seeds before recommendation. The nearby-state unit test also
passes, checking normalization under changed coordinate units, deterministic
ordering of duplicate states, canonical positions and buffer-capacity reuse.

The full 112-stage converted round-1687 case is now under a separate local short
comparison: two iterations/eight forwards, by-scenario four-worker reference,
eight-worker candidate, then eight-worker complete-state deduplication. Simulation
is disabled; this is a timing/solver-work diagnostic, not a trained-policy quality
gate. The original risk/input files are retained. Manager SSH is still pending.

The by-scenario comparison's terminal storage difference is concentrated in hydro
ID 2: mean 2,088.408 → 1,916.823 hm3, or -0.3312 percentage points of that hydro's
51,806.1 hm3 usable capacity. Hydro IDs 0, 1 and 3 have unchanged mean terminal
storage. See `v3-local-point-tiles-terminal-by-hydro.json`. This per-hydro result
is more informative than diluting the difference across total system capacity;
no operational acceptance margin has been inferred from it.

## Full-case local short comparison

The serial native macOS comparison completed on the original 112-stage round-1687
converted case, preserving the input risk settings. It uses only two iterations
and eight forward trajectories; one timing per arm, no simulation. All arms use
by-scenario scheduling and LML1 selection each iteration.

| Arm | Threads | Training | LPs | Generated cuts | Final LB |
|---|---:|---:|---:|---:|---:|
| Reference | 4 | 504.874 s | 37,352 | 1,776 | 87,794,918,239.24646 |
| Candidate | 8 | 310.796 s | 37,352 | 1,776 | 87,794,918,239.24646 |
| Candidate, full-state deduplication | 8 | 308.017 s | 37,352 | 1,776 | 87,794,918,239.24646 |

Eight workers reduced training by 38.44% in this single local comparison.
Deduplication removed no solves or cuts; its additional 0.89% timing difference
is not persuasive evidence of a benefit. No solve failed; all arms retried 130
solves. All 112 exported cut files are byte-identical across the three arms; their
SHA-256 hashes are recorded separately. This is a short training
throughput measurement, not evidence about converged-policy quality or a server
speedup. It does not extrapolate the compact 47% progressive-population gain.

The deduplicated backward phase attributed 28% of its time to worker wait. A
follow-up keeps eight workers and the same complete set of points/openings, using
ten-opening/two-point chains to investigate work distribution and basis reuse.
That follow-up completed; its regression is recorded below.

### Full-case cross-point regression

Ten-opening/two-point chains with eight workers took 412.284 seconds, versus
310.796 seconds for eight-worker by-scenario (+32.65%). The LP/cut counts remained
37,352/1,776 and no solve failed. Final lower bound was 87,428,504,896.98741,
versus 87,794,918,239.24646 (-0.4173%); after just two iterations this is not a
comparison of converged policies. Do not enable this grouped setting for round
1687 on the strength of the compact-case result. It remains opt-in, default off.

The grouped backward phase took 386 seconds and attributed 24% to worker wait,
versus roughly 28% in the prior deduplicated pass. A lower wait fraction did not
compensate for more solver work: the grouped run performed 26,277,601 backward
simplex iterations versus 18,911,125 by-scenario (+38.95%). The public preview retains the capability for further study,
not a universal speed recommendation.

## Published opt.3 package checks

The immutable prerelease targets `3e3bdbc4081c247611465f8bd849a75a5de31bff`.
Both x86_64 and ARM64 CLI/wheel builds succeeded. All four downloaded files match
their GitHub asset SHA-256 digests; latest remains `v0.15.0-myria.4`.
Myria's actual `cobre-worker-base` Docker stage built locally on Linux ARM64 with
the updated opt.3 hash lock. The stable wheel accepts the plain case and rejects
experimental fields; the isolated wheel trains four iterations with progressive
population, point grouping and audited selection, with LML1 and DCS. The stable
package files remain unchanged.

The real ARM64 installer succeeded twice in a disposable Ubuntu 24.04 container:
second installation idempotent, stable and prior-opt.2 sentinel hashes unchanged,
and the installed CLI trained the combined fixture. The reusable
`scripts/benchmarks/verify_runtime.py` also passed natively on ARM64, checking
CLI/Python bound agreement, Python one/two-worker invariance, and invalid schedule
rejection for both cut-selection methods. These are smoke tests, not policy
non-inferiority tests.

An initial local x86_64 emulation check failed with illegal instruction. A control
with the published opt.2 CLI failed identically (exit 132); this VirtualApple CPU
exposes no AVX2/FMA, which `.cargo/config.toml` requests for both builds. Native
x86_64 package execution required the separate check reported below. The release workflow
now has a read-only `validate_only` mode for existing optimized releases, using
native x86_64/ARM64 runners, digest verification and the reusable artifact check;
it does not publish or replace release assets.

Native artifact verification passed on both Linux architectures in
[run 35546083787](https://github.com/Myria-solucoes/cobre/actions/runs/35546083787).
The publishing job was skipped. The initial verification run exposed a probe
construction mistake: Python overrides replace the full training section, so the
invalid-schedule probe initially removed selection. The corrected probe preserves
training and changes only the initial population; it now verifies the intended
`forward_schedule` error. No runtime artifact was changed.

## Aggressive combined configuration: quality tradeoff

The combined experiment uses eight workers, 12 initial forwards doubling every
eight iterations with full population by 27, audited/deduplicated backward
selection and two-point/ten-opening warm chains. Reference is four-worker
by-scenario with all 96 points for all 40 iterations. CVaR remains alpha 0.15,
lambda 0.4, with 1,024 common out-of-sample evaluation paths per trained pair.
Seed 42 has three timing repetitions; the other seeds have one each.

| Training seed | Reference | Combined | Reduction | Paired mean cost change | Mean deficit change |
|---|---:|---:|---:|---:|---:|
| 42 | 17.472 s | 5.624 s | 67.81% | +0.1112% | +2.1244% |
| 7 | 19.751 s | 7.452 s | 62.27% | -0.0180% | +0.2188% |
| 91 | 19.322 s | 6.673 s | 65.46% | +0.0061% | +0.0825% |

Seed 42's paired cost interval is [+0.0483%, +0.1742%], excluding zero; its mean
terminal storage changes 2,820.218 → 2,704.567 hm3. The speedup is real in this
compact experiment, but the observed cost/deficit tradeoff prevents describing
this combination as preserving quality. Mean cost is not the nested CVaR
objective, and no nested-risk superiority/inferiority claim is made. Do not adopt
this aggressive combination solely for its time reduction. A more conservative
ramp (24 initial, full by 17), all backward points and independent chains is being
compared next at eight workers, using the same evaluation protocol.

## Conservative combined configuration

The follow-up retains every backward point and independent by-scenario chains,
uses eight workers, and starts with 24 forwards, doubling every eight iterations
and reaching 96 by iteration 17. The reference uses four workers and 96 forwards
throughout. All other protocol settings match the aggressive comparison above,
including 1,024 common out-of-sample paths; seed 42 has three timing repetitions,
seeds 7 and 91 one each. Training performs 351,760 LPs versus 468,880 (-24.98%).

| Training seed | Reference | Conservative | Reduction | Paired mean cost change | Mean deficit change |
|---|---:|---:|---:|---:|---:|
| 42 | 18.426 s | 10.272 s | 44.25% | -0.0175% | -0.0202% |
| 7 | 21.202 s | 10.910 s | 48.54% | -0.0264% | +0.3946% |
| 91 | 19.854 s | 10.122 s | 49.02% | -0.0016% | +0.0465% |

Paired mean-cost 95% intervals are respectively [-0.0617%, +0.0268%],
[-0.0459%, -0.0068%], and [-0.0150%, +0.0119%]. These intervals are conditional
on each trained pair and do not include training uncertainty. Terminal storage
changes 2,820.218 → 2,703.666 hm3, 2,911.040 → 2,931.077 hm3, and
2,959.577 → 2,965.520 hm3. The first seed therefore still ends with less water;
seed 7 has more deficit despite lower mean cost. This is a more promising speed/
quality tradeoff than the aggressive combination, but it is not a certificate
of operational or nested-CVaR non-inferiority. Retain fixed population as the
default and validate risk and individual-reservoir trajectories before promotion.
These are local measurements; manager validation remains pending SSH access.
Evidence: `v3-conservative-combined-seed-*-{runs,summary}.json`.


## Paired deficit follow-up

For the local conservative three-seed comparison, summing deficit per common
scenario before computing paired differences gives 95% normal intervals of
[-0.3187%, +0.2783%], [+0.2664%, +0.5227%], and [-0.0751%, +0.1680%],
respectively. The seed-7 increase is therefore not merely an unquantified point
estimate; its paired interval excludes zero. These intervals are conditional on
the trained policies, without adjustment for multiple comparisons, and are not
a nested-risk test. Keep the conservative ramp experimental. The adjacent
`v3-conservative-paired-deficit.json` preserves scenario counts and MWh intervals.
