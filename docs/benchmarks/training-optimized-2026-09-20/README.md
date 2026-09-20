# Compact optimized-training benchmarks — 2026-09-20

Baseline: `v0.15.0-myria.4` (`289ec383`). Candidate: `b244a3ee`.
Both locally benchmarked binaries were built in the same Rust 1.94 / Debian
Bookworm container with HiGHS 1.13.1 and the release profile. Published release
artifacts use their release workflow toolchains and therefore have different
binary hashes. CLI version alone does not distinguish these runtimes.

Runs used `examples/4ree` (12 stages, 4 hydros, 126 thermals), on
`myria-manager`, serially, with 4 solver threads, an 8-CPU / 20-GiB container
limit, and a 300-second timeout per run. The host also runs Myria services;
small timing differences are not evidence of a speedup. Simulation used 256
common scenarios within each matched comparison. Timing repeats keep the same
seed and are not independent policy-quality observations.

The selected arm starts at one quarter of the forward points plus two exploration
points, processes all points every fourth iteration, and switches to full coverage
at two thirds of the iteration limit. It uses the same forward count, successor
openings and risk parameters. CVaR cases use alpha=0.15, lambda=0.4.

| Case: iterations × forwards | Repeats | Baseline training (s) | Selected training (s) | Reduction | Paired mean cost change | Mean deficit change |
|---|---:|---:|---:|---:|---:|---:|
| 12 × 24, expectation, LML1 | 3 | 1.994 | 2.060 | -3.31% | -0.0576% | +0.0233% |
| 40 × 96, CVaR, LML1 | 3 | 39.894 | 33.343 | 16.42% | -0.0198% | -0.0240% |
| 24 × 48, CVaR, DCS | 3 | 21.544 | 19.292 | 10.45% | +0.0027% | +0.0120% |
| 30 × 64, CVaR, LML1, tree seed 7, out of sample | 1 | 17.988 | 16.321 | 9.27% | -0.0118% | -0.2606% |
| 30 × 64, CVaR, LML1, tree seed 91, out of sample | 1 | 17.345 | 14.907 | 14.06% | -0.0102% | -0.0783% |
| 24 × 48, CVaR, by-node / block 5 | 1 | 9.457 | 7.625 | 19.37% | -0.0461% | +0.0295% |

Times are medians where repeats > 1. End-to-end times, solve counts, lower
bounds, conditional paired-cost intervals and terminal storage are in the
adjacent JSON summaries. Raw commands and binary hashes are in `*-runs.json`.
Remote raw cases, outputs and logs are retained under
`/home/jackson/tmp/cobre-optimized-20260920/benchmark-*`.

## Interpretation and limits

- On the main CVaR/LML1 comparison, training fell from 39.894 s to 33.343 s
  (16.42%). End-to-end time fell from 40.864 s to 34.350 s (15.94%). LP solves
  fell from 468,880 to 391,880. Terminal stored volume changed by -2.08%.
- On two independent training-tree seeds, with out-of-sample inflows, training
  fell by approximately 9–14%; mean cost changed by about -0.01%. Conditional
  paired-cost 95% interval upper endpoints were +0.0152% and +0.0413%. These
  intervals cover simulation sampling for the given policy pair, not variation
  over training seeds or other decks.
- Exact DCS changes alone had identical reported bounds and simulated costs,
  and no measurable speed benefit in these small cases (21.544 s baseline vs
  21.634 s candidate). The fallback regression demonstrates a correctness fix;
  it does not demonstrate that real runs hit the old 50-round fallback.
- The tiny expectation case became slightly slower despite fewer solves.
  Selection has overhead and is not universally faster.
- Deficit sometimes increased slightly, and storage/dispatch can change even
  when cost improves. These runs do **not** certify non-inferiority on every
  relevant operational metric. No nested conditional risk evaluation was
  performed on these simulated policies. Mean or total-cost-tail comparisons
  cannot substitute for that evaluation.
- The by-node result is a single compact comparison, not a scheduler tuning
  conclusion. DCS is not combined with by-node in this implementation.

The point budget remains opt-in and experimental. Before promoting a default,
validate representative dry/wet/constrained decks, more independent training
seeds, operational deficit/storage margins and nested risk. Do not interpret
lower-bound stability or these timings as a convergence-quality certificate.

## Reproduction

Use `scripts/benchmarks/training.py` and `summarize.py`. Each case uses the
iteration/forward count and cut method in the table. Add `--cvar` for risk cases,
`--seed 7` or `--seed 91 --simulation-scheme out_of_sample` for the independent
seed cases (both seed cases use out-of-sample simulation), and
`--scheduler by_node --block-size 5` for the node case. Simulation seed is 8675309.
The harness pins both `training.tree_seed` and `training.scenario_source.seed`.

Validation included 2,338 SDDP unit tests, 1,766 I/O unit tests, 41 existing
integration/oracle tests, a new selected-point CVaR refinement/thread regression,
and two CLI checkpoint/resume tests. Three pre-existing ignored tests were not
run. See the implementation guide for the single-rank and sparse-basis limits.
