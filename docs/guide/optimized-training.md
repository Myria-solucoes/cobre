# Optimized training runtime

The optimized runtime is a separate build. Install it in a version-specific
location and select it explicitly; do not replace an existing runtime executable
or its current-version manifest. Keep input and output directories separate when
comparing policies.

## Exact cut selection

Dynamic cut selection retains its existing configuration surface. Candidate
scoring uses the pool's coefficient matrix directly when most rows are candidates,
and gathers a compact matrix otherwise. Only the selected prefix of violated
rows is sorted, with the same violation/slot ordering. At the inner-loop limit,
the fallback loads every eligible missing row before solving: a row that is not
violated at the previous solution may bind after reoptimization.

```json
{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 24 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 12 }],
    "cut_selection": {
      "selection": {
        "method": "dynamic",
        "candidate_recency": null,
        "seed_window": 2,
        "max_added_per_round": 20
      }
    }
  }
}
```

A finite `candidate_recency` excludes older rows and remains an approximation.
Dynamic selection supports both schedulers. Each by-node block starts with its
own resident set and retains canonical opening aggregation. Smaller resident LPs can require more solves, so measure
the whole pass rather than assuming that enabling dynamic selection helps.

## Experimental point budget

`training.backward_selection` is opt-in. Its absence processes every visited
point as before. The option supports sampled training with one MPI rank and
multiple worker threads. Enumerated training and multiple ranks are rejected.

```json
{
  "training": {
    "selection": { "method": "sampled", "forward_passes": 24 },
    "stopping_rules": [{ "type": "iteration_limit", "limit": 12 }],
    "backward_selection": {
      "initial_points": 6,
      "exploration_points": 2,
      "full_every": 4,
      "full_from_iteration": 8
    },
    "cut_selection": {
      "selection": { "method": "lml1", "check_frequency": 1 }
    }
  }
}
```

All four point-budget fields are positive integers. The diverse-point budget
increases linearly from `initial_points` toward full coverage at
`full_from_iteration`, separately at each node. Deterministic farthest-point
selection uses the complete state vector, with each coordinate normalized by
its observed range. `exploration_points` adds pseudo-random remaining states;
its ordering depends on iteration and original scenario identity, not worker
scheduling. Every `full_every` iterations, and from `full_from_iteration` onward,
every point is processed. The refinement iteration must fit within the configured
iteration limit.

Every selected point still integrates all successor openings with the configured
risk measure. Forward trajectories and their statistics are not subsampled.
Optional `deduplicate: true` collapses bit-identical complete states within a
node and iteration, preserving the smallest scenario identity. Full passes cover
every distinct state when this option is enabled.

Optional `audit_relative_tolerance` (finite, nonnegative) compares exploration
cuts with the envelope excluding this iteration's exploration cuts, at the probed
states. Improvement above the tolerance doubles that node's minimum point budget,
capped by its available distinct states. The denominator is `max(abs(probe_value),
1)`. All exploration cuts remain in the policy. This local diagnostic does not
certify policy quality or bound improvement elsewhere. Budget floors grow only;
they are not checkpointed, so an audited run resumed after iteration 1 processes
all points conservatively.

Periodic complete passes provide additional coverage; there is no automatic
non-inferiority certificate. Time limits, shutdown, or another stopping rule can interrupt before
refinement. Inspect completion before treating a policy as refined.

Unused cut slots stay unoccupied, are excluded from row selection, and are not
exported as artificial zero-valued cuts. Checkpoints retain the generated active
and inactive cuts. Basis export falls back to no cached basis for sparse pools,
since policy reload compacts slot identities. Resume therefore preserves valid
cuts but can follow a different numerical trajectory.

This option changes the training trajectory and is experimental. Compare policy
quality out of sample on common scenarios and multiple training seeds. Include
cost, deficit, storage and an evaluation compatible with nested conditional risk;
the CVaR of total simulated cost alone is not that nested objective. A stable lower
bound is not a quality certificate.

## Reproducible compact comparisons

Run benchmarks serially on an otherwise idle machine. Both executables should
come from the same toolchain, solver and build profile. The harness copies the
case, freezes configuration, uses a separate simulation seed, alternates arm
order, applies a timeout and records binary hashes, commands, exit codes and wall
time. It never edits the source case or an installed runtime.

```sh
python3 scripts/benchmarks/training.py \
  --baseline /absolute/path/to/cobre-baseline \
  --candidate /absolute/path/to/cobre-candidate \
  --case examples/4ree \
  --output /absolute/path/to/new-comparison-directory \
  --iterations 12 --forwards 24 --threads 4 --repeats 3
```

The default comparison uses LML1 for baseline, candidate with exhaustive points,
and candidate with experimental point selection. Inspect the simulation and
training outputs as well as `results.json`; short runs establish regressions and
measurement plumbing, not production-scale non-inferiority. Scheduler/profile
sweeps and dynamic-selection comparisons should use matched case configurations.


## Preview artifacts

The `Release Myria runtime` workflow accepts `prerelease=true` when publishing an
opt-in runtime. It creates a prerelease without changing GitHub's latest release.
The runtime tag must be new. Keep the platform's recommended runtime unchanged
when registering the preview's CLI and Python artifacts.

For independent training repetitions, change `--seed` (timing `--repeats` keeps
that seed fixed). Use `--simulation-scheme out_of_sample` to generate inflows
outside the training opening set. Keep `--simulation-seed` fixed across arms.
`--scheduler by_node --block-size 10` supports matched scheduler experiments;
DCS runs directly in either scheduler.

With `pyarrow` available, summarize an output directory with
`python3 scripts/benchmarks/summarize.py /absolute/path/to/comparison`.
The paired cost interval is conditional on one pair of trained policies; it does
not cover variation between training seeds or certify nested CVaR quality.

## Further preview optimizations

Dynamic selection accepts `adaptive_max_added_per_round`. When provided, it must
be at least `max_added_per_round`: the initial batch doubles after unsuccessful
separation rounds up to that ceiling. Candidate eligibility, violation tolerance
and the complete eligible-row fallback are unchanged. Omit it for fixed batches.

Frozen backward LPs reuse their matrix when consecutive work units on one worker
use the same child within one node dispatch. HiGHS clears basis, factorization and
pricing history before each independent unit, then all state/noise bounds are
patched. The cache is invalidated at every dispatch and child change. DCS retains
its own resident-set lifecycle. Backends without a model-preserving cold reset
reload normally. This does not reuse an arbitrary prior point's simplex basis.

Harness flags `--deduplicate`, `--audit-relative-tolerance 0.01` and
`--adaptive-max-added-per-round 80` enable these optional controls. Record each
configuration separately: a more conservative audit can sacrifice speed to
increase coverage.
