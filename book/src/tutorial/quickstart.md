# Quickstart

This page takes you from zero to a completed SDDP study in three commands using the
built-in `1dtoy` template. The template models a single-bus hydrothermal system with
one hydro plant and two thermal units over a 4-stage finite planning horizon — small
enough to run in seconds, complete enough to demonstrate every stage of the workflow.

If you have not installed Cobre yet, start with [Installation](../guide/installation.md).

![Quick Start Demo](../images/quickstart.gif)

---

## Step 1: Scaffold a Case Directory

```bash
cobre init --template 1dtoy my_first_study
```

Cobre writes 11 input files into a new `my_first_study/` directory and prints a
summary to stderr:

```
 ━━━━━━━━━━━●
 ━━━━━━━━━━━●⚡  COBRE v0.9.1
 ━━━━━━━━━━━●   Power systems in Rust

Created my_first_study case directory from template '1dtoy':

  ✔ config.json                    Algorithm configuration: training (forward passes, stopping rules) and simulation settings
  ✔ initial_conditions.json        Initial reservoir storage volumes for each hydro plant at the start of the planning horizon
  ✔ penalties.json                 Global penalty costs for constraint violations (deficit, excess, spillage, storage bounds, etc.)
  ✔ stages.json                    Planning horizon definition: policy graph type, discount rate, stage dates, time blocks, and scenario counts
  ✔ system/buses.json              Electrical bus definitions with deficit cost segments
  ✔ system/hydros.json             Hydro plant definitions: reservoir bounds, outflow limits, turbine model, and generation limits
  ✔ system/hydro_production_models.json  Per-(hydro, stage) production-model configuration carrying the productivity coefficient
  ✔ system/lines.json              Transmission line definitions (empty in this single-bus example)
  ✔ system/thermals.json           Thermal plant definitions with piecewise cost segments and generation bounds
  ✔ scenarios/inflow_seasonal_stats.parquet  Seasonal PAR(p) statistics for hydro inflow scenario generation (mean, std, lag correlations)
  ✔ scenarios/load_seasonal_stats.parquet    Seasonal PAR(p) statistics for electrical load scenario generation (mean, std, lag correlations)

Next steps:
  -> cobre validate my_first_study
  -> cobre run my_first_study --output my_first_study/results
```

The directory structure is:

```
my_first_study/
  config.json
  initial_conditions.json
  penalties.json
  stages.json
  system/
    buses.json
    hydros.json
    hydro_production_models.json
    lines.json
    thermals.json
  scenarios/
    inflow_seasonal_stats.parquet
    load_seasonal_stats.parquet
```

---

## Step 2: Validate the Case

```bash
cobre validate my_first_study
```

The validation pipeline checks all layers — schema, references, physical
feasibility, stochastic consistency, and solver feasibility — and prints entity
counts on success:

```
Valid case: 1 buses, 1 hydros, 2 thermals, 0 lines
  buses: 1
  hydros: 1
  thermals: 2
  lines: 0
```

If any layer fails, Cobre prints each error prefixed with `error:` and exits with
code 1. The 1dtoy template always passes validation.

---

## Step 3: Run the Study

```bash
cobre run my_first_study --output my_first_study/results
```

Cobre runs the SDDP training loop (128 iterations, 1 forward pass each) followed by
a simulation pass (100 scenarios). Output is written to `my_first_study/results/`.
The banner, a progress bar, and a post-run summary are printed to stderr:

```
Training complete in 0.5s (128 iterations, iteration_limit)
  Lower bound:  1.55955e7 $/stage
  Upper bound:  5.79592e5 +/- 0.00000e0 $/stage
  Gap:          -2590.8% (started at 70.5%)
  Policy rows:  384 active / 384 generated
  LP solves:    5632 (5632 first-try, 0 retried, 0 failed)

Simulation complete in 0.6s (100 scenarios)
  Completed: 100  Failed: 0

Output written to my_first_study/results/
```

> **Why is the gap a large negative number?**
> The 1dtoy config uses `forward_passes: 1`, which means each training iteration
> draws a single scenario trajectory for the upper-bound estimate. A single scenario
> is an extremely noisy sample of the true expected cost — one unlucky trajectory
> can land far below the lower bound, driving the gap deeply negative. This is
> expected behavior, not a solver error. The gap only becomes well-behaved and
> stable when training runs with multiple forward passes, because
> averaging over more scenarios produces a reliable upper-bound estimate. The 1dtoy
> template keeps `forward_passes: 1` for speed; in a production study you would
> increase this value and add a convergence-based stopping rule so training halts
> when the gap truly stabilizes.

Exact numerical values (bounds, gap, policy row counts, timing) will vary across
runs because scenario sampling is stochastic. The gap and iteration count depend on
the random seed and the convergence tolerance configured in `config.json`.

The results directory contains training convergence data, a FlatBuffers policy
checkpoint, and Hive-partitioned Parquet files for simulation dispatch results:

```
my_first_study/results/
  training/
    metadata.json
    convergence.parquet
    dictionaries/
    timing/
  policy/
    cuts/
      stage_000.bin  ...  stage_003.bin
    basis/
      stage_000.bin  ...  stage_003.bin
    metadata.json
  simulation/
    metadata.json
    costs/
    hydros/
    thermals/
    buses/
```

---

## What's Next

You have completed a full SDDP study from case setup to results. The following pages
go deeper into how the case is structured and how to interpret the output:

- [Anatomy of a Case](./anatomy-of-a-case.md) — what each input file controls
- [Understanding Results](./understanding-results.md) — how to read Parquet output and convergence metrics
- [CLI Reference](../guide/cli-reference.md) — all flags, subcommands, and exit codes
- [Configuration](../guide/configuration.md) — every `config.json` field documented
