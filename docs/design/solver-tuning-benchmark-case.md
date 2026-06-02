---
title: "Solver-tuning benchmark case — specification"
date: 2026-06-02
status: design
tags: [cobre, benchmark, case-spec, sddp, wall-clock, tuning]
companion: ["docs/design/solver-parameter-tuning.md"]
---

# Solver-tuning benchmark case — specification

This document specifies the study to build for the solver/accelerator tuning
research. The tuning **procedure** that consumes it is in
`docs/design/solver-parameter-tuning.md`.

## Design requirements

The case must satisfy three requirements at once:

1. **Reproduce correct results accurately** — a deterministic configuration with
   a frozen reference (lower bound + first-stage cost) so every variant can be
   checked for correctness, not just speed.
2. **Be fast on small problems** — a small/cold configuration that runs in
   seconds–minutes, so regressions on small problems are caught cheaply.
3. **Scale to large problems** — large stage LPs with a **deep cut pool**, so a
   1–2-iteration warm-started probe reproduces the production per-solve regime
   (192 forward passes warm-starting against a fat pool on the HPC cluster).

The binding constraint behind all three: Cobre is **solver-bound**, and the
dominant variable cost is the **backward pass solving many LPs sequentially per
worker** against the cut pool. The case is sized so that LP solve time dominates
wall time and the cut-row count (not the base LP) is the main scaling axis.

## System (entity files)

| Dimension                         | Target                            | Rationale                                                                                               |
| --------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Reservoir hydros (storage states) | **15–25**                         | State dimension → cut-row width → pricing/BTRAN cost. The primary scaling axis.                         |
| FPHA hydros (variable head)       | several of the above              | Adds the average-storage generation constraint; richer LP structure.                                    |
| Thermals                          | 15–30                             | Realistic dispatch breadth; more columns.                                                               |
| Buses / lines                     | 1 bus, or 3–5 buses + a few lines | Keep the network modest — the focus is hydro state + cuts, not transmission.                            |
| Horizon                           | **60 monthly stages** (5 years)   | Many backward LPs per pass; production-like depth.                                                      |
| PAR(p) inflow order               | **1–2**                           | Autocorrelation; PAR lags add to the state dimension.                                                   |
| Backward openings per stage       | **20–40**                         | The count of sequential LPs in the backward pass — the source of the backward-step solve-time variance. |

State dimension ≈ (#reservoirs) + (PAR lags). Aim for an effective state
dimension of ~20–50: large enough that cut rows are wide and the warm-start
basis is non-trivial, small enough to run locally in minutes.

Provide reproducible stochastics: a fixed `scenario_source.seed`, a PAR(p)
inflow model, and in-sample schemes. Domain realism (inflow magnitudes, FPHA
coefficients, costs) is best supplied from real data scaled to the sizes above —
the synthetic structure is fine, but the numbers should be physically plausible
so the policy converges cleanly and the duals stay well-scaled.

## Acceptance criteria (is the case good enough?)

Before using the case for tuning, confirm on a baseline run:

- **Solver-bound**: `solve_stats.(forward+backward)_solve_seconds` is a large
  fraction (target ≥ 70%) of `duration_seconds`. If overhead/IO dominates, the
  case is too small — scale up stages/openings/state.
- **Backward-dominated**: `backward_solve_seconds` ≥ `forward_solve_seconds`.
- **Deep pool achievable**: the pre-build run reaches the target active-cut depth
  `K` per stage (hundreds–~1000). Measure with the `probe_k_disaggregated`
  example.
- **Convergence is clean**: lower bound stabilizes; no excessive solver retries
  (`solve_stats.retried` ≈ 0 on the baseline config).

## Three run modes (same system, different `config.json`)

The same entity files are reused; only `config.json` changes.

### Mode A — correctness (the gate)

Deterministic, single trajectory, run to convergence. Its converged lower bound
and first-stage cost are the **frozen reference**. Every variant must reproduce
them within tolerance (set the tolerance from baseline run-to-run drift; exact
bit-identity is not expected across solver/accelerator configs).

```jsonc
{
  "training": {
    "forward_passes": 1,
    "stopping_rules": [{ "type": "iteration_limit", "limit": <converged-N> }],
    "scenario_source": { "seed": 42, "inflow": { "scheme": "in_sample" } }
  },
  "simulation": { "enabled": true, "num_scenarios": 100 },
  "policy": { "mode": "fresh" }
}
```

### Mode B — small / fast (regression)

Small forward count, modest cap, cold start. Confirms a variant does not regress
small/cold problems and captures fixed overhead + cold-solve cost.

```jsonc
{
  "training": {
    "forward_passes": 4,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 50 }],
    "scenario_source": { "seed": 42 },
  },
  "policy": { "mode": "fresh" },
}
```

### Mode C — scale probe (the primary timing signal)

Two phases.

**Phase 1 — pre-build the deep pool once** (baseline config), storing bases:

```jsonc
{
  "training": {
    "forward_passes": 16,
    "stopping_rules": [{ "type": "iteration_limit", "limit": <N to reach K> }],
    "scenario_source": { "seed": 42 }
  },
  "policy": {
    "path": "./policy",
    "mode": "fresh",
    "checkpointing": { "enabled": true, "store_basis": true }
  }
}
```

**Phase 2 — probe each variant** from the frozen pool, 1–2 iterations,
`forward_passes` = local core count:

```jsonc
{
  "training": {
    "forward_passes": <local-cores>,
    "stopping_rules": [{ "type": "iteration_limit", "limit": 2 }],
    "scenario_source": { "seed": 42 }
  },
  "policy": { "path": "./policy", "mode": "warm_start" }
}
```

`warm_start` loads the policy from `<output_dir>/<policy.path>`. The harness
copies the frozen master policy into each variant's own output dir before the
run, so variants never mutate the master and every variant starts identically.

## Directory layout the harness expects

```
<case>/
  config.json                 # base; harness templates the 3 modes from it
  buses/ hydros/ thermals/ …  # entity files (the system)
  <stochastic inputs>
MASTER_POLICY/policy/         # produced once by Mode-C phase 1 (cuts + bases)
runs/<backend>/<variant>/     # per-variant output dirs (harness-created)
```

## Build steps

1. Author the entity files to the sizes above and a base `config.json`.
2. Run Mode A to convergence; freeze the reference LB + first-stage cost.
3. Verify the acceptance criteria on the Mode-A/baseline run.
4. Run Mode-C phase 1 to build `MASTER_POLICY/policy/`; confirm `K` via
   `probe_k_disaggregated`.
5. Hand off to the tuning procedure (`solver-parameter-tuning.md`).
