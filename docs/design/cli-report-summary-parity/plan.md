# `cobre summary` / `cobre report` — full live-run parity

Status: IN PROGRESS (approved 2026-05-30; D4 = full topology fidelity)

## Goal

- `cobre summary <DIR>` reproduces the **entire** `cobre run` end-block from a
  completed output directory, without re-running: Execution topology, Hydro
  models, Model provenance, Training (+ time split), Simulation (+ expected
  cost, time split).
- `cobre report <DIR>` JSON gains the headline numbers it currently lacks:
  final bounds and simulation expected cost (free once the metadata carries
  them).

## Why (current state, verified)

Both commands work and paths are current. But:

- `summary` simulation section is degraded: `total_time_ms` hardcoded `0`
  (`commands/summary.rs:155`) → "0.0s"; `mean_cost`/`std_cost` hardcoded `None`
  (`:152-153`) → no "Expected cost" line.
- `report` is metadata-only → no bound values, no expected cost.
- Root cause: the live run builds fully-populated `TrainingSummary`
  (`run.rs:1142`) and `SimulationSummary` (`run.rs:1467`) in memory, but the
  persisted `cobre-io` metadata captures only a subset. The cost aggregate
  (`cobre_sddp::SimulationSummary`: mean/std/cvar/...) is computed live and
  never persisted; `simulation/metadata.json` holds only scenario counts +
  duration.

## Data gaps & sources (section → what summary needs → where it is today)

| Live section        | Needs                                                              | Persisted today?                                  |
| ------------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| Execution topology  | backend, threads, layout, mpi/slurm, **per-host rank lists**      | `DistributionInfo` (scalars+mpi+slurm); **no hosts** |
| Hydro models        | `HydroModelSummary` (n_constant/n_fpha/planes/evaporation/refs)   | **Not persisted**                                 |
| Model provenance    | `ModelProvenanceReport`                                            | `training/model_provenance.json` ✓ — **no reader**|
| Training core       | iters, bounds, gap, rows, LP solves                               | metadata + `convergence.parquet` ✓ (summary reads)|
| Training time split | fwd/bwd solve seconds, parallelism, first/retried/failed          | **Not persisted**                                 |
| Simulation core     | scenario counts, duration                                         | metadata ✓ (duration ignored — bug)               |
| Simulation cost     | mean/std (→ CI95), cvar                                            | **Not persisted** (live only)                     |
| Simulation solve    | lp_solves/first_try/retried/failed, solve_seconds, parallelism    | **Not persisted**                                 |

Schema-freshness gate (`ci.yml:160`) is **input-only** — output metadata has no
`schemars` derive, so extending these structs does not trip it.

## Design decisions (finalized 2026-05-30)

- **D1 — Persist in the existing metadata structs** (not a new sidecar). Extend
  `TrainingMetadata` + `SimulationMetadata` in `cobre-io/output/manifest.rs`.
  `report` (dumps metadata) gains the numbers for free; `summary` reads them
  back O(1). New fields are additive + `#[serde(default)]` so old output dirs
  still deserialize.
- **D2 — Unify provenance as a cross-model concept (chosen over "persist
  as-is").** Provenance ("user-given vs. internally-fitted") applies to BOTH
  the inflow model and the FPHA/hydro production model. Restructure
  `ModelProvenanceReport` (cobre-sddp) into two sub-sections:
  - `inflow`: the existing seasonal-stats / AR / correlation / opening-tree
    sources (+ ar_method/order, white-noise fallbacks, history digest).
  - `hydro_production`: **aggregated from the already-existing
    `HydroModelProvenance`** (`hydro_models.rs:305-315`:
    `production_sources`, `evaporation_reference_sources`) — FPHA planes
    `computed_from_geometry` vs `precomputed_hyperplanes` counts; evaporation
    reference `default_midpoint` vs `user_supplied` counts.
  The hydro **structural** summary (`HydroModelSummary`: n_constant/n_fpha/
  total_planes/evaporation-linearized counts, kappa_warnings) stays its own
  artifact, but **loses the source qualifiers from its display** (they move to
  the provenance section). This changes the live `cobre run` output too — by
  design (approved).
- **D2a — Provenance reader.** Add `read_provenance_report` (cobre-io) for the
  now cross-model report.
- **D3 — Hydro structural summary: persist `training/hydro_models.json`** +
  reader. Requires `Serialize`/`Deserialize` on the structural parts of
  `HydroModelSummary` (drop/relocate the source-bearing `fpha_details` to
  provenance, or keep counts only). Mirrors the provenance sidecar pattern.
- **D4 — Topology: FULL fidelity (chosen).** Add `hosts: Vec<HostLayout
  { hostname, ranks: Vec<u32> }>` to `DistributionInfo` so `cobre summary`
  reproduces the exact multi-node per-host breakdown. Additive + serde default.
- **D5 — Parity:** every new persisted field/file must be written by BOTH
  `cobre-cli/run.rs` and `cobre-python/run.rs` (hard rule). Both already share
  `write_checkpoint`/`write_provenance_report`; the cross-model provenance
  builder + metadata writing paths must be aligned in both.

## Phases

1. **cobre-sddp — unify provenance (foundation).** Restructure
   `ModelProvenanceReport` into `inflow` + `hydro_production` sub-sections;
   extend `build_provenance_report` to take `&HydroModelProvenance` and
   aggregate FPHA-plane and evaporation-reference source counts. Make the
   structural `HydroModelSummary` (counts only) `Serialize`/`Deserialize` and
   relocate the source qualifiers out of its display contract. Update
   cobre-sddp unit tests + the cobre-cli `summary.rs` display fns/tests that
   assert "loaded"/"computed"/"v_ref" strings.
2. **cobre-io — persistence layer.** `DistributionInfo.hosts` (HostLayout);
   `TrainingMetadata` bounds + training solve stats (fwd/bwd seconds,
   parallelism, first/retried/failed); `SimulationMetadata` cost
   (mean/std/cvar) + solve stats + parallelism. All additive + `serde(default)`.
   Add `read_provenance_report`; add `hydro_models.json` writer+reader.
   Tests: roundtrip + back-compat (old metadata without new fields).
3. **cobre-cli — wiring + display.** `run.rs`: build cross-model provenance
   (pass hydro provenance), populate new metadata fields, persist
   `hydro_models.json` + topology hosts. `summary.rs`: restructure
   `print_provenance_summary` (Inflow / Hydro-production sub-sections), strip
   source from `print_hydro_model_summary`, fix the `0.0s` duration.
   `commands/summary.rs`: assemble all five sections via the shared print fns
   (parity by construction); graceful degradation when a file is absent.
4. **cobre-python — parity.** Mirror the cross-model provenance build + new
   metadata fields + `hydro_models.json` + topology hosts in
   `cobre-python/run.rs`.
5. **report.rs.** Free enrichment via metadata dump; optionally surface
   explicit `bounds`/`cost` keys in `ReportOutput`.
6. **Tests + e2e.** cobre-io roundtrip + back-compat; CLI integration test
   (run 1dtoy → `summary` emits every section incl. expected cost + unified
   provenance; `report | jq .simulation.cost`); parity check on a Python-run
   output dir.

## Verification gates

- `cargo test -p cobre-io -p cobre-cli`; clippy + fmt clean.
- End-to-end: `init 1dtoy` → `run` → `summary` reproduces the live end-block
  (diff sections against the live `run` output); `report | jq .simulation.cost`.
- Parity: a Python-run output dir feeds `cobre summary` identically.

## Out of scope

- Persisting per-scenario detail beyond the aggregate (already in parquet).
- CVaR/category breakdown in the summary unless trivially free from the struct.
