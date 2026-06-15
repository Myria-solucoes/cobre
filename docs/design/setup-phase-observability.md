# Setup-phase observability — timing, fitting metadata & FPHA validation

**Status:** Design / scope decided (ready for `/plan`)
**Scope:** `cobre-sddp`, `cobre-io`, `cobre-cli`, `cobre-python`, schemas,
`CHANGELOG.md`
**Author:** design loop, 2026-06-15

## Summary

Cobre richly instruments the **training** and **simulation** phases (per-iteration
timing, `duration_seconds`, convergence parquet) but the **setup** phase — system
load, PAR(p)/stochastic fitting, FPHA fitting, evaporation fitting, broadcast —
is observability-dark: the CLI shows counts only, nothing is timed, and some
fitted models are never exported at all. This effort closes that gap with four
deliverables:

1. **Setup-phase timing** — time each setup step; render a CLI **"Setup"** block;
   export a **non-hashed** `setup` section in `metadata.json`.
2. **Evaporation coefficient export** — persist the linear evaporation
   coefficients (today computed then discarded).
3. **FPHA validation / deviation report** — export the FPHA-vs-exact deviation,
   both as a per-`(hydro, stage)` aggregate and (opt-in) a per-sampled-point
   table (NEWAVE `oper_desvio_fpha` parity), built on the `compute_fit_deviation`
   diagnostic already shipped.
4. **Enriched "Hydro models" CLI section** — per-source FPHA/productivity detail,
   an inflow-fitting line, and a worst-deviation summary.

## Part A — Current state (verified against HEAD)

**CLI "Hydro models" section** (`crates/cobre-cli/src/summary.rs`,
`print_hydro_model_summary`): two counts-only lines —
`Production: N FPHA (X planes), M constant` and `Evaporation: A linearized,
B without` — backed by `HydroModelSummary`
(`crates/cobre-sddp/src/production/hydro_models/summary.rs`,
`build_hydro_model_summary`). No timing; no inflow/PAR(p) line.

**Exports related to fitting:**

- FPHA: `hydro_models/fpha_hyperplanes.parquet` (fitted planes only;
  `crates/cobre-io/src/output/hydro_models.rs`, `write_fpha_hyperplanes`).
- Evaporation: **nothing.** `resolve_evaporation_models`
  (`crates/cobre-sddp/src/production/hydro_models/evaporation.rs`) fits
  `Q_ev = k_evap0 + k_evap_v·v` around `v_ref = (v_min + v_max)/2` and keeps the
  coefficients only in `EvaporationModelSet` — never persisted.
- PAR(p): `stochastic/fitting_report.json`
  (`crates/cobre-io/src/output/stochastic.rs`, `write_fitting_report`) carries
  `selected_order` + per-season coefficients + order-reduction reasons; no
  timing, and AR coefficients are empty under the current `max_order = 0`
  (white-noise) support. `inflow_*` parquets carry seasonal stats / AR coeffs.

**Timing:** training writes `duration_seconds` in `training/metadata.json` plus
per-iteration timing (`training/convergence.parquet`,
`training/timing/iterations.parquet`); simulation writes `duration_seconds`.
**Setup is entirely untimed.**

**Output-writer path:** schema → `RecordBatch` builder → `write_parquet_atomic`
→ public writer in `crates/cobre-io/src/output/` → call site in
`crates/cobre-cli/src/commands/run/outputs.rs` (`write_training_outputs`). The
Python bindings (`crates/cobre-python/src/run.rs`) reuse the same `cobre-io`
writers, so a new output is mirrored in Python automatically — but the call site
parity (CLI ⇄ Python) must be verified per the project Python-parity rule.

## Part B — Decisions

| #     | Decision           | Resolution                                                                                                                                                                                                                                                                         |
| ----- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A** | Effort scope       | All four deliverables (timing, evaporation export, FPHA validation report, CLI enrichment).                                                                                                                                                                                        |
| **B** | FPHA report grain  | **Both.** Per-`(hydro, stage)` aggregate (mean/max/signed deviation, relative, `α`) **always** written to the metadata setup section; per-sampled-point table (`hydro_id, stage_id, v, q, fph_exact, fpha_fitted, deviation, relative`) as an **opt-in** parquet (off by default). |
| **C** | Timing granularity | **Per-phase totals** (load, stochastic/PAR(p) fit, production/FPHA fit, evaporation fit, broadcast). Not per-plant — FPHA fitting is rayon-parallel, so per-plant wall time is noise.                                                                                              |
| **D** | Timing home        | A **non-hashed** `setup` section in the existing `metadata.json` (reuse the manifest), plus a compact CLI "Setup" block. Timings are non-deterministic and must never enter a parity-hashed artifact.                                                                              |

## Part C — Workstreams

### WS1 — Setup timing backbone

Instrument the setup steps in the run orchestration (`cobre-cli` /
`cobre-sddp` setup path) with `Instant`/`elapsed`, collect a `SetupTimings`
value (per-phase `Duration`s), and:

- Render a compact CLI **"Setup"** block (one line per timed phase).
- Export the timings as a `setup` section of `metadata.json` (manifest), kept
  **out of any parity hash**.

Constraint: any timing type that lands in `cobre-io`/`cobre-core` must be named
generically (no `fpha`/`par` tokens); the per-phase labels can be specific where
collected in the orchestration layer.

### WS2 — Evaporation coefficient export

New parquet `hydro_models/evaporation_models.parquet` (or a section of an
existing hydro-models output) with `hydro_id`, `k_evap0`, `k_evap_v`, `v_ref_hm3`
(and source provenance). Wire the writer in `cobre-io` + the call site in
`cobre-cli`; verify Python parity. Deterministic and additive (default runs gain
a new file, existing parity files unchanged).

### WS3 — FPHA validation / deviation report

Reuse `fpha_fitting::compute_fit_deviation` (already on the fit path):

- **Aggregate** per `(hydro, stage/entry)`: `mean_abs_mw`, `max_abs_mw`,
  `mean_signed_mw`, `relative`, `alpha`. Carried up from the fit (it already
  computes the aggregate) into the metadata setup section and the CLI summary.
- **Per-point** (opt-in): a parquet sampling the fit grid —
  `hydro_id, stage_id, v_hm3, q_m3s, fph_exact_mw, fpha_fitted_mw, deviation_mw,
relative`. Gated by a CLI flag / config field, **off by default**. The
  deviation values are deterministic, so the file is reproducible; it is opt-in
  purely for size (`hydros × stages × grid`).

### WS4 — Enriched "Hydro models" CLI section

Extend `print_hydro_model_summary` / `HydroModelSummary`:

- Per-source FPHA / constant-productivity / precomputed detail.
- An **inflow-fitting** line (PAR(p) selected order summary, from the existing
  fitting report).
- A **worst-deviation** summary line (the max relative FPHA deviation across
  plants, from WS3's aggregate) + the per-phase setup timings from WS1.

## Cross-cutting constraints

- **Determinism.** Timings are non-deterministic → non-hashed metadata only.
  Deviation values and evaporation coefficients are deterministic → safe to
  hash, but the new outputs are **additive** (default parity baselines unchanged;
  no re-bless unless a baseline opts into hashing a new file).
- **Inert defaults.** The per-point report is off by default; the always-on
  additions (evaporation parquet, metadata setup section) do not alter existing
  output files, so default runs stay byte-identical on the parity-hashed set.
- **Python parity.** Every new CLI-written output mirrored in
  `cobre-python/src/run.rs` (automatic via shared `cobre-io` writers; verify the
  call site).
- **Generic infra naming.** No algorithm tokens in `cobre-core`/`cobre-io`
  identifiers, comments, or shipped docs.
- **Both LP backends green** (HiGHS default; CLP via
  `--no-default-features --features clp`).

## Testing

- Timing: a unit/integration check that the `setup` metadata section is present
  and well-formed; timings excluded from any parity hash.
- Evaporation export: round-trip the parquet; coefficients match the fitted
  model; Python writes the same file.
- FPHA validation: aggregate matches `compute_fit_deviation`; per-point table
  shape/columns; opt-in flag default-off leaves outputs unchanged;
  determinism (bit-identical across input ordering / rank count).
- CLI: snapshot of the enriched "Hydro models" + "Setup" blocks.

## Sequencing

```
WS1 (timing backbone)
  └─▶ WS4 (CLI enrichment — consumes WS1 timings + WS3 aggregate)
WS2 (evaporation export)        — independent, small
WS3 (FPHA validation report)    — independent of WS1; aggregate feeds WS4
```

WS1 and WS2 are low-risk and land first; WS3 reuses the shipped deviation
diagnostic; WS4 is the display layer that consumes WS1 + WS3 and lands last.

## Related follow-up — parallelize PAR(p) fitting (gated on WS1 timing)

A likely-worthwhile parallelization target this effort should quantify, not a
deliverable here. Today the FPHA fit runs per-hydro in parallel
(`resolve_production_models_from_artifacts`: `system.hydros().par_iter().map(fit_one_hydro).collect::<Result<Vec<_>, _>>()`
then a sequential in-canonical-order flatten — bit-deterministic, covered by
`fit_is_thread_count_invariant`), but PAR(p) inflow fitting is **sequential**
except the already-parallel residual step (`par/fitting/correlation.rs`,
`compute_hydro_residuals`).

With the live default `estimation.max_order = 6`, the per-hydro path that runs —
`par/fitting/estimation.rs::estimate_all_hydro_ar_coefficients` (per-hydro
periodic PACF order selection + a non-Toeplitz Yule-Walker LU solve per season),
and the per-`(hydro, season)` `estimate_seasonal_stats` — is genuinely expensive
and structurally independent per hydro. **Inspiration: mirror the FPHA pattern** —
`par_iter` over the canonical hydro slice, each task owning local state, a
canonical-index `collect()` reassembly, with the inner per-season arithmetic
reproduced verbatim so output is bit-identical across thread/rank counts. The
deterministic idiom already exists in this very module (`compute_hydro_residuals`
is the proven exemplar the FPHA comment cites), so the lift is low and the
determinism gate (shuffle/rank-count invariance) is the same one FPHA passes.

**Decision: defer the implementation, but quantify it with WS1.** Once the
`setup` timing section reports the PAR-fitting wall-time slice on a large case
(151 plants × decades of monthly history at order 6), parallelize
`estimate_all_hydro_ar_coefficients` (and, if it shows up, `estimate_seasonal_stats`)
with the FPHA idiom if the readout justifies it. The prior is "worth doing"; WS1
turns it from a guess into a measured call.
