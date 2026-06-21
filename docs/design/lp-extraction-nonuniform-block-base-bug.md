# Latent bug: simulation extraction reads stage-0 equipment column bases under non-uniform block counts

## Status

**Confirmed latent correctness bug.** Pre-existing (not introduced by the
`StateLayout` extraction work). Discovered during the role-(b) consumer
enumeration while removing the global `StageIndexer`. Filed for a dedicated
fix with owner re-baseline — it is **not** hash-neutral, so it does not belong
in the hash-neutral `StateLayout`-extraction spine.

## Symptom

`simulation/extraction.rs` (`StageExtractionSpec`, `extract_hydro_per_block`
and its ~20 sibling reads) resolves per-block primal columns as
`grid.flat(<base>.start, h, b)` where:

- the **stride** (`grid` / `StageExtractionSpec.n_blks`) is correctly
  **per-stage** — its rustdoc explicitly notes that using the global
  `indexer.n_blks` "would misread any stage whose block count differs"; but
- the **base** (`spillage.start`, `diversion.start`, `thermal`,
  `withdrawal_slack_*`, the four operational-violation slacks,
  `generation.start`, `evap_indices`) is read from the **global stage-0**
  geometry instance, where e.g.
  `spillage.start = turbine.start + hydro_count * n_blks_stage0`.

For any stage whose block count differs from stage 0's (a non-uniform block
schedule such as `[1, 3, 2]`), the stage-0-derived base is offset wrong, so
extraction reads the **wrong primal column** for that stage's per-block
quantities.

`turbine.start` (= `theta + 1`, the control-region start) and
`water_balance.start` (a row base) are stage-invariant and are read correctly;
the bug is specific to the equipment bases that sit **after** the first
block-major family, whose offset rides on stage-0's `n_blks`.

## Why current regression coverage misses it

The deterministic parity hash for the non-uniform-block cases (D33, D34) hashes
only stage-invariant reported fields (`storage_final_hm3`, derived from
`storage.start = 0`; and `water_value_per_hm3`, a row dual). The
`n_blks`-dependent equipment outputs (per-block spillage, turbine flow,
operational slacks) are reported into other `SimulationHydroResult` fields that
the parity hash does not cover. So the gap is real but invisible to the existing
D33/D34 regression.

## Defect class

This is the same root as the `anticipated_state_out` relocation bug fixed in the
`StateLayout`-extraction work: a **global stage-0, `n_blks`-dependent column
offset applied at a stage whose block count differs**. The relocation fixed the
cut-target instance; this is the simulation-extraction instance of the same
class.

## Fix direction (for the dedicated ticket)

The per-stage geometry that knows each stage's correct equipment bases is
`StageLayout`, but it is **ephemeral** — built inside
`lp/builder/template.rs::build_single_stage_template` and dropped after the CSC
is baked, so it does not exist at simulation-solve time. The fix therefore
requires **persisting per-stage equipment bases** (the per-stage `StageLayout`
geometry, or a compact per-stage equipment-base table) and repointing
`simulation/extraction.rs` onto the stage-correct base for the stage being
extracted.

This changes reported simulation outputs for non-uniform-block studies, so it
is **not hash-neutral**: it needs an owner re-baseline (`COBRE_PARITY_REGEN`) and
an `sddp-specialist` correctness sign-off, and the parity hash should be
**widened** to cover at least one `n_blks`-dependent equipment field so the class
cannot silently regress again.

## Interim disposition

While unfixed, the `StateLayout`-extraction work keeps a slim role-(b) geometry
descriptor that `simulation/extraction.rs` reads, preserving the existing
stage-0 base behavior **unchanged** (hash-neutral). The bug is neither widened
nor fixed by that work; it is preserved exactly as it stands today until the
dedicated fix lands.
