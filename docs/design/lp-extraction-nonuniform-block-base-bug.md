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

## Disposition after the LP-architecture-simplification residual review

The residual cross-cutting review confirmed this bug is the one
**non-hash-neutral** item among the otherwise subtractive, test/doc-only work
that closed the `StateLayout`-extraction effort. The owner directed it to be
**fixed in-plan** (rather than deferred to a separate follow-up), accepting the
non-hash-neutral re-baseline. The fix is specified and tracked as the
extraction-base correction ticket; the requirements below are its blueprint. The
bug is preserved exactly until that fix lands.

### Confirmed scope at the current tree

The buggy bases are the `indexer.<family>.start` reads in
`simulation/extraction.rs` for every equipment family that sits **after** the
first block-major family — i.e. whose offset rides on stage-0's `n_blks`:
`spillage`, `diversion`, `thermal`, `anticipated_decision`, `generation`,
`line_fwd`, `line_rev`, `excess`, `deficit`, the operational-violation slacks
(`turbine_below_slack`, `outflow_below_slack`, `outflow_above_slack`,
`generation_below_slack`), and the evaporation columns (`evap_indices`, anchored
at the `n_blks`-dependent FPHA-generation-block end). The single-block
`inflow_slack` and `withdrawal_slack_{neg,pos}` reads use a `+ h` offset with no
block stride, but their BASE still rides on the `n_blks`-dependent prior
families, so they shift under a non-uniform schedule and are in scope too. Only
the genuinely stage-invariant bases are correct and unaffected: `turbine.start =
theta + 1` (the control-region start), the row bases `water_balance.start` /
`load_balance.start`, and the family emptiness predicates / `max_deficit_segments`
constant.

**Second sub-class (same root, different access pattern).** `compute_cost_result`
in the same file sums the **whole global stage-0 range**
(`range_sum(indexer.<family>.clone())`) for the reported cost breakdown
(`thermal_cost`, anticipated/spillage_cost, deficit/excess/exchange/turbined/
inflow-penalty/withdrawal/evaporation/op-violation costs) — wrong **base AND
length** at any stage whose block count differs from stage 0's. It is a reported
output (`cobre-io` output schemas).

The owner-approved fix (Option B) **removes the entire stage-0/`n_blks` defect
class from `extraction.rs` by construction**: a per-stage `StageEquipmentGeometry`
(carrying a `Range` per family plus the per-stage `evap_indices`) is built from
each stage's `StageLayout`, threaded through `StageExtractionSpec`, and every
block-major / `n_blks`-dependent-base read — the per-block `grid.flat` reads, the
cost `range_sum` reads, the `anticipated_decision` decision-MW read, and the
single-block `inflow_slack` / `withdrawal_slack` / `evap_indices` reads — is
repointed onto it. After the fix the only `indexer.<family>` reads left in
`simulation/extraction.rs` are the genuinely stage-invariant `turbine.start`
(`theta + 1`), the row bases, the family emptiness predicates, and the
`max_deficit_segments` constant. The parity hash is widened to cover a per-block
equipment field (`spillage_m3s`), a cost-breakdown field (`spillage_cost`), and
the anticipated-decision field (`anticipated_decision_mw`), so all three sub-classes
are regression-visible. The cost-reconciliation invariant
`Σ(breakdown) == immediate_cost` is preserved and pinned by a non-uniform-block
regression test.

### Established fix pattern (precedent already in the tree)

The simulation pipeline already threads **per-stage** column-base slices into
`StageExtractionSpec` for two families: `ncs_col_starts: &[usize]` and
`pumping_col_starts: &[usize]` (each indexed by stage `t`, sourced from the
persisted per-stage geometry rather than the global stage-0 `StageIndexer`).
The fix mirrors this exactly: persist per-stage equipment bases for the
block-major families above (a compact per-stage `&[usize]` table per family, or
the per-stage `StageLayout` geometry) and repoint each `grid.flat(base, …)` call
onto the stage-correct base. The per-stage `n_blks` stride is already correct;
only the base must become per-stage.

### Non-negotiable fix requirements

- **NOT hash-neutral**: changes reported simulation outputs for non-uniform-block
  studies. Requires an **owner re-baseline** (`COBRE_PARITY_REGEN`) for the
  affected cases.
- **Widen the parity hash** to cover at least one `n_blks`-dependent equipment
  field (e.g. per-block spillage or turbine flow) so the defect class cannot
  silently regress again — the current D33/D34 hashes cover only stage-invariant
  fields, which is why the bug is invisible today.
- **`sddp-specialist` correctness sign-off** on the re-baselined outputs.
- Add or extend a deterministic non-uniform-block case asserting a per-block
  equipment value at a stage whose block count differs from stage 0.

Until that dedicated fix lands, the bug is preserved exactly as it stands today
(neither widened nor fixed).
