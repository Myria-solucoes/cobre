# Chronological Blocks — Design

## 1. Purpose and scope

Cobre models each SDDP stage as a set of **blocks** (load levels / patamares). Two
block modes were always intended; only one is implemented today.

- **Parallel blocks (current).** Each block carries its own load balance. Hydro
  storage is modeled once per stage: a single initial storage `V_in` and a single
  final storage `V_out` per hydro. Turbined/spilled/diverted flows are per-block,
  but they all feed **one** water-balance row per hydro; the stage inflow is a
  single value scaled to the whole stage.

- **Chronological blocks (this design).** Blocks happen **sequentially** inside a
  stage, so storage evolves block-by-block: `S⁰ → S¹ → … → Sᴷ`. The stage's
  initial storage is the first block's initial storage (`S⁰`); the stage's final
  storage is the last block's final storage (`Sᴷ`). Inflow stays a single **rate**
  (m³/s) for the stage; the **volume** entering each block is that rate times the
  block's duration in hours. This lets the model capture intra-stage hydraulic
  dynamics (drawdown/refill across load levels) that parallel mode averages away.

The input surface already exists: `BlockMode::{Parallel, Chronological}`
(`cobre_core::temporal`) is parsed per-stage from `stages.json`
(`block_mode: "parallel" | "chronological"`, default `parallel`) and validated.
`Block { index, name, duration_hours }` carries the per-block hours, validated
`> 0` and (semantically) to sum to the stage duration. Today the solver simply
ignores the flag and always builds the parallel LP. This is therefore a
**solver-side** feature.

This document specifies, to implementation depth:

1. The chronological-mode LP column/row layout and how it mirrors the parallel
   design's hot-path choices (§4).
2. The cut / state-transfer interface that makes a trained policy **portable
   across block modes** (§5).
3. Simulation extraction and output of per-block storage (§6).
4. Generic constraints over per-block storage (ramp constraints) and hydro
   bounds (§7).

It closes with the threading of the per-stage mode (§8), the correctness
contracts (§9), interactions and non-goals (§10), and a phased implementation
outline (§11).

## 2. The governing invariant

The entire design rests on one fact about the existing architecture, confirmed
across the cut, layout, and state-transfer paths:

> **A Benders cut is a function of the state vector only**
> `{ storage (one per hydro), inflow lags, anticipated thermal state }`,
> and every column it touches lives in the **state region `[0, theta]`** whose
> offsets are a pure function of `(N, L, A, k_max)` — **independent of `n_blks`**.
> All per-block columns live in the **control region after `theta`**.

`StateLayout` owns `[0, theta]`; its module doc states the `n_blks`-independence
explicitly. The cut pool (`cut::pool`), wire format (`cut::wire`), and on-disk
policy are keyed only on `n_state = N·(1+L) + A·k_max`, which carries no `n_blks`
factor. `theta = N·(3+L) + A·k_max + A`, also `n_blks`-free.

**Design rule (load-bearing).** Chronological mode's intra-stage storage dynamics
must be added **strictly in the control region** (new per-block columns and rows)
and must **not** become state-vector dimensions. The state vector stays "one
storage per hydro per stage", where for a chronological stage:

- the **incoming** state `storage_in[h]` is the first block's initial storage `S⁰`;
- the **outgoing** state `storage[h]` (column `h`, range `[0,N)`) is the last
  block's final storage `Sᴷ`.

Violating this — inserting any per-block storage column inside `[0, theta]` —
would shift `theta`, move every state column, and make **every stored cut land on
the wrong column**: a silent wrong-bound corruption that still compiles. This is
the inverse of the `CutStateProjection` default-identity contract and the reason
`StudyDimensions` refuses to hold a global `n_blks`.

A corollary we get for free: because the inter-stage interface (`S⁰` in, `Sᴷ`
out) is mode-independent, **block mode can differ per stage** with no special
handling — a parallel stage's `V_out` flows into a chronological stage's `S⁰`, and
a chronological stage's `Sᴷ` flows into the next stage's `V_in`, identically.

## 3. How the parallel LP is built (recap, with the hot-path rationale)

The per-stage LP has two regions, and the ordering is deliberate.

**State region `[0, theta]`** — owned by `StateLayout`, stage-invariant:

```
[0, N)          storage      — outgoing storage Sᴷ (= V_out)        → outgoing state / cut target
[N, N(1+L))     inflow_lags  — AR lag variables (lag-major)
…anticipated_state, anticipated_state_out…
z_inflow[N]     — realized inflow (auxiliary; defines next-stage lag 0)
storage_in[N]   — incoming storage S⁰ (= V_in)                     → incoming state (pinned)
theta           — future cost scalar
```

It comes first for three hot-path reasons we want to inherit unchanged:

- **Fast state transfer.** The outgoing state is the prefix slice
  `primal[..n_state]`; the next stage is pinned by `set_col_bounds` on the
  contiguous `storage_in` block (resolved via
  `StateLayout::state_to_lp_incoming_column`). State pinning uses column bounds,
  not equality rows.
- **Fast cut evaluation.** A cut row touches only the `render_pairs` state columns
  plus `theta` — a tiny dense block-independent set. DCS scoring (`gemm`) spans the
  same outgoing projection.
- **Block geometry is quarantined after `theta`**, in the per-stage `StageLayout`
  (`StageGeometry`), the one place that already strides by this stage's `n_blks`.

**Control region `[theta+1, …)`** — owned by `StageLayout`, strided by `n_blks`
(`K`) via the `BlockGrid` typed address primitive (`flat(start, entity, blk) =
start + entity·K + blk`, entity-outer/block-inner):

```
turbine[H·K], spillage[H·K], diversion[H·K], thermal[T·K],
anticipated_decision[A], line_fwd/rev[L·K], deficit[B·S·K], excess[B·K],
[inflow_slack[N]], generation[Hf·K], evaporation cols,
[withdrawal slacks, 4 operational-violation slack families], ncs[·K],
pumping[·K], contracts[·K], generic slacks, σ_fill, σ^{v−}
```

**Water balance (parallel).** One equality row per hydro, summing all blocks
(`fill_state_and_water_entries`):

```
Sᴷ_h − S⁰_h + Σ_k τ_k·(q+s+d)_{h,k} − Σ_k Σ_{u∈up(h)} τ_k·(q+s)_{u,k}
            − Σ_l ζ·ψ_l·lag_{h,l}  (± ζ·slacks, + ζ·evap)  =  ζ·(base_h − withdrawal_h)
```

with two duration factors:

- `τ_k = blocks[k].duration_hours · M3S_TO_HM3` — per-block, on **flow** decisions;
- `ζ = (Σ_k duration_hours_k) · M3S_TO_HM3 = Σ_k τ_k` — stage-total, on the
  **inflow / loss** side (AR-lag ψ, inflow-penalty slack, evaporation flow,
  withdrawal). `M3S_TO_HM3 = 3600/1e6`.

**FPHA** (`fill_fpha_entries`, contract D06) uses **average** storage: the
`−γᵥ/2` coefficient lands on **both** `V_out` (column `h`) and `V_in` (column
`storage_in.start + h`), per block per plane. Evaporation (`fill_evaporation_
entries`) is analogous (one row per evap hydro, `−slope/2` on both storage
columns).

**PreFilling** hydros (no dam yet) get a frozen identity `V_out − V_in = 0` (only
those two entries) plus a short-circuit that reroutes their incremental inflow to
the first non-PreFilling downstream hydro. This keeps `∂Q/∂V̂_h = 0` (a valid flat
cut); leaving any flow/inflow coupling on that row produces a stale-nonzero cut
coefficient (a wrong cut that compiles). Spillage freeze is PreFilling-only;
turbine/diversion are frozen in both PreFilling and Filling (contracts D38–D42).

## 4. Chronological LP layout

The change is confined to the control region and the water-balance rows. The
state region, `theta`, `n_state`, and every cut/transfer mechanism are untouched.

### 4.1 New column family: interior block storage

Storage evolves through `K+1` boundaries `S⁰ … Sᴷ`. The endpoints reuse the
existing state columns (`S⁰ = storage_in[h]`, `Sᴷ = storage[h]`); only the `K−1`
**interior** boundaries need new columns. Add one control-region family:

```
storage_internal — length N·(K−1) in chronological mode, 0 in parallel mode
                   entry (h, i) = interior boundary Sⁱ⁺¹ of hydro h, i ∈ [0, K−1)
```

Placement: **first in the control region**, at `control_region_start()` (=
`theta+1`), shifting `turbine_start` to `storage_internal.end`:

```
let storage_internal_start = state.control_region_start();
let storage_internal_end   = storage_internal_start + n_h * n_interior;   // n_interior = K−1 (chrono) | 0 (parallel)
let turbine_start          = storage_internal_end;                        // was control_region_start()
// … rest of the equipment chain unchanged, each anchored at the previous .end …
```

Because `n_interior = 0` in parallel mode **and** when `K = 1` in chronological
mode, the family is empty in both, so `turbine_start == control_region_start()`
and the **entire LP is byte-identical to today** (§9). Placement is a
clarity/maintainability choice, not a correctness one (the accessor in §4.2 makes
it invisible to every consumer); "first in the control region" is chosen
deliberately:

- **Distinct stride, distinct region.** This family's stride is `K−1` (interior
  boundaries), **not** `K` like every other equipment family. Giving it its own
  clearly-delimited region — rather than nesting it among the stride-`K` flow
  columns (`turbine`/`spillage`/`diversion`) — avoids the "assumed to stride like
  its neighbors" footgun that `BlockGrid` exists to prevent. It does **not**
  address through `BlockGrid`; it gets its own accessor.
- **Stable anchor.** `storage_internal_start = control_region_start()` depends on
  nothing downstream, so future equipment-column additions cannot shift it; only
  one line (`turbine_start`) re-anchors, and the rest of the `prev.end` chain is
  untouched.
- **Cohesion with the state seam.** It places the interior storage trajectory
  immediately adjacent to the state region that owns its endpoints (`S⁰`, `Sᴷ`),
  so all storage lives at the `[0, theta]`/control seam.

(The considered alternative — grouping it after `diversion` with the per-hydro
flows — reads cohesively too, but mixes a stride-`K−1` family into stride-`K`
neighbors and perturbs more downstream offsets, so it loses on both counts above.)

### 4.2 The boundary-storage accessor (the "smart" addressing)

Following the `BlockGrid` single-owner philosophy — one typed method owns a
stride so no caller open-codes it — add a `StageLayout` method resolving any of
the `K+1` storage boundaries to its LP column, hiding the endpoints-are-state /
interiors-are-control split:

```rust
/// LP column of hydro `h`'s storage at block boundary `k` (k ∈ 0..=K).
/// k == 0 → S⁰ (incoming state); k == K → Sᴷ (outgoing state);
/// otherwise the interior control column.
fn block_storage_col(&self, h: usize, k: usize) -> usize {
    match k {
        0              => self.col_storage_in_start() + h,        // state region
        k if k == self.n_blks => h,                               // state region (storage.start + h)
        _              => self.storage_internal_start + h * (self.n_blks - 1) + (k - 1),
    }
}
```

Every water-balance fill, FPHA storage reference, extraction read, and
generic-constraint per-block storage term goes through this one method. In
parallel mode it is never called (the single-row balance path is used instead);
calling it with `K = 1` returns the endpoints only (no interior), preserving the
`K = 1 ≡ parallel` identity.

### 4.3 Water-balance rows: `N·K` chained rows

Replace the single `water_balance` row range (`N` rows) with `N·K` rows addressed
block-major like the load balance (`grid.flat(water_balance_start, h, k)`):

```
load_balance_start moves to water_balance_start + N·K   (parallel: N·1 = N, unchanged)
```

For an **Operating/Filling** hydro, block `k`'s row (1-indexed `k = 1..K`):

```
Sᵏ_h − Sᵏ⁻¹_h + τ_k·(q+s+d)_{h,k} − Σ_{u∈up(h)} τ_k·(q+s)_{u,k} − Σ_{d→h} τ_k·div_{d,k}
        − Σ_l τ_k·ψ_l·lag_{h,l}  (± τ_k·slacks, + τ_k·evap_{h,k})  =  τ_k·(base_h − withdrawal_h)
```

i.e. the parallel row, but per block, with **`τ_k` replacing `ζ` everywhere** —
the inflow rate, AR-lag ψ, evaporation, withdrawal, and inflow-penalty slack are
all apportioned to block `k` by its share of stage hours, and `Sᵏ`/`Sᵏ⁻¹` come
from `block_storage_col(h, k)` / `block_storage_col(h, k-1)`. The lag columns are
state columns (unchanged); only the coefficient placed on them is now `−τ_k·ψ_l`.

**Telescoping property.** Summing the `K` chained rows yields exactly the parallel
single row: the `Sᵏ` terms cancel pairwise to `Sᴷ − S⁰`, and `Σ_k τ_k = ζ`
recovers every `ζ`-scaled term. So chronological mode is **parallel mode plus the
interior storage-path constraints** (interior bounds + per-block FPHA and
evaporation). This is the foundation of the `K = 1` identity (§9) and a strong
differential oracle: a chronological run whose interior storage bounds never bind
and whose production/loss is storage-insensitive (`γᵥ = 0`, no storage-dependent
evaporation) matches parallel to LP tolerance.

**Inflow apportionment** is the modeling decision the brief fixes: a single stage
rate, volume by block hours (`τ_k·rate`). This design extends the **same**
apportionment to every other `ζ`-scaled quantity (AR-lag, evaporation, withdrawal,
inflow penalty) so the stage total is conserved exactly and the telescoping
identity holds. The arrival within a block is modeled as net volume on that
block's balance (no intra-block timing); intra-block routing and water travel time
are out of scope (§10).

The inflow **state** is untouched. The realized-inflow auxiliary `z_inflow` and
its definition row (which feed the next stage's lag-0 and the AR chain) remain
**stage-level** — `z_h` is still the total stage inflow, and only its _volume
contribution_ is split across blocks in the water-balance rows. Per-block inflow
must **not** become a per-block `z_inflow` (that would enlarge the state region and
break §2); the apportionment lives entirely in the control-region water rows.

**PreFilling under chronological mode.** Every hydro still emits `K` water rows.
A PreFilling hydro's block-`k` row is the per-block frozen identity
`Sᵏ_h − Sᵏ⁻¹_h = 0` (no flow/inflow/loss terms) plus the per-block short-circuit
reroute of its incremental inflow to the downstream target. The chain
`S⁰ = S¹ = … = Sᴷ` pins the interior columns and keeps `V̂_h` dead
(`∂Q/∂V̂_h = 0`), preserving the flat-cut contract per block. This mirrors the
parallel PreFilling branch exactly, replicated `K` times.

**Filling phase under chronological mode.** A `Filling`-phase hydro is **not**
frozen — it takes the standard per-block chained balance above (the freeze branch
is keyed on `PreFilling` alone), so its `K` interior storages evolve normally. Two
phase-specific rows stay **stage-level on the outgoing storage `Sᴷ`** and need no
per-block split: the filling-target soft floor (`σ_fill`: `v_h + σ_fill ≥
V_target`) and the filled-min-storage floor (`σ^{v−}`), both one row per hydro on
column `h = block_storage_col(h, K)`. The backward filling-target fold
(`build_filling_v_target`) is keyed `(hydro, stage)` and scaled by **stage-total
`ζ`**, so the `τ_k`-replaces-`ζ` water-balance change leaves it untouched —
`V_target` remains a target on the genuine stage-final storage that the telescoped
chain produces. Filling-phase spillage stays the legitimate **D40 relief valve**,
now per block: each block's spillage enters its own balance with `+τ_k`, so it
draws against that block's storage rather than injecting phantom water. Net: the
Filling phase composes correctly with chaining, but because it is non-obvious it is
pinned as a §9 contract rather than left implicit.

### 4.4 FPHA and evaporation: block-local average storage

FPHA already builds one row per `(hydro, block, plane)`. The only change: the two
`−γᵥ/2` coefficients move from the stage endpoints to **block `k`'s own
boundaries**:

```
col_v_in ← block_storage_col(h, k-1) = Sᵏ⁻¹      // was storage_in.start + h
col_v    ← block_storage_col(h, k)   = Sᵏ          // was h
```

so block `k`'s generation cap uses its local average `(Sᵏ⁻¹ + Sᵏ)/2`. This is
strictly more accurate than parallel mode, which uses the single stage average
`(S⁰ + Sᴷ)/2` for every block. The D06 "−γᵥ/2 on **both** storage columns"
contract is preserved — both columns now simply resolve through the accessor. For
`K = 1`, block 1's boundaries are `(S⁰, Sᴷ)`, identical to parallel.

**Evaporation is per-block**, for the same reason FPHA is: with a distinct water
balance per block, each block's evaporation must reflect that block's own storage.
The single stage-level evaporation row per hydro becomes `K` rows, and the single
evaporation flow/slack column triple becomes `K` triples (one per block). Block
`k`'s evaporation row reads `evaporation_flow_{h,k} − slope/2·Sᵏ⁻¹ − slope/2·Sᵏ +
f⁺_{h,k} − f⁻_{h,k} = intercept`, using its local average `(Sᵏ⁻¹ + Sᵏ)/2` (the
`−slope/2`-on-both-storage-columns structure of the parallel evaporation row,
resolved through the accessor), and its flow enters block `k`'s water balance with
`+τ_k`. For `K = 1` this collapses to the single parallel evaporation row on
`(S⁰, Sᴷ)`.

### 4.5 Bounds and scaling

- **Interior storage bounds.** Each `Sᵏ` inherits the per-`(hydro, stage)`
  `[min_storage_hm3, max_storage_hm3]` applied to the single column today (no new
  input). Column-bound application loops `h × interior_boundary`.
- **Objective coefficient: `0.0`.** Interior storage columns carry **zero**
  objective cost, exactly like the endpoint storage columns (`fill_storage_columns`
  writes only bounds; the outgoing and incoming storage columns are objective-`0.0`,
  and the sole storage-adjacent cost — `storage_violation_below_cost` — lands on the
  `σ^{v−}` slack, not on a storage column). Storage is priced only indirectly
  (FPHA head, evaporation, the soft `σ_fill`/`σ^{v−}` floors); an interior column
  given a nonzero objective would distort the stage cost.
- **Scaling.** Interior storage columns inherit the **storage column scale**
  (same physical quantity, hm³); the new per-block water rows inherit the
  water-balance row scale. Identical scaling on the state columns across modes is
  what keeps rendered cut rows (`−coeff·col_scale[col]`) byte-comparable (§5).

## 5. Cut portability and the state interface

This section states what the §2 invariant buys, bounds what it promises, and
flags the one persisted artifact that is mode-dependent (the warm-start basis).

**Structural portability (guaranteed, automatic).** A policy trained in one block
mode loads and applies in the other:

- The cut argument is the stage-initial storage per hydro, with identical meaning
  in both modes (`S⁰`); the cut is stored against the canonical
  `storage → lag → anticipated` index order via `CutStateProjection`, whose
  default-identity contract makes it index-identical to the global `StateLayout`.
- `n_state`, `theta`, the wire format, and the on-disk `OwnedPolicyCutRecord` are
  `n_blks`-independent, so the bytes are interchangeable. Stored cut coefficients
  are unscaled (the `col_scale` division happens at extraction), so the disk
  policy is scale-invariant; only the in-LP rendering must reconstruct the same
  state-column scale, which §4.5 ensures.
- State transfer (`primal[..n_state]` → `set_col_bounds` on `storage_in`) is
  inherited verbatim; cut evaluation touches only state columns + `theta`.

**What it does not promise (by design).** For `K ≥ 2`,
`Q_parallel(x) ≠ Q_chronological(x)` — they are different subproblems (chronological
adds interior storage-path constraints and per-block FPHA). "Train parallel,
simulate chronological" is therefore a deliberate **coarse-policy / fine-simulation**
workflow (the cut is a valid lower-approximation of the cost-to-go _under its
training dynamics_, evaluated against finer simulation dynamics), not a claim that
the cut equals the chronological cost-to-go. Training chronological yields a policy
fitted to the finer dynamics at higher per-stage LP cost; both are supported.

**Warm-start basis — the one mode-dependent persisted artifact.** The on-disk
policy stores more than cuts: it also persists a per-stage simplex basis
(`StageBasis`), reloaded on the "simulate from a saved policy" path
(`build_basis_cache_from_checkpoint` → `CapturedBasis` → warm-start). Unlike the
cuts, this basis is **column-count-dependent and therefore mode-dependent** — a
chronological LP has the extra `storage_internal` and per-block evaporation
columns. Cross-mode load does **not** corrupt results: `reconstruct_col_statuses`
resizes the stored statuses to the current LP's `num_cols` (BASIC-padding the new
columns), so it cannot hit the backend `col_status.len() == num_cols` assertion,
and a warm-start basis only _seeds_ the simplex — it cannot change the optimum.
The cost is a **degraded warm-start** (the parallel statuses are positionally
misaligned to chronological columns → extra pivots, a possible HiGHS
`basis_rejection`), not a wrong answer. So cross-mode policy reuse is sound, but it
is the cuts that are portable, not the basis; an implementation may choose to skip
the persisted basis entirely on a detected cross-mode load rather than pay the
repair. (Within a single chronological run the basis is fully consistent — the
broadcast wire format is length-prefixed and sizes to the live LP, §9.)

**Provenance.** Record the training block mode (and per-stage modes if mixed) in
the policy provenance report for traceability. Cross-mode load is **allowed**, not
rejected — that is the feature. No hard compatibility gate beyond the existing
`state_dimension` consistency check.

## 6. Simulation extraction and outputs

Today storage is extracted once per stage (`view.primal[storage_in.start + h]`,
`view.primal[storage.start + h]`) and the single pair is **repeated across every
block row** of `SimulationHydroResult` (which is already emitted one row per
`(hydro, stage, block)`). The per-block extraction pattern already exists for
turbined/spilled/generation (`extract_hydro_per_block`, addressing via
`grid.flat`).

**Change.** In chronological mode, fill each block row's storage from that block's
boundaries via the accessor:

```
storage_initial ← view.primal[block_storage_col(h, b)]      // Sᵇ
storage_final   ← view.primal[block_storage_col(h, b + 1)]  // Sᵇ⁺¹
```

So `block b`'s row reports `(Sᵇ, Sᵇ⁺¹)`. In parallel mode the existing behavior is
unchanged (every block row reports `(S⁰, Sᴷ)`). Derived per-block stored-energy
follows from the per-block initial storage.

**Output schema.** The existing `storage_initial_hm3` / `storage_final_hm3`
columns in the `hydros` parquet become genuinely block-specific in chronological
mode (same columns, block-resolved values) — no new columns required, and the
`block_id`-partitioned shape already accommodates it. CLI/Python parity is
automatic: both call `cobre_io::write_simulation_results`; the change lives in
`SimulationHydroResult` population (shared extraction), not the writers.

## 7. Generic constraints and hydro bounds

The generic-constraint engine already supports per-block terms: most `VariableRef`
variants carry `block_id: Option<usize>` and the row machinery expands a
`block_id = None` term per block (or collapses to one stage row when **every** term
is block-independent). Turbined and outflow are already per-block, so their
within-stage variation is already expressible. The one gap is storage: today
`VariableRef::HydroStorage` resolves to the single outgoing column
(`state.storage.start + h`) and is classified block-independent.

**Two new variants — initial and final storage, block-indexed.** Mirror the way
flow variables expose a block index: a user references either boundary of a block,
and the block index follows the same `0..K-1` convention as `HydroTurbined` et al.
(initial of block `b` spans boundaries `0..K-1`; final of block `b` spans
boundaries `1..K`; together they reach every boundary `0..K`).

```
VariableRef::HydroStorageInitial { hydro_id, block_id: Option<usize> }
    eff_blk = block_id.unwrap_or(block_idx)
    column  = block_storage_col(h, eff_blk)          // Sᵉᶠᶠ_ᵇˡᵏ  (start of block), coeff +1.0

VariableRef::HydroStorageFinal { hydro_id, block_id: Option<usize> }
    eff_blk = block_id.unwrap_or(block_idx)
    column  = block_storage_col(h, eff_blk + 1)      // Sᵉᶠᶠ_ᵇˡᵏ⁺¹ (end of block), coeff +1.0
```

This gives the modeler:

- The **stage-initial anchor** the brief calls for: `HydroStorageInitial{block_id:
0}` = `S⁰`, the pinned incoming storage — available in both modes (boundary 0
  always exists). Referencing it is sound: the cut subgradient is the reduced cost
  of that pinned column, which by duality absorbs any constraint loaded onto it.
  (A constraint on `S⁰` alone only tests the incoming state; the useful case
  couples `S⁰` to decisions or later boundaries.)
- The **stage-final** value `HydroStorageFinal{block_id: K-1}` = `Sᴷ`, equal to the
  existing `HydroStorage{hydro_id}` (retained, block-independent, for back-compat).
- **Within-stage ramp / variation constraints**, e.g.
  `HydroStorageFinal{b} − HydroStorageInitial{b} ≤ Δ` (within block `b`) or
  `HydroStorageInitial{b+1} − HydroStorageInitial{b}` (across the boundary).

Both variants are **block-dependent** — they must return `false` from
`variable_ref_is_block_independent` (whose match is exhaustive, so adding a variant
forces an explicit classification at compile time), so a `block_id = None` term
expands per block (referencing each row's own boundary) rather than collapsing to
one mis-priced stage row.

**Validation.** A per-block storage reference resolving to an **interior** boundary
(`k ∈ 1..K-1`) exists only in chronological mode. So on a `Parallel` stage with
`K > 1`, reject any `HydroStorageInitial`/`HydroStorageFinal` term **except** the
two stage endpoints (`Initial{0}` → `S⁰`, `Final{K-1}` → `Sᴷ`), with a clear
message ("per-block storage references an interior boundary, which requires
chronological block mode at stage N"). `block_id = None` on a parallel `K > 1`
stage expands onto interior boundaries and is likewise rejected. For `K = 1` and
all chronological stages, every reference is valid.

**Hydro bounds.** Per the scope decision, interior boundaries inherit the existing
per-`(hydro, stage)` storage bounds; no `hydro_bounds.parquet` schema change. A
future per-block storage-bound override (optional `block_id` + min/max columns)
remains possible but is out of scope here.

## 8. Threading the per-stage mode

`BlockMode` is read per stage from `Stage.block_mode` during template build:

- `StageLayout::new` reads it to size `storage_internal` (`N·(K−1)` chronological,
  `0` parallel/`K=1`) and the water-balance row count (`N·K` vs `N`).
- `fill_state_and_water_entries` branches on it: the existing single-row path for
  parallel; the chained per-block path (§4.3) for chronological. PreFilling and
  Operating/Filling sub-branches exist in both.
- `fill_fpha_entries` / `fill_evaporation_entries` resolve storage columns through
  the accessor (which is mode-aware), so they need no explicit branch beyond the
  per-block evaporation row count.

Per-stage granularity is honored end to end (mixed-mode runs are valid); the
common case (all stages one mode) is the degenerate uniform case. No study-global
`BlockMode` field is introduced — it stays per-stage, alongside `n_blks`.

## 9. Correctness contracts

These are the regression anchors the implementation must pin (named tests /
deterministic cases), in the spirit of the existing D-case contracts.

- **`K = 1` ⇒ byte-identical to parallel.** A chronological stage with a single
  block produces an LP byte-identical to the parallel build (empty
  `storage_internal`, one water row, FPHA on `(S⁰, Sᴷ)`). Pin with a differential
  test asserting identical CSC/bounds/objective.
- **Telescoping ⇒ parallel agreement when interiors are inert.** A chronological
  run matches the parallel run's bounds to LP tolerance when nothing makes the
  interior storage path bind: non-binding interior storage bounds **and**
  storage-insensitive production/loss (`γᵥ = 0` and no storage-dependent
  evaporation). The water-balance rows telescope exactly (`Sᵏ` cancels to
  `Sᴷ − S⁰`, `Σ τ_k = ζ`) unconditionally; the per-block FPHA/evaporation
  storage references are the only legitimate divergence once `γᵥ ≠ 0` or the
  evaporation slope is nonzero.
- **Cross-mode policy load.** A policy trained parallel loads into a chronological
  run (and vice versa) and evaluates `theta` without error; the cut bytes are
  unchanged. Pin with a round-trip test over `cut::wire` / policy load.
- **`theta` / `n_state` invariance.** Assert `theta` and `n_state` are independent
  of `block_mode` for a fixed `(N, L, A, k_max)` — the §2 invariant, pinned
  directly so a future layout change that violates it fails loudly.
- **Declaration-order invariance.** Per-block water-balance assembly iterates
  blocks and hydros in fixed (index) order; results stay bit-identical regardless
  of input entity ordering (the workspace-wide hard rule).
- **D06 / D38–D42 preserved.** The average-storage coefficient lands on both
  block-local storage columns; the PreFilling per-block frozen identity preserves
  the spillage/turbine freeze and flat-cut contracts.
- **Filling-phase target on `Sᴷ`.** A `Filling`-phase hydro with `K ≥ 2` emits `K`
  standard chained rows; its `σ_fill` / `σ^{v−}` floors stay on the stage-final
  `Sᴷ` and its `ζ`-scaled `V_target` fold is unchanged. Pin with a `K ≥ 2` Filling
  case asserting the target binds on `Sᴷ` and per-block spillage stays free (D40).

## 10. Interactions and non-goals

- **Water travel time** (`docs/design/water-travel-time-sddp-analysis.md`).
  Chronological blocks are the natural substrate for intra-stage travel time
  (an upstream block-`k` release arriving at a downstream block `k+m`), but travel
  time is **out of scope** here; this design routes upstream releases within the
  same block, as parallel mode does.
- **Risk measures, convergence, DCS / cut selection.** Unaffected: they operate on
  the cut pool and state vector, both unchanged.
- **MPI basis broadcast / within-run warm-start.** Unaffected in mechanism: the
  `CapturedBasis` broadcast wire format is length-prefixed and sizes to the live LP
  (§5), so the larger chronological column/row counts flow through. The one
  caveat is **cross-mode** reuse of a _persisted_ basis (§5): mode-dependent,
  positionally reloaded, harmless to correctness but a degraded warm-start.
- **Pre-existing CLP basis bug (track separately).** A latent CLP-backend defect
  (`docs/design/clp-basis-status-code-bug.md`) installs demoted cut rows with the
  wrong status code (degraded warm-start, correct optimum). Chronological mode
  exercises the cut-row demotion path harder, so fix it before relying on CLP
  warm-starts at scale. Not a blocker — HiGHS is the default backend and is
  unaffected.
- **Anticipated thermals.** Their state/decision columns are in the state region
  and stage-level; unaffected. `storage_internal` sits between `theta` and the
  equipment block, leaving `anticipated_state_out` (state region) untouched.
- **Intra-block dynamics.** Net-volume-per-block only; no sub-block timing.

## 11. Phased implementation outline

Each phase is independently testable and leaves the build green.

**Phase 1 — chronological LP core (makes training & simulation run).**
`storage_internal` column family + `block_storage_col` accessor; `N·K`
water-balance rows with `τ_k` apportionment; per-block FPHA/evaporation storage
via the accessor; PreFilling per-block frozen identity; interior bounds inherit
stage bounds; scaling; `BlockMode` threading in `StageLayout`/entries/rows.
Anchors: `K=1` byte-identity, `theta`/`n_state` invariance, telescoping agreement.

**Phase 2 — simulation extraction & outputs.** Per-block storage read via the
accessor in `extract_hydro_per_block`; block-resolved `storage_initial/final`;
per-block stored-energy; CLI/Python parity (shared writer). Anchors: per-block
storage extraction test; parallel-mode extraction unchanged.

**Phase 3 — generic constraints & validation.** `HydroStorageInitial` /
`HydroStorageFinal` variants + resolvers via the accessor + exhaustive
`is_block_independent` classification; the stage-initial anchor (`Initial{0}`) and
back-compat `HydroStorage`; ramp-constraint expressibility; semantic validation
rejecting interior-boundary storage references under parallel mode. Anchors:
ramp-constraint LP test; stage-initial-anchor test; parallel-mode interior
rejection test.

**Cross-cutting — policy portability.** Provenance records training block mode;
cross-mode load test; documentation/book update describing the two modes and the
coarse-train/fine-simulate workflow.

## 12. Resolved decisions

- **Evaporation is per-block (§4.4).** Modeled consistently with per-block FPHA,
  using each block's local average storage; the stage-level fallback is dropped.
- **`storage_internal` is its own family, first in the control region (§4.1).**
  Chosen for clarity/maintainability: distinct stride (`K−1`) in a distinct region,
  a stable anchor, and adjacency to the state seam — not nested among the stride-`K`
  flow columns.
- **Per-block initial _and_ final storage references, block-indexed (§7).** Two
  variants (`HydroStorageInitial` / `HydroStorageFinal`) following the `0..K-1`
  block-index convention of the existing flow variables; the stage-initial anchor
  is `HydroStorageInitial{0}` (`S⁰`); `HydroStorage` is retained as the
  block-independent stage-final alias.

### To finalize during implementation

- Final `VariableRef` names and the `block_id = None` expansion semantics (current
  convention: `None` = the row's own block) — naming only; the column mapping and
  classification above are fixed.
- The per-block evaporation slack/penalty structure simply replicates the existing
  stage-level one `K` times; confirm no penalty-weight rescaling is needed beyond
  the `τ_k` apportionment of the flow into the balance.
