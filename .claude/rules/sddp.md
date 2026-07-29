---
paths:
  - "crates/cobre-sddp/**/*.rs"
---

# SDDP Numerical & Algorithm Conventions

Hard-won correctness contracts of the SDDP solver. Each one is a _contract_, not
a style preference: a plausible-looking deviation produces wrong bounds, rejected
warm-starts, or silently understated cuts that still compile and pass most tests.
Verify against the cited code before changing any of them.

## Benders cut sign & subgradient extraction

The FCF stores the **raw subgradient** `∂Q/∂x` as a cut's `coefficients` (it is
_not_ negated at storage). That subgradient is the incoming-state column's
reduced cost **divided** by `col_scale`:
`∂Q/∂x_orig = rc_scaled / col_scale[col]` — divided, not multiplied, because the
pin sets `v_scaled = v_orig / col_scale`. Cut-row construction then negates the
gradient so the LP row reads `−∇·x + θ ≥ intercept`, yielding the Benders cut
`θ ≥ Q(x̂) + π'(x − x̂)`.
Read: `training/backward/duals_extraction.rs` (`extract_duals_from_view`), `cut/fcf.rs`, and
`cut::row::push_scaled_coefficient`, where `batch.values.push(-coeff * d)`
applies the negation.

## State pinning uses column bounds, not equality rows

Incoming state is pinned with `set_col_bounds` on the incoming-state LP column;
there is no state-fixing row range in the LP. Always resolve the LP column —
for both pinning and dual extraction — via
`StateSpace::state_to_lp_incoming_column`; never assume a fixing-row index.
Read: `lp/indexer/state_space.rs`.

## FPHA uses average storage

The FPHA generation constraint is
`g ≤ γ₀ + (γᵥ/2)·(V_in + V_out) + γ_q·q (+ γ_s·s)`. The `−γᵥ/2` coefficient
appears on **both** the incoming and outgoing storage columns — not on `V_out`
alone. (Discovered during deterministic case D06.)
Read: `lp/builder/entries.rs` (`fill_fpha_entries` — pushes `−γᵥ/2` onto both the
incoming- and outgoing-storage columns), `lp/builder/rows.rs` (`fill_fpha_rows`),
and `lp/builder/template.rs`.

## Hydro-cell aggregation assumes one production map per cell

`HydroCellIndex` partitions a plant's `unit_groups` into `bus_id`-equivalence
cells; a same-bus group pair's bounds sum exactly into one cell's LP columns
only because every group sharing a cell also shares the plant's production
map and objective coefficients. `HydroGenerationModel` is a field on `Hydro`,
never on `HydroUnitGroup`, and the resolved coefficients
(`ResolvedProductionModel`, `FphaPlane`'s `gamma_v`/`gamma_q`/`gamma_s`) are
keyed `[hydro][stage]` (`ProductionModelSet::model`) — there is no group or
cell axis to key on, and `HydroUnitGroup` itself carries no productivity,
efficiency, or cost field. That is what makes a same-bus group pair a
segment (one shared production ray) rather than a 2-D zonotope, so summing
member bounds is exact for the turbined-flow and generation-MW box
constraints considered independently — see the fold-order sub-contract below
for the one place that independence breaks down.

A per-group productivity `ρ_g` would break this: pricing the cell at
`ρ_cell = max_g ρ_g` lets the LP draw the efficient unit's MW from the
inefficient unit's water, understating cost — an invalid lower bound that
still converges and still looks plausible. The fix, if a per-group
production field is ever introduced, is to widen the cell partition key to
`(bus_id, production-coefficient signature)`; partitioning by `bus_id` alone
would then silently misprice any mixed-productivity cell.

Read: `crates/cobre-sddp/src/production/hydro_models/types.rs`
(`ProductionModelSet::model`), `crates/cobre-core/src/entities/hydro.rs`
(`HydroGenerationModel` on `Hydro`, absent from `HydroUnitGroup`),
`crates/cobre-sddp/src/lp/indexer/hydro_cell.rs` (`HydroCellIndex::build`).
Pinned by `production_model_set_model_returns_correct_variant` (the
`(hydro, stage)` lookup has no group or cell dimension to key on) and
`test_multi_bus_plant_splits_into_bus_ordered_cells` (partitioning depends on
`bus_id` alone, blind to differing group bounds).

### ConstantProductivity's bound fold must run per group, then sum the cell

A `ConstantProductivity` plant has no separate generation column: its MW cap
folds into the turbine bound as
`min(max_turbined_m3s, max_generation_mw / ρ)` (`fill_turbine_columns`,
`lp/builder/columns.rs`). That fold is exact for a single-group cell; once it
resolves per cell instead of per plant, the fold ORDER becomes load-bearing
the moment a cell holds more than one group. The correct cell bound is
**fold-then-sum** — each group's own `min(q̄_g, p̄_g / ρ)` computed first, then
summed over the cell's groups — because each group is independently limited
by whichever of its own two caps binds first. **Sum-then-fold**
(`min(Σ_g q̄_g, (Σ_g p̄_g) / ρ)`) is the wrong-but-compiling alternative: `min`
does not distribute over a sum of independent terms
(`min(Σa, Σb) ≥ Σ min(a, b)`, strict whenever the binding side — flow-limited
or MW-limited — differs across the cell's groups), so it silently overstates
the cell's true capacity, producing an invalid, too-loose bound that still
converges: with `ρ = 1` and groups `(q̄=100, p̄=50)` and `(q̄=10, p̄=100)`,
fold-then-sum gives `50 + 10 = 60` while sum-then-fold gives
`min(110, 150) = 110` — with one shared `ρ` and no per-group productivity at
all.

A bound multiplier applied to BOTH `q̄_g` and `p̄_g` identically (a shared
availability derate, a per-unit nominal-capability scalar) commutes with the
per-group fold (`min(k·q̄_g, k·p̄_g/ρ) = k·min(q̄_g, p̄_g/ρ)` for `k > 0`), so
fold-then-sum stays exact under any number of such multipliers. A multiplier
on only one side of the pair (an MW-only forced-outage derate against a
fixed mechanical flow limit) does not commute the same way: it can flip
which side binds for a group at a given stage — harmless under
fold-then-sum, which re-folds independently per group regardless, but it
makes sum-then-fold's overstatement vary stage-to-stage instead of vanishing.

Live: `cell_max_turbined` (`fill_turbine_columns`, `lp/builder/columns.rs`)
resolves this bound per CELL, folding each of the cell's own member groups
before summing them, exactly as this sub-contract requires — each group's
`q̄_g`/`p̄_g` is its RESOLVED per-block value (the override when the study
supplies one, the declaration otherwise, via `GroupBoundLookup`), never the
bare declared value.

Read: `crates/cobre-sddp/src/lp/builder/columns.rs` (`cell_max_turbined`).
Pinned by `test_same_bus_groups_sum_into_one_cell_box`, mutation-verified
against sum-then-fold on a two-group fixture whose groups bind on opposite
sides.

### Both terms of the cell bound's closing `min` are load-bearing

`cell_max_turbined`/`cell_max_generation` close with `sum.min(hb...)`: `sum`
folds/sums the cell's OWN member groups; `hb...` is the plant's resolved
bound. Neither term is a redundant guard over the other — each dominates a
disjoint regime, and dropping either one compiles and passes today's
single-group fixtures.

Drop the plant term (`min` degenerates to `sum`) and every lowering
`hydro_bounds` override in a study is silently discarded: a mid-horizon
capacity cut, declared exactly the way the no-raising rule's own rejection
message prescribes (declare the plant at final capacity, tighten the earlier
stages with override rows), stops reaching the LP the moment the plant
declares more than one group.

Drop the group term (`min` degenerates to `hb...`) and a multi-cell plant can
turbine or generate past its own declared capacity, because
`cell_max_turbined`/`cell_max_generation` are the ONLY consumers of
`hb.max_turbined_m3s`/`hb.max_generation_mw` in the hydro LP path — no
plant-level aggregate-max row exists to catch the overshoot. Three
independent, fully-valid-input mechanisms reach the group term, the third
strictly inside a single cell:

- **Cell subsetting.** A two-bus plant's cell sums only its own bus's groups,
  necessarily less than the plant total whenever the other bus's groups are
  nonzero.
- **Rule 41 slack.** The declaring-plant sum check is `Σ_g g.max_* ≤
declared`, not `=` — groups summing to less than the declared value satisfy
  it.
- **Fold-then-sum vs. raw-sum, on a SINGLE cell.** Rule 41 checks the RAW
  group sum; `cell_max_turbined` checks the FOLD-then-sum. These can diverge
  even at rule-41 EQUALITY with no override at all, because "one cell" is not
  "one group": with ρ = 1, a plant declaring `(110, 150)` and two SAME-BUS
  groups `(q̄ 100, p̄ 50)` and `(q̄ 10, p̄ 100)` satisfies rule 41 exactly on
  both columns (100+10=110, 50+100=150), yet the folded group side is
  `min(100,50) + min(10,100) = 60` against the plant's own
  `min(110, 150) = 110`. The group term binds by 50 m³/s with no override, no
  cell split, and no rule-41 slack — the same non-distributivity that
  motivates fold-then-sum over sum-then-fold, surfacing on the OTHER side of
  the `min`.

The plant term collapses to a no-op only for a plant with **no declared
groups** (the implicit single group mirrors the plant's declared value
exactly, and the fold is monotone in both its inputs) — never merely "one
cell", which a same-bus multi-group plant also has while still hitting the
third mechanism above. This is inert on TODAY'S fixtures, not provably inert:
rule 41 and the no-raising rule both admit `value ≤ declared +
ENVELOPE_TOLERANCE`, so even a no-declared-groups plant's resolved value may
sit up to that tolerance above declared — the plant term could tighten by
that same margin. No shipped fixture exercises this; do not round it up to
"provably inert."

Read: `crates/cobre-sddp/src/lp/builder/columns.rs` (`cell_max_turbined`,
`cell_max_generation`), `crates/cobre-io/src/validation/semantic/block_bounds.rs`
(`check_bound_raises_declared_capacity`, the no-raising rule),
`crates/cobre-io/src/validation/semantic/hydro.rs` (rule 41). Pinned by
`test_same_bus_groups_sum_into_one_cell_box`'s third plant (a same-bus pair at
rule-41 equality, no override), which pins the group term binding, and by
`test_cell_columns_take_their_own_group_box`'s block-2 override, which pins
the plant term binding.

## Cut pool is append-only; basis matches by slot identity

Cuts are never removed from the LP. Deactivation toggles a cut row's RHS bounds
to the `±f64::INFINITY` sentinel (trivially satisfied); every cut keeps a stable
slot index for the lifetime of the run. The per-iteration template refreeze
encodes **only active cuts** (one row per `active_cuts()` entry), not inactive
cuts at sentinel bounds. Warm-start basis reconstruction therefore matches stored
cut rows to current LP rows by **`CutPool` slot identity**, never by row count.
`reconstruct_basis` is the single hot-path entry point — never bypass it.
Read: `cut/pool.rs`, `cut/basis_reconstruct.rs`.

## NCS stochastic availability is a dimensionless factor

Non-controllable-source availability `α_r(ω) ∈ [0, 1]` is dimensionless. The
realized cap is `A_r = max_gen · clamp(mean + std·η, 0, 1)`. The
`non_controllable_models.parquet` stores `(mean, std)` **as factors**, not as MW.
Read: `stochastic/noise.rs` (`transform_ncs_noise`, `compute_effective_eta`).

## Lower-bound evaluation must patch NCS

`evaluate_lower_bound` patches NCS column bounds per opening via
`StageSolvePrep::run`'s internal `transform_ncs_noise` call, exactly as the
forward and backward passes do. Skipping the patch understates the bound (a
real bug caught during D15). The patch inputs ride on `StageContext`
(`ncs_max_gen`, `ncs_allow_curtailment`), the same struct every other solve
site reads.
Read: `training/lower_bound.rs`, `training/stage_solve_prep.rs`.

## Per-stage exchange in the backward pass

`exchange()` is called inside the backward loop, once per stage, not in a
separate pre-pass before the loop.
Read: `training/backward_pass_state.rs`.

## Backward opening order is warm-start-only

A trial point's backward openings are SOLVED in the installed `solve_order`
permutation (`OpeningTree::set_solve_order`, keyed by
`noise_key::build_noise_key_table` — the intrinsic shortest-chain order, a
nearest-neighbor + 2-opt minimum-distance path over the openings'
inflow-noise vectors; a stage below 3 openings keeps its σ-weighted key, the
live fallback that also owns the noise-dimension validation) but each
opening's outcome is WRITTEN and AGGREGATED by **canonical ω**. The
aggregation therefore carries no solve-order dependence: results are
declaration-order-invariant and run-to-run reproducible across thread and
rank shapes (the pinned gates). No config field selects the order.
CHANGING the order (a code change to `noise_key`) changes the warm-start
chain each opening's solve starts from, and at a degenerate optimum a
differently-warmed solve may settle on a different-but-equally-valid vertex
with different duals — the hot≠cold divergence the Cobre determinism contract
permits — so an order change re-checks the golden parity baselines instead of
assuming byte-identical outputs. Aggregating the outcome slice indexed by
solve position — or handing solve-order-permuted probabilities to
`RiskMeasure::aggregate_cut_into` — is the wrong-but-compiling alternative: it
makes the cut depend on solve order, silently
breaking declaration-order invariance and run-to-run reproducibility.
Read: `stochastic/noise_key.rs` (`build_noise_key_table`, `apply_chain_order`),
`training/backward/trial_point.rs` (`process_trial_point_backward` — solves by
`solve_order`, aggregates by canonical ω), `training/backward/outcome_aggregation.rs`
(`write_opening_outcome`). Pinned by the `opening_order_determinism` gate in
`tests/mpi_wire.rs` (threads=k / threads=1 / a same-shape repeat / a 2-rank
stub, bitwise `final_lb`) and the MPI SLURM Integration job's rank-invariance
comparison on `examples/4ree`.

## Opening-block scheduler is warm-start-only

The opt-in opening-block scheduler
(`training.parallelism.backward_scheduler = { method = opening_block }`)
reassigns the backward pass's work unit from a whole trial point to an
opening-block: workers claim `(trial point, block)` units in any order from a
shared atomic counter, warm-chaining each block's openings from a fresh
frozen-LP load. Units are SOLVED in claim order — dependent on worker count and
scheduling timing — but each opening's outcome is WRITTEN into a per-`(m, ω)`
arena and AGGREGATED per trial point over CANONICAL ω, in ASCENDING m. The
generated cut set is therefore independent of claim order and worker count:
reordering claims changes only which worker warms which block, never which cut
is produced. Aggregating the arena in claim/solve-position order, or keying it
on the claim index instead of `(m, ω)`, is the wrong-but-compiling
alternative — CVaR's tail weighting is order-sensitive, so it silently breaks
CVaR reproducibility and declaration-order invariance the same way a
solve-order-keyed aggregation would break the trial-point path above. An
active Dynamic Cut Selection iteration always falls back to the trial-point
path: the opening-block scheduler's frozen-LP load is incompatible with
DCS's cut-free lazy core.
Read: `training/backward/opening_block.rs`
(`process_stage_backward_opening_block`'s claim loop,
`opening_block_finish`'s per-`(m, ω)` arena and ascending-m aggregation),
`training/backward_pass_state.rs` (`run_one_backward_stage`'s
`use_opening_block` dispatch). Pinned by
`opening_block_scheduler_determinism_expectation` and
`opening_block_scheduler_determinism_cvar` in `tests/mpi_wire.rs` (threads=4
/ a same-shape threads=4 repeat / threads=2 / threads=1 / a `Rank0Of2`
2-rank stub, bitwise `final_lb`, on both an expectation and a `CVaR`
configuration), `opening_block_degenerates_on_single_opening`
(opening-block-vs-trial-point equality on a single-opening case whose
resolved block count is `1`), and
`opening_block_handles_non_uniform_cut_projection`
(opening-block-vs-trial-point equality on a case whose per-stage cut-state
projection dimension varies across stages).

**Hardest-first claim order is result-neutral.** Under `OpeningBlock`,
claims are further ordered hardest-`(stage, block)`-first
(longest-processing-time, LPT) by the PREVIOUS iteration's per-`(stage,
block)` mean `simplex_iterations` pivot — never per-`(m, block)`, since
resampled trial points make per-m hardness noise where the opening-block
component is iteration-stable. The hardest-first order touches only the
claim decode: the per-`(m, ω)` write and the ascending-m aggregation above
are unchanged, so hardest-first-on and the canonical identity order produce
a bit-identical cut set and `final_lb`. Keying the order on per-`(m, block)`
pivots, reordering the arena or the aggregation instead of only the claim
decode, and a tie-break that leaves equal-mean blocks unordered (not a total
order) are each wrong-but-compiling: the first two reintroduce a
claim-order dependence the invariant above forbids; the third makes the
claim order itself nondeterministic across otherwise-identical runs.
`block_pivots_prev` is the previous iteration's fully-merged row —
`BackwardPassState::run` swaps it in from `block_pivots` once per call, never
per stage; reading `block_pivots` instead during the sweep is stale
(reset-then-partially-filled).
Read: `training/backward/opening_block.rs`
(`process_stage_backward_opening_block`'s `block_order`-indexed decode,
`hardest_first_block_order`, `identity_block_order`),
`training/backward_pass_state.rs` (`run_one_backward_stage`'s block-order
computation, the `run` swap). Pinned by
`hardest_first_claim_order_is_result_neutral` in `tests/mpi_wire.rs`
(hardest-first on vs off, bitwise `final_lb`).

## No EWMA upper bound

`ConvergenceMonitor::upper_bound()` returns the raw per-iteration upper bound —
there is no exponentially-weighted smoothing. Gap closure is immediate for
deterministic cases.
Read: `convergence/convergence.rs`.

## Spillage is frozen `[0, 0]` during PreFilling

A `PreFilling` hydro's spillage column is pinned `[0, 0]` — no dam exists yet to
spill from, and its incremental inflow has already left via the short-circuit, so a
free spillage column injects phantom water onto the first active downstream hydro's
water-balance row (a conservation violation). The freeze is gated on
`Phase::PreFilling` ALONE. Two wrong-but-compiling alternatives: extending the
freeze to `Filling` removes the legitimate over-dam relief valve an impounding
reservoir needs (D40); gating on `filling.is_none()` leaves the phantom-spill hole
open for a filling hydro in its own `PreFilling` sub-phase (D38, D39). Turbine and
diversion differ — they are frozen in BOTH `PreFilling` and `Filling` (no installed
machinery), whereas spillage is legitimately free in `Filling`.
Read: `lp/builder/columns.rs` (`fill_spillage_columns`). Cases: D38, D39, D42
(phantom PreFilling spill removed); D40 (legitimate Filling-phase spill retained).

## Policy-load compatibility validation is mandatory

Every policy load — full-FCF warm-start/resume/simulation-only and terminal
boundary-cut injection — routes through `validate_policy_load`, the single
entry point; there is no opt-out or bypass path. Its check matrix keys off
`PolicyLoadKind`: `state_dimension` equality and per-slot `slot_identity`
(`entity_type`, `entity_id`, `subindex`) are hard-rejected for both `FullFcf`
and `BoundaryInjection`; `num_stages` equality is hard-rejected only for
`FullFcf` — a `BoundaryInjection` load skips it deliberately, since a monthly
source study may legitimately feed a weekly+monthly current study.
`col_scale`/LP prescaling is explicitly NOT a compatibility dimension: a state
variable's identity and physical unit are independent of how the LP happens to
scale its column, so comparing `col_scale` would falsely reject a policy whose
entities genuinely match but whose scaling strategy or magnitude differs from
the current study's — the forbidden alternative this contract rules out.
Read: `policy/policy_load.rs` (`validate_policy_load`, `slot_identity`). Pinned
by the `validate_policy_load_full_fcf_*` and
`validate_policy_load_boundary_injection_*` tests in that module's test suite.

## Initial-state seeding resolves IDs through a position map, never `binary_search`

`System::hydros()`/`thermals()` sort canonically by `(operational_start_date,
id)`, which is id-ascending only when every entity shares one operational
start date. A staggered-commissioning system (filling reservoirs, future-entry
plants — the entire point of `operational_start_date`) breaks that
coincidence, so `binary_search_by_key` over the canonical slice — which
requires id-ascending order — silently returns `Err` (or the wrong index) for
an out-of-id-order entity, dropping its seed to the default `0.0`. Every
id-keyed initial-condition lookup (`storage`, `filling_storage`, thermal
`past_anticipated_commitments`) resolves through an `id -> position` map built
once from the canonical slice, never a `binary_search_by_key` call. The map is
built from the canonical order, but every write still iterates the IC record
list (not the map) — a map iteration order is unspecified and would violate
declaration-order invariance if used to drive writes.

The derived inflow lag seed (`derive_inflow_seeds`) satisfies the same
invariant a different way: it carries no id->position map at all — it
iterates `hydros` directly, so the loop index IS the canonical position, then
filters each hydro's own historical windows by id. `build_initial_state`'s lag
block trusts this pre-ordering and does a plain positional read, with no id
lookup of its own.
Read: `setup/mod.rs` (`id_to_position`, `build_initial_state`),
`crates/cobre-stochastic/src/seeds.rs` (`derive_inflow_seeds`). Pinned by
`test_initial_state_seeds_correctly_under_staggered_commissioning_dates`,
`build_initial_state_anticipated_seed_correct_under_staggered_commissioning_dates`,
and `test_seed_correct_under_staggered_commissioning_dates`, each using a
staggered-date fixture where the canonical order is id-descending.

## Water travel time

A declared upstream→downstream arc introduces in-transit "bucket" state: one
Markov-1 volume slot per `(downstream plant, lag)` absorbs water in flight. With
the feature compiled in but no arc declared (`n_buckets == 0`), every path below
collapses to the pre-bucket layout byte-for-byte; the moment any arc is
declared, each of the following is a contract.

### Shared lagged-delivery ring skeleton

The water in-transit bucket ring and the anticipated-thermal ring are one
lagged-delivery ring construct, owned by `DeliveryRing`: a borrowed outgoing
block (identity-resolved, contributing to `n_state`) and a separate borrowed
incoming block (pinned via `state_to_lp_incoming_column`), advanced one
Markov-1 slot per stage by the same interior shift row
(`DeliveryRing::emit_shift_rows`) and the same paired row-cap/column-freeze
masking (`DeliveryRing::freeze_masked_columns`). The two rings differ only in
how each deposits into its newest slot and in what a masked terminal slot
means — both differences live entirely at each ring's own call site, never a
second skeleton implementation:

- **Deposit.** Water's block-mode-coupled per-lag deposit share is emitted at
  its own call site (`fill_arc_release_block_entries`), never through
  `DeliveryRing::emit_deposit`. Anticipated's deposit IS `emit_deposit`: it
  pins the ring's newest slot to a single decision column, `+1` on
  `out_col(slot, lane)` and `−1` on `decision_col`.
- **Masked terminal slot.** Water's masked slot discards a genuine share the
  ring would otherwise deposit — an admitted target-stage imprecision (see
  Terminal credit deferred below). Anticipated's masked slot never held a
  value in the first place, because no anticipated commitment is ever created
  past the horizon (see End-of-horizon masking below). Both render the SAME
  masking output (frozen `[0, 0]`, scale-independent) — only the per-ring
  subsection below states what the masked slot MEANS.

The masking contract is always two-sided and ships together: a masked
position (`row_pos[i] == None`) gets NO definition row (the row-cap side) AND
a frozen `[0, 0]` outgoing column (`freeze_masked_columns`, the column-freeze
side) in the SAME pass — wiring only one side leaves either a dangling row
referencing a frozen column or a free column with no defining constraint, both
wrong-but-compiling. Water instantiates one ring per downstream plant
(`transit_bucket_ring`, `n_lanes = 1`, over that plant's ragged contiguous
sub-range); anticipated instantiates ONE dense ring spanning every plant
(`anticipated_ring`, `n_lanes = n_anticipated`, slot-major/plant-minor) — both
addressing schemes resolve through the same `out_col`/`in_col` formula
(`block.start + slot * n_lanes + lane`).
Read: `lp/builder/delivery_ring.rs` (`DeliveryRing::emit_shift_rows`,
`freeze_masked_columns`, `emit_deposit`, `out_col`/`in_col`, `slot_target`),
`lp/builder/entries.rs` (`transit_bucket_ring`, `anticipated_ring`).

### In-transit bucket dynamics & sign

`fill_transit_bucket_definition_entries` routes the bucket-definition ring
shift through `DeliveryRing::emit_shift_rows` (the shared skeleton above,
`b_d^out = b_{d+1}^in + k_d·D_i`); `fill_arc_release_block_entries` deposits
the arc's `k_d`-weighted release from the SAME release column that also
carries `k_0` onto the balance row — never a separate once-per-stage family
(the deposit share itself is emitted at the call site, never through
`DeliveryRing::emit_deposit`, which only the anticipated ring calls). Incoming
buckets are pinned via column bounds, resolved through
`StateSpace::state_to_lp_incoming_column`'s explicit bucket arm, never the
`anticipated_state` catch-all. Subgradient extraction
divides the incoming bucket column's reduced cost by `col_scale`
(`extract_duals_from_view`, the same rc/col_scale contract as storage); the
cut row renders the **outgoing** bucket column through
`StateSpace::lp_column_for_state`'s identity arm and multiplies `col_scale`
back on via `push_scaled_coefficient` — divided on extract, multiplied on
render, identical to storage. Swapping which column is pinned/read, or
dividing on render instead of extract, prices the in-transit water in the
wrong direction — a wrong bound that still compiles. A fold implementation
(crossing mass absorbed same-stage, no bucket at all) can reach the same total
cost as the correct one, so total cost alone cannot discriminate — only the
dual's sign/magnitude and the per-stage delivery split do.
Read: `lp/builder/entries.rs` (`fill_transit_bucket_definition_entries`,
`fill_arc_release_block_entries`, `transit_bucket_ring`), `lp/indexer/state_space.rs`
(`StateSpace::state_to_lp_incoming_column`, `StateSpace::lp_column_for_state`),
`training/backward/duals_extraction.rs` (`extract_duals_from_view`), `cut/row.rs`
(`push_scaled_coefficient`, `push_cut_row`). Pinned by the bucket-arm
column-resolution tests (outgoing resolves by identity, incoming resolves to the
pinned column via an explicit arm, never the anticipated catch-all) and the
per-stage-visit bucket-pinning regressions in the backward pass and lower-bound
evaluation; a sub-stage-delay bucket-dual regression is the fold-discriminating
pin for the sign/magnitude itself.

### k-factor conservation

`resolve_spread` sums the stage-clock weights to `Σ_d k_d = 1` per arc per
anchor stage (`debug_assert`-enforced), and `fill_arc_release_block_entries`
asserts the same sum immediately before it deposits. A closed-form ceiling
depth (e.g. `⌈t_v/h_t⌉`) is a plausible-looking replacement for the resolver's
overlap-based depth and silently drops trailing mass on a non-uniform calendar
— conservation violated, not a compile error.
Read: `lead_time/mod.rs` (`resolve_spread`), `lp/builder/entries.rs`
(`fill_arc_release_block_entries`). Pinned by the resolver's monthly-then-weekly
counterexample regression (asserting the correct, deeper depth against the
closed-form ceiling's shallower, wrong one) and the stage-level conservation
regression exercising the `Σ_d k_d = 1` debug_assert directly across
non-uniform calendars; a mixed-calendar end-to-end regression extends the pin
to delivered-plus-horizon-drop equalling released, per arc, to floating-point
tolerance.

A plant's turbined flow is `Σ_c q_c` over its `HydroCellIndex` cells — a
disjoint CSR partition, not a duplicate representation — so an arc's `k_d`
prices the plant's TOTAL release and is REPLICATED onto every cell's turbine
column at the same magnitude, never apportioned (divided) across them:
apportioning by `1/|C|` discards `(1 − 1/|C|)` of the released mass, an
under-delivery no less wrong than the ceiling-depth bug above. Conservation
holds PER CELL, not merely in the aggregate — every cell of a plant feeds the
same arc at the same travel time, so `stage_weights` is cell-invariant by
construction and the `Σ_d k_d = 1` debug_assert stays exactly where it is
(once per arc per stage), never moved inside a per-cell loop. This holds only
while travel time is an ARC (plant) attribute; if a cell ever acquires its own
`t_v`, each cell needs its own weight vector (each still summing to 1) and the
assertion moves inside the per-cell loop — still never apportioned even then.
Read: `lp/builder/entries.rs` (`fill_arc_release_block_entries`,
`fill_arc_release_chrono_block_entries`). Pinned by
`test_cascade_release_sums_the_upstream_plants_cells` (same-magnitude,
not-divided per cell) and `test_plant_total_release_is_invariant_to_cell_partition`
(a solved-LP objective/dual comparison between a one-cell and an evenly-split
two-cell plant releasing the same total).

### Canonical bucket ordering

Bucket columns sort by the downstream plant's canonical
`(operational_start_date, id)` index — the same order `System::hydros` already
carries — then by lag; never by raw declared id, never by cascade-traversal
order. `build_transit_bucket_topology` derives `column_order` from that canonical
iteration alone. Emitting buckets in traversal order instead makes the state
layout input-declaration-order-dependent, breaking the
declaration-order-invariance hard rule.
Read: `setup/bucket_topology.rs` (`build_transit_bucket_topology`,
`TransitBucketTopology::column_order`). Pinned by the bucket column-order
declaration-invariance regression: two systems differing only in the
declaration order of their hydros produce identical `column_order`,
`per_plant_depth`, and `n_buckets`.

### Stage-0 seed: windowed IC anchor

`build_initial_transit_bucket_state` seeds every declared arc's stage-0
incoming buckets directly from its `past_defluences` windows — never a
positional walk over a fixed pre-study calendar. For upstream hydro `i`'s
window `[start_date, end_date)`, `e_off = start_0 − end_date` and
`width = end_date − start_date` feed `ic_anchor_k` exactly as it already
takes `(cumulative_before, period_duration)`: the windowed derivation lives
entirely in how the caller computes those two offsets from calendar dates,
never inside `ic_anchor_k` itself. A hydro may carry multiple, non-contiguous
windows; the seed must `filter` over every window with a matching `hydro_id`
and deposit each one independently
(`volume = width · M3S_TO_HM3 · value_m3s`, `seed[start+d] += k[d] · volume`)
— a `.find()` would silently keep only the first window and drop the rest,
understating the seed with no error. There is no fallback for incomplete
coverage: `cobre-io`'s `validate_travel_time` row-5 gate guarantees every
declared arc's windows cover `[start_0 − t_v, start_0)` before setup ever
runs this seed.
Read: `setup/bucket_seed.rs` (`build_initial_transit_bucket_state`),
`setup/bucket_topology.rs` (`ic_anchor_k`). Pinned by the single-window
unroll regression (the `k`-weighted deposit matches the closed-form
half-share), the gapped-two-window additive regression (two non-contiguous
windows for one arc contribute independently), and the seed's own
declaration-order-invariance regression (distinct from, and in addition to,
the topology-level ordering pin above).

### Terminal credit deferred

`horizon_cap_active` caps each stage's active lag at `n_stages − 1 − t`, the
deepest lag whose target stage still lands inside the horizon;
`build_transit_bucket_row_pos` gates the per-stage LP fill on that cap, so a lag beyond
it gets no bucket-definition row at that stage — dropped by construction, not
retained and silently zeroed elsewhere. `fill_arc_release_block_entries` /
`fill_arc_release_chrono_block_entries` drop the matching deposit share rather
than write it to a stale row index, and `fill_transit_bucket_columns` freezes the
masked slot's outgoing column `[0, 0]` (the commissioning-dormant-column
convention) so no row is needed to define it. The complementary guarantee is
why dropping the row is safe: the finite horizon's zero terminal value
(`HorizonMode::Finite`, the only implemented mode) makes a masked slot's cut
coefficient structurally zero, so no solution loses value by never routing
water into it — the residual mass has no receiving stage either way. This
under-values end-of-horizon upstream release; it is a documented target-stage
imprecision, not a bug to patch by capping
`TransitBucketTopology::per_plant_depth`/`column_order` too — those size from the
global max over every anchor and must retain what the earliest stages need.
Read: `setup/bucket_topology.rs` (`horizon_cap_active`), `lp/builder/layout.rs`
(`build_transit_bucket_row_pos`), `lp/builder/columns.rs` (`fill_transit_bucket_columns`).
Pinned by the horizon-depth-cap regression (the last stage's active-lag cap
reaches zero, so no slot targets past the horizon), `build_transit_bucket_row_pos`'s
own consumption regression (that same cap sequence emitting correspondingly
fewer rows), and a sub-stage-delay case's last-stage release, whose dropped
share surfaces as an uneven per-stage delivery split rather than a credited
one.

### Sub-contracts: mode-independent sizing, aggregation consistency, fixed delivery density

The bucket state stays a pure function of stage lengths, never of
`n_blks`/`block_mode`, only because each of the following holds:

- **Depth from stage lengths alone.** Bucket depth and `n_buckets` derive from
  the per-stage calendar and the pre-study anchor alone
  (`study_stage_durations`, `build_transit_bucket_topology`) — never from `n_blks` or
  `block_mode`. Deriving any part of the depth inside a block-aware code path
  re-couples the state dimension to how a stage happens to be resolved.
- **Shared arrival density.** A chronological stage's per-block deposit shares
  `block_deposits`/`within_stage_routing` and the stage-level `stage_weights`
  come from the same shared arrival density (`resolve_spread`'s
  `stage_weights`/`block_deposits`/`within_stage_routing`,
  `resolve_block_factors`'s `BlockFactors`), so `Σ_b w_b·χ_{b,d} = k_d` holds
  by construction. Building `block_deposits`/`within_stage_routing` from one
  density and `stage_weights` from another lets the chronological and
  parallel cuts diverge and silently breaks conservation.
- **Fixed delivery density.** A maturing bucket delivers into its arrival
  stage's blocks through a fixed, `block_mode`-independent `arrival_density`
  looked up from the setup-precomputed per-`(arc, arrival stage)` table
  (`resolve_chrono_arrival_density` reading
  `TemplateBuildCtx::arc_arrival_density`, built by `build_arc_arrival_density`
  as a blend over every contributing source stage's lag, resolved in the
  ARRIVAL stage's own frame), never by tracking which origin block a unit came
  from. Tracking origin-to-arrival-block correlation would grow the bucket
  into a per-block vector whose length scales with the receiving stage's
  `n_blks` — re-violating the depth-from-stage-lengths property above.

Read: `lead_time/mod.rs` (`resolve_spread`'s
`block_deposits`/`within_stage_routing`/`arrival_density` fields,
`resolve_block_factors`'s `BlockFactors`, `resolve_arrival_density_at`),
`setup/bucket_topology.rs` (`build_arc_arrival_density`), `lp/builder/entries.rs`
(`fill_chronological_water_entries`, `resolve_chrono_arrival_density`). Pinned
by the shared-density-consistency regression exercising the aggregation
debug_assert directly, the chronological block-table regression matching the
worked kappa/chi numbers, and the `K = 1` chronological-vs-parallel
byte-identity regression; a state-dimension-equality regression across
parallel and chronological builds is the direct pin for mode-independent
sizing. The arrival-frame lookup regression (the resolved density equals the
precomputed `arc_arrival_density` table entry verbatim) is the direct pin for
the fixed-delivery-density clause itself; the parallel-fill regression (the
maturing bucket keeps a single `-1.0` regardless of the table's contents)
pins that `fill_parallel_water_entries` never reads it.

## Anticipated thermal commitments

### Pre-study anticipated commitments: calendar-derived coverage

`AnticipatedCommitmentHistory::values_mw` (`cobre-core`) is an ordinal,
delivery-stage-indexed vector — `values_mw[j]` is the MW delivered at the
`j`-th pre-study-committed delivery stage — never date-windowed like
`past_defluences`. Its length must equal the calendar-derived count of
pre-study-committed delivery stages: `LeadStages(l)` clamps to
`min(l, n_stages)`; `LeadTime(delta)` counts the leading study stages whose
stage-end cumulative hours are `<= delta` (tie-inclusive). `cobre-io`'s
`check_anticipated_thermals` computes this count itself, via
`required_anticipated_commitment_count`, rather than calling into the
solver crate's point-commitment resolver (cobre-io is upstream and cannot
depend on it), and hard-rejects any length mismatch as a
`BusinessRuleViolation`. A `len == lead_stages` gate is a plausible-looking
alternative that silently mis-covers a `LeadTime`-configured plant on a
non-uniform calendar, since the required count is calendar-derived, not a
constant stage count; there is no fallback comparable to the one already
rejected for `past_defluences` coverage above.
Read: `crates/cobre-io/src/validation/semantic/thermal.rs`
(`required_anticipated_commitment_count`, `check_anticipated_thermals`).
Pinned by `test_anticipated_lead_time_coverage_pmo_calendar` and
`test_anticipated_lead_time_coverage_pmo_calendar_under_coverage_rejected`.

### In-LP anticipated ring: definition-row sign & two-sided masking

The anticipated ring is `DeliveryRing`'s other instantiation (the shared
skeleton above): an outgoing block (`StateSpace::anticipated_slots_out`,
identity-resolved by `state_to_lp_column`, contributing to `n_state`) and a
separate incoming block (`StateSpace::anticipated_state`, pinned via
`state_to_lp_incoming_column`) — never one dual-purpose range shifted
out-of-LP. There is no Rust-side shift step: the ring transition is resolved
entirely by the definition rows below, and `current_state`/`state_at_capture`
read the outgoing block by the same plain copy already used for storage and
travel-time buckets.

An interior slot's outgoing column is pinned to the next slot's incoming value
by the shared ring-shift row, `slot_k^out − slot_{k+1}^in = 0`
(`fill_anticipated_slot_definition_entries` routes it through
`DeliveryRing::emit_shift_rows`); the plant's own newest slot (`k = k_i − 1`)
is pinned instead to the fresh decision column, `slot_{k_i-1}^out =
decision_col`, via the shared skeleton's deposit primitive
(`fill_anticipated_state_out_def_entries` calls `DeliveryRing::emit_deposit`
directly). Both row families render `[0, 0]`
(`fill_anticipated_slot_definition_rows` / `fill_anticipated_state_out_def_rows`):
the `+1`/`−1` structural coefficients on each side of the row do the shift,
never the bounds.

A slot beyond the horizon-reachable window (`build_anticipated_slot_row_pos`'s
per-slot `Option<usize>`, `None` when unreachable) gets BOTH sides of the
shared masking contract together via `DeliveryRing::freeze_masked_columns`: no
definition row (the row-cap side) AND a frozen `[0, 0]` outgoing column
(`fill_anticipated_slot_columns`, the column-freeze side — the same
commissioning-dormant-column convention as NCS/thermal/line/station/contract).

A slot beyond a plant's OWN `StateSpace::anticipated_lead_stages[plant]`
bound is structural padding even when `t + slot_idx` itself still lands
inside the horizon — the multi-plant heterogeneous-lead case, where two
plants sharing one `k_max`-wide ring have different per-plant reachable
widths. `policy::policy_export::build_stage_entity_manifest` applies this
same bound before populating `EntitySlot::delivery_anchor`, never a depth- or
decider-only check: `AnticipatedResolution::decision_sets`/`depth` count only
within-study-decided commitments and silently exclude a still-draining
pre-study seed, undercounting a ring position that legitimately holds one. The
manifest resolves a ring column back to `(slot, plant)` via
`DeliveryRing::slot_lane_at` — the exact inverse of `out_col`/`in_col` — never
a re-derived `offset % n_anticipated`/`offset / n_anticipated` pair.

Read: `lp/indexer/state_space.rs` (`StateSpace::state_to_lp_column`,
`state_to_lp_incoming_column`), `lp/builder/layout.rs`
(`build_anticipated_slot_row_pos`), `lp/builder/entries.rs`
(`fill_anticipated_slot_definition_entries`,
`fill_anticipated_state_out_def_entries`, `anticipated_ring`), `lp/builder/rows.rs`
(`fill_anticipated_slot_definition_rows`), `lp/builder/columns.rs`
(`fill_anticipated_slot_columns`), `policy/policy_export.rs`
(`build_stage_entity_manifest`). Pinned by the `state_to_lp_column`
`anticipated_slots_out`-identity regressions, the combined row-cap-and-
column-freeze regression asserting both sides in one test, the backward-cut
coefficient propagation regressions (K=1, K=2, K=3) confirming the ring-routed
definition rows produce the correct subgradient values, and the manifest's
padding-vs-reachable delivery-anchor regression.

### End-of-horizon masking is exact, never a dropped commitment

Unlike the water ring's Terminal credit deferred subsection, no anticipated
commitment is ever discarded at the horizon boundary — none is created there
in the first place. `is_anticipated_decision_active`/
`is_anticipated_decision_active_for_delivery` gate a decision column's
existence on the strict clause `stage_idx + K_i < n_stages`;
`PointResolution::decider` itself has a fixed domain `m in [0, n_stages)`, so
no code path ever computes a commitment targeting a delivery past the
horizon and then truncates it. `build_anticipated_slot_row_pos`'s per-slot
`None` (no definition row) and `fill_anticipated_slot_columns`'s frozen
`[0, 0]` outgoing column, at a `(stage_idx, slot)` pair whose target
`m = stage_idx + slot + 1 >= n_stages`, are therefore always vacuous: the
masked slot is provably zero for every valid configuration, never a real
value the model declines to route anywhere. This differs in kind from
water's masking: a masked bucket discards a genuine non-zero `k_d`-weighted
release share deposited every stage regardless of the arc's travel time — an
admitted target-stage imprecision — while the anticipated gate prevents the
decision from ever existing, so nothing of value is lost. Crediting a masked
slot as if it held a dropped commitment would introduce value the model
never computed, for a delivery stage that does not exist.
Read: `lp/indexer/anticipated_gate.rs` (`is_anticipated_decision_active`,
`is_anticipated_decision_active_for_delivery`), `lead_time/mod.rs`
(`PointResolution::decider`), `lp/builder/layout.rs`
(`build_anticipated_slot_row_pos`), `lp/builder/columns.rs`
(`fill_anticipated_slot_columns`). Pinned by
`a1c_lead_stages_is_pure_index_shift`'s empty-`decision_sets`-past-horizon
assertion.

### In-LP anticipated ring: single-decider deposit & `K = 0` exclusion

Each anticipated plant gets AT MOST ONE decision column per stage
(`col_anticipated_decision_start + local_idx`), driven by
`PointResolution::genuine_decisions_at(stage_idx).next()` (a `K = 0`
self-delivery already excluded — see below). That decision deposits into its
OWN ring slot, `slot = delivery_stage − stage_idx − 1` — computed DIRECTLY from
the decision's own delivery stage, never from a `depth`-derived boundary.

**`depth[t]` is not the ring's per-stage occupancy boundary.** `depth[t]`
(`PointResolution::depth`) counts only IN-STUDY decided items still in flight
— `build_decision_sets_and_depth`'s sweep adds a delta only for `Some(t)`
deciders, structurally excluding pre-study (`None`, IC-seeded) occupancy. A
plant can have BOTH an IC-seeded item and a fresh in-study decision occupying
the ring at the same stage (e.g. a constant-lead plant's stage 0), so
`depth[t] − genuine_count(t)` under-counts and mis-targets the slot — the
wrong-but-plausible shortcut `PointResolution::is_ready_at`'s doc comment
warns against. The correct interior/deposit/padding split is checked PER SLOT
directly: slot `k`'s target `m = stage_idx + k + 1` is a deposit iff
`decider[m] == Some(stage_idx)`, an interior shift iff `is_ready_at(m,
stage_idx)` and not a deposit, else padding. `decider` is nondecreasing in
`m`, so readiness is monotonic and slots are ready in a contiguous prefix from
slot 0 — the property that makes the per-slot check well-founded without
needing an aggregate boundary.

**`K = 0` (sub-stage lead, `c(m) = m`) is excluded from the ring entirely —
exclude-with-advisory, never a hard error, never an underflow.** A
delivery whose physical lead is shorter than its own stage's duration is
decided inside its own delivery stage; `PointResolution::self_delivered_stages`
identifies these, and `genuine_decisions_at`/`is_anticipated_at` filter them
out of the decision and fishing gates respectively — the plant's ordinary
thermal generation column is priced and bounded normally (no fishing
coupling, no anticipated row at all) at that stage. A setup-time
`tracing::warn!` (`setup::warn_on_sub_stage_lead`, the same channel
`StudyParams::from_config`'s budget advisory uses) names the plant, the
stage, and the `lead_stages == 0` alternative — never emitted per-scenario or
per-trajectory.

Read: `lead_time/mod.rs` (`PointResolution::genuine_decisions_at`,
`self_delivered_stages`, `is_anticipated_at`, `is_ready_at`),
`lp/indexer/anticipated_gate.rs`
(`is_anticipated_decision_active_for_delivery`,
`anticipated_resolution_for`), `lp/builder/layout.rs`
(`build_anticipated_slot_row_pos`, `build_anticipated_decision_row_pos`,
`build_anticipated_fishing_row_pos`), `lp/builder/columns.rs`
(`fill_anticipated_columns`), `lp/builder/entries.rs`
(`fill_anticipated_state_out_def_entries`, `fill_anticipated_fishing_entries`),
`setup/mod.rs` (`warn_on_sub_stage_lead`). Pinned by the `K = 0`
zero-emission-plus-advisory regression (no anticipated slot/row/fishing
coupling at any stage, one advisory per self-delivered stage).

### Fan-out configurations are rejected at setup

The LP builder has no fan-out representation: every anticipated plant gets at
most one decision column per stage (above). A `LeadTime` plant whose
resolution would fan out (`|genuine C(t)| > 1` at any decision stage) never
reaches it — `resolve_state_layout` rejects any
`AnticipatedResolution::max_fanout > 1` configuration with
`SddpError::Validation`, naming the fanning plant (`first_fanned_plant_id`)
before a study's stage templates exist. This is the SOLE fan-out guard, not a
belt-and-braces check backed by column/entry/row-position handling that no
longer exists.
Read: `setup/mod.rs` (`resolve_state_layout`, `first_fanned_plant_id`). Pinned
by `lead_time_fanout_rejected_at_setup` (asserts `SddpError::Validation`, not a
panic, after confirming the fixture genuinely fans out).

### Delivery-anchoring preservation

Every anticipated plant's decision column is bounded, costed, and
commissioning-gated at ITS OWN delivery stage `m` (its
`genuine_decisions_at(t)` target, when one exists), never the decision stage
`t`. `fill_anticipated_columns` reads `thermal_bounds(thermal_idx,
delivery_stage)` for the column's `[min, max]` bounds,
`total_hours_per_stage[delivery_stage]` and
`cumulative_discount_factors[delivery_stage]` for its present-value objective,
and `is_anticipated_decision_active_for_delivery` (the plant's window at
`delivery_stage`) for its dormancy — each at the plant's own genuine delivery
stage, never at `stage_idx`. The delivered commitment is a hard equality with
no slack (the fishing coupling pins the plant's delivery-stage generation to
the committed value), so relatively-complete recourse requires the committed
value always lie within the delivery stage's own generation bounds. A
DECISION-anchored read (`thermal_bounds(thermal_idx, stage_idx)`) is the
forbidden alternative: it
reintroduces the capacity-drop infeasibility — a commitment placed under the
decision stage's larger capacity that no scenario can deliver under the delivery
stage's smaller one, stranded with no feasibility cut to absorb it — and still
compiles, since constant-across-lead bounds make the two reads indistinguishable.

Residual audit complete: no mechanism other than `thermal_bounds` can strand a
delivered commitment. The only generic-constraint handle on an anticipated plant,
`VariableRef::AnticipatedDecision` (`resolve_anticipated_decision`), binds the
fresh decision column at its own decision stage (the recourse variable, already
delivery-anchored here), never an in-flight matured commitment (no `VariableRef`
targets the ring state slots) nor the delivery-stage generation; constraining it
cannot strand a delivered value. The one path that touches the delivery-stage
generation, `VariableRef::ThermalGeneration` on an anticipated plant, is already
surfaced by `warn_thermal_generation_on_anticipated_thermal` and is the general
"a hard generic constraint may be infeasible" class, not an anticipated-specific
hole.

Read: `lp/builder/columns.rs` (`fill_anticipated_columns`),
`lp/indexer/anticipated_gate.rs` (`is_anticipated_decision_active_for_delivery`),
`lp/generic_constraints.rs` (`resolve_anticipated_decision`),
`cobre-io` `validation/semantic/thermal.rs`
(`warn_thermal_generation_on_anticipated_thermal`). Pinned by
`test_anticipated_decision_delivery_anchored_bounds` (stage-varying delivery
bounds/cost, mutation-verified against the decision-anchored read), the
end-to-end
`a1b_lead_time_equals_lead_stages_uniform_calendar` (the same
decision-anchored mutation turns the forward solve infeasible; pinned by
training and simulating both `LeadTime` and `LeadStages` configurations of
the same calendar to bit-identical solutions), and
`a1c_lead_stages_is_pure_index_shift` (pins the delivery-anchored decider
`c(m) = m - lead` those bounds are read against).

### Delivered commitments reconcile against solver drift; exactness is unreachable

Delivery-anchoring keeps the committed value inside the delivery stage's
generation bounds **in exact arithmetic only**. The value that actually reaches
the delivery stage is the solver's computed value for a **basic** ring-slot
column: `slot_out` is defined by an equality row (`slot_out − decision = 0`, or
the interior shift), so the simplex produces it through the basis factorization,
and it is accurate only to the backend's `primal_feasibility_tolerance` (`1e-9`
on HiGHS and CLP) — never to 1 ULP. A commitment at its cap therefore arrives a
hair outside it, and the fishing equality's no-slack pin turns that hair into
`SddpError::Infeasible`: a false infeasibility over a physically meaningless
quantity that aborts training outright.

`StageSolvePrep::run` therefore reconciles every pinned commitment against the
delivery generation column's **enforced** bound (`col_upper * col_scale`, the
round-tripped value the solver applies — not the template's raw `max_gen`),
relaxing the column just far enough to admit drift within `drift_margin`. Drift
beyond that margin is `SddpError::AnticipatedCommitmentOutOfBounds`, never
absorbed: the margin is the discrimination line between solver noise and a
modelling error, and a guard that relaxes for ANY overshoot silently admits a
plant generating past its cap.

Two forbidden alternatives, both of which have shipped:

- **Deleting the reconciliation on the premise that unscaling makes it
  redundant.** `apply_anticipated_col_scale_unscale` (`col_scale = 1.0` on
  `anticipated_slots_out ∪ anticipated_state`) removes the ring _carry_ drift and
  is retained — the carry is bit-exact and the decision column's own value is
  bit-exact at its bound. It cannot remove the drift the basis factorization
  introduces at the deposit row, because exactness there is the solver's to give
  and it does not give it. No amount of unscaling closes this.
- **Making the reconciliation an opt-in hook.** It is not a variation point and
  takes no parameter: `run` derives its own gate, so all four solve sites (forward,
  backward, lower bound, simulation) get it and none can opt out. An
  `Option<..Ctx>` hook threaded per call site is what let all four silently lose it
  in one commit.

Read: `lp/builder/commitment_reconcile.rs` (`reconcile_commitment`,
`fill_bound_relaxations`, `drift_margin`), `training/stage_solve_prep.rs`
(`StageSolvePrep::reconcile_commitments`), `lp/builder/scaling.rs`
(`apply_anticipated_col_scale_unscale`). Pinned by
`anticipated_commitment_drifted_over_cap_is_absorbed` (a seed a hair past the cap
trains; it returns `Infeasible` the moment the reconciliation is disabled) and
`anticipated_commitment_over_cap_seed_is_refused` (a genuine over-commitment is
named, not absorbed). `anticipated_commitment_at_cap_survives_ring_carry` does
NOT pin this contract and must never be mistaken for it: a seed exactly at the cap
carries zero drift, never reaches the reconciliation, and stays green with the
guard deleted — an at-cap-only suite is what let this regression ship.
