# `crates/cobre-sddp/src/lp/indexer/` — Design Analysis (Simplification Lens)

**Scope:** the LP index-map submodule of `cobre-sddp` (`crates/cobre-sddp/src/lp/indexer/`),
centred on `StageIndexer`, plus its direct owners (`setup::StageData`, `StageLayout`) and
consumer paths (cut/state-vector path, matrix-fill/geometry path, simulation extraction).
**Question (the lens):** is the machinery proportionate to the problem, and what is the
_cleanest decomposition_? This is deliberately a **different** question from the prior
degradation assessment's anti-drift lens ("where can silent breakage slip in?") — an
anti-drift optimizer can only ever answer "add enforcement," so it cannot see
"untangle / remove." This pass asks the parsimony question that one did not.
**Method:** first-hand reads of the indexer sources + an empirical consumer map across
`cobre-sddp/src` and `tests/` (construction sites, owners, per-field read frequency, the
`StageIndexer`↔`StageLayout` boundary).
**Status:** analysis only — no code changed. Point-in-time snapshot; **symbol anchors are
stable, line anchors will drift.**
**Date:** 2026-06-20.

This is the **fourth pass** on this cluster, and the first scoped to the indexer with a
simplification lens:

1. `lp-construction-bloat-analysis.md` — "is there removable bloat?" → ~85–90% essential.
2. `lp-layout-architecture-spike.md` — "redesign the `StageIndexer`/`StageLayout` split?"
   → no; flagged a latent correctness bug.
3. `lp-architecture-degradation-assessment.md` — "what degradations remain?" →
   "convention → construction" upgrades; called the `StateLayout`/`StageStructure` split
   _unjustified_.
4. **This doc** — "is `StageIndexer` the cleanest possible design, and if not, what is the
   untangling?" It **partially revises** pass 3 (§5).

---

## 0. Verdict up front

`StageIndexer` is **not the best-possible design and the architecture around it is not
clean.** The gap is real, but its root is the opposite of what the "over-engineered" feeling
suggests:

> The root cause is **under-separation plus a half-finished per-stage migration — not
> over-abstraction.** The things that _feel_ like over-engineering (a test-only second
> construction mode, a shadow recovery API, ~30 repeated "empty when …" field caveats) are
> **symptoms** of two distinct concerns being welded into one type while a refactor that
> pulls geometry out to per-stage owners is left partly done.

**The recurrence is itself the diagnosis (owner's read).** This pass independently surfaced a
_third_ instance of one root cause — global stage-0 geometry used where per-stage geometry is
required — after the NCS bound-patch and the simulation-extraction strides: the anticipated
cut-column (§7.2). A sound architecture forecloses a bug _class_ on first contact; one that lets
the same class recur three times — silently, reachable from ordinary input, invisible to the
test suite, and (per §7.3) even after the exact hazard was written into a doc comment for a
sibling column family — is itself evidence that the global-vs-per-stage split is a **structural
defect**, not a run of independent oversights. The cost of the current design is paid in
**latent correctness bugs**, not merely readability. The remediation preference recorded
throughout this doc is therefore **structural — remove the ambiguity by construction, accepting
a larger blast radius — over point-patching each instance as it surfaces** (point-patching is
what produced three). This is a long-term-maintainability objective, not a quick-result one.

The clean move is therefore to **separate** (likely two smaller single-purpose types) and
to **finish** the per-stage migration so no global-vs-per-stage ambiguity remains. Net
source surface goes **down** — the dual mode, the recovery accessors, the redundant
per-stage recompute, and the caveat layer all disappear — even though the type _count_ may
rise from one to two. "Simplify a lot" here means **untangle**, not "fewer types." None of
the sacred numerical contracts (`.claude/rules/sddp.md`) are touched.

---

## 1. What `StageIndexer` is (literally)

One struct (`indexer/layout.rs`) with roughly **60 fields**, overwhelmingly `Range<usize>`
— one per LP column-family and row-family — plus scalars, index `Vec`s, one reverse
`HashMap`, and two cut-path caches. It carries an explicit
`#[allow(clippy::struct_excessive_bools)]` (four independent presence flags).

| Group              | Representative fields                                                                                                 | Kind             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------- | ---------------- |
| State-vector core  | `storage`, `inflow_lags`, `storage_in`, `theta`, `n_state`, `anticipated_state`                                       | ranges + scalars |
| Equipment columns  | `turbine`, `spillage`, `diversion`, `thermal`, `line_fwd/rev`, `deficit`, `excess`, `anticipated_decision/_state_out` | ranges + counts  |
| Rows               | `water_balance`, `load_balance`, `z_inflow_rows`, `min_*_rows` (×4), `anticipated_fishing`                            | ranges           |
| Optional slacks    | `inflow_slack`, `withdrawal_slack_{neg,pos}`, `{outflow,turbine,generation}_*_slack`                                  | ranges           |
| FPHA / evaporation | `generation`, `fpha_rows`, `evap_indices` (+ index vecs)                                                              | ranges + `Vec`   |
| Presence flags     | `has_inflow_penalty`, `has_withdrawal`, `has_operational_violations`, `has_ncs`                                       | `bool`           |
| Cut caches         | `nonzero_state_indices`, `state_to_lp_column_map`                                                                     | `Vec`            |

Two construction modes exist: `StageIndexer::new(N, L)` (state-only; leaves ~two-thirds of
the fields degenerate `0..0` / `0`) and `with_equipment[_and_evaporation]` (full).

---

## 2. What it is (empirically) — role from its consumers

Production builds a `StageIndexer` in **exactly two places**, both via
`with_equipment_and_evaporation`:

- `setup::build_wired_indexer` → the global/stage-0 instance held in `StageData.indexer`.
- `StageLayout::new` → a **fresh per-stage** instance, rebuilt every stage (it does **not**
  derive from the global one — it recomputes the whole struct from per-stage counts).

The two instances are read by **disjoint** consumer sets along a real seam:

| Instance                          | Built     | Lifetime  | Consumed by                                                                            | Load-bearing fields (read frequency)                                                                                                |
| --------------------------------- | --------- | --------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `StageData.indexer` (global)      | once      | whole run | **cut / state-vector path** (`cut/row.rs`, `cut/dcs.rs`, `cut/wire.rs`, `cut_sync.rs`) | `n_state` (~284), `state_to_lp_column[_map]` (~44), `storage_in` (~32), `inflow_lags`, `anticipated_state`, `nonzero_state_indices` |
| `StageLayout.indexer` (per-stage) | per stage | per stage | **matrix-fill / geometry path** (`builder/*`, `generic_constraints`, `extraction`)     | `n_blks` (~161), `thermal` (~147), `generation` (~87), `block_grid()`, `deficit`                                                    |

The split is not incidental: the global instance's load-bearing fields are
**stage-invariant** (the state dimension does not change across stages — one copy suffices),
while the per-stage instance's load-bearing fields are exactly those that **do** vary per
stage (which is why `StageLayout` rebuilds rather than shares).

---

## 3. Findings (simplification defects)

### F1 — Two responsibilities fused into one type (SRP), provable from consumers

`StageIndexer` is two cohesive types in a trench coat: (a) a **stage-invariant state-vector
layout + resolver** (`n_state`, `storage_in`, `inflow_lags`, `anticipated_state`,
`state_to_lp_column[_map]`, `state_to_lp_incoming_column`, `nonzero_state_indices`) and
(b) a **per-stage LP geometry map** (the ~40 equipment/slack/row ranges, `n_blks`,
`block_grid()`). The two halves have **disjoint consumers, different owners, and different
lifetimes** (§2). That is the definition of two types, not one.

### F2 — A whole branch of the contract exists only for tests

`StageIndexer::new` (the state-only mode) has **zero production callers** — every
`StageIndexer::new(` site across the crate is inside a `#[cfg(test)]` module. The ~30 field
docs each repeating _"Empty (`0..0`) when built via `StageIndexer::new`"_ and the
`#[allow(clippy::struct_excessive_bools)]` are tax the **production** type pays for a
**test-fixture** convenience. A clean type does not bake a test-only mode into its documented
production contract; this belongs in a fixture builder.

### F3 — The "single source of truth" is not single

There is no single instance: the same constructor runs once globally and once per stage,
producing parallel objects. The per-stage instance **recomputes the stage-invariant half
from scratch** (redundant representation + redundant startup work), and — worse — geometry
callers and state callers read **different instances under the same type name**. That
ambiguity is the structural generator of the realized silent-wrong-output bug class (a
global stage-0 range strided per stage). A genuine SSOT cannot produce that failure mode by
construction; this one did.

### F4 — The representation generates corrective machinery

Empty ranges are normalised to `0..0`, which discards the cursor (where the block _would_
start). A shadow API exists purely to recover it: `generation_col_start`, `evap_col_start`,
`post_equipment_col_start`, `fpha_rows_end`, `post_equipment_row_start`. Code whose only job
is to undo a sibling field's representation choice is dead weight a cleaner representation
(carry the cursor; do not collapse) would delete.

### F5 — The type is caught mid-migration, and the half-done state is the hazard

Geometry is migrating out of the indexer family-by-family, leaving scars: `has_ncs` is a
bare presence flag (NCS column base already moved to `StageContext::ncs_col_starts`), and
`EquipmentCounts::n_pumping` is _"accepted for structural symmetry … but not read"_ (pumping
geometry owned by `StageLayout`). The inconsistency **between** what has migrated and what
has not is exactly where silent bugs live. Finishing the migration (all per-stage geometry
owned per-stage; the global instance carries only the state-vector half) removes the
ambiguity rather than documenting around it.

---

## 4. What is **not** a defect (do not over-correct)

- **The state-pinning resolver is correct and must be preserved.** `state_mapping.rs`
  (`state_to_lp_incoming_column` / `state_to_lp_column`) carries the `sddp.md` column-bound
  pinning + subgradient contract. The problem is that it is _tangled with_ geometry, not
  that it is wrong. Any separation must preserve its semantics byte-for-byte.
- **The precomputed caches are a legitimate hot-path optimization.**
  `state_to_lp_column_map` trades memory for speed on the cut-row hot path; not bloat.
- **The entity-family count is irreducible.** A layout map of _some_ size is intrinsic to a
  model with this many column/row families; the type is large because two real types + a
  test mode are stacked, not because it is gold-plated.
- **Per-stage geometry is a real requirement**, not speculative generality (commissioning
  windows, anticipated-decision gating, per-stage active generic constraints, FPHA hydro
  selection all vary geometry per stage).

---

## 5. Engagement with the degradation assessment (pass 3)

**Agree:** the domain complexity is irreducible; the dual built-in/generic-constraint path
is correctly two doors into one room; the `sddp.md` numerical contracts are sacred.

**Revise:** pass 3 declared the `StateLayout`/`StageStructure` split **"unjustified."** Its
stated reason was that per-stage-varying geometry did not really exist — a claim it then
**retracted** in its own §3 ("per-stage-varying geometry already exists today"). The
consumer evidence here makes the split's case stronger than pass 3 credited: the two roles
already have **different owners, different lifetimes, and non-overlapping consumer sets**
(§2). The split is therefore not a speculative new abstraction — it is **naming a seam the
code has already torn along**, and it _removes_ machinery (the dual mode, the recovery API,
the redundant recompute), so it reads as simplification, not addition. Pass 3's anti-drift
lens structurally could not surface this; the parsimony lens does.

---

## 6. Direction (target shape — a shape, not a plan)

1. **Separate the two concerns.** A small stage-invariant _state-vector layout_ type
   (owning `n_state`, the incoming/lag/anticipated-state ranges, the resolver, and the cut
   caches) distinct from a _per-stage geometry layout_ (the equipment/slack/row ranges,
   `n_blks`, `block_grid()`). The global `StageData` instance carries only the former; the
   per-stage `StageLayout` owns the latter.
2. **Demote the test-only mode** to a fixture builder; delete the ~30 "empty when `new`"
   caveats and the dual-mode contract from the production type.
3. **Finish the per-stage migration** so geometry has exactly one (per-stage) owner — which
   discharges F3/F5 and lets the F4 recovery accessors be deleted.
4. **Guardrail (anti-simplification):** preserve `state_to_lp_incoming_column` semantics
   exactly; verify every step **hash-neutral** on the uniform-count deterministic cases (the
   established `COBRE_PARITY_REGEN` net) before trusting it.

What this explicitly does **not** include: collapsing the built-in/generic dual path
(correctly two); weakening any `sddp.md` contract; or a user-blocking validation guard.

---

## 7. Per-file analysis (the living part of this doc)

| Indexer file                                                     | Status               | Note                                                                                                 |
| ---------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------- |
| `layout.rs` (`StageIndexer` struct + accessors)                  | **analyzed**         | §1–§4                                                                                                |
| `state_mapping.rs` (resolver / pinning contract — role (a))      | **analyzed**         | role (a) resolver — cleanest seam; see §7.1                                                          |
| `constructors.rs` (the three constructors + range build helpers) | **analyzed**         | dual-mode + redundant recompute; NCS lesson not applied → §7.3                                       |
| `anticipated.rs` (per-stage anticipated iterators/predicates)    | **analyzed**         | confirmed latent bug + dead iterator → §7.2                                                          |
| `sparse_state.rs` (`set_nonzero_mask`)                           | **analyzed**         | clean role-(a) member; fully stage-invariant → §7.4                                                  |
| `block_grid.rs` (`BlockGrid` primitive)                          | analyzed (cross-ref) | see the whole-`lp` simplification notes: high ceremony-to-arithmetic ratio; defensible but secondary |

### 7.1 `state_mapping.rs` — role (a), the state↔column resolver

**What it is.** An `impl StageIndexer` block in its own file (not a separate type) holding
four methods: `state_to_lp_column` (outgoing state-vector index → the cut-row column used in
forward-pass coefficient construction), `state_to_lp_incoming_column` (state-vector index →
the incoming column pinned by `set_col_bounds` and read back for the cut subgradient),
`finalize_state_column_map` (precompute the outgoing map into the `state_to_lp_column_map`
cache), and `lp_column_for_state` (cache reader with a live-resolver fallback). The file is
~167 LOC of production code (≈110 of it doc comment) and ~500 LOC of tests (≈73% of the file).

**Two mappings, an irreducible asymmetry.** The _incoming_ resolver is a straight range→range
offset (storage→`storage_in`, lag→`inflow_lags`, anticipated→`anticipated_state`). The
_outgoing_ resolver is materially more complex because it encodes the **state transition**:
lag-0→`z_inflow`, lag≥1→previous lag, and the anticipated ring-buffer shift (Equal→
`anticipated_state_out`, Less→shifted slot, Greater→identity padding). That complexity is
domain-essential — the forward-pass cut row references next-stage state columns, so the shift
logic must live somewhere.

**Heaviness here is EARNED — the contrast with `BlockGrid`.** Three things look like the same
ceremony flagged elsewhere but are justified: (i) the ~20-line padding-slot 5-step contract
comment is load-bearing (it justifies the `Greater => j` identity that would otherwise read
as a bug); (ii) the dual mappings are irreducible; (iii) the ≈73% test ratio is appropriate
for a `sddp.md`-sacred contract whose deviations are silent-wrong. This is the inverse of
`BlockGrid`'s ceremony-to-arithmetic ratio: comparable surface weight, but here it guards
genuinely subtle, silently-failing arithmetic.

**SoC verdict — the cleanest concern in the indexer, and almost cleanly separable.** This
file is the strongest evidence that role (a) is real and cohesive: it is already
file-isolated, single-responsibility, owns a named contract, and `state_to_lp_incoming_column`
reads **only stage-invariant state-vector fields** (`hydro_count`, `max_par_order`,
`storage_in.start`, `inflow_lags.start`, `anticipated_state.start`). The only thing keeping it
from being a standalone `StateLayout` type is that those fields live on the 60-field struct.
Its sole sin is being an `impl` block on a too-large host.

**The one wrinkle that complicates the clean split (load-bearing).** The _outgoing_ resolver's
anticipated branch returns `anticipated_state_out.start + plant`, and `anticipated_state_out`
lives in the **per-stage control/equipment region** (its offset depends on `n_blks` and
`n_thermals`). So `state_to_lp_column` is **not** purely stage-invariant: for anticipated
thermals it couples the state-resolver concern to per-stage geometry. Two consequences:

- A `StateLayout` extraction is ~90% clean (storage / lag / the entire incoming resolver lift
  freely) but must handle this one coupling — import the `anticipated_state_out` base, or
  thread it in.
- It is a **latent-ambiguity smell of the F3/F5 family**: the cut path reads
  `state_to_lp_column_map`, finalized once from the **global stage-0** indexer in
  `build_wired_indexer`. If anticipated thermals co-occur with per-stage-varying block counts,
  the cached stage-0 `anticipated_state_out.start` would mis-address later stages — the same
  failure pattern as the NCS bug. **Now confirmed (§7.2): a real latent correctness bug** (no
  shipped case is known to exercise anticipated thermals + varying block counts together).

**Minor smell (F2 family).** `lp_column_for_state` carries a hot-path `unwrap_or_else(live
resolver)` fallback whose stated purpose is to tolerate _un-finalized test indexers_, and
`finalize_state_column_map` is a deferred two-phase-init step even though the outgoing map is a
pure function of construction-time layout (it appears computable in the constructor — verify).
Both are the "production code defensively tolerates a partially-built object" pattern;
computing the cache at construction would delete the fallback branch and the finalize step.

### 7.2 `anticipated.rs` + the anticipated-thermal machinery (verified by full cut-path trace)

The predicate file is lean — `anticipated.rs` is ~67 LOC of production code (two methods,
`is_anticipated_decision_active` and the `anticipated_decision_active_at_stage` iterator) and
~90% tests. The anticipated _weight_ lives in the column/row machinery (`anticipated_decision`,
the `anticipated_state` ring buffer, `anticipated_state_out` + its definition row, and the
always-active `anticipated_fishing` rows). Three findings, in priority order.

**F-bug (CONFIRMED latent correctness bug — third instance of the F3/F5 root).** Cut rows are
baked from the **single global stage-0 `StageData.indexer`** at all four construction sites
(forward rebake `training/session/mod.rs`, simulation rebake `simulation/state.rs`, lower bound
`training/lower_bound.rs`, backward delta-cut `training/backward_pass_state.rs`). A stored cut
holds only its raw state-indexed subgradient; the state→column mapping is applied at
materialization via `state_to_lp_column` / the stage-0-finalized `state_to_lp_column_map`. For
the matured anticipated slot the Equal branch returns `anticipated_state_out.start + plant`, and
`anticipated_state_out.start = thermal_start + n_thermals·n_blks + n_anticipated` —
**`n_blks`-dependent**. So when a stage's `n_thermals·n_blks` differs from stage 0's, the
(generically non-zero, in-mask) anticipated cut coefficient is written to the **wrong column** of
that stage's LP → silently wrong cuts (wrong policy/bounds), compiling and passing the suite.

> **The crystallizing observation:** the matured anticipated slot is the **lone state dimension
> whose cut-mapping column lives in the per-stage geometry region.** Storage maps identity; lags
> map to `z_inflow`/incoming-lag columns; the incoming pin/dual read uses `anticipated_state`
> (all in the `n_blks`-independent state region). Only `anticipated_state_out` reaches into the
> equipment region. **An SoC violation (a state concern reaching into the geometry region)
> produces the correctness bug directly** — this is F1/F3/F5 made concrete, the third sibling of
> the NCS bound-patch and simulation-extraction instances documented in pass 3 §1.

Dormant only because no shipped input combines anticipated thermals with non-uniform block
counts (no deterministic example uses anticipated thermals at all; `n_blks` _is_ allowed to vary
— `d33` ships `[1, 3, 2]`), and **no validator forbids the combination.** **Preferred fix (owner, deferred until the current plan completes): option (B)** — relocate
`anticipated_state_out` into the `n_blks`-independent state region so the global cut-map is
correct **by construction**. Chosen deliberately over the lighter (A) per-stage cut-column patch
(which would mirror the NCS `ncs_col_starts[t]` fix): the goal is to remove the ambiguity
structurally, not to add a third per-stage patch to a design that keeps regenerating this bug
class. A larger blast radius is accepted in exchange for long-term maintainability. (C) a
blocking validator is rejected by the project stance (pass 3 §1.4: complete support, don't
forbid input); (D) record-only is the interim state. **No code change now — recorded and
deferred.**

**F-dead (CONFIRMED dead production code).** `anticipated_decision_active_at_stage`
(`anticipated.rs:27`, the iterator) has **zero non-test callers** — production uniformly uses the
per-plant `is_anticipated_decision_active`. It survives only because it is `pub`, so the
`dead_code` lint never fires — the same visibility-masks-dead-code mechanism as the test-only
`StageIndexer::new` (F2). Safe to delete or `#[cfg(test)]`-gate.

**F-redundant (`anticipated_state_out` + its definition row — strongest simplification, but
escalate).** The trace confirms `anticipated_state_out` is a **pure alias** of
`anticipated_decision`: its def row is the identity `state_out[i] − decision[i] = 0`, it has no
objective coefficient, no other matrix entries, and is read nowhere in `training/`/`cut/` except
the Equal branch. The leanest design routes the Equal branch straight to
`anticipated_decision.start + plant`, deleting A columns + up to A equality rows **per stage**.
**But this is not a mechanical edit:** collapsing puts a θ-bearing future-cost coefficient onto a
column that also carries bound-driven activation, an NPV objective term, generic-constraint
coefficients, and per-stage-flipping bounds. Whether the combined column's reduced cost still
yields the correct Benders subgradient and a stable warm-start basis (`reconstruct_basis`,
slot identity) is exactly the `sddp.md`-class hazard. **Requires an SDDP reduced-cost/basis proof
(`/assess`), not a refactor.** Note it does **not** discharge F-bug on its own — `decision.start`
is _also_ `n_blks`-dependent; the per-stage-vs-global fix is orthogonal.

**Update (sddp-specialist verdict).** The subgradient is harvested from the **incoming ring-buffer
column** (`state_to_lp_incoming_column`), not `state_out` — so `state_out` is a cut _target_, not a
cut _source_; removing it threatens only basis stability, not subgradient validity. Verdict:
**relocation (option B) strictly dominates collapse** — collapse neither fixes F-bug nor preserves
the free-carrier basis uniformity (it risks dual-vertex drift + `solve_with_basis` rejections when
the future-cost gradient lands on the bound-active, NPV-costed `decision` column). So `state_out`
should **move** into the state region, not be deleted; the `StateLayout` extraction (§8) is the
vehicle, and the "delete A cols + A rows" leanest-design line above is **superseded** by relocation.
See `lp-builder-simplification-assessment.md` §2.8 for the full formulation rationale (delivery
obligation vs consumed coefficient).

**Minor.** A third copy of the `stage + K_i < n_stages` activation rule is inlined at
`simulation/extraction.rs` (comment claims it "matches" the predicate) instead of calling the
single owner — a drift seam. `anticipated_local_by_sys_pos` is live (generic-constraint resolver,
build-time) but built unconditionally — mild over-eagerness, not dead. The lag-shift vs
anticipated-shift logic is **not** duplicated (genuinely distinct semantics — leave separate).

### 7.3 `constructors.rs` — the three constructors + range helpers

**Shape.** ~640 LOC production (≈67% tests). Two free range helpers
(`build_inflow_slack_range`, `build_oper_violation_ranges`), two build helpers
(`build_fpha_rows`, `build_evap_indices`), and three constructors:
`with_equipment_and_evaporation` (~259 LOC — the real one, a one-shot sequential offset chain),
`with_equipment` (a 1-line wrapper), and `new` (~181 LOC).

**`new`'s dual role; two more pub-masks-unused items.** `new` is both the external state-only
mode (test-only externally, per §1/F2) **and** the internal zero-initializer:
`with_equipment_and_evaporation` calls `Self::new(..)` as `base` and inherits
`storage`/`inflow_lags`/`hydro_count`/`max_par_order`/`nonzero_state_indices` via `..base`. So
`new` is not dead, but its _public dual-mode_ contract is test-only. `with_equipment` is a thin
public wrapper with **zero production callers** — a third pub-masks-unused item alongside the
test-only external `new` (F2) and the dead `anticipated_decision_active_at_stage` iterator
(§7.2 F-dead). Three `pub` items in the indexer survive only because visibility suppresses the
`dead_code` lint.

**Redundant state-vector recompute.** The full constructor calls `new` (which lays out
`z_inflow`/`storage_in`/`theta`/`n_state` for the _no-anticipated_ layout) and then recomputes
all four locally to absorb the `anticipated_state` shift, overriding `base`'s. For the common
no-anticipated case (`n_ant_state == 0`) the recomputes are byte-identical to what `base` already
holds — compute-twice-discard-once. Startup-only (not a perf issue); a clarity/redundancy smell
and a sibling of F3's per-stage recompute.

**The decisive evidence for the architecture critique (§0).** The constructor doc carries an
~18-line block stating the NCS lesson precisely: NCS columns must **not** be anchored at a global
stage-0 base because that base "assumed stage 0's block count, landing the bound patch on the
wrong columns for any stage whose ... block count" differs. A few lines of code later it places
`anticipated_decision` and `anticipated_state_out` at exactly such a base
(`thermal_end = thermal_start + n_thermals·n_blks`), feeding the §7.2 F-bug. **The exact hazard
analysis was written down for one column family and not applied to the two sitting in the same
region.** Nothing structural connects "this column lives in the `n_blks`-dependent region" to
"its global cut-mapping is therefore unsafe," so the knowledge stayed inert and the bug shipped.
This is the clearest single illustration of §0's owner read: a sound design would make this a
compile-time impossibility, not a doc comment one family obeys and two ignore.

**Not smells (calibration).** The `#[allow(clippy::too_many_lines)]` on the full constructor is
earned — a sequential offset chain where each range derives from the previous reads better whole,
and four sub-regions are already extracted to helpers; do not split it further. The
`if n > 0 { range } else { 0..0 }` normalization, repeated per optional block, is the **root of
F4** (it discards the cursor, forcing the `*_col_start` recovery accessors); it is the one
representation choice worth revisiting in the redesign.

**Residual.** `// Per Decision 3` (~line 441) is a plan-token leak (N4), already on the
residual-drift cleanup list.

### 7.4 `sparse_state.rs` — `set_nonzero_mask` (the cut-sparsity cache)

**Shape.** ~127 LOC production (≈60 doc), ≈70% tests. One production method, `set_nonzero_mask`,
plus a correctly `#[cfg(test)]`-gated `finalize_for_test` helper. It computes
`nonzero_state_indices` — the state-vector indices that may carry non-zero cut coefficients —
which drives the mask-driven cut-row hot path (`build_cut_row_batch_into`). A legitimate
performance cache (sparse iteration over state dims), not bloat; the heavy test ratio is
**earned** (it pins the PAR(p)-A padding-exclusion contract whose violation over-estimates cuts →
LB > UB, a real past bug).

**Cleanest confirmation of the role-(a) bundle.** `set_nonzero_mask` reads **only stage-invariant
state-vector fields** (`hydro_count`, `max_par_order`, `inflow_lags.start`, `k_max`,
`n_anticipated`, `anticipated_state.start`) — notably `anticipated_state` (the stage-invariant
incoming ring buffer), **not** `anticipated_state_out` (the per-stage column behind the §7.2 bug).
So this file is 100% within role (a), with none of the §7.1 anticipated wrinkle. Together with
`state_mapping.rs` (resolvers), the two caches (`state_to_lp_column_map`, `nonzero_state_indices`),
and their two post-construction finalizers (`finalize_state_column_map`, `set_nonzero_mask`), it
forms a cohesive, fully stage-invariant state-vector/cut concern — the concrete extraction target
for the `StateLayout` type proposed in §6.

**F2 again (deferred finalization + dense fallback).** The mask is empty by default and populated
post-construction in `build_wired_indexer`; the cut-row consumer treats an empty mask as the
"dense path" (all `n_state` indices). That dense fallback exists to tolerate un-finalized test
indexers — the same partially-built-object tolerance as `lp_column_for_state` (§7.1). Computing
both caches at construction would remove the two-phase init and the dense-fallback branch.

**Positive contrast (the model the `pub` items should copy).** `finalize_for_test` is correctly
`#[cfg(test)]`-gated — exactly the discipline the three `pub`-masks-dead-code items (`new`'s
external mode, `with_equipment`, the `anticipated_decision_active_at_stage` iterator) fail to
follow. The codebase knows the right pattern; those three are the exceptions.

**D2 (minor).** The lag (`inflow_lags.start + lag*hydro_count + h`) and anticipated
(`anticipated_state.start + slot*n_anticipated + plant`) strides are hand-rolled here, as at other
sites — a different stride family from `BlockGrid`'s `n_blks` block-major shapes (state-vector
strides, not equipment strides), so `BlockGrid` does not cover them.

### 7.5 Indexer synthesis (all files analyzed)

The indexer cluster is now fully mapped. The through-line:

- **Role (a) — state-vector / cut concern (stage-invariant, cohesive, cleanly extractable):**
  `state_mapping.rs` (§7.1) + `sparse_state.rs` (§7.4) + the `n_state` / state-region ranges + the
  two caches + their two finalizers. Reads only stage-invariant fields — the lone exception is
  `state_to_lp_column`'s anticipated branch reaching `anticipated_state_out` (§7.1), which is
  exactly the §7.2 bug. This bundle is the `StateLayout` the redesign should lift out.
- **Role (b) — per-stage LP geometry:** the ~40 equipment/slack/row ranges built by
  `constructors.rs` (§7.3) and rebuilt per stage by `StageLayout`. This is where the
  `n_blks`-dependent offsets live, the F4 normalization root, and the §7.2/F-bug region.
- **The fusion of (a) and (b) into one 60-field struct, plus the test-only public dual-mode, is
  the indexer's central defect** — and it is the direct cause of the recurring global-vs-per-stage
  bug class (§0).

Net: the indexer needs **no new abstraction**; it needs **separation (a from b) and completion of
the per-stage migration**, both of which _remove_ code. `BlockGrid` (cross-ref) is the one piece
that _added_ machinery; §6 judges its cost/benefit secondary.

---

## 8. `StateLayout` extraction — concrete design (target for the post-plan redesign)

Turns §6/§7.5 from a shape into a type boundary. **Design only; deferred until the in-flight
plan completes; no code now.** Field assignments are grounded in a consumer grep of the
boundary fields.

### 8.1 The boundary — what moves where

**`StateLayout`** (built once, stage-invariant) — the role-(a) bundle:

| Member                                                              | Kind       | Why role (a)                                          |
| ------------------------------------------------------------------- | ---------- | ----------------------------------------------------- |
| `storage`, `inflow_lags`, `anticipated_state`, `storage_in`         | col ranges | state-region columns; pinning + cut targets           |
| `z_inflow`                                                          | col range  | `state_to_lp_column` maps outgoing lag-0 here         |
| `anticipated_state_out`                                             | col range  | **after relocation (§8.3)**; cut target, matured slot |
| `theta`                                                             | usize      | future-cost column (cut-referenced) + control marker  |
| `n_state`, `hydro_count`, `max_par_order`, `n_anticipated`, `k_max` | scalars    | state-vector dimensions                               |
| `anticipated_lead_stages`                                           | Vec        | per-plant K_i; drives ring occupancy + mask           |
| `nonzero_state_indices`, `state_to_lp_column_map`                   | caches     | cut-sparsity + column precompute                      |

Methods: `state_to_lp_column`, `state_to_lp_incoming_column`, `lp_column_for_state`,
`set_nonzero_mask`, `finalize_state_column_map`, and `control_region_start()` (= `theta + 1`).

**Geometry layer** (built per stage, in `StageLayout`) — role (b): the ~40 equipment/slack/row
ranges (`turbine`…`excess`, `anticipated_decision`, the slacks, `generation`, `evap_indices`,
`water_balance`, `load_balance`, `z_inflow_rows`/`z_inflow_row_start`, `fpha_rows`, `min_*_rows`,
`anticipated_fishing`, the `anticipated_state_out_def` row), the
`n_blks`/`n_thermals`/`n_lines`/`n_buses`/`max_deficit_segments` counts, the `has_*` flags, and the
extraction/generic-constraint identity maps `anticipated_thermal_indices` +
`anticipated_local_by_sys_pos`.

Grep-confirmed assignment of the non-obvious fields:

- `theta` — read by the cut path (`cut/row`, `cut/dcs`) **and** geometry build (`constructors`,
  `builder/layout`) → owned by `StateLayout`; geometry reads `control_region_start()`.
- `z_inflow` **columns** (role a) vs `z_inflow_rows`/`z_inflow_row_start` (role b — consumed by
  `PatchBuffer` + row-fill, never the cut path) → **split between the two layers**.
- `anticipated_lead_stages` — read by role a (`state_mapping`, `sparse_state`) **and** role b
  (`template`, `extraction`, the predicates) → owned by `StateLayout`; geometry reads it there.
- `anticipated_thermal_indices` / `anticipated_local_by_sys_pos` — sole production readers are
  simulation extraction and the generic-constraint resolver → role (b).

### 8.2 The dependency is one-way (why the cut is clean)

After §8.3 the graph is acyclic and single-directional: **geometry → `StateLayout`** (geometry
reads `control_region_start()`, `n_anticipated`, `k_max`, `anticipated_lead_stages`);
`StateLayout` reads nothing from geometry. That is the property the fused struct lacks today: the
state resolver currently reaches _into_ the geometry region (`anticipated_state_out`), making the
dependency cyclic — which is the §7.2 bug.

### 8.3 The one seam — and why the extraction IS the bug fix

The lone a→b reach is `state_to_lp_column`'s Equal branch returning `anticipated_state_out.start`,
a control-region (per-stage, `n_blks`-dependent) column. Two resolutions:

- **Thread it in (hash-neutral):** pass `anticipated_state_out_start` into `state_to_lp_column` /
  its cache build, sourced per stage. Keeps the column in place; the cache becomes per-stage. This
  is §7.2 option (A) — mechanical, hash-neutral, but leaves the per-stage-cache fragility.
- **Relocate it (owner's choice, option B):** move `anticipated_state_out` into the stage-invariant
  state region so `StateLayout` owns it. The seam **vanishes**, the dependency becomes one-way, and
  the §7.2 bug is fixed **by construction** (the column is no longer `n_blks`-dependent). Under
  option B the `StateLayout` extraction and the bug fix are **a single operation**: the clean type
  boundary _requires_ the column to be stage-invariant, and making it stage-invariant _is_ the fix.

Relocation keeps a dedicated state-out column (preserving the warm-start-basis uniformity argued in
§7.2 F-redundant) rather than collapsing it onto `anticipated_decision` (which would not fix the
bug — `decision` is also `n_blks`-dependent — and is basis-unproven). So option B supersedes the
F-redundant simplification.

### 8.4 What the extraction deletes

- the test-only public dual-mode (`new`) and the `with_equipment` wrapper — `StateLayout` has one
  constructor, geometry has its own (F2 + §7.3);
- the ~30 "empty when built via `new`" caveats on the state fields;
- the empty-range **recovery accessors** for state ranges (F4) — `StateLayout` carries real
  cursors, not collapsed `0..0`;
- the **redundant state-vector recompute** in `with_equipment_and_evaporation` (§7.3) — geometry no
  longer rebuilds the state half per stage;
- the **two-phase init + dense fallback** (F2) — `StateLayout` finalizes both caches in its
  constructor (each is a pure function of the layout), so `lp_column_for_state`'s fallback and the
  empty-mask dense path can go.

Net: a _smaller_ codebase with a _new_ type — separation, not addition.

### 8.5 Owners and construction flow

- `StageData.indexer: StageIndexer` → `StageData.state: StateLayout` (the cut path consumes role
  (a) only; audit any residual global geometry read of `StageData.indexer` first).
- `StageLayout` stops embedding a freshly-rebuilt full `StageIndexer`; it holds the per-stage
  geometry + a handle to the shared `StateLayout` (for `control_region_start()` + anticipated
  metadata). The redundant per-stage state recompute disappears.

### 8.6 Verification & sequencing (honest)

Two regimes, because the two moves carry different risk:

1. **Type extraction, no column move** — mechanical; **hash-neutral**; verify on the uniform-count
   D-cases via `COBRE_PARITY_REGEN`.
2. **`anticipated_state_out` relocation (option B)** — a column-layout change → **not hash-neutral**
   (re-baseline required) and gated on **SDDP-specialist sign-off** for reduced-cost / warm-start-
   basis correctness (the §7.2 caveat), plus the new deterministic case combining anticipated
   thermals + per-stage-varying block counts.

Sane order: do (1) first (safe, immediate cleanup), then (2) to close the seam and fix the bug.
Both **after** the in-flight plan completes.

---

## 9. Key symbols

- `crates/cobre-sddp/src/lp/indexer/layout.rs` — `StageIndexer`, `EquipmentCounts`
  (`n_pumping` unread), the empty-range recovery accessors (`generation_col_start`,
  `evap_col_start`, `post_equipment_col_start`, `fpha_rows_end`, `post_equipment_row_start`).
- `crates/cobre-sddp/src/lp/indexer/state_mapping.rs` — `state_to_lp_incoming_column`,
  `state_to_lp_column`, `finalize_state_column_map` (role (a), the sacred contract).
- `crates/cobre-sddp/src/lp/indexer/sparse_state.rs` — `set_nonzero_mask`.
- `crates/cobre-sddp/src/setup/mod.rs` — `build_wired_indexer` (the one global construction
  site; sets `has_ncs`, finalizes the state map).
- `crates/cobre-sddp/src/setup/stage_data.rs` — `StageData.indexer` (global owner).
- `crates/cobre-sddp/src/lp/builder/layout.rs` — `StageLayout` (per-stage owner; `indexer`
  field; rebuilds per stage; delegating `col_*` accessors).
- `crates/cobre-sddp/src/cut/{row,dcs,wire,cut_sync}.rs` — the state-vector consumers
  (read `n_state`, `state_to_lp_column[_map]`).
- `docs/design/lp-architecture-degradation-assessment.md` — pass 3 (engaged in §5).
