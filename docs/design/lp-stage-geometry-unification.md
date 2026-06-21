# `crates/cobre-sddp/src/lp/` — Post-Redesign Assessment: Collapsing the Duplicated Layout Bags

> **Status: design approved (disjoint-owner model). No code yet beyond this
> blueprint.** Post-redesign continuation of
> `lp-architecture-degradation-assessment.md` (§D1, §6, §7) and
> `lp-indexer-simplification-assessment.md` (§8). The role-(a) `StateLayout`
> extraction, the `anticipated_state_out` relocation, and the per-stage
> simulation-extraction base fix are **done**. This doc specifies the next
> structural move: collapse the duplicated layout bags so **every layout fact has
> exactly one owner**. Grounded in a field-and-consumer trace of the live tree.

## 0. Verdict up front

The redesign cleaned role (a) (`StateLayout`) but left **two** duplication axes:

1. **Role-(b) geometry triplicated.** The ~18 equipment/slack ranges live in
   `StageIndexer` (stage-0), `StageLayout` (per-stage ephemeral), and
   `StageEquipmentGeometry` (per-stage persisted), synced by two copy
   constructors.
2. **Study-invariant scalars duplicated across 4–5 bags.** `hydro_count`,
   `n_thermals`, `n_lines`, `n_buses`, `n_blks`, `max_deficit_segments`,
   `n_anticipated`, `k_max`, `has_*` are each declared in **`EquipmentCounts`**
   (the constructor-input bag), **`StageIndexer`**, **`StateLayout`** (the state
   subset), and **`StageLayout`** (builder copies). `EquipmentCounts` is the
   input-side twin of `StageIndexer`'s persisted scalars.

The target gives each fact exactly one owner across **three disjoint persisted
owners + one ephemeral builder**, deleting `StageIndexer` and `EquipmentCounts`.
The critical structural point: **`n_blks` is per-stage, not study-invariant** —
it must cease to exist as a persisted global fact, which is what kills the
stage-0/`n_blks` defect class by construction. The move is **expected
hash-neutral**: it relocates _which type holds_ a fact, never the fact.

> A rejected first attempt introduced a new `StudyLpMeta` struct holding a subset
> of `EquipmentCounts`' fields — a _sixth_ owner of the same scalars. The lesson,
> baked into this revision: the fix is to **collapse** the existing bags into
> single owners, never to add another.

---

## 1. The cast today (post-plan)

| Type                                             | Holds                                                                                                                                                                     | Lifetime                         |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `EquipmentCounts` (`indexer/layout.rs`)          | The study-invariant scalar dims + flags + anticipated identity lists, as a **constructor-input** bag for `StageIndexer`.                                                  | Transient input                  |
| `StageIndexer` (`indexer/layout.rs`)             | Role-(b) ranges anchored at stage-0 `n_blks`; a **persisted copy** of the same scalars/flags `EquipmentCounts` carries; identity lists; row ranges.                       | Long-lived (`StageData.indexer`) |
| `StateLayout` (`indexer/state_layout.rs`)        | Role (a): state ranges, resolvers, caches **+ the state-defining dims** (`hydro_count`, `max_par_order`, `n_anticipated`, `k_max`, `anticipated_lead_stages`, `n_state`). | Long-lived; shared `&`           |
| `StageLayout<'a>` (`builder/layout.rs`)          | Borrows `&StateLayout`; **own copies** of `n_h`/`lag_order`/`n_anticipated`/`k_max`; per-stage role-(b) ranges; transient build state.                                    | Ephemeral                        |
| `StageEquipmentGeometry` (`builder/template.rs`) | Persisted snapshot of `StageLayout`'s 18 equipment ranges.                                                                                                                | Long-lived per-stage             |

Each study scalar appears in up to **five** of these. That is the duplication to
remove.

---

## 2. The two duplication axes (the defect)

### 2.1 Role-(b) geometry — three carriers, two copy constructors

The block-major equipment/slack families each have a `Range<usize>` in
`StageIndexer`, `StageLayout`, and `StageEquipmentGeometry`, kept in sync by
`StageEquipmentGeometry::from_layout` (production, per-stage) and `::from_indexer`
(test/uniform-block only). Adding or moving a family is an edit in three struct
defs + a copy constructor, none compiler-enforced.

### 2.2 Study-invariant scalars — `EquipmentCounts` is the input-side root

`EquipmentCounts` (`#[derive(Clone, Default)]`, `pub`) is passed to
`StageIndexer::with_equipment_and_evaporation`; `build_wired_indexer` then **copies
its scalars into the persisted `StageIndexer`**, `StateLayout` holds the state
subset, and `StageLayout::new` re-derives its own `n_h`/`lag_order`/… copies. So
`hydro_count` lives in four owners, `n_thermals` in two, `n_blks` in three, etc.
The scalars have no single source of truth — the §D1 prediction, on the _scalar_
axis.

### 2.3 `StageIndexer` is a grab-bag

Its production readers split into unrelated groups: stage-invariant row bases
(`water_balance.start`, `load_balance.start`, `z_inflow_row_start`,
`turbine.start` = `theta + 1`); study-wide flags/dims (several duplicating
`StateLayout`); stage-0 identity lists (`fpha_/evap_hydro_indices`,
`anticipated_thermal_indices` — a per-stage footgun where FPHA/evap membership
varies by stage); and vestigial equipment ranges (read in production only where
stage-0 == per-stage, or via the test-only `from_indexer`).

### 2.4 The `n_blks` footgun is embedded in the persisted types

`EquipmentCounts.n_blks` and `StageIndexer.n_blks` are the **global stage-0**
block count, carried as a stride next to global bases — the exact shape of the
non-uniform-block bug the extraction fix just dodged. A persisted global `n_blks`
must not survive the redesign.

---

## 3. Root cause

The long-lived context structs persist the **stage-0-global** `StageIndexer` (fed
by the equally-global `EquipmentCounts`), while the **stage-correct** geometry
(`StageLayout`) is computed at bake and dropped. Post-bake consumers that need
correct per-stage geometry were served by _adding_ a third carrier
(`StageEquipmentGeometry`) rather than making the per-stage geometry the single
persisted role-(b) owner. On the scalar axis, the same global bag
(`EquipmentCounts`) is simply copied into every consumer that wants a count.

---

## 4. Target: three disjoint owners + one ephemeral builder

Approved model: **disjoint single-ownership** — each fact owned by exactly one
type, split by concern (no shared dims bag, so the cut hot path keeps reading its
state dims directly with no indirection).

| Owner                                         | Owns — and nothing another owns                                                                                                                                                                                              | Derivation                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| **`StateLayout`** (role a)                    | State column ranges, resolvers, caches; **state-defining dims** `hydro_count`, `max_par_order`, `n_anticipated`, `k_max`, `anticipated_lead_stages`, `n_state`.                                                              | exists — unchanged                         |
| **`StudyDimensions`** (non-state study shape) | `n_thermals`, `n_lines`, `n_buses`, `max_deficit_segments`; `has_ncs`, `has_inflow_penalty`; `anticipated_thermal_indices`; `n_pumping`. Persisted once, threaded like `StateLayout`; also serves as the construction input. | **`EquipmentCounts` repurposed + renamed** |
| **`StageGeometry`** (role b, per-stage)       | Equipment/slack **column ranges + row ranges + per-stage identity lists** (`fpha_/evap_hydro_indices`, `evap_indices`) + **per-stage `n_blks`**. Persisted as a per-stage `Vec`.                                             | **`StageEquipmentGeometry` widened**       |
| **`StageLayout<'a>`** (ephemeral builder)     | Transient build state; **borrows** `&StateLayout` + `&StudyDimensions`; **produces** one `StageGeometry` per stage. No stored dim copies.                                                                                    | keep; drop the `n_h`/`lag_order`/… copies  |

**Deleted:** `StageIndexer`, `EquipmentCounts` (it _becomes_ `StudyDimensions`),
and both copy constructors. No new bag is introduced — `StudyDimensions` is the
existing `EquipmentCounts` promoted to a single owner.

### 4.1 No-duplication proof

`hydro_count` → `StateLayout` only. `n_thermals` → `StudyDimensions` only.
`n_blks` → `StageGeometry` only (per-stage; no global copy survives).
`max_deficit_segments` → `StudyDimensions` only. Every layout scalar resolves to
exactly one type.

### 4.2 Ownership rules that make it stick

- **State-defining vs equipment dims are disjoint sets** — a count is either
  state-defining (→ `StateLayout`) or equipment-shape (→ `StudyDimensions`),
  never both. `hydro_count` is state-defining; per-stage row counts that need it
  read `state.hydro_count`.
- **`n_blks` is per-stage by definition** — it appears only on `StageGeometry`.
  Any consumer reading a "global `n_blks`" is reading a footgun and is repointed
  to the per-stage value.
- **Identity lists by variance:** `anticipated_thermal_indices` is study-invariant
  (→ `StudyDimensions`); `fpha_/evap_hydro_indices`/`evap_indices` vary per stage
  (→ `StageGeometry`).
- The dependency graph stays one-way: `StageGeometry`/`StageLayout` →
  {`StateLayout`, `StudyDimensions`}; the leaf owners read nothing back.

---

## 5. What is **not** a defect (do not over-correct)

- **`StateLayout` keeping its own state dims is correct**, not duplication — under
  the disjoint model it is the _sole_ owner of those dims; no other type holds
  them after the collapse.
- **`StageLayout` being a fat builder is fine** — it borrows dims and produces a
  `StageGeometry`; the defect was the _copied, separately-maintained_ role-(b)
  half, not its breadth.
- **The two cut representations stay two** (`cut/row` vs `cut/dcs`).
- **No new user-blocking validation** — the non-uniform-block path is supported,
  not forbidden.

---

## 6. Sequencing (each phase hash-neutral, independently revertible)

1. **`EquipmentCounts` → `StudyDimensions`.** Promote the input bag to the single
   persisted owner of the non-state study scalars/flags + `anticipated_thermal_indices`;
   move those fields off `StageIndexer`; delete the `n_state`/`hydro_count`
   `StageIndexer` duplicates (read `StateLayout`); thread `StudyDimensions` through
   the contexts; repoint readers. **No new struct** — `EquipmentCounts` is
   renamed and promoted, not supplemented.
2. **`StageEquipmentGeometry` → `StageGeometry`.** Widen with per-stage row ranges
   - per-stage identity lists + per-stage `n_blks`; repoint the per-stage readers
     (extraction, the setup-derived specs); make `StageLayout` borrow dims instead
     of copying them.
3. **Delete `StageIndexer`** + both copy constructors; replace
   `StageData.indexer` / `TrainingContext.indexer` with the `StudyDimensions`
   handle + the per-stage `StageGeometry` table + the existing `StateLayout`
   handle. The compiler proves completeness (the type becomes dead).

**Verification each phase:** `cargo fmt`; build + clippy `-D warnings` on **both**
default and `test-support` features; `RUSTDOCFLAGS=-Dwarnings cargo doc`;
`cargo test` (test-support); and `COBRE_PARITY_REGEN` **neutrality** across all
D-cases (including non-uniform D33/D34). Any digest movement is a _found bug_
surfaced by the move (not expected churn) and gates an `sddp-specialist` sign-off
for that case. A focused `sddp-specialist` review follows Phase 2 (the
cut/anticipated/extraction-touching phase) and the final phase.

---

## 7. Key symbols

- `crates/cobre-sddp/src/lp/indexer/layout.rs` — `EquipmentCounts` (→
  `StudyDimensions`), `StageIndexer` (to dissolve), the stage-0 `n_blks` stride,
  the `has_*` flags, the identity lists.
- `crates/cobre-sddp/src/lp/indexer/state_layout.rs` — `StateLayout`
  (`control_region_start()`, the state-defining dims; unchanged owner).
- `crates/cobre-sddp/src/lp/builder/template.rs` — `StageEquipmentGeometry` (→
  `StageGeometry`), `from_layout` (the per-stage nucleus), `from_indexer` (to
  delete), `equipment_geometry_per_stage`.
- `crates/cobre-sddp/src/lp/builder/layout.rs` — `StageLayout` (the per-stage
  builder; drop the dim copies; borrow `StudyDimensions`).
- `crates/cobre-sddp/src/setup/mod.rs` — `build_wired_indexer` (the single
  construction site; builds the dims/flags, sets `has_ncs`).
- `crates/cobre-sddp/src/setup/stage_data.rs`,
  `crates/cobre-sddp/src/workspace/context.rs` — `StageData` / `TrainingContext`
  (the persisted/hot-path handles to repoint).
- `crates/cobre-sddp/src/simulation/extraction.rs` — the heaviest per-stage
  consumer.
- `docs/design/lp-architecture-degradation-assessment.md` §D1/§6/§7;
  `docs/design/lp-indexer-simplification-assessment.md` §8;
  `docs/design/lp-extraction-nonuniform-block-base-bug.md` — the prior passes.
