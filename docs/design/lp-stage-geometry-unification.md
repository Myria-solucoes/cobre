# `crates/cobre-sddp/src/lp/` — Post-Redesign Assessment: Unifying the Role-(b) Geometry Carriers

> **Status: design only. No code.** This is the post-redesign continuation of
> `lp-architecture-degradation-assessment.md` (§D1, §6, §7) and
> `lp-indexer-simplification-assessment.md` (§8). The role-(a) `StateLayout`
> extraction and the `anticipated_state_out` relocation those docs designed are
> **done**; the per-stage simulation-extraction base fix is **done**. This doc
> assesses the _resulting_ shape and specifies the next structural move:
> collapsing the role-(b) per-stage geometry from three hand-synced carriers to
> one, and retiring `StageIndexer` as a distinct persisted type. Grounded in a
> consumer trace of the live tree, not in the doc-comments.

## 0. Verdict up front

The redesign improved one axis and regressed the other.

- **Role (a) is now genuinely clean.** `StateLayout` is a single-concern,
  stage-invariant owner of the state-vector columns, the cut/pin resolvers, and
  the two caches; every offset is a pure function of `(N, L, A, k_max)`. The
  `anticipated_state_out` relocation fixed the cut-target latent bug _by
  construction_. This is a real reduction in coupling and a real correctness win.

- **Role (b) is now more fragmented than before the redesign.** The same ~18
  equipment/slack column ranges are declared in **three** types
  (`StageIndexer`, `StageLayout`, `StageEquipmentGeometry`) with **three**
  validity scopes and **two** field-by-field copy constructors. Before the
  redesign there were two role-(b) carriers; there are now three. `StageIndexer`
  has degenerated from "the layout source of truth" into a grab-bag of
  stage-invariant row bases + study-wide flags + stage-0 identity lists + a
  vestigial copy of the equipment ranges.

This realizes the exact prediction of degradation §D1 ("geometry lives in 4+
hand-synced owners"). The fix is the §6 target taken one step further: make the
**per-stage** geometry the single owner and dissolve `StageIndexer`. The move is
**expected hash-neutral** — it relocates _which type holds_ an index, not the
index itself.

---

## 1. What the redesign delivered (the post-plan map)

Four types now carry what `StageIndexer` alone used to:

| Type                                             | What it actually holds (verified from fields + readers)                                                                                                                                                                                                                                | Lifetime                                                                |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `StateLayout` (`indexer/state_layout.rs`)        | Role (a): `storage`, `inflow_lags`, `anticipated_state`, `anticipated_state_out`, `storage_in`, `z_inflow` columns; `theta`; `n_state`; state dims; `anticipated_lead_stages`; the two caches; the resolvers + `is_anticipated_decision_active`. Every offset is `n_blks`-independent. | Long-lived; shared by `&` (`StageData.state`, `TrainingContext.state`). |
| `StageIndexer` (`indexer/layout.rs`)             | Role (b) **anchored at stage-0's global `n_blks`**. Equipment/slack column ranges, row ranges, counts, `has_*` flags, FPHA/evap/anticipated identity lists. Its own rustdoc admits it is "valid only at stages whose block count equals stage 0's."                                    | Long-lived (`StageData.indexer`, `TrainingContext.indexer`).            |
| `StageLayout<'a>` (`builder/layout.rs`)          | Borrows `&StateLayout` (role a, **not** duplicated) **+** its own per-stage role-(b) equipment/row ranges (correct `n_blks`) **+** transient build state (identity maps, `generic_constraint_rows`, `zeta`, `num_cols/num_rows`).                                                      | Ephemeral; dropped after the CSC is baked.                              |
| `StageEquipmentGeometry` (`builder/template.rs`) | A persisted **snapshot of `StageLayout`'s 18 equipment ranges**, for the simulation read path.                                                                                                                                                                                         | Long-lived (`StageTemplates::equipment_geometry_per_stage`).            |

The role-(a) boundary is exactly the one designed in indexer §8.1 and it holds.
The new cost is entirely on the role-(b) side.

---

## 2. The realized degradation: role-(b) geometry triplicated

### 2.1 The same ranges, declared three times

The block-major equipment/slack families each have a `Range<usize>` field in all
three role-(b) carriers:

| Family                                                                    | `StageIndexer` | `StageLayout`               | `StageEquipmentGeometry` | Scope of each                                                        |
| ------------------------------------------------------------------------- | -------------- | --------------------------- | ------------------------ | -------------------------------------------------------------------- |
| `turbine`, `spillage`, `diversion`, `thermal`                             | ✓              | ✓                           | ✓                        | indexer = stage-0; layout = per-stage; geometry = per-stage snapshot |
| `line_fwd`, `line_rev`, `deficit`, `excess`                               | ✓              | ✓                           | ✓                        | ""                                                                   |
| `generation`, `evap_indices`                                              | ✓              | ✓                           | ✓                        | ""                                                                   |
| `inflow_slack`, `withdrawal_slack_{neg,pos}`                              | ✓              | ✓                           | ✓                        | ""                                                                   |
| `outflow_below`/`outflow_above`/`turbine_below`/`generation_below` slacks | ✓              | ✓                           | ✓                        | ""                                                                   |
| `anticipated_decision`                                                    | ✓              | ✓ (via `AnticipatedLayout`) | ✓                        | ""                                                                   |

Synchronisation between them is two field-by-field copy constructors:

- `StageEquipmentGeometry::from_layout(&StageLayout)` — the **production** path
  (per-stage, correct). Built in `build_single_stage_template`, transposed into
  `StageTemplates::equipment_geometry_per_stage`.
- `StageEquipmentGeometry::from_indexer(&StageIndexer)` — **tests + uniform-block
  single-stage convenience only**; copies the stage-0 ranges verbatim. Its own
  rustdoc names it "the bug this type exists to forbid" when misused at a stage
  whose block count differs.

Adding or moving any equipment family is now an edit in three struct definitions
plus one (or two) copy constructors, none enforced by the compiler — the same
"thread it through ≥N sites in agreement" hazard degradation §D1 flagged, now
with an extra site.

### 2.2 `StageIndexer` has become a grab-bag

A consumer trace of the persisted `StageIndexer` (excluding `#[cfg(test)]` and
the `indexer/` module itself) shows its production readers fall into four
unrelated groups:

1. **Genuinely stage-invariant row geometry** (correct to read globally):
   `water_balance.start`, `load_balance.start`, `z_inflow_row_start`,
   `turbine.start` (= `theta + 1`, the control-region anchor).
2. **Study-wide scalars/flags:** `has_inflow_penalty`, `has_withdrawal`,
   `has_operational_violations`, `has_ncs`, `max_deficit_segments`, `n_state`,
   `hydro_count` — several of which **duplicate `StateLayout`** (`n_state`,
   `hydro_count`).
3. **Stage-0 identity lists:** `fpha_hydro_indices`, `fpha_rows`,
   `evap_hydro_indices`, `evap_indices`, `anticipated_thermal_indices`,
   `anticipated_fishing_start`. These are the **stage-0** sets; FPHA/evap
   membership can vary per stage, so a global read here is a latent footgun of
   the same class as §2.3 (currently masked because the production per-stage
   consumers read the recomputed `StageLayout`/persisted per-stage tables).
4. **Vestigial equipment ranges:** the §2.1 column ranges, now consumed in
   production only where stage-0 == per-stage (uniform studies), or via the
   test-only `from_indexer`.

Group (4) is dead weight on the hot path; groups (1)–(3) are three different
concerns wearing one type. `StageIndexer` is no longer a single source of
truth — it is the residue left after role (a) and the correct role-(b) geometry
moved out.

### 2.3 The `n_blks` footgun is still embedded in the type

`StageIndexer.n_blks` is the **global stage-0** block count. The type carries it
as a stride for its own equipment ranges, so the type _invites_ the very
non-uniform-block bug the redesign just fixed in `extraction.rs`. The fix
repointed the simulation reads onto the per-stage `StageEquipmentGeometry`, but
the trap (a persisted global stride next to persisted global bases) is still
present in the type for the next consumer to step on.

---

## 3. Root cause: persisted is stage-0-global; correct is ephemeral

The single decision that produced the triplication: **the geometry the long-lived
context structs persist (`StageData.indexer`, `TrainingContext.indexer`) is the
stage-0-global `StageIndexer`, while the stage-correct geometry (`StageLayout`)
is computed at bake time and dropped.**

When a post-bake consumer (simulation extraction) needed correct per-stage
geometry, neither persisted source served it — `StageIndexer` is stage-0-wrong
for non-uniform blocks, and `StageLayout` no longer exists. The extraction fix
closed the correctness gap by **persisting a third carrier**
(`StageEquipmentGeometry`) instead of making the per-stage geometry _the_
persisted role-(b) representation and retiring the stage-0 one. That was the
correct, surgical, hash-bounded move for a correctness fix under an in-flight
plan; it was explicitly **not** the structural unification, which was deferred to
here.

---

## 4. Target shape: one per-stage geometry owner; dissolve `StageIndexer`

A shape, not a ticket list. The model is already right; this is one more
"convention/copy → construction" upgrade in the spirit of degradation §6.

**(T1) One persisted per-stage role-(b) geometry table.** Generalize
`StageEquipmentGeometry` (or a renamed `StageGeometry`) into the single owner of
"where does family X live at stage `t`": the equipment/slack column ranges it
already holds, **plus** the per-stage row ranges and the per-stage identity lists
that consumers currently reach into `StageIndexer` for. Built once per stage from
`StageLayout` (the existing `from_layout` path), persisted as the existing
per-stage `Vec`, indexed by `t`. This is the same per-stage-`&[…]` pattern
`ncs_col_starts` / `pumping_col_starts` already established.

**(T2) Study-invariant scalars move to where they belong.** The `has_*` flags and
study-wide dims become a small study-level metadata holder (or fold onto
`StateLayout` where they are state-related — `n_state`, `hydro_count` already live
there, so the `StageIndexer` copies are deleted, not moved).

**(T3) Stage-invariant row anchors fold into the state/metadata layer.** The row
_bases_ (`water_balance.start`, `load_balance.start`, `z_inflow_row_start`) and
the control anchor (`theta + 1`, already `StateLayout::control_region_start()`)
are `n_blks`-independent; they belong on `StateLayout`/the metadata holder, read
through the handle the hot paths already carry.

**(T4) Retire `StageIndexer`.** With T1–T3 it has no remaining concern of its own:
`StageData.indexer` / `TrainingContext.indexer` are replaced by `state:
&StateLayout` (already present) + the per-stage geometry table (T1) + the
metadata holder (T2). Both copy constructors (`from_layout`, `from_indexer`) and
the stage-0 `n_blks` footgun (§2.3) disappear with the type.

### 4.1 Boundary — what each residual `StageIndexer` member becomes

| Residual member (group from §2.2)              | Destination                                                                                         |
| ---------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| Equipment/slack column ranges (4)              | The per-stage geometry table (T1) — already there as the `from_layout` snapshot.                    |
| Per-stage row ranges + identity lists (3)      | The per-stage geometry table (T1), made per-stage-correct (closes the §2.2-group-3 latent footgun). |
| Stage-invariant row bases + control anchor (1) | `StateLayout` / metadata holder (T3).                                                               |
| `has_*` flags, study dims (2)                  | Metadata holder (T2); `n_state`/`hydro_count` deleted as `StateLayout` duplicates.                  |

### 4.2 The dependency stays one-way

The acyclic graph indexer §8.2 established is preserved: geometry → `StateLayout`
(reads `control_region_start()`, anticipated metadata); `StateLayout` reads
nothing back. T1–T4 only remove a redundant node (`StageIndexer`) and a redundant
edge (the stage-0 copies); they introduce no new reverse dependency.

---

## 5. What is **not** a defect (do not over-correct)

- **`StageLayout` being a fat builder is fine.** It legitimately holds a borrowed
  role-(a) handle + its own role-(b) geometry + transient build state. A builder
  is allowed to be wide; the defect is that its role-(b) half is _copied out and
  separately maintained_, not that it holds it. Do **not** try to slim
  `StageLayout` itself.
- **Do not merge role (a) back in.** The `StateLayout` boundary is the redesign's
  win; T1–T4 keep it untouched.
- **The two cut representations stay two** (`cut/row` baked path vs `cut/dcs`) —
  degradation §3. This doc is about role-(b) geometry ownership only.
- **No new user-blocking validation.** The non-uniform-block path is _supported_,
  not forbidden (degradation §1.4); T1 widens support, it does not gate input.

---

## 6. Verification, sequencing, and blast radius (honest)

**Hash-neutrality.** T1–T4 relocate _which type owns_ an index; they do not change
any column or row address. The move is therefore **expected hash-neutral**.
Verify by `COBRE_PARITY_REGEN` neutrality across **all** D-cases, including the
non-uniform D33/D34. Any digest movement is not "expected churn" — it flags a
residual stage-0 read that the relocation surfaced (a found bug of the §2.2
group-3 class), and that case alone requires `sddp-specialist` sign-off. This is
the opposite risk profile from the `anticipated_state_out` relocation, which was
_intentionally_ non-hash-neutral.

**Blast radius.** `StageIndexer` is read across the persisted context structs and
~10 production files. The heavy consumer is `simulation/extraction.rs` (already
repointed onto the per-stage geometry for the `n_blks`-dependent reads; the
remainder are the stage-invariant group-1 reads, the easy part of T3). The hot
paths (`training/forward/stage_solve.rs`, `training/backward/lp_setup.rs`,
`training/lower_bound.rs`) read the stage-invariant subset (`has_*`, row bases,
`n_blks`) — folded by T2/T3. The cut/warm-start path already consumes role (a)
via `StateLayout::state*`, so T1–T4 do not touch the basis wire format
(`CapturedBasis`) or cut storage.

**Sane order.**

1. **T3 + T2 first (stage-invariant fold)** — move row bases, the control anchor,
   and the flags onto `StateLayout`/metadata; delete the `n_state`/`hydro_count`
   duplicates. Pure substitution; hash-neutral; smallest risk.
2. **T1 (widen the per-stage table)** — add the per-stage row ranges + identity
   lists to the geometry table; repoint the remaining `StageIndexer` group-3
   reads onto it. Hash-neutral; closes the §2.2-group-3 footgun.
3. **T4 (delete `StageIndexer` + both copy constructors)** — falls out once
   1–2 leave it with no readers. The compiler proves completeness (dead type).

Each step is independently hash-verifiable and independently revertible.

**What this buys.** Three role-(b) carriers → one (+ the ephemeral builder, which
is correct to keep); two copy constructors → zero; the stage-0 `n_blks` footgun
deleted with its type; `StageIndexer`'s grab-bag dissolved into single-concern
owners. Net subtractive — a smaller codebase, the §D1 prediction discharged.

---

## 7. Key symbols

- `crates/cobre-sddp/src/lp/indexer/layout.rs` — `StageIndexer` (the type to
  dissolve; the stage-0 `n_blks` stride; the `has_*` flags; the identity lists).
- `crates/cobre-sddp/src/lp/indexer/state_layout.rs` — `StateLayout`
  (`control_region_start()`, `n_state`, `hydro_count`; the T2/T3 destination).
- `crates/cobre-sddp/src/lp/builder/template.rs` — `StageEquipmentGeometry`,
  `from_layout` (the production per-stage path; the T1 nucleus), `from_indexer`
  (the test/uniform copy to delete), `equipment_geometry_per_stage`.
- `crates/cobre-sddp/src/lp/builder/layout.rs` — `StageLayout` (the per-stage
  geometry source; keep as the builder).
- `crates/cobre-sddp/src/setup/stage_data.rs` — `StageData.{indexer,state}` (the
  persisted owners; `indexer` to be replaced by the geometry table + metadata).
- `crates/cobre-sddp/src/workspace/context.rs` — `TrainingContext.{indexer,state}`
  (the hot-path handles).
- `crates/cobre-sddp/src/simulation/extraction.rs` — the heaviest consumer;
  already per-stage for equipment, stage-invariant remainder is the T3 tail.
- `docs/design/lp-architecture-degradation-assessment.md` — §D1 (the prediction),
  §6/§7 (the target shape this continues).
- `docs/design/lp-indexer-simplification-assessment.md` — §8 (the role-(a)
  extraction this builds on; §8.5's "audit residual global geometry reads").
- `docs/design/lp-extraction-nonuniform-block-base-bug.md` — the correctness fix
  that added the third carrier under the in-flight plan.
