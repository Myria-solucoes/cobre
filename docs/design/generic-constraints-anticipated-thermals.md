# Generic Constraints — Anticipated Thermal Support (F2-006)

**Status:** Design / not yet implemented
**Owner:** TBD
**Tracking:** assessment finding F2-006 (see `assessment-report.md`)
**Effort estimate:** Medium (~2–3 days for a focused implementer)
**Last revised:** 2026-05-22

---

## Abstract

Extend the `VariableRef` enum and the generic-constraint expression parser to
support direct references to anticipated-thermal LP columns. Add a semantic
validator warning when a user references an anticipated thermal via the
existing per-block `thermal_generation` variable (the column's meaning shifts
across stages and the user almost certainly meant the commitment column
instead).

Add **one** new `VariableRef` variant:

```rust
VariableRef::AnticipatedDecision { thermal_id: EntityId }
```

The companion `AnticipatedState { thermal_id, slot }` variant is **explicitly
deferred** (see § 11 "Open Questions / Deferred").

---

## 1. Background

### 1.1 What generic constraints are

Cobre users define linear LP constraints in `system/generic_constraints.json`
using an expression-string DSL:

```json
{
  "constraints": [
    {
      "id": 7,
      "name": "thermal_cap",
      "stage_id": null,
      "expression": "thermal_generation(5, 0) + thermal_generation(6, 0) <= 200",
      "sense": "le",
      "rhs": 200.0
    }
  ]
}
```

The parser at `crates/cobre-io/src/constraints/generic.rs` tokenises the
expression and resolves each identifier (`thermal_generation`,
`hydro_generation`, etc.) into a `VariableRef` enum variant. At LP-build time
`crates/cobre-sddp/src/generic_constraints.rs::resolve_variable_ref` maps each
`VariableRef` to one or more `(column_index, coefficient_multiplier)` pairs and
the CSC matrix entries are emitted.

`VariableRef` lives in `crates/cobre-core/src/generic_constraint.rs` (the
core data model). It currently has **20 variants** covering the canonical
column catalog: hydro storage/turbine/spillage/diversion/outflow/generation/
evaporation/withdrawal, thermal generation, line direct/reverse/exchange, bus
deficit/excess, pumping flow/power, contract import/export, NCS generation/
curtailment.

### 1.2 The anticipated-thermals LP layout

After the anticipated-thermals feature landed (commits `6e6e308..69f08a6`,
v0.6.x), the per-stage LP gained three new column blocks per stage `t`:

| Block                               | Layout                                   | Cardinality         | What it represents                                                                      |
| ----------------------------------- | ---------------------------------------- | ------------------- | --------------------------------------------------------------------------------------- |
| `anticipated_decision`              | `[start_dec, start_dec + n_anticipated)` | `A = n_anticipated` | Per-plant commitment scalar `d_t,i` placed at stage `t` for delivery at stage `t + K_i` |
| `anticipated_state`                 | `[start_state, start_state + A * K_max)` | `A * K_max`         | Slot-major ring buffer; `state[slot, plant]` lives at `start_state + slot * A + plant`  |
| `thermal` (per-block, pre-existing) | `[indexer.thermal.start, ...)`           | `T * n_blks`        | Per-block generation `g_blk` for ALL thermals (anticipated and non-anticipated)         |

The fishing equality row at delivery stages couples per-block thermal
generation to slot 0 of the ring buffer:

```text
sum_blk hours[blk] * g_t,i,blk  =  hours_total * state_t[slot=0, plant=i]
```

(See `crates/cobre-sddp/src/lp_builder/matrix.rs`, the
`fill_anticipated_fishing_*` family for the row-construction code.)

### 1.3 The gap

`VariableRef::ThermalGeneration { thermal_id, block_id }` is the only way
a user can reference any thermal column today. For an anticipated plant
this column has **stage-dependent semantics**:

| Stage                           | Predicate        | What `g_blk` means in the LP                                                                                                                                             |
| ------------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `t < K_i` (pre-delivery)        | fishing inactive | Free dispatch in `[min_gen, max_gen]`. User constraint behaves like any normal thermal cap.                                                                              |
| `t >= K_i` (delivery and after) | fishing active   | Coupled to `slot_0` via the fishing equality. A user constraint `g_blk <= X` may bind `slot_0 <= X * hours[blk] / hours_total` instead — non-obvious and stage-specific. |

Additionally, **there is no way for the user to reference the commitment
column** `anticipated_decision`. So constraints like "limit how much we can
commit forward in stage `t`" cannot be expressed.

The F2-006 finding (assessment report, § Major) describes this as a quiet
trap: `ThermalGeneration` on an anticipated plant doesn't crash, but its
LP meaning shifts across stages, and the documented commitment column has
no public reference.

---

## 2. Goals

1. Let users express linear constraints over the `anticipated_decision`
   column with the same ergonomics as existing variables.
2. Emit a semantic-validator **warning** (not error) when a user references
   `thermal_generation(i, blk)` for a thermal `i` whose
   `anticipated_config: Some(_)`, alerting them to the stage-dependent
   semantics and suggesting `anticipated_decision(i)` if the commitment
   was the intended target.
3. Maintain backward compatibility: every existing case file that uses
   `thermal_generation(i, blk)` continues to parse and build the same LP.
4. Keep the JSON expression DSL consistent with existing variant names
   (snake_case, parenthesised arguments, optional block index).
5. Preserve bit-for-bit declaration-order invariance.

## 3. Non-goals

- **Do not** introduce `VariableRef::AnticipatedState { thermal_id, slot }`
  in this ticket. See § 11.1 for the deferral rationale.
- **Do not** change the semantics of `thermal_generation` on anticipated
  plants. The warning is purely informational; the LP behavior stays the
  same.
- **Do not** auto-rewrite user expressions. The validator warning suggests
  the alternative; the user manually edits.
- **Do not** add new public APIs to `cobre-sddp` (the SDDP crate). All
  public-API additions live in `cobre-core` and `cobre-io`.

---

## 4. Detailed design

### 4.1 New `VariableRef` variant

In `crates/cobre-core/src/generic_constraint.rs`, after the existing
`ThermalGeneration` variant (current line 122-128):

```rust
/// Forward-commitment decision MW for an anticipated thermal unit (MW).
///
/// References the commitment placed at the current stage `t` for delivery
/// at stage `t + lead_stages`. This is a per-plant per-stage scalar — it has
/// **no `block_id`** because the commitment is uniform across blocks.
///
/// The column exists in the LP only for plants whose `anticipated_config`
/// is `Some(_)`. Referencing this variant for a non-anticipated thermal is
/// a referential-validation error (see
/// `cobre-io::validation::referential::validate_variable_ref_entity`).
///
/// The column also has `[0.0, 0.0]` bounds at boundary stages where
/// `t + K_i >= n_stages` (the F2-002 strict predicate); a constraint
/// referencing the column at the boundary is structurally a no-op.
AnticipatedDecision {
    /// Thermal unit identifier. Must satisfy `anticipated_config: Some(_)`.
    thermal_id: EntityId,
},
```

**Why no `block_id` field**: the commitment is set once per stage per
plant. Carrying an `Option<usize>` would be syntactically misleading.
Documented existing precedents that follow the same pattern:
`HydroStorage`, `HydroEvaporation`, `HydroWithdrawal` (all stage-level).

### 4.2 Expression DSL update

In `crates/cobre-io/src/constraints/generic.rs`, the
`build_variable_ref` function (current lines 800–894) maps each
identifier name to its `VariableRef` variant.

#### 4.2.1 Add the name → variant mapping

After the existing `"thermal_generation"` arm (line 843), add:

```rust
"anticipated_decision" => {
    // Stage-level (no block_id). Reject explicit block_id with a clear error.
    if block_id.is_some() {
        return Err(format!(
            "variable \"anticipated_decision\" is a stage-level scalar and \
             does not accept a block_id — write \"anticipated_decision({})\", \
             not \"anticipated_decision({}, ...)\"",
            entity_id.0, entity_id.0,
        ));
    }
    Ok(VariableRef::AnticipatedDecision {
        thermal_id: entity_id,
    })
}
```

The expression-language signature is therefore `anticipated_decision(N)`
where `N` is the thermal id.

#### 4.2.2 Update the doc-comment list

The top-of-file doc comment in `crates/cobre-io/src/constraints/generic.rs`
enumerates the supported variable names in examples. Add
`anticipated_decision` to that enumeration so users discover it by reading
the module docs.

#### 4.2.3 Update the JSON schema

`book/src/schemas/generic_constraints.schema.json` carries the JSON
schema for the file. Find the `expression` field's description (currently
line 15) and update its example to mention the new identifier:

```diff
- "description": "Expression string to be parsed. E.g. `\"2.5 * thermal_generation(5) - hydro_generation(3)\"`.",
+ "description": "Expression string to be parsed. E.g. `\"2.5 * thermal_generation(5) - hydro_generation(3)\"`. To constrain an anticipated thermal's commitment, use `\"anticipated_decision(5)\"` (stage-level scalar, no block index).",
```

### 4.3 LP resolver mapping

In `crates/cobre-sddp/src/generic_constraints.rs::resolve_variable_ref`
(current lines 81-220), add the new arm. Suggested placement:
immediately after the existing `VariableRef::ThermalGeneration` arm
(current lines 141-152).

```rust
// ── Anticipated thermal commitment ─────────────────────────────────
VariableRef::AnticipatedDecision { thermal_id } => {
    resolve_anticipated_decision(*thermal_id, indexer, thermal_pos)
}
```

And add the helper function near the bottom of the file (after
`resolve_hydro_evaporation` style helpers):

```rust
/// Resolve `AnticipatedDecision` to the per-plant commitment LP column.
///
/// The column index is
/// `indexer.anticipated_decision.start + local_idx`
/// where `local_idx` is the position of `thermal_id` inside
/// `indexer.anticipated_thermal_indices` (the parallel array maintained by
/// `StudySetup`).
///
/// Returns an empty vec when:
/// - `thermal_id` is not found in `thermal_pos` (defence-in-depth — the
///   referential validator should have caught this), OR
/// - the thermal exists but is not anticipated (its system-level position
///   does not appear in `indexer.anticipated_thermal_indices`).
///
/// `block_idx` is ignored: the commitment is a stage-level scalar that
/// does not vary across blocks.
fn resolve_anticipated_decision<S: BuildHasher>(
    thermal_id: EntityId,
    indexer: &StageIndexer,
    thermal_pos: &HashMap<EntityId, usize, S>,
) -> Vec<(usize, f64)> {
    let Some(&sys_pos) = thermal_pos.get(&thermal_id) else {
        return vec![];
    };
    let Some(local_idx) = indexer
        .anticipated_thermal_indices
        .iter()
        .position(|&p| p == sys_pos)
    else {
        return vec![];
    };
    vec![(indexer.anticipated_decision.start + local_idx, 1.0)]
}
```

### 4.4 Referential validation

In
`crates/cobre-io/src/validation/referential.rs::validate_variable_ref_entity`
(current lines 883-...), extend the thermal arm to cover the new variant:

```rust
VariableRef::ThermalGeneration { thermal_id, .. }
| VariableRef::AnticipatedDecision { thermal_id, .. } => {
    if !ids.thermal.contains(&thermal_id.0) {
        ctx.add_error(
            ErrorKind::InvalidReference,
            file,
            Some(label.to_string()),
            format!("{label} references non-existent Thermal {}", thermal_id.0),
        );
    }
}
```

This handles "thermal id does not exist" for both variants uniformly.

The follow-on check ("thermal exists but is not anticipated") lives in
the semantic validator (see § 4.5).

### 4.5 Semantic-validator additions

In `crates/cobre-io/src/validation/semantic/thermal.rs`, add **two** new
checks. Both iterate over every `LinearTerm` in every parsed
`GenericConstraint` and inspect the `VariableRef`.

#### Check A — `anticipated_decision` requires an anticipated thermal (HARD ERROR)

```rust
fn check_anticipated_decision_target_is_anticipated(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let anticipated_ids: HashSet<EntityId> = data
        .thermals
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| t.id)
        .collect();

    for constraint in &data.generic_constraints {
        for term in &constraint.expression.terms {
            if let VariableRef::AnticipatedDecision { thermal_id } = term.variable {
                if !anticipated_ids.contains(&thermal_id) {
                    ctx.add_error(
                        ErrorKind::BusinessRuleViolation,
                        "system/generic_constraints.json",
                        Some(format!("constraints[id={}]", constraint.id.0)),
                        format!(
                            "Constraint {}: anticipated_decision({}) references Thermal {} \
                             which is not anticipated (has no `anticipated_config`). The \
                             commitment column exists only for anticipated thermals. \
                             Use `thermal_generation({}, ...)` for the per-block generation \
                             column, or add `anticipated_config` to the thermal.",
                            constraint.id.0, thermal_id.0, thermal_id.0, thermal_id.0,
                        ),
                    );
                }
            }
        }
    }
}
```

#### Check B — `thermal_generation` on an anticipated plant gets a WARNING

```rust
fn warn_thermal_generation_on_anticipated_thermal(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    let anticipated_ids: HashSet<EntityId> = data
        .thermals
        .iter()
        .filter(|t| t.anticipated_config.is_some())
        .map(|t| t.id)
        .collect();

    for constraint in &data.generic_constraints {
        for term in &constraint.expression.terms {
            if let VariableRef::ThermalGeneration { thermal_id, .. } = term.variable {
                if anticipated_ids.contains(&thermal_id) {
                    ctx.add_warning(
                        ErrorKind::SemanticAmbiguity,
                        "system/generic_constraints.json",
                        Some(format!("constraints[id={}]", constraint.id.0)),
                        format!(
                            "Constraint {}: thermal_generation({}, ...) references \
                             anticipated Thermal {}. The per-block generation column \
                             has stage-dependent semantics for anticipated thermals: \
                             pre-delivery (t < K_i) it is free dispatch within \
                             [min_gen, max_gen], at delivery and after (t >= K_i) it \
                             is coupled to the matured commitment via the fishing row \
                             `sum_blk hours[blk] * g_blk = hours_total * slot_0`. If \
                             you meant to constrain the commitment scalar, use \
                             `anticipated_decision({})` instead.",
                            constraint.id.0, thermal_id.0, thermal_id.0, thermal_id.0,
                        ),
                    );
                }
            }
        }
    }
}
```

Both functions are registered in `cobre-io/src/validation/semantic/mod.rs`
alongside the existing thermal checks.

> **New error kind**: `ErrorKind::SemanticAmbiguity` is a new warning class.
> If the existing enum does not have a suitable variant, add it. Candidates
> already in the enum: `BusinessRuleViolation` (probably the wrong tone for
> a warning), `UnusedEntity` (used for the stub-entity warnings), or a new
> `SemanticAmbiguity`. Verify before implementation; if a suitable warning
> kind exists, use it.

---

## 5. Schema / wire-format impact

### 5.1 Postcard MPI broadcast

`VariableRef` is `Serialize + Deserialize` via serde derives. Postcard is a
**positional** wire format — appending a new variant at the end of the enum
is safe **only if** the existing variants retain their discriminant order.
Serde encodes enum variants by their **index in declaration order**, so
appending the new variant after the last (`NonControllableCurtailment`) is
safe.

**Action**: declare `AnticipatedDecision` at the **end** of `VariableRef`,
not after `ThermalGeneration` as suggested in § 4.1. Rationale: keeps the
wire format strictly append-only. The doc-comment for the variant can
still point at the conceptual grouping ("forward-commitment decision").

**Regression guard**: extend
`crates/cobre-core/tests/generic_constraint_serde_*.rs` (or the appropriate
test file — verify) to round-trip a `VariableRef::AnticipatedDecision`
value through postcard and assert structural equality.

### 5.2 FlatBuffers policy

`VariableRef` is **not** part of the FlatBuffers policy schema (verify by
grepping `crates/cobre-io/src/output/policy/`). Policy serialisation
covers cuts and basis caches, not user-authored constraints. No
FlatBuffers-side action expected; verify during implementation.

### 5.3 JSON schema file

Update `book/src/schemas/generic_constraints.schema.json` per § 4.2.3.
The schema file is published in the book; the change is user-visible
documentation, not a wire-format constraint.

---

## 6. Documentation updates

### 6.1 mdBook reference

`book/src/reference/case-format.md` documents the
`generic_constraints.json` schema. Add a row to the variable-name table
for `anticipated_decision(N)` with the same shape as the existing
`thermal_generation(N, blk)` row:

| Variable name          | Signature                          | LP column                                | Semantics                                                                                                                       |
| ---------------------- | ---------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `anticipated_decision` | `anticipated_decision(thermal_id)` | `anticipated_decision.start + local_idx` | Per-plant commitment scalar at the current stage. Stage-level; no block index. Thermal must have `anticipated_config: Some(_)`. |

### 6.2 mdBook guide

`book/src/guide/thermal-units.md` already documents anticipated thermal
units. Add a subsection under the existing "Anticipated Dispatch
Configuration" header explaining how to constrain commitments via generic
constraints:

```markdown
### Constraining commitments via generic constraints

To express an LP constraint on the commitment column directly (rather
than on per-block dispatch), use `anticipated_decision(thermal_id)` in
the expression string in `system/generic_constraints.json`:

    "expression": "anticipated_decision(5) + anticipated_decision(6) <= 150"

This caps the SUM of stage-0 commitments for thermals 5 and 6 at 150 MW
at every stage. Note that the commitment column has `[0, 0]` bounds at
boundary stages where `t + K_i >= n_stages` (the strict-active predicate
introduced by F2-002), so the constraint is structurally a no-op at
those stages.

Referencing a non-anticipated thermal via `anticipated_decision` is a
validation error. Referencing an anticipated thermal via
`thermal_generation(...)` is allowed (the column exists) but the validator
emits a warning because the column's meaning is stage-dependent — see the
warning text for the alternative.
```

### 6.3 CHANGELOG

Under `[Unreleased]` `### Added`:

```markdown
- `generic_constraints.json` accepts a new variable identifier
  `anticipated_decision(thermal_id)` which references an anticipated
  thermal's per-stage commitment column directly. Use this when you
  want to constrain commitments rather than per-block dispatch. The
  semantic validator emits a warning when `thermal_generation(...)`
  is used on an anticipated thermal, surfacing the stage-dependent
  semantics of that column.
```

---

## 7. Testing strategy

### 7.1 Unit tests in `cobre-core`

In `crates/cobre-core/src/generic_constraint.rs` test module:

- **AC-1**: `VariableRef::AnticipatedDecision { thermal_id: EntityId(5) }`
  constructs successfully.
- **AC-2**: Postcard round-trip preserves the variant and its `thermal_id`.

### 7.2 Parser tests in `cobre-io`

In `crates/cobre-io/src/constraints/generic.rs` test module:

- **AC-3**: `parse_expression("anticipated_decision(5)", &HashMap::new())`
  succeeds and yields a single `LinearTerm` with the expected variant.
- **AC-4**: `parse_expression("2.5 * anticipated_decision(5)", ...)`
  applies the literal scale.
- **AC-5**: `parse_expression("anticipated_decision(5, 0)", ...)` returns
  a `Err` whose message names `anticipated_decision` and explains that
  it does not accept a `block_id`.
- **AC-6**: Two ways to spell the same constraint produce identical
  `LinearTerm` vectors (whitespace and parenthesisation insensitivity —
  test that the existing tokenizer treats the new identifier like the
  others).

### 7.3 Referential-validation tests

In `crates/cobre-io/src/validation/referential.rs` test module:

- **AC-7**: A constraint with `anticipated_decision(99)` where Thermal 99
  does not exist yields `ErrorKind::InvalidReference` naming Thermal 99
  and the constraint id.

### 7.4 Semantic-validation tests

In `crates/cobre-io/src/validation/semantic/thermal.rs` test module:

- **AC-8**: A constraint with `anticipated_decision(5)` where Thermal 5
  exists but has `anticipated_config: None` yields
  `ErrorKind::BusinessRuleViolation` naming Thermal 5 and the constraint
  id. The error message must include the remediation hint
  (`thermal_generation(5, ...)` or add `anticipated_config`).
- **AC-9**: A constraint with `thermal_generation(5, 0)` where Thermal 5
  has `anticipated_config: Some(_)` yields a **warning** (not error)
  with the F2-006 explanation text and the
  `anticipated_decision(5)` suggestion. Verify `ctx.errors().is_empty()`
  and `ctx.warnings().len() == 1`.
- **AC-10**: A constraint with `thermal_generation(5, 0)` where Thermal 5
  is NOT anticipated produces **no** warning (the existing behaviour is
  unambiguous in that case).
- **AC-11**: A constraint with `anticipated_decision(5)` where Thermal 5
  IS anticipated produces **no** warning and **no** error.

### 7.5 LP-builder resolver tests

In `crates/cobre-sddp/src/generic_constraints.rs` test module:

- **AC-12**: Resolving `VariableRef::AnticipatedDecision { thermal_id: 5 }`
  in a study where Thermal 5 is anticipated and occupies local-index 1
  returns `vec![(indexer.anticipated_decision.start + 1, 1.0)]`.
- **AC-13**: Resolving the same variant in a study where Thermal 5 is
  not anticipated returns `vec![]` (defence-in-depth).
- **AC-14**: `block_idx` argument is ignored — passing `block_idx = 0`
  and `block_idx = 2` returns identical results.

### 7.6 End-to-end integration test

Create
`crates/cobre-sddp/tests/anticipated_generic_constraint_e2e.rs`:

- **AC-15**: A 3-stage K=2 fixture with one anticipated thermal and a
  generic constraint `anticipated_decision(thermal_id) <= 25.0`. Train
  for 8 iterations. Assert that every stage's
  `anticipated_decision_mw` parquet value is `<= 25.0 + 1e-9`. Compare
  against a baseline training run WITHOUT the constraint and verify
  that the constrained run's LB is strictly worse (more expensive) —
  proving the constraint actually binds.
- **AC-16**: Same fixture, but the constraint references a
  non-anticipated thermal — verify that case loading FAILS at the
  semantic-validator layer with the expected error message
  (the AC-8 error).

### 7.7 Declaration-order-invariance regression

Reuse the harness added by F3-004 in
`crates/cobre-sddp/src/lp_builder/template.rs::tests::lp_template_invariant_under_anticipated_index_permutation`.
Add a parallel test that includes a generic constraint over
`anticipated_decision(N)` and asserts the LP coefficient matrix is
bit-for-bit identical under two permutations of
`anticipated_thermal_indices`.

### 7.8 Verification gate

- `cargo nextest run --workspace --all-features` — all pass
- `cargo clippy --workspace --all-features --tests -- -D warnings` — clean
- `cargo fmt --all --check` — clean
- `mdbook build book` — clean

---

## 8. Backward compatibility

- All existing case files using `thermal_generation(i, blk)` continue to
  parse identically. The semantic-validator warning is informational
  only — `cargo run` continues to succeed; the warning appears in the
  log output and any `--validate` report.
- Postcard MPI broadcast is forward-compatible **only** if
  `VariableRef::AnticipatedDecision` is appended at the end of the enum
  (see § 5.1).
- JSON schema is forward-compatible: the new identifier name is additive;
  cases that don't use it are unaffected.
- No version bump strictly required; v0.7.x minor-feature release is
  appropriate.

---

## 9. Implementation plan (atomic tickets)

The work decomposes into 7 sequential atomic tickets. Each must compile
and pass tests independently before the next ticket starts. Suggested
branch: `feat/anticipated-decision-variableref`.

### Ticket 1 — Core enum variant

- **Files**: `crates/cobre-core/src/generic_constraint.rs`
- **What**: Add `VariableRef::AnticipatedDecision { thermal_id }` at the
  END of the enum (postcard wire-format safety, § 5.1). Add doc comment
  per § 4.1.
- **Tests**: AC-1, AC-2 (construction + postcard round-trip).
- **Effort**: S

### Ticket 2 — Expression parser

- **Files**: `crates/cobre-io/src/constraints/generic.rs`
- **What**: Add `anticipated_decision` arm to `build_variable_ref` per
  § 4.2.1. Update module doc comments per § 4.2.2.
- **Tests**: AC-3, AC-4, AC-5, AC-6.
- **Effort**: S

### Ticket 3 — JSON schema

- **Files**: `book/src/schemas/generic_constraints.schema.json`
- **What**: Update the `expression` field description per § 4.2.3.
- **Tests**: none required (schema is informational).
- **Effort**: S

### Ticket 4 — Referential validator

- **Files**: `crates/cobre-io/src/validation/referential.rs`
- **What**: Extend the thermal arm of `validate_variable_ref_entity` per
  § 4.4. Verify error-message stability (existing tests that pin
  message wording shouldn't drift).
- **Tests**: AC-7.
- **Effort**: S

### Ticket 5 — Semantic validators

- **Files**: `crates/cobre-io/src/validation/semantic/thermal.rs`,
  `crates/cobre-io/src/validation/semantic/mod.rs`
- **What**: Add `check_anticipated_decision_target_is_anticipated` and
  `warn_thermal_generation_on_anticipated_thermal` per § 4.5. Register
  both in `mod.rs`. Add `ErrorKind::SemanticAmbiguity` if a suitable
  warning kind does not already exist.
- **Tests**: AC-8, AC-9, AC-10, AC-11.
- **Effort**: M

### Ticket 6 — LP resolver

- **Files**: `crates/cobre-sddp/src/generic_constraints.rs`
- **What**: Add the `AnticipatedDecision` match arm and the
  `resolve_anticipated_decision` helper per § 4.3.
- **Tests**: AC-12, AC-13, AC-14.
- **Effort**: S

### Ticket 7 — Integration tests + documentation

- **Files**:
  - `crates/cobre-sddp/tests/anticipated_generic_constraint_e2e.rs` (new)
  - `book/src/reference/case-format.md`
  - `book/src/guide/thermal-units.md`
  - `CHANGELOG.md`
  - Optionally: extend the F3-004 order-invariance probe in
    `crates/cobre-sddp/src/lp_builder/template.rs::tests`
- **What**: AC-15, AC-16; book and CHANGELOG updates per § 6.
- **Tests**: full workspace gate (`cargo nextest run --workspace --all-features`).
- **Effort**: M

**Total effort**: ~2–3 days for a focused implementer; the medium
tickets (5 and 7) carry most of the work.

---

## 10. Risks

### 10.1 Postcard enum-discriminant drift

If `VariableRef` is appended-to in some future PR without checking the
declaration order, MPI broadcast of generic constraints could silently
deserialise the wrong variant on workers. Mitigation: add a postcard
round-trip regression test (AC-2) that pins the discriminant order by
asserting the serialised byte sequence for each variant. **Required**
for this change to land safely.

### 10.2 Warning fatigue

If a case has many `thermal_generation(...)` references against
anticipated thermals (which IS a legitimate use case — capping
per-block dispatch in addition to commitment), the warning fires once
per term. Consider deduplicating: emit one warning per
`(constraint_id, thermal_id)` pair rather than one per term. Implementation
detail to verify during ticket 5; not a design blocker.

### 10.3 LP solution drift

Adding the resolver arm changes nothing about existing constraints,
but the new resolver helper must produce a column index consistent with
the rest of the LP builder. Bit-for-bit determinism of pre-existing
training runs is the regression gate (AC-15's baseline-comparison test
implicitly checks this — if a fixture without the new constraint
produced a different LB, the resolver is broken).

### 10.4 Existing CLI / examples missing the warning

The new warning will fire on real production cases that already use
`thermal_generation` on anticipated thermals. Audit `examples/`,
`crates/cobre-cli/templates/`, and any D-case fixture that uses generic
constraints with anticipated thermals. Either:

- (a) update fixtures to use `anticipated_decision` if the constraint
  was meant to bind the commitment, or
- (b) silence the warning per-constraint via a future `#[serde(rename =
"intentional_per_block_cap")]` annotation (out of scope for this
  ticket).

For now: simply audit and document. Most likely there are zero current
fixtures that hit this case (the anticipated-thermals feature is recent;
no production fixture used generic constraints with anticipated thermals
before this work).

---

## 11. Open Questions / Deferred

### 11.1 Why is `AnticipatedState { slot }` deferred?

The ring buffer's slot index is an LP-internal concept (slot 0 = oldest
pending = matures this stage; slot K_i - 1 = newest = just placed).
Exposing it to user constraints invites several footguns:

- Users must know that slot 0 has different meaning per plant (it
  matures THIS stage, so it equals the commitment placed K stages ago).
- Slots `K_i..K_max` are deterministically zero padding for plants
  whose `K_i < K_max` — a user constraint over those slots is
  structurally meaningless.
- The slot indexing convention may change in a future release if we
  add features like "freeze a specific historical commitment" or
  "stochastic anticipated commitments". Exposing the current layout
  pins it.

The 80% use case is "constrain commitments" — `AnticipatedDecision`
covers that cleanly. If a real use case for slot-level constraints
emerges, open a follow-up ticket then; the design space for that
variant is meaningfully different (per-slot block index, validator
warning for padding slots, etc.) and warrants its own RFC.

### 11.2 Should the warning be silenceable?

If the user explicitly wants the stage-dependent semantics of
`thermal_generation(i, blk)` on an anticipated thermal (e.g., capping
per-block dispatch to handle ramping), the warning is noise. Options:

- (a) Leave it always-on; user filters at the log level.
- (b) Add a per-constraint `silence_anticipated_warning: true` flag in
  JSON.
- (c) Add a global `cobre.toml` or CLI flag to silence the warning class.

**Recommendation**: (a). Warnings are advisory; users who deliberately
use `thermal_generation` on anticipated thermals can grep them out of
log output. (b) and (c) add config surface area that is hard to justify
without a real complaint. Defer (b)/(c) to a future ticket if user
feedback demands it.

### 11.3 Should `anticipated_decision` accept a `lead_offset` argument?

If a future feature lets users constrain commitments destined for
specific delivery stages (e.g., "limit total commitment delivering at
stage 5"), the variant would need a `delivery_stage` field. **Decision**:
out of scope for v1. The current commitment column is a per-stage
scalar; multi-stage aggregations can be expressed by writing N
constraints (one per stage) with the per-stage filter
(`stage_id: Some(5)`).

---

## 12. Acceptance criteria summary

The ticket is complete when all of the following hold:

- [ ] `VariableRef::AnticipatedDecision { thermal_id }` exists in
      `cobre-core` and is `Serialize + Deserialize` round-trip safe
      via postcard.
- [ ] `parse_expression("anticipated_decision(N)", ...)` produces the
      expected `LinearTerm`.
- [ ] `parse_expression("anticipated_decision(N, blk)", ...)` returns
      an actionable error.
- [ ] Referential validation rejects unknown thermal ids in
      `anticipated_decision(...)`.
- [ ] Semantic validation rejects `anticipated_decision(...)` on a
      non-anticipated thermal with a clear error message.
- [ ] Semantic validation **warns** when `thermal_generation(...)` is
      used on an anticipated thermal, suggesting
      `anticipated_decision(...)`.
- [ ] LP resolver maps the new variant to the correct column index.
- [ ] An end-to-end test demonstrates the constraint actually binds the
      commitment (LB worsens vs unconstrained baseline).
- [ ] Order-invariance probe extended to cover the new variant.
- [ ] `cargo nextest run --workspace --all-features` passes.
- [ ] `cargo clippy --workspace --all-features --tests -- -D warnings`
      clean.
- [ ] `cargo fmt --all --check` clean.
- [ ] `mdbook build book` clean.
- [ ] CHANGELOG updated.
- [ ] book `case-format.md` and `thermal-units.md` updated.

---

## 13. References

- `assessment-report.md` — original F2-006 finding (Major,
  architecture).
- `crates/cobre-core/src/generic_constraint.rs` — the `VariableRef`
  enum and `LinearTerm`/`GenericConstraint` data model.
- `crates/cobre-io/src/constraints/generic.rs` — the expression-string
  parser and the name → variant mapping.
- `crates/cobre-io/src/validation/referential.rs` — entity-id
  referential validation for variable refs.
- `crates/cobre-io/src/validation/semantic/thermal.rs` — thermal-
  domain semantic checks (where the new checks live).
- `crates/cobre-sddp/src/generic_constraints.rs` — LP-builder resolver
  from `VariableRef` to LP column index.
- `crates/cobre-sddp/src/indexer.rs` —
  `StageIndexer::anticipated_decision`,
  `StageIndexer::anticipated_thermal_indices`.
- `book/src/schemas/generic_constraints.schema.json` — published JSON
  schema (user-visible).
- `book/src/reference/case-format.md` and
  `book/src/guide/thermal-units.md` — user-facing documentation.
- F2-002 commit `1ee0416` — strict boundary predicate; the new column
  has `[0, 0]` bounds at the boundary stage.
- F3-004 commit `e3f9e47` — declaration-order-invariance probe; extend
  for the new variant.
