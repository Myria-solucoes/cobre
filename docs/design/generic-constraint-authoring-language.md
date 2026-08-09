# Generic-Constraint Authoring Language — Findings & Design Proposal

**Status:** evaluation + design proposal. Not approved, not implemented.
**Date started:** 2026-08-08.
**Motivation:** cobre-bridge is a _transition-only_ CLI (DECOMP → cobre migration).
The end state is users authoring constraint and parameter data **directly in cobre**.
Anything a DECOMP user relies on today, a future cobre-native author must be able to
express directly and ergonomically. This document records what we learned evaluating
DECOMP's "restrições elétricas especiais" (LIBs format) and proposes how to port its
authoring ergonomics into a coherent cobre-native surface.

This is a design record, not a plan. It captures findings, the target design, verified
gaps, open decisions, and a desiderata list, so the direction is not lost.

---

## 1. What the DECOMP feature actually is

DECOMP "restrições elétricas especiais" (CEPEL LIBs manual,
`restricoes_operativas/restricoes_eletrica/`) are **not** piecewise-linear constraints
and carry **no concavity/convexity requirement** — a full read of the spec and of a real
deck (`decomp-abr-26-lpp`) found zero piecewise / convex / segment terminology. ("LPP" in
the deck header is a study label, never a card.)

They are a small **algebraic DSL for plain linear constraints** over model quantities:

```
LI(t,p) ≤ Σ kₕ·GHₕ + Σ kₜ·GTₜ + Σ kᵢ·Intᵢ + Σ k_c·g_c ≤ LS(t,p)
```

per stage `t` and load block `p`, layered with authoring ergonomics:

- **Named linear sub-expressions** (`EXPRESSAO-ELETRICA`), composable and referenceable.
- **Named scalar aliases** varying by (period, patamar) (`ALIAS-ELETRICO` + `…-VALOR-PERIODO-PATAMAR`).
- **Two-sided bounds** (`…-FORMULA` + `…-LIMITES-FORMULA-…`) or **one-sided inequations**
  with an **arbitrary linear RHS** (`RESTRICAO-ELETRICA-INEQUACAO`).
- **Data-conditional bound selection** (`se(cond, X, Y)`).
- **Conditional activation** (`REGRA-ATIVACAO` + `HABILITA`), enable/disable per
  (period, patamar).
- **Soft/hard + penalty cost** (`TRATAMENTO-VIOLACAO`).

**The property that makes the whole thing portable to cobre's LP-only engine:** both the
`se()` conditionals and the activation-rule conditions are restricted _by spec_ to **input
variables** ("variáveis de entrada"). They resolve at **build time** — nothing needs MILP,
indicators, or big-M. "Hard" is modeled as slack + penalty plus an infeasibility flag if
the slack binds.

---

## 2. cobre already has ~85% of the engine

cobre's generic-constraint subsystem is close to a superset of the DECOMP semantics:

- **Domain model** — `crates/cobre-core/src/constraints/generic_constraint.rs`:
  `GenericConstraint { id, name, description, expression, sense, slack }`;
  `ConstraintExpression { terms: Vec<LinearTerm> }`;
  `LinearTerm { coefficient: CoefficientRef, scale, variable: VariableRef }`;
  `ConstraintSense { GreaterEqual, LessEqual, Equal }`; a **24-variant `VariableRef`**
  catalog.
- **Parser** — `crates/cobre-io/src/constraints/generic.rs`: expression is a parsed string;
  grammar supports `coeff * @name * variable`, a `bus=` selector on two hydro variables,
  and `@name` scalar-parameter references.
- **Bounds** — `constraints/generic_constraint_bounds.parquet`, per `(constraint, stage, block)`;
  **row presence = activation** (absent ⇒ inactive that stage/block).
- **Scalar parameters** — `crates/cobre-core/src/model/parameters.rs`: `ScalarParameter { id,
name, kind }`, `ParameterKind ∈ { Constant, PerStage, Seasonal, Computed }`, referenced via
  `@name`; resolved by `ResolvedParameters` keyed `(param_id, stage_idx)`.
- **Soft constraints** — `SlackConfig { enabled, penalty }`; `Equal` already allocates
  **both** `slack_plus` and `slack_minus`.
- **LP seam** — `crates/cobre-sddp/src/lp/generic_constraints.rs` +
  `builder/{layout,entries}.rs`; generic rows appended after structural rows at
  `row_generic_start`.

The bridge already converts the **old** `restricao-eletrica.csv` (only `ger_usih` +
`ener_interc`, constant bounds) into cobre generic constraints. The new LIBs format is the
richer surface; the direction is to make cobre author it natively, not to keep it in the
bridge.

---

## 3. Vocabulary verification — abr-26 deck vs cobre (VERIFIED against live code)

Token inventory extracted from `lib_restricao-eletrica-especial.csv` (1,192 lines) and
mapped against cobre's accepted parser tokens (`generic.rs`) and `VariableRef`
(`generic_constraint.rs`).

### Decision-variable tokens

| DECOMP token       | uses | cobre representation             | Status                                                                                                                                                                                                                                                                                                                                                                                                                            |
| ------------------ | ---: | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ger_usih(X)`      |  150 | `hydro_generation(X)`            | ✅ direct                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ener_interc(X,Y)` |   50 | `line_exchange(line_id)`         | ✅ **Gap A resolved** — DECOMP submarket = cobre bus (1:1); `ener_interc(X,Y)` is the `line_exchange` of the line whose source bus is X and target bus is Y (sign per line orientation). Notation item only: add a `(source_bus, target_bus)` addressing form; detailed multi-line boundaries use a named-expression sum                                                                                                          |
| `ger_usit(X)`      |   41 | `thermal_generation(X)`          | ✅ direct                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `ger_pee(X)`       |   41 | `non_controllable_generation(X)` | ✅ direct (wind park = non-controllable source)                                                                                                                                                                                                                                                                                                                                                                                   |
| `ger_conjh(X,Y)`   |   22 | `hydro_generation(X, bus=b)`     | ✅ **Gap B resolved** (Itaipu case) — cobre's granularity is "the unit group connected to a bus": **same modeling, different notation.** Map DECOMP group `Y` to its cobre bus `b` and address `hydro_generation(X, bus=b)`; needs a group→bus lookup. Sole caveat: two groups on one bus collapse to a single column (`generic_constraint.rs:27`), so the group→bus map must be 1:1 — it is for Itaipu (distinct 50/60 Hz buses) |

### Input-quantity tokens (exogenous constants/parameters)

| DECOMP token                  | uses | nature                                                                                                                                                 | Status                                                                                                                                                                           |
| ----------------------------- | ---: | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `disp_usih(X)`                |   24 | available hydro power (capacity × availability)                                                                                                        | ➡️ expose as **builtin read-only parameter** — available power is cobre-derivable                                                                                                |
| `demanda_sin`                 |   21 | whole-system load = sum of all bus loads                                                                                                               | ➡️ expose as **builtin read-only parameter** (system load) — but see stochastic-load note (§6)                                                                                   |
| `val_demanda(X)`              |    7 | bus-X load **value** (constant; no dual contribution — the CMO-safe form)                                                                              | ➡️ expose as **builtin read-only parameter** (bus load) — see §6                                                                                                                 |
| `demanda(X)`                  |    6 | bus-X load (dual-contributing in DECOMP)                                                                                                               | ➡️ value → builtin load parameter; the CMO dual-leak semantics (**Gap D**) are NOT portable (§5)                                                                                 |
| `carga_ande`                  |    2 | ANDE (Paraguay) load served by Itaipu 50 Hz; per-patamar constant from the DECOMP **RI** (or **IT**) register, bundled with Itaipu 50/60 Hz gen limits | ➡️ deterministic external per-(stage,block) constant → **user-declared parameter** (concrete driver for the (stage,block) param axis, feature #3); NOT stochastic, NOT a builtin |
| `peq_N_{PCH,PCT,EOL,UFV}gd_N` |    4 | small distributed generation aggregates (North) — used but undefined ⇒ builtin                                                                         | ➡️ external — user-declared parameters (cobre has no distributed-generation aggregate)                                                                                           |

### Ergonomics tokens

| DECOMP construct                             |                                                      count | cobre status                                                                                                                                                                                             |
| -------------------------------------------- | ---------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| named expressions (`EXPRESSAO-ELETRICA`)     | **50 defined**, heavily reused (`FNESE` 62×, `FNS` 56×, …) | ❌ **ABSENT** — no named sub-expressions (feature #1)                                                                                                                                                    |
| per-(stage,block) aliases (`ALIAS-ELETRICO`) |                             **13 defined**, 234 value rows | ⚠️ params are **per-stage only**, no block axis (feature #3). In this deck the aliases are used as **constants** (demand/MMGD values) on the value/bound side, not as coefficients on decision variables |
| `re(X)` (reference another RE's formula)     |                                             0 in this deck | subsumed by named expressions                                                                                                                                                                            |

**Headline (post owner review, 2026-08-08):** the three highest-use decision-variable tokens
(`ger_usih`, `ger_usit`, `ger_pee` — 232 uses) map **directly** today. Gaps A and B are
**notation only, resolved:** DECOMP submarkets are cobre buses so `ener_interc(X,Y)` is the
line between bus X and bus Y; DECOMP unit groups are cobre unit-groups-on-a-bus (same model,
addressed via `bus=`). For input quantities (Gap C), quantities cobre owns (demand, available
power) are exposed as **builtin read-only parameters**; DECOMP mnemonics with no distinct
cobre meaning (`carga_ande`, distributed-generation aggregates) are just load/parameter
values, not special builtins. The deck is overwhelmingly built on **named expressions** and
**per-(stage,block) aliases**, confirming features #1 and #3 are the backbone. One new
consideration surfaced: cobre supports **stochastic load**, so demand references are not
automatically build-time constants (§6).

---

## 4. Target design — one cobre-native authoring language

Organizing principle: **all sugar lives in an authoring/desugaring layer in
`cobre-io`/setup and compiles down to the existing core** (flat `Vec<LinearTerm>` +
per-(stage,block) bounds + `SlackConfig`) **byte-identically, before the LP is built.**
The hot path, determinism, and cut machinery never see the sugar. Invariant test for every
feature: **the resolved LP is byte-identical to the hand-flattened version.** This protects
declaration-order invariance and run-to-run reproducibility no matter how much sugar is
added.

Components (one shared expression grammar, one `@name` namespace):

1. **Named expressions** — define a linear combination once, reference by name anywhere
   (LHS, bounds, conditions, other expressions). Inlined at setup with cycle detection and
   coefficient distribution. Pure pre-LP flattening; zero solver risk. (Gap A's base case is
   a direct line lookup by bus pair; named expressions cover multi-line boundaries in
   detailed models.)
2. **Two-sided / range constraints** — one authored `LI ≤ expr ≤ LS`, one id, one reported
   band. See §6 for the internal-representation decision.
3. **Parameters with a (stage, block) axis + user-valued tables** — extend `ResolvedParameters`
   from `(param, stage)` to `(param, stage, block)`; usable in coefficient, bound, and
   condition positions. Enables "declare `demanda_sin` once, reference `@demanda_sin` in many
   bounds" (DRY, single-source) instead of pre-baking numbers into every parquet row.
4. **Arbitrary linear RHS** — `flow <= base + margin`, normalized to LHS at parse time.
5. **Conditional bounds** (`se(cond, X, Y)`-equivalent) — resolved at build time, where
   `cond` references `@name` **parameters** (reuse the parameter resolver; no parallel
   input-binding).
6. **Activation rules** — an authored condition over `@name` parameters, resolved at build
   time into the existing row-presence gating (the ergonomic wrapper over "omit the row").
7. **Echo / eco output** — dump the fully-resolved flat constraints per (stage, block):
   the debuggability mitigation for all the indirection, doubles as the bridge-comparison
   artifact and a Python-parity output. (DECOMP itself emits `oper_re_formulas.csv` etc.)

**Builtin read-only quantities (design insight from Gap C).** For quantities cobre already
owns — system/bus demand, available hydro power — prefer exposing **builtin parameters**
the author can reference without re-declaring, rather than forcing a hand-copied value that
can drift from cobre's actual load/availability. Genuinely external quantities (`carga_ande`,
distributed-generation aggregates) stay user-declared parameters.

---

## 5. What is NOT portable (DECOMP model accidents, not ergonomics)

- **Gap D — `demanda(X)` vs `val_demanda(X)` → CMO.** DECOMP's `demanda(X)` makes the
  constraint's dual contribute to the submarket marginal cost; `val_demanda(X)` does not.
  This exists only because DECOMP treats demand as a variable whose dual can leak into the
  price. cobre's load is exogenous, so both collapse to a constant term and there is no
  dual-leak distinction to reproduce. Document it; do not port it.
  (Submarket interchange is **not** in this list — it is resolved: a DECOMP submarket is a
  cobre bus, so `ener_interc(X,Y)` is the line between bus X and bus Y. See §3 / §6.2.)

---

## 6. Open decisions / correctness forks

0. **Bounds/sense model — DECIDED (owner 2026-08-08, FINAL, supersedes the interim F2): F3 —
   the interval IS the constraint; drop the authored `sense`.** A generic constraint carries two
   nullable endpoint columns, `bound_lower` / `bound_upper` (both `Option<f64>`), per
   `(stage, block)`. `sense` is **removed from the authored `generic_constraints.json`** — it is
   redundant, because the shape is fully derivable from which endpoints are finite: lower-only ⇒
   `>=`, upper-only ⇒ `<=`, both-equal ⇒ `==`, both-differ ⇒ range, both-null ⇒ error. Internally,
   shape is **derived from the bounds** wherever the LP builder / slack allocation / violation
   reporting need it (a one-sided row ⇒ one slack + single-slack report; a two-sided row ⇒ two
   slacks + net report) — never a second authored field that could disagree with the bounds.
   `generic_constraints.json` becomes `{id, name, description, expression, slack}`.

   **Why F3 over F2 (the reversal, owner-confirmed):** F2 kept `sense` as a per-constraint fixed
   shape, which (a) is a second source of truth the whole per-sense validation existed only to
   reconcile, and (b) **cannot represent a constraint whose active sides vary by period** — exactly
   what DECOMP produces, since its per-(period,patamar) `LI`/`LS` allow either side to be
   independently unbounded (`±1e21`). F3 models that natively (a row leaves an endpoint `null`);
   F2 would force a `1e21` sentinel or a constraint split. F3 also simplifies the LP builder to
   `row_lower = bound_lower.unwrap_or(-inf); row_upper = bound_upper.unwrap_or(+inf)` and dissolves
   the sense-token question.

   **Accepted costs:** no author-declared shape to catch a typo (a stray endpoint silently reshapes
   a row — mitigated by requiring ≥1 endpoint per row and `bound_upper >= bound_lower` when both
   present); shape is read from the bounds / the epic-03 echo output rather than a JSON field; a
   declared `==` is indistinguishable from a degenerate `[v,v]` band (LP-identical, moot).

   **Rejected:** F2 (keep `sense`) — the ratified-then-reversed interim; and `==`-uses-lower-only
   (asymmetric). This is a **clean break** to the released `generic_constraints.json` (drops the
   `sense` field) and its schema — sanctioned pre-release, all writers in-house, matches the
   clean-break directive.

   **Blast radius / sequencing:** larger than F2 — it partially unwinds ticket-006 (removes the
   `Range` sense variant/token/schema entry and the per-sense validation) and drops `sense` from
   every `generic_constraints.json` fixture + the `sense` field in
   `schemas/generic_constraints.schema.json` (schema **is** regenerated here, unlike F2). Landed by
   a re-scoped ticket-041 as the epic-02 capstone (after tickets 005–010, all uncommitted). The
   two-nullable-column work from 005/006 is reused; the sense machinery is removed. Lock-step
   cobre-bridge follow-up: its writer drops the `sense` output and emits `bound_lower`/`bound_upper`
   per the interval (4 sites: `pipeline.py:678`, `converters/constraints.py:735/1444/1765`, plus
   wherever it writes `sense`).

1. **Two-sided constraint internal representation — DECIDED (owner 2026-08-08, confirmed after
   specialist review): (B) native range row.** Implement as a `ConstraintSense::Range` arm that
   reuses the existing `Equal` two-slack / net-reporting stack. This reverses the initial (A)
   lean — the specialist showed (A) is the heavier, more-polluting lift and (B) carries no
   cut/basis/determinism risk. Evidence trail below.
   The owner's initial lean to (A) desugar-to-two-rows rested on "one-sided rows keep an
   unambiguous dual sign for the cut." The read-only specialist review found that premise
   **false in cobre**:
   - **Cut correctness is sign-agnostic.** The backward cut reads the incoming-state
     **column's reduced cost** (`training/backward/duals_extraction.rs:21-29`), not
     generic-row duals; by LP duality the reduced cost already folds in every row's dual
     regardless of which bound is active. Generic template-row duals are never read by cut
     assembly (only the always-one-sided FCF cut-row duals are). `Equal` already builds
     two-bounded rows whose active bound is not fixed at build time, and its cuts are already
     correct via this route — so there is no fixed-sign assumption to break.
   - **Warm-start is safe.** Range rows are template rows, copied verbatim by
     `reconstruct_template_row_statuses` (`cut/basis_reconstruct.rs:216-230`); the
     active-bound flip is carried by `BasisStatus::Upper`, a first-class round-tripping
     variant; `enforce_basic_count_invariant` counts only `Basic`. `Equal` proves this path
     already handles two-bounded rows.
   - **Determinism.** (B) is byte-closest to today's single-row-per-(constraint,stage,block)
     layout. (A) adds an ordering obligation (two rows at canonical positions, two
     deterministic ids).
   - **(A) is the MORE polluting option here**, precisely because `Equal` already provides
     two-bounded + two-slack + net-reporting: (A) doubles generic row count, forces a
     reporting re-collapse (extraction keys off `constraint.id`), and fragments band violation
     across two slacks/objective terms. (B) reuses the `Equal` stack near-verbatim (one id,
     one row, one report line).
   - **Shared io surface (needed by A _or_ B):** the bounds parquet carries one `bound` per
     (constraint, stage, block) (`GenericConstraintRowEntry.bound`); a band needs two values
     (LI, LS) → add a second bound column, a `ConstraintSense::Range` variant + a `Range{LI,LS}`
     arm in the row-bound match (`entries.rs:1288`), and extend two-slack allocation
     (`layout.rs:905-923`) to include `Range`.

   **Outcome: (B) confirmed** (see decision line above). (A) is recorded only as the rejected
   alternative and the reason it was rejected.

2. **Gap A — RESOLVED (owner).** DECOMP submarket = cobre bus (1:1). `ener_interc(X,Y)` is the
   `line_exchange` of the line whose source/target buses are X/Y. Remaining work is a notation
   convenience: a `(source_bus, target_bus)` addressing form for lines (mapping to the same
   column, with sign per line orientation). Named expressions cover multi-line boundaries in
   more detailed cobre models.
3. **Gap B — RESOLVED (owner, Itaipu case).** cobre's granularity is "unit-group-connected-to-a-bus":
   same modeling as DECOMP's `ger_conjh`, different notation. `ger_conjh(X,Y)` →
   `hydro_generation(X, bus=b)` via a group→bus lookup. Constraint: the group→bus map must be
   1:1 (two groups on one bus collapse to one column) — holds for Itaipu (distinct 50/60 Hz
   buses). Not a structural gap.
4. **Stochastic load vs demand references — DECIDED (owner 2026-08-08): resolve demand
   references to the load expected value.** A constraint referencing `demanda`/`demanda_sin`
   uses the deterministic **expected value** of the (possibly stochastic) load, resolved at
   build time — DECOMP-faithful (DECOMP uses a deterministic demand forecast). Per-realization
   demand-dependence is a recorded known limit, not supported. (`carga_ande` is deterministic,
   so it is unaffected.)
5. **`carga_ande` modeling — DECIDED direction (owner): model as a bus load OR an export
   contract; final choice driven by cobre-bridge discovery.** Discovery (2026-08-08): the
   bridge does **not** currently convert the DECOMP RI-register ANDE load into a cobre input;
   it only accounts for ANDE comparator-side, where NEWAVE treats it as **C_ADIC must-take /
   additional load** (`comparators/newave_readers.py:703-758`: `net_load = mercado_energia +
c_adic − Σ NCS`), and the comment notes **cobre's simulation output already includes
   C_ADIC**. So cobre already models ANDE-class must-take energy as **load** → the **bus-load**
   interpretation is the more consistent choice. Counter-signal: DECOMP's RI register treats it
   as a per-patamar **deduction from Itaipu 50 Hz** (`ger_conjh(66,1) − carga_ande`), which is
   export-like. Recommendation: **bus load** (consistent with cobre's existing C_ADIC handling);
   the export-contract alternative mirrors the DECOMP RI deduction more literally. Final call
   deferred to the bridge conversion design.
6. **Parameter file naming/location** (see §7).

---

## 7. Desiderata (feature list)

Ranked, all desugaring to the unchanged core at build time:

1. **Named expressions** (+ cycle detection, coefficient distribution) + **echo output**.
2. **Two-sided / range constraints** — DECIDED (B) native range row: a `ConstraintSense::Range`
   arm reusing the `Equal` two-slack/net-reporting stack (§6.1).
3. **(stage, block) + user-valued parameters**; allow `@name` in bound positions.
4. **Arbitrary linear RHS** normalization.
5. **Conditional bounds** over `@name` params (build-time resolved).
6. **Activation rules** over `@name` params → row-presence gating, with a **guard-rail
   validator** rejecting any condition that references a decision variable (the permanent
   LP-only vs MILP boundary; DECOMP already lives within it, so the port loses nothing).
7. **Builtin read-only quantities** for demand / available power (Gap C).
8. **Soft/hard ergonomics** — optional penalized-hard-with-diagnostic instead of raw
   infeasibility (secondary).
9. **`min_diversion_m3s` bound axis** — pairs the existing `max_diversion_m3s` to make
   hydro diversion a two-sided entity bound. The diversion LP column already exists
   (`lp/builder/columns.rs:fill_diversion_columns`, `col_upper` from resolved
   `max_diversion_m3s`); add an optional input field in `cobre-io/src/constraints/bounds.rs` +
   resolver + validator + one `col_lower` assignment. **NOT small (planner-measured):** the
   bound-row structs are constructed at ~192 exhaustive struct-literal sites across ~32 files
   (`HydroBlockBounds` 135, `HydroBoundsRow` 47, `HydroBlockOverride` 10) with no `Default`
   impl, so adding a field is a compile-atomic union edit. Owner directive (correct architecture
   over band-aid): extend the types uniformly **and add `Default`** so the next axis is O(1);
   the override-only shortcut is recorded as rejected. Lets DECOMP RHQ `QDES` lower limits
   lower to a bound instead of splitting one constraint across bound(upper)+generic(lower).
   (Surfaced by the bridge spec §6 M3 — 11 rv3 rows.)
10. **`min/max_spillage_m3s` bound axis** — cobre has **no** spillage bound axis today; the
    spillage LP column exists (`lp/builder/columns.rs:fill_spillage_columns`, currently
    `[0, ∞)`). Add optional inputs + resolver + validator + `col_lower`/`col_upper`. **Must
    respect** the `PreFilling` `[0,0]` pin and the frozen-storage/spillage determinism contract
    (columns.rs D38/D39). Lets DECOMP RHQ `QVER` lower to a bound. Low-priority tail (0 rv3
    rows). (Surfaced by the bridge spec §6 M5.)

**File move + rename (owner request, 2026-08-08).** Move the scalar-parameters input from
`system/scalar_parameters.json` into the `constraints/` folder alongside
`generic_constraints.json` + `generic_constraint_bounds.parquet`, and rename it to
`generic_parameters.json` (parallels `generic_constraints`; the params are conceptually part
of the generic-constraint feature, referenced by `@name`). Touch points: `cobre-io`
(`schema.rs` path table, `extensions/scalar_parameters.rs` path + docs, `lib.rs` doc),
`schemas/scalar_parameters.schema.json` → `schemas/generic_parameters.schema.json`
(regenerate via `cobre schema export`), the bridge writer (`converters/scalar_parameters.py`),
Python parity, and a migration note. Breaking input-layout change. _Naming caveat:_ if
parameters ever parameterize non-constraint inputs, a broader home may be preferable; current
usage is constraint-scoped, so `constraints/generic_parameters.json` is coherent.

---

## 8. Suggested sequencing

Substrate first, heaviest/lowest-frequency last; each phase ships with the byte-identity test.

1. Unified grammar + **named expressions** + **echo output**; **two-sided constraints**.
2. **(stage, block) parameters** + user-valued tables + **arbitrary RHS**; the
   `constraints/generic_parameters.json` move+rename.
3. **Conditional bounds** + **activation rules** + guard-rail validator (minimal condition
   grammar: comparisons + `&`/`|`, matched to real decks; steer per-stage/block value
   variation toward the bounds table, reserve conditionals for input-_driven_ rules).
4. Builtin read-only quantities; soft/hard ergonomics.

---

## 9. Relationship to the bridge conversion spec (cobre-side features surfaced)

The cobre-bridge feature spec `~/git/cobre-bridge/plans/decomp-special-constraints-feature.md`
converts the **classic `dadger`** special-constraint families (RE / RHQ / RHV / RHE; RHA
blocked on idecomp) into cobre inputs. It is **complementary** to this design and meets it at
one shared contract; it surfaces a few genuine cobre-side features to fold in here.

- **Shared flat-core contract (validates our organizing invariant).** The bridge is a machine
  generator: it emits the flat core directly — `expression` string + per-(constraint,stage,block)
  `generic_constraint_bounds.parquet` + `SlackConfig` — and needs **none** of the authoring
  sugar (named expressions, aliases, `se()`, activation). That flat core is exactly the
  byte-identical desugar target of §4. Two independent consumers (bridge output, cobre-native
  authoring) landing on the same core is confirmation the core is the right contract.
- **Format boundary.** The bridge owns the classic `dadger` registers; the richer **LIBs**
  electrical DSL (`lib_restricao-eletrica-especial.csv`) is the cobre-native authoring target
  of this design (idecomp cannot read LIBs). The bridge only **detects and warns** on LIBs
  usage; it does not convert it.
- **Standing stance (owner, carried in the bridge spec §4):** prefer a _cobre-side feature_
  over a bridge workaround, and prefer an _entity bound_ over a generic constraint wherever the
  mapping is faithful. This is the rationale for the two bound-axis additions (§7 #9, #10):
  rather than the bridge splitting a `QDES`/`QVER` constraint across a bound and a generic, add
  the missing bound axis in cobre so it lowers cleanly. Cross-check: these bound-axis gaps
  (`min_diversion_m3s` absent; **no** spillage axis) are verified against
  `cobre-io/src/constraints/bounds.rs` on `feat/rung1-tree`.
- **Native range row is a cross-benefit to the bridge.** The bridge spec currently emits a
  two-sided limit as **two** one-sided generic constraints (it predates our §6.1 decision and
  assumed the old two-row default; "no ranged form in cobre" is stated in its §7/G6). With
  `ConstraintSense::Range` (DECIDED, §6.1), the bridge can optionally collapse that pattern to a
  single range constraint. Equivalent either way, so no rework is forced — but the bridge spec's
  "no ranged form / two constraints" language should be updated once `Range` lands. **Action:
  flag this staleness to the bridge-spec owner** (this design doc does not edit the bridge spec).
- **File move (G11) is a shared dependency.** The bridge's RHE→VminOP emitter writes its
  `@rho_acum_h{id}` per-stage overrides into the parameters file; its E5 targets the renamed
  `constraints/generic_parameters.json` (§7 file move). Sequence the cobre rename and the bridge
  writer together.
- **Scalar-param scope clarification.** The bridge's RHE need is satisfied by the **existing
  per-stage** scalar parameters (`@rho_acum_h{id}`); it does **not** require the (stage,block)
  parameter axis. That axis (§7 #3) is a cobre-native authoring need (DECOMP `ALIAS-ELETRICO`
  per-patamar aliases, `carga_ande`), not a bridge requirement — keep it scoped as such to avoid
  over-building.

---

## Appendix — verification provenance

- Spec: CEPEL LIBs manual, `restricoes_operativas/restricoes_eletrica/restricoes_eletrica.html`.
- Deck: `~/git/cobre-bridge/example/decomp-abr-26-lpp/lib_restricao-eletrica-especial.csv`
  (indices key `RESTRICAO-ELETRICA-ESPECIAL`).
- cobre parser tokens: `crates/cobre-io/src/constraints/generic.rs` (string → `VariableRef`).
- cobre variable catalog: `crates/cobre-core/src/constraints/generic_constraint.rs`
  (`VariableRef`, and the bus-not-group addressability note at line 27).
- Unit groups: `crates/cobre-core/src/entities/hydro.rs` (`HydroUnitGroup`).
- Parameters: `crates/cobre-core/src/model/parameters.rs`,
  `crates/cobre-sddp/src/policy/resolved_parameters.rs`,
  `crates/cobre-io/src/extensions/scalar_parameters.rs`.
- Entity bound axes (§7 #9/#10, §9): `crates/cobre-io/src/constraints/bounds.rs`
  (`max_diversion_m3s` present, `min_diversion_m3s` and all spillage axes absent);
  `crates/cobre-sddp/src/lp/builder/columns.rs` (`fill_diversion_columns`,
  `fill_spillage_columns` — LP columns exist; spillage `PreFilling [0,0]` pin + D38/D39
  frozen-storage contract).
- Bridge conversion spec: `~/git/cobre-bridge/plans/decomp-special-constraints-feature.md`
  (classic RE/RHQ/RHV/RHE families; §0 shared-core alignment; §6 bound-axis menu M3/M5).
