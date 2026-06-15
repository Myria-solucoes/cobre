# Hydro-topology parity — gap tracker

**Purpose.** Track the remaining gaps between cobre's hydro cascade / production
model and DECOMP's, so we can plan and sequence the work. This is a **living
document** — update the status cells as items land.

**Scope.** Two gap families surfaced while reading the DECOMP user manual
(§3.4.6.x) and the methodology reference (ch. 4–5):

- **Family A — FPHA tailrace lateral flow.** DECOMP's "lateral flow" registers
  (VL/VU/VA, manual §3.4.6.5) compose a _tailrace flow_ `Q_jus` that feeds the
  level polynomial → net head → FPHA. They are an **FPHA-head** feature; they do
  **not** add mass to any reservoir balance.
- **Family B — mass-balance / cascade routing.** Everything that changes _where
  water mass goes and when_. These are **water-balance** features, independent of
  FPHA.

> **The boundary that matters for scoping.** Family A is intrinsically
> FPHA-coupled — the influence-graph _data_ can be modeled FPHA-agnostically, but
> the _behavior_ only appears once the FPHA `Q_jus` consumes it. Family B is the
> work that improves the hydro balance **without touching FPHA**.

**Status legend:** ✅ supported / done · 🟡 partial / low-priority · ❌ real gap
**Size:** S (additive field/config) · M (new LP terms / input stream) · L (structural — touches balance/state)

---

## Family A — FPHA tailrace lateral flow (manual §3.4.6.5) — **DEFERRED**

> **[DEFERRED]** Parked by decision. Family A is FPHA-coupled (≈ the deferred
> `plans/fpha-tailrace-modeling/` epic-04). The table below is the future-version
> spec; revisit when a case needs lateral-confluence tailrace composition.

DECOMP composes the tailrace flow that the level polynomial consumes as:

```
Q_jus(i) = f_self · Outflow_i
         + Σ_{j ≤ 3} f_j · Outflow_j        (other plants' defluence)
         + Σ_{g ≤ 3} f_g · IncInflow_g       (flow-gauge incremental inflow)
```

where `Outflow = turbined + spilled` (spill counts only if the plant's cadastre
flags it — manual obs. 4), `cota = poly(Q_jus)`, and `poly` is the backwater
family. Constraints: ≤3 contributing plants and ≤3 gauges per plant; a plant may
influence only one other plant's tailrace; for gauges, the **mean** incremental
flow over scenarios is used (manual obs. 1–5).

**cobre today.** `Q_jus = own (turbined + spilled)` only. The **backwater
(remanso)** mechanism is already present: `tailrace_curves` families keyed by
`downstream_reference_level_m` are DECOMP's FJ/CURVAJUS curves. What is missing is
the **lateral-confluence composition** — other plants' defluence and gauge
incremental flow joining `Q_jus`. The fix is mostly **fields on the hydro
production / FPHA model definition** (`production_models.rs`) plus wiring the
composed `Q_jus` into the FPHA row.

| #   | Gap item                              | DECOMP                       | Proposed cobre field                                         | Status | Size |
| --- | ------------------------------------- | ---------------------------- | ------------------------------------------------------------ | :----: | :--: |
| A1  | Downstream-level backwater families   | FJ / CURVAJUS + remanso flag | `tailrace_curves` + `downstream_reference_level_m` (shipped) |   ✅   |  —   |
| A2  | Own-defluence participation factor    | VL field 3                   | `own_outflow_factor: f64` (default 1.0)                      |   ❌   |  S   |
| A3  | Spill-counts-in-tailrace flag         | VL obs. 4                    | `spill_affects_tailrace: bool`                               |   🟡   |  S   |
| A4  | Other-plant defluence contributors    | VU (≤3)                      | `tailrace_outflow_contributors: [{source_hydro_id, factor}]` |   ❌   |  M   |
| A5  | Gauge incremental-inflow contributors | VA (≤3)                      | `tailrace_inflow_contributors: [{gauge_id, factor}]`         |   ❌   |  M   |
| A6  | Per-gauge incremental-inflow input    | VA data + obs. 5             | new optional per-gauge incremental-inflow stream             |   ❌   |  M   |

---

## Family B — mass-balance / cascade routing

cobre's balance is built per hydro per stage in
`crates/cobre-sddp/src/lp/builder/matrix.rs::fill_state_and_water_entries`. Routing
edges are `downstream_id` (factor 1.0) and `DiversionChannel{downstream_id,
max_flow}`; all routing is **same-stage**.

> **Audit status (verified against the code).** Every "already supported?"
> hypothesis below was checked with file-level evidence. The recurring enabler is
> the **per-stage hydro-bounds override** path
> (`constraints/hydro_bounds.parquet` → `HydroBoundsRow` → `resolve_bounds` →
> `ResolvedBounds.hydro` → LP column bounds): all 11 hydro bound fields are
> individually overridable per `(hydro, stage)`, sparse. This is what makes
> B5/B7/B10 work today. Consistent limitation: these overrides are
> **per-stage, not per-block**.

### Core routing / balance

| #   | Gap item                                  | DECOMP                                                                                            | cobre status (audited)                                                                                                                                                                                                                          | Status |  Size   |
| --- | ----------------------------------------- | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----: | :-----: |
| B1  | Water travel time between plants          | VI/QI; methodology §4.5.14.2 + ch. 5.3 (`t−tv` delayed-release terms in every balance equation)   | same-stage routing only                                                                                                                                                                                                                         |   ❌   |  **L**  |
| B2  | Consumptive withdrawal with return (DA)   | §4.5.8.2 — soft withdrawal (non-attendance cost) + return fraction α to same/other plant          | soft withdrawal at same plant ✅ (slacks + penalty); **cross-plant coupled return ✗** — but emulatable via signed withdrawals (`+Qda` at i, `−α·Qda` at j)                                                                                      | 🟡 low |   (M)   |
| B3  | Withdrawal / irrigation, signed return    | §4.5.8.1 — per-stage withdrawal, negative = return                                                | **FULLY SUPPORTED per-stage**, signed (test `test_water_withdrawal_negative_accepted`); RHS `ζ·(base − withdrawal)`, sign-aware slacks. Not per-block.                                                                                          |   ✅   |    —    |
| B4  | Per-block water balance (run-of-river)    | §4.5.15 — optional per-plant; stops storage-less plants buffering across intra-day blocks         | per-**stage** balance only (flows summed across blocks) → RoR plants can currently buffer intra-stage                                                                                                                                           |   ❌   |    M    |
| B5  | Minimum defluence per stage               | §4.5.11 / RQ — hard, `%` of historical min or fixed                                               | **FULLY SUPPORTED per-stage** via `min_outflow_m3s` override → constraint RHS. Not per-block (row replicated, shares stage value).                                                                                                              |   ✅   |    —    |
| B6  | Generic linear restrictions (RHA/RHQ/RHV) | §4.5.13 — flows per block (RHQ), volumes per stage (RHV), inflow (RHA), two-sided, stage validity | **PARTIAL.** `generic_constraint` is per-block, two-sided, stage-gated; covers turbined/spilled/diverted/stored/generation. **Gaps:** B6a RHA inflow not expressible (no inflow `VariableRef`); B6b pumping terms silently no-op (gated on B9). |   🟡   | S / +B9 |

### Adjacent (audited)

| #   | Gap item                          | DECOMP                                                          | cobre status (audited)                                                                                                                                                         | Status | Size |
| --- | --------------------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :----: | :--: |
| B7  | Flood-control / wait volume       | §4.5.12 / VE — per-stage max-storage cap                        | **FULLY SUPPORTED** — per-stage `max_storage_hm3` override → outgoing-storage column upper bound (doc: "flood control level")                                                  |   ✅   |  —   |
| B8  | Reservoir / dead-volume filling   | §4.5.9 / VM·DF — min storage target + min defluence during fill | **real, UNFINISHED gap** — `FillingConfig` + I/O + `HydroStageBounds.filling_inflow_m3s` exist but the LP **never consumes** them (all reads literal `0.0`; penalties unwired) |   ❌   | S–M  |
| B9  | Pumping stations / pumped storage | §4.4.3 / QBOM·VBOM (unidades elevatórias)                       | **real gap** — `PumpingStation` is an explicit **NO-OP stub**: full input pipeline (JSON, `pumping_bounds.parquet`, bounds slot) but **zero LP columns/rows**                  |   ❌   | M–L  |
| B10 | Maintenance / availability        | §4.5.10 / MP·FD — per-stage availability factor                 | **FULLY SUPPORTED** — per-stage `max_generation_mw` / `max_turbined_m3s` overrides → LP column bounds (cobre-bridge converts NEWAVE factors this way)                          |   ✅   |  —   |

### Audit conclusions

- **Closed by the audit (no work needed):** **B3, B5, B7, B10** — served by the
  per-stage `hydro_bounds.parquet` override path or existing slacks. Your
  hypotheses were correct.
- **Downgraded:** **B2** → low-priority / emulatable. DA is a _consumptive
  withdrawal with return_, not cascade routing; cobre's soft withdrawal already
  covers the same-plant case, and a cross-plant return is emulatable with signed
  withdrawals. The only thing genuinely missing is the _coupled_ return-shrink
  when a withdrawal is infeasible — rare in practice.
- **Confirmed remaining real gaps:** **B1** (travel time, L) · **B4** (per-block
  run-of-river balance, M) · **B8** (filling — finish LP consumption, S–M) ·
  **B9** (pumping — wire LP columns/rows onto the existing stub, M–L). Plus two
  small **B6** items: **B6a** expose realized inflow as a `VariableRef` (enables
  RHA) and **B6b** reference pumping in generic constraints (gated on B9).
- **Cross-cutting sub-theme — per-block hydro bounds.** B3 and B5 are per-_stage_
  only; B4 (per-block RoR) and full RHQ (per-block flows) both want per-_block_
  bounds/balance. A per-block hydro-bounds capability would serve several items at
  once and is worth designing as a shared primitive.
- B1 (travel time) is the only item that reshapes the balance + state; everything
  else is additive.

---

## Cross-cutting requirements (apply to every item)

- **Declaration-order bit-determinism.** Canonical ordering for any new
  contributor/destination/gauge list; cross-plant/cross-stage coupling must be
  order-invariant (mirror the existing cascade coupling).
- **Inert defaults.** Every new field defaults to today's behavior so existing
  deterministic cases stay **byte-identical** — no parity re-bless unless a case
  opts in. Re-bless both `parity_baselines/` and `parity_baselines_clp/` plus the
  `EXPECTED_HASHES` tripwire when a case does opt in.
- **Python parity.** Any new input/output the CLI reads/writes must be mirrored in
  `cobre-python`.
- **Genericity rule.** Names in `cobre-core`/`cobre-io` stay generic English (no
  CEPEL/NEWAVE/DECOMP terms in identifiers, comments, or shipped docs); the DECOMP
  register is cited here only for traceability.
- **Both LP backends green** (HiGHS default; CLP via `--no-default-features --features clp`).

---

## Suggested sequencing (Family B, FPHA-free)

```
1. B8  filling — finish LP consumption of the already-modeled FillingConfig   (S–M)
2. B6a expose realized inflow as a VariableRef → enables RHA restrictions      (S)
3. B4  per-block run-of-river balance (+ the shared per-block-bounds primitive) (M)
4. B9  pumping — wire LP columns/rows onto the existing stub                   (M–L)
   └─ B6b pumping in generic constraints follows for free
5. B1  travel time — structural (balance + state + pre-study lags); design first (L)
—  B2  emulate via signed withdrawals if/when a case needs cross-plant return  (opt)
```

Rationale: B8 and B6a are low-risk finishes of work already half-present; B4
introduces the per-block-bounds primitive that B-family per-block items reuse; B9
unlocks B6b; B1 is the structural one and should be designed deliberately (it is
the only item that touches the state vector).

---

## References

- DECOMP user manual §3.4.6.5 (VL/VU/VA), §3.4.6.6 (VI/QI travel time),
  §3.4.6.8–3.4.6.9 (TI/DA), §3.4.6.10 (VM/DF), §3.4.6.14–3.4.6.18
  (RQ/VE/RHA/RHQ/RHV).
- DECOMP methodology reference §4.4.3 (pumping units), §4.4.6 / 5.3 (travel time),
  §4.5.8 (withdrawals/diversions), §4.5.9–4.5.13 (filling/maintenance/min
  defluence/flood-control/special restrictions), §4.5.15 (per-block RoR balance).
- cobre per-stage override path: `crates/cobre-io/src/resolution/bounds.rs`
  (`resolve_bounds`), `crates/cobre-io/src/constraints/bounds.rs` (`HydroBoundsRow`
  / `PumpingBoundsRow`), `crates/cobre-core/src/model/resolved/bounds.rs`
  (`HydroStageBounds`).
- cobre LP: `crates/cobre-sddp/src/lp/builder/matrix.rs`
  (`fill_state_and_water_entries`, withdrawal slacks, column bounds),
  `crates/cobre-sddp/src/lp/generic_constraints.rs` (`VariableRef` resolution),
  `crates/cobre-core/src/entities/{hydro.rs,pumping_station.rs}`,
  `crates/cobre-core/src/constraints/generic_constraint.rs`.
- Deferred plan: `plans/fpha-tailrace-modeling/epic-04-lateral-composition/`
  (≈ Family A).
