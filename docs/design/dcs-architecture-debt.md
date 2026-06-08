# SDDP architecture review after Dynamic Cut Selection

> **Status**: Assessment / tech-debt note (2026-06-08). Read-only review; **no code
> changed**. Records where introducing Dynamic Cut Selection (DCS) added
> comprehension cost and mixed responsibilities to the `cobre-sddp` crate, with
> concrete code references and a remediation sketch. Line numbers are a snapshot of
> a tree under active edit — anchor on the symbol names, not the exact lines.
>
> **Companions:**
>
> - `docs/design/dynamic-cut-selection-design.md` — the DCS design this reviews.
> - `docs/design/backward-opening-ordering.md`, `docs/design/clp-hot-start-backward-pass.md`
>   — perf work in the same hot paths.
> - `.claude/rules/sddp.md` — the determinism / sign-scale contracts (all intact).

## TL;DR

The hypothesis — "DCS made the code harder to understand and mixed some
responsibilities" — is **directionally right, but the damage is concentrated, not
pervasive.** The DCS algorithm itself is cleanly encapsulated, the
infrastructure-crate genericity rule held, and every numerical contract survived.
What degraded is at **two joints**:

1. **Dual code paths in the three passes** — every pass now branches
   `if dcs { lazy } else { baked }`, and the backward pass carries two parallel,
   _drifted_ opening-solve functions. (≈18 DCS conditionals in shipped code.)
2. **Config-layer semantic overloading** — one `RowSelectionConfig` serves four
   methods via precedence-chained field reuse, including a deprecated field
   resurrected as an alias and a cadence field that doubles as a window.

Neither is a correctness bug; both are maintainability debt, fixable with two
well-scoped refactors (§7) once the perf direction settles.

## 1. What stayed healthy (so we don't over-correct)

- **`dcs.rs` is cohesive.** ~883 production lines, single concern: `DcsParams`,
  the scoring kernel (`score_violated_candidates`), the scratch types,
  `lazy_solve_preloaded`, `build_initial_resident_set`, `result_view`. The
  algorithm is not smeared across the crate.
- **Infrastructure genericity preserved.** No `dcs` / `dynamic` / `lazy_solve`
  references leaked into `cobre-core` / `cobre-io` (beyond the config method
  list) / `cobre-solver` / `cobre-stochastic` / `cobre-comm`. The rule most at
  risk under a feature like this held.
- **Closed-enum dispatch, not `Box<dyn>`.** `CutSelectionStrategy` is a sum type
  (`Level1` / `Lml1` / `Dominated` / `Dynamic`); `DcsParams::from_strategy`
  bridges to the algorithm without trait objects.
- **Contracts intact.** Determinism (bit-identical across rank counts), exactness
  (every lazy solve terminates at `violated == 0` or the TC-exact set), and the
  cut sign/scale conventions all held under repeated stress this cycle.

The bones are sound. The findings below are about the joints.

## 2. Finding 1 — Dual-path proliferation across the passes (highest comprehension cost)

DCS was added as a **parallel strategy beside** the baked path rather than
**behind** a shared abstraction, so each pass now opens with a strategy branch and
a reader must hold two models of "how a stage is solved."

**Forward pass** — `forward.rs:1062`:

```rust
let (view_objective, unscaled_primal): (f64, Vec<f64>) = if let Some(params) = dcs {
    build_initial_resident_set(pool, iteration, params.k2, &mut ws.backward_accum.dcs_initial_resident);
    let dcs_ctx = DcsSolveContext { /* ... */ continue_carry: false };
    lazy_solve_preloaded(&mut ws.solver, &ctx.templates[t], pool, /* ... */, &mut ws.backward_accum.dcs_solve, dcs_ctx)?;
    let view = ws.backward_accum.dcs_solve.result_view();
    let unscaled = /* unscale view.primal by col_scale */;   // (A)
    (view.objective, unscaled)
} else {
    let inputs = crate::stage_solve::StageInputs { stored_basis: basis_slice.get_mut(m, t).as_ref(), /* ... */ };
    let view = crate::stage_solve::run_stage_solve(ws, &inputs)?;
    let unscaled = /* unscale view.primal by col_scale */;   // (A') duplicated
    (view.objective, unscaled)
};
```

The two arms produce the _same_ `(objective, unscaled_primal)` but via different
solve entry points, and the unscaling block (A/A') is duplicated. **Simulation
pass** (`simulation/pipeline.rs:518`) has the identical shape.

**Backward pass — two parallel, drifted functions.** `solve_opening_baked`
(`backward.rs:703`) and `solve_opening_dcs` (`backward.rs:824`) take nearly the
same arguments and both "solve one opening, extract duals, accumulate," but have
**diverged** in three ways that a reader must track:

| concern              | `solve_opening_baked`                              | `solve_opening_dcs`                                          |
| -------------------- | -------------------------------------------------- | ------------------------------------------------------------ |
| solve                | `run_stage_solve` (baked all-cuts LP)              | `lazy_solve_preloaded` (lazy resident set)                   |
| warm start (ω=0)     | `resolve_backward_basis` → stored cross-iter basis | `stored_basis = None` → **cold**                             |
| dual extraction      | `extract_duals_from_view` (state **+ cut** duals)  | `extract_state_duals_only` + `accumulate_dcs_binding_counts` |
| basis capture (ω=0)  | `save_basis_at_omega_zero`                         | **none** ("captured basis would describe the baked layout")  |
| outcome accumulation | `accumulate_opening_outcome`                       | `write_opening_outcome`                                      |

`process_trial_point_backward` (`backward.rs:945`) then dispatches between them and
hoists DCS-only setup behind back-to-back conditionals:

```rust
let dcs_params = training_ctx.dcs.filter(|p| p.is_active(iteration));
if let Some(params) = dcs_params {            // backward.rs:982
    ws.solver.load_model(&ctx.templates[s]);  // DCS loads the cut-free core itself
    build_initial_resident_set(/* ... */);
}
for omega in 0..succ.probabilities.len() {
    if let Some(params) = dcs_params { solve_opening_dcs(/* ... */, omega != 0) }
    else { solve_opening_baked(/* ... */) }
}
```

**DCS-awareness even leaks into the pass driver.** `backward_pass_state.rs:1004`
computes `dcs_active` and then _skips the normal LP load_ in two places because
DCS does its own load:

```rust
let dcs_active = training_ctx.dcs.is_some_and(|p| p.is_active(iteration));
// ...
if !dcs_active { load_backward_lp(ws, succ); }   // backward_pass_state.rs:1017 and again at :1068
```

So the orchestration layer — which "should" only know "run the backward pass" —
now encodes "if DCS is active, don't do my normal setup because the strategy will."
That is the mixed responsibility the hypothesis points at: _which strategy_ and
_how to solve_ are interleaved at every level (driver → trial-point → opening).

**Cost.** ≈18 DCS conditionals in shipped code across `backward.rs` (9),
`backward_pass_state.rs` (5), `forward.rs` (2), `simulation/pipeline.rs` (2), plus
the two drifted `solve_opening_*` functions. The drift (different dual extraction
and outcome accumulation) is the real hazard: a change to "how an opening's result
is recorded" must be made — and kept consistent — in two places.

## 3. Finding 2 — Config-layer semantic overloading (the clearest mixed responsibility)

One struct, `RowSelectionConfig` (`crates/cobre-io/src/config/training.rs:68`),
serves all four methods, and DCS was bolted on by **reusing existing fields via
precedence chains** rather than giving DCS its own fields. Three concrete smells:

**(a) The struct's own doc is now stale.** `method` (training.rs:73) reads:

```rust
/// Method: `"level1"`, `"lml1"`, or `"domination"`.
pub method: Option<String>,
```

— it does not even list `"dynamic"`, the method this whole feature added.

**(b) Deprecated fields resurrected as DCS aliases.** `threshold` (training.rs:84)
and `memory_window` (training.rs:93) are both documented **"Deprecated. Silently
ignored."** Yet the DCS parse consumes them as fallbacks:

```rust
let k1    = config.candidate_window.or(config.memory_window);             // cut_selection.rs:705
let nadic = config.nadic.or(config.threshold).unwrap_or(DEFAULT_NADIC);   // cut_selection.rs:719
```

So a field marked "silently ignored" is, for `method = "dynamic"`, silently
_honored_ — a direct contradiction between the doc and the behavior.

**(c) A cadence field doubles as a window, forcing a method-specific carve-out.**
`check_frequency` is the periodic-pruning cadence for level1/lml1/domination, and
**also** k2 (the active-set seed window) for dynamic, with no first-class name of
its own:

```rust
let check_frequency = config.check_frequency.unwrap_or(5);
// must special-case dynamic, where 0 is meaningful (k2=0, matches NEWAVE):
if check_frequency == 0 && method != "dynamic" {            // cut_selection.rs:657
    return Err("cut_selection.check_frequency must be > 0".to_string());
}
// ...
let k2 = config.check_frequency.unwrap_or(DEFAULT_K2);      // cut_selection.rs:717
```

The `&& method != "dynamic"` carve-out in a shared validation is itself the
symptom: one field means two different things, so the validator has to ask which
method it is. (This footgun was hit in practice — a `check_frequency` left from a
level-1 config silently became k2 under `dynamic`.)

Net: a reader cannot answer "what does `check_frequency: 0` do?" or "is
`memory_window` used?" without first knowing `method` and the precedence order.
The cut-selection config has become a four-method union resolved by undocumented
field overloading.

## 4. Finding 3 — Stateful temporal coupling via opening-reuse

The backward opening-reuse path threads mutable state _across openings within a
trial point_, which is subtler to reason about than a stateless per-opening solve.
The carrier is `DcsSolveScratch.row_map` (`dcs.rs:453`):

```rust
/// ... and so the backward opening-reuse path can carry residency across a trial
/// point's openings: a fresh solve resets it (see DcsSolveContext::continue_carry),
/// a continued solve leaves it intact.
pub row_map: CutRowMap,
```

The behavior is selected by `DcsSolveContext.continue_carry`, threaded from the
callers (`omega != 0` in backward, `false` in forward/simulation). The invariants a
maintainer must hold:

- the cold/fresh solve is "ω == 0" today, but conceptually "first solved" — these
  will diverge the moment opening ordering lands (`backward-opening-ordering.md`);
- the residency set monotonically grows across a trial point's openings and must be
  reset exactly at the trial-point boundary;
- the DCS path deliberately captures **no** ω=0 basis (asymmetry vs the baked
  path's `save_basis_at_omega_zero`), because the residency layout differs from the
  baked layout.

None of this is wrong, but it is _temporal_ coupling encoded in a shared scratch
buffer plus a boolean, with correctness resting on resets happening at the right
boundary — exactly the kind of state that makes the hot path harder to follow.

## 5. Finding 4 — `lazy_solve_preloaded` and `DcsSolveScratch` carry many responsibilities

`lazy_solve_preloaded` (`dcs.rs:664`, ~155 lines) is a single function that does:
(1) two entry modes — `continue_carry` warm vs fresh `reset → append seed →
uniform-BASIC reconstruct`; (2) the initial solve; (3) the bounded lazy
add-and-resolve inner loop; (4) the TC fallback; (5) copying the final solve into
the result buffers. Each is justified, but their concentration is why the
redundant-solve removal and the Polonius/borrow work this cycle needed such careful
reasoning — the function is doing solve orchestration, basis reconstruction, and
result marshalling at once.

`DcsSolveScratch` (`dcs.rs:437`) is correspondingly a kitchen-sink: add-row
construction (`batch`), scoring scratch (`scoring`, `out_selected`), the
reconstruction `recon_basis`, the cross-opening residency `row_map`, the result
mirror (`res_primal`/`res_dual`/`res_reduced_costs` + scalars), **and** an
instrumentation accumulator (`scoring_time_seconds`). Five distinct concerns —
solve inputs, residency state, result outputs, and measurement — in one struct.

## 6. Root tension

Every finding traces to one decision: **DCS lives as a second strategy that the
passes choose between, instead of behind a uniform "solve this stage's openings"
seam.** `lazy_solve_preloaded` is pass-agnostic (good — all three passes call it),
but the _unification stopped one level too low_: the callers still branch on DCS,
so the branching, the drifted twin functions, the driver-level `dcs_active` skips,
and the config union all follow from there.

## 7. Remediation sketch (deferred — no changes now)

Two refactors would dissolve most of the debt without touching the algorithm or the
contracts (both are enum dispatch, honoring the no-`Box<dyn>` rule):

1. **A uniform opening-solve seam.** Introduce an enum-dispatched
   `StageOpeningSolver` (variants `Baked` / `Lazy`) that owns "solve one opening,
   extract duals, accumulate outcome." The passes call it uniformly; the
   `if dcs … else …` branches, the two `solve_opening_*` functions, and the
   `dcs_active` load-skips collapse to one construction site + one call site per
   pass. The baked-vs-DCS divergences (dual extraction, basis capture, outcome
   accumulation) become methods on the two variants, so they can no longer drift
   silently.

2. **First-class DCS config.** Give DCS its own config struct (or at least
   first-class `active_window` for k2 and stop honoring deprecated
   `threshold`/`memory_window`), so no field means two things and the
   `method != "dynamic"` validation carve-out disappears. Update the `method` doc
   to list `"dynamic"`.

3. **(Optional) decompose `lazy_solve_preloaded`** into `prepare` (entry mode +
   seed + reconstruct), `lazy_loop`, and `finalize` (result marshalling), and split
   the instrumentation accumulator out of `DcsSolveScratch`.

## 8. Severity & sequencing

- **None of these are correctness or determinism bugs** — they are comprehension
  and maintainability costs. The drifted `solve_opening_*` pair (§2) is the highest
  _risk_ (silent divergence on future edits); the config overloading (§3) is the
  highest _user-facing_ cost (it has already caused a real misconfiguration).
- **Sequencing:** do not start while the perf work (opening ordering, CLP
  hot-start) is in flight in these same files — land or shelve those first.
  Refactor (1) is best attempted _after_ opening ordering, since the "first solved
  vs ω==0" change naturally wants the same seam. Refactor (2) is independent and
  low-risk and could go any time.
