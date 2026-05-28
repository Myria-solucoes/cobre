# Basis Reconstruction Simplification and Related Cleanups — Design Analysis

**Status**: Proposal (pending experimental validation)
**Date**: 2026-05-28
**Authors**: Investigation triggered by Epic 03 (in-backward selection) production benchmark

---

## 1. Abstract

Three related findings emerged from analysing the 50-iteration production benchmark of the in-backward selection design (Epic 03), which reported +320 s slower wall time vs the baseline (+3.1%):

1. **Basis reconstruction over-engineering (§3-§7)**: the current `basis_reconstruct.rs` is a 3,648-line module that predicts cut-row activity via a sliding-window bitmap of recent binding observations, maintained by per-stage MPI `allreduce(BitwiseOr)` and balanced by Scheme 1 / Scheme 2 promotion machinery. The design assumes recent cut activity predicts future activity — an assumption **anti-aligned with SDDP's structural intent**, which deliberately varies trial-point states across iterations. The §10 experiment will determine whether a hybrid simplification (keep slot-identity preservation; drop the activity classifier and Scheme 1/2) loses material wall time. Estimated benefit: -2,800 LOC, -25 MB peak memory, -1 MPI allreduce per stage per iter, ~0 to +25 s wall delta on the baseline path.

2. **Basis-store staleness under mid-sweep rebake (§8)**: a structural issue where the in-backward hook's per-stage rebake of `baked_templates[t]` happens between two stored-basis lifecycle events (iter N capture vs iter N+1 apply), making the captured basis suboptimal relative to the apply-time LP. Empirical signature: +4.5% per-pivot cost in backward, +18% in forward under in-back mode. Recommendation: rely on the §6 basis simplification as the primary mitigation — the simplified classifier reads only the stored row status (insensitive to mid-sweep state).

3. **Double rebake / ticket-019 cleanup (§9)**: with `IN_BACKWARD_ENABLED = true` and the post-backward block still active, every `baked_templates[t]` is rebuilt twice per iteration (once by the hook, once by the end-of-iter loop, producing identical bytes). The actual cost is dominated by the no-op cut-selection scan that the post-backward block still runs (~104 s on the production benchmark). Recommendation: bundle the cleanup with the full ticket-019 (remove the post-backward block, narrow end-of-iter rebake to stages 0 and T-1).

The three findings combine into a path that could flip Epic 03's verdict from UNFAVORABLE (+320 s) to roughly NEUTRAL (~+100 s) at 50 iters production scale (Workstream A + Workstream B, §11). A clearly FAVORABLE verdict would require an additional structural rebake-cost mitigation (out of scope for this document). A 4-configuration sweep on `cobre_set_24_sc2` (~30 minutes wall) would empirically resolve Workstream A's recommendation; Workstream B's recommendation is grounded in mechanical reasoning and the existing per-phase telemetry.

---

## 2. Motivation

The investigation into the in-backward cut selection design (the "selection-inside-backward" architectural change) produced a 50-iteration production benchmark with surprising results:

| Mode                                 |  Wall time |                Delta |
| ------------------------------------ | ---------: | -------------------: |
| end-of-backward selection (baseline) | 10,378.7 s |                    — |
| in-backward selection                | 10,698.9 s | **+320.2 s (+3.1%)** |

The expected mechanism (intra-iteration cut pruning produces leaner LPs, saving simplex pivots) **did** activate — in-backward used 102M fewer total simplex iterations than baseline, and the forward phase ran 6,549 s faster in aggregate. But these wins were dominated by a backward-phase regression that traced partly to **basis disruption**: the per-pivot cost in in-backward mode was 4.5% higher in backward and 18% higher in forward than baseline. The hypothesis is that mid-sweep rebakes of `baked_templates[s]` invalidate the warm-start basis stored at `(scenario m, stage s)` from the prior iteration, so the simplex pays extra refactorization / phase-1 work per pivot.

This raised a deeper question: is the basis-reconstruction machinery itself over-engineered for the SDDP setting? Specifically:

- The classifier predicts which cuts are likely tight at the new trial point using a sliding-window bitmap of recent binding observations.
- SDDP forward sampling deliberately produces _different_ trial points each iteration — that variation is essential to policy coverage.
- If trial points scatter across iterations, then "this cut was binding in iter N − 1" is a weak predictor of "this cut is binding in iter N at a different state".

The classifier is, structurally, fighting against the iteration model. Examining its empirical value is overdue.

---

## 3. Current implementation overview

The basis warm-start path runs once per `(scenario, stage)` at opening 0 of each multi-opening backward solve. For every later opening (omega ≥ 1), HiGHS's internal warm-start carries the basis forward without our involvement. The empirical ratio at the production benchmark is **5% of LP solves use `reconstruct_basis`** (604,800 of 12,096,000 backward solves).

`reconstruct_basis` runs in five phases:

1. **Column statuses** — copy verbatim from the stored basis (resize/pad with BASIC if target column count differs).
2. **Template (non-cut) row statuses** — copy verbatim.
3. **Cut row statuses, preserved slots** — for each cut row in the target LP whose pool slot ID appears in the stored basis, copy the stored row status (BASIC or LOWER).
4. **Cut row statuses, new slots** — for cuts not in the stored basis, classify via `CutMetadata.active_window`. If any of the low `basis_activity_window` (default 5) bits is set, the cut was binding in one of the recent iterations and is classified LOWER (tight guess). Otherwise BASIC (safe slack default).
5. **Basic-count invariant repair** — HiGHS rejects bases that violate `col_basic + row_basic == num_row`. The classifier in step 4 may over-predict LOWER; Scheme 1 promotes preserved-LOWER candidates to BASIC and Scheme 2 reverts the most-recent new-LOWER classifications back to BASIC until the invariant holds. A separate `enforce_basic_count_invariant` post-pass handles the dropped-BASIC case on the forward path.

The classifier feeds on infrastructure that lives across modules:

- `CutMetadata.active_window: u32` — a 32-bit sliding bitmap per cut, populated by per-stage MPI `allreduce(BitwiseOr)` so any rank's binding observation flips the global bit. Shifted left by 1 at iteration boundary. Bit 31 (`SEED_BIT`) is a transient mark used for within-iteration retention.
- `CapturedBasis.cut_row_slots: Vec<u32>` — slot ID per stored cut row, used for the slot-identity match in step 3.
- `PromotionScratch` — scratch buffers for the Scheme 1 partial sort and the Scheme 2 tail-override list.
- Per-stage `sync_stage_metadata` performs **two** allreduces: a `Sum` over `active_count` (binding counters, used elsewhere for budget enforcement and staleness) and a `BitwiseOr` over the iteration-level activity bit. The second allreduce exists solely for the classifier.

Total LOC: 3,648 in `basis_reconstruct.rs` plus ~200 LOC of supporting plumbing (workspace metadata fields, MPI wire format updates, scratch initialisation).

---

## 4. Empirical evidence

### 4.1 Coverage

At the production benchmark (50 iterations, convertido scale, 192 forwards × 2 MPI ranks × 96 threads per rank):

- Total backward LP solves: **12,096,000**
- Basis warm-starts offered: **604,800** (= 50 iters × 192 scenarios × 63 stages × 1 opening-0 per scenario × 2 ranks ÷ rank distribution)
- Ratio: **5.0%**

The remaining 95% of solves rely on HiGHS's internal warm-start, which carries the previous opening's basis automatically without going through our reconstruction code. The classifier's wall-time leverage is therefore bounded to the 5% of solves that initiate a new scenario at opening 0.

### 4.2 Acceptance rate

- `basis_offered`: 604,800
- `basis_consistency_failures`: **0**
- `basis_reconstructions` (HiGHS internal): **0**

The current implementation produces 100% valid bases (HiGHS's `isBasisConsistent` check never fails). The Scheme 1 / Scheme 2 invariant repair works. There is no correctness defect to fix.

### 4.3 Per-pivot cost

Aggregate solve time divided by simplex pivot count yields a proxy for per-pivot solver cost:

| Phase    | baseline ms/pivot | in-back ms/pivot |  Delta |
| -------- | ----------------: | ---------------: | -----: |
| Backward |            0.3853 |           0.4026 |  +4.5% |
| Forward  |            0.3037 |           0.3585 | +18.0% |

The pivot count itself drops under in-back mode (3,296M backward vs 3,398M baseline; 285M forward vs 358M baseline) — the §8.3 LP-shrinking mechanism IS delivering smaller LPs and fewer pivots. But each pivot is more expensive, meaning the basis warm-start quality is lower. The forward phase is hit much harder because forward at iter N+1 receives a basis from iter N's forward capture, but the templates loaded at iter N+1 have been mid-sweep-rebaked by iter N's backward hook — the basis is consistent but far from optimal.

### 4.4 Pivot accounting

A typical backward LP at the production benchmark takes ~281 pivots (baseline) or ~273 pivots (in-back). The first opening per scenario is the only opening where our basis is offered; opening 0 of a typical scenario likely takes more pivots than openings 1-19 (because openings 1-19 have a near-optimal HiGHS-internal warm-start). Per-opening pivot counts are already exported to `training/solver/iterations.parquet` (one row per `(iteration, phase, stage, opening, rank, worker_id)` tuple with `simplex_iterations` populated by the per-omega `SolverStatsDelta`). Filtering by `phase == "backward"` and grouping by `opening == 0` vs `opening > 0` yields the breakdown directly; the §10 experiment runs consume this artifact.

The classifier's wall-time value depends on how many pivots it saves at opening 0 relative to a cold start. Plausible bands:

- If classifier saves ~30% of opening-0 pivots: ~120M pivots saved × 0.385 ms = ~46 ks aggregate ≈ **+240 s wall-equivalent** at 192-way parallelism.
- If classifier saves ~10% of opening-0 pivots: ~40M pivots saved ≈ **+80 s wall-equivalent**.
- If classifier saves ~5%: ~20M pivots ≈ **+40 s wall-equivalent**.

The total run is 10,378 s; the classifier is plausibly worth somewhere between 0.4% and 2.5% of wall time. The §10 experiment reads the per-opening breakdown from `training/solver/iterations.parquet` and tightens the band against the 4-config sweep.

---

## 5. Structural analysis: why the classifier under-delivers in SDDP

The classifier's signal — `active_window` — is set by **any rank's** observation of binding in **any of the last K iterations**. Three properties of SDDP make this signal weaker than it looks.

### 5.1 Trial-point variation by design

SDDP's forward pass evaluates the policy at trial points drawn from a stochastic process. Successive iterations produce **different** trial-point sets — this variation is precisely what gives the policy generalisation. After a few iterations the policy stabilises and the trial-point distribution tightens, but at any iteration the active set "which cuts bind at this exact x*hat" depends on the specific x_hat being solved. The bitmap aggregates across trial points, so "cut k was binding in iter N − 1" really means "cut k was binding at some trial point in iter N − 1". Whether that signal predicts binding at iter N's \_different* trial point is a state-space proximity question, not an iteration-recency one.

### 5.2 Bitmap saturation

In production with 192 trial points per iteration, the bitmap saturates quickly. Bit 0 is set when **any** of the 192 trial points × 64 stages × 2 ranks ever observes the cut binding. After 5 iterations (the default window), most cuts in the active pool have at least one bit set. The classifier therefore predicts LOWER (tight) for most cuts.

But there is an algebraic ceiling on how many LOWER cuts the basis can hold (the `col_basic + row_basic == num_row` invariant). Scheme 1/2 enforces this ceiling by demoting "excess" LOWER predictions back to BASIC. The number of cuts actually classified LOWER is therefore **not** controlled by the classifier — it is controlled by the invariant math. The classifier just **picks which cuts** to mark, not how many.

In effect, the classifier-plus-Scheme-1/2 chooses a small (~L = number of openings ≈ 20) subset of cuts to mark LOWER, sorted by a popcount-and-staleness criterion. Whether this subset matches the true binding set at the new x_hat depends on the prediction accuracy of the popcount sort, which is essentially "cuts that have been binding the most recently and most frequently are more likely tight now".

### 5.3 Simplex correction cost is small relative to the solve

Each misclassified LOWER cut costs roughly 2 simplex pivots to correct (one pivot flips the wrongly-LOWER cut to BASIC, one flips the actually-binding cut from BASIC to LOWER). For ~20 candidate LOWERs per opening-0 solve, even a 50%-wrong classifier costs ~20 extra pivots, against a backdrop of ~280 pivots/solve. The classifier is competing against a baseline that is already cheap.

### 5.4 Anti-alignment with the in-backward design

The in-backward design (Epic 03) rebakes `baked_templates[s]` mid-backward-sweep. At iter N+1's stage-s solve at opening 0, the basis stored at `(m, s)` was captured in iter N against an OLDER `baked_templates[s]` (the one in effect after iter N's hook ran at stage s, before iter N's end-of-iter rebake). The slot-identity match in step 3 of the classifier still works (slot IDs are stable). But the `active_window` bitmap is updated continuously across iter N's backward sweep — bits get OR'd in as later-numbered stages observe binding. So by the time iter N+1's stage-s solve runs the classifier, the bitmap reflects observations made _across multiple stages of the prior iteration's backward sweep_. The signal is even noisier than the iter-boundary picture suggests.

A simpler classifier (or none at all) is **insensitive to this disruption**.

---

## 6. Proposed simplification: hybrid `reconstruct_basis`

The proposal keeps the most predictive component (slot-identity preservation of cuts already in the stored basis) and removes the noisier components (activity-window classifier for new cuts; Scheme 1/2 invariant repair).

### 6.1 What stays

- **Phase a, column statuses**: copy from stored basis verbatim, pad with BASIC if target column count differs. (Unchanged.)
- **Phase b, template row statuses**: copy from stored verbatim, fill missing with BASIC if shorter. (Unchanged.)
- **Slot identity preservation**: for each cut row in the target LP whose slot ID appears in `CapturedBasis.cut_row_slots`, copy the stored row status verbatim. This is the highest-fidelity prediction we have — those cuts were binding (or non-binding) at the actual stored x_hat. If the new x_hat is even somewhat similar, the cut's status is likely the same.
- **`enforce_basic_count_invariant`**: kept as a safety net on the forward path. Cut selection can drop a cut whose stored status was BASIC, leaving `col_basic + row_basic > num_row`. The post-pass demotes trailing BASIC cut rows to LOWER until the invariant holds. The post-pass is also short (~30 LOC) and addresses a different failure mode than the classifier.

### 6.2 What goes away

- **Activity-window classifier**: `classify_cut_rows` and all logic referencing `active_window`. New cuts (slot not in stored basis) are unconditionally classified BASIC.
- **`CutMetadata.active_window: u32`**: field removed. Per-cut memory drops by 4 bytes; aggregate savings at production scale ≈ 25 MB peak.
- **`SEED_BIT` and within-iter retention** : the transient seed bit is no longer needed because the classifier doesn't exist.
- **Per-stage `allreduce(BitwiseOr)` on activity bits**: removed from `sync_stage_metadata`. The `Sum` allreduce on `active_count` (used for budget enforcement, not the classifier) stays.
- **Scheme 1 (preserved-LOWER promotion)**: removed. With "new cuts always BASIC", the invariant balances by construction (see §6.4).
- **Scheme 2 (new-LOWER tail override)**: removed for the same reason.
- **`PromotionScratch`**: struct deleted; the `candidates` and `new_lower_indices` vectors are no longer allocated.
- **`basis_activity_window` config knob and validation**: removed from `StudyParams`, `cobre-io` schema, and CLI.

### 6.3 Code sketch

The new `reconstruct_basis` is approximately:

```rust
pub fn reconstruct_basis<'a, I>(
    stored: &CapturedBasis,
    target: ReconstructionTarget,
    current_cut_rows: I,
    out: &mut Basis,
    slot_lookup: &mut Vec<Option<u32>>,
) -> ReconstructionStats
where
    I: Iterator<Item = (usize, f64, &'a [f64])>,
{
    // Phase a: column statuses (unchanged from current)
    reconstruct_col_statuses(stored, target, out);

    // Phase b: template row statuses (unchanged)
    reconstruct_template_row_statuses(stored, target, out);

    // Phase c: slot lookup
    build_slot_lookup(&stored.cut_row_slots, slot_lookup);

    // Phase d: cut row statuses — preserved slots copy stored; new slots BASIC.
    let mut stats = ReconstructionStats::default();
    for (target_slot, _, _) in current_cut_rows {
        let status = match slot_lookup.get(target_slot).and_then(|o| *o) {
            Some(pos) => {
                stats.preserved += 1;
                stored.basis.row_status[stored.base_row_count + pos as usize]
            }
            None => {
                stats.new_slack += 1;
                HIGHS_BASIS_STATUS_BASIC
            }
        };
        out.row_status.push(status);
    }
    stats
}
```

Approximately **80 LOC** for the core path, plus the small `reconstruct_col_statuses` / `reconstruct_template_row_statuses` / `build_slot_lookup` helpers already in the file. `enforce_basic_count_invariant` is kept verbatim as a separate function.

### 6.4 Why the invariant balances naturally

The HiGHS invariant `col_basic + row_basic == num_row` is preserved by the stored basis. When the target LP differs from the stored LP, three churn events can perturb it:

1. **Preserved cut row, BASIC→BASIC**: no change to `row_basic`. Invariant preserved.
2. **Preserved cut row, LOWER→LOWER**: same.
3. **Dropped cut row that was BASIC**: `row_basic` decreases by 1; the cut row is gone, `num_row` also decreases by 1. Invariant preserved.
4. **Dropped cut row that was LOWER**: neither side changes. Invariant preserved.
5. **New cut row, classified BASIC** (proposal): `row_basic` increases by 1; `num_row` also increases by 1. Invariant preserved.
6. **New cut row, classified LOWER** (current implementation): `row_basic` unchanged; `num_row` increased by 1. **Invariant broken** — `col_basic + row_basic < num_row` by 1. Scheme 1/2 exists to demote a preserved BASIC somewhere else by 1 to compensate.

The current implementation breaks the invariant on every new-LOWER classification and repairs it via Scheme 1/2. The proposal never breaks it, so the repair machinery becomes unnecessary.

The forward path can still produce `col_basic + row_basic > num_row` via event 3 in a specific shape: a preserved cut row that was BASIC gets dropped, but the column whose basis status was implicitly counted in `col_basic` doesn't get touched. This is the legitimate use case for `enforce_basic_count_invariant`, which we retain.

### 6.5 What this does NOT change

- Slot-identity correctness under cut-set churn (the original motivation for `reconstruct_basis`).
- Bit-determinism: there is no rank-dependent state in the simplified path.
- The forward path's `enforce_basic_count_invariant` safety net.
- The `CutMetadata.active_count` and `last_active_iter` fields, which are used by budget enforcement and staleness ordering elsewhere.
- The `Sum` allreduce on `active_count` (budget) — only the `BitwiseOr` allreduce goes away.

---

## 7. Estimated cost-benefit

### 7.1 Wall-time savings (positive)

| Component                                                                       | Estimated saving |
| ------------------------------------------------------------------------------- | ---------------: |
| `reconstruct_basis` runtime simplification (smaller hot path)                   |       ~1-3 s/run |
| Per-stage MPI `allreduce(BitwiseOr)` removal                                    |       ~3-5 s/run |
| `CapturedBasis::to_broadcast_payload` size reduction (fewer slots to broadcast) |       ~1-2 s/run |
| Less cache pollution in scratch buffers (PromotionScratch removed)              | ~unknown, modest |

Total: **~5-10 s/run** direct wall-time savings on the production benchmark.

### 7.2 Wall-time cost (negative)

The cost comes from extra simplex pivots at opening 0 when a new cut would have been correctly classified LOWER under the current implementation but is classified BASIC under the proposal.

Bound: out of ~20 cuts the current classifier marks LOWER, perhaps half are correct (≈ 10 cuts). Under the proposal, those 10 misses each cost ~2 pivots to correct. So ~20 extra pivots per opening-0 solve.

20 pivots × 604,800 opening-0 solves × 0.385 ms/pivot ≈ **46 ks aggregate solve time** ≈ **+240 s wall-equivalent** at 192-way parallelism (worst case if classifier is currently strong).

If the classifier is currently weak (the structural argument in §5 suggests this), the cost is much smaller — possibly ~10-50 s wall-equivalent.

### 7.3 Maintenance and footprint savings (independent of wall)

- **LOC**: -2,800 (basis_reconstruct.rs from 3,648 to ~800; supporting plumbing in workspace.rs, cut_selection.rs, sync_stage_metadata simplified)
- **Memory**: ~25 MB peak at production scale (active_window field × peak populated cuts × num stages)
- **Test surface**: ~30 test cases removed (Scheme 1, Scheme 2, classifier edge cases)
- **Config surface**: `basis_activity_window` knob removed from `StudyParams`, `cobre-io` schema, CLI flag, docs
- **MPI wire format**: `CapturedBasis::to_broadcast_payload` shrinks by `cut_row_slots.len() × 4` bytes per basis

### 7.4 Robustness gains

The simplified path is **insensitive** to mid-sweep rebakes of `baked_templates[s]`. The activity bitmap is the main source of staleness in the current implementation (it accumulates observations across the sweep). Removing it removes a category of subtle prediction-vs-reality drift.

This may **flip the in-backward design's verdict at production scale** from UNFAVORABLE (+320 s) to FAVORABLE. Specifically:

- The +18% per-pivot cost in forward (under in-back mode) is the basis-disruption signature. If the simplified classifier reduces this to baseline-equivalent per-pivot cost, the forward phase alone gains ~5,400 s of aggregate solve time relative to current in-back (≈ +28 s wall-equivalent).
- The +4.5% per-pivot cost in backward similarly traces to disruption. Recovery here is worth ~15,000 s of aggregate solve time ≈ +78 s wall-equivalent.
- Combined potential recovery: **~+100 s wall-equivalent**, which would shrink the +320 s deficit to ~+220 s. Still not favorable at 50 iters, but the trajectory is right.

When combined with the post-backward block removal (the §9 ticket-019 cleanup, ~100 s additional saving), the in-backward design's net could land at **+20 s slower**, essentially break-even.

---

## 8. Basis-store staleness under mid-sweep rebake

This section is independent of the basis-simplification proposal: the staleness mechanism exists regardless of how `reconstruct_basis` predicts cut activity. But the §6 simplification is the most natural mitigation, so the two are tightly related.

### 8.1 The mechanism

The basis store maps `(scenario m, stage s)` pairs to a `CapturedBasis`. The lifecycle:

1. **Capture** — at the end of a successful LP solve, `ws.solver.get_basis(&mut captured.basis)` reads the simplex's final basis (column statuses + row statuses) and `write_capture_metadata` records the cut row slots and the state vector at capture. The capture happens on every solve (forward and backward), overwriting the prior capture for that `(m, s)` pair.
2. **Apply** — at the start of the next solve at the same `(m, s)` pair (typically next iteration, opening 0), `resolve_backward_basis(basis_slice, m, s)` returns the stored basis, and `reconstruct_basis` rebuilds a target-LP basis from it before `solver.solve(Some(&basis))` installs it.

The basis is "consistent" with the target LP if `col_basic + row_basic == num_row`. It is "optimal-ish" if it is also close to the new LP's optimal basis. Consistency is necessary; optimality-ish is what determines pivot count.

Between capture and apply, two events can degrade the optimal-ish-ness:

- **A**: The trial point `x_hat` at the apply site differs from the trial point at the capture site. Cuts that were tight at capture may be slack at apply, and vice versa. This is the **state-variation source** of staleness.
- **B**: The LP itself has changed shape — rows (cuts) have been added, removed, or had their data mutated. This is the **template-mutation source** of staleness.

In baseline mode (end-of-backward selection), source B operates only at iteration boundaries. The end-of-iter rebake updates every `baked_templates[t]` from the active cuts; the next iteration's solves load the new templates. Between iter N's capture and iter N+1's apply, the template changed once. Slot identity tracks which cuts survived; preserved cuts keep their captured row statuses; new cuts get classified.

In in-backward mode, source B operates **mid-backward-sweep**. Iter N's backward sweep proceeds high-to-low (`t = T-2 → 0`). At stage t, the hook fires and rebakes `baked_templates[t]`. The next stage processed (t-1) loads the freshly-rebaked `baked_templates[t]` for its LP. So **within a single iteration's backward sweep**, the templates that downstream stages will load are being mutated by upstream stages.

For the basis store specifically:

- Iter N capture at `(m, t)` happens at end of iter N's backward solve for that pair. At capture time, `baked_templates[t+1]` (the FCF that was loaded for this solve) was the **post-hook** version of stage t+1's template (because the hook at t+1 ran _before_ the backward solve at t).
- Iter N+1 apply at `(m, t)` loads `baked_templates[t+1]` again. By this point, iter N's end-of-iter rebake (still present because the post-backward block runs) has rebuilt `baked_templates[t+1]` from the active cuts at end of iter N — which is the SAME state as what the hook produced (the post-backward block found 0 deactivations).

So in steady state, the capture's template and the apply's template are **the same**: both reflect "iter N final pool[t+1] after all deactivations". The staleness from source B is bounded to the within-iter trajectory: the iter N capture sees a template containing some subset of iter N's cuts (only those added by stages later than t+1 are visible — there are no later stages because t+1 is the successor and stages process high-to-low ... wait, this is the place where careful tracing matters).

Let me re-trace the timeline precisely. Iter N's backward sweep at stage t (loading templates at t+1):

1. Stage T-2's hook runs first (since `for t in (0..num_stages-1).rev()` starts at T-2). The hook rebakes `baked_templates[T-2]`.
2. Stage T-3's backward solves run, loading `baked_templates[T-2]` (the just-rebaked one).
3. Stage T-3's hook rebakes `baked_templates[T-3]`.
4. Stage T-4's backward solves run, loading `baked_templates[T-3]`.
5. ... etc.

So at iter N's stage T-3 solve at `(m, T-3)`:

- The template loaded is `baked_templates[T-2]` — the post-iter-N-stage-T-2-hook version.
- The basis captured at this solve reflects the column/row statuses of this LP.

At iter N+1's stage T-3 solve at `(m, T-3)`:

- The template loaded is again `baked_templates[T-2]`, but by now iter N's end-of-iter rebake has rebuilt it from the active cuts at end of iter N. The active cuts at end of iter N include everything from iter N's stages T-2 and T-3 hook-deactivations. The captured iter-N template at stage T-3 included only stage T-2's deactivations (because the hook at T-3 hadn't run yet at the time of stage T-3's capture).

So the iter-N-captured template and the iter-N+1-applied template **differ by the deactivations the hook produced at stages t+1 and beyond** (where t+1 is the successor of the captured `(m, ?)` pair).

For most stages, the difference is small (a few cuts). But for high-numbered stages (those processed first), the iter-N capture saw a template that was the "lightly pruned" version; the iter-N+1 apply sees the "heavily pruned" version. The basis stored at those captures has row statuses for cuts that have since been deactivated. The slot-identity match drops those rows; the basic-count invariant repair compensates. Numerically valid, but suboptimal.

In baseline mode, there is no analogous mismatch: the iter-N capture sees the same template that the iter-N+1 apply will see (modulo iter-boundary changes). The basis is captured against the post-iter-N-final-rebake state.

### 8.2 Empirical signature

The per-pivot solve cost is the cleanest signature of staleness. From Appendix A.4:

| Phase    | baseline ms/pivot | in-back ms/pivot |  Delta |
| -------- | ----------------: | ---------------: | -----: |
| Backward |            0.3853 |           0.4026 |  +4.5% |
| Forward  |            0.3037 |           0.3585 | +18.0% |

The forward phase is hit much harder (+18% vs +4.5%) because forward solves load templates that have been **fully** rebaked by iter N's backward sweep. Every `baked_templates[t]` was touched by the hook + the end-of-iter rebake in iter N. The basis stored at iter N's forward capture against the **pre-iter-N-backward** template state is now applied against the **post-iter-N-backward** template. That's a larger delta than the within-backward-sweep variation that backward solves experience.

The retry count corroborates: backward `lp_retries` went from 4,313 (baseline) to 8,950 (in-back) — a doubling. Retries trigger when an LP solve fails on the first attempt (typically numerical issues or infeasibility detection during phase-1) and the wrapper retries with different settings. More retries = more solves that started from a basis far enough from optimal that the simplex went sideways.

### 8.3 Mitigations

Three mitigation strategies, in order of preference:

**M1: §6 basis simplification (primary mitigation).** The activity-window classifier consumes the cumulative bitmap that has been OR'd across iter N's entire backward sweep by the time iter N+1's apply runs. Removing the classifier removes a source of cross-iteration coupling — the only state the simplified classifier consumes is the stored row status of preserved cuts, which is **insensitive to mid-sweep rebakes** (slot identity is stable; the stored status reflects the capture-time LP shape, not the apply-time LP shape). This is the cleanest fix and is independently justified by §5.

**M2: Capture the basis later.** Move the basis capture from end-of-solve to end-of-iteration (or end-of-stage-after-hook). The capture would then reflect the final iter-N pool state, matching what iter N+1 will load. Tradeoffs:

- The basis captured at end-of-iter is the basis from the **last solve at this (m, t) pair**, but the post-hook template differs from that solve's template (because the hook ran after the solve and changed the template). Re-resolving the LP against the post-hook template at end-of-iter to capture a fresh basis would be a full LP solve per `(m, t)` per iter — far too expensive.
- A cheaper version: at end-of-iter, **patch the captured basis** to reflect the cuts that have been deactivated since capture. Set the deactivated cuts' row statuses to BASIC (their LP rows now have sentinel ±INF bounds) and demote enough other BASIC entries to maintain the invariant. This is conceptually similar to `enforce_basic_count_invariant` but applied to the **captured** basis at end of iter, not the **reconstructed** basis at apply time. Effort: medium; correctness risk: moderate.

**M3: Disable basis warm-start at the disrupted stages.** Cold-start the LP at opening 0 of `(m, t)` when iter N's hook modified `baked_templates[t+1]`. A cold start costs maybe 200-500 extra pivots per disrupted solve. Crude but easy to implement. Probably worse than M1 in practice.

### 8.4 Recommendation for staleness

**M1 (basis simplification) is the recommended path.** It removes the staleness source most affected by mid-sweep rebake (the activity bitmap), preserves the high-fidelity component (slot-identity match), and is independently justified by §5. The estimated wall-time recovery on the production benchmark is ~+100 s (recovering the +18% forward per-pivot cost and most of the +4.5% backward per-pivot cost). M2 (capture-time patching) is a second-line option if M1 is insufficient. M3 (cold-start the disrupted stages) is a fallback if both M1 and M2 fail.

---

## 9. Double rebake (Epic 03 ticket-019 cleanup)

### 9.1 The mechanism

With `IN_BACKWARD_ENABLED = true` and the post-backward block still active (ticket-019 not yet done), every `baked_templates[t]` is rebuilt twice per iteration:

1. **Per-stage hook rebake** (`backward_pass_state.rs:968-979`) — after the hook's selection at stage t produces the deactivation updates and `apply_updates` settles `pool[t]`, the hook calls `build_cut_row_batch_into(cut_batches[t], fcf, t, ...)` + `cobre_solver::bake_rows_into_template(templates[t], cut_batches[t], &mut baked[t])`. The result is the full-active-cut version of `baked_templates[t]` after this iteration's per-stage selection.
2. **End-of-iter rebake in `run_cut_management`** (`training_session/mod.rs:1020-1042`) — after the backward sweep completes, the post-backward selection block iterates `1..num_sel_stages` and calls `select_for_stage_with_scratch` on every stage's pool. With the in-back hook having already pruned, the post-backward selection finds zero new deactivations on every stage. The block then proceeds to its rebake loop: for every stage t, `build_cut_row_batch_into(scratch.bake_row_batches[t], fcf, t, ...)` + `bake_rows_into_template(..., &mut scratch.baked_templates[t])`. The result is bitwise-identical to the hook's per-stage rebake (because `pool[t]` did not mutate between the hook's rebake and now).

The two rebakes write to the **same** `baked_templates[t]` (it lives on `IterationScratch`; the hook borrows it as `&mut [StageTemplate]` via `BackwardPassInputs.baked`; the end-of-iter loop borrows it via `&mut self.scratch.baked_templates`). The second write overwrites the first with identical bytes.

### 9.2 Cost quantification

The bake cost is dominated by:

- `build_cut_row_batch_into`: per stage, scans the active cuts and copies their coefficients into the row batch. Cost: O(`active_count` × `n_state`) bytes copied.
- `bake_rows_into_template`: writes the row batch into the baked template's CSC/CSR storage, fixing up `row_starts` and `values`. Cost: O(`active_count` × `n_state`).

At production scale (iter 50, ~520 active cuts/stage average, 64 stages, 155 hydros + few more state vars ≈ 156):

- Per-stage rebake: 520 × 156 ≈ 81 K cell-writes ≈ a few hundred microseconds with HiGHS's CSC packing.
- Per-iter total: 63 stages × few hundred μs ≈ ~30 ms per iter per rank.
- Across 50 iters × 2 ranks: ~3 s aggregate.

Wall-equivalent: ~1.5 s on the production benchmark. Tiny.

But the `cut_batch_build_ms` and the `cut_selection_ms` columns capture two larger costs that are tied to the same code path:

| Cost component                                                                          | baseline | in-back | Delta (s) |
| --------------------------------------------------------------------------------------- | -------: | ------: | --------: |
| `cut_batch_build_ms` (build_delta_cut_row_batch_into for the backward-sweep delta path) |    6,163 |   3,110 |      -3.1 |
| `cut_selection_ms` (post-backward block scan + rebake)                                  |  231,240 | 103,921 |    -127.3 |

In-back saves 127.3 s on `cut_selection_ms` because the post-backward block's selection scan finds 0 deactivations and exits early on the selection compute (the gemm-based dominance check still runs but processes a globally-pruned pool that yields no candidates).

**The actual cost being paid in in-back mode is the no-op scan, not the rebake.** The rebake is a small fraction of the 103.9 s in `cut_selection_ms` (most of it is the gemm-based dominance scan over the full active pool).

### 9.3 What ticket-019 removes

The full ticket-019 (per the Epic 03 plan, paused at user direction) removes the post-backward selection block in `run_cut_management` entirely. Specifically:

- Lines ~800-1000 of `training_session/mod.rs` (the selection-and-apply loop over `1..num_sel_stages`): removed.
- Lines 1020-1042 (the end-of-iter rebake loop): removed for the stages that the hook handled. The hook already rebaked them.
- The `PolicySelectionComplete` event emission: rerouted to fire from `BackwardResult.selection_records` populated by the hook.

The remaining post-iter work in `run_cut_management` reduces to: budget enforcement, metric aggregation, event emission rerouting, and (possibly) a rebake of stage 0 if the hook excludes stage 0 via `STAGE_0_EXCLUSION_GUARD = 1`.

### 9.4 Cost recovery from ticket-019

Three components recovered:

1. **The no-op `cut_selection_ms` scan**: ~104 s aggregate at production scale. This is the bulk of the savings.
2. **The duplicate rebake**: ~3-5 s. Modest.
3. **The `cut_batch_build_ms` for the end-of-iter path**: ~3 s. The hook builds these batches per-stage already; the end-of-iter rebuild is redundant.

Total ticket-019 recovery: **~+110 s** on the production deficit of +320 s.

### 9.5 Subtleties

Two stages need careful handling:

**Stage T-1 (terminal)**. The backward sweep never visits stage T-1 (the loop is `0..num_stages-1`). The hook never runs at stage T-1. Today's end-of-iter rebake loop rebakes `baked_templates[T-1]` — but does any solve actually load `baked_templates[T-1]`? Stage T-2's forward / backward solves load `baked_templates[T-1]` as the FCF of the terminal stage. So yes, the rebake at T-1 is needed. Ticket-019 must preserve it (the hook can't handle it).

**Stage 0 (initial)**. The hook excludes stage 0 via `STAGE_0_EXCLUSION_GUARD = 1` (which mirrors the post-backward block's `1..num_sel_stages` exclusion). So no per-stage hook rebake at stage 0. Today's end-of-iter rebake handles `baked_templates[0]` — needed by stage 1's solves which load FCF[1] = stage 1's value function approximation... wait, FCF[1] is rebaked at stage 1's hook, not at stage 0's. Stage 0's `baked_templates[0]` is loaded by... nothing? Stage 0 is the initial stage; its solve loads `baked_templates[1]` (FCF of stage 1). So `baked_templates[0]` is never loaded for backward solves. It IS loaded for the lower-bound computation at stage 0 (forward at stage 0 with the initial state).

So the post-ticket-019 rebake loop reduces to:

```rust
// Rebake stages that the hook did not handle.
// Stage 0: needed by lower-bound computation (forward at stage 0).
// Stage T-1: needed by stage T-2's solves (terminal FCF).
// Stages 1..T-2: rebaked by the hook.
build_cut_row_batch_into(&mut scratch.bake_row_batches[0], fcf, 0, ...);
bake_rows_into_template(&ctx.templates[0], &scratch.bake_row_batches[0], &mut scratch.baked_templates[0]);

build_cut_row_batch_into(&mut scratch.bake_row_batches[T-1], fcf, T-1, ...);
bake_rows_into_template(&ctx.templates[T-1], &scratch.bake_row_batches[T-1], &mut scratch.baked_templates[T-1]);
```

Two rebakes per iter, not 64. Most of the work is eliminated.

### 9.6 Recommendation for the double rebake

**Bundle the double-rebake elimination with the ticket-019 cleanup**, not as a separate change. The double rebake by itself is a small win (~5 s/run); the post-backward block removal is the much larger win (~104 s/run). Splitting them would mean two PRs against the same code area with overlapping concerns. One clean PR that:

1. Removes the post-backward block in `run_cut_management`.
2. Reduces the end-of-iter rebake loop to the two boundary stages (0 and T-1).
3. Reroutes `PolicySelectionComplete` to fire from `BackwardResult.selection_records`.
4. Removes the redundant `cut_batches[t]` end-of-iter builds (the hook owns these now for stages 1..T-2).

Effort: ~1 day of hpc-rust-developer work plus careful test updates. The smoke test on `cobre_set_24_sc2` (run `cobre run` with `--enable-inside-backward` and confirm LB matches reference within 1e-3 relative tolerance) is the integration check.

### 9.7 Order of operations

The recommended order is:

1. **First**: §6 basis simplification (independent of in-backward, smaller blast radius, validated by the §10 experiment).
2. **Second**: §9 ticket-019 cleanup (depends on the in-backward design being kept; would be moot if in-backward is reverted).
3. **Re-run the 50-iter production benchmark** after both land to record the §14.5 verdict.

If the basis simplification alone closes the gap to <+100 s on the production benchmark, the ticket-019 work has a clearer FAVORABLE projection (because ~+110 s of recovery would land it at -10 s, a small net win).

If the basis simplification leaves the gap at +200 s or more, the ticket-019 work might bring it to roughly break-even but the §8.3 mitigations (incremental rebake, deferred rebake) would be needed for a clear FAVORABLE.

---

## 9. Risks and unknowns

### 9.1 Empirical bounds on classifier value

The strongest unknown is the actual per-opening-0 pivot count saved by the classifier vs a cold-start opening 0. Per-opening pivot counts are exported to `training/solver/iterations.parquet`, so direct measurement is possible from a single run. The §7.2 estimate of ~20 extra pivots/solve is a heuristic upper bound derived from the openings count and the structure of the LP. The actual cost could be smaller (if the classifier is mostly picking wrong cuts anyway, as the §5 structural argument suggests) or larger (if there are second-order effects we haven't modelled).

### 9.2 Forward-path interactions

The forward path is more affected by basis quality than backward — forward solves are larger LPs and run with looser tolerances. The cobre forward path captures basis at the end of each stage solve for use by the next iteration. If we drop the activity-window classifier, the forward solves at iter N+1 see a stored basis from iter N's forward capture, but with the simplified classifier predicting "new cuts BASIC". For cuts added between iter N and iter N+1 (via the backward of iter N), these would all start BASIC under the proposal. Currently the activity classifier might mark them LOWER if they bound during the prior backward.

This is a real degradation on the forward path. We don't have a clean way to quantify it without the experiment.

### 9.3 Cold-start fallback for new scenarios

The basis warm-start path requires a stored basis. The first iteration (or first time a `(m, s)` pair is solved) cold-starts. The simplification doesn't affect cold start. The first solve of each `(m, s)` pair pays the same cost in both implementations.

### 9.4 Bit-determinism preservation

The simplified path has no rank-dependent state. Slot lookups are deterministic. The classifier was the only source of MPI-dependent state (`allreduce(BitwiseOr)`); removing it removes one MPI dependency entirely. Bit-determinism is **strictly easier** to preserve under the simplification.

### 9.5 Configuration churn

The `basis_activity_window` config knob is documented and shipped. Removing it is a breaking change to user configs (existing configs would fail validation if the field is still present). A clean removal path: deprecate the field for one release (log a warning, ignore the value), then remove. Acceptable churn.

---

## 10. Experimental plan

A 4-configuration benchmark sweep on `cobre_set_24_sc2` (the documented local benchmark case, ~2 min 30 s baseline) at 50 iterations resolves the recommendation. The case is small but exercises all relevant paths: backward sweep, in-backward hook, cut selection, basis reconstruction, multi-rank MPI.

| Run | Selection mode          | Basis path                  |                                    Expected baseline |
| --- | ----------------------- | --------------------------- | ---------------------------------------------------: |
| A   | end-backward (baseline) | current `reconstruct_basis` | ~150 s (4-iter scale × 50 iters / 4 ≈ extrapolation) |
| B   | in-backward             | current `reconstruct_basis` |                                               ~177 s |
| C   | end-backward            | hybrid simplified           |                                                  TBD |
| D   | in-backward             | hybrid simplified           |                                                  TBD |

(At 50 iters on the local case, total wall is much smaller than convertido. Adjust expectations: the local case is 4 forwards × 5 iters configured; would need to bump config to 50 iters and accept ~30 min wall for the sweep, OR run on a 10-iter intermediate which would be ~5 min × 4 = 20 min.)

### 10.1 Decision criteria

- **Criterion 1**: If `(C)` wall is within 2% of `(A)`, the simplification is benign on the baseline path. **Simplify.**
- **Criterion 2**: If `(D)` wall is materially better than `(B)` (e.g. by ≥ 5%), the simplification mitigates the in-back basis disruption.
- **Criterion 3**: If `(D)` wall is _not_ worse than `(C)` (i.e. in-back catches up to baseline under the simpler classifier), the in-backward design becomes viable.

Decision matrix:

| Criterion 1 result     | Criterion 2 result | Action                                                                     |
| ---------------------- | ------------------ | -------------------------------------------------------------------------- |
| Pass (within 2%)       | Pass (≥ 5% better) | **Simplify and ship in-backward**                                          |
| Pass                   | Fail               | Simplify; in-backward stays parked                                         |
| Fail (worse > 5%)      | Either             | Keep current implementation; do a targeted profile of where the cost lives |
| Fail (within 5%, > 2%) | Pass               | Borderline — needs human judgement on the LOC vs wall tradeoff             |

### 10.2 Implementation effort

The hybrid simplified `reconstruct_basis` is a ~half-day implementation behind a `BASIS_HYBRID` cargo feature flag:

1. Duplicate `reconstruct_basis` into a `reconstruct_basis_hybrid` function (or use a feature-gated module).
2. The simplified version is the code in §6.3 with the existing helpers.
3. `enforce_basic_count_invariant` retained verbatim.
4. Wire `stage_solve.rs` to pick the implementation via the feature.

The MPI allreduce and `active_window` updates remain enabled under the feature (we don't tear out the data plane until the experiment validates the path). Once the experiment passes, a follow-up cleanup removes the unused infrastructure.

### 10.3 Instrumentation available and optional additions

Per-opening pivot counts are already exported to `training/solver/iterations.parquet` (one row per `(iteration, phase, stage, opening, rank, worker_id)` with `simplex_iterations`, `solve_time_ms`, `basis_offered`, `basis_consistency_failures`, `basis_reconstructions` columns). The experiment reads this artifact directly — no new instrumentation is required to settle §4.4.

Optional diagnostics that could be added under the same feature flag if the verdict is ambiguous:

- Reconstruction-stats delta accumulator (preserved vs new for the hybrid path).
- Classifier-accuracy probe: optionally, run _both_ classifiers on the same input and log the symmetric difference of their LOWER predictions. (Diagnostic only; not for performance comparison.)

### 10.4 Stretch experiment

If the 4-config sweep results are ambiguous, run a 6-config extension at convertido scale (the production benchmark):

| Run  | Mode         | Basis path                  | Wall (est.) |
| ---- | ------------ | --------------------------- | ----------: |
| A50  | end-backward | current                     |   ~10,400 s |
| B50  | in-backward  | current                     |   ~10,700 s |
| C50  | end-backward | hybrid                      |         TBD |
| D50  | in-backward  | hybrid                      |         TBD |
| C50' | end-backward | hybrid + ticket-019 cleanup |         TBD |
| D50' | in-backward  | hybrid + ticket-019 cleanup |         TBD |

This is a ~12-hour sweep on the user's 192-core hardware, run unattended. The user owns this execution per the documented production-benchmark policy.

---

## 11. Recommendation

Three distinct workstreams are recommended. They have independent merit but reinforce each other; the right sequencing is below.

### 11.1 Workstream A: basis simplification (§6)

**Recommendation: run the §10 experiment, then simplify if it passes.**

- The structural argument (§5) and empirical signature (§4.3) point toward the classifier being more expensive than it is worth in the SDDP setting.
- The cost-benefit estimate (§7) is wide enough that the right call hinges on actual measurement.
- The 4-config experiment costs ~half a day to implement (feature flag only — per-opening data already exported) and ~30 minutes to run on `cobre_set_24_sc2`.
- Decision criteria are in §10.1. If Criterion 1 passes (simplification within 2% of current on baseline), proceed. If Criterion 2 also passes (in-back recovery ≥ 5%), the Epic 03 verdict math shifts materially.
- Workstream-A blast radius: `basis_reconstruct.rs`, `workspace.rs` (CapturedBasis fields), `cut_selection.rs` (CutMetadata), `sync_stage_metadata` (drop BitwiseOr allreduce), config schema. Independent of in-backward; ships in baseline mode safely.

### 11.2 Workstream B: ticket-019 cleanup (§9)

**Recommendation: bundle the double-rebake fix with the full post-backward block removal.**

- The double rebake by itself is ~5 s/run — too small to justify a standalone change.
- The full ticket-019 cleanup (remove post-backward block, narrow end-of-iter rebake to stages 0 and T-1, reroute `PolicySelectionComplete`) recovers ~+110 s on the production benchmark.
- The cleanup is **conditional on keeping the in-backward design**. If Epic 03 is reverted, ticket-019 is moot.
- Effort: ~1 day of work + careful test updates. The smoke test on `cobre_set_24_sc2` (LB matches baseline reference within 1e-3 relative tolerance after running with `--enable-inside-backward`) is the integration check.
- Workstream-B blast radius: `training_session/mod.rs` (run_cut_management), event emission paths. Tightly coupled to Epic 03.

### 11.3 Workstream C: basis-store staleness (§8)

**Recommendation: rely on Workstream A as the primary mitigation; defer M2/M3 unless A is insufficient.**

- The staleness mechanism (§8.1) exists structurally under the current `reconstruct_basis` design and is exacerbated by in-back mode's mid-sweep template rebake.
- The §6 simplification eliminates the activity-window component of staleness because the simplified classifier reads only the stored row status (insensitive to mid-sweep state) — that is the bulk of the disruption signature.
- M2 (capture-time basis patching) is a second-line option; M3 (cold-start disrupted stages) is a fallback.
- Decision point: after Workstream A is committed and re-benchmarked, if the in-back per-pivot cost inflation persists at > +2% (vs the current +4.5% backward / +18% forward), then escalate to M2 or M3. If it drops to ≤ +2%, the staleness is sufficiently addressed by Workstream A.
- Workstream-C effort: zero in the recommended path (A covers it). If M2 is needed, ~2 days. If M3, ~half a day.

### 11.4 Combined Epic 03 path

To make Epic 03 (in-backward selection) net favorable at production scale, run Workstreams A and B together:

| Workstream                                             | Estimated recovery |
| ------------------------------------------------------ | -----------------: |
| A: Basis simplification (per-pivot cost)               |            ~+100 s |
| B: Ticket-019 cleanup (no-op scan + rebake reductions) |            ~+110 s |
| Optional: skip rebake when 0 deactivations             |             ~+10 s |
| **Cumulative recovery**                                |        **~+220 s** |
| Remaining deficit from +320 s baseline                 |            ~+100 s |

A neutral verdict at 50 iters is achievable with A+B; a clearly FAVORABLE verdict (e.g. -50 s net win) requires one of the structural rebake-cost mitigations (incremental rebake, deferred rebake, or skip-at-small-pool). Those are not in this document's scope; they would be follow-on design work if needed.

### 11.5 Sequencing

The recommended order is:

1. **Implement Workstream A behind a feature flag.** (~half a day; per-opening pivot data is already in `training/solver/iterations.parquet`.)
2. **Run the §10 experiment** on `cobre_set_24_sc2`. (~30 min wall + decision time.)
3. **If Workstream A passes**: strip the legacy classifier and supporting infrastructure. Commit as a single PR. (~1 day cleanup + test updates.)
4. **Implement Workstream B** (ticket-019 cleanup). The Epic 03 plan already has this ticket refined; the planner can dispatch it. (~1 day.)
5. **Re-run the 50-iter production benchmark** at convertido scale. User-owned; ~3 hours wall time.
6. **Record the §14.5 verdict** in the cut-selection-parallelism design doc. If FAVORABLE, dispatch Epic 03's remaining tickets (016, 018, 020, 021). If still UNFAVORABLE, decide whether to invest in structural rebake mitigations or close Epic 03.
7. **Workstream C is a backstop**: only invoked if Workstream A leaves residual in-back staleness > +2%.

### 11.6 If Workstream A's experiment fails

If Criterion 1 fails (simplification materially worse on baseline), the fallback paths preserved by this design:

- Investigate WHERE the wall-time cost lives. Use per-opening pivot data to identify whether the cost is concentrated at opening 0 or distributed.
- Consider a **narrower simplification**: keep the activity classifier but eliminate the MPI `allreduce(BitwiseOr)` (sample binding info from a single rank instead of all). This keeps the predictive value but removes the most expensive infrastructure (~3-5 s/run savings, modest LOC reduction).
- Consider a **classifier replacement**: instead of the sliding-window bitmap, use a simpler heuristic (e.g. "always BASIC for cuts added in the last iteration; copy stored status for older cuts"). The bitmap may be over-engineered even if cut activity prediction is useful.

The fallback paths preserve the option to revisit without committing to the maximal simplification. They also preserve the workstream-B and workstream-C analyses (those are independent of A's outcome).

---

## 12. Summary table

| Item                                           | Current                               | Proposed                                |
| ---------------------------------------------- | ------------------------------------- | --------------------------------------- |
| `basis_reconstruct.rs` LOC                     | 3,648                                 | ~800                                    |
| `CutMetadata` size per cut                     | 32 B (4 fields)                       | 28 B (active_window dropped)            |
| Peak memory at production                      | ~25 MB extra                          | —                                       |
| Per-stage MPI allreduces                       | 2 (Sum + BitwiseOr)                   | 1 (Sum only)                            |
| Phase a (column statuses)                      | unchanged                             | unchanged                               |
| Phase b (template row statuses)                | unchanged                             | unchanged                               |
| Phase c (slot lookup)                          | unchanged                             | unchanged                               |
| Phase d (cut row classification)               | activity-window classifier            | preserved → copy stored; new → BASIC    |
| Phase e (Scheme 1/2 invariant repair)          | required                              | removed (math balances by construction) |
| `enforce_basic_count_invariant` (forward path) | kept                                  | kept                                    |
| Bit-determinism                                | preserved                             | preserved                               |
| Tests                                          | ~50 cases                             | ~20 cases                               |
| Config knob `basis_activity_window`            | exposed                               | removed                                 |
| Robustness to mid-sweep rebake                 | susceptible (active_window staleness) | insensitive                             |
| Wall-time impact (estimated)                   | —                                     | -5 to +25 s/run (within noise)          |
| Wall-time impact under in-backward (estimated) | —                                     | -50 to -100 s/run (recovery)            |

---

## Appendix A: Empirical data from the 50-iter production benchmark

### A.1 Wall-time

| Metric             | end-backward | in-backward |                                   Delta |
| ------------------ | -----------: | ----------: | --------------------------------------: |
| Total wall time    |   10,378.7 s |  10,698.9 s |                                +320.2 s |
| Final cuts_active  |       33,356 |      33,444 |                                     +88 |
| Peak cuts_active   |       45,549 |      33,444 | -12,105 (in-back pools never overshoot) |
| Final LB (iter 50) |   4.7734e+12 |  4.7687e+12 |                             -1.0e-3 rel |
| Final gap_percent  |    -1083.98% |   -1093.01% |                                       — |

### A.2 Per-phase aggregate timing (sum across 50 iters)

| Phase                                  | baseline ms | in-back ms | Delta (s) |
| -------------------------------------- | ----------: | ---------: | --------: |
| cut_selection_ms (post-backward block) |     231,240 |    103,921 |    -127.3 |
| cut_sync_ms                            |     136,641 |    155,123 |     +18.5 |
| lower_bound_ms                         |     415,723 |    429,402 |     +13.7 |
| state_exchange_ms                      |       6,934 |      6,789 |      -0.1 |
| cut_batch_build_ms                     |       6,163 |      3,110 |      -3.1 |
| bwd_load_imbalance_ms                  |   1,531,407 |  1,551,798 |     +20.4 |
| bwd_scheduling_overhead_ms             |     264,847 |    318,668 |     +53.8 |
| fwd_load_imbalance_ms                  |      61,745 |     70,422 |      +8.7 |
| fwd_scheduling_overhead_ms             |      17,081 |     13,923 |      -3.2 |
| overhead_ms                            |      49,949 |     49,936 |      -0.0 |

### A.3 Per-worker aggregate (sum across 96 workers × 2 ranks × 50 iters)

| Phase            | baseline ms |  in-back ms | Delta (s) |
| ---------------- | ----------: | ----------: | --------: |
| forward_wall_ms  |  64,227,082 |  60,662,115 |  -3,565.0 |
| backward_wall_ms | 674,706,070 | 685,022,624 | +10,316.6 |
| bwd_setup_ms     |  12,982,556 |  11,170,793 |  -1,811.8 |

### A.4 Solver telemetry (sum across 50 iters)

| Metric                              |      baseline |   in-backward |
| ----------------------------------- | ------------: | ------------: |
| backward lp_solves                  |    12,096,000 |    12,096,000 |
| backward basis_offered              |       604,800 |       604,800 |
| backward basis_consistency_failures |             0 |             0 |
| backward basis_reconstructions      |             0 |             0 |
| backward lp_retries                 |         4,313 |         8,950 |
| backward simplex_iterations         | 3,398,838,023 | 3,296,781,004 |
| backward solve_time_total (s)       |     1,309,664 |     1,327,191 |
| backward avg pivots/LP              |        280.99 |        272.55 |
| backward ms/pivot                   |        0.3853 |        0.4026 |
| forward simplex_iterations          |   358,810,767 |   285,690,564 |
| forward solve_time_total (s)        |       108,956 |       102,407 |
| forward avg pivots/LP               |        584.00 |        464.99 |
| forward ms/pivot                    |        0.3037 |        0.3585 |
| backward basis_set_time_total (s)   |          10.7 |           9.8 |
| forward basis_set_time_total (s)    |           7.9 |           8.2 |

### A.5 Per-iteration-bucket critical-path delta

| Iter range         | `bwd_max` delta (s) | `fwd_max` delta (s) | `bwd_agg` delta (s) | Average cuts_active (baseline) |
| ------------------ | ------------------: | ------------------: | ------------------: | -----------------------------: |
| 1–5                |               +80.0 |                +0.1 |              +7,452 |                         19,540 |
| 6–10               |                +2.2 |                -2.0 |                 +39 |                         21,107 |
| 11–20              |               -10.1 |                -8.7 |                -855 |                         25,334 |
| 21–30              |               +27.2 |                -5.1 |              +1,678 |                         27,451 |
| 31–40              |               +36.4 |                -7.8 |              +2,602 |                         29,152 |
| 41–50              |                -2.7 |                -5.6 |                -599 |                         32,266 |
| **Sum (50 iters)** |          **+132.9** |           **-29.1** |         **+10,316** |                              — |

---

## Appendix B: References

- `crates/cobre-sddp/src/basis_reconstruct.rs` — current implementation (3,648 LOC)
- `crates/cobre-sddp/src/workspace.rs` — `CapturedBasis` definition and broadcast wire format
- `crates/cobre-sddp/src/cut_selection.rs` — `CutMetadata` definition
- `crates/cobre-sddp/src/backward_pass_state.rs:516-595` — `sync_stage_metadata` (the per-stage MPI allreduces)
- `crates/cobre-sddp/src/stage_solve.rs:160-275` — `run_stage_solve` (the call site of `reconstruct_basis`)
- `crates/cobre-sddp/src/backward.rs:577-...` — `process_trial_point_backward` (the opening loop)
- `docs/design/cut-selection-parallelism-redesign.md` — Epic 03 (in-backward) design
- `docs/design/cut-selection-parallelism-redesign.md §14.5` — Epic 03 gate decision template
- 50-iter benchmark: `plans/cut-selection-benchmarks/{selection-end-backward,selection-in-backward}/` (gitignored)
- Local benchmark case: `~/git/cobre-bridge/example/cobre_set_24_sc2/`
- Local benchmark reference outputs: `~/git/cobre-bridge/example/cobre_set_24_sc/output/`

---

_This document captures a proposal pending experimental validation. The recommendation is to run the §10 experiment before committing to the simplification. If the experiment results are not as predicted, the §11.4 fallback paths remain open._
