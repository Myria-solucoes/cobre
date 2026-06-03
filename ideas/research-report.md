# LP-Solver Parameter Tuning for SDDP Backward-Pass Dispatch (Cobre): HiGHS and CLP — A Source-Grounded Technical Reference

## TL;DR
- For the SDDP deep-cut-pool warm-started backward pass, the canonical fast path on **both** backends is **dual simplex + warm-started basis + presolve OFF + perturbation OFF + tight-enough tolerances**, because adding Benders cuts adds rows that leave the prior optimal basis primal-feasible but dual-infeasible — exactly the case dual simplex re-optimizes in a handful of pivots, while presolve and cost perturbation are the two knobs that both destroy the warm start and threaten the dual accuracy the cuts depend on.
- The correctness-sensitive knobs (must NOT be traded for speed) are: `dual_simplex_cost_perturbation_multiplier` / CLP `perturbation`, the dual feasibility tolerance, and aggressive `presolve`. The pure speed levers are pricing strategy (Devex vs steepest edge vs Dantzig), price strategy, scaling mode, and factorization frequency.
- The one mapping the user flagged — HiGHS `simplex_price_strategy` integer→name — could **not** be confirmed verbatim from source within this effort; the strongly-expected mapping is `0=Col, 1=Row, 2=RowSwitch, 3=RowSwitchColSwitch (default)`, consistent with the design doc, but this remains an explicit open item requiring a direct read of `highs/simplex/SimplexConst.h`.

## Key Findings

1. **Dual simplex is the workhorse for re-solving after adding cuts.** A Benders cut θ + πᵀE x ≥ g is an added row. Appending a row to a previously optimal LP keeps the old basis primal-feasible-ish but generally dual-infeasible — precisely the starting condition under which dual simplex re-optimizes efficiently from the retained basis. HiGHS defaults to dual simplex: its official options list gives "Strategy for simplex solver 0 ⇒ Choose; 1 ⇒ Dual (serial); 2 ⇒ Dual (SIP); 3 ⇒ Dual (PAMI); 4 ⇒ Primal … Range: {0, 4} Default: 1" (the dual engine is `HEkkDual`). CLP's `model.dual()` invokes `ClpSimplexDual`. The SDDP.jl performance guide ("Improve computational performance", sddp.dev) states directly: "forcing solvers to use the dual simplex algorithm (e.g., Method=1 in Gurobi) is usually a performance win."

2. **Presolve OFF is critical for warm-start** because presolve transforms the problem (substitutions, bound tightening, redundant-row/col removal) so the saved basis no longer corresponds index-for-index to the LP actually handed to simplex; HiGHS would have to presolve → solve reduced → postsolve, discarding the cheap warm start. Verified: HiGHS presolve is `HPresolve` with a `HighsPostsolveStack` (DeepWiki source index over `highs/lp_data/`); the default `presolve` option is `"choose"` (official options docs).

3. **The LU factorization is the expensive shared resource.** HiGHS factorizes the basis matrix B via the `HFactor` class; CLP via `ClpFactorization` (built on CoinUtils `CoinFactorization`/`CoinAbcFactorization`). Between refactorizations both use product-form / Forrest–Tomlin-style updates. A warm-started re-solve that takes few pivots should ideally avoid a fresh INVERT — hence the HiGHS distinction between loading a basis that forces refactorization ("alien") and one that does not.

4. **Cost perturbation is the single most dangerous speed knob for SDDP** because it perturbs the dual values, and the Benders cut gradient IS the dual vector. A perturbed-but-still-valid cut is harmless (still a lower bound under convexity); an invalid cut that slices off part of the true value function breaks the SDDP lower-bound guarantee. Both backends perturb only to escape degeneracy/cycling, so turning it off costs little when warm-starting takes few pivots.

5. **Scaling interacts with the warm start.** Re-deriving scale factors changes the numerical representation the basis weights were computed against; in HiGHS scaling is internal and a saved basis is expressed in user-model terms, so re-scaling perturbs edge weights more than the basis identity. Verified defaults: HiGHS `simplex_scale_strategy` default = 2 (equilibration), from the official options list ("Simplex scaling strategy: off / choose / equilibration (default) / forced equilibration / max value (0/1/2/3/4) … Default: 2"); CLP `scalingFlag_` default = 3 (auto), from `ClpSimplex.cpp`.

## Details

### Part 1 — Conceptual primer

**Revised simplex and the basis.** An LP min cᵀx s.t. L ≤ Ax ≤ U, l ≤ x ≤ u is solved by maintaining a *basis*: a selection of m = num_row basic variables whose columns form a square, invertible matrix B; the remaining variables are nonbasic, fixed at a bound. The revised simplex method never forms B⁻¹A explicitly; it keeps a factorization of B and solves Bx=b (FTRAN) and Bᵀy=c (BTRAN) on demand. HiGHS encodes the basis with `nonbasicFlag_`, `nonbasicMove_`, and `basicIndex_` arrays (verified in `src/simplex/HSimplex.cpp`, class `SimplexBasis`). The **cardinality invariant** is that the number of basic columns plus basic rows (slacks) equals num_row; HiGHS repairs a passed basis that violates this by flipping variables nonbasic or adding slacks (verified, HiGHS "Further features" docs: "There can be more basic variables than the number of rows in the model. HiGHS will identify a set of basic variables of the correct dimension by making some basic variables nonbasic … fewer … by adding basic variables corresponding to slacks").

**Warm starting.** Saving the optimal basis from one solve and restoring it for a related solve means simplex starts at or near a vertex of the new problem instead of from scratch. The savings come from (a) far fewer pivots and (b) avoiding re-discovery of the active set. The cost a warm start cannot avoid by itself is the LU factorization of B for the new problem — unless the solver can reuse the previous factorization.

**LU factorization and its updates / refactorization.** Each simplex iteration swaps one basic for one nonbasic variable, a rank-one change to B. Rather than refactorize, the solver *updates* the factors (product-form, or the more stable Forrest–Tomlin update). Updates accumulate fill-in and numerical error, so after some number of updates the solver *refactorizes* (INVERT) from scratch. HiGHS: `HFactor`; the `simplex_update_limit` option caps updates ("Limit on the number of simplex UPDATE operations · Range {0, 2147483647} · Default: 5000", official options). CLP: `ClpFactorization` with `setFactorizationFrequency`; the CLP User Guide states "The default is to refactor every 200 iterations, but it may make more sense to use something such as 100 + the number of rows divided by 50." CoinUtils supplies the underlying `CoinFactorization`/`CoinAbcFactorization` machinery.

**Why dual simplex after adding rows.** Primal simplex needs a primal-feasible start; dual simplex needs a dual-feasible start. After adding a cut row to an optimal LP, the reduced costs (dual feasibility) are essentially unchanged but the new row may be violated (primal infeasibility) — the natural dual-simplex starting point. Dual simplex drives out the primal infeasibility in a few pivots. This is why dual + warm start + presolve-off is the canonical fast path; primal simplex is preferable mainly when you changed objective coefficients (destroying dual feasibility while keeping primal feasibility).

**Why accurate duals matter in SDDP.** The cost-to-go approximation is built from cuts θ ≥ g + πᵀ(E x − e) where π are the optimal dual multipliers of the stage subproblem. Under convexity any supporting hyperplane from a feasible dual solution is a valid lower bound, so a slightly suboptimal-but-feasible dual gives a valid (if weaker) cut. The failure mode is a dual that is *infeasible* for the true problem (e.g., because the solver perturbed costs or relaxed dual feasibility past tolerance), which can yield a cut that overestimates the value function and cuts off the true optimum — breaking the lower-bound guarantee on which SDDP convergence rests (Pereira & Pinto 1991). de Matos, Philpott & Finardi (2015, *J. Comput. Appl. Math.* 290:196–208) and Diniz et al. (2020, SBPO) both emphasize cut management and basis recovery (warm start) as fundamental to performance on the Brazilian hydrothermal system; SDDP.jl implements the de Matos "Level-One" cut selection (`DematosCutOracle`), and Guigues (2017, *EJOR* 258(1):47–57) provides the Limited-Memory Level-1 variant with a convergence proof.

### Part 2 — HiGHS, parameter by parameter
For each: (a) mechanism + source/enum status, (b) general tuning effect, (c) SDDP verdict.

**2.1 `presolve` ("on"/"off"/"choose", default "choose")**
(a) Controls whether `HPresolve` reduces the model before solving and `HighsPostsolveStack` maps the solution back. VERIFIED from official options docs and DeepWiki source index (`highs/lp_data/HighsOptions.cpp`).
(b) Presolve usually shrinks large LPs dramatically and the reduced solve more than pays for presolve time on a cold start. But presolve renames/removes variables and constraints, so a basis saved for the original model is meaningless for the presolved model; HiGHS must then solve the reduced LP from a crash basis and postsolve.
(c) **SDDP verdict: OFF.** In the backward pass you re-solve the *same* subproblem structure thousands of times with a known good basis. Presolve would discard the warm start every solve. Turning it off preserves the basis correspondence — the documented reason "presolve OFF is critical for warm-start." (Recent HiGHS keeps the basis when rows/cols are deleted, but for a growing cut pool the regime is presolve-off + dual + warm start.)

**2.2 `simplex_dual_edge_weight_strategy` (-1 choose, 0 Dantzig, 1 Devex, 2 SteepestEdge; default -1)**
(a) VERIFIED from the official HiGHS options page: "Strategy for simplex dual edge weights: Choose / Dantzig / Devex / Steepest Edge (-1/0/1/2) · Range: {-1, 2} · Default: -1." (NOTE a discrepancy: the third-party OPTANO C# binding reports `range {-1, 3}`. Treat the official `{-1,2}` as authoritative for current HiGHS; the extra value in a binding may reflect an internal/auxiliary code. Confirm for your pinned version.) Selects how the leaving variable is priced in dual simplex: Dantzig = most-violated basic variable (cheapest/iteration, most iterations); Devex (Harris 1973) = approximate steepest-edge reference weights, updated cheaply; exact dual steepest edge (Goldfarb & Reid 1977; Forrest & Goldfarb 1992) = violation ÷ true edge norm (fewest iterations, most expensive/iteration). The default (-1=choose) selects a steepest/Devex hybrid (SciPy exposes it as "steepest-devex": exact steepest edge until too costly/inexact, then Devex).
(b) On large cold solves, steepest-edge-class pricing wins by cutting iteration count enough to offset per-iteration cost; Dantzig is competitive only on small/easy problems. Edge weights are state that must be reinitialized or re-derived when the basis changes; fresh steepest-edge initialization is itself expensive.
(c) **SDDP verdict: Devex (1) or leave at choose (-1).** A warm-started re-solve after adding one cut takes very few pivots, so the up-front cost of exact steepest-edge weight initialization is poorly amortized; Devex's cheap reference weights are the better trade in the few-pivot regime. HYPOTHESIS — confirm empirically against -1, since HiGHS's adaptive choice may already pick well. None of these threaten dual accuracy.

**2.3 `simplex_scale_strategy` (0 off, 1 choose, 2 equilibration [default], 3 forced equilibration, 4 max value)**
(a) VERIFIED from official docs ("… equilibration (default) … (0/1/2/3/4) · Range {0,4} · Default: 2"). Scaling multiplies rows/cols by factors to compress the dynamic range of |A_ij|, improving conditioning of B and reliability of the LU factorization. Equilibration ≈ scaling so the max abs entry per row/col is near 1. (An older HiGHS C# binding reported default 1 / range {0,5}; current source default is 2.)
(b) Good scaling reduces numerical trouble and refactorization frequency on badly-scaled matrices; on already-well-scaled problems it is near-neutral. Scaling is computed from the matrix, so if A is unchanged between solves the factors are unchanged.
(c) **SDDP verdict: keep equilibration (2), but be aware of the warm-start interaction.** Because adding a cut adds a row with new coefficients, equilibration factors *can* shift slightly solve-to-solve, perturbing the scaled representation against which edge weights and tolerances are measured — a second-order effect on the warm start, not on basis identity. Hydro-dispatch LPs (reservoir balances, generation limits) are usually moderately scaled, so HiGHS scaling rarely hurts; turning it OFF is worth A/B testing only if you observe scale factors drifting per solve. Scaling does not by itself invalidate the basis (stored in user-model terms).

**2.4 `simplex_price_strategy` (advanced; values 0–3) — ENUM UNCONFIRMED**
(a) This is an *advanced* HiGHS option (not on the public options page) defined in `highs/simplex/SimplexConst.h` as `enum SimplexPriceStrategy`. **NOT VERIFIED VERBATIM** (the source file could not be fetched; path moved from `src/` to `highs/` in current layout). The strongly-expected mapping, consistent with the design doc and HiGHS `k…` naming, is: `0 = kSimplexPriceStrategyCol` (column-wise PRICE), `1 = kSimplexPriceStrategyRow` (row-wise PRICE), `2 = kSimplexPriceStrategyRowSwitch` (row-wise, switch to column when not hyper-sparse), `3 = kSimplexPriceStrategyRowSwitchColSwitch` (row+col switching; the default). "PRICE" forms the pivotal row / updated reduced costs (αᵀ = e_pᵀ B⁻¹A). Row-wise price exploits hyper-sparsity of B⁻¹ (Huangfu & Hall 2018, *Math. Prog. Comp.* 10(1):119–142); column-wise is better when the result is dense; the "switch" variants choose dynamically by measured density. Partial pricing and the PAMI suboptimization scheme are related machinery from the same paper.
(b) The switching strategies (2,3) are generally best because they adapt to hyper-sparsity — exactly what makes HiGHS dual simplex fast on large sparse LPs.
(c) **SDDP verdict: leave at default (expected 3).** Hydro-dispatch backward-pass LPs are large and sparse with hyper-sparse B⁻¹, so the row-wise-with-switch default is well-matched. Pure speed knob, no dual-accuracy risk. **Action item: confirm the integer→name mapping by reading `highs/simplex/SimplexConst.h` directly before relying on any specific integer.**

**2.5 `dual_simplex_cost_perturbation_multiplier` (0.0 = off; default ≈ 1.0, INFERRED)**
(a) Advanced option in `HighsOptions`. A multiplier on the magnitude of random cost perturbations the dual simplex adds to break degeneracy/cycling; 0.0 disables. (Default ≈1.0 is INFERRED, not source-verified here.)
(b) Perturbation helps on highly degenerate LPs where simplex stalls; the cost is that returned duals are duals of the *perturbed* problem until a clean-up pass removes the perturbation.
(c) **SDDP verdict: 0.0 (OFF) — correctness-sensitive.** Hydro-dispatch LPs are frequently degenerate (many binding capacity/balance constraints), so HiGHS may perturb by default. Perturbed costs ⇒ perturbed reduced costs ⇒ perturbed duals ⇒ a cut gradient built from the wrong π. Even with clean-up, the safe choice for valid cuts is to disable perturbation and rely on the warm start (few pivots ⇒ little cycling risk). If you see stalls/cycling, prefer a different pricing rule or a slightly looser feasibility tolerance over re-enabling perturbation.

**2.6 `primal_feasibility_tolerance` and `dual_feasibility_tolerance` (default 1e-7; range [1e-10, ∞])**
(a) VERIFIED: HiGHS official docs give range [1e-10, inf]; the CRAN `highs` package manual (8 May 2026) confirms defaults `primal_feasibility_tolerance = 1e-7` and `dual_feasibility_tolerance = 1e-7`. Primal tolerance: how far a basic variable may exceed its bound and still count feasible. Dual tolerance: how negative a reduced cost may be and still count optimal (dual-feasible).
(b) Tighter (e.g., 1e-9) ⇒ more accurate primal/dual solution, generally more iterations and more sensitivity to degeneracy; looser ⇒ faster but the duals (and thus cut gradients) are correct only to that tolerance.
(c) **SDDP verdict: tighten the dual tolerance toward 1e-9, accept the iteration cost — correctness-sensitive.** The cut gradient is the dual vector, so dual feasibility tolerance directly bounds the cut's error. A dual tolerance of 1e-7 lets reduced costs be wrong at the 1e-7 level (usually cuts valid-but-slightly-loose); tightening to 1e-9 buys cut accuracy at modest extra pivots in a warm-started solve. Primal tolerance matters less for cut *validity* but affects the intercept g; 1e-7 is typically fine, 1e-9 if the intercept matters. Floor is 1e-10. With perturbation off, very tight tolerances on a degenerate LP can increase pivot counts — trade carefully.

**2.7 Iteration limit (`simplex_iteration_limit`, default 2147483647).** Per-solve cap. A warm-started re-solve should converge in tens of pivots; a per-attempt cap is a useful watchdog to detect a pathological solve (e.g., a numerically broken warm start) and trigger a fall-back (refactorize / cold start / loosen tolerance) rather than a guarantee-relevant knob.

### Part 3 — CLP, parameter by parameter

**3.1 `perturbation` (50 / 100 / 101 / 102; default 100)**
(a) VERIFIED VERBATIM from `ClpSimplex.hpp` (line 714, Doxygen) and `ClpSimplex.cpp` constructor (`perturbation_(100)`): "Perturbation: 50 - switch on perturbation; 100 - auto perturb if takes too long (1.0e-6 largest nonzero); 101 - we are perturbed; 102 - don't try perturbing again; default is 100; others are for playing." So **102 = off** (CLP will not attempt perturbation). The AbcSimplex variant additionally documents "-50 to +50 — perturb by this power of ten."
(b) 100 (default) lets CLP self-perturb if it detects slow progress/degeneracy; 50 forces it on; 102 forbids it.
(c) **SDDP verdict: 102 (OFF) — correctness-sensitive, same reasoning as HiGHS 2.5.** Perturbation perturbs duals → risks the cut gradient. Call `setPerturbation(102)`. Warm-started few-pivot re-solves rarely need perturbation.

**3.2 Dual pricing mode / `dualPivot` (0–3; 1 = full Dual Steepest Edge)**
(a) CLP's dual row pricing is selected by installing a `ClpDualRowPivot` instance: `ClpDualRowDantzig` (most-violated) or `ClpDualRowSteepest` (DSE). VERIFIED from the CLP User Guide: "[Dantzig] is easily dominated by the Steepest instance which should normally be used." The `ClpDualRowSteepest` constructor takes a mode: VERIFIED "`ClpDualRowSteepest steep(1); // 0 uninitialized, 1 compute weights`." The numeric 0–3 `dualPivot` in the Cobre design doc is the wrapper's enumeration (typically auto/dantzig/partial/steepest); the exact integer→instance mapping is wrapper-specific and should be confirmed in Cobre's CLP binding. CLP also offers partial-pricing variants of steepest for long-thin problems.
(b) DSE minimizes iteration count and is the recommended default for hard LPs; Dantzig is cheaper per iteration but takes many more.
(c) **SDDP verdict: DSE for cold/first solves; consider uninitialized-weights or partial for warm few-pivot re-solves.** As with HiGHS Devex (2.2), the cost of computing full steepest-edge weights is poorly amortized over a handful of pivots; CLP's "uninitialized weights" (initial weight 1.0, correct for an all-slack basis) or a Dantzig step can be competitive when the warm start is close. Pure speed knob, no dual-accuracy risk. Confirm empirically.

**3.3 `factorization_frequency` (default 200; e.g., 100, 200, 400)**
(a) VERIFIED from the CLP User Guide and `ClpSimplex.hpp`: `setFactorizationFrequency(int)`; "The default is to refactor every 200 iterations, but it may make more sense to use something such as 100 + number of rows / 50." Controls how many product-form/Forrest–Tomlin updates accumulate before a fresh INVERT. Underpinned by CoinUtils `CoinFactorization`.
(b) Refactorizing too often wastes time on a fresh O(nnz) factorization; too rarely lets update fill-in and numerical error grow, slowing FTRAN/BTRAN and risking accuracy. The optimum balances refactorization cost against per-pivot update cost.
(c) **SDDP verdict: largely moot when warm-started re-solves take few pivots.** If each re-solve does, say, 5–50 pivots before optimal, you essentially never reach a frequency threshold of 100–400, so the dominant factorization cost is the *single* INVERT at the start of each re-solve (the basis load), not periodic refactorization. The lever that matters is avoiding that INVERT (see 3.6 and Part 3.5), not the frequency. Set frequency to a moderate value (default 200) and tune elsewhere. HYPOTHESIS: optimal frequency has negligible effect in this regime; confirm by measuring pivots-per-resolve.

**3.4 `scaling` (-1 leave as is, 0 off, 1 equilibrium, 2 geometric, 3 auto [default], 4 dynamic)**
(a) VERIFIED VERBATIM from `ClpSimplex.hpp`: "May scale depending on mode -1 leave mode as is, 0 -off, 1 equilibrium, 2 geometric, 3, auto, 4 dynamic(later)"; default `scalingFlag_(3)`.
(b) Geometric/auto improve conditioning on badly-scaled matrices; off is fine for well-scaled ones. Auto (3) lets CLP pick.
(c) **SDDP verdict: leave at auto (3), or geometric (2); test OFF.** Same logic as HiGHS scaling: hydro LPs are moderately scaled; scaling rarely hurts and stabilizes the factorization. Re-deriving factors as cuts are added is a minor warm-start perturbation, not a basis invalidation. Pure speed/stability knob.

**3.5 Primal / dual feasibility tolerance.** CLP exposes `setPrimalTolerance` / `setDualTolerance`. Same correctness logic as HiGHS 2.6: tighten the dual tolerance for cut accuracy, accept extra pivots. For SDDP set the dual tolerance tight (e.g., 1e-9) and verify cut validity. Correctness-sensitive.

**3.6 `algorithm` (dual vs primal).** VERIFIED: CLP `algorithm_` data member, "+ for primal variations and − for dual variations," default 0; `model.dual()` invokes `ClpSimplexDual`, `model.primal()` invokes `ClpSimplexPrimal`. `dual()`/`primal()` accept `startFinishOptions` bits: "1 - do not delete work areas and factorization at end; 2 - use old factorization if same number of rows; 4 - skip as much initialization of work areas as possible." These are exactly the warm-start/factorization-reuse levers for a tight re-solve loop.
(c) **SDDP verdict: DUAL.** Adding cut rows ⇒ dual-feasible/primal-infeasible start ⇒ dual simplex re-optimizes in few pivots. Use `startFinishOptions` bits 1 (retain work areas/factorization) and 4 (skip re-initialization) to minimize per-solve overhead. Note bit 2 ("use old factorization if same number of rows") will NOT apply across a cut addition because adding a row changes the row count — another reason the per-resolve INVERT is the cost to manage.

**3.7 Iteration limit.** `setMaximumIterations(int)`. Same watchdog role as HiGHS 2.7.

### Part 3.5 — The HiGHS basis-cardinality invariant and alien vs non-alien loading
HiGHS requires the loaded basis to satisfy col_basic + row_basic == num_row; it repairs violations by flipping variables nonbasic or adding slacks (VERIFIED, HiGHS "Further features" docs). When you `setBasis` with a basis HiGHS regards as consistent with the incumbent factorization, it can avoid a full INVERT; when the basis is "alien" (not matching, e.g., after structural change), HiGHS computes a fresh LU factorization via `HFactor`. The design-doc's `set_basis_non_alien` corresponds to loading a basis flagged compatible so the factorization is reused — saving one INVERT per re-solve, which (per 3.3/3.6) is the dominant factorization cost in the backward pass. Confirm the exact API/flag name against the HiGHS version Cobre links; the principle (skip refactorization when the basis is a small perturbation) is verified by the role of `HFactor` and the update/refactorize design.

## Recommendations

**Stage 1 — Set the correctness floor (do first, non-negotiable):**
- HiGHS: `presolve=off`; `dual_simplex_cost_perturbation_multiplier=0.0`; `dual_feasibility_tolerance=1e-9` (primal 1e-7 or 1e-9); `simplex_strategy=1` (dual serial).
- CLP: `setPerturbation(102)`; `algorithm=dual`; dual tolerance ≈1e-9; use `startFinishOptions` bits 1+4.
- Validate every change with a cut-validity check: re-solve a stage subproblem at a known state and confirm the cut does not cut off a previously feasible value-function point.

**Stage 2 — Speed levers (tune empirically; measure pivots-per-resolve and wall-time):**
- HiGHS: try `simplex_dual_edge_weight_strategy=1` (Devex) vs `-1` (choose). Leave `simplex_price_strategy` at default (expected 3) — but FIRST confirm its enum. Leave scaling at 2; A/B test 0.
- CLP: try `ClpDualRowSteepest` with uninitialized weights vs full DSE vs Dantzig for the warm re-solve; leave scaling at 3, test 0/2; set factorization frequency 200 (expect little sensitivity).

**Stage 3 — Reduce the per-resolve INVERT and cut-pool size:**
- Ensure the basis is loaded "non-alien" so HiGHS/CLP reuse the factorization where possible.
- Implement cut selection (de Matos Level-1 / Guigues Limited-Memory Level-1) to bound cut-pool growth; fewer active rows ⇒ smaller B ⇒ faster FTRAN/BTRAN and INVERT. This is the highest-leverage structural change beyond solver knobs.

**Thresholds that change the recommendation:**
- If pivots-per-resolve is consistently high (hundreds), the warm start is failing — investigate scaling drift, tolerance-induced cycling, or an alien basis load; only then consider re-enabling perturbation as a last resort (and re-validate cuts).
- If you observe invalid cuts (lower bound exceeding a known feasible cost), immediately tighten the dual tolerance and confirm perturbation is off.
- If cold/first-stage solves dominate wall-time, switch pricing back to steepest-edge for those while keeping Devex/uninitialized for warm re-solves.

### Recommended values

**HiGHS (SDDP backward pass)**

| Option | Value | Type | Rationale |
|---|---|---|---|
| presolve | off | correctness/speed | Preserves warm-start basis correspondence |
| simplex_strategy | 1 (dual serial) | speed | Dual simplex re-optimizes after added rows |
| dual_simplex_cost_perturbation_multiplier | 0.0 | correctness | Perturbation corrupts duals → cut gradient |
| dual_feasibility_tolerance | 1e-9 | correctness | Bounds cut-gradient error |
| primal_feasibility_tolerance | 1e-7 (or 1e-9) | correctness | Cut intercept accuracy |
| simplex_dual_edge_weight_strategy | 1 (Devex) or -1 (choose) | speed | Few-pivot resolves under-amortize steepest-edge init |
| simplex_scale_strategy | 2 (equilibration) | speed/stability | Hydro LPs moderately scaled; test 0 |
| simplex_price_strategy | default (expected 3) — CONFIRM ENUM | speed | Hyper-sparse row-wise+switch suits large sparse LPs |
| simplex_iteration_limit | moderate watchdog | safety | Detect broken warm start |

**CLP (SDDP backward pass)**

| Option | Value | Type | Rationale |
|---|---|---|---|
| algorithm | dual (`model.dual()`) | speed | Dual-feasible start after added rows |
| perturbation | 102 (off) | correctness | Protects dual values |
| dualPivot | DSE cold; uninitialized/partial warm | speed | Amortization of weight init over few pivots |
| factorization frequency | 200 (default) | speed | Near-irrelevant when resolves take few pivots |
| scaling | 3 (auto) | speed/stability | Test 0/2 |
| dual tolerance | ~1e-9 | correctness | Cut accuracy |
| startFinishOptions | bits 1+4 | speed | Retain work areas/factorization, skip re-init |
| maximum iterations | moderate watchdog | safety | Detect broken warm start |

## Caveats / Open Questions (confirm empirically or from source)
1. **`simplex_price_strategy` enum is NOT source-verified.** Read `highs/simplex/SimplexConst.h` (path moved from `src/` to `highs/`) to confirm `0=Col, 1=Row, 2=RowSwitch, 3=RowSwitchColSwitch` and the default. The design doc's "3 = RowSwitchColSwitch" is consistent with expectation but unconfirmed.
2. **`dual_simplex_cost_perturbation_multiplier` default** (≈1.0) is inferred, not source-verified; confirm in `highs/lp_data/HighsOptions.cpp`.
3. **HiGHS `simplex_dual_edge_weight_strategy` range discrepancy.** Official HiGHS docs say range `{-1,2}`, default `-1`; the third-party OPTANO C# binding reports `{-1,3}`. Treat `{-1,2}` as authoritative; confirm for your pinned version.
4. **HiGHS `simplex_scale_strategy` default**: current source/docs say 2; an older C# binding reported 1/range{0,5}. Confirm for the pinned HiGHS version.
5. **Does equilibration scaling perturb the warm start enough to matter?** Whether re-derived scale factors per cut addition measurably increase pivot counts is empirical; A/B test scaling 2 vs 0 (HiGHS) and 3 vs 0 (CLP).
6. **Optimal factorization frequency given few-pivot resolves** — HYPOTHESIS "negligible effect"; verify by measuring pivots-per-resolve.
7. **CLP `dualPivot` 0–3 numeric mapping** is wrapper-specific; confirm in Cobre's CLP binding which integer selects full DSE vs partial.
8. **`set_basis_non_alien` exact API/flag** should be confirmed against the linked HiGHS version; the factorization-reuse principle is verified, the specific call is from the design doc.
9. **Devex vs steepest-edge vs choose** for warm re-solves is a HYPOTHESIS favoring Devex; only measurement on Cobre's instances settles it.