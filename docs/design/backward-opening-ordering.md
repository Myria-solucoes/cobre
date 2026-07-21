# Warm-start-friendly opening ordering in the backward pass

> **Terminology**: this note predates a naming pass. In current code the
> "TSP"/σ-key orders are the intrinsic shortest-chain order with a σ-key
> fallback below 3 openings (`noise_key::apply_chain_order`); the
> `backward_opening_order` config field was removed.

> **Status**: Implemented (default-on, descending). The backward pass always
> solves openings in descending noise-key order; configuration options were
> removed after the A/B measurement confirmed the gain (§8.6–8.7). Wired at
> `setup/mod.rs:317` (precompute), `opening_tree.rs:152-207`
> (`set_solve_order`), and `backward.rs:1146-1163` (hot-path iteration).
> The original proposal text (§1–§7) and validation record (§8) are
> preserved below as design rationale and measurement history.
>
> **Companions:**
>
> - `docs/design/backward-pass-performance-analysis.md` — backward-pass cost
>   structure (this is a new lever within it).
> - `docs/design/clp-hot-start-backward-pass.md` — factorization reuse per
>   opening; **orthogonal and complementary** to this (that reduces
>   refactorizations, this reduces pivots).
> - `docs/design/backward-node-parallelism.md` — the load-imbalance / per-worker
>   direction (a separate, orthogonal lever).
> - `docs/design/dynamic-cut-selection-design.md` — the DCS lazy loop whose
>   opening-reuse warm chain this reorders.
> - `.claude/rules/sddp.md` — the cut sign/scale and determinism contracts this
>   change must not disturb.

## TL;DR

In the backward pass, each trial point solves the stage LP once per opening
`ω ∈ Ω` (the stochastic inflow realizations) on **one resident LP**, warm-
continuing the dual simplex from the previous opening's basis — only the RHS
(inflows) and bounds change between openings.

A warm dual-simplex re-solve does work roughly proportional to **how far the new
optimal basis is from the current one**, which tracks **how much the inflow RHS
changed** between consecutive openings. Today the openings are visited in their
natural index order, so consecutive realizations can be a dry year next to a wet
year — a large RHS jump. Measured on the `cobre_tuning` case (HiGHS, k2=0):

| solves              | count  | simplex iters/solve | share of backward solve time |
| ------------------- | ------ | ------------------- | ---------------------------- |
| ω≥1 (warm-continue) | 112216 | **~220**            | **88%**                      |
| ω=0 (cold head)     | 7560   | ~705                | 12%                          |

**Solving the openings in an order where consecutive ones are similar** (sorted by
a scalar derived from the per-opening **noise vector** — not the realized inflow,
which the LP couples to the state; see §2.1) makes each hop small, so each
re-solve starts close to its optimum and needs far fewer pivots. Because this attacks the term that is
~88% of backward solve time, even a moderate pivot reduction is a large backward
win. The proposal is gated on measurement (the gain is inflow-structure
dependent), but the case's strong spatial inflow correlation (the
"12/13 indefinite correlation matrices" warning) makes a 1-D ordering key
especially effective here.

## 1. What the backward opening loop does today

For each successor stage `s` and each trial point `m` (a forward-visited state
`x̂`), `process_trial_point_backward` (`crates/cobre-sddp/src/backward.rs`)
iterates the openings serially on one worker:

```
load the stage LP once for this trial point
for ω in 0..|Ω|:
    patch_opening_bounds(ω)      // install opening ω's inflow noise as RHS/bounds
    solve:
        ω == 0  → fresh solve (cold; DCS seeds + cold-starts)
        ω  > 0  → continue_carry: solve(None), warm from ω−1's basis + retained
                  factorization, lazily adding only the cuts ω additionally violates
    store the opening's dual π_ω, intercept, objective into per_opening_stats[ω]
aggregate the StagedCut from per_opening_stats over ω = 0..|Ω|
```

The matrix is fixed across openings; only the RHS/bounds move. This is the
textbook "fixed matrix, change bounds, re-solve" pattern, and the warm chain is
exactly why HiGHS retains its factorization (INVERT) across `ω>0`
(see `clp-hot-start-backward-pass.md`).

## 2. The cost: per-opening re-solve ∝ inter-opening RHS distance

A warm dual-simplex re-solve after an RHS change re-optimizes from the current
optimal basis. The number of dual pivots is, to first order, proportional to how
far the new optimal basis is from the current one — i.e. to the **magnitude of
the RHS/bound perturbation** between the two openings.

When openings are visited in arbitrary index order, consecutive perturbations are
arbitrary — often large (dry↔wet). The measured ~220 pivots per warm solve (vs
~705 for the cold head) reflects this: warm-starting already cuts pivots ~3×
versus cold, but 220 is still high for a "warm" re-solve, and it is paid
`(|Ω|−1)` times per trial point, on every trial point, every iteration.

### 2.1 What to measure the distance on — the noise, not the realized inflow

It is tempting to order by "total realized inflow per opening," but the realized
inflow at stage `t` is a function of the **incoming state** `x̂` (the PAR lags),
not a quantity we hold cheaply before solving. We do not need it. The thing that
actually varies between openings — and the single pre-image of _every_ per-opening
change `patch_opening_bounds` installs (inflow RHS via `transform_inflow_noise`,
load RHS via `transform_load_noise`, NCS bounds via `transform_ncs_noise`, which
**all read `raw_noise`**) — is the per-opening **noise vector `ξ_ω` (`raw_noise`)**.
It is the loop's input, available for all openings before any solve.

Ordering by the noise is not an approximation; it reproduces the realized-RHS
distance order **exactly**. With the PAR structure

```
inflow_{t,h}(ω) = [ μ + Σ_k φ_k·(a_{t-k,h} − μ) ]  +  σ_{t,h}·ξ_{ω,h}
                   └────────── C_h(x̂) ──────────┘     └ opening-dependent ┘
```

the bracket `C_h(x̂)` depends on the trial point's state but is **identical across
all openings** at that trial point. A common additive shift cannot change the
openings' relative order, and it cancels outright in pairwise distances:

```
‖rhs(ω) − rhs(ω′)‖ = ‖σ ⊙ (ξ_ω − ξ_ω′)‖      // C(x̂) cancels exactly
```

So the inter-opening distances — hence the optimal visiting order — are a function
of the **σ-weighted noise alone**, with the state term gone.

**Key consequence.** The ordering key is therefore **state-independent**: it
depends only on the stage's noise set, not on `x̂`. The backward openings come
from the fixed `OpeningTree` (`tree_view.opening(s, ω)`), built once at setup and
unchanged across iterations and trial points, and the σ weights are fixed
parameters — so the permutation is **run-constant**. It can be **precomputed once
at study setup** for every stage and merely indexed in the hot path (no sort, no
key computation, no per-iteration/per-trial-point/per-stage recompute), and it is
a pure, rank-invariant function of the synchronized tree. See §5.

## 3. The idea: order the openings to minimize the warm-start path

Visit the openings in an order that minimizes the total perturbation distance
traversed by the warm chain. The simplest effective form is a **monotone sweep**:
sort by a scalar key derived from the per-opening noise vector and solve in that
order, so each hop is a small step.

Why a 1-D key suffices here: inflows are **strongly spatially correlated** (the
spectral-decomposition warning flags 12 of 13 correlation matrices as near-
indefinite — wet/dry moves are basin-wide), so a single projection of the noise
orders realizations driest→wettest and consecutive ones really are similar. The
warm chain then climbs a smooth ramp instead of hopping randomly.

### 3.1 Ordering-key options (all functions of `raw_noise` only)

1. **σ-weighted aggregate noise** `key(ω) = Σ_h σ_{t,h}·ξ_{ω,h}` — cheapest, and
   exactly equal (up to the common shift) to ordering by aggregate realized
   inflow. Recommended starting point. Include the load/NCS noise components with
   their own scales for a unified key if those perturbations are material.
2. **First principal component** of the σ-weighted per-opening noise vectors —
   sort by the PC1 coordinate. Captures the dominant inter-opening spread in one
   number; given the correlation structure, PC1 ≈ the wet→dry axis. Better for
   multi-basin systems where a single aggregate hides offsetting moves.
3. **Nearest-neighbor path (small TSP) on the σ-weighted noise vectors** — orders
   by true multivariate proximity rather than a 1-D projection, using the exact
   `‖σ ⊙ (ξ_ω − ξ_ω′)‖` distances above. `O(|Ω|²)` to build (trivial for
   `|Ω|~20`); likely marginal over PC1 given the correlation. Hold in reserve.
4. **Median-start, sweep outward** — start the chain at the central (median-key)
   opening and expand in both directions, so the single cold solve is the most
   representative and both warm sweeps stay short. A refinement over plain
   ascending sort; worth trying if the extreme tails are expensive.

Start with option 1; escalate to PC1/TSP only if measurement shows residual pivot
cost. Monotone transforms in the noise→RHS map (log-space, σ-scaling, the
configured non-negativity truncation) preserve order, so the noise-based key stays
faithful to the realized distances; the only mismatch would be strong
per-component anisotropy between noise variance and simplex sensitivity, which the
σ-weighting (or, at most, embedding the actual patched RHS-delta vectors) absorbs.

### 3.2 Sweep direction — a measured flag, not a hard choice

Whether to sweep **ascending** (dry→wet) or **descending** (wet→dry) is left as a
configuration flag, with the default to be fixed once measured (§8). The rationale:

- **First order it is symmetric.** Ascending and descending traverse the same
  points reversed, so the total hop distance `Σ |key(i+1) − key(i)|` — and hence
  the aggregate warm-pivot cost — is identical. There is no first-order reason to
  prefer one.
- **Two second-order effects pull opposite ways, both small and problem-dependent:**
  1. _Cold-head placement._ The chain's first opening bears the cold solve (DCS) or
     the cross-iteration warm anchor (frozen). The driest opening is the most
     constrained / penalty-heavy / nearest the feasibility edge — the costliest and
     least numerically robust to cold-solve (the class that motivated the
     cold-retry escalation in `clp-hot-start`-adjacent work); the wettest is the
     most slack and easiest. This **favors descending** (cold head at the wettest).
  2. _Dual-simplex relax-vs-tighten asymmetry._ Under dual simplex an RHS step is
     repaired by re-feasibilizing the basic variables it perturbs. A relaxing step
     (ascending, +water absorbed by spill/storage) and a tightening step
     (descending, −water pushed into thermal caps / deficit) generally perturb
     different counts of variables; **which is cheaper is problem-dependent** (it
     hinges on whether spill, deficit, or thermal limits bind) and not determinable
     a priori. Hops outnumber the cold head `(|Ω|−1) : 1`, so this term dominates
     whenever it is non-trivial.

**Decision rule.** Expose `sweep_direction` as a flag; measure both on the
diagnostic (§8) — the per-opening `(noise_key, simplex_iterations)` log reveals the
asymmetry directly by regressing pivots on the _signed_ step — and fix the default
from data. Pending that, the blind default is **descending** (protects the cold
head; the warm-hop asymmetry is a coin-flip the cold-retry net covers either way).

If both tails prove expensive, prefer the **median-anchor outward sweep**
(§3.1 option 4) over either pure direction: cold-solve the central opening and
warm-sweep outward both ways, halving the max anchor distance and keeping the cold
solve off both extremes — at the cost of a basis capture/restore at the turn (the
single-chain `continue_carry` cannot continue linearly across the anchor).

## 4. Determinism-safe design (the crux)

The cut is a probability-weighted expectation over openings:

```
gradient  = Σ_ω p_ω · π_ω
intercept = Σ_ω p_ω · (objective_ω − π_ω · x̂)      (schematically)
```

Floating-point addition is **not associative**, so changing the order in which
the openings are _summed_ changes the cut bit-for-bit — which would violate the
`sddp.md` / workspace contract that results be **bit-identical within a mode
across MPI rank counts** and **invariant to input declaration order**.

The proposal therefore **decouples solve order from aggregation order**:

- **Solve** the openings in the warm-start-optimal (sorted) order.
- **Store** each opening's `(π_ω, intercept_ω, objective_ω)` into
  `per_opening_stats[ω]` indexed by its **original** `ω` (the buffer is already
  `|Ω|`-sized and `ω`-indexed).
- **Aggregate** the `StagedCut` by iterating `ω = 0..|Ω|` in **canonical order**,
  exactly as today — untouched.

With this decoupling the generated cut is **bit-identical to the current
release**. The change becomes purely about the _sequence of solves_, not their
values:

- Each opening's `π_ω` is its own exact optimum. Under DCS the lazy loop
  terminates at `violated == 0` for each opening, so the returned dual reproduces
  the all-cuts optimum **independently** of how many resident cuts the warm chain
  has accumulated or in what order the chain visited the openings. Reordering
  cannot change any per-opening value — only the pivot count to reach it.
- The sort key is a **deterministic, pure function of the opening noise** (which
  is itself the same on every rank), so the solve order is identical across rank
  counts and independent of worker assignment. Rank-invariance holds.
- Because the aggregation is canonical, the result is also invariant to the order
  in which openings are declared in the input — strengthening, not weakening,
  declaration-order invariance.

Net: exactness preserved, cross-rank determinism preserved, cuts unchanged to the
bit. This is what makes the optimization safe to enable by default once measured.

## 5. Implementation

**No FFI change, no new dependency, no solver-trait change.** The work splits into
a one-time setup precompute and a one-line hot-path change.

Because the ordering key is run-constant (§2.1) — the backward openings come from
the fixed `OpeningTree` read by `tree_view.opening(s, ω)`, built once at setup and
unchanged across iterations and trial points, and the σ weights are fixed
seasonal/AR parameters — the permutation is **precomputed once at study setup**,
not in the hot path:

1. **Setup-time precompute (once per run).** During study setup, where both the
   `OpeningTree` and the per-stage σ are in hand, compute for each stage `s` a
   permutation `solve_order[s]` = `0..|Ω_s|` sorted by the key (option 1: the
   σ-weighted aggregate of each opening's `raw_noise`) in the direction given by the
   `sweep_direction` flag (§3.2; default descending, fixed after measurement),
   tie-broken by `ω` for full determinism. Store a compact per-stage table —
   `Σ_s |Ω_s|` `u32` indices (≈ a few KB; e.g. 115 stages × 20 ≈ 2300 entries) —
   alongside the `OpeningTree` in the stochastic context / `StudySetup`, exposed as
   e.g. `tree_view().solve_order(s) -> &[u32]`. Derived from the already-
   synchronized tree, so it is **bit-identical on every rank by construction** —
   no per-worker computation.
2. **Hot-path change (the only one).** In `process_trial_point_backward`
   (`crates/cobre-sddp/src/backward.rs`), iterate the precomputed slice instead of
   the range: `for &omega in tree_view.solve_order(s)` rather than
   `for omega in 0..|Ω|`. No sort, no key computation, no `raw_noise` scan in the
   loop — just an indexed read. The **first** entry is the fresh/cold solve;
   replace the `ω == 0` test that selects `continue_carry` (and any ω=0 basis
   handling) with "is the first entry of `solve_order`."
3. **Index by canonical `ω`.** Continue writing each opening's result to
   `per_opening_stats[ω]` using its original `ω` (not its position in
   `solve_order`). Leave the cut aggregation loop unchanged (`ω = 0..|Ω|`) — this
   is what keeps cuts bit-identical (§4).
4. **Tie-in points to check:** `patch_opening_bounds` already takes the opening's
   noise via the same `tree_view.opening(s, ω)`; ensure the cold/fresh-solve
   selection and the DCS `continue_carry` flag key off "first solved" rather than
   literal `ω == 0`.

The precomputed table and the "first solved" flag are the only new state; the
solve, scoring, and aggregation bodies are otherwise unchanged, and the hot path
gains an indexed read, not a computation.

## 6. Scope boundary

In scope:

- Reordering the backward opening solve sequence within a trial point, with
  canonical-order cut aggregation.
- The per-stage ordering-key computation and its reuse buffer.

Out of scope (separate efforts, noted for clarity):

- **CLP factorization reuse** (`clp-hot-start-backward-pass.md`) — orthogonal;
  composes (see §7).
- **Load imbalance / per-worker persistence** (`backward-node-parallelism.md`).
- **Cross-iteration ω-head basis capture** for the DCS cold solve — composes
  (see §7), but is its own change.
- The forward and simulation passes (single solve per (stage, scenario); no
  opening chain to reorder).

## 7. Interaction with other work

- **CLP hot-start (orthogonal, composes).** Hot-start retains the _factorization_
  per opening (fewer refactorizations); ordering reduces the _pivots_ per opening.
  They stack. Ordering also benefits HiGHS, where the 220-pivot measurement was
  taken — HiGHS already retains INVERT, so its remaining inner-loop cost _is_
  pivots, which is exactly what ordering targets.
- **DCS lazy loop (composes, bonus).** Solving similar openings consecutively
  means each opening finds fewer _new_ violated cuts than its predecessor (similar
  inflows bind similar cuts), so the lazy add-and-resolve loop runs fewer rounds —
  fewer solves on top of fewer pivots per solve.
- **Cold-head basis capture (composes).** With a fixed sort key, the first-solved
  (cold) opening is the _same_ opening every iteration, so a future cross-
  iteration basis capture for that head warms a stable target.

## 8. Validation plan

The gain is inflow-structure dependent, so measure before defaulting:

1. **Diagnostic (cheapest, predicts the ceiling).** Instrument the current
   backward to log per opening `(noise_key, simplex_iterations)`, where `noise_key`
   is the σ-weighted aggregate of §3.1. Check whether pivot count correlates with
   the consecutive-opening noise distance in natural order, and what the distance
   distribution looks like under the sorted order. High correlation + large
   natural-order jumps ⇒ ordering will pay.
2. **Flagged prototype.** Implement the reorder behind a default-off flag. Run the
   `cobre_tuning` local case and compare backward warm-solve `simplex_iterations`
   (target: 220 → materially lower) and backward wall time. Because the cut is
   bit-identical (the determinism design above), the exactness and determinism
   suites must stay green unchanged under both backends — a strong correctness
   check that the decoupling was done right.
3. **Decide `sweep_direction` (§3.2).** Run the prototype both ascending and
   descending and compare total backward warm-pivots/wall. Use the per-opening
   `(noise_key, simplex_iterations)` log to confirm the relax-vs-tighten asymmetry
   (regress pivots on the signed step) and the cold-head cost at each extreme. Fix
   the default from the winner; if both tails are expensive, evaluate the
   median-anchor outward sweep instead.
4. **Production confirmation.** Confirm on a production-scale case that the
   per-warm-solve pivot count drops and that backward wall scales better with the
   cut pool.
5. **Default switch.** If the prototype shows a material, robust reduction with
   bit-identical cuts, make ordering the default (it cannot regress correctness,
   only speed). Otherwise keep it flagged and record the measurement.

### 8.6 Measured results — `cobre_tuning` A/B (2026-06-08)

Controlled A/B on the `cobre_tuning` case (HiGHS backend, training only, 5
forward passes × 10 iterations, DCS active, simulation off, threads=8). Three
variants identical except the reorder config; backward `simplex_iterations`
(pivots) is the primary signal — it is a deterministic, work-distribution-invariant
function of the config, so the differences below are exact, not noise.

| Variant            | Backward total pivots | mean/solve | Δ vs off | bw retries | fwd pivots        | bw wall |
| ------------------ | --------------------- | ---------- | -------- | ---------- | ----------------- | ------- |
| off (no reorder)   | 30,092,513            | 256.28     | —        | 19         | 5,789,046         | 662 s   |
| reorder ascending  | 27,663,119            | 235.59     | −8.1%    | 17         | 5,835,951 (+0.8%) | 615 s   |
| reorder descending | 27,640,611            | 235.40     | −8.1%    | 14         | 5,788,738 (≈off)  | 616 s   |

All three solve the identical 117,420 backward openings (clean comparison).

**Findings.**

- **Reorder is material**: ~8.1% fewer backward pivots and ~7% less backward wall,
  either direction.
- **Sweep direction is first-order symmetric** (§3.2 confirmed): ascending vs
  descending differ by only 0.08% on backward pivots. The tie breaks toward
  **descending** on every secondary signal — marginally fewer backward pivots
  (deterministic, so real), fewest LP retries (14 vs 17), and **forward-neutral**
  (its marginally-different cuts ripple into the forward pass ≈0%, whereas
  ascending perturbs forward +0.8%). Cold-head/relax-vs-tighten asymmetry is
  therefore small and favors descending, matching the blind default.
- **Decision: `sweep_direction` default = `descending`** (unchanged from the code
  default). The per-opening signed-step regression (`COBRE_W1_DIAG`) was not
  separately required — the aggregate near-tie already confirms the predicted
  first-order symmetry, and the secondary signals decide descending.
- Note (owner clarification 2026-06-08): the requirement is work-distribution
  invariance, not bit-identity vs the previous (off) version — reorder-on cuts
  differ marginally from off by warm-start path, which is why forward pivots shift
  slightly. The thread/rank-count determinism suites stay green with reorder on
  (the actual correctness gate).

### 8.7 Made unconditional & descending; config removed (2026-06-08)

On the strength of the §8.6 A/B (material ~8.1% backward-pivot reduction,
descending best on every secondary signal, forward-neutral), the owner chose to
make the backward opening reorder **unconditional** in the **descending**
direction and **remove its configuration options entirely** (clean delete rather
than keep-as-ignored):

- The `training.reorder_backward_openings` and
  `training.backward_opening_sweep_direction` config keys, the
  `OpeningSweepDirection` enum, and the `sweep_direction_from_config` mapping are
  gone. Setup now always installs the solve order with
  `SweepDirection::Descending`, and the backward pass always iterates it. The
  generic `SweepDirection` enum stays in `cobre-stochastic` (always passed
  `Descending`); the canonical-ω cut aggregation is unchanged, so cuts remain
  bit-identical across thread/rank counts.
- The `cobre_tuning` example config was migrated to drop the two now-deleted keys
  (`deny_unknown_fields` would otherwise reject it).
- `parity_hash_highs` (HiGHS) was re-baselined under the always-descending path
  (`COBRE_PARITY_REGEN=1`, owner-approved). The regenerated hashes came out
  **byte-identical** to the prior canonical-order baselines: the deterministic
  D-cases are single-opening per stage, so descending equals the identity
  permutation and the output cannot drift. CLP parity baselines are separately
  stale and were left untouched (out of scope).

## 9. Risks & open questions

- **Gain is data-dependent.** For tightly clustered inflows, natural order is
  already cheap and ordering helps little; for high-variance / multimodal inflows
  it helps a lot. Mitigation: the §8 diagnostic predicts this before
  implementation.
- **1-D key adequacy.** The σ-weighted aggregate noise may be a poor proxy if RHS
  sensitivity is dominated by a single basin out of phase with the aggregate.
  Mitigation: the PC1 or nearest-neighbor refinements (§3.1).
- **Sweep direction / cold-head placement.** Direction is first-order symmetric;
  the second-order effects (cold-head extreme, dual-simplex relax-vs-tighten) are
  small and problem-dependent. Handled as a measured flag (§3.2, §8 step 3),
  default descending; median-anchor sweep if both tails are expensive.
- **Aggregation-order regression risk.** The entire determinism guarantee rests on
  keeping the cut aggregation in canonical `ω` order while only the solve loop is
  reordered. The flagged prototype's green determinism/exactness suites are the
  guard; any drift there means the decoupling leaked.
- **Bounds-patch coupling.** Confirm nothing in `patch_opening_bounds` or the DCS
  `continue_carry` path implicitly assumes monotone or natural `ω` order beyond
  the basis it warm-starts from.
