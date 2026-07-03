# Water Travel Time in Cobre SDDP — Formulation & Convergence Analysis

Design memo (formulation link of a specialist chain; a Rust implementation plan
follows). Scope: the **future cost function (FCF)** and **Benders cut** impact of
adding water travel time between cascade hydro plants, the **initial-condition**
question, the **end-of-horizon** coupling, and the **determinism/contract** risks.
No code is prescribed; the math and the state/cut structure are. Every formulation
claim is grounded against the cited cobre symbols.

> **Revision note — chronological block mode.** This memo was first written before
> the `chronological` block mode landed (`BlockMode::Chronological` in
> `cobre-core`'s `temporal.rs`; per-block sequential storage `S⁰ → S¹ → … → Sᴷ`
> documented in `book/src/guide/block-modes.md`). The revision (i) re-grounds
> §2.1 against the **split** water-balance fills
> (`fill_parallel_water_entries` / `fill_chronological_water_entries`) and the
> `ζ`-vs-`τ_k` reality, and re-scopes §8 to the stage-clock bucket layer,
> (ii) adds **§2.5**, which establishes the
> **block-mode-independent state principle**, (iii) re-validates the §0.1 locks,
> and (iv) reframes the **scope** discussion (§8.5). The state/cut layout facts
> (§1, §3) survive unchanged — chronological mode was deliberately
> state-**preserving** (per-block storage lives in the CONTROL region, not the
> state region), so the cut vector is untouched by it.
>
> **Hard requirement (from the chronological-blocks work), enforced throughout the
> revision:** the state vector and cut layout at every stage are **identical
> regardless of `block_mode`**. The bucket count `B` is a pure function of
> `(travel times, stage lengths)` on the **stage clock**, computed **before**
> `block_mode` is consulted; `block_mode` changes only **where within a stage** an
> arrival lands (a stateless coefficient-placement concern). §2.5 states and proves
> this; the earlier "two regimes with different state footprints" framing was a bug
> (it let a chronological stage create a bucket a parallel stage would not) and has
> been removed.

---

## 0. Executive verdict

| #   | Question                                                                                                     | Verdict                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| --- | ------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | In-transit-bucket Markov-1 reframing equivalent to the textbook multi-lag E^k cut (term formalized in §2.1)? | **VALIDATED** (proof in §2). The bucket cut _is_ the textbook cut in lifted coordinates. Convexity preserved. `k₀>0` handled cleanly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 2   | FCF/Benders impact under column-bound pinning?                                                               | Buckets are ordinary Benders state: subgradient = `rc/col_scale` on the incoming bucket column, identical to storage. LB validity, monotonicity, a.s. finite convergence **preserved** (no new randomness). Cost: `+B` cut dimensions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 3   | Is "water in transit at study start" a required input?                                                       | **YES, genuinely required** . Recommend requiring it; a derive-from-`past_inflows` fallback is acceptable with a documented caveat. Genuine fork — §4.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| 4   | Does the augmented state value residual buckets at horizon end?                                              | **No, not automatically** when `V_{T+1}=0`. An explicit terminal credit (`V_eff = V + V_in_transit`) is required to avoid penalizing end-of-horizon upstream release. §5.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 5   | Determinism / contract risks?                                                                                | Canonical bucket-column sort, `col_scale` sizing, GEMM `d`, broadcast payload length, and one new reduction site to keep order-fixed. New `sddp.md` contracts proposed. §6.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| 6   | Best representation?                                                                                         | **RESOLVED: k-weighted volume buckets aggregated per downstream plant** (§7, §8.4 lock #2). Minimizes cut dimensionality (a sufficient statistic), and the chronological block-resolved deposit ($\sum_b \chi_{b,d}\,\tau_b\,D^b$, §2.5.2) is intrinsically a weighted volume the raw-lagged form cannot carry.                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| 7   | Variable stage length via `StageLagTransition`-style accumulation?                                           | **Yes** — the calendar-fraction overlap arithmetic produces volume-conserving, stage-dependent k-factors and depths. Stage-varying active-bucket sets reuse the `anticipated_state` (k_max global / per-stage active / padding-masked) pattern. §8.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| 8   | Does the chronological substrate change the state vector? (new)                                              | **No — by design (§2.5).** The bucket count `B` is a **stage-clock** quantity (the §2.5.1 window-overlap depth per arc, computed against the real stage calendar — `⌈t_v/h_t⌉` only on a uniform calendar; every declared arc, `t_v>0`, carries ≥1 bucket, the resolved §2.5.5 fork), computed before `block_mode` is consulted, so the state vector and cut layout are **`block_mode`-independent**. `block_mode` changes only **how the same stage's arrival mass is placed and attributed** (parallel: single row, stage-uniform deposit `k_d`; chronological: `κ`-routed rows, block-resolved deposit `χ_{b,d}`) — coefficient concerns, both stateless. Cross-stage transport is a **stage-level scalar bucket**, identical in both modes. |

---

## 0.1 Locked decisions (owner sign-off)

The original owner sign-off locked the five decisions below (current status lives
in the **§8.4 disposition table**; the note above records the amendments):

> **Revision — the sub-stage fork is resolved: exact overlap, both modes.** The
> locks below were signed off against a single-water-row stage LP (parallel mode
> only). The block-mode-independent state principle (§2.5) keeps every lock's
> object intact, and the owner has resolved the §2.5.5 sub-stage fork to **exact
> overlap**: a sub-stage delay ($0 < t_v < h_t$) carries **one bucket in both
> modes**. The fold convention (zero state, boundary-crossing mass lumped into the
> arrival stage) is **rejected** — §2.5.5 records why (chronological last-block
> distortion of $t_v/h_t$ of the release; silent loss of all inter-stage transport
> in parallel). Consequences for the locks: **lock #2 (representation) resolves
> decisively to k-weighted volume buckets** — the chronological deposit is a
> per-block-weighted volume ($\chi_{b,d}$, §2.5.2) that the raw-lagged form cannot
> carry. **Lock #4 (per-stage activation) is amended** — long stages no longer
> degenerate to "no outgoing buckets" ($k^0\approx1$): every declared arc has
> $L_{\text{arc}}(t) \ge 1$ at every stage (§2.5.1 window-overlap depth), with depth
> stage-varying (variable stage lengths) under the global-max + per-stage-mask
> discipline. **Locks #1, #3, #5 stand** and now bind for every declared arc — the
> IC seed and terminal treatment apply to sub-stage buckets too. §8.4 tabulates
> the per-lock disposition.

1. **Initial condition (§4.3):** REQUIRE `past_defluences` (registro VI); when
   absent, derive a logged proxy from `past_inflows`; never silently zero-seed.
   Seed at stage-0 incoming bucket column bounds (mirror `build_initial_state`).
   (Scope: this governs the **stage-0 IC seed** only; a mid-horizon upstream
   entrant's ring self-seeds to zero by conservation — §4.2, round-3 review.)
2. **Representation (§7.3):** ~~raw-lagged-defluence per source plant~~ —
   **SUPERSEDED, see §8.4 lock #2**: resolved to **k-weighted volume buckets
   aggregated per downstream plant** (the chronological block-resolved deposit is a
   weighted volume the raw-lagged form cannot carry). Original text kept struck
   through as the decision record.
3. **End-of-horizon (§5):** DEFER the `V_eff` terminal credit. Water still in
   transit past the horizon end (into/after a boundary FCF) is dropped — an
   accepted, documented imprecision, not silently zeroed elsewhere. Revisit when a
   terminal/cyclic value is in scope.
4. **Per-stage activation (§8):** ~~long stages degenerate to `k⁰≈1` (no outgoing
   buckets)~~ — **AMENDED, see §8.4 lock #4**: under exact overlap every declared arc
   has $L_{\text{arc}}(t) \ge 1$ at every stage. What stands from this lock: state
   dimension is stage-varying (global max, per-stage masked); inactive bucket slots
   are excluded from `nonzero_state_indices` by **reachability** (the §8.2
   over-estimation guard).
5. **Modeled→unmodeled boundary:** a stage still **receives** residual in-transit
   water that matures within it (collapses to a single incoming slot when the stage
   ≫ `τ`), preserving conservation in the upstream modeled stages and the
   backward-cut coupling to their releases. Only water maturing past the horizon end
   is dropped (per decision 3). (~~"a stage that generates no outgoing buckets"~~ —
   no such stage exists for a declared arc under exact overlap, §8.4.)

The implementation plan builds against the **§8.4 disposition table**, which
supersedes this list where they differ (locks #2 resolved, #4 amended, #1/#3/#5
broadened to every declared arc); this list is retained as the original sign-off
record.

---

## 1. Notation and the current cobre state model

Following SDDP.jl conventions, augmented with cobre symbols.

| Symbol                                | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $t$                                   | stage index, $t=1,\dots,T$                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| $v_h^t$                               | outgoing storage of hydro $h$ (hm³) — Benders state                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| $\hat v$                              | trial (visited) state                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| $D_i^t$                               | total _defluence_ of plant $i$ at stage $t$ = $\sum_{\text{blk}} \tau_{\text{blk}}\,(u_{i,\text{blk}}+s_{i,\text{blk}})$ (hm³) on the **main cascade arc**; turbine $u$, spill $s$. Diversion is a **separate arc** (the `diversion_upstream` map routes it to its own target plant in both water-balance fills), excluded from the v1 travel-time scope (unification memo §7.3); folding `div` into the cascade $D_i$ would misroute the diverted share onto the cascade bucket (round-2 review) |
| $\tau_k$                              | per-block flow→volume factor $=\,$`stage.blocks[k].duration_hours · M3S_TO_HM3`; the per-block coefficient in **both** water-balance fills (`fill_parallel_water_entries` uses `tau_h` on the single row's flow terms, `fill_chronological_water_entries` uses `tau_k` on block `k`'s row for **every** term)                                                                                                                                                                                     |
| $\zeta$                               | per-stage rate factor `StageLayout::zeta` $=\sum_k \tau_k$ (the stage total). In parallel mode it scales the once-per-stage families (inflow / AR-lag / withdrawal / evaporation); in chronological mode there is **no** `ζ` — each block's row carries `τ_k` and the rows telescope so `Σ_k τ_k = ζ`                                                                                                                                                                                             |
| $k_d$                                 | propagation factor: fraction of an upstream release arriving $d$ stages later, $\sum_d k_d = 1$                                                                                                                                                                                                                                                                                                                                                                                                   |
| $L$                                   | max travel-time lag in stages                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| $b_d^t$                               | in-transit _bucket_: volume that will arrive downstream in $d$ stages (incoming state at start of $t$)                                                                                                                                                                                                                                                                                                                                                                                            |
| $V_t(\cdot)$, $\underline V_t(\cdot)$ | true / lower-approximated recourse                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| $\theta$                              | epigraph (future-cost) variable, the global scalar column `StateLayout::theta`                                                                                                                                                                                                                                                                                                                                                                                                                    |

**The current cobre state vector** (`StateLayout`, file
`crates/cobre-sddp/src/lp/indexer/state_layout.rs`) is, in column order:

```
[storage (N)] ⊕ [inflow_lags (N·L_par, lag-major)] ⊕ [anticipated_state (A·k_max)]
   ⊕ [anticipated_state_out (A)] ⊕ [z_inflow (N, aux)] ⊕ [storage_in (N, incoming)] ⊕ θ
```

with `n_state = N·(1+L_par) + A·k_max`. **This is unchanged by chronological block
mode** (confirmed against `StateLayout::new`: `n_state = n*(1+l) + n_anticipated*k_max`,
no `n_blks` term). Chronological mode's interior per-block storage columns `S¹…Sᴷ⁻¹`
live in the **CONTROL** region (`StageLayout::storage_internal`, anchored at
`state.control_region_start()`), and the two block endpoints alias the existing state
columns — `block_storage_col(h, 0) = col_storage_in_start + h` (= `S⁰`, incoming state)
and `block_storage_col(h, K) = h` (= `Sᴷ`, outgoing state). So the state vector — and
therefore the cut byte layout — is identical across modes (this is what makes a policy
portable across modes, `book/src/guide/block-modes.md`). The bucket augmentation of
this memo therefore composes with either block mode: it adds state, and chronological
mode does not.

Three facts are load-bearing for this memo:

1. **Every state-region offset is a pure function of $(N, L_{par}, A, k_{max})$** —
   independent of `n_blks`/`n_thermals`. This is _exactly_ what lets a single global
   stage-0 cut map resolve onto the right column at every stage (`StateLayout` module
   doc). Any new bucket block must preserve this property.

2. **Incoming state is pinned via column bounds**, not equality rows
   (`set_col_bounds` on `storage_in` / `inflow_lags` / `anticipated_state`;
   `sddp.md` "State pinning uses column bounds"). The subgradient is the **incoming
   column's reduced cost divided by `col_scale`**:
   $\partial Q/\partial x = \text{rc\_scaled}/\text{col\_scale}[\text{col}]$
   (`training/backward/duals_extraction.rs::extract_duals_from_view`).

3. **The cut row negates and scales** the stored raw subgradient
   (`cut/row.rs::push_scaled_coefficient`: `batch.values.push(-coeff * col_scale[j])`,
   θ column gets `+col_scale[θ]`), so the row reads
   $-\nabla\!\cdot x + \theta \ge \text{intercept}$, i.e.
   $\theta \ge Q(\hat x) + \pi^\top (x - \hat x)$ with
   $\text{intercept} = Q(\hat x) - \nabla^\top \hat x$.

cobre already carries **two** finite-memory dynamics collapsed to Markov-1 state by
exactly the lifting this memo proposes:

- **PAR inflow lags** (`inflow_lags`): a per-hydro depth-$L_{par}$ ring buffer, shifted
  each stage in `stochastic/noise.rs::shift_lag_state`, seeded from `past_inflows` in
  `setup/mod.rs::build_initial_state`, with calendar-fraction weighting for short
  stages in `stochastic/noise.rs::accumulate_and_shift_lag_state` driven by
  `StageLagTransition`.
- **Anticipated-thermal commitments** (`anticipated_state`): a per-plant depth-$K_i$
  **k-injected** ring buffer (a decision is _deposited_ at slot $K_i-1$ and shifts
  down), `noise.rs::shift_anticipated_state`, seeded from
  `past_anticipated_commitments`, with a horizon gate
  `stage_idx + K_i < n_stages`.

The travel-time bucket is the same object as these two. That is the whole point.

---

## 2. Q1 — The in-transit-bucket reframing is mathematically exact

### 2.1 Setup — grounded against the SPLIT water-balance fills

`lp/builder/entries.rs::fill_state_and_water_entries` now **dispatches on
`stage.block_mode`** to one of two fills; both route the **full** upstream defluence
onto the downstream balance in the **same stage** (today's no-travel-time case), but
they build the row(s) differently, and the travel-time injection must be wired into
**both**:

- **Parallel** (`fill_parallel_water_entries`): one equality row per hydro,
  `row = row_water_balance_start() + h_idx`. Per-block flow terms
  (turbine / spillage / diversion, and upstream releases at `−`) carry the **per-block**
  factor `τ_h = stage.blocks[blk].duration_hours · M3S_TO_HM3`; the once-per-stage
  families (AR-lag `ψ`, inflow-penalty slack, withdrawal `±`, evaporation) carry the
  **stage total** `ζ = StageLayout::zeta`. The downstream balance of
  $j=\mathrm{downstream}(i)$ is

  $$
  v_j^t - v_j^{t-1} + \sum_{k}\tau_k\,(u_{j,k}+s_{j,k}+\text{div}_{j,k})
  - \underbrace{\sum_{k}\tau_k\,(u_{i,k}+s_{i,k})}_{=\,D_i^t\ \text{(arrives now)}}
  - \zeta\!\sum_{l}\psi\,a_{i,t-l} = \zeta(\text{base}-\text{withdrawal}).
  $$

- **Chronological** (`fill_chronological_water_entries`): `K` chained rows per hydro,
  `row = row_water_balance_start() + h_idx·K + (k−1)` for `k ∈ 1..=K`, coupling the
  per-block storage columns `block_storage_col(h, k)` and `block_storage_col(h, k−1)`
  (chain `S⁰ → … → Sᴷ`). Here `τ_k` replaces `ζ` **everywhere** — the per-block flow,
  the AR-lag `ψ`, the inflow-penalty slack, and the withdrawal slacks all carry
  `τ_k` on block `k`'s row. Summing the `K` rows telescopes the interior storage
  (`Sᵏ` cancels) back to the parallel single row because `Σ_k τ_k = ζ`; at `K = 1`
  it is byte-identical to parallel. The upstream defluence arrives on the
  **same block** row as it is released:
  `col_entries[turbine_col(u_idx, blk)].push((row, −τ_k))` on
  `row = row_water + j·K + (k−1)`.

Note the **`ζ`-vs-`τ_k` distinction is load-bearing for travel time**: a travel-time
coefficient placed on a flow term must follow the flow convention (`τ_k` per block in
either mode), while a coefficient placed on the once-per-stage inflow families follows
`ζ` (parallel) or `τ_k` (chronological). The original memo's "`ζ` on the upstream
defluence" was stale — flow has always been `τ`-scaled per block; only the
once-per-stage families are `ζ`-scaled. §2.5 shows that the within-stage travel-time
routing lives entirely inside the chronological chain's `τ_k` flow terms (a stateless
coefficient re-placement) and needs no `ζ`-family change.

Travel time replaces "$-D_i^t$ now" with the **propagation curve** (referência
§5.3.1, DESSEM):

$$
A_j^t \;=\; \sum_{d=0}^{L} k_d\, D_i^{t-d},\qquad \sum_{d=0}^L k_d = 1 .
$$

Here $k_0 D_i^t$ is a _current decision_; $k_d D_i^{t-d}$ for $d\ge1$ depend on
**past** defluences. In a DP, the recourse therefore depends on the storage **and**
on the defluence memory $\mathcal D_{t-1}=(D_i^{t-1},\dots,D_i^{t-L})$:

$$
V_t^{F}\big(v^{t-1},\,\mathcal D_{t-1}\big),
$$

and a Benders cut on $V_t^F$ carries subgradient components against **several prior
stages' decision vectors** — the textbook
$\alpha_5 \ge w_6 + \pi_6\!\left[E_5^1\Delta x_5 + E_4^2\Delta x_4 + E_3^3\Delta x_3\right]$
(the term "textbook" is formalized in the box below; that display is its
$t = 6$, $L = 3$ instance).

> **Definition — the "textbook" multi-lag formulation.** Wherever this memo (and
> the unification memo) says **"textbook"**, it means the classical treatment of
> **time-lagged linking variables in nested Benders / SDDP** — the hydrothermal
> formulation lineage of Pereira & Pinto (1991), whose authoritative reference
> instance for this design is the DECOMP manual's travel-time formulation
> (§5.3.1–§5.3.2; the external anchor mapped in the unification memo §1). Formally:
> the stage-$t$ subproblem's constraints receive earlier stages' decisions through
> **fixed lag-coupling matrices** $E_j^{k}$ — the block through which the stage-$j$
> decision vector $x_j$ enters the stage-$(j{+}k)$ constraint set at lag $k$ —
>
> $$
> A_t\,x_t \;=\; b_t \;+\; \sum_{k=1}^{L} E_{t-k}^{k}\,x_{t-k}
> $$
>
> (sign convention: lagged terms as RHS credit, matching arrivals entering the
> water balance as inflow). The value function therefore carries the **raw
> decision memory**, $V_t^F(x_{t-1},\dots,x_{t-L})$, and LP duality gives the
> Benders cut generated at stage $t$ around trial values $\hat x_{t-1},\dots$:
>
> $$
> \alpha_{t-1} \;\ge\; w_t \;+\; \pi_t^\top \sum_{k=1}^{L}
> E_{t-k}^{k}\,\big(x_{t-k}-\hat x_{t-k}\big),
> $$
>
> with $w_t$ the subproblem objective at the trial point and $\pi_t$ its
> constraint duals — **one subgradient block per lagged stage**, which is the
> defining feature (and the bookkeeping cost) of the textbook form. For water
> travel time the lag blocks are the propagation factors — $E_{t-d}^{d}$ carries
> $k_d$ on the upstream release columns — and the textbook **state** is the
> un-lifted raw-lagged stack, DECOMP's $x_t = (v_1, v_2, d_1, d_2)$ (unification
> memo §1, row 2). §2.2 proves the bucket lifting is this same object in Markov-1
> coordinates: $b = M\mathcal D$, and the $E^k$ blocks reappear as the rows of
> $M^\top$. Reference: Pereira, M.V.F. & Pinto, L.M.V.G. (1991), "Multi-stage
> stochastic optimization applied to energy planning", _Mathematical Programming_
> 52; _Modelo DECOMP — Manual de Referência_ (CEPEL, Outubro 2021), §5.3.

### 2.2 The lifting

Define the bucket state $b_t=(b_1^t,\dots,b_L^t)$ with $b_d^t = \sum_{m\ge1} k_{d-1+m}\,D_i^{t-m}$ ($k_e=0$ for $e>L$). Equivalently $b_t = M\,\mathcal D_{t-1}$ with the **Hankel** map $[M]_{d,m}=k_{d-1+m}$ (constant along anti-diagonals; support $m \le L{+}1{-}d$, i.e. anti-triangular — not lower-triangular Toeplitz; invertibility below rests on the anti-diagonal of $k_L$'s, $\det M = \pm k_L^L$).

**Calendar generalization (read before applying this section on a non-uniform
calendar).** The algebra below is written for **fixed** $k_d$ (uniform calendar).
Under stage-varying factors $k_d^{(t)}$ (§8, the non-uniform case), every result
survives with the stage superscript: the transition becomes
$b_d^{t+1} = b_{d+1}^t + k_d^{(t)} D_i^t$ (each stage deposits with **its own**
factors), the arrival identity is $A_j^t = k_0^{(t)} D_i^t + b_1^t$, and the
pullback runs through a stage-dependent $M_t$ (no longer Hankel). The
load-bearing fact is calendar-agnostic either way: the buckets are the **minimal
sufficient statistic** for future arrivals, which is all $V_t$ depends on —
invertibility of any particular $M_t$ is a remark, not a requirement.

**Claim (state-augmentation theorem).**

1. **Markov-1 transition.** From the definition,

   $$
   b_d^{t+1} \;=\; b_{d+1}^{t} \;+\; k_d\,D_i^{t}\qquad(b_{L+1}\equiv 0).
   $$

   _Proof._ $b_d^{t+1}=\sum_{m\ge1}k_{d-1+m}D_i^{t+1-m}
   = k_d D_i^{t} + \sum_{m'\ge1}k_{d+m'}D_i^{t-m'} = k_d D_i^{t} + b_{d+1}^{t}.\ \square$
   This is exactly the proposed update, and it is the **shift-with-injection** ring
   buffer of `anticipated_state`.

2. **Current arrival is bucket 1.**

   $$
   A_j^t \;=\; k_0 D_i^t + \underbrace{\sum_{d=1}^{L}k_d D_i^{t-d}}_{=\,b_1^t}
   \;=\; k_0 D_i^t + b_1^t .
   $$

   So the downstream balance "receives $b_1^t$ now, plus $k_0\cdot$(current defluence)" —
   exactly the proposal, and **$k_0>0$ co-exists cleanly**: the _same_ upstream
   release column $D_i^t$ appears (i) on the downstream balance with coefficient
   $-k_0$ (same-stage injection, decision), and (ii) in the bucket transition rows
   with coefficients $k_1,\dots,k_L$ (feeding future state). Today's code is the
   special case $k_0=1,\ k_{\ge1}=0$.

3. **The bucket cut IS the textbook cut (§2.1 definition) — the design-equivalence
   claim.** Why this claim exists (it is the §0 Q1 question, and the reason §2 was
   written at all): cobre's cut pool, `StateLayout`, and broadcast payload all
   assume a **one-step** state vector, so cobre must generate ordinary one-step
   cuts on $(v, b)$ — while the reference model (DECOMP) generates multi-lag cuts
   on $(v, \mathcal D)$ with subgradient blocks against several prior stages. If
   the two cut families were not the same object, the lifting would buy the
   one-step machinery at the price of a weaker (or merely different) FCF
   approximation, and DECOMP parity would be structurally out of reach. The claim
   rules that out: the two families are identical up to the fixed linear change of
   coordinates $b = M\mathcal D$.

   **The two value functions, plainly.** There is only **one** cost-to-go being
   approximated. Beyond storage, the travel-time physics requires exactly one
   piece of memory at the start of stage $t$: what water is still in the channel
   and when it arrives. The two symbols are the same quantity in two
   parameterizations of that memory:

   - $V_t^F(v, \mathcal D)$ — memory carried as the **raw past releases**
     $\mathcal D = (D^{t-1},\dots,D^{t-L})$: DECOMP's state. Future arrivals are
     recomputed from $\mathcal D$ on demand via the k-factors.
   - $V_t^A(v, b)$ — the same memory carried as the **already-mixed arrival
     schedule** $b_d$ = volume maturing in $d$ stages: cobre's state. The
     k-factors were applied at deposit time; nothing needs recomputing.

   Since $b = M\mathcal D$ is a fixed linear re-encoding,
   $V_t^A(v,b) := V_t^F(v, M^{-1}b)$ is the **same function with re-labeled
   axes**, not a second object — the "transformation" in the proof is that
   relabeling and nothing else. Per arc the dimension is $L$ either way, so the
   choice of coordinates is not about size; it is about transition/deposit
   structure (§7: buckets aggregate per downstream plant and carry the
   chronological block-resolved deposit; raw defluences do neither). And $M$,
   $M^{-1}$ never appear in code — no bucket is ever converted back to defluences
   at run time; the map is purely a **proof device**: because it is linear and
   invertible, hyperplanes map to hyperplanes with validity and support points
   preserved, which is exactly what the proof below needs in order to identify
   the two cut families.

   _Proof._ If $k_L\neq0$, $M$ is invertible, and
   $V_t^A(v,b):=V_t^F(v,M^{-1}b)$. Both are convex (§2.3). Any SDDP cut on $V_t^A$,

   $$
   \theta \ge \alpha + \beta_v^\top v + \beta_b^\top b,
   $$

   pulls back through $b=M\mathcal D$ to

   $$
   V_t^F(v,\mathcal D) \ge \alpha + \beta_v^\top v + (M^\top\beta_b)^\top \mathcal D,
   $$

   whose $\mathcal D$-subgradient $M^\top\beta_b$ has a nonzero component on **every**
   prior-stage defluence $D_i^{t-1},\dots,D_i^{t-L}$ — a multi-lag cut of exactly
   the §2.1 textbook shape. Conversely, $\mathcal D = M^{-1}b$ maps any textbook
   cut forward to a one-step bucket cut. $\square$

   **Where the $E^k$ matrices land in this identification.** Recall (§2.1 box)
   that $E_{t-m}^{m}$ is fixed constraint **data**, not a cut coefficient: the
   block through which the lag-$m$ release $D^{t-m}$ enters stage $t$'s constraint
   rows — for travel time, its water-balance row carries the single factor $k_m$
   (τ-scaled onto the release columns). The remaining shares of that same release,
   $k_{m+1}, k_{m+2}, \dots$, sit in the E-matrices of the **later** arrival
   stages ($E_{t-m}^{m+1}$ couples $D^{t-m}$ into stage $t{+}1$'s balance, and so
   on). Row $m$ of $M^\top$ is $\big(k_m,\ k_{m+1},\ \dots,\ k_{L-1+m}\big)$
   (zeros past $k_L$) — that release's **k-factor profile collected across its
   arrival stages**, i.e. the $D^{t-m}$-entries of the whole E-family stacked into
   one vector. So "the $E^k$ structure" and "the rows of $M^\top$" carry the same
   numbers in two bookkeepings: the textbook lag-$m$ cut coefficient contracts
   those entries against the arrival stages' duals (directly via
   $\pi_t^\top E_{t-m}^{m}$, and through the nested cuts for the later shares),
   while the pulled-back bucket coefficient contracts the same profile against
   the bucket subgradients,
   $(M^\top\beta_b)_m = \sum_{d} k_{d-1+m}\,\beta_{b,d}$ — with $\beta_{b,d}$,
   the marginal value of a unit of in-transit water at maturity slot $d$, playing
   the role the telescoped future-stage duals play in the textbook form.

**Interpretation.** "The cut depends on many prior stages" and "an ordinary one-step
cut on the augmented state" are the **same object in two coordinate systems**, related
by the fixed invertible linear map $M$. The reframing is _not_ a heuristic
approximation; it is the standard lifting that converts a finite-memory dynamic to
Markov-1, which cobre already performs twice (`inflow_lags`, `anticipated_state`).

Invertibility ($k_L\neq0$) is not required for correctness — the buckets are the
**minimal sufficient statistic** for future arrivals, which is all $V_t$ depends on.
With $L$ = true memory length, $k_L\neq0$ by definition; invertibility merely confirms
the bucket state retains no less and no more than the defluence memory.

### 2.3 Convexity is preserved

The augmented stage problem is still an LP in (controls, outgoing augmented state, $\theta$):

$$
\begin{aligned}
V_t(v^{t-1},b_t) = \min\ & c_t^\top x_t + \theta\\
\text{s.t. } & b_d^{t,\text{out}} - b_{d+1}^{t,\text{in}} - k_d D_i^t = 0 && (\text{linear transition})\\
& b_d^{t,\text{in}} \ \text{pinned via column bounds} && (\text{incoming state})\\
& \theta \ge \underline V_{t+1}(v^t, b_{t+1}) && (\text{PWL convex})\\
& (x_t, v^t)\ \text{feasible.}
\end{aligned}
$$

The transition rows are linear equalities; the recourse $\underline V_{t+1}$ is a max
of linear cuts (convex). $V_t$ is therefore convex piecewise-linear in the incoming
augmented state, exactly as storage and inflow-lag states already are. Augmenting the
state with linearly-evolving buckets preserves convexity. **Q1: validated.**

---

## 2.5 The block-mode-independent state principle

The original draft of this section split travel time into "intra-stage" and
"inter-stage" **regimes** and let a chronological stage's within-stage overflow spill
into a bucket. That framing had a latent bug against a **hard requirement** of the
chronological-blocks work, and this section is rewritten to eliminate it. The
requirement, and the single principle the whole section is now built around:

> **The state vector and cut layout at every stage are identical regardless of
> `block_mode`.** The bucket count $B$ is a pure function of `(travel times, stage
lengths)` computed on the **stage clock**, **before** `block_mode` is consulted.
> `block_mode` never changes $B$, the state vector, or the cut layout. It changes only
> **where within a stage** an arrival lands — a template-coefficient concern, which is
> stateless.

### 2.5.0 Why the requirement is non-negotiable, and the bug it kills

cobre already enforces the analogous rule for **block count**. The `StateLayout` module
doc (`crates/cobre-sddp/src/lp/indexer/state_layout.rs`) states it verbatim:

> "every offset here is a pure function of `N` (`hydro_count`), `L` (`max_par_order`),
> `A` (`n_anticipated`), and `k_max` — independent of `n_blks`/`n_thermals` — so a
> single global stage-0 layout resolves onto the correct column at every stage
> regardless of per-stage block counts."

Travel-time state must obey the **same** invariant, extended to `block_mode`: the cut
vector must be a pure function of `(N, L, A, k_max, B)` and **independent of both
`n_blks` and `block_mode`**. This is not a nicety — it is what makes a policy trained in
one mode loadable in the other (`book/src/guide/block-modes.md`), what keeps the
`K = 1`-chronological ≡ parallel byte-identity anchor intact, and what lets the single
global stage-0 cut map resolve at every stage.

**The bug (old §2.5.3).** The old text said a chronological block's arrival window
spilling past the stage end "feeds an inter-stage bucket." If a **chronological** stage
created a bucket that the **same** stage in **parallel** mode would not, then $B$ —
hence the state dimension — would depend on `block_mode`. That is exactly the violation
the requirement forbids. Killing it is the point of the rewrite.

**The root cause of the fix.** Buckets exist for **one** reason: SDDP stages communicate
**only** through the state vector, so any release mass that **crosses a stage boundary**
must be carried as state. "Crosses a stage boundary" is a **stage-clock** property — it
depends on the travel time $t_v$ and the stage length $h_t = \sum_k
\text{duration\_hours}_k$, and on **nothing** about how the stage is chopped into blocks.
So $B$ is **intrinsically** block-mode-independent, provided we (a) **compute** it on the
stage clock before any block reasoning, and (b) **forbid** any within-stage mechanism
from adding to it. The rewrite makes both explicit.

### 2.5.1 $B$ is a stage-clock quantity — the four-case identity

Let one arc carry travel time $t_v > 0$ anchored at stage $t$ (a _declared_ arc is one
with $t_v > 0$; $t_v = 0$ is today's instantaneous model and creates no bucket). Model
the release as a **uniform arrival density** over the delayed window
$[\text{start}_t + t_v,\ \text{end}_t + t_v)$ on the calendar clock (the same
uniform-over-the-period assumption the AR-lag machinery makes — density grounding in
§2.5.4, `compute_recent_observation_seed` seeding precedent in §4.2). The arc's
**bucket depth at stage $t$** is the **deepest future
stage** that window reaches, computed against the **real per-stage calendar** (the
§8.1 overlap arithmetic):

$$
L_{\text{arc}}(t) \;=\; \max\{\, d \ge 1 \;:\; [\text{start}_t + t_v,\ \text{end}_t + t_v)
\,\cap\, [\text{start}_{t+d},\ \text{end}_{t+d}) \ne \emptyset \,\},
$$

the **max**, NOT the count of overlapping stages: the window can skip an
intermediate stage entirely ($k_1 = 0$ when $t_v > h_t + h_{t+1}$), yet the ring must
keep a **contiguous** maturity index $d = 1..L$ — the shift
$b_d^{t+1} = b_{d+1}^t$ moves a $d{=}2$ deposit through slot 1 on its way out, so a
zero-deposit low slot cannot be dropped (counting overlaps instead of taking the max
would size $L = 2$ for a $k_2, k_3$-only arc and lose the $k_3$ mass — the same
non-conservation bug class in a new coat).

On a **uniform** calendar ($h_{t+d} = h_t$ for all $d$) this reduces to the closed form
$L_{\text{arc}} = \lceil t_v/h_t\rceil$. **The closed form is the uniform special case
only — it is WRONG on a non-uniform calendar**, because the depth depends on the
_downstream_ stage lengths, not the anchor's: a 720 h anchor with $t_v = 360$ h followed
by **weekly** stages has a window tail $[h_t, h_t + 360)$ spanning **three** 168 h
stages (true depth 3), while $\lceil 360/720\rceil = 1$ — under-counting drops the
lag-2/lag-3 mass (water non-conservation). The mirror case over-counts
($\lceil 360/168\rceil = 3$ where a weekly anchor before a monthly stage needs 1–2).
Every downstream use of $\lceil t_v/h_t\rceil$ in this memo means $L_{\text{arc}}(t)$;
the worked examples (§2.5.7) use uniform monthly calendars where the two coincide.

**Every declared arc carries at least one bucket at every stage**
($L_{\text{arc}}(t) \ge 1$ for $t_v > 0$ — the window always crosses the anchor's end
boundary; the resolved §2.5.5 fork). The depth uses only $t_v$ and the **stage
calendar**; **`n_blks` and `block_mode` do not appear**. The global bucket count is
$B = \sum_{\text{arc}} \max\big(\max_t L_{\text{arc}}(t),\ L_{\text{arc}}(\text{IC})\big)$
(or the aggregated-per-downstream-plant form of §7), where
$L_{\text{arc}}(\text{IC})$ is the depth of the **pre-study anchors** (§4.2). At
study start the in-transit residual is the mass released in $[-t_v, 0)$, arriving
over the study-clock interval $[0, t_v)$, so
$L_{\text{arc}}(\text{IC}) = \lvert\texttt{window\_period\_overlaps}(0,\ t_v,\
\text{study\_durations})\rvert$ — anchored at **study start** against the
**study** calendar, with **no** $k_0$ same-stage exemption (pre-study water has
no same-stage share), unlike the in-study $L_{\text{arc}}(t)$ which subtracts 1.
This exceeds $\max_t L_{\text{arc}}(t)$ by **at most one bucket**, and only on a
**fine-first / coarse-next** study calendar ($h_0 < t_v$ with the stage
containing $t_v$ wider than $h_0$; e.g. study durations $[24, 720, 720, \dots]$ h,
$t_v = 30$ h: $L_{\text{arc}}(\text{IC}) = 2$ while every in-study anchor has
$L_{\text{arc}} = 1$). On a uniform calendar the two coincide (no deepening).
Omitting the IC anchor from the max truncates that one legitimate early bucket on
fine-first calendars — water non-conservation in exactly the early stages
short-term scheduling cares about. The coarseness of the pre-study
`past_defluences`/`past_inflows` data is a **seeding-value** precision concern
(how the coarse aggregate is distributed into these buckets — the IC seed step),
not a depth concern: the bucket **count** depends only on $t_v$ and the study
calendar, never on the pre-study period width. The per-stage/arc **active mask** rides the
$k_{max}$-global / $K_i$-active / padding-masked discipline `anticipated_state`
uses (`state_layout.rs::set_nonzero_mask`, §8.2). Mask by the **full outgoing
reachable set** — own-release deposits ($L_{\text{arc}}(t)$ is only this
component) **∪ in-flight residual from earlier anchors ∪ IC residual** — not by
$k_d \ne 0$: a zero-deposit transit slot still carries mass through the ring shift
(e.g. $t_v = 2.5\,h_t$ uniform: deposits hit slots 2–3 only, yet slot 1 passes
their mass onward). For in-study anchors the union adds nothing (arrival times
are monotone in the anchor, so older anchors never reach deeper); the IC residual
is the one genuine extension, and masking too tightly is validity-safe but drops
legitimate cut coefficients. The full reachable set stays **contiguous**
$\{1..L_{\text{active}}(t)\}$ (a computed per-stage max over own-release and IC
residual, not a constant), so `set_nonzero_mask`-style contiguous emission carries
over unchanged. **The mask contract has two sides** (insurance FCF review): the cut
**row** renders only unmasked slots (§8.2's guard), while the **intercept**
computation dots the FULL coefficient vector against the global trial state
($\bar\alpha = Q(\hat x) - \sum_{\text{all } j}\beta_j\hat x_j$) — validity requires
every masked slot to carry **zero dual and zero trial value**, which reachability
masking guarantees (a beyond-$L_{\text{active}}$ slot has no rows and no deposits)
but a test must pin on the intercept side, not only the row side.

The four-case identity (**uniform calendar shown** — depths are the closed-form values;
on a non-uniform calendar the depth column becomes $L_{\text{arc}}(t)$ above, and the
mode-identity conclusion is unchanged because the definition never consults
`block_mode`):

| Case               | $t_v$ vs $h_t$  | $L_{\text{arc}}$ (buckets/arc) | Parallel mode                                                                                                         | Chronological mode                                                                                                            | Same $B$? |
| ------------------ | --------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------- |
| **Instant**        | $t_v = 0$       | $0$                            | `−τ` on the single row, same stage (today)                                                                            | `−τ_k` on the **same block** row (today)                                                                                      | ✅ (0=0)  |
| **Sub-stage**      | $0 < t_v < h_t$ | $1$                            | same-stage mass on the single row at $k_0$; crossing mass ($k_1 = t_v/h_t$ of each release) deposited into the bucket | same-stage mass **routed across blocks** $b\to b'$; per-block crossing shares $\chi_{b,1}$ deposited into the **same** bucket | ✅        |
| **Multi-stage**    | $t_v > h_t$     | $\lceil t_v/h_t\rceil$ = span  | no same-stage mass ($k_0=0$); `span` buckets deliver into future stages                                               | no same-stage mass; the **same** `span` buckets, delivered by template density                                                | ✅        |
| **Exact multiple** | $t_v = m\,h_t$  | $m$                            | $k_0=0$; `m` buckets                                                                                                  | $k_0=0$; the **same** `m` buckets                                                                                             | ✅        |

The load-bearing column is the last: **$L_{\text{arc}}$, hence $B$, is identical across
modes in every case.** The two modes differ only in the **middle columns** — _where the
same-stage portion of the arrival lands_ — and that is a coefficient-placement concern
with no state footprint (§2.5.2).

### 2.5.2 What `block_mode` actually changes: coefficient placement only (stateless)

The same-stage portion of an arrival (the mass that does **not** cross a stage boundary)
is placed differently by the two fills, but neither placement touches state:

- **Parallel** (`fill_parallel_water_entries`): the stage has **one** water row per
  hydro, `row_water_balance_start() + h_idx`. All same-stage arrival mass lands on that
  single row with coefficient `−τ` (the aggregate flow→volume factor). There is no
  within-stage time axis (`temporal.rs`: "independent sub-periods solved
  simultaneously"), so a sub-stage delay simply lands in the same aggregated balance —
  timing within the stage is invisible by construction.
- **Chronological** (`fill_chronological_water_entries`): the stage has **`K` chained
  rows** per hydro, `row_water_balance_start() + h_idx·K + (k−1)`, blocks ordered by
  `Block::index` with cumulative offsets $H_b = \sum_{k<b}\text{duration\_hours}_k$.
  Today the upstream release in block `b` lands on the downstream **block-`b`** row
  (`col_entries[turbine_col(u_idx, blk)].push((row_water + j·K + (k−1), −τ_k))`).
  **Nothing forces same-block arrival** — the `−τ_k` may instead be split across
  **later blocks of the same stage** using block-overlap factors
  $$
  \kappa_{b\to b'} \;=\; \frac{\big|\,[H_b + t_v,\ H_{b+1} + t_v)\ \cap\ [H_{b'},\ H_{b'+1})\,\big|}{\text{duration\_hours}_b},
  $$
  landing $(\text{row}_{j,b'},\ -\tau_b\,\kappa_{b\to b'})$ on the downstream block-`b'`
  row $\text{row}_{j,b'} = \texttt{row\_water\_balance\_start()} + j\cdot K + b'$, for
  every downstream block `b'` **within the same stage** ($b' \le K-1$).

**The crossing mass — mode-refined deposit coefficients on the same bucket.** The mass
that **does** cross a stage boundary deposits into the stage-clock bucket of §2.5.1, and
here the two modes differ in **attribution**, not in state:

- **Parallel** cannot attribute a release to a position within the stage (blocks are
  simultaneous sub-periods), so every release column deposits at the **stage-level**
  factor: coefficient $k_d\,\tau_{\text{blk}}$ on the bucket-$d$ definition row, with
  the same-stage remainder $k_0\,\tau_{\text{blk}}$ on the single water row
  ($k_0 = 1 - \sum_{d\ge1} k_d$; for a sub-stage delay $k_1 = t_v/h_t$).
- **Chronological** attributes **by block**: block `b` deposits at its own crossing
  share into future stage $t+d$,
  $$
  \chi_{b,d} \;=\; \frac{\big|\,[H_b + t_v,\ H_{b+1} + t_v)\ \cap\ \big[\text{stage } t{+}d\text{'s interval}\big]\,\big|}{\text{duration\_hours}_b},
  $$
  i.e. coefficient $\chi_{b,d}\,\tau_b$ on the bucket-$d$ row, with the in-stage
  remainder $\kappa$-routed per the bullet above. Per-column conservation:
  $\sum_{b'}\kappa_{b\to b'} + \sum_{d\ge1}\chi_{b,d} = 1$.

The bucket **variable** is identical in both modes — same meaning (volume crossing the
boundary, maturing in $d$ stages), same count, same column, same cut coordinate. Only
the deposit rows refine. Three facts pin this down:

1. **Aggregation consistency** (sub-contract 2): $\sum_b w_b\,\chi_{b,d} = k_d$ with
   $w_b = \text{duration\_hours}_b / h_t$ — automatic when $\chi$ and $k$ are read off
   **one** shared uniform arrival density (§2.5.4).
2. **The forbidden alternative — stage-uniform deposits in chronological mode.**
   Depositing every chronological block at the stage-level $k_d$ compiles and conserves
   mass, but credits a late-block release as mostly-in-stage when its arrival is almost
   entirely next stage (example (iii): with $t_v = 250\text{ h}$ against a 720 h stage,
   a last-third-of-the-stage release arrives wholly next month, yet stage-uniform would
   keep $k_0 \approx 65\%$ of it in-stage) — the same distortion class as the rejected
   fold. Block-resolved $\chi_{b,d}$ is the point of chronological mode.
3. **Cross-mode cuts are structurally portable, not numerically identical.** The
   deposit coefficients differ between modes for the same stage — they must, because
   chronological knows the block clock and parallel cannot — exactly the precedent the
   chronological feature set for $K \ge 2$ (`book/src/guide/block-modes.md`,
   coarse-train/fine-simulate). At $K = 1$ the single block spans the stage, so
   $\chi_{0,d} = k_d$ and the two modes are **byte-identical** — the existing parity
   anchor extends.

In **both** modes all of this is a `T_t`-matrix edit: the release column is the same
decision; only which rows it couples to (and with what fractions) changes. `n_state`,
`col_scale`, GEMM `d`, and the MPI basis payload are all untouched. Convexity and cut
validity are preserved trivially — $V_t$ remains an LP value function in
$(v^{t-1}, \text{inflow lags}, \text{buckets})$, and the Benders extraction
(`rc/col_scale` on the incoming columns) is byte-for-byte the storage/§3 machinery.

### 2.5.3 The three sub-contracts that make the invariant hold

These are the candidate `sddp.md` entries. Each is a contract — a plausible-looking
deviation reintroduces `block_mode`-dependence in the state vector.

**Sub-contract 1 — $B$ from stage lengths only.** The per-arc bucket depth
$L_{\text{arc}}(t)$ (the §2.5.1 window-overlap count against the real calendar) and the global
$B = \sum \max\big(\max_t L_{\text{arc}}(t),\ L_{\text{arc}}(\text{IC})\big)$ are
computed from $t_v$, the stage lengths $h_t = \sum_k \text{duration\_hours}_k$, and
the **pre-study period lengths** (the IC anchors, §2.5.1) **alone** — never from
`n_blks` and never from `block_mode`. $B$ is the global maximum over all anchors
(in-study and IC) with a per-stage/arc active mask, the `anticipated_state`
$k_{max}$-global / $K_i$-active pattern (`state_layout.rs::set_nonzero_mask`). The forbidden alternative:
deriving any part of $B$ inside a block-aware code path — that couples the state
dimension to `n_blks`/`block_mode` and breaks the `state_layout.rs` invariant quoted in
§2.5.0. (Grounds the `n_blks`-independence directly against that module doc.)

**Sub-contract 2 — the chronological tail feeds the PRE-EXISTING stage-clock bucket; it
never allocates a new one.** This is the fix for the original-draft bug (§2.5.0). When a
chronological block's arrival window overflows the stage end, that overflow mass does
**not** create a bucket — the bucket for that arc/lag **already exists** (allocated by
sub-contract 1 on the stage clock, and present identically in parallel mode). The block
clock supplies
only finer **origin structure** that **aggregates back** to the mode-independent
stage-level $k_d$. The consistency requirement making this exact:

$$
\sum_{b:\,H_b\in[0,h_t)} w_b\,\chi_{b,d} \;=\; k_d,
\qquad w_b = \frac{\text{duration\_hours}_b}{h_t},
$$

i.e. the per-block crossing shares $\chi_{b,d}$ (§2.5.2), weighted by block share,
**must aggregate to** the stage-level $k_d$ — these are the deposit coefficients on the
bucket-$d$ definition row in chronological ($\chi_{b,d}\,\tau_b$) and parallel
($k_d\,\tau_{\text{blk}}$) mode respectively. This holds **iff** the block-level
$\chi$/$\kappa$ **and** the stage-level $k_d$ are built from **one shared
uniform-release arrival density** over $[t_v, t_v+h_t)$ (§2.5.4). State that
shared-density consistency as the contract: _the block partition of a stage is a
refinement of that stage's arrival density; refining a partition cannot change the
integral over a coarser cell._ The forbidden alternative — building $\chi/\kappa$ from
one density and $k_d$ from another — makes the chronological and parallel cuts diverge
and silently violates conservation.

**Sub-contract 3 — buckets are stage-level SCALARS; cross-stage delivery uses a FIXED
template density, not per-arrival-block state (the sufficient-statistic boundary).**
This is the least obvious and the most important. A bucket $b_d$ is **one scalar per
`(arc, lag)`** — the total volume maturing $d$ stages ahead. When it **delivers** into
its arrival stage $t+d$, that stage may itself be chronological with its own $K$ blocks;
the delivered water must be spread over that stage's rows. **It is spread by a
precomputed, `block_mode`-fixed density** (uniform-over-the-early-blocks, or the same
arrival-density restricted to $[t_v-d\,h_t,\ t_v-d\,h_t+h_t)$), **not** by tracking which
origin block each unit came from. Tracking origin-block ↔ arrival-block correlation
across a stage boundary would require the bucket to carry a per-block vector, and its
length would scale with the delivery stage's `n_blks` — **violating sub-contract 1**. So
the deliberate boundary is:

- **Within a stage**, routing is **exact**: block `b` → block `b'` with the true
  $\kappa_{b\to b'}$ (§2.5.2), because it is all one LP, no state involved.
- **Across a stage boundary**, the state is a **scalar** and the origin-block↔arrival-block
  correlation is **aggregated away** into a fixed template density — a **bounded,
  documented approximation** (the cross-stage water loses its sub-stage timing detail) in
  exchange for an `n_blks`- and `block_mode`-independent state vector.

This is the sufficient-statistic line: the minimal cross-stage sufficient statistic for
future cost is the **scalar** maturing volume per lag; sub-stage arrival timing is only
recoverable within the arrival stage's own LP, spread by the template. The forbidden
alternative — a per-block bucket vector — is a state explosion that reintroduces exactly
the `n_blks`-dependence the requirement forbids.

### 2.5.4 The shared arrival density — grounded in the AR-lag precedent

Sub-contracts 2 and 3 both rest on **one** uniform-release arrival density, and cobre
already computes exactly this shape for AR lags. `compute_monthly_transition`
(`crates/cobre-sddp/src/stochastic/lag_transition.rs`) splits a stage's contribution
across calendar periods by **interval-overlap fractions**: `days_in_period` intersects
`[stage_start, stage_end)` with `[period_start, period_end)`, and the weight is
`overlap_days · 24 / period_hours` — a normalized overlap of a **uniform-over-the-stage**
quantity with a target period. The travel-time density is the identical construction with
the target intervals **shifted by $t_v$**:

- **Stage-level $k_d$** = overlap of the shifted window $[t_v, t_v+h_t)$ with future
  **stage** interval $[d\,h_t, (d{+}1)h_t)$, normalized — the `compute_monthly_transition`
  arithmetic on the **stage** clock.
- **Block-level $\kappa_{b\to b'}$** = overlap of block `b`'s shifted window
  $[H_b+t_v, H_{b+1}+t_v)$ with block `b'`'s interval $[H_{b'}, H_{b'+1})$, normalized —
  the **same** arithmetic on the **block** clock.

Because both are overlaps of the **same** shifted uniform density against two nested
partitions (stages ⊃ blocks), the block factors **necessarily** aggregate to the stage
factors (sub-contract 2's equality). The travel-time overlap precompute should be a
**sibling** of `compute_monthly_transition` (a different cadence — arrival windows, not
AR periods — reusing the interval-overlap **pattern**; the existing day-granular
functions are insufficient, as established above), **not** overloaded onto it.
The determinism contract is the canonical per-stage construction of both densities in a
fixed (sorted) order, identical in spirit to the existing lag-transition precompute.
Note the density shapes **duals as well as primal mass**: the $k_d$ sit on the
**release columns** in the definition rows (the incoming bucket column carries a
plain $-1$; its reduced cost is the definition-row dual, $rc(b_{d+1}^{\text{in}}) =
\pi_{\text{def}(d)}$), so a different density changes which rows the release feeds
and hence the equilibrium duals — the extracted subgradient $\beta_b$ and the
marginal value of in-transit water the policy acts on. A routing curve
(`temporal-lag-unification.md` §2.3) changes the water values, not only the volume
split; the uniform assumption is a modeling choice about prices too.

**DECOMP consistency.** This is also the reference model's construction, with each
claim resting on its own section of the manual (CEPEL, Outubro 2021). **Defluence
anchors:** §5.3 propagates upstream **defluence** with fixed fractional factors
$k_0^t, k_1^{t-1}, \dots$ (symbolic in the manual's figure) **including a same-stage
$k_0 > 0$** — exact overlap, no fold; its Benders cut is the textbook multi-lag form (§2.1)
over a state carrying raw defluence volumes (the un-lifted coordinates of §2.2); and
the lagged **defluence** arrival is spread over the arrival stage's patamares
duration-proportionally, $(Q^{t-tv}+S^{t-tv})_k = \tfrac{d_k}{D}(Q^{t-tv}+S^{t-tv})$ —
sub-contract 3's $\rho$ with the uniform density. **Inflow cross-check (separate
claim):** the manual's worked non-uniform-calendar example ($T_v = 15$ d over weekly
stages, §4.5.14.2, Figs. 5.5b/5.5c) is an **ENA/inflow-propagation** computation, not a
defluence balance; it yields $k_2 = 6/7,\ k_3 = 1/7$ — the same numbers this uniform
density produces — evidence that DECOMP uses one calendar-overlap arithmetic for both
quantities, cited here as arithmetic cross-validation, not as a defluence-balance
citation. Full mapping, divergences (lifted coordinates; block-resolved $\chi$ where
DECOMP has no chronological mode), and the shared-resolver proposal:
`docs/design/temporal-lag-unification.md`.

### 2.5.5 The sub-stage fork — RESOLVED: exact overlap, both modes

For $0 < t_v < h_t$ the arrival window straddles exactly **one** stage boundary, so the
question was whether the boundary-crossing mass is carried as state
($L_{\text{arc}} = 1$, "exact overlap") or lumped into the arrival stage
($L_{\text{arc}} = 0$, "fold"). **Resolved by the owner: exact overlap, in both
modes** — every declared arc carries the bucket; the fold is rejected. Both choices
satisfy the reconciliation principle (the same $L$ in both modes); the resolution is
about accuracy, not `block_mode`-independence. The decision record:

| Convention                  | $L_{\text{arc}}$ for $0<t_v<h_t$ | What happens to the boundary spill                                                                                                                                        | Trade-off                                                                                                                          |
| --------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Fold / whole-stage lump** | **0**                            | The spill is **folded into the arrival stage** — in chronological mode onto the **last block** (so the timing error is one-block small); in parallel onto the single row. | Delivers the owner's "no state when $t_v < h_t$"; error bounded, grows as $t_v \to h_t$; chronological routing already shrinks it. |
| **Exact overlap**           | **1**                            | One bucket carries the physically-real spill into the next stage.                                                                                                         | Conserves cross-stage timing exactly; costs $+1$ bucket dim **per arc** even for hours-scale delays.                               |

**Why the fold is rejected** — it has two failure modes, one per mode:

- **Chronological:** the crossing mass lumps onto the **last block's** row —
  systematically at stage-end, so block-resolved storage, FPHA head, and evaporation
  exposure are distorted in a consistent direction, and the distortion is exactly the
  within-stage timing chronological mode exists to resolve. It is not small: the
  crossing fraction is $t_v/h_t$ of the stage release under the uniform density — e.g.
  $t_v = 250\text{ h}$ against a 720 h stage misplaces $\approx 35\%$ of the upstream
  release (worked in example (iii)).
- **Parallel:** the crossing mass is silently absorbed into the same-stage balance — a
  sub-stage delay produces **no inter-stage transport at all**, even when a third of
  the water physically arrives next stage.

Against that, the fold's only benefit was one saved state dimension per declared arc.
Travel-time arcs are declared per arc precisely where the delay matters, so the saving
is small and the error is paid exactly where the feature is wanted. A $t_v$ that is
negligible against every stage length is better handled by **not declaring it** (setup
can log an advisory when $\max_t t_v/h_t$ is tiny) than by folding it away silently.

Enumerating the alternatives for the crossing mass closes the question: fold in both
modes (the distortions above); bucket in chronological only (state depends on
`block_mode` — the §2.5.0 bug); push into the next stage's RHS (impossible — the mass
is decision-dependent, the RHS is template); **bucket in both modes** — the only
accurate, mode-independent option.

With exact overlap, $L_{\text{arc}}(t) \ge 1$ holds for every declared arc (§2.5.1)
and the sub-stage case is not special anywhere downstream: the bucket seeds from the
IC (§4), joins the cut vector (§3), and delivers by the template density
(sub-contract 3) like any other.

### 2.5.6 Invariance anchor (the byte-identity guarantees)

The principle yields concrete regression anchors the implementation plan must pin:

1. **Travel-time OFF is byte-identical to today.** When no arc carries travel time,
   $B = 0$ (sub-contract 1), the arrival density is the instantaneous $\kappa_{b\to b}=1$
   / $k_0=1$ special case, and every fill emits exactly today's `−τ`/`−τ_k` on the
   same-stage/same-block row. The `StateLayout`, the LP, and the cuts are unchanged.
2. **$K=1$-chronological ≡ parallel remains byte-identical** with travel time off, and
   the **state dimension** stays equal across modes with travel time **on**
   (sub-contract 1) — extending the existing chronological parity anchor to cover the
   travel-time-off layout.
3. **$B = 0$ iff no arc has travel time**, independent of how many stages are
   chronological and how many blocks they carry; conversely every declared arc
   ($t_v > 0$) contributes $L_{\text{arc}}(t) \ge 1$ (§2.5.1, §2.5.5).

### 2.5.7 Worked numeric examples

Three reproducible cases. Each takes **one arc** (upstream `u` → downstream `j`), a
single travel time $t_v$, and monthly stages. Convention: exact overlap (§2.5.5), so
every declared arc carries $L_{\text{arc}}(t) \ge 1$ buckets — all three examples use
**uniform monthly calendars**, where the closed form $\lceil t_v/h_t\rceil$ applies
(§2.5.1; non-uniform calendars need the general window-overlap count). Volumes in the balance rows
are the release column's coefficient `−τ · κ` (in-stage) or `τ · χ` (bucket deposit);
only the routing (which row, what fraction) is shown — the release column itself is
unchanged.

#### Example (i) — sub-stage delay: monthly stage, 30 daily blocks, $t_v = 2$ days

**Setup.** Stage $t$ = one 30-day month, $h_t = 720\text{ h}$. Chronological, $K = 30$
daily blocks, each $\text{duration\_hours}_b = 24\text{ h}$, so $H_b = 24b$ hours for
$b = 0,\dots,30$. Travel time $t_v = 2\text{ days} = 48\text{ h}$.

**Stage-clock $B$ (computed first, before blocks).** Sub-stage case
($0 < t_v = 48 < h_t = 720$): $L_{\text{arc}} = \lceil 48/720\rceil = 1$ — **one bucket,
in both modes**. The arrival window $[48, 768)\text{ h}$ lies inside the stage except
its last $768 - 720 = 48\text{ h}$; that boundary-crossing mass is what the bucket
carries. **So $B_{\text{arc}} = 1$, identically in parallel.**

**Block routing (chronological, `block_mode`-dependent placement only).** Block `b`'s
release arrives over $[H_b + 48,\ H_{b+1} + 48) = [24b+48,\ 24b+72)$, which is exactly
downstream block $b' = b+2$'s interval $[24(b{+}2),\ 24(b{+}3)) = [24b+48, 24b+72)$. So
$\kappa_{b\to b+2} = 1$ and all others 0 — a clean **$b \to b+2$** shift:

$$
\text{col\_entries}[\text{turbine\_col}(u, b)]\ \text{gets}\ \big(\text{row}_{j,\,b+2},\ -\tau_b\big),\quad \tau_b = 24\cdot\text{M3S\_TO\_HM3}.
$$

For $b = 0,\dots,27$ this lands inside the stage (`row_water + j·30 + (b+2)`): pure
in-stage routing, $\chi_b = 0$. For $b = 28, 29$ the arrival windows $[720, 744)$ and
$[744, 768)$ lie **entirely past the stage end**, so these two blocks route nothing
in-stage and deposit **fully into the bucket**: $\chi_{28,1} = \chi_{29,1} = 1$,
coefficient $\chi\,\tau_b$ on the bucket-definition row. Duration-weighted deposit
$= 2/30 = 48/720 = t_v/h_t$ — the aggregation consistency of sub-contract 2, verified.

**Parallel comparison.** Parallel cannot attribute releases to blocks: **every** release
column deposits at the stage-level $k_1 = 48/720 = 1/15$ (coefficient
$k_1\,\tau_{\text{blk}}$ on the bucket row) and keeps $k_0 = 14/15$ on the single water
row `row_water + j`. Same bucket, same $L_{\text{arc}} = 1$ — different columns feed it
(all 30 columns at $6.7\%$ each vs. two blocks at $100\%$): the sending-side refinement
of §2.5.2, coinciding when the release profile is uniform across blocks.

**Delivery.** The crossing mass arrives over $[720, 768)$ — the next stage's first two
daily blocks. If that stage is chronological, the bucket delivers by the template
density $\rho = (\tfrac12, \tfrac12)$ over those blocks; if parallel, onto its single
row. **Key numbers: $L_{\text{arc}} = 1$, shift $b \to b+2$ for $b \le 27$,
$\chi_{28,1} = \chi_{29,1} = 1$, parallel $k_1 = 1/15$, $\rho = (\tfrac12, \tfrac12)$.**

#### Example (ii) — multi-stage delay: monthly stages, $t_v = 40$ days

**Setup.** Uniform monthly stages of 30 days, $h_t = 720\text{ h}$. Travel time
$t_v = 40\text{ days} = 960\text{ h}$. Same result in parallel and chronological.

**Stage-clock $B$ and the span arithmetic (explicit).** The release of stage $t$ is
uniform over $[0, 720)\text{ h}$ on the stage clock; delayed by $t_v$ it arrives over
$[960,\ 960 + 720) = [960,\ 1680)\text{ h}$. Overlay the future stage intervals
(stage $t+d$ = $[720d,\ 720(d{+}1))$):

- stage $t+1$ = $[720, 1440)$: overlap with $[960,1680)$ is $[960,1440)$ → $480\text{ h}$
  → $k_1 = 480/720 = 2/3$.
- stage $t+2$ = $[1440, 2160)$: overlap is $[1440,1680)$ → $240\text{ h}$
  → $k_2 = 240/720 = 1/3$.
- stage $t$ (same stage, $[0,720)$): overlap with $[960,1680)$ is empty → $k_0 = 0$.

So $k_0 = 0,\ k_1 = 2/3,\ k_2 = 1/3$ (sum $= 1$, conserved). The arrival reaches stages
$t+1$ and $t+2$, i.e. **span $= 2$**, so
$L_{\text{arc}} = \lceil t_v/h_t\rceil = \lceil 960/720\rceil = \lceil 1.333\rceil = 2$
buckets — **identical in both modes** (it used only $t_v$ and $h_t$). Note $k_0 = 0$:
because $t_v > h_t$, **no** mass stays in the release stage, so there is no same-stage
routing term at all — the whole release is bucketed.

**Bucket dynamics (both modes, §2.2 transition).** Two scalar buckets per arc,
$b_1$ and $b_2$. At stage $t$ the release $D_u^t$ deposits $k_1 D_u^t = \tfrac23 D_u^t$
maturing in 1 stage and $k_2 D_u^t = \tfrac13 D_u^t$ maturing in 2 stages; the ring
shifts $b_d^{t+1} = b_{d+1}^t + k_d D_u^t$. At stage $t+1$, $b_1^{t+1}$ delivers
$\tfrac23 D_u^t$ into $j$'s water balance; at $t+2$, the remaining $\tfrac13 D_u^t$
delivers. **This is byte-identical in parallel and chronological mode** — the bucket is a
scalar, the deposit coefficients $k_1, k_2$ came from the stage clock, and the transition
is the §2 lifting with no block reference.

**Cross-stage delivery under chronological (sub-contract 3).** When $b_1^{t+1}$ delivers
$\tfrac23 D_u^t$ into stage $t+1$, and $t+1$ happens to be chronological with $K'$ blocks,
that scalar is spread over $t+1$'s early blocks by the **fixed template density** — e.g.
the overlap of $[960,1440)$ (restricted to stage $t+1$) with $t+1$'s blocks, precomputed,
`block_mode`-fixed. The origin-block detail of stage $t$ is **not** carried; only the
scalar $\tfrac23 D_u^t$ crosses. If $t+1$ is parallel, the scalar lands on its single row.
Either way the **state** is the same two scalars $b_1, b_2$. **Key numbers:
$k_0=0,\ k_1=2/3,\ k_2=1/3$, span $=2$, $L_{\text{arc}} = 2$, identical $B$ across modes.**

#### Example (iii) — sub-stage delay with fractional shares: 3 blocks of 240 h, $t_v = 250$ h

The case that decided the §2.5.5 fork: a delay slightly longer than one block, in a
coarse-block chronological stage.

**Setup.** Stage $t$ = 720 h, chronological with $K = 3$ blocks of 240 h
($H_b = 240b$); $t_v = 250\text{ h}$. Sub-stage ($250 < 720$), so
$L_{\text{arc}} = \lceil 250/720\rceil = 1$ bucket — both modes.

**Chronological routing and deposits.** Per-block arrival windows and their splits
(fractions of $\tau_b$):

| block | released over | arrives over  | $\kappa_{b\to B1}$ | $\kappa_{b\to B2}$ | $\chi_{b,1}$ (bucket) |
| ----- | ------------- | ------------- | ------------------ | ------------------ | --------------------- |
| B0    | $[0, 240)$    | $[250,\ 490)$ | $230/240$          | $10/240$           | $0$                   |
| B1    | $[240, 480)$  | $[490,\ 730)$ | $0$                | $230/240$          | $10/240$              |
| B2    | $[480, 720)$  | $[730,\ 970)$ | $0$                | $0$                | $1$                   |

Each row sums to $1$ (per-column conservation, §2.5.2). Duration-weighted deposit:
$\tfrac13\,(0 + 10/240 + 1) = 250/720 = t_v/h_t$ — the stage-level $k_1$, which is
exactly what **parallel** deposits from every release column
($k_0 = 470/720 \approx 65\%$ stays on the single row). Aggregation consistency
verified.

**What the downstream chained rows receive** (per unit of each block's release
$D^b$): B0's row — nothing from this arc; B1's row — $\tfrac{230}{240} D^0$; B2's
row — $\tfrac{10}{240} D^0 + \tfrac{230}{240} D^1$; the bucket —
$\tfrac{10}{240} D^1 + D^2$.

**Delivery.** The bucket's content arrives over $[720, 970)$: against the next
monthly stage's 240 h blocks, $\rho = (240/250,\ 10/250) = (96\%,\ 4\%)$ over its
first two blocks (or the single row if that stage is parallel).

**Why this case rejected the fold (§2.5.5).** Under fold, the bucket would not exist
and its content — $\tfrac{10}{240} D^1 + D^2$, $\approx 35\%$ of the stage release at
a uniform profile — would lump onto B2's row (chronological) or dissolve into the
same-stage aggregate (parallel). Under stage-uniform deposits in chronological (the
§2.5.2 forbidden alternative), B2's release — which arrives wholly next month — would
keep $k_0 \approx 65\%$ of itself in-stage. Block-resolved $\chi$ removes both errors.
**Key numbers: $L_{\text{arc}} = 1$, $\chi = (0,\ 10/240,\ 1)$,
$k_1 = 250/720 \approx 35\%$, $\rho = (96\%,\ 4\%)$.**

### 2.5.8 Summary

$B$ and the cut layout are a pure function of `(travel times, stage lengths)` on the
stage clock — every declared arc carries $L_{\text{arc}}(t) \ge 1$ buckets (§2.5.1, §2.5.5) —
and `block_mode` changes only coefficient placement and attribution: where same-stage
mass lands (single row vs. $\kappa$-routed rows) and how deposits are attributed
(stage-uniform $k_d$ vs. block-resolved $\chi_{b,d}$), never the state. **Answer to the
regime question, re-stated correctly:** within-stage routing (the chronological
refinement) is **exact and stateless**; cross-stage transport is a **stage-level scalar
bucket** whose count is `block_mode`-independent by construction. There are not two
regimes with different state footprints — there is **one** stage-clock skeleton
(buckets) with a `block_mode`-dependent **within-stage refinement** (routing + deposit
attribution) layered on top, and only the skeleton touches state.

---

## 3. Q2 — FCF and Benders cut impact under column-bound pinning

### 3.1 Column structure (mirror `storage`/`anticipated_state`)

Each bucket dimension contributes **one state dimension** and **two LP columns**,
exactly like storage (`storage` + `storage_in`):

- $b_d^{\text{out}}$ — outgoing column, the **cut target** (lives in the state region,
  analogous to `storage[0..N)` and `anticipated_state_out`).
- $b_d^{\text{in}}$ — incoming column, **pinned via `set_col_bounds`** to the previous
  stage's $b_d^{\text{out}}$ (analogous to `storage_in`); its reduced cost is the
  subgradient source.

No aux total-defluence column is required under the resolved k-weighted representation:
the bucket-**definition row** references the release columns directly —
$b_d^{\text{out}} - b_{d+1}^{\text{in}} - \sum_{\text{blk}} c_{d,\text{blk}}\,
\tau_{\text{blk}}(u_i + s_i)_{\text{blk}} = 0$ with $c_{d,\text{blk}} = k_d$ (parallel,
stage-uniform) or $\chi_{\text{blk},d}$ (chronological, block-resolved, §2.5.2). An aux
$D_i^t$ column (the `z_inflow` analogue) is compatible with the **parallel** deposit
only — a single aggregate cannot carry the chronological per-block weights — so it is
at most a parallel-mode implementation convenience, not part of the formulation.
One physical note (round-2 review): evaporation remains a function of the plant's
**own** average storage — in-transit bucket volume is river water, not reservoir
surface, and correctly carries no evaporation exposure.

The augmented layout, preserving the "pure function of dimensions" invariant (§1, fact 1),
inserts a bucket block at a **fixed** position (recommend immediately after
`inflow_lags`, before `anticipated_state`, so anticipated offsets shift by a constant
$B$ and stay dimension-pure):

```
… ⊕ [inflow_lags (N·L_par)] ⊕ [buckets_out (B)] ⊕ [anticipated_state (A·k_max)] ⊕ …
                                ⊕ [buckets_in (B, incoming)] ⊕ …
```

with `n_state = N·(1+L_par) + B + A·k_max`.

**Insertion is invariant-preserving in principle but NOT a code no-op** (verified
against the resolvers): `StateLayout::state_to_lp_incoming_column` resolves the
anticipated block through a **catch-all `else`** (any state index ≥ the lag end maps to
an anticipated column), so bucket indices inserted at that position would silently
resolve to anticipated columns — wrong reduced costs that still compile.
`state_to_lp_column`'s anticipated branch guard, `set_nonzero_mask`'s
storage→lag→anticipated loop order, and the `n_state` formula all hardcode the current
block sequence. Two further consumers complete the rewrite inventory (state-transfer
review): **`CutStateProjection::new`** — the walk the backward pass
(`extract_duals_from_view`) and cut scoring actually iterate — hardcodes the same
three-block sequence, and `StageStateConfig` carries no bucket flag; the bucket block
joins the walk **always-included** (full range in `n_state`, per-stage masking via
`nonzero_state_indices` — the anticipated discipline), never behind a per-stage flag.
And **`build_stage_entity_manifest`** (policy export) classifies every state slot by
layout region into the FlatBuffers per-slot manifest — bucket slots need their own
branch and a **new `EntityType` variant** in `policy.fbs` (a new byte-enum value is
FlatBuffers-forward-compatible; the writer and any `entity_type` switch gain the arm).
**The most load-bearing missing site (FCF review): `PatchBuffer` — the owner of the
pinning itself.** `fill_col_state_patches` does NOT iterate
`state_to_lp_incoming_column`; it hardcodes the storage/lag/anticipated offsets, with
`fill_anticipated_state_col_patches` anchored at `cat6_start = n·(1+l)` and
`state_col_patch_count()` returning `N(1+L)+A·K`. Inserting buckets shifts
`anticipated_state.start` by `B` while `cat6_start` stays put — anticipated patches
misalign AND bucket incoming columns are **never pinned**, making their reduced costs
(the subgradients) garbage: a wrong-bound bug that compiles. Fix: add the bucket block
to the patch fill and count, and prefer routing all three patch sites through
`state_to_lp_incoming_column` so the pinning contract has a **single owner**.
Adding the bucket block therefore rewrites **both column resolvers, the mask builder,
the projection walk, the manifest writer, the patch buffer, and `n_state` together**,
pinned by regression tests asserting (a) a bucket state index resolves to a bucket
column, (b) every bucket incoming column is actually pinned per stage, and (c) a
manifest round-trip covering the new variant. One shape note for the resolver
branches (insurance FCF review): the bucket branch in `state_to_lp_column` is an
**identity mapping** — outgoing bucket columns sit at their state-vector indices,
storage-like — NOT the `z_inflow`-style lag remap and NOT the anticipated
shift-remap; copying either neighboring branch compiles and mis-places every bucket
cut coefficient.

**Round-2 inventory additions (verified against the tree).** The manifest
**consumer** joins the writer: `load_boundary_cuts` validates per-slot
`(entity_type, entity_id, subindex)` identity (via `slot_identity`) and is where a
bucket `EntityType` is consumed; `build_terminal_entity_manifest` delegates to the
writer, so the new arm propagates. `StateLayout::new`'s signature (and setup's
`build_state_layout`) gains the `B` parameter threaded from the resolver. Sites
that **auto-scale** from `n_state` and need no rewrite — the trial-state MPI
allgatherv (`state_exchange.rs`), the backward accumulators
(`agg_coefficients` / `state_duals_buf`), and `StageGeometry::theta_col` (fully
layout-derived; the discount fold reads it, so the v0.9.1 hand-derived-theta bug
class does not recur) — are named so the plan does not mistake them for untouched:
they move with the layout by construction, and the audit obligation is for any
OTHER hand-derived offset past the insertion point (the `PatchBuffer` `cat6_start`
above is the known instance). Simulation output extraction
(`simulation/extraction.rs`, the per-entity state readers like
`compute_anticipated_decision_mw`) needs a new bucket reader only if D7 ships
bucket outputs — cross-referenced there.

**Two state-transfer contracts, stated explicitly (round-2 review; both are the
D15 bug class — per-opening vs per-visit confusion):**

1. **Bucket incoming columns are pinned once per stage-visit** via the state-col
   patch (`fill_col_state_patches`), never per opening — buckets are
   decision-driven state, unlike the per-opening NCS availability patch. The
   **lower-bound evaluation path** (`evaluate_lower_bound` → `lb_evaluate_stage_0`)
   and the **backward trial-point path** both consume the same `PatchBuffer` fill
   and inherit the single-owner fix, and each is named in the plan's test net.
2. The **forward and simulation state-assembly sites**
   (`training/forward/stage_solve.rs` and its `simulation/pipeline.rs` twin)
   plain-copy `unscaled_primal[..n_state]` and then overwrite the lag and
   anticipated blocks in place — the bucket block must sit in the **shift-gap**
   those overwrites do not touch, which the storage-convention identity mapping
   (outgoing bucket column = state index) guarantees; assert it.

The DCS scoring/basis path needs no bucket-specific code once the
`CutStateProjection` walk and `n_state` include the buckets
(`score_violated_candidates` iterates the projection;
`reconstruct_basis_uniform_basic` copies the widened column block).

### 3.2 The cut

The cut added to stage $t-1$, in cobre's stored form (raw subgradient $\beta$,
intercept $\bar\alpha$):

$$
\theta_{t-1} \ \ge\ \bar\alpha
+ \sum_{h} \beta^{v}_{h}\,v_h
+ \sum_{h,l} \beta^{\text{lag}}_{h,l}\,(\cdot)
+ \underbrace{\sum_{d} \beta^{b}_{d}\,b_d}_{\text{new}}
+ \sum_{\text{ant}} \beta^{\text{ant}}(\cdot),
$$

with the bucket subgradient extracted **identically** to storage
(`extract_duals_from_view`, iterating `CutStateProjection::n_state()` and reading the
incoming column):

$$
\beta^{b}_{d} \;=\; \frac{\text{rc\_scaled}\big[\text{col}(b_d^{\text{in}})\big]}
{\text{col\_scale}\big[\text{col}(b_d^{\text{in}})\big]} .
$$

The cut **row** lands the negated, scaled coefficient on the _outgoing_ bucket column
via `push_scaled_coefficient`:

$$
-\,\beta^{b}_{d}\,\text{col\_scale}\big[\text{col}(b_d^{\text{out}})\big]\,b_d
\ +\ \dots\ +\ \text{col\_scale}[\theta]\,\theta \ \ge\ \bar\alpha .
$$

Sign convention is automatic — in-transit water is a "good" (it raises future cheap
hydro), so $\partial Q/\partial b_d \le 0$ like storage; no special-casing. The bucket
dimensions simply join the `CutStateProjection` walk in **storage → lag → buckets →
anticipated** order, sized into the per-stage `CutPool` via
`FutureCostFunction::new_per_stage`, and rendered by `render_pairs`. Padding/inactive
bucket slots are **excluded from `nonzero_state_indices`** (see §6, §8) so they emit no
cut-row entry — the same discipline `set_nonzero_mask` applies to PAR padding and
anticipated padding (the explicit PAR(p)-A over-estimation guard).

### 3.3 Validity, monotonicity, convergence

- **Lower-bound validity.** Each bucket cut is a valid supporting hyperplane of the
  convex $V_t$ in the augmented state (§2.3). It underestimates $V_t$ everywhere, so
  the stage-1 lower bound remains valid.
- **Monotonicity.** The cut pool is append-only (`sddp.md` "Cut pool is append-only");
  the larger state does not change this — cuts are only added, so $\underline z^k$ is
  non-decreasing.
- **A.s. finite convergence.** The crucial point: **travel time adds deterministic
  linear dynamics driven by decisions, not new randomness.** The noise process and
  its stagewise-independence are untouched (buckets are functions of $D_i$, a
  decision). Boundedness of the augmented state comes from **conservation on the
  reachable set, NOT from column bounds**: spillage columns are deliberately
  unbounded (`fill_spillage_columns` sets upper = $+\infty$ — the same freedom that
  gives relatively complete recourse, since a plant can always pass arriving water
  on), so defluence is not capacity-capped. But water is conserved: any stage's
  total defluence is bounded by total system water (initial storage + the
  finite-support inflows realized so far), and $\sum_d k_d = 1$ gives
  $\sum_d b_d^t \le \sum_{m=1}^{L} D^{t-m}$ — an accumulation over at most $L$
  stages of conservation-bounded releases. The reachable state set is therefore
  compact and convex, and the standard SDDP convergence hypotheses (compact convex
  reachable states, relatively complete recourse via free spillage,
  stagewise-independent finite noise) hold. Convergence is preserved — **conditional
  on one recourse corner (FCF review)**: free spillage is the relief valve, but the
  `sddp.md` commissioning contracts freeze spillage `[0,0]` in `PreFilling` (and
  turbine/diversion in `PreFilling` **and** `Filling`), so an arc delivering a
  pinned, unavoidable arrival ($b_1$) into a plant inside those windows can hit a
  balance with every valve frozen — an infeasible stage LP, and cobre generates no
  feasibility cuts: a relatively-complete-recourse violation, not merely a run
  failure. The plan must adopt one of: **validation** (reject a declared arc whose
  downstream plant is in `PreFilling`/`Filling` — or **before its entry window**,
  where the plant is absent from the LP entirely and the arrival has no balance row
  at all — during any of the arc's arrival windows) or **routing** (send the
  delivery through the incremental-inflow short-circuit those phases already use,
  landing it on the first non-frozen downstream — noting, round-2 review, that the
  shipped short-circuit (`fill_prefilling_shortcircuit`) is
  **same-stage/instantaneous**: it re-routes current-stage terms onto the
  substitute row and cannot carry a delayed bucket delivery as-is, so the routing
  arm requires new delivery-row re-targeting machinery, which tilts the default
  toward validation). Until one lands, the recourse
  claim is conditional. (Surfaced in the unification memo's validation inventory.)

### 3.4 Practical convergence cost of a larger state vector

The honest cost is **cut dimensionality**: $V_t$ is now PWL convex in
$\mathbb R^{n\_state + B}$, and SDDP's cut count to reach a target gap grows with state
dimension (polynomially in practice for structured hydro, worst-case worse).
Mitigations and facts:

- **cobre already pays a multi-dimensional state cost** — `inflow_lags` adds $N\cdot
  L_{par}$ dimensions for PAR(p) (default order up to 6) and converges; buckets are the
  same kind of cost, incremental not novel.
- **Aggregation (§7) is the primary lever**: aggregating buckets per _downstream
  plant_ instead of per _arc_ reduces $B$ from $\sum_{\text{arc}}L$ to $\sum_j L_j$
  with no loss of information (a sufficient statistic).
- **GEMM cut selection** (`gemm.rs::gemm_block`) scales $O(K\cdot M\cdot d)$ with
  $d=n\_state$; larger $d$ is linear extra work, and cut selection becomes _more_
  valuable (more near-parallel cuts to prune). The existing GEMM kernel absorbs the
  larger $d$ with no structural change (dimensions flow from `n_state`).
- **Warm-start basis** grows by $2B$ columns and (raw-lagged) $0$ or (k-weighted) $B$
  rows; the resolve-loop warm-start still applies (one solver instance per worker,
  basis reused — see the solver-integration discipline).

---

## 4. Q3 — Initial condition: "water in transit at study start"

### 4.1 It is genuinely required

The downstream balances at study stages $t=0,\dots,L-1$ reference buckets
$b_1^0,\dots,b_L^0$ that were filled by **pre-study** upstream releases
$D_i^{-1},\dots,D_i^{-L}$:

$$
b_d^{0} \;=\; \sum_{m\ge1} k_{d-1+m}\,D_i^{-m}
\qquad\Longleftrightarrow\qquad
A_j^{0} = k_0 D_i^{0} + \underbrace{\textstyle\sum_{d\ge1}k_d D_i^{-d}}_{b_1^0}.
$$

The $D_i^{-m}$ are **not decisions** of the optimization — they happened before the
horizon. Omitting them sets $b^0=0$, asserting that _no_ upstream water arrives during
the first $L$ stages: a conservation error that under-credits downstream inflow and
biases the early policy (over-conservative downstream, wrong marginal water values).
This is structurally identical to — and as required as — `ic.storage` (initial
reservoir levels), `ic.past_inflows` (PAR lag seed), and
`ic.past_anticipated_commitments` (anticipated ring-buffer seed), all consumed in
`setup/mod.rs::build_initial_state`.

### 4.2 How to seed

Mirror `build_initial_state`. Add an `InitialConditions` field (e.g. `past_defluences`,
per source plant or per arc). At setup compute the incoming bucket state and inject it
as the **stage-0 incoming bucket column bounds** (`set_col_bounds`), exactly as initial
storage pins `storage_in`. Under the resolved k-weighted representation the seed is the
unrolled $b_d^0=\sum_{m\ge1}k_{d-1+m}D_i^{-m}$ (a deterministic setup precompute,
single-threaded, fixed lag order); the raw-lagged verbatim seed applies only to the
rejected §7 alternative. The volume-accumulating precedent already exists:
`stochastic/lag_transition.rs::compute_recent_observation_seed` accumulates
`value_m3s · observation_hours` and a coverage `weight_seed`.

**Input granularity.** `past_defluences` must cover the window
$[\text{start}_0 - t_v^{\max},\ \text{start}_0)$ per arc, supplied **per pre-study
calendar period** following the same period convention `past_inflows` uses (the AR-lag
periods `build_initial_state` already consumes) — the seed then applies the same
uniform-within-period density and overlap arithmetic as the in-study $k_d$
(§2.5.4), so pre-study and in-study water obey one discretization. A $t_v^{\max}$
that reaches further back than the supplied history is a validation error under the
§4.3 REQUIRE option (or triggers the derived fallback with its logged caveat).

**Upstream commissioning (round-2 review).** The seed and the deposits must
respect the UPSTREAM plant's commissioning state — the memos elsewhere cover only
the downstream side (unification memo validation rows 11–12):

- an upstream that **enters mid-horizon** ($t_e > 0$) seeds **zero** buckets at
  entry — conservation-FORCED, not a fork (round-3 review): during stages
  $0..t_e$ the reach's water already reaches downstream **same-stage** via the
  pre-entry short-circuit, so seeding the arc's buckets from a `past_inflows`
  proxy at $t_e$ would deliver that water a second time (water creation). The
  in-study ring self-seeds to zero naturally (no deposits before entry);
- a plant **active from stage 0** whose pre-study operation is unrecorded is the
  ordinary §4.3 fork (decided with D5): no pre-study short-circuit ever
  delivered the pre-study flow, so the derived `past_inflows` proxy is clean
  there and is the operative default when `past_defluences` is absent;
- an upstream **in `Filling`** has turbine and diversion frozen but spillage
  **free** (the D40 relief valve), and spillage is part of the $(u+s)$ deposit — a
  Filling upstream therefore **legitimately deposits spill** into the bucket;
  never freeze the deposit rows wholesale during Filling;
- an upstream that **exits mid-horizon** stops releasing and its buckets drain
  through the ring shift over the following $L$ stages — no special handling, but
  the plan asserts it.

### 4.3 Derive vs require — a genuine fork

| Option                                         | What                                                                                               | Trade-off                                                                                                                                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Require `past_defluences`** (registro VI) | User supplies pre-study upstream releases                                                          | Most correct; Cost: one new input field. **Recommended.**                                                                                                                                   |
| **B. Derive from `past_inflows`**              | Proxy $D_i^{-m}\approx$ pre-study natural inflow (pass-through assumption), reusing existing input | No new required input; correct only if upstream operated ≈ run-of-river pre-study; wrong for plants that were storing/drawing down. **Recommended fallback default**, with a logged caveat. |
| **C. Steady-state**                            | $D_i^{-m}=\bar D_i$ (long-run average)                                                             | Cheap; biased if the pre-study period was wet/dry.                                                                                                                                          |
| **D. Zero-seed**                               | $b^0=0$                                                                                            | Simplest; systematically under-credits the first $L$ stages. Last resort, with a warning.                                                                                                   |

**Correctness cost of deriving.** The initial-condition error is a **transient that
decays after $L$ stages** (once buckets fill with in-study decisions it washes out).
So for long-horizon planning ($T\gg L$, steady-state policy of interest) a derived or
zero seed perturbs only the first $L$ stages and is tolerable. But the _primary
use-case for travel time is short/medium-term scheduling_ (DESSEM-like), where the
first $L$ stages **are** the stages of interest — there, requiring the VI record
matters. **Recommendation: implement A as the contract, with B as a documented
fallback default when no VI data is present; never silently zero-seed.** (Per the
anti-simplification rule this fork is surfaced, not silently resolved.)

---

## 5. Q4 — End-of-horizon coupling

Buckets $b_1^T,\dots,b_L^T$ outgoing at the last stage hold volume that would arrive at
$T+1,\dots,T+L$ — **beyond the window**. credits residual transit volume to
downstream storage for FCF coupling: $V_{\text{eff}} = V + V_{\text{in\_transit}}$.

**Does the augmented state value the residual buckets automatically?** It depends on
the terminal value function:

- **If a genuine terminal FCF exists** ($V_{T+1}\neq0$ — a terminal storage valuation,
  a long-term coupling, or a cyclic/infinite-horizon policy graph; cf.
  `horizon_mode.rs`): then provided $V_{T+1}$ is defined over the **full outgoing
  augmented state** (including buckets) with the correct downstream water value
  ($\partial V_{T+1}/\partial b_d < 0$), the residual buckets **are** valued and
  end-of-horizon upstream release is correctly credited. SDDP generates cuts on the
  bucket dimensions of $V_{T+1}$ naturally. In a **cyclic** horizon the issue
  dissolves: post-$T$ arrivals land in the next cycle's early stages, so the buckets
  wrap around with no special treatment.

- **If $V_{T+1}=0$ (finite-horizon default):** the outgoing buckets at $T$ have **zero
  value**. The optimizer sees no benefit to upstream release whose water lands in a
  worthless bucket → it **under-releases upstream in the last $L$ stages**. This is the
  exact end-effect distortion `V_eff` fixes, and the augmented state does **not** fix
  it on its own — it faithfully represents that the water arrives after $T$, and if
  nothing values post-$T$ arrivals the buckets are correctly-but-undesirably worthless.

**Correct terminal treatment (required for finite horizon).** Inject value into the
residual buckets, mirroring $V_{\text{eff}}$. Options:

1. **Terminal bucket credit (recommended).** At stage $T$ only, credit all in-transit
   volume to the relevant downstream storage:
   $V_{\text{eff},j} = v_j^{T} + \sum_{\text{arcs}\to j}\sum_d b_d$. Concretely, at $T$
   route the bucket deposits and the incoming buckets into $v_j$ (or into the terminal
   storage argument of whatever terminal value exists) rather than into worthless
   buckets, so the in-transit water inherits the downstream storage's marginal value.
2. **Terminal FCF over the augmented state.** If cobre grows a terminal/cyclic value,
   define it over buckets directly with the downstream energy value. More general;
   requires that value to know the energy worth of in-transit water.
3. **Horizon padding** ($+L$ drain stages). Cleanest theoretically, but adds compute
   and changes horizon semantics — overkill.

**Deferral honesty (FCF review):** `HorizonMode::Finite` with $V_{T+1}=0$ is the only
implemented mode — the "genuine terminal FCF" branch above is hypothetical — and the
deferred drop lands on the **near-tail stages that ARE the primary short/medium-term
use case**, not on a far-horizon nicety. The deferral stays LB-valid for the
model-as-formulated and `V_eff` is a later extension (buckets are already state), but
the plan documents it as a target-stage imprecision.

**Recommendation: option 1** for the finite-horizon mode; for a cyclic horizon, ensure
the bucket state is part of the graph's recurring state so it wraps naturally.

---

## 6. Q5 — Determinism and contract risks

1. **Declaration-order invariance / bit reproducibility (hard rule).** New bucket
   columns **must sort canonically** by a key derived from sorted entity IDs — e.g.
   `(downstream_id, upstream_id, lag)` (per-arc) or `(downstream_id, lag)`
   (aggregated) — **never** arc-discovery / HashMap-iteration order. Risk: emitting
   buckets in cascade-traversal order makes the layout input-order-dependent and breaks
   the declaration-order-invariance rule. The block must stay a pure function of the
   (sorted) topology and dimensions to keep the stage-invariant offset property (§1, fact 1).

2. **Append-only cut pool + slot-identity basis reconstruction.** `reconstruct_basis`
   matches cut **rows** to pool **slots** by slot identity — independent of state
   dimension — so it is unaffected by adding bucket dims, provided `n_state` and the
   per-stage `CutStateProjection`/`CutPool` sizing include the buckets consistently.
   Stage-varying active-bucket sets (§8) use the **`anticipated_state` discipline
   ONLY — always included at full global `B` in every pool, per-stage reachability
   mask — NEVER the `inflow_lags:false` dimension-reduction route** (FCF review):
   that route shrinks a pool's `state_dimension`, and a reduced coefficient vector
   zipped against the GLOBAL trial state `x̂` in the intercept computation
   ($\bar\alpha = Q(\hat x) - \sum_j \beta_j \hat x_j$) misaligns. Pool dimension is
   global; masking is the only per-stage variation.

3. **`col_scale` of bucket columns.** Buckets are volumes (hm³); reuse storage-like
   scaling. The extraction divides by `col_scale[col(b_d^in)]`; the cut row multiplies
   by `col_scale[col(b_d^out)]` — the in/out columns may carry different scales (as
   `storage_in` vs `storage` already do) and the cut math stays in original units.
   `col_scale` must be **sized to include the bucket columns**; an undersized
   `col_scale` would misindex `col_scale[theta_col]` in `push_cut_row`.

4. **GEMM dimensions.** `gemm_block` asserts `coef.len()==k_rows·d` and
   `state_block.len()==m_len·d` with `d=n_state`; consistent once `n_state` includes
   buckets. GEMM determinism is preserved (single-threaded `matrixmultiply`, fixed
   cache-blocked algorithm). Only effect: larger $d$.

5. **MPI basis payload.** `CapturedBasis::state_at_capture` has length `n_state`; the
   wire format (`BASIS_BROADCAST_WIRE_VERSION = 1`) encodes the length explicitly, so a
   larger `n_state` round-trips **without a version bump**. But a **cross-version policy
   load** (an old policy/cut set without buckets, loaded by bucket-aware code) has a
   mismatched `state_dimension` and **must be rejected**. Caveat (corrected, round-2
   review): the `FutureCostFunction::from_deserialized` / `CutPool::from_deserialized`
   checks validate only **internal** consistency (cross-stage agreement, coefficient
   length vs the **recorded** `state_dimension`) — but a dedicated
   policy-compatibility layer already performs the cross-version rejection:
   `validate_policy_compatibility` rejects `metadata.state_dimension != current` on
   the CLI and Python warm-start paths, and `load_boundary_cuts` checks the dimension
   unconditionally plus per-slot `(entity_type, entity_id, subindex)` manifest
   identity. Two residuals for the plan: the warm-start check is **opt-in**
   (`config.policy.validate_compatibility`) — a bucket study must force it on, or the
   failure mode reverts to a coefficient-length panic when a user disables it; and
   the bucket dims must land in `state_dimension` (nearly automatic once `n_state`
   includes them) so both layers catch a stale policy. The `cut/wire.rs` record
   carries the coefficient length, so within a version it round-trips.

6. **Bit-reproducible reductions.** The bucket subgradient components join the
   **existing** expectation/CVaR reduction over openings — no new reduction class, but
   they must use the **same fixed-order/compensated** reduction cobre already applies to
   the storage subgradients (verify it is order-fixed; see the numerical-reproducibility
   discipline). The initial-condition unroll $\sum_m k_{d-1+m}D^{-m}$ and the
   per-downstream injection $\sum_i k_{d,i}D_i^t$ are computed at **setup / per-stage
   template build on one rank** in a fixed canonical (sorted) order — keep them out of
   any rank-count-dependent parallel reduction.

### 6.1 New `sddp.md` contracts to add

- **In-transit bucket dynamics & sign.** "The in-transit bucket ring evolves
  Markov-1 via the in-LP definition rows
  $b_d^{\text{out}} = b_{d+1}^{\text{in}} + k_d D_i$ (k-weighted volume form — the
  resolved representation). Incoming buckets are pinned via column bounds; the subgradient
  is $\text{rc}/\text{col\_scale}$ on the **incoming** bucket column and the cut row
  multiplies by `col_scale` on the **outgoing** bucket column — divided on extract,
  multiplied on render, identical to storage. Pin to a named regression case."
- **k-factor conservation.** "The propagation factors satisfy $\sum_d k_d = 1$ per
  source per stage (volume conservation); a `debug_assert` enforces it. A drift from 1
  silently creates or destroys water."
- **Canonical bucket ordering.** "Bucket columns sort by the downstream hydro's
  **canonical entity index** (the `(operational_start_date, id)` order every state
  block follows) then lag — never raw user id, never traversal order —
  declaration-order invariance."
- **Terminal credit.** "In the finite-horizon mode the last stage credits residual
  buckets to downstream storage ($V_{\text{eff}}$); without it end-of-horizon upstream
  release is under-valued (pin to a regression case)."

---

## 7. Q6 — Representation recommendation

> **Decision record.** This question is RESOLVED to **k-weighted volume buckets
> aggregated per downstream plant** (§8.4 lock #2, §0.1 note): the chronological
> block-resolved deposit is a weighted volume the raw-lagged form cannot carry.
> §7.1–§7.2 are retained as the analysis that fed the decision; read the
> raw-lagged arguments (including "wins on both for a linear cascade") as the
> record of the rejected alternative, superseded by the exact-overlap +
> chronological-deposit resolution.

Two orthogonal axes.

### 7.1 Axis A — raw-lagged-defluence vs k-weighted volume buckets

**Raw-lagged-defluence** ($D^{\text{lag}}_{d}=D_i^{t-d}$, k applied on the downstream
balance):

- **Structurally identical to `inflow_lags`** — a pure-shift ring buffer
  ($D^{\text{lag}}_{d}\!\leftarrow\!D^{\text{lag}}_{d-1}$, freshest $\leftarrow$ current
  defluence). Maximal reuse of `shift_lag_state`, the lag-major layout, the
  `state_to_lp_column` lag mapping, and `StageLagTransition` weighting.
- k-factors live as **coefficients on the downstream balance row**, stage-dependent,
  rebuilt in the template per stage exactly like $\tau_{\text{blk}}$ and $\zeta$ already
  are.
- Initial condition is **verbatim** (raw past defluences, like `past_inflows`).
- Needs **1 aux total-defluence column + 1 definition row per source plant** (the
  freshest-bucket materialization, the `z_inflow` analogue); **no per-bucket transition
  rows**.

**k-weighted volume buckets** ($b_d$ = volume maturing in $d$ stages):

- $B$ outgoing + $B$ incoming columns + $B$ transition rows; k-factors live in the
  transition rows.
- Closest code analogue is `anticipated_state` (a k-injected ring buffer with a deposit
  at slot $K_i-1$), _not_ `inflow_lags`.
- Initial condition is the unrolled $\sum_m k_{d-1+m}D^{-m}$.

State (cut) dimension is **$L$ per arc either way**; the difference is LP rows/columns
and code reuse. Raw-lagged wins on both **for a linear cascade**.

### 7.2 Axis B — per-arc vs aggregated-per-downstream-plant

The future cost depends only on the **arrival schedule at $j$**, not on which upstream
contributed. Define the aggregate
$B_d^{j,t} = \sum_{i\in\text{up}(j)} b_{d,i}^t = \sum_{i\in\text{up}(j)}\sum_{m\ge1} k_{d-1+m,i}\,D_i^{t-m}$ — the §2.2
per-arc bucket summed over arcs (an earlier draft's single-term
$\sum_i k_{d,i}D_i^{t-d}$ is inconsistent with the transition below; corrected,
round-2 review). It evolves Markov-1:

$$
B_d^{j,t+1} = B_{d+1}^{j,t} + \sum_{i\in\text{up}(j)} k_{d,i}\,D_i^{t}.
$$

So aggregation is a **sufficient statistic** — it loses **no** information and reduces
the state from $\sum_{\text{arc}}L$ to $\sum_j L_j$. The saving is real at
**confluences** (a plant $j$ with several travel-time tributaries); for a strictly
linear cascade (one upstream per plant) per-arc and per-plant coincide.

The tension: aggregation **forces the k-weighted form** — the injection
$\sum_i k_{d,i}D_i^t$ is a weighted sum, so raw per-arc defluences cannot be kept. So:

- **Raw-lagged** ⇒ naturally **per source plant**, max reuse, $B=\sum_{\text{src}}L$.
- **k-weighted aggregate** ⇒ **per downstream plant**, minimal $B=\sum_j L_j$, reuse the
  `anticipated_state` ring buffer instead of `inflow_lags`.

### 7.3 Recommendation

| Topology                                                | Recommendation                                                 | Why                                                                                                       |
| ------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| Linear cascade (no travel-time confluence)              | **Raw-lagged-defluence, per source plant**                     | per-arc = aggregate, so take the maximal-reuse form (`inflow_lags` machinery, verbatim seeding)           |
| Confluent (multiple travel-time upstreams into one $j$) | **k-weighted volume buckets, aggregated per downstream plant** | minimizes cut dimensionality (sufficient statistic); reuse the `anticipated_state` k-injected ring buffer |

A robust single design that is always correct and always minimal-dimension is
**aggregate-per-downstream-plant k-weighted buckets**, modeled on `anticipated_state`.
Because cut dimensionality is the dominant SDDP cost (§3.4), this was already the
primary recommendation; the exact-overlap resolution (§2.5.5) **settles it** — the
chronological deposit is the block-weighted volume $\sum_b \chi_{b,d}\,\tau_b\,D^b$
(§2.5.2), which the raw-lagged form (a verbatim copy of one lagged defluence) cannot
carry. The table above stays as the record of the rejected max-reuse alternative
(§8.4 lock #2).

---

## 8. Q7 — Variable stage length (the stage-clock bucket skeleton)

> **Scope (post-revision).** This section addresses the **stage-clock** layer of
> §2.5 — the cross-stage buckets and their variable-stage-length depth. Its k-factors
> are **stage-clock** overlaps and its buckets are the state-carrying skeleton. The
> **within-stage routing** layer (the `block_mode`-dependent placement of same-stage
> mass) uses the **block-clock** overlap factors $\kappa_{b\to b'}$ of §2.5.2 and adds
> no state; do not conflate the two. Both use the same overlap arithmetic on nested
> partitions (stages ⊃ blocks) built from one shared arrival density (§2.5.4), but
> only the stage-clock layer sizes the state vector — so the variable-stage-length
> concerns below (ring-buffer depth, masking, `n_state` sizing) belong to the bucket
> skeleton, which is `block_mode`-independent by §2.5's sub-contract 1.

Travel time is a scalar in **hours**; stages are days/weeks/months. When the delay
$t_v$ **exceeds a stage's own length** $h_t$, the arrival splits across multiple
downstream **stages**, and the stage-level k-factors are therefore **stage-dependent**.
(When $t_v \lesssim h_t$ in a chronological stage, the same-stage mass additionally
splits across **blocks** of the stage — §2.5; the boundary-crossing mass still carries
the arc's bucket, per the resolved exact-overlap convention.)

### 8.1 The k-factors are calendar-fraction overlaps — mirror `StageLagTransition`

A release $D_i^t$ spread over stage $t$ (hours $h_t$) arrives downstream over the window
$[\text{start}_t + t_v,\ \text{end}_t + t_v)$. The fraction reaching downstream stage
$t+m$ is the **overlap of that arrival window with stage $t+m$'s calendar interval**,
normalized to sum to 1. These overlap fractions **are** $k_d^t$ — this is the
authoritative depth/weight definition §2.5.1 references. The reusable precedent is the
**interval-overlap PATTERN**, not the existing functions: `days_in_period` /
`month_total_hours` are **day-granular** (`days · 24`), while travel time is
intrinsically **hour-scale** — sub-day $t_v$ (a 6 h delay), and every block-clock
window ($\kappa/\chi$ with $H_b + t_v$ endpoints like $[250, 490)$ h) land off day
boundaries; the day-aligned worked examples ($t_v = 15$ d against weekly stages) fit
only coincidentally. And `compute_monthly_transition` computes exactly **two**
adjacent-period weights (accumulate + spillover), which cannot express a ≥3-way split
($k_2 = 6/7, k_3 = 1/7$). Two further disqualifiers (state-transfer review): the
existing transition machinery is **monthly-cycle-gated** (`Weekly`/`Custom` season
cycles take a no-op path — exactly the calendars the DECOMP validation targets), and
`Stage::start_date`/`end_date` are day-resolution `NaiveDate`, so a sub-day $t_v$
cannot anchor on dates at all. The resolver therefore needs a **new hour-resolution,
multi-slot overlap primitive** that builds its stage clock from **cumulative
`duration_hours`** (cycle-agnostic, hour-exact), mirroring only the overlap _pattern_;
reusable code from the existing path is near-nil. Mirroring the
calendar-fraction logic produces:

- **Volume-conserving k-factors**: $\sum_d k_d^t = 1$ because the overlap fractions
  partition the arrival window — a `debug_assert`, like the existing transitions.
  Conservation is the load-bearing property (a new `sddp.md` contract, §6.1).
- **Stage-dependent depth**: long stages ($h_t \ge t_v$) need shallow buckets
  ($L=1$, possibly with $k_0<1$); short stages (weekly, $t_v\sim$ monthly) need deep
  buckets. The ring buffer is sized to the **maximum** depth over stages (the $k_{max}$
  analogue), with per-stage active depth $\le$ max.

`StageLagTransition` already proves cobre handles "a stage shorter than the lag period"
_and_ "a coarser ring buffer maintained in parallel with stage-dependent finalization"
(the multi-resolution downstream-accumulation path). The travel-time factor precompute
should be a **sibling** precompute (a different cadence — arrival windows, not AR lag
periods), following the `days_in_period`/`month_total_hours`/`find_season_year_monthly`
**pattern** at hour resolution (the existing day-granular functions are insufficient,
§8.1), **not** overloaded onto `StageLagTransition`.

### 8.2 Stage-dependent state dimension — a solved pattern

Buckets that don't exist (or are inactive) at some stages is the **same** situation
cobre already handles three ways:

1. `inflow_lags:false` stages — fewer cut-state dims, sized via
   `FutureCostFunction::new_per_stage` / `CutStateProjection` — a **precedent
   buckets must NOT copy** (dimension reduction misaligns coefficients against the
   global trial state; see §6 item 2 — buckets are always-included + masked).
2. The anticipated horizon gate `stage_idx + K_i < n_stages`.
3. Commissioning-window per-stage column omission (NCS/pumping/etc.).

Correctness requirements for buckets:

- **Inactive bucket slots must be excluded from `nonzero_state_indices`** so they emit
  no cut-row entry. Including a structurally-zero deep bucket in a cut **over-estimates**
  it — the exact PAR(p)-A / anticipated-padding bug class that `set_nonzero_mask`'s docs
  call out. This is the single most important correctness pitfall here.
- **Global `n_state` is sized to the maximum bucket depth** (the broadcast payload and
  cut storage need a fixed global dimension), with **per-stage masking** of inactive
  slots — exactly `anticipated_state` ($k_{max}$ global, per-plant $K_i$ active, padding
  masked).
- **Buckets maturing beyond $T$** are handled by the terminal credit (§5), not silently
  dropped.

So variable stage length is covered by mirroring two already-shipped patterns:
`StageLagTransition` (calendar-fraction k-factors) + `anticipated_state` ($k_{max}$
global depth / per-stage active / padding-masked). No new SDDP correctness concern
arises that cobre has not already solved.

### 8.3 Risk-profile contrast: two axes, not one

The chronological block feature is **`block_mode`-independent AND state-preserving**:
its interior per-block storage lives in the CONTROL region, `n_state` is untouched, cuts
are portable across block modes, and `K = 1` is byte-identical to parallel
(`book/src/guide/block-modes.md`; §1 above). Travel time keeps the first property and
gives up the second — and it is important not to conflate the two axes:

- **`block_mode`-independence is PRESERVED (§2.5).** The bucket count `B` is a stage-clock
  quantity; the same study has the **same** state vector whether its stages are parallel
  or chronological. The within-stage routing layer only moves same-stage coefficients
  between rows of one LP — no state footprint. So a policy is still cross-**mode**
  loadable: train parallel, simulate chronological, byte-compatible cut vector.
- **State-preservation is GIVEN UP whenever `B > 0`.** A travel-time policy with any
  cross-stage bucket has a **larger** state vector ($n\_state + B$). It is therefore
  **not loadable into a non-travel-time policy**, and a non-bucket policy is not loadable
  into bucket-aware code: the recorded `state_dimension` differs and the
  **policy-compatibility layer** rejects the mismatch — `validate_policy_compatibility`
  (opt-in; force it on for bucket studies) plus `load_boundary_cuts`' unconditional
  check; the `from_deserialized` / `new_with_warm_start` internal checks alone cannot
  catch it (§6 item 5, round-3 review). This is cross-**version** (with-vs-without travel
  time), **not** cross-**mode** (parallel-vs-chronological) — the latter is preserved.
  Under the resolved exact-overlap convention (§2.5.5) every declared arc carries at
  least one bucket, so **any** study that uses travel time is state-expanding and pays
  the cross-version break; a study with no declared arc has `B = 0` and gives up
  nothing.
- **Cross-CALENDAR portability is also given up (round-2 review).** Sub-contract 1
  makes $B$ a function of the stage **lengths**, so the same declared arcs on a
  different stage calendar generally yield a different $B$, hence a different
  `n_state` — a policy trained on one calendar is rejected on another by the same
  `state_dimension` check (mechanically safe, never silent). The portability
  taxonomy is therefore three axes: cross-**mode** preserved, cross-**version** and
  cross-**calendar** broken-by-rejection.

**Pre-existing risk the plan must track (not a blocker).** The added cross-stage cut
dimensions amplify a known latent CLP-backend issue: cut-row demotion in
`enforce_basic_count_invariant` writes a HiGHS-convention basis status code that the
CLP backend misreads (CLP reads code `0` as "free" where HiGHS means "at-lower"), so
demoted cut rows are installed free — a silently degraded warm-start (no crash,
correct optimum, CLP self-repairs). It is **off-by-default CLP only** and is already
amplified by chronological mode's larger cut counts; travel-time buckets add more cut
rows and amplify it further. The fix is to translate status codes at the CLP boundary
rather than pass raw `i32` through. Flag it in the implementation plan's risk register;
it does not block the formulation.

---

## 8.4 Re-validation of the §0.1 locks against the state principle

The §0.1 locks were signed off before chronological block mode landed and against a
single-water-row stage LP. The block-mode-independent state principle (§2.5) plus the
resolved exact-overlap convention (§2.5.5 — every declared arc carries a bucket) leave
three locks standing and settle the two that the principle had put in question.

| Lock                                              | Disposition                             | Why                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **#1 Initial condition** (`past_defluences`)      | **STANDS, broadened**                   | The IC seeds the stage-0 **incoming buckets** — identical in both modes. Under exact overlap every declared arc has a bucket, so the seed applies to **every** travel-time study (a sub-stage arc's stage-0 bucket = the crossing mass of pre-study releases). No change to the requirement itself.                                                                                                                                                    |
| **#2 Representation** (raw-lagged per source)     | **RESOLVED: k-weighted volume buckets** | The chronological deposit is a per-block-weighted volume ($\sum_b \chi_{b,d}\,\tau_b\,D^b$, §2.5.2) — intrinsically a weighted sum over block release columns, not a raw copy of a lagged defluence. The raw-lagged form cannot carry block-resolved sending, so the reopened lock resolves to the k-weighted volume-bucket representation (the §7.3 primary recommendation), per downstream plant.                                                    |
| **#3 End-of-horizon** (defer `V_eff`)             | **STANDS, broadened**                   | Under exact overlap **every declared arc** — sub-stage included — carries $b_1$, so every arc may leave an unvalued residual bucket at stage $T$ (a sub-stage arc's $b_1^T$ holds $t_v/h_T$ of the final stage's release). The deferred-`V_eff` imprecision therefore applies to all declared arcs, not only multi-stage ones; its magnitude per arc is bounded by the crossing fraction. Deferring remains the accepted, documented choice.           |
| **#4 Per-stage activation** (`k⁰≈1` degeneration) | **AMENDED**                             | The lock's "long stages degenerate to $k^0\approx1$ (no outgoing buckets)" clause is dead — under exact overlap every declared arc has $L_{\text{arc}}(t) \ge 1$ at every stage (§2.5.1). What survives is per-stage **depth variation** (variable stage lengths: the window-overlap depth depends on the DOWNSTREAM stage lengths, §2.5.1) under the global-max + `nonzero_state_indices` **reachability** masking discipline (§8.2, sub-contract 1). |
| **#5 Modeled→unmodeled boundary**                 | **STANDS**                              | Concerns residual in-transit water maturing within a downstream stage — a cross-stage conservation property carried by the scalar buckets (§2.5 sub-contract 3), untouched by within-stage routing.                                                                                                                                                                                                                                                    |

**Net:** locks #1, #3, #5 stand and now bind for every declared arc; lock #2 is
resolved to k-weighted volume buckets; lock #4 is amended to depth-variation +
masking. No lock remains open pending the §2.5.5 convention — that fork is resolved
(exact overlap, both modes).

## 8.5 Scope / shipping recommendation

**A and B are not two mechanisms to sequence — they are two LAYERS of ONE mechanism.**
The single mechanism is: build the arrival curve for each arc, split it at the stage
boundary. What lands **inside** the release stage is the **within-stage routing layer**
(A) — a `block_mode`-dependent, stateless coefficient placement (parallel: single row;
chronological: block-routed). What crosses **out** of the release stage is the
**stage-clock bucket layer** (B) — the state-carrying skeleton, `block_mode`-independent
by §2.5's sub-contract 1. **One preprocessing (the shared arrival density of §2.5.4)
produces both layers**; the block factors $\kappa$ and the stage factors $k_d$ are the
same density read against nested partitions. So the question is not "which regime is in
scope" but "in which **order** do we ship the two layers of the one mechanism," which is
a **shipping** choice, not a mechanism split.

With the §2.5.5 fork resolved to exact overlap, a routing-only first landing is no
longer a complete feature (there is no zero-bucket convention to ship it under). Two
viable orders remain:

| Shipping order               | What ships first                                                                                                                                                                                                                    | Gains                                                                                                                                                                                                                                        | Costs / caveats                                                                                                                                                                                     |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Core first (stage clock)** | The full bucket skeleton — the §2.5.1 window-overlap depths, IC seeding, ring transition, cut/broadcast sizing — with **stage-uniform attribution in both modes** ($k_d$ deposits on every release column; lumped/uniform delivery) | The mode-independent state machinery lands first and never changes again; **parallel-mode studies are exact from day one** (stage-uniform IS the parallel attribution); the block refinement that follows touches only template coefficients | Chronological stages interim-carry the §2.5.2 stage-uniform attribution error (late-block releases under-deposited) until the refinement lands — a documented interim limitation; $K=1$ stays exact |
| **Both together**            | One landing: the shared arrival density (§2.5.4) emitting $k_d$ (skeleton) and $\chi_{b,d}/\kappa/\rho$ (block-resolved refinement), split at the stage boundary                                                                    | No interim approximation anywhere; the §2.5.2 forbidden alternative never ships; one consistent preprocessing validated once                                                                                                                 | Largest single increment; sub-contracts 1–3 and both parity anchors (§2.5.6) must land in the same increment                                                                                        |

**Recommendation: both together.** The block-resolved refinement carries no state
machinery of its own — it is template-coefficient work ($\chi$, $\kappa$, $\rho$) on
top of the same preprocessing that the core needs anyway — so the increment it adds is
modest, while core-first would temporarily ship, for chronological stages, exactly the
attribution error (§2.5.2's forbidden alternative) this design exists to prevent.
**Core-first** is the fallback if the plan needs a smaller first landing; its interim
chronological distortion must then be documented as a known limitation, never silent.

This is a surfaced fork for owner decision, not a silent resolution (anti-simplification
rule); it is a shipping-order choice only — no §8.4 lock depends on it.

---

## 9. Consolidated design forks (surface, do not silently resolve)

0. **Shipping order (§8.5) — the one remaining structural fork.** The sub-stage
   bucket-depth fork is **RESOLVED**: exact overlap in both modes,
   $L_{\text{arc}}(t) \ge 1$ per declared arc via the §2.5.1 window-overlap depth
   (decision record in §2.5.5; the fold is rejected). The representation question is also **RESOLVED** to
   k-weighted volume buckets (§8.4 lock #2 — block-resolved deposits are weighted
   volumes the raw-lagged form cannot carry). What remains is the landing order of the
   one mechanism's two layers: core-first (stage-clock skeleton with stage-uniform
   attribution as a documented interim) vs both-together (_recommended: both
   together_). Forks 1 and 3 below bind for **every** travel-time study — every
   declared arc carries a bucket.
1. **Initial condition (§4.3):** Require `past_defluences` (registro VI) — _recommended_
   — vs derive from `past_inflows` (fallback default with caveat) vs zero-seed
   (last resort). Affects short-term scheduling accuracy in the first $L$ stages.
   Binds for every declared arc (§8.4 lock #1) — a sub-stage arc's stage-0 bucket is
   the crossing mass of pre-study releases.
2. **Representation (§7.3): RESOLVED — k-weighted volume buckets aggregated per
   downstream plant** (§8.4 lock #2). The chronological block-resolved deposit
   $\sum_b \chi_{b,d}\,\tau_b\,D^b$ is intrinsically a weighted volume; the raw-lagged
   alternative survives only as the record of the rejected option.
3. **Terminal coupling (§5):** terminal bucket credit ($V_{\text{eff}}$, recommended)
   vs terminal FCF over augmented state vs horizon padding — and whether the finite or
   cyclic horizon mode is in scope.

---

## 10. What the implementation chain inherits from this memo

- **The block-mode-independent state principle (§2.5) is the governing invariant.**
  The bucket count `B` and the cut layout are a pure function of `(travel times, stage
lengths)` on the stage clock, **independent of `n_blks` and `block_mode`** — the same
  invariant `state_layout.rs` already enforces for block count. Three sub-contracts
  (§2.5.3) keep it: `B` from stage lengths only; the chronological overflow feeds the
  pre-existing stage-clock bucket (never allocates a new one); cross-stage delivery uses
  a fixed template density (buckets are stage-level scalars). Candidates for `sddp.md`.
- **The mechanism is one preprocessing, two layers (§8.5).** The within-stage routing
  layer is a template-construction change inside `fill_chronological_water_entries` /
  `fill_parallel_water_entries` — place same-stage arrival mass with block-overlap
  factors $\kappa_{b\to b'}$ (chronological, block clock) or on the single row
  (parallel). **Stateless**: `n_state`, `col_scale`, GEMM `d`, the MPI basis payload, and
  cross-**mode** policy portability all unchanged. The stage-clock bucket layer carries
  cross-stage water as state. Both layers come from the one shared arrival density
  (§2.5.4).
- The cross-stage bucket augmented state is **exact** (not approximate); buckets are the
  textbook multi-lag cut (§2.1 definition) in lifted coordinates; convexity, LB validity, monotonicity and
  a.s. finite convergence all hold (§2–§3). State expansion (`+B`) is `block_mode`-
  independent and paid for **every declared arc** ($L_{\text{arc}}(t) \ge 1$, §2.5.1 +
  the resolved §2.5.5 convention); a study with no declared arc pays nothing.
- **The bucket transition is realized in-LP** (definition rows), so state transfer is
  the storage mechanism verbatim: forward/simulation plain-copy captures
  $b_d^{\text{out}}$, `fill_col_state_patches` pins $b_d^{\text{in}}$, and **no
  `noise.rs` shift code exists for buckets** — the "anticipated_state template"
  language everywhere in this memo means the **sizing/masking discipline only**
  ($k_{max}$-global / per-stage-active / padding-masked), never an out-of-LP shift.
  Contract: bucket **outgoing** columns sit at LP columns equal to their state-vector
  indices (the storage convention), which is what makes the plain copy correct.
- Buckets are ordinary Benders state: **same** `rc/col_scale` extraction and negate-and-
  scale cut-row construction as storage; they slot into the
  storage→lag→bucket→anticipated `CutStateProjection` walk and the per-stage `CutPool`
  sizing (§3).
- `anticipated_state` ($k_{max}$/$K_i$/padding mask, k-injected ring buffer) is **the**
  implementation template for the resolved k-weighted aggregate representation and the
  stage-varying active set; `StageLagTransition` / `compute_recent_observation_seed`
  remain the templates for the overlap-factor arithmetic ($k_d$, $\chi$, $\kappa$,
  $\rho$) and the volume-accumulating IC seed (§7–§8).
- Initial state seeds at stage-0 incoming bucket column bounds, mirroring
  `build_initial_state` (§4); the terminal stage needs an explicit residual-bucket
  credit (§5); new determinism contracts go into `sddp.md` (§6.1).
