# Water Travel Time in Cobre SDDP — Formulation & Convergence Analysis

Design memo (formulation link of a specialist chain; a Rust implementation plan
follows). Scope: the **future cost function (FCF)** and **Benders cut** impact of
adding water travel time between cascade hydro plants, the **initial-condition**
question, the **end-of-horizon** coupling, and the **determinism/contract** risks.
No code is prescribed; the math and the state/cut structure are. Every formulation
claim is grounded against the cited cobre symbols.

---

## 0. Executive verdict

| #   | Question                                                                           | Verdict                                                                                                                                                                                                                                             |
| --- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | In-transit-bucket Markov-1 reframing equivalent to the textbook multi-lag E^k cut? | **VALIDATED** (proof in §2). The bucket cut _is_ the textbook cut in lifted coordinates. Convexity preserved. `k₀>0` handled cleanly.                                                                                                               |
| 2   | FCF/Benders impact under column-bound pinning?                                     | Buckets are ordinary Benders state: subgradient = `rc/col_scale` on the incoming bucket column, identical to storage. LB validity, monotonicity, a.s. finite convergence **preserved** (no new randomness). Cost: `+B` cut dimensions.              |
| 3   | Is "water in transit at study start" a required input?                             | **YES, genuinely required** . Recommend requiring it; a derive-from-`past_inflows` fallback is acceptable with a documented caveat. Genuine fork — §4.                                                                |
| 4   | Does the augmented state value residual buckets at horizon end?                    | **No, not automatically** when `V_{T+1}=0`. An explicit terminal credit (`V_eff = V + V_in_transit`) is required to avoid penalizing end-of-horizon upstream release. §5.                                                                  |
| 5   | Determinism / contract risks?                                                      | Canonical bucket-column sort, `col_scale` sizing, GEMM `d`, broadcast payload length, and one new reduction site to keep order-fixed. New `sddp.md` contracts proposed. §6.                                                                         |
| 6   | Best representation?                                                               | **k-weighted volume buckets aggregated per downstream plant** minimizes cut dimensionality (a sufficient statistic). Raw-lagged-defluence per source plant maximizes code reuse. Genuine fork — §7.                                                 |
| 7   | Variable stage length via `StageLagTransition`-style accumulation?                 | **Yes** — the calendar-fraction overlap arithmetic produces volume-conserving, stage-dependent k-factors and depths. Stage-varying active-bucket sets reuse the `anticipated_state` (k_max global / per-stage active / padding-masked) pattern. §8. |

---

## 0.1 Locked decisions (owner sign-off)

The three forks of §9, plus the end-of-horizon and per-stage-activation questions,
are resolved:

1. **Initial condition (§4.3):** REQUIRE `past_defluences` (registro VI); when
   absent, derive a logged proxy from `past_inflows`; never silently zero-seed.
   Seed at stage-0 incoming bucket column bounds (mirror `build_initial_state`).
2. **Representation (§7.3):** raw-lagged-defluence **per source plant** —
   `k`-factors as stage-dependent coefficients on the downstream water-balance row;
   reuse the `inflow_lags` ring buffer, lag-major layout, `shift_lag_state`, and
   `StageLagTransition` weighting; verbatim IC seeding.
3. **End-of-horizon (§5):** DEFER the `V_eff` terminal credit. Water still in
   transit past the horizon end (into/after a boundary FCF) is dropped — an
   accepted, documented imprecision, not silently zeroed elsewhere. Revisit when a
   terminal/cyclic value is in scope.
4. **Per-stage activation (§8):** travel-time modeling is per-stage, derived from
   stage length vs `τ` via the `k`-factor overlap — long stages (e.g. monthly)
   degenerate to `k⁰≈1` (no outgoing buckets); short stages (weekly) model fully;
   the boundary FCF carries no buckets. State dimension is stage-varying (global
   max, per-stage masked); inactive bucket slots are excluded from
   `nonzero_state_indices` (the §8.2 over-estimation guard).
5. **Modeled→unmodeled boundary:** a stage that generates no outgoing buckets still
   **receives** residual in-transit water that matures within it (collapses to a
   single incoming slot when the stage ≫ `τ`), preserving conservation in the
   upstream modeled stages and the backward-cut coupling to their releases. Only
   water maturing past the horizon end is dropped (per decision 3).

The implementation plan builds against these.

---

## 1. Notation and the current cobre state model

Following SDDP.jl conventions, augmented with cobre symbols.

| Symbol                                | Meaning                                                                                                                                                           |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $t$                                   | stage index, $t=1,\dots,T$                                                                                                                                        |
| $v_h^t$                               | outgoing storage of hydro $h$ (hm³) — Benders state                                                                                                               |
| $\hat v$                              | trial (visited) state                                                                                                                                             |
| $D_i^t$                               | total _defluence_ of plant $i$ at stage $t$ = $\sum_{\text{blk}} \tau_{\text{blk}}\,(u_{i,\text{blk}}+s_{i,\text{blk}}+\text{div})$ (hm³). Turbine $u$, spill $s$ |
| $\tau_{\text{blk}}$                   | $=\,$`duration_hours · M3S_TO_HM3` (flow→volume per block)                                                                                                        |
| $\zeta$                               | per-stage rate factor (`StageLayout::zeta`)                                                                                                                       |
| $k_d$                                 | propagation factor: fraction of an upstream release arriving $d$ stages later, $\sum_d k_d = 1$                                                                   |
| $L$                                   | max travel-time lag in stages                                                                                                                                     |
| $b_d^t$                               | in-transit _bucket_: volume that will arrive downstream in $d$ stages (incoming state at start of $t$)                                                            |
| $V_t(\cdot)$, $\underline V_t(\cdot)$ | true / lower-approximated recourse                                                                                                                                |
| $\theta$                              | epigraph (future-cost) variable, the global scalar column `StateLayout::theta`                                                                                    |

**The current cobre state vector** (`StateLayout`, file
`crates/cobre-sddp/src/lp/indexer/state_layout.rs`) is, in column order:

```
[storage (N)] ⊕ [inflow_lags (N·L_par, lag-major)] ⊕ [anticipated_state (A·k_max)]
   ⊕ [anticipated_state_out (A)] ⊕ [z_inflow (N, aux)] ⊕ [storage_in (N, incoming)] ⊕ θ
```

with `n_state = N·(1+L_par) + A·k_max`. Three facts are load-bearing for this memo:

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

### 2.1 Setup

Today (`lp/builder/entries.rs::fill_state_and_water_entries`, upstream loop) the
downstream water-balance row of $j=\mathrm{downstream}(i)$ receives the **full**
upstream defluence in the **same** stage:

$$
v_j^t - v_j^{t-1} + \sum_{\text{blk}}\tau_{\text{blk}}(u_j+s_j+\text{div})
- \underbrace{\sum_{\text{blk}}\tau_{\text{blk}}(u_i+s_i)}_{=\,D_i^t\ \text{(arrives now)}}
- \zeta\!\sum_{l}\psi\,a_{i,t-l} = \zeta(\text{base}-\text{withdrawal}).
$$

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
$\alpha_5 \ge w_6 + \pi_6\!\left[E_5^1\Delta x_5 + E_4^2\Delta x_4 + E_3^3\Delta x_3\right]$.

### 2.2 The lifting

Define the bucket state $b_t=(b_1^t,\dots,b_L^t)$ with $b_d^t = \sum_{m\ge1} k_{d-1+m}\,D_i^{t-m}$ ($k_e=0$ for $e>L$). Equivalently $b_t = M\,\mathcal D_{t-1}$ with the lower-triangular Toeplitz map $[M]_{d,m}=k_{d-1+m}$.

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

3. **The bucket cut IS the textbook cut.** If $k_L\neq0$, $M$ is invertible, and
   $V_t^A(v,b):=V_t^F(v,M^{-1}b)$. Both are convex (§2.3). Any SDDP cut on $V_t^A$,
   $$
   \theta \ge \alpha + \beta_v^\top v + \beta_b^\top b,
   $$
   pulls back through $b=M\mathcal D$ to
   $$
   V_t^F(v,\mathcal D) \ge \alpha + \beta_v^\top v + (M^\top\beta_b)^\top \mathcal D,
   $$
   whose $\mathcal D$-subgradient $M^\top\beta_b$ has a nonzero component on **every**
   prior-stage defluence $D_i^{t-1},\dots,D_i^{t-L}$. The textbook $E^k$ matrices are
   precisely the rows of $M^\top$ (the k-factor structure). $\square$

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

## 3. Q2 — FCF and Benders cut impact under column-bound pinning

### 3.1 Column structure (mirror `storage`/`anticipated_state`)

Each bucket dimension contributes **one state dimension** and **two LP columns**,
exactly like storage (`storage` + `storage_in`):

- $b_d^{\text{out}}$ — outgoing column, the **cut target** (lives in the state region,
  analogous to `storage[0..N)` and `anticipated_state_out`).
- $b_d^{\text{in}}$ — incoming column, **pinned via `set_col_bounds`** to the previous
  stage's $b_d^{\text{out}}$ (analogous to `storage_in`); its reduced cost is the
  subgradient source.

A single **freshest-bucket materialization** is needed (the injection $k_d D_i^t$ makes
the outgoing bucket a _sum_, not a single pre-existing column). This is the direct
analogue of `z_inflow`: `z_inflow` materializes lag-0's realized inflow so the lag cut
can target a single column; here an aux **total-defluence column** $D_i^t$ (defined by
$D_i - \sum_{\text{blk}}\tau_{\text{blk}}(u_i+s_i)=0$) materializes the injection. §7
discusses whether the buckets are then a pure shift (raw-lagged) or carry transition
rows (k-weighted).

The augmented layout, preserving the "pure function of dimensions" invariant (§1.1),
inserts a bucket block at a **fixed** position (recommend immediately after
`inflow_lags`, before `anticipated_state`, so anticipated offsets shift by a constant
$B$ and stay dimension-pure):

```
… ⊕ [inflow_lags (N·L_par)] ⊕ [buckets_out (B)] ⊕ [anticipated_state (A·k_max)] ⊕ …
                                ⊕ [buckets_in (B, incoming)] ⊕ …
```

with `n_state = N·(1+L_par) + B + A·k_max`.

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
  decision). The augmented state is bounded (defluence is turbine+spill-capacity
  bounded; buckets are storage-scale volumes) and convex, so the standard SDDP
  convergence hypotheses (compact convex state, relatively complete recourse,
  stagewise-independent finite noise) still hold. Convergence is preserved.

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
per source plant or per arc, the last $L$ stage defluences). At setup compute the
incoming bucket state and inject it as the **stage-0 incoming bucket column bounds**
(`set_col_bounds`), exactly as initial storage pins `storage_in`. In the raw-lagged
representation (§7) the seed is **verbatim** ($D^{0}_{\text{lag},d}=D_i^{-d}$, like
`past_inflows`); in the k-weighted representation it is the unrolled
$b_d^0=\sum_{m\ge1}k_{d-1+m}D_i^{-m}$ (a deterministic setup precompute, single-threaded,
fixed lag order). The volume-accumulating precedent already exists:
`stochastic/lag_transition.rs::compute_recent_observation_seed` accumulates
`value_m3s · observation_hours` and a coverage `weight_seed`.

### 4.3 Derive vs require — a genuine fork

| Option                                         | What                                                                                               | Trade-off                                                                                                                                                                                   |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Require `past_defluences`** (registro VI) | User supplies pre-study upstream releases                                                          | Most correct; Cost: one new input field. **Recommended.**                                                                                                                   |
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
   (sorted) topology and dimensions to keep the stage-invariant offset property (§1.1).

2. **Append-only cut pool + slot-identity basis reconstruction.** `reconstruct_basis`
   matches cut **rows** to pool **slots** by slot identity — independent of state
   dimension — so it is unaffected by adding bucket dims, provided `n_state` and the
   per-stage `CutStateProjection`/`CutPool` sizing include the buckets consistently.
   Stage-varying active-bucket sets (§8) ride the existing per-stage pool dimension
   support (`FutureCostFunction::new_per_stage`, already used for `inflow_lags:false`
   stages).

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
   mismatched `state_dimension` and **must be rejected** — the existing
   `FutureCostFunction::from_deserialized` / `new_with_warm_start` consistency check
   covers this _iff_ the bucket dims are part of the recorded `state_dimension`. The
   `cut/wire.rs` record carries the coefficient length, so within a version it
   round-trips; ensure the bucket dims are in `state_dimension` so a stale policy is
   caught.

6. **Bit-reproducible reductions.** The bucket subgradient components join the
   **existing** expectation/CVaR reduction over openings — no new reduction class, but
   they must use the **same fixed-order/compensated** reduction cobre already applies to
   the storage subgradients (verify it is order-fixed; see the numerical-reproducibility
   discipline). The initial-condition unroll $\sum_m k_{d-1+m}D^{-m}$ and the
   per-downstream injection $\sum_i k_{d,i}D_i^t$ are computed at **setup / per-stage
   template build on one rank** in a fixed canonical (sorted) order — keep them out of
   any rank-count-dependent parallel reduction.

### 6.1 New `sddp.md` contracts to add

- **In-transit bucket dynamics & sign.** "The in-transit bucket ring buffer evolves
  Markov-1 as $b_d^{\text{out}} = b_{d+1}^{\text{in}} + k_d D_i$ (raw-lagged:
  $D^{\text{out}}_{\text{lag},d}=D^{\text{in}}_{\text{lag},d-1}$, freshest =
  total-defluence aux). Incoming buckets are pinned via column bounds; the subgradient
  is $\text{rc}/\text{col\_scale}$ on the **incoming** bucket column and the cut row
  multiplies by `col_scale` on the **outgoing** bucket column — divided on extract,
  multiplied on render, identical to storage. Pin to a named regression case."
- **k-factor conservation.** "The propagation factors satisfy $\sum_d k_d = 1$ per
  source per stage (volume conservation); a `debug_assert` enforces it. A drift from 1
  silently creates or destroys water."
- **Canonical bucket ordering.** "Bucket columns sort by sorted-ID key
  `(downstream_id, [upstream_id,] lag)`, never traversal order — declaration-order
  invariance."
- **Terminal credit.** "In the finite-horizon mode the last stage credits residual
  buckets to downstream storage ($V_{\text{eff}}$); without it end-of-horizon upstream
  release is under-valued (pin to a regression case)."

---

## 7. Q6 — Representation recommendation

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
$B_d^{j} = \sum_{i\in\text{up}(j)} k_{d,i} D_i^{t-d}$. It evolves Markov-1:

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
Because cut dimensionality is the dominant SDDP cost (§3.4), I lean to this as the
**primary recommendation**, with raw-lagged-per-source as the max-reuse alternative
when the cascade is linear. This is a genuine fork (reuse vs minimal dimension) and is
surfaced for decision rather than silently chosen.

---

## 8. Q7 — Variable stage length

Travel time is a scalar in **hours**; stages are days/weeks/months. A single delay
$t_v$ must split across multiple downstream stages when a stage is shorter than $t_v$,
and the k-factors are therefore **stage-dependent**.

### 8.1 The k-factors are calendar-fraction overlaps — mirror `StageLagTransition`

A release $D_i^t$ spread over stage $t$ (hours $h_t$) arrives downstream over the window
$[\text{start}_t + t_v,\ \text{end}_t + t_v)$. The fraction reaching downstream stage
$t+m$ is the **overlap of that arrival window with stage $t+m$'s calendar interval**,
normalized to sum to 1. These overlap fractions **are** $k_d^t$, and they are exactly
the `days_in_period`/`month_total_hours` arithmetic in
`stochastic/lag_transition.rs::compute_monthly_transition`
(accumulate_weight/spillover_weight). Mirroring that calendar-fraction logic produces:

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
periods), reusing `days_in_period`/`month_total_hours`/`find_season_year_monthly`, **not**
overloaded onto `StageLagTransition`.

### 8.2 Stage-dependent state dimension — a solved pattern

Buckets that don't exist (or are inactive) at some stages is the **same** situation
cobre already handles three ways:

1. `inflow_lags:false` stages — fewer cut-state dims, sized via
   `FutureCostFunction::new_per_stage` / `CutStateProjection`.
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

---

## 9. Consolidated design forks (surface, do not silently resolve)

1. **Initial condition (§4.3):** Require `past_defluences` (registro VI) — _recommended_
   — vs derive from `past_inflows` (fallback default with caveat) vs zero-seed
   (last resort). Affects short-term scheduling accuracy in the first $L$ stages.
2. **Representation (§7.3):** raw-lagged-defluence per source (max reuse, linear
   cascades) vs k-weighted aggregate per downstream plant (minimal dimension, confluent
   cascades). Reuse vs cut-dimensionality.
3. **Terminal coupling (§5):** terminal bucket credit ($V_{\text{eff}}$, recommended)
   vs terminal FCF over augmented state vs horizon padding — and whether the finite or
   cyclic horizon mode is in scope.

---

## 10. What the implementation chain inherits from this memo

- The augmented state is **exact** (not approximate); buckets are the textbook
  multi-lag cut in lifted coordinates; convexity, LB validity, monotonicity and a.s.
  finite convergence all hold (§2–§3).
- Buckets are ordinary Benders state: **same** `rc/col_scale` extraction and negate-and-
  scale cut-row construction as storage; they slot into the
  storage→lag→bucket→anticipated `CutStateProjection` walk and the per-stage `CutPool`
  sizing (§3).
- The two existing ring-buffer patterns are the implementation templates:
  `inflow_lags` (+`StageLagTransition`, `compute_recent_observation_seed`) for the
  raw-lagged form and the variable-stage k-factors; `anticipated_state`
  ($k_{max}$/$K_i$/padding mask) for the k-weighted aggregate form and the
  stage-varying active set (§7–§8).
- Initial state seeds at stage-0 incoming bucket column bounds, mirroring
  `build_initial_state` (§4); the terminal stage needs an explicit residual-bucket
  credit (§5); new determinism contracts go into `sddp.md` (§6.1).
