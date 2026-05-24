# Anticipated-Thermal Dispatch — Methodology Reference (current code)

Source of truth: branch `feat/anticipated-thermals-pre-horizon-seeding`,
HEAD `241c057`. The Epic 03 always-active fishing predicate flip is reverted
in commit `29ebd1e`; the LP is back to the legacy `K_i <= stage_idx`
semantics. This document describes what the LP _actually_ does today.

Notation:

- $T$ = number of study stages, $t \in \{0, \dots, T-1\}$ (zero-indexed).
- $A$ = number of anticipated thermals; $i \in \{0, \dots, A-1\}$ (anticipated-local index).
- $K_i \ge 1$ = lead stages for plant $i$.
- $K = K_{\max} = \max_i K_i$, the ring-buffer width.
- $B$ = blocks per stage; $h_b$ = block-$b$ duration in hours; $H = \sum_b h_b$.
- $d^{\mathrm{NPV}}_t$ = cumulative discount factor at stage $t$.
- $c_i(t)$ = unit cost (\$/MWh) for plant $i$ at stage $t$.

---

## 1. Entity surface

`cobre_core::Thermal.anticipated_config: Option<AnticipatedConfig>` with
`AnticipatedConfig.lead_stages: u32` (file
`crates/cobre-core/src/entities/thermal.rs:25-28`). Physically:
$K_i = \texttt{lead\_stages}$ is the **number of stages between commitment and
delivery**. A decision $d_t^i$ taken at stage $t$ delivers physical generation
at stage $t + K_i$. `lead_stages >= 1` is enforced by the IO validator.

`cobre_core::AnticipatedCommitmentHistory.values_mw: Vec<f64>`
(file `crates/cobre-core/src/initial_conditions.rs:136-146`). Nominal intent:
`values_mw[k]` is the externally-decided MW the plant must deliver at study
stage $k$, for $0 \le k < K_i$. **Current LP-enforcement gap** (docstring
lines 97-108): non-zero entries are accepted by the validator but the LP does
**not** pin generation to them. The fishing constraint that would do so is
gated by `K_i <= stage_idx` and never reads slot 0 before the ring-buffer
shift has overwritten it (see Section 6).

---

## 2. LP columns

Three column ranges are introduced per stage. Anticipated thermals also reuse
the standard per-block thermal columns; the LP does **not** distinguish them
at the column level (confirmed in `matrix.rs:202-220` —
`fill_thermal_columns` iterates over _all_ `ctx.thermals` with no anticipation
check).

### 2.1 `col_anticipated_decision_start + i`, $i \in \{0, \dots, A-1\}$

One column per anticipated plant per stage (`layout.rs:158-163`,
`matrix.rs:235-261`). Bounds depend on the **strict horizon predicate**
$t + K_i < T$:

$$
d_t^i \in
\begin{cases}
[\,g_{\min}^i(t+K_i),\ g_{\max}^i(t+K_i)\,] & \text{if } t + K_i < T \\
[\,0,\ 0\,] & \text{otherwise (presolved out)}
\end{cases}
$$

The boundary case $t + K_i = T$ is excluded: the delivery stage would fall
outside the study horizon (`matrix.rs:228-232`).

### 2.2 `col_anticipated_state_start + slot * A + i`, slot-major

Ring-buffer state columns, $A \cdot K$ per stage (`layout.rs:164-171`,
`matrix.rs:102-110`). Bounds are $(-\infty, +\infty)$; the slot value is
pinned by the Category 6 state-fixing row (Section 3). Slot $s$ of plant $i$
represents the MW committed $K_i - 1 - s$ stages ago, scheduled for delivery
$K_i - s$ stages from now (slot 0 = "matures _this_ stage if fishing fires").

### 2.3 `col_thermal_start + g * B + b`, per-block generation $g_b^g$

Shared with standard thermals (`matrix.rs:213`). For an anticipated plant $i$
with global thermal index $g$, these columns are the _physical_ MW dispatched
in each block. The fishing equality couples them to the ring-buffer slot 0
when fishing is active.

---

## 3. LP rows

### 3.1 Category 6 — anticipated-state-fixing equality, one row per `(slot, i)`

Built by `fill_anticipated_state_fixing_entries` (`matrix.rs:1014-1026`),
`fill_anticipated_decision_state_write_entries` (`matrix.rs:1040-1062`), and
RHS-patched at solve time by `fill_anticipated_state_patches`
(`patch.rs:407-434`).

For each $(s, i) \in [0, K) \times [0, A)$, the row at
`row_anticipated_state_fixing_start + s*A + i` reads:

$$
\underbrace{1.0}_{\text{diagonal}} \cdot x^{\mathrm{state}}_{s,i,t}
+ \underbrace{\mathbb{1}\{t + K_i < T \land s = K_i - 1\}}_{\text{decision write}} \cdot d_t^i
\;=\; \widehat{x}^{\mathrm{state}}_{s,i,t}
$$

where the RHS $\widehat{x}^{\mathrm{state}}_{s,i,t}$ is patched in from the
incoming `state[anticipated_state.start + s*A + i]`:

- **Stage $t = 0$**: the RHS is whatever `build_initial_state`
  (`setup/mod.rs:880-919`) wrote — either zero, or the seed
  `values_mw[s]` if a non-zero entry was supplied.
- **Stage $t \ge 1$**: the RHS is the previous stage's outgoing state
  after `shift_anticipated_state` ran (Section 5).

The decision-write column contributes $+1.0$ at row $s = K_i - 1$ only when
the plant is active at $t$ (`matrix.rs:1052-1060`). For $K_i = 1$ the decision
writes into slot 0 — same slot the fishing row reads. **This collision is
exactly what broke Epic 03's predicate flip.**

### 3.2 Anticipated-fishing equality, one row per active plant

Built by `fill_anticipated_fishing_rows` (`matrix.rs:927-950`) and
`fill_anticipated_fishing_entries` (`matrix.rs:965-1000`). Active iff
**$K_i \le t$** (the legacy stage-dependent predicate restored by the revert,
`layout.rs:737-741`). For each active $i$:

$$
\sum_{b=0}^{B-1} h_b \cdot g_{b,t}^{i}
\;-\; H \cdot x^{\mathrm{state}}_{0, i, t}
\;=\; 0
$$

Equivalently $\sum_b h_b \, g_{b,t}^i = H \cdot x^{\mathrm{state}}_{0,i,t}$:
total stage MWh dispatched by plant $i$ equals the slot-0 MW level scaled by
total hours. Slot 0 therefore acts as a stage-wide MW pin on the per-block
generation. When **inactive** ($K_i > t$), the row does not exist; the
per-block columns are free to vary within their standard $[g_{\min}, g_{\max}]$
bounds.

Inflow lag rows, generic constraints, load balance, water balance, etc. are
unchanged by anticipation and out of scope here.

---

## 4. Objective contributions

Three terms touch anticipated plants. Anticipation costs are NPV-discounted
**at the delivery stage** but **charged at the decision stage**, mirroring
how a real-world option payment is priced.

### 4.1 Per-block dispatch cost (zeroed at delivery)

`fill_thermal_columns` writes the standard cost on each per-block column
(`matrix.rs:217`):

$$
\sum_{b=0}^{B-1} c_i(t) \cdot h_b \cdot d^{\mathrm{NPV}}_t \cdot g_{b,t}^i
$$

This is then **zeroed for anticipated plants at delivery stages** by
`zero_anticipated_delivery_thermal_cost` (`matrix.rs:325-347`), gated by
the same legacy predicate $K_i \le t$. Reason: at $t \ge K_i$ the fishing
row binds $g_{b,t}^i$ to a past decision $d_{t - K_i}^i$ whose cost has
already been priced (Section 4.2). Charging it again per block would
double-count.

### 4.2 Anticipated-decision cost (NPV-discounted to delivery stage)

`fill_anticipated_decision_objective` (`matrix.rs:283-306`) writes the
coefficient on `col_anticipated_decision_start + i` only when the plant is
active at $t$ ($t + K_i < T$):

$$
\text{obj}\!\left[d_t^i\right]
\;=\; c_i(t+K_i) \cdot H_{t+K_i} \cdot d^{\mathrm{NPV}}_{t+K_i}
$$

where $H_{t+K_i} = \texttt{total\_hours\_per\_stage}[t+K_i]$. Note the
discount uses the **delivery-stage** factor: the LP sees the present value
of paying for the energy at delivery, even though the commitment is locked
at $t$. Inactive plants (those at $t + K_i \ge T$) have a zero objective
coefficient and $[0, 0]$ bounds, so the column is presolved out.

### 4.3 State-column objective

Anticipated-state columns carry zero objective coefficient. They are pure
state variables.

---

## 5. Ring-buffer evolution between stages

`shift_anticipated_state` (`noise.rs:253-297`) runs after each forward-pass
solve. Given:

- `incoming_anticipated[s * A + i]` = the _previous_ stage's outgoing state
  slot $s$ for plant $i$ (saved into `ws.scratch.anticipated_state_buf`
  before being overwritten).
- `unscaled_primal[anticipated_decision.start + i]` = the LP's optimal
  $d_t^i$ at this stage.

For each plant $i$ with lead $K_i$:

$$
x^{\mathrm{state}}_{s, i, t+1} =
\begin{cases}
\text{incoming}_{s+1, i} & 0 \le s < K_i - 1 \quad \text{(shift down)} \\
d_t^i & s = K_i - 1 \quad \text{(write newest)} \\
0 & K_i \le s < K \quad \text{(padding)}
\end{cases}
$$

The slot-0 incoming value falls out at every transition. For inactive plants
the LP enforces $d_t^i = 0$ via $[0, 0]$ bounds, so writing the primal at
slot $K_i - 1$ still produces a zero — the comment at `noise.rs:233-244`
explains the invariant.

---

## 6. The seeded `values_mw` — what actually happens

`build_initial_state` (`setup/mod.rs:880-919`) writes:

$$
\texttt{state}[\,\texttt{anticipated\_state.start} + s \cdot A + i\,]
= \texttt{history.values\_mw}[s], \quad 0 \le s < K_i
$$

At stage 0, the Category 6 RHS patches (Section 3.1) carry these values into
the LP. **But** the fishing row at stage 0 is active **only** for plants with
$K_i = 0$ — and $K_i \ge 1$ is mandatory. So at stage 0 _no_ fishing row
reads slot 0. The LP sees the slot-0 state column pinned to `values_mw[0]`
by the state-fixing row, but no other constraint references it. The shift at
end of stage 0 then overwrites:

- slot 0 $\leftarrow$ incoming slot 1 $= \texttt{values\_mw}[1]$.
- slot $K_i - 1$ $\leftarrow d_0^i$.
- the seed `values_mw[0]` is discarded.

By the time slot 0 carries the _original_ `values_mw[K_i - 1]` (after $K_i - 1$
shifts), $t = K_i - 1$, and fishing is _still_ inactive ($K_i > K_i - 1$).
One more shift and the LP-decided $d_0^i$ has taken slot 0's place.

**Conclusion**: under the current legacy code, the LP silently discards
seeded `values_mw[k]` values for every $k$. The validator emits a
`SemanticAmbiguity` warning when non-zero values are supplied (see
`AnticipatedCommitmentHistory` docstring) so the user is notified, but the
dispatch is identical to the all-zeros seed.

For the NEWAVE-parity bridge case (ST.CRUZ NOVA with $K=1$,
`values_mw = [204.5647]`), the LP behaves identically to
`values_mw = [0.0]`.

---

## 7. Cut machinery

Cuts are generated by the backward pass (`backward.rs`) and applied as LP
rows at the predecessor stage. Conventions documented in
`crates/cobre-sddp/src/backward.rs:6-39`:

- The dual on the anticipated-state-fixing row $(s, i)$ at stage $t+1$ is
  $\partial Q_{t+1} / \partial \widehat{x}^{\mathrm{state}}_{s, i, t+1}$.
- Convention: `coefficients = dual` (raw HiGHS dual, no negation). The cut
  row builder (`forward.rs::push_scaled_coefficient`, called via
  `build_cut_row_batch`) writes $-\beta_j x_j + \theta \ge \alpha$ so the
  cut reads $\theta \ge \alpha + \beta^\top x$.
- The dual on slot 0 at stage $t+1$ is non-zero only when fishing has bound
  the LP there — requiring $K_i \le t + 1$.

`state_to_lp_column` (`indexer.rs:1388-1443`) maps the **predecessor's
outgoing state index $j$** to the LP column in the predecessor's stage
problem:

- For an anticipated-state slot $(s, i)$ with $s + 1 = K_i$
  (`Ordering::Equal`, line 1402): map to
  `anticipated_decision.start + i`. The cut subgradient targets the
  predecessor's _decision_ column directly. This is the only branch that
  fires for $K_i = 1$ (slot 0 maps to decision).
- For $s + 1 < K_i$ (`Ordering::Less`, line 1403): map to
  `anticipated_state.start + (s+1) * A + i`. The cut targets the
  predecessor's _outgoing-state_ slot $s+1$, which is where the same
  commitment lived one stage earlier.
- For padding slots $s \ge K_i$ (`Ordering::Greater`, line 1426): identity
  return. The padding-slot invariant (`indexer.rs:1406-1418`) guarantees the
  dual is zero because the slot is pinned to zero by its state-fixing row.

The net effect: a non-zero $\partial Q / \partial(\text{slot 0 at stage } t+1)$
propagates backward to the _correct_ upstream variable — either the
predecessor's decision (for $K_p = 1$) or the predecessor's slot 1 (for
$K_p \ge 2$), which in turn propagates back via its own cut at stage $t-1$,
and so on until it reaches the original decision column.

---

## 8. What anticipated dispatch DOES deliver today

- Plants with `anticipated_config: Some(K)` get a separate decision column
  $d_t^i$ at every stage $t$ with $t + K_i < T$.
- The decision cost is paid at stage $t$ but priced at delivery-stage NPV
  (Section 4.2).
- At delivery stage $t + K_i$, the fishing row pins
  $\sum_b h_b g_b^i = H \cdot \text{slot 0}$, where slot 0 carries
  $d_{t}^i$ after $K_i$ ring-buffer shifts.
- Per-block dispatch cost is zeroed at delivery stages (Section 4.1) to
  avoid double-counting.
- The SDDP cut machinery propagates
  $\partial Q_{t+1} / \partial d_t^i$ correctly through the state-fixing
  row duals and `state_to_lp_column` remapping.

## What it does NOT deliver

- Pre-horizon seeded values `values_mw[k]` are loaded into the initial
  state but discarded by the ring-buffer shift before any fishing
  constraint reads them.
- For NEWAVE parity ($K = 1$, `values_mw = [204.5647]`), the LP behaves
  identically to `values_mw = [0.0]`.
- The validator's `SemanticAmbiguity` warning is the only user-facing
  signal that the seeded values are inert.

---

## 9. End-to-end numerical trace ($T = 3$, single anticipated plant, $K = 2$)

System:

- Anticipated plant $i = 0$: $g_{\max} = 100$ MW, $c = 10$ \$/MWh, $K = 2$.
- Backup thermal (standard): $g_{\max} = 500$ MW, $c = 5000$ \$/MWh.
- Single block per stage with $h = 1$ h, so $H = 1$, $d^{\mathrm{NPV}} = 1$.
- Load = 150 MW per stage.
- Seed `values_mw = [0, 0]`.

State vector after `build_initial_state`:
$\big[\,\text{slot 0} = 0,\ \text{slot 1} = 0\,\big]$.

### Stage $t = 0$ (decision)

Active predicate $t + K < T$: $0 + 2 < 3$ → **active**, $d_0 \in [0, 100]$.
Fishing predicate $K \le t$: $2 \le 0$ → **inactive**, no fishing row.
Decision write at $s = K - 1 = 1$: yes.

Category 6 state-fixing equations:

- slot 0: $x^{\mathrm{state}}_{0,0,0} = 0$ (no decision write, RHS = 0).
- slot 1: $x^{\mathrm{state}}_{1,0,0} + d_0 = 0$.

The slot-1 equation implies $x^{\mathrm{state}}_{1,0,0} = -d_0$. Since
this state column is unbounded in sign, that is feasible.

Power balance: $g^{\text{backup}}_0 + g_{0,0}^{i=0} = 150$, with
$g_{0,0}^{i=0} \in [0, 100]$ (no fishing constraint binds it). The LP
prefers cheap thermal: $g_{0,0}^{i=0} = 100$, $g^{\text{backup}}_0 = 50$
**at the per-block level**. Per-block dispatch cost for the anticipated
plant is **not** zeroed at $t = 0$ (predicate $K \le t$ is $2 \le 0$ → false),
so this 100 MW costs $100 \times 1 \times 10 = 1000$. The backup costs
$50 \times 1 \times 5000 = 250{,}000$.

The decision cost: $c_i(2) \cdot H \cdot d^{\mathrm{NPV}}_2 \cdot d_0
= 10 \cdot 1 \cdot 1 \cdot d_0$. The LP wants to set $d_0$ as low as
possible to minimize cost — and **nothing binds $d_0$ to physical
delivery yet**. So the LP picks $d_0 = 0$. _(This is the symptom of the
no-pre-horizon-enforcement gap: at $t = 0$ the LP can both dispatch the
cheap plant at full capacity per-block AND set $d_0 = 0$, because the
fishing row that would couple them is inactive.)_

State after shift:

- slot 0 $\leftarrow$ incoming slot 1 $= 0$.
- slot 1 $\leftarrow d_0 = 0$.
- State at start of $t = 1$: $[0, 0]$.

### Stage $t = 1$ (decision)

Active $1 + 2 < 3$? No. So $d_1 \in [0, 0]$ (presolved out).
Fishing $K \le t$: $2 \le 1$? No. No fishing row.

Same LP geometry as $t = 0$ but without a fresh decision: per-block
$g_{0,1}^{i=0} = 100$, backup = 50. State after shift remains $[0, 0]$.

### Stage $t = 2$ (delivery of the never-made commitment)

Active $2 + 2 < 3$? No. $d_2 \in [0, 0]$.
Fishing $K \le t$: $2 \le 2$? **Yes**, fishing row active.

Fishing equation: $1 \cdot g_{0,2}^{i=0} - 1 \cdot x^{\mathrm{state}}_{0,0,2} = 0$,
with $x^{\mathrm{state}}_{0,0,2}$ pinned to incoming slot 0 = 0. So
$g_{0,2}^{i=0} = 0$.

Per-block dispatch cost is **zeroed** at $t = 2$ (predicate $K \le t$ holds),
so this would have been free — but the fishing row pins it to zero anyway.
Backup carries the full 150 MW: $150 \times 5000 = 750{,}000$.

### Total

$$
J_{\text{total}} = \underbrace{1000 + 250{,}000}_{t=0} + \underbrace{1000 + 250{,}000}_{t=1} + \underbrace{750{,}000}_{t=2}
= 1{,}252{,}000
$$

**Key observation**: with the legacy predicate, the LP both dispatches the
anticipated plant freely _and_ never commits via $d_t$, because the cost of
$d_t$ is paid in present value at $t$ but its enforcement only kicks in at
$t + K$. The plant pays nothing for $d_t$ at $t$, and at $t + K$ it has no
delivery obligation (slot 0 = 0). The cheap energy at $t = 0, 1$ comes from
the _standard_ per-block thermal cost path, not the anticipation pipeline.
This is exactly why the original Epic 03 wanted to flip the predicate — and
exactly why the LP-geometry regression at $K = 1$ forced the revert. The
correct fix (decouple fishing-read from decision-write slots) is documented
as future work and is not yet implemented.

---

## Cross-references

| Topic                                                | File                                          | Lines     |
| ---------------------------------------------------- | --------------------------------------------- | --------- |
| `AnticipatedConfig`                                  | `crates/cobre-core/src/entities/thermal.rs`   | 22-28     |
| `AnticipatedCommitmentHistory` (with gap docstring)  | `crates/cobre-core/src/initial_conditions.rs` | 88-146    |
| `fill_anticipated_decision_columns`                  | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 235-261   |
| `fill_anticipated_decision_objective`                | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 283-306   |
| `zero_anticipated_delivery_thermal_cost`             | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 325-347   |
| `fill_anticipated_fishing_rows` (predicate)          | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 927-950   |
| `fill_anticipated_fishing_entries`                   | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 965-1000  |
| `fill_anticipated_state_fixing_entries`              | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 1014-1026 |
| `fill_anticipated_decision_state_write_entries`      | `crates/cobre-sddp/src/lp_builder/matrix.rs`  | 1040-1062 |
| `n_anticipated_fishing_rows` (`K_i <= t` count)      | `crates/cobre-sddp/src/lp_builder/layout.rs`  | 737-741   |
| `fill_anticipated_state_patches` (RHS at solve time) | `crates/cobre-sddp/src/lp_builder/patch.rs`   | 407-434   |
| `shift_anticipated_state` (ring buffer)              | `crates/cobre-sddp/src/noise.rs`              | 253-297   |
| `build_initial_state` (seed write)                   | `crates/cobre-sddp/src/setup/mod.rs`          | 847-919   |
| `state_to_lp_column` (cut remap)                     | `crates/cobre-sddp/src/indexer.rs`            | 1388-1443 |
| Backward-pass dual convention                        | `crates/cobre-sddp/src/backward.rs`           | 6-66      |
| Epic 03 revert commit                                | git                                           | `29ebd1e` |
