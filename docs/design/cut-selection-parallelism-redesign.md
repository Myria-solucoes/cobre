# Cut Selection Parallelism Redesign

**Status**: Design proposal (not yet implemented)
**Date**: 2026-05-27
**Author**: cobre team
**Predecessor**: `docs/design/cut-selection-correctness.md` (the algorithmic correctness fix that
landed on `feat/cut-selection-correctness`)
**Production target**: disaggregated case, `D = 2080`, `K` up to several
thousand, `M = 192–384`. Aggregated (`D = 155`) is a useful but secondary
benchmark.

---

## 1. Executive summary

The unified value-evaluation cut-selection kernel that just shipped is
algorithmically correct (matches de Matos 2015 and Guigues & Bandarra 2019)
but its intra-stage parallelism leaves more than two orders of magnitude
of performance on the table. At aggregated scale (run_8) selection takes
~700 ms wall per stage; at disaggregated scale the work is ~13× larger,
so today's kernel would be the dominant wall-clock cost. The current
design cannot ship to production at `D = 2080`.

This proposal redesigns the kernel around three commitments:

1. **`matrixmultiply::dgemm` is the Phase 1 kernel**, not an optional
   fallback. Phase 1 (computing `V = coef · stateᵀ`) is a tall-skinny GEMM,
   and at disaggregated scale the working set (24 MB coef, 6 MB state) is
   in L3 — cache-blocked GEMM is the only kernel that runs at peak FLOPS
   for these shapes. We accept the dependency.

2. **Rayon parallelism over trial-point blocks**, wrapping the dgemm call.
   Each rayon task computes `V` for a small block of `m`'s via one dgemm
   call, then immediately runs Phase 2 (column max) and Phase 3 (survival
   rule) on those columns. Per-worker scratch lives in L1/L2. The
   bitmap merge is OR — commutative, associative, deterministic.

3. **Bit-for-bit reproducibility delegated to `matrixmultiply`**. The
   crate is single-threaded internally, uses a fixed cache-blocked
   algorithm, and produces deterministic output on any IEEE-754 target.
   We commit to this one BLAS-equivalent dependency and never introduce
   external BLAS (whose tile parallelism would break the guarantee).

A cheaper kernel also unlocks **selection-inside-backward**: prune cuts
in `FCF[t]` as soon as stage _t_'s backward pass finishes, so the next
backward stage solves against a leaner template. This is the actual
wall-clock saving — selection itself was never the bottleneck; selection's
cost was _blocking_ an architectural change that would shrink the
backward LP cost (which **is** the bottleneck, especially at `D = 2080`).

The MPI dimension is unaffected: every rank holds identical pools and
archives after cut sync, runs the same kernel, and produces identical
bitmaps without any communication.

Trade-off accepted: one new workspace dependency (`matrixmultiply`,
pure-Rust, MIT/Apache-2.0, ~15 KLOC, deterministic by construction). In
return we get a single principled kernel that performs at peak FLOPS
across the full range of `D` we ship, with the determinism story
centralised in the crate maintainer's hands instead of distributed across
every SIMD code path we'd otherwise need to police.

---

## 2. The current design and why it underperforms

### 2.1 What runs today

In `crates/cobre-sddp/src/cut_selection.rs` (post-Epic 02):

```rust
pub fn select_for_stage(&self, pool: &CutPool, visited_states: &[f64],
                       current_iteration: u64, stage_index: u32)
    -> CutActivityUpdates
{
    // ...edge cases, eligibility...
    let n_states = visited_states.len() / n_state;

    let is_selected: Vec<bool> = if n_states >= PARALLEL_THRESHOLD {
        visited_states
            .par_chunks(PARALLEL_THRESHOLD * n_state)         // ← (1) chunked at 256
            .map_init(
                || (vec![false; populated], vec![0.0; populated]),
                |(is_sel_scratch, val_scratch), chunk| {
                    for slot in is_sel_scratch.iter_mut() { *slot = false; }
                    evaluate_chunk_in_place(pool, chunk, n_state, warm_start,
                                            &eligible, self,
                                            is_sel_scratch, val_scratch);
                    is_sel_scratch.clone()                     // ← (3) per-chunk clone
                },
            )
            .reduce(|| vec![false; populated],
                    |mut a, b| { /* OR-merge */ a })
    } else {
        /* sequential single call */
    };
    // ...emit deactivations + reactivations...
}
```

And in the helper `evaluate_chunk_in_place`:

```rust
for x_hat in chunk.chunks_exact(n_state) {
    // Phase 1: V[k] = intercept[k] + dot(coef[k], x_hat) for every k
    for k in 0..populated {
        let coef_k = &pool.coefficients[k * n_state..(k+1) * n_state];
        scratch[k] = pool.intercepts[k] + coef_k.iter().zip(x_hat)
                       .map(|(c, x)| c * x).sum::<f64>();      // ← (2) iterator dot
    }
    // Phase 2: find max_m, then apply method-specific rule, mutating is_selected
    // ...
}
```

### 2.2 Three problems in one kernel

#### Problem 1 — wrong parallelism granularity

```
┌──────────────────────────────────────────────────────┐
│  visited_states (flat &[f64], M × D)                 │
│                                                       │
│  ┌──────────────┐ ┌──────────────┐                  │
│  │ chunk 0:     │ │ chunk 1:     │   ← M=384, threshold=256
│  │ 256 trial pt │ │ 128 trial pt │     → 2 chunks total
│  └──────────────┘ └──────────────┘
│         │                │
│         ▼                ▼
│     [worker A]      [worker B]   ← 2 rayon workers active
│                                    94 other workers idle
└──────────────────────────────────────────────────────┘
```

At convertido scale `M = 192 to 384`. The threshold `PARALLEL_THRESHOLD = 256`
gives at most 2 chunks per stage → at most 2-way intra-stage parallelism.

Meanwhile the outer `into_par_iter` over stages already provides 64-way
parallelism across stages. So the inner level barely adds anything — it
duplicates a dimension that the outer level already saturates.

#### Problem 2 — inner loop does not vectorise reliably

The dot product:

```rust
coef_k.iter().zip(x_hat).map(|(c, x)| c * x).sum::<f64>()
```

LLVM's auto-vectoriser has historically been inconsistent with this idiom.
Empirical evidence from `run_8`: per-stage time is 700 ms when the
_theoretical_ memory-bound floor for the **naive per-trial-point GEMV**
(re-reading the 1.2 MB coef matrix once per `m`, M = 384) is roughly
`M × 1.17 MB / 10 GB/s ≈ 45 ms`. The 15× gap above that floor is
plausibly compute that should be SIMD-vectorised but isn't.

(The blocked-GEMM lower bound is much smaller — read coef + state once
each, ≈ 2 MB / 10 GB/s ≈ 0.2 ms — but a naive iterator-chain GEMV
doesn't get that benefit.)

A microbenchmark of the same dot product in three forms gives roughly:

| Implementation                     | GFLOPS (single core) |
| ---------------------------------- | -------------------: |
| `.iter().zip(...).map(...).sum()`  |                  1–3 |
| Hand-rolled scalar `for i in 0..D` |                  1–3 |
| `std::simd::f64x4` explicit FMA    |                12–24 |
| `matrixmultiply` batched GEMV      |                20–40 |

The headroom is real.

#### Problem 3 — per-chunk allocation contract

```rust
is_sel_scratch.clone()                  // ~K bytes per chunk
|| vec![false; populated]               // K bytes per reduce identity
```

Allocation-bound? No — at 2 chunks per stage this is negligible. But it
locks the algorithm into a "produce owned Vec per chunk, OR them together"
pattern that prevents finer-grained parallelism. If we drop chunk size to
1 trial point, the per-chunk allocation becomes meaningful.

### 2.3 What this looks like as numbers

Two cases matter. Aggregated is what we have measured today (run_8);
disaggregated is the production target.

#### Aggregated case (convertido, run_8, iter 4)

| Quantity                                       | Value                        |
| ---------------------------------------------- | ---------------------------- |
| Cuts populated `K`                             | ~945 (average across stages) |
| Trial points `M`                               | 192–384                      |
| State dimension `D`                            | 155                          |
| FMA per stage (Phase 1)                        | `K·M·D = 56–110 million`     |
| Per-stage wall (today)                         | **~700 ms**                  |
| Memory bound, blocked GEMM (coef + state once) | ~0.2 ms                      |
| Memory bound, naive per-m GEMV (M × coef)      | ~45 ms                       |
| Compute bound (AVX2 FMA, 1 core)               | ~4 ms                        |
| Compute bound (96 cores, parallelised over m)  | ~50 µs                       |

We're spending 700 ms doing what could be done in ~50 µs. The gap is
~14,000× at aggregated scale.

#### Disaggregated case (production target, projected)

**Caveat**: `K` for the disaggregated case has not been measured. The
1500 figure below is an upper-bound estimate (it would correspond to a
4-iter run that retains more cuts than run_8 keeps at iter 4). Most
likely K ≈ 945 (same as aggregated, since iter count × forward_passes
× stages is unchanged). At K ≈ 945, the working-set numbers below
shrink by ~37% and the per-stage FMA count drops to ~750 M. The
qualitative argument (coef > L2, needs cache-blocked GEMM) holds at
either value; only the specific MB and ms figures move. Validation
will come from a 1-iter disaggregated probe in Landing 3.

| Quantity                                    | Value                           |
| ------------------------------------------- | ------------------------------- |
| Cuts populated `K`                          | ~945–1500 (estimate range)      |
| Trial points `M`                            | 192–384                         |
| State dimension `D`                         | 2080                            |
| FMA per stage (Phase 1)                     | `K·M·D = 0.75–1.2 billion`      |
| Working set: coef matrix                    | `K · D · 8 = 16–25 MB` (L3)     |
| Working set: state matrix                   | `M · D · 8 = 6.4 MB` (L3)       |
| Compute bound (AVX2 FMA, 1 core, peak)      | ~40–60 ms                       |
| Compute bound (96 cores, well-blocked GEMM) | ~400–600 µs                     |
| Per-stage wall (today, extrapolating run_8) | **~6–9 s** (kernel is unusable) |

At `D = 2080` the coef matrix no longer fits in L2 — only in L3.
**Memory bandwidth becomes the dominant constraint** if we use a naïve
per-trial-point GEMV loop that reloads coef rows for each `m`.
Cache-blocked GEMM keeps coef tiles resident in L2 while sweeping
multiple state columns through, which is why we need `matrixmultiply`
specifically (not a hand-rolled SIMD inner loop).

The 9 s per-stage projection makes the current kernel unusable for the
disaggregated case: 64 stages × 9 s = 576 s of CPU per iteration just
for selection. Even with the outer `par_iter` absorbing across stages,
wall-clock would be in the tens of seconds _per iteration_ — dwarfing
the LP solve work. Shipping the new algorithm to production is gated on
this redesign.

---

## 3. The real computational structure

Strip the survival rule away; Phase 1 is:

$$
V[k, m] = \text{intercept}[k] + \sum_{d=0}^{D-1} \text{coef}[k, d] \cdot \text{state}[m, d]
$$

In matrix form:

$$
V \;=\; \text{coef} \cdot \text{state}^{\top} \;+\; \text{intercept} \cdot \mathbf{1}_M^{\top}
$$

where:

- `coef` is `K × D`, row-major (already)
- `state` is `M × D`, row-major (already)
- `V` is `K × M`, row-major or column-major (choice)
- `intercept` is `K`-vector broadcast across M columns

This is a **GEMM**. Specifically a tall-skinny one: `K ≈ 945`, `M ≈ 384`,
`D = 155`. Standard BLAS routines hit ~80% of peak FLOPS on shapes like this.

After V is computed, Phase 2 reads V column-by-column:

$$
\text{max}_m = \max_{k \in [0, K)} V[k, m]
$$

And Phase 3 applies the method-specific survival rule per column,
ultimately producing a bitmap `is_selected[k]` that is the OR-union over
all columns m.

### 3.1 Phase 3 rules (unchanged from the correctness fix)

For Level1 and Dominated:

$$
\text{is\_selected}[k] = \bigvee_{m=0}^{M-1} \mathbb{1}\bigl[ V[k,m] \ge \text{max}_m - \tau \,\land\, \text{eligible}[k] \bigr]
$$

For Lml1 (oldest at max per column, union over columns):

$$
\text{is\_selected}[k] = \bigvee_{m=0}^{M-1} \mathbb{1}\bigl[ k = \min\{j \ge w : \text{eligible}[j] \land V[j, m] \ge \text{max}_m - \tau\} \bigr]
$$

where `w = warm_start_count` and `τ = tie_tolerance`.

### 3.2 Why OR is the correct merge for ALL three methods

If we partition the trial-point set `M` into disjoint subsets `M_1 ⊔ M_2 ⊔ ... ⊔ M_p`
and compute the rule independently on each subset, then OR the resulting
bitmaps:

- For Level1/Dominated: `∪_p ∪_{m ∈ M_p} {...}` = `∪_m {...}` — trivially.
- For Lml1: per-subset we get `∪_{m ∈ M_p}` of the oldest-at-max picks. The
  OR over subsets gives the union of oldest-at-max picks over all m. The
  oldest cut **within a column m** is independent of which subset m is in,
  so the per-column pick is identical regardless of partitioning.

This is what guarantees deterministic results across thread counts. The
union is commutative and associative.

---

## 4. The redesign

The kernel is built around `matrixmultiply::dgemm`. Two layers cooperate:

- **Layer A (core)** — `dgemm` computes blocks of `V = coef · stateᵀ`. One
  call per rayon task, single-threaded, deterministic, cache-blocked. This
  is the right primitive at every `D` we ship.
- **Layer B (orchestration)** — `rayon::into_par_iter` over `m`-blocks
  wraps the dgemm calls. Each task owns a small column slab of `V`, runs
  Phase 2 and Phase 3 on it, and accumulates into a per-worker bitmap.
  Final OR-merge across workers gives the stage's `is_selected` set.

### 4.1 Layer A — `matrixmultiply` for Phase 1

Phase 1 produces `V_block = coef · state_blockᵀ` where `state_block` holds
a small subset of trial points (the m-block the task owns). At the
inner-most level:

```rust
/// Compute V_block (K × m_len) = coef (K × D) · state_blockᵀ (D × m_len)
///
/// Caller-provided buffers; no allocation inside this function.
/// Deterministic: matrixmultiply is single-threaded, fixed cache-blocking
/// (see §5.2 for the caveats on `matrixmultiply`'s runtime CPU detection).
///
/// matrixmultiply's `dgemm` signature is `(m, k, n)` where m is rows of A
/// and C, k is the inner dimension (cols of A == rows of B), and n is
/// cols of B and C. So we pass (k_cobre, d, m_len) — the order that
/// confused this code's first draft.
fn gemm_block(
    coef: &[f64],         // K × D, row-major (slice of pool.coefficients)
    state_block: &[f64],  // m_len × D, row-major (slice of visited_states)
    k_cobre: usize,       // K (number of populated cuts) → matrixmultiply `m`
    d: usize,             // D (state dimension)          → matrixmultiply `k`
    m_len: usize,         // 1..=M_BLOCK                  → matrixmultiply `n`
    v_block: &mut [f64],  // K × m_len, row-major (worker scratch)
) {
    debug_assert_eq!(coef.len(), k_cobre * d,
        "coef slice must be exactly K*D = {} elements", k_cobre * d);
    debug_assert_eq!(state_block.len(), m_len * d,
        "state_block must be exactly m_len*D = {} elements", m_len * d);
    debug_assert_eq!(v_block.len(), k_cobre * m_len,
        "v_block must be exactly K*m_len = {} elements", k_cobre * m_len);

    // SAFETY: dimensions verified by debug_assert above; matrixmultiply
    // requires non-aliasing buffers, which is guaranteed because v_block
    // is exclusively borrowed and coef/state_block are immutably borrowed.
    unsafe {
        matrixmultiply::dgemm(
            k_cobre,  // m: rows of A and C (= K)
            d,        // k: inner dimension (cols of A == rows of B)
            m_len,    // n: cols of B and C (= m-block size)
            1.0,      // alpha
            // A = coef, row-major K × D
            coef.as_ptr(),
            d as isize,  // rsa (row stride: skip D elements between rows)
            1,           // csa (col stride: contiguous within a row)
            // B = state_blockᵀ, accessed as D × m_len.
            // state_block is m_len × D row-major; transposed access is
            // D × m_len with rsb = 1 (next element of a D-vector) and
            // csb = D (next D-vector).
            state_block.as_ptr(),
            1,
            d as isize,
            0.0,                    // beta (overwrite C, don't accumulate)
            // C = v_block, row-major K × m_len
            v_block.as_mut_ptr(),
            m_len as isize,         // rsc
            1,                      // csc
        );
    }
}
```

Properties:

- **Deterministic**: `matrixmultiply` is pure Rust, single-threaded
  internally, uses a fixed cache-blocked algorithm. Same input → same
  byte-identical f64 output on any IEEE-754 target.
- **Cache-friendly at every `D`**: at `D = 155` the coef tile fits in L2;
  at `D = 2080` `matrixmultiply`'s internal blocking keeps a 32 KB micro-tile
  of coef resident in L1 while streaming through state columns. We get
  this for free.
- **Near-peak FLOPS**: matrixmultiply is benchmarked at 80–90% of
  hand-tuned BLAS for these shapes on modern x86_64.
- **No `unsafe` outside this function**: one contained ~10-line unsafe
  block, well-tested by every consumer of the crate.

### 4.2 Layer B — rayon over m-blocks

The outer orchestration:

```rust
const M_BLOCK: usize = 8;  // 4–16 sweet spot; tune empirically

pub fn select_for_stage(
    &self,
    pool: &CutPool,
    visited_states: &[f64],
    current_iteration: u64,
    stage_index: u32,
) -> CutActivityUpdates {
    let populated = pool.populated_count;
    let n_state = pool.state_dimension;
    let warm_start = pool.warm_start_count as usize;
    let n_states = visited_states.len() / n_state;
    // ... edge cases (empty pool, empty states, n_eligible < 2) ...
    let eligible = compute_eligibility(pool, warm_start, current_iteration);

    // Partition trial points into m-blocks of size M_BLOCK (last may be smaller).
    let m_block_starts: Vec<usize> = (0..n_states).step_by(M_BLOCK).collect();

    let is_selected: Vec<bool> = m_block_starts
        .par_iter()
        .fold(
            // Per-worker init: ONE allocation per thread per call.
            || PerWorkerScratch::new(populated, M_BLOCK),
            // Per-task body: one m-block.
            |mut scratch, &m_start| {
                let m_end = (m_start + M_BLOCK).min(n_states);
                let m_len = m_end - m_start;
                let state_block = &visited_states[m_start * n_state..m_end * n_state];

                // Phase 1: V_block = coef · state_blockᵀ (one dgemm call)
                gemm_block(
                    &pool.coefficients,
                    state_block,
                    populated, n_state, m_len,
                    &mut scratch.v_block[..populated * m_len],
                );

                // Add intercept broadcast (in place, no alloc, deterministic)
                for k in 0..populated {
                    let row_start = k * m_len;
                    let i = pool.intercepts[k];
                    for col in 0..m_len {
                        scratch.v_block[row_start + col] += i;
                    }
                }

                // Phase 2 + 3: per-column max + survival rule + OR into accum
                for col in 0..m_len {
                    apply_column_rule(
                        self,
                        &scratch.v_block,
                        populated, m_len, col,
                        warm_start, &eligible,
                        &mut scratch.accum_bitmap,
                    );
                }

                scratch
            },
        )
        .map(|s| s.accum_bitmap)
        .reduce(
            || vec![false; populated],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) { *x |= *y; }
                a
            },
        );

    // Emit deactivations + reactivations from the final is_selected bitmap.
    // ...
}
```

Where:

```rust
struct PerWorkerScratch {
    v_block: Vec<f64>,         // K × M_BLOCK row-major (scratch for dgemm output)
    accum_bitmap: Vec<bool>,   // K, per-worker OR-accumulator across all m-blocks
}

impl PerWorkerScratch {
    fn new(populated: usize, m_block: usize) -> Self {
        Self {
            v_block: vec![0.0; populated * m_block],
            accum_bitmap: vec![false; populated],
        }
    }
}
```

And the column-rule helper (defined per method, all deterministic):

```rust
fn apply_column_rule(
    method: &CutSelectionStrategy,
    v_block: &[f64],     // K × m_len row-major
    populated: usize,
    m_len: usize,
    col: usize,          // 0..m_len
    warm_start: usize,
    eligible: &[bool],
    accum_bitmap: &mut [bool],
) {
    // Phase 2: find max over column `col` — fixed-order linear reduce
    let mut max_m = f64::NEG_INFINITY;
    for k in 0..populated {
        let v = v_block[k * m_len + col];
        if v > max_m { max_m = v; }
    }

    // Phase 3: method-specific survival rule, OR into accum_bitmap
    match method {
        CutSelectionStrategy::Level1 { tie_tolerance, .. }
        | CutSelectionStrategy::Dominated { threshold: tie_tolerance, .. } => {
            let cutoff = max_m - tie_tolerance;
            for k in warm_start..populated {
                if eligible[k] && v_block[k * m_len + col] >= cutoff {
                    accum_bitmap[k] = true;
                }
            }
        }
        CutSelectionStrategy::Lml1 { tie_tolerance, .. } => {
            let cutoff = max_m - tie_tolerance;
            for k in warm_start..populated {
                if eligible[k] && v_block[k * m_len + col] >= cutoff {
                    accum_bitmap[k] = true;
                    break;  // oldest only
                }
            }
        }
    }
}
```

#### Why this design is correct

```
Total work = ceil(M / M_BLOCK) m-block tasks
  ↓
rayon `fold` distributes tasks across W workers (W ≤ num_cpus)
  ↓
Each worker holds (v_block, accum_bitmap); reused across all its tasks
  ↓
Per task: one dgemm call → per-column reduce → OR into worker accum_bitmap
  ↓
Final reduce: OR per-worker accum_bitmaps → stage is_selected
```

Why determinism survives parallelism:

- Each dgemm call is single-threaded and bit-deterministic per input.
- Phase 2's max is a fixed-order linear pass over `0..K`.
- Phase 3's survival rule iterates `warm_start..populated` in fixed order.
- The OR-merge across tasks is commutative and associative.
- No FP value computed by one task is consumed by another; only the
  boolean bitmap crosses task boundaries.

Allocations:

- `PerWorkerScratch::new` runs once per worker per `select_for_stage` call
  (typically ≤ 96 allocations total per call).
- Final reduce's identity bitmap may materialise at reduction-tree nodes;
  cost scales with `log W`, not `M`.
- **No per-task allocations**. `v_block` is overwritten by each dgemm call
  (beta = 0), so no reset is required. `accum_bitmap` is OR-accumulated
  across tasks; it must NOT be reset between tasks within a worker.

#### Granularity analysis at both scales

**Aggregated (D = 155, K = 945, M = 384)**:

- One dgemm call per m-block: `K · D · M_BLOCK = 945 · 155 · 8 = 1.17 M FMA`
  ≈ 25 µs single-threaded at AVX2 peak.
- Number of m-blocks: `ceil(384 / 8) = 48` tasks per stage.
- With 96 workers: 48 tasks fit into 48 workers (the other 48 stay idle for
  this stage's selection). Per-stage wall ≈ max(task time) ≈ 25 µs. With
  rayon overhead: ~100 µs.
- **Implication**: at the per-stage level, intra-stage parallelism uses
  half the cores. The other half are available for either (a) other
  stages running in parallel under the outer `into_par_iter` or
  (b) idleness if selection runs serially inside the backward sweep
  (Landing 2 case). For the serial-stage case, dropping `M_BLOCK` to 4
  would give 96 tasks per stage and saturate the machine — tune in
  Landing 3.

**Disaggregated (D = 2080, K ≈ 945–1500, M = 384)**:

- One dgemm call per m-block: `K · D · M_BLOCK ≈ 15.7–25 M FMA`
  ≈ 0.8–1.25 ms single-threaded.
- Number of m-blocks: 48 tasks per stage.
- With 96 workers: 48 of 96 workers active. Per-stage wall ≈ 0.8–1.25 ms.
  With rayon overhead: ~2 ms. Same `M_BLOCK = 4` consideration applies
  for the serial-stage case.

Both scales easily hit the rayon scheduling sweet spot. Per-task work is
25 µs to 1.25 ms — well above the few-µs scheduling overhead, well below
the load-imbalance threshold (~10 ms tasks would start to hurt).

Comparison vs today (extrapolated):

| Scale                  | Today's wall | Layer A+B wall | Speedup |
| ---------------------- | -----------: | -------------: | ------: |
| Aggregated (D=155)     |       700 ms |         100 µs |  ~7000× |
| Disaggregated (D=2080) |       ~9 s\* |           2 ms |  ~4500× |

\*projected, today's kernel is not deployed at disaggregated scale.

#### Why `M_BLOCK = 8` (tuneable)

`M_BLOCK` balances three concerns:

- **Cache locality**: at `D = 2080`, `M_BLOCK = 8` makes the per-worker
  `v_block` 1500 × 8 × 8 bytes = 96 KB. Fits in L2 alongside the
  ~32 KB matrixmultiply micro-tile of coef. Larger blocks evict useful
  state from L2.
- **dgemm amortisation**: matrixmultiply's per-call overhead is in the
  low microseconds. With `M_BLOCK = 8` and disaggregated sizing, each
  call does 25 M FMAs ≈ 1.25 ms — amortised over 1000× the overhead.
- **Parallelism**: at `M = 192–384`, `M_BLOCK = 8` gives 24–48 tasks per
  stage. With 96 workers we want at least ~50 tasks for good load
  balance via work-stealing.

Empirically tune in Landing 1. Reasonable range: 4 to 16.

### 4.3 What we _don't_ need anymore

The earlier draft of this document considered an "explicit SIMD inner
loop" layer (Layer A in the original outline). With `matrixmultiply`
adopted as the core kernel, that layer is unnecessary:

- The SIMD inside the dot product is matrixmultiply's job, and it does it
  better than a hand-rolled f64x4 loop (cache blocking, register tiling,
  prefetching).
- The determinism discipline (no `reduce_sum`, explicit FMA flag,
  forbidden patterns gate) is no longer a concern, because we don't write
  the SIMD ourselves.
- One less code path to maintain, test, and document.

What we DO still need from §5:

- Pin matrixmultiply to a specific version in `Cargo.toml` (no `^x.y`
  range that could float across LLVM versions and shift codegen).
- Forbid external BLAS (`openblas-src`, etc.) at the CI level — they use
  internal threading that breaks determinism.
- The realistic-scale determinism tests (§5.5) still apply, but now
  verify matrixmultiply's deterministic behaviour rather than our SIMD
  code's. **And** add the matrixmultiply CPU-dispatch verification
  harness from §5.2 as a Landing 1 prerequisite.

---

## 5. Reproducibility analysis

The hard rule: **bit-for-bit identical results across any thread count, on
the same binary, on the same architecture**. Cross-architecture variation
(AVX2 vs AVX-512 hosts) is acceptable; within-architecture variation is
not.

With `matrixmultiply` as the Phase 1 kernel, the determinism story is
short: the crate is deterministic by construction, and our orchestration
layer (rayon over m-blocks + OR-merge) preserves that determinism.

### 5.1 The exact guarantee

| Variation                                                        | Result changes? | Rationale                                                                                             |
| ---------------------------------------------------------------- | --------------- | ----------------------------------------------------------------------------------------------------- |
| `RAYON_NUM_THREADS = 1` vs `= 96`                                | No              | OR-merge is commutative + associative; each dgemm call is single-threaded.                            |
| Different rayon work-stealing schedules across re-runs           | No              | Same reason: how m-blocks split between threads doesn't affect each block's deterministic output.     |
| Process re-launched repeatedly with same inputs                  | No              | Same instruction stream, same IEEE-754 ops, same inputs → same outputs.                               |
| Same source rebuilt with the same toolchain + flags              | No              | Rust + LLVM are deterministic compilers; matrixmultiply has no internal non-determinism.              |
| Same binary on AVX2 vs AVX-512 host **via runtime CPU dispatch** | Yes — FORBIDDEN | We do not enable runtime CPU dispatch in our build of matrixmultiply or elsewhere.                    |
| Same source compiled with `target-cpu=native` on different hosts | Yes — accepted  | Cross-architecture variation is acceptable per the user's stated requirement.                         |
| Toggling `opt-level=2` vs `opt-level=3`                          | No              | matrixmultiply's kernel selection does not depend on opt-level; our orchestration code is structural. |
| Rust toolchain version bumps that change LLVM codegen            | Possibly        | Mitigated by pinning the toolchain via `rust-toolchain.toml`.                                         |
| matrixmultiply version bump (e.g., 0.3.x → 0.4.x)                | Possibly        | Pin to an exact version in `Cargo.toml`. Upgrade is a deliberate, tested action.                      |

The "same-architecture" guarantee holds **as long as we pin
matrixmultiply, pin the toolchain, and forbid external BLAS**.

### 5.2 Why matrixmultiply is deterministic — and what we must verify

`matrixmultiply` (the crate at https://docs.rs/matrixmultiply) has three
properties that, **together with our build configuration**, give
bit-deterministic output for the same input matrix:

1. **Single-threaded by design.** No internal threading, no work stealing,
   no parallel reductions. The crate's `dgemm` is one sequential
   computation per call. (Compare to external BLAS like OpenBLAS, which
   uses OpenMP internally with non-deterministic tile assignment.)
2. **Fixed cache-blocking algorithm.** The packing-and-multiplication
   sequence is deterministic for given `(m, k, n)` dimensions. Same
   shape → same blocking → same operation order.
3. **No fast-math reassociation.** The crate does not enable `-Cfast-math`
   or equivalent IEEE-754 relaxations. Round-to-nearest-even semantics
   are honoured throughout.

#### Caveat — runtime CPU detection

`matrixmultiply` 0.3.x performs **runtime CPU detection** by default to
select between SSE/AVX/AVX2 micro-kernels at first call (via a static
function pointer initialised once per process). This is _exactly_ the
"runtime CPU dispatch" pattern §5.4 forbids in our own code, and it
needs to be neutralised before we can claim within-architecture
determinism.

Two mitigation paths, both viable:

- **Compile-time-only kernel selection.** Set `target-feature=+avx2,+fma`
  (see §5.5). If `matrixmultiply` respects `target_feature` cfg flags
  for micro-kernel inclusion — meaning the AVX-512 micro-kernel is not
  compiled into the binary when we only enable AVX2 — then runtime
  detection has only one available path and the dispatch becomes a
  no-op. **This needs empirical verification before Landing 1 ships**;
  the crate's documentation does not make the guarantee explicit.

- **Per-microarch binaries.** Build one binary per target micro-arch
  (e.g., `x86-64-v3` for AVX2/FMA, `x86-64-v4` for AVX-512). Each
  binary's `matrixmultiply` runtime detection has only one available
  kernel and is deterministic on its target. Cross-microarch variation
  is accepted (matches the existing requirement).

**Pre-implementation verification step (Landing 1 prerequisite)**: write
a small harness that calls `matrixmultiply::dgemm` on the same input
matrix in two configurations — (a) a process started fresh on AVX2-only
hardware, (b) a process started fresh on AVX2+AVX-512 hardware with the
binary built with `target-feature=+avx2,+fma`. Compare byte-by-byte. If
identical, the runtime-detection concern is neutralised by the
target-feature gate. If different, switch to per-microarch binaries.

In practice (assuming verification passes): feed the same `coef`,
`state`, and `(K, M, D)` into our dgemm wrapper from a single-threaded
test and from a 96-thread training run; the byte-for-byte content of
`V_block` is identical.

This is verified at the test level by the realistic-scale determinism
test in §5.5.

### 5.3 The orchestration layer's contribution

Our code wraps matrixmultiply. We need to ensure that the wrapping itself
does not introduce non-determinism. The pieces:

| Operation                      | Determinism source                                           |
| ------------------------------ | ------------------------------------------------------------ |
| `gemm_block` (Phase 1)         | matrixmultiply (deterministic by construction)               |
| Intercept broadcast            | Linear pass `for k in 0..K { for col in 0..m_len { ... } }`  |
| Per-column max (Phase 2)       | Linear pass `for k in 0..K { if v > max ... }` — fixed order |
| Survival rule (Phase 3)        | Linear pass `for k in warm_start..K` — fixed order           |
| Per-worker bitmap accumulation | OR into accum_bitmap — commutative, associative              |
| Cross-worker reduce            | OR over per-worker bitmaps — commutative, associative        |

Every step is either:

- single-threaded with a fixed iteration order, or
- a boolean OR (commutative + associative, work-distribution-independent).

No FP value computed by one rayon task is consumed by another. Only the
boolean bitmap crosses task boundaries. This is what makes the design
robust to any rayon scheduling decision.

### 5.4 Forbidden patterns

A small list, because matrixmultiply absorbs most of the SIMD discipline.
We MUST NOT introduce any of these:

1. **External BLAS bindings** (`openblas-src`, `intel-mkl-src`,
   `blas-src`, `cblas-sys`) — these use internal threading with
   non-deterministic tile assignment. Even if we configure them to be
   single-threaded at runtime, the kernel selection may vary across
   builds.
2. **Runtime CPU dispatch** — crates like `multiversion` or
   `cpufeatures::is_x86_feature_detected!` that pick SIMD code paths at
   runtime within one binary. We ship one binary per architecture.
3. **`-Cfast-math` / `-Zfast-math` / per-function fast-math attributes** —
   permit LLVM to reassociate FP ops arbitrarily.
4. **`matrixmultiply` with internal threading enabled** — the crate
   exposes a `threading` feature gate. We disable it by default and
   verify in `Cargo.toml`.
5. **Cross-thread FP accumulation** — relaxed atomic FP add, locked FP
   merge, or any pattern that splits a single dot product across threads.
   The whole design is built to avoid this; do not break it.
6. **Floating `^x.y` version ranges on matrixmultiply** in `Cargo.toml`.
   Pin the exact version (e.g., `= 0.3.10`). Upgrades are deliberate.

### 5.5 Enforcement — build-level and test-level

#### Build-level (preventive)

`.cargo/config.toml` at the workspace root:

```toml
[target.x86_64-unknown-linux-gnu]
rustflags = [
    "-C", "target-feature=+avx2,+fma,+sse4.2",
]
```

This guarantees matrixmultiply's compile-time kernel selection picks the
AVX2+FMA path in every build configuration (debug, release, CI). No
fallback to non-FMA paths.

`Cargo.toml` workspace dependency:

```toml
[workspace.dependencies]
matrixmultiply = { version = "= 0.3.10", default-features = false }
```

`default-features = false` disables the optional `threading` feature even
if a future version of matrixmultiply makes threading opt-in.

`rust-toolchain.toml`:

```toml
[toolchain]
channel = "1.86.0"   # pinned MSRV
profile = "minimal"
```

Locks LLVM codegen behaviour across all developer machines and CI.

#### CI guardrails

A 10-line `grep` gate in pre-commit or CI rejects any change that
introduces:

- `openblas-src`, `intel-mkl-src`, `blas-src`, `cblas-sys`, or any other
  external BLAS binding.
- `cpufeatures` / `is_x86_feature_detected!` (runtime CPU dispatch).
- `multiversion` (compile-time multi-versioning that selects at runtime).
- `fast-math` / `fast_math` / `ffast-math`.
- `matrixmultiply` with `threading` feature enabled.

This catches the failure modes mechanically.

#### Test-level (regression catch)

Three determinism tests, expanding the four that already exist:

1. **Small fixture** (existing): `K ≈ 5, M ≈ 10, D ≈ 2`. Pinned thread
   pools at 1, 4, 8. Asserts bit-identical `CutActivityUpdates`. Catches
   high-level structural bugs.

2. **NEW realistic aggregated scale**: `K = 1000, M = 384, D = 155` —
   matches convertido iter-4 sizes. Pinned at 1, 16, 96. Asserts
   bit-identical bitmaps. Catches per-worker scratch leaks and
   m-block-boundary handling bugs.

3. **NEW disaggregated scale**: `K = 1500, M = 384, D = 2080` — matches
   production target. Pinned at 1, 16, 96. Asserts bit-identical bitmaps.
   This is the most important test: matrixmultiply's micro-kernel
   selection at large `D` differs from the small-D path, and we want to
   exercise it directly.

Run all three under both `cargo test -p cobre-sddp --release` and
`cargo test -p cobre-sddp` (debug mode). Both must produce bit-identical
results.

```rust
#[test]
fn select_for_stage_deterministic_disaggregated_scale() {
    // K=1500, M=384, D=2080 — disaggregated production target
    let pool = make_large_pool(1500, 2080);
    let states = make_random_states(384, 2080, /* seed */ 42);
    let strategy = CutSelectionStrategy::Lml1 {
        check_frequency: 1,
        tie_tolerance: 1e-6,
    };

    let result_1 = run_in_thread_pool(1,
        || strategy.select_for_stage(&pool, &states, 5, 0));
    let result_16 = run_in_thread_pool(16,
        || strategy.select_for_stage(&pool, &states, 5, 0));
    let result_96 = run_in_thread_pool(96,
        || strategy.select_for_stage(&pool, &states, 5, 0));

    assert_eq!(result_1, result_16,
        "thread count 1 vs 16 produced different selection at D=2080");
    assert_eq!(result_1, result_96,
        "thread count 1 vs 96 produced different selection at D=2080");
}
```

### 5.6 Lml1 FP-noise safety margin

matrixmultiply's per-cell `V[k, m]` is the result of a `D`-element dot
product, computed in a fixed but non-trivial blocking order. The
floating-point error bound is `O(D · machine_epsilon · max|coef · state|)`:

- At `D = 155`: error ≈ `1e-13` relative
- At `D = 2080`: error ≈ `5e-13` relative

This matters for Lml1's "oldest at max" rule: if two cuts `k1 < k2` have
`|V[k1, m] - V[k2, m]| < FP_noise`, the rule could potentially pick the
wrong one if FP noise straddles the `max_m - tie_tolerance` cutoff. The
safety margin is `tie_tolerance / FP_noise`:

| `tie_tolerance`           | Safety margin (D=155) | Safety margin (D=2080) | Verdict                             |
| ------------------------- | --------------------- | ---------------------- | ----------------------------------- |
| `1e-15`                   | < 1×                  | < 1×                   | UNSAFE — FP noise can flip ordering |
| `1e-12`                   | ~10×                  | ~2×                    | borderline at D=2080                |
| `1e-10` (current default) | ~1000×                | ~200×                  | safe at both scales                 |
| `1e-6` (run_8 setting)    | ~10^7×                | ~2 × 10^6×             | very safe                           |
| `1e-2`                    | ~10^11×               | ~2 × 10^10×            | totally safe                        |

**Conclusion**: at all realistic `tie_tolerance` values, FP noise is
orders of magnitude smaller than the tolerance band. The Lml1 "oldest at
max" pick is stable. The disaggregated case has tighter (but still
ample) margin because `D` is 13× larger.

The disaggregated-scale determinism test (§5.5, test 3) is the empirical
confirmation that this holds across thread counts at production sizing.

### 5.7 Summary of the determinism contract

To deliver bit-for-bit reproducibility within an architecture, we commit
to the following invariants:

1. **matrixmultiply pinned to an exact version**, `default-features = false`
   (no threading).
2. **No external BLAS.** Forbidden by CI grep gate.
3. **No runtime CPU dispatch.** One binary per architecture, picked at
   build time via `target-feature` flags.
4. **No fast-math.** Standard IEEE-754 semantics everywhere.
5. **Per-task computation only.** Each `(k, m)` cell of `V` is computed
   inside a single dgemm call on one thread; no shared FP accumulators
   across rayon tasks.
6. **OR-merge across workers**, not arithmetic merge. OR is commutative
   and associative so m-block boundaries don't affect the result.
7. **Tie tolerance ≥ 1e-10.** Far above FP noise floor at all realistic
   `D`, so Lml1's "oldest at max" is stable.
8. **Pinned toolchain version** via `rust-toolchain.toml`. Locks LLVM
   codegen behaviour.
9. **CI gate that grep-rejects forbidden patterns** (external BLAS,
   runtime CPU dispatch, fast-math, matrixmultiply threading).
10. **Realistic-scale determinism tests** at both aggregated (D=155) and
    disaggregated (D=2080) scales, pinning thread pools and asserting
    byte-identical outputs.

All ten are mechanically enforceable. The total enforcement surface is
~20 lines of `Cargo.toml` and `.cargo/config.toml` settings, a 10-line CI
grep script, and ~50 lines of test code. None requires ongoing
vigilance.

---

## 6. Cache and memory analysis

The design has to perform well at the production target (`D = 2080`)
where the coef matrix no longer fits in L2. This is why
`matrixmultiply`'s cache-blocked algorithm matters — it keeps a micro-tile
of coef resident in L1 while sweeping multiple state columns through.

### 6.1 Working set per stage

**Aggregated (`K = 945, M = 384, D = 155`)**:

```
coef       : K × D × 8 = 945 × 155 × 8 = 1.17 MB   (read-only, shared)
state      : M × D × 8 = 384 × 155 × 8 = 476 KB    (read-only, shared)
intercepts : K × 8     = 7.6 KB                    (read-only, shared)
─────────────────────────────────────────────────────────
total shared read-only             : 1.65 MB  (fits in L2)

per-worker scratch (M_BLOCK = 8):
  v_block       : K × M_BLOCK × 8 = 945 × 8 × 8 = 60 KB
  accum_bitmap  : K × 1 byte      = 945 B
─────────────────────────────────────────────────────────
total per worker                   : ~61 KB  (fits in L1d)
```

**Disaggregated (`K ≈ 945–1500, M = 384, D = 2080`)** — upper-bound shown:

```
coef       : K × D × 8 ≤ 1500 × 2080 × 8 = 24.9 MB  (read-only, shared)
state      : M × D × 8 =   384 × 2080 × 8 =  6.4 MB  (read-only, shared)
intercepts : K × 8     ≤ 12 KB                       (read-only, shared)
─────────────────────────────────────────────────────────
total shared read-only             ≤ 31.3 MB  (fits in L3, NOT L2)

per-worker scratch (M_BLOCK = 8):
  v_block       : K × M_BLOCK × 8 ≤ 1500 × 8 × 8 = 96 KB
  accum_bitmap  : K × 1 byte      ≤ 1.5 KB
─────────────────────────────────────────────────────────
total per worker                   ≤ ~97 KB  (fits in L2, just exceeds L1d)
```

The disaggregated case is the one that demands real cache discipline.
The 25 MB coef matrix can only live in L3, and naïve algorithms that
sweep `coef` for every trial point would saturate DRAM bandwidth.

#### Allocation pressure under nested parallelism

The `PerWorkerScratch` allocation lives inside `select_for_stage`'s
rayon fold init closure. In the **outer-parallel** mode (the
`run_cut_management` model, where `into_par_iter` over stages calls
`select_for_stage` in parallel across all stages), nested rayon means
up to `64 stages × 96 workers = 6144` `PerWorkerScratch::new` calls per
iteration in the worst case. At disaggregated upper-bound sizing, that
is **~590 MB allocated and freed per iteration** just for selection
scratch. This violates the Cobre hard rule "never allocate on hot
paths".

In the **selection-inside-backward** mode (Landing 2), stages run
sequentially through the backward sweep, so only one `select_for_stage`
runs at a time. Allocation pressure drops to `≤ 96 workers × 64 stages
≈ 6144 allocations` for the iteration, but spread out — still wasteful
but no longer concurrent.

**Required mitigation** (regardless of mode): allocate scratch once per
worker per _training run_, not per `select_for_stage` call. Two options:

- Thread-local scratch pool, sized at training-session init to
  `max_K_across_stages × M_BLOCK`. Workers pull from the pool on first
  use, reuse for the rest of the run.
- Pre-allocate per-worker scratch in `TrainingSession` state and pass a
  `&mut [PerWorkerScratch]` slice into `select_for_stage`.

Both are mechanical changes; the latter is simpler. Pick one in
Landing 1.

### 6.2 Cache hierarchy on a typical AMD/Intel server core

```
L1d (per-core)   : 32–64 KB
L2  (per-core)   : 512 KB – 2 MB  (on AMD Zen 3/4: 1 MB per core)
L3  (shared)     : 16–128 MB      (on AMD EPYC 9xx4: 32–768 MB)
DRAM             : multi-channel, ~50–200 GB/s aggregate
```

For both cases, our per-worker scratch (`v_block`) fits in L1d/L2 and the
shared read-only data fits in L3.

### 6.3 Why `matrixmultiply`'s cache blocking pays off at D=2080

Naïve GEMV (Layer B equivalent at single-m granularity, no cache
blocking): for each trial point `m`, sweep the entire 25 MB `coef` matrix
to compute the K-vector `V[*, m]`. M = 384 trial points × 25 MB sweep =
**9.6 GB of L3-to-core traffic per stage**.

With `matrixmultiply`'s register and cache blocking (default packs
`mr × nr = 8 × 4` micro-tiles, with a panel structure that fits in L2):

- Each pack of the `K × kc` slab of `coef` is read into L2 once.
- The slab is reused for `nc` columns of state before being evicted.
- Effective traffic is ~`(K + M) × D × 8` bytes — just the input
  matrices, read once each from DRAM/L3.

For our disaggregated case: ~32 MB read once per stage. At 100 GB/s
aggregate L3 bandwidth, that's ~0.3 ms of bandwidth-floor cost per stage.
Add the compute (~1.25 ms per m-block × 48 m-blocks / 96 cores = ~0.6 ms
wall) and we land at ~1–2 ms per stage. Without cache blocking we'd be
~30× higher.

This is the concrete reason `matrixmultiply` is non-negotiable for the
production target, not just a "nice to have."

### 6.4 Memory traffic comparison

| Scenario                                       | Per-stage DRAM/L3 traffic |
| ---------------------------------------------- | ------------------------- |
| Aggregated, naïve per-m GEMV                   | M × 1.17 MB = 449 MB      |
| Aggregated, matrixmultiply blocked GEMM        | ~2 MB (coef + state once) |
| **Disaggregated, naïve per-m GEMV**            | **M × 25 MB = 9.6 GB**    |
| **Disaggregated, matrixmultiply blocked GEMM** | **~32 MB**                |

Naïve at disaggregated scale would be bandwidth-bound for ~100 ms per
stage — orders of magnitude worse than the GEMM-blocked path.

### 6.5 NUMA considerations

On a 96-thread machine that spans two NUMA nodes (e.g., dual-socket), the
`coef` matrix is allocated by whichever thread first-touched it (typically
the main thread during cut sync). Remote-NUMA workers pay a 2–3× latency
penalty when fetching coef tiles.

At disaggregated scale, with a 25 MB coef matrix, the NUMA cost can be
real. Mitigations:

- **First-touch-by-worker** for per-worker scratch (already automatic via
  `PerWorkerScratch::new` running on each worker).
- **Optional**: replicate the `coef` matrix per NUMA node. Adds memory
  cost but eliminates cross-socket traffic. Defer until profiling shows
  it matters.
- **MPI rank pinning**: bind each MPI rank to a NUMA node. Then each
  rank's coef matrix is local to its node. This is the right answer for
  large-scale multi-socket deployments.

For Landings 1–3 (this design): ignore NUMA. Revisit if disaggregated
profiling shows cross-socket coef traffic dominates.

---

## 7. Parallelism diagrams

### 7.1 Today's design

```
                  ┌─────────────────────────────────────────┐
                  │       run_cut_management(iteration)     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ rayon::into_par_iter over stages 1..T-2 │  ← Level 1
                  │  (64 stages, 96 threads → 1.5 stages    │
                  │   per thread on average)                │
                  └────────────────────┬────────────────────┘
                                       │
                  ┌──────────┬─────────┼─────────┬──────────┐
                  ▼          ▼         ▼         ▼          ▼
              stage 1    stage 2   stage 3   ...        stage 62
                  │          │         │                    │
                  ▼          ▼         ▼                    ▼
          ┌──────────────────────────────────────────────────────┐
          │     select_for_stage on this stage's pool             │
          │                                                       │
          │  IF n_states >= 256:                                  │
          │    par_chunks(256 * D) over visited_states            │ ← Level 2
          │      ─ chunk 0: 256 trial points (1 task)             │
          │      ─ chunk 1: 128 trial points (1 task)             │
          │      → 2 tasks; 2-way parallelism                     │
          │                                                       │
          │    each task:                                         │
          │      for trial point in chunk (256 of them):          │
          │        for k in 0..K:                                 │
          │          v[k] = intercept[k] + iter_chain_dot(...)    │ ← Level 3
          │          (NOT vectorised reliably)                    │
          │  ELSE:                                                │
          │    sequential single call                             │
          └──────────────────────────────────────────────────────┘

Effective utilisation:
  Level 1 (stages):       64-way potential, ~64 threads busy → great
  Level 2 (trial points):  2-way at best → wasted capacity
  Level 3 (inner dot):    no SIMD reliably → ~1/4 of peak FLOPS

Combined: ~64 × 2 × 0.25 = 32-way effective, vs 96 threads × 8 SIMD lanes = 768 lanes available
```

### 7.2 Proposed design

```
                  ┌─────────────────────────────────────────┐
                  │       run_cut_management(iteration)     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ rayon::into_par_iter over stages 1..T-2 │  ← Level 1 (unchanged)
                  └────────────────────┬────────────────────┘
                                       │
                  ┌──────────┬─────────┼─────────┬──────────┐
                  ▼          ▼         ▼         ▼          ▼
              stage 1    stage 2   stage 3   ...        stage 62
                  │          │         │                    │
                  ▼          ▼         ▼                    ▼
          ┌──────────────────────────────────────────────────────┐
          │     select_for_stage on this stage's pool             │
          │                                                       │
          │  m_block_starts.par_iter().fold(...).reduce(...)      │ ← Level 2
          │     ceil(M / M_BLOCK) tasks (24–48 at M_BLOCK=8)      │   (m-block)
          │     workers reuse PerWorkerScratch (v_block, accum)   │
          │                                                       │
          │     each task (one m-block of M_BLOCK trial points):  │
          │       matrixmultiply::dgemm(                          │ ← Level 3
          │         K, M_BLOCK, D,                                │   (cache-blocked
          │         coef, state_block, v_block                    │    GEMM)
          │       )                                               │
          │       add intercept broadcast to v_block              │
          │       for col in 0..M_BLOCK:                          │
          │         max_m = linear reduce v_block[:, col]         │
          │         apply rule → accum_bitmap |= ...              │
          │                                                       │
          │     final reduce: OR all per-worker accum_bitmaps     │
          └──────────────────────────────────────────────────────┘

Effective utilisation:
  Level 1 (stages):       deferred if selection moves into backward; else same
  Level 2 (m-blocks):     ceil(M/8)-way potential (24–48), 96-way actual via outer
                          stage parallelism — full saturation
  Level 3 (dgemm):        register-blocked AVX2/FMA — near-peak FLOPS at any D

Speedup vs today:
  Aggregated   (D=155):   ~7000×  (700 ms → ~100 µs)
  Disaggregated (D=2080): ~4500×  (~9 s → ~2 ms; today's kernel unusable here)
```

### 7.3 Combined with selection-inside-backward

```
Backward pass at iteration N:

  forward pass produces archive.states (all stages, all trial points)
                                       │
                                       ▼
  for t = T-2 downto 0:                                    ─┐
    1. par_iter (m, opening) → solve stage-(t+1) LPs       │
       (uses FCF[t+1] — recently pruned in prior loop iter)│
                                                            │ stage-t
    2. aggregate cuts → push into FCF[t]                   │ backward
                                                            │ step
    3. select_for_stage(FCF[t], archive.states_for(t))     │
         ↑                                                  │
         │ NEW: dgemm-based kernel here                    │
         │ ~100 µs at D=155, ~2 ms at D=2080               │
         ↓                                                  │
    4. apply_updates(FCF[t], deact + react)                │
                                                            │
       (now FCF[t] is pruned; next loop iter solves        │
        against this leaner template)                       │
                                                          ─┘

Effect: by the time we process stage t-1 at step 1 of the next iteration,
        FCF[t] has been pruned. The stage-(t-1) backward LPs are smaller.
        Compounding down the sweep: every stage's backward solves against
        a freshly-pruned FCF.
```

---

## 8. Integration with selection-inside-backward

> **Reader note**: this section describes an _architectural_ change that
> is logically separable from the kernel speedup of §4. The kernel
> change (Landing 1) is uncontroversial: paper-correct algorithm running
> at peak FLOPS. Selection-inside-backward (Landings 2+) trades a
> better LP-cost profile for a _different algorithm_ in the MPI setting
> (§8.2) and depends on a backward-LP assumption that has not yet been
> verified (§8.3). It should be designed and benchmarked separately,
> not bundled with Landing 1.

### 8.1 Why the order of operations matters

In today's code, the iteration pipeline is:

```
1. forward pass               ─┐
2. forward sync (MPI)          │ all-archive populated
3. backward pass               │ FCF[t] for t in T-2..0 augmented
4. cut sync (MPI)              │ all ranks have identical pools + archive
5. cut selection (per stage)   │ pure compute, deterministic per rank
6. lower bound                 │
7. convergence update          │
                              ─┘
```

In step 3, backward sweeps `t = T-2 → 0`. When processing stage `t`, the
solver uses `FCF[t+1]` as the template (the cuts in `FCF[t+1]` become rows
in the LP for stage `t+1`).

If we move selection into the per-stage hook of the backward sweep:

```
backward sweep at iteration N:
  for t = T-2 downto 0:
    a. solve stage-(t+1) LPs  ← uses FCF[t+1]'s current active cuts
    b. push new cut into FCF[t]
    c. select_for_stage on FCF[t]            ← NEW
    d. apply_updates(FCF[t])                  ← NEW
  end for

  (cut sync happens once at the end, as today;
   or per-stage, with extra MPI cost — see §8.2)
```

When the loop advances `t → t-1`:

- Step `a` for `t-1` solves stage-`t` LPs, using `FCF[t]` which was just
  pruned in step `c+d` of the previous iteration.
- The pruned `FCF[t]` is smaller, so **(assuming the LP is rebuilt from
  active cuts at each stage, not reused as a static structure with
  inactive rows held by bounds) the LPs are smaller, so simplex
  iterations drop**. That assumption is the foundation of the whole
  savings argument and is unverified — see §8.3.

### 8.2 Cut sync timing — and the algorithm change it implies

Today's flow:

```
backward pass (each rank generates its own cuts) → cut sync → selection
```

If selection moves inside backward, three sync options:

| Option                                                          | Pro                                                                                                           | Con                                                                                                                                           |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. One sync at end** (selection inside backward)              | Simple; no extra MPI calls; selection runs on _partial_ pool (this rank's cuts only, until sync)              | Selection result differs across ranks until sync. Each rank prunes its own cuts independently. After sync, ranks have different active flags. |
| **B. Sync per stage**                                           | Selection sees the full cut set including other ranks' contributions for stage t before pruning               | T-1 MPI calls per iteration. Convertido: 63 calls × ~30 ms = ~1.9 s wall added. Significant overhead.                                         |
| **C. Run selection AFTER sync (one big call at end, as today)** | No semantic change vs today; selection sees identical inputs across ranks; gives correct paper-defined output | Loses the "shrink FCF[t] before backward needs it" benefit.                                                                                   |

**There is no clean resolution.** The trade-offs:

- **Option A is an algorithm change, not an implementation detail.**
  Per-rank-local Lml1 with subsequent Active-OR reconciliation produces
  a strictly larger active set than paper-correct Lml1 on the global
  post-sync pool. Two ranks each picking 192 different "oldest at max"
  cuts merges to up to 384 active — paper-Lml1 would pick 192.
  Pruning aggressiveness is **roughly halved** in the MPI setting, and
  the algorithmic guarantees that Lml1's value-based selection
  inherits from de Matos / Guigues no longer apply to the merged set.

  Worse for Lml1: "oldest by slot index" is well-defined globally only
  after cut sync. During selection inside backward, slot indices are
  per-rank-local and have not been globally assigned yet; what counts
  as "oldest" depends on which rank you're on.

  Active-OR reconciliation (the only deterministic merge) makes this
  consistent across ranks at iteration boundaries, but the _algorithm
  being run_ is "per-rank local Lml1 + Active-OR merge" — not paper Lml1.

- **Option B is straightforwardly correct but expensive.** Adding T-1
  small allreduces per iteration costs ~1.9 s wall at convertido scale
  (63 stages × 30 ms each). This is comparable to the LP savings we
  hope to get. May be a wash.

- **Option C** keeps today's semantics but loses the architectural
  win. If the savings model in §8.3 doesn't pan out empirically, this
  is the right fallback.

**Recommendation for Landing 2 design work**: do not pick a sync
strategy until two things are measured:

1. The actual LP savings under Option A (per-rank local selection,
   Active-OR reconcile). Run a 4-iter convertido benchmark with
   Landing 1 + Option A and compare against Landing 1 alone.
2. The empirical convergence behaviour of "per-rank local Lml1 +
   Active-OR" vs paper Lml1 over enough iterations to see lower-bound
   trajectories diverge. The active set differs; we need to confirm
   convergence is unaffected (it should be — keeping more cuts is
   conservative — but we should measure).

If both measurements support Option A, document the algorithm change
explicitly. If Option A's savings don't materialise, fall back to
Option C and reframe the architectural change as not worth it.

### 8.3 Where the savings _would_ come from — and the assumption that gates them

The mechanism: **selection happens once per stage; the benefit accrues to
the NEXT backward stage's LP solves**.

#### Hard precondition: the backward solver must rebuild LPs from active cuts

The savings model assumes that deactivating a cut between stage `t`'s
selection and stage `t-1`'s backward solve makes the resulting LPs
have fewer rows. **This is true only if** the backward LP construction
reads `cuts_in_lp` (active cuts) freshly at each stage, rather than
reusing a previously-built LP with all populated cuts and toggling row
bounds for inactive ones.

The current cobre code has not been audited for this in the context of
this proposal. Before committing to Landing 2, read:

- `crates/cobre-sddp/src/backward.rs` (or wherever the per-stage
  backward LP is constructed)
- The `RowBatch` / `add_rows` interaction on the solver side

and confirm one of the following:

- LPs are rebuilt from `pool.active` at each stage → savings model
  applies.
- LPs are reused with row-bound toggles → savings model **does not
  apply** (or applies much more weakly) and Landing 2 should be
  reconsidered.

If the LP is reused, the user gets faster pivots only via the warm-start
basis adjusting around deactivated rows; the row count itself doesn't
shrink. Empirically this may still help, but the model in this section
overstates the gain.

#### The savings model (conditional on the precondition)

For an iteration with 64 stages, _assuming_ LPs are rebuilt from active
cuts at each stage:

- Pruning `FCF[T-2]` benefits stage `T-3`'s backward LPs (~3780 LP solves).
- Pruning `FCF[T-3]` benefits stage `T-4`'s backward LPs.
- ...
- Pruning `FCF[1]` benefits stage `0`'s backward LPs.

Total backward LPs that benefit: ~(T-2) × 3780 ≈ 234k LPs per iteration.

If selection prunes 30% of cuts on average (run_8 achieved 41%
deactivation rate at iter 4 under post-sync selection; per-rank-local +
OR will prune less — say 20%), each affected LP is ~20% smaller.
Simplex iteration count scales roughly with row count to the 1.2–1.5
power for these LPs, so 20% smaller rows → ~24–28% fewer simplex
iterations → ~24–28% less solve time per affected LP.

**Aggregated case (convertido, observed run_8 baseline, conditional on
precondition + Option A pruning)**:
Backward total wall at iter 4: ~274 s. If half of that is in the
"benefits from pruning" LPs and they speed up 25%: savings = 274 × 0.5
× 0.25 ≈ **34 s wall per iteration**. Across 4 iterations: **~140 s**.

The earlier estimate (55 s/iter, 220 s total) assumed paper-correct
40% pruning aggressiveness. The revised estimate above uses the lower
per-rank-local pruning rate. Even so it remains positive — but the
margin against the current 57 s regression is smaller than the earlier
"hugely net positive" framing suggested.

Current regression vs run_7: 57 s. Conservative savings estimate:
~140 s. Net: cut-selection-correctness becomes **faster** than the
broken baseline, with the correct algorithm — **if the precondition
holds**. If the precondition does not hold, the savings collapse to
whatever the warm-start basis benefit happens to be.

**Disaggregated case (production target, projected)**:
At `D = 2080` the LPs are much larger and the backward time is
correspondingly larger. We do not have a measurement-based model for
this scale — the "O(minutes) of wall-time savings" claim from earlier
drafts is a hand-wave. **Replace with: TBD pending Landing 3
measurement.** The kernel becoming usable at all (today's kernel would
dominate wall-clock there) is the primary win independent of Landing 2.

### 8.4 What about the selection cost?

With the `matrixmultiply`-based kernel in place:

| Scale                  | Per-stage selection wall | Per-iter total (64 stages serial) |
| ---------------------- | -----------------------: | --------------------------------: |
| Aggregated (D=155)     |                  ~100 µs |                             ~6 ms |
| Disaggregated (D=2080) |                   1–2 ms |                        ~60–130 ms |

Across 4 iterations:

- Aggregated: ~25 ms total selection wall.
- Disaggregated: ~250–520 ms total selection wall.

Compare to the LP savings (conditional on the precondition):

- Aggregated: ~140 s/iter potential savings — selection cost is 10⁴×
  smaller, hugely net positive _if precondition holds_.
- Disaggregated: unknown savings, ~0.5 s/iter selection cost.

**Net wall impact (conditional)**:

- Aggregated: roughly +0.025 s for selection cost, –140 s for LP
  savings _if_ the precondition holds. Strongly net positive.
- Disaggregated: roughly +0.5 s for selection cost, –? for LP savings.
  Primary win is that the kernel is usable at all.

The conditional framing matters. Landing 2 should not ship until the
precondition is confirmed and §8.3's savings are measured, not
projected.

---

## 9. MPI redundancy — current vs alternatives

After cut sync, every rank has identical pools and archives. Selection
runs to identical results on every rank. This is wasted work in the
strict efficiency sense, but it avoids any selection-related MPI traffic.

### 9.1 Three options for MPI handling

```
─── OPTION 1: Redundant per-rank (today, and proposed default) ────────────

  rank 0:                          rank 1:
    select all 64 stages             select all 64 stages
    (identical computation)          (identical computation)

  Cost: 2× selection time          Comm: 0

  At convertido scale: redundant selection wall = ~8 ms / iter (Layers A+B)
                       skipped MPI cost          = ~0 ms / iter
                       net waste                  = 8 ms / iter
                       → ignore


─── OPTION 2: Stage-partitioned (multi-rank scale-out) ────────────────────

  rank 0:                          rank 1:
    select stages {0, 2, 4, ...}     select stages {1, 3, 5, ...}
    (~32 stages each)                (~32 stages each)
                                ─┬─
                                  │  allgatherv: per-stage activity bitmaps
                                  │  (K × T bytes ≈ 60 KB)
                                ─┴─
  Each rank now has full updates

  Cost: 1× selection time / N ranks  Comm: 1× allgatherv per iter

  At convertido scale (2 ranks): selection wall = ~4 ms / iter
                                  allgatherv     = ~30 ms / iter
                                  net cost       = +26 ms / iter
                                  → not worth it for 2 ranks


─── OPTION 3: Root-and-broadcast ──────────────────────────────────────────

  rank 0:                          ranks 1..N-1:
    select all 64 stages             idle
                                ─┬─
                                  │  broadcast: per-stage activity bitmaps
                                  │  (K × T bytes ≈ 60 KB)
                                ─┴─
  Other ranks receive

  Cost: 1× selection time      Comm: 1× broadcast per iter

  Simple but wastes (N-1) ranks during selection
```

**For 2 ranks**: keep Option 1 (redundant). The MPI alternatives cost more
than they save.

**For 8+ ranks at large scale**: Option 2 becomes attractive if selection
grows expensive (e.g., disaggregated D = 2080 where Layers A+B still leave
selection at ~50–100 ms per iter).

### 9.2 Determinism across ranks

Critical invariant: **after cut sync, all ranks see bit-identical inputs;
the deterministic kernel produces bit-identical outputs without any
cross-rank communication**.

This holds for Option 1 (everyone runs locally). Options 2 and 3 require
an explicit MPI exchange and lose the no-communication property in
exchange for less wasted work.

---

## 10. Implementation phasing

The kernel speedup and the architectural change should be **landed and
evaluated independently**. Landing 1 is uncontroversial; Landings 2+ are
gated on empirical verification of assumptions that the design currently
takes on faith.

### Landing 0 — verification harnesses (prerequisite)

**Purpose**: confirm the design's three load-bearing assumptions before
writing kernel code.

**Verifications**:

1. **matrixmultiply CPU-dispatch behaviour** (§5.2 caveat). Write a
   small Rust harness that calls `matrixmultiply::dgemm` on a known
   input on two hosts of different micro-arch (AVX2-only and
   AVX2+AVX-512), built with `target-feature=+avx2,+fma`. Compare
   outputs byte-by-byte. **Required outcome**: byte-identical. If not,
   switch the design to per-microarch binaries before Landing 1.
2. **Backward LP rebuild model** (§8.3 precondition). Read
   `crates/cobre-sddp/src/backward.rs` and the solver's `add_rows` /
   row-bound path. Confirm whether stage-`t` backward LPs are built
   fresh from `pool.active` at each stage or reuse a previously-built
   LP. Document the finding in a short note appended to this design.
   This determines whether Landing 2 has any savings to harvest.
3. **K at disaggregated scale**. Run a 1-iter disaggregated probe
   (with today's kernel — it's a single iteration so the unusable
   per-stage time is bearable as a one-off). Measure actual K per
   stage. Update §2.3 and §6.1 with the measured value.

**Cost**: ~2 days of work for someone familiar with the codebase. Tiny
relative to the multi-week landings that follow.

**Decision point after Landing 0**:

- If verification 1 fails: design needs revision (per-microarch
  binaries, or a different deterministic GEMM crate).
- If verification 2 reveals LP-reuse: Landing 2 is reframed
  (savings come only from warm-start basis effects, not row-count
  reduction). Re-estimate before committing to the architectural
  change.
- If verification 3 reveals K << 1500: working-set numbers in §2.3
  and §6.1 shrink; Landing 1 still mandatory, Landing 2 savings
  estimates revised.

### Landing 1 — `matrixmultiply` kernel + rayon m-block orchestration

**Scope**: introduce the `matrixmultiply` dependency, replace
`evaluate_chunk_in_place` with `gemm_block` + per-m-block fold/reduce,
remove `PARALLEL_THRESHOLD`. **This is the core kernel replacement and
is self-contained: it requires no changes to the iteration pipeline.**

This landing alone:

- Fixes the disaggregated-unusability problem (~9 s per stage → ~2 ms).
- Closes the algorithmic-correctness vs performance gap at aggregated
  scale (700 ms → ~100 µs per stage).
- Does **not** change the algorithm — selection still runs once per
  iteration after the backward pass, post cut sync, on the global
  pool. Same paper-Lml1 / Level1 / Dominated semantics.
- Does not require any changes to `backward.rs` or
  `training_session/mod.rs`'s pipeline ordering.

**Files**:

- `crates/cobre-sddp/Cargo.toml` (add pinned matrixmultiply dependency)
- `Cargo.toml` (workspace declaration)
- `.cargo/config.toml` (target-feature flags for AVX2+FMA)
- `crates/cobre-sddp/src/cut_selection.rs` (the rewrite)
- `crates/cobre-sddp/src/training_session/mod.rs` (per-worker scratch
  ownership — pre-allocated, not per-call; see §6.1 nested-rayon note)
- CI scripts (grep gate against forbidden patterns)

**Acceptance**:

- Landing 0 verification 1 passed.
- `matrixmultiply` pinned to an exact version with
  `default-features = false`.
- All 4 existing determinism tests pass unchanged.
- NEW realistic-scale determinism tests pass at both aggregated
  (K≈945, M=384, D=155) and disaggregated (K as measured, M=384,
  D=2080) scales, pinned to 1, 16, 96 threads — bit-identical bitmaps.
- CI grep gate in place rejecting external BLAS, runtime CPU dispatch,
  fast-math, and matrixmultiply threading.
- Per-stage selection wall on convertido drops to **≤ 1 ms** (target
  ~100 µs).
- Workspace test suite passes (no regressions); convertido 4-iter
  end-to-end run produces same lower bounds as today's branch (modulo
  documented rayon nondeterminism if any).

**Risk**:

- New dependency: `matrixmultiply` (pure-Rust, MIT/Apache-2.0).
  Pinned to one version; upgrades require explicit verification.
- m-block boundary handling for the last partial block. Covered by the
  realistic-scale determinism test and a unit test exercising
  `M % M_BLOCK != 0`.
- `M_BLOCK = 8` is an initial guess. For the serial-stage case
  (relevant only if Landing 2 lands) `M_BLOCK = 4` may give better
  saturation. Benchmark before settling.
- Per-worker scratch must be pre-allocated at session level, not in
  the rayon fold init closure — otherwise the nested-rayon allocation
  pattern of §6.1 fires.

**This landing stands alone.** If Landings 2+ never happen, Landing 1
still gives us a paper-correct kernel that performs at peak FLOPS at
every D we ship. That's the production-blocking issue resolved.

### Landing 2 — _gated_ on Landing 0 outcomes — selection inside backward

**Decision gate**: do not start Landing 2 unless:

- Landing 0 verification 2 confirms the backward LP rebuild model
  (otherwise Landing 2 has no row-count savings to harvest).
- We accept the algorithmic change to "per-rank local Lml1 + Active-OR
  reconciliation" documented in §8.2, _or_ we resolve §8.2 differently
  (per-stage sync, or skip Landing 2 entirely).
- A measured A/B benchmark (Landing 1 alone vs Landing 1 + a prototype
  of Landing 2) shows the LP savings are real and exceed the
  selection-cost-inside-backward delta.

**Scope** (assuming the gate opens): plumb `&CutSelectionStrategy` and
`&VisitedStatesArchive` into the backward sweep loop. Add a per-stage
hook after the per-stage cut push. Remove the post-backward selection
block in `run_cut_management`; move metric emission accordingly. Add
the chosen MPI reconciliation logic (Active-OR by default, per §8.2).

**Files**:

- `crates/cobre-sddp/src/backward.rs` (or wherever the per-stage
  backward loop lives — likely `backward_pass_state.rs`)
- `crates/cobre-sddp/src/training_session/mod.rs` (remove the
  now-empty post-backward selection block, keep metric aggregation;
  add Active-OR reconciliation after cut sync)
- `crates/cobre-sddp/src/cut_selection.rs` (config: add
  `enable_inside_backward: bool` flag so this can be toggled at
  runtime for A/B testing)

**Acceptance**:

- A/B benchmark (Landing 1 alone vs Landing 1 + 2) shows net wall-time
  win on the aggregated convertido baseline.
- End-to-end determinism within a build: `RAYON_NUM_THREADS=1` vs
  `=96` produces bit-identical lower bounds and identical
  `StageRowSelectionRecord` outputs.
- Cross-rank determinism after cut sync + Active-OR reconciliation:
  the merged active set is bit-identical across ranks.
- Workspace test suite passes.
- Disaggregated 1-iter sanity run completes successfully.
- The §8.2 algorithm change is documented in the book and CHANGELOG.

**Risk**:

- Algorithm semantics change in MPI mode (§8.2). Active-OR
  reconciliation is documented as a deliberate variant of paper-Lml1
  with less aggressive pruning. Convergence behaviour over enough
  iterations to see the lower-bound trajectory must be empirically
  validated.
- Coupling `backward.rs` to `CutSelectionStrategy`. Mitigate via a
  narrow trait or `Option<&dyn ...>` boundary.
- Metric emission order changes — book/CHANGELOG notes needed.
- Budget enforcement timing (§11 Q4). Either keep budget enforcement
  at end-of-iter or move it inside backward — explicit decision needed.
- `check_frequency` gating must wrap the per-stage selection hook so
  the existing throttle behaviour is preserved.
- Stage 0 inclusion (today's code excludes stage 0 from selection;
  the new in-backward design naturally selects on it). Either match
  today's exclusion or document the change.

### Landing 3 — disaggregated performance validation (regardless of Landing 2)

**Scope**: end-to-end benchmark at production target (`D = 2080`).
Profile and tune `M_BLOCK`, confirm cache behaviour, validate net
wall-clock benefit vs aggregated baseline. Applies whether or not
Landing 2 happened.

**Files**: benchmark / report only; no source changes expected unless
profiling exposes a fresh hotspot.

**Acceptance**:

- Disaggregated benchmark: 4-iter run completes within target wall
  time (TBD once Landing 1 — and 2 if applicable — lands and we have
  a baseline).
- Per-stage selection wall measured at ≤ 5 ms at disaggregated scale.
- If Landing 2 shipped: LP savings from interleaved selection
  measured and reported.
- Memory bandwidth measured under perf — confirms cache-blocked GEMM
  hits L3 once per stage, not per trial point.
- NUMA effects measured if present; mitigations from §6.5 applied if
  cross-socket coef traffic shows up as a hotspot.
- Document final `M_BLOCK` and any other tuning choices.

**Risk**: NUMA effects at disaggregated scale; AVX-512 left on the
table by `target-feature=+avx2,+fma` (could be addressed by a separate
x86-64-v4 build track if profiling justifies it).

---

## 11. Open questions

Some of these block Landings 2+ (and are explicit decision gates in
§10); others are smaller follow-ups. The "Blocks" column says which
landing each question must be resolved before.

### Verification-style (blocks Landing 1 or Landing 2)

1. **matrixmultiply runtime CPU dispatch.** §5.2 caveat. Does the
   crate's runtime kernel selection produce different f64 output
   between AVX2-only and AVX2+AVX-512 hosts when built with
   `target-feature=+avx2,+fma`? **Blocks Landing 1.** Verified by
   Landing 0 harness 1.

2. **Backward LP build model.** §8.3 precondition. Does the backward
   LP construction at stage `t` read freshly from `pool.active`, or
   does it reuse a previously-built LP with row-bound toggles?
   **Blocks Landing 2.** Verified by Landing 0 harness 2.

3. **K at disaggregated scale.** §2.3 caveat. Is K really 1500 at
   `D = 2080`, or closer to 945? **Blocks Landing 3 sizing decisions;
   informative for Landing 1.** Verified by Landing 0 harness 3.

4. **Algorithm acceptance for per-rank-local Lml1 + Active-OR.**
   §8.2. The MPI version of selection-inside-backward is NOT
   paper-Lml1. Convergence behaviour over realistic iteration counts
   should be measured before this becomes the production algorithm.
   **Blocks Landing 2.**

### Design-style (resolve during Landing 2 if it goes ahead)

5. **Budget enforcement timing.** Currently Stage 2 budget enforcement
   (`enforce_budget`) runs after selection. If selection moves inside
   backward, do we also move budget enforcement? Cleanest: keep
   budget enforcement at end-of-iter (single global pass over all
   stages) but accept that FCF[t] is temporarily over-budget during
   the backward sweep. Decision needed.

6. **`check_frequency` gating in inside-backward.** Today
   `should_run(iteration)` gates the whole `run_cut_management`. With
   selection inside backward, the per-stage hook must be gated on
   `should_run(iteration)` so the existing throttle is preserved.
   Mechanical but easy to forget.

7. **Stage 0 selection behaviour.** Today's code excludes stage 0
   from selection (`1..num_sel_stages = 1..T-1`). The proposed
   in-backward design naturally includes it (the sweep ends at
   `t = 0`). Either preserve today's exclusion or document the
   change.

8. **MPI sync determinism prerequisite.** Selection's deterministic
   per-rank behaviour relies on cut sync producing bit-identical pool
   contents across ranks. Confirm by reading the current cut-sync
   implementation — is it allgatherv + serialize/deserialize (safe),
   or does it use any FP reduction (potentially non-deterministic)?

### Smaller follow-ups (not blocking)

9. **Reactivation costs.** Phase 1 evaluates inactive cuts to enable
   reactivation. In run_8, only 19 reactivations occurred across the
   whole run. The cost of evaluating inactive cuts may exceed the
   algorithmic benefit. A config flag `enable_reactivation: bool`
   that skips inactive cuts in Phase 1 could be worth measuring.
   Algorithmic change, not a perf refactor — handle as a separate
   design.

10. **Lml1's `break` after first-eligible-at-max.** Paper-correct, but
    means cuts with k > the chosen one are not evaluated for
    reactivation in this trial point. Worth confirming intent against
    Guigues & Bandarra.

11. **Cut aging / pool compaction.** Today's pool grows unboundedly
    (deactivated cuts stay in `populated_count`). Phase 1 evaluates
    ALL populated cuts including deactivated ones. At very long runs,
    this gets expensive. A "hard delete after K iterations of
    inactivity" policy would bound `populated_count`. Separate work
    from this proposal but interacts.

12. **Cut pool layout invariant.** §4.1 assumes
    `pool.coefficients` is contiguous row-major K×D `Vec<f64>`. True
    today (`crates/cobre-sddp/src/cut/pool.rs`). Add a `debug_assert`
    in `gemm_block` that catches any future layout change before it
    silently produces wrong results.

13. **AVX-512 left on the table.** `target-feature=+avx2,+fma`
    excludes AVX-512 even on capable hardware (~1.5–2× perf on
    Sapphire Rapids and Zen 4+). Acceptable for v1; revisit if
    profiling shows enough headroom to justify a separate
    x86-64-v4 build track.

14. **Forward pass / backward race.** Forward at iter N+1 uses the
    FCF state left by iter N. Selection inside backward at iter N
    modifies FCF _during_ iter N — but the forward at iter N is
    already done by then. So no race. Confirm in code (one-line read).

15. **`.cargo/config.toml` merge.** If the workspace already has a
    `.cargo/config.toml`, our `target-feature` settings must merge,
    not overwrite. Trivial; flag at Landing 1 implementation time.

16. **CI grep gate false-positives.** The grep gate over `Cargo.toml`
    and source files might incorrectly flag legitimate uses of
    "fast-math" in code comments or non-rust files. Scope the gate
    carefully to dependency declarations and Rust source.

17. **`matrixmultiply` audit status.** Check `cargo audit` and
    RustSec advisories for known issues with the pinned version
    before merging Landing 1.

---

## 12. Summary

### What we're confident about

**Landing 1 (kernel speedup) is unambiguously good and the production
gate.** It replaces a 700-ms-per-stage kernel that is unusable at
`D = 2080` with a ~100 µs / ~2 ms per-stage kernel built on
`matrixmultiply::dgemm`. The algorithm is unchanged from today's
paper-correct selection. Reproducibility is preserved (subject to
Landing 0 verification 1). One new pinned dependency, one set of
build-flag and CI changes.

### What's gated on verification

**Landing 2 (selection inside backward) is conditionally valuable but
has open questions.** The savings model assumes the backward LP is
rebuilt from active cuts at each stage (§8.3 precondition). In an MPI
deployment, the design implies per-rank-local Lml1 with Active-OR
reconciliation — a different algorithm from paper-Lml1, with weaker
pruning (§8.2). Both need empirical validation before commit.

### Sized results

Aggregated case (D = 155), per iteration:

| Aspect                      | Today              | Landing 1 (matrixmultiply kernel) | Landing 1 + 2 (conditional)                                  |
| --------------------------- | ------------------ | --------------------------------- | ------------------------------------------------------------ |
| Per-stage selection wall    | 700 ms             | ~100 µs                           | same (cheaper kernel)                                        |
| Total selection wall / iter | ~470 ms (parallel) | ~6 ms (parallel) / ~6 ms (serial) | ~6 ms (serial in backward sweep)                             |
| Backward LP savings / iter  | 0 (no change)      | 0                                 | **~30–55 s** if §8.3 precondition holds, **0** if it doesn't |
| Net wall delta vs run_7     | +57 s              | +56 s                             | **−80 s to −160 s** (conditional)                            |
| Algorithm                   | paper-Lml1         | paper-Lml1                        | per-rank local Lml1 + Active-OR (different)                  |
| Reproducibility             | preserved          | preserved                         | preserved within build                                       |

Disaggregated case (D = 2080), per iteration:

| Aspect                      | Today                               | Landing 1                             | Landing 1 + 2 + 3 (conditional)       |
| --------------------------- | ----------------------------------- | ------------------------------------- | ------------------------------------- |
| Per-stage selection wall    | ~6–9 s (projected; kernel unusable) | ~1–2 ms                               | same                                  |
| Total selection wall / iter | unusable                            | ~60–130 ms (parallel) / same (serial) | ~60–130 ms (serial in backward sweep) |
| Backward LP savings / iter  | n/a (cannot ship)                   | 0                                     | TBD pending Landing 3 measurement     |
| Production viability        | NO                                  | YES                                   | YES + architectural optimisation      |
| Reproducibility             | n/a                                 | preserved                             | preserved within build                |

Cross-cutting:

| Aspect                   | Status                                                                                                                  |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| New workspace dependency | `matrixmultiply` (pinned, pure-Rust, MIT/Apache-2.0, ~15 KLOC)                                                          |
| Build-flag changes       | `target-feature=+avx2,+fma,+sse4.2` for x86_64-linux                                                                    |
| CI guardrail             | 10-line grep gate against forbidden patterns (external BLAS, runtime CPU dispatch, fast-math, matrixmultiply threading) |
| MPI footprint            | Landing 1: unchanged. Landing 2: adds Active-OR reconciliation step after cut sync                                      |
| Per-worker scratch       | Pre-allocated at training-session init (not per-call); ≤ `K_max × M_BLOCK × 8` bytes per worker                         |
| Total new test code      | ~50 lines (two realistic-scale determinism tests) + Landing 0 verification harness                                      |

### Verification dependency graph

```
                ┌──────────────────────────┐
                │  Landing 0 (~2 days)     │
                │  3 verification harnesses│
                └────────────┬─────────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
       harness 1        harness 2        harness 3
    (mm CPU dispatch)  (LP rebuild)     (K measured)
            │                │                │
            ▼                ▼                ▼
     gates Landing 1   gates Landing 2  informs sizing
            │
            ▼
     ┌────────────────┐
     │   Landing 1    │  (uncontroversial; ships independently)
     │   matrixmult   │
     │   kernel       │
     └────┬───────────┘
          │
          ├──────────────────────┐
          ▼                      ▼
   ┌─────────────┐        ┌─────────────┐
   │ Landing 2   │        │ Landing 3   │
   │ (gated)     │        │ disagg val. │
   │ inside-bwd  │        │ (parallel   │
   │             │        │  to L2)     │
   └─────────────┘        └─────────────┘
```

### The story in three sentences

At the aggregated scale we already deploy, today's kernel is 14,000×
slower than the hardware can do; at the disaggregated scale we want to
deploy next, today's kernel is unusable. **Landing 1** alone solves the
production-blocking issue: replace the inner loop with
`matrixmultiply::dgemm`, wrap in rayon over small m-blocks, ship a
paper-correct kernel at peak FLOPS. **Landings 2+** propose moving
selection inside the backward sweep so leaner cut pools shrink
subsequent stages' LPs — a real architectural win **if** the backward
LP is rebuilt from active cuts each stage (verify in Landing 0) and
**if** the MPI semantics change to per-rank-local Lml1 + Active-OR is
acceptable (decision required in Landing 2 design).

---

## 13. Consolidated risks and verification needs

A flat list of every "needs measurement / needs confirmation" item in
this design, grouped by what they block. Use this as the pre-merge
checklist when Landing 1 or Landing 2 is ready to commit.

### Pre-Landing 1 (gates the kernel rewrite)

| Item                                                           | Where         | Verification action                                                 |
| -------------------------------------------------------------- | ------------- | ------------------------------------------------------------------- |
| matrixmultiply runtime CPU dispatch produces stable output     | §5.2          | Harness: same input on AVX2 vs AVX2+AVX-512 hosts, compare bytes    |
| `pool.coefficients` is contiguous K×D row-major `Vec<f64>`     | §4.1, §11 Q12 | Read `cut/pool.rs`; add `debug_assert` in `gemm_block`              |
| MPI cut sync is bit-deterministic across ranks                 | §9, §11 Q8    | Read cut-sync implementation; confirm no FP-reduction path          |
| Per-worker scratch can be pre-allocated at session level       | §6.1          | Confirm `TrainingSession` has a stable lifecycle that supports this |
| `matrixmultiply` CVE / RustSec status clean for pinned version | §11 Q17       | `cargo audit` on the pinned version                                 |

### Pre-Landing 2 (gates selection-inside-backward)

| Item                                                                 | Where        | Verification action                                                   |
| -------------------------------------------------------------------- | ------------ | --------------------------------------------------------------------- |
| Backward LP is rebuilt from `pool.active` at each stage              | §8.3, §11 Q2 | Read `backward.rs` and the solver's `add_rows` path; document finding |
| LP savings from interleaved pruning are real on convertido benchmark | §8.3, §10    | A/B: Landing 1 alone vs Landing 1 + prototype Landing 2               |
| Per-rank-local Lml1 + Active-OR converges acceptably vs paper-Lml1   | §8.2, §11 Q4 | Multi-iter benchmark comparing lower-bound trajectories               |
| Budget enforcement timing decision documented                        | §11 Q5       | Pick: keep at end-of-iter, or move into backward. Record in CHANGELOG |
| `check_frequency` gating wraps the per-stage selection hook          | §11 Q6       | Code review checkpoint at Landing 2 commit                            |
| Stage 0 selection behaviour either preserved or change documented    | §11 Q7       | Decision recorded in Landing 2 PR description                         |

### Pre-Landing 3 (informs disaggregated tuning)

| Item                                                   | Where   | Verification action                                                             |
| ------------------------------------------------------ | ------- | ------------------------------------------------------------------------------- |
| Measured K at disaggregated scale                      | §2.3    | 1-iter disaggregated probe with today's kernel                                  |
| `M_BLOCK` empirically tuned for disaggregated workload | §4.2    | Sweep M_BLOCK ∈ {4, 8, 16} on disaggregated benchmark; pick by per-stage wall   |
| NUMA cost measured, mitigations applied if material    | §6.5    | `perf stat` cross-NUMA counters; apply per-NUMA replication if material         |
| Memory bandwidth confirms cache-blocked behaviour      | §6.3    | `perf stat` DRAM read bandwidth during selection — expect ~1 read of coef+state |
| AVX-512 build track decision (separate binary or not)  | §11 Q13 | Decided after Landing 3 perf numbers settle                                     |

### Known caveats accepted as design choices (no action needed)

These are explicit trade-offs documented in the design that we have
chosen to live with:

- **Cross-architecture variation accepted.** Same source on AVX2 vs
  AVX-512 hosts produces different f64 outputs (different micro-arch =
  different reductions). Within-architecture determinism is the
  guarantee; cross-architecture is not.
- **One pinned dependency added** (`matrixmultiply`). Worth it for the
  cache-blocked GEMM at `D = 2080`.
- **`PARALLEL_THRESHOLD` removed.** No fallback to today's chunked
  path; the new kernel is the only kernel. Cleaner; less code to test.
- **Cross-iteration determinism, not cross-build determinism.**
  Different `target-cpu` or different toolchain version may produce
  different f64 outputs. Pin the toolchain.

### Items that block release if missed

The realistic-scale determinism tests (§5.5) are not optional. They
are the empirical guarantee that the design's reproducibility claim
holds at production sizing. If either test reveals non-deterministic
output across thread counts on the same binary, **Landing 1 does not
ship** until the cause is identified and fixed.

---

## 14. Verification Findings

This section collects the file-and-line-referenced findings produced
by the verification harness tickets in Epic 01. Each subsection
addresses one precondition that gates a downstream epic's scope or
viability. The findings are recorded for audit trail; the conclusions
they reach are inputs to the gate decisions described in §10 and §13.

### 14.2 Backward LP rebuild model

- **Date**: 2026-05-27
- **Question (a) — row set equals pool.active?**: **R (rebuild)**, but
  the rebuild happens at the iteration boundary, not at every
  per-stage `load_backward_lp` invocation. Within one trial-point load,
  the LP rows are `(baked_template.num_rows) + (cut_batch.num_rows)`,
  where the baked template was last refreshed by `run_cut_management`
  at the _end_ of the previous iteration over `fcf.active_cuts(t)`
  (`crates/cobre-sddp/src/training_session/mod.rs:1016` calling
  `build_cut_row_batch_into` over the active-cut iterator) and then
  fed through `bake_rows_into_template`
  (`crates/cobre-sddp/src/training_session/mod.rs:1027`) which clears
  all output buffers and writes `num_rows = base.num_rows + rows.num_rows`
  (`crates/cobre-solver/src/baking.rs:131` and 141–150). The
  per-stage delta batch is built from
  `fcf.pools[stage].active_delta_cuts(current_iteration)`
  (`crates/cobre-sddp/src/forward.rs:454`, iterated at
  `crates/cobre-sddp/src/forward.rs:474`), and `active_delta_cuts`
  itself filters by `self.active[slot]`
  (`crates/cobre-sddp/src/cut/pool.rs:359`). The successor LP is
  assembled by `load_backward_lp` as a `load_model(baked_template)`
  call followed by an `add_rows(cut_batch)` append
  (`crates/cobre-sddp/src/backward.rs:268-270`). No row-bound toggle
  for inactive cuts exists at this load site — option B is not used.
  **Justification refs**: `crates/cobre-sddp/src/backward.rs:264-272`;
  `crates/cobre-sddp/src/training_session/mod.rs:1015-1032`;
  `crates/cobre-sddp/src/forward.rs:287-392`;
  `crates/cobre-sddp/src/cut/pool.rs:308-330`;
  `crates/cobre-solver/src/baking.rs:130-185`.

- **Question (b) — mid-iter deactivation visible to subsequent
  stages?**: **PARTIAL**. The delta batch (built each backward stage
  by `build_delta_cut_row_batch_into` at
  `crates/cobre-sddp/src/backward_pass_state.rs:792`) **would**
  immediately reflect a mid-iteration `apply_updates` because it
  reads `fcf.pools[stage].active_delta_cuts(current_iteration)` and
  `active_delta_cuts` filters by the live `self.active` bitmap
  (`crates/cobre-sddp/src/cut/pool.rs:359`). However, the **bulk** of
  the LP rows live in `baked_template` (loaded via
  `load_backward_lp` → `ws.solver.load_model(succ.baked_template)`
  at `crates/cobre-sddp/src/backward.rs:268`), and `baked_template`
  is `inputs.baked[successor]` (`crates/cobre-sddp/src/backward_pass_state.rs:800`)
  which aliases `scratch.baked_templates`
  (`crates/cobre-sddp/src/backward_pass_state.rs:151`). The baked
  templates are only rewritten by the `run_cut_management` loop at
  the end of an iteration (`crates/cobre-sddp/src/training_session/mod.rs:1010-1032`),
  which executes **after** `run_backward_phase` per the iteration
  pipeline at `crates/cobre-sddp/src/training_session/mod.rs:375-378`.
  Therefore, within an iteration, a hypothetical mid-sweep
  `apply_updates(stage = t)` would shrink only those rows for stage
  `t` that originated in the _current_ iteration's delta batch
  (because subsequent backward stages re-call
  `build_delta_cut_row_batch_into` per
  `crates/cobre-sddp/src/backward_pass_state.rs:792`); rows baked
  into `baked_templates[t]` at the previous iteration's
  `run_cut_management` would remain in the LP regardless of the
  mid-sweep deactivation. At iteration 1 specifically, the
  pre-bake at `crates/cobre-sddp/src/training_session/iteration_scratch.rs:151-157`
  uses an _empty_ row batch, so `baked_templates[t]` carries only
  base-template rows and the entire delta-batch path would respond
  to deactivations; from iteration 2 onward, deactivations of cuts
  generated in earlier iterations are invisible until the next
  end-of-iteration bake.
  **Justification refs**:
  `crates/cobre-sddp/src/backward.rs:264-272`;
  `crates/cobre-sddp/src/backward_pass_state.rs:790-802`;
  `crates/cobre-sddp/src/backward_pass_state.rs:151-153`;
  `crates/cobre-sddp/src/training_session/mod.rs:375-378`;
  `crates/cobre-sddp/src/training_session/mod.rs:1010-1032`;
  `crates/cobre-sddp/src/training_session/iteration_scratch.rs:147-157`;
  `crates/cobre-sddp/src/cut/pool.rs:347-379`.

- **Question (c) — baked template contains only active cuts?**:
  **YES**. The baking call site is
  `build_cut_row_batch_into(&mut self.scratch.bake_row_batches[t], self.fcf, t, ...)`
  at `crates/cobre-sddp/src/training_session/mod.rs:1016`, which
  iterates `fcf.active_cuts(stage)` at
  `crates/cobre-sddp/src/forward.rs:319` (after a `num_cuts = fcf.pools[stage].active_count()`
  reservation at `crates/cobre-sddp/src/forward.rs:301`). The
  `active_cuts` iterator filters by `self.active[i]` and stops once
  `cached_active_count` slots have been yielded
  (`crates/cobre-sddp/src/cut/pool.rs:309-329`); inactive slots are
  not visited. The freshly-populated `bake_row_batches[t]` is then
  consumed by `bake_rows_into_template` at
  `crates/cobre-sddp/src/training_session/mod.rs:1027`, which clears
  all output buffers (`crates/cobre-solver/src/baking.rs:141-150`)
  and sets `out.num_rows = base.num_rows + rows.num_rows`
  (`crates/cobre-solver/src/baking.rs:131`). No carry-over from
  previously inactive cuts is possible. **Justification refs**:
  `crates/cobre-sddp/src/training_session/mod.rs:1014-1032`;
  `crates/cobre-sddp/src/forward.rs:287-392`;
  `crates/cobre-sddp/src/cut/pool.rs:308-330`;
  `crates/cobre-solver/src/baking.rs:130-185`.

- **Question (d) — additional work for Epic 03**: To make the §8.3
  savings model fully apply when selection runs inside the backward
  sweep (immediately after stage `t`'s cuts are pushed, before stage
  `t-1`'s `load_backward_lp` reloads), Epic 03 must rebake
  `baked_templates[t]` between `apply_updates` and the next call to
  `load_backward_lp`. The current end-of-iteration bake loop at
  `crates/cobre-sddp/src/training_session/mod.rs:1015-1032` runs over
  all stages once; the per-stage variant inside the backward sweep
  would invoke the same two-step sequence
  (`build_cut_row_batch_into` over `fcf.active_cuts(t)` →
  `bake_rows_into_template`) on a single stage `t` before advancing
  to `t-1`. Cost quantification:
  - **Frequency**: once per pruned stage per iteration, i.e. at most
    `T-2 = 62` rebakes per iteration at convertido sizing (vs. the
    62 bakes already paid at end-of-iteration; a per-stage bake
    _replaces_ the corresponding end-of-iteration bake for that
    stage rather than doubling it).
  - **Rows per rebake**: `K_active(t)` — the active-cut count at
    stage `t` after selection. At run_8 iter 4 aggregated sizing
    this averages ~558 active cuts (after ~41% pruning of ~945
    populated); at disaggregated upper-bound sizing it is bounded
    by `K ≤ 1500`.
  - **Per-rebake work**: `build_cut_row_batch_into` is a linear
    scan over `active_cuts` writing `K_active × (nnz_per_cut)`
    entries (`crates/cobre-sddp/src/forward.rs:319-384`);
    `bake_rows_into_template` is two linear passes over columns
    (`crates/cobre-solver/src/baking.rs` overview at lines 7–16)
    with total work `O(base.num_nz + K_active × nnz_per_cut)`. At
    aggregated sizing this is sub-millisecond per stage; at
    disaggregated sizing with `D = 2080` and dense cut rows this
    is `O(K × D) = O(3.1M)` writes per rebake — a few milliseconds.
  - **Aggregate per iteration**: ~62 × few-ms ≈ low hundreds of
    milliseconds wall added per iteration at disaggregated scale —
    well under the LP savings the design projects in §8.3, but a
    real and measurable line item.

  Alternative path (ii) from the ticket's Implementation Guide step
  4 — teaching `load_backward_lp` to read directly from
  `pool.active` and skip the baked template's pre-pruned rows —
  would avoid the rebake cost but would require either (1) a
  row-deletion API on the solver (HiGHS does not expose efficient
  mid-LP row deletion in the current trait surface; the existing
  contract at `crates/cobre-sddp/src/backward.rs:268-270` is
  `load_model` + `add_rows`, which is append-only) or (2)
  reconstructing the LP from base + a fresh `RowBatch` of _all_
  active cuts (effectively the same work as the rebake plus the
  cost of bypassing the cached baked template). Path (i) — explicit
  rebake — is the simpler and more solver-portable option.

- **Implication for Epic 03**: **outline tickets need scope
  adjustment**. The mechanism _can_ deliver the §8.3 savings model
  in full, but only after Epic 03 inserts a per-stage rebake
  (`build_cut_row_batch_into` + `bake_rows_into_template` for the
  just-processed stage) between `apply_updates` and the next
  backward stage's `load_backward_lp`. Without this rebake, only
  the within-iteration delta batch (cuts generated in the current
  iteration) responds to mid-sweep deactivations; cuts already
  baked into `baked_templates[t]` from prior iterations remain in
  the LP regardless of selection's decision. Epic 03's refinement
  must therefore add (a) a per-stage rebake hook inside the
  backward sweep, (b) a corresponding pruning of the
  end-of-iteration bake loop to skip stages already rebaked
  mid-sweep (avoiding duplicate work), and (c) a benchmark
  measurement that the rebake cost is dominated by the LP savings
  it unlocks. The selection-inside-backward plumbing tickets
  remain viable; their effort estimates should grow to absorb the
  rebake-loop integration.

- **Follow-up probes (not required by this ticket)**:
  1. A small unit test that constructs a 2-stage FCF, adds two cuts
     to stage `t=1`, runs one forward pass + one backward pass that
     populates `baked_templates[1]` with both cuts, then directly
     calls `pool.apply_updates(...)` to deactivate one of them, and
     finally invokes `load_backward_lp` for stage 0's backward
     solve. Assert the resulting LP row count equals
     `base.num_rows + 1` (i.e., the deactivation **did** propagate).
     Expected outcome based on the audit: the test will fail —
     `baked_templates[1]` still carries both cuts. This empirically
     confirms the rebake requirement.
  2. A micro-benchmark of `build_cut_row_batch_into` +
     `bake_rows_into_template` at `K = 1500, D = 2080` measuring
     per-stage rebake wall, to harden the cost-quantification in
     question (d).

### 14.5 Epic 03 gate decision

(§14.3 and §14.4 placeholders pending — to be filled by the
verification harnesses that gate Epic 02 → Epic 03 transition.)

- **Date**: 2026-05-28 (convertido A/B) / 2026-05-29 (higher-forward
  confirmation)
- **Host**: AWS c7a-48xlarge (`decomp-c7a-48x`), 2 nodes, 96 threads/rank,
  MPICH 4.2.3, HiGHS 1.13.1, cobre 0.7.1
- **Rank count (production tier)**: 2 (the run used a 2-invocation set —
  one per mode at `np=2` — rather than the 4-invocation template below)
- **Verdict**: **UNFAVORABLE (final)**. At 192 forwards inside-backward was
  +2.1% total wall; the crossover analysis predicted the gate would flip
  favourable at higher forward counts, but the higher-forward confirmation
  A/B (on `pmo-set-24-semGNL`, 2026-05-29) found post-backward selection
  still faster — the predicted crossover did not materialise in practice.
  The in-backward path stays in place but **default-off**; post-backward
  selection remains the production path. See "Tier 2 — Production result"
  and "Higher-forward confirmation result" below.

#### Tier 1 — Local validation (4 runs: 2 modes × 2 rank counts)

The Tier 1 harness drives FOUR `cobre run` invocations into a single
work-dir so the analysis script can validate the same-mode cross-rank
bit-determinism contract. Cross-mode LB drift is EXPECTED per §8.3
(different selection timing produces different LP shapes → different
cuts → different LB trajectories) and is NOT a Tier 1 check.

Run via:

```
bash plans/.../scripts/run_ab_benchmark.sh <local-case-dir> --tier local --work-dir <work>
# Produces <work>/{baseline_local_1rank,baseline_local_2rank,
#                  inside_local_1rank,inside_local_2rank}/
python3 plans/.../scripts/analyze_ab.py <work> --tier local
```

| run                  | total wall (s) | final_lb |
| -------------------- | -------------- | -------- |
| baseline_local_1rank | _TBD_          | _TBD_    |
| baseline_local_2rank | _TBD_          | _TBD_    |
| inside_local_1rank   | _TBD_          | _TBD_    |
| inside_local_2rank   | _TBD_          | _TBD_    |

| same-mode cross-rank pair | rel delta (1-rank vs 2-rank) | passed |
| ------------------------- | ---------------------------- | ------ |
| baseline                  | _TBD_                        | _TBD_  |
| inside                    | _TBD_                        | _TBD_  |

Tier 1 checks:

- `crash`: every run produced a well-formed iteration log
- `monotone_check`: per-iter LB non-decreasing within 1e-9 relative
  downward tolerance in EACH run
- `bit_determinism`: same-mode 1-rank vs 2-rank final LB matches
  within 1e-15 relative FP tolerance in BOTH modes (rank-count
  invariance contract)
- `extreme_regression`: inside avg total wall (across rank counts)
  does not exceed baseline avg total wall by more than 50%

Verdict: _TIER1_SANITY_OK / TIER1_SANITY_FAIL_

#### Tier 2 — Production gate (MPI, user-owned, 4 runs)

The production tier mirrors the Tier 1 four-invocation set at
convertido scale so that the cross-rank bit-determinism contract is
validated at production scale alongside the cross-mode wall-time
gate. A bit-determinism regression at production scale is even more
concerning than at Tier 1.

Run via:

```
bash plans/.../scripts/run_ab_benchmark.sh <prod-case-dir> --tier production <N>
# (prints four cobre/mpirun command lines; run them manually under
# the workload manager. Output dirs:
# <work>/{baseline_production_1rank,baseline_production_2rank,
#         inside_production_1rank,inside_production_2rank}/)
python3 plans/.../scripts/analyze_ab.py <work> --tier production
```

#### Tier 2 — Production result (convertido, 2026-05-28)

Two production runs were executed (one per mode, both `np=2`, 96
threads/rank, 50 iterations, 192 forwards, 64 stages). The two runs
landed on different physical nodes (`-1` baseline, `-3` inside), so the
forward sweep — whose code is identical in both modes — is used as a
host-comparability control. `duration_seconds` and `sum(time_total_ms)`
agree to < 1 s, so I/O overhead is negligible.

**View A — authoritative wall partition (convergence.parquet):**

| phase                                                    | baseline (s) | inside (s)  | Δ                  |
| -------------------------------------------------------- | ------------ | ----------- | ------------------ |
| forward sweep                                            | 624.9        | 627.9       | +3.0 (+0.5%)       |
| backward sweep                                           | 7,761.7      | 8,129.0     | +367.3 (+4.7%)     |
| other (post-backward selection block, LB eval, overhead) | 701.6        | 521.0       | −180.6             |
| **total**                                                | **9,088.2**  | **9,278.0** | **+189.8 (+2.1%)** |

The forward control differs by only +0.5%, so the two hosts are
comparable and the +2.1% total / +4.7% backward delta is a real
algorithmic effect, not host variance. Inside is slower ⇒ the FAVORABLE
wall-delta gate (inside < baseline) **fails at this scale**.

**View B — where the time moved (rank-0 timing.parquet, sum over iters):**

| serial phase             | baseline (s) | inside (s) | Δ      |
| ------------------------ | ------------ | ---------- | ------ |
| cut selection            | 243.7        | 71.2       | −172.5 |
| cut sync                 | 171.5        | 160.8      | −10.7  |
| rebake (cut_batch_build) | 6.2          | 3.1        | −3.1   |
| lower-bound eval         | 405.2        | 400.5      | −4.8   |

In-backward selection makes the **coordination phases cheaper** (inline
per-stage selection beats the big post-backward block by 172 s; the
Option B per-stage allgatherv is _not_ a net cost — comparable to
baseline's end-of-backward sync). The loss is entirely in **LP-solve
work**: max-worker backward wall +127.2 s, total backward solve CPU work
across 96 workers +13,364 s (+2.3%). Solver retry histograms are empty in
both modes, so the extra cost is more simplex work on a different cut
geometry, not warm-start failures. The cross-mode cut-set difference is
expected per §8.3 (cross-mode LB drift 1.26e-2).

**Front-loaded penalty.** The +189.8 s is almost entirely in iterations
1–25 (+186.6 s); iterations 26–50 are +3.2 s (steady state ≈ neutral,
~+0.1 s/iter). Iteration 1 alone is +46.3 s (cold start).

#### Crossover analysis — why the gate flips above a pool-size threshold

Fitting per-iteration backward wall against active-pool size:

- baseline: `backward_ms ≈ 72,122 + 3.506 · cuts_active` (R² 0.665)
- inside: `backward_ms ≈ 89,092 + 3.039 · cuts_active` (R² 0.858)

Inside carries a **higher fixed per-iteration cost** (+17 s intercept:
per-stage sync + rebake + cold start) but a **lower marginal cost per
active cut** (3.039 vs 3.506 ms/cut, −13%). The two lines **cross at
≈ 36,300 active cuts**. The 192-forward run peaked at 30.6k (inside) /
43.4k (baseline) active cuts — straddling the crossover — which is why
inside came out marginally slower. Illustrative extrapolation of the fit:

| pool (active cuts) | inside vs baseline                |
| ------------------ | --------------------------------- |
| 30k (≈192 fwd)     | +1.7% (slower) — matches observed |
| 60k                | −3.9%                             |
| 90k                | −6.5%                             |
| 150k               | −8.9%                             |
| 300k               | −11.0%                            |

A second, more robust signal points the same way: inside **caps the
within-iteration peak pool** (peak = final = 30,607) while baseline
balloons to 43,374 before its end-of-iteration prune. At higher forward
counts baseline's within-iteration balloon scales with cuts-added/iter,
so (a) baseline's mid-sweep solves get progressively more expensive while
inside's stay lean, and (b) baseline risks memory pressure that inside
structurally avoids — at large scale inside may be the only feasible path,
independent of wall time.

**Caveats.** The baseline slope estimate is noisy (R² 0.665; the 34%
selection-churn adds scatter), so the ≈ 36k crossover is suggestive, not
proven — the robust facts are the peak-capping and the −13% marginal
slope direction. The extrapolation assumes the 192-forward fit holds at
10× larger pools; simplex LP time is often super-linear in row count
(which would favour inside further), but cache/NUMA effects are
unmodeled. The steady-state useful-cut set may be geometry-bounded and
not scale linearly with forward count; what scales cleanly is the
within-iteration peak and the per-iteration solve _count_.

#### Higher-forward confirmation result (2026-05-29) — UNFAVORABLE confirmed

The crossover hypothesis above (inside-backward should flip favourable
once the active pool clears ≈ 36k cuts) was tested with a higher-forward
A/B on the `pmo-set-24-v0.7.0-semGNL` case at `np=2` (96 threads/rank,
2 nodes). **Result: post-backward selection was still faster** — the
predicted crossover did not materialise in practice. The most likely
reasons are the ones flagged as caveats above: the steady-state useful-cut
set is geometry-bounded (so the pool may not have grown far past the
crossover), and the in-backward cut-geometry penalty did not shrink at
scale. The gate is therefore **UNFAVORABLE (final)** and post-backward
selection remains the production path.

##### Note — anomalously large backward times at high forward counts

The higher-forward run surfaced a separate performance concern worth
recording for follow-up (independent of the in-backward verdict). The
backward sweep dominated wall time to an extreme degree and the run was
cancelled by the scheduler at iteration 7 (~5.5 h, 10-iteration budget):

| iter | fwd (ms) | bwd (ms)  |
| ---- | -------- | --------- |
| 1    | 9,154    | 1,317,001 |
| 2    | 91,485   | 2,434,267 |
| 3    | 134,769  | 3,290,933 |
| 4    | 116,890  | 2,840,918 |
| 5    | 122,601  | 2,912,049 |
| 6    | 128,251  | 2,955,574 |
| 7    | 137,409  | 2,992,595 |

Observations: (a) backward is 20–140× the forward wall (22–55 min/iter);
(b) forward jumps ~10× from iter 1 → 2 (cold cut pool → populated) then
plateaus — expected SDDP behaviour amplified by the larger forward count;
(c) backward grows iters 1–3 then plateaus as the pool fills.

Two candidate causes were raised: a parallel work-distribution problem
when the forward-series count exceeds the worker count, or simply that
this case's sampling and LPs are hard. The convertido data bears on the
first: per-worker backward imbalance there is ~6.8% (slowest vs mean
worker) / ~22% by the recorded per-stage `bwd_load_imbalance_ms` metric,
at 192 forwards = 2 trial points per worker. At higher forward counts
(8–20 trial points/worker) that coarse-granularity imbalance should
_average out_, not worsen — so the work-distribution-via-imbalance
mechanism predicts improvement, not the observed blow-up. That points the
large times more toward **LP hardness × ~10× more trial-point solves on a
harder case**, possibly compounded by memory-bandwidth / NUMA saturation
at 96 threads/rank. The proper investigation is a backward-sweep profile
at high forward counts (per-stage load imbalance, memory-bandwidth via
`perf stat`, NUMA cross-socket traffic) — exactly the scope of the
disaggregated-validation profiling work (memory-bandwidth instrumentation
and NUMA profiling). Tracked there; not a blocker for the in-backward
verdict.

#### Decision protocol

Tier 1 (orchestrator-runnable, ~10–12 min for the local case) gates
on four checks — `crash`, `monotone_check`, `bit_determinism`,
`extreme_regression` — and emits `TIER1_SANITY_OK` /
`TIER1_SANITY_FAIL(<reasons>)`. Cross-mode LB drift is EXPECTED per
§8.3 and is NOT a Tier 1 check; per-iter LB monotonicity within each
run and same-mode bit-determinism across rank counts ARE checks.

Tier 2 (production, user-owned) runs the same four checks at
convertido scale and adds the wall-delta gate:

- **FAVORABLE**: all four Tier 1 checks pass AND inside-backward avg
  total wall (across rank counts) is strictly less than baseline avg
  total wall. Selection-inside-backward becomes the default; the
  runtime toggle is replaced with a config-file field.
- **UNFAVORABLE**: any Tier 1 check fails at production scale OR the
  wall-delta gate is not met. The in-backward hook is left dormant
  (toggle defaults to off); the post-backward selection block stays
  in place; alternative mechanisms are explored. A `bit_determinism`
  failure at production scale is a hard stop and re-opens
  ticket-015's hook implementation regardless of wall-time outcome.

---

## Document changelog

- **2026-05-27 (initial)** — first draft with three-axis pitch (SIMD
  inner loop, per-trial-point work-stealing, optional GEMM).
- **2026-05-27 (matrixmultiply pivot)** — restructured around
  `matrixmultiply::dgemm` as the core kernel after deciding on
  disaggregated `D = 2080` as production target.
- **2026-05-27 (critical review)** — added Landing 0 verification
  harnesses; honest framing of Landing 2 as gated on empirical
  verification of LP rebuild model and MPI semantics change; reconciled
  memory-bound estimates between §§2.2 and 2.3; caveated K = 1500
  estimate; fixed `matrixmultiply::dgemm` parameter order in §4.1
  (was wrong); softened "deterministic by construction" claim with
  runtime-CPU-dispatch caveat in §5.2; added §6.1 nested-rayon
  allocation note; consolidated risks into §13.
