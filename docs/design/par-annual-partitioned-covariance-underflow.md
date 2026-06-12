# PAR annual partitioned-covariance: potential `usize` underflow in `ref_season`

**Status:** Open — deferred investigation. Pre-existing; not a regression.
**Severity:** Potential panic (debug) / silent wraparound (release) on the annual
PAR fitting path. Reachability unconfirmed.
**Path exposure:** Annual order-selection only. The `examples/1dtoy` determinism
oracle does **not** exercise it (1dtoy uses non-annual selection), so the bug is
oracle-blind — only the `cobre-stochastic` annual unit tests touch this code, and
none currently drive `max_order` into the failing regime.

## Symptom

In `crates/cobre-stochastic/src/par/fitting/partitioned_covariance.rs`,
`assemble_partitioned_covariance` builds the Σ₁₂ row-1 block:

```rust
for (j, entry) in sigma_12[k..k + k.saturating_sub(1)].iter_mut().enumerate() {
    let ref_season = (season + n_seasons - 1 - j) % n_seasons;
    let lag = k.saturating_sub(1).saturating_sub(j);
    let rho = periodic_autocorrelation(ref_season, lag, n_seasons, obs_z, stats_z);
    *entry = rho;
}
```

`ref_season` is computed in `usize`. `j` ranges over `0..=k-2`. When
`season + n_seasons - 1 < j` the subtraction underflows:

- in debug builds: panic (`attempt to subtract with overflow`);
- in release builds: wraps to a garbage `usize`, then `% n_seasons` yields a
  wrong season index — a **silent** wrong correlation, not a crash.

For `season = 0`, `n_seasons = 12`, the underflow first occurs at `j = 12`, which
is reachable once `j` can reach `n_seasons`, i.e. `k - 2 >= n_seasons`
(`k >= n_seasons + 2`). `k` grows to the classical AR `max_order`.

## Why it may be reachable

`crates/cobre-stochastic/src/par/precompute.rs` notes that the classical AR order
"may already exceed 12 in unusual configurations." If `max_order` exceeds
`n_seasons + 1`, the loop reaches the underflowing `j`. Whether real input data
plus the order-selection guards actually drive `max_order` that high is the open
question.

## Proposed fix (not yet applied)

A wrapping-safe rewrite that is arithmetically identical for the non-underflowing
range:

```rust
let ref_season = (season + n_seasons * 2 - 1 - j % n_seasons) % n_seasons;
```

Adding an extra `n_seasons` multiple keeps the subtraction non-negative for any
`j`, since `j % n_seasons < n_seasons`.

## Why deferred

1. **Pre-existing, not introduced here.** This code was relocated verbatim during
   the `par/fitting.rs` → `fitting/` directory split; the arithmetic is unchanged
   from before that move.
2. **Oracle-blind.** A fix changes numerics on a path the `1dtoy` bit-identity
   oracle cannot guard, so it must not be applied as a drive-by cleanup.
3. **Needs its own coverage.** Before fixing, confirm reachability (can
   `max_order` reach `n_seasons + 2` under real inputs and the order-selection
   guards?) and add an annual-path regression test that constructs the failing
   regime, so the fix can be verified against a known-correct expectation rather
   than blind.

## Investigation checklist

- [ ] Trace the realistic upper bound of classical `max_order` vs `n_seasons`
      through `precompute.rs` order selection and any caller-side caps.
- [ ] Construct a minimal annual fixture that drives `k >= n_seasons + 2`.
- [ ] Add a `cobre-stochastic` unit test asserting `ref_season` correctness across
      the full `j` range (parameterized over `season`, `n_seasons`, `k`).
- [ ] Apply the wrapping-safe form; confirm the new test passes and existing
      annual tests are unchanged.
