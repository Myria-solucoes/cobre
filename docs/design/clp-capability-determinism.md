# CLP Capability Determinism

> **Status**: Delivered (2026-06-02). Records the determinism contract, the
> verified CLP API surface, and the capability decisions the CLP backend rests
> on. The three C++-class-only knobs — dual-steepest-edge pricing,
> factorization-frequency, and the hot-start snapshot/restore trio — are
> delivered and determinism-verified. It accompanies
> `backend-selection-and-mutual-exclusivity.md`: that note settles _which_
> backend a build selects; this one settles _how_ the CLP backend's
> capability surface stays bit-for-bit reproducible.

## The determinism contract

Every solver path in this workspace owes two guarantees, and the CLP backend is
no exception:

- **Declaration-order invariance** — solving a model and a row-permuted copy of
  the same model yields the identical objective, primal vector, and reduced
  costs, and, after the duals are mapped back through the inverse permutation,
  the identical row duals. Input ordering must never change the answer.
- **Bit-for-bit re-solve reproducibility** — the same mutate-then-solve sequence,
  run twice on one solver instance and once on a fresh instance, yields
  objective and solution buffers that are identical down to the last mantissa
  bit.

These are verified by the comparison being **bit-for-bit** (`f64::to_bits`
equality), never a tolerance. A tolerance would mask exactly the drift the
contract forbids. The reusable harness in
`crates/cobre-solver/tests/clp_determinism.rs` encodes both guarantees and
exposes a `MutationStep` vocabulary plus an
`assert_solve_sequence_deterministic` helper so that any new CLP solve
capability can be checked against the contract by handing the helper a solver
factory and a step list — without modifying the comparison core.

The contract is about a path being **self-consistent**: the same inputs in the
same declaration order always produce the same output, and a permuted input
produces the correspondingly permuted output. It is **not** a requirement that
two different solve methods agree with each other. In particular, a warm or
hot-started re-solve is **not** required to land on the same dual vertex as a
cold solve of the same model. A degenerate LP has many optimal dual vertices;
any of them is a valid answer. What the contract demands is that whichever
vertex a given path reports, it reports it **reproducibly and order-invariantly**.
So the right test for a new capability is "does this path reproduce itself and
stay order-invariant", never "does this path match a cold solve".

Two solver settings are preconditions for the contract and must remain in force
on every CLP path:

- **Perturbation stays off.** The CLP profile default sends the
  perturbation-off mode (`102`, not CLP's own auto-perturbation default). With
  auto-perturbation on, CLP injects pseudo-random bound perturbations and no
  re-solve is bit-reproducible.
- **Scaling stays off and the dual-sign convention is identity.** The prescaler
  already conditions the matrix, and the row prices are forwarded with no
  negation, so duals compare directly against the canonical convention.

## API-surface verdict

The vendored CLP C interface
(`crates/cobre-solver/vendor/Clp/Clp/src/Clp_C_Interface.h`) was re-checked
against the capability surface CLP needs:

- **The C API exposes incremental mutation and both simplex directions.**
  `Clp_addRows` (CSR form: `rowStarts`/`columns`/`elements`, with `rowStarts`
  typed `const CoinBigIndex*`, which is `int` in the default
  `COIN_BIG_INDEX == 0` build), `Clp_chgRowLower`, `Clp_chgRowUpper`,
  `Clp_chgColumnLower`, `Clp_chgColumnUpper`, and `Clp_primal` are all present.
  A compile-time guard already asserts `CoinBigIndex` is 32-bit so the Rust
  `i32` offset arrays are ABI-compatible.
- **The C API does _not_ expose the hot-start loop, factorization frequency, or
  dual-steepest-edge pricing.** `markHotStart` / `solveFromHotStart` /
  `unmarkHotStart`, `setFactorizationFrequency`, and
  `setDualRowPivotAlgorithm` / `ClpDualRowSteepest` appear zero times in the C
  header. They exist only on the C++ `ClpSimplex` class
  (`ClpSimplex.hpp`: the hot-start trio, factorization frequency, and the
  dual-row pivot setter).

**Routing consequence.** Incremental mutation can be implemented entirely
through the C route — no C++ shim is required to append rows or patch bounds.
Reaching hot-start, factorization frequency, or steepest-edge pricing requires a
small C++ shim that casts the opaque model handle to the C interface's
`Clp_Simplex` struct (whose `model_` member is the C++ `ClpSimplex`) and calls
the class methods directly.

## Incremental mutation stays the source-of-truth model, reconciled natively

The first-round CLP backend mutates by patching Rust-retained CSC/bounds buffers
and re-issuing a full `Clp_loadProblem` from scratch on every mutation, which
discards CLP's factorization and basis each time. The deterministic replacement
keeps the **retained-model buffers as the source of truth** and reconciles them
with the native incremental calls (`Clp_chgRowLower`/`Upper`,
`Clp_chgColumnLower`/`Upper` for bound patches, `Clp_addRows` for appended rows)
on each mutation, rather than reloading.

A de-risking measurement confirmed this is bit-for-bit safe: on the two-cut
fixture, patching a row bound through `Clp_chgRowLower`/`Upper` followed by a
dual re-solve produces row duals that are **bit-for-bit identical** to loading
the already-patched model from scratch and solving cold. Keeping the retained
buffers authoritative also keeps the two independently-checkable acceptance
properties intact:

- the conformance fixtures stay exact (append two rows → objective `162`;
  warm-start round-trip → objective `100`), and
- declaration-order invariance is checkable directly against the retained
  buffers, because they are the canonical, ordering-defined copy of the model.

The retained buffers double as the basis for any future cross-check against a
full reload, so a regression that silently diverges the native path from the
reload path is caught.

## The three class-only knobs are delivered through the C++ shim

The determinism contract for these knobs is non-negotiable and unchanged:
**self-consistent reproducibility** (the same mutate-then-solve sequence, run
twice on one instance and once on a fresh instance, agrees bit-for-bit on
objective, primals, reduced costs, and row duals) plus **declaration-order
invariance** (a row-permuted run yields the inverse-permuted identical result).
A hot-started re-solve is **not** required to reproduce a cold solve's dual
vertex: a degenerate optimum has many valid optimal dual vertices, and a
hot-started re-solve may report a different but equally valid one. What the
contract demands is that whichever vertex the path reports, it reports
reproducibly and order-invariantly. Perturbation stays off (`102`) and scaling
stays off across the whole lifetime; those are preconditions for any
reproducibility.

The hot-start composition is: per-(worker, stage) persistent solver instances,
each solved once to leave the simplex rim and factorization alive, then
snapshotting once with `markHotStart`, re-solving each bound-patched model with
`solveFromHotStart`, and releasing with `unmarkHotStart` on teardown — composed
with the native bound-mutation path (`Clp_chg*` patches preserve the
factorization across the patch). The snapshot token (`saveStuff`) is CLP-owned;
the solver holds it opaquely, never dereferences it, pairs every `markHotStart`
with exactly one `unmarkHotStart` (an explicit release, or one issued from
`Drop`), and `debug_assert!`s the token is non-null. The hot-start methods are
inherent `ClpSolver` capabilities, separate from the shared `solve` entry point,
so they compose with the persistent instance without altering the cold-solve
contract.

### Root-cause correction: the earlier SIGSEGVs were a shim cast bug, not a freed rim

An earlier de-risking pass reported that none of the three knobs was deliverable
and attributed the SIGSEGVs to a torn-down rim left dangling by a finished
`Clp_dual` — concluding that only a substantially larger `OsiClpSolverInterface`
persistent-rim wrapper could reach them. **That diagnosis was wrong.** A code
review found the real cause in the C++ shim: it cast the opaque model handle —
which is a `Clp_Simplex` **wrapper** struct
(`{ ClpSimplex* model_; CMessageHandler* handler_ }`, what `Clp_newModel()`
returns) — **directly** to `ClpSimplex*`, instead of going through `->model_`.
Every class method therefore dispatched on a garbage `this` built from the
wrapper's two leading pointers, faulting in a memory-layout-dependent way. The
gdb "dangling `dualRowPivot_`" observation was the misinterpreted wrapper data,
not a freed rim object.

With the corrected cast (`static_cast<Clp_Simplex*>(model)->model_`, which is how
every `Clp_*` C-API call already reaches the model), all three knobs were
re-tested across every arrangement and **work** — zero faults under a
per-process test runner, correct objectives, and hot-start re-solves that are
bit-for-bit reproducible run-to-run and cross-instance. No persistent-rim Osi
wrapper is required; the native incremental-mutation path already keeps the
factorization alive across the bound patches and solves these knobs operate on.

### What is wired

- **Dual-steepest-edge pricing and refactorization cadence are driven by
  `apply_profile`.** When the profile's `dual_pricing_mode` selects a non-default
  rule it is installed through the shim (`setDualRowPivotAlgorithm`); mode `3` is
  CLP's own `ClpDualRowSteepest` constructor default and acts as the "leave CLP
  default — do not call" sentinel, so the default/forward/simulation profiles
  issue no pricing call and stay byte-identical to a build that never set
  pricing. When `factorization_frequency` is non-zero it is set through the shim
  (`setFactorizationFrequency`); `0` is the "leave CLP's internal default"
  sentinel and issues no call. The backward-pass profile pins full
  dual-steepest-edge pricing (`dual_pricing_mode = 1`) and a tuned refactorization
  cadence (`factorization_frequency = 200`); both are independently chosen
  CLP-native values, not a translation of the `HiGHS` per-phase profiles, and
  both drive their shim setters when that profile is applied.
- **Hot-start is delivered as an inherent `ClpSolver` capability**
  (`mark_hot_start` / `solve_from_hot_start` / `unmark_hot_start`) composing with
  the native bound-mutation path as described above.

Each knob is held to the determinism contract, not a hot-vs-cold cross-check: the
reusable harness asserts run-to-run, cross-instance, and same-instance-reused
bit-for-bit reproducibility plus declaration-order invariance for the
DSE-tuned-profile sequence, the refactorization-cadence sequence, the combined
backward-tuned profile, and the `mark`/`solveFromHotStart`/`unmark` sequence —
all green. The CLP backend ships functional across the full surface: the dual and
primal simplex algorithms, native incremental mutation with basis warm-start
across solves, the per-phase perturbation / scaling / feasibility-tolerance /
iteration-limit profiles, dual-steepest-edge pricing, factorization cadence, and
hot-start.

## Escalation contract

Determinism is a correctness contract, not a performance preference. When a
de-risking measurement shows that a capability cannot meet the determinism
contract as designed, the response is to **stop and escalate to the owner** with
the observation and the available options — never to silently drop the
capability, silently relax the contract from bit-for-bit to "close enough," or
proceed on a path known to break reproducibility or order-invariance. Equally,
what the contract _requires_ is a question for the owner: when a measurement
turns up behavior that looks surprising (such as a warm path reporting a
different optimal dual vertex than a cold solve), the resolution is to confirm
the contract's intent with the owner rather than to assume the stricter reading
and block a capability that actually satisfies the contract. A capability ships
only after it is demonstrated to satisfy declaration-order invariance and
bit-for-bit re-solve reproducibility under the contract as the owner defines it.
