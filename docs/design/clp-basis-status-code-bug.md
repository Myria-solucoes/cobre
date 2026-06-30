# CLP Basis Status-Code Namespace Bug — Finding

**Status:** open, unfixed. Surfaced during the adversarial assessment of
`docs/design/chronological-blocks.md` and verified firsthand against the code on
2026-06-30. Pre-existing; independent of chronological blocks (but amplified by
them — see §5).

**Severity:** correctness-adjacent but bounded — no wrong optimum and no crash;
the symptom is a silently degraded warm-start on the CLP backend, which is
off-by-default.

## 1. Summary

The warm-start basis machinery in `cobre-sddp` is written entirely in the **HiGHS**
status-code namespace, but it runs unconditionally for **both** solver backends.
When it demotes an excess-basic cut row, it writes the HiGHS code for
"nonbasic-at-lower" (`0`). On the CLP backend that integer means **free**, not
at-lower (CLP's at-lower is `3`). The demoted cut row is therefore installed into
CLP as a _free_ row, producing an inconsistent warm-start basis. CLP's `Clp_dual`
tolerates and repairs it, so the solve still reaches the correct optimum — but the
warm-start is degraded (extra pivots / weaker basis reuse) on exactly the demoted
rows, which is why no test catches it.

## 2. Root cause

The two backends number simplex statuses differently, and the shared `Basis` type
(`cobre_solver::types::Basis`) is documented as holding _raw solver-native_ status
codes with **no canonical numbering**:

| meaning           | HiGHS (`ffi/highs.rs`)         | CLP (`backends/clp/solver.rs`)       |
| ----------------- | ------------------------------ | ------------------------------------ |
| basic             | `HIGHS_BASIS_STATUS_BASIC = 1` | `CLP_BASIS_BASIC = 1`                |
| nonbasic at lower | `HIGHS_BASIS_STATUS_LOWER = 0` | `CLP_BASIS_AT_LOWER = 3`             |
| value `0` means…  | at-lower                       | **free** (COIN `ClpSimplex::isFree`) |

"basic" coincides (`1` in both), so the _reads_ in the reconstruction path
(`reconstruct_basis` tests only `== HIGHS_BASIS_STATUS_BASIC`) and the new-cut-row
seeding (writes `HIGHS_BASIS_STATUS_BASIC = 1`) and the column padding
(`reconstruct_col_statuses` pads with `1`) are all namespace-safe. The single
unsafe write is the demotion in `enforce_basic_count_invariant`
(`crates/cobre-sddp/src/cut/basis_reconstruct.rs`):

```rust
if out.row_status[idx] == HIGHS_BASIS_STATUS_BASIC {   // 1 == 1, OK in both
    out.row_status[idx] = HIGHS_BASIS_STATUS_LOWER;     // writes 0 → CLP reads "free"
    ...
}
```

`enforce_basic_count_invariant` runs unconditionally after every reconstruction on
the shared `run_stage_solve<S: SolverInterface>` path, so it executes for CLP with
a CLP-captured basis (`get_basis` returns CLP-native codes — preserved nonbasic
rows are `3`). `ClpSolver::install_basis` then passes `b.row_status[r]` **verbatim**
to `cobre_clp_set_row_status` with no translation, so the demoted row enters CLP as
status `0` (free). The resulting `row_status` vector mixes two namespaces:
preserved rows in CLP's `3`, demoted rows in HiGHS's `0`.

## 3. Why it is not catastrophic

`ClpSolver::install_basis` documents (and relies on) the fact that CLP's per-element
status setters silently accept an inconsistent offered basis and `Clp_dual` repairs
it — there is no consistency check and no rejection on the CLP path (unlike HiGHS,
which asserts and can return `BasisInconsistent`). A warm-start basis only _seeds_
the simplex; it cannot change the LP's optimum. So the effect is confined to
warm-start quality (pivot count), not the solution.

## 4. Detection gap

No test catches this because (a) the optimum is unaffected, and (b) warm-start
_efficiency_ is not asserted anywhere. A `Basis` produced under one backend is
silently semantically invalid under the other despite the shared type. Any future
test should assert that a round-tripped CLP basis contains no `0`/free row status
where an at-lower (`3`) is intended, or compare pivot counts against a clean basis.

## 5. Relevance to chronological blocks

Chronological mode adds many more cut rows per stage (the cut pool is unchanged,
but the larger per-stage LP and more frequent cut selection mean more
demotion events), so it exercises this demotion path harder. The bug should be
fixed before relying on CLP warm-starts at chronological scale. It does **not**
block the chronological-blocks design (HiGHS is the default backend and is
unaffected; correctness holds on CLP too).

## 6. Fix direction

Translate basis status codes at the CLP boundary rather than passing raw `i32`
through. Options, in rough order of preference:

1. Make `Basis` carry a **canonical** status enum and have each backend's
   `get_basis`/`install_basis` map to/from its native codes — the type's doc
   already admits the current "raw native codes" design is the trap.
2. Failing that, have `ClpSolver::install_basis` (and `get_basis`) translate the
   HiGHS-namespace codes the reconstruction path emits into CLP codes
   (`0 → CLP_BASIS_AT_LOWER`), so the demotion lands on at-lower, not free.

Either fix is byte-neutral for the HiGHS path and for the LP itself.

## 7. References

- `crates/cobre-sddp/src/cut/basis_reconstruct.rs` — `enforce_basic_count_invariant`
  (the demotion write), `reconstruct_basis`, `reconstruct_col_statuses`.
- `crates/cobre-solver/src/ffi/highs.rs` — `HIGHS_BASIS_STATUS_LOWER` / `_BASIC`.
- `crates/cobre-solver/src/backends/clp/solver.rs` — `CLP_BASIS_AT_LOWER` /
  `CLP_BASIS_BASIC`, `install_basis` (raw row/col passthrough + the `Clp_dual`
  self-repair note).
- `crates/cobre-solver/src/types.rs` — `Basis` (the "raw solver-native codes,
  no canonical numbering" contract).

Related CLP investigations: `docs/design/` CLP hot-start / floor-infeasibility
work (separate issues — dual-simplex pathologies, not this status-code mismatch).
