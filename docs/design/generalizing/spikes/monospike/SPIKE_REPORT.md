# D10 Monomorphization Spike — Results

**Setup.** Synthetic Cargo workspace mirroring the proposed architecture
(rustc 1.94.1, `[profile.release] strip="symbols"`, no LTO, 14-core WSL2):
12 device types, 5 network fidelities, 3 commitment forms, 2 engines
(direct = build-once-solve-once; sddp-like = 20-iteration patch/solve loop),
solver behind `SolverInterface` + `ProducesDuals`/`SupportsWarmStart`
capability traits, one feature-selected backend per binary (mirrors Cobre).
Emission blocks carry distinct constants per (device, formulation) so LLVM
cannot fold instances. Same seed across variants → identical emission logic.

| Variant | Shape                                                                                                         | Src lines (model+engines) | Cold build | Binary | LLVM IR lines |
| ------- | ------------------------------------------------------------------------------------------------------------- | ------------------------- | ---------- | ------ | ------------- |
| base    | today: 1 engine, hardwired formulation                                                                        | 372                       | 0.54 s     | 424 KB | 14,230        |
| A       | doc's recommendation: `ProblemTemplate` as runtime VALUE; `BuildProblem` per device; 2 engines generic over S | 1,810                     | 1.48 s     | 554 KB | 64,144        |
| B       | worst-credible: type-level `<N: NetForm, C: ComForm>` builds; ALL 15 (N,C) tuples legalized × 2 engines       | 1,790                     | 4.78 s     | 898 KB | 205,768       |
| B4      | same as B but only 4 tuples legalized (the doc's "legalize only built tuples")                                | 1,790                     | 1.62 s     | 534 KB | 58,620        |

Incremental rebuild after touching the model crate: base 0.47 s / A 1.35 s / B 4.74 s.

**Findings.**

1. The recommended shape (A) has NO monomorphization multiplier from the
   formulation matrix at all — formulations are runtime values; codegen scales
   with source size only. Its 4.5× IR over base is feature code (5 fidelity
   arms × 12 devices vs 1 arm), not generics overhead. Engine count ≈ ×2 on
   builder codegen (each engine crate monomorphizes the shared builders over S
   once).
2. The type-level worst case (B, 15 tuples) costs 3.2× compile time and 1.6×
   binary vs A — noticeable, not explosive. Cost is LINEAR in
   (legalized tuples × engines).
3. The doc's own mitigation works exactly as claimed: restricting to 4 built
   tuples (B4) collapses cost to variant-A levels (1.62 s / 534 KB — binary
   actually smaller than A because dead fidelity branches are const-folded per
   instantiation).
4. Extrapolation to real scale (cobre-sddp `lp/` ≈ 9.7k lines ≈ 5.4× spike
   volume): variant A's cold-build contribution stays in single-digit seconds;
   even a fully-legalized 15-tuple type-level matrix would add ~25 s cold —
   acceptable, but wasteful for no benefit.

**Verdict for D10.** The monomorphization risk the doc gates on is BOUNDED and
CONTROLLABLE. The closed-enum + monomorphized-generics design is fundable from
the compile-cost axis, with the value-based `ProblemTemplate` (A) as the
default shape and type-level formulation parameters reserved for cases where
per-instance specialization demonstrably pays. Caveats: synthetic code
underestimates per-instance type-checking cost of real builder code
(borrows/iterators/complex types), and the spike does not model incremental
compilation interaction with a large dependency graph — but the RATIOS, which
are what D10 asks about, are robust.
