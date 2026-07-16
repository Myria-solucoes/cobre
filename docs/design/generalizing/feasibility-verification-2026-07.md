# Beyond-SDDP Generalization Plan — Independent Feasibility Verification

**Date:** 2026-07-16
**Subject:** `docs/design/generalizing/beyond-sddp-generalization.md` (post-QA-convergence revision)
**Status:** Verification report — dated evidence record. Measurements below are
run snapshots regenerable from the harnesses in `spikes/` (this directory);
they inform forks D3/D9/D10 but do not resolve them.

**Method.** A fresh skeptical pass, independent of the prior QA loop:
(1) direct re-verification of load-bearing code claims; (2) **execution of the
two empirical spikes the roadmap itself gates on** — D10 (monomorphization
cost) and D3 (HiGHS MIP determinism); (3) five research streams on ground the
prior passes did not cover (HiGHS determinism primary sources; policy-artifact
and case-format precedents; UC formulation libraries; Rust sparse linear
algebra; 2025–26 ecosystem deltas); (4) adversarial review through five lenses
the prior audit did not use (MPI×engine interaction, maintenance economics,
results model, migration burden, sequencing).

---

## 1. Executed spikes — the roadmap's own gates

### 1.1 D10 monomorphization spike — PASS

Synthetic Cargo workspace mirroring the proposed architecture (rustc 1.94.1,
release profile as workspace, 12 device types × 5 network fidelities × 3
commitment forms, 2 engines generic over `SolverInterface` + capability
traits, one feature-selected backend per binary). Emission blocks carry
distinct constants per (device, formulation) so LLVM cannot fold instances.
Harness: `spikes/monospike/gen_spike.py` + `measure.sh`.

| Variant | Shape                                                                                              | Cold build | Binary | LLVM IR lines |
| ------- | -------------------------------------------------------------------------------------------------- | ---------- | ------ | ------------- |
| base    | today: 1 engine, hardwired formulation                                                             | 0.54 s     | 424 KB | 14,230        |
| A       | doc's recommendation: `ProblemTemplate` as runtime **value**, `BuildProblem` per device, 2 engines | 1.48 s     | 554 KB | 64,144        |
| B       | worst-credible: type-level `<NetForm, ComForm>` builds, **all 15 tuples** legalized × 2 engines    | 4.78 s     | 898 KB | 205,768       |
| B4      | B with only **4 tuples** legalized ("legalize only built tuples")                                  | 1.62 s     | 534 KB | 58,620        |

Findings: the recommended value-based template has **no formulation-matrix
multiplier at all** (codegen scales with source, ~×2 per engine); the
type-level worst case costs 3.2× compile / 1.6× binary and is **linear in
legalized tuples × engines**; the doc's own mitigation verifiably collapses
it back to baseline (B4 ≈ A). Extrapolated to the real `lp/` volume (~9.7k
lines ≈ 5.4× the spike), variant A's contribution stays single-digit seconds.
**The D10 gate can close: the dispatch design is fundable-to-build from the
compile-cost axis** (D9 remains an owner call, but cost forces neither arm).
Caveat: synthetic code under-represents type-checking cost of real builder
code; the ratios — which are what D10 asks about — are robust.

### 1.2 D3 HiGHS MIP determinism spike — evidence FOR threads=1 policy

Harness: `spikes/mipdet/highs_det.c` — links the **exact vendored HiGHS
1.13.1** (submodule build), uses the full C API (`Highs_passMip`; the curated
in-tree wrapper exposes no MIP surface, confirming I.3). Seeded UC-family
MIPs with symmetric unit groups (deliberately many optimal integer
solutions), min-up/down, ramps, reserves; `mip_rel_gap=0`; objective bit
pattern + FNV-1a solution hash + node count compared.

| Probe                                                                                   | Result                                                                                                                                    |
| --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 5 easy instances × 3 runs, threads=1, fresh instance each                               | bit-identical                                                                                                                             |
| Hard instance (G=24, T=48, **2,466-node B&B tree**, proven optimal) × 4 runs in-process | bit-identical (obj bits, solution hash, node count)                                                                                       |
| Node-limit-truncated instance (G=28, T=60, stopped at exactly 3,000 nodes) × 4 runs     | bit-identical truncated incumbent — **node-limit termination is determinism-safe**, unlike wall-clock limits                              |
| Same instance, **two separate processes**                                               | bit-identical across processes                                                                                                            |
| Same instance, **threads=8 + parallel=on** × 2 runs                                     | identical to threads=1 (same 2,466 nodes → parallel tree search did not engage in 1.13.1; **not** evidence that parallel is safe)         |
| **Column-permutation probe**                                                            | objective differs at the last ULP (…78792 vs …79932), different node counts (11 vs 3) — **input order changes the B&B path and the bits** |

Conclusions for D3: (i) run-to-run, fixed-order determinism of HiGHS 1.13.1
MIP at threads=1 is **empirically supported** — the evidence D3 required,
since upstream documents no guarantee; (ii) the permutation divergence proves
**canonical entity ordering before the solver is load-bearing for MIP**, so
declaration-order invariance must be delivered by Cobre's canonical sort, as
today; (iii) scope limits: one instance family, one platform, one solver
version — the production gate remains extending the real
`clp_determinism.rs`-style harness (V.4), and this spike is that harness's
prototype. Policy facts surfaced: HiGHS **defaults** are `threads=0` +
`parallel="choose"` — the deterministic configuration must be explicitly
pinned (`threads=1`, `parallel="off"`, pinned `random_seed`); wall-clock
limits are inherently nondeterministic → deterministic termination must use
node/gap limits only.

---

## 2. Research deltas the roadmap should absorb

### 2.1 HiGHS determinism (primary sources)

- The doc's claim "no determinism statement even single-threaded" is TRUE and
  understated: the maintainer explicitly calls the concurrent path
  "(theoretically) non-deterministic", with "deterministic pseudo-clocks" as
  future work (scipy#16897, 2022-08). Para-B&B (arXiv:2604.09556) confirmed:
  April 2026, Southeast University, **not merged** into HiGHS mainline —
  "single-paper-thin" is fair.
- Calibration: Gurobi and CPLEX ship default-deterministic **parallel** MIP
  via deterministic work clocks; CBC has an opt-in deterministic mode; SCIP is
  deterministic under fixed seeds. **HiGHS is the only one of the five with no
  deterministic parallel mode at all** — `threads=1` is mandatory there, not
  merely conservative.
- **New risk class the doc lacks:** HiGHS MIP _correctness_ regressions across
  versions (GitHub #2957: wrong MIP optimum in 1.14 vs 1.13; #1273/#907/#2171:
  presolve changes the reported optimum). Version pinning and a
  presolve-correctness regression guard are **correctness** gates, independent
  of reproducibility. HiGHS 1.15 ships a **prototype multithreaded MIP
  solver** and 1.12+ a multithreaded IPM: the vendored-solver upgrade path is
  now determinism-hostile — every upgrade must re-run the determinism +
  correctness harness.

### 2.2 Policy-artifact and case-format precedents

- **Deferring `ValueFunctionArtifact` is strongly validated.** StochOptFormat
  serializes problems and deliberately excludes the cost-to-go; it is frozen
  at v1.0.0 (Oct 2023) with SDDP.jl the only real implementation. SDDP.jl's
  cut files are an undocumented, Julia-coupled, single-tool round-trip.
  MSPFormat is MSPLib's benchmark problem format (experimental read support in
  SDDP.jl), not a policy format. **No open, typed FCF/value-function schema
  exists anywhere** — CEPEL's `cortes.dat` and PSR's FCF exports are
  proprietary binaries decoded by community readers.
- **FCF as the composition boundary is validated in production**: DESSEM
  couples directly to the NEWAVE FCF (since 2023); PSR NCP/OptGen read the
  SDDP future-cost function. When Cobre eventually builds the artifact it will
  be **defining** the first open schema, not adopting one — so provenance,
  versioning, and auditability must be first-class (FCF files move markets and
  are corrected under ANEEL deadlines).
- **Case format v2 struct-first is validated** (Sienna independently converged
  on structs → JSON + HDF5 bulk series; v5.0 Nov 2025). One contestable
  sub-decision surfaced: the **bulk time-series binary layer** —
  FlatBuffers/postcard (house rules) vs netCDF/HDF5 (domain-dominant,
  cross-language, lazy-loading; PyPSA explicitly argues netCDF). Needs an
  explicit trade-off note, not a silent default.

### 2.3 UC formulation precedents

- "UC as a composable formulation feature over one data model" is **the
  dominant design**: UnitCommitment.jl (abstract slot per axis + paper-named
  marker structs — maps 1:1 to Rust closed enums), Egret (`UCFormulation`
  namedtuple of nine slots; its stringly-typed `getattr` selection is the
  anti-pattern contrast), Sienna (per-device formulation tree,
  Dispatch-vs-UnitCommitment as formulation choice = the exact precedent for
  `CommitmentConfig::None` emitting zero binaries). Nobody builds UC as a
  separate engine.
- **Design improvement:** assign commitment/formulation **per entity class**
  (Sienna's `DeviceModel` map), not one global `CommitmentConfig`; one enum
  per formulation axis (ramping / piecewise costs / startup / generation
  limits / min-up-down), mirroring UnitCommitment.jl's slots; the
  tight-and-compact default is exactly the field's default (KOW piecewise +
  Morales-España ramping + Rajan-Takriti UT/DT + Pan-Guan/Gentile limits).
- **Unit-under-Plant** is unprecedented in the OSS libraries (all flatten
  unit = generator) but precedented in **DESSEM** (unidades geradoras under
  usinas — the production tool of Cobre's audience) and PLEXOS (identical-unit
  clustering). Multi-start and hydro forbidden-zone data are per-unit —
  affirmative evidence for the hierarchy. Frame the choice as
  production-dispatch alignment. No tool anywhere documents a MILP
  reproducibility contract — Cobre would be first (and has no prior art to
  lean on; the D3 harness is the evidence).

### 2.4 `cobre-network` linear-algebra backend (gap now filled)

- The roadmap names no backend. Recommendation: **faer** (pure Rust,
  MIT/Apache, actively maintained) on its **single-threaded path**
  (`Par::Seq` / `no_rayon`) — the only pure-Rust option with sparse LU +
  Cholesky + QR (AMD ordering). Fallback: vendored **KLU** FFI (what
  PowerNetworkMatrices.jl uses) with its LGPL-2.1 relink obligation noted.
  Rejected: sprs (no LU; LGPL-gated Cholesky), russell_sparse (UMFPACK is
  GPL-2.0+ since 5.2 — license trap), nalgebra-sparse (Cholesky-only).
- **Contract addition required:** a canonical slack bus is **not sufficient**
  — canonical bus/branch ordering **before matrix assembly** must join the
  determinism contract (AMD's elimination order is input-order-dependent →
  ULP-level order variance otherwise; the D3 permutation probe shows the same
  physics on the MIP side). Pin: single-threaded factorization, AMD ordering,
  no FMA contraction (faer guarantees), version pin, golden tests vs a dense
  reference. faer's sparse module is younger than its dense core (correctness
  fixes as recent as 0.24) — property-test it.
- The lazy `VirtualPTDF` row cache is exactly what PowerNetworkMatrices.jl
  ships (KLU factors + on-demand rows + LRU) — validated. Dense PTDF at
  Brazilian scale (~9.6k buses × ~14.7k branches ≈ 1.1 GB f64) is feasible
  but wasteful; lazy is right.

### 2.5 Ecosystem deltas (2025 → Jul 2026)

- **SCIP is fully Apache-2.0** and russcip ships a `bundled` precompiled
  build — a credible third MIP backend (same thin-FFI-to-vendored-C philosophy
  as Cobre's). Worth one line in III.6 as the documented fallback if HiGHS
  MIP proves weak on performance or correctness.
- **PyPSA v1.0**: curated first-class Components class, **no** plugin
  mechanism for user component types → the closed-set posture is strengthened;
  PyPSA also added native two-stage stochastic + CVaR (two-stage only — no
  SDDP competition). The doc's stale "UC cannot combine with expansion"
  limitation was reversed in current PyPSA docs — do not cite it.
- **Sienna**: PowerSystems.jl is on v5 (not 4.x); DeviceModel/ProblemTemplate
  durable; org migrated to Sienna-Platform; NREL was renamed (National Lab of
  the Rockies) — treat Sienna as a design precedent, not a stable roadmap
  anchor.
- **NEWAVE "Híbrido" is in production since Jan-2025** (individualized plants
  in the first 12 months; 2025 CVaR recalibrated to α=15 %, λ=40 %); DESSEM
  v22 authorized Jan-2026. **No CEPEL open-sourcing** — only a
  transparency/governance process (ANEEL REN 1.144/2025 consultation). The
  doc's "REE characterization is going obsolete" is already consummated.
- No Rust JuMP/MOI analog emerged; linopy pays substantial engineering to
  claw back its IR tax → "no MOI-style IR" further validated.

---

## 3. Fresh-lens adversarial findings and dispositions

Confirmed real (all grounded at file:line by a read-only reviewer, key claims
re-verified in this session):

1. **Engine::Direct under MPI is undefined** (HIGH). IV.4's dispatch gives
   `cobre_direct::run` no communicator while dispatch sits after
   rank-collective setup (`broadcast_and_build_setup`; rank-0-only output
   writes per `run/mod.rs`). Phase 0 must define n>1 semantics (reject /
   rank-0-solves-others-idle / redundant deterministic solve) and the Phase-0
   gate must include an MPI run. ED through the current pipeline would also
   execute the stochastic non-root reconstruction — engine-tagged setup stages
   must skip it. → proposed **new fork D14**.
2. **Results-model mechanism misdiagnosed** (HIGH). Python-parity duplication
   lives in the hand-mirrored **call-site lists** of `cobre-io` writers
   conditioned on config/system state — not in a missing results _type_. The
   fix is **one shared output-orchestration entry point** in `cobre-io`
   (single call-site list consumed by CLI and Python) with per-engine results
   values; generalize the results _schema_ only when the second engine exists
   (pull-don't-push applies to outputs too). III.7/V.1 should be re-specified
   accordingly.
3. **Kernel-refit sequencing** (HIGH). Under both D12 options the kernel is
   carved against the **un-purified** core that Phase 1 then reshapes, and
   nothing re-gates the kernel's engine-neutrality afterward. → add **D12
   option (c)**: carve after Phase-1 purification (ED rides the bespoke
   builder through Phase 1; kernel carved once against the clean v2 core,
   pulled by two live engines), and add an explicit post-purification re-gate
   regardless of the option chosen.
4. **Bit-for-bit v1→v2 gate freezes numeric paths** (MED). v2 is structurally
   greenfield but **numerically frozen** for the shim's life (no reduction
   -order or scaling changes to existing quantities; each relocated quantity —
   e.g. `Switchable` Scalar arm vs the stats path — carries an exact-arithmetic
   equivalence obligation). State this general constraint in V.2; note the v1
   schema CI gate persists alongside v2 for the shim lifetime.
5. **Maintenance economics under-evidenced** (MED). The D10 spike bounds
   compile cost but not the per-formulation touch count (~enum variant +
   `BuildProblem` impl + admission-gate entry + schema regen + `.pyi` +
   determinism-harness case) or CI wall-time trajectory. State the touch list
   and CI budget explicitly; the dyn-plugin steelman is otherwise neutralized
   by the spike (the closed-enum premium is measured small).

---

## 4. Verdict

**Feasible: yes.** No fatal flaw found on a pass specifically designed to
find one. The two empirical unknowns the roadmap itself declared blocking
(D10, D3) were executed and both landed on the plan's side: monomorphization
cost is bounded and controllable exactly as designed, and HiGHS 1.13.1
single-threaded MIP determinism — which upstream refuses to document — held
bit-for-bit across processes on a 2,466-node proof tree.

**Sound: yes, and now multiply validated.** Independent precedent lines
converged on each core choice this session: the formulation-template
mechanism (UnitCommitment.jl / Egret / Sienna), the closed component set
(PyPSA v1.0), the FCF composition boundary (CEPEL/PSR in production), the
lazy derived-matrix design (PowerNetworkMatrices.jl), the struct-first case
format (Sienna v5), and the defer of `ValueFunctionArtifact` (no open FCF
schema exists anywhere to conform to).

**Best-possible: with amendments.** The verification surfaced no superior
alternative architecture — but it did surface (i) one genuinely missing
decision (Engine::Direct MPI semantics, D14), (ii) one mis-specified
mechanism (the results seam), (iii) one missing sequencing option (D12c),
(iv) one missing contract clause (canonical ordering before network-matrix
assembly), (v) one unnamed dependency (the sparse-LA backend), and (vi) a
solver-upgrade correctness/determinism gate. With those folded in — plus the
per-entity-class formulation map and the factual refreshes (NEWAVE Híbrido in
production, SCIP/russcip option, Sienna/PyPSA status) — the design is, on the
evidence gathered, the strongest available shape for Cobre's constraints.

The remaining genuinely open items are the owner forks the doc already names
(D1 universal-vs-cadastre, D2 timing, D3 sign-off — now evidence-backed, D9,
D12 — now three-armed, D13), which is exactly where a roadmap should leave
its uncertainty.

---

_Amendment application to `beyond-sddp-generalization.md` awaits owner
sign-off per the anti-simplification discipline: several items add or modify
forks (D12c, D14) or change recommendations (results seam, per-class
formulation map)._
