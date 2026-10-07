# Myria runtime on Cobre 0.18

This fork keeps the upstream CLI and Python ABI version while identifying every
output and policy as `cobre-myria` / `0.18.0-myria.1`. Policies are native format 3.
Resume requires the same software identity, release and compatible model. Do not
reuse a policy from the official variant or an older release. Convert DECOMP
boundary policies with the same Python wheel used to execute the model.

Automatically fitted nonstationary PAR models receive the Myria regularization
and an explicit validation warning. User-supplied coefficients are not silently
changed. Historical seasonal coverage uses recurring seasons even when the
history lies outside the simulation horizon.

The optimizations described in [optimized training](optimized-training.md) are
available. Approximate point budgets and progressive forward populations remain
opt-in; the default uses all configured points. CPU binding defaults to `none`.
Native checkpoint writes replace `output/policy` atomically. Sparse point budgets
omit unused cut slots and cached bases whose row positions cannot survive export.

```sh
cobre validate case --output output
cobre run case --output output --threads 4
```

A resumed validation must see the same output directory as execution. SIGTERM or
SIGINT requests a graceful stop at an iteration boundary: a partial run exits 5,
retains its policy and skips simulation. An interrupted run is not a successful
complete study. Iteration limits are absolute across resumes.

The release workflow builds matching Linux CLI archives and Python wheels on
native x86_64 and ARM64 runners. It executes targeted regressions for PAR repair,
sparse cut export, point selection, progressive populations, dynamic selection,
work scheduling, affinity, native checkpointing and CLI/Python parity. The shipped
pair is also exercised with:

```sh
python scripts/benchmarks/verify_runtime_018.py --cli target/release/cobre --case examples/1dtoy
```

These checks exercise numerical correctness and persistence. They do not measure
performance at convergence or replace a platform canary with real decks.

The native ARM64 test link also verifies Qhull archive ordering with GNU ld.
The wrapper must precede the archive providing its symbols; success with LLD
on x86_64 alone does not prove this build contract.
