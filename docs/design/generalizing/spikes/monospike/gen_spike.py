#!/usr/bin/env python3
"""D10 monomorphization spike generator.

Generates three Cargo workspaces simulating the proposed cobre-model design:
  base : single engine, hardwired formulation (mimics today's shape)
  a    : ProblemTemplate as runtime VALUE; BuildProblem per device; 2 engines
         generic over SolverInterface (the doc's recommended shape, III.3)
  b    : worst-credible type-level matrix: device builds generic over
         <N: NetForm, C: ComForm>; 15 (N,C) combos instantiated per engine
         crate via a dispatch registry (5x3 x 2 engines x D devices)

Each variant gets: spike-solver (trait + 1 feature-selected backend of 2),
spike-model (devices + builders), spike-direct + spike-sddp engines (a/b only),
spike-cli binary. Code volume and constants vary per (device, form, fidelity)
so LLVM cannot trivially fold instances.
"""
import random
import sys
from pathlib import Path

DEVICES = [
    ("Bus", 6), ("TransportLine", 7), ("ElectricalBranch", 9), ("Hydro", 12),
    ("Thermal", 8), ("Unit", 10), ("Load", 5), ("Storage", 8),
    ("Contract", 5), ("Pump", 6), ("Reserve", 5), ("Fuel", 6),
]
NETFORMS = ["CopperPlate", "Transport", "DcBtheta", "DcPtdf", "Lpac"]
COMFORMS = ["NoCommit", "Relaxed", "Binary"]
# devices whose build depends on the commitment axis
COMMIT_DEVICES = {"Thermal", "Unit", "Hydro"}


def snake(name):
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i > 0:
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def emit_block(rng, dev, fields, depth_tag):
    """One emission block: ~18-30 lines of loop-heavy builder calls with
    block-specific constants so instances differ."""
    f = lambda: f"e.f{rng.randrange(fields)}"
    c = lambda: f"{rng.uniform(0.05, 3.0):.4}"
    k1 = rng.randrange(3, 7)
    k2 = rng.randrange(2, 5)
    lines = []
    lines.append(f"    // {depth_tag}")
    lines.append(f"    for (i, e) in sys.{snake(dev)}s.iter().enumerate() {{")
    lines.append(f"        let c0 = b.add_col(0.0, {f()} * {c()}, {c()});")
    lines.append(f"        let c1 = b.add_col(-{f()}, {f()} + {c()}, {c()});")
    lines.append(f"        let r0 = b.add_row({f()} * {c()}, f64::INFINITY);")
    lines.append(f"        b.set(r0, c0, {c()});")
    lines.append(f"        b.set(r0, c1, -{c()});")
    lines.append(f"        for k in 0..{k1} {{")
    lines.append(f"            let rk = b.add_row(-1.0e30, {f()} + k as f64 * {c()});")
    lines.append(f"            b.set(rk, c0, {c()} + k as f64 * {c()});")
    lines.append(f"            if k % 2 == 0 {{ b.set(rk, c1, {c()}); }}")
    lines.append("        }")
    lines.append(f"        for k in 0..{k2} {{")
    lines.append(f"            let ck = b.add_col(0.0, {f()} * (1.0 + k as f64), {c()});")
    lines.append(f"            b.set(r0, ck, {c()} * (k as f64 + 1.0));")
    lines.append("        }")
    lines.append(f"        if i % {rng.randrange(5, 11)} == 0 {{")
    lines.append(f"            let cx = b.add_col(0.0, 1.0e30, {f()} * {c()});")
    lines.append(f"            b.set(r0, cx, {c()});")
    lines.append("        }")
    lines.append("    }")
    return "\n".join(lines)


def solver_crate(root):
    d = root / "spike-solver"
    (d / "src").mkdir(parents=True, exist_ok=True)
    (d / "Cargo.toml").write_text(
        '[package]\nname = "spike-solver"\nversion = "0.1.0"\nedition = "2021"\n'
        "[features]\ndefault = [\"backend-h\"]\nbackend-h = []\nbackend-c = []\n"
    )
    backends = []
    for tag, mul, feat in [("H", 1.618, "backend-h"), ("C", 2.414, "backend-c")]:
        rows = []
        for m in range(6):
            rows.append(
                f"    fn kernel{m}(&mut self) -> f64 {{\n"
                f"        let mut acc = {0.1 * (m + 1):.3} * {mul};\n"
                f"        for (i, v) in self.p.obj.iter().enumerate() {{\n"
                f"            acc += v * ((i % {m + 2}) as f64) * {mul / (m + 1):.4};\n"
                f"            if i % {m + 3} == 0 {{ acc *= 0.999_9; }}\n"
                f"        }}\n        acc\n    }}"
            )
        backends.append(f'''
#[cfg(feature = "{feat}")]
pub struct Backend{tag} {{ p: LpProblem, x: Vec<f64>, y: Vec<f64>, obj: f64 }}
#[cfg(feature = "{feat}")]
impl Backend{tag} {{
    pub fn new() -> Self {{ Self {{ p: LpProblem::default(), x: vec![], y: vec![], obj: 0.0 }} }}
{chr(10).join(rows)}
}}
#[cfg(feature = "{feat}")]
impl SolverInterface for Backend{tag} {{
    fn load(&mut self, p: &LpProblem) {{ self.p = p.clone(); self.x = vec![0.0; p.obj.len()]; self.y = vec![0.0; p.row_lower.len()]; }}
    fn solve(&mut self) -> i32 {{
        let mut o = 0.0;
        for m in 0..40 {{ o += self.kernel{0}() * (m as f64).mul_add(1e-3, 1.0); }}
        for (i, xi) in self.x.iter_mut().enumerate() {{ *xi = (self.p.col_lower[i].max(0.0) + i as f64 * 1e-6) * {mul}; }}
        for (j, yj) in self.y.iter_mut().enumerate() {{ *yj = (j as f64).mul_add(1e-7, 0.5) * {mul}; }}
        self.obj = o; 0
    }}
    fn objective(&self) -> f64 {{ self.obj }}
    fn primal(&self, i: usize) -> f64 {{ self.x[i] }}
    fn n_cols(&self) -> usize {{ self.x.len() }}
    fn set_col_bounds(&mut self, i: usize, lo: f64, hi: f64) {{ self.p.col_lower[i] = lo; self.p.col_upper[i] = hi; }}
    fn set_row_bounds(&mut self, i: usize, lo: f64, hi: f64) {{ self.p.row_lower[i] = lo; self.p.row_upper[i] = hi; }}
}}
#[cfg(feature = "{feat}")]
impl ProducesDuals for Backend{tag} {{ fn dual(&self, i: usize) -> f64 {{ self.y[i] * {mul} }} }}
#[cfg(feature = "{feat}")]
impl SupportsWarmStart for Backend{tag} {{ fn warm(&mut self, s: &[f64]) {{ for (a, b) in self.x.iter_mut().zip(s) {{ *a = *b; }} }} }}
''')
        # replace kernel index cycling: use kernel (m % 6)
    lib = f'''#![allow(clippy::all, dead_code)]
#[derive(Default, Clone)]
pub struct LpProblem {{
    pub col_lower: Vec<f64>, pub col_upper: Vec<f64>, pub obj: Vec<f64>,
    pub row_lower: Vec<f64>, pub row_upper: Vec<f64>,
    pub a: Vec<(u32, u32, f64)>,
}}
pub struct LpBuilder {{ pub p: LpProblem }}
impl LpBuilder {{
    pub fn new() -> Self {{ Self {{ p: LpProblem::default() }} }}
    #[inline]
    pub fn add_col(&mut self, lo: f64, hi: f64, obj: f64) -> usize {{
        self.p.col_lower.push(lo); self.p.col_upper.push(hi); self.p.obj.push(obj);
        self.p.obj.len() - 1
    }}
    #[inline]
    pub fn add_row(&mut self, lo: f64, hi: f64) -> usize {{
        self.p.row_lower.push(lo); self.p.row_upper.push(hi);
        self.p.row_lower.len() - 1
    }}
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {{ self.p.a.push((r as u32, c as u32, v)); }}
    pub fn finish(self) -> LpProblem {{ self.p }}
}}
pub trait SolverInterface {{
    fn load(&mut self, p: &LpProblem);
    fn solve(&mut self) -> i32;
    fn objective(&self) -> f64;
    fn primal(&self, i: usize) -> f64;
    fn n_cols(&self) -> usize;
    fn set_col_bounds(&mut self, i: usize, lo: f64, hi: f64);
    fn set_row_bounds(&mut self, i: usize, lo: f64, hi: f64);
}}
pub trait ProducesDuals: SolverInterface {{ fn dual(&self, i: usize) -> f64; }}
pub trait SupportsWarmStart: SolverInterface {{ fn warm(&mut self, s: &[f64]); }}
{"".join(backends)}
#[cfg(feature = "backend-h")]
pub type DefaultBackend = BackendH;
#[cfg(all(feature = "backend-c", not(feature = "backend-h")))]
pub type DefaultBackend = BackendC;
'''
    # fix kernel call: solve() references kernel{0} literal; patch to sum all kernels
    lib = lib.replace(
        "for m in 0..40 { o += self.kernel0() * (m as f64).mul_add(1e-3, 1.0); }",
        "for m in 0..40 { o += match m % 6 { 0 => self.kernel0(), 1 => self.kernel1(), 2 => self.kernel2(), 3 => self.kernel3(), 4 => self.kernel4(), _ => self.kernel5() } * (m as f64).mul_add(1e-3, 1.0); }",
    )
    (d / "src" / "lib.rs").write_text(lib)


def model_common(rng):
    """System + device structs + template enums (shared by all variants)."""
    parts = ["#![allow(clippy::all, dead_code)]\nuse spike_solver::LpBuilder;\n"]
    for dev, nf in DEVICES:
        fields = ", ".join(f"pub f{i}: f64" for i in range(nf))
        parts.append(f"#[derive(Clone)]\npub struct {dev} {{ {fields} }}")
    vecs = ", ".join(f"pub {snake(d)}s: Vec<{d}>" for d, _ in DEVICES)
    parts.append(f"pub struct System {{ {vecs} }}")
    ctor_fields = []
    for dev, nf in DEVICES:
        init = ", ".join(f"f{i}: (n + {i}) as f64 * {rng.uniform(0.1, 2.0):.4}" for i in range(nf))
        ctor_fields.append(
            f"        {snake(dev)}s: (0..n).map(|n| {dev} {{ {init} }}).collect()"
        )
    parts.append(
        "impl System {\n    pub fn synthetic(n: usize) -> Self {\n        Self {\n"
        + ",\n".join(ctor_fields)
        + "\n        }\n    }\n}"
    )
    parts.append(f"#[derive(Clone, Copy, PartialEq, Eq)]\npub enum NetForm {{ {', '.join(NETFORMS)} }}")
    parts.append(f"#[derive(Clone, Copy, PartialEq, Eq)]\npub enum ComForm {{ {', '.join(COMFORMS)} }}")
    parts.append("#[derive(Clone, Copy)]\npub struct ProblemTemplate { pub net: NetForm, pub com: ComForm }")
    return "\n\n".join(parts)


def model_variant_a(rng):
    """BuildProblem per device; runtime match on template enums."""
    parts = ["pub trait BuildProblem { fn build(sys: &System, t: &ProblemTemplate, b: &mut LpBuilder); }"]
    for dev, nf in DEVICES:
        arms = []
        for net in NETFORMS:
            block = emit_block(rng, dev, nf, f"{dev}/{net}")
            arms.append(f"            NetForm::{net} => {{\n{block}\n            }}")
        commit = ""
        if dev in COMMIT_DEVICES:
            carms = []
            for com in COMFORMS:
                block = emit_block(rng, dev, nf, f"{dev}/commit/{com}")
                carms.append(f"            ComForm::{com} => {{\n{block}\n            }}")
            commit = "\n        match t.com {\n" + "\n".join(carms) + "\n        }"
        parts.append(f'''pub struct {dev}F;
impl BuildProblem for {dev}F {{
    fn build(sys: &System, t: &ProblemTemplate, b: &mut LpBuilder) {{
        match t.net {{
{chr(10).join(arms)}
        }}{commit}
    }}
}}''')
    calls = "\n    ".join(f"{d}F::build(sys, t, b);" for d, _ in DEVICES)
    parts.append(f"pub fn build_all(sys: &System, t: &ProblemTemplate, b: &mut LpBuilder) {{\n    {calls}\n}}")
    return "\n\n".join(parts)


def model_variant_b(rng):
    """Type-level forms; device builds generic over <N, C>; dispatch registry."""
    parts = []
    parts.append("pub trait NetFormT { const ID: usize; fn w(k: usize) -> f64; }")
    for i, net in enumerate(NETFORMS):
        parts.append(
            f"pub struct T{net};\nimpl NetFormT for T{net} {{ const ID: usize = {i}; "
            f"#[inline] fn w(k: usize) -> f64 {{ {rng.uniform(0.3, 1.7):.4} + k as f64 * {rng.uniform(0.01, 0.2):.4} }} }}"
        )
    parts.append("pub trait ComFormT { const ID: usize; fn g(k: usize) -> f64; }")
    for i, com in enumerate(COMFORMS):
        parts.append(
            f"pub struct T{com};\nimpl ComFormT for T{com} {{ const ID: usize = {i}; "
            f"#[inline] fn g(k: usize) -> f64 {{ {rng.uniform(0.3, 1.7):.4} + k as f64 * {rng.uniform(0.01, 0.2):.4} }} }}"
        )
    for dev, nf in DEVICES:
        # generic body: one block per fidelity guarded by const ID (dead-code
        # eliminated per instantiation) + N::w / C::g calls keeping N,C live.
        blocks = []
        for i, net in enumerate(NETFORMS):
            block = emit_block(rng, dev, nf, f"{dev}/{net}/generic")
            block = block.replace("b.set(r0, c0, ", "b.set(r0, c0, N::w(1) * ")
            blocks.append(f"    if N::ID == {i} {{\n{block}\n    }}")
        commit = ""
        if dev in COMMIT_DEVICES:
            cblocks = []
            for i, com in enumerate(COMFORMS):
                block = emit_block(rng, dev, nf, f"{dev}/commit/{com}/generic")
                block = block.replace("b.set(r0, c1, ", "b.set(r0, c1, C::g(2) * ")
                cblocks.append(f"    if C::ID == {i} {{\n{block}\n    }}")
            commit = "\n" + "\n".join(cblocks)
        parts.append(
            f"pub fn build_{snake(dev)}<N: NetFormT, C: ComFormT>(sys: &System, b: &mut LpBuilder) {{\n"
            + "\n".join(blocks) + commit + "\n}"
        )
    calls = "\n    ".join(f"build_{snake(d)}::<N, C>(sys, b);" for d, _ in DEVICES)
    parts.append(f"pub fn build_all_t<N: NetFormT, C: ComFormT>(sys: &System, b: &mut LpBuilder) {{\n    {calls}\n}}")
    arms = []
    for net in NETFORMS:
        for com in COMFORMS:
            arms.append(
                f"        (NetForm::{net}, ComForm::{com}) => build_all_t::<T{net}, T{com}>(sys, b),"
            )
    parts.append(
        "pub fn build_dispatch(sys: &System, t: &ProblemTemplate, b: &mut LpBuilder) {\n"
        "    match (t.net, t.com) {\n" + "\n".join(arms) + "\n    }\n}"
    )
    return "\n\n".join(parts)


def model_base(rng):
    """Hardwired single formulation (Transport, NoCommit) — mimics today."""
    parts = []
    for dev, nf in DEVICES:
        block = emit_block(rng, dev, nf, f"{dev}/hardwired")
        parts.append(f"pub fn build_{snake(dev)}(sys: &System, b: &mut LpBuilder) {{\n{block}\n}}")
    calls = "\n    ".join(f"build_{snake(d)}(sys, b);" for d, _ in DEVICES)
    parts.append(f"pub fn build_all(sys: &System, _t: &ProblemTemplate, b: &mut LpBuilder) {{\n    {calls}\n}}")
    return "\n\n".join(parts)


ENGINE_TOML = '[package]\nname = "{name}"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\nspike-solver = {{ path = "../spike-solver" }}\nspike-model = {{ path = "../spike-model" }}\n'


def engine_direct(build_call):
    return f'''#![allow(clippy::all)]
use spike_model::{{ProblemTemplate, System}};
use spike_solver::{{LpBuilder, ProducesDuals, SolverInterface}};

pub fn run<S: SolverInterface + ProducesDuals>(sys: &System, t: &ProblemTemplate, s: &mut S) -> f64 {{
    let mut b = LpBuilder::new();
    {build_call}(sys, t, &mut b);
    let p = b.finish();
    s.load(&p);
    let _ = s.solve();
    let mut acc = s.objective();
    for i in 0..s.n_cols().min(64) {{ acc += s.primal(i) * 1e-3; }}
    acc += s.dual(0) * 0.5;
    acc
}}
'''


def engine_sddp(build_call):
    return f'''#![allow(clippy::all)]
use spike_model::{{ProblemTemplate, System}};
use spike_solver::{{LpBuilder, ProducesDuals, SolverInterface}};

pub fn run<S: SolverInterface + ProducesDuals>(sys: &System, t: &ProblemTemplate, s: &mut S) -> f64 {{
    let mut b = LpBuilder::new();
    {build_call}(sys, t, &mut b);
    let p = b.finish();
    s.load(&p);
    let mut bound = 0.0f64;
    for it in 0..20 {{
        for i in 0..s.n_cols().min(32) {{
            s.set_col_bounds(i, 0.0, 10.0 + it as f64 + i as f64 * 0.1);
        }}
        let _ = s.solve();
        bound = bound.max(s.objective());
        for i in 0..8 {{ bound += s.dual(i) * 1e-4; }}
    }}
    bound
}}
'''


def write_variant(root, variant):
    rng = random.Random(42)  # same seed → same emission blocks across variants
    root.mkdir(parents=True, exist_ok=True)
    solver_crate(root)
    members = ["spike-solver", "spike-model", "spike-cli"]
    md = root / "spike-model"
    (md / "src").mkdir(parents=True, exist_ok=True)
    (md / "Cargo.toml").write_text(
        '[package]\nname = "spike-model"\nversion = "0.1.0"\nedition = "2021"\n'
        '[dependencies]\nspike-solver = { path = "../spike-solver" }\n'
    )
    common = model_common(rng)
    if variant == "base":
        body = model_base(rng)
        build_call = "spike_model::build_all"
    elif variant == "a":
        body = model_variant_a(rng)
        build_call = "spike_model::build_all"
    else:
        body = model_variant_b(rng)
        build_call = "spike_model::build_dispatch"
    (md / "src" / "lib.rs").write_text(common + "\n\n" + body + "\n")

    if variant == "base":
        cli = f'''#![allow(clippy::all)]
use spike_model::{{ComForm, NetForm, ProblemTemplate, System}};
use spike_solver::{{DefaultBackend, LpBuilder, ProducesDuals, SolverInterface}};
fn main() {{
    let sys = System::synthetic(std::env::args().len() + 199);
    let t = ProblemTemplate {{ net: NetForm::Transport, com: ComForm::NoCommit }};
    let mut b = LpBuilder::new();
    spike_model::build_all(&sys, &t, &mut b);
    let p = b.finish();
    let mut s = DefaultBackend::new();
    s.load(&p);
    let _ = s.solve();
    let mut bound = 0.0f64;
    for it in 0..20 {{
        for i in 0..s.n_cols().min(32) {{ s.set_col_bounds(i, 0.0, 10.0 + it as f64 + i as f64 * 0.1); }}
        let _ = s.solve();
        bound = bound.max(s.objective());
        for i in 0..8 {{ bound += s.dual(i) * 1e-4; }}
    }}
    println!("{{bound}}");
}}
'''
    else:
        for name, gen in [("spike-direct", engine_direct(build_call)), ("spike-sddp", engine_sddp(build_call))]:
            ed = root / name
            (ed / "src").mkdir(parents=True, exist_ok=True)
            (ed / "Cargo.toml").write_text(ENGINE_TOML.format(name=name))
            (ed / "src" / "lib.rs").write_text(gen)
        members += ["spike-direct", "spike-sddp"]
        nets = ", ".join(f'"{n}" => NetForm::{n}' for n in NETFORMS)
        coms = ", ".join(f'"{c}" => ComForm::{c}' for c in COMFORMS)
        cli = f'''#![allow(clippy::all)]
use spike_model::{{ComForm, NetForm, ProblemTemplate, System}};
use spike_solver::DefaultBackend;
enum Engine {{ Direct, Sddp }}
fn main() {{
    let args: Vec<String> = std::env::args().collect();
    let engine = if args.get(1).map(String::as_str) == Some("direct") {{ Engine::Direct }} else {{ Engine::Sddp }};
    let net = match args.get(2).map(String::as_str).unwrap_or("Transport") {{ {nets}, _ => NetForm::Transport }};
    let com = match args.get(3).map(String::as_str).unwrap_or("NoCommit") {{ {coms}, _ => ComForm::NoCommit }};
    let sys = System::synthetic(200);
    let t = ProblemTemplate {{ net, com }};
    let mut s = DefaultBackend::new();
    let out = match engine {{
        Engine::Direct => spike_direct::run(&sys, &t, &mut s),
        Engine::Sddp => spike_sddp::run(&sys, &t, &mut s),
    }};
    println!("{{out}}");
}}
'''
    cd = root / "spike-cli"
    (cd / "src").mkdir(parents=True, exist_ok=True)
    deps = 'spike-solver = { path = "../spike-solver" }\nspike-model = { path = "../spike-model" }\n'
    if variant != "base":
        deps += 'spike-direct = { path = "../spike-direct" }\nspike-sddp = { path = "../spike-sddp" }\n'
    (cd / "Cargo.toml").write_text(
        '[package]\nname = "spike-cli"\nversion = "0.1.0"\nedition = "2021"\n[dependencies]\n' + deps
    )
    (cd / "src" / "main.rs").write_text(cli)
    members_s = ", ".join(f'"{m}"' for m in members)
    (root / "Cargo.toml").write_text(
        f'[workspace]\nresolver = "2"\nmembers = [{members_s}]\n'
        "[profile.release]\nstrip = \"symbols\"\n"
    )


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for v in ["base", "a", "b"]:
        write_variant(out / v, v)
    print("generated", out)
