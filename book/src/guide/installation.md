# Installation

Cobre is a statically linked binary available for the platforms listed below.
Choose the method that best fits your environment.

---

## Pre-built Binaries (Recommended)

No Rust toolchain or C compiler required.

### Linux and macOS

```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cobre-rs/cobre/releases/latest/download/cobre-cli-installer.sh | sh
```

The installer places the `cobre` binary in `$CARGO_HOME/bin` (typically
`~/.cargo/bin`). Add that directory to your `PATH` if it is not already present.

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://github.com/cobre-rs/cobre/releases/latest/download/cobre-cli-installer.ps1 | iex"
```

### Supported Platforms

| Platform              | Target Triple               |
| --------------------- | --------------------------- |
| macOS (Apple Silicon) | `aarch64-apple-darwin`      |
| macOS (Intel)         | `x86_64-apple-darwin`       |
| Linux (x86-64)        | `x86_64-unknown-linux-gnu`  |
| Linux (ARM64)         | `aarch64-unknown-linux-gnu` |
| Windows (x86-64)      | `x86_64-pc-windows-msvc`    |

You can also download individual archives directly from the
[GitHub Releases page](https://github.com/cobre-rs/cobre/releases/latest).

### Verify the Installation

```bash
cobre version
```

Expected output (exact versions and arch will vary):

```
cobre   v0.9.1
solver: HiGHS
comm:   local
zstd:   enabled
arch:   x86_64-linux
build:  release (lto=thin)
```

---

## From crates.io

```bash
cargo install cobre-cli
```

Requires Rust 1.88+ and build prerequisites (see Build from Source below).
Installs to `$CARGO_HOME/bin`.

---

## Build from Source

For contributors or unsupported platforms.

### Prerequisites

| Dependency     | Minimum Version         | Notes                                   |
| -------------- | ----------------------- | --------------------------------------- |
| Rust toolchain | 1.88 (stable)           | Install via [rustup](https://rustup.rs) |
| C compiler     | any recent GCC or Clang | Required for the HiGHS LP solver        |
| CMake          | 3.15                    | Required for the HiGHS build system     |
| Git            | any                     | Required for submodule initialization   |

### Steps

```bash
# Clone the repository
git clone https://github.com/cobre-rs/cobre.git
cd cobre

# Initialize HiGHS submodule (required for the solver backend)
git submodule update --init --recursive

# Build the release binary
cargo build --release -p cobre-cli
```

The binary is written to `target/release/cobre`. Optionally install to `$CARGO_HOME/bin`:

```bash
cargo install --path crates/cobre-cli
```

Verify:

```bash
./target/release/cobre version
cargo test --workspace
```

---

## Choosing the LP Backend

Cobre supports two LP solver backends, selected at build time via Cargo features.
Exactly one backend is compiled into any given binary.

| Backend | Feature flag | License | Notes                                                   |
| ------- | ------------ | ------- | ------------------------------------------------------- |
| HiGHS   | `highs`      | MIT     | Default. No extra steps required.                       |
| CLP     | `clp`        | EPL-2.0 | COIN-OR. Opt-in; requires the CLP/CoinUtils submodules. |

### Default build (HiGHS)

```bash
cargo build --release -p cobre-cli
```

No flags are needed. HiGHS is the default backend and the one shipped in
pre-built binaries.

### CLP build

```bash
# Initialize the CLP and CoinUtils submodules first
git submodule update --init --recursive

# Build with CLP, disabling the HiGHS default
cargo build --release -p cobre-cli --no-default-features --features clp
```

### Mutual exclusivity

The `highs` and `clp` features are mutually exclusive — exactly one LP backend
is compiled into a binary, and enabling both at once is a compile error. Because
`highs` is the default feature, selecting CLP requires `--no-default-features`
to suppress the default before `--features clp` is applied; a plain
`--features clp` leaves the `highs` default on and fails the build. Enabling
neither backend is also a compile error, so a backend is always chosen
explicitly. The default build (no extra flags) uses HiGHS.

### Identifying the active backend

The `cobre version` banner shows which backend is compiled in:

```
cobre   v0.9.1
solver: CLP 1.17.11
comm:   local
...
```

The `solver` and `solver_version` fields in each run's
[output metadata](../reference/output-format.md) record the active backend
identifier (`"highs"` or `"clp"`) and its library version string. These fields
are written by both the CLI and the Python bindings.

### Determinism

Each backend is internally deterministic: the same input, solved twice, produces
bit-for-bit identical results; permuting the input entities produces the
correspondingly permuted output. Switching from one backend to the other may
legitimately change numerical results — the two simplex implementations can reach
different optimal vertices on degenerate problems, all of which are valid. No
cross-backend numerical equality is guaranteed; each backend maintains its own
parity baselines.

### Migration note

Existing builds are unaffected. The default backend is HiGHS, unchanged from
prior releases. The CLP backend is strictly opt-in: users who do not pass
`--no-default-features --features clp` continue to build and run against HiGHS
exactly as before.

### Known limitation

Re-loading a fresh model into a CLP solver instance after a hot-start snapshot
has been taken is unsupported and guarded against at runtime. This situation does
not arise on the production solve paths; it is relevant only to callers that
construct solver instances directly and interleave `load_model` calls with
hot-start operations.

---

## Next Steps

- [Quickstart](../tutorial/quickstart.md) — run a complete study end to end using the built-in `1dtoy` template
- [Running Studies](./running-studies.md) — validate, run, and inspect results for any case directory
- [CLI Reference](./cli-reference.md) — complete flag and subcommand reference
