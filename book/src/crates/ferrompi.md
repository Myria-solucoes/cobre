# ferrompi

<span class="status-alpha">alpha</span>

Safe MPI 4.x bindings for Rust, used by `cobre-comm` as the MPI communication backend. This is a separate repository at [github.com/cobre-rs/ferrompi](https://github.com/cobre-rs/ferrompi).

ferrompi provides type-safe wrappers around MPI collective operations (`allgatherv`, `allreduce`, `broadcast`, `barrier`) with RAII-managed `MPI_Init_thread` / `MPI_Finalize` lifecycle. It supports `ThreadLevel::Funneled` initialization, which matches the Cobre execution model where only the main thread issues MPI calls.

See the [ferrompi README](https://github.com/cobre-rs/ferrompi) and the [cobre-comm crate README](https://github.com/cobre-rs/cobre/blob/main/crates/cobre-comm/README.md) for the backend integration.
