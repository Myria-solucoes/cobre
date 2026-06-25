//! Factory function for creating the active communication backend.
//!
//! [`create_communicator`] is the single runtime entry point for constructing a
//! [`Communicator`](crate::Communicator). Selection order: the
//! `COBRE_COMM_BACKEND` environment variable, then compiled-in Cargo features
//! (`mpi`), then a fallback to [`LocalBackend`](crate::LocalBackend).

/// Programmatic backend selector for library-mode callers that pass a
/// `BackendKind` to [`create_communicator`] instead of using environment
/// variables.
///
/// # Examples
///
/// ```rust
/// use cobre_comm::BackendKind;
///
/// let kind = BackendKind::Auto;
/// let copy = kind;
/// assert_eq!(copy, kind);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Auto-detect the best available backend (same order as the env-var path).
    Auto,
    /// MPI backend; fails at runtime if the `mpi` feature is not compiled in.
    Mpi,
    /// Single-process local backend; always available.
    Local,
}

/// Enum-dispatched communicator wrapping any available concrete backend.
///
/// Enum dispatch (not `Box<dyn>`) because [`crate::Communicator`] carries
/// generic methods that make it non-object-safe; dispatch overhead is negligible
/// against the MPI collective or LP solve it wraps. `CommBackend: Send + Sync`
/// because all inner backends are. Only present in `mpi` builds; no-feature
/// builds use [`crate::LocalBackend`] directly.
#[cfg(feature = "mpi")]
pub enum CommBackend {
    /// MPI backend powered by ferrompi.
    Mpi(Box<crate::FerrompiBackend>),

    /// Single-process local backend (always-available fallback).
    Local(crate::LocalBackend),
}

#[cfg(feature = "mpi")]
// Rationale: never called; its monomorphisation is the compile-time check that
// `CommBackend: Send + Sync`, catching any future field that breaks it.
#[allow(dead_code)]
const fn _assert_comm_backend_send_sync() {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CommBackend>();
}

#[cfg(feature = "mpi")]
impl crate::Communicator for CommBackend {
    fn allgatherv<T: crate::CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        counts: &[usize],
        displs: &[usize],
    ) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.allgatherv(send, recv, counts, displs),
            Self::Local(backend) => backend.allgatherv(send, recv, counts, displs),
        }
    }

    fn allreduce<T: crate::CommData>(
        &self,
        send: &[T],
        recv: &mut [T],
        op: crate::ReduceOp,
    ) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.allreduce(send, recv, op),
            Self::Local(backend) => backend.allreduce(send, recv, op),
        }
    }

    fn broadcast<T: crate::CommData>(
        &self,
        buf: &mut [T],
        root: usize,
    ) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.broadcast(buf, root),
            Self::Local(backend) => backend.broadcast(buf, root),
        }
    }

    fn barrier(&self) -> Result<(), crate::CommError> {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.barrier(),
            Self::Local(backend) => backend.barrier(),
        }
    }

    fn rank(&self) -> usize {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.rank(),
            Self::Local(backend) => backend.rank(),
        }
    }

    fn size(&self) -> usize {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.size(),
            Self::Local(backend) => backend.size(),
        }
    }

    fn abort(&self, error_code: i32) -> ! {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.abort(error_code),
            Self::Local(backend) => backend.abort(error_code),
        }
    }
}

#[cfg(all(feature = "mpi", feature = "shared-memory"))]
impl crate::SharedMemoryProvider for CommBackend {
    /// `HeapRegion<T>` directly (not an enum wrapper): both inner backends
    /// already unify on it as their `Region<T>`.
    type Region<T: crate::CommData> = crate::HeapRegion<T>;

    fn create_shared_region<T: crate::CommData>(
        &self,
        count: usize,
    ) -> Result<Self::Region<T>, crate::CommError> {
        match self {
            Self::Mpi(backend) => backend.create_shared_region(count),
            Self::Local(backend) => backend.create_shared_region(count),
        }
    }

    fn split_local(&self) -> Result<crate::traits::LocalCommKind, crate::CommError> {
        match self {
            Self::Mpi(backend) => backend.split_local(),
            Self::Local(backend) => backend.split_local(),
        }
    }

    fn is_leader(&self) -> bool {
        match self {
            Self::Mpi(backend) => backend.is_leader(),
            Self::Local(backend) => backend.is_leader(),
        }
    }
}

#[cfg(feature = "mpi")]
impl crate::TopologyProvider for CommBackend {
    fn topology(&self) -> &crate::ExecutionTopology {
        match self {
            #[cfg(feature = "mpi")]
            Self::Mpi(backend) => backend.topology(),
            Self::Local(backend) => backend.topology(),
        }
    }
}

/// Returns all backend names compiled into this binary.
///
/// Always includes `"local"`. Conditionally includes `"mpi"` (feature `mpi`).
///
/// # Examples
///
/// ```rust
/// use cobre_comm::available_backends;
///
/// let backends = available_backends();
/// assert!(backends.contains(&"local".to_string()));
/// ```
#[must_use]
#[allow(clippy::vec_init_then_push)] // cfg-gated push pattern
pub fn available_backends() -> Vec<String> {
    let mut backends = Vec::new();
    #[cfg(feature = "mpi")]
    backends.push("mpi".to_string());
    backends.push("local".to_string());
    backends
}

/// Returns `true` if any MPI launcher environment variable is present (checked
/// via `var_os`, so non-UTF-8 values still count).
///
/// Always compiled (no cfg gate) so it is testable in no-feature builds, where
/// it is unused outside tests — hence the dead-code allow.
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
fn mpi_launch_detected() -> bool {
    const MPI_ENV_VARS: [&str; 6] = [
        "PMI_RANK",
        "PMI_SIZE",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "MPI_LOCALRANKID",
        "SLURM_PROCID",
    ];
    MPI_ENV_VARS
        .iter()
        .any(|var| std::env::var_os(var).is_some())
}

/// Construct the active communication backend (no-feature build).
///
/// When the `mpi` feature is not compiled in, this function always returns a
/// [`crate::LocalBackend`] or an error:
///
/// - `COBRE_COMM_BACKEND` unset, `"auto"`, or `"local"` → `Ok(LocalBackend)`
/// - A known distributed backend name (`"mpi"`) →
///   `Err(BackendError::BackendNotAvailable)`
/// - An unknown name → `Err(BackendError::InvalidBackend)`
///
/// # Errors
///
/// - [`crate::BackendError::BackendNotAvailable`]: a known backend was requested
///   but not compiled into this binary.
/// - [`crate::BackendError::InvalidBackend`]: `COBRE_COMM_BACKEND` contains an
///   unrecognized value.
///
/// # Examples
///
/// ```rust
/// # #[cfg(not(feature = "mpi"))]
/// # {
/// use cobre_comm::create_communicator;
///
/// // With no distributed features, the factory always returns LocalBackend.
/// let backend = create_communicator().expect("local backend must succeed");
/// # use cobre_comm::Communicator;
/// assert_eq!(backend.rank(), 0);
/// assert_eq!(backend.size(), 1);
/// # }
/// ```
#[cfg(not(feature = "mpi"))]
pub fn create_communicator() -> Result<crate::LocalBackend, crate::BackendError> {
    let requested = std::env::var("COBRE_COMM_BACKEND").unwrap_or_else(|_| "auto".to_string());
    match requested.as_str() {
        "auto" | "local" => Ok(crate::LocalBackend),
        "mpi" => Err(crate::BackendError::BackendNotAvailable {
            requested,
            available: available_backends(),
        }),
        _ => Err(crate::BackendError::InvalidBackend {
            requested,
            available: vec!["auto", "mpi", "local"]
                .into_iter()
                .map(String::from)
                .collect(),
        }),
    }
}

/// Construct the active communication backend (MPI build).
///
/// When the `mpi` feature is compiled in, this function returns a
/// [`CommBackend`] selected according to the `COBRE_COMM_BACKEND` environment
/// variable:
///
/// - Unset or `"auto"` → auto-detect priority chain
/// - `"mpi"` → `CommBackend::Mpi(FerrompiBackend::new()?)`
/// - `"local"` → `CommBackend::Local(LocalBackend)`
/// - Unknown name → `Err(BackendError::InvalidBackend)`
///
/// # Errors
///
/// - [`crate::BackendError::InvalidBackend`]: `COBRE_COMM_BACKEND` contains an
///   unrecognized value.
/// - [`crate::BackendError::InitializationFailed`]: the selected backend failed
///   to initialize (propagated from [`crate::FerrompiBackend::new`]).
///
/// # Examples
///
/// ```rust
/// # #[cfg(feature = "mpi")]
/// # {
/// use cobre_comm::{create_communicator, Communicator};
///
/// // With COBRE_COMM_BACKEND unset or "local", returns CommBackend::Local.
/// // std::env::remove_var is unsafe in multi-threaded contexts (Rust 2024).
/// // SAFETY: this doctest runs single-threaded; no concurrent env mutation.
/// unsafe { std::env::set_var("COBRE_COMM_BACKEND", "local") };
/// let backend = create_communicator().expect("local backend must succeed");
/// assert_eq!(backend.rank(), 0);
/// unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
/// # }
/// ```
#[cfg(feature = "mpi")]
pub fn create_communicator() -> Result<CommBackend, crate::BackendError> {
    let requested = std::env::var("COBRE_COMM_BACKEND").unwrap_or_else(|_| "auto".to_string());
    match requested.as_str() {
        "auto" => auto_detect(),
        "mpi" => Ok(CommBackend::Mpi(Box::new(crate::FerrompiBackend::new()?))),
        "local" => Ok(CommBackend::Local(crate::LocalBackend)),
        _ => Err(crate::BackendError::InvalidBackend {
            requested,
            available: vec!["auto", "mpi", "local"]
                .into_iter()
                .map(String::from)
                .collect(),
        }),
    }
}

/// Auto-detect the backend: MPI when [`mpi_launch_detected`] is `true`,
/// otherwise the local fallback.
///
/// # Errors
///
/// Returns [`crate::BackendError::InitializationFailed`] if the MPI backend is
/// selected but [`crate::FerrompiBackend::new`] fails.
#[cfg(feature = "mpi")]
fn auto_detect() -> Result<CommBackend, crate::BackendError> {
    #[cfg(feature = "mpi")]
    if mpi_launch_detected() {
        return Ok(CommBackend::Mpi(Box::new(crate::FerrompiBackend::new()?)));
    }
    Ok(CommBackend::Local(crate::LocalBackend))
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::BackendKind;

    use super::{available_backends, mpi_launch_detected};

    /// Serialises tests that mutate `COBRE_COMM_BACKEND`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// `available_backends()` always contains `"local"` regardless of features.
    #[test]
    fn test_available_backends_contains_local() {
        let backends = available_backends();
        assert!(
            backends.contains(&"local".to_string()),
            "expected 'local' in {backends:?}"
        );
    }

    /// In a no-feature build `available_backends()` returns exactly `["local"]`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_available_backends_no_feature_exact() {
        assert_eq!(available_backends(), vec!["local".to_string()]);
    }

    /// `mpi_launch_detected()` returns `false` when none of the MPI env vars
    /// are set.
    #[test]
    fn test_mpi_launch_detected_false_by_default() {
        const MPI_VARS: [&str; 6] = [
            "PMI_RANK",
            "PMI_SIZE",
            "OMPI_COMM_WORLD_RANK",
            "OMPI_COMM_WORLD_SIZE",
            "MPI_LOCALRANKID",
            "SLURM_PROCID",
        ];
        // Hold ENV_LOCK to prevent races with tests that set/remove MPI vars.
        let _guard = ENV_LOCK.lock().unwrap();
        let any_set = MPI_VARS.iter().any(|v| std::env::var_os(v).is_some());
        if any_set {
            // Running inside a real MPI launch; skip rather than fail.
            return;
        }
        assert!(!mpi_launch_detected());
    }

    /// `mpi_launch_detected()` returns `true` when `PMI_RANK` is set.
    #[test]
    fn test_mpi_launch_detected_pmi_rank() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK; no concurrent env var access.
        unsafe { std::env::set_var("PMI_RANK", "0") };
        let result = mpi_launch_detected();
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("PMI_RANK") };
        assert!(
            result,
            "expected mpi_launch_detected() == true when PMI_RANK is set"
        );
    }

    /// `mpi_launch_detected()` returns `true` when `OMPI_COMM_WORLD_RANK` is set.
    #[test]
    fn test_mpi_launch_detected_ompi() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: serialised by ENV_LOCK.
        unsafe { std::env::set_var("OMPI_COMM_WORLD_RANK", "0") };
        let result = mpi_launch_detected();
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("OMPI_COMM_WORLD_RANK") };
        assert!(
            result,
            "expected mpi_launch_detected() == true when OMPI_COMM_WORLD_RANK is set"
        );
    }

    /// No-feature build: unset `COBRE_COMM_BACKEND` → `Ok(LocalBackend)` with
    /// rank 0 and size 1.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_auto() {
        use crate::Communicator;

        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        let backend = super::create_communicator().expect("LocalBackend construction must succeed");
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.size(), 1);
    }

    /// No-feature build: `COBRE_COMM_BACKEND=foobar` → `Err(InvalidBackend)`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_invalid() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "foobar") };
        let err = super::create_communicator().expect_err("unknown backend must return Err");
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        assert!(
            matches!(
                err,
                crate::BackendError::InvalidBackend { ref requested, .. }
                    if requested == "foobar"
            ),
            "unexpected error: {err:?}"
        );
    }

    /// No-feature build: `COBRE_COMM_BACKEND=mpi` → `Err(BackendNotAvailable)`
    /// where `requested == "mpi"` and `available` contains `"local"`.
    #[test]
    #[cfg(not(feature = "mpi"))]
    fn test_create_communicator_no_feature_unavailable() {
        let _guard = ENV_LOCK.lock().unwrap();
        unsafe { std::env::set_var("COBRE_COMM_BACKEND", "mpi") };
        let err = super::create_communicator().expect_err("unavailable backend must return Err");
        // SAFETY: symmetric with set_var above.
        unsafe { std::env::remove_var("COBRE_COMM_BACKEND") };
        assert!(
            matches!(err, crate::BackendError::BackendNotAvailable { .. }),
            "expected BackendNotAvailable, got {err:?}"
        );
        if let crate::BackendError::BackendNotAvailable {
            ref requested,
            ref available,
        } = err
        {
            assert_eq!(requested, "mpi");
            assert!(
                available.contains(&"local".to_string()),
                "available should contain 'local', got {available:?}"
            );
        }
    }

    /// Verify that `BackendKind` derives `Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`.
    #[test]
    fn test_backend_kind_derives() {
        let kind = BackendKind::Auto;

        let s = format!("{kind:?}");
        assert!(s.contains("Auto"), "Debug output should contain 'Auto'");

        // Rationale: explicit Clone::clone on a Copy type proves Clone is derived.
        #[allow(clippy::clone_on_copy)]
        let cloned = kind.clone();
        assert_eq!(cloned, kind);

        let copied = kind;
        assert_eq!(copied, kind);

        assert_eq!(BackendKind::Mpi, BackendKind::Mpi);
        assert_ne!(BackendKind::Mpi, BackendKind::Local);
        assert_eq!(BackendKind::Local, BackendKind::Local);
    }

    #[cfg(feature = "mpi")]
    #[allow(clippy::float_cmp)]
    mod comm_backend {
        use super::super::CommBackend;
        use crate::{Communicator, LocalBackend, ReduceOp};

        #[cfg(feature = "shared-memory")]
        use crate::{LocalCommunicator, SharedMemoryProvider, SharedRegion};

        /// Compile-time assertion that `CommBackend: Send + Sync`.
        #[test]
        fn test_comm_backend_send_sync() {
            fn assert_send_sync<T: Send + Sync>() {}
            assert_send_sync::<CommBackend>();
        }

        /// `CommBackend::Local` delegates `rank()` → 0 and `size()` → 1.
        #[test]
        fn test_comm_backend_local_rank_size() {
            let backend = CommBackend::Local(LocalBackend);
            assert_eq!(backend.rank(), 0);
            assert_eq!(backend.size(), 1);
        }

        /// `CommBackend::Local` delegates `barrier()` → `Ok(())`.
        #[test]
        fn test_comm_backend_local_barrier() {
            let backend = CommBackend::Local(LocalBackend);
            assert!(backend.barrier().is_ok());
        }

        /// `CommBackend::Local` delegates `allreduce` with identity-copy semantics.
        #[test]
        fn test_comm_backend_local_allreduce() {
            let backend = CommBackend::Local(LocalBackend);
            let send = [1.0_f64, 2.0, 3.0];
            let mut recv = [0.0_f64; 3];
            backend.allreduce(&send, &mut recv, ReduceOp::Sum).unwrap();
            assert_eq!(recv, [1.0, 2.0, 3.0]);
        }

        /// `CommBackend::Local` delegates `allgatherv` with identity-copy semantics.
        #[test]
        fn test_comm_backend_local_allgatherv() {
            let backend = CommBackend::Local(LocalBackend);
            let send = [7.0_f64, 8.0, 9.0];
            let mut recv = [0.0_f64; 3];
            backend.allgatherv(&send, &mut recv, &[3], &[0]).unwrap();
            assert_eq!(recv, [7.0, 8.0, 9.0]);
        }

        /// `CommBackend::Local` delegates `broadcast` as a no-op for root 0.
        #[test]
        fn test_comm_backend_local_broadcast() {
            let backend = CommBackend::Local(LocalBackend);
            let mut buf = [1.0_f64, 2.0];
            assert!(backend.broadcast(&mut buf, 0).is_ok());
            assert_eq!(buf, [1.0, 2.0]);
        }

        /// `CommBackend::Local` delegates `SharedMemoryProvider` methods correctly.
        ///
        /// Covers: `create_shared_region`, `split_local`, `is_leader`.
        #[cfg(feature = "shared-memory")]
        #[test]
        fn test_comm_backend_local_shared_memory() {
            let backend = CommBackend::Local(LocalBackend);

            let mut region = backend.create_shared_region::<f64>(10).unwrap();
            assert_eq!(region.as_slice().len(), 10);
            region.as_mut_slice().fill(42.0);
            assert_eq!(region.as_slice(), &[42.0_f64; 10]);
            assert!(region.fence().is_ok());

            let local_comm = backend.split_local().unwrap();
            assert_eq!(local_comm.rank(), 0);
            assert_eq!(local_comm.size(), 1);
            assert!(local_comm.barrier().is_ok());

            assert!(backend.is_leader());
        }
    }
}
