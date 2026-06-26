//! Cut synchronization across MPI ranks after the backward pass.
//!
//! Each rank's newly generated cuts are exchanged via a per-stage `allgatherv`
//! of serialized records in the [`cut::wire`] format, so the FCF is bit-for-bit
//! identical across all ranks at the end of each iteration (the next forward
//! pass rebuilds the LP from it).
//!
//! `sync_cuts` inserts only **remote** cuts: the backward pass already inserted
//! the local rank's own cuts, so the local segment of the receive buffer is
//! skipped — re-inserting it would double-count cuts.
//!
//! The `allgatherv` acts as an implicit barrier; no explicit `comm.barrier()`
//! is needed. Buffers pre-allocated in [`CutSyncBuffers::new`] are reused so the
//! per-stage exchange is allocation-free.
//!
//! Serialization/version handling is delegated to [`cut::wire`] (wire version 1);
//! the version-reject contract lives there, not here.
//!
//! [`cut::wire`]: crate::cut::wire

use cobre_comm::Communicator;

use crate::{
    FutureCostFunction, SddpError,
    cut::wire::{CutWireHeader, cut_wire_size, deserialize_cuts_from_buffer_into, serialize_cut},
};

/// Pre-allocated byte buffers for gathering cut wire records across all MPI
/// ranks via [`Communicator::allgatherv`] with `T = u8`.
///
/// # Buffer layout
///
/// | Buffer      | Capacity                                             | Description                                                   |
/// |-------------|------------------------------------------------------|---------------------------------------------------------------|
/// | `send_buf`  | `max_cuts_per_rank * cut_wire_size(n_state)`         | This rank's serialized cut records                            |
/// | `recv_buf`  | `max_cuts_per_rank * num_ranks * cut_wire_size(n_state)` | All ranks' serialized cut records in rank-major order     |
/// | `counts`    | `num_ranks`                                          | Per-rank byte count (`actual_cuts * record_size`)             |
/// | `displs`    | `num_ranks`                                          | Per-rank byte displacement (`sum of preceding counts`)        |
///
/// # Examples
///
/// ```rust
/// use cobre_comm::LocalBackend;
/// use cobre_sddp::cut_sync::CutSyncBuffers;
/// use cobre_sddp::cut::fcf::FutureCostFunction;
///
/// // Single rank, 2 state dimensions, 2 cuts per rank.
/// // max_cuts_per_rank must equal the number of cuts actually passed to
/// // sync_cuts so that per_rank_cuts[0] matches local_cuts.len().
/// let mut bufs = CutSyncBuffers::new(2, 2, 1);
///
/// let mut fcf = FutureCostFunction::new(2, 2, 2, 10, &[0; 2]);
/// let comm = LocalBackend;
///
/// let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[
///     (0, 1, 0, 10.0, &[1.0, 2.0]),
///     (0, 1, 1, 20.0, &[3.0, 4.0]),
/// ];
///
/// let remote_count = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();
/// // Single-rank: no remote cuts inserted.
/// assert_eq!(remote_count, 0);
/// ```
#[derive(Debug, Clone)]
pub struct CutSyncBuffers {
    /// This rank's serialized records; only the leading
    /// `actual_cuts * record_size` bytes are sent each call.
    send_buf: Vec<u8>,

    /// All ranks' serialized records; rank `r`'s records occupy
    /// `recv_buf[displs[r]..displs[r] + counts[r]]` after `allgatherv`.
    recv_buf: Vec<u8>,

    /// Per-rank byte count for `allgatherv`, recomputed each `sync_cuts` call.
    counts: Vec<usize>,

    /// Per-rank byte displacement for `allgatherv`: entry `r` = sum of
    /// `counts[0..r]`.
    displs: Vec<usize>,

    /// Length of the state vector (number of cut coefficients).
    n_state: usize,

    /// Total number of MPI ranks.
    num_ranks: usize,

    /// Cached wire record size: `cut_wire_size(n_state)`.
    record_size: usize,

    /// Per-rank expected cut counts: entry `r` is the number of cuts rank `r`
    /// generates per stage per iteration, sizing each rank's `allgatherv` slot.
    per_rank_cuts: Vec<usize>,

    /// Deserialization scratch for cut headers; grown lazily, never shrunk.
    deserialize_headers_buf: Vec<CutWireHeader>,

    /// Deserialization scratch for coefficients (flat layout); grown lazily,
    /// never shrunk.
    deserialize_coefficients_buf: Vec<f64>,
}

impl CutSyncBuffers {
    /// Construct pre-allocated cut synchronization buffers for the given
    /// topology, assuming a uniform distribution of `max_cuts_per_rank` cuts
    /// per rank.
    ///
    /// # Arguments
    ///
    /// - `n_state` — state dimension (number of cut coefficients per cut).
    /// - `max_cuts_per_rank` — maximum number of cuts any rank generates per
    ///   stage per iteration. Used to pre-allocate buffer capacity.
    /// - `num_ranks` — total number of MPI ranks (`comm.size()`).
    #[must_use]
    pub fn new(n_state: usize, max_cuts_per_rank: usize, num_ranks: usize) -> Self {
        Self::with_distribution(
            n_state,
            max_cuts_per_rank,
            num_ranks,
            max_cuts_per_rank * num_ranks,
        )
    }

    /// Construct buffers for non-uniform work distribution.
    ///
    /// When the total number of forward passes does not divide evenly among
    /// ranks, the first `total_forward_passes % num_ranks` ranks each handle
    /// one extra forward pass. This constructor sizes buffers for the maximum
    /// per-rank count and records each rank's expected count for correct
    /// `allgatherv` displacements.
    ///
    /// # Arguments
    ///
    /// - `n_state` — state dimension (number of cut coefficients per cut).
    /// - `max_cuts_per_rank` — maximum cuts any rank generates per stage per
    ///   iteration. Used to size the send buffer.
    /// - `num_ranks` — total number of MPI ranks.
    /// - `total_forward_passes` — total forward passes across all ranks. Used
    ///   to compute per-rank expected cut counts.
    #[must_use]
    pub fn with_distribution(
        n_state: usize,
        max_cuts_per_rank: usize,
        num_ranks: usize,
        total_forward_passes: usize,
    ) -> Self {
        let record_size = cut_wire_size(n_state);
        let send_cap = max_cuts_per_rank * record_size;

        let base = total_forward_passes / num_ranks;
        let remainder = total_forward_passes % num_ranks;
        let per_rank_cuts: Vec<usize> = (0..num_ranks)
            .map(|r| base + usize::from(r < remainder))
            .collect();
        let recv_cap: usize = per_rank_cuts.iter().sum::<usize>() * record_size;

        let counts: Vec<usize> = per_rank_cuts.iter().map(|&c| c * record_size).collect();
        let mut displs = vec![0usize; num_ranks];
        for r in 1..num_ranks {
            displs[r] = displs[r - 1] + counts[r - 1];
        }

        Self {
            send_buf: vec![0u8; send_cap],
            recv_buf: vec![0u8; recv_cap],
            counts,
            displs,
            n_state,
            num_ranks,
            record_size,
            per_rank_cuts,
            deserialize_headers_buf: Vec::new(),
            deserialize_coefficients_buf: Vec::new(),
        }
    }

    /// Synchronize locally generated cuts across all MPI ranks for one stage,
    /// inserting only remote cuts into the FCF (local cuts already inserted by
    /// the backward pass are skipped).
    ///
    /// # Arguments
    ///
    /// - `stage` — 0-based stage index for which cuts are being synchronized.
    /// - `local_cuts` — locally generated cuts as `(slot_index, iteration,
    ///   forward_pass_index, intercept, coefficients)` tuples. The backward
    ///   pass has already inserted these cuts into the FCF; they are serialized
    ///   here to send to remote ranks, but are **not** re-inserted locally.
    /// - `fcf` — Future Cost Function to receive remote cuts.
    /// - `comm` — communicator for the `allgatherv` call.
    ///
    /// # Returns
    ///
    /// `Ok(n)` where `n` is the number of remote cuts inserted into `fcf`.
    /// In single-rank mode, returns `Ok(0)` because the only rank's segment
    /// is skipped during deserialization.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Validation(_))` if `local_cuts.len()` does not
    /// equal `per_rank_cuts[my_rank]`, the expected cut count for this rank as
    /// established by the cut-distribution plan. Allowing a mismatch to proceed
    /// would corrupt remote ranks' deserialized cut buffers, so this invariant
    /// is enforced in both debug and release builds.
    ///
    /// Returns `Err(SddpError::Communication(_))` if the underlying
    /// `allgatherv` call fails. The FCF and buffer contents are unspecified
    /// on error.
    ///
    /// # Panics (debug builds only)
    ///
    /// - Panics if `local_cuts.len() * record_size > send_buf.len()`.
    /// - Panics if any cut's coefficient slice length does not equal `n_state`.
    pub fn sync_cuts<C: Communicator>(
        &mut self,
        stage: usize,
        local_cuts: &[(u32, u32, u32, f64, &[f64])],
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<usize, SddpError> {
        let n_local = local_cuts.len();
        let my_rank = comm.rank();
        let expected_for_me = self.per_rank_cuts[my_rank];
        if n_local != expected_for_me {
            return Err(SddpError::Validation(format!(
                "sync_cuts invariant violated at stage {stage}: rank \
                 {my_rank} produced {n_local} cuts, expected \
                 {expected_for_me} per the cut-distribution plan. \
                 Releasing this divergence to allgatherv would corrupt \
                 remote ranks' deserialized cut buffers."
            )));
        }

        let send_len = n_local * self.record_size;

        debug_assert!(
            send_len <= self.send_buf.len(),
            "send_len {send_len} exceeds send_buf capacity {}",
            self.send_buf.len()
        );

        for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
            local_cuts.iter().enumerate()
        {
            debug_assert!(
                coefficients.len() == self.n_state,
                "cut {i} coefficient length {} != n_state {}",
                coefficients.len(),
                self.n_state,
            );
            let start = i * self.record_size;
            serialize_cut(
                &mut self.send_buf[start..start + self.record_size],
                slot_index,
                iteration,
                forward_pass_index,
                intercept,
                coefficients,
            );
        }

        for r in 0..self.num_ranks {
            let cuts_for_r = if r == my_rank {
                n_local
            } else {
                self.per_rank_cuts[r]
            };
            self.counts[r] = cuts_for_r * self.record_size;
        }
        self.displs[0] = 0;
        for r in 1..self.num_ranks {
            self.displs[r] = self.displs[r - 1] + self.counts[r - 1];
        }

        let recv_len: usize = self.counts.iter().sum();
        debug_assert!(
            recv_len <= self.recv_buf.len(),
            "recv_len {recv_len} exceeds recv_buf capacity {}",
            self.recv_buf.len()
        );

        comm.allgatherv(
            &self.send_buf[..send_len],
            &mut self.recv_buf[..recv_len],
            &self.counts,
            &self.displs,
        )?;

        let mut remote_count = 0usize;

        for r in 0..self.num_ranks {
            if r == my_rank {
                continue;
            }

            let start = self.displs[r];
            let end = start + self.counts[r];
            let slice = &self.recv_buf[start..end];
            deserialize_cuts_from_buffer_into(
                slice,
                self.n_state,
                &mut self.deserialize_headers_buf,
                &mut self.deserialize_coefficients_buf,
            )?;
            for (i, header) in self.deserialize_headers_buf.iter().enumerate() {
                let coeff_start = i * self.n_state;
                fcf.add_cut(
                    stage,
                    u64::from(header.iteration),
                    header.forward_pass_index,
                    header.intercept,
                    &self.deserialize_coefficients_buf[coeff_start..coeff_start + self.n_state],
                );
                remote_count += 1;
            }
        }

        Ok(remote_count)
    }

    /// Pack the current iteration's local cuts into the send buffer, returning
    /// the number of cut records packed.
    ///
    /// Reads coefficients directly from the pool's `coefficients` slice to
    /// avoid per-cut `Vec<f64>` clones. Only cuts generated at the given
    /// `iteration` and currently active are included.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the number of eligible cuts exceeds the send
    /// buffer capacity.
    #[allow(clippy::cast_possible_truncation)]
    pub fn pack_local_records(
        &mut self,
        fcf: &FutureCostFunction,
        stage: usize,
        iteration: u64,
    ) -> usize {
        let pool = &fcf.pools[stage];

        let mut n_cuts = 0usize;
        for slot in 0..pool.populated_count {
            if !pool.active[slot] {
                continue;
            }
            let meta = &pool.metadata[slot];
            if meta.iteration_generated != iteration {
                continue;
            }

            let required = (n_cuts + 1) * self.record_size;
            debug_assert!(
                required <= self.send_buf.len(),
                "pack_local_records: {required} bytes required, exceeds send_buf capacity {}",
                self.send_buf.len()
            );

            let start = n_cuts * self.record_size;
            let coeffs =
                &pool.coefficients[slot * pool.state_dimension..(slot + 1) * pool.state_dimension];
            serialize_cut(
                &mut self.send_buf[start..start + self.record_size],
                slot as u32,
                iteration as u32,
                meta.forward_pass_index,
                pool.intercepts[slot],
                coeffs,
            );
            n_cuts += 1;
        }

        n_cuts
    }

    /// Exchange records pre-packed via
    /// [`pack_local_records`](Self::pack_local_records), inserting only remote
    /// cuts into the FCF (the local rank's segment is skipped).
    ///
    /// # Arguments
    ///
    /// - `stage` — 0-based stage index for which cuts are being synchronized.
    /// - `n_local` — number of cuts packed into the send buffer.
    /// - `fcf` — Future Cost Function to receive remote cuts.
    /// - `comm` — communicator for the `allgatherv` call.
    ///
    /// # Returns
    ///
    /// `Ok(n)` where `n` is the number of remote cuts inserted into `fcf`.
    ///
    /// # Errors
    ///
    /// Returns `Err(SddpError::Validation(_))` if `n_local` does not equal
    /// `per_rank_cuts[my_rank]`, the expected cut count for this rank as
    /// established by the cut-distribution plan. Allowing a mismatch to proceed
    /// would corrupt remote ranks' deserialized cut buffers, so this invariant
    /// is enforced in both debug and release builds.
    ///
    /// Returns `Err(SddpError::Communication(_))` if the underlying
    /// `allgatherv` call fails.
    pub fn sync_packed_records<C: Communicator>(
        &mut self,
        stage: usize,
        n_local: usize,
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<usize, SddpError> {
        let send_len = n_local * self.record_size;

        debug_assert!(
            send_len <= self.send_buf.len(),
            "send_len {send_len} exceeds send_buf capacity {}",
            self.send_buf.len()
        );

        let my_rank = comm.rank();
        let expected_for_me = self.per_rank_cuts[my_rank];
        if n_local != expected_for_me {
            return Err(SddpError::Validation(format!(
                "sync_cuts invariant violated at stage {stage}: rank \
                 {my_rank} produced {n_local} cuts, expected \
                 {expected_for_me} per the cut-distribution plan. \
                 Releasing this divergence to allgatherv would corrupt \
                 remote ranks' deserialized cut buffers."
            )));
        }

        for r in 0..self.num_ranks {
            let cuts_for_r = if r == my_rank {
                n_local
            } else {
                self.per_rank_cuts[r]
            };
            self.counts[r] = cuts_for_r * self.record_size;
        }
        self.displs[0] = 0;
        for r in 1..self.num_ranks {
            self.displs[r] = self.displs[r - 1] + self.counts[r - 1];
        }

        let recv_len: usize = self.counts.iter().sum();
        debug_assert!(
            recv_len <= self.recv_buf.len(),
            "recv_len {recv_len} exceeds recv_buf capacity {}",
            self.recv_buf.len()
        );

        comm.allgatherv(
            &self.send_buf[..send_len],
            &mut self.recv_buf[..recv_len],
            &self.counts,
            &self.displs,
        )?;

        let mut remote_cut_count = 0usize;

        for r in 0..self.num_ranks {
            if r == my_rank {
                continue;
            }

            let start = self.displs[r];
            let end = start + self.counts[r];
            let slice = &self.recv_buf[start..end];

            deserialize_cuts_from_buffer_into(
                slice,
                self.n_state,
                &mut self.deserialize_headers_buf,
                &mut self.deserialize_coefficients_buf,
            )?;

            for (i, header) in self.deserialize_headers_buf.iter().enumerate() {
                let coeff_start = i * self.n_state;
                fcf.add_cut(
                    stage,
                    u64::from(header.iteration),
                    header.forward_pass_index,
                    header.intercept,
                    &self.deserialize_coefficients_buf[coeff_start..coeff_start + self.n_state],
                );
                remote_cut_count += 1;
            }
        }

        Ok(remote_cut_count)
    }

    /// Return the send buffer capacity in bytes.
    #[must_use]
    pub fn send_capacity(&self) -> usize {
        self.send_buf.len()
    }

    /// Return the receive buffer capacity in bytes.
    #[must_use]
    pub fn recv_capacity(&self) -> usize {
        self.recv_buf.len()
    }
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]

    use cobre_comm::{CommData, CommError, Communicator, LocalBackend, ReduceOp};

    use super::CutSyncBuffers;
    use crate::{
        SddpError,
        cut::{
            fcf::FutureCostFunction,
            wire::{cut_wire_size, deserialize_cuts_from_buffer, serialize_cut},
        },
    };

    // ── Unit tests ────────────────────────────────────────────────────────────

    #[test]
    fn new_deserialize_scratch_bufs_start_empty() {
        // AC3: deserialize_headers_buf and deserialize_coefficients_buf both
        // have capacity == 0 immediately after construction (grown lazily).
        let bufs = CutSyncBuffers::new(2, 3, 4);
        assert_eq!(
            bufs.deserialize_headers_buf.capacity(),
            0,
            "deserialize_headers_buf must start with capacity 0"
        );
        assert_eq!(
            bufs.deserialize_coefficients_buf.capacity(),
            0,
            "deserialize_coefficients_buf must start with capacity 0"
        );
    }

    #[test]
    fn new_send_buf_capacity_is_max_cuts_times_record_size() {
        let bufs = CutSyncBuffers::new(2, 3, 1);
        let expected = 3 * cut_wire_size(2);
        assert_eq!(bufs.send_capacity(), expected);
    }

    #[test]
    fn new_recv_buf_capacity_is_max_cuts_times_num_ranks_times_record_size() {
        // 10 * 4 * cut_wire_size(3) = 40 * 49 = 1960
        let bufs = CutSyncBuffers::new(3, 10, 4);
        let expected = 10 * 4 * cut_wire_size(3);
        assert_eq!(bufs.recv_capacity(), expected);
        assert_eq!(expected, 1960);
    }

    #[test]
    fn new_counts_length_equals_num_ranks() {
        let bufs = CutSyncBuffers::new(3, 10, 4);
        assert_eq!(bufs.counts.len(), 4);
    }

    #[test]
    fn new_displs_length_equals_num_ranks() {
        let bufs = CutSyncBuffers::new(3, 10, 4);
        assert_eq!(bufs.displs.len(), 4);
    }

    #[test]
    fn new_counts_and_displs_initialized_to_max_uniform_values() {
        // Construction sets max uniform capacity; sync_cuts recomputes per call.
        let bufs = CutSyncBuffers::new(2, 3, 2);
        let per_rank = 3 * cut_wire_size(2); // 123
        assert_eq!(bufs.counts[0], per_rank);
        assert_eq!(bufs.counts[1], per_rank);
        assert_eq!(bufs.displs[0], 0);
        assert_eq!(bufs.displs[1], per_rank);
    }

    #[test]
    fn new_n_state_zero_record_size_is_25() {
        // Edge case: n_state = 0, record_size = 25.
        let bufs = CutSyncBuffers::new(0, 5, 1);
        assert_eq!(bufs.send_capacity(), 5 * 25);
        assert_eq!(bufs.recv_capacity(), 5 * 25);
    }

    #[test]
    fn send_buf_serialization_round_trip_two_cuts() {
        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (1, 1, 1, 20.0, &[3.0, 4.0])];

        let record_size = cut_wire_size(2);
        let send_len = local_cuts.len() * record_size;
        assert_eq!(send_len, 82);

        // Serialize manually into send_buf using the same logic as sync_cuts.
        for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
            local_cuts.iter().enumerate()
        {
            let start = i * record_size;
            serialize_cut(
                &mut bufs.send_buf[start..start + record_size],
                slot_index,
                iteration,
                forward_pass_index,
                intercept,
                coefficients,
            );
        }

        let recovered = deserialize_cuts_from_buffer(&bufs.send_buf[..send_len], 2).unwrap();
        assert_eq!(recovered.len(), 2);

        let (h0, c0) = &recovered[0];
        assert_eq!(h0.slot_index, 0);
        assert_eq!(h0.iteration, 1);
        assert_eq!(h0.forward_pass_index, 0);
        assert_eq!(h0.intercept, 10.0);
        assert_eq!(c0, &[1.0, 2.0]);

        let (h1, c1) = &recovered[1];
        assert_eq!(h1.slot_index, 1);
        assert_eq!(h1.iteration, 1);
        assert_eq!(h1.forward_pass_index, 1);
        assert_eq!(h1.intercept, 20.0);
        assert_eq!(c1, &[3.0, 4.0]);
    }

    #[test]
    fn counts_and_displs_computation_for_various_cut_counts() {
        // 2 local cuts, n_state=2: per_rank_bytes = 2 * 41 = 82; 3 ranks →
        // counts = [82, 82, 82], displs = [0, 82, 164].
        let mut bufs = CutSyncBuffers::new(2, 5, 3);

        let n_local = 2usize;
        let record_size = cut_wire_size(2); // 41
        let per_rank = n_local * record_size; // 82

        // Simulate what sync_cuts does to counts and displs.
        for r in 0..3 {
            bufs.counts[r] = per_rank;
            bufs.displs[r] = r * per_rank;
        }

        assert_eq!(bufs.counts, vec![82, 82, 82]);
        assert_eq!(bufs.displs, vec![0, 82, 164]);
    }

    // ── Integration tests (round-trip with LocalBackend) ──────────────────────

    #[test]
    fn sync_cuts_single_rank_returns_zero_remote_cuts() {
        // AC: Given CutSyncBuffers::new(n_state=2, max_cuts_per_rank=2,
        // num_ranks=1), when sync_cuts is called with 2 local cuts in
        // single-rank mode, then it returns Ok(0) — the single rank's own
        // cuts are skipped. (max_cuts_per_rank must equal the actual cut count
        // so per_rank_cuts[0] == n_local.)
        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 2, 10, &[0; 2]);
        let comm = LocalBackend;

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (0, 1, 1, 20.0, &[3.0, 4.0])];

        let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();
        assert_eq!(result, 0, "expected zero remote cuts in single-rank mode");
    }

    #[test]
    fn sync_cuts_single_rank_does_not_insert_local_cuts_into_fcf() {
        let mut bufs = CutSyncBuffers::new(2, 2, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 2, 10, &[0; 2]);
        let comm = LocalBackend;

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (0, 1, 1, 20.0, &[3.0, 4.0])];

        bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();

        // FCF must remain empty — local cuts are intentionally NOT inserted.
        assert_eq!(
            fcf.total_active_cuts(),
            0,
            "sync_cuts must not insert local cuts into FCF"
        );
    }

    #[test]
    fn sync_cuts_serialization_round_trip_via_allgatherv_identity() {
        // max_cuts_per_rank=1 matches the 1 cut actually sent (per_rank_cuts[0]=1).
        let mut bufs = CutSyncBuffers::new(2, 1, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);
        let comm = LocalBackend;

        // The backward pass inserts this rank's own cut before sync_cuts runs.
        fcf.add_cut(0, 1, 0, 10.0, &[1.0, 2.0]);

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[(0, 1, 0, 10.0, &[1.0, 2.0])];

        let remote_inserted = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();

        assert_eq!(remote_inserted, 0);
        assert_eq!(fcf.total_active_cuts(), 1);
    }

    #[test]
    fn sync_cuts_zero_local_cuts_returns_zero() {
        // total_forward_passes=0 so per_rank_cuts[0]=0 matches n_local=0.
        let mut bufs = CutSyncBuffers::with_distribution(2, 5, 1, 0);
        let mut fcf = FutureCostFunction::new(2, 2, 5, 10, &[0; 2]);
        let comm = LocalBackend;

        let result = bufs.sync_cuts(0, &[], &mut fcf, &comm).unwrap();
        assert_eq!(result, 0);
        assert_eq!(fcf.total_active_cuts(), 0);
    }

    #[test]
    fn sync_cuts_error_maps_to_sddp_communication_error() {
        struct FailingComm;

        impl Communicator for FailingComm {
            fn allgatherv<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                Err(CommError::CollectiveFailed {
                    operation: "allgatherv",
                    mpi_error_code: 42,
                    message: "simulated failure".to_string(),
                })
            }

            fn allreduce<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn barrier(&self) -> Result<(), CommError> {
                unreachable!()
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                1
            }

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let mut bufs = CutSyncBuffers::new(2, 1, 1);
        let mut fcf = FutureCostFunction::new(2, 2, 1, 10, &[0; 2]);

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[(0, 1, 0, 5.0, &[1.0, 2.0])];

        let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &FailingComm);
        assert!(
            matches!(result, Err(SddpError::Communication(_))),
            "expected SddpError::Communication, got: {result:?}",
        );
    }

    #[test]
    fn sync_cuts_three_ranks_returns_four_remote_cuts() {
        // Pre-populate recv_buf with remote data; the mock allgatherv copies
        // only the local (rank-0) segment, leaving remote segments untouched.
        // Tests the deserialization path without unsafe pointer operations.

        /// 3-rank mock; rank 0 is local. `allgatherv` copies only the rank-0
        /// segment, relying on remote segments pre-populated in `recv_buf`.
        struct ThreeRankComm;

        impl Communicator for ThreeRankComm {
            fn allgatherv<T: CommData>(
                &self,
                send: &[T],
                recv: &mut [T],
                counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                let r0_len = counts[0];
                recv[..r0_len].copy_from_slice(&send[..r0_len]);
                Ok(())
            }

            fn allreduce<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn barrier(&self) -> Result<(), CommError> {
                unreachable!()
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                3
            }

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let n_state = 2;
        let record_size = cut_wire_size(n_state); // 41
        let n_local = 2;
        let per_rank_bytes = n_local * record_size; // 82

        // FCF: 1 stage, n_state=2, forward_passes=6, max_iterations=10,
        // warm_start=0 → capacity = 0 + 10*6 = 60 slots.
        let mut fcf = FutureCostFunction::new(1, n_state, 6, 10, &[0; 1]);
        let mut bufs = CutSyncBuffers::new(n_state, n_local, 3);

        // Pre-populate recv_buf with remote rank data at the exact offsets
        // that sync_cuts will compute (displs[1] = 82, displs[2] = 164).
        let r1_start = per_rank_bytes; // 82
        serialize_cut(
            &mut bufs.recv_buf[r1_start..r1_start + record_size],
            10,
            1,
            10,
            100.0,
            &[1.0, 2.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[r1_start + record_size..r1_start + 2 * record_size],
            11,
            1,
            11,
            200.0,
            &[3.0, 4.0],
        );

        let r2_start = 2 * per_rank_bytes; // 164
        serialize_cut(
            &mut bufs.recv_buf[r2_start..r2_start + record_size],
            20,
            1,
            20,
            300.0,
            &[5.0, 6.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[r2_start + record_size..r2_start + 2 * record_size],
            21,
            1,
            21,
            400.0,
            &[7.0, 8.0],
        );

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 50.0, &[0.1, 0.2]), (1, 1, 1, 60.0, &[0.3, 0.4])];

        let remote_inserted = bufs
            .sync_cuts(0, local_cuts, &mut fcf, &ThreeRankComm)
            .unwrap();
        assert_eq!(remote_inserted, 4, "expected 4 remote cuts inserted");
        assert_eq!(fcf.total_active_cuts(), 4);
    }

    #[test]
    fn sync_cuts_preserves_cut_fields_after_deserialization() {
        // Single-rank: no remote insertions to observe, so inspect recv_buf
        // directly after the allgatherv identity copy.
        let n_state = 2usize;
        let mut bufs = CutSyncBuffers::new(n_state, 1, 1);
        let mut fcf = FutureCostFunction::new(1, n_state, 1, 10, &[0; 1]);
        let comm = LocalBackend;

        let coeffs = [7.5_f64, -3.25_f64];
        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[(5, 3, 2, 99.0, &coeffs)];

        bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();

        // After LocalBackend allgatherv (identity copy), recv_buf[0..record_size]
        // contains rank 0's serialized cut. Deserialize and verify.
        let record_size = cut_wire_size(n_state);
        let recovered =
            deserialize_cuts_from_buffer(&bufs.recv_buf[..record_size], n_state).unwrap();
        assert_eq!(recovered.len(), 1);

        let (header, rec_coeffs) = &recovered[0];
        assert_eq!(header.slot_index, 5);
        assert_eq!(header.iteration, 3);
        assert_eq!(header.forward_pass_index, 2);
        assert_eq!(header.intercept, 99.0);
        assert_eq!(rec_coeffs[0].to_bits(), coeffs[0].to_bits());
        assert_eq!(rec_coeffs[1].to_bits(), coeffs[1].to_bits());
    }

    // ── Invariant check tests ─────────────────────────────────────────────────

    #[test]
    fn sync_cuts_invariant_passes_when_local_matches_expected() {
        // per_rank_cuts == [3] (single rank), n_local == 3 → Ok(0).
        let mut bufs = CutSyncBuffers::new(2, 3, 1);
        let mut fcf = FutureCostFunction::new(1, 2, 3, 10, &[0; 1]);
        let comm = LocalBackend;

        let local_cuts: &[(u32, u32, u32, f64, &[f64])] = &[
            (0, 1, 0, 10.0, &[1.0, 2.0]),
            (1, 1, 1, 20.0, &[3.0, 4.0]),
            (2, 1, 2, 30.0, &[5.0, 6.0]),
        ];

        let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &comm).unwrap();
        assert_eq!(result, 0, "single rank: no remote cuts expected");
    }

    #[test]
    fn sync_cuts_invariant_rejects_local_mismatch() {
        // per_rank_cuts == [3, 3] but rank 0 supplies only 2 cuts.
        // allgatherv is unreachable — the invariant check returns before it.
        struct TwoRankStubComm;

        impl Communicator for TwoRankStubComm {
            fn allgatherv<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                unreachable!("allgatherv must not be reached when invariant fails")
            }

            fn allreduce<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn barrier(&self) -> Result<(), CommError> {
                unreachable!()
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                2
            }

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let mut bufs = CutSyncBuffers::with_distribution(2, 3, 2, 6);
        let mut fcf = FutureCostFunction::new(1, 2, 6, 10, &[0; 1]);

        // Only 2 cuts; expected 3 per per_rank_cuts[0].
        let local_cuts: &[(u32, u32, u32, f64, &[f64])] =
            &[(0, 1, 0, 10.0, &[1.0, 2.0]), (1, 1, 1, 20.0, &[3.0, 4.0])];

        let result = bufs.sync_cuts(0, local_cuts, &mut fcf, &TwoRankStubComm);
        match result {
            Err(SddpError::Validation(ref msg)) => {
                assert!(
                    msg.contains("sync_cuts invariant violated"),
                    "message missing 'sync_cuts invariant violated': {msg}"
                );
                assert!(
                    msg.contains("rank 0 produced 2 cuts, expected 3"),
                    "message missing 'rank 0 produced 2 cuts, expected 3': {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got: {other:?}"),
        }
    }

    #[test]
    fn sync_packed_cuts_invariant_rejects_local_mismatch() {
        // sync_packed_records counterpart of the sync_cuts mismatch test:
        // n_local=2 with per_rank_cuts[0]=3 rejects before any allgatherv.
        struct TwoRankStubComm;

        impl Communicator for TwoRankStubComm {
            fn allgatherv<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _counts: &[usize],
                _displs: &[usize],
            ) -> Result<(), CommError> {
                unreachable!("allgatherv must not be reached when invariant fails")
            }

            fn allreduce<T: CommData>(
                &self,
                _send: &[T],
                _recv: &mut [T],
                _op: ReduceOp,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn broadcast<T: CommData>(
                &self,
                _buf: &mut [T],
                _root: usize,
            ) -> Result<(), CommError> {
                unreachable!()
            }

            fn barrier(&self) -> Result<(), CommError> {
                unreachable!()
            }

            fn rank(&self) -> usize {
                0
            }

            fn size(&self) -> usize {
                2
            }

            fn abort(&self, error_code: i32) -> ! {
                std::process::exit(error_code)
            }
        }

        let mut bufs = CutSyncBuffers::with_distribution(2, 3, 2, 6);
        let mut fcf = FutureCostFunction::new(1, 2, 6, 10, &[0; 1]);

        let result = bufs.sync_packed_records(0, 2, &mut fcf, &TwoRankStubComm);
        match result {
            Err(SddpError::Validation(ref msg)) => {
                assert!(
                    msg.contains("sync_cuts invariant violated"),
                    "message missing 'sync_cuts invariant violated': {msg}"
                );
                assert!(
                    msg.contains("rank 0 produced 2 cuts, expected 3"),
                    "message missing 'rank 0 produced 2 cuts, expected 3': {msg}"
                );
            }
            other => panic!("expected SddpError::Validation, got: {other:?}"),
        }
    }
}
