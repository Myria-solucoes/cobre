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

    /// Maximum cut-coefficient count across all pools (the global
    /// `StateSpace::n_state`); sizes buffer capacity only.
    ///
    /// The wire stride for a pool's exchange is
    /// `cut_wire_size(fcf.pools[pool].state_dimension)` — the pool's own
    /// dimension — never this value. A reduced pool carries fewer
    /// coefficients than this max, so packing at `max_record_size` while reading
    /// per-pool coefficient slices would mis-stride every record and corrupt
    /// remote ranks' deserialized cuts.
    max_n_state: usize,

    /// Total number of MPI ranks.
    num_ranks: usize,

    /// Maximum wire record size `cut_wire_size(max_n_state)`; the buffer-capacity
    /// upper bound, not the per-stage stride (see [`max_n_state`](Self::max_n_state)).
    max_record_size: usize,

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
        let max_record_size = cut_wire_size(n_state);
        let send_cap = max_cuts_per_rank * max_record_size;

        let base = total_forward_passes / num_ranks;
        let remainder = total_forward_passes % num_ranks;
        let per_rank_cuts: Vec<usize> = (0..num_ranks)
            .map(|r| base + usize::from(r < remainder))
            .collect();
        let recv_cap: usize = per_rank_cuts.iter().sum::<usize>() * max_record_size;

        let counts: Vec<usize> = per_rank_cuts.iter().map(|&c| c * max_record_size).collect();
        let mut displs = vec![0usize; num_ranks];
        for r in 1..num_ranks {
            displs[r] = displs[r - 1] + counts[r - 1];
        }

        Self {
            send_buf: vec![0u8; send_cap],
            recv_buf: vec![0u8; recv_cap],
            counts,
            displs,
            max_n_state: n_state,
            num_ranks,
            max_record_size,
            per_rank_cuts,
            deserialize_headers_buf: Vec::new(),
            deserialize_coefficients_buf: Vec::new(),
        }
    }

    /// Synchronize locally generated cuts across all MPI ranks for one pool,
    /// inserting only remote cuts into the FCF (local cuts already inserted by
    /// the backward pass are skipped).
    ///
    /// # Arguments
    ///
    /// - `pool` — 0-based pool id for which cuts are being synchronized.
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
    /// - Panics if any cut's coefficient slice length does not equal the pool's
    ///   own dimension `fcf.pools[pool].state_dimension`.
    pub fn sync_cuts<C: Communicator>(
        &mut self,
        pool: usize,
        local_cuts: &[(u32, u32, u32, f64, &[f64])],
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<usize, SddpError> {
        let n_local = local_cuts.len();
        let my_rank = comm.rank();
        let expected_for_me = self.per_rank_cuts[my_rank];
        if n_local != expected_for_me {
            return Err(SddpError::Validation(format!(
                "sync_cuts invariant violated at pool {pool}: rank \
                 {my_rank} produced {n_local} cuts, expected \
                 {expected_for_me} per the cut-distribution plan. \
                 Releasing this divergence to allgatherv would corrupt \
                 remote ranks' deserialized cut buffers."
            )));
        }

        let pool_n_state = fcf.pools[pool].state_dimension;
        debug_assert!(
            pool_n_state <= self.max_n_state,
            "pool {pool} dimension {pool_n_state} exceeds buffer-capacity max {}",
            self.max_n_state
        );
        let record_size = cut_wire_size(pool_n_state);

        let send_len = n_local * record_size;

        debug_assert!(
            send_len <= self.send_buf.len(),
            "send_len {send_len} exceeds send_buf capacity {}",
            self.send_buf.len()
        );

        for (i, &(slot_index, iteration, forward_pass_index, intercept, coefficients)) in
            local_cuts.iter().enumerate()
        {
            debug_assert!(
                coefficients.len() == pool_n_state,
                "cut {i} coefficient length {} != pool dimension {pool_n_state}",
                coefficients.len(),
            );
            let start = i * record_size;
            serialize_cut(
                &mut self.send_buf[start..start + record_size],
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
            self.counts[r] = cuts_for_r * record_size;
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
                pool_n_state,
                &mut self.deserialize_headers_buf,
                &mut self.deserialize_coefficients_buf,
            )?;
            for (i, header) in self.deserialize_headers_buf.iter().enumerate() {
                let coeff_start = i * pool_n_state;
                fcf.add_cut(
                    pool,
                    u64::from(header.iteration),
                    header.forward_pass_index,
                    header.intercept,
                    &self.deserialize_coefficients_buf[coeff_start..coeff_start + pool_n_state],
                );
                remote_count += 1;
            }
        }

        Ok(remote_count)
    }

    /// Pack the current iteration's local cuts into the send buffer, returning
    /// the number of cut records packed.
    ///
    /// Reads coefficients via [`CutPool::coefficient_row`] to avoid per-cut
    /// `Vec<f64>` clones. Only cuts generated at the given `iteration` and
    /// currently active are included.
    ///
    /// [`CutPool::coefficient_row`]: crate::cut::CutPool::coefficient_row
    ///
    /// # Panics
    ///
    /// Panics in debug builds if the number of eligible cuts exceeds the send
    /// buffer capacity.
    #[allow(clippy::cast_possible_truncation)]
    pub fn pack_local_records(
        &mut self,
        fcf: &FutureCostFunction,
        pool: usize,
        iteration: u64,
    ) -> usize {
        let cut_pool = &fcf.pools[pool];

        // Wire stride is the pool's own dimension, never the cached
        // `max_record_size` (see the `max_n_state` field doc): packing at the global
        // max while the slice below reads `cut_pool.state_dimension` coefficients
        // mis-strides every record once a pool is reduced.
        let record_size = cut_wire_size(cut_pool.state_dimension);
        debug_assert!(
            record_size <= self.max_record_size,
            "record size {record_size} exceeds buffer-capacity max {}",
            self.max_record_size
        );

        let mut n_cuts = 0usize;
        for slot in 0..cut_pool.populated() {
            if !cut_pool.is_active(slot) {
                continue;
            }
            let meta = cut_pool.metadata(slot);
            if meta.iteration_generated != iteration {
                continue;
            }

            let required = (n_cuts + 1) * record_size;
            debug_assert!(
                required <= self.send_buf.len(),
                "pack_local_records: {required} bytes required, exceeds send_buf capacity {}",
                self.send_buf.len()
            );

            let start = n_cuts * record_size;
            let coeffs = cut_pool.coefficient_row(slot);
            serialize_cut(
                &mut self.send_buf[start..start + record_size],
                slot as u32,
                iteration as u32,
                meta.forward_pass_index,
                cut_pool.intercept(slot),
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
    /// - `pool` — 0-based pool id for which cuts are being synchronized.
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
        pool: usize,
        n_local: usize,
        fcf: &mut FutureCostFunction,
        comm: &C,
    ) -> Result<usize, SddpError> {
        let my_rank = comm.rank();
        let expected_for_me = self.per_rank_cuts[my_rank];
        if n_local != expected_for_me {
            return Err(SddpError::Validation(format!(
                "sync_cuts invariant violated at pool {pool}: rank \
                 {my_rank} produced {n_local} cuts, expected \
                 {expected_for_me} per the cut-distribution plan. \
                 Releasing this divergence to allgatherv would corrupt \
                 remote ranks' deserialized cut buffers."
            )));
        }

        // Per-pool wire stride, matching what `pack_local_records` packed at (see
        // the `max_n_state` field doc).
        let pool_n_state = fcf.pools[pool].state_dimension;
        debug_assert!(
            pool_n_state <= self.max_n_state,
            "pool {pool} dimension {pool_n_state} exceeds buffer-capacity max {}",
            self.max_n_state
        );
        let record_size = cut_wire_size(pool_n_state);

        let send_len = n_local * record_size;

        debug_assert!(
            send_len <= self.send_buf.len(),
            "send_len {send_len} exceeds send_buf capacity {}",
            self.send_buf.len()
        );

        for r in 0..self.num_ranks {
            let cuts_for_r = if r == my_rank {
                n_local
            } else {
                self.per_rank_cuts[r]
            };
            self.counts[r] = cuts_for_r * record_size;
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
                pool_n_state,
                &mut self.deserialize_headers_buf,
                &mut self.deserialize_coefficients_buf,
            )?;

            for (i, header) in self.deserialize_headers_buf.iter().enumerate() {
                let coeff_start = i * pool_n_state;
                fcf.add_cut(
                    pool,
                    u64::from(header.iteration),
                    header.forward_pass_index,
                    header.intercept,
                    &self.deserialize_coefficients_buf[coeff_start..coeff_start + pool_n_state],
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

    /// Two pools of differing dimension exchanged via the production
    /// `pack_local_records` + `sync_packed_records` path: pool 0 carries
    /// `pools[0].state_dimension == 2`, pool 1 carries
    /// `pools[1].state_dimension == 5`. Each pool's wire stride is its own
    /// dimension; the cached global max (5) sizes buffers only. Asserts each
    /// pool's remote cuts deserialize with the correct per-pool coefficient
    /// count and exact bit-for-bit intercepts/coefficients.
    #[test]
    // RATIONALE: one test drives two pools of differing dimension through the
    // shared recv_buf so the per-pool-stride contract is exercised end-to-end;
    // splitting per pool would not share the buffer and so would not test it.
    #[allow(clippy::too_many_lines)]
    fn sync_packed_records_two_pools_differing_dimension_per_pool_stride() {
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

        // Global max dimension = 5 (pool[1]); pool[0] is reduced to 2. Buffer
        // capacity is sized by the max via CutSyncBuffers::new(5, ...); the
        // per-pool stride is each pool's own dimension.
        let global_n_state = 5;
        let dims = [2usize, 5usize];
        let n_local = 2;
        let mut fcf =
            FutureCostFunction::new_per_pool(&dims, global_n_state, 6, 10, &[0; 2], &[6; 2]);
        let mut bufs = CutSyncBuffers::new(global_n_state, n_local, 3);

        // Drive each pool through the production path. Each iteration: insert
        // this rank's own cuts (the backward pass does this before sync), pack
        // them, pre-populate the remote segments at the per-pool stride, then
        // sync and assert per-pool recovery.
        let remote_iteration = 1u64;

        // Pool 0 (dimension 2).
        let s0_record_size = cut_wire_size(dims[0]); // 41
        fcf.add_cut(0, remote_iteration, 0, 50.0, &[0.1, 0.2]);
        fcf.add_cut(0, remote_iteration, 1, 60.0, &[0.3, 0.4]);
        let s0_packed = bufs.pack_local_records(&fcf, 0, remote_iteration);
        assert_eq!(s0_packed, 2, "pool 0 should pack both local cuts");

        // Remote ranks at the per-pool stride (per_rank_bytes = 2 * 41 = 82).
        let s0_per_rank_bytes = n_local * s0_record_size;
        let s0_r1 = s0_per_rank_bytes;
        serialize_cut(
            &mut bufs.recv_buf[s0_r1..s0_r1 + s0_record_size],
            10,
            remote_iteration as u32,
            10,
            100.0,
            &[1.0, 2.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[s0_r1 + s0_record_size..s0_r1 + 2 * s0_record_size],
            11,
            remote_iteration as u32,
            11,
            200.0,
            &[3.0, 4.0],
        );
        let s0_r2 = 2 * s0_per_rank_bytes;
        serialize_cut(
            &mut bufs.recv_buf[s0_r2..s0_r2 + s0_record_size],
            20,
            remote_iteration as u32,
            20,
            300.0,
            &[5.0, 6.0],
        );
        serialize_cut(
            &mut bufs.recv_buf[s0_r2 + s0_record_size..s0_r2 + 2 * s0_record_size],
            21,
            remote_iteration as u32,
            21,
            400.0,
            &[7.0, 8.0],
        );

        let s0_remote = bufs
            .sync_packed_records(0, s0_packed, &mut fcf, &ThreeRankComm)
            .unwrap();
        assert_eq!(s0_remote, 4, "pool 0: 4 remote cuts inserted");

        // Verify pool 0's recv_buf recovery NOW, before pool 1 reuses the
        // shared buffer at its (larger) stride. Remote rank-1 cut deserializes at
        // the per-pool dimension (2) with exact intercept/coefficients.
        let s0_recovered_r1 =
            deserialize_cuts_from_buffer(&bufs.recv_buf[s0_r1..s0_r1 + s0_record_size], dims[0])
                .unwrap();
        assert_eq!(s0_recovered_r1.len(), 1);
        assert_eq!(s0_recovered_r1[0].1.len(), 2, "pool 0 coeff count is 2");
        assert_eq!(
            s0_recovered_r1[0].0.intercept.to_bits(),
            100.0_f64.to_bits()
        );
        assert_eq!(s0_recovered_r1[0].1[0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(s0_recovered_r1[0].1[1].to_bits(), 2.0_f64.to_bits());

        // Pool 1 (dimension 5).
        let s1_record_size = cut_wire_size(dims[1]); // 65
        fcf.add_cut(1, remote_iteration, 0, 70.0, &[1.1, 1.2, 1.3, 1.4, 1.5]);
        fcf.add_cut(1, remote_iteration, 1, 80.0, &[2.1, 2.2, 2.3, 2.4, 2.5]);
        let s1_packed = bufs.pack_local_records(&fcf, 1, remote_iteration);
        assert_eq!(s1_packed, 2, "pool 1 should pack both local cuts");

        let s1_per_rank_bytes = n_local * s1_record_size;
        let s1_r1_coeffs0 = [10.0, 11.0, 12.0, 13.0, 14.0];
        let s1_r1_coeffs1 = [20.0, 21.0, 22.0, 23.0, 24.0];
        let s1_r1 = s1_per_rank_bytes;
        serialize_cut(
            &mut bufs.recv_buf[s1_r1..s1_r1 + s1_record_size],
            30,
            remote_iteration as u32,
            30,
            500.0,
            &s1_r1_coeffs0,
        );
        serialize_cut(
            &mut bufs.recv_buf[s1_r1 + s1_record_size..s1_r1 + 2 * s1_record_size],
            31,
            remote_iteration as u32,
            31,
            600.0,
            &s1_r1_coeffs1,
        );
        let s1_r2_coeffs0 = [30.0, 31.0, 32.0, 33.0, 34.0];
        let s1_r2_coeffs1 = [40.0, 41.0, 42.0, 43.0, 44.0];
        let s1_r2 = 2 * s1_per_rank_bytes;
        serialize_cut(
            &mut bufs.recv_buf[s1_r2..s1_r2 + s1_record_size],
            40,
            remote_iteration as u32,
            40,
            700.0,
            &s1_r2_coeffs0,
        );
        serialize_cut(
            &mut bufs.recv_buf[s1_r2 + s1_record_size..s1_r2 + 2 * s1_record_size],
            41,
            remote_iteration as u32,
            41,
            800.0,
            &s1_r2_coeffs1,
        );

        let s1_remote = bufs
            .sync_packed_records(1, s1_packed, &mut fcf, &ThreeRankComm)
            .unwrap();
        assert_eq!(s1_remote, 4, "pool 1: 4 remote cuts inserted");

        // Pool 1's recv_buf recovery at its own per-pool dimension (5): the
        // remote rank-2 cut deserializes with 5 coefficients, bit-for-bit.
        let s1_recovered_r2 =
            deserialize_cuts_from_buffer(&bufs.recv_buf[s1_r2..s1_r2 + s1_record_size], dims[1])
                .unwrap();
        assert_eq!(s1_recovered_r2.len(), 1);
        assert_eq!(s1_recovered_r2[0].1.len(), 5, "pool 1 coeff count is 5");
        assert_eq!(
            s1_recovered_r2[0].0.intercept.to_bits(),
            700.0_f64.to_bits()
        );
        for (got, want) in s1_recovered_r2[0].1.iter().zip(&s1_r2_coeffs0) {
            assert_eq!(got.to_bits(), want.to_bits());
        }

        // Each pool's cuts carry its own coefficient count.
        assert_eq!(fcf.pools[0].state_dimension, 2);
        assert_eq!(fcf.pools[1].state_dimension, 5);
        // 2 local (skipped on insert) + 4 remote per pool = 6 active cuts each.
        assert_eq!(fcf.pools[0].active_count(), 6);
        assert_eq!(fcf.pools[1].active_count(), 6);

        // The inserted FCF cuts evaluate to the per-pool coefficient·state dot
        // product — proving the remote coefficients landed in the right pool with
        // the right length. The max over pool-1 cuts at unit state is the
        // rank-2 cut with the largest intercept: 800 + sum(s1_r2_coeffs1).
        let s1_state = [1.0, 1.0, 1.0, 1.0, 1.0];
        let expected_max = 800.0 + s1_r2_coeffs1.iter().sum::<f64>();
        assert_eq!(fcf.evaluate_at_state(1, &s1_state), expected_max);
    }

    /// All-enabled regression: every pool's `state_dimension` equals the global
    /// `n_state`, so the per-pool stride equals the cached max and the serialized
    /// bytes are bit-identical to the pre-change path. Packs through the
    /// production path and compares the send buffer against an independent
    /// `serialize_cut` reference.
    #[test]
    fn pack_local_records_all_enabled_bytes_match_global_stride() {
        let n_state = 4;
        let record_size = cut_wire_size(n_state);
        let mut fcf = FutureCostFunction::new(2, n_state, 2, 10, &[0; 2]);
        let mut bufs = CutSyncBuffers::new(n_state, 2, 1);

        let c0 = [1.0, 2.0, 3.0, 4.0];
        let c1 = [5.0, 6.0, 7.0, 8.0];
        fcf.add_cut(0, 1, 0, 10.0, &c0);
        fcf.add_cut(0, 1, 1, 20.0, &c1);

        let packed = bufs.pack_local_records(&fcf, 0, 1);
        assert_eq!(packed, 2);

        // Independent reference at the global stride. pack_local_records writes
        // each cut's deterministic pool slot as slot_index: with forward_passes=2
        // and iteration=1, slots are 1*2+0=2 and 1*2+1=3.
        let mut reference = vec![0u8; 2 * record_size];
        serialize_cut(&mut reference[0..record_size], 2, 1, 0, 10.0, &c0);
        serialize_cut(
            &mut reference[record_size..2 * record_size],
            3,
            1,
            1,
            20.0,
            &c1,
        );

        assert_eq!(
            &bufs.send_buf[..2 * record_size],
            &reference[..],
            "all-enabled stage packs bit-identically to the global-stride path"
        );
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
