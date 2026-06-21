use crate::indexer::{BlockGrid, StateLayout};

/// Pre-allocated row-bound and column-bound patch arrays for one SDDP stage LP solve.
///
/// The buffer is reused across all iterations.  It carries two regions:
///
/// - **Row-bound region** (`indices` / `lower` / `upper`): sized for
///   `N + M*B + N` patches, where `N` is the number of hydro plants,
///   `M` is the number of stochastic load buses, and `B` is the maximum
///   block count across stages.  The row buffer holds only Categories 3,
///   4, and 5 — noise, load, and z-inflow patches.
///
/// - **Column-bound region** (`col_indices` / `col_lower` / `col_upper`):
///   sized for `N*(1+L) + A*K` entries, where `L` is the maximum PAR order
///   and `A*K` is the anticipated-thermal state count.  This region carries
///   the state-fixing slots for Categories 1 (storage), 2 (lag), and
///   6 (anticipated-state).  It is populated by `fill_col_state_patches`.
///
/// # Memory layout — row-bound region
///
/// | Entry range          | Category                          | LP row indices                |
/// | -------------------- | --------------------------------- | ----------------------------- |
/// | `[0, N)`             | AR dynamics / noise (Category 3)  | `base_rows[s]`                |
/// | `[N, N + M*B_act)`   | Load balance patches (Category 4) | per-stage                     |
/// | `[N + M*B, 2*N + M*B)` | Z-inflow definition (Category 5)| `z_inflow_row_start + h`     |
///
/// State fixing (Categories 1, 2, 6) is applied exclusively via column bounds
/// and lives in the column-bound region.
///
/// [`fill_load_patches`](PatchBuffer::fill_load_patches) writes Category 4
/// and records `active_load_patches` for the current stage's block count.
/// When `n_load_buses == 0`, Category 4 is empty and `forward_patch_count`
/// returns `N` unchanged.
///
/// Generic-constraint rows are not in this list; their coefficients,
/// including those resolved from [`ResolvedParameters`](crate::resolved_parameters::ResolvedParameters),
/// are immutable after stage-template construction.
#[derive(Debug, Clone)]
pub struct PatchBuffer {
    /// Row indices to patch.
    ///
    /// Length `N + M*max_blocks + N`.  Entries are `usize` to match
    /// the `set_row_bounds(&[usize], ...)` interface directly.
    pub indices: Vec<usize>,

    /// New lower bounds for each patched row.
    ///
    /// Length `N + M*max_blocks + N`.  For equality constraints,
    /// `lower[i] == upper[i]`.
    pub lower: Vec<f64>,

    /// New upper bounds for each patched row.
    ///
    /// Length `N + M*max_blocks + N`.  For equality constraints,
    /// `upper[i] == lower[i]`.
    pub upper: Vec<f64>,

    /// Column indices to patch in the column-bound region.
    ///
    /// Length `N*(1+L) + A*K` — one entry for each state-fixing slot covering
    /// Categories 1 (storage, N entries), 2 (lag, N*L entries), and
    /// 6 (anticipated-state, A*K entries).  Populated by `fill_col_state_patches`;
    /// zero-initialised at construction.
    pub col_indices: Vec<usize>,

    /// New lower bounds for each patched column in the column-bound region.
    ///
    /// Length `N*(1+L) + A*K`.  Populated together with `col_indices` and
    /// `col_upper` by `fill_col_state_patches`.
    pub col_lower: Vec<f64>,

    /// New upper bounds for each patched column in the column-bound region.
    ///
    /// Length `N*(1+L) + A*K`.  For tight bound constraints,
    /// `col_upper[i] == col_lower[i]`.
    pub col_upper: Vec<f64>,

    /// Number of operating hydro plants (N).
    hydro_count: usize,

    /// Maximum PAR order across all operating hydros (L).
    max_par_order: usize,

    /// Number of buses with stochastic load noise (M).
    load_bus_count: usize,

    /// Maximum block count across all stages.
    ///
    /// Determines the Category 4 capacity: `load_bus_count * max_blocks`.
    max_blocks: usize,

    /// Number of anticipated thermals (A).
    n_anticipated: usize,

    /// Maximum lead-time horizon across anticipated thermals (K).
    k_max: usize,

    /// Number of load patches written by the most recent [`fill_load_patches`] call.
    ///
    /// Equals `load_bus_count * n_blocks` for the stage solved most recently.
    /// Zero when `fill_load_patches` has not yet been called or when
    /// `load_bus_count == 0`.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    active_load_patches: usize,

    /// Number of z-inflow patches written by the most recent [`fill_z_inflow_patches`] call.
    ///
    /// Equals `hydro_count` when z-inflow patches are active, zero otherwise.
    ///
    /// [`fill_z_inflow_patches`]: PatchBuffer::fill_z_inflow_patches
    active_z_inflow_patches: usize,
}

impl PatchBuffer {
    /// Construct a [`PatchBuffer`] pre-allocated for `N + M*B + N` row patches.
    ///
    /// - `hydro_count` — number of operating hydro plants (N).
    /// - `max_par_order` — maximum PAR order across all operating hydros (L).
    ///   Not used in the row-buffer capacity; still used in the col-buffer
    ///   capacity (`N*(1+L) + A*K`).
    /// - `n_load_buses` — number of buses with stochastic load noise (M).
    ///   Pass `0` when there is no stochastic load.
    /// - `max_blocks` — maximum block count across all stages (B).
    ///   Pass `0` when there is no stochastic load.
    /// - `n_anticipated` — number of anticipated thermals (A).
    ///   Pass `0` when there are no anticipated thermals.
    /// - `k_max` — maximum lead-time horizon across anticipated thermals (K).
    ///   Pass `0` when there are no anticipated thermals.
    ///
    /// The row-bound region (`indices`, `lower`, `upper`) is sized to
    /// `N + M*B + N` and zero-initialised.  The column-bound region
    /// (`col_indices`, `col_lower`, `col_upper`) is sized to `N*(1+L) + A*K`
    /// and zero-initialised; it is populated by `fill_col_state_patches`.
    /// Call [`fill_forward_patches`] and [`fill_load_patches`] to populate the
    /// row-bound region before each LP solve.
    ///
    /// # Examples
    ///
    /// ```
    /// use cobre_sddp::lp_builder::PatchBuffer;
    ///
    /// // 3-hydro AR(2) system, no stochastic load, no anticipated thermals
    /// // Row capacity = N + M*B + N = 3 + 0 + 3 = 6
    /// // Col capacity = N*(1+L) + A*K = 3*(1+2) + 0 = 9
    /// let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
    /// assert_eq!(buf.indices.len(), 6);
    /// assert_eq!(buf.col_indices.len(), 9);
    ///
    /// // 3-hydro AR(2) system with 2 stochastic load buses, up to 3 blocks
    /// // Row capacity = N + M*B + N = 3 + 6 + 3 = 12
    /// let buf_load = PatchBuffer::new(3, 2, 2, 3, 0, 0);
    /// assert_eq!(buf_load.indices.len(), 12);
    ///
    /// // Production scale: N = 160, L = 12, no stochastic load
    /// // Row capacity = N + N = 160 + 160 = 320
    /// let big = PatchBuffer::new(160, 12, 0, 0, 0, 0);
    /// assert_eq!(big.indices.len(), 320);
    ///
    /// // Edge case: no lags (L = 0)
    /// // Row capacity = N + N = 5 + 5 = 10
    /// let no_lag = PatchBuffer::new(5, 0, 0, 0, 0, 0);
    /// assert_eq!(no_lag.indices.len(), 10);
    ///
    /// // Anticipated thermals: 1 plant, K=2 — row capacity unchanged (A*K is col-only)
    /// // Row capacity = N + N = 3 + 3 = 6
    /// let ant = PatchBuffer::new(3, 2, 0, 0, 1, 2);
    /// assert_eq!(ant.indices.len(), 6);
    /// ```
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    #[must_use]
    // Rationale: each argument sizes an independent buffer region — noise/z-inflow rows
    // (hydro_count), lag-state column slots (max_par_order), load patches (n_load_buses,
    // max_blocks), and anticipated-state column slots (n_anticipated, k_max); the capacity
    // formula for each region is different, so there is no shared sub-struct to collapse them.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hydro_count: usize,
        max_par_order: usize,
        n_load_buses: usize,
        max_blocks: usize,
        n_anticipated: usize,
        k_max: usize,
    ) -> Self {
        // Row buffer carries only noise (N) + load (M*B) + z_inflow (N) patches.
        // State fixing is applied via column bounds and lives in the col buffer.
        let capacity = hydro_count + n_load_buses * max_blocks + hydro_count;
        // Column-bound region covers state-fixing slots for Categories 1, 2, and 6:
        // N*(1+L) + A*K entries.
        let col_capacity = hydro_count * (1 + max_par_order) + n_anticipated * k_max;
        Self {
            indices: vec![0; capacity],
            lower: vec![0.0; capacity],
            upper: vec![0.0; capacity],
            col_indices: vec![0; col_capacity],
            col_lower: vec![0.0; col_capacity],
            col_upper: vec![0.0; col_capacity],
            hydro_count,
            max_par_order,
            load_bus_count: n_load_buses,
            max_blocks,
            n_anticipated,
            k_max,
            active_load_patches: 0,
            active_z_inflow_patches: 0,
        }
    }

    /// Fill `N` noise patches (Category 3) for a forward-pass solve.
    ///
    /// Writes `N` noise-fixing patches at the start of the row buffer:
    /// row `base_row + h` ← `noise[h]` for `h ∈ [0, N)`.
    ///
    /// Category 3 is NOT prescaled by `row_scale` because `noise[h]` is computed
    /// from `template.row_lower` (already row-scaled) plus an unscaled noise term.
    /// Prescaling would double-scale the base component.
    ///
    /// All patches are equality constraints: `lower[i] == upper[i] == noise[h]`.
    ///
    /// State-fixing (Categories 1, 2, 6) is applied separately via
    /// `fill_col_state_patches` and `set_col_bounds`.
    ///
    /// After this call, pass `&buf.indices[..pc]`, `&buf.lower[..pc]`,
    /// `&buf.upper[..pc]` where `pc = forward_patch_count()` to
    /// `SolverInterface::set_row_bounds`.
    ///
    /// # Arguments
    ///
    /// - `layout` — stage-invariant role-(a) state layout (provides `n_state`
    ///   and `hydro_count` for the length checks).
    /// - `state` — incoming state vector of length `n_state = N*(1+L) + A*K`.
    ///   Only used for the `debug_assert` length check; not read during noise write.
    /// - `noise` — stochastic noise innovations of length `N`, one per hydro.
    /// - `base_row` — first row index of the AR dynamics constraints in the
    ///   static non-dual region of the LP ([Solver Abstraction SS2.2]).
    ///   Computed during stage template construction.
    /// - `row_scale` — per-row scaling factors from the stage template.
    ///   Accepted for API compatibility; not applied to Category 3 values.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != layout.n_state` or
    /// `noise.len() != layout.hydro_count`.
    pub fn fill_forward_patches(
        &mut self,
        layout: &StateLayout,
        state: &[f64],
        noise: &[f64],
        base_row: usize,
        _row_scale: &[f64],
    ) {
        debug_assert_eq!(
            state.len(),
            layout.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = layout.n_state,
        );
        debug_assert!(
            noise.len() == layout.hydro_count || noise.is_empty(),
            "noise slice length {got} must equal hydro_count {expected} or be empty",
            got = noise.len(),
            expected = layout.hydro_count,
        );

        // Category 3: AR dynamics rows in the static non-dual region.
        // The noise value is computed by the caller as:
        //   noise[h] = template.row_lower[base_row + h] + noise_scale[h] * eta
        // where `template.row_lower` is already scaled (by `apply_row_scale`).
        // The `noise_scale` factor IS pre-scaled by the row scaling factor
        // during LP setup (see `setup.rs`: noise_scale[h] *= row_scale[base_row + h]),
        // so `noise[h]` is already in the correct scaled units and must be
        // written as-is without additional prescaling here.
        for (h, &nv) in noise.iter().enumerate() {
            // AR dynamics row = base_row + h (hydro-major). This is in the static
            // non-dual region, NOT the inflow-lag column `N + ℓ·N + h` of Category 2,
            // despite the shared `+ h` shape.
            self.indices[h] = base_row + h;
            self.lower[h] = nv;
            self.upper[h] = nv;
        }
    }

    /// Fill `N*(1+L) + A*K` column-bound patches for a state-fixing solve.
    ///
    /// Populates the column-bound region (`col_indices`, `col_lower`, `col_upper`)
    /// with the column-bound counterparts of Categories 1, 2, and 6:
    ///
    /// | Entry range              | Category                              | Column targets                            |
    /// | ------------------------ | ------------------------------------- | ----------------------------------------- |
    /// | `[0, N)`                 | Storage-fixing (Category 1)           | `storage_in.start + h` for `h ∈ [0, N)`  |
    /// | `[N, N*(1+L))`           | AR lag-fixing (Category 2)            | `inflow_lags.start + lag*N + h`           |
    /// | `[N*(1+L), N*(1+L)+A*K)` | Anticipated-state-fixing (Category 6) | `anticipated_state.start + slot*A + plant`|
    ///
    /// Column-bound state fixing enforces `x == v` by setting `lb = ub = v / col_scale[col]`
    /// in the scaled LP (contrast with the row-equality path, which multiplies by `row_scale[row]`).
    ///
    /// All patches write equality bounds: `col_lower[i] == col_upper[i]` for every entry.
    ///
    /// After this call, pass
    /// `&buf.col_indices[..state_col_patch_count()]`,
    /// `&buf.col_lower[..state_col_patch_count()]`,
    /// `&buf.col_upper[..state_col_patch_count()]`
    /// to `SolverInterface::set_col_bounds`.
    ///
    /// # Arguments
    ///
    /// - `state_layout` — stage-invariant state-vector layout (role (a)),
    ///   the source of the `storage_in`, `inflow_lags`, and `anticipated_state`
    ///   column starts. A single global stage-0 layout resolves the correct
    ///   column at every stage because these three starts are pure functions of
    ///   `N`, `L`, `A`, `k_max` (independent of `n_blks`).
    /// - `state` — incoming state vector of length `n_state = N*(1+L) + A*K`.
    ///   Prefix `[0, N)` is incoming storage, `[N, N*(1+L))` is AR lags,
    ///   `[N*(1+L), N*(1+L)+A*K)` is the anticipated-state ring-buffer.
    /// - `col_scale` — per-column scaling factors from the stage template.
    ///   Pass `&[]` when no column scaling is active. When non-empty, must
    ///   be at least `state_layout.anticipated_state.end` entries long so every
    ///   Category 1, 2, and 6 column can be indexed.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `state.len() != state_layout.n_state`.
    pub fn fill_col_state_patches(
        &mut self,
        state_layout: &StateLayout,
        state: &[f64],
        col_scale: &[f64],
    ) {
        debug_assert_eq!(
            state.len(),
            state_layout.n_state,
            "state slice length {got} != n_state {expected}",
            got = state.len(),
            expected = state_layout.n_state,
        );

        let n = self.hydro_count;
        let l = self.max_par_order;

        // Category 1: storage-fixing (col = storage_in.start + h)
        // The incoming storage column is distinct from the outgoing storage column
        // at [0, N). Using the wrong column would pin the water-balance output
        // variable rather than the state-carrying variable.
        let storage_in_start = state_layout.storage_in.start;
        for (h, &sv) in state[..n].iter().enumerate() {
            let col = storage_in_start + h;
            let scaled = if col_scale.is_empty() {
                sv
            } else {
                sv / col_scale[col]
            };
            self.col_indices[h] = col;
            self.col_lower[h] = scaled;
            self.col_upper[h] = scaled;
        }

        // Category 2: AR lag-fixing (col = inflow_lags.start + lag*N + h)
        let inflow_lags_start = state_layout.inflow_lags.start;
        for lag in 0..l {
            for h in 0..n {
                let slot = n + lag * n + h;
                let col = inflow_lags_start + lag * n + h;
                let sv = state[slot];
                let scaled = if col_scale.is_empty() {
                    sv
                } else {
                    sv / col_scale[col]
                };
                self.col_indices[slot] = col;
                self.col_lower[slot] = scaled;
                self.col_upper[slot] = scaled;
            }
        }

        // Category 6: anticipated-state-fixing (col = anticipated_state.start + slot*A + plant)
        let cat6_start = n * (1 + l);
        self.fill_anticipated_state_col_patches(state_layout, state, col_scale, cat6_start);
    }

    /// Write the `A*K` Category 6 anticipated-state-fixing column-bound patches
    /// into the column-bound buffer starting at `cat6_start`.
    ///
    /// Each patch targets the column `anticipated_state.start + slot * A + plant`
    /// and sets `lb = ub = state[col] / col_scale[col]` (or `state[col]` when
    /// `col_scale` is empty). Iteration order is slot-major, plant-minor —
    /// matching the LP ring-buffer layout and the row-equality counterpart
    /// `fill_anticipated_state_patches`.
    ///
    /// When `n_anticipated == 0` or `k_max == 0` this is a no-op.
    fn fill_anticipated_state_col_patches(
        &mut self,
        state_layout: &StateLayout,
        state: &[f64],
        col_scale: &[f64],
        cat6_start: usize,
    ) {
        let n_ant = self.n_anticipated;
        let k = self.k_max;
        let ant_state_col_start = state_layout.anticipated_state.start;
        for slot in 0..k {
            for plant in 0..n_ant {
                let off = slot * n_ant + plant;
                let buf_slot = cat6_start + off;
                let col = ant_state_col_start + off;
                let sv = state[col];
                let scaled = if col_scale.is_empty() {
                    sv
                } else {
                    sv / col_scale[col]
                };
                self.col_indices[buf_slot] = col;
                self.col_lower[buf_slot] = scaled;
                self.col_upper[buf_slot] = scaled;
            }
        }
    }

    /// Fill Category 4 load balance row patches for a forward-pass solve.
    ///
    /// Writes `n_load_buses * n_blocks` equality patches into the Category 4
    /// region starting at offset `N` (immediately after Category 3 noise patches).
    /// Each patch targets the exact load balance row for bus `bus_positions[i]`
    /// and block `blk`:
    ///
    /// ```text
    /// row = load_row_start + bus_positions[i] * n_blocks + blk
    /// ```
    ///
    /// where `n_blocks` is the per-stage block count carried by `grid`. The row
    /// address is computed through [`BlockGrid::flat`] (bus-outer / block-inner),
    /// the single owner of block-major address strides.
    ///
    /// The `load_rhs` slice is laid out as `[bus0_blk0, bus0_blk1, …, bus1_blk0, …]`
    /// (bus-major, block-minor), matching `bus_positions` order.
    ///
    /// When `row_scale` is non-empty, each patch value is prescaled by
    /// `row_scale[row]` before being stored.  Pass `&[]` when no row scaling
    /// has been applied.
    ///
    /// After this call, [`forward_patch_count`] returns
    /// `N + n_load_buses * n_blocks` so that the correct slice is
    /// passed to `set_row_bounds`.
    ///
    /// # Arguments
    ///
    /// - `load_row_start` — first row index of the load-balance block in the LP.
    /// - `grid` — the per-stage [`BlockGrid`]; it must carry this stage's block
    ///   count (the value the LP template was built with), NOT a global grid.
    ///   A global grid would stride by the wrong block count at any stage whose
    ///   count differs.
    /// - `load_rhs` — patched RHS values; length must equal
    ///   `self.load_bus_count * grid.n_blks()`.
    /// - `bus_positions` — LP bus position for each stochastic load bus;
    ///   length must equal `self.load_bus_count`.
    /// - `row_scale` — per-row scaling factors from the stage template.
    ///   Pass `&[]` when no scaling is active.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if:
    /// - `load_rhs.len() != self.load_bus_count * grid.n_blks()`
    /// - `bus_positions.len() != self.load_bus_count`
    /// - `grid.n_blks() > self.max_blocks`
    ///
    /// [`forward_patch_count`]: PatchBuffer::forward_patch_count
    pub fn fill_load_patches(
        &mut self,
        load_row_start: usize,
        grid: BlockGrid,
        load_rhs: &[f64],
        bus_positions: &[usize],
        row_scale: &[f64],
    ) {
        // `n_blocks` here is a buffer-size count (no `+ blk` address term), so it
        // reads the scalar via `grid.n_blks()` rather than routing through `flat`;
        // the LP row address below routes through `grid.flat` instead.
        let n_blocks = grid.n_blks();
        debug_assert_eq!(
            load_rhs.len(),
            self.load_bus_count * n_blocks,
            "load_rhs length {got} != load_bus_count*n_blocks {expected}",
            got = load_rhs.len(),
            expected = self.load_bus_count * n_blocks,
        );
        debug_assert_eq!(
            bus_positions.len(),
            self.load_bus_count,
            "bus_positions length {got} != load_bus_count {expected}",
            got = bus_positions.len(),
            expected = self.load_bus_count,
        );
        debug_assert!(
            n_blocks <= self.max_blocks,
            "n_blocks {n_blocks} exceeds max_blocks {mb}",
            mb = self.max_blocks,
        );

        // Category 4 follows Category 3 (noise, N entries).
        let cat4_start = self.hydro_count;
        let mut slot = cat4_start;

        for (i, &bus_pos) in bus_positions.iter().enumerate() {
            for blk in 0..n_blocks {
                let row = grid.flat(load_row_start, bus_pos, blk);
                // The host-array index shares the flat layout `i * n_blks + blk`,
                // so it routes through `grid.flat(0, i, blk)` (start = 0) — the
                // same primitive, keeping a single owner for the strided arithmetic.
                let rhs = load_rhs[grid.flat(0, i, blk)];
                let scaled = if row_scale.is_empty() {
                    rhs
                } else {
                    rhs * row_scale[row]
                };
                self.indices[slot] = row;
                self.lower[slot] = scaled;
                self.upper[slot] = scaled;
                slot += 1;
            }
        }

        self.active_load_patches = self.load_bus_count * n_blocks;
    }

    /// Fill Category 5 patches: z-inflow definition row RHS.
    ///
    /// Updates N rows starting at `z_inflow_row_start` with the realized-inflow
    /// RHS values from `z_inflow_rhs`. Each row is an equality constraint:
    /// `lower[i] = upper[i] = z_inflow_rhs[h]`.
    ///
    /// This method must be called after `fill_forward_patches` (which fills
    /// categories 1-3) and [`fill_load_patches`] (category 4), before
    /// `solver.set_row_bounds`.
    ///
    /// When `row_scale` is non-empty, each patch value is prescaled by
    /// `row_scale[row]`.  Pass `&[]` when no row scaling is active.
    ///
    /// # Arguments
    ///
    /// - `z_inflow_row_start` - first row index of the z-inflow definition rows.
    /// - `z_inflow_rhs` - per-hydro RHS values (length >= `hydro_count`).
    /// - `row_scale` - per-row scaling factors. Pass `&[]` when no scaling.
    ///
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    pub fn fill_z_inflow_patches(
        &mut self,
        z_inflow_row_start: usize,
        z_inflow_rhs: &[f64],
        row_scale: &[f64],
    ) {
        let n = self.hydro_count;
        if n == 0 || z_inflow_rhs.is_empty() {
            self.active_z_inflow_patches = 0;
            return;
        }

        // Category 5 follows Categories 3 (N) and 4 (active load patches).
        let cat5_start = self.hydro_count + self.active_load_patches;

        for (h, &rhs) in z_inflow_rhs.iter().enumerate().take(n) {
            let slot = cat5_start + h;
            let row = z_inflow_row_start + h;
            let scaled = if row_scale.is_empty() {
                rhs
            } else {
                rhs * row_scale[row]
            };
            self.indices[slot] = row;
            self.lower[slot] = scaled;
            self.upper[slot] = scaled;
        }

        self.active_z_inflow_patches = n;
    }

    /// Number of active patches after [`fill_forward_patches`], (optionally)
    /// [`fill_load_patches`], and (optionally) [`fill_z_inflow_patches`]:
    /// `N + active_load_patches + active_z_inflow_patches`.
    ///
    /// Use this to pass the full forward-pass buffer to `set_row_bounds`.
    ///
    /// [`fill_forward_patches`]: PatchBuffer::fill_forward_patches
    /// [`fill_load_patches`]: PatchBuffer::fill_load_patches
    /// [`fill_z_inflow_patches`]: PatchBuffer::fill_z_inflow_patches
    #[must_use]
    #[inline]
    pub fn forward_patch_count(&self) -> usize {
        self.hydro_count + self.active_load_patches + self.active_z_inflow_patches
    }

    /// Capacity of the column-bound region: `N*(1+L) + A*K`.
    ///
    /// Returns the number of entries allocated in `col_indices`, `col_lower`,
    /// and `col_upper` — one for each state-fixing slot covering Categories 1
    /// (storage, N entries), 2 (lag, N*L entries), and 6 (anticipated-state,
    /// A*K entries).
    #[must_use]
    #[inline]
    pub fn state_col_patch_count(&self) -> usize {
        self.hydro_count * (1 + self.max_par_order) + self.n_anticipated * self.k_max
    }
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
mod tests {
    use super::PatchBuffer;
    use crate::indexer::test_fixtures::{state_layout, state_layout_for, state_layout_full};
    use crate::indexer::{BlockGrid, StageIndexer, StateLayout};

    /// Convenience: make a role-(a) state layout without repeating N/L everywhere.
    fn idx(n: usize, l: usize) -> StateLayout {
        state_layout(n, l)
    }

    // -------------------------------------------------------------------------
    // Capacity formulas (row + column buffers) across scales
    // -------------------------------------------------------------------------

    /// Row capacity is `N + n_load_buses*max_blocks + N` and column capacity is
    /// `N*(1+L) + A*K`. Both formulas are exercised at zero / unit-anticipated /
    /// combined / production scales in one table so each scale stays legible via
    /// the tuple-naming failure message.
    #[test]
    fn patch_buffer_capacity_formulas() {
        // (n, l, n_load_buses, max_blocks, a, k, expected_row_cap, expected_col_cap)
        let cases = [
            (
                0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
            ),
            (3, 2, 0, 0, 0, 0, 6, 9),
            (0, 0, 0, 0, 1, 2, 0, 2),
            (3, 2, 0, 0, 2, 3, 6, 15),
            (160, 12, 0, 0, 0, 0, 320, 2080),
        ];

        for (n, l, n_load_buses, max_blocks, a, k, expected_row_cap, expected_col_cap) in cases {
            let buf = PatchBuffer::new(n, l, n_load_buses, max_blocks, a, k);

            for (label, len) in [
                ("col_indices", buf.col_indices.len()),
                ("col_lower", buf.col_lower.len()),
                ("col_upper", buf.col_upper.len()),
            ] {
                assert_eq!(
                    len, expected_col_cap,
                    "{label} col cap mismatch for (n={n}, l={l}, a={a}, k={k})"
                );
            }

            for (label, len) in [
                ("indices", buf.indices.len()),
                ("lower", buf.lower.len()),
                ("upper", buf.upper.len()),
            ] {
                assert_eq!(
                    len, expected_row_cap,
                    "{label} row cap mismatch for (n={n}, l={l}, n_load_buses={n_load_buses}, max_blocks={max_blocks}, a={a}, k={k})"
                );
            }
        }
    }

    /// `state_col_patch_count` returns N*(1+L) + A*K.
    #[test]
    fn state_col_patch_count_returns_n_times_one_plus_l() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 1, 2);
        // N*(1+L) + A*K = 3*3 + 1*2 = 11
        assert_eq!(buf.state_col_patch_count(), 11);
    }

    /// Column buffer is zero-initialised at construction.
    #[test]
    fn col_buffer_zero_initialised() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        assert_eq!(buf.col_indices.len(), 9);
        assert!(
            buf.col_indices.iter().all(|&v| v == 0),
            "col_indices not zero-initialised"
        );
        assert!(
            buf.col_lower.iter().all(|&v| v == 0.0),
            "col_lower not zero-initialised"
        );
        assert!(
            buf.col_upper.iter().all(|&v| v == 0.0),
            "col_upper not zero-initialised"
        );
    }

    /// forward_patch_count without fill_z_inflow_patches returns N.
    #[test]
    fn forward_patch_count_without_z_inflow_fill() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        // forward_patch_count = N + 0 + 0 = 3
        assert_eq!(buf.forward_patch_count(), 3);
    }

    /// Category 3 (noise) indices start at slot 0.
    ///
    /// `fill_forward_patches` writes only Category 3 (noise) at `[0, N)`.
    #[test]
    fn fill_forward_patches_writes_only_noise() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        // Category 3 at slots 0..3: base_row + h = 50 + h
        assert_eq!(buf.indices[0], 50);
        assert_eq!(buf.indices[1], 51);
        assert_eq!(buf.indices[2], 52);
        assert_eq!(buf.lower[0], 0.1);
        assert_eq!(buf.upper[0], 0.1);
        assert_eq!(buf.lower[1], 0.2);
        assert_eq!(buf.upper[1], 0.2);
        assert_eq!(buf.lower[2], 0.3);
        assert_eq!(buf.upper[2], 0.3);
    }

    #[test]
    fn fill_forward_patches_all_equality_constraints() {
        // Every patch must satisfy lower == upper (equality constraint)
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        for i in 0..buf.forward_patch_count() {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    /// After fill_forward_patches, forward_patch_count == N (no load, no z_inflow).
    #[test]
    fn forward_patches_zero_lags_only_noise() {
        let n = 2;
        let mut buf = PatchBuffer::new(n, 0, 0, 0, 0, 0);
        let state = [5.0, 7.0];
        let noise = [0.5, 0.6];
        buf.fill_forward_patches(&idx(n, 0), &state, &noise, 10, &[]);

        // forward_patch_count = N = 2 (noise only; no load, no z-inflow)
        assert_eq!(buf.forward_patch_count(), 2);

        // Category 3 at slots 0, 1
        assert_eq!(buf.indices[0], 10); // base_row + 0 = 10
        assert_eq!(buf.lower[0], 0.5);
        assert_eq!(buf.indices[1], 11); // base_row + 1 = 11
        assert_eq!(buf.lower[1], 0.6);
    }

    #[test]
    fn production_scale_forward_patch_count() {
        // Without fill_z_inflow_patches, forward_patch_count = N = 160.
        // Row buffer capacity = N + N = 320.
        let buf = PatchBuffer::new(160, 12, 0, 0, 0, 0);
        assert_eq!(buf.forward_patch_count(), 160);
        assert_eq!(buf.indices.len(), 320);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // fixture: values are small integers, no precision lost
    fn production_scale_fill_forward_patches_smoke() {
        let n = 160;
        let l = 12;
        let mut buf = PatchBuffer::new(n, l, 0, 0, 0, 0);
        let n_state = n * (1 + l);
        let state: Vec<f64> = (0..n_state).map(|i| i as f64).collect();
        let noise: Vec<f64> = (0..n).map(|h| h as f64 * 0.01).collect();
        buf.fill_forward_patches(&idx(n, l), &state, &noise, 500, &[]);

        // Category 3 starts at slot 0 (no Cat 1/2/6 in row buffer).
        assert_eq!(buf.indices[0], 500); // base_row + 0 = 500
        assert_eq!(buf.lower[0], 0.0); // noise[0]
        assert_eq!(buf.indices[159], 659); // base_row + 159 = 659
        assert_eq!(buf.lower[159], 159.0 * 0.01);

        // All patches must be equality constraints
        for i in 0..buf.forward_patch_count() {
            assert_eq!(buf.lower[i], buf.upper[i], "patch {i} not equality");
        }
    }

    #[test]
    fn clone_and_debug() {
        let buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let cloned = buf.clone();
        assert_eq!(cloned.indices.len(), buf.indices.len());

        let s = format!("{buf:?}");
        assert!(s.contains("PatchBuffer"));
    }

    // -------------------------------------------------------------------------
    // Category 4 (load balance) unit tests
    // -------------------------------------------------------------------------

    /// AC (capacity): `PatchBuffer::new(2, 1, 1, 3, 0, 0)` → row capacity = N + M*B + N = 2 + 3 + 2 = 7.
    #[test]
    fn new_with_load_allocates_correct_capacity() {
        let buf = PatchBuffer::new(2, 1, 1, 3, 0, 0);
        // N + M*B + N = 2 + 1*3 + 2 = 7
        assert_eq!(buf.indices.len(), 7);
        assert_eq!(buf.lower.len(), 7);
        assert_eq!(buf.upper.len(), 7);
    }

    /// Category 4 row indices follow `row = load_row_start + bus_positions[i] * n_blocks + blk`.
    ///
    /// With `n_load_buses=2, n_blocks=2, bus_positions=[0,1], load_row_start=100`, N=0:
    /// Cat 4 starts at slot N=0 so indices[0..4] = [100, 101, 102, 103].
    #[test]
    fn fill_load_patches_correct_indices() {
        // N=0, L=0, M=2, B=2, A=0, K=0 → row capacity = 0 + 2*2 + 0 = 4
        let mut buf = PatchBuffer::new(0, 0, 2, 2, 0, 0);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(100, BlockGrid::new(2, 1), &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.indices[0], 100); // bus 0, blk 0
        assert_eq!(buf.indices[1], 101); // bus 0, blk 1
        assert_eq!(buf.indices[2], 102); // bus 1, blk 0
        assert_eq!(buf.indices[3], 103); // bus 1, blk 1
    }

    /// Category 4 lower and upper bounds equal the corresponding `load_rhs` value.
    #[test]
    fn fill_load_patches_correct_values() {
        let mut buf = PatchBuffer::new(0, 0, 2, 2, 0, 0);
        let load_rhs = [300.0_f64, 280.0, 500.0, 450.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(100, BlockGrid::new(2, 1), &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.lower[0], 300.0);
        assert_eq!(buf.upper[0], 300.0);
        assert_eq!(buf.lower[1], 280.0);
        assert_eq!(buf.upper[1], 280.0);
        assert_eq!(buf.lower[2], 500.0);
        assert_eq!(buf.upper[2], 500.0);
        assert_eq!(buf.lower[3], 450.0);
        assert_eq!(buf.upper[3], 450.0);
    }

    /// Every load patch must be an equality constraint: `lower[i] == upper[i]`.
    #[test]
    fn fill_load_patches_equality_constraints() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, BlockGrid::new(3, 1), &load_rhs, &bus_positions, &[]);

        let count = buf.forward_patch_count();
        for i in 0..count {
            assert_eq!(
                buf.lower[i],
                buf.upper[i],
                "patch {i}: lower {lo} != upper {up}",
                lo = buf.lower[i],
                up = buf.upper[i],
            );
        }
    }

    /// `forward_patch_count` includes Category 4 after `fill_load_patches`.
    ///
    /// N=3, M=2, n_blocks=3 → forward_patch_count = N + M*n_blocks = 3 + 6 = 9.
    #[test]
    fn forward_patch_count_includes_load() {
        let mut buf = PatchBuffer::new(3, 2, 2, 3, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        let load_rhs = [100.0_f64, 90.0, 80.0, 200.0, 190.0, 180.0];
        let bus_positions = [0_usize, 1];
        buf.fill_load_patches(20, BlockGrid::new(3, 1), &load_rhs, &bus_positions, &[]);

        assert_eq!(buf.forward_patch_count(), 9); // N=3 + M*n_blocks=6
    }

    /// When `n_load_buses == 0`, `forward_patch_count` equals `N`.
    #[test]
    fn zero_load_buses_no_category4() {
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(3, 2), &state, &noise, 50, &[]);

        // No fill_load_patches call: forward_patch_count = N = 3
        assert_eq!(buf.forward_patch_count(), 3);
    }

    // -------------------------------------------------------------------------
    // Category 6 col-patch unit tests (col-side path, row-side deleted)
    // -------------------------------------------------------------------------

    /// fill_forward_patches with N=0, A=1, K=2 writes zero noise patches
    /// (N=0 hydros → no Category 3 entries in row buffer).
    #[test]
    fn fill_forward_patches_zero_hydros_zero_noise_patches() {
        // N=0, A=1, K=2 anticipated-only state layout.
        let state_layout = state_layout_full(0, 0, 1, 2, vec![2]);

        let mut state = vec![0.0_f64; state_layout.n_state];
        state[state_layout.anticipated_state.start] = 7.0;
        state[state_layout.anticipated_state.start + 1] = 11.0;

        // N=0, A=1, K=2 → row capacity = 0 + 0 + 0 = 0
        let mut buf = PatchBuffer::new(0, 0, 0, 0, 1, 2);

        // forward_patch_count = N = 0 (state goes in col buffer, not row buffer)
        assert_eq!(
            buf.forward_patch_count(),
            0,
            "forward_patch_count before fill"
        );

        buf.fill_forward_patches(&state_layout, &state, &[], 0, &[]);

        assert_eq!(
            buf.forward_patch_count(),
            0,
            "forward_patch_count after fill"
        );
    }

    /// fill_forward_patches with N=3, A=0, K=0 still writes exactly N noise patches at [0, N).
    #[test]
    fn fill_forward_patches_no_anticipated_noise_at_slot_zero() {
        let n = 3;
        let l = 2;
        let mut buf = PatchBuffer::new(n, l, 0, 0, 0, 0);
        let state = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let noise = [0.1, 0.2, 0.3];
        buf.fill_forward_patches(&idx(n, l), &state, &noise, 50, &[]);

        // forward_patch_count = N = 3 (no load, no z-inflow)
        assert_eq!(buf.forward_patch_count(), 3);

        // Category 3 at slots 0..N (no Cat 1/2/6 in row buffer).
        assert_eq!(buf.indices[0], 50); // base_row + 0 = 50
        assert_eq!(buf.lower[0], 0.1);
        assert_eq!(buf.indices[1], 51);
        assert_eq!(buf.lower[1], 0.2);
        assert_eq!(buf.indices[2], 52);
        assert_eq!(buf.lower[2], 0.3);
    }

    // -------------------------------------------------------------------------
    // fill_col_state_patches unit tests
    // -------------------------------------------------------------------------

    /// Build an augmented indexer for N=3, L=2, A=0, K=0 to use across
    /// Category 1 and 2 column-patch tests.
    fn idx_augmented_3_2() -> StageIndexer {
        use crate::indexer::{EquipmentCounts, EvapConfig, FphaColumnLayout};
        StageIndexer::with_equipment_and_evaporation(
            &EquipmentCounts {
                hydro_count: 3,
                max_par_order: 2,
                n_thermals: 0,
                n_lines: 0,
                n_buses: 1,
                n_blks: 1,
                has_inflow_penalty: false,
                max_deficit_segments: 1,
                n_anticipated: 0,
                k_max: 0,
                anticipated_lead_stages: vec![],
                anticipated_thermal_indices: vec![],
                n_pumping: 0,
            },
            &FphaColumnLayout {
                hydro_indices: vec![],
                planes_per_hydro: vec![],
            },
            &EvapConfig {
                hydro_indices: vec![],
            },
        )
    }

    /// Category 1 col_indices[0..3] = [storage_in.start, +1, +2].
    #[test]
    fn fill_col_state_patches_category1_indices() {
        let indexer = idx_augmented_3_2();
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state_layout = state_layout_for(&indexer);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        let s = state_layout.storage_in.start;
        assert_eq!(buf.col_indices[0], s);
        assert_eq!(buf.col_indices[1], s + 1);
        assert_eq!(buf.col_indices[2], s + 2);
    }

    /// Category 1 col_lower[0..3] == col_upper[0..3] == [10.0, 20.0, 30.0].
    #[test]
    fn fill_col_state_patches_category1_values() {
        let indexer = idx_augmented_3_2();
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state_layout = state_layout_for(&indexer);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        assert_eq!(buf.col_lower[0], 10.0);
        assert_eq!(buf.col_upper[0], 10.0);
        assert_eq!(buf.col_lower[1], 20.0);
        assert_eq!(buf.col_upper[1], 20.0);
        assert_eq!(buf.col_lower[2], 30.0);
        assert_eq!(buf.col_upper[2], 30.0);
    }

    /// Category 2 col_indices[3..9] matches lag column targets; col_lower matches lag values.
    ///
    /// Lag-column formula: `inflow_lags.start + lag*N + h`.
    /// - lag=0: cols `[il, il+1, il+2]`
    /// - lag=1: cols `[il+3, il+4, il+5]`
    ///
    /// State layout: lags at `state[3..9]` = [1,2,3,4,5,6].
    #[test]
    fn fill_col_state_patches_category2_indices_and_values() {
        let indexer = idx_augmented_3_2();
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state_layout = state_layout_for(&indexer);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        let il = state_layout.inflow_lags.start;
        // lag=0
        assert_eq!(buf.col_indices[3], il);
        assert_eq!(buf.col_indices[4], il + 1);
        assert_eq!(buf.col_indices[5], il + 2);
        // lag=1
        assert_eq!(buf.col_indices[6], il + 3);
        assert_eq!(buf.col_indices[7], il + 4);
        assert_eq!(buf.col_indices[8], il + 5);

        assert_eq!(buf.col_lower[3], 1.0);
        assert_eq!(buf.col_upper[3], 1.0);
        assert_eq!(buf.col_lower[6], 4.0);
        assert_eq!(buf.col_upper[6], 4.0);
        assert_eq!(buf.col_lower[8], 6.0);
        assert_eq!(buf.col_upper[8], 6.0);
    }

    /// Category 6 column-bound patches for N=0, L=0, A=1, K=2.
    ///
    /// col_indices[0..2] = [anticipated_state.start, +1],
    /// col_lower[0..2] == col_upper[0..2] == [7.0, 11.0] (slot-major / plant-minor).
    #[test]
    fn fill_col_state_patches_anticipated_category6() {
        // N=0, A=1, K=2 anticipated-only state layout.
        let state_layout = state_layout_full(0, 0, 1, 2, vec![2]);

        // n_state = 0 + 1*2 = 2; anticipated_state.start = 0
        let ant_start = state_layout.anticipated_state.start;
        let mut state = vec![0.0_f64; state_layout.n_state];
        state[ant_start] = 7.0;
        state[ant_start + 1] = 11.0;

        let mut buf = PatchBuffer::new(0, 0, 0, 0, 1, 2);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        assert_eq!(buf.col_indices[0], ant_start);
        assert_eq!(buf.col_indices[1], ant_start + 1);
        assert_eq!(buf.col_lower[0], 7.0);
        assert_eq!(buf.col_upper[0], 7.0);
        assert_eq!(buf.col_lower[1], 11.0);
        assert_eq!(buf.col_upper[1], 11.0);
    }

    /// Every patch in the active col region has col_lower[i] == col_upper[i].
    #[test]
    fn fill_col_state_patches_equality_constraints() {
        let indexer = idx_augmented_3_2();
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state_layout = state_layout_for(&indexer);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        let count = buf.state_col_patch_count();
        for i in 0..count {
            assert_eq!(
                buf.col_lower[i],
                buf.col_upper[i],
                "col patch {i}: lower {lo} != upper {up}",
                lo = buf.col_lower[i],
                up = buf.col_upper[i],
            );
        }
    }

    /// col_scale divides: col_lower[h] == state[h] / col_scale[col] for Category 1.
    ///
    /// With col_scale[storage_in.start + h] = 2.0 for all h, and state = [10, 20, 30, ...],
    /// expected col_lower[0..3] = [5.0, 10.0, 15.0].
    #[test]
    fn fill_col_state_patches_unscaled_with_col_scale() {
        let indexer = idx_augmented_3_2();
        let state_layout = state_layout_for(&indexer);
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);

        // Build a col_scale long enough to cover anticipated_state.end.
        // Fill with 1.0 everywhere, then override the storage_in columns to 2.0.
        let ncols = state_layout
            .anticipated_state
            .end
            .max(state_layout.storage_in.end);
        let mut col_scale = vec![1.0_f64; ncols];
        let s = state_layout.storage_in.start;
        col_scale[s] = 2.0;
        col_scale[s + 1] = 2.0;
        col_scale[s + 2] = 2.0;

        buf.fill_col_state_patches(&state_layout, &state, &col_scale);

        assert_eq!(buf.col_lower[0], 5.0);
        assert_eq!(buf.col_upper[0], 5.0);
        assert_eq!(buf.col_lower[1], 10.0);
        assert_eq!(buf.col_upper[1], 10.0);
        assert_eq!(buf.col_lower[2], 15.0);
        assert_eq!(buf.col_upper[2], 15.0);
    }

    /// When n_anticipated == 0, Category 6 is empty and state_col_patch_count() == N*(1+L).
    #[test]
    fn fill_col_state_patches_zero_anticipated_collapses_correctly() {
        let indexer = idx_augmented_3_2();
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state_layout = state_layout_for(&indexer);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        // N*(1+L) + A*K = 3*3 + 0 = 9
        assert_eq!(buf.state_col_patch_count(), 9);
        assert_eq!(buf.col_indices.len(), 9);
    }

    /// After fill_col_state_patches, the row buffer (indices/lower/upper) is untouched.
    ///
    /// Catches accidental cross-buffer writes; the row buffer must remain
    /// zero-initialised since no row-equality filler has been called.
    #[test]
    fn row_buffer_unchanged_after_fill_col_state_patches() {
        let indexer = idx_augmented_3_2();
        let state = [10.0_f64, 20.0, 30.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut buf = PatchBuffer::new(3, 2, 0, 0, 0, 0);
        let state_layout = state_layout_for(&indexer);
        buf.fill_col_state_patches(&state_layout, &state, &[]);

        assert!(
            buf.indices.iter().all(|&v| v == 0),
            "row indices modified by fill_col_state_patches"
        );
        assert!(
            buf.lower.iter().all(|&v| v == 0.0),
            "row lower modified by fill_col_state_patches"
        );
        assert!(
            buf.upper.iter().all(|&v| v == 0.0),
            "row upper modified by fill_col_state_patches"
        );
    }
}
