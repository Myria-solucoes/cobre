//! The [`StudyDimensions`] single owner of the study-invariant, non-state LP
//! shape.
//!
//! [`StudyDimensions`] owns the scalar entity counts, presence flags, and
//! anticipated-thermal identity list that are constant across every stage AND
//! every block of a study and are **not** part of the state vector. It is the
//! single persisted owner of those facts: no other long-lived type holds them.

/// Study-invariant, non-state LP shape for an SDDP study.
///
/// Single owner of the layout facts that are invariant across both stages and
/// blocks and are not state-defining: the non-state entity counts, the
/// optional-column presence flags, and the anticipated-thermal identity list.
/// Every reader of one of these facts resolves it here; no other long-lived
/// type carries a copy.
///
/// The exclusions are deliberate, and each excluded dim lives elsewhere because
/// it is owned by a different concern — duplicating it here would reintroduce
/// the multi-owner drift this type exists to remove:
///
/// - **State-defining dims** (`hydro_count`, `max_par_order`, `n_anticipated`,
///   `k_max`, `anticipated_lead_stages`) define the state vector and its cut
///   layout, so they are owned solely by
///   [`StateLayout`](super::StateLayout). A count is either state-defining (→
///   `StateLayout`) or non-state study shape (→ here), never both.
/// - **`n_blks`** is a *per-stage* quantity: a stage's block count is read from
///   its own template, not from a study-global field. A persisted global
///   `n_blks` is the exact footgun that mis-strides equipment columns at any
///   stage whose block count differs from stage 0's, so it is owned by the
///   per-stage geometry, never here.
///
/// `anticipated_thermal_indices` is study-invariant (the anticipated plant set
/// does not change across stages), so it is owned here; the per-stage FPHA /
/// evaporation identity lists vary by stage and are owned by the per-stage
/// geometry instead.
// Rationale: the four bool fields (`has_ncs`, `has_inflow_penalty`,
// `has_withdrawal`, `has_operational_violations`) are independent presence flags
// for optional column groups, not states of one machine; folding them into a
// two-variant enum or state machine — the lint's suggested refactor — would
// obscure that they vary independently.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Default)]
pub struct StudyDimensions {
    /// Number of thermal units (T).
    pub n_thermals: usize,
    /// Number of transmission lines (`L_n`).
    pub n_lines: usize,
    /// Number of buses (B).
    pub n_buses: usize,
    /// Maximum number of deficit segments across all buses (S).
    pub max_deficit_segments: usize,
    /// Whether the study has NCS generation columns.
    ///
    /// `true` when NCS entities are active. The per-(ncs, block) column base is
    /// addressed per stage, never from a study-global base, so only presence is
    /// recorded here.
    pub has_ncs: bool,
    /// Whether inflow non-negativity penalty slack columns are present.
    pub has_inflow_penalty: bool,
    /// Whether withdrawal slack columns are present (`hydro_count > 0`).
    pub has_withdrawal: bool,
    /// Whether operational violation slack columns are present.
    pub has_operational_violations: bool,
    /// Mapping from anticipated-local position to global thermal index.
    ///
    /// `anticipated_thermal_indices[i]` is the position within
    /// `system.thermals[]` of the i-th anticipated plant. Empty when there are
    /// no anticipated thermals.
    pub anticipated_thermal_indices: Vec<usize>,
    /// Number of pumping stations.
    pub n_pumping: usize,
}
