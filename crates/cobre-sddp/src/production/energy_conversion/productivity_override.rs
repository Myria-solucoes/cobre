//! User-supplied productivity override table: the
//! [`HydroEnergyProductivityOverride`] lookup struct and its
//! [`build_hydro_energy_productivity_override`] constructor.
//!
//! The struct fields are private, so the constructor — which scatters parsed
//! rows into the per-stage / per-hydro-default lookup tables via a
//! private-field struct literal — must live in the same module.

use std::collections::{HashMap, HashSet};

use cobre_core::EntityId;
use cobre_io::{HydroEnergyProductivityRow, LoadError};

/// Per-`(hydro, stage)` override table loaded from
/// `system/hydro_energy_productivity.parquet`.
///
/// Each of the four override columns (`ρ_eq`, `V_ref`, `Q_ref`, `ρ_esp`) is
/// stored in two parallel lookup tables: one for stage-specific rows and one
/// for per-hydro defaults (rows whose `stage_id` is NULL in the source file).
///
/// Accessors apply a per-stage → per-hydro-default → `None` fallback chain.
/// `Default` yields an override where every accessor returns `None`.
#[derive(Debug, Default, Clone)]
pub struct HydroEnergyProductivityOverride {
    rho_eq_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    rho_eq_per_hydro_default: HashMap<EntityId, f64>,
    v_ref_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    v_ref_per_hydro_default: HashMap<EntityId, f64>,
    q_ref_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    q_ref_per_hydro_default: HashMap<EntityId, f64>,
    rho_esp_per_hydro_stage: HashMap<(EntityId, i32), f64>,
    rho_esp_per_hydro_default: HashMap<EntityId, f64>,
}

impl HydroEnergyProductivityOverride {
    /// Returns the user-supplied `ρ_eq` for `(hydro, stage)` if any.
    ///
    /// Looks up the stage-specific entry first, then falls back to the
    /// per-hydro default (NULL `stage_id` rows), then returns `None`.
    /// An out-of-range `stage` (larger than `i32::MAX`) always returns `None`.
    #[must_use]
    pub fn equivalent_productivity(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.rho_eq_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.rho_eq_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `V_ref` \[hm³\] for `(hydro, stage)` if any.
    ///
    /// Applies the same per-stage → per-hydro-default → `None` fallback as
    /// [`equivalent_productivity`](Self::equivalent_productivity).
    #[must_use]
    pub fn reference_volume(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.v_ref_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.v_ref_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `Q_ref` \[m³/s\] for `(hydro, stage)` if any.
    ///
    /// Applies the same per-stage → per-hydro-default → `None` fallback as
    /// [`equivalent_productivity`](Self::equivalent_productivity).
    #[must_use]
    pub fn reference_outflow(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.q_ref_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.q_ref_per_hydro_default.get(&hydro).copied()
    }

    /// Returns the user-supplied `ρ_esp` \[MW/(m³/s)/m\] for `(hydro, stage)` if any.
    ///
    /// Applies the same per-stage → per-hydro-default → `None` fallback as
    /// [`equivalent_productivity`](Self::equivalent_productivity).
    #[must_use]
    pub fn specific_productivity(&self, hydro: EntityId, stage: usize) -> Option<f64> {
        let s = i32::try_from(stage).ok()?;
        if let Some(&v) = self.rho_esp_per_hydro_stage.get(&(hydro, s)) {
            return Some(v);
        }
        self.rho_esp_per_hydro_default.get(&hydro).copied()
    }
}

/// Build a [`HydroEnergyProductivityOverride`] from parsed rows.
///
/// Each non-`None` override column is scattered into the matching per-stage or
/// per-hydro-default lookup table. Duplicate `(hydro_id, stage_id)` pairs
/// (with NULL `stage_id` distinct from any concrete value) are rejected.
///
/// # Errors
///
/// Returns [`LoadError::SchemaError`] with
/// `field = "hydro_energy_productivity.duplicate_entry"` when the same
/// `(hydro_id, stage_id)` key appears more than once in `rows`.
pub fn build_hydro_energy_productivity_override(
    rows: &[HydroEnergyProductivityRow],
) -> Result<HydroEnergyProductivityOverride, LoadError> {
    let mut seen: HashSet<(EntityId, Option<i32>)> = HashSet::with_capacity(rows.len());
    let mut out = HydroEnergyProductivityOverride {
        rho_eq_per_hydro_stage: HashMap::new(),
        rho_eq_per_hydro_default: HashMap::new(),
        v_ref_per_hydro_stage: HashMap::new(),
        v_ref_per_hydro_default: HashMap::new(),
        q_ref_per_hydro_stage: HashMap::new(),
        q_ref_per_hydro_default: HashMap::new(),
        rho_esp_per_hydro_stage: HashMap::new(),
        rho_esp_per_hydro_default: HashMap::new(),
    };

    for row in rows {
        let key = (row.hydro_id, row.stage_id);
        if !seen.insert(key) {
            let stage_label = row
                .stage_id
                .map_or_else(|| "NULL".to_string(), |s| s.to_string());
            return Err(LoadError::SchemaError {
                path: std::path::PathBuf::from("<hydro_energy_productivity>"),
                field: "hydro_energy_productivity.duplicate_entry".to_string(),
                message: format!(
                    "duplicate (hydro_id={}, stage_id={}) key",
                    row.hydro_id.0, stage_label
                ),
            });
        }

        if let Some(s) = row.stage_id {
            if let Some(v) = row.equivalent_productivity_mw_per_m3s {
                out.rho_eq_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.reference_volume_hm3 {
                out.v_ref_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.reference_outflow_m3s {
                out.q_ref_per_hydro_stage.insert((row.hydro_id, s), v);
            }
            if let Some(v) = row.specific_productivity_mw_per_m3s_per_m {
                out.rho_esp_per_hydro_stage.insert((row.hydro_id, s), v);
            }
        } else {
            if let Some(v) = row.equivalent_productivity_mw_per_m3s {
                out.rho_eq_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.reference_volume_hm3 {
                out.v_ref_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.reference_outflow_m3s {
                out.q_ref_per_hydro_default.insert(row.hydro_id, v);
            }
            if let Some(v) = row.specific_productivity_mw_per_m3s_per_m {
                out.rho_esp_per_hydro_default.insert(row.hydro_id, v);
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::expect_used,
    clippy::float_cmp,
    clippy::panic,
    clippy::unwrap_used
)]
mod tests {
    use cobre_core::EntityId;
    use cobre_io::HydroEnergyProductivityRow;

    use super::*;

    #[test]
    fn test_override_four_column_lookup_precedence() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.6),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(4.0),
                reference_volume_hm3: Some(120.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: Some(0.009),
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(2),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(5.0),
                reference_volume_hm3: None,
                reference_outflow_m3s: Some(200.0),
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let o = build_hydro_energy_productivity_override(&rows).expect("override builds");
        assert_eq!(o.equivalent_productivity(EntityId(1), 0), Some(3.6));
        assert_eq!(o.equivalent_productivity(EntityId(1), 1), Some(4.0));
        assert_eq!(o.equivalent_productivity(EntityId(2), 0), Some(5.0));
        assert_eq!(o.equivalent_productivity(EntityId(3), 0), None);
        assert_eq!(o.reference_volume(EntityId(1), 0), Some(120.0));
        assert_eq!(o.reference_volume(EntityId(1), 1), Some(120.0));
        assert_eq!(o.reference_volume(EntityId(2), 0), None);
        assert_eq!(o.reference_outflow(EntityId(2), 0), Some(200.0));
        assert_eq!(o.reference_outflow(EntityId(1), 0), None);
        assert_eq!(o.specific_productivity(EntityId(1), 0), Some(0.009));
        assert_eq!(o.specific_productivity(EntityId(1), 1), Some(0.009));
        assert_eq!(o.specific_productivity(EntityId(2), 0), None);
    }

    #[test]
    fn test_build_override_rejects_duplicate_hydro_stage() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.6),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: None,
                reference_volume_hm3: Some(120.0),
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let err = build_hydro_energy_productivity_override(&rows).unwrap_err();
        match err {
            cobre_io::LoadError::SchemaError { field, .. } => {
                assert_eq!(field, "hydro_energy_productivity.duplicate_entry");
            }
            other => panic!("expected SchemaError, got: {other:?}"),
        }
    }

    #[test]
    fn test_build_override_distinguishes_null_and_concrete_stages() {
        let rows = vec![
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: None,
                equivalent_productivity_mw_per_m3s: Some(2.0),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
            HydroEnergyProductivityRow {
                hydro_id: EntityId(1),
                stage_id: Some(0),
                equivalent_productivity_mw_per_m3s: Some(3.0),
                reference_volume_hm3: None,
                reference_outflow_m3s: None,
                specific_productivity_mw_per_m3s_per_m: None,
            },
        ];
        let o = build_hydro_energy_productivity_override(&rows).expect("override builds");
        assert_eq!(o.equivalent_productivity(EntityId(1), 0), Some(3.0));
        assert_eq!(o.equivalent_productivity(EntityId(1), 1), Some(2.0));
    }

    #[test]
    fn test_default_override_returns_none_for_every_accessor() {
        let o = HydroEnergyProductivityOverride::default();
        assert_eq!(o.equivalent_productivity(EntityId(1), 0), None);
        assert_eq!(o.reference_volume(EntityId(1), 0), None);
        assert_eq!(o.reference_outflow(EntityId(1), 0), None);
        assert_eq!(o.specific_productivity(EntityId(1), 0), None);
    }
}
