//! Offline LP prescaling: geometric-mean column/row scale factors applied to
//! stage templates for numerical conditioning (`D_r * A * D_c` form), plus the
//! noise pre-scaling helper. Invoked from `setup/template_postprocess::postprocess_templates`.

use cobre_core::{Hydro, Stage};
use cobre_solver::StageTemplate;
use cobre_stochastic::par::precompute::PrecomputedPar;

use super::M3S_TO_HM3;

/// Per-column geometric-mean scaling factors from a CSC matrix:
/// `1 / sqrt(max|A_ij| * min|A_ij|)` over nonzeros, `1.0` for an empty column.
/// Length `num_cols`.
#[must_use]
#[allow(clippy::cast_sign_loss)] // col_starts are non-negative by CSC construction
pub(crate) fn compute_col_scale(num_cols: usize, col_starts: &[i32], values: &[f64]) -> Vec<f64> {
    let mut scale = vec![1.0_f64; num_cols];
    for j in 0..num_cols {
        let start = col_starts[j] as usize;
        let end = col_starts[j + 1] as usize;
        if start == end {
            continue;
        }
        let mut max_abs = 0.0_f64;
        let mut min_abs = f64::INFINITY;
        for &v in &values[start..end] {
            let abs_val = v.abs();
            if abs_val > 0.0 {
                max_abs = max_abs.max(abs_val);
                min_abs = min_abs.min(abs_val);
            }
        }
        if max_abs > 0.0 && min_abs < f64::INFINITY {
            let d = 1.0 / (max_abs * min_abs).sqrt();
            scale[j] = d;
        }
    }
    scale
}

/// Apply column scaling in-place. After this call, per column `j`: `values` and
/// `objective` are MULTIPLIED by `col_scale[j]`, while `col_lower`/`col_upper` are
/// DIVIDED by it (the scaled variable is `x̃ = x / d_j`).
pub(crate) fn apply_col_scale(template: &mut StageTemplate, col_scale: &[f64]) {
    let num_cols = template.num_cols;
    debug_assert_eq!(col_scale.len(), num_cols);

    #[allow(clippy::needless_range_loop, clippy::cast_sign_loss)]
    // j+1 access; col_starts non-negative by construction
    for j in 0..num_cols {
        let start = template.col_starts[j] as usize;
        let end = template.col_starts[j + 1] as usize;
        let d = col_scale[j];
        for v in &mut template.values[start..end] {
            *v *= d;
        }
    }

    for (obj, &d) in template.objective.iter_mut().zip(col_scale) {
        *obj *= d;
    }

    // Inverse-scale column bounds: `x̃ = x / d_j`, so bounds become `[lo/d, hi/d]`.
    for ((lo, hi), &d) in template
        .col_lower
        .iter_mut()
        .zip(template.col_upper.iter_mut())
        .zip(col_scale)
    {
        *lo /= d;
        *hi /= d;
    }
}

/// Per-row geometric-mean scaling factors from a CSC matrix:
/// `1 / sqrt(max|A_ij| * min|A_ij|)` over a row's nonzeros, `1.0` for an empty row.
/// Length `num_rows`.
///
/// MUST be called on the ALREADY column-scaled matrix to obtain the standard
/// `D_r * A * D_c` form (column scaling before row scaling).
#[must_use]
#[allow(clippy::cast_sign_loss)] // col_starts and row_indices are non-negative by CSC construction
pub(crate) fn compute_row_scale(
    num_rows: usize,
    num_cols: usize,
    col_starts: &[i32],
    row_indices: &[i32],
    values: &[f64],
) -> Vec<f64> {
    let mut row_max = vec![0.0_f64; num_rows];
    let mut row_min = vec![f64::INFINITY; num_rows];

    #[allow(clippy::needless_range_loop)] // j+1 access on col_starts requires index
    for j in 0..num_cols {
        let start = col_starts[j] as usize;
        let end = col_starts[j + 1] as usize;
        for k in start..end {
            let row = row_indices[k] as usize;
            let abs_val = values[k].abs();
            if abs_val > 0.0 {
                row_max[row] = row_max[row].max(abs_val);
                row_min[row] = row_min[row].min(abs_val);
            }
        }
    }

    let mut scale = vec![1.0_f64; num_rows];
    for (s, (&rmax, &rmin)) in scale.iter_mut().zip(row_max.iter().zip(row_min.iter())) {
        if rmax > 0.0 && rmin < f64::INFINITY {
            *s = 1.0 / (rmax * rmin).sqrt();
        }
    }
    scale
}

/// Apply row scaling in-place: per row `i`, `values` and `row_lower`/`row_upper`
/// are MULTIPLIED by `row_scale[i]`. Objective and column bounds are NOT touched —
/// those are column-domain quantities handled by [`apply_col_scale`].
pub(crate) fn apply_row_scale(template: &mut StageTemplate, row_scale: &[f64]) {
    let num_rows = template.num_rows;
    debug_assert_eq!(row_scale.len(), num_rows);

    let num_cols = template.num_cols;
    #[allow(clippy::needless_range_loop, clippy::cast_sign_loss)]
    // j+1 access; values non-negative by construction
    for j in 0..num_cols {
        let start = template.col_starts[j] as usize;
        let end = template.col_starts[j + 1] as usize;
        for k in start..end {
            let row = template.row_indices[k] as usize;
            template.values[k] *= row_scale[row];
        }
    }

    for ((lo, hi), &d) in template
        .row_lower
        .iter_mut()
        .zip(template.row_upper.iter_mut())
        .zip(row_scale)
    {
        *lo *= d;
        *hi *= d;
    }
}

/// Pre-compute `ζ * σ` per `(stage, hydro)` for noise transformation, returning
/// `(noise_scale, zeta_per_stage, block_hours_per_stage)`. `noise_scale` is flat,
/// `[s_idx * n_hydros + h_idx]`, so the forward pass indexes it without branching.
///
/// A `PreFilling` hydro gets `noise_scale = 0`: its water-balance row is the
/// frozen-storage identity `v_h − v_h_in = 0`, so leaving `noise_scale` nonzero
/// would patch `noise_scale·eta` onto that RHS and unfreeze the storage column. This
/// is the ONLY noise channel touched — the `z_inflow` RHS is NOT zeroed, so realized
/// inflow `z_h` still flows downstream in the `PreFilling` cascade short-circuit. A
/// non-filling hydro is `Operating` at every stage (parity-neutral).
pub(super) fn compute_noise_scale(
    study_stages: &[&Stage],
    hydros: &[Hydro],
    n_hydros: usize,
    par_lp: &PrecomputedPar,
) -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let n = study_stages.len();
    let mut noise_scale = vec![0.0_f64; n * n_hydros];
    let mut zeta_per_stage = Vec::with_capacity(n);
    let mut block_hours_per_stage = Vec::with_capacity(n);

    for (s_idx, stage) in study_stages.iter().enumerate() {
        let total_hours: f64 = stage.blocks.iter().map(|b| b.duration_hours).sum();
        let zeta_s = total_hours * M3S_TO_HM3;
        zeta_per_stage.push(zeta_s);
        block_hours_per_stage.push(stage.blocks.iter().map(|b| b.duration_hours).collect());
        for h_idx in 0..n_hydros {
            // `hydros` shorter than `n_hydros` (a direct-construction test path)
            // defaults missing entries to `Operating` via `.get()`, never a panic.
            let is_prefilling = hydros.get(h_idx).is_some_and(|hydro| {
                matches!(
                    crate::lp_builder::filling_phase(
                        hydro.filling.as_ref(),
                        hydro.entry_stage_id,
                        stage.id,
                    ),
                    crate::lp_builder::Phase::PreFilling
                )
            });
            let sigma = if !is_prefilling && par_lp.n_stages() > 0 && par_lp.n_hydros() == n_hydros
            {
                par_lp.sigma(s_idx, h_idx)
            } else {
                0.0
            };
            noise_scale[s_idx * n_hydros + h_idx] = zeta_s * sigma;
        }
    }

    (noise_scale, zeta_per_stage, block_hours_per_stage)
}

#[cfg(test)]
#[allow(
    clippy::doc_markdown,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    clippy::unwrap_used,
    clippy::expect_used
)]
mod tests {
    use cobre_solver::StageTemplate;

    // =========================================================================
    // Row scaling tests
    // =========================================================================

    /// Build a minimal `StageTemplate` for row-scaling unit tests.
    ///
    /// The matrix is given in CSC form.  All non-LP-semantic fields are zeroed
    /// so the helpers under test only touch the fields they care about.
    fn minimal_template(
        num_rows: usize,
        num_cols: usize,
        col_starts: Vec<i32>,
        row_indices: Vec<i32>,
        values: Vec<f64>,
        row_lower: Vec<f64>,
        row_upper: Vec<f64>,
    ) -> StageTemplate {
        let num_nz = values.len();
        StageTemplate {
            num_cols,
            num_rows,
            num_nz,
            col_starts,
            row_indices,
            values,
            col_lower: vec![0.0; num_cols],
            col_upper: vec![f64::INFINITY; num_cols],
            objective: vec![0.0; num_cols],
            row_lower,
            row_upper,
            n_transfer: 0,
            n_dual_relevant: 0,
            n_hydro: 0,
            max_par_order: 0,
            col_scale: Vec::new(),
            row_scale: Vec::new(),
            n_state: 0,
        }
    }

    /// AC E3-001-1: a matrix where every row has min_abs == max_abs gives scale 1.0.
    ///
    /// Matrix (2 rows × 2 cols, column-major), all nonzeros |value| = 1.0:
    ///
    /// ```text
    /// col 0: row 0 → 1.0, row 1 → 1.0
    /// col 1: row 0 → 1.0, row 1 → 1.0
    /// ```
    ///
    /// For each row: min_abs = max_abs = 1.0 → scale = 1/sqrt(1*1) = 1.0.
    #[test]
    fn row_scale_identity_for_uniform_matrix() {
        // All nonzeros have |value| = 1.0.  min_abs = max_abs = 1.0.
        // scale[i] = 1/sqrt(1.0 * 1.0) = 1.0 for every row.
        let col_starts = vec![0, 2, 4];
        let row_indices = vec![0, 1, 0, 1];
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let scale = super::compute_row_scale(2, 2, &col_starts, &row_indices, &values);
        assert_eq!(scale.len(), 2);
        assert!(
            (scale[0] - 1.0).abs() < 1e-15,
            "row 0 scale should be 1.0, got {}",
            scale[0]
        );
        assert!(
            (scale[1] - 1.0).abs() < 1e-15,
            "row 1 scale should be 1.0, got {}",
            scale[1]
        );
    }

    /// AC E3-001-2: geometric-mean scale matches expected value for known matrix.
    ///
    /// Matrix (2 rows × 2 cols):
    ///
    /// ```text
    /// col 0: row 0 → 1.0
    /// col 1: row 0 → 100.0, row 1 → 4.0
    /// ```
    ///
    /// Row 0: min_abs = 1.0, max_abs = 100.0 → scale = 1/sqrt(100) = 0.1
    /// Row 1: min_abs = max_abs = 4.0         → scale = 1/sqrt(16)  = 0.25
    #[test]
    fn row_scale_geometric_mean() {
        let col_starts = vec![0, 1, 3];
        let row_indices = vec![0, 0, 1];
        let values = vec![1.0, 100.0, 4.0];
        let scale = super::compute_row_scale(2, 2, &col_starts, &row_indices, &values);
        assert_eq!(scale.len(), 2);
        let expected_row0 = 1.0_f64 / (1.0_f64 * 100.0_f64).sqrt(); // 0.1
        let expected_row1 = 1.0_f64 / (4.0_f64 * 4.0_f64).sqrt(); // 0.25
        assert!(
            (scale[0] - expected_row0).abs() < 1e-14,
            "row 0 scale: expected {expected_row0}, got {}",
            scale[0]
        );
        assert!(
            (scale[1] - expected_row1).abs() < 1e-14,
            "row 1 scale: expected {expected_row1}, got {}",
            scale[1]
        );
    }

    /// AC E3-001-3: `apply_row_scale` multiplies matrix values and row bounds.
    ///
    /// Uses the same 2×2 matrix as `row_scale_geometric_mean` so the expected
    /// values are easily verified by hand.
    #[test]
    fn apply_row_scale_scales_values_and_bounds() {
        // CSC: col 0 has one nonzero (row 0, val 1.0); col 1 has two (row 0→100.0, row 1→4.0).
        let col_starts = vec![0_i32, 1, 3];
        let row_indices = vec![0_i32, 0, 1];
        let values = vec![1.0_f64, 100.0, 4.0];
        let row_lower = vec![-5.0_f64, 7.0];
        let row_upper = vec![f64::INFINITY, 7.0];

        let mut tmpl =
            minimal_template(2, 2, col_starts, row_indices, values, row_lower, row_upper);

        // Row 0: scale = 1/sqrt(1*100) = 0.1
        // Row 1: scale = 1/sqrt(4*4)   = 0.25
        let row_scale = vec![0.1_f64, 0.25];
        super::apply_row_scale(&mut tmpl, &row_scale);

        // Matrix values: entry (row 0, col 0) = 1.0 * 0.1 = 0.1
        assert!((tmpl.values[0] - 0.1).abs() < 1e-15, "value[0] wrong");
        // Entry (row 0, col 1) = 100.0 * 0.1 = 10.0
        assert!((tmpl.values[1] - 10.0).abs() < 1e-15, "value[1] wrong");
        // Entry (row 1, col 1) = 4.0 * 0.25 = 1.0
        assert!((tmpl.values[2] - 1.0).abs() < 1e-15, "value[2] wrong");

        // Row bounds: row 0 lower = -5.0 * 0.1 = -0.5
        assert!(
            (tmpl.row_lower[0] - (-0.5)).abs() < 1e-15,
            "row_lower[0] wrong"
        );
        // Row 0 upper is INFINITY — must remain INFINITY after scaling.
        assert!(
            tmpl.row_upper[0].is_infinite() && tmpl.row_upper[0] > 0.0,
            "row_upper[0] must remain +inf"
        );
        // Row 1 lower = 7.0 * 0.25 = 1.75
        assert!(
            (tmpl.row_lower[1] - 1.75).abs() < 1e-15,
            "row_lower[1] wrong"
        );
        // Row 1 upper = 7.0 * 0.25 = 1.75 (equality constraint: lower == upper after scaling)
        assert!(
            (tmpl.row_upper[1] - 1.75).abs() < 1e-15,
            "row_upper[1] wrong"
        );

        // Column bounds and objective must be untouched.
        assert_eq!(tmpl.col_lower, vec![0.0; 2]);
        assert!(tmpl.col_upper[0].is_infinite());
        assert!(tmpl.col_upper[1].is_infinite());
        assert_eq!(tmpl.objective, vec![0.0; 2]);
    }

    /// AC E3-001-4: a row with no nonzeros receives scale factor 1.0.
    ///
    /// Matrix (3 rows × 1 col): only row 1 has a nonzero.
    /// Rows 0 and 2 are structurally empty → scale = 1.0.
    #[test]
    fn row_scale_empty_row_gets_one() {
        // col 0 has one nonzero: (row 1, val 8.0)
        let col_starts = vec![0_i32, 1];
        let row_indices = vec![1_i32];
        let values = vec![8.0_f64];
        let scale = super::compute_row_scale(3, 1, &col_starts, &row_indices, &values);
        assert_eq!(scale.len(), 3);
        // Rows 0 and 2 are empty → scale 1.0
        assert!(
            (scale[0] - 1.0).abs() < 1e-15,
            "empty row 0 scale should be 1.0"
        );
        // Row 1 has min_abs = max_abs = 8.0 → scale = 1/8
        let expected = 1.0_f64 / 8.0;
        assert!(
            (scale[1] - expected).abs() < 1e-15,
            "row 1 scale: expected {expected}, got {}",
            scale[1]
        );
        assert!(
            (scale[2] - 1.0).abs() < 1e-15,
            "empty row 2 scale should be 1.0"
        );
    }

    // =========================================================================
    // PreFilling noise-scale zeroing
    // =========================================================================

    use chrono::NaiveDate;
    use cobre_core::scenario::InflowModel;
    use cobre_core::{
        Block, BlockMode, EntityId, FillingConfig, Hydro, HydroGenerationModel, HydroPenalties,
        NoiseMethod, ScenarioSourceConfig, Stage, StageRiskConfig, StageStateConfig,
    };
    use cobre_stochastic::par::precompute::PrecomputedPar;

    fn zero_hydro_penalties() -> HydroPenalties {
        HydroPenalties {
            spillage_cost: 0.0,
            diversion_cost: 0.0,
            turbined_cost: 0.0,
            storage_violation_below_cost: 0.0,
            filling_target_violation_cost: 0.0,
            turbined_violation_below_cost: 0.0,
            outflow_violation_below_cost: 0.0,
            outflow_violation_above_cost: 0.0,
            generation_violation_below_cost: 0.0,
            evaporation_violation_cost: 0.0,
            water_withdrawal_violation_cost: 0.0,
            water_withdrawal_violation_pos_cost: 0.0,
            water_withdrawal_violation_neg_cost: 0.0,
            evaporation_violation_pos_cost: 0.0,
            evaporation_violation_neg_cost: 0.0,
            inflow_nonnegativity_cost: 0.0,
        }
    }

    /// A constant-productivity hydro with optional filling at `start_stage_id = 2`,
    /// `entry_stage_id = 4`. With `filling = true` a stage id `< 2` is PreFilling.
    fn noise_hydro(id: i32, filling: bool) -> Hydro {
        Hydro {
            id: EntityId(id),
            name: format!("H{id}"),
            bus_id: EntityId(1),
            downstream_id: None,
            entry_stage_id: filling.then_some(4),
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 100.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 50.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 45.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: filling.then_some(FillingConfig {
                start_stage_id: 2,
                filling_min_rate_m3s: 0.0,
            }),
            penalties: zero_hydro_penalties(),
        }
    }

    fn one_block_stage(id: i32) -> Stage {
        Stage {
            index: usize::try_from(id).unwrap(),
            id,
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: Some(0),
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: false,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
        }
    }

    fn white_noise_model(id: i32, stage_id: i32, std: f64) -> InflowModel {
        InflowModel {
            hydro_id: EntityId(id),
            stage_id,
            mean_m3s: 50.0,
            std_m3s: std,
            ar_coefficients: vec![],
            residual_std_ratio: 1.0,
            annual: None,
        }
    }

    /// A PreFilling hydro's water-balance `noise_scale` is zeroed for that
    /// `(stage, hydro)`, while an Operating hydro at the same stage (with σ > 0)
    /// keeps its `ζ·σ` value. The forbidden alternative — leaving the PreFilling
    /// noise_scale nonzero — would patch `noise_scale·eta ≠ 0` onto the
    /// frozen-storage-identity row and unfreeze the storage column.
    #[test]
    fn prefilling_water_noise_scale_is_zeroed_operating_is_not() {
        // H1 (id 1): non-filling (Operating). H2 (id 2): filling, PreFilling at id 0.
        let hydros = vec![noise_hydro(1, false), noise_hydro(2, true)];
        let n_hydros = 2;
        let std = 4.0_f64;
        let stage = one_block_stage(0); // id 0 < start_stage_id 2 ⇒ H2 PreFilling.
        let inflow_models = vec![white_noise_model(1, 0, std), white_noise_model(2, 0, std)];
        let par_lp = PrecomputedPar::build(
            &inflow_models,
            std::slice::from_ref(&stage),
            &[EntityId(1), EntityId(2)],
            None,
        )
        .expect("white-noise PrecomputedPar build must succeed");

        let study_stages = [&stage];
        let (noise_scale, zeta_per_stage, _bh) =
            super::compute_noise_scale(&study_stages, &hydros, n_hydros, &par_lp);

        let zeta = zeta_per_stage[0];
        let h1_op = 0; // id 1 → system index 0 (Operating).
        let h2_pf = 1; // id 2 → system index 1 (PreFilling at stage id 0).

        // Operating H1 keeps ζ·σ (σ = std > 0 for white noise).
        assert_eq!(
            noise_scale[h1_op],
            zeta * par_lp.sigma(0, h1_op),
            "Operating hydro's noise_scale is the standard ζ·σ"
        );
        assert!(
            noise_scale[h1_op] > 0.0,
            "control precondition: Operating σ > 0 so the contrast is real"
        );

        // PreFilling H2 is zeroed despite σ > 0.
        assert!(
            par_lp.sigma(0, h2_pf) > 0.0,
            "precondition: H2 has σ > 0, so the zeroing is not vacuous"
        );
        assert_eq!(
            noise_scale[h2_pf], 0.0,
            "PreFilling hydro's water-balance noise_scale is zeroed (frozen-identity freeze)"
        );
    }

    /// At a Filling stage (id == start_stage_id), the filling hydro is NOT
    /// PreFilling, so its noise_scale is NOT zeroed — the freeze is PreFilling-only.
    /// (Filling keeps PAR/noise per the retention design.)
    #[test]
    fn filling_stage_noise_scale_is_not_zeroed() {
        let hydros = vec![noise_hydro(2, true)];
        let n_hydros = 1;
        let std = 4.0_f64;
        // Stage id 2 == start_stage_id ⇒ Filling (not PreFilling).
        let stage = one_block_stage(2);
        let inflow_models = vec![white_noise_model(2, 2, std)];
        let par_lp = PrecomputedPar::build(
            &inflow_models,
            std::slice::from_ref(&stage),
            &[EntityId(2)],
            None,
        )
        .expect("build");

        let study_stages = [&stage];
        let (noise_scale, zeta_per_stage, _bh) =
            super::compute_noise_scale(&study_stages, &hydros, n_hydros, &par_lp);

        assert_eq!(
            noise_scale[0],
            zeta_per_stage[0] * par_lp.sigma(0, 0),
            "Filling-stage noise_scale keeps ζ·σ (only PreFilling is zeroed)"
        );
        assert!(
            noise_scale[0] > 0.0,
            "Filling stage keeps a nonzero noise_scale"
        );
    }
}
