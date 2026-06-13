//! Throwaway, env-gated backward-pass diagnostic for the opening-ordering work.
//!
//! When the `COBRE_W1_DIAG` environment variable is set, the backward pass emits
//! one record per backward opening pairing a σ-weighted aggregate **noise key**
//! with the just-consumed `simplex_iterations` of that opening's warm
//! dual-simplex re-solve. Offline, the per-opening pivots are regressed on the
//! consecutive-opening noise-key distance in natural order to predict whether
//! reordering the opening solve sequence (so consecutive openings have similar
//! noise) would shrink the warm-start hops — the gate for the ordering ticket.
//!
//! ## Why this lives entirely in `cobre-sddp`
//!
//! `noise_key[stage][ω]` is a pure function of setup-constant data — the synced
//! opening tree's `raw_noise` and the fixed per-(hydro, stage) seasonal standard
//! deviation `std_m3s`. It is therefore precomputed **once at setup**, where the
//! inflow models and the synced tree are both in scope, and the backward hot
//! path only **looks it up** by canonical ω (an indexed read, no compute). σ
//! never reaches the hot path.
//!
//! The table is built **only** when `COBRE_W1_DIAG` is set
//! (`NoiseKeyDiag::from_keys_if_enabled` returns `None` otherwise), so the
//! default path allocates nothing and computes
//! nothing new. No infrastructure crate (`cobre-core`, `cobre-stochastic`) is
//! touched and no parquet column / Arrow schema is added — this is throwaway
//! instrumentation, mirroring the reverted `COBRE_CLP_DIAG` pattern.
//!
//! ## σ layout alignment
//!
//! The opening noise vector's first `n_hydros` components are the per-hydro
//! inflow residuals η, in canonical `System::hydros()` order. The σ weight for
//! each is the seasonal `InflowModel::std_m3s` for that `(hydro, stage)` pair.
//! `build_sigma_table` aligns σ to this layout by keying the inflow models on
//! `(hydro_id, stage_id)` exactly as the PAR precompute does, indexed
//! `[stage * n_hydros + h]`. A `(hydro, stage)` pair with no inflow model
//! contributes σ = 0 (matching the PAR's deterministic-zero-inflow fallback).

use cobre_core::{EntityId, System};
use cobre_stochastic::StochasticContext;

use crate::error::SddpError;

/// Environment variable that enables the throwaway backward `noise_key`
/// diagnostic. Presence (any value, including empty) enables it.
const DIAG_ENV_VAR: &str = "COBRE_W1_DIAG";

/// Precomputed per-(stage, canonical-ω) σ-weighted noise keys for the backward
/// diagnostic.
///
/// Built once at setup by `NoiseKeyDiag::from_keys_if_enabled` and borrowed read-only by
/// the backward pass via [`TrainingContext`](crate::context::TrainingContext).
/// `keys[stage][omega]` is `Σ_h std_m3s_{stage,h} · raw_noise_{stage,omega}[h]`
/// over the `n_hydros` inflow components of the opening noise vector.
#[derive(Debug, Clone)]
pub struct NoiseKeyDiag {
    /// `keys[stage][omega]` — the σ-weighted aggregate noise key.
    keys: Vec<Vec<f64>>,
}

impl NoiseKeyDiag {
    /// Build the diagnostic table from the already-computed solve-order key
    /// table **iff** the `COBRE_W1_DIAG` environment variable is set; otherwise
    /// return `None`.
    ///
    /// Reuses the caller's precomputed σ-weighted key table (the same table the
    /// backward solve order sorts by) so the diagnostic and the ordering cannot
    /// drift, and so the table is built once — not twice — at setup. When `None`
    /// (the default), the backward pass performs zero key computation and the
    /// default solve path is byte-identical to `main`.
    pub(crate) fn from_keys_if_enabled(keys: &[Vec<f64>]) -> Option<Self> {
        // Enabled only when the env var is present (any value, incl. empty);
        // absent → `None` (the default, zero-cost path).
        std::env::var_os(DIAG_ENV_VAR)?;
        Some(Self {
            keys: keys.to_vec(),
        })
    }

    /// Look up the precomputed noise key for `(stage, omega)`.
    ///
    /// Returns `None` for an out-of-range `(stage, omega)` rather than panicking,
    /// so a malformed diagnostic request never aborts a run.
    #[must_use]
    pub(crate) fn key(&self, stage: usize, omega: usize) -> Option<f64> {
        self.keys.get(stage).and_then(|s| s.get(omega).copied())
    }
}

/// Build the per-(stage, canonical-ω) σ-weighted noise key table from
/// setup-constant data.
///
/// `table[stage][omega] = Σ_h std_m3s_{stage,h} · raw_noise_{stage,omega}[h]`
/// over the `n_hydros` inflow components of the opening noise vector — the SAME
/// key the throwaway diagnostic ([`NoiseKeyDiag`]) records and the ordering work
/// sorts by, so the two cannot drift.
///
/// σ is aligned to the noise vector layout `[stage * n_hydros + h]` by
/// [`build_sigma_table`], keying the inflow models on `(hydro_id, stage_id)`
/// exactly as the PAR precompute does. A `(hydro, stage)` pair with no inflow
/// model contributes σ = 0.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when the per-(hydro,stage) σ slice is longer
/// than an opening's noise vector (the σ-layout vs noise-dim mismatch), naming
/// both lengths and never silently truncating or zero-padding.
pub(crate) fn build_noise_key_table(
    system: &System,
    stochastic: &StochasticContext,
) -> Result<Vec<Vec<f64>>, SddpError> {
    let n_hydros = stochastic.n_hydros();
    let hydro_ids: Vec<EntityId> = system.hydros().iter().map(|h| h.id).collect();
    let study_stage_ids: Vec<i32> = system
        .stages()
        .iter()
        .filter(|s| s.id >= 0)
        .map(|s| s.id)
        .collect();
    let n_stages = study_stage_ids.len();

    // σ aligned to the noise vector layout: [stage * n_hydros + h].
    let sigma = build_sigma_table(system, &hydro_ids, &study_stage_ids, n_hydros);

    let tree = stochastic.tree_view();
    let mut keys = Vec::with_capacity(n_stages);
    for stage in 0..n_stages {
        let n_openings = tree.n_openings(stage);
        let sigma_stage = &sigma[stage * n_hydros..stage * n_hydros + n_hydros];
        let mut stage_keys = Vec::with_capacity(n_openings);
        for omega in 0..n_openings {
            let raw_noise = tree.opening(stage, omega);
            stage_keys.push(noise_key(sigma_stage, raw_noise)?);
        }
        keys.push(stage_keys);
    }

    Ok(keys)
}

/// Compute the σ-weighted aggregate noise key `Σ_h σ_h · raw_noise[h]`.
///
/// `sigma` is the per-hydro seasonal `std_m3s` slice for one stage, aligned 1:1
/// with the inflow span of the opening `raw_noise` vector. Only the first
/// `sigma.len()` components of `raw_noise` (the inflow η) are weighted; trailing
/// load/NCS noise components are ignored.
///
/// # Errors
///
/// Returns [`SddpError::Validation`] when `raw_noise` is shorter than `sigma`
/// (the σ-layout vs noise-dim mismatch), naming both lengths. The key is never
/// computed over a truncated or zero-padded slice.
pub(crate) fn noise_key(sigma: &[f64], raw_noise: &[f64]) -> Result<f64, SddpError> {
    if raw_noise.len() < sigma.len() {
        return Err(SddpError::Validation(format!(
            "noise_key σ-layout mismatch: σ length {} exceeds opening noise dimension {}; \
             refusing to truncate or zero-pad",
            sigma.len(),
            raw_noise.len(),
        )));
    }
    Ok(sigma.iter().zip(raw_noise.iter()).map(|(s, n)| s * n).sum())
}

/// Build the per-(stage, hydro) seasonal `std_m3s` table, aligned to the opening
/// noise vector layout `[stage * n_hydros + h]`.
///
/// Keys the system's inflow models on `(hydro_id, stage_id)` exactly as the PAR
/// precompute does, so the σ index order matches the canonical hydro order of
/// the noise vector. A `(hydro, stage)` pair with no inflow model contributes
/// σ = 0 (the PAR's deterministic-zero-inflow fallback).
fn build_sigma_table(
    system: &System,
    hydro_ids: &[EntityId],
    study_stage_ids: &[i32],
    n_hydros: usize,
) -> Vec<f64> {
    use std::collections::HashMap;

    let model_std: HashMap<(i32, i32), f64> = system
        .inflow_models()
        .iter()
        .map(|m| ((m.hydro_id.0, m.stage_id), m.std_m3s))
        .collect();

    let mut sigma = vec![0.0_f64; study_stage_ids.len() * n_hydros];
    for (s_idx, &stage_id) in study_stage_ids.iter().enumerate() {
        for (h_idx, hydro_id) in hydro_ids.iter().enumerate() {
            if let Some(&std) = model_std.get(&(hydro_id.0, stage_id)) {
                sigma[s_idx * n_hydros + h_idx] = std;
            }
        }
    }
    sigma
}

#[cfg(test)]
mod tests {
    use super::noise_key;

    #[test]
    fn test_noise_key_sums_sigma_weighted_components() {
        // noise_key = Σ σ_h · raw_noise[h] over a hand-constructed 3-element pair.
        let sigma = [30.0, 20.0, 10.0];
        let raw_noise = [1.5, -2.0, 0.5];
        // 30*1.5 + 20*(-2.0) + 10*0.5 = 45 - 40 + 5 = 10.
        let key = noise_key(&sigma, &raw_noise).expect("dims aligned");
        assert!((key - 10.0).abs() < 1e-12, "expected 10.0, got {key}");
    }

    #[test]
    fn test_noise_key_ignores_trailing_noise_components() {
        // Trailing load/NCS components beyond σ.len() are not weighted.
        let sigma = [2.0, 4.0];
        let raw_noise = [1.0, 1.0, 100.0, -50.0];
        // 2*1 + 4*1 = 6; the 100.0 and -50.0 tail is ignored.
        let key = noise_key(&sigma, &raw_noise).expect("dims aligned");
        assert!((key - 6.0).abs() < 1e-12, "expected 6.0, got {key}");
    }

    #[test]
    fn test_noise_key_hard_errors_on_sigma_longer_than_noise() {
        // σ longer than the noise vector must hard-error, naming both lengths,
        // never silently truncate or zero-pad.
        let sigma = [1.0, 2.0, 3.0];
        let raw_noise = [1.0, 1.0];
        let err = noise_key(&sigma, &raw_noise).expect_err("must reject mismatch");
        let msg = err.to_string();
        assert!(msg.contains('3'), "message must name σ length 3: {msg}");
        assert!(msg.contains('2'), "message must name noise dim 2: {msg}");
    }
}
