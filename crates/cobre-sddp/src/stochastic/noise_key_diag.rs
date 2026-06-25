//! Throwaway, env-gated backward-pass diagnostic for opening-ordering analysis.
//!
//! When `COBRE_W1_DIAG` is set, the backward pass emits one record per opening
//! pairing a σ-weighted aggregate **noise key** with the just-consumed
//! `simplex_iterations` of that opening's warm dual-simplex re-solve, to predict
//! whether reordering openings by noise similarity would shrink warm-start hops.
//!
//! `noise_key[stage][ω]` is a pure function of setup-constant data (the synced
//! tree's `raw_noise` and the fixed per-(hydro, stage) `std_m3s`), so it is
//! precomputed once at setup and the backward hot path only looks it up by
//! canonical ω — σ never reaches the hot path. The table is built only when the
//! env var is set, so the default path allocates and computes nothing new.
//!
//! ## σ layout alignment
//!
//! The opening noise vector's first `n_hydros` components are the per-hydro
//! inflow residuals η, in canonical `System::hydros()` order, weighted by the
//! seasonal `InflowModel::std_m3s` for that `(hydro, stage)` pair.
//! `build_sigma_table` keys the inflow models on `(hydro_id, stage_id)` exactly
//! as the PAR precompute does, indexed `[stage * n_hydros + h]`; a `(hydro,
//! stage)` pair with no inflow model contributes σ = 0 (the PAR's
//! deterministic-zero-inflow fallback).

use cobre_core::{EntityId, System};
use cobre_stochastic::StochasticContext;

use crate::error::SddpError;

/// Enables the diagnostic when present, with any value including empty.
const DIAG_ENV_VAR: &str = "COBRE_W1_DIAG";

/// Precomputed per-(stage, canonical-ω) σ-weighted noise keys for the backward
/// diagnostic.
///
/// Built once at setup and borrowed read-only by the backward pass via
/// [`TrainingContext`](crate::context::TrainingContext).
/// `keys[stage][omega]` is `Σ_h std_m3s_{stage,h} · raw_noise_{stage,omega}[h]`
/// over the `n_hydros` inflow components of the opening noise vector.
#[derive(Debug, Clone)]
pub struct NoiseKeyDiag {
    keys: Vec<Vec<f64>>,
}

impl NoiseKeyDiag {
    /// Build the diagnostic table iff `COBRE_W1_DIAG` is set, else `None`.
    ///
    /// Reuses the caller's precomputed σ-weighted key table (the same one the
    /// backward solve order sorts by) so the diagnostic and the ordering cannot
    /// drift and the table is built once, not twice.
    pub(crate) fn from_keys_if_enabled(keys: &[Vec<f64>]) -> Option<Self> {
        std::env::var_os(DIAG_ENV_VAR)?;
        Some(Self {
            keys: keys.to_vec(),
        })
    }

    /// Look up the precomputed noise key for `(stage, omega)`, returning `None`
    /// out of range rather than panicking, so a malformed request never aborts.
    #[must_use]
    pub(crate) fn key(&self, stage: usize, omega: usize) -> Option<f64> {
        self.keys.get(stage).and_then(|s| s.get(omega).copied())
    }
}

/// Build the per-(stage, canonical-ω) σ-weighted noise key table from
/// setup-constant data — the SAME key [`NoiseKeyDiag`] records and the ordering
/// work sorts by, so the two cannot drift.
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
/// Only the first `sigma.len()` components of `raw_noise` (the inflow η) are
/// weighted; trailing load/NCS noise components are ignored.
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

/// Build the per-(stage, hydro) seasonal `std_m3s` table, indexed
/// `[stage * n_hydros + h]`, keyed on `(hydro_id, stage_id)` exactly as the PAR
/// precompute so σ matches the canonical hydro order; a `(hydro, stage)` pair
/// with no inflow model contributes σ = 0.
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
