//! Policy persistence, provenance, and orchestration for the SDDP solver.
//!
//! This module clusters the read/write surface around a trained policy and the
//! resolved inputs that produced it: loading and validating policy checkpoints,
//! exporting active cuts and basis data for `cobre-io`, reporting which data
//! sources fed the preprocessing pipeline, materialising the resolved-parameter
//! lookup table, recording LP scaling diagnostics, and the shared output-writing
//! helpers consumed by both the CLI binary and the Python bindings.
//!
//! ## Contents
//!
//! - [`policy_load`] — checkpoint compatibility validation and solver-state
//!   reconstruction from deserialized checkpoint data.
//! - [`policy_export`] — conversion of a trained [`FutureCostFunction`](crate::FutureCostFunction)
//!   and [`TrainingResult`](crate::TrainingResult) into the `cobre-io` policy types.
//! - [`provenance`] — [`ModelProvenanceReport`](provenance::ModelProvenanceReport),
//!   the JSON summary of which data sources fed each preprocessing role.
//! - [`resolved_parameters`] — the `(parameter_id, stage_idx) → f64` lookup table
//!   built once before LP construction. Owns its own wire-format version byte: a
//!   round-trip and a reject-old-version test guard the codec, so the version byte
//!   advances in lockstep with the envelope layout rather than reading a stale
//!   payload as current.
//! - [`scaling_report`] — pre/post-scaling coefficient-range diagnostics for the
//!   stage template LPs, written once after template build.
//! - [`orchestration`] — the shared policy-checkpoint writer and stochastic-artifacts
//!   exporter that the CLI and Python bindings both call, so the two front ends
//!   write byte-identical outputs from one implementation.

pub mod orchestration;
pub mod policy_export;
pub mod policy_load;
pub mod provenance;
pub mod resolved_parameters;
pub mod scaling_report;
