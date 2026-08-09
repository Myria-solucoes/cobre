//! Layer 5a — generic-constraint-vs-stage-mode semantic validation.
//!
//! Rejects per-block variable references that cannot resolve to a real column on a
//! stage: interior storage boundaries (`HydroStorageInitial` / `HydroStorageFinal`)
//! on parallel stages, and out-of-range block selectors on any per-block reference.

use cobre_core::{VariableRef, temporal::BlockMode};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

/// Layer 5a — rejects per-block variable references that address a column a stage
/// cannot expose, honoring per-`(constraint, stage)` activation.
///
/// `HydroStorageInitial{Some(b)}` references boundary `b` (start of block `b`);
/// `HydroStorageFinal{Some(b)}` references boundary `b + 1` (end of block `b`); a
/// boundary is interior iff `0 < k < K`. On a `Parallel` stage with `K > 1` only
/// the two endpoints (`0`, `K`) exist — the `storage_internal` family is empty — so
/// an interior `Some(..)` reference resolves outside the storage family and is
/// rejected. A `None` selector is the stage endpoint (`S⁰` / `Sᴷ`), which always
/// exists, so it always passes. `HydroEvaporation{Some(b)}` addresses one of the
/// `K` per-block evaporation columns (present in both modes), so only out-of-range
/// `b >= K` is rejected there; a bare `HydroEvaporation{None}` is the stage
/// evaporation in parallel mode (every block shares the same endpoints) but is
/// ambiguous on a chronological stage with `K > 1` (the blocks differ), so it is
/// rejected there — a block must be named.
///
/// An out-of-range `block_id` (`b >= K`) is rejected on every stage: it would
/// otherwise resolve to no column and silently drop the term.
///
/// A `(constraint, stage)` pair the constraint does not activate (no bound row for
/// that stage) is skipped: no LP row is emitted there, so validating it would
/// false-reject a legitimate mixed-mode study.
pub(super) fn check_per_block_storage_interior_reference(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    for stage in data.stages.stages.iter().filter(|s| s.id >= 0) {
        let k = stage.blocks.len();
        let stage_id = stage.id;
        for constraint in &data.generic_constraints {
            if !constraint_active_on_stage(data, constraint.id.0, stage_id) {
                continue;
            }
            for term in &constraint.expression.terms {
                validate_block_ref(
                    constraint,
                    &term.variable,
                    k,
                    stage_id,
                    stage.block_mode,
                    ctx,
                );
            }
        }
    }
}

/// Whether `constraint` has a bound row at `stage_id` — mirrors
/// `ResolvedGenericConstraintBounds::is_active`; only an active `(constraint, stage)`
/// pair contributes an LP row.
fn constraint_active_on_stage(data: &ParsedData, constraint_id: i32, stage_id: i32) -> bool {
    data.generic_constraint_bounds
        .iter()
        .any(|r| r.constraint_id == constraint_id && r.stage_id == stage_id)
}

/// Which per-block storage boundary a term references, and the block selector.
///
/// `Initial` references boundary `block_id`; `Final` references `block_id + 1`.
#[derive(Clone, Copy)]
enum StorageRef {
    Initial(Option<usize>),
    Final(Option<usize>),
}

/// Dispatch a term to its per-block validity check. Storage boundaries get the
/// full interior + out-of-range check; evaporation gets out-of-range only (its `K`
/// per-block columns exist in both modes); every other variant is unrestricted.
fn validate_block_ref(
    constraint: &cobre_core::GenericConstraint,
    variable: &VariableRef,
    k: usize,
    stage_id: i32,
    block_mode: BlockMode,
    ctx: &mut ValidationContext,
) {
    match variable {
        VariableRef::HydroStorageInitial { block_id, .. } => {
            validate_storage_ref(
                constraint,
                StorageRef::Initial(*block_id),
                k,
                stage_id,
                block_mode,
                ctx,
            );
        }
        VariableRef::HydroStorageFinal { block_id, .. } => {
            validate_storage_ref(
                constraint,
                StorageRef::Final(*block_id),
                k,
                stage_id,
                block_mode,
                ctx,
            );
        }
        VariableRef::HydroEvaporation {
            block_id: Some(b), ..
        } if *b >= k => {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "constraints/generic_constraints.json",
                Some(format!("constraint[id={}]", constraint.id.0)),
                format!(
                    "Constraint \"{}\": per-block evaporation reference \
                     `hydro_evaporation({b})` at stage {stage_id} references block {b} \
                     which does not exist at stage {stage_id} (K = {k})",
                    constraint.name
                ),
            );
        }
        VariableRef::HydroEvaporation { block_id: None, .. }
            if block_mode == BlockMode::Chronological && k > 1 =>
        {
            ctx.add_error(
                ErrorKind::BusinessRuleViolation,
                "constraints/generic_constraints.json",
                Some(format!("constraint[id={}]", constraint.id.0)),
                format!(
                    "Constraint \"{}\": stage-level `hydro_evaporation` at chronological \
                     stage {stage_id} is ambiguous — evaporation is per-block there \
                     (K = {k}); name a block, e.g. `hydro_evaporation(<hydro>, 0)`",
                    constraint.name
                ),
            );
        }
        _ => {}
    }
}

fn validate_storage_ref(
    constraint: &cobre_core::GenericConstraint,
    storage: StorageRef,
    k: usize,
    stage_id: i32,
    block_mode: BlockMode,
    ctx: &mut ValidationContext,
) {
    let (accessor, block_id) = match storage {
        StorageRef::Initial(b) => ("hydro_storage_initial", b),
        StorageRef::Final(b) => ("hydro_storage_final", b),
    };

    if let Some(b) = block_id
        && b >= k
    {
        ctx.add_error(
            ErrorKind::BusinessRuleViolation,
            "constraints/generic_constraints.json",
            Some(format!("constraint[id={}]", constraint.id.0)),
            format!(
                "Constraint \"{}\": per-block storage reference `{accessor}({b})` at \
                 stage {stage_id} references block {b} which does not exist at \
                 stage {stage_id} (K = {k})",
                constraint.name
            ),
        );
        return;
    }

    match block_mode {
        BlockMode::Chronological => {}
        BlockMode::Parallel => {
            if k <= 1 {
                return;
            }
            let interior = match (storage, block_id) {
                (_, None) => false,
                (StorageRef::Initial(_), Some(b)) => boundary_is_interior(b, k),
                (StorageRef::Final(_), Some(b)) => boundary_is_interior(b + 1, k),
            };
            if interior {
                let block_label = match block_id {
                    Some(b) => format!("{accessor}({b})"),
                    None => format!("{accessor}(all blocks)"),
                };
                ctx.add_error(
                    ErrorKind::BusinessRuleViolation,
                    "constraints/generic_constraints.json",
                    Some(format!("constraint[id={}]", constraint.id.0)),
                    format!(
                        "Constraint \"{}\": per-block storage reference `{block_label}` at \
                         stage {stage_id} resolves to an interior boundary, which requires \
                         chronological block mode (stage {stage_id} is parallel with {k} blocks)",
                        constraint.name
                    ),
                );
            }
        }
    }
}

/// A boundary `k` is interior iff it is neither the stage-initial anchor
/// (`k == 0`) nor the stage-final boundary (`k == K`).
fn boundary_is_interior(k: usize, num_blocks: usize) -> bool {
    k > 0 && k < num_blocks
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use cobre_core::{
        ConstraintExpression, EntityId, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
        temporal::{Block, BlockMode},
    };

    use super::super::test_support::*;
    use super::super::validate_semantic_hydro_thermal;
    use crate::ValidationEntry;
    use crate::constraints::GenericConstraintBoundsRow;
    use crate::validation::schema::ParsedData;
    use crate::validation::{ErrorKind, ValidationContext};

    fn make_blocks(k: usize) -> Vec<Block> {
        (0..k)
            .map(|i| Block {
                index: i,
                name: format!("B{i}"),
                duration_hours: 168.0,
            })
            .collect()
    }

    /// Build `ParsedData` with a single stage of `k` blocks in `block_mode` and a
    /// generic constraint whose sole term references the given variant, active on
    /// stage 0 (a bound row exists, so the per-block check runs).
    fn make_data_storage_ref(block_mode: BlockMode, k: usize, variable: VariableRef) -> ParsedData {
        let mut data = make_data(
            vec![make_hydro(1, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.stages.stages[0].block_mode = block_mode;
        data.stages.stages[0].blocks = make_blocks(k);
        data.generic_constraints = vec![GenericConstraint {
            id: EntityId::from(1),
            name: "storage_constraint".to_string(),
            description: None,
            expression: ConstraintExpression {
                terms: vec![LinearTerm::literal(1.0, variable)],
            },
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
            bound_lower_ref: None,
            bound_upper_ref: None,
        }];
        data.generic_constraint_bounds = vec![GenericConstraintBoundsRow {
            constraint_id: 1,
            stage_id: 0,
            block_id: None,
            bound_lower: Some(0.0),
            bound_upper: None,
        }];
        data
    }

    fn evaporation(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroEvaporation {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn interior_errors(data: &ParsedData) -> Vec<ValidationEntry> {
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(data, &mut ctx);
        ctx.errors()
            .iter()
            .filter(|e| {
                e.kind == ErrorKind::BusinessRuleViolation
                    && e.file
                        .to_string_lossy()
                        .contains("constraints/generic_constraints.json")
            })
            .map(|e| (*e).clone())
            .collect()
    }

    fn initial(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroStorageInitial {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    fn final_(block_id: Option<usize>) -> VariableRef {
        VariableRef::HydroStorageFinal {
            hydro_id: EntityId::from(1),
            block_id,
        }
    }

    #[test]
    fn parallel_k3_interior_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(1)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("storage_constraint") && msg.contains("interior boundary"),
            "message should name the constraint and the interior requirement, got: {msg}"
        );
        assert!(
            msg.contains("parallel with 3 blocks"),
            "message should state the parallel mode and block count, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_interior_final_rejected() {
        // Final{0} references boundary k=1, interior for K=3.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(0)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
    }

    #[test]
    fn parallel_k3_endpoint_initial_accepted() {
        // Initial{0} references boundary k=0, the S⁰ endpoint.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_endpoint_final_accepted() {
        // Final{2} references boundary k=3=K, the Sᴷ endpoint.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(2)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_none_initial_accepted() {
        // Initial{None} is the S⁰ endpoint, which exists on a parallel stage.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k3_none_final_accepted() {
        // Final{None} is the Sᴷ endpoint, which exists on a parallel stage.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn parallel_k1_all_references_accepted() {
        for variable in [
            initial(Some(0)),
            final_(Some(0)),
            initial(None),
            final_(None),
        ] {
            let data = make_data_storage_ref(BlockMode::Parallel, 1, variable);
            assert!(
                interior_errors(&data).is_empty(),
                "K=1 parallel reference must be accepted"
            );
        }
    }

    #[test]
    fn chronological_k3_all_references_accepted() {
        for variable in [
            initial(Some(0)),
            initial(Some(1)),
            initial(Some(2)),
            final_(Some(0)),
            final_(Some(1)),
            final_(Some(2)),
            initial(None),
            final_(None),
        ] {
            let data = make_data_storage_ref(BlockMode::Chronological, 3, variable);
            assert!(
                interior_errors(&data).is_empty(),
                "chronological reference must be accepted"
            );
        }
    }

    #[test]
    fn parallel_k3_out_of_range_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("block 5 which does not exist") && msg.contains("K = 3"),
            "message should be the out-of-range message, got: {msg}"
        );
    }

    #[test]
    fn parallel_k3_out_of_range_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("block 5 which does not exist") && msg.contains("K = 3"),
            "message should be the out-of-range message, got: {msg}"
        );
    }

    #[test]
    fn chronological_k3_out_of_range_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, initial(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(errors[0].message.contains("block 5 which does not exist"));
    }

    #[test]
    fn chronological_k3_out_of_range_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, final_(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(errors[0].message.contains("block 5 which does not exist"));
    }

    #[test]
    fn parallel_k3_interior_ref_on_inactive_constraint_accepted() {
        // An interior reference the constraint never activates on this stage
        // (no bound row) emits no LP row, so it must not be rejected.
        let mut data = make_data_storage_ref(BlockMode::Parallel, 3, initial(Some(1)));
        data.generic_constraint_bounds.clear();
        assert!(
            interior_errors(&data).is_empty(),
            "inactive (constraint, stage) pair must be skipped"
        );
    }

    #[test]
    fn mixed_mode_interior_ref_active_only_on_chronological_stage_accepted() {
        // Stage 0 chronological (K=3), stage 1 parallel (K=3); the constraint
        // references an interior boundary but is active only on the chronological
        // stage, so the parallel stage must not be flagged.
        let mut data = make_data_storage_ref(BlockMode::Chronological, 3, initial(Some(1)));
        data.stages.stages.push({
            let mut s = data.stages.stages[0].clone();
            s.id = 1;
            s.block_mode = BlockMode::Parallel;
            s
        });
        assert!(
            interior_errors(&data).is_empty(),
            "constraint inactive on the parallel stage must not be flagged"
        );
    }

    #[test]
    fn parallel_k3_evaporation_block_accepted() {
        // Per-block evaporation columns exist in both modes, so a valid block is fine.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(2)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_accepted_in_parallel() {
        // In parallel mode every block shares the stage endpoints, so bare
        // evaporation is the well-defined stage rate.
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_accepted_in_chronological_k1() {
        // K == 1: the single block is the whole stage, so bare is unambiguous.
        let data = make_data_storage_ref(BlockMode::Chronological, 1, evaporation(None));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_block_accepted_in_chronological() {
        let data = make_data_storage_ref(BlockMode::Chronological, 3, evaporation(Some(0)));
        assert!(interior_errors(&data).is_empty());
    }

    #[test]
    fn evaporation_bare_rejected_in_chronological_multiblock() {
        // K > 1 chronological: blocks differ, so a bare (stage-level) reference is
        // ambiguous and must name a block.
        let data = make_data_storage_ref(BlockMode::Chronological, 3, evaporation(None));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        assert!(
            errors[0].message.contains("ambiguous") && errors[0].message.contains("name a block"),
            "message should ask for an explicit block, got: {}",
            errors[0].message
        );
    }

    #[test]
    fn evaporation_out_of_range_block_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, evaporation(Some(5)));
        let errors = interior_errors(&data);
        assert_eq!(errors.len(), 1, "expected one error, got: {errors:?}");
        let msg = &errors[0].message;
        assert!(
            msg.contains("hydro_evaporation(5)") && msg.contains("block 5 which does not exist"),
            "message should be the evaporation out-of-range message, got: {msg}"
        );
    }
}
