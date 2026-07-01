//! Layer 5a — generic-constraint-vs-stage-mode semantic validation.
//!
//! Rejects per-block storage references (`HydroStorageInitial` /
//! `HydroStorageFinal`) that cannot resolve to a real storage column on a stage.

use cobre_core::{VariableRef, temporal::BlockMode};

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

/// Layer 5a — rejects per-block storage references that address a boundary a
/// stage cannot expose.
///
/// `HydroStorageInitial{b}` references boundary `k = b` (start of block `b`);
/// `HydroStorageFinal{b}` references boundary `k = b + 1` (end of block `b`).
/// A boundary is interior iff `0 < k < K`. On a `Parallel` stage with `K > 1`
/// only the two endpoints (`k = 0`, `k = K`) exist — the `storage_internal`
/// family is empty — so an interior reference resolves to a column outside the
/// storage family. A `block_id = None` term expands onto every block's boundary,
/// hitting interiors when `K > 1`, so it is rejected there too. On `K == 1` and
/// on every `Chronological` stage every boundary exists, so all references pass.
///
/// An out-of-range `block_id` (`b >= K`) is rejected on every stage: it would
/// otherwise reach `block_storage_col` and produce a bad column index at LP
/// build.
pub(super) fn check_per_block_storage_interior_reference(
    data: &ParsedData,
    ctx: &mut ValidationContext,
) {
    for stage in data.stages.stages.iter().filter(|s| s.id >= 0) {
        let k = stage.blocks.len();
        let stage_id = stage.id;
        for constraint in &data.generic_constraints {
            for term in &constraint.expression.terms {
                let Some(storage) = classify_storage_ref(&term.variable) else {
                    continue;
                };
                validate_storage_ref(constraint, storage, k, stage_id, stage.block_mode, ctx);
            }
        }
    }
}

/// Which per-block storage boundary a term references, and the block selector.
///
/// `Initial` references boundary `block_id`; `Final` references `block_id + 1`.
#[derive(Clone, Copy)]
enum StorageRef {
    Initial(Option<usize>),
    Final(Option<usize>),
}

fn classify_storage_ref(variable: &VariableRef) -> Option<StorageRef> {
    match variable {
        VariableRef::HydroStorageInitial { block_id, .. } => Some(StorageRef::Initial(*block_id)),
        VariableRef::HydroStorageFinal { block_id, .. } => Some(StorageRef::Final(*block_id)),
        _ => None,
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
                (_, None) => true,
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
        ConstraintExpression, ConstraintSense, EntityId, GenericConstraint, LinearTerm,
        SlackConfig, VariableRef,
        temporal::{Block, BlockMode},
    };

    use super::super::test_support::*;
    use super::super::validate_semantic_hydro_thermal;
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
    /// generic constraint whose sole term references the given storage variant.
    fn make_data_storage_ref(
        block_mode: BlockMode,
        k: usize,
        variable: VariableRef,
    ) -> crate::validation::schema::ParsedData {
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
            sense: ConstraintSense::LessEqual,
            slack: SlackConfig {
                enabled: false,
                penalty: None,
            },
        }];
        data
    }

    fn interior_errors(
        data: &crate::validation::schema::ParsedData,
    ) -> Vec<crate::ValidationEntry> {
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
    fn parallel_k3_none_initial_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, initial(None));
        assert_eq!(interior_errors(&data).len(), 1);
    }

    #[test]
    fn parallel_k3_none_final_rejected() {
        let data = make_data_storage_ref(BlockMode::Parallel, 3, final_(None));
        assert_eq!(interior_errors(&data).len(), 1);
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
}
