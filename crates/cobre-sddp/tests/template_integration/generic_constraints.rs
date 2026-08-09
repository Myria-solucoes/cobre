//! `generic_constraints` section tests.

use super::*;

/// Parse `json` (a `generic_constraints.json` body) through the real
/// `cobre_io::constraints::parse_generic_constraints` path — the same loader the
/// CLI uses — into the flat `Vec<GenericConstraint>`. `name_to_id` is empty (these
/// fixtures carry no `@param` coefficients) and the line topology is empty (no
/// `line_exchange(source_bus=…, target_bus=…)` form).
fn parse_generic_from_str(json: &str) -> Vec<cobre_core::GenericConstraint> {
    use std::collections::HashMap;
    use std::io::Write;

    let mut file = tempfile::NamedTempFile::new().expect("create tempfile");
    file.write_all(json.as_bytes()).expect("write json fixture");
    cobre_io::constraints::parse_generic_constraints(
        file.path(),
        &HashMap::new(),
        &cobre_io::constraints::LineBusPairIndex::default(),
    )
    .expect("parse generic constraints")
}

/// Discharge the desugaring invariant on one twin pair: a sugared study and its
/// hand-flattened equivalent must desugar to the same flat form AND build a
/// byte-identical LP. Asserts both the parsed `Vec<GenericConstraint>` (flat-form
/// identity — implies echo identity, the echo being a pure function of the flat
/// terms) and the full `StageTemplates` Debug digest (coefficients, bounds, and
/// slacks — not just row/col counts).
fn assert_lp_byte_identical(
    sugared_json: &str,
    flat_json: &str,
    n_blks: usize,
    bounds: &cobre_core::ResolvedGenericConstraintBounds,
) {
    let sugared = parse_generic_from_str(sugared_json);
    let flat = parse_generic_from_str(flat_json);
    assert_eq!(
        sugared, flat,
        "desugared flat form must equal the hand-flattened twin"
    );

    let sugared_tpl = build_templates_for(&one_bus_system_n_blks_with_generic(
        n_blks,
        sugared,
        bounds.clone(),
    ));
    let flat_tpl = build_templates_for(&one_bus_system_n_blks_with_generic(
        n_blks,
        flat,
        bounds.clone(),
    ));
    assert_eq!(
        format!("{sugared_tpl:?}"),
        format!("{flat_tpl:?}"),
        "sugared and hand-flattened LP templates must be byte-identical"
    );
}

/// Byte-identity twin: a single-column named-expression case (the `d13`/`d54`
/// anchor shape, `thermal_generation(0)`) declared as `@fnese` and its
/// hand-flattened twin desugar identically and build the same LP.
#[test]
fn desugared_named_expression_lp_matches_hand_flattened_twin() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let sugared = r#"{
      "expressions": [ { "name": "fnese", "expression": "thermal_generation(0)" } ],
      "constraints": [
        { "id": 1, "name": "cap", "expression": "@fnese", "slack": { "enabled": false } }
      ]
    }"#;
    let flat = r#"{
      "constraints": [
        { "id": 1, "name": "cap", "expression": "thermal_generation(0)", "slack": { "enabled": false } }
      ]
    }"#;

    let id_map: HashMap<i32, usize> = [(1_i32, 0)].into_iter().collect();
    let rows = vec![(1_i32, 0_i32, None::<i32>, None, Some(10.0_f64))];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    assert_lp_byte_identical(sugared, flat, 1, &bounds);
}

/// Declaration-order invariance: three named expressions, each a single term on the
/// SAME LP column (`bus_excess` of the one bus) with DISTINCT literal coefficients
/// (1, 2, 4). Distinct coefficients give distinct `canonical_term_key`s and distinct
/// CSC entries, and with `n_blks = 1` all three land on the SAME generic-constraint
/// row. The constraint sums all three in the permuted order, so their CSC `values`
/// sequence for that column is observable and order-sensitive. Every one of the
/// `3! = 6` permutations of the declaration-and-reference order must build a
/// byte-identical `StageTemplates`.
///
/// Guarded mutation: skipping `ConstraintExpression::canonicalize` lets the
/// inline/reference order — a pure function of write order — leak into the term
/// sequence, so the six permutations produce six different CSC `values` orderings
/// and this assertion fails. `canonicalize` sorts the terms by content, collapsing
/// every write order to one; that is what this test pins.
#[test]
fn declaration_order_permutation_is_invariant_in_lp() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let defs = [
        ("e1", "1 * bus_excess(1)"),
        ("e2", "2 * bus_excess(1)"),
        ("e3", "4 * bus_excess(1)"),
    ];
    let perms: [[usize; 3]; 6] = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];

    let id_map: HashMap<i32, usize> = [(1_i32, 0)].into_iter().collect();
    let rows = vec![(1_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let digest = |perm: &[usize; 3]| -> String {
        let expressions: Vec<serde_json::Value> = perm
            .iter()
            .map(|&i| serde_json::json!({ "name": defs[i].0, "expression": defs[i].1 }))
            .collect();
        let reference = perm
            .iter()
            .map(|&i| format!("@{}", defs[i].0))
            .collect::<Vec<_>>()
            .join(" + ");
        let file = serde_json::json!({
            "expressions": expressions,
            "constraints": [
                { "id": 1, "name": "c", "expression": reference, "slack": { "enabled": false } }
            ]
        });
        let parsed = parse_generic_from_str(&file.to_string());
        let system = one_bus_system_n_blks_with_generic(1, parsed, bounds.clone());
        format!("{:?}", build_templates_for(&system))
    };

    let first = digest(&perms[0]);
    for perm in &perms[1..] {
        assert_eq!(
            digest(perm),
            first,
            "declaration-order permutation {perm:?} changed the LP"
        );
    }
}

#[test]
fn generic_constraints_zero_does_not_change_layout() {
    let system = one_bus_system_n_blks(1);
    let templates = build_templates_for(&system);
    let t = &templates[0];
    assert!(t.num_cols > 0);
    assert!(t.num_rows > 0);
    // A second build must be bit-for-bit identical (determinism).
    let templates2 = build_templates_for(&system);
    assert_eq!(
        templates2[0].num_cols, t.num_cols,
        "second build must not change num_cols"
    );
    assert_eq!(
        templates2[0].num_rows, t.num_rows,
        "second build must not change num_rows"
    );
}

#[test]
fn generic_constraint_no_slack_block_id_none_3_blocks_collapses() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
    let rows = vec![(10_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(10, false);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (block-independent block_id=None collapses to a single row)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

#[test]
fn generic_constraint_no_slack_block_id_none_3_blocks_block_level_per_block() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(11_i32, 0)].into_iter().collect();
    let rows = vec![(11_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(11, false, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + n_blks,
        "num_rows must increase by n_blks={n_blks} (block-level expression keeps one row per block)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

#[test]
fn generic_constraint_le_slack_enabled_2_blocks_collapses() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
    let rows = vec![(20_i32, 0_i32, None::<i32>, None, Some(300.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(20, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 1,
        "num_cols must increase by 1 (one `<=` slack on the single collapsed row)"
    );
}

#[test]
fn generic_constraint_le_slack_enabled_2_blocks_block_level_per_block() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(21_i32, 0)].into_iter().collect();
    let rows = vec![(21_i32, 0_i32, None::<i32>, None, Some(300.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(21, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (one slack per per-block row for `<=`)"
    );
}

#[test]
fn generic_constraint_degenerate_band_two_slacks_collapses() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
    let rows = vec![(30_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(100.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(30, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (plus+minus slacks on the single collapsed row)"
    );
}

#[test]
fn generic_constraint_two_sided_two_slacks_collapses() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(70_i32, 0)].into_iter().collect();
    let rows = vec![(70_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(70, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by 1 (collapsed stage-level row)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 2,
        "num_cols must increase by 2 (plus+minus slacks on the single collapsed row)"
    );
}

#[test]
fn generic_constraint_degenerate_band_two_slacks_block_level_per_block() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(31_i32, 0)].into_iter().collect();
    let rows = vec![(31_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(100.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(31, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 4,
        "num_cols must increase by 4 (two slacks per per-block row for `==`)"
    );
}

#[test]
fn generic_constraint_two_sided_two_slacks_block_level_per_block() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(71_i32, 0)].into_iter().collect();
    let rows = vec![(71_i32, 0_i32, None::<i32>, Some(100.0_f64), Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint_with_expr(71, true, block_level_excess_expr(1));
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 2,
        "num_rows must increase by 2 (block-level keeps one row per block)"
    );
    assert_eq!(
        t_cols,
        baseline_cols + 4,
        "num_cols must increase by 4 (two slacks per per-block row for range)"
    );
}

#[test]
fn generic_constraint_collapsed_slack_priced_by_total_stage_hours() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let block_hours = 720.0_f64; // one_bus_system_n_blks uses 720 h per block
    let penalty = 5000.0_f64; // make_constraint sets penalty = 5000 when slack enabled
    let total_stage_hours = block_hours * (n_blks as f64);

    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(60_i32, 0)].into_iter().collect();
    let rows = vec![(60_i32, 0_i32, None::<i32>, None, Some(500.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(60, true);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_cols,
        baseline_cols + 1,
        "collapsed `<=` row must add exactly one slack column"
    );

    let slack_col = baseline_cols; // new slack appended after the baseline cols
    let generic_row = t.num_rows - 1; // the single collapsed generic row is last

    let expected_obj = penalty * total_stage_hours / COST_SCALE_FACTOR;
    let per_block_sum = penalty * block_hours * (n_blks as f64) / COST_SCALE_FACTOR;
    assert!(
        (expected_obj - per_block_sum).abs() < 1e-9,
        "penalty conservation identity must hold: {expected_obj} vs {per_block_sum}"
    );
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-9,
        "collapsed slack objective must be penalty × total_stage_hours / scale = {expected_obj}, \
         got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    let total: f64 = entries.iter().sum();
    assert!(
        (total - (-1.0)).abs() < f64::EPSILON,
        "collapsed `<=` slack must have -1.0 at the generic row, got {total}"
    );
}

#[test]
fn generic_constraint_specific_block_id_generates_one_row() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;
    let baseline_cols = build_templates_for(&baseline_system)[0].num_cols;

    let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
    let rows = vec![(40_i32, 0_i32, Some(1_i32), None, Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let constraint = make_constraint(40, false);
    let system = one_bus_system_n_blks_with_generic(n_blks, vec![constraint], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;
    let t_cols = build_templates_for(&system)[0].num_cols;

    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "num_rows must increase by exactly 1 (only the specified block)"
    );
    assert_eq!(
        t_cols, baseline_cols,
        "num_cols must be unchanged (no slack columns)"
    );
}

#[test]
fn generic_constraint_inactive_does_not_contribute_rows() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use std::collections::HashMap;

    let n_blks = 2_usize;
    let baseline_system = one_bus_system_n_blks(n_blks);
    let baseline_rows = build_templates_for(&baseline_system)[0].num_rows;

    // Constraint 51 has no bounds row → inactive.
    let id_map: HashMap<i32, usize> = [(50_i32, 0), (51_i32, 1)].into_iter().collect();
    let rows = vec![(50_i32, 0_i32, None::<i32>, None, Some(400.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let c_active = make_constraint(50, false);
    let c_inactive = make_constraint(51, false);
    let system =
        one_bus_system_n_blks_with_generic(n_blks, vec![c_active, c_inactive], generic_bounds);
    let t_rows = build_templates_for(&system)[0].num_rows;

    // The active constraint's trivial expression collapses to one row (without the
    // collapse it would be baseline_rows + n_blks); the test's point is that the
    // inactive constraint stays absent under either representation.
    assert_eq!(
        t_rows,
        baseline_rows + 1,
        "only the active constraint must contribute rows (collapsed to one)"
    );
}

#[test]
fn generic_constraint_thermal_le_row_bounds_and_csc_entry() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(10),
        name: "gc_thermal_le".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(10_i32, 0)].into_iter().collect();
    let rows = vec![(10_i32, 0_i32, None::<i32>, None, Some(50.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1: theta=0, thermal col 0/block 0 = col 1,
    // load_balance row 0 → generic row at row 1.
    let thermal_col = 1_usize;
    let generic_row = 1_usize;

    assert!(
        t.row_lower[generic_row].is_infinite() && t.row_lower[generic_row] < 0.0,
        "row_lower must be -INF for an upper-only row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 50.0).abs() < f64::EPSILON,
        "row_upper must be 50.0, got {}",
        t.row_upper[generic_row]
    );

    let entries = csc_entries_at(t, thermal_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={thermal_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 total at thermal col / generic row, got {total}"
    );
}

#[test]
fn generic_constraint_thermal_le_slack_column_and_csc_entry() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let block_hours = 744.0_f64;
    let penalty = 5000.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(20),
        name: "gc_thermal_le_slack".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };

    let id_map: HashMap<i32, usize> = [(20_i32, 0)].into_iter().collect();
    let rows = vec![(20_i32, 0_i32, None::<i32>, None, Some(50.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1, 1 slack col:
    //   withdrawal_slack_start = col_evap_start + 0 evap cols
    //   = col_generation_start + 0 generation cols = inflow_slack_end
    //   = excess_end = 1(theta)+1(thermal)+1(deficit)+1(excess) = 4
    //   col_generic_slack_start = withdrawal_slack_start + n_h(=0) = 4
    //   slack_plus_col = 4
    let slack_col = 4_usize;
    let generic_row = 1_usize;

    assert!(
        t.col_lower[slack_col].abs() < f64::EPSILON,
        "slack col_lower must be 0.0, got {}",
        t.col_lower[slack_col]
    );
    assert!(
        t.col_upper[slack_col].is_infinite() && t.col_upper[slack_col] > 0.0,
        "slack col_upper must be +INF, got {}",
        t.col_upper[slack_col]
    );

    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "slack objective must be {expected_obj}, got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={slack_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - (-1.0)).abs() < f64::EPSILON,
        "expected -1.0 at slack col / generic row for an upper-only row, got {total}"
    );
}

#[test]
fn generic_constraint_thermal_ge_row_bounds() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(30),
        name: "gc_thermal_ge".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(30_i32, 0)].into_iter().collect();
    let rows = vec![(30_i32, 0_i32, None::<i32>, Some(10.0_f64), None)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let generic_row = 1_usize;
    assert!(
        (t.row_lower[generic_row] - 10.0).abs() < f64::EPSILON,
        "row_lower must be 10.0 for a lower-only row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        t.row_upper[generic_row].is_infinite() && t.row_upper[generic_row] > 0.0,
        "row_upper must be +INF for a lower-only row, got {}",
        t.row_upper[generic_row]
    );
}

/// A lower-only row (`bound_lower` present, `bound_upper` absent) on a
/// slack-enabled constraint allocates exactly ONE slack column with `+1.0` at
/// the generic row — the mirror image of the upper-only `-1.0` case above.
#[test]
fn generic_constraint_thermal_ge_slack_column_and_csc_entry() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let block_hours = 744.0_f64;
    let penalty = 5000.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(35),
        name: "gc_thermal_ge_slack".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };

    let id_map: HashMap<i32, usize> = [(35_i32, 0)].into_iter().collect();
    let rows = vec![(35_i32, 0_i32, None::<i32>, Some(10.0_f64), None)];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Same layout as the upper-only slack test: theta=0, thermal=1, deficit=1,
    // excess=1 → col_generic_slack_start = 4, slack_plus_col = 4.
    let slack_col = 4_usize;
    let generic_row = 1_usize;

    assert_eq!(
        t.row_lower[generic_row], 10.0,
        "row_lower must be 10.0 for a lower-only row"
    );
    assert!(
        t.row_upper[generic_row].is_infinite() && t.row_upper[generic_row] > 0.0,
        "row_upper must be +INF for a lower-only row"
    );

    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_col] - expected_obj).abs() < 1e-12,
        "slack objective must be {expected_obj}, got {}",
        t.objective[slack_col]
    );

    let entries = csc_entries_at(t, slack_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={slack_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 at slack col / generic row for a lower-only row, got {total}"
    );
}

#[test]
fn generic_constraint_two_sided_thermal_row_bounds_and_csc_entry() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);

    let constraint = GenericConstraint {
        id: EntityId(72),
        name: "gc_thermal_range".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![LinearTerm::literal(
                1.0,
                VariableRef::ThermalGeneration {
                    thermal_id: thermal_entity_id,
                    block_id: None,
                },
            )],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(72_i32, 0)].into_iter().collect();
    let rows = vec![(72_i32, 0_i32, None::<i32>, Some(5.0_f64), Some(20.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    // Layout for N=0, T=1, B=1, K=1 (same as generic_constraint_thermal_le_row_bounds_and_csc_entry).
    let thermal_col = 1_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 5.0).abs() < f64::EPSILON,
        "row_lower must be 5.0 (bound_lower) for a two-sided row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 20.0).abs() < f64::EPSILON,
        "row_upper must be 20.0 (bound_upper) for a two-sided row, got {}",
        t.row_upper[generic_row]
    );

    let entries = csc_entries_at(t, thermal_col, generic_row);
    assert!(
        !entries.is_empty(),
        "no CSC entry found at (col={thermal_col}, row={generic_row})"
    );
    let total: f64 = entries.iter().sum();
    assert!(
        (total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 total at thermal col / generic row, got {total}"
    );
}

#[test]
fn generic_constraint_degenerate_band_thermal_two_slacks() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{ConstraintExpression, GenericConstraint, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let block_hours = 744.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(40),
        name: "gc_thermal_eq_slack".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };

    let id_map: HashMap<i32, usize> = [(40_i32, 0)].into_iter().collect();
    let rows = vec![(40_i32, 0_i32, None::<i32>, Some(80.0_f64), Some(80.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let slack_plus_col = 4_usize;
    let slack_minus_col = 5_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_lower must be 80.0 for a degenerate band, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_upper must be 80.0 for a degenerate band, got {}",
        t.row_upper[generic_row]
    );

    // Two slack columns: num_cols baseline is 4 (theta=0, thermal=1, deficit=1, excess=1,
    // withdrawal_slack=0), with 2 slacks → num_cols = 6.
    assert_eq!(t.num_cols, 6, "num_cols must be 6 with 2 slack columns");

    assert!(
        t.col_upper[slack_plus_col].is_infinite() && t.col_upper[slack_plus_col] > 0.0,
        "plus slack col_upper must be +INF"
    );
    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_plus_col] - expected_obj).abs() < 1e-12,
        "plus slack objective must be {expected_obj}, got {}",
        t.objective[slack_plus_col]
    );
    let plus_entries = csc_entries_at(t, slack_plus_col, generic_row);
    assert!(
        !plus_entries.is_empty(),
        "no CSC entry at plus slack col / generic row"
    );
    let plus_total: f64 = plus_entries.iter().sum();
    assert!(
        (plus_total - 1.0).abs() < f64::EPSILON,
        "plus slack CSC must be +1.0 for a degenerate band, got {plus_total}"
    );

    assert!(
        t.col_upper[slack_minus_col].is_infinite() && t.col_upper[slack_minus_col] > 0.0,
        "minus slack col_upper must be +INF"
    );
    assert!(
        (t.objective[slack_minus_col] - expected_obj).abs() < 1e-12,
        "minus slack objective must be {expected_obj}"
    );
    let minus_entries = csc_entries_at(t, slack_minus_col, generic_row);
    assert!(
        !minus_entries.is_empty(),
        "no CSC entry at minus slack col / generic row"
    );
    let minus_total: f64 = minus_entries.iter().sum();
    assert!(
        (minus_total - (-1.0)).abs() < f64::EPSILON,
        "minus slack CSC must be -1.0 for a degenerate band, got {minus_total}"
    );
}

#[test]
fn generic_constraint_two_sided_thermal_two_slacks() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{ConstraintExpression, GenericConstraint, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let block_hours = 744.0_f64;

    let constraint = GenericConstraint {
        id: EntityId(73),
        name: "gc_thermal_range_slack".to_string(),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };

    let id_map: HashMap<i32, usize> = [(73_i32, 0)].into_iter().collect();
    let rows = vec![(73_i32, 0_i32, None::<i32>, Some(30.0_f64), Some(80.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);
    let t = &build_templates_for(&system)[0];

    let slack_plus_col = 4_usize;
    let slack_minus_col = 5_usize;
    let generic_row = 1_usize;

    assert!(
        (t.row_lower[generic_row] - 30.0).abs() < f64::EPSILON,
        "row_lower must be 30.0 for a two-sided row, got {}",
        t.row_lower[generic_row]
    );
    assert!(
        (t.row_upper[generic_row] - 80.0).abs() < f64::EPSILON,
        "row_upper must be 80.0 for a two-sided row, got {}",
        t.row_upper[generic_row]
    );

    // Two slack columns: num_cols baseline is 4 (theta=0, thermal=1, deficit=1, excess=1,
    // withdrawal_slack=0), with 2 slacks → num_cols = 6.
    assert_eq!(t.num_cols, 6, "num_cols must be 6 with 2 slack columns");

    assert!(
        t.col_upper[slack_plus_col].is_infinite() && t.col_upper[slack_plus_col] > 0.0,
        "plus slack col_upper must be +INF"
    );
    let expected_obj = penalty * block_hours / COST_SCALE_FACTOR;
    assert!(
        (t.objective[slack_plus_col] - expected_obj).abs() < 1e-12,
        "plus slack objective must be {expected_obj}, got {}",
        t.objective[slack_plus_col]
    );
    let plus_entries = csc_entries_at(t, slack_plus_col, generic_row);
    assert!(
        !plus_entries.is_empty(),
        "no CSC entry at plus slack col / generic row"
    );
    let plus_total: f64 = plus_entries.iter().sum();
    assert!(
        (plus_total - 1.0).abs() < f64::EPSILON,
        "plus slack CSC must be +1.0 for a two-sided row, got {plus_total}"
    );

    assert!(
        t.col_upper[slack_minus_col].is_infinite() && t.col_upper[slack_minus_col] > 0.0,
        "minus slack col_upper must be +INF"
    );
    assert!(
        (t.objective[slack_minus_col] - expected_obj).abs() < 1e-12,
        "minus slack objective must be {expected_obj}"
    );
    let minus_entries = csc_entries_at(t, slack_minus_col, generic_row);
    assert!(
        !minus_entries.is_empty(),
        "no CSC entry at minus slack col / generic row"
    );
    let minus_total: f64 = minus_entries.iter().sum();
    assert!(
        (minus_total - (-1.0)).abs() < f64::EPSILON,
        "minus slack CSC must be -1.0 for a two-sided row, got {minus_total}"
    );
}

/// A degenerate two-sided band (`bound_lower == bound_upper`) is still
/// two-sided by the null-pattern (both endpoints present), so it must get an
/// equality row (`row_lower == row_upper`) AND the full two-slack allocation
/// — the shape derivation must not special-case equal endpoints down to a
/// one-sided row.
#[test]
fn generic_constraint_degenerate_band_gets_equality_row_and_two_slacks() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{ConstraintExpression, GenericConstraint, SlackConfig};
    use std::collections::HashMap;

    let thermal_entity_id = EntityId(2);
    let penalty = 5000.0_f64;
    let id = 74;

    let constraint = GenericConstraint {
        id: EntityId(id),
        name: format!("gc_{id}"),
        description: None,
        expression: ConstraintExpression { terms: vec![] },
        slack: SlackConfig {
            enabled: true,
            penalty: Some(penalty),
        },
    };
    let id_map: HashMap<i32, usize> = [(id, 0_usize)].into_iter().collect();
    let rows = vec![(id, 0_i32, None::<i32>, Some(10.0_f64), Some(10.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());
    let system = one_bus_one_thermal_system(thermal_entity_id, vec![constraint], generic_bounds);

    let t = &build_templates_for(&system)[0];
    let generic_row = 1_usize;

    assert_eq!(
        t.row_lower[generic_row], 10.0,
        "degenerate band row_lower must equal the shared endpoint value"
    );
    assert_eq!(
        t.row_upper[generic_row], 10.0,
        "degenerate band row_upper must equal the shared endpoint value"
    );
    assert_eq!(
        t.num_cols, 6,
        "degenerate two-sided band must still allocate BOTH slack columns"
    );
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_two_hydros_sum_csc_entries() {
    use chrono::NaiveDate;
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::entities::hydro::{HydroGenerationModel, HydroPenalties};
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, BlockMode, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };
    use cobre_core::{
        ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let h1_id = EntityId(5);
    let h2_id = EntityId(10);
    let prod_h1 = 2.5_f64;
    let prod_h2 = 3.0_f64;

    let hydros = vec![
        make_hydro(
            h1_id,
            HydroSpec {
                name: format!("H{}", h1_id.0),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                ..Default::default()
            },
        ),
        make_hydro(
            h2_id,
            HydroSpec {
                name: format!("H{}", h2_id.0),
                operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
                bus_id: EntityId(1),
                downstream_id: None,
                entry_stage_id: None,
                exit_stage_id: None,
                min_storage_hm3: 0.0,
                max_storage_hm3: 200.0,
                min_outflow_m3s: 0.0,
                max_outflow_m3s: None,
                generation_model: HydroGenerationModel::ConstantProductivity,
                min_turbined_m3s: 0.0,
                max_turbined_m3s: 100.0,
                specific_productivity_mw_per_m3s_per_m: None,
                min_generation_mw: 0.0,
                max_generation_mw: 250.0,
                tailrace: None,
                hydraulic_losses: None,
                efficiency: None,
                evaporation_coefficients_mm: None,
                evaporation_reference_volumes_hm3: None,
                diversion: None,
                filling: None,
                penalties: HydroPenalties {
                    spillage_cost: 0.01,
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
                    inflow_nonnegativity_cost: 1000.0,
                },
                ..Default::default()
            },
        ),
    ];

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: chrono::NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks: vec![Block {
                index: 0,
                name: "BLK0".to_string(),
                duration_hours: 744.0,
            }],
            block_mode: BlockMode::Parallel,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        },
    );

    let inflow_models = vec![
        InflowModel {
            hydro_id: h1_id,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 0.0,
            annual: None,
        },
        InflowModel {
            hydro_id: h2_id,
            stage_id: 0,
            mean_m3s: 50.0,
            std_m3s: 0.0,
            ar_coefficients: vec![],
            residual_std_ratio: 0.0,
            annual: None,
        },
    ];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    // Generic constraint: hydro_generation(H1) + hydro_generation(H2) <= 200
    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_sum_gen".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: h1_id,
                        block_id: None,
                        bus_id: None,
                    },
                ),
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroGeneration {
                        hydro_id: h2_id,
                        block_id: None,
                        bus_id: None,
                    },
                ),
            ],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(100_i32, 0_i32, None::<i32>, None, Some(200.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 2,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 2,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let system = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(hydros)
        .stages(vec![stage])
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties)
        .generic_constraints(vec![constraint])
        .resolved_generic_bounds(generic_bounds)
        .build()
        .expect("two_hydro_system: valid");

    // Productivities are supplied explicitly (prod_h1=2.5 at index 0, prod_h2=3.0
    // at index 1) so the generic-row LP coefficients equal the productivities.
    let pm = production_set(&[prod_h1, prod_h2], 1);
    let t = &build_stage_templates_resolving_layout(
        &system,
        no_penalty_config(),
        &PrecomputedPar::default(),
        &PrecomputedNormal::default(),
        &pm,
        &default_evaporation(&system),
        &ResolvedParameters::default(),
    )
    .expect("two_hydro_system: valid")
    .templates[0];

    // For constant-productivity hydros, HydroGeneration maps to the turbine column
    // with multiplier = productivity; the generic constraint row is the last row.
    let generic_row = t.num_rows - 1;

    // N=2: storage=[0..2], z_inflow=[2..4], storage_in=[4..6], theta=6, turbine_start=7.
    let turbine_h1_col = 7_usize;
    let turbine_h2_col = 8_usize;

    let entries_h1 = csc_entries_at(t, turbine_h1_col, generic_row);
    assert!(
        !entries_h1.is_empty(),
        "no CSC entry found for H1 turbine at generic constraint row"
    );
    let total_h1: f64 = entries_h1.iter().sum();
    assert!(
        (total_h1 - prod_h1).abs() < f64::EPSILON,
        "expected coefficient {prod_h1} for H1, got {total_h1}"
    );

    let entries_h2 = csc_entries_at(t, turbine_h2_col, generic_row);
    assert!(
        !entries_h2.is_empty(),
        "no CSC entry found for H2 turbine at generic constraint row"
    );
    let total_h2: f64 = entries_h2.iter().sum();
    assert!(
        (total_h2 - prod_h2).abs() < f64::EPSILON,
        "expected coefficient {prod_h2} for H2, got {total_h2}"
    );
}

/// Build a single-hydro, one-bus, `n_blks`-block, one-stage system under
/// `block_mode` with `storage` state on, optionally carrying `constraint` +
/// `bounds`. Column layout (N=1, no thermal/FPHA/evap): storage=[0,1),
/// z_inflow=[1,2), storage_in=[2,3), theta=3, control region from col 4.
#[allow(clippy::cast_possible_wrap)]
fn one_hydro_system(
    n_blks: usize,
    block_mode: cobre_core::BlockMode,
    constraint: Option<cobre_core::GenericConstraint>,
    bounds: cobre_core::ResolvedGenericConstraintBounds,
) -> cobre_core::System {
    use chrono::NaiveDate;
    use cobre_core::HydroGenerationModel;
    use cobre_core::scenario::{InflowModel, LoadModel};
    use cobre_core::temporal::{
        Block, NoiseMethod, ScenarioSourceConfig, StageRiskConfig, StageStateConfig,
    };

    let hydro_id = EntityId(5);

    let hydro = make_hydro(
        hydro_id,
        HydroSpec {
            name: "H1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            bus_id: EntityId(1),
            min_storage_hm3: 0.0,
            max_storage_hm3: 200.0,
            max_turbined_m3s: 100.0,
            generation_model: HydroGenerationModel::ConstantProductivity,
            max_generation_mw: 250.0,
            ..Default::default()
        },
    );

    let bus = make_bus(
        EntityId(1),
        BusSpec {
            name: "B1".to_string(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            deficit_segments: vec![DeficitSegment {
                depth_mw: None,
                cost_per_mwh: 500.0,
            }],
            excess_cost: 0.0,
            ..Default::default()
        },
    );

    let blocks: Vec<Block> = (0..n_blks)
        .map(|i| Block {
            index: i,
            name: format!("BLK{i}"),
            duration_hours: 720.0,
        })
        .collect();

    let stage = make_stage(
        0,
        StageSpec {
            start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            end_date: NaiveDate::from_ymd_opt(2024, 2, 1).unwrap(),
            season_id: None,
            blocks,
            block_mode,
            state_config: StageStateConfig {
                storage: true,
                inflow_lags: false,
            },
            risk_config: StageRiskConfig::Expectation,
            scenario_config: ScenarioSourceConfig {
                branching_factor: 1,
                noise_method: NoiseMethod::Saa,
            },
            ..Default::default()
        },
    );

    let inflow_models = vec![InflowModel {
        hydro_id,
        stage_id: 0,
        mean_m3s: 50.0,
        std_m3s: 0.0,
        ar_coefficients: vec![],
        residual_std_ratio: 0.0,
        annual: None,
    }];

    let load_models = vec![LoadModel {
        bus_id: EntityId(1),
        stage_id: 0,
        mean_mw: 100.0,
        std_mw: 0.0,
    }];

    let resolved_bounds = ResolvedBounds::new(
        &BoundsCountsSpec {
            n_hydros: 1,
            n_thermals: 0,
            n_lines: 0,
            n_pumping: 0,
            n_contracts: 0,
            n_stages: 1,
            k_max: 0,
        },
        &BoundsDefaults {
            hydro: default_hydro_bounds(),
            hydro_block: default_hydro_block_bounds(),
            thermal: ThermalStageBounds { cost_per_mwh: 0.0 },
            thermal_block: ThermalBlockBounds {
                min_generation_mw: 0.0,
                max_generation_mw: 0.0,
            },
            line_block: LineBlockBounds {
                direct_mw: 0.0,
                reverse_mw: 0.0,
            },
            pumping_block: PumpingBlockBounds {
                min_flow_m3s: 0.0,
                max_flow_m3s: 0.0,
            },
            contract_block: ContractBlockBounds {
                min_mw: 0.0,
                max_mw: 0.0,
                price_per_mwh: 0.0,
            },
        },
    );
    let penalties = ResolvedPenalties::new(
        &PenaltiesCountsSpec {
            n_hydros: 1,
            n_buses: 1,
            n_lines: 0,
            n_ncs: 0,
            n_stages: 1,
        },
        &PenaltiesDefaults {
            hydro: default_hydro_penalties(),
            bus: BusStagePenalties { excess_cost: 0.0 },
            line: LineStagePenalties { exchange_cost: 0.0 },
            ncs: NcsStagePenalties {
                curtailment_cost: 0.0,
            },
        },
    );

    let mut builder = SystemBuilder::new()
        .buses(vec![bus])
        .hydros(vec![hydro])
        .stages(vec![stage])
        .inflow_models(inflow_models)
        .load_models(load_models)
        .bounds(resolved_bounds)
        .penalties(penalties);
    if let Some(c) = constraint {
        builder = builder
            .generic_constraints(vec![c])
            .resolved_generic_bounds(bounds);
    }
    builder.build().expect("one_hydro_system: valid")
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_chronological_stage_net_storage_one_row() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        BlockMode, ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let hydro_id = EntityId(5);

    let baseline = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        None,
        ResolvedGenericConstraintBounds::new(&HashMap::new(), std::iter::empty()),
    );
    let baseline_rows = build_templates_for(&baseline)[0].num_rows;

    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_ramp".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorageFinal {
                        hydro_id,
                        block_id: None,
                    },
                ),
                LinearTerm::literal(
                    -1.0,
                    VariableRef::HydroStorageInitial {
                        hydro_id,
                        block_id: None,
                    },
                ),
            ],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(100_i32, 0_i32, None::<i32>, None, Some(50.0_f64))];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        Some(constraint),
        generic_bounds,
    );
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_rows,
        baseline_rows + 1,
        "stage net storage (block_id=None) collapses to exactly one row"
    );

    // S⁰ = storage_in.start + 0 = 2; Sᴷ = storage.start + 0 = 0.
    let generic_row = baseline_rows;
    let final_total: f64 = csc_entries_at(t, 0, generic_row).iter().sum();
    assert!(
        (final_total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 on stage-final Sᴷ (col 0), got {final_total}"
    );
    let initial_total: f64 = csc_entries_at(t, 2, generic_row).iter().sum();
    assert!(
        (initial_total + 1.0).abs() < f64::EPSILON,
        "expected -1.0 on stage-initial S⁰ (col 2), got {initial_total}"
    );
}

#[test]
#[allow(clippy::cast_possible_wrap)]
fn generic_constraint_chronological_specific_block_ramp_one_row() {
    use cobre_core::ResolvedGenericConstraintBounds;
    use cobre_core::{
        BlockMode, ConstraintExpression, GenericConstraint, LinearTerm, SlackConfig, VariableRef,
    };
    use std::collections::HashMap;

    let n_blks = 3_usize;
    let hydro_id = EntityId(5);
    let target_block = 1_usize;

    let baseline = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        None,
        ResolvedGenericConstraintBounds::new(&HashMap::new(), std::iter::empty()),
    );
    let baseline_rows = build_templates_for(&baseline)[0].num_rows;

    let constraint = GenericConstraint {
        id: EntityId(100),
        name: "gc_ramp_b1".to_string(),
        description: None,
        expression: ConstraintExpression {
            terms: vec![
                LinearTerm::literal(
                    1.0,
                    VariableRef::HydroStorageFinal {
                        hydro_id,
                        block_id: Some(target_block),
                    },
                ),
                LinearTerm::literal(
                    -1.0,
                    VariableRef::HydroStorageInitial {
                        hydro_id,
                        block_id: Some(target_block),
                    },
                ),
            ],
        },
        slack: SlackConfig {
            enabled: false,
            penalty: None,
        },
    };

    let id_map: HashMap<i32, usize> = [(100_i32, 0)].into_iter().collect();
    let rows = vec![(
        100_i32,
        0_i32,
        Some(target_block as i32),
        None,
        Some(50.0_f64),
    )];
    let generic_bounds = ResolvedGenericConstraintBounds::new(&id_map, rows.into_iter());

    let system = one_hydro_system(
        n_blks,
        BlockMode::Chronological,
        Some(constraint),
        generic_bounds,
    );
    let t = &build_templates_for(&system)[0];

    assert_eq!(
        t.num_rows,
        baseline_rows + 1,
        "block_id = Some(b) ramp must produce exactly one row"
    );

    // block 1 boundaries: initial k=1 = S¹ = 4, final k=2 = S² = 5.
    let generic_row = baseline_rows;
    let final_total: f64 = csc_entries_at(t, 5, generic_row).iter().sum();
    assert!(
        (final_total - 1.0).abs() < f64::EPSILON,
        "expected +1.0 on S² (col 5), got {final_total}"
    );
    let initial_total: f64 = csc_entries_at(t, 4, generic_row).iter().sum();
    assert!(
        (initial_total + 1.0).abs() < f64::EPSILON,
        "expected -1.0 on S¹ (col 4), got {initial_total}"
    );
}
