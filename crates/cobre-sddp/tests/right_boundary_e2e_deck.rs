//! Real-deck read-back of the right-boundary mechanism: a converted case
//! declaring `future_anticipated_deliveries` must expose its post-horizon
//! commitment lane through the setup-only terminal manifest, dated and kept
//! live for a boundary FCF to price. Two tiers:
//!
//! - `deck_independent_fanout` — CI-run, no deck: pure [`AnticipatedResolution`]
//!   structural checks over calendars shaped like the deck's own and like the
//!   fan-out geometry the setup-time reject targets.
//! - `deck_smoke` — a single test guarded on a real converted deck's presence
//!   on this machine; skips loudly and passes when the deck is absent OR when
//!   the deck's post-horizon lane is not yet live (the bridge's converted
//!   lead is currently too short for the window to survive setup), and
//!   asserts the full carried-state and fanned-lane read-back once the lane
//!   is live. Never runs in CI.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp,
    clippy::doc_markdown
)]

mod common;

mod deck_independent_fanout {
    //! `max_fanout` on calendars shaped like the real deck (uniform weekly,
    //! two-stage lead) and like the fan-out geometry `resolve_state_layout`
    //! rejects at setup time (pinned by `lead_time_fanout_rejected_at_setup`,
    //! not re-pinned here).

    use cobre_sddp::lead_time::{AnticipatedResolution, DeliveryAxis, LeadTime};

    /// The deck's plant carries a uniform two-week anticipation lag on a
    /// weekly calendar: one delivery stage anchors to exactly one decision
    /// stage everywhere, so the fan-out width stays at the independent-slot
    /// value.
    #[test]
    fn deck_shaped_weekly_calendar_has_max_fanout_one() {
        let stage_lengths_hours = [168.0; 6];
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            DeliveryAxis {
                stage_lengths_hours: &stage_lengths_hours,
                n_decision: stage_lengths_hours.len(),
                n_delivery: stage_lengths_hours.len(),
            },
        );

        assert_eq!(
            resolution.max_fanout, 1,
            "a uniform weekly calendar with a two-stage lead must resolve to the \
             independent-slot fan-out width"
        );
    }

    /// A coarse stage followed by several finer stages anchors more than one
    /// delivery stage to the same coarse decision stage, fanning out — the
    /// geometry `resolve_state_layout` rejects.
    #[test]
    fn coarse_then_fine_calendar_fans_out_beyond_one() {
        let stage_lengths_hours = [720.0, 168.0, 168.0, 168.0, 168.0];
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Time(720.0)],
            DeliveryAxis {
                stage_lengths_hours: &stage_lengths_hours,
                n_decision: stage_lengths_hours.len(),
                n_delivery: stage_lengths_hours.len(),
            },
        );

        assert!(
            resolution.max_fanout > 1,
            "a coarse-then-fine calendar must fan out beyond the independent-slot width; \
             got {}",
            resolution.max_fanout
        );
    }
}

mod deck_smoke {
    //! Setup-only terminal-manifest read-back against a real converted deck.
    //! No training solve: [`cobre_sddp::StudySetup::build_terminal_entity_manifest`]
    //! reads the terminal cut pool's projection directly.

    use std::path::PathBuf;

    use cobre_io::ENTITY_SLOT_DELIVERY_DATE_SENTINEL;
    use cobre_sddp::indexer::StateDim;

    use crate::common::fresh_setup_with;

    /// `EntityType::AnticipatedThermalState`'s raw discriminant
    /// (`schemas/policy.fbs`); mirrors `cobre_sddp::policy_export`'s own
    /// same-named constant, which is private to its module and so
    /// unreachable from here.
    const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;

    /// The GNL plant's cobre thermal id in the converted deck.
    const DECK_THERMAL_ID: i32 = 94;

    fn deck_dir() -> PathBuf {
        let home = std::env::var("HOME").expect("HOME must be set to resolve the converted deck");
        PathBuf::from(home).join("git/cobre-bridge/example/cobre-mar-26-rv2")
    }

    /// A converted case declaring one `future_anticipated_deliveries`
    /// window must expose it in the terminal manifest, dated and kept live.
    /// A canary, not a fixed assertion: when the deck's post-horizon lane
    /// isn't live yet (today's reality — the bridge's converted GNL lead is
    /// still the stale 168h, so the window is dropped at setup), this skips
    /// loudly and passes instead of hard-failing; it starts asserting the
    /// real read-back the moment a corrected deck lands, no code change
    /// needed.
    #[test]
    fn real_deck_terminal_manifest_lists_live_dated_post_horizon_lane() {
        let deck = deck_dir();
        if !deck.exists() {
            eprintln!("skipping deck smoke: {deck:?} absent");
            return;
        }

        let setup = fresh_setup_with(&deck, |_| {});
        let system =
            cobre_io::load_case(&deck).expect("load_case must succeed on the converted deck");
        let state = setup.stage_state();

        let manifest = setup.build_terminal_entity_manifest(&system);
        let lane_slot = manifest.iter().find(|slot| {
            slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
                && usize::try_from(slot.subindex).unwrap_or(usize::MAX) >= state.k_max
                && slot.entity_id == DECK_THERMAL_ID
        });
        let live_lane = state.n_commitment >= 1
            && lane_slot
                .is_some_and(|slot| slot.delivery_date != ENTITY_SLOT_DELIVERY_DATE_SENTINEL);

        if !live_lane {
            eprintln!(
                "skipping deck smoke boundary read-back: n_commitment={}, thermal {DECK_THERMAL_ID} \
                 manifest slot {}. The deck's converted GNL lead is still the stale 168h, so its \
                 future_anticipated_deliveries window is dropped at setup — pending the bridge \
                 re-conversion with the faithful ~2-month lead.",
                state.n_commitment,
                match lane_slot {
                    Some(slot) => format!("found, delivery_date={}", slot.delivery_date),
                    None => "absent".to_string(),
                },
            );
            return;
        }

        let lane_slot = lane_slot.expect("live_lane guarantees Some");
        assert_ne!(
            lane_slot.delivery_date, ENTITY_SLOT_DELIVERY_DATE_SENTINEL,
            "a live post-horizon lane must carry a real delivery date, not the sentinel"
        );
        assert!(
            lane_slot.delivery_date >= 20_260_501,
            "the delivery date must land at or after the study's post-horizon start; got {}",
            lane_slot.delivery_date
        );

        let lane = state.commit_out.end - state.n_commitment;
        let lane_out_col = state.state_to_lp_column(StateDim::new(lane)).get();
        let terminal_stage = setup.num_stages() - 1;
        let template = &setup.stage_ctx().templates[terminal_stage];

        assert_eq!(
            template.col_lower[lane_out_col],
            f64::NEG_INFINITY,
            "the post-horizon lane's commit_out column must stay open, never frozen [0, 0]"
        );
        assert_eq!(
            template.col_upper[lane_out_col],
            f64::INFINITY,
            "the post-horizon lane's commit_out column must stay open, never frozen [0, 0]"
        );
    }
}

mod anticipated_fanout_readback {
    //! rv3 mixed-granularity terminal-manifest read-back: the same
    //! deck-guarded, setup-only pattern as `deck_smoke` (see the module doc
    //! above), retargeted at the `decomp-jul-26-rv3` deck and thermal 86
    //! (SANTA CRUZ), extended with a synthetic-source fan-out cross-check
    //! against the read-back manifest via [`load_boundary_cuts`].

    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use chrono::NaiveDate;
    use cobre_io::{
        ENTITY_SLOT_DELIVERY_DATE_SENTINEL, EntitySlot, FORMAT_VERSION, GraphManifest,
        ManifestEdge, ManifestNode, PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock,
        StageCutsPayload, write_policy_checkpoint,
    };
    use cobre_sddp::load_boundary_cuts;

    use crate::common::fresh_setup_with;

    /// Mirrors `deck_smoke`'s same-named constant (`EntityType::AnticipatedThermalState`'s
    /// raw discriminant, `schemas/policy.fbs`); private to `cobre_sddp::policy_export`.
    const ENTITY_TYPE_ANTICIPATED_THERMAL_STATE: u8 = 2;

    /// SANTA CRUZ's cobre thermal id in the converted deck.
    const DECK_THERMAL_ID: i32 = 86;

    fn deck_dir() -> PathBuf {
        let home = std::env::var("HOME").expect("HOME must be set to resolve the converted deck");
        PathBuf::from(home).join("git/cobre-bridge/example/decomp-jul-26-rv3")
    }

    fn producer_block() -> ProducerBlock {
        ProducerBlock {
            completed_iterations: 0,
            final_lower_bound: 0.0,
            best_upper_bound: None,
            max_iterations: 1,
            forward_passes: 1,
            warm_start_cuts: 0,
            warm_start_counts: vec![],
            rng_seed: 0,
            total_visited_states: 0,
            training_block_mode: "parallel".to_string(),
            training_block_mode_per_stage: vec![],
            cost_scale_factor: None,
        }
    }

    /// A 1-stage chain graph manifest (node id == stage id == pool id) — the
    /// synthetic source checkpoint's own graph, unrelated to the deck's.
    fn single_stage_manifest() -> GraphManifest {
        GraphManifest {
            n_pools: 1,
            nodes: vec![ManifestNode {
                id: 0,
                stage_id: 0,
                pool_id: 0,
            }],
            edges: Vec::<ManifestEdge>::new(),
        }
    }

    fn anticipated_source_slot(thermal_id: i32, ring_slot: u32, delivery_date: i32) -> EntitySlot {
        EntitySlot {
            entity_type: ENTITY_TYPE_ANTICIPATED_THERMAL_STATE,
            entity_id: thermal_id,
            subindex: ring_slot,
            was_active: true,
            delivery_date,
        }
    }

    /// Mirrors `boundary_reconcile_defaults.rs`'s same-named helper: a single-stage,
    /// single-cut checkpoint whose one cut carries `coefficients`, one per `manifest`
    /// slot in the same order.
    fn write_checkpoint(dir: &Path, manifest: &[EntitySlot], coefficients: &[f64]) {
        let state_dimension = u32::try_from(coefficients.len()).expect("small coefficient count");
        let cut = PolicyCutRecord {
            cut_id: 0,
            slot_index: 0,
            iteration: 0,
            forward_pass_index: 0,
            intercept: 1.0,
            coefficients,
            is_active: true,
        };
        let cuts = vec![cut];
        let payload = StageCutsPayload {
            stage_id: 0,
            state_dimension,
            capacity: 1,
            warm_start_count: 0,
            cuts: &cuts,
            active_cut_indices: &[0],
            populated_count: 1,
            entity_manifest: manifest,
        };
        let metadata = PolicyCheckpointMetadata {
            format_version: FORMAT_VERSION,
            cobre_version: "0.14.0".to_string(),
            created_at: "2026-08-11T00:00:00Z".to_string(),
            num_stages: 1,
            graph_manifest: single_stage_manifest(),
            producer: producer_block(),
        };
        write_policy_checkpoint(dir, &[payload], &[], &metadata, &[]).expect("write checkpoint");
    }

    /// Reads back thermal 86's live post-horizon lane, asserts its
    /// mixed weekly-then-monthly slot geometry as ranges, then cross-checks
    /// the date-driven anticipated fan-out by reconciling a synthetic 2-month
    /// source against the read-back lane via [`load_boundary_cuts`]
    /// (`build_rebind` itself is `pub(crate)`, unreachable from this
    /// integration test). A canary, exactly like `deck_smoke`: skips loudly
    /// and passes when the deck is absent or the lane is not yet live,
    /// asserting the full read-back only once a corrected deck lands.
    #[test]
    fn rv3_thermal_86_anticipated_lane_matches_mixed_granularity_fanout() {
        let deck = deck_dir();
        if !deck.exists() {
            eprintln!(
                "skipping rv3 anticipated fan-out read-back: {deck:?} absent, pending the \
                 decomp-jul-26-rv3 bridge conversion"
            );
            return;
        }

        let setup = fresh_setup_with(&deck, |_| {});
        let system =
            cobre_io::load_case(&deck).expect("load_case must succeed on the converted deck");
        let state = setup.stage_state();

        let manifest = setup.build_terminal_entity_manifest(&system);
        let intervals = setup.build_terminal_anticipated_delivery_intervals(&system);

        let lane_positions: Vec<usize> = manifest
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
                    && usize::try_from(slot.subindex).unwrap_or(usize::MAX) >= state.k_max
                    && slot.entity_id == DECK_THERMAL_ID
            })
            .map(|(i, _)| i)
            .collect();
        let first_lane_slot = lane_positions.first().map(|&pos| &manifest[pos]);
        let live_lane = state.n_commitment >= 1
            && first_lane_slot
                .is_some_and(|slot| slot.delivery_date != ENTITY_SLOT_DELIVERY_DATE_SENTINEL);

        if !live_lane {
            eprintln!(
                "skipping rv3 anticipated fan-out read-back: n_commitment={}, thermal \
                 {DECK_THERMAL_ID} manifest slot {}. Pending the decomp-jul-26-rv3 bridge regen \
                 with the faithful multi-month GNL lead.",
                state.n_commitment,
                match first_lane_slot {
                    Some(slot) => format!("found, delivery_date={}", slot.delivery_date),
                    None => "absent".to_string(),
                },
            );
            return;
        }

        let dated_positions: Vec<usize> = lane_positions
            .iter()
            .copied()
            .filter(|&pos| manifest[pos].delivery_date != ENTITY_SLOT_DELIVERY_DATE_SENTINEL)
            .collect();

        let mut by_month: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for &pos in &dated_positions {
            by_month
                .entry(manifest[pos].delivery_date)
                .or_default()
                .push(pos);
        }
        let per_month_counts: Vec<(i32, usize)> = by_month
            .iter()
            .map(|(&anchor, positions)| (anchor, positions.len()))
            .collect();

        let count = dated_positions.len();
        assert!(
            (5..=7).contains(&count),
            "thermal {DECK_THERMAL_ID} must carry 5..=7 dated post-horizon lane slots (mixed \
             weekly-then-monthly granularity); observed {count}, grouped by month anchor as \
             {per_month_counts:?}"
        );
        assert_eq!(
            by_month.len(),
            2,
            "thermal {DECK_THERMAL_ID}'s dated slots must anchor to exactly two calendar months \
             (a nearer weekly month, then a farther monthly one); observed anchors \
             {per_month_counts:?}"
        );
        let mut anchors = by_month.keys().copied();
        let near_anchor = anchors.next().expect("by_month.len() == 2 asserted above");
        let far_anchor = anchors.next().expect("by_month.len() == 2 asserted above");
        let near_positions = &by_month[&near_anchor];
        let far_positions = &by_month[&far_anchor];

        assert!(
            (4..=6).contains(&near_positions.len()),
            "the nearer post-horizon month (weekly granularity, the September role) must carry \
             4..=6 slots; observed {} at anchor {near_anchor}",
            near_positions.len()
        );
        assert!(
            (1..=2).contains(&far_positions.len()),
            "the farther post-horizon month (monthly granularity, the October role) must carry \
             1..=2 slots; observed {} at anchor {far_anchor}",
            far_positions.len()
        );

        // build_rebind/RebindOp (crate::policy::reconcile) are pub(crate); the
        // fan-out cross-check therefore round-trips through load_boundary_cuts
        // and a synthetic on-disk checkpoint instead of calling build_rebind
        // directly. Since RebindOp's Blend/Renormalize distinction is not
        // observable through load_boundary_cuts's coefficient-only return, the
        // nonzero-vs-Zero split below is the reachable proxy: Zero is the only
        // resolution that produces an exact 0.0 for a live, dated slot fully
        // covered by the synthetic source.
        let id86_manifest: Vec<EntitySlot> = dated_positions
            .iter()
            .map(|&pos| manifest[pos].clone())
            .collect();
        let id86_intervals: Vec<Option<(NaiveDate, NaiveDate)>> =
            dated_positions.iter().map(|&pos| intervals[pos]).collect();

        let near_coefficient = 100.0_f64;
        let far_coefficient = 42.5_f64;
        let source_manifest = vec![
            anticipated_source_slot(DECK_THERMAL_ID, 0, near_anchor),
            anticipated_source_slot(DECK_THERMAL_ID, 1, far_anchor),
        ];
        let source_coefficients = vec![near_coefficient, far_coefficient];

        let tmp = tempfile::tempdir().expect("tempdir");
        write_checkpoint(tmp.path(), &source_manifest, &source_coefficients);

        let source_state_dimension =
            u32::try_from(source_coefficients.len()).expect("small coefficient count");
        let cuts = load_boundary_cuts(
            tmp.path(),
            0,
            source_state_dimension,
            &id86_manifest,
            &id86_intervals,
            None,
            1_000_000.0,
            &mut |_| {},
        )
        .expect("the synthetic 2-month source must reconcile against the read-back id-86 lane");
        assert_eq!(cuts.len(), 1);
        let fanned = &cuts[0].coefficients;
        assert_eq!(fanned.len(), id86_manifest.len());

        for (j, &pos) in dated_positions.iter().enumerate() {
            assert_ne!(
                fanned[j], 0.0,
                "a live dated lane slot covered by the synthetic source must fan out to a \
                 nonzero coefficient (Blend/Renormalize), never default to Zero: slot {j} \
                 (delivery_date={})",
                manifest[pos].delivery_date
            );
        }

        assert_eq!(
            far_positions.len(),
            1,
            "the farther month must resolve to exactly one target slot for the unit-weight \
             Blend check"
        );
        let far_j = dated_positions
            .iter()
            .position(|&pos| pos == far_positions[0])
            .expect("far_positions is a subset of dated_positions");
        assert_eq!(
            fanned[far_j].to_bits(),
            far_coefficient.to_bits(),
            "the farther month's single monthly target slot must reproduce the source \
             coefficient bit-for-bit (Blend with one unit-weight term)"
        );
    }
}
