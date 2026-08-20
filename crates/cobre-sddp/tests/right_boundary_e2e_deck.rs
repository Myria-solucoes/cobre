//! Real-deck read-back of the right-boundary mechanism: a converted case
//! whose in-study decider reaches a post-study delivery must expose the
//! carrying ring slot through the setup-only terminal manifest, dated and
//! kept live for a boundary FCF to price. Two tiers:
//!
//! - `deck_independent_fanout` — CI-run, no deck: pure [`AnticipatedResolution`]
//!   structural checks over calendars shaped like the deck's own and like the
//!   fan-out geometry the setup-time reject targets.
//! - `deck_smoke` — a single test guarded on a real converted deck's presence
//!   on this machine; skips loudly and passes when the deck is absent OR when
//!   the deck's post-study-targeted ring slot is not yet live (the bridge's
//!   converted lead is currently too short for any in-study decider to reach
//!   it), and asserts the full carried-state read-back once the slot is
//!   live. Never runs in CI.

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

    /// A converted case declaring a post-horizon-reaching lead must expose a
    /// post-study-targeted ring slot in the terminal manifest, dated and kept
    /// live. A canary, not a fixed assertion: when the deck's post-study
    /// target isn't live yet (today's reality — the bridge's converted GNL
    /// lead is still the stale 168h, so no in-study decider reaches a
    /// post-study delivery), this skips loudly and passes instead of
    /// hard-failing; it starts asserting the real read-back the moment a
    /// corrected deck lands, no code change needed.
    ///
    /// The manifest position of a found slot IS its global state-dimension
    /// index: the terminal pool's manifest walks an all-enabled projection
    /// over the FULL state (`build_terminal_entity_manifest`), the identity
    /// case `right_boundary_pricing.rs`'s `freeze_terminal_template` also
    /// relies on — never a hand-rolled `commit_out`-relative offset.
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
        let ring_slot = manifest.iter().enumerate().find(|(_, slot)| {
            slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
                && slot.entity_id == DECK_THERMAL_ID
                && slot.delivery_date >= 20_260_501
        });

        let Some((state_dim, _slot)) = ring_slot else {
            eprintln!(
                "skipping deck smoke boundary read-back: thermal {DECK_THERMAL_ID} carries no \
                 ring slot dated at or after the study's post-horizon start. The deck's \
                 converted GNL lead is still the stale 168h, so no in-study decider yet reaches \
                 a post-study delivery — pending the bridge re-conversion with the faithful \
                 ~2-month lead."
            );
            return;
        };

        let out_col = state.state_to_lp_column(StateDim::new(state_dim)).get();
        let terminal_stage = setup.num_stages() - 1;
        let template = &setup.stage_ctx().templates[terminal_stage];

        assert_eq!(
            template.col_lower[out_col],
            f64::NEG_INFINITY,
            "the post-study-targeted ring slot's commit_out column must stay open, never \
             frozen [0, 0]"
        );
        assert_eq!(
            template.col_upper[out_col],
            f64::INFINITY,
            "the post-study-targeted ring slot's commit_out column must stay open, never \
             frozen [0, 0]"
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
        EntitySlot, FORMAT_VERSION, GraphManifest, ManifestEdge, ManifestNode,
        PolicyCheckpointMetadata, PolicyCutRecord, ProducerBlock, StageCutsPayload,
        write_policy_checkpoint,
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

    /// Reads back thermal 86's live post-study-targeted ring slots, asserts
    /// their mixed weekly-then-monthly slot geometry as ranges, then
    /// cross-checks the date-driven anticipated fan-out by reconciling a
    /// synthetic 2-month source against the read-back slots via
    /// [`load_boundary_cuts`] (`build_rebind` itself is `pub(crate)`,
    /// unreachable from this integration test). A canary, exactly like
    /// `deck_smoke`: skips loudly and passes when the deck is absent or no
    /// slot is yet live, asserting the full read-back only once a corrected
    /// deck lands.
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

        let manifest = setup.build_terminal_entity_manifest(&system);
        let intervals = setup.build_terminal_anticipated_delivery_intervals(&system);

        // Every ring slot is dated (the day-01 anchor of its modular delivery
        // target's stage) regardless of whether that target lands in-study or
        // post-study; a slot carries the thermal's POST-HORIZON commitment
        // only when its anchor matches a declared post-study stage — the
        // subindex >= k_max lane convention this test used to key on no
        // longer exists (every commitment-hold slot is now a ring slot).
        let post_study_anchors: std::collections::HashSet<i32> = system
            .post_study_stages()
            .map(|post_study| {
                use chrono::Datelike;
                post_study
                    .stages
                    .iter()
                    .map(|s| {
                        s.start_date.year() * 10_000
                            + i32::try_from(s.start_date.month()).unwrap_or(1) * 100
                            + 1
                    })
                    .collect()
            })
            .unwrap_or_default();

        let dated_positions: Vec<usize> = manifest
            .iter()
            .enumerate()
            .filter(|(_, slot)| {
                slot.entity_type == ENTITY_TYPE_ANTICIPATED_THERMAL_STATE
                    && slot.entity_id == DECK_THERMAL_ID
                    && post_study_anchors.contains(&slot.delivery_date)
            })
            .map(|(i, _)| i)
            .collect();

        if dated_positions.is_empty() {
            eprintln!(
                "skipping rv3 anticipated fan-out read-back: thermal {DECK_THERMAL_ID} carries \
                 no ring slot dated to a declared post-study month. Pending the \
                 decomp-jul-26-rv3 bridge regen with the faithful multi-month GNL lead."
            );
            return;
        }

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
