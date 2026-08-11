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
//!   asserts the full AC2/AC3 read-back once the lane is live. Never runs in
//!   CI.

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

    use cobre_sddp::lead_time::{AnticipatedResolution, LeadTime};

    /// The deck's plant carries a uniform two-week anticipation lag on a
    /// weekly calendar: one delivery stage anchors to exactly one decision
    /// stage everywhere, so the fan-out width stays at the independent-slot
    /// value.
    #[test]
    fn deck_shaped_weekly_calendar_has_max_fanout_one() {
        let stage_lengths_hours = [168.0; 6];
        let resolution = AnticipatedResolution::resolve(
            &[LeadTime::Stages(2)],
            &stage_lengths_hours,
            stage_lengths_hours.len(),
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
            &stage_lengths_hours,
            stage_lengths_hours.len(),
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

    /// AC2 + AC3: a converted case declaring one `future_anticipated_deliveries`
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
                "skipping deck smoke AC2/AC3 read-back: n_commitment={}, thermal {DECK_THERMAL_ID} \
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
