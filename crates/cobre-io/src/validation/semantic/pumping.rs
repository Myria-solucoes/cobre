//! Layer 5a — pumping-station-domain semantic validation.
//!
//! Referential integrity is discharged in Layer 3 (`check_pumping_references`);
//! this module covers the remaining invariant: a station must move water between
//! two *distinct* reservoirs.

use super::super::{ErrorKind, ValidationContext, schema::ParsedData};

/// Rejects pumping stations whose source and destination hydros are identical.
///
/// A `source_hydro_id == destination_hydro_id` station passes referential
/// validation (the ID resolves) yet models a degenerate self-transfer: a
/// self-cancelling `+τ`/`−τ` pair on one water-balance row while still drawing
/// power — a silent modeling error, not a dangling reference. The IDs are valid,
/// so the kind is [`ErrorKind::InvalidValue`], not `InvalidReference`.
pub(super) fn check_pumping_semantics(data: &ParsedData, ctx: &mut ValidationContext) {
    for station in &data.pumping_stations {
        if station.source_hydro_id == station.destination_hydro_id {
            let entity_str = format!("PumpingStation {}", station.id.0);
            ctx.add_error(
                ErrorKind::InvalidValue,
                "system/pumping_stations.json",
                Some(&entity_str),
                format!(
                    "{entity_str} has source_hydro_id == destination_hydro_id ({}): a station must transfer water between two distinct reservoirs",
                    station.source_hydro_id.0
                ),
            );
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic, clippy::doc_markdown)]
mod tests {
    use cobre_core::{EntityId, entities::PumpingStation};

    use super::super::test_support::{make_data, make_hydro, make_stages};
    use super::super::validate_semantic_hydro_thermal;
    use crate::validation::{ErrorKind, ValidationContext};

    /// Build a `PumpingStation` with the given bus and source/destination hydros.
    fn make_pumping(id: i32, bus_id: i32, src_hydro: i32, dst_hydro: i32) -> PumpingStation {
        PumpingStation {
            id: EntityId::from(id),
            name: format!("Pump_{id}"),
            bus_id: EntityId::from(bus_id),
            source_hydro_id: EntityId::from(src_hydro),
            destination_hydro_id: EntityId::from(dst_hydro),
            entry_stage_id: None,
            exit_stage_id: None,
            consumption_mw_per_m3s: 0.5,
            min_flow_m3s: 0.0,
            max_flow_m3s: 100.0,
        }
    }

    /// A station whose source and destination hydros are identical (hydro 10)
    /// yields exactly one `InvalidValue` mentioning the degenerate condition and
    /// the shared hydro ID.
    #[test]
    fn test_pumping_source_equals_destination_invalid_value() {
        let mut data = make_data(
            vec![make_hydro(10, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping(1, 1, 10, 10)];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        let inv: Vec<_> = ctx
            .errors()
            .into_iter()
            .filter(|e| e.kind == ErrorKind::InvalidValue)
            .collect();
        assert_eq!(
            inv.len(),
            1,
            "expected exactly 1 InvalidValue error, got: {:?}",
            inv.iter().map(|e| &e.message).collect::<Vec<_>>()
        );
        let msg = &inv[0].message;
        assert!(
            msg.contains("source_hydro_id == destination_hydro_id"),
            "message should describe the degenerate transfer, got: {msg}"
        );
        assert!(
            msg.contains("10"),
            "message should contain the shared hydro id '10', got: {msg}"
        );
    }

    /// A station with distinct existing source/destination hydros produces no error.
    #[test]
    fn test_pumping_distinct_endpoints_no_error() {
        let mut data = make_data(
            vec![make_hydro(10, None), make_hydro(20, None)],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping(1, 1, 10, 20)];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "distinct endpoints should produce no errors, got: {:?}",
            ctx.errors()
        );
    }

    /// A run-of-river endpoint (an existing hydro with no reservoir capacity) is
    /// accepted: only the source-equals-destination condition is rejected here.
    /// `make_hydro` yields a hydro with no geometry/FPHA — an RoR-like endpoint
    /// from the semantic check's perspective; its distinct ID resolves.
    #[test]
    fn test_pumping_run_of_river_endpoint_no_error() {
        let mut ror = make_hydro(30, None);
        ror.max_storage_hm3 = 0.0; // run-of-river: no reservoir storage
        let mut data = make_data(
            vec![make_hydro(10, None), ror],
            vec![],
            vec![],
            make_stages(vec![0]),
            vec![],
            vec![],
        );
        data.pumping_stations = vec![make_pumping(1, 1, 10, 30)];
        let mut ctx = ValidationContext::new();
        validate_semantic_hydro_thermal(&data, &mut ctx);
        assert!(
            !ctx.has_errors(),
            "run-of-river endpoint should produce no errors, got: {:?}",
            ctx.errors()
        );
    }
}
