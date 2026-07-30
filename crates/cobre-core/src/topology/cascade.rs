//! Resolved hydro cascade topology.
//!
//! `CascadeTopology` holds the validated, cycle-free directed graph of hydro plant
//! relationships. It is built during case loading after all `Hydro` entities have
//! been validated and their `downstream_id` cross-references verified.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::{EntityId, Hydro};

/// Resolved hydro cascade graph for water balance traversal.
///
/// A directed forest where each hydro has at most one downstream plant; built
/// from hydro `downstream_id` fields during System construction and immutable
/// thereafter. Traversing in topological order guarantees every upstream
/// inflow is computed before the downstream plant that receives it.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CascadeTopology {
    /// Terminal nodes (no downstream) are absent from the map.
    downstream: HashMap<EntityId, EntityId>,

    /// Headwaters (no upstream) are absent from the map.
    upstream: HashMap<EntityId, Vec<EntityId>>,

    /// Upstream before downstream; ties broken by `EntityId`'s inner i32 for determinism.
    topological_order: Vec<EntityId>,
}

impl CascadeTopology {
    /// Build cascade topology from hydro entities, assumed to be in canonical ID order.
    ///
    /// Does not validate: cycles and `downstream_id`s referencing a non-existent
    /// hydro are stored as-is for the separate validation layer to catch.
    #[must_use]
    pub fn build(hydros: &[Hydro]) -> Self {
        let mut downstream: HashMap<EntityId, EntityId> = HashMap::new();
        for hydro in hydros {
            if let Some(ds_id) = hydro.downstream_id {
                downstream.insert(hydro.id, ds_id);
            }
        }

        let mut upstream: HashMap<EntityId, Vec<EntityId>> = HashMap::new();
        for (from, to) in &downstream {
            upstream.entry(*to).or_default().push(*from);
        }
        for upstream_list in upstream.values_mut() {
            upstream_list.sort_by_key(|id| id.0);
        }

        let mut in_degree: HashMap<EntityId, usize> = HashMap::new();
        for hydro in hydros {
            in_degree.insert(hydro.id, 0);
        }
        for to in downstream.values() {
            if let Some(deg) = in_degree.get_mut(to) {
                *deg += 1;
            }
        }

        let mut ready: BinaryHeap<Reverse<i32>> = in_degree
            .iter()
            .filter(|&(_, deg)| *deg == 0)
            .map(|(id, _)| Reverse(id.0))
            .collect();

        let mut topological_order: Vec<EntityId> = Vec::with_capacity(hydros.len());

        while let Some(Reverse(current_raw)) = ready.pop() {
            let current = EntityId(current_raw);
            topological_order.push(current);

            if let Some(&ds_id) = downstream.get(&current)
                && let Some(deg) = in_degree.get_mut(&ds_id)
            {
                *deg -= 1;
                if *deg == 0 {
                    ready.push(Reverse(ds_id.0));
                }
            }
        }

        Self {
            downstream,
            upstream,
            topological_order,
        }
    }

    /// Returns the downstream hydro, or `None` for a terminal node.
    #[must_use]
    pub fn downstream(&self, hydro_id: EntityId) -> Option<EntityId> {
        self.downstream.get(&hydro_id).copied()
    }

    /// Returns the upstream hydros, or an empty slice for a headwater.
    #[must_use]
    pub fn upstream(&self, hydro_id: EntityId) -> &[EntityId] {
        self.upstream.get(&hydro_id).map_or(&[], Vec::as_slice)
    }

    /// Returns the topological ordering of all hydro IDs (upstream before downstream).
    #[must_use]
    pub fn topological_order(&self) -> &[EntityId] {
        &self.topological_order
    }

    /// Returns true if the given hydro is a headwater (has no upstream plants).
    #[must_use]
    pub fn is_headwater(&self, hydro_id: EntityId) -> bool {
        !self.upstream.contains_key(&hydro_id)
    }

    /// Returns true if the given hydro is a terminal node (has no downstream plant).
    #[must_use]
    pub fn is_terminal(&self, hydro_id: EntityId) -> bool {
        !self.downstream.contains_key(&hydro_id)
    }

    /// Returns the number of hydros in the cascade.
    #[must_use]
    pub fn len(&self) -> usize {
        self.topological_order.len()
    }

    /// Returns true if the cascade has no hydros.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.topological_order.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::CascadeTopology;
    use crate::{
        EntityId, Hydro,
        entities::{HydroGenerationModel, HydroPenalties},
    };
    use chrono::NaiveDate;

    fn make_hydro(id: i32, downstream_id: Option<i32>) -> Hydro {
        let zero_penalties = HydroPenalties {
            spillage_cost: 0.0,
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
        };
        let mut hydro = Hydro {
            unit_groups: Vec::new(),
            id: EntityId(id),
            name: String::new(),
            operational_start_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(),
            downstream_id: downstream_id.map(EntityId),
            travel_time_hours: None,
            entry_stage_id: None,
            exit_stage_id: None,
            min_storage_hm3: 0.0,
            max_storage_hm3: 1.0,
            min_outflow_m3s: 0.0,
            max_outflow_m3s: None,
            generation_model: HydroGenerationModel::ConstantProductivity,
            min_turbined_m3s: 0.0,
            max_turbined_m3s: 1.0,
            specific_productivity_mw_per_m3s_per_m: None,
            min_generation_mw: 0.0,
            max_generation_mw: 1.0,
            tailrace: None,
            hydraulic_losses: None,
            efficiency: None,
            evaporation_coefficients_mm: None,
            evaporation_reference_volumes_hm3: None,
            diversion: None,
            filling: None,
            penalties: zero_penalties,
        };
        hydro.declare_mirror_unit_group(EntityId(0));
        hydro
    }

    #[test]
    fn test_empty_cascade() {
        let topo = CascadeTopology::build(&[]);
        assert_eq!(topo.len(), 0);
        assert!(topo.is_empty());
        assert_eq!(topo.topological_order(), &[]);
    }

    #[test]
    fn test_single_hydro_terminal() {
        let hydros = vec![make_hydro(1, None)];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 1);
        assert!(!topo.is_empty());
        assert!(topo.is_headwater(EntityId(1)));
        assert!(topo.is_terminal(EntityId(1)));
        assert_eq!(topo.topological_order(), &[EntityId(1)]);
        assert_eq!(topo.downstream(EntityId(1)), None);
        assert_eq!(topo.upstream(EntityId(1)), &[]);
    }

    #[test]
    fn test_linear_chain() {
        // A(0) -> B(1) -> C(2)
        let hydros = vec![
            make_hydro(0, Some(1)),
            make_hydro(1, Some(2)),
            make_hydro(2, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);

        assert_eq!(topo.downstream(EntityId(0)), Some(EntityId(1)));
        assert_eq!(topo.downstream(EntityId(1)), Some(EntityId(2)));
        assert_eq!(topo.downstream(EntityId(2)), None);

        assert_eq!(topo.upstream(EntityId(0)), &[]);
        assert_eq!(topo.upstream(EntityId(1)), &[EntityId(0)]);
        assert_eq!(topo.upstream(EntityId(2)), &[EntityId(1)]);

        let order = topo.topological_order();
        let pos_a = order.iter().position(|&id| id == EntityId(0)).unwrap();
        let pos_b = order.iter().position(|&id| id == EntityId(1)).unwrap();
        let pos_c = order.iter().position(|&id| id == EntityId(2)).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_fork_merge() {
        // A(0)->C(2), B(1)->C(2): two headwaters merge at C
        let hydros = vec![
            make_hydro(0, Some(2)),
            make_hydro(1, Some(2)),
            make_hydro(2, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);

        assert_eq!(topo.downstream(EntityId(0)), Some(EntityId(2)));
        assert_eq!(topo.downstream(EntityId(1)), Some(EntityId(2)));
        assert_eq!(topo.downstream(EntityId(2)), None);

        let upstream_c = topo.upstream(EntityId(2));
        assert_eq!(upstream_c.len(), 2);
        assert!(upstream_c.contains(&EntityId(0)));
        assert!(upstream_c.contains(&EntityId(1)));
        // Sorted by inner i32: A(0) before B(1)
        assert_eq!(upstream_c[0], EntityId(0));
        assert_eq!(upstream_c[1], EntityId(1));

        // Topo order: A and B before C
        let order = topo.topological_order();
        let pos_a = order.iter().position(|&id| id == EntityId(0)).unwrap();
        let pos_b = order.iter().position(|&id| id == EntityId(1)).unwrap();
        let pos_c = order.iter().position(|&id| id == EntityId(2)).unwrap();
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_parallel_chains() {
        // A(0)->B(1) and C(2)->D(3): two independent chains
        let hydros = vec![
            make_hydro(0, Some(1)),
            make_hydro(1, None),
            make_hydro(2, Some(3)),
            make_hydro(3, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 4);

        let order = topo.topological_order();
        let pos_a = order.iter().position(|&id| id == EntityId(0)).unwrap();
        let pos_b = order.iter().position(|&id| id == EntityId(1)).unwrap();
        let pos_c = order.iter().position(|&id| id == EntityId(2)).unwrap();
        let pos_d = order.iter().position(|&id| id == EntityId(3)).unwrap();

        // A before B, C before D
        assert!(pos_a < pos_b);
        assert!(pos_c < pos_d);
        // All hydros are represented
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn test_all_terminal() {
        // Three hydros, all terminal -- topological order is canonical ID order
        let hydros = vec![
            make_hydro(1, None),
            make_hydro(2, None),
            make_hydro(3, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);

        // All headwaters and all terminals
        for id in [1, 2, 3] {
            assert!(topo.is_headwater(EntityId(id)));
            assert!(topo.is_terminal(EntityId(id)));
        }

        // Topological order is canonical ID order (all headwaters, sorted by i32)
        assert_eq!(
            topo.topological_order(),
            &[EntityId(1), EntityId(2), EntityId(3)]
        );
    }

    #[test]
    fn test_deterministic_ordering() {
        // Same input built twice must produce identical results
        let hydros = vec![
            make_hydro(5, Some(10)),
            make_hydro(3, Some(10)),
            make_hydro(10, None),
        ];
        let topo_a = CascadeTopology::build(&hydros);
        let topo_b = CascadeTopology::build(&hydros);
        assert_eq!(topo_a, topo_b);
        assert_eq!(topo_a.upstream(EntityId(10)), topo_b.upstream(EntityId(10)));
        assert_eq!(topo_a.topological_order(), topo_b.topological_order());
    }

    #[test]
    fn test_is_headwater() {
        // A(0)->C(2), B(1)->C(2): A and B are headwaters, C is not
        let hydros = vec![
            make_hydro(0, Some(2)),
            make_hydro(1, Some(2)),
            make_hydro(2, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert!(topo.is_headwater(EntityId(0)));
        assert!(topo.is_headwater(EntityId(1)));
        assert!(!topo.is_headwater(EntityId(2)));
    }

    #[test]
    fn test_is_terminal() {
        // A(0)->C(2), B(1)->C(2): C is terminal, A and B are not
        let hydros = vec![
            make_hydro(0, Some(2)),
            make_hydro(1, Some(2)),
            make_hydro(2, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert!(!topo.is_terminal(EntityId(0)));
        assert!(!topo.is_terminal(EntityId(1)));
        assert!(topo.is_terminal(EntityId(2)));
    }

    #[test]
    fn test_len() {
        let hydros = vec![
            make_hydro(0, Some(1)),
            make_hydro(1, Some(2)),
            make_hydro(2, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        assert_eq!(topo.len(), 3);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_topology_serde_roundtrip_cascade() {
        // Linear cascade: A(0) -> B(1) -> C(2, terminal).
        let hydros = vec![
            make_hydro(0, Some(1)),
            make_hydro(1, Some(2)),
            make_hydro(2, None),
        ];
        let topo = CascadeTopology::build(&hydros);
        let json = serde_json::to_string(&topo).unwrap();
        let deserialized: CascadeTopology = serde_json::from_str(&json).unwrap();
        assert_eq!(topo, deserialized);
    }
}
