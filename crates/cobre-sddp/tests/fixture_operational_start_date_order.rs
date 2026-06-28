//! Guards that every shipped deterministic-case entity fixture assigns
//! `operational_start_date` so the `(operational_start_date, name)` build order
//! reproduces the ascending-`id` order. A future fixture edit that breaks the
//! date-monotonic-with-id property (e.g. a shared sentinel date that lets the
//! `name` tiebreak reorder a collection) fails here instead of silently moving a
//! parity hash.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::Value;

const ENTITY_FILES: &[&str] = &[
    "buses.json",
    "hydros.json",
    "thermals.json",
    "lines.json",
    "non_controllable_sources.json",
    "pumping_stations.json",
    "energy_contracts.json",
];

fn deterministic_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/deterministic")
}

fn entity_array(doc: &Value) -> Option<&Vec<Value>> {
    doc.as_object()?
        .iter()
        .find(|(key, value)| key.as_str() != "$schema" && value.is_array())
        .and_then(|(_, value)| value.as_array())
}

fn check_file(path: &Path) {
    let text = fs::read_to_string(path).expect("read entity fixture");
    let doc: Value = serde_json::from_str(&text).expect("parse entity fixture");
    let Some(entities) = entity_array(&doc) else {
        return;
    };
    if entities.is_empty() {
        return;
    }

    let mut by_id: Vec<&Value> = entities.iter().collect();
    by_id.sort_by_key(|e| e["id"].as_i64().expect("entity id"));

    let mut by_build: Vec<&Value> = entities.iter().collect();
    by_build.sort_by(|a, b| {
        let da = a["operational_start_date"]
            .as_str()
            .expect("operational_start_date present");
        let db = b["operational_start_date"]
            .as_str()
            .expect("operational_start_date present");
        let na = a["name"].as_str().expect("entity name");
        let nb = b["name"].as_str().expect("entity name");
        (da, na).cmp(&(db, nb))
    });

    let ids_by_id: Vec<i64> = by_id.iter().map(|e| e["id"].as_i64().unwrap()).collect();
    let ids_by_build: Vec<i64> = by_build.iter().map(|e| e["id"].as_i64().unwrap()).collect();
    assert_eq!(
        ids_by_build,
        ids_by_id,
        "{}: (operational_start_date, name) order must equal ascending-id order",
        path.display()
    );
}

#[test]
fn deterministic_fixtures_preserve_id_order_under_build_sort() {
    let root = deterministic_root();
    let mut checked = 0usize;
    for case in fs::read_dir(&root).expect("read deterministic root") {
        let system = case.expect("case dir entry").path().join("system");
        if !system.is_dir() {
            continue;
        }
        for fname in ENTITY_FILES {
            let path = system.join(fname);
            if path.exists() {
                check_file(&path);
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "no deterministic entity fixtures found under {}",
        root.display()
    );
}
