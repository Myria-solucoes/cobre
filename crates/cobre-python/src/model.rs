//! `PyO3` wrapper classes exposing `cobre-core` entity types in `cobre.model`.
//!
//! Entity collections are cloned into the wrappers because entity data is small
//! (strings + floats) and cloning avoids lifetime complexity at the `PyO3` boundary.
//!
//! Stub entities (`EnergyContract`, `PumpingStation`, `NonControllableSource`) are
//! data-complete but contribute no LP variables or constraints.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

// ─── Bus ─────────────────────────────────────────────────────────────────────

/// Electrical network node where energy balance is maintained.
///
/// Each bus carries a power balance constraint satisfied at every stage and block.
#[pyclass(name = "Bus", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyBus {
    inner: cobre_core::Bus,
}

#[pymethods]
impl PyBus {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// Pre-resolved piecewise-linear deficit cost segments.
    ///
    /// Each segment is returned as a dict with keys `"depth_mw"` (float or
    /// `None` for the final unbounded segment) and `"cost_per_mwh"` (float).
    #[getter]
    fn deficit_segments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .deficit_segments
            .iter()
            .map(|seg| {
                let d = PyDict::new(py);
                match seg.depth_mw {
                    Some(v) => d.set_item("depth_mw", v)?,
                    None => d.set_item("depth_mw", py.None())?,
                }
                d.set_item("cost_per_mwh", seg.cost_per_mwh)?;
                Ok(d)
            })
            .collect()
    }

    /// Cost per `MWh` for surplus generation absorption [$/`MWh`].
    #[getter]
    fn excess_cost(&self) -> f64 {
        self.inner.excess_cost
    }

    fn __repr__(&self) -> String {
        format!("Bus(id={}, name='{}')", self.inner.id.0, self.inner.name)
    }
}

impl PyBus {
    pub(crate) fn from_rust(bus: cobre_core::Bus) -> Self {
        Self { inner: bus }
    }
}

// ─── Line ────────────────────────────────────────────────────────────────────

/// Transmission interconnection carrying bidirectional power between two buses.
#[pyclass(name = "Line", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyLine {
    inner: cobre_core::Line,
}

#[pymethods]
impl PyLine {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn source_bus_id(&self) -> i32 {
        self.inner.source_bus_id.0
    }

    #[getter]
    fn target_bus_id(&self) -> i32 {
        self.inner.target_bus_id.0
    }

    /// Maximum flow from source to target [MW].
    #[getter]
    fn direct_capacity_mw(&self) -> f64 {
        self.inner.direct_capacity_mw
    }

    /// Maximum flow from target to source [MW].
    #[getter]
    fn reverse_capacity_mw(&self) -> f64 {
        self.inner.reverse_capacity_mw
    }

    /// Transmission losses as percentage (e.g., 2.5 means 2.5%).
    #[getter]
    fn losses_percent(&self) -> f64 {
        self.inner.losses_percent
    }

    /// Regularization cost per `MWh` exchanged [$/`MWh`].
    #[getter]
    fn exchange_cost(&self) -> f64 {
        self.inner.exchange_cost
    }

    fn __repr__(&self) -> String {
        format!(
            "Line(id={}, name='{}', source_bus_id={}, target_bus_id={})",
            self.inner.id.0,
            self.inner.name,
            self.inner.source_bus_id.0,
            self.inner.target_bus_id.0
        )
    }
}

impl PyLine {
    pub(crate) fn from_rust(line: cobre_core::Line) -> Self {
        Self { inner: line }
    }
}

// ─── Thermal ─────────────────────────────────────────────────────────────────

/// Thermal power plant with a scalar marginal cost.
#[pyclass(name = "Thermal", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyThermal {
    inner: cobre_core::Thermal,
}

#[pymethods]
impl PyThermal {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn bus_id(&self) -> i32 {
        self.inner.bus_id.0
    }

    /// Minimum electrical generation (minimum stable load) [MW].
    #[getter]
    fn min_generation_mw(&self) -> f64 {
        self.inner.min_generation_mw
    }

    /// Maximum electrical generation (installed capacity) [MW].
    #[getter]
    fn max_generation_mw(&self) -> f64 {
        self.inner.max_generation_mw
    }

    /// Marginal cost of generation [$/`MWh`].
    #[getter]
    fn cost_per_mwh(&self) -> f64 {
        self.inner.cost_per_mwh
    }

    fn __repr__(&self) -> String {
        format!(
            "Thermal(id={}, name='{}', bus_id={})",
            self.inner.id.0, self.inner.name, self.inner.bus_id.0
        )
    }
}

impl PyThermal {
    pub(crate) fn from_rust(thermal: cobre_core::Thermal) -> Self {
        Self { inner: thermal }
    }
}

// ─── Hydro ───────────────────────────────────────────────────────────────────

/// Hydroelectric power plant with reservoir storage and cascade topology.
///
/// Plants form a cascade via `downstream_id` references.
#[pyclass(name = "Hydro", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyHydro {
    inner: cobre_core::Hydro,
}

#[pymethods]
impl PyHydro {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[getter]
    fn bus_id(&self) -> i32 {
        self.inner.bus_id.0
    }

    #[getter]
    fn downstream_id(&self) -> Option<i32> {
        self.inner.downstream_id.map(|id| id.0)
    }

    /// Minimum operational storage (dead volume) [hm³].
    #[getter]
    fn min_storage_hm3(&self) -> f64 {
        self.inner.min_storage_hm3
    }

    /// Maximum operational storage (flood control level) [hm³].
    #[getter]
    fn max_storage_hm3(&self) -> f64 {
        self.inner.max_storage_hm3
    }

    /// Minimum turbined flow [m³/s].
    #[getter]
    fn min_turbined_m3s(&self) -> f64 {
        self.inner.min_turbined_m3s
    }

    /// Maximum turbined flow (installed turbine capacity) [m³/s].
    #[getter]
    fn max_turbined_m3s(&self) -> f64 {
        self.inner.max_turbined_m3s
    }

    /// Power output per unit of turbined flow [MW/(m³/s)].
    ///
    /// Always `None`: the productivity scalar is resolved per-stage from
    /// `system/hydro_production_models.json`; this getter exists for ABI continuity.
    #[getter]
    fn productivity_mw_per_m3s(&self) -> Option<f64> {
        match self.inner.generation_model {
            cobre_core::HydroGenerationModel::ConstantProductivity
            | cobre_core::HydroGenerationModel::LinearizedHead
            | cobre_core::HydroGenerationModel::Fpha => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Hydro(id={}, name='{}', bus_id={})",
            self.inner.id.0, self.inner.name, self.inner.bus_id.0
        )
    }
}

impl PyHydro {
    pub(crate) fn from_rust(hydro: cobre_core::Hydro) -> Self {
        Self { inner: hydro }
    }
}

// ─── EnergyContract (minimal stub) ───────────────────────────────────────────

/// Bilateral energy contract with an external system.
#[pyclass(name = "EnergyContract", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyEnergyContract {
    inner: cobre_core::EnergyContract,
}

#[pymethods]
impl PyEnergyContract {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn __repr__(&self) -> String {
        format!(
            "EnergyContract(id={}, name='{}')",
            self.inner.id.0, self.inner.name
        )
    }
}

impl PyEnergyContract {
    pub(crate) fn from_rust(contract: cobre_core::EnergyContract) -> Self {
        Self { inner: contract }
    }
}

// ─── PumpingStation (minimal stub) ───────────────────────────────────────────

/// Pumping station that transfers water between hydro reservoirs.
#[pyclass(name = "PumpingStation", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPumpingStation {
    inner: cobre_core::PumpingStation,
}

#[pymethods]
impl PyPumpingStation {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn __repr__(&self) -> String {
        format!(
            "PumpingStation(id={}, name='{}')",
            self.inner.id.0, self.inner.name
        )
    }
}

impl PyPumpingStation {
    pub(crate) fn from_rust(station: cobre_core::PumpingStation) -> Self {
        Self { inner: station }
    }
}

// ─── NonControllableSource (minimal stub) ────────────────────────────────────

/// Intermittent generation source that cannot be dispatched.
#[pyclass(name = "NonControllableSource", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyNonControllableSource {
    inner: cobre_core::NonControllableSource,
}

#[pymethods]
impl PyNonControllableSource {
    #[getter]
    fn id(&self) -> i32 {
        self.inner.id.0
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn __repr__(&self) -> String {
        format!(
            "NonControllableSource(id={}, name='{}')",
            self.inner.id.0, self.inner.name
        )
    }
}

impl PyNonControllableSource {
    pub(crate) fn from_rust(source: cobre_core::NonControllableSource) -> Self {
        Self { inner: source }
    }
}

// ─── System ──────────────────────────────────────────────────────────────────

/// Top-level system representation wrapping a loaded Cobre case.
///
/// Cannot be constructed from Python — use `cobre.io.load_case()` to obtain one.
#[pyclass(name = "System", frozen)]
pub struct PySystem {
    inner: Arc<cobre_core::System>,
}

#[pymethods]
impl PySystem {
    /// All buses in canonical ID order.
    #[getter]
    fn buses(&self) -> Vec<PyBus> {
        self.inner
            .buses()
            .iter()
            .cloned()
            .map(PyBus::from_rust)
            .collect()
    }

    /// All transmission lines in canonical ID order.
    #[getter]
    fn lines(&self) -> Vec<PyLine> {
        self.inner
            .lines()
            .iter()
            .cloned()
            .map(PyLine::from_rust)
            .collect()
    }

    /// All thermal plants in canonical ID order.
    #[getter]
    fn thermals(&self) -> Vec<PyThermal> {
        self.inner
            .thermals()
            .iter()
            .cloned()
            .map(PyThermal::from_rust)
            .collect()
    }

    /// All hydro plants in canonical ID order.
    #[getter]
    fn hydros(&self) -> Vec<PyHydro> {
        self.inner
            .hydros()
            .iter()
            .cloned()
            .map(PyHydro::from_rust)
            .collect()
    }

    /// All energy contracts in canonical ID order.
    #[getter]
    fn contracts(&self) -> Vec<PyEnergyContract> {
        self.inner
            .contracts()
            .iter()
            .cloned()
            .map(PyEnergyContract::from_rust)
            .collect()
    }

    /// All pumping stations in canonical ID order.
    #[getter]
    fn pumping_stations(&self) -> Vec<PyPumpingStation> {
        self.inner
            .pumping_stations()
            .iter()
            .cloned()
            .map(PyPumpingStation::from_rust)
            .collect()
    }

    /// All non-controllable sources in canonical ID order.
    #[getter]
    fn non_controllable_sources(&self) -> Vec<PyNonControllableSource> {
        self.inner
            .non_controllable_sources()
            .iter()
            .cloned()
            .map(PyNonControllableSource::from_rust)
            .collect()
    }

    /// Number of buses in the system.
    #[getter]
    fn n_buses(&self) -> usize {
        self.inner.n_buses()
    }

    /// Number of transmission lines in the system.
    #[getter]
    fn n_lines(&self) -> usize {
        self.inner.n_lines()
    }

    /// Number of hydro plants in the system.
    #[getter]
    fn n_hydros(&self) -> usize {
        self.inner.n_hydros()
    }

    /// Number of thermal plants in the system.
    #[getter]
    fn n_thermals(&self) -> usize {
        self.inner.n_thermals()
    }

    /// Number of stages (study and pre-study) in the system.
    #[getter]
    fn n_stages(&self) -> usize {
        self.inner.n_stages()
    }

    fn __repr__(&self) -> String {
        format!(
            "System(n_buses={}, n_lines={}, n_hydros={}, n_thermals={}, n_stages={})",
            self.inner.n_buses(),
            self.inner.n_lines(),
            self.inner.n_hydros(),
            self.inner.n_thermals(),
            self.inner.n_stages(),
        )
    }
}

impl PySystem {
    pub(crate) fn from_rust(system: cobre_core::System) -> Self {
        Self {
            inner: Arc::new(system),
        }
    }

    /// Wrap an already-shared [`cobre_core::System`] via a refcount bump rather
    /// than a full clone (used by [`crate::study::Study`]'s `system` getter).
    pub(crate) fn from_arc(inner: Arc<cobre_core::System>) -> Self {
        Self { inner }
    }
}
