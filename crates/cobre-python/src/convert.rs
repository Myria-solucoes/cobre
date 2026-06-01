//! Conversions between Python objects and `serde_json` values.
//!
//! These helpers are the inverse of [`crate::results::json_value_to_py`]: they
//! take a Python object (typically a value from a `config_overrides` mapping)
//! and produce a [`serde_json::Value`]. The dotted-key override map accepted by
//! `cobre.run.run` and `cobre.io.validate` is converted into a
//! [`serde_json::Map`] via [`pydict_to_json_map`] **under the GIL**, before the
//! solver lifecycle releases it with `py.detach`.
//!
//! ## Supported value types
//!
//! | Python type      | JSON value                         |
//! |------------------|------------------------------------|
//! | `None`           | `null`                             |
//! | `bool`           | `bool`                             |
//! | `int`            | number (i64 / u64)                 |
//! | `float`          | number (f64; rejects NaN/Inf)      |
//! | `str`            | string                             |
//! | `list` / `tuple` | array (recursive)                  |
//! | `dict`           | object (string keys, recursive)    |
//!
//! Any other Python type raises [`PyValueError`]. Non-`str` dict keys also raise
//! [`PyValueError`].

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

/// Convert an arbitrary JSON-compatible Python object into a [`serde_json::Value`].
///
/// The conversion is the dual of [`crate::results::json_value_to_py`]. The type
/// dispatch checks `bool` before `int` because in Python `bool` is a subclass of
/// `int`; checking `int` first would coerce `True`/`False` into `1`/`0`.
///
/// # Errors
///
/// Returns [`PyValueError`] when:
/// - `obj` is of an unsupported type (anything not in the table in the module docs).
/// - a `float` is `NaN` or infinite (not representable in JSON).
/// - an integer is outside the `i64`/`u64` range.
/// - a `dict` key is not a `str`.
pub fn py_to_json_value(obj: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if obj.is_none() {
        return Ok(serde_json::Value::Null);
    }

    // `bool` must be checked before `int`: Python `bool` is an `int` subclass.
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(serde_json::Value::Bool(b.is_true()));
    }

    if let Ok(i) = obj.cast::<PyInt>() {
        // Try signed first, then unsigned for values above i64::MAX.
        if let Ok(v) = i.extract::<i64>() {
            return Ok(serde_json::Value::Number(v.into()));
        }
        if let Ok(v) = i.extract::<u64>() {
            return Ok(serde_json::Value::Number(v.into()));
        }
        return Err(PyValueError::new_err(
            "integer override value is out of the supported range (i64/u64)",
        ));
    }

    if let Ok(f) = obj.cast::<PyFloat>() {
        let v = f.value();
        let number = serde_json::Number::from_f64(v).ok_or_else(|| {
            PyValueError::new_err(
                "float override value (NaN or infinity) is not representable in JSON",
            )
        })?;
        return Ok(serde_json::Value::Number(number));
    }

    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(serde_json::Value::String(s.to_str()?.to_owned()));
    }

    if let Ok(list) = obj.cast::<PyList>() {
        let arr = list
            .iter()
            .map(|item| py_to_json_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(tuple) = obj.cast::<PyTuple>() {
        let arr = tuple
            .iter()
            .map(|item| py_to_json_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(serde_json::Value::Array(arr));
    }

    if let Ok(dict) = obj.cast::<PyDict>() {
        return Ok(serde_json::Value::Object(pydict_to_json_map(dict)?));
    }

    Err(PyValueError::new_err(format!(
        "unsupported override value type `{}`: expected one of None, bool, int, \
         float, str, list, tuple, or dict",
        obj.get_type().name()?,
    )))
}

/// Convert a Python `dict` into a [`serde_json::Map`].
///
/// Keys must be Python `str`; a non-`str` key raises [`PyValueError`]. Values are
/// converted recursively via [`py_to_json_value`].
///
/// # Errors
///
/// Returns [`PyValueError`] when a key is not a `str`, or when any value fails
/// [`py_to_json_value`] conversion.
pub fn pydict_to_json_map(
    dict: &Bound<'_, PyDict>,
) -> PyResult<serde_json::Map<String, serde_json::Value>> {
    let mut map = serde_json::Map::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key_str = key.cast::<PyString>().map_err(|_| {
            PyValueError::new_err(format!(
                "override map keys must be strings, got `{}`",
                key.get_type()
                    .name()
                    .map_or_else(|_| "<unknown>".to_owned(), |n| n.to_string()),
            ))
        })?;
        map.insert(key_str.to_str()?.to_owned(), py_to_json_value(&value)?);
    }
    Ok(map)
}

// NOTE: `py_to_json_value` / `pydict_to_json_map` are GIL-bound: a Rust
// `#[cfg(test)]` module that calls `Python::attach` cannot link against the
// extension-module build (no embedded interpreter). Their per-type and
// rejection coverage lives in `tests/test_convert.py`, which exercises them
// through `cobre.io.validate(..., config_overrides=...)` and
// `cobre.run.run(..., config_overrides=...)` — mirroring how
// `results::json_value_to_py` is covered from Python.
