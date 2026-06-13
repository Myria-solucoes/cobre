//! Running statistical accumulators.
//!
//! Groups the online mean/variance accumulator ([`welford`]) used for
//! streaming statistics where values arrive one at a time.

pub mod welford;
