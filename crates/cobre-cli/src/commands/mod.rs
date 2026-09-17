//! Subcommand implementations for the `cobre` binary.
//!
//! Each module pairs a clap-derived `Args` struct with an `execute` function.

pub(crate) mod broadcast;
pub mod init;
pub mod run;
pub mod schema;
pub mod validate;
pub mod version;
