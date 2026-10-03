//! The identity this build stamps into every output it writes.

/// Recorded as `software` in output metadata and policy checkpoints.
pub const SOFTWARE_NAME: &str = "cobre";

/// Recorded as `software_version`. The workspace shares one version, so this
/// crate's version is the product's.
pub const SOFTWARE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// The software an output records as its writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SoftwareIdentity<'a> {
    /// `None` when the output predates recording the name.
    pub name: Option<&'a str>,
    /// The writer's version string, compared verbatim.
    pub version: &'a str,
}

impl SoftwareIdentity<'static> {
    /// This build.
    pub const THIS_BUILD: Self = Self {
        name: Some(SOFTWARE_NAME),
        version: SOFTWARE_VERSION,
    };
}
